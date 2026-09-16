"""Relation inference for the caption experiment, matched to the relation evaluation.

The relation model's reported number (geometry+CLIP+union, 64.24% mean Top-1)
was produced by eval_gt_relations.py with: the checkpoint rebuilt by
`load_checkpoint`, geometry from `geo_extractor(model_config.geo_mode)` on the
real image size, CLIP features that build_clip_cache.py encoded through
`CLIPExtractor.encode_crops` on boxes clipped by `_clamp_box` (the union region
being the bounding box of the two RAW boxes, clipped afterwards), and the raw
argmax over the non-PAD/UNK predicates - no temperature, no priors, no
overrides.

The legacy caption path (relation_prediction.predict.infer_relationships_semantic,
used by grounded_caption_pipeline.py) is NOT that model: it softmaxes at T=2,
adds hand-set prior bonuses of up to +0.22 to the scores, can REPLACE the
model's argmax with a semantic predicate the model did not choose ("consistency
override"), and encodes every crop again for every ordered pair. Whatever it
feeds BLIP is not the prediction whose accuracy was measured. This module
reuses the evaluation code path instead, so the relation reaching the caption
is the one the relation experiment scored.

Selection policy (predeclared, see CAPTION_EXPERIMENTS.md)
----------------------------------------------------------
* candidate pairs: every ordered pair of distinct verified detections whose
  label is in the model vocabulary and whose box is at least MIN_BOX_SIZE on
  both sides (the training population drops smaller boxes);
* prediction: argmax over valid predicates; confidence = softmax probability
  of that predicate over the valid predicates at T=1 (uncalibrated);
* semantic filter: the repository's `_is_extreme_nonsense` rule set rejects
  the pair if its top-1 triple is implausible (the predicate is not swapped);
* selection: the single highest-confidence surviving pair (ties: lower
  subject index, then lower object index);
* use: injected into the BLIP prefix only if confidence >= threshold.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from relation_prediction.clip_extractor import _clamp_box
from relation_prediction.vg_dataset import (
    MIN_BOX_SIZE,
    Vocab,
    geo_extractor,
    normalize_label,
)

SELECTED_VARIANT = "geometry_clip_union"
SEEDS = (42, 43, 44)
DEFAULT_CHECKPOINT_ROOT = "checkpoints_gpu"
RELATION_CONFIDENCE_THRESHOLD = 0.5

# The feature signature a geometry+CLIP+union checkpoint must carry. Anything
# else is a different model than the one the experiment reports.
EXPECTED_CONFIG: Dict[str, object] = {
    "model_type": "mlp",
    "geo_mode": "ext",
    "geo_dim": 19,
    "geo_norm": True,
    "clip_dim": 768,
    "union_dim": 768,
    "pose_dim": 0,
    "embed_dim": 64,
    "input_dim": 2 * 64 + 19 + 2 * 768 + 768,   # 2451
}


class CheckpointMismatch(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Checkpoint selection - validation only
# ---------------------------------------------------------------------------

def select_checkpoint(checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT,
                      variant: str = SELECTED_VARIANT,
                      seeds: Sequence[int] = SEEDS) -> Dict:
    """Pick one seed of `variant` using ONLY what training recorded on validation.

    Rule (fixed before any caption was generated):
      1. highest validation Top-1 (`val_acc` in training_meta.json) - the same
         metric train_full_visual_semantic.py used to pick the epoch that was
         saved (select_metric="top1"), so seed selection and epoch selection
         use one criterion;
      2. tie -> higher `val_macro_f1`;
      3. tie -> lower seed.

    This function never opens results_gpu/ or any test-set result. Picking the
    seed with the best TEST Top-1 would leak the test set into the caption
    experiment's model choice.
    """
    candidates = []
    problems = []
    for seed in seeds:
        tag = f"{variant}_seed{seed}"
        ckpt_dir = Path(checkpoint_root) / tag
        meta_path = ckpt_dir / "training_meta.json"
        weights = ckpt_dir / "relation_mlp.pt"
        if not meta_path.is_file() or not weights.is_file():
            problems.append(f"{tag}: missing {'training_meta.json' if not meta_path.is_file() else 'relation_mlp.pt'}")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        val_acc = meta.get("val_acc")
        if val_acc is None:
            problems.append(f"{tag}: training_meta.json has no val_acc")
            continue
        if meta.get("select_metric", "top1") != "top1":
            problems.append(f"{tag}: trained with select_metric={meta.get('select_metric')!r}")
        if int(meta.get("seed", seed)) != seed:
            problems.append(f"{tag}: training_meta seed {meta.get('seed')} != {seed}")
        candidates.append({
            "seed": seed,
            "tag": tag,
            "checkpoint_dir": str(ckpt_dir).replace("\\", "/"),
            "val_top1": float(val_acc),
            "val_macro_f1": (float(meta["val_macro_f1"])
                             if meta.get("val_macro_f1") is not None else float("-inf")),
            "best_epoch": meta.get("epoch"),
            "n_samples_val": (meta.get("split") or {}).get("n_samples_val"),
        })
    if problems:
        raise CheckpointMismatch("checkpoint selection cannot proceed:\n  "
                                 + "\n  ".join(problems))
    if not candidates:
        raise CheckpointMismatch(f"no {variant} checkpoints under {checkpoint_root}")
    ordered = sorted(candidates, key=lambda c: (-c["val_top1"], -c["val_macro_f1"], c["seed"]))
    chosen = ordered[0]
    for c in candidates:
        if c["val_macro_f1"] == float("-inf"):
            c["val_macro_f1"] = None
    return {
        "variant": variant,
        "rule": ("max validation top-1 (training_meta.val_acc); tie -> max "
                 "val_macro_f1; tie -> lowest seed. Test metrics are not read."),
        "candidates": candidates,
        "selected": chosen,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class RelationBundle:
    def __init__(self, model, label_vocab: Vocab, pred_vocab: Vocab, info: Dict,
                 device: torch.device):
        self.model = model
        self.label_vocab = label_vocab
        self.pred_vocab = pred_vocab
        self.info = info
        self.config = info["inferred_config"]
        self.device = device
        self.geo_fn, geo_dim = geo_extractor(self.config["geo_mode"])
        if geo_dim != self.config["geo_dim"]:
            raise CheckpointMismatch(
                f"geo_mode {self.config['geo_mode']!r} yields {geo_dim} columns but "
                f"the checkpoint expects {self.config['geo_dim']}")
        self.valid_idxs = [i for i in range(len(pred_vocab))
                           if pred_vocab.token(i) not in (Vocab.PAD, Vocab.UNK)]

    @property
    def clip_dim(self) -> int:
        return int(self.config["clip_dim"])

    @property
    def union_dim(self) -> int:
        return int(self.config["union_dim"])


def load_relation_bundle(checkpoint_dir: str, device: Optional[torch.device] = None,
                         expect: Optional[Mapping] = EXPECTED_CONFIG) -> RelationBundle:
    """Rebuild a checkpoint with the SAME loader eval_gt_relations.py scored it with."""
    from eval_gt_relations import load_checkpoint

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_vocab, pred_vocab, info = load_checkpoint(checkpoint_dir, device)
    cfg = info["inferred_config"]
    if expect:
        diffs = {k: (v, cfg.get(k)) for k, v in expect.items() if cfg.get(k) != v}
        if diffs:
            raise CheckpointMismatch(
                "checkpoint does not have the geometry+CLIP+union feature signature: "
                + ", ".join(f"{k}: expected {e}, got {g}" for k, (e, g) in diffs.items()))
    if getattr(model, "input_dim", None) != cfg.get("input_dim"):
        raise CheckpointMismatch(f"model input_dim {getattr(model, 'input_dim', None)} "
                                 f"!= config {cfg.get('input_dim')}")
    return RelationBundle(model, label_vocab, pred_vocab, info, device)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def union_box(box_a: Sequence[float], box_b: Sequence[float]) -> Tuple[float, float, float, float]:
    """Union region exactly as build_clip_cache.enumerate_work and
    CLIPExtractor.extract_union_embedding define it: bbox of the RAW boxes,
    clipped to the image only afterwards."""
    return (min(box_a[0], box_b[0]), min(box_a[1], box_b[1]),
            max(box_a[2], box_b[2]), max(box_a[3], box_b[3]))


def eligible_detections(detections: Sequence[Mapping], label_vocab: Vocab,
                        image_size: Tuple[int, int]) -> Tuple[List[Dict], List[Dict]]:
    """Detections the relation model can take, plus why the others were dropped."""
    w, h = image_size
    kept, dropped = [], []
    for d in detections:
        label = normalize_label(str(d["label"]).replace("_", " "))
        box = [float(v) for v in d["box"]]
        reason = None
        if label == "UNK" or label_vocab[label] == label_vocab[Vocab.UNK]:
            reason = "label_not_in_relation_vocab"
        elif (box[2] - box[0]) < MIN_BOX_SIZE or (box[3] - box[1]) < MIN_BOX_SIZE:
            reason = f"box_smaller_than_{MIN_BOX_SIZE}px"
        elif _clamp_box(box, w, h) is None:
            reason = "degenerate_box"
        entry = {"label": label, "box": box, "score": float(d.get("score", 0.0))}
        if reason:
            dropped.append({**entry, "reason": reason})
        else:
            kept.append(entry)
    return kept, dropped


def build_pair_batch(image, detections: Sequence[Mapping], bundle: RelationBundle,
                     encode_crops, batch_size: int = 64) -> Dict:
    """All ordered pairs of `detections` as one model batch.

    `encode_crops(list_of_PIL_crops) -> Tensor[n, CLIP_DIM]` is
    CLIPExtractor.encode_crops in production. Each object crop is encoded once
    and each unordered union region once (the (i, j) and (j, i) unions are the
    same pixels), so an image with n detections costs n + n(n-1)/2 CLIP
    forwards in batches, not the 3 * n(n-1) single forwards of the legacy path.
    """
    w, h = image.size
    n = len(detections)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    out = {"pairs": pairs, "n_object_crops": 0, "n_union_crops": 0}
    if not pairs:
        return out

    lv = bundle.label_vocab
    subj_idx = torch.tensor([lv[detections[i]["label"]] for i, _ in pairs], dtype=torch.long)
    obj_idx = torch.tensor([lv[detections[j]["label"]] for _, j in pairs], dtype=torch.long)
    geo = torch.tensor([bundle.geo_fn(tuple(detections[i]["box"]), tuple(detections[j]["box"]),
                                      float(w), float(h)) for i, j in pairs],
                       dtype=torch.float32)
    out.update(subj_idx=subj_idx, obj_idx=obj_idx, geo=geo)

    def _encode(boxes: List[Tuple[float, ...]]) -> torch.Tensor:
        crops = [image.crop(_clamp_box(b, w, h)) for b in boxes]
        chunks = [encode_crops(crops[s:s + batch_size]) for s in range(0, len(crops), batch_size)]
        return torch.cat(chunks, dim=0).float()

    if bundle.clip_dim > 0:
        obj_embs = _encode([tuple(d["box"]) for d in detections])
        out["subj_feat"] = obj_embs[[i for i, _ in pairs]]
        out["obj_feat"] = obj_embs[[j for _, j in pairs]]
        out["n_object_crops"] = n
    if bundle.union_dim > 0:
        unordered = sorted({(min(i, j), max(i, j)) for i, j in pairs})
        boxes, keep_rows = [], {}
        for k, (i, j) in enumerate(unordered):
            ub = union_box(detections[i]["box"], detections[j]["box"])
            if _clamp_box(ub, w, h) is None:      # cannot happen for valid object boxes
                raise ValueError(f"degenerate union box {ub}")
            keep_rows[(i, j)] = len(boxes)
            boxes.append(ub)
        union_embs = _encode(boxes)
        out["union_feat"] = union_embs[[keep_rows[(min(i, j), max(i, j))] for i, j in pairs]]
        out["n_union_crops"] = len(boxes)
    return out


@torch.no_grad()
def predict_pairs(bundle: RelationBundle, batch: Mapping, top_k: int = 3) -> List[Dict]:
    """Raw-argmax predictions per pair, mirroring eval_gt_relations.py."""
    if not batch.get("pairs"):
        return []
    dev = bundle.device
    kwargs = {}
    for name in ("subj_feat", "obj_feat", "union_feat"):
        if name in batch:
            kwargs[name] = batch[name].to(dev)
    bundle.model.check_inputs(batch["geo"].to(dev), **kwargs)
    logits = bundle.model(batch["subj_idx"].to(dev), batch["obj_idx"].to(dev),
                          batch["geo"].to(dev), **kwargs)
    valid = torch.tensor(bundle.valid_idxs, dtype=torch.long, device=logits.device)
    probs = torch.softmax(logits.index_select(1, valid).float(), dim=-1).cpu()
    k = min(top_k, probs.shape[1])
    top_p, top_i = probs.topk(k, dim=-1)
    results = []
    for row, (i, j) in enumerate(batch["pairs"]):
        ranked = [(bundle.pred_vocab.token(bundle.valid_idxs[int(ix)]), float(p))
                  for p, ix in zip(top_p[row], top_i[row])]
        results.append({"subject_index": i, "object_index": j,
                        "predicate": ranked[0][0], "confidence": ranked[0][1],
                        "top_k": ranked})
    return results


def select_relation(detections: Sequence[Mapping], predictions: Sequence[Mapping],
                    nonsense_fn=None) -> Tuple[Optional[Dict], List[Dict]]:
    """Apply the semantic filter and pick the single most confident relation."""
    if nonsense_fn is None:
        from relation_prediction.predict import _is_extreme_nonsense as nonsense_fn
    annotated = []
    for p in predictions:
        s = detections[p["subject_index"]]
        o = detections[p["object_index"]]
        rejected = bool(nonsense_fn(s["label"], p["predicate"], o["label"]))
        annotated.append({**p, "subject": s["label"], "object": o["label"],
                          "status": "rejected_semantic_filter" if rejected else "candidate"})
    survivors = [p for p in annotated if p["status"] == "candidate"]
    if not survivors:
        return None, annotated
    best = min(survivors, key=lambda p: (-p["confidence"], p["subject_index"], p["object_index"]))
    s = detections[best["subject_index"]]
    o = detections[best["object_index"]]
    selected = {
        "subject": s["label"], "predicate": best["predicate"], "object": o["label"],
        "confidence": best["confidence"],
        "subject_box": s["box"], "object_box": o["box"],
        "subject_index": best["subject_index"], "object_index": best["object_index"],
        "top_k": best["top_k"],
    }
    best["status"] = "selected"
    return selected, annotated


def relation_decision(selected: Optional[Mapping], n_eligible: int, n_predictions: int,
                      threshold: float) -> Dict:
    """Whether the grounded arm injects a relation for this image, and if not, why."""
    if n_eligible < 2:
        return {"use_relation": False, "fallback_reason": "fewer_than_2_eligible_detections"}
    if selected is None:
        reason = ("all_pairs_rejected_by_semantic_filter" if n_predictions
                  else "no_pairs")
        return {"use_relation": False, "fallback_reason": reason}
    if not (isinstance(selected.get("confidence"), float) and math.isfinite(selected["confidence"])):
        return {"use_relation": False, "fallback_reason": "invalid_confidence"}
    if selected["confidence"] < threshold:
        return {"use_relation": False, "fallback_reason": "below_confidence_threshold"}
    return {"use_relation": True, "fallback_reason": None}

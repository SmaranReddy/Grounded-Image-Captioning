"""
eval_gt_relations.py — E0: image-disjoint ground-truth VG relation evaluation.

Purpose
-------
Produce a trustworthy measurement of relation-prediction quality on Visual
Genome ground-truth annotations, with a split that is disjoint AT THE IMAGE
LEVEL. Every existing split in this repository is a sample-level
`random_split`, which leaks: VG averages ~2.5 kept relations per image, so
sibling relations from the same image land on both sides of the split.

This script is additive. It modifies no existing file and imports the
repository's existing filtering / normalisation / model code unchanged.

Primary evaluation uses RAW LOGITS ONLY. No temperature, no priors, no logit
adjustment, no calibration, no adaptive margins, no heuristic correction, no
gating.

The only restriction applied to the logits is that the PAD and UNK vocabulary
slots are removed from the candidate set (they are not predicates and carry no
ground truth). This is label-space restriction, not calibration.

Usage:
    python eval_gt_relations.py
    python eval_gt_relations.py --checkpoint-dir ./checkpoints --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch

# --- existing repository logic, imported unchanged --------------------------
from relation_prediction.model import RelationMLP
from relation_prediction.relation_transformer import RelationTransformer
from relation_prediction.vg_dataset import (
    VGRelationshipDataset,
    Vocab,
    GEO_DIM,
    GEO_DIM_EXT,
    MIN_BOX_SIZE,
    POSE_FEATURE_DIM,
    UNION_FEATURE_DIM,
    ALLOWED_PREDICATES,
    normalize_predicate,
    normalize_predicate_v2,
    predicate_normalizer,
    reachable_predicates,
    PREDICATE_SCHEMES,
    normalize_label,
    extract_geo_features,
    extract_geo_features_ext,
    geo_extractor,
    _get_name,
    _xywh_to_xyxy,
)
from relation_prediction.predict import _infer_hidden_dims, _infer_clip_dim
from relation_prediction.model import format_feature_blocks
from utils.console import configure_safe_stdio


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

VG_ROOT        = "./data/visual_genome"
CHECKPOINT_DIR = "./checkpoints"
SPLIT_MANIFEST = "./splits/e0_image_split.json"
RESULTS_DIR    = "./results/e0"

SEED       = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test fraction is the remainder (0.15), computed so the three always sum to 1.

BATCH_SIZE = 4096
TOPK       = (1, 3, 5)


# ---------------------------------------------------------------------------
# Streaming loader for relationships.json
# ---------------------------------------------------------------------------

def stream_relationship_records(path: str):
    """Yield one top-level per-image record at a time.

    relationships.json is ~709 MB; json.load() needs several GB of RAM. This
    incremental decoder keeps peak memory to one record plus a bounded buffer.
    """
    dec = json.JSONDecoder()
    ws = re.compile(r"[\s,]*")
    with open(path, "r", encoding="utf-8") as f:
        buf = f.read(1 << 22)
        pos = buf.index("[") + 1
        while True:
            while True:
                pos = ws.match(buf, pos).end()
                if pos < len(buf) and buf[pos] == "]":
                    return
                try:
                    obj, pos = dec.raw_decode(buf, pos)
                    break
                except ValueError:
                    chunk = f.read(1 << 22)
                    if not chunk:
                        return
                    buf = buf[pos:] + chunk
                    pos = 0
            yield obj
            if len(buf) - pos < (1 << 20):
                chunk = f.read(1 << 22)
                if chunk:
                    buf = buf[pos:] + chunk
                    pos = 0


def _geo_mode_for_dim(geo_dim: int) -> str:
    """Name the geometry mode a checkpoint's width implies.

    Only used when a checkpoint predates the geo_mode field. The 0 case
    matters: without it a visual-only model (geo_dim=0) is silently read back
    as "basic" and then fed 5 geometry columns it has no weights for.
    """
    if geo_dim == GEO_DIM_EXT:
        return "ext"
    if geo_dim == GEO_DIM:
        return "basic"
    if geo_dim == 0:
        return "none"
    raise ValueError(f"cannot infer geo_mode from geo_dim={geo_dim}")


def build_samples(vg_root: str, label_vocab: Vocab, pred_vocab: Vocab,
                  geo_mode: str = "basic", predicate_scheme: str = "v1"):
    """Apply the repository's existing filtering behaviour to every relation.

    Mirrors VGRelationshipDataset._load exactly: predicate normalisation +
    allowlist, COCO label normalisation (UNK dropped), MIN_BOX_SIZE rejection,
    and the same 5-dim geometry features normalised by image width/height.

    Returns:
        samples: list of (image_id, subj_idx, obj_idx, geo(list[5]), pred_idx)
        stats:   dict of census counters
    """
    geo_fn, _ = geo_extractor(geo_mode)
    pred_fn = predicate_normalizer(predicate_scheme)

    img_json = os.path.join(vg_root, "image_data.json")
    rel_json = os.path.join(vg_root, "relationships.json")
    for p in (img_json, rel_json):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required VG annotation file missing: {p}")

    with open(img_json, "r", encoding="utf-8") as f:
        img_meta = json.load(f)
    img_size: Dict[int, Tuple[int, int]] = {
        m["image_id"]: (int(m["width"]), int(m["height"])) for m in img_meta
    }
    del img_meta

    samples: List[Tuple] = []
    total_raw = 0
    skipped_small = 0
    n_images_seen = 0

    for img in stream_relationship_records(rel_json):
        n_images_seen += 1
        iid = img.get("image_id")
        img_w, img_h = img_size.get(iid, (1, 1))

        for r in img.get("relationships", []):
            total_raw += 1
            pred = pred_fn(r.get("predicate", ""))
            if pred is None:
                continue

            subj_d = r.get("subject", {})
            obj_d = r.get("object", {})

            subj_name = normalize_label(_get_name(subj_d))
            obj_name = normalize_label(_get_name(obj_d))
            if subj_name == "UNK" or obj_name == "UNK":
                continue

            subj_box = _xywh_to_xyxy(
                subj_d.get("x", 0), subj_d.get("y", 0),
                subj_d.get("w", 1), subj_d.get("h", 1),
            )
            obj_box = _xywh_to_xyxy(
                obj_d.get("x", 0), obj_d.get("y", 0),
                obj_d.get("w", 1), obj_d.get("h", 1),
            )

            if (subj_box[2] - subj_box[0]) < MIN_BOX_SIZE or (subj_box[3] - subj_box[1]) < MIN_BOX_SIZE:
                skipped_small += 1
                continue
            if (obj_box[2] - obj_box[0]) < MIN_BOX_SIZE or (obj_box[3] - obj_box[1]) < MIN_BOX_SIZE:
                skipped_small += 1
                continue

            geo = geo_fn(subj_box, obj_box, img_w, img_h)

            samples.append((
                int(iid),
                label_vocab[subj_name],
                label_vocab[obj_name],
                geo,
                pred_vocab[pred],
            ))

    stats = {
        "n_images_in_relationships_json": n_images_seen,
        "raw_count": total_raw,
        "kept_count": len(samples),
        "skipped_small_box": skipped_small,
        "retention": round(len(samples) / max(total_raw, 1), 6),
    }
    return samples, stats


# ---------------------------------------------------------------------------
# Image-disjoint split
# ---------------------------------------------------------------------------

def build_image_split(image_ids, seed: int, train_frac: float, val_frac: float):
    """Partition IMAGE IDS (never samples) into train / val / test."""
    ids = sorted(int(i) for i in image_ids)   # deterministic base ordering
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def assert_disjoint(train_ids, val_ids, test_ids) -> Dict:
    """Hard assertion: the three image sets must not intersect. Abort if they do."""
    s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
    tr_va = s_tr & s_va
    tr_te = s_tr & s_te
    va_te = s_va & s_te

    if tr_va or tr_te or va_te:
        print("\n[E0] FATAL: image-disjoint assertion FAILED", file=sys.stderr)
        print(f"  train n val : {len(tr_va)} overlapping image ids", file=sys.stderr)
        print(f"  train n test: {len(tr_te)} overlapping image ids", file=sys.stderr)
        print(f"  val   n test: {len(va_te)} overlapping image ids", file=sys.stderr)
        raise SystemExit(2)

    return {
        "train_n_val": 0,
        "train_n_test": 0,
        "val_n_test": 0,
        "status": "PASS",
        "note": "image-level disjointness verified; no image appears in two splits",
    }


def format_overlap_line(overlap: Dict) -> str:
    """The overlap-check status line, ASCII only.

    It used to print U+2229 (the intersection sign). Windows encodes redirected
    stdout with the ANSI code page, cp1252, which has no such character, so
    every evaluation launched by run_visual_experiment.py died on this line
    with UnicodeEncodeError - after the checkpoint, the relationships and the
    split had all loaded.
    """
    return (f"      overlap check: {overlap['status']} "
            f"(train&val={overlap['train_n_val']}, train&test={overlap['train_n_test']}, "
            f"val&test={overlap['val_n_test']})")


# ---------------------------------------------------------------------------
# Checkpoint loading (bare state_dict, geometry-only)
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(checkpoint_dir: str, device: torch.device,
                    checkpoint_file: str = "relation_mlp.pt"):
    model_path = os.path.join(checkpoint_dir, checkpoint_file)
    lv_path = os.path.join(checkpoint_dir, "label_vocab.json")
    pv_path = os.path.join(checkpoint_dir, "pred_vocab.json")
    for p in (model_path, lv_path, pv_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Checkpoint file missing: {p}")

    label_vocab = Vocab.load(lv_path)
    pred_vocab = Vocab.load(pv_path)

    raw = torch.load(model_path, map_location=device, weights_only=True)

    # ---- Transformer (E1): reconstruct straight from the stored config -----
    cfg_in = raw.get("model_config", {}) if isinstance(raw, dict) else {}
    if cfg_in.get("model_type") == "transformer":
        state = raw["model_state_dict"]
        embed_dim = cfg_in.get("embed_dim", 64)
        d_model = cfg_in.get("d_model", 256)
        clip_dim = cfg_in.get("clip_dim", 0)
        pose_dim = cfg_in.get("pose_dim", 0)
        union_dim = cfg_in.get("union_dim", 0)
        # Checkpoints written before the extended geometry descriptor existed
        # carry no geo_dim; they are 5-dim by definition.
        geo_dim = cfg_in.get("geo_dim", GEO_DIM)
        geo_norm = bool(cfg_in.get("geo_norm", False))

        model = RelationTransformer(
            num_labels=len(label_vocab),
            num_predicates=len(pred_vocab),
            d_model=d_model,
            embed_dim=embed_dim,
            clip_dim=clip_dim,
            pose_dim=pose_dim,
            union_dim=union_dim,
            geo_dim=geo_dim,
            geo_norm=geo_norm,
        )
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        ckpt_info = {
            "path": os.path.abspath(model_path).replace("\\", "/"),
            "sha256": sha256_file(model_path),
            "inferred_config": {
                "wrapper_format": "model_state_dict+model_config",
                "model_type": "transformer",
                "num_labels": len(label_vocab),
                "num_predicates": len(pred_vocab),
                "embed_dim": embed_dim,
                "d_model": d_model,
                "clip_dim": clip_dim,
                "pose_dim": pose_dim,
                "union_dim": union_dim,
                "geo_dim": geo_dim,
                "geo_norm": geo_norm,
                "geo_mode": cfg_in.get("geo_mode", _geo_mode_for_dim(geo_dim)),
                "feature_mode": "+".join(
                    ["geometry"]
                    + (["clip"] if clip_dim else [])
                    + (["union"] if union_dim else [])
                    + (["pose"] if pose_dim else [])
                ),
            },
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "label_vocab_path": os.path.abspath(lv_path).replace("\\", "/"),
            "pred_vocab_path": os.path.abspath(pv_path).replace("\\", "/"),
        }
        return model, label_vocab, pred_vocab, ckpt_info

    # ---- MLP: mirrors predict.py's bare-state-dict branch ------------------
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
        config = raw.get("model_config", {})
        embed_dim = config.get("embed_dim", state["label_emb.weight"].shape[1])
        hidden_dims = _infer_hidden_dims(state)
        clip_dim = config.get("clip_dim", 0)
        pose_dim = config.get("pose_dim", 0)
        union_dim = config.get("union_dim", 0)
        geo_dim = config.get("geo_dim", GEO_DIM)
        geo_norm = bool(config.get("geo_norm", False))
        wrapper = "model_state_dict+model_config"
    else:
        state = raw
        embed_dim = state["label_emb.weight"].shape[1]
        hidden_dims = _infer_hidden_dims(state)
        pose_dim = 0
        union_dim = 0
        # A bare state_dict carries no config, so the geometry width has to be
        # recovered from the weights. When the model was trained with
        # --geo-norm the BatchNorm buffer IS the geometry width; only without
        # it must we fall back to the legacy 5.
        #
        # This previously hardcoded geo_dim = GEO_DIM (5) and then called
        # _infer_clip_dim without it, so a bare 19-dim geometry checkpoint was
        # silently reconstructed as geo_dim=5 + clip_dim=7 — a model that loads
        # without error and computes nonsense. predict.py already had this fix;
        # this loader did not, and the two must agree.
        geo_norm = "geo_norm.weight" in state
        geo_dim = int(state["geo_norm.weight"].shape[0]) if geo_norm else GEO_DIM
        clip_dim = _infer_clip_dim(state, embed_dim, geo_dim)
        wrapper = "bare_state_dict"

    model = RelationMLP(
        num_labels=len(label_vocab),
        num_predicates=len(pred_vocab),
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        clip_dim=clip_dim,
        pose_dim=pose_dim,
        union_dim=union_dim,
        geo_dim=geo_dim,
        geo_norm=geo_norm,
    )
    # Checkpoints written since the union-dimension fix record input_dim. When
    # present it must agree with the model rebuilt from the other fields;
    # otherwise the config describes a different model than the one scored.
    saved_input_dim = config.get("input_dim") if wrapper != "bare_state_dict" else None
    if saved_input_dim is not None and int(saved_input_dim) != model.input_dim:
        raise SystemExit(
            f"[E0] FATAL: checkpoint model_config records input_dim={saved_input_dim}, "
            f"but the model rebuilt from it has input_dim={model.input_dim} "
            f"[{format_feature_blocks(model.feature_blocks())}]. Refusing to evaluate."
        )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    ckpt_info = {
        "path": os.path.abspath(model_path).replace("\\", "/"),
        "sha256": sha256_file(model_path),
        "inferred_config": {
            "wrapper_format": wrapper,
            "model_type": "mlp",
            "num_labels": len(label_vocab),
            "num_predicates": len(pred_vocab),
            "embed_dim": embed_dim,
            "hidden_dims": list(hidden_dims),
            "clip_dim": clip_dim,
            "pose_dim": pose_dim,
            "union_dim": union_dim,
            "geo_dim": geo_dim,
            "geo_norm": geo_norm,
            "geo_mode": (config.get("geo_mode") if wrapper != "bare_state_dict" else None)
                        or _geo_mode_for_dim(geo_dim),
            "input_dim": model.input_dim,
            "feature_mode": "geometry-only" if clip_dim == 0 else "+".join(
                ["geometry", "clip"]
                + (["union"] if union_dim else [])
                + (["pose"] if pose_dim else [])
            ),
        },
        "parameter_count": param_count,
        "label_vocab_path": os.path.abspath(lv_path).replace("\\", "/"),
        "pred_vocab_path": os.path.abspath(pv_path).replace("\\", "/"),
    }
    return model, label_vocab, pred_vocab, ckpt_info


# ---------------------------------------------------------------------------
# Visual test set (E1: CLIP / union / pose checkpoints)
# ---------------------------------------------------------------------------

def build_visual_test_set(vg_root, vg_image_dir, clip_cache_path, cfg,
                          test_ids, label_vocab, pred_vocab):
    """Build the TEST-split tensors for a checkpoint that needs visual features.

    Uses VGRelationshipDataset unchanged, with the CHECKPOINT'S vocabularies
    (passing them in sets _build_vocab=False, so indices match the trained
    model). Samples are then restricted to the frozen manifest's test image
    IDs, recovered from sample_keys ("<image_id>_obj_<object_id>").

    No YOLO, no captioning — ground-truth boxes and labels only.
    """
    use_pose = cfg["pose_dim"] > 0
    use_union = cfg["union_dim"] > 0

    ds = VGRelationshipDataset(
        relationships_json=os.path.join(vg_root, "relationships.json"),
        image_data_json=os.path.join(vg_root, "image_data.json"),
        vg_image_dir=vg_image_dir,
        label_vocab=label_vocab,
        pred_vocab=pred_vocab,
        use_visual=True,
        clip_cache_path=clip_cache_path,
        require_visual=True,
        use_pose=use_pose,
        use_union=use_union,
        geo_mode=cfg.get("geo_mode", "basic"),
    )

    test_set = {int(i) for i in test_ids}
    keep: List[int] = []
    images: set = set()
    for i, (subj_key, obj_key) in enumerate(ds.sample_keys):
        key = subj_key or obj_key
        if not key or "_obj_" not in key:
            continue
        try:
            iid = int(key.split("_obj_")[0])
        except ValueError:
            continue
        if iid in test_set:
            keep.append(i)
            images.add(iid)

    if not keep:
        raise SystemExit(
            "[E0] FATAL: no test-split samples survived visual filtering. "
            "The VG images / CLIP cache required by this checkpoint are absent."
        )

    subj, obj, geo, gt = [], [], [], []
    subj_f, obj_f, union_f, pose_f = [], [], [], []
    for i in keep:
        item = ds[i]
        subj.append(item[0]); obj.append(item[1]); geo.append(item[2]); gt.append(int(item[3]))
        idx = 4
        subj_f.append(item[idx]); obj_f.append(item[idx + 1]); idx += 2
        if use_union:
            union_f.append(item[idx]); idx += 1
        if use_pose:
            pose_f.append(item[idx]); idx += 1

    out = {
        "subj": torch.stack(subj),
        "obj": torch.stack(obj),
        "geo": torch.stack(geo),
        "gt": gt,
        "subj_feat": torch.stack(subj_f),
        "obj_feat": torch.stack(obj_f),
        "union_feat": torch.stack(union_f) if use_union else None,
        "pose_feat": torch.stack(pose_f) if use_pose else None,
        "n_images": len(images),
        "n_dataset_total": len(ds),
    }
    return out


def _feature_kwargs(vis, start, end, device):
    """Visual-feature kwargs for test rows [start, end).

    Shared by the inference loop and its preflight so the two cannot build
    different batches. Returns {} for a checkpoint scored without visual data.
    """
    if vis is None:
        return {}
    kwargs = {
        "subj_feat": vis["subj_feat"][start:end].to(device),
        "obj_feat": vis["obj_feat"][start:end].to(device),
    }
    if vis["union_feat"] is not None:
        kwargs["union_feat"] = vis["union_feat"][start:end].to(device)
    if vis["pose_feat"] is not None:
        kwargs["pose_feat"] = vis["pose_feat"][start:end].to(device)
    return kwargs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gt_idxs, top_idxs, pred_vocab, valid_idxs, topk=TOPK):
    """Raw top-1 / recall@k / per-predicate P,R,F1 from the confusion matrix.

    gt_idxs:   list[int]            ground-truth predicate vocab indices
    top_idxs:  list[list[int]]      per-sample ranked predicate indices (desc)
    valid_idxs: predicate vocab indices excluding PAD and UNK
    """
    n = len(gt_idxs)
    top1 = [t[0] for t in top_idxs]

    correct1 = sum(1 for g, p in zip(gt_idxs, top1) if g == p)
    recall_at = {}
    for k in topk:
        hit = sum(1 for g, t in zip(gt_idxs, top_idxs) if g in t[:k])
        recall_at[k] = hit / max(n, 1)

    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    support = Counter()
    predicted_count = Counter()
    tp = Counter()
    for g, p in zip(gt_idxs, top1):
        gname = pred_vocab.token(g)
        pname = pred_vocab.token(p)
        confusion[gname][pname] += 1
        support[gname] += 1
        predicted_count[pname] += 1
        if g == p:
            tp[gname] += 1

    per_predicate: Dict[str, Dict] = {}
    for idx in valid_idxs:
        name = pred_vocab.token(idx)
        s = support[name]
        pc = predicted_count[name]
        t = tp[name]
        precision = t / pc if pc > 0 else 0.0
        recall = t / s if s > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_predicate[name] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": s,
            "predicted_count": pc,
            "true_positives": t,
        }

    scored = [n_ for n_ in per_predicate if per_predicate[n_]["support"] > 0]
    zero_support = sorted(n_ for n_ in per_predicate if per_predicate[n_]["support"] == 0)

    macro_f1 = (sum(per_predicate[n_]["f1"] for n_ in scored) / len(scored)) if scored else 0.0
    total_support = sum(per_predicate[n_]["support"] for n_ in scored)
    weighted_f1 = (
        sum(per_predicate[n_]["f1"] * per_predicate[n_]["support"] for n_ in scored)
        / total_support
    ) if total_support else 0.0

    metrics = {
        "top1": round(correct1 / max(n, 1), 6),
        "recall@3": round(recall_at[3], 6),
        "recall@5": round(recall_at[5], 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "n_test_samples": n,
        "n_correct_top1": correct1,
        "macro_f1_note": "mean F1 over predicates with support > 0",
        "zero_support_predicates": zero_support,
        "n_zero_support_predicates": len(zero_support),
    }
    conf_out = {g: dict(sorted(d.items(), key=lambda kv: -kv[1])) for g, d in confusion.items()}
    return metrics, per_predicate, conf_out


def compute_baselines(train_gt, test_gt, pred_vocab, valid_idxs, seed: int):
    """Majority-class and deterministic uniform-random baselines on the test set."""
    train_counter = Counter(train_gt)
    majority_idx = max(valid_idxs, key=lambda i: train_counter.get(i, 0))
    maj_correct = sum(1 for g in test_gt if g == majority_idx)

    rng = random.Random(seed)
    choices = sorted(valid_idxs)
    rand_correct = sum(1 for g in test_gt if g == rng.choice(choices))

    n = max(len(test_gt), 1)
    return {
        "majority_baseline": round(maj_correct / n, 6),
        "majority_class": pred_vocab.token(majority_idx),
        "majority_class_source": "most frequent predicate in the TRAIN split",
        "random_baseline": round(rand_correct / n, 6),
        "random_baseline_note": (
            f"uniform over {len(choices)} valid predicates, random.Random({seed}), "
            "deterministic"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E0: image-disjoint ground-truth VG relation evaluation.",
    )
    parser.add_argument("--vg-root", default=VG_ROOT)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--split-manifest", default=SPLIT_MANIFEST)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--predicate-scheme", type=str, default=None,
                        choices=list(PREDICATE_SCHEMES),
                        help="Override the predicate normaliser. Default: the "
                             "scheme recorded in the checkpoint, else 'v1'.")
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--rebuild-split", action="store_true",
                        help="Recompute the split manifest even if it exists")
    parser.add_argument("--vg-image-dir", type=str, default=None,
                        help="VG image dir (visual checkpoints only; "
                             "default <vg-root>/images)")
    parser.add_argument("--clip-cache-path", type=str, default=None,
                        help="CLIP cache (visual checkpoints only; "
                             "default <vg-root>/clip_cache_proper.pt)")
    parser.add_argument("--force-visual-population", action="store_true",
                        help="Restrict the test set to samples with complete "
                             "CLIP features even for a geometry-only "
                             "checkpoint. Use this to score a geometry model "
                             "on the same population as a visual one.")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Result basename without .json. Default: "
                             "'transformer_e1' for transformer checkpoints, "
                             "else the checkpoint filename.")
    parser.add_argument("--checkpoint-file", type=str, default="relation_mlp.pt",
                        help="Checkpoint filename inside --checkpoint-dir")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting an existing result file that "
                             "belongs to a different checkpoint")
    args = parser.parse_args()

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  E0 - IMAGE-DISJOINT GROUND-TRUTH VG RELATION EVALUATION")
    print("=" * 70)
    print(f"  device        : {device}")
    print(f"  vg root       : {args.vg_root}")
    print(f"  checkpoint dir: {args.checkpoint_dir}")
    print(f"  seed          : {args.seed}")
    print()

    # --- 1. checkpoint --------------------------------------------------
    print("[1/6] Loading checkpoint ...")
    model, label_vocab, pred_vocab, ckpt_info = load_checkpoint(
        args.checkpoint_dir, device, args.checkpoint_file,
    )
    cfg = ckpt_info["inferred_config"]
    print(f"      format      : {cfg['wrapper_format']}")
    print(f"      model type  : {cfg['model_type']}")
    print(f"      labels      : {cfg['num_labels']}   predicates: {cfg['num_predicates']}")
    if cfg["model_type"] == "transformer":
        print(f"      embed_dim   : {cfg['embed_dim']}   d_model: {cfg['d_model']}")
    else:
        print(f"      embed_dim   : {cfg['embed_dim']}   hidden: {cfg['hidden_dims']}")
        print(f"      input_dim   : {cfg['input_dim']}")
    print(f"      geometry    : {cfg.get('geo_mode', 'basic')} "
          f"({cfg['geo_dim']}-dim, batchnorm={cfg.get('geo_norm', False)})")
    print(f"      features    : {cfg['feature_mode']} "
          f"(clip={cfg['clip_dim']}, union={cfg['union_dim']}, pose={cfg['pose_dim']})")
    print(f"      parameters  : {ckpt_info['parameter_count']:,}")
    print(f"      sha256      : {ckpt_info['sha256'][:16]}...")

    valid_idxs = [
        i for i in range(len(pred_vocab))
        if pred_vocab.token(i) not in (Vocab.PAD, Vocab.UNK)
    ]
    print(f"      scored classes: {len(valid_idxs)} (PAD/UNK excluded)")

    # --- 2. samples -----------------------------------------------------
    print("\n[2/6] Streaming relationships.json and applying repo filters ...")
    # The evaluation MUST reconstruct samples with the same predicate
    # normaliser the checkpoint was trained with, or the test population
    # silently differs from the training population.
    pred_scheme = args.predicate_scheme or cfg.get("predicate_scheme") or "v1"
    cfg["predicate_scheme"] = pred_scheme
    print(f"      predicate scheme: {pred_scheme}"
          f"{' (from checkpoint)' if not args.predicate_scheme else ' (from --predicate-scheme)'}")
    samples, data_stats = build_samples(
        args.vg_root, label_vocab, pred_vocab, geo_mode=cfg.get("geo_mode", "basic"),
        predicate_scheme=pred_scheme,
    )
    qualifying_images = sorted({s[0] for s in samples})
    data_stats["qualifying_image_count"] = len(qualifying_images)
    print(f"      raw relationships : {data_stats['raw_count']:,}")
    print(f"      kept samples      : {data_stats['kept_count']:,}")
    print(f"      retention         : {data_stats['retention']:.4f}")
    print(f"      qualifying images : {data_stats['qualifying_image_count']:,}")

    # --- 3. image-disjoint split ---------------------------------------
    print("\n[3/6] Building image-disjoint split ...")
    manifest_path = args.split_manifest
    reuse = os.path.isfile(manifest_path) and not args.rebuild_split
    if reuse:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        train_ids = manifest["train_ids"]
        val_ids = manifest["val_ids"]
        test_ids = manifest["test_ids"]
        print(f"      reusing frozen manifest: {manifest_path}")
    else:
        train_ids, val_ids, test_ids = build_image_split(
            qualifying_images, args.seed, args.train_frac, args.val_frac,
        )
        manifest = None

    overlap = assert_disjoint(train_ids, val_ids, test_ids)
    print(format_overlap_line(overlap))

    split_of: Dict[int, str] = {}
    for i in train_ids:
        split_of[i] = "train"
    for i in val_ids:
        split_of[i] = "val"
    for i in test_ids:
        split_of[i] = "test"

    buckets: Dict[str, List] = {"train": [], "val": [], "test": []}
    for s in samples:
        b = split_of.get(s[0])
        if b is not None:
            buckets[b].append(s)

    counts = {
        "n_images_train": len(train_ids),
        "n_images_val": len(val_ids),
        "n_images_test": len(test_ids),
        "n_samples_train": len(buckets["train"]),
        "n_samples_val": len(buckets["val"]),
        "n_samples_test": len(buckets["test"]),
    }
    for k, v in counts.items():
        print(f"      {k:16s}: {v:,}")

    if manifest is None:
        os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "fractions": {
                "train": args.train_frac,
                "val": args.val_frac,
                "test": round(1.0 - args.train_frac - args.val_frac, 6),
            },
            "split_unit": "image_id",
            "method": (
                "sorted(image_ids) -> random.Random(seed).shuffle -> "
                "contiguous 70/15/15 slice; every sample inherits its image's split"
            ),
            "filter_fingerprint": {
                "source": "relation_prediction/vg_dataset.py (imported unchanged)",
                "min_box_size": MIN_BOX_SIZE,
                "predicate_scheme": pred_scheme,
                "n_allowed_predicates": len(ALLOWED_PREDICATES),
                "allowed_predicates": sorted(ALLOWED_PREDICATES),
                "reachable_predicates": reachable_predicates(pred_scheme),
                "label_space": "COCO_LABELS via normalize_label; UNK dropped",
            },
            "data": data_stats,
            "counts": counts,
            "overlap_check": overlap,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"      manifest written: {manifest_path}")

    # --- 4. inference on the TEST split ---------------------------------
    print("\n[4/6] Running inference on the TEST split (raw logits only) ...")
    # A checkpoint trained with --visual-filter-only has clip_dim=0 but was
    # fitted on the visual-complete subset, so it must be SCORED on that subset
    # too. Deciding from clip_dim alone would silently score the ablation's
    # geometry control on a larger test set than its CLIP counterparts.
    needs_visual = (cfg["clip_dim"] > 0 or cfg["pose_dim"] > 0 or cfg["union_dim"] > 0
                    or bool(cfg.get("visual_filter_only"))
                    or args.force_visual_population)

    if needs_visual:
        # E1 path: CLIP / union / pose features, ground-truth boxes only.
        print(f"      checkpoint needs visual features ({cfg['feature_mode']}); "
              "building visual test set ...")
        vis = build_visual_test_set(
            args.vg_root,
            args.vg_image_dir or os.path.join(args.vg_root, "images"),
            args.clip_cache_path or os.path.join(args.vg_root, "clip_cache_proper.pt"),
            cfg, test_ids, label_vocab, pred_vocab,
        )
        subj_t, obj_t, geo_t = vis["subj"], vis["obj"], vis["geo"]
        gt_idxs = vis["gt"]
        counts["n_samples_test_geometry_census"] = counts["n_samples_test"]
        counts["n_samples_test"] = len(gt_idxs)
        counts["n_images_test_evaluated"] = vis["n_images"]
        print(f"      visual test samples: {len(gt_idxs):,} "
              f"over {vis['n_images']:,} test images")

        # A visual checkpoint is scored only on test samples whose VG image and
        # CLIP embedding exist. When coverage is partial, that is a DIFFERENT,
        # easier-or-harder test set than the geometry census, and its top-1 is
        # NOT comparable with a geometry-only result. Make that explicit rather
        # than letting the two numbers be put side by side in a table.
        census = counts["n_samples_test_geometry_census"]
        coverage = len(gt_idxs) / max(census, 1)
        counts["test_set_coverage_vs_geometry_census"] = round(coverage, 6)
        counts["comparable_to_geometry_census"] = coverage >= 0.99
        print(f"      coverage vs geometry census: {coverage:.1%} "
              f"({len(gt_idxs):,} / {census:,})")
        if coverage < 0.99:
            print()
            print("      " + "!" * 62)
            print("      WARNING: this checkpoint was evaluated on a SUBSET of the")
            print("      frozen test split, because VG images / CLIP embeddings are")
            print("      missing for the rest. Top-1 here is NOT comparable with a")
            print("      geometry-only result measured on the full census.")
            print("      Either complete the VG image download and rebuild the CLIP")
            print("      cache, or compare only against another model scored on this")
            print("      same subset.")
            print("      " + "!" * 62)
            print()
    else:
        test = buckets["test"]
        if not test:
            raise SystemExit("[E0] FATAL: test split is empty.")
        subj_t = torch.tensor([s[1] for s in test], dtype=torch.long)
        obj_t = torch.tensor([s[2] for s in test], dtype=torch.long)
        geo_t = torch.tensor([s[3] for s in test], dtype=torch.float32)
        gt_idxs = [s[4] for s in test]
        vis = None

    n_test = len(gt_idxs)

    # Restrict the candidate set to real predicates (PAD/UNK are not classes).
    valid_t = torch.tensor(valid_idxs, dtype=torch.long, device=device)
    kmax = max(TOPK)

    # Preflight: the test batch's feature blocks must match the checkpoint
    # before any logits are computed, so a mismatch is reported by block name
    # (FeatureDimensionError) rather than as a matmul shape error.
    if hasattr(model, "check_inputs") and n_test:
        first = min(args.batch_size, n_test)
        actual = model.check_inputs(geo_t[:first].to(device),
                                    **_feature_kwargs(vis, 0, first, device))
        print(f"      feature blocks: {format_feature_blocks(actual)}")
        print(f"      input dim     : batch {sum(w for _, w in actual)} == "
              f"model {model.input_dim}")

    top_idxs: List[List[int]] = []
    with torch.no_grad():
        for start in range(0, n_test, args.batch_size):
            end = min(start + args.batch_size, n_test)
            kwargs = _feature_kwargs(vis, start, end, device)
            logits = model(
                subj_t[start:end].to(device),
                obj_t[start:end].to(device),
                geo_t[start:end].to(device),
                **kwargs,
            )
            # Raw logits. No temperature, priors, calibration or gating.
            sub = logits.index_select(1, valid_t)
            k = min(kmax, sub.size(1))
            order = sub.topk(k, dim=-1).indices
            mapped = valid_t[order]
            top_idxs.extend(mapped.cpu().tolist())
            if (start // args.batch_size) % 5 == 0:
                print(f"      [{end:,}/{n_test:,}]")

    # --- 5. metrics ------------------------------------------------------
    print("\n[5/6] Computing metrics ...")
    metrics, per_predicate, confusion = compute_metrics(
        gt_idxs, top_idxs, pred_vocab, valid_idxs,
    )
    baselines = compute_baselines(
        [s[4] for s in buckets["train"]], gt_idxs, pred_vocab, valid_idxs, args.seed,
    )
    metrics.update({
        "majority_baseline": baselines["majority_baseline"],
        "random_baseline": baselines["random_baseline"],
    })

    print(f"      Top-1        : {metrics['top1']:.4f}")
    print(f"      Recall@3     : {metrics['recall@3']:.4f}")
    print(f"      Recall@5     : {metrics['recall@5']:.4f}")
    print(f"      Macro-F1     : {metrics['macro_f1']:.4f}")
    print(f"      Weighted-F1  : {metrics['weighted_f1']:.4f}")
    print(f"      Majority     : {baselines['majority_baseline']:.4f} "
          f"(class '{baselines['majority_class']}')")
    print(f"      Random       : {baselines['random_baseline']:.4f}")
    if metrics["zero_support_predicates"]:
        print(f"      Zero-support predicates ({metrics['n_zero_support_predicates']}): "
              f"{', '.join(metrics['zero_support_predicates'])}")

    # --- 6. write result --------------------------------------------------
    print("\n[6/6] Writing result JSON ...")
    elapsed = time.time() - t_start
    if args.output_name:
        ckpt_name = args.output_name
    elif cfg["model_type"] == "transformer":
        ckpt_name = "transformer_e1"
    else:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_info["path"]))[0]
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, f"{ckpt_name}.json")

    # Output safety: never silently clobber another checkpoint's result.
    if os.path.isfile(out_path) and not args.force:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                prev_sha = json.load(f).get("checkpoint", {}).get("sha256")
        except Exception:
            prev_sha = None
        if prev_sha and prev_sha != ckpt_info["sha256"]:
            raise SystemExit(
                f"[E0] Refusing to overwrite {out_path}: it belongs to a "
                f"different checkpoint (sha256 {prev_sha[:16]}...). "
                "Use --output-name or --force."
            )

    report = {
        "experiment": "E0",
        "description": (
            "Image-disjoint ground-truth Visual Genome relation evaluation. "
            "Raw logits only: no temperature, priors, calibration, adaptive "
            "margins, heuristic correction or gating. PAD/UNK excluded from "
            "the candidate set and from scoring."
        ),
        "checkpoint": ckpt_info,
        "split": {
            "manifest_path": os.path.abspath(manifest_path).replace("\\", "/"),
            "seed": args.seed,
            "split_unit": "image_id",
            "fractions": manifest["fractions"],
            **counts,
            "image_overlap_check": overlap,
        },
        "data": data_stats,
        "metrics": metrics,
        "baselines": baselines,
        "per_predicate": per_predicate,
        "confusion_matrix": confusion,
        "runtime": {
            "device": str(device),
            "elapsed_seconds": round(elapsed, 2),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"      wrote {out_path}")
    print(f"\n[E0] Complete in {elapsed:.1f}s")


if __name__ == "__main__":
    configure_safe_stdio()
    main()

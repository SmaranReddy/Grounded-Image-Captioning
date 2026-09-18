"""Caption experiment: how should a predicted relation reach the caption?

Same images, same BLIP checkpoint, same preprocessing, same decoding
parameters across every arm.

Experiment 1 (CAPTION_EXPERIMENTS.md, CLOSED) varies only the text prefix BLIP
continues from:

    baseline      "a photo of"
    grounded      "a photo of a person riding a bicycle"   (relation >= threshold)
    objects_only  "a photo of a person and a bicycle"      (control)

Its result: the relation prefix increases object hallucination and the
objects-only control matches it on every metric, so the predicate contributes
nothing. A prefix is a mandatory assertion - the system cannot decline a wrong
prediction.

Experiment 2 (CAPTION_RERANKING.md) generates candidates from all three
prefixes and CHOOSES between them with a deterministic, evidence-based score:

    object_reranked    score without the relation term
    relation_reranked  score with a confidence-weighted relation term

Stages (each resumable; see the runbooks for the full procedure)
----------------------------------------------------------------
    preflight          environment, data, split integrity, checkpoint, weights
    select-checkpoint  pick the geometry+CLIP+union seed by VALIDATION top-1
    detect             YOLO -> CLIP verification -> relation model  (relations.jsonl)
    generate --arm A   BLIP caption for one prefix arm              (captions_A.jsonl)
    candidates         multi-candidate BLIP + uniform scoring       (candidates.jsonl)
    tune-rerank        pick the rerank weights on VALIDATION, then LOCK them
    rerank --arm A     choose one candidate per image with the locked weights
    smoke              all of the above on ~5 images, with PASS/FAIL checks
    examples           side-by-side qualitative examples (not evidence)
    report             print the scored results

Scoring is `python hallucination_eval.py --run-dir <run dir>`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

from build_caption_eval_set import FROZEN_SPLIT_IDS_SHA256, split_ids_sha256  # noqa: E402
from utils.caption_rerank import SOURCE_ORDER  # noqa: E402
from utils.caption_relations import (  # noqa: E402
    DEFAULT_CHECKPOINT_ROOT,
    RELATION_CONFIDENCE_THRESHOLD,
)

DEFAULT_EVAL_SET = "splits/caption_eval_test_250.json"
DEFAULT_RESULTS_ROOT = "results_caption"
DEFAULT_SELECTION = f"{DEFAULT_RESULTS_ROOT}/relation_checkpoint_selection.json"
DEFAULT_RUN_DIR = f"{DEFAULT_RESULTS_ROOT}/main"
DEFAULT_SPLIT_MANIFEST = "splits/e0_image_split.json"
DEFAULT_WEIGHTS = f"{DEFAULT_RESULTS_ROOT}/rerank_weights.json"
DEFAULT_NUM_CANDIDATES = 4        # == EXPERIMENT_GENERATION_CONFIG["num_beams"]
DEFAULT_SCORE_BATCH_SIZE = 12
DEFAULT_VAL_EVAL_SET = "splits/caption_eval_val_200.json"
DEFAULT_VAL_RUN_DIR = f"{DEFAULT_RESULTS_ROOT}/val_tuning"
YOLO_CONF = 0.5            # grounded_caption_pipeline.py's detection threshold
YOLO_TOP_K = 10
PREFIX_ARMS = ("baseline", "grounded", "objects_only")
RERANK_ARMS = ("object_reranked", "relation_reranked")
ARMS = PREFIX_ARMS
SYNTHETIC_RELATION = {"subject": "person", "predicate": "riding", "object": "bicycle",
                      "confidence": 1.0}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git() -> Dict:
    from run_visual_experiment import git_commit
    return git_commit()


def _versions() -> Dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("torch", "transformers", "ultralytics", "PIL", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    try:
        import torch
        out["cuda"] = torch.cuda.is_available()
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return out


def _set_seed(seed: int) -> None:
    from utils.seed import set_seed
    set_seed(seed)


def _device(name: Optional[str]):
    import torch
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _blip_dtype(name: str):
    import torch
    return {"float32": torch.float32, "float16": torch.float16}[name]


def _write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _append_jsonl(fh, obj) -> None:
    fh.write(json.dumps(obj) + "\n")
    fh.flush()


def _done_ids(path: Path) -> set:
    if not path.is_file():
        return set()
    from utils.caption_experiment_eval import read_jsonl
    return {str(r["image_id"]) for r in read_jsonl(path)}


def load_eval_set(path: str, allow_small: bool, allow_val: bool = False) -> Dict:
    if not Path(path).is_file():
        raise SystemExit(f"evaluation set {path} not found - run build_caption_eval_set.py")
    es = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = es["meta"]
    if meta.get("split_ids_sha256") != FROZEN_SPLIT_IDS_SHA256:
        raise SystemExit(f"{path} was not built from the frozen E0 split")
    if meta.get("split") != "test" and not (allow_small or allow_val):
        raise SystemExit(f"{path} draws from split {meta.get('split')!r}, not test")
    if meta.get("split") == "val" and not (allow_small or allow_val):
        raise SystemExit(f"{path} is a VALIDATION set; pass --tuning to use it")
    if meta.get("status") != "OK" and not allow_small:
        raise SystemExit(f"{path} has status {meta.get('status')} "
                         f"({meta.get('final')} usable images) - STOP")
    return es


def load_selection(path: str) -> Dict:
    if not Path(path).is_file():
        raise SystemExit(f"{path} not found - run `python run_caption_experiment.py "
                         "select-checkpoint` first")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_run_config(run_dir: Path) -> Dict:
    p = run_dir / "run_config.json"
    if not p.is_file():
        raise SystemExit(f"{p} not found - run the `detect` stage first")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def cmd_preflight(args) -> int:
    ok = True

    def line(name, good, detail="", optional=False):
        nonlocal ok
        mark = " OK " if good else (" OPT" if optional else "MISS")
        if not good and not optional:
            ok = False
        print(f"  [{mark}] {name:<34} {detail}")

    print("=" * 78)
    print("CAPTION EXPERIMENT PREFLIGHT")
    print("=" * 78)
    v = _versions()
    line("python", True, v["python"])
    for mod in ("torch", "transformers", "ultralytics", "PIL", "numpy"):
        line(mod, v.get(mod) is not None, str(v.get(mod)))
    line("CUDA GPU", bool(v.get("cuda")), str(v.get("gpu") or "none - stages will run on CPU (slow)"),
         optional=True)
    if v.get("cuda"):
        import torch
        mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        line("GPU memory", mem >= 3.5, f"{mem:.1f} GiB")

    line("yolo11m.pt", (PROJ_ROOT / "yolo11m.pt").is_file(),
         "present" if (PROJ_ROOT / "yolo11m.pt").is_file() else "ultralytics will download it",
         optional=True)
    try:
        from huggingface_hub import try_to_load_from_cache
        for repo in ("Salesforce/blip-image-captioning-base", "openai/clip-vit-base-patch32"):
            cached = isinstance(try_to_load_from_cache(repo, "config.json"), str)
            line(repo, cached, "cached" if cached else "will download on first use", optional=True)
    except Exception:
        pass

    vg = Path(args.vg_root)
    for name, optional in (("objects.json", False), ("relationships.json", False),
                           ("image_data.json", False)):
        p = vg / name
        line(f"VG {name}", p.is_file(), f"{p.stat().st_size / 2**20:,.0f} MB" if p.is_file()
             else "python download_vg.py --objects", optional)

    manifest = Path(args.split_manifest)
    if manifest.is_file():
        sha = split_ids_sha256(json.loads(manifest.read_text(encoding="utf-8")))
        line("frozen split unchanged", sha == FROZEN_SPLIT_IDS_SHA256, sha[:16] + "...")
    else:
        line("frozen split", False, f"{manifest} missing")

    if Path(args.eval_set).is_file():
        es = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
        m = es["meta"]
        line("caption eval set", m.get("status") == "OK",
             f"status {m.get('status')}: requested {m.get('requested')}, usable {m.get('usable')}, "
             f"missing {m.get('missing')}, final {m.get('final')}")
    else:
        line("caption eval set", False, f"{args.eval_set} missing - build_caption_eval_set.py")

    if Path(args.selection).is_file():
        sel = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        ck = sel["selected"]["checkpoint_dir"]
        try:
            from utils.caption_relations import load_relation_bundle
            import torch
            b = load_relation_bundle(ck, device=torch.device("cpu"))
            line("relation checkpoint", True,
                 f"{ck} input_dim={b.config['input_dim']} geo={b.config['geo_dim']} "
                 f"clip={b.clip_dim} union={b.union_dim}")
        except Exception as exc:
            line("relation checkpoint", False, f"{ck}: {exc}")
    else:
        line("relation checkpoint selection", False,
             f"{args.selection} missing - run select-checkpoint")

    if Path(args.val_eval_set).is_file():
        vm = json.loads(Path(args.val_eval_set).read_text(encoding="utf-8"))["meta"]
        line("validation caption set (tuning)", vm.get("split") == "val",
             f"split {vm.get('split')}, {vm.get('final')} images, status {vm.get('status')}",
             optional=True)
    else:
        line("validation caption set (tuning)", False,
             f"{args.val_eval_set} missing - build_caption_eval_set.py --split val "
             "(only needed for the reranking arms)", optional=True)

    if Path(args.weights).is_file():
        w = json.loads(Path(args.weights).read_text(encoding="utf-8"))
        sel = w.get("selected_on") or {}
        line("locked rerank weights", sel.get("split") == "val",
             f"{w.get('weights')} selected on {sel.get('n_images')} {sel.get('split')} images",
             optional=True)
    else:
        line("locked rerank weights", False,
             f"{args.weights} missing - run tune-rerank on the validation run "
             "(only needed for the reranking arms)", optional=True)

    free = shutil.disk_usage(PROJ_ROOT).free / 2 ** 30
    line("free disk", free > 2, f"{free:.1f} GiB")
    print("-" * 78)
    print("PREFLIGHT: READY" if ok else "PREFLIGHT: NOT READY")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# checkpoint selection
# ---------------------------------------------------------------------------

def cmd_select_checkpoint(args) -> int:
    import torch
    from utils.caption_relations import load_relation_bundle, select_checkpoint

    sel = select_checkpoint(args.checkpoint_root)
    chosen = sel["selected"]
    print("=" * 78)
    print("RELATION CHECKPOINT SELECTION (validation metrics only; test metrics not read)")
    print("=" * 78)
    print(f"  rule: {sel['rule']}")
    print(f"  {'seed':>6}{'val top-1':>12}{'val macro-F1':>15}{'best epoch':>12}")
    for c in sel["candidates"]:
        f1 = "n/a" if c["val_macro_f1"] is None else f"{c['val_macro_f1']:.4f}"
        mark = "  <- selected" if c["seed"] == chosen["seed"] else ""
        print(f"  {c['seed']:>6}{c['val_top1']:>12.4f}{f1:>15}{str(c['best_epoch']):>12}{mark}")

    bundle = load_relation_bundle(chosen["checkpoint_dir"], device=torch.device("cpu"))
    cfg = bundle.config
    sel["verified_config"] = cfg
    sel["feature_blocks"] = [[n, w] for n, w in bundle.model.feature_blocks()]
    sel["weights_sha256"] = _sha256(Path(chosen["checkpoint_dir"]) / "relation_mlp.pt")
    sel["pred_vocab"] = [bundle.pred_vocab.token(i) for i in range(len(bundle.pred_vocab))]
    sel["created_utc"] = _now()
    sel["git"] = _git()
    print(f"\n  loaded {chosen['checkpoint_dir']}: input_dim={cfg['input_dim']} "
          f"blocks={sel['feature_blocks']}")
    print(f"  weights sha256 {sel['weights_sha256'][:16]}...")

    out = Path(args.output)
    if out.is_file() and not args.force:
        old = json.loads(out.read_text(encoding="utf-8"))
        if old.get("weights_sha256") != sel["weights_sha256"]:
            raise SystemExit(f"{out} already records a DIFFERENT checkpoint "
                             f"({old['selected']['checkpoint_dir']}). Pass --force to replace.")
        print(f"  {out} already records this checkpoint; left unchanged.")
        return 0
    _write_json(out, sel)
    print(f"  wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# detect: YOLO -> CLIP verification -> relation model
# ---------------------------------------------------------------------------

def _image_ids(es: Dict, limit: Optional[int], image_ids: Optional[List[str]] = None) -> List[str]:
    ids = [str(i) for i in es["usable_ids"]]
    if image_ids:
        unknown = [i for i in image_ids if i not in set(ids)]
        if unknown:
            raise SystemExit(f"image ids not usable in the eval set: {unknown}")
        ids = [i for i in ids if i in set(image_ids)]
    return ids[:limit] if limit else ids


def run_detect(run_dir: Path, es: Dict, eval_set_path: str, selection: Dict, ids: List[str],
               threshold: float, device_name: Optional[str], checkpoint_dir: Optional[str] = None,
               seed: int = 42) -> List[Dict]:
    import torch
    from PIL import Image
    from relation_prediction.clip_extractor import CLIPExtractor
    from utils.caption_relations import (build_pair_batch, eligible_detections,
                                         load_relation_bundle, predict_pairs,
                                         relation_decision, select_relation)
    from utils.detection_verifier import verify_detections
    from utils.yolo_detector import format_detections, load_model

    run_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    device = _device(device_name)
    ck_dir = checkpoint_dir or selection["selected"]["checkpoint_dir"]
    weights_sha = _sha256(Path(ck_dir) / "relation_mlp.pt")
    override = weights_sha != selection.get("weights_sha256")

    cfg_path = run_dir / "run_config.json"
    run_cfg = {
        "created_utc": _now(),
        "git": _git(),
        "versions": _versions(),
        "eval_set": {"path": eval_set_path.replace("\\", "/"),
                     "usable_ids_sha256": es["meta"]["usable_ids_sha256"],
                     "split": es["meta"]["split"], "status": es["meta"]["status"],
                     "final": es["meta"]["final"]},
        "image_ids": ids,
        "relation_threshold": threshold,
        "relation_threshold_is_default": threshold == RELATION_CONFIDENCE_THRESHOLD,
        "relation_checkpoint": {
            "checkpoint_dir": ck_dir, "weights_sha256": weights_sha,
            "matches_selection_file": not override,
            "selection_rule_applied": selection.get("rule"),
            "selected_seed": selection["selected"]["seed"],
        },
        "detection": {"model": "yolo11m.pt", "conf_thres": YOLO_CONF, "top_k": YOLO_TOP_K,
                      "verification": "utils.detection_verifier.verify_detections"},
        "relation_policy": ("argmax over valid predicates (as eval_gt_relations.py); "
                            "confidence = softmax prob at T=1; semantic filter = "
                            "predict._is_extreme_nonsense; select max confidence; "
                            "inject iff confidence >= threshold"),
        "seed": seed,
    }
    if cfg_path.is_file():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        for key in ("image_ids", "relation_threshold"):
            if old.get(key) != run_cfg[key]:
                raise SystemExit(f"{cfg_path} records a different {key}; use a new --run-dir")
        if old["eval_set"]["usable_ids_sha256"] != run_cfg["eval_set"]["usable_ids_sha256"] \
                or old["relation_checkpoint"]["weights_sha256"] != weights_sha:
            raise SystemExit(f"{cfg_path} was produced with a different eval set or checkpoint")
        run_cfg = old
    else:
        _write_json(cfg_path, run_cfg)
    if override:
        print(f"!! WARNING: {ck_dir} is NOT the checkpoint recorded in the selection file")

    out_path = run_dir / "relations.jsonl"
    done = _done_ids(out_path)
    todo = [i for i in ids if i not in done]
    print(f"[detect] {len(ids)} images, {len(done)} already done, {len(todo)} to process "
          f"on {device}")
    if not todo:
        return []

    bundle = load_relation_bundle(ck_dir, device=device)
    print(f"[detect] relation model: input_dim={bundle.config['input_dim']} "
          f"blocks={bundle.model.feature_blocks()}")
    extractor = CLIPExtractor(device)
    yolo = load_model(str(device) if device.type == "cpu" else "cuda")

    written = []
    with open(out_path, "a", encoding="utf-8") as fh:
        for n, iid in enumerate(todo, 1):
            entry = es["images"][iid]
            t0 = time.time()
            image = Image.open(entry["file"]).convert("RGB")
            raw = format_detections(yolo(image, verbose=False), conf_thres=YOLO_CONF, top_k=YOLO_TOP_K)
            t_yolo = time.time()
            verified = verify_detections(copy.deepcopy(raw), image, debug=False)
            t_verify = time.time()
            eligible, dropped = eligible_detections(verified, bundle.label_vocab, image.size)
            batch = build_pair_batch(image, eligible, bundle, extractor.encode_crops)
            preds = predict_pairs(bundle, batch)
            selected, annotated = select_relation(eligible, preds)
            decision = relation_decision(selected, len(eligible), len(preds), threshold)
            rec = {
                "image_id": iid,
                "file": entry["file"],
                "image_size": list(image.size),
                "raw_detections": [{"label": d["label"], "box": [round(v, 2) for v in d["box"]],
                                    "score": round(d["score"], 4)} for d in raw],
                "verified_detections": [{"label": d["label"], "box": [round(v, 2) for v in d["box"]],
                                         "score": round(d["score"], 4),
                                         "clip_similarity": d.get("clip_similarity"),
                                         "verification_score": d.get("verification_score")}
                                        for d in verified],
                "eligible_detections": eligible,
                "dropped_detections": dropped,
                "pair_predictions": annotated,
                "selected_relation": selected,
                "decision": decision,
                "threshold": threshold,
                "n_object_crops": batch.get("n_object_crops", 0),
                "n_union_crops": batch.get("n_union_crops", 0),
                "seconds": {"yolo": round(t_yolo - t0, 3), "verify": round(t_verify - t_yolo, 3),
                            "relations": round(time.time() - t_verify, 3)},
            }
            _append_jsonl(fh, rec)
            written.append(rec)
            rel = (f"{selected['subject']} {selected['predicate']} {selected['object']} "
                   f"({selected['confidence']:.2f})" if selected else "-")
            print(f"[detect] {n}/{len(todo)} {iid}: raw {len(raw)} verified {len(verified)} "
                  f"eligible {len(eligible)} pairs {len(preds)} relation {rel} "
                  f"{'USED' if decision['use_relation'] else decision['fallback_reason']}")
    return written


def cmd_detect(args) -> int:
    es = load_eval_set(args.eval_set, args.allow_small, allow_val=getattr(args, "tuning", False))
    selection = load_selection(args.selection)
    ids = _image_ids(es, args.limit)
    run_detect(Path(args.run_dir), es, args.eval_set, selection, ids, args.relation_threshold,
               args.device, args.checkpoint_dir, args.seed)
    return 0


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def run_generate(run_dir: Path, arm: str, blip_dtype: str, seed: int = 42) -> List[Dict]:
    from PIL import Image
    from utils.blip_captioner import (BASELINE_PREFIX, blip_model_info, build_blip_prefix,
                                      build_objects_only_prefix, generate_blip_from_prefix)
    from utils.caption_experiment_eval import keyed, read_jsonl

    cfg = load_run_config(run_dir)
    ids = [str(i) for i in cfg["image_ids"]]
    threshold = float(cfg["relation_threshold"])
    es = json.loads(Path(cfg["eval_set"]["path"]).read_text(encoding="utf-8"))
    if es["meta"]["usable_ids_sha256"] != cfg["eval_set"]["usable_ids_sha256"]:
        raise SystemExit("evaluation set changed since detect ran; refusing to generate")
    files = {i: es["images"][i]["file"] for i in ids}
    relations = {}
    if arm != "baseline":
        rel_path = run_dir / "relations.jsonl"
        if not rel_path.is_file():
            raise SystemExit(f"{rel_path} missing - run detect first")
        relations = keyed(read_jsonl(rel_path), "relations.jsonl")
        missing = [i for i in ids if i not in relations]
        if missing:
            raise SystemExit(f"relations.jsonl is incomplete ({len(missing)} images missing); "
                             "finish detect first")

    out_path = run_dir / f"captions_{arm}.jsonl"
    meta_path = run_dir / f"captions_{arm}.meta.json"
    from utils.blip_captioner import EXPERIMENT_GENERATION_CONFIG
    meta = {"arm": arm, "blip_dtype": blip_dtype, "generation_config": EXPERIMENT_GENERATION_CONFIG,
            "threshold": threshold, "seed": seed}
    if meta_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in meta} != meta:
            raise SystemExit(f"{meta_path} records different generation settings; "
                             "refusing to mix them in one arm")
    done = _done_ids(out_path)
    todo = [i for i in ids if i not in done]
    print(f"[generate:{arm}] {len(ids)} images, {len(done)} done, {len(todo)} to caption")
    if not todo:
        return []

    _set_seed(seed)
    dtype = _blip_dtype(blip_dtype)
    written = []
    with open(out_path, "a", encoding="utf-8") as fh:
        for n, iid in enumerate(todo, 1):
            t0 = time.time()
            image = Image.open(files[iid]).convert("RGB")
            selected = relations.get(iid, {}).get("selected_relation") if arm != "baseline" else None
            decision = relations.get(iid, {}).get("decision", {}) if arm != "baseline" else {}
            use = bool(decision.get("use_relation"))

            def prefix_for(rel):
                if arm == "grounded":
                    prefix, used = build_blip_prefix([], [rel], min_confidence=0.0)
                    if not used:
                        raise RuntimeError(f"{iid}: relation {rel} dropped by build_blip_prefix")
                    return prefix
                return build_objects_only_prefix(rel)

            rec = {"image_id": iid, "arm": arm, "relation_used": use,
                   "fallback_reason": None if (arm == "baseline" or use) else decision.get("fallback_reason"),
                   "relation": ({k: selected[k] for k in ("subject", "predicate", "object", "confidence")}
                                if selected else None)}
            relation_out = None
            if arm != "baseline" and selected is not None:
                relation_out = generate_blip_from_prefix(image, prefix_for(selected), dtype=dtype)
                rec["relation_prefix"] = relation_out["prefix"]
                rec["relation_caption"] = relation_out["caption"]
            if arm == "baseline" or not use:
                out = generate_blip_from_prefix(image, BASELINE_PREFIX, dtype=dtype)
            else:
                out = relation_out
            rec.update({"prefix": out["prefix"], "caption": out["caption"],
                        "input_ids": out["input_ids"], "prefix_echoed": out["prefix_echoed"],
                        "seconds": round(time.time() - t0, 3)})
            _append_jsonl(fh, rec)
            written.append(rec)
            print(f"[generate:{arm}] {n}/{len(todo)} {iid}: {out['caption']!r}"
                  + (f"  [relation: {rec['relation']['predicate']} "
                     f"{rec['relation']['confidence']:.2f}]" if use else ""))
    meta["blip"] = blip_model_info()
    _write_json(meta_path, meta)
    return written


def cmd_generate(args) -> int:
    run_generate(Path(args.run_dir), args.arm, args.blip_dtype, args.seed)
    return 0


# ---------------------------------------------------------------------------
# candidates: multi-candidate BLIP generation (the reranking arms' generator)
# ---------------------------------------------------------------------------

def run_candidates(run_dir: Path, num_candidates: int, batch_size: int, blip_dtype: str,
                   seed: int = 42) -> List[Dict]:
    """Generate and uniformly score the candidate pool for every image.

    One pass over the images produces the pool BOTH reranking arms choose
    from, so the two arms differ in nothing but their scoring weights. The
    stage is resumable and refuses to mix generation settings inside one run
    directory, exactly like `generate`.
    """
    from PIL import Image
    from utils.blip_captioner import EXPERIMENT_GENERATION_CONFIG, blip_model_info
    from utils.caption_candidates import generate_and_score
    from utils.caption_experiment_eval import keyed, read_jsonl

    cfg = load_run_config(run_dir)
    ids = [str(i) for i in cfg["image_ids"]]
    rel_path = run_dir / "relations.jsonl"
    if not rel_path.is_file():
        raise SystemExit(f"{rel_path} missing - run detect first")
    relations = keyed(read_jsonl(rel_path), "relations.jsonl")
    missing = [i for i in ids if i not in relations]
    if missing:
        raise SystemExit(f"relations.jsonl is incomplete ({len(missing)} images missing); "
                         "finish detect first")
    es = json.loads(Path(cfg["eval_set"]["path"]).read_text(encoding="utf-8"))
    if es["meta"]["usable_ids_sha256"] != cfg["eval_set"]["usable_ids_sha256"]:
        raise SystemExit("evaluation set changed since detect ran; refusing to generate")
    files = {i: es["images"][i]["file"] for i in ids}

    out_path = run_dir / "candidates.jsonl"
    meta_path = run_dir / "candidates.meta.json"
    meta = {"num_candidates": num_candidates, "blip_dtype": blip_dtype,
            "generation_config": EXPERIMENT_GENERATION_CONFIG,
            "score_batch_size": batch_size, "seed": seed,
            "candidate_sources": list(SOURCE_ORDER),
            "policy": ("beams of the baseline prefix always; of the objects-only and "
                       "relation prefixes whenever a relation was selected, at ANY "
                       "confidence - the confidence is scored, not gated"),
            "lm_score": ("mean per-token log P(caption | image), teacher-forced with "
                         "identical conditioning for every candidate")}
    if meta_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        differing = [k for k in ("num_candidates", "blip_dtype", "generation_config", "seed")
                     if old.get(k) != meta[k]]
        if differing:
            raise SystemExit(f"{meta_path} records different settings {differing}; "
                             "refusing to mix them in one run directory")

    done = _done_ids(out_path)
    todo = [i for i in ids if i not in done]
    print(f"[candidates] {len(ids)} images, {len(done)} done, {len(todo)} to generate "
          f"(K={num_candidates} per prefix)")
    if not todo:
        return []

    _set_seed(seed)
    dtype = _blip_dtype(blip_dtype)
    written = []
    with open(out_path, "a", encoding="utf-8") as fh:
        for n, iid in enumerate(todo, 1):
            t0 = time.time()
            image = Image.open(files[iid]).convert("RGB")
            selected = relations[iid].get("selected_relation")
            result = generate_and_score(image, selected, num_candidates=num_candidates,
                                        batch_size=batch_size, dtype=dtype)
            rec = {"image_id": iid, "prefixes": result["prefixes"],
                   "relation": ({k: selected[k] for k in
                                 ("subject", "predicate", "object", "confidence")}
                                if selected else None),
                   "candidates": result["candidates"],
                   "seconds": round(time.time() - t0, 3)}
            _append_jsonl(fh, rec)
            written.append(rec)
            print(f"[candidates] {n}/{len(todo)} {iid}: {len(rec['candidates'])} candidates "
                  f"from {len(rec['prefixes'])} prefixes in {rec['seconds']:.1f}s")
    meta["blip"] = blip_model_info()
    _write_json(meta_path, meta)
    return written


def cmd_candidates(args) -> int:
    run_candidates(Path(args.run_dir), args.num_candidates, args.batch_size,
                   args.blip_dtype, args.seed)
    return 0


# ---------------------------------------------------------------------------
# rerank: choose one candidate per image (pure CPU, no model, no ground truth)
# ---------------------------------------------------------------------------

def load_weights(path: str) -> Dict:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"{p} not found - run `tune-rerank` on the VALIDATION run first. "
                         "Weights must never be chosen on the frozen test set.")
    return json.loads(p.read_text(encoding="utf-8"))


def run_rerank(run_dir: Path, arm: str, weights_file: Dict, weights_path: str) -> List[Dict]:
    """Select one caption per image from the cached pool.

    This stage opens run_config.json, relations.jsonl and candidates.jsonl and
    nothing else. In particular it never opens the evaluation manifest, which
    is the only file in the run that carries human annotations.
    """
    from utils.caption_experiment_eval import keyed, read_jsonl
    from utils.caption_rerank import (RerankWeights, evidence_objects, first_candidate,
                                      rerank)

    if arm not in RERANK_ARMS:
        raise SystemExit(f"{arm} is not a reranking arm ({RERANK_ARMS})")
    cfg = load_run_config(run_dir)
    ids = [str(i) for i in cfg["image_ids"]]
    relations = keyed(read_jsonl(run_dir / "relations.jsonl"), "relations.jsonl")
    cand_path = run_dir / "candidates.jsonl"
    if not cand_path.is_file():
        raise SystemExit(f"{cand_path} missing - run the candidates stage first")
    pools = keyed(read_jsonl(cand_path), "candidates.jsonl")
    missing = [i for i in ids if i not in pools]
    if missing:
        raise SystemExit(f"candidates.jsonl is incomplete ({len(missing)} images missing)")

    full = RerankWeights.from_dict(weights_file["weights"][arm])
    if arm == "object_reranked" and full.w_rel:
        raise SystemExit("object_reranked must have w_rel = 0 in the weights file")
    weights = full.without_relation() if arm == "object_reranked" else full
    contrast = full.without_relation() if arm == "relation_reranked" else None

    meta_path = run_dir / f"captions_{arm}.meta.json"
    meta = {"arm": arm, "weights": weights.as_dict(), "weights_sha256": weights.sha256(),
            "weights_file": weights_path.replace("\\", "/"),
            "weights_file_sha256": _sha256(weights_path),
            "weights_selected_on": weights_file.get("selected_on"),
            "objective": weights_file.get("objective")}
    if meta_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("weights_sha256") != meta["weights_sha256"]:
            raise SystemExit(f"{meta_path} records DIFFERENT weights ({old.get('weights')}); "
                             "use a new --run-dir rather than re-selecting on this one")

    out_path = run_dir / f"captions_{arm}.jsonl"
    written = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for iid in ids:
            pool = pools[iid]["candidates"]
            selected = relations[iid].get("selected_relation")
            evidence = evidence_objects(relations[iid].get("verified_detections", []))
            choice = rerank(pool, evidence, selected, weights)
            first = first_candidate(pool)
            alt = (rerank(pool, evidence, selected, contrast)["caption"]
                   if contrast is not None else None)
            prefix = pools[iid]["prefixes"][choice["source"]]
            rec = {
                "image_id": iid, "arm": arm,
                "caption": choice["caption"],
                "prefix": prefix,
                "source": choice["source"], "beam_rank": choice["beam_rank"],
                "score": choice["score"], "terms": choice["terms"],
                "relation": pools[iid].get("relation"),
                "relation_used": bool(choice["relation_stated"]),
                "evidence_objects": sorted(evidence),
                "n_supported": choice["n_supported"],
                "n_unsupported": choice["n_unsupported"],
                "n_candidates": len(pool),
                "n_candidates_scored": choice["n_candidates_scored"],
                "n_candidates_dropped": choice["n_candidates_dropped"],
                "dropped": choice["dropped"],
                "first_candidate": None if first is None else first["text"],
                "changed_from_first_candidate":
                    first is not None and choice["caption"] != first["text"],
                "changed_by_relation_term": alt is not None and alt != choice["caption"],
                "prefix_echoed": choice["caption"].lower().startswith(prefix.lower()),
                "scores": [{k: r[k] for k in ("text", "source", "beam_rank", "score")}
                           for r in choice["ranking"]],
            }
            _append_jsonl(fh, rec)
            written.append(rec)
    _write_json(meta_path, meta)
    changed = sum(r["changed_from_first_candidate"] for r in written)
    stated = sum(r["relation_used"] for r in written)
    print(f"[rerank:{arm}] {len(written)} images, weights {weights.as_dict()}; "
          f"{changed} captions differ from BLIP's first candidate; "
          f"{stated} state the predicted relation -> {out_path}")
    return written


def cmd_rerank(args) -> int:
    weights_file = load_weights(args.weights)
    for arm in (RERANK_ARMS if args.arm == "all" else (args.arm,)):
        run_rerank(Path(args.run_dir), arm, weights_file, args.weights)
    return 0


# ---------------------------------------------------------------------------
# tune-rerank: choose the weights on the VALIDATION split, then lock them
# ---------------------------------------------------------------------------

# Pre-declared grid and objective (fixed before any weight was scored).
W_OBJ_GRID = (0.0, 0.1, 0.25, 0.5, 1.0)
W_HALL_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
W_REL_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
TUNING_OBJECTIVE = (
    "maximise POPE-adversarial F1 on the validation images; ties -> lower CHAIR_i; "
    "ties -> smaller L1 norm of the weights (prefer the simpler scorer); "
    "ties -> lexicographically smallest (w_obj, w_hall, w_rel). POPE F1 is used "
    "because it is symmetric: asserting an absent object costs a false positive "
    "and staying silent about a present one costs a false negative, so a caption "
    "cannot win by saying less - which CHAIR_i alone would reward."
)


def _tuning_candidate_pools(run_dir: Path, ids: List[str]):
    from utils.caption_experiment_eval import keyed, read_jsonl
    from utils.caption_rerank import evidence_objects

    relations = keyed(read_jsonl(run_dir / "relations.jsonl"), "relations.jsonl")
    pools = keyed(read_jsonl(run_dir / "candidates.jsonl"), "candidates.jsonl")
    missing = [i for i in ids if i not in pools or i not in relations]
    if missing:
        raise SystemExit(f"validation run is incomplete ({len(missing)} images missing)")
    return [
        {"image_id": i,
         "candidates": pools[i]["candidates"],
         "relation": relations[i].get("selected_relation"),
         "evidence": evidence_objects(relations[i].get("verified_detections", []))}
        for i in ids
    ]


def _score_weights(items, gt, probes, weights):
    """Validation metrics for one weight vector. Ground truth is used HERE only -
    to score a candidate selection that was made without it."""
    from utils.caption_hallucination import aggregate_chair, caption_object_record, score_pope
    from utils.caption_rerank import rerank

    captions = {it["image_id"]: rerank(it["candidates"], it["evidence"], it["relation"],
                                       weights)["caption"] for it in items}
    records = [caption_object_record(captions[i], gt[i]) for i in captions]
    mentioned = {i: set(r["mentioned"]) for i, r in zip(captions, records)}
    chair = aggregate_chair(records)
    pope = score_pope(probes["adversarial"], mentioned)
    return {"weights": weights.as_dict(), "pope_adversarial_f1": pope["f1"],
            "chair_i": chair["chair_i"], "chair_s": chair["chair_s"],
            "object_recall": chair["object_recall"],
            "mean_mentioned_objects": chair["mean_mentioned_objects"],
            "mean_hallucinated_objects": chair["mean_hallucinated_objects"],
            "pope_adversarial_accuracy": pope["accuracy"],
            "l1": sum(abs(v) for v in weights.as_dict().values())}


def _best(rows):
    return sorted(rows, key=lambda r: (-round(r["pope_adversarial_f1"], 9),
                                       round(r["chair_i"], 9), round(r["l1"], 9),
                                       r["weights"]["w_obj"], r["weights"]["w_hall"],
                                       r["weights"]["w_rel"]))[0]


def cmd_tune_rerank(args) -> int:
    from utils.caption_hallucination import build_pope_probes
    from utils.caption_rerank import RerankWeights

    run_dir = Path(args.run_dir)
    cfg = load_run_config(run_dir)
    if cfg["eval_set"].get("split") == "test":
        raise SystemExit("refusing to tune on a TEST run directory. Weights are selected "
                         "on the validation split only (see CAPTION_RERANKING.md).")
    es = json.loads(Path(cfg["eval_set"]["path"]).read_text(encoding="utf-8"))
    ids = [str(i) for i in cfg["image_ids"]]
    gt = {i: set(es["images"][i]["objects"]) for i in ids}
    probes = build_pope_probes(gt, seed=args.seed)
    items = _tuning_candidate_pools(run_dir, ids)

    print("=" * 78)
    print(f"RERANK WEIGHT SELECTION on {len(ids)} VALIDATION images ({cfg['eval_set']['path']})")
    print("=" * 78)
    print(f"  objective: {TUNING_OBJECTIVE}")

    object_grid = [_score_weights(items, gt, probes, RerankWeights(o, h, 0.0))
                   for o in W_OBJ_GRID for h in W_HALL_GRID]
    object_best = _best(object_grid)
    ow = RerankWeights.from_dict(object_best["weights"])
    print(f"\n  object_reranked  -> w_obj={ow.w_obj} w_hall={ow.w_hall} "
          f"(POPE-adv F1 {object_best['pope_adversarial_f1']:.4f}, "
          f"CHAIR_i {object_best['chair_i']:.4f})")

    # The relation arm inherits the object weights and tunes ONE parameter, so
    # the two arms differ by exactly the relation term.
    relation_grid = [_score_weights(items, gt, probes, RerankWeights(ow.w_obj, ow.w_hall, r))
                     for r in W_REL_GRID]
    relation_best = _best(relation_grid)
    rw = RerankWeights.from_dict(relation_best["weights"])
    print(f"  relation_reranked-> w_rel={rw.w_rel} "
          f"(POPE-adv F1 {relation_best['pope_adversarial_f1']:.4f}, "
          f"CHAIR_i {relation_best['chair_i']:.4f})")

    out = {
        "created_utc": _now(),
        "git": _git(),
        "selected_on": {
            "run_dir": str(run_dir).replace("\\", "/"),
            "eval_set": cfg["eval_set"],
            "n_images": len(ids),
            "split": cfg["eval_set"].get("split"),
            "seed": args.seed,
        },
        "objective": TUNING_OBJECTIVE,
        "procedure": (
            "1. w_rel = 0: grid over (w_obj, w_hall) -> object_reranked weights. "
            "2. w_obj, w_hall fixed at that optimum: grid over w_rel -> "
            "relation_reranked weights. The two arms therefore differ by one "
            "parameter and one scoring term, nothing else."),
        "grid": {"w_obj": list(W_OBJ_GRID), "w_hall": list(W_HALL_GRID),
                 "w_rel": list(W_REL_GRID)},
        "weights": {"object_reranked": ow.as_dict(), "relation_reranked": rw.as_dict()},
        "weights_sha256": {"object_reranked": ow.sha256(), "relation_reranked": rw.sha256()},
        "validation_metrics": {"object_reranked": object_best,
                               "relation_reranked": relation_best},
        "object_grid": object_grid,
        "relation_grid": relation_grid,
        "candidates_meta": json.loads((run_dir / "candidates.meta.json").read_text("utf-8"))
        if (run_dir / "candidates.meta.json").is_file() else None,
    }
    path = Path(args.output)
    if path.is_file() and not args.force:
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("weights") != out["weights"]:
            raise SystemExit(f"{path} already LOCKS different weights {old.get('weights')}. "
                             "Re-tuning after the frozen test run would be post-hoc "
                             "optimisation; pass --force only if the test run has not "
                             "happened yet.")
        print(f"\n  {path} already locks these weights; left unchanged.")
        return 0
    _write_json(path, out)
    print(f"\n  LOCKED -> {path}")
    print("  Do not re-run this after the frozen test evaluation.")
    return 0


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def cmd_smoke(args) -> int:
    from PIL import Image
    from utils.blip_captioner import (BASELINE_PREFIX, build_blip_prefix,
                                      generate_blip_from_prefix)
    from utils.caption_experiment_eval import evaluate_run, expected_prefix, keyed, read_jsonl

    run_dir = Path(args.run_dir)
    marker = run_dir / ".smoke_run"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not marker.is_file():
            raise SystemExit(f"{run_dir} is not empty and was not created by `smoke`; "
                             "refusing to touch it. Choose another --run-dir.")
        if not args.force:
            raise SystemExit(f"{run_dir} holds a previous smoke run; pass --force to redo it")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text("created by run_caption_experiment.py smoke\n", encoding="utf-8")

    checks: List[Dict] = []

    def check(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})
        print(f"  [{status:^4}] {name}: {detail}")

    es = load_eval_set(args.eval_set, allow_small=True)
    ids = _image_ids(es, args.n)
    selection = load_selection(args.selection)
    print("=" * 78)
    print(f"SMOKE TEST on {len(ids)} images -> {run_dir}")
    print("=" * 78)

    try:
        import torch
        from utils.caption_relations import load_relation_bundle
        b = load_relation_bundle(args.checkpoint_dir or selection["selected"]["checkpoint_dir"],
                                 device=torch.device("cpu"))
        check("relation model loads", "PASS",
              f"input_dim={b.config['input_dim']} blocks={b.model.feature_blocks()}")
    except Exception as exc:
        check("relation model loads", "FAIL", repr(exc))
        return _finish_smoke(run_dir, checks)

    rel = run_detect(run_dir, es, args.eval_set, selection, ids, args.relation_threshold,
                     args.device, args.checkpoint_dir, args.seed)
    rel = keyed(read_jsonl(run_dir / "relations.jsonl"), "relations")
    per_img = {i: (len(rel[i]["raw_detections"]), len(rel[i]["verified_detections"]),
                   len(rel[i]["pair_predictions"])) for i in ids}
    check("YOLO detects objects",
          "PASS" if any(v[0] for v in per_img.values()) else "FAIL",
          "raw/verified/pairs per image " + str(per_img))
    check("CLIP verification runs",
          "PASS" if any(v[1] for v in per_img.values()) else "FAIL",
          f"{sum(v[1] for v in per_img.values())} verified of {sum(v[0] for v in per_img.values())} raw")
    n_sel = sum(1 for i in ids if rel[i]["selected_relation"])
    check("relations predicted",
          "PASS" if n_sel else "FAIL",
          f"{n_sel}/{len(ids)} images with a selected relation; confidences "
          + str([round(rel[i]["selected_relation"]["confidence"], 3) for i in ids
                 if rel[i]["selected_relation"]]))
    n_used = sum(1 for i in ids if rel[i]["decision"]["use_relation"])
    check(f"relations above threshold {args.relation_threshold}",
          "PASS" if n_used else "WARN",
          f"{n_used}/{len(ids)} images inject a relation"
          + ("" if n_used else " (not a failure on 5 images; try --n 20 to see one used)"))

    # synthetic, deterministic relation through the real BLIP
    first = Image.open(es["images"][ids[0]]["file"]).convert("RGB")
    dtype = _blip_dtype(args.blip_dtype)
    syn_prefix, used = build_blip_prefix([], [SYNTHETIC_RELATION], min_confidence=0.5)
    b1 = generate_blip_from_prefix(first, BASELINE_PREFIX, dtype=dtype)
    b2 = generate_blip_from_prefix(first, BASELINE_PREFIX, dtype=dtype)
    g = generate_blip_from_prefix(first, syn_prefix, dtype=dtype)
    import utils.blip_captioner as blip
    span = blip._processor.tokenizer(" a person riding a bicycle", add_special_tokens=False)["input_ids"]
    expected_ids = b1["input_ids"][:-1] + span + b1["input_ids"][-1:]
    ok = bool(used) and g["input_ids"] == expected_ids and g["prefix_echoed"]
    check("synthetic relation reaches BLIP", "PASS" if ok else "FAIL",
          f"prefix {syn_prefix!r}; baseline ids {b1['input_ids']} -> grounded ids "
          f"{g['input_ids']} (expected {expected_ids}); caption {g['caption']!r}")
    check("baseline generation deterministic", "PASS" if b1["caption"] == b2["caption"] else "FAIL",
          f"{b1['caption']!r} vs {b2['caption']!r}")

    for arm in ARMS:
        run_generate(run_dir, arm, args.blip_dtype, args.seed)
    caps = {arm: keyed(read_jsonl(run_dir / f"captions_{arm}.jsonl"), arm) for arm in ARMS}
    for arm in ("baseline", "grounded"):
        empty = [i for i in ids if not caps[arm][i]["caption"].strip()]
        check(f"{arm} captions generated", "FAIL" if empty else "PASS",
              f"{len(ids) - len(empty)}/{len(ids)}; e.g. {caps[arm][ids[0]]['caption']!r}")
    bad_base = [i for i in ids if caps["baseline"][i]["prefix"] != BASELINE_PREFIX]
    check("baseline uses baseline prefix", "FAIL" if bad_base else "PASS",
          f"{len(ids) - len(bad_base)}/{len(ids)} use {BASELINE_PREFIX!r}")
    problems = []
    for i in ids:
        want = expected_prefix("grounded", rel[i]["selected_relation"], args.relation_threshold)
        g_rec = caps["grounded"][i]
        if g_rec["prefix"] != want:
            problems.append(f"{i}: prefix {g_rec['prefix']!r} != {want!r}")
        if g_rec["relation_used"]:
            if g_rec["input_ids"] == caps["baseline"][i]["input_ids"]:
                problems.append(f"{i}: grounded BLIP input identical to baseline")
            if not g_rec["prefix_echoed"]:
                problems.append(f"{i}: grounded caption does not start with relation prefix")
            if g_rec["caption"] == caps["baseline"][i]["caption"]:
                problems.append(f"{i}: grounded caption identical to baseline despite relation")
    check("grounded path uses relation prefix exactly when decided",
          "FAIL" if problems else ("PASS" if n_used else "WARN"),
          "; ".join(problems) if problems else
          (f"{n_used} image(s) with relation prefix, rest baseline prefix" if n_used
           else "no image crossed the threshold, so only the fallback path was exercised "
                "on real relations (the synthetic check above covers injection)"))
    fb = [i for i in ids if not caps["grounded"][i]["relation_used"]]
    fb_diff = [i for i in fb if caps["grounded"][i]["caption"] != caps["baseline"][i]["caption"]]
    check("fallback reproduces baseline caption", "WARN" if fb_diff else "PASS",
          f"{len(fb) - len(fb_diff)}/{len(fb)} fallback captions identical to baseline")

    # --- reranking arms -----------------------------------------------------
    if not args.no_rerank:
        _smoke_rerank(run_dir, ids, rel, caps, dtype, args, check)

    try:
        res = evaluate_run(run_dir, args.eval_set, allow_small=True, n_bootstrap=200,
                           seed=args.seed, clipscore=not args.no_clipscore, clip_device=None)
        ch = {a: round(res["metrics_all_images"][a]["chair"]["chair_i"], 4) for a in res["arms_present"]}
        check("hallucination evaluation accepts all arms", "PASS",
              f"CHAIR_i on {len(ids)} images (plumbing only, not a result): {ch}")
    except Exception as exc:
        check("hallucination evaluation accepts all arms", "FAIL", repr(exc))
    return _finish_smoke(run_dir, checks)


# Probe weights for the smoke test only. They exercise every term of the score
# and are NOT the experiment's weights: those are locked by `tune-rerank` on the
# validation split. Nothing measured during a smoke run is a result.
SMOKE_PROBE_WEIGHTS = {"w_obj": 0.5, "w_hall": 0.5, "w_rel": 1.0}


def _smoke_rerank(run_dir, ids, rel, caps, dtype, args, check) -> None:
    """Prove the reranking path works on the real model: the pool contains each
    prefix arm's own caption, the uniform scorer agrees with its fallback, and
    selection is deterministic."""
    import torch
    from utils.caption_candidates import (candidate_prefixes, generate_candidates,
                                          score_candidates)
    from utils.caption_experiment_eval import keyed, read_jsonl
    from utils.caption_rerank import RerankWeights

    try:
        run_candidates(run_dir, args.num_candidates, args.batch_size, args.blip_dtype,
                       args.seed)
        pools = keyed(read_jsonl(run_dir / "candidates.jsonl"), "candidates")
        sizes = {i: len(pools[i]["candidates"]) for i in ids}
        check("candidate pools generated",
              "PASS" if all(sizes.values()) else "FAIL",
              f"candidates per image {sizes} (K={args.num_candidates} per prefix)")
    except Exception as exc:
        check("candidate pools generated", "FAIL", repr(exc))
        return

    # The pool must literally contain what each single-caption arm produced.
    mismatches = []
    for iid in ids:
        pool = pools[iid]["candidates"]
        for arm, source in (("baseline", "baseline"), ("grounded", "relation"),
                            ("objects_only", "objects_only")):
            rec = caps[arm][iid]
            if arm != "baseline" and not rec.get("relation_used"):
                continue
            rank0 = [c for c in pool if c["source"] == source and c["beam_rank"] == 0]
            if not rank0:
                mismatches.append(f"{iid}: no {source} candidate for the {arm} arm")
            elif rank0[0]["text"] != rec["caption"]:
                mismatches.append(f"{iid}/{arm}: pool has {rank0[0]['text']!r}, "
                                  f"arm has {rec['caption']!r}")
    check("pool contains every prefix arm's own caption",
          "FAIL" if mismatches else "PASS",
          "; ".join(mismatches) if mismatches else
          "beam 0 of each prefix is byte-identical to that arm's single caption")

    # The shared-image-embedding scorer must agree with the public forward.
    try:
        from PIL import Image
        import utils.caption_candidates as cc
        image = Image.open(rel[ids[0]]["file"]).convert("RGB")
        cands = generate_candidates(image, candidate_prefixes(rel[ids[0]]["selected_relation"]),
                                    num_candidates=args.num_candidates, dtype=dtype)
        fast = {c["text"]: c["lm_score"]
                for c in score_candidates(image, cands, batch_size=args.batch_size, dtype=dtype)}
        original = cc._decode_logits

        def forward_only(model, input_ids, attention_mask, image_embeds, pixel_values, batch):
            return model(pixel_values=pixel_values.expand(batch, -1, -1, -1),
                         input_ids=input_ids, attention_mask=attention_mask).logits

        cc._decode_logits = forward_only
        try:
            slow = {c["text"]: c["lm_score"]
                    for c in score_candidates(image, cands, batch_size=args.batch_size,
                                              dtype=dtype)}
        finally:
            cc._decode_logits = original
        worst = max(abs(fast[t] - slow[t]) for t in fast)
        check("candidate scoring matches the reference forward",
              "PASS" if worst < 1e-3 else "FAIL",
              f"max |shared-embedding - public forward| = {worst:.2e} over {len(fast)} candidates")
    except Exception as exc:
        check("candidate scoring matches the reference forward", "FAIL", repr(exc))

    weights_path = run_dir / "smoke_probe_weights.json"
    _write_json(weights_path, {
        "objective": "SMOKE PROBE ONLY - not selected on any data",
        "selected_on": {"split": "none", "n_images": 0},
        "weights": {"object_reranked": {**SMOKE_PROBE_WEIGHTS, "w_rel": 0.0},
                    "relation_reranked": dict(SMOKE_PROBE_WEIGHTS)},
    })
    weights_file = load_weights(str(weights_path))
    try:
        rows = {arm: run_rerank(run_dir, arm, weights_file, str(weights_path))
                for arm in RERANK_ARMS}
        pool_texts = {i: {c["text"] for c in pools[i]["candidates"]} for i in ids}
        bad = [f"{arm}/{r['image_id']}" for arm, rs in rows.items() for r in rs
               if r["caption"] not in pool_texts[r["image_id"]]]
        check("reranked captions come from the pool", "FAIL" if bad else "PASS",
              "; ".join(bad) if bad else
              f"{sum(len(r) for r in rows.values())} selections, all from the generated pool")
        changed = sum(r["changed_from_first_candidate"] for r in rows["relation_reranked"])
        check("reranker can depart from BLIP's first candidate", "PASS",
              f"{changed}/{len(ids)} changed with the probe weights {SMOKE_PROBE_WEIGHTS} "
              "(plumbing only, not a result)")
    except Exception as exc:
        check("reranked captions come from the pool", "FAIL", repr(exc))
        return

    for arm in RERANK_ARMS:
        (run_dir / f"captions_{arm}.meta.json").unlink(missing_ok=True)
    again = {arm: run_rerank(run_dir, arm, weights_file, str(weights_path))
             for arm in RERANK_ARMS}
    same = all(json.dumps(rows[a], sort_keys=True) == json.dumps(again[a], sort_keys=True)
               for a in RERANK_ARMS)
    check("reranking is deterministic", "PASS" if same else "FAIL",
          "re-running the selection reproduces every caption and score" if same
          else "a second selection differed")


def _finish_smoke(run_dir: Path, checks: List[Dict]) -> int:
    failed = [c for c in checks if c["status"] == "FAIL"]
    status = "FAIL" if failed else "PASS"
    _write_json(run_dir / "smoke_report.json",
                {"status": status, "created_utc": _now(), "checks": checks,
                 "versions": _versions(), "git": _git()})
    print("-" * 78)
    print(f"SMOKE TEST: {status}  ({len(failed)} failed, "
          f"{sum(c['status'] == 'WARN' for c in checks)} warnings) -> {run_dir / 'smoke_report.json'}")
    if failed:
        print("Do NOT run the full evaluation until every FAIL is resolved.")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------

EXAMPLE_STRATA = (
    ("relation_removed_hallucination", "relation injected; grounded caption has FEWER hallucinated objects"),
    ("relation_added_hallucination", "relation injected; grounded caption has MORE hallucinated objects"),
    ("relation_changed_caption_same_count", "relation injected; caption changed, same hallucination count"),
    ("below_threshold_fallback", "relation predicted but below threshold; baseline prefix used"),
    ("no_relation_fallback", "no usable relation (too few detections or all filtered)"),
)


def _stratum(row: Dict) -> str:
    base = row["arms"]["baseline"]
    grd = row["arms"]["grounded"]
    if grd["relation_used"]:
        if grd["n_hallucinated"] < base["n_hallucinated"]:
            return "relation_removed_hallucination"
        if grd["n_hallucinated"] > base["n_hallucinated"]:
            return "relation_added_hallucination"
        return "relation_changed_caption_same_count"
    if row["decision"].get("fallback_reason") == "below_confidence_threshold":
        return "below_threshold_fallback"
    return "no_relation_fallback"


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


def cmd_examples(args) -> int:
    from PIL import Image, ImageDraw, ImageFont
    from utils.caption_experiment_eval import keyed, read_jsonl

    run_dir = Path(args.run_dir)
    per_image_path = run_dir / "per_image.jsonl"
    if not per_image_path.is_file():
        raise SystemExit(f"{per_image_path} missing - run hallucination_eval.py --run-dir first")
    rows = keyed(read_jsonl(per_image_path), "per_image")
    rel = keyed(read_jsonl(run_dir / "relations.jsonl"), "relations")

    by_stratum: Dict[str, List[str]] = {k: [] for k, _ in EXAMPLE_STRATA}
    for iid in sorted(rows, key=int):
        by_stratum[_stratum(rows[iid])].append(iid)
    rng = random.Random(f"examples:{args.seed}")
    per = max(1, math.ceil(args.n / len(EXAMPLE_STRATA)))
    chosen: List[str] = []
    for key, _ in EXAMPLE_STRATA:
        pool = list(by_stratum[key])
        rng.shuffle(pool)
        chosen += pool[:per]
    leftovers = [i for k, _ in EXAMPLE_STRATA for i in by_stratum[k] if i not in chosen]
    rng.shuffle(leftovers)
    chosen = (chosen + leftovers)[:args.n]

    out_dir = run_dir / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    md = ["# Qualitative examples - NOT statistical evidence\n",
          "Examples are drawn with a fixed seed from five strata so that failures appear "
          "alongside successes. Stratum sizes over the whole run are shown for context; "
          "quantitative claims must come from caption_results.md.\n",
          "| stratum | images in run |", "|---|---|"]
    for key, desc in EXAMPLE_STRATA:
        md.append(f"| {desc} | {len(by_stratum[key])} |")
    md.append("")

    for iid in chosen:
        row, r = rows[iid], rel[iid]
        image = Image.open(r["file"]).convert("RGB")
        scale = 640 / max(image.size)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        draw = ImageDraw.Draw(image)
        sel = r.get("selected_relation")
        for k, d in enumerate(r["eligible_detections"]):
            color = "gray"
            if sel and k == sel["subject_index"]:
                color = "lime"
            elif sel and k == sel["object_index"]:
                color = "deepskyblue"
            box = [v * scale for v in d["box"]]
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0] + 3, box[1] + 2), f"{d['label']} {d['score']:.2f}", fill=color, font=font)
        rel_text = (f"{sel['subject']} {sel['predicate']} {sel['object']}  conf={sel['confidence']:.3f}"
                    f"  ({'USED' if r['decision']['use_relation'] else r['decision']['fallback_reason']})"
                    if sel else f"no relation ({r['decision']['fallback_reason']})")
        lines = [f"image {iid}   stratum: {_stratum(row)}",
                 f"GT objects: {', '.join(row['gt_objects'])}",
                 f"detected: {', '.join(d['label'] for d in r['eligible_detections']) or '-'}",
                 f"relation: {rel_text}"]
        for arm in ("baseline", "grounded", "objects_only"):
            if arm in row["arms"]:
                a = row["arms"][arm]
                lines += _wrap(f"{arm.upper()}: {a['caption']}   [hallucinated: "
                               f"{', '.join(a['hallucinated']) or 'none'}]", 80)
        panel = Image.new("RGB", (max(image.width, 700), image.height + 22 * len(lines) + 16), "white")
        panel.paste(image, (0, 0))
        pd = ImageDraw.Draw(panel)
        for k, text in enumerate(lines):
            pd.text((8, image.height + 8 + 22 * k), text, fill="black", font=font)
        fname = f"{iid}.jpg"
        panel.save(out_dir / fname, quality=90)

        md.append(f"## Image {iid} - {dict(EXAMPLE_STRATA)[_stratum(row)]}\n")
        md.append(f"![{iid}]({fname})\n")
        md.append("| | |")
        md.append("|---|---|")
        md.append(f"| baseline caption | {row['arms']['baseline']['caption']} |")
        md.append(f"| grounded caption | {row['arms']['grounded']['caption']} |")
        if "objects_only" in row["arms"]:
            md.append(f"| objects-only control | {row['arms']['objects_only']['caption']} |")
        md.append(f"| detected objects | {', '.join(d['label'] for d in r['eligible_detections']) or '-'} |")
        md.append(f"| predicted relation | {rel_text} |")
        md.append(f"| human GT objects | {', '.join(row['gt_objects'])} |")
        md.append(f"| hallucinated (baseline / grounded) | "
                  f"{', '.join(row['arms']['baseline']['hallucinated']) or 'none'} / "
                  f"{', '.join(row['arms']['grounded']['hallucinated']) or 'none'} |\n")
    (out_dir / "examples.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[examples] {len(chosen)} examples -> {out_dir / 'examples.md'}")
    print("[examples] strata: " + ", ".join(f"{k}={len(v)}" for k, v in by_stratum.items()))
    return 0


def cmd_report(args) -> int:
    p = Path(args.run_dir) / "caption_results.md"
    if not p.is_file():
        raise SystemExit(f"{p} missing - run hallucination_eval.py --run-dir {args.run_dir}")
    print(p.read_text(encoding="utf-8"))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, run_dir=DEFAULT_RUN_DIR):
        p.add_argument("--run-dir", default=run_dir)
        p.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
        p.add_argument("--selection", default=DEFAULT_SELECTION)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")

    p = sub.add_parser("preflight", help="check environment, data and checkpoint")
    common(p)
    p.add_argument("--vg-root", default="data/visual_genome")
    p.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    p.add_argument("--val-eval-set", default=DEFAULT_VAL_EVAL_SET)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.set_defaults(fn=cmd_preflight)

    p = sub.add_parser("select-checkpoint", help="choose the relation checkpoint by validation top-1")
    p.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    p.add_argument("--output", default=DEFAULT_SELECTION)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_select_checkpoint)

    def relation_args(p):
        p.add_argument("--relation-threshold", type=float, default=RELATION_CONFIDENCE_THRESHOLD,
                       help=f"pre-declared: {RELATION_CONFIDENCE_THRESHOLD}; any other value is "
                            "recorded as a deviation")
        p.add_argument("--checkpoint-dir", default=None,
                       help="override the selected checkpoint (recorded as an override)")
        p.add_argument("--allow-small", action="store_true", help=argparse.SUPPRESS)

    p = sub.add_parser("detect", help="YOLO + CLIP verification + relation model")
    common(p)
    relation_args(p)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--tuning", action="store_true",
                   help="allow a VALIDATION-split evaluation set (weight tuning only; "
                        "the frozen test set is never tuned on)")
    p.set_defaults(fn=cmd_detect)

    p = sub.add_parser("generate", help="BLIP captions for one arm")
    common(p)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--blip-dtype", choices=("float32", "float16"), default="float32",
                   help="float32 default: GTX 16xx cards are prone to fp16 NaNs")
    p.set_defaults(fn=cmd_generate)

    def rerank_gen_args(p):
        p.add_argument("--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES,
                       help=f"beams returned per prefix (default {DEFAULT_NUM_CANDIDATES} = "
                            "num_beams, so beam 0 of each prefix is exactly what the "
                            "corresponding single-caption arm produces)")
        p.add_argument("--batch-size", type=int, default=DEFAULT_SCORE_BATCH_SIZE,
                       help="candidates scored per forward pass (VRAM knob)")
        p.add_argument("--blip-dtype", choices=("float32", "float16"), default="float32",
                       help="float32 default: GTX 16xx cards are prone to fp16 NaNs")

    p = sub.add_parser("candidates", help="multi-candidate BLIP generation for the "
                                          "reranking arms")
    common(p)
    rerank_gen_args(p)
    p.set_defaults(fn=cmd_candidates)

    p = sub.add_parser("rerank", help="select one candidate per image with LOCKED weights")
    common(p)
    p.add_argument("--arm", choices=RERANK_ARMS + ("all",), default="all")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help="weights locked by tune-rerank on the validation split")
    p.set_defaults(fn=cmd_rerank)

    p = sub.add_parser("tune-rerank", help="select the rerank weights on VALIDATION images")
    common(p, run_dir=DEFAULT_VAL_RUN_DIR)
    p.add_argument("--output", default=DEFAULT_WEIGHTS)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_tune_rerank)

    p = sub.add_parser("smoke", help="end-to-end check on a few images")
    common(p, run_dir=f"{DEFAULT_RESULTS_ROOT}/smoke")
    relation_args(p)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--no-clipscore", action="store_true")
    p.add_argument("--no-rerank", action="store_true",
                   help="skip the candidate-generation and reranking checks")
    rerank_gen_args(p)
    p.add_argument("--force", action="store_true", help="wipe an existing smoke run dir")
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("examples", help="qualitative side-by-side examples")
    common(p)
    p.add_argument("--n", type=int, default=16)
    p.set_defaults(fn=cmd_examples)

    p = sub.add_parser("report", help="print caption_results.md")
    common(p)
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    from utils.console import configure_safe_stdio
    configure_safe_stdio()
    raise SystemExit(main())

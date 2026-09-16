"""
hallucination_eval.py — hallucination evaluation.

TWO MODES
=========
1. Caption experiment (use this):

       python hallucination_eval.py --run-dir results_caption/main

   Scores the captions run_caption_experiment.py generated (baseline /
   relation-grounded / objects-only control, identical images and decoding)
   against human Visual Genome object annotations, with corpus-level CHAIR, a
   balanced caption-independent POPE probe set, paired bootstrap confidence
   intervals and validity checks that the relation actually reached BLIP.
   Implementation: utils/caption_experiment_eval.py.

2. Legacy stress test (below; kept for reference, DO NOT report its numbers):
   - default ground truth is YOLO's own detections, circular for a system
     conditioned on YOLO;
   - its "grounded" system adds YOLO-based caption gating/repair and a
     template fallback, so it differs from the baseline in more than the
     relation;
   - its POPE (utils/pope.py) builds positive probes from the objects the
     caption mentions and negative probes from objects it does not, so every
     negative is a true negative by construction and the probe set depends on
     the caption;
   - utils.metrics.detect_coco_objects misses plurals and double-counts
     "hot dog" as hot dog + dog.

Legacy description:

Runs all three systems on the hallucination-prone image set:
    1. BLIP-2 baseline (implicit captioning)
    2. BLIP-2 + CLIP reranking (semantic visual alignment)
    3. Grounded pipeline (YOLO + relations + evidence gating)

Prioritizes groundedness metrics: CHAIR, POPE.
Skips reference-based metrics (no captions available for this set).

Usage:
    # Full evaluation (all 250 images, all 3 systems)
    python hallucination_eval.py

    # Limit images
    python hallucination_eval.py --num-samples 50

    # Limit systems
    python hallucination_eval.py --systems blip2 grounded

Output:
    results/hallucination/
        per_image/           # One JSON per image with all captions + metrics
        system_summary.json  # Aggregate metrics per system
        failure_analysis.json  # Cross-system failure analysis
        comparison.json       # Side-by-side per-image comparison
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple



def _legacy_imports() -> None:
    """The legacy mode pulls in YOLO, BLIP and NLTK-backed metrics; the caption
    experiment mode needs none of them, so they are imported only on demand."""
    global Image, tqdm, run_system_blip2, run_system_blip2_clip, run_system_grounded
    global run_yolo, yolo_detections_to_objects, _SYSTEM_NAMES, _list_image_paths
    global compute_chair, validate_chair_schema, compute_pope, get_clip_scorer
    from PIL import Image
    from tqdm import tqdm
    from evaluate import (
        run_system_blip2,
        run_system_blip2_clip,
        run_system_grounded,
        run_yolo,
        yolo_detections_to_objects,
        _SYSTEM_NAMES,
        _list_image_paths,
    )
    from utils.metrics import compute_chair, validate_chair_schema
    from utils.pope import compute_pope
    from utils.clip_scorer import get_clip_scorer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HALLUCINATION_IMAGE_DIR = "hallucinating images"
OUTPUT_DIR = "results/hallucination"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# CLIP singleton
# ---------------------------------------------------------------------------

_clip_scorer = None

# ---------------------------------------------------------------------------
# Object ground truth
# ---------------------------------------------------------------------------
# Default: YOLO's own detections stand in for ground truth. That is CIRCULAR
# for the grounded system, which is conditioned on and gated against exactly
# those detections — it is graded against its own input. Supply
# --gt-objects-json (see build_caption_eval_set.py) for an independent,
# human-annotated ground truth; the source is recorded in every result file.

_GT_OBJECTS: Optional[Dict[str, Set[str]]] = None
_GT_SOURCE: str = "yolo_proxy"
_GT_META: Dict = {}


def load_gt_objects(path: Optional[str]) -> None:
    """Load an external image_id -> object-set ground truth."""
    global _GT_OBJECTS, _GT_SOURCE, _GT_META
    if not path:
        _GT_OBJECTS = None
        _GT_SOURCE = "yolo_proxy"
        _GT_META = {
            "warning": (
                "CHAIR/POPE scored against YOLO's own detections. The grounded "
                "system is conditioned and gated on those same detections, so "
                "this comparison is circular and must not be reported as "
                "evidence of hallucination reduction."
            ),
        }
        return
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    images = payload.get("images", payload)
    _GT_OBJECTS = {
        str(k): set(v["objects"] if isinstance(v, dict) else v)
        for k, v in images.items()
    }
    _GT_SOURCE = "external"
    _GT_META = payload.get("meta", {})
    _GT_META["path"] = path
    print(f"[hallucination_eval] external object ground truth: {path} "
          f"({len(_GT_OBJECTS):,} images)")


def resolve_gt_objects(image_id: str, yolo_objects: Set[str]) -> Tuple[Set[str], str]:
    """Ground-truth objects for one image, plus the source actually used."""
    if _GT_OBJECTS is None:
        return yolo_objects, "yolo_proxy"
    key = str(image_id)
    if key in _GT_OBJECTS:
        return set(_GT_OBJECTS[key]), "external"
    stem = os.path.splitext(os.path.basename(key))[0]
    if stem in _GT_OBJECTS:
        return set(_GT_OBJECTS[stem]), "external"
    return yolo_objects, "yolo_proxy_fallback"


def _ensure_clip():
    global _clip_scorer
    if _clip_scorer is None:
        _clip_scorer = get_clip_scorer()


# ---------------------------------------------------------------------------
# Per-image analysis
# ---------------------------------------------------------------------------

def analyze_image(
    image_path: str,
    image_id: str,
) -> Dict:
    """Run all systems on a single image and compute hallucination metrics.

    Returns a rich dict with:
        - image_id, file_path
        - blip2:          caption + CHAIR/POPE/CLIP sim
        - blip2_clip:     caption, candidates, ranked + CHAIR/POPE/CLIP sim
        - grounded:       caption, relations + CHAIR/POPE/CLIP sim
        - yolo_detections: ground-truth objects (YOLO proxy)
    """
    result: Dict = {
        "image_id": image_id,
        "file_path": image_path,
    }

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        result["error"] = f"Cannot open image: {e}"
        return result

    # ── YOLO detections (once, shared across all systems) ──────────────
    try:
        dets = run_yolo(image)
        gt_objects = yolo_detections_to_objects(dets)
    except Exception as e:
        dets = []
        gt_objects = set()
        result["yolo_error"] = str(e)

    result["yolo_detections"] = [
        {"label": d["label"], "score": round(d["score"], 3)}
        for d in dets
    ]
    result["yolo_objects"] = sorted(gt_objects)

    # Swap in the independent ground truth when one was supplied.
    gt_objects, gt_source = resolve_gt_objects(image_id, gt_objects)
    result["gt_objects"] = sorted(gt_objects)
    result["gt_source"] = gt_source

    # ── System 1: BLIP-2 baseline ─────────────────────────────────────
    try:
        blip2_caption = run_system_blip2(image)
    except Exception as e:
        blip2_caption = f"[error: {e}]"

    result["blip2"] = {
        "caption": blip2_caption,
    }
    _add_metrics(result["blip2"], blip2_caption, gt_objects)

    # CLIP similarity
    try:
        _ensure_clip()
        clip_sim = _clip_scorer.compute_similarity(image, blip2_caption)
        result["blip2"]["clip_similarity"] = round(clip_sim, 4)
    except Exception:
        result["blip2"]["clip_similarity"] = None

    # ── System 2: BLIP-2 + CLIP reranking ─────────────────────────────
    try:
        blip2_clip_caption, candidates, ranked = run_system_blip2_clip(image)
    except Exception as e:
        blip2_clip_caption = f"[error: {e}]"
        candidates = []
        ranked = []

    result["blip2_clip"] = {
        "caption": blip2_clip_caption,
        "candidates": candidates,
        "ranked": [
            {"caption": cap, "clip_similarity": round(score, 4)}
            for cap, score in ranked
        ],
    }
    _add_metrics(result["blip2_clip"], blip2_clip_caption, gt_objects)

    try:
        _ensure_clip()
        csim = _clip_scorer.compute_similarity(image, blip2_clip_caption)
        result["blip2_clip"]["clip_similarity"] = round(csim, 4)
    except Exception:
        result["blip2_clip"]["clip_similarity"] = None

    # ── System 3: Grounded pipeline ──────────────────────────────────
    try:
        grounded_caption, relations = run_system_grounded(image, dets)
    except Exception as e:
        grounded_caption = f"[error: {e}]"
        relations = []

    result["grounded"] = {
        "caption": grounded_caption,
        "relations": [f"{s} {p} {o}" for s, p, o in relations],
        "relation_triples": [{"subject": s, "predicate": p, "object": o}
                             for s, p, o in relations],
    }
    _add_metrics(result["grounded"], grounded_caption, gt_objects)

    try:
        _ensure_clip()
        gsim = _clip_scorer.compute_similarity(image, grounded_caption)
        result["grounded"]["clip_similarity"] = round(gsim, 4)
    except Exception:
        result["grounded"]["clip_similarity"] = None

    # Detect grounded model conservative fallback
    result["grounded"]["is_safe_fallback"] = (
        "The scene contains" in grounded_caption
        or "No objects detected" in grounded_caption
    )

    return result


def _add_metrics(system_result: Dict, caption: str, gt_objects: Set[str]):
    """Compute CHAIR and POPE for a caption and set all result fields."""
    if not gt_objects:
        system_result["chair_i"] = 0.0
        system_result["chair_s"] = 0.0
        system_result["hallucinated_objects"] = []
        system_result["missed_objects"] = []
        system_result["pope"] = None
        return

    chair = compute_chair(caption, gt_objects)
    validate_chair_schema(chair)

    pope = compute_pope(caption, gt_objects)

    system_result["chair_i"] = chair["chair_i"]
    system_result["chair_s"] = chair["chair_s"]
    system_result["hallucinated_objects"] = chair["hallucinated_objects"]
    system_result["missed_objects"] = chair["missed_objects"]

    system_result["pope"] = {
        "precision": pope["pope_precision"],
        "recall": pope["pope_recall"],
        "f1": pope["pope_f1"],
        "accuracy": pope["pope_accuracy"],
        "tp": pope["pope_tp"],
        "fp": pope["pope_fp"],
        "fn": pope["pope_fn"],
        "tn": pope["pope_tn"],
        "num_positive_probes": pope["pope_num_positive_probes"],
        "num_negative_probes": pope["pope_num_negative_probes"],
        "hallucinated_objects": pope["pope_hallucinated_objects"],
        "missed_objects": pope["pope_missed_objects"],
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(
    per_image: List[Dict],
) -> Dict:
    """Compute aggregate metrics across all images per system."""
    systems = ["blip2", "blip2_clip", "grounded"]
    agg: Dict = {}

    for sk in systems:
        captions = [r[sk]["caption"] for r in per_image if sk in r and "error" not in r]

        chair_i_vals = [r[sk].get("chair_i", 0.0) for r in per_image if sk in r]
        chair_s_vals = [r[sk].get("chair_s", 0.0) for r in per_image if sk in r]
        clip_sims = [
            r[sk].get("clip_similarity") for r in per_image
            if sk in r and r[sk].get("clip_similarity") is not None
        ]

        pope_f1_vals = [
            r[sk].get("pope", {}).get("f1", 0.0) for r in per_image
            if sk in r and r[sk].get("pope") is not None
        ]
        pope_prec_vals = [
            r[sk].get("pope", {}).get("precision", 0.0) for r in per_image
            if sk in r and r[sk].get("pope") is not None
        ]
        pope_rec_vals = [
            r[sk].get("pope", {}).get("recall", 0.0) for r in per_image
            if sk in r and r[sk].get("pope") is not None
        ]
        pope_acc_vals = [
            r[sk].get("pope", {}).get("accuracy", 0.0) for r in per_image
            if sk in r and r[sk].get("pope") is not None
        ]

        safe_fallback_count = sum(
            1 for r in per_image
            if sk in r and r[sk].get("is_safe_fallback", False)
        )

        total_hallucinated_assertions = sum(
            r[sk].get("pope", {}).get("fp", 0) for r in per_image
            if sk in r and r[sk].get("pope") is not None
        )
        total_missed_objects = sum(
            r[sk].get("pope", {}).get("fn", 0) for r in per_image
            if sk in r and r[sk].get("pope") is not None
        )

        avg_clip = round(sum(clip_sims) / len(clip_sims), 4) if clip_sims else 0.0

        agg[sk] = {
            "system_name": _SYSTEM_NAMES[sk],
            "num_images": len([r for r in per_image if sk in r]),
            "avg_chair_i": round(sum(chair_i_vals) / len(chair_i_vals), 4) if chair_i_vals else 0.0,
            "avg_chair_s": round(sum(chair_s_vals) / len(chair_s_vals), 4) if chair_s_vals else 0.0,
            "hallucination_rate": round(sum(chair_s_vals) / len(chair_s_vals), 4) if chair_s_vals else 0.0,
            "total_images_with_hallucination": int(sum(chair_s_vals)),
            "avg_pope_f1": round(sum(pope_f1_vals) / len(pope_f1_vals), 4) if pope_f1_vals else 0.0,
            "avg_pope_precision": round(sum(pope_prec_vals) / len(pope_prec_vals), 4) if pope_prec_vals else 0.0,
            "avg_pope_recall": round(sum(pope_rec_vals) / len(pope_rec_vals), 4) if pope_rec_vals else 0.0,
            "avg_pope_accuracy": round(sum(pope_acc_vals) / len(pope_acc_vals), 4) if pope_acc_vals else 0.0,
            "avg_clip_similarity": avg_clip,
            "total_hallucinated_assertions": int(total_hallucinated_assertions),
            "total_missed_objects": int(total_missed_objects),
            "safe_fallback_count": safe_fallback_count,
            # A system can score a perfect CHAIR by refusing to say anything
            # informative. These two make that failure mode visible instead of
            # letting it read as a hallucination win.
            "safe_fallback_rate": round(safe_fallback_count / len(per_image), 4)
                                  if per_image else 0.0,
            "avg_caption_words": round(
                sum(len(str(r[sk].get("caption", "")).split())
                    for r in per_image if sk in r) / max(
                        sum(1 for r in per_image if sk in r), 1), 2),
        }

    return agg


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def failure_analysis(per_image: List[Dict]) -> Dict:
    """Analyze failure modes across all systems."""
    systems = ["blip2", "blip2_clip", "grounded"]

    analysis: Dict = {
        "hallucinated_objects_by_system": {},
        "missed_objects_by_system": {},
        "grounded_safe_fallbacks": [],
        "relation_analysis": {
            "total_images_with_relations": 0,
            "images_with_relations": [],
        },
        "pope_confusion_by_system": {},
        "clip_similarity_winners": {},
        "hallucination_frequency": {},  # most commonly hallucinated objects
    }

    # Hallucinated object frequency
    for sk in systems:
        hal_counter: Counter = Counter()
        miss_counter: Counter = Counter()

        for r in per_image:
            if sk not in r:
                continue
            for obj in r[sk].get("hallucinated_objects", []):
                hal_counter[obj] += 1
            for obj in r[sk].get("missed_objects", []):
                miss_counter[obj] += 1

        analysis["hallucinated_objects_by_system"][sk] = dict(
            hal_counter.most_common(20)
        )
        analysis["missed_objects_by_system"][sk] = dict(
            miss_counter.most_common(20)
        )

    # Grounded model safe fallbacks
    for r in per_image:
        if r.get("grounded", {}).get("is_safe_fallback", False):
            analysis["grounded_safe_fallbacks"].append({
                "image_id": r["image_id"],
                "blip2_caption": r.get("blip2", {}).get("caption", ""),
                "grounded_caption": r.get("grounded", {}).get("caption", ""),
            })

    # Relation analysis (grounded model)
    for r in per_image:
        rels = r.get("grounded", {}).get("relation_triples", [])
        if rels:
            analysis["relation_analysis"]["total_images_with_relations"] += 1
            analysis["relation_analysis"]["images_with_relations"].append({
                "image_id": r["image_id"],
                "relations": rels,
            })

    # POPE confusion matrix totals
    for sk in systems:
        tp = sum(
            r[sk].get("pope", {}).get("tp", 0) for r in per_image
            if sk in r and r[sk].get("pope")
        )
        fp = sum(
            r[sk].get("pope", {}).get("fp", 0) for r in per_image
            if sk in r and r[sk].get("pope")
        )
        fn = sum(
            r[sk].get("pope", {}).get("fn", 0) for r in per_image
            if sk in r and r[sk].get("pope")
        )
        tn = sum(
            r[sk].get("pope", {}).get("tn", 0) for r in per_image
            if sk in r and r[sk].get("pope")
        )
        analysis["pope_confusion_by_system"][sk] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    # CLIP similarity winners (which system's caption best matches image)
    clip_wins = Counter()
    for r in per_image:
        sims = {}
        for sk in systems:
            if sk in r and r[sk].get("clip_similarity") is not None:
                sims[sk] = r[sk]["clip_similarity"]
        if sims:
            winner = max(sims, key=sims.get)
            clip_wins[winner] += 1
    analysis["clip_similarity_winners"] = dict(clip_wins.most_common())

    # Most hallucinated objects overall
    all_hal: Counter = Counter()
    for r in per_image:
        for sk in systems:
            if sk in r:
                for obj in r[sk].get("hallucinated_objects", []):
                    all_hal[obj] += 1
    analysis["hallucination_frequency"] = dict(all_hal.most_common(20))

    return analysis


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_hallucination_eval(args):
    # Honour --output-dir; this used to always write to the module constant.
    output_root = Path(getattr(args, "output_dir", None) or OUTPUT_DIR)
    per_image_dir = output_root / "per_image"
    os.makedirs(str(per_image_dir), exist_ok=True)

    # ── Load images ──────────────────────────────────────────────────
    if getattr(args, "images_from_gt", False):
        # Take the image set straight from the ground-truth manifest, so the
        # images evaluated and the objects scored against always agree.
        if _GT_OBJECTS is None:
            raise SystemExit(
                "[hallucination_eval] --images-from-gt requires --gt-objects-json"
            )
        with open(args.gt_objects_json, "r", encoding="utf-8") as f:
            gt_payload = json.load(f)
        image_paths = [
            Path(entry["file"])
            for entry in gt_payload.get("images", {}).values()
            if isinstance(entry, dict) and entry.get("file")
            and os.path.isfile(entry["file"])
        ]
        image_paths.sort(key=lambda q: q.stem)
        source_desc = f"ground-truth manifest '{args.gt_objects_json}'"
    else:
        image_paths = _list_image_paths(args.image_dir)
        source_desc = f"'{args.image_dir}/'"

    if args.num_samples is not None:
        image_paths = image_paths[:args.num_samples]

    print(f"[hallucination_eval] Loaded {len(image_paths)} images from {source_desc}")
    print(f"[hallucination_eval] Object ground truth: {_GT_SOURCE}")
    if _GT_SOURCE == "yolo_proxy":
        print("[hallucination_eval] WARNING: CHAIR/POPE are being scored against "
              "YOLO's own detections. The grounded system is conditioned and "
              "gated on those detections, so the comparison is CIRCULAR. Pass "
              "--gt-objects-json for an independent ground truth.")
    print(f"[hallucination_eval] Systems: {args.systems or 'all three'}")
    print()

    # ── Analyze each image ──────────────────────────────────────────
    per_image: List[Dict] = []
    pbar = tqdm(image_paths, desc="Hallucination evaluation")

    for p in pbar:
        image_id = p.stem
        result = analyze_image(str(p), image_id)

        if "error" in result:
            print(f"\n[skip] {image_id}: {result['error']}")
            continue

        per_image.append(result)

        # Save per-image analysis
        per_image_path = per_image_dir / f"{image_id}_analysis.json"
        _save_json(result, str(per_image_path))

        # Quick summary in progress bar
        blip2_has_hal = result.get("blip2", {}).get("chair_s", 0) > 0
        clip_has_hal = result.get("blip2_clip", {}).get("chair_s", 0) > 0
        grd_has_hal = result.get("grounded", {}).get("chair_s", 0) > 0
        pbar.set_postfix({
            "B2": "H" if blip2_has_hal else ".",
            "B2C": "H" if clip_has_hal else ".",
            "GRD": "H" if grd_has_hal else ".",
            "img": image_id,
        })

    if not per_image:
        print("[hallucination_eval] No images analyzed. Exiting.")
        return

    print(f"\n[hallucination_eval] Analyzed {len(per_image)} images successfully.")

    # ── Compute aggregates ──────────────────────────────────────────
    agg = aggregate_results(per_image)
    agg["ground_truth"] = {
        "source": _GT_SOURCE,
        "circular": _GT_SOURCE == "yolo_proxy",
        "n_images_scored_against_external_gt": sum(
            1 for r in per_image if r.get("gt_source") == "external"
        ),
        "n_images_fell_back_to_yolo": sum(
            1 for r in per_image if r.get("gt_source", "").startswith("yolo_proxy")
        ),
        "meta": _GT_META,
    }
    summary_path = output_root / "system_summary.json"
    _save_json(agg, str(summary_path))
    print(f"[hallucination_eval] Saved system summary to {summary_path}")

    # ── Print aggregate comparison ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  HALLUCINATION STRESS-TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"  Images evaluated: {len(per_image)}")

    metrics_display = [
        ("avg_chair_i", "CHAIR_i (↓)"),
        ("hallucination_rate", "Hall. Rate (↓)"),
        ("total_images_with_hallucination", "Hall. Images (↓)"),
        ("avg_pope_f1", "POPE F1 (↑)"),
        ("avg_pope_precision", "POPE Prec (↑)"),
        ("avg_pope_recall", "POPE Rec (↑)"),
        ("avg_pope_accuracy", "POPE Acc (↑)"),
        ("avg_clip_similarity", "CLIP Sim (↑)"),
        ("total_hallucinated_assertions", "Hal. Assert. (↓)"),
        ("total_missed_objects", "Missed Obj. (↓)"),
        ("safe_fallback_count", "Safe Fallback"),
    ]

    print()
    header = f"{'Metric':<30}"
    for sk in agg:
        header += f" {_SYSTEM_NAMES[sk]:<25}"
    print(f"  {header}")

    for key, label in metrics_display:
        row = f"  {label:<30}"
        for sk in agg:
            val = agg[sk].get(key, "N/A")
            if isinstance(val, float):
                row += f" {val:<25.4f}"
            else:
                row += f" {str(val):<25}"
        print(row)

    # ── Failure analysis ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  FAILURE ANALYSIS")
    print(f"{'=' * 60}")

    fa = failure_analysis(per_image)
    fa_path = output_root / "failure_analysis.json"
    _save_json(fa, str(fa_path))
    print(f"[hallucination_eval] Saved failure analysis to {fa_path}")

    # Print top hallucinated objects
    print(f"\n  Top 10 most hallucinated objects (all systems):")
    for obj, count in list(fa["hallucination_frequency"].items())[:10]:
        print(f"    {obj:<20} {count:>4} images")

    # Print grounded safe fallbacks
    if fa["grounded_safe_fallbacks"]:
        n_fb = len(fa["grounded_safe_fallbacks"])
        print(f"\n  Grounded safe fallbacks: {n_fb}/{len(per_image)} images")
        print(f"    (Grounded model fell back to conservative template)")
        print(f"    First 5 fallback cases:")
        for fb in fa["grounded_safe_fallbacks"][:5]:
            print(f"    [{fb['image_id']}]")
            print(f"      BLIP-2:  {fb['blip2_caption'][:80]}")
            print(f"      Grounded: {fb['grounded_caption'][:80]}")
    else:
        print(f"\n  Grounded safe fallbacks: 0/{len(per_image)}")

    # Print POPE confusion
    print(f"\n  POPE Confusion Matrix (total across all images):")
    for sk, cm in fa["pope_confusion_by_system"].items():
        total = cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"]
        acc = (cm["tp"] + cm["tn"]) / total if total > 0 else 0
        print(f"    {_SYSTEM_NAMES[sk]:<25}"
              f"  TP={cm['tp']:>4} FP={cm['fp']:>4}"
              f"  FN={cm['fn']:>4} TN={cm['tn']:>4}"
              f"  Acc={acc:.4f}")

    # Print CLIP similarity winners
    print(f"\n  CLIP Similarity Winners (best caption per image):")
    for sk, count in fa["clip_similarity_winners"].items():
        pct = 100 * count / len(per_image)
        print(f"    {_SYSTEM_NAMES[sk]:<25}  {count:>4} images ({pct:.1f}%)")

    # ── Cross-system per-image comparison ────────────────────────────
    comparison = {
        "num_samples": len(per_image),
        "systems": list(agg.keys()),
        "system_names": {sk: _SYSTEM_NAMES[sk] for sk in agg},
        "aggregate": agg,
        "failure_analysis": fa,
        "per_image": [
            {
                "image_id": r["image_id"],
                "blip2_caption": r.get("blip2", {}).get("caption", ""),
                "blip2_clip_caption": r.get("blip2_clip", {}).get("caption", ""),
                "grounded_caption": r.get("grounded", {}).get("caption", ""),
                "yolo_objects": r.get("yolo_objects", []),
                "grounded_relations": r.get("grounded", {}).get("relations", []),
                "blip2_chair_i": r.get("blip2", {}).get("chair_i", 0),
                "blip2_clip_chair_i": r.get("blip2_clip", {}).get("chair_i", 0),
                "grounded_chair_i": r.get("grounded", {}).get("chair_i", 0),
                "blip2_hallucinated": r.get("blip2", {}).get("hallucinated_objects", []),
                "blip2_clip_hallucinated": r.get("blip2_clip", {}).get("hallucinated_objects", []),
                "grounded_hallucinated": r.get("grounded", {}).get("hallucinated_objects", []),
                "blip2_clip_similarity": r.get("blip2", {}).get("clip_similarity"),
                "blip2_clip_rerank_similarity": r.get("blip2_clip", {}).get("clip_similarity"),
                "grounded_clip_similarity": r.get("grounded", {}).get("clip_similarity"),
            }
            for r in per_image
        ],
    }

    comp_path = output_root / "comparison.json"
    _save_json(comparison, str(comp_path))
    print(f"\n[hallucination_eval] Saved comparison to {comp_path}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  HALLUCINATION STRESS-TEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Output: {output_root.resolve()}")
    print(f"  Images: {len(per_image)}")
    print(f"  Systems: {', '.join(_SYSTEM_NAMES[sk] for sk in agg)}")
    print(f"  Metrics: CHAIR_i, CHAIR_s, POPE (F1/Prec/Rec/Acc), CLIP sim")
    print(f"  Failure analysis: per-object hallucination frequency, safe fallbacks, POPE confusion")
    print()

    # Cross-system verdict
    print(f"  GROUNDEDNESS VERDICT (based on CHAIR_i):")
    systems_sorted = sorted(agg.keys(), key=lambda sk: agg[sk].get("avg_chair_i", 1.0))
    for i, sk in enumerate(systems_sorted):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"  {i+1}."
        print(f"    {medal} {_SYSTEM_NAMES[sk]}: CHAIR_i={agg[sk]['avg_chair_i']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hallucination evaluation. Use --run-dir for the caption experiment.",
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help="Caption-experiment run directory (run_caption_experiment.py). "
             "Selects the fair evaluation; all legacy options are ignored.",
    )
    parser.add_argument(
        "--eval-set", type=str, default=None,
        help="Caption evaluation set (default: the one recorded in run_config.json)",
    )
    parser.add_argument("--bootstrap", type=int, default=10000,
                        help="Paired bootstrap resamples (caption experiment)")
    parser.add_argument("--no-clipscore", action="store_true",
                        help="Skip CLIPScore (caption experiment)")
    parser.add_argument("--allow-small", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument(
        "--image-dir", type=str, default=HALLUCINATION_IMAGE_DIR,
        help=f"Directory with test images (default: {HALLUCINATION_IMAGE_DIR}/)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}/)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Limit number of images to evaluate",
    )
    parser.add_argument(
        "--systems", type=str, nargs="+",
        choices=["blip2", "blip2_clip", "grounded"], default=None,
        help="Systems to evaluate (default: all three)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--images-from-gt", dest="images_from_gt", action="store_true",
        help="Take the image list from --gt-objects-json instead of "
             "--image-dir, so the evaluated images and the scored ground "
             "truth are guaranteed to match",
    )
    parser.add_argument(
        "--gt-objects-json", type=str, default=None,
        help="Independent object ground truth produced by "
             "build_caption_eval_set.py. Without it CHAIR/POPE fall back to "
             "YOLO's own detections, which is CIRCULAR for the grounded "
             "system and is flagged as such in the output.",
    )

    args = parser.parse_args()

    if args.run_dir:
        from utils.caption_experiment_eval import evaluate_run
        start = time.time()
        results = evaluate_run(args.run_dir, args.eval_set, allow_small=args.allow_small,
                               n_bootstrap=args.bootstrap, seed=args.seed,
                               clipscore=not args.no_clipscore)
        print((Path(args.run_dir) / "caption_results.md").read_text(encoding="utf-8"))
        print(f"[hallucination_eval] wrote {Path(args.run_dir) / 'caption_results.json'} "
              f"({results['n_images']} images, {time.time() - start:.1f}s)")
        return

    print("[hallucination_eval] LEGACY MODE: circular YOLO ground truth by default, "
          "gating confounds and a non-balanced POPE. Do not report these numbers; "
          "use --run-dir for the caption experiment.")
    _legacy_imports()
    from utils.seed import set_seed
    set_seed(args.seed)
    load_gt_objects(args.gt_objects_json)

    start = time.time()
    run_hallucination_eval(args)
    elapsed = time.time() - start
    print(f"[hallucination_eval] Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    from utils.console import configure_safe_stdio
    configure_safe_stdio()
    main()

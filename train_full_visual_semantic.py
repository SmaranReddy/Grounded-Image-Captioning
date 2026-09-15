"""
Full Visual-Semantic Relation MLP Training
===========================================

Trains the relation MLP on VG images with CLIP visual features.

Supports two modes:
  1. Mixed/Fallback Mode (--use-visual only)
     - Allows zero-vector fallback for missing-image samples
     - Legacy behavior (54.7% real CLIP coverage)

  2. Pure Visual Mode (--use-visual --require-visual)
     - ONLY retains samples with valid non-zero CLIP embeddings
     - 100% real visual-semantic supervision
     - No geometry-only fallback samples
     - Scientifically clean appearance-driven relation learning

Usage:
  python train_full_visual_semantic.py                          (geometry-only)
  python train_full_visual_semantic.py --use-visual             (mixed/fallback)
  python train_full_visual_semantic.py --use-visual --require-visual  (pure visual)
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

from relation_prediction.model import (
    FeatureDimensionError, RelationMLP, format_feature_blocks,
)
from relation_prediction.relation_transformer import RelationTransformer
from relation_prediction.vg_dataset import (
    VGRelationshipDataset, Vocab, GEO_DIM, GEO_DIM_EXT,
    POSE_FEATURE_DIM, UNION_FEATURE_DIM, geo_extractor,
)
from utils.console import configure_safe_stdio

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VG_ROOT = PROJ_ROOT / "data/visual_genome"
CHECKPOINT_DIR = PROJ_ROOT / "checkpoints"
VG_IMAGE_DIR = VG_ROOT / "images"
CLIP_CACHE_PATH = VG_ROOT / "clip_cache_proper.pt"

BATCH_SIZE = 256
EPOCHS = 25
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.1
EMBED_DIM = 64
HIDDEN_DIMS = (256, 128)
DROPOUT = 0.3
MIN_PRED_COUNT = 0       # 0 = keep every predicate class (was a silent no-op at 50)
MAX_SAMPLES = None
SEED = 42
USE_VISUAL = True
REQUIRE_VISUAL = False
VISUAL_FILTER_ONLY = False  # load the CLIP cache to fix the sample population,
                            # but give the model clip_dim=0 (geometry control)
USE_POSE = False
USE_UNION = False
MODEL_TYPE = "mlp"
SPLIT_MANIFEST = None   # path to a frozen image-disjoint split manifest (E0 protocol)
GEO_MODE = "basic"      # "basic" = 5-dim legacy geometry, "ext" = 19-dim extended
PREDICATE_SCHEME = "v1" # "v1" = frozen E0 normaliser, "v2" = surface-form normaliser
GEO_NORM = False        # standardise the geometry block with BatchNorm1d
LOSS = "focal"          # "ce" (plain cross-entropy) or "focal"
CLASS_WEIGHT_ALPHA = None  # None = legacy effective-number weights; float = 1/count**alpha
SELECT_METRIC = "top1"  # checkpoint selection metric: "top1" or "macro_f1"
D_MODEL = 256

SEMANTIC_PREDICATES = frozenset({
    "riding", "carrying", "holding", "wearing", "sitting on", "standing on",
})

QUALITATIVE_PAIRS = [
    ("person", "bicycle"),
    ("person", "horse"),
    ("person", "backpack"),
    ("person", "chair"),
    ("person", "umbrella"),
    ("person", "surfboard"),
    ("person", "cell phone"),
    ("person", "car"),
    ("person", "dog"),
    ("person", "bottle"),
    ("person", "skateboard"),
    ("person", "couch"),
    ("dog", "cat"),
    ("cat", "couch"),
    ("person", "horse"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collate(batch):
    subj_idxs, obj_idxs, geos, preds = [], [], [], []
    subj_feats, obj_feats = [], []
    union_feats, pose_feats = [], []
    for item in batch:
        subj_idxs.append(item[0])
        obj_idxs.append(item[1])
        geos.append(item[2])
        preds.append(item[3])
        idx = 4
        if len(item) > idx:
            subj_feats.append(item[idx]); obj_feats.append(item[idx + 1])
            idx += 2
            # The dataset appends union BEFORE pose, but only for the modes it
            # was asked for. Distinguishing them by position alone is wrong when
            # only ONE of the two is enabled (pose-only used to be collated into
            # the union slot), so dispatch on the tensor width instead.
            while len(item) > idx:
                feat = item[idx]
                if feat.shape[-1] == POSE_FEATURE_DIM and UNION_FEATURE_DIM != POSE_FEATURE_DIM:
                    pose_feats.append(feat)
                else:
                    union_feats.append(feat)
                idx += 1
    result = (
        torch.stack(subj_idxs),
        torch.stack(obj_idxs),
        torch.stack(geos),
        torch.stack(preds),
    )
    if subj_feats:
        result = result + (torch.stack(subj_feats), torch.stack(obj_feats))
    if union_feats:
        result = result + (torch.stack(union_feats),)
    if pose_feats:
        result = result + (torch.stack(pose_feats),)
    return result


def resolve_feature_dims(*, use_visual, visual_filter_only, use_union, use_pose,
                         geo_mode):
    """Width of every optional feature block for a run's feature flags.

    VISUAL_FILTER_ONLY is what makes the four-way feature ablation a valid
    comparison. Turning visual features on changes the sample population
    (require_visual drops pairs whose crops are missing or degenerate), so a
    geometry run with use_visual=False would be scored on a DIFFERENT and
    larger test set than the CLIP runs - the variants would differ in two
    ways at once. With this flag the control loads the same cache and keeps
    the same samples, and only the model's input width changes.
    """
    _, geo_dim = geo_extractor(geo_mode)
    return {
        "geo_dim": geo_dim,
        "clip_dim": 0 if visual_filter_only else (
            VGRelationshipDataset.CLIP_DIM if use_visual else 0),
        "union_dim": UNION_FEATURE_DIM if use_union else 0,
        "pose_dim": POSE_FEATURE_DIM if use_pose else 0,
    }


def _unpack_batch(batch, device, *, has_visual, use_union, use_pose):
    """Split a collated batch into model inputs. The ONLY copy of the layout.

    The training loop, per-epoch validation and the post-training best-model
    analysis each used to carry their own unpacking, and the post-training
    call omitted use_union: union_feat came back None, the model silently
    skipped that block, and the union arm crashed with "mat1 and mat2 shapes
    cannot be multiplied (512x1683 and 2451x256)" only after its last epoch.
    The flags are keyword-only with no defaults so a caller cannot leave one
    out, and the tensor count is checked so the layout cannot be misread.
    """
    n_expected = 4 + (2 if has_visual else 0) + int(use_union) + int(use_pose)
    if len(batch) != n_expected:
        raise FeatureDimensionError(
            f"collated batch carries {len(batch)} tensors, but has_visual="
            f"{has_visual}, use_union={use_union}, use_pose={use_pose} "
            f"means {n_expected}"
        )
    subj, obj, geo, pred = (t.to(device) for t in batch[:4])
    subj_feat = obj_feat = union_feat = pose_feat = None
    idx = 4
    if has_visual:
        subj_feat = batch[idx].to(device)
        obj_feat = batch[idx + 1].to(device)
        idx += 2
    if use_union:
        union_feat = batch[idx].to(device)
        idx += 1
    if use_pose:
        pose_feat = batch[idx].to(device)
    return subj, obj, geo, pred, subj_feat, obj_feat, union_feat, pose_feat


PREFLIGHT_SAMPLES = 8


def preflight_feature_dims(model, splits, device, *, has_visual, use_union,
                           use_pose):
    """Fail before epoch 1 if any split's batches do not fit the model.

    Each split gets a small batch built through the same _collate and
    _unpack_batch that the loaders and the post-training analysis use. Every
    feature block's width is checked against model.feature_blocks(), and one
    no-grad forward is run in eval mode.

    Samples are fetched by index rather than by iterating a DataLoader, so no
    RNG is consumed: the shuffle order and every seeded draw after this are
    exactly what they would have been without the check.
    """
    print(f"\n{'=' * 78}")
    print("  PREFLIGHT - FEATURE DIMENSIONS")
    print(f"{'=' * 78}")
    expected = model.feature_blocks() if hasattr(model, "feature_blocks") else None
    if expected is not None:
        print(f"  model expects : {format_feature_blocks(expected)} = {model.input_dim}")
    was_training = model.training
    model.eval()
    try:
        for name, ds in splits.items():
            try:
                n = min(len(ds), PREFLIGHT_SAMPLES)
                if n == 0:
                    raise FeatureDimensionError(f"the {name} split is empty")
                batch = _collate([ds[i] for i in range(n)])
                subj, obj, geo, _, sf, of, uf, pf = _unpack_batch(
                    batch, device, has_visual=has_visual,
                    use_union=use_union, use_pose=use_pose)
                if expected is not None:
                    actual = model.check_inputs(geo, sf, of, uf, pf)
                    actual_dim = sum(w for _, w in actual)
                    if actual_dim != model.input_dim:
                        raise FeatureDimensionError(
                            f"{name} batch supplies {actual_dim} columns, "
                            f"model expects {model.input_dim}")
                    print(f"  {name:<5} batch   : {format_feature_blocks(actual)} "
                          f"= {actual_dim}  OK")
                with torch.no_grad():
                    logits = model(subj, obj, geo, subj_feat=sf, obj_feat=of,
                                   union_feat=uf, pose_feat=pf)
                if logits.shape[0] != n:
                    raise FeatureDimensionError(
                        f"{name} forward returned {tuple(logits.shape)} for {n} samples")
            except FeatureDimensionError:
                print(f"\n  PREFLIGHT FAILED on the {name} split; training was "
                      "not started.", file=sys.stderr)
                raise
    finally:
        model.train(was_training)
    print("  PREFLIGHT PASSED: every split's batch matches the model input width")


_VALID_IDX_CACHE = {}


def _valid_predicate_indices(pred_vocab, device):
    """Vocabulary indices of real predicates (PAD/UNK excluded), on `device`."""
    key = (len(pred_vocab), str(device))
    cached = _VALID_IDX_CACHE.get(key)
    if cached is None:
        idxs = [i for i in range(len(pred_vocab))
                if pred_vocab.token(i) not in (Vocab.PAD, Vocab.UNK)]
        cached = torch.tensor(idxs, dtype=torch.long, device=device)
        _VALID_IDX_CACHE[key] = cached
    return cached


def macro_f1_from_metrics(metrics, preds, targets, pred_vocab):
    """Macro-F1 over predicates with support > 0 — the E0 definition."""
    preds = preds.numpy() if hasattr(preds, "numpy") else preds
    targets = targets.numpy() if hasattr(targets, "numpy") else targets
    f1s = []
    for i in range(len(pred_vocab)):
        token = pred_vocab.token(i)
        if token in (Vocab.PAD, Vocab.UNK):
            continue
        support = int((targets == i).sum())
        if support == 0:
            continue
        tp = int(((preds == i) & (targets == i)).sum())
        predicted = int((preds == i).sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support
        f1s.append(2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def compute_predicate_metrics(model, loader, device, pred_vocab, *, has_visual,
                              use_union, use_pose):
    model.eval()
    per_pred_correct = defaultdict(int)
    per_pred_total = defaultdict(int)
    confusion_counts = defaultdict(lambda: defaultdict(int))

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            subj, obj, geo, target, subj_feat, obj_feat, union_feat, pose_feat = _unpack_batch(
                batch, device, has_visual=has_visual, use_union=use_union,
                use_pose=use_pose)

            logits = model(subj, obj, geo,
                           subj_feat=subj_feat, obj_feat=obj_feat,
                           union_feat=union_feat, pose_feat=pose_feat)

            # Restrict the candidate set to real predicates, exactly as the
            # frozen E0 test protocol does, so validation numbers reported here
            # are on the same footing as the numbers eval_gt_relations.py
            # produces. PAD/UNK are vocabulary slots, not classes.
            valid = _valid_predicate_indices(pred_vocab, logits.device)
            preds = valid[logits.index_select(1, valid).argmax(dim=-1)]
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

            for p, t in zip(preds.cpu().numpy(), target.cpu().numpy()):
                pred_token = pred_vocab.token(int(p))
                true_token = pred_vocab.token(int(t))
                per_pred_total[true_token] += 1
                if p == t:
                    per_pred_correct[true_token] += 1
                if true_token != Vocab.PAD:
                    confusion_counts[true_token][pred_token] += 1

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = {}
    for token in sorted(set(list(per_pred_total.keys()) + list(per_pred_correct.keys()))):
        total = per_pred_total.get(token, 0)
        correct = per_pred_correct.get(token, 0)
        metrics[token] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
        }

    return metrics, confusion_counts, all_preds, all_targets


def print_predicate_table(metrics, header="Predicate-wise Validation Metrics"):
    print(f"\n  {header}")
    print(f"  {'Predicate':<20} {'Total':>8} {'Correct':>8} {'Accuracy':>10}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
    sorted_preds = sorted(metrics.items(), key=lambda x: -x[1]["total"])
    for token, m in sorted_preds:
        if token in (Vocab.PAD, Vocab.UNK):
            continue
        print(f"  {token:<20} {m['total']:>8} {m['correct']:>8} {m['accuracy']:>10.3f}")


def print_confusion_analysis(confusion_counts, pred_vocab, top_n=10):
    print(f"\n  Top {top_n} Most Confused Predicate Pairs (true -> predicted):")
    print(f"  {'True Pred':<20} {'Predicted As':<20} {'Count':>8}")
    print(f"  {'-'*20} {'-'*20} {'-'*8}")

    all_confusions = []
    for true_token, pred_dict in confusion_counts.items():
        for pred_token, count in pred_dict.items():
            if pred_token != true_token:
                all_confusions.append((true_token, pred_token, count))

    all_confusions.sort(key=lambda x: -x[2])
    for true_tok, pred_tok, count in all_confusions[:top_n]:
        print(f"  {true_tok:<20} {pred_tok:<20} {count:>8}")

    print(f"\n  Most Dominant Predicates (by total frequency):")
    totals = {}
    for true_token, pred_dict in confusion_counts.items():
        total_preds = sum(pred_dict.values())
        totals[true_token] = total_preds
    sorted_totals = sorted(totals.items(), key=lambda x: -x[1])
    for token, count in sorted_totals[:10]:
        print(f"  {token:<20} {count:>8}")


def qualitative_test(model, label_vocab, pred_vocab, device, has_visual, top_k=5,
                     geo_mode="basic"):
    """Probe the model on canonical subject/object layouts.

    The geometry probe is built from actual boxes through the dataset's own
    extractor rather than being hard-coded, so it stays correct for any
    geo_mode (the old literal 5-element vectors crashed a 19-dim model), and
    the zero visual features are sized from the model rather than assumed 768.
    """
    print(f"\n  Qualitative Relation Predictions (top-{top_k}):")
    print(f"  {'Subject':<12} {'Object':<14} {'Predictions':<60}")
    print(f"  {'-'*12} {'-'*14} {'-'*60}")

    geo_fn, _ = geo_extractor(geo_mode)
    # Subject and object of similar size, object slightly above and overlapping
    # the subject — a neutral "interacting" layout on a 640x480 frame.
    probe_img_w, probe_img_h = 640.0, 480.0
    subj_box = (220.0, 180.0, 380.0, 420.0)
    obj_box = (240.0, 140.0, 400.0, 380.0)
    geo_default = torch.tensor(
        [geo_fn(subj_box, obj_box, probe_img_w, probe_img_h)],
        dtype=torch.float32, device=device,
    )

    clip_dim = getattr(model, "clip_dim", 0)
    union_dim = getattr(model, "union_dim", 0)
    pose_dim = getattr(model, "pose_dim", 0)

    model.eval()
    with torch.no_grad():
        for subj_name, obj_name in QUALITATIVE_PAIRS:
            s_idx = label_vocab[subj_name]
            o_idx = label_vocab[obj_name]
            s_t = torch.tensor([s_idx], dtype=torch.long, device=device)
            o_t = torch.tensor([o_idx], dtype=torch.long, device=device)

            kwargs = {}
            if has_visual and clip_dim:
                kwargs["subj_feat"] = torch.zeros((1, clip_dim), device=device)
                kwargs["obj_feat"] = torch.zeros((1, clip_dim), device=device)
            if union_dim:
                kwargs["union_feat"] = torch.zeros((1, union_dim), device=device)
            if pose_dim:
                kwargs["pose_feat"] = torch.zeros((1, pose_dim), device=device)

            logits = model(s_t, o_t, geo_default, **kwargs)

            probs = F.softmax(logits, dim=-1)
            top_probs, top_idxs = probs[0].topk(top_k)
            pred_str = " | ".join(
                f"{pred_vocab.token(int(idx))} ({prob:.3f})"
                for idx, prob in zip(top_idxs, top_probs)
            )
            print(f"  {subj_name:<12} {obj_name:<14} {pred_str:<60}")


def analyze_clip_coverage(full_ds):
    print(f"\n  CLIP Coverage Analysis:")
    total = len(full_ds)
    real_clip = 0
    zero_clip = 0
    for idx in range(min(total, 5000)):
        item = full_ds[idx]
        if len(item) > 4:
            sf, of = item[4], item[5]
            if sf.sum().item() == 0 and of.sum().item() == 0:
                zero_clip += 1
            else:
                real_clip += 1
    if total > 5000:
        ratio = real_clip / max(real_clip + zero_clip, 1)
        real_clip = int(total * ratio)
        zero_clip = total - real_clip
        print(f"  (extrapolated from first 5000 samples, ratio={ratio:.3f})")

    coverage_pct = 100.0 * real_clip / max(total, 1)
    print(f"  Total samples:          {total:,}")
    print(f"  With real CLIP embeds:  {real_clip:,} ({coverage_pct:.1f}%)")
    print(f"  Fallback zero vectors:  {zero_clip:,} ({100 - coverage_pct:.1f}%)")
    return real_clip, zero_clip, coverage_pct


# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Frozen image-disjoint split (E0 protocol)
# ---------------------------------------------------------------------------

def _sample_image_ids(dataset):
    """Recover the VG image_id of every RETAINED dataset sample.

    VGRelationshipDataset does not expose image_id directly, but sample_keys
    carries it as "<image_id>_obj_<object_id>" (built in _load and kept in
    sync by _filter_strict_visual). Returns a list aligned with
    dataset.samples; None where no object_id was present in the annotation.
    """
    ids = []
    for subj_key, obj_key in dataset.sample_keys:
        key = subj_key or obj_key
        if key and "_obj_" in key:
            try:
                ids.append(int(key.split("_obj_")[0]))
            except ValueError:
                ids.append(None)
        else:
            ids.append(None)
    return ids


def split_by_manifest(dataset, manifest_path):
    """Split the dataset with the frozen E0 image-disjoint manifest.

    The manifest is authoritative and is never regenerated here. Assignment
    uses the FINAL retained samples' image_id, so no sample from a test image
    can reach train or validation regardless of the dataset's own filtering.

    Returns (train_subset, val_subset, info).
    """
    if not os.path.isfile(manifest_path):
        raise SystemExit(f"[split] Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for key in ("train_ids", "val_ids", "test_ids"):
        if key not in manifest:
            raise SystemExit(f"[split] Manifest missing '{key}': {manifest_path}")

    train_ids = {int(i) for i in manifest["train_ids"]}
    val_ids = {int(i) for i in manifest["val_ids"]}
    test_ids = {int(i) for i in manifest["test_ids"]}

    overlap_tr_va = len(train_ids & val_ids)
    overlap_tr_te = len(train_ids & test_ids)
    overlap_va_te = len(val_ids & test_ids)
    overlap_total = overlap_tr_va + overlap_tr_te + overlap_va_te

    sample_iids = _sample_image_ids(dataset)
    train_idx, val_idx = [], []
    n_test_excluded = 0
    n_unattributed = 0
    train_imgs, val_imgs = set(), set()

    for i, iid in enumerate(sample_iids):
        if iid is None:
            n_unattributed += 1
            continue
        if iid in train_ids:
            train_idx.append(i)
            train_imgs.add(iid)
        elif iid in val_ids:
            val_idx.append(i)
            val_imgs.add(iid)
        elif iid in test_ids:
            n_test_excluded += 1
        else:
            n_unattributed += 1

    print(f"\n{'=' * 78}")
    print("  FROZEN IMAGE-DISJOINT SPLIT (E0 protocol)")
    print(f"{'=' * 78}")
    print(f"  Manifest:                 {manifest_path}")
    print(f"  Manifest seed:            {manifest.get('seed')}")
    print(f"  Split unit:               {manifest.get('split_unit')}")
    print(f"  Train image count:        {len(train_imgs):,}  (manifest: {len(train_ids):,})")
    print(f"  Validation image count:   {len(val_imgs):,}  (manifest: {len(val_ids):,})")
    print(f"  Train sample count:       {len(train_idx):,}")
    print(f"  Validation sample count:  {len(val_idx):,}")
    print(f"  Overlap count:            {overlap_total}"
          f"  (train n val={overlap_tr_va}, train n test={overlap_tr_te}, val n test={overlap_va_te})")
    print(f"  Test samples excluded:    {n_test_excluded:,}  (held out, never seen in training)")
    print(f"  Unattributed samples:     {n_unattributed:,}  (no image_id -> dropped)")

    if overlap_total != 0:
        print("\n[split] FATAL: manifest image sets overlap. Aborting.", file=sys.stderr)
        raise SystemExit(2)

    realised_overlap = len(train_imgs & val_imgs)
    if realised_overlap != 0:
        print(f"\n[split] FATAL: {realised_overlap} images landed in both train and val. Aborting.",
              file=sys.stderr)
        raise SystemExit(2)

    if not train_idx or not val_idx:
        print("\n[split] FATAL: train or validation split is empty. Aborting.", file=sys.stderr)
        raise SystemExit(2)

    info = {
        "manifest_path": manifest_path,
        "manifest_seed": manifest.get("seed"),
        "n_images_train": len(train_imgs),
        "n_images_val": len(val_imgs),
        "n_samples_train": len(train_idx),
        "n_samples_val": len(val_idx),
        "overlap_count": overlap_total,
        "n_test_samples_excluded": n_test_excluded,
        "n_unattributed_samples": n_unattributed,
    }
    return Subset(dataset, train_idx), Subset(dataset, val_idx), info


def main():
    print("=" * 78)
    print("  FULL VISUAL-SEMANTIC RELATION MLP TRAINING")
    print("=" * 78)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(SEED)

    # -----------------------------------------------------------------------
    # STEP 1 - Dataset & Configuration Verification
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 1 — TRAINING CONFIGURATION")
    print(f"{'=' * 78}")

    # Determine training mode
    mode_parts = []
    if USE_VISUAL and REQUIRE_VISUAL:
        mode_parts.append("PURE VISUAL-SEMANTIC (strict)")
    elif USE_VISUAL:
        mode_parts.append("VISUAL-SEMANTIC (mixed/fallback)")
    else:
        mode_parts.append("GEOMETRY-ONLY")
    if VISUAL_FILTER_ONLY:
        mode_parts.append("clip cache used for FILTERING ONLY (clip_dim=0)")
    if USE_POSE:
        mode_parts.append("pose")
    if USE_UNION:
        mode_parts.append("union")
    mode_label = " + ".join(mode_parts)
    print(f"\n  *** MODE: {mode_label} ***")

    t0 = time.time()
    full_ds = VGRelationshipDataset(
        relationships_json=str(VG_ROOT / "relationships.json"),
        image_data_json=str(VG_ROOT / "image_data.json"),
        vg_image_dir=str(VG_IMAGE_DIR) if USE_VISUAL else None,
        min_pred_count=MIN_PRED_COUNT,
        max_samples=MAX_SAMPLES,
        use_visual=USE_VISUAL,
        clip_cache_path=str(CLIP_CACHE_PATH) if USE_VISUAL else None,
        require_visual=REQUIRE_VISUAL,
        use_pose=USE_POSE,
        use_union=USE_UNION,
        geo_mode=GEO_MODE,
        predicate_scheme=PREDICATE_SCHEME,
    )
    load_time = time.time() - t0

    label_vocab = full_ds.label_vocab
    pred_vocab = full_ds.pred_vocab

    dataset_size = len(full_ds)
    num_labels = len(label_vocab)
    num_predicates = len(pred_vocab)
    dims = resolve_feature_dims(
        use_visual=USE_VISUAL, visual_filter_only=VISUAL_FILTER_ONLY,
        use_union=USE_UNION, use_pose=USE_POSE, geo_mode=GEO_MODE,
    )
    clip_dim, pose_dim, union_dim = dims["clip_dim"], dims["pose_dim"], dims["union_dim"]

    # CLIP coverage
    print(f"\n  Dataset Statistics:")
    print(f"  Dataset size:              {dataset_size:,} samples")
    print(f"  Load time:                 {load_time:.1f}s")
    print(f"  Number of labels:          {num_labels}")
    print(f"  Number of predicates:      {num_predicates}")
    print(f"  CLIP dimension:            {clip_dim}")
    print(f"  Pose dimension:            {pose_dim}")
    print(f"  Union dimension:           {union_dim}")
    print(f"  Use visual:                {USE_VISUAL}")
    print(f"  Require visual:            {REQUIRE_VISUAL}")
    print(f"  Use pose:                  {USE_POSE}")
    print(f"  Use union:                 {USE_UNION}")
    print(f"  Visual filter only:        {VISUAL_FILTER_ONLY}")

    # Pre-training validation: verify no zero embeddings in pure visual mode
    if REQUIRE_VISUAL and USE_VISUAL:
        _validate_pure_visual(full_ds)
    elif USE_VISUAL and not REQUIRE_VISUAL:
        real_clip, zero_clip, clip_coverage = analyze_clip_coverage(full_ds)

    geo_dim = dims["geo_dim"]
    if full_ds.geo_dim != geo_dim:
        raise SystemExit(f"[train] dataset geometry width {full_ds.geo_dim} != "
                         f"{geo_dim} expected for geo_mode={GEO_MODE}")

    if MODEL_TYPE == "transformer":
        model = RelationTransformer(
            num_labels=num_labels,
            num_predicates=num_predicates,
            d_model=D_MODEL,
            embed_dim=EMBED_DIM,
            clip_dim=clip_dim,
            pose_dim=pose_dim,
            union_dim=union_dim,
            geo_dim=geo_dim,
            geo_norm=GEO_NORM,
            dropout=DROPOUT,
        ).to(device)
    else:
        model = RelationMLP(
            num_labels=num_labels,
            num_predicates=num_predicates,
            embed_dim=EMBED_DIM,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
            clip_dim=clip_dim,
            pose_dim=pose_dim,
            union_dim=union_dim,
            geo_dim=geo_dim,
            geo_norm=GEO_NORM,
        ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # RelationMLP sizes its first Linear from its own feature_blocks(); report
    # that value rather than re-deriving it, so the log cannot disagree with
    # the weights.
    input_dim = getattr(model, "input_dim",
                        2 * EMBED_DIM + geo_dim + 2 * clip_dim + union_dim + pose_dim)

    print(f"\n  Model Configuration:")
    print(f"  Model type:               {MODEL_TYPE}")
    print(f"  Input dimension:          {input_dim}")
    print(f"  Embedding dimension:      {EMBED_DIM}")
    print(f"  Hidden dims:              {HIDDEN_DIMS if MODEL_TYPE == 'mlp' else 'N/A'}")
    print(f"  Dropout:                  {DROPOUT}")
    print(f"  Geometry mode:            {GEO_MODE} ({geo_dim}-dim)")
    print(f"  Predicate scheme:         {PREDICATE_SCHEME}")
    print(f"  Geometry BatchNorm:       {GEO_NORM}")
    print(f"  Loss:                     {LOSS}")
    print(f"  Class-weight alpha:       {CLASS_WEIGHT_ALPHA}")
    print(f"  Selection metric:         {SELECT_METRIC}")
    print(f"  Parameters:               {param_count:,}")
    print(f"  Batch size:               {BATCH_SIZE}")
    print(f"  Epochs:                   {EPOCHS}")
    print(f"  Learning rate:            {LR}")
    print(f"  Weight decay:             {WEIGHT_DECAY}")
    print(f"  Validation fraction:      {VAL_FRACTION}")

    # -----------------------------------------------------------------------
    # STEP 2 - Data Splits & Loaders
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 2 — DATA SPLITS")
    print(f"{'=' * 78}")

    if SPLIT_MANIFEST:
        # Frozen image-disjoint split (E0 protocol). Authoritative; never
        # regenerated. Test image IDs are excluded entirely.
        train_ds, val_ds, _split_info = split_by_manifest(full_ds, SPLIT_MANIFEST)
    else:
        _split_info = {"manifest_path": None,
                       "note": "sample-level random_split (image-leaking)"}
        # Legacy behaviour: sample-level random split (image-leaking).
        n_val = max(1, int(dataset_size * VAL_FRACTION))
        n_train = dataset_size - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(SEED),
        )
        print("  [WARNING] sample-level random_split: train/val share images (leakage).")
    print(f"  Training samples:   {len(train_ds):,}")
    print(f"  Validation samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=_collate,
    )

    # One definition of the batch layout for every consumer below: preflight,
    # the training loop, per-epoch validation and the best-model analysis.
    feature_flags = dict(has_visual=USE_VISUAL, use_union=USE_UNION, use_pose=USE_POSE)
    preflight_feature_dims(model, {"train": train_ds, "val": val_ds}, device,
                           **feature_flags)

    # -----------------------------------------------------------------------
    # STEP 3 - Training Setup
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 3 — TRAINING VISUAL-SEMANTIC MLP")
    print(f"{'=' * 78}")

    # Class statistics must come from the TRAIN SPLIT, not from the whole
    # pre-filter corpus: with a frozen image-disjoint manifest the two differ,
    # and weighting a loss by counts the model never sees is simply wrong.
    train_pred_counter = Counter()
    for i in getattr(train_ds, "indices", range(len(full_ds))):
        train_pred_counter[pred_vocab.token(int(full_ds.samples[i][3]))] += 1

    if CLASS_WEIGHT_ALPHA is None:
        class_weights = compute_class_weights(train_pred_counter, pred_vocab, num_predicates)
        weight_desc = "effective-number, beta=0.999"
    elif CLASS_WEIGHT_ALPHA == 0.0:
        class_weights = None
        weight_desc = "none (unweighted)"
    else:
        class_weights = compute_inverse_frequency_weights(
            train_pred_counter, pred_vocab, num_predicates, alpha=CLASS_WEIGHT_ALPHA,
        )
        weight_desc = f"inverse-frequency ** {CLASS_WEIGHT_ALPHA}"

    print(f"\n  Train-split predicate counts, class weights ({weight_desc}):")
    for i in range(num_predicates):
        tok = pred_vocab.token(i)
        if tok in (Vocab.PAD, Vocab.UNK):
            continue
        w = class_weights[i].item() if class_weights is not None else 1.0
        print(f"    {tok:<15} count={train_pred_counter.get(tok, 0):<8} weight={w:.4f}")

    alpha_t = class_weights.to(device) if class_weights is not None else None
    if LOSS == "focal":
        criterion = FocalLoss(gamma=2.0, alpha=alpha_t, ignore_index=0)
    elif LOSS == "ce":
        criterion = nn.CrossEntropyLoss(weight=alpha_t, ignore_index=0)
    else:
        raise SystemExit(f"[train] unknown --loss {LOSS!r}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    CLF_L2_WEIGHT = 1e-4

    os.makedirs(str(CHECKPOINT_DIR), exist_ok=True)
    best_val_acc = 0.0
    best_select_score = -1.0
    best_epoch = 0
    has_visual = USE_VISUAL
    all_batch_times = []
    epoch_metrics_log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()

        for batch in train_loader:
            batch_start = time.time()
            subj, obj, geo, pred, subj_feat, obj_feat, union_feat, pose_feat = _unpack_batch(
                batch, device, **feature_flags)

            optimizer.zero_grad()

            logits = model(subj, obj, geo,
                           subj_feat=subj_feat, obj_feat=obj_feat,
                           union_feat=union_feat, pose_feat=pose_feat)

            loss = criterion(logits, pred)
            loss = loss + classifier_l2_loss(model, weight=CLF_L2_WEIGHT)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_time = time.time() - batch_start
            all_batch_times.append(batch_time)

            train_loss += loss.item() * pred.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == pred).sum().item()
            train_total += pred.size(0)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        train_acc = train_correct / max(train_total, 1)
        avg_loss = train_loss / max(train_total, 1)

        # Validation
        val_metrics, confusion_counts, val_preds, val_targets = compute_predicate_metrics(
            model, val_loader, device, pred_vocab, **feature_flags,
        )

        overall_val_correct = sum(m["correct"] for m in val_metrics.values())
        overall_val_total = sum(m["total"] for m in val_metrics.values())
        val_acc = overall_val_correct / max(overall_val_total, 1)
        val_macro_f1 = macro_f1_from_metrics(
            val_metrics, val_preds, val_targets, pred_vocab,
        )
        select_score = val_macro_f1 if SELECT_METRIC == "macro_f1" else val_acc

        epoch_log = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "lr": scheduler.get_last_lr()[0],
        }
        epoch_metrics_log.append(epoch_log)

        print(f"\n  Epoch {epoch:3d}/{EPOCHS} | "
              f"loss {avg_loss:.4f} | "
              f"train {train_acc:.3f} | "
              f"val {val_acc:.3f} | mF1 {val_macro_f1:.3f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | "
              f"{epoch_time:.1f}s")

        # Semantic predicates
        print(f"  Semantic predicates:")
        for sp in sorted(SEMANTIC_PREDICATES):
            m = val_metrics.get(sp, {"total": 0, "correct": 0, "accuracy": 0.0})
            marker = " ***" if m["total"] > 0 else ""
            print(f"    {sp:<15} acc={m['accuracy']:.3f}  ({m['correct']}/{m['total']}){marker}")

        if select_score > best_select_score:
            best_select_score = select_score
            best_val_acc = val_acc
            best_epoch = epoch
            _save_checkpoint(model, label_vocab, pred_vocab, str(CHECKPOINT_DIR), epoch, val_acc,
                             dataset=full_ds, mode_label=mode_label,
                             val_macro_f1=val_macro_f1, split_info=_split_info)
            print(f"  >>> New best model saved ({SELECT_METRIC}={select_score:.4f}, "
                  f"val_acc={val_acc:.3f}, val_macro_f1={val_macro_f1:.3f}, epoch={epoch})")

    # -----------------------------------------------------------------------
    # STEP 4 - Training Complete
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 4 — TRAINING COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Best validation accuracy: {best_val_acc:.3f} (epoch {best_epoch})")

    # -----------------------------------------------------------------------
    # Confusion Analysis
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 5 — CONFUSION ANALYSIS (best model)")
    print(f"{'=' * 78}")

    _load_best_model(model, str(CHECKPOINT_DIR), device)
    # This call used to omit use_union/use_pose: the union arm's best model
    # was handed union_feat=None here, after every epoch had finished.
    best_metrics, best_confusion, _, _ = compute_predicate_metrics(
        model, val_loader, device, pred_vocab, **feature_flags,
    )

    print_predicate_table(best_metrics, "Per-Predicate Validation Accuracy (Best Model)")
    print_confusion_analysis(best_confusion, pred_vocab, top_n=15)

    # -----------------------------------------------------------------------
    # STEP 6 - Qualitative Tests
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 6 — QUALITATIVE RELATION TESTS")
    print(f"{'=' * 78}")

    qualitative_test(model, label_vocab, pred_vocab, device, has_visual, top_k=5,
                     geo_mode=GEO_MODE)

    # -----------------------------------------------------------------------
    # STEP 7 - CLIP Analysis
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 7 — CLIP IMPACT ANALYSIS")
    print(f"{'=' * 78}")

    # Only meaningful when the model actually consumed CLIP. Called
    # unconditionally, this section asserted "CLIP features ARE contributing"
    # in runs built with clip_dim=0, which is a claim about a feature the model
    # never saw.
    if getattr(model, "clip_dim", 0) > 0:
        analyze_clip_impact(best_metrics, full_ds)

    # -----------------------------------------------------------------------
    # STEP 8 - Save Artifacts
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 7b — SAVING TRAINING ARTIFACTS")
    print(f"{'=' * 78}")

    _save_training_logs(epoch_metrics_log, str(CHECKPOINT_DIR))
    _save_validation_metrics(best_metrics, str(CHECKPOINT_DIR))
    _save_confusion_analysis(best_confusion, pred_vocab, str(CHECKPOINT_DIR))

    print(f"  Checkpoints saved to: {CHECKPOINT_DIR}")
    ckpt_files = os.listdir(str(CHECKPOINT_DIR))
    for f in sorted(ckpt_files):
        fpath = os.path.join(str(CHECKPOINT_DIR), f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:<30} {size_kb:>8.1f} KB")

    # -----------------------------------------------------------------------
    # STEP 8 - FINAL REPORT
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("  STEP 8 — FINAL REPORT")
    print(f"{'=' * 78}")

    generate_final_report(
        model, best_metrics, val_loader, device, pred_vocab, has_visual, label_vocab,
        all_batch_times, dataset_size, EPOCHS, best_val_acc, best_epoch,
    )


# ---------------------------------------------------------------------------
# Pure Visual Pre-training Validation
# ---------------------------------------------------------------------------

def _validate_pure_visual(dataset: VGRelationshipDataset) -> None:
    """Verify no zero-vector samples leak through in pure visual mode."""
    print(f"\n{'=' * 65}")
    print("  PRE-TRAINING PURE VISUAL VALIDATION")
    print(f"{'=' * 65}")
    total = len(dataset)
    print(f"  Total samples after strict filtering: {total}")

    zero_samples = 0
    subj_norms = []
    obj_norms = []
    for idx in range(total):
        item = dataset[idx]
        sf = item[4]
        of = item[5]
        sn = sf.norm().item()
        on = of.norm().item()
        subj_norms.append(sn)
        obj_norms.append(on)
        if sn < 0.001 or on < 0.001:
            zero_samples += 1

    all_norms = subj_norms + obj_norms
    mean_norm = sum(all_norms) / len(all_norms) if all_norms else 0.0
    min_norm = min(all_norms) if all_norms else 0.0
    max_norm = max(all_norms) if all_norms else 0.0

    print(f"  Zero-vector samples:              {zero_samples}/{total * 2} features")
    print(f"  Mean feature norm:                {mean_norm:.4f}")
    print(f"  Min feature norm:                 {min_norm:.4f}")
    print(f"  Max feature norm:                 {max_norm:.4f}")

    assert zero_samples == 0, (
        f"[PURE VISUAL] Found {zero_samples} zero-vector features in strict mode! "
        "Samples with missing CLIP embeddings leaked through."
    )

    # Predicate distribution
    pred_counter = Counter()
    for idx in range(total):
        item = dataset[idx]
        pred_name = dataset.pred_vocab.token(item[3].item())
        pred_counter[pred_name] += 1

    print(f"  Real CLIP coverage:              100.00% (strict mode)")
    print(f"  Retained sample count:           {total}")
    print(f"\n  Predicate distribution (pure visual):")
    for pred, count in pred_counter.most_common():
        print(f"    {pred}: {count}")
    print(f"{'=' * 65}\n")


# ---------------------------------------------------------------------------
# Checkpoint Helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model, label_vocab, pred_vocab, ckpt_dir, epoch, val_acc,
                     dataset=None, mode_label="unknown", val_macro_f1=None,
                     split_info=None):
    state = model.state_dict()
    model_type = "transformer" if isinstance(model, RelationTransformer) else "mlp"

    if model_type == "transformer":
        config = {
            "model_type": "transformer",
            "num_labels": model.label_emb.num_embeddings,
            "num_predicates": model.num_predicates,
            "d_model": model.d_model,
            "embed_dim": model.embed_dim,
            "clip_dim": model.clip_dim,
            "pose_dim": model.pose_dim,
            "union_dim": model.union_dim,
        }
    else:
        config = {
            "model_type": "mlp",
            "num_labels": model.label_emb.num_embeddings,
            "num_predicates": model.mlp[-1].out_features,
            "embed_dim": model.label_emb.embedding_dim,
            "clip_dim": model.clip_dim,
            "pose_dim": model.pose_dim,
            "union_dim": model.union_dim,
        }
        # RelationMLP lays its layers out as Linear/ReLU/Dropout triples, so the
        # Linear weights are mlp.0, mlp.3, mlp.6, ... and hidden_dims is the
        # out_features of every Linear EXCEPT the final classifier.
        #
        # The previous version skipped "mlp.0.weight" and kept everything else,
        # which dropped the first hidden width and kept the output width: a
        # (256, 128) model was recorded as hidden_dims=[128, 21]. Every
        # checkpoint written by this script carries that wrong value. Nothing
        # currently reads it back (predict.py and eval_gt_relations.py both
        # re-derive the widths from the weights via _infer_hidden_dims), so no
        # published number is affected — but model_config was not describing
        # the model, and rebuilding from it raised a shape error.
        hdims = []
        idx = 0
        while f"mlp.{idx}.weight" in state:
            hdims.append(int(state[f"mlp.{idx}.weight"].shape[0]))
            idx += 3
        config["hidden_dims"] = hdims[:-1]

    # Geometry descriptor width and normalisation are part of the model
    # signature: without them a 19-dim checkpoint silently mismatches a 5-dim
    # feature builder at load time.
    config["geo_dim"] = getattr(model, "geo_dim", GEO_DIM)
    config["geo_norm"] = getattr(model, "geo_norm", None) is not None
    config["geo_mode"] = GEO_MODE
    config["predicate_scheme"] = PREDICATE_SCHEME
    # The evaluator decides whether to restrict the test set to the
    # visual-complete population from the model_config alone, so the geometry
    # control has to advertise that it was trained on the filtered population
    # even though its clip_dim is 0. Without this it would be scored on all
    # 10,227 test pairs while the CLIP variants are scored on the subset, and
    # the ablation would be comparing two different test sets.
    config["visual_filter_only"] = VISUAL_FILTER_ONLY
    config["require_visual"] = REQUIRE_VISUAL
    # The first layer's exact width and the blocks that make it up, read off
    # the model. eval_gt_relations.py refuses a checkpoint whose rebuilt model
    # disagrees with input_dim.
    if hasattr(model, "feature_blocks"):
        config["input_dim"] = model.input_dim
        config["feature_blocks"] = [[name, width] for name, width in model.feature_blocks()]

    torch.save({"model_state_dict": state, "model_config": config},
               os.path.join(ckpt_dir, "relation_mlp.pt"))
    label_vocab.save(os.path.join(ckpt_dir, "label_vocab.json"))
    pred_vocab.save(os.path.join(ckpt_dir, "pred_vocab.json"))

    meta = {
        "epoch": epoch,
        "val_acc": val_acc,
        "timestamp": time.time(),
        "mode": mode_label,
        "model_type": model_type,
        "use_visual": USE_VISUAL,
        "require_visual": REQUIRE_VISUAL,
        "visual_filter_only": VISUAL_FILTER_ONLY,
        "use_pose": USE_POSE,
        "use_union": USE_UNION,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "embed_dim": EMBED_DIM,
        "hidden_dims": list(HIDDEN_DIMS) if model_type == "mlp" else None,
        "d_model": model.d_model if model_type == "transformer" else None,
        "dropout": DROPOUT,
        "seed": SEED,
        "geo_mode": GEO_MODE,
        "predicate_scheme": PREDICATE_SCHEME,
        "geo_dim": config["geo_dim"],
        "geo_norm": config["geo_norm"],
        # Read off the model: the geometry control loads CLIP but has
        # clip_dim=0, and this used to record 768 for it.
        "clip_dim": getattr(model, "clip_dim", 0),
        "union_dim": getattr(model, "union_dim", 0),
        "pose_dim": getattr(model, "pose_dim", 0),
        "input_dim": getattr(model, "input_dim", None),
        "loss": LOSS,
        "class_weight_alpha": CLASS_WEIGHT_ALPHA,
        "select_metric": SELECT_METRIC,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "val_macro_f1": val_macro_f1,
        "split": split_info,
        "val_acc_note": (
            "Validation numbers are NOT comparable to the frozen E0 test "
            "numbers. Report test metrics from eval_gt_relations.py."
        ),
    }
    if dataset is not None:
        meta["dataset_size"] = len(dataset)
        meta["num_labels"] = len(dataset.label_vocab)
        meta["num_predicates"] = len(dataset.pred_vocab)
        # Real CLIP coverage
        if USE_VISUAL:
            real, total, pct = dataset.compute_clip_coverage()
            meta["real_clip_coverage_pct"] = round(pct, 2)
            meta["real_clip_samples"] = real
            meta["total_samples"] = total
        meta["retained_sample_count"] = len(dataset)
        # Predicate distribution. Read the raw sample tuples rather than calling
        # dataset[idx], which materialises (and clones) the CLIP tensors for
        # every sample — that ran on every new-best epoch and dominated the
        # checkpoint-saving cost in visual mode.
        pred_counter = Counter(
            dataset.pred_vocab.token(int(sample[3])) for sample in dataset.samples
        )
        meta["predicate_distribution"] = dict(pred_counter.most_common())

    with open(os.path.join(ckpt_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _load_best_model(model, ckpt_dir, device):
    """Reload the best checkpoint written by _save_checkpoint.

    _save_checkpoint writes {"model_state_dict": ..., "model_config": ...}, but
    this helper used to hand that whole wrapper dict to load_state_dict(), which
    raises on the unexpected "model_config" key and aborted the run before the
    post-training analysis and artifact-saving steps. Unwrap it, and keep
    accepting a bare state_dict for older checkpoints.
    """
    ckpt_path = os.path.join(ckpt_dir, "relation_mlp.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [warn] no checkpoint at {ckpt_path}; keeping in-memory weights")
        return
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model.load_state_dict(state)
    model.eval()


def _save_training_logs(epoch_logs, ckpt_dir):
    with open(os.path.join(ckpt_dir, "training_logs.json"), "w") as f:
        json.dump(epoch_logs, f, indent=2)


def _save_validation_metrics(metrics, ckpt_dir):
    serializable = {}
    for token, m in metrics.items():
        serializable[token] = {
            "total": m["total"],
            "correct": m["correct"],
            "accuracy": round(m["accuracy"], 4),
        }
    with open(os.path.join(ckpt_dir, "validation_metrics.json"), "w") as f:
        json.dump(serializable, f, indent=2)


def _save_confusion_analysis(confusion_counts, pred_vocab, ckpt_dir, top_n=20):
    all_confusions = []
    for true_token, pred_dict in confusion_counts.items():
        for pred_token, count in pred_dict.items():
            if pred_token != true_token:
                all_confusions.append({
                    "true": true_token,
                    "predicted": pred_token,
                    "count": count,
                })
    all_confusions.sort(key=lambda x: -x["count"])
    with open(os.path.join(ckpt_dir, "confusion_analysis.json"), "w") as f:
        json.dump({"top_confusions": all_confusions[:top_n]}, f, indent=2)


# ---------------------------------------------------------------------------
# CLIP Impact Analysis
# ---------------------------------------------------------------------------

def analyze_clip_impact(best_metrics, full_ds):
    semantic = ["riding", "carrying", "holding", "wearing", "sitting on", "standing on"]
    spatial = ["on", "near", "under", "above", "next to", "behind", "in front of", "over", "inside"]

    print(f"\n  Semantic Predicate Performance (CLIP-informed):")
    sem_total = 0
    sem_correct = 0
    for p in semantic:
        m = best_metrics.get(p, {"total": 0, "correct": 0, "accuracy": 0.0})
        sem_total += m["total"]
        sem_correct += m["correct"]
        print(f"    {p:<15} acc={m['accuracy']:.3f}  ({m['correct']}/{m['total']})")
    print(f"    {'-- All Semantic --':<15} acc={sem_correct / max(sem_total, 1):.3f}  ({sem_correct}/{sem_total})")

    print(f"\n  Spatial Predicate Performance (geometry-dominated):")
    spa_total = 0
    spa_correct = 0
    for p in spatial:
        m = best_metrics.get(p, {"total": 0, "correct": 0, "accuracy": 0.0})
        spa_total += m["total"]
        spa_correct += m["correct"]
        print(f"    {p:<15} acc={m['accuracy']:.3f}  ({m['correct']}/{m['total']})")
    print(f"    {'-- All Spatial --':<15} acc={spa_correct / max(spa_total, 1):.3f}  ({spa_correct}/{spa_total})")

    print(f"\n  Assessment:")
    if sem_total > 0:
        sem_acc = sem_correct / sem_total
    else:
        sem_acc = 0.0
    if spa_total > 0:
        spa_acc = spa_correct / spa_total
    else:
        spa_acc = 0.0

    print(f"    Semantic accuracy:  {sem_acc:.3f}")
    print(f"    Spatial accuracy:   {spa_acc:.3f}")
    print(f"    Gap:                {abs(sem_acc - spa_acc):.3f}")

    if sem_acc > spa_acc * 0.5:
        print(f"    CLIP features ARE contributing to semantic predictions.")
    else:
        print(f"    CLIP features have limited impact on semantic predictions.")
    print(f"    Geometry still dominates overall accuracy due to spatial predicate frequency.")


# ---------------------------------------------------------------------------
# Final Report
# ---------------------------------------------------------------------------

def generate_final_report(
    model, best_metrics, val_loader, device, pred_vocab, has_visual, label_vocab,
    all_batch_times, dataset_size, epochs, best_val_acc, best_epoch,
):
    # Read the feature configuration off the model rather than restating it:
    # this block used to print a hardcoded "visual-semantic (1669-dim input)"
    # no matter how the run was configured.
    clip_dim = int(getattr(model, "clip_dim", 0) or 0)
    union_dim = int(getattr(model, "union_dim", 0) or 0)
    pose_dim = int(getattr(model, "pose_dim", 0) or 0)
    geo_dim = int(getattr(model, "geo_dim", 0) or 0)
    in_dim = (model.mlp[0].weight.shape[1] if hasattr(model, "mlp")
              else 2 * EMBED_DIM + geo_dim + 2 * clip_dim + union_dim + pose_dim)
    mode = "+".join(["labels"]
                    + (["geo"] if geo_dim else [])
                    + (["clip"] if clip_dim else [])
                    + (["union"] if union_dim else [])
                    + (["pose"] if pose_dim else []))

    avg_batch_time = sum(all_batch_times) / max(len(all_batch_times), 1)
    samples_per_sec = BATCH_SIZE / max(avg_batch_time, 1e-6)
    total_train_time = sum(all_batch_times)

    print(f"\n  1. Final Validation Metrics:")
    overall_correct = sum(m["correct"] for m in best_metrics.values())
    overall_total = sum(m["total"] for m in best_metrics.values())
    print(f"     Overall accuracy:       {overall_correct / max(overall_total, 1):.4f}")
    print(f"     Best epoch:             {best_epoch}")
    print(f"     Best val accuracy:      {best_val_acc:.4f}")

    print(f"\n  2. Predicate-Wise Metrics:")
    print_predicate_table(best_metrics, "")

    print(f"\n  3. Training Speed:")
    print(f"     Batch size:             {BATCH_SIZE}")
    print(f"     Total batches:          {len(all_batch_times)}")
    print(f"     Avg batch time:         {avg_batch_time:.4f}s")
    print(f"     Samples/sec:            {samples_per_sec:.1f}")
    print(f"     Total training time:    {total_train_time:.1f}s")
    print(f"     Epochs:                 {epochs}")

    print(f"\n  4. GPU Utilization Summary:")
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e6
        mem_reserved = torch.cuda.memory_reserved() / 1e6
        mem_peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"     VRAM allocated:         {mem_allocated:.0f} MB")
        print(f"     VRAM reserved:          {mem_reserved:.0f} MB")
        print(f"     VRAM peak:              {mem_peak:.0f} MB")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e6
        print(f"     VRAM utilization:       {mem_peak / total_vram * 100:.1f}%")
    else:
        print(f"     (CPU training)")

    print(f"\n  5. Best/Worst Predicates:")
    sorted_by_acc = sorted(
        [(t, m) for t, m in best_metrics.items() if t not in (Vocab.PAD, Vocab.UNK) and m["total"] > 0],
        key=lambda x: -x[1]["accuracy"],
    )
    print(f"     Top 5 best:")
    for token, m in sorted_by_acc[:5]:
        print(f"       {token:<20} acc={m['accuracy']:.3f} ({m['correct']}/{m['total']})")
    print(f"     Bottom 5 worst:")
    for token, m in sorted_by_acc[-5:]:
        print(f"       {token:<20} acc={m['accuracy']:.3f} ({m['correct']}/{m['total']})")

    print(f"\n  6. Qualitative Prediction Examples:")
    qualitative_test(model, label_vocab, pred_vocab, device, has_visual, top_k=3,
                     geo_mode=GEO_MODE)

    print(f"\n  7. Honest Assessment of Relation Quality:")
    sem_preds = {"riding", "carrying", "holding", "wearing", "sitting on", "standing on"}
    sem_metrics = {p: best_metrics.get(p, {"total": 0, "correct": 0, "accuracy": 0.0}) for p in sem_preds}
    sem_total = sum(m["total"] for m in sem_metrics.values())
    sem_correct = sum(m["correct"] for m in sem_metrics.values())
    sem_acc = sem_correct / max(sem_total, 1)

    spatial_preds = {"on", "near", "under", "above", "next to", "behind", "in front of"}
    spa_metrics = {p: best_metrics.get(p, {"total": 0, "correct": 0, "accuracy": 0.0}) for p in spatial_preds}
    spa_total = sum(m["total"] for m in spa_metrics.values())
    spa_correct = sum(m["correct"] for m in spa_metrics.values())
    spa_acc = spa_correct / max(spa_total, 1)

    print(f"     Semantic predicates:                         acc={sem_acc:.3f} ({sem_correct}/{sem_total})")
    print(f"     Spatial predicates:                          acc={spa_acc:.3f} ({spa_correct}/{spa_total})")
    print(f"     Overall:                                     acc={overall_correct / max(overall_total, 1):.3f} ({overall_correct}/{overall_total})")

    if sem_total > 50:
        print(f"     Spatial/semantic frequency ratio:            "
              f"{spa_total / max(sem_total, 1):.1f}x")
    # Splitting accuracy by predicate group says nothing on its own about
    # whether CLIP helped. This block used to announce "CLIP features ARE
    # providing meaningful semantic signal" whenever semantic accuracy cleared
    # 0.3 - in runs with no CLIP at all. Attribution needs the matched
    # no-CLIP control, which is what run_visual_experiment.py measures.
    if clip_dim > 0:
        print("     Attribution to CLIP requires the matched geometry control; "
              "see run_visual_experiment.py --collect.")

    print(f"\n  8. Pipeline Readiness for Grounded Captioning:")
    print(f"     Model:     {'READY' if best_val_acc > 0.3 else 'NEEDS IMPROVEMENT'}")
    print(f"     Checkpoint: relation_mlp.pt directly loadable by infer_relationships_learned()")
    print(f"     Vocab:     label_vocab.json + pred_vocab.json present")
    print(f"     Mode:      {mode} ({in_dim}-dim input)")
    print(f"     Coverage:  {dataset_size:,} training samples")

    print(f"\n{'=' * 78}")
    print("  TRAINING COMPLETE")
    print(f"{'=' * 78}")


# ---------------------------------------------------------------------------
# Focal Loss with class weighting
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        if self.alpha is not None:
            alpha_w = self.alpha.gather(0, targets) * (targets != self.ignore_index).float()
            loss = alpha_w * loss
        if self.reduction == 'mean':
            valid = (targets != self.ignore_index).float()
            return loss.sum() / valid.sum().clamp(min=1)
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def compute_class_weights(pred_counter, pred_vocab, num_predicates, ignore_index=0, beta=0.999):
    counts = torch.zeros(num_predicates)
    for pred_str, count in pred_counter.items():
        idx = pred_vocab[pred_str]
        counts[idx] = count

    weights = torch.ones(num_predicates)
    for i in range(num_predicates):
        n = counts[i].item()
        if n > 0:
            weights[i] = (1.0 - beta) / (1.0 - beta ** n)
        else:
            weights[i] = 0.0

    weights[ignore_index] = 0.0
    if weights.sum() > 0:
        weights = weights / weights.sum() * num_predicates
    return weights


def compute_inverse_frequency_weights(pred_counter, pred_vocab, num_predicates,
                                      ignore_index=0, alpha=0.5):
    """Class weights proportional to (1 / count) ** alpha, normalised to mean 1.

    alpha=0 is unweighted, alpha=1 is full inverse frequency. Values around
    0.25-0.5 trade a little top-1 accuracy for a sizeable macro-F1 gain. The
    effective-number scheme (beta=0.999) is far more aggressive than that on
    this predicate distribution and costs several points of top-1.
    """
    counts = torch.zeros(num_predicates)
    for pred_str, count in pred_counter.items():
        counts[pred_vocab[pred_str]] = count

    weights = torch.zeros(num_predicates)
    valid = []
    for i in range(num_predicates):
        if i == ignore_index or pred_vocab.token(i) in (Vocab.PAD, Vocab.UNK):
            continue
        if counts[i].item() > 0:
            weights[i] = (1.0 / counts[i].item()) ** alpha
            valid.append(i)

    if valid:
        weights = weights / weights[valid].sum() * len(valid)
    weights[ignore_index] = 0.0
    return weights


def classifier_l2_loss(model, weight=1e-4):
    if isinstance(model, RelationTransformer):
        return weight * model.output.weight.norm(2).pow(2) * 0.5
    last_linear = None
    for module in model.mlp:
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        return weight * last_linear.weight.norm(2).pow(2) * 0.5
    return torch.tensor(0.0, device=next(model.parameters()).device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pure Visual-Semantic Relation MLP Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Geometry-only MLP, extended geometry, frozen E0 split (recommended baseline)
  python train_full_visual_semantic.py --no-visual --geo-mode ext --geo-norm \
      --loss ce --class-weight-alpha 0 --split-manifest splits/e0_image_split.json \
      --checkpoint-dir checkpoints_e2_geo

  # Geometry-only MLP, legacy 5-dim geometry (E0 reproduction)
  python train_full_visual_semantic.py --no-visual

  # MLP with visual-semantic features (mixed/fallback)
  python train_full_visual_semantic.py --use-visual

  # PURE visual-semantic MLP (strict, no zero-vectors allowed)
  python train_full_visual_semantic.py --use-visual --require-visual

  # TRANSFORMER with visual-semantic features
  python train_full_visual_semantic.py --use-visual --model transformer

  # TRANSFORMER with full interaction-aware features
  python train_full_visual_semantic.py --use-visual --require-visual --use-union --use-pose --model transformer
        """,
    )
    parser.add_argument("--use-visual", dest="use_visual", action="store_true",
                        default=USE_VISUAL,
                        help="Enable CLIP visual features")
    # --use-visual defaults to True, so without an explicit opposite flag the
    # geometry-only baseline documented above was unreachable from the CLI.
    parser.add_argument("--no-visual", dest="use_visual", action="store_false",
                        help="Disable CLIP visual features (geometry-only baseline)")
    parser.add_argument("--visual-filter-only", action="store_true", default=False,
                        help="Load the CLIP cache to fix the sample population but "
                             "build the model with clip_dim=0. This is how the "
                             "geometry control of the feature ablation is run on "
                             "exactly the same samples as the CLIP variants.")
    parser.add_argument("--require-visual", action="store_true", default=False,
                        help="Strict mode: drop samples with missing CLIP embeddings")
    parser.add_argument("--use-pose", action="store_true", default=False,
                        help="Enable pose features (requires MediaPipe)")
    parser.add_argument("--use-union", action="store_true", default=False,
                        help="Enable union-region CLIP features")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument("--model", type=str, default=MODEL_TYPE, choices=["mlp", "transformer"],
                        help="Model architecture (mlp or transformer)")
    parser.add_argument("--vg-root", type=str, default=str(VG_ROOT))
    parser.add_argument("--clip-cache", type=str, default=None,
                        help="Path to the CLIP feature cache built by "
                             "build_clip_cache.py (default: <vg-root>/clip_cache_proper.pt)")
    parser.add_argument("--geo-mode", type=str, default=GEO_MODE,
                        choices=["none", "basic", "ext"],
                        help="Geometry descriptor: 'none' = 0-dim (visual-only "
                             "control), 'basic' = 5-dim legacy, "
                             "'ext' = 19-dim extended (adds subject-relative "
                             "offsets, asymmetric containment, absolute scale "
                             "and position, signed vertical gaps, aspect "
                             "ratios). Same boxes, no new data.")
    parser.add_argument("--geo-norm", action="store_true", default=GEO_NORM,
                        help="Standardise the geometry block with a "
                             "BatchNorm1d whose running statistics are stored "
                             "in the checkpoint (recommended with --geo-mode ext)")
    parser.add_argument("--predicate-scheme", type=str, default=PREDICATE_SCHEME,
                        choices=["v1", "v2"],
                        help="Predicate normalisation. 'v1' = frozen E0 "
                             "behaviour (exact-match map + allowlist). 'v2' "
                             "additionally recovers inflections, trailing "
                             "articles and paraphrases onto the SAME 19 "
                             "classes (+18.4%% annotations) and drops passives "
                             "instead of labelling the pair backwards.")
    parser.add_argument("--loss", type=str, default=LOSS, choices=["ce", "focal"],
                        help="Training objective (default: focal, legacy)")
    parser.add_argument("--class-weight-alpha", type=float, default=None,
                        help="Class weights = (1/count)**alpha computed on the "
                             "TRAIN split. 0 = unweighted. Omit to keep the "
                             "legacy effective-number (beta=0.999) weights.")
    parser.add_argument("--select-metric", type=str, default=SELECT_METRIC,
                        choices=["top1", "macro_f1"],
                        help="Validation metric used to pick the best epoch")
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--d-model", type=int, default=D_MODEL,
                        help="Transformer width (ignored for --model mlp)")
    parser.add_argument("--split-manifest", type=str, default=None,
                        help="Frozen image-disjoint split manifest (e.g. "
                             "splits/e0_image_split.json). When supplied it "
                             "replaces random_split and is treated as "
                             "authoritative; test image IDs are never used.")
    args = parser.parse_args()
    configure_safe_stdio()

    USE_VISUAL = args.use_visual
    REQUIRE_VISUAL = args.require_visual
    VISUAL_FILTER_ONLY = args.visual_filter_only
    if VISUAL_FILTER_ONLY and not args.use_visual:
        parser.error("--visual-filter-only requires visual loading (drop --no-visual)")
    USE_POSE = args.use_pose
    USE_UNION = args.use_union
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    SEED = args.seed
    MODEL_TYPE = args.model
    SPLIT_MANIFEST = args.split_manifest
    GEO_MODE = args.geo_mode
    PREDICATE_SCHEME = args.predicate_scheme
    GEO_NORM = args.geo_norm
    LOSS = args.loss
    CLASS_WEIGHT_ALPHA = args.class_weight_alpha
    SELECT_METRIC = args.select_metric
    DROPOUT = args.dropout
    WEIGHT_DECAY = args.weight_decay
    D_MODEL = args.d_model
    CHECKPOINT_DIR = Path(args.checkpoint_dir)
    VG_ROOT = Path(args.vg_root)
    VG_IMAGE_DIR = VG_ROOT / "images"
    CLIP_CACHE_PATH = (Path(args.clip_cache) if args.clip_cache
                       else VG_ROOT / "clip_cache_proper.pt")

    main()

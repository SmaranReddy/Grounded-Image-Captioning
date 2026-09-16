"""Object-hallucination metrics for the baseline-vs-grounded caption experiment.

Everything here is a pure function of (caption text, human ground truth) and
is shared by every arm, so a difference between arms can only come from the
captions.

CHAIR (Rohrbach et al., 2018), computed at corpus level
-------------------------------------------------------
Each caption contributes its SET of mentioned COCO-80 objects (a repeated
mention counts once, so repetition cannot inflate or deflate a score).

    CHAIR_i = sum_images |mentioned - GT| / sum_images |mentioned|
    CHAIR_s = fraction of captions with at least one hallucinated object

CHAIR_i is a ratio of hallucinated to mentioned objects, so a caption is not
penalised merely for mentioning more objects - but CHAIR_s is: a caption that
names more objects has more chances to name a wrong one. That is why the mean
number of mentioned objects and hallucinated objects per caption are always
reported next to them.

Caption POPE (Li et al., 2023, adapted to captions)
---------------------------------------------------
POPE asks "Is there a <object> in the image?" for a BALANCED set of present
and absent objects and scores the yes/no answers. A captioner answers "yes"
exactly when its caption mentions the object.

The legacy utils/pope.py was not this: its positive probes were the objects
the caption mentioned (so the probe set itself depended on the caption, and
longer captions got more probes), and its negative probes were drawn from
objects the caption did NOT mention, so every negative was a true negative by
construction and a hallucination could never appear as a false positive there.

Here the probe set is built from the ground truth ALONE, once per image, before
any caption is looked at, and is byte-identical for every arm:

    positives: up to k GT objects (seeded sample)
    negatives: exactly as many absent COCO-80 objects, chosen by
        random      - uniform over absent classes
        popular     - absent classes most frequent across the eval set
        adversarial - absent classes that co-occur most with this image's GT

A false positive is an absent object the caption asserts - a hallucination.
Caveats that apply equally to all arms: Visual Genome annotations are not
exhaustive, so an "absent" object may in fact be present; and caption POPE
recall is bounded by how many objects a one-sentence caption can mention.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from utils.coco_mentions import COCO_80, mentions

POPE_SETTINGS: Tuple[str, ...] = ("random", "popular", "adversarial")
POPE_POSITIVES_PER_IMAGE = 3


# ---------------------------------------------------------------------------
# CHAIR
# ---------------------------------------------------------------------------

def caption_object_record(caption: str, gt_objects: Iterable[str]) -> Dict:
    """Per-caption object bookkeeping. Pure function of caption + GT."""
    gt = set(gt_objects)
    unknown = gt - COCO_80
    if unknown:
        raise ValueError(f"ground truth contains non-COCO labels: {sorted(unknown)}")
    mentioned = mentions(caption)
    hallucinated = mentioned - gt
    correct = mentioned & gt
    return {
        "mentioned": sorted(mentioned),
        "hallucinated": sorted(hallucinated),
        "correct": sorted(correct),
        "n_mentioned": len(mentioned),
        "n_hallucinated": len(hallucinated),
        "n_correct": len(correct),
        "n_gt": len(gt),
        "n_words": len(caption.split()),
    }


def aggregate_chair(records: Sequence[Mapping]) -> Dict:
    n = len(records)
    if n == 0:
        raise ValueError("no captions to aggregate")
    sum_m = sum(r["n_mentioned"] for r in records)
    sum_h = sum(r["n_hallucinated"] for r in records)
    sum_c = sum(r["n_correct"] for r in records)
    sum_gt = sum(r["n_gt"] for r in records)
    return {
        "n_captions": n,
        "chair_i": sum_h / sum_m if sum_m else 0.0,
        "chair_s": sum(1 for r in records if r["n_hallucinated"] > 0) / n,
        "object_recall": sum_c / sum_gt if sum_gt else 0.0,
        "mean_mentioned_objects": sum_m / n,
        "mean_hallucinated_objects": sum_h / n,
        "mean_correct_objects": sum_c / n,
        "mean_caption_words": sum(r["n_words"] for r in records) / n,
        "captions_with_no_object_mention": sum(1 for r in records if r["n_mentioned"] == 0),
        "total_mentioned": sum_m,
        "total_hallucinated": sum_h,
    }


# ---------------------------------------------------------------------------
# POPE
# ---------------------------------------------------------------------------

def _popularity(gt_by_image: Mapping[str, Set[str]]) -> Counter:
    counts: Counter = Counter()
    for objs in gt_by_image.values():
        counts.update(set(objs))
    return counts


def _cooccurrence(gt_by_image: Mapping[str, Set[str]]) -> Dict[str, Counter]:
    co: Dict[str, Counter] = defaultdict(Counter)
    for objs in gt_by_image.values():
        objs = set(objs)
        for a in objs:
            for b in objs:
                if a != b:
                    co[a][b] += 1
    return co


def build_pope_probes(
    gt_by_image: Mapping[str, Iterable[str]],
    k: int = POPE_POSITIVES_PER_IMAGE,
    seed: int = 42,
) -> Dict[str, Dict[str, List[Tuple[str, bool]]]]:
    """setting -> image_id -> [(object, present), ...]. Takes NO caption input.

    Deterministic across machines and processes: every random choice uses a
    random.Random seeded with a string (hashed with SHA-512 by CPython, not by
    the per-process str hash), over sorted inputs.
    """
    gt = {str(i): set(objs) for i, objs in gt_by_image.items()}
    popularity = _popularity(gt)
    co = _cooccurrence(gt)
    probes: Dict[str, Dict[str, List[Tuple[str, bool]]]] = {s: {} for s in POPE_SETTINGS}

    for image_id in sorted(gt, key=lambda x: (len(x), x)):
        present = sorted(gt[image_id])
        absent = sorted(COCO_80 - gt[image_id])
        rng = random.Random(f"pope-positives:{seed}:{image_id}")
        n = min(k, len(present), len(absent))
        positives = sorted(rng.sample(present, n)) if n else []

        negatives_by_setting = {
            "random": sorted(random.Random(f"pope-random:{seed}:{image_id}").sample(absent, n)),
            "popular": sorted(absent, key=lambda o: (-popularity[o], o))[:n],
            "adversarial": sorted(
                absent,
                key=lambda o: (-sum(co[g][o] for g in present), -popularity[o], o),
            )[:n],
        }
        for setting in POPE_SETTINGS:
            negs = negatives_by_setting[setting]
            assert len(negs) == len(positives), "POPE probes must be balanced"
            probes[setting][image_id] = (
                [(o, True) for o in positives] + [(o, False) for o in negs]
            )
    return probes


def score_pope(
    probes: Mapping[str, Sequence[Tuple[str, bool]]],
    mentioned_by_image: Mapping[str, Set[str]],
) -> Dict:
    """Micro-averaged POPE over every probe of every image in `probes`."""
    tp = fp = tn = fn = 0
    for image_id, image_probes in probes.items():
        said = mentioned_by_image[image_id]
        for obj, present in image_probes:
            yes = obj in said
            if present and yes:
                tp += 1
            elif present:
                fn += 1
            elif yes:
                fp += 1
            else:
                tn += 1
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n_probes": total,
        "n_positive_probes": tp + fn,
        "n_negative_probes": tn + fp,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "yes_ratio": (tp + fp) / total if total else 0.0,
        "negative_false_positive_rate": fp / (tn + fp) if tn + fp else 0.0,
    }


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(
    counts_a: Sequence[Sequence[float]],
    counts_b: Sequence[Sequence[float]],
    statistic,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict:
    """Bootstrap CI of statistic(B) - statistic(A), resampling IMAGES jointly.

    counts_a[i] / counts_b[i] are the per-image count vectors of the same image
    under the two arms; the same resampled image indices are used for both, so
    the interval reflects the paired design.
    """
    import numpy as np

    a = np.asarray(counts_a, dtype=float)
    b = np.asarray(counts_b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] == 0:
        raise ValueError(f"paired arrays must share a non-empty (n, d) shape: {a.shape} vs {b.shape}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.shape[0], size=(n_resamples, a.shape[0]))
    sums_a = a[idx].sum(axis=1)
    sums_b = b[idx].sum(axis=1)
    deltas = statistic(sums_b) - statistic(sums_a)
    point = float(statistic(b.sum(axis=0)[None, :])[0] - statistic(a.sum(axis=0)[None, :])[0])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": point, "ci95": [float(lo), float(hi)], "n_resamples": n_resamples,
            "seed": seed}


def chair_i_from_sums(sums):
    """sums[:, 0] = hallucinated, sums[:, 1] = mentioned."""
    import numpy as np
    return np.divide(sums[:, 0], sums[:, 1], out=np.zeros(len(sums)), where=sums[:, 1] > 0)


def mean_from_sums(n_images: int):
    def _stat(sums):
        return sums[:, 0] / n_images
    return _stat


# ---------------------------------------------------------------------------
# Pairing guard
# ---------------------------------------------------------------------------

def check_pairing(eval_ids: Sequence[str], arms: Mapping[str, Mapping[str, Mapping]]) -> None:
    """Every arm must caption exactly the evaluation images, each record keyed
    by its own image_id. Raises ValueError otherwise - a comparison over
    different image sets is not a comparison."""
    expected = set(map(str, eval_ids))
    if len(expected) != len(eval_ids):
        raise ValueError("evaluation image list contains duplicates")
    for arm, records in arms.items():
        got = set(records)
        if got != expected:
            missing = sorted(expected - got)[:5]
            extra = sorted(got - expected)[:5]
            raise ValueError(
                f"arm {arm!r} does not cover the evaluation set exactly: "
                f"{len(expected - got)} missing (e.g. {missing}), "
                f"{len(got - expected)} unexpected (e.g. {extra})"
            )
        for key, rec in records.items():
            if str(rec.get("image_id")) != key:
                raise ValueError(f"arm {arm!r}: record keyed {key} carries image_id "
                                 f"{rec.get('image_id')}")

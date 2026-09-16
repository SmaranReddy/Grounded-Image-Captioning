"""
POPE: Polling-based Object Probing Evaluation  (LEGACY - not used by the caption experiment)

Known limitation, still present: the positive probes are the objects the
caption MENTIONS and the negative probes are objects it does NOT mention, so
(a) the probe set depends on the caption - a caption naming more objects gets
more probes - and (b) every negative probe is a true negative by construction,
so a hallucination can never register as a false positive on a negative probe
(it only shows up as fp among the positives, i.e. it duplicates CHAIR).
Fixing the negative count at NUM_NEGATIVE_PROBES removed the length-dependent
TN padding, not these two problems. The caption experiment uses the balanced,
caption-independent probe set in utils/caption_hallucination.py instead.

Evaluates object hallucination by constructing probing questions about
whether objects mentioned in a caption are actually present in the image.

POPE differs from CHAIR conceptually:
  CHAIR measures hallucination rate among mentioned objects (chair_i, chair_s)
  POPE measures precision, recall, F1, and accuracy of object assertions

POPE complements SPICE + CHAIR:
  SPICE -> relation grounding quality
  CHAIR -> hallucinated caption objects
  POPE  -> object hallucination probing behavior

Probing paradigm:
  For each candidate object from COCO-80:
    - "Is there a [object] in the image based on the caption?"
    - Positive probe: object is MENTIONED in the caption
    - Negative probe: object is NOT mentioned (sampled, balanced)
    - Answer is correct if it matches ground-truth presence

Key distinction from CHAIR:
  CHAIR only evaluates caption-output overlap with GT objects.
  POPE additionally evaluates recall (missed objects) and overall
  accuracy including true negatives via balanced probing.
"""

from __future__ import annotations

import random
from typing import Dict, Set

from utils.metrics import COCO_80, detect_coco_objects

# Fixed number of negative probes per image. Must NOT depend on the caption:
# see the note in compute_pope. Every negative probe is a true negative by
# construction, so this constant only sets how much TN padding enters
# pope_accuracy — it is identical for every system, which is what makes the
# accuracies comparable. pope_precision / recall / f1 do not use TN at all and
# are the metrics to report.
NUM_NEGATIVE_PROBES = 10


def compute_pope(
    candidate: str,
    gt_objects: Set[str],
    seed: int = 42,
) -> Dict:
    """Compute POPE metrics for a single caption.

    Constructs probing questions by:
      1. Extracting mentioned COCO-80 objects from the caption (positive probes)
      2. Sampling non-mentioned, non-GT COCO-80 objects (negative probes)
      3. Computing the confusion matrix against GT object annotations

    All three systems are evaluated with IDENTICAL probing logic and
    object vocabulary, ensuring fair comparison.

    Args:
        candidate:  Generated caption string.
        gt_objects: Set of ground-truth COCO-80 object labels in the image.
        seed:       Random seed for deterministic negative probe sampling.

    Returns:
        Dict with POPE metrics:
            pope_precision, pope_recall, pope_f1, pope_accuracy,
            pope_tp, pope_fp, pope_fn, pope_tn,
            pope_num_positive_probes, pope_num_negative_probes,
            pope_hallucinated_objects, pope_missed_objects
    """
    mentioned = detect_coco_objects(candidate)

    # Confusion matrix building blocks.
    tp_set = mentioned & gt_objects
    fp_set = mentioned - gt_objects
    fn_set = gt_objects - mentioned

    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)

    # -- Negative probes --------------------------------------------------
    # Pool = objects neither mentioned by the caption NOR present in GT, so
    # every negative probe is a true negative by construction.
    #
    # The count is FIXED (NUM_NEGATIVE_PROBES) and must not depend on the
    # caption. It used to be `max(10, len(mentioned))`, which made a caption
    # that names more objects earn more free true negatives; since TN enters
    # pope_accuracy, a system whose captions are simply longer scored higher
    # at identical correctness. The grounded system's captions ARE longer than
    # the baseline's (the injected relation prefix adds object mentions), so
    # that bias ran in exactly the direction of the result being claimed.
    #
    # `num_positives` was also only bound inside the `if negative_pool:`
    # branch while being read unconditionally at the return, so an empty pool
    # raised NameError instead of returning a result.
    num_positives = len(mentioned)
    negative_pool = sorted(COCO_80 - mentioned - gt_objects)
    num_negatives = min(len(negative_pool), NUM_NEGATIVE_PROBES)

    if num_negatives:
        rng = random.Random(seed)
        rng.shuffle(negative_pool)
        negative_probes = set(negative_pool[:num_negatives])
    else:
        negative_probes = set()

    tn = len(negative_probes)

    # -- Metrics ----------------------------------------------------------
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (
        (tp + tn) / (tp + fp + tn + fn)
        if (tp + fp + tn + fn) > 0
        else 0.0
    )

    return {
        "pope_precision": round(precision, 4),
        "pope_recall": round(recall, 4),
        "pope_f1": round(f1, 4),
        "pope_accuracy": round(accuracy, 4),
        "pope_tp": tp,
        "pope_fp": fp,
        "pope_fn": fn,
        "pope_tn": tn,
        "pope_num_positive_probes": num_positives,
        "pope_num_negative_probes": num_negatives,
        "pope_hallucinated_objects": sorted(fp_set),
        "pope_missed_objects": sorted(fn_set),
    }

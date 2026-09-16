"""Hallucination metrics for the caption experiment must be fair between arms.

Covers the mention extractor shared by captions and ground truth, CHAIR
arithmetic, the balanced caption-independent POPE probe set (the legacy POPE
built its probes from the caption), pairing guards and the paired statistics.
Pure Python: no model, no NLTK, no GPU.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.caption_hallucination import (  # noqa: E402
    POPE_SETTINGS,
    aggregate_chair,
    build_pope_probes,
    caption_object_record,
    check_pairing,
    mcnemar_exact,
    paired_bootstrap,
    score_pope,
)
from utils.coco_mentions import COCO_80, mentions, objects_from_names  # noqa: E402


# ---------------------------------------------------------------------------
# mention extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("a dog on a couch", {"dog", "couch"}),
    ("two dogs and three cats", {"dog", "cat"}),            # plurals
    ("a man eating a hot dog", {"person", "hot dog"}),      # not also `dog`
    ("a teddy bear on a bed", {"teddy bear", "bed"}),       # not also `bear`
    ("people waiting at a bus stop", {"person"}),           # bus stop is not a bus
    ("a train on the train tracks", {"train"}),
    ("wine glasses on a table", {"wine glass", "dining table"}),
    ("a woman holding a cell phone", {"person", "cell phone"}),
    ("a baby elephant", {"elephant"}),
    ("knives, forks and spoons", {"knife", "fork", "spoon"}),
    ("a photo of", set()),
    ("", set()),
])
def test_mentions(text, expected):
    assert mentions(text) == expected


def test_surface_table_only_names_coco_classes():
    from utils.coco_mentions import _PHRASES
    assert {c for c in _PHRASES.values() if c is not None} <= COCO_80


def test_coco_vocabulary_matches_the_rest_of_the_repository():
    from relation_prediction.vg_dataset import COCO_LABELS
    assert COCO_80 == COCO_LABELS


def test_ground_truth_and_captions_share_one_vocabulary():
    """A VG name and a caption word for the same thing map to the same class."""
    for name, caption in [("table", "a table"), ("bikes", "a bike"), ("phone", "a phone"),
                          ("man", "a woman"), ("sofa", "a couch")]:
        assert objects_from_names([name]) == mentions(caption)


# ---------------------------------------------------------------------------
# CHAIR
# ---------------------------------------------------------------------------

def test_chair_counts_each_object_once_per_caption():
    rec = caption_object_record("a dog and a dog and a dog", {"cat"})
    assert rec["n_mentioned"] == 1 and rec["n_hallucinated"] == 1


def test_chair_corpus_level_arithmetic():
    recs = [
        caption_object_record("a person riding a horse", {"person", "horse"}),   # 0/2
        caption_object_record("a dog on a couch", {"dog"}),                      # 1/2
        caption_object_record("a sunny day", {"car"}),                           # 0/0
    ]
    agg = aggregate_chair(recs)
    assert agg["chair_i"] == pytest.approx(1 / 4)          # micro, not mean of ratios
    assert agg["chair_s"] == pytest.approx(1 / 3)
    assert agg["object_recall"] == pytest.approx(3 / 4)
    assert agg["mean_mentioned_objects"] == pytest.approx(4 / 3)
    assert agg["captions_with_no_object_mention"] == 1


def test_chair_rejects_non_coco_ground_truth():
    with pytest.raises(ValueError):
        caption_object_record("a dog", {"shirt"})


# ---------------------------------------------------------------------------
# POPE
# ---------------------------------------------------------------------------

GT = {
    "1": {"person", "bicycle", "car"},
    "2": {"person", "dog"},
    "3": {"cat", "couch", "tv", "remote"},
    "4": {"person", "surfboard"},
}


def test_pope_probes_are_balanced_and_disjoint_from_truth():
    probes = build_pope_probes(GT, k=3, seed=42)
    for setting in POPE_SETTINGS:
        for iid, p in probes[setting].items():
            pos = [o for o, present in p if present]
            neg = [o for o, present in p if not present]
            assert len(pos) == len(neg) == min(3, len(GT[iid]))
            assert set(pos) <= GT[iid]
            assert not set(neg) & GT[iid]
            assert set(neg) <= COCO_80


def test_pope_probe_set_is_deterministic():
    assert build_pope_probes(GT, seed=42) == build_pope_probes(dict(reversed(list(GT.items()))), seed=42)


def test_pope_probes_do_not_depend_on_any_caption():
    """The function takes no caption; scoring two very different captions uses one probe set."""
    import inspect
    assert "caption" not in " ".join(inspect.signature(build_pope_probes).parameters)
    probes = build_pope_probes(GT, seed=42)
    short = {i: mentions("a photo") for i in GT}
    long_ = {i: mentions("a person, a dog, a cat, a car, a bicycle, a couch and a tv") for i in GT}
    s, l = score_pope(probes["random"], short), score_pope(probes["random"], long_)
    assert s["n_probes"] == l["n_probes"]
    assert s["n_positive_probes"] == l["n_positive_probes"] == s["n_negative_probes"]


def test_pope_length_alone_does_not_buy_accuracy():
    """Two captions that say exactly the true objects score identically, however
    many filler words one of them has."""
    probes = build_pope_probes(GT, seed=42)
    exact = {i: set(GT[i]) for i in GT}
    padded = {i: mentions(" and ".join(f"a very nice {o}" for o in sorted(GT[i])) + " on a sunny day")
              for i in GT}
    assert padded == exact
    for setting in POPE_SETTINGS:
        assert score_pope(probes[setting], exact) == score_pope(probes[setting], padded)


def test_pope_hallucinating_an_absent_probe_is_a_false_positive():
    probes = build_pope_probes(GT, seed=42)
    negatives = {i: {o for o, present in probes["adversarial"][i] if not present} for i in GT}
    clean = score_pope(probes["adversarial"], {i: set(GT[i]) for i in GT})
    halluc = score_pope(probes["adversarial"], {i: set(GT[i]) | negatives[i] for i in GT})
    assert clean["fp"] == 0 and clean["accuracy"] == 1.0
    assert halluc["fp"] == halluc["n_negative_probes"] > 0
    assert halluc["negative_false_positive_rate"] == 1.0
    assert halluc["accuracy"] == pytest.approx(0.5)


def test_pope_adversarial_negatives_prefer_co_occurring_objects():
    gt = {"a": {"person", "dog"}, "b": {"person", "dog"}, "c": {"person", "frisbee"},
          "d": {"person"}}
    probes = build_pope_probes(gt, k=1, seed=0)
    neg = [o for o, present in probes["adversarial"]["d"] if not present]
    assert neg == ["dog"]


# ---------------------------------------------------------------------------
# pairing and statistics
# ---------------------------------------------------------------------------

def test_pairing_rejects_mismatched_image_sets():
    ids = ["1", "2"]
    good = {"1": {"image_id": "1"}, "2": {"image_id": "2"}}
    check_pairing(ids, {"baseline": good, "grounded": dict(good)})
    with pytest.raises(ValueError):
        check_pairing(ids, {"baseline": good, "grounded": {"1": {"image_id": "1"}}})
    with pytest.raises(ValueError):
        check_pairing(ids, {"baseline": good, "grounded": {"1": {"image_id": "1"},
                                                            "2": {"image_id": "1"}}})


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)


def test_paired_bootstrap_identical_arms_give_zero():
    a = [[1, 2], [0, 3], [2, 2], [0, 1]]
    ratio = lambda s: s[:, 0] / s[:, 1]  # noqa: E731
    r = paired_bootstrap(a, a, ratio, n_resamples=500, seed=1)
    assert r["delta"] == 0 and r["ci95"] == [0.0, 0.0]


def test_paired_bootstrap_point_estimate_is_the_corpus_difference():
    a = [[1, 2], [1, 2]]
    b = [[0, 2], [1, 2]]
    ratio = lambda s: s[:, 0] / s[:, 1]  # noqa: E731
    r = paired_bootstrap(a, b, ratio, n_resamples=500, seed=1)
    assert r["delta"] == pytest.approx(0.25 - 0.5)
    assert r["ci95"][0] <= r["delta"] <= r["ci95"][1]

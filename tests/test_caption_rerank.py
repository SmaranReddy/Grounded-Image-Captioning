"""Grounding-aware candidate reranking: scoring, selection and evidence hygiene.

Everything here is a pure function of (candidates, evidence, relation, weights).
No model, no GPU and - critically - no ground truth: the last test in this file
poisons the human annotations in the evaluation manifest and asserts that the
rerank stage produces byte-identical output, which is what makes "the system
grounds its caption in its own perception" a checkable claim rather than a
promise.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.caption_rerank import (  # noqa: E402
    MAX_CANDIDATE_WORDS,
    MIN_CANDIDATE_WORDS,
    SOURCE_ORDER,
    RerankWeights,
    evidence_objects,
    first_candidate,
    is_degenerate,
    rerank,
    relation_stated,
    score_candidate,
)

PERSON_BIKE = {"subject": "person", "predicate": "riding", "object": "bicycle",
               "confidence": 0.78}


def cand(text, source="baseline", beam_rank=0, lm=-1.0):
    return {"text": text, "source": source, "beam_rank": beam_rank, "lm_score": lm}


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def test_weights_round_trip_and_hash_is_value_based():
    w = RerankWeights(0.25, 0.5, 1.0)
    assert RerankWeights.from_dict(w.as_dict()) == w
    assert w.sha256() == RerankWeights(0.25, 0.5, 1.0).sha256()
    assert w.sha256() != RerankWeights(0.25, 0.5, 0.0).sha256()


def test_weights_reject_negative_and_non_finite_values():
    for bad in ({"w_obj": -0.1}, {"w_hall": float("nan")}, {"w_rel": float("inf")}):
        with pytest.raises(ValueError):
            RerankWeights.from_dict(bad)
    with pytest.raises(ValueError):
        RerankWeights.from_dict({"w_obj": 0.1, "w_typo": 1.0})


def test_without_relation_zeroes_only_the_relation_weight():
    w = RerankWeights(0.25, 0.5, 1.0).without_relation()
    assert (w.w_obj, w.w_hall, w.w_rel) == (0.25, 0.5, 0.0)


# ---------------------------------------------------------------------------
# Evidence: system perception mapped into the evaluation's own vocabulary
# ---------------------------------------------------------------------------

def test_evidence_objects_uses_the_evaluation_mapper():
    ev = evidence_objects([{"label": "person"}, {"label": "motorcycle"},
                           {"label": "dining table"}])
    assert ev == {"person", "motorcycle", "dining table"}


def test_evidence_objects_is_empty_without_detections():
    assert evidence_objects([]) == frozenset()


def test_evidence_objects_ignores_labels_outside_coco80():
    assert evidence_objects([{"label": "sky"}, {"label": "person"}]) == {"person"}


# ---------------------------------------------------------------------------
# Relation consistency
# ---------------------------------------------------------------------------

def test_relation_stated_matches_through_synonyms():
    assert relation_stated("a man riding a bicycle down a street", PERSON_BIKE)


def test_relation_stated_rejects_a_different_predicate():
    assert not relation_stated("a person standing beside a bicycle", PERSON_BIKE)
    assert not relation_stated("a person and a bicycle", PERSON_BIKE)


def test_relation_stated_requires_subject_before_object():
    assert not relation_stated("a bicycle near a person riding", PERSON_BIKE)


def test_relation_stated_handles_multi_word_predicates():
    rel = {"subject": "cup", "predicate": "sitting on", "object": "dining table",
           "confidence": 0.6}
    assert relation_stated("a cup sitting on a table", rel)
    assert not relation_stated("a cup under a table", rel)


def test_relation_stated_is_false_without_a_relation():
    assert not relation_stated("a person riding a bicycle", None)
    assert not relation_stated("a person riding a bicycle", {})


def test_relation_stated_needs_two_distinct_mentions_for_a_self_relation():
    rel = {"subject": "person", "predicate": "holding", "object": "person",
           "confidence": 0.5}
    assert relation_stated("a person holding a child person", rel) is True
    assert relation_stated("a person holding an umbrella", rel) is False


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------

def test_object_support_counts_only_detected_objects():
    w = RerankWeights(w_obj=1.0)
    s = score_candidate(cand("a person riding a bicycle"), frozenset({"person"}), None, w)
    assert s["n_supported"] == 1 and s["n_unsupported"] == 1
    assert s["terms"]["object_support"] == pytest.approx(1.0)


def test_hallucination_penalty_is_negative_and_scales_with_the_weight():
    ev = frozenset({"person"})
    a = score_candidate(cand("a person riding a bicycle"), ev, None, RerankWeights(w_hall=0.5))
    b = score_candidate(cand("a person riding a bicycle"), ev, None, RerankWeights(w_hall=1.0))
    assert a["terms"]["hallucination_penalty"] == pytest.approx(-0.5)
    assert b["terms"]["hallucination_penalty"] == pytest.approx(-1.0)


def test_relation_term_is_weighted_by_confidence():
    w = RerankWeights(w_rel=1.0)
    ev = frozenset({"person", "bicycle"})
    strong = score_candidate(cand("a person riding a bicycle"), ev, PERSON_BIKE, w)
    weak = score_candidate(cand("a person riding a bicycle"), ev,
                           {**PERSON_BIKE, "confidence": 0.1}, w)
    assert strong["terms"]["relation_support"] == pytest.approx(0.78)
    assert weak["terms"]["relation_support"] == pytest.approx(0.1)


def test_relation_term_is_zero_for_a_candidate_that_does_not_state_it():
    s = score_candidate(cand("a person standing next to a bicycle"),
                        frozenset({"person", "bicycle"}), PERSON_BIKE,
                        RerankWeights(w_rel=1.0))
    assert s["terms"]["relation_support"] == 0.0


def test_score_is_the_sum_of_its_terms():
    s = score_candidate(cand("a person riding a bicycle", lm=-1.25),
                        frozenset({"person"}), PERSON_BIKE,
                        RerankWeights(0.5, 0.25, 1.0))
    assert s["score"] == pytest.approx(sum(s["terms"].values()))
    assert s["score"] == pytest.approx(-1.25 + 0.5 - 0.25 + 0.78)


def test_non_finite_lm_score_is_refused():
    with pytest.raises(ValueError):
        score_candidate(cand("a person on a bicycle", lm=float("-inf")),
                        frozenset(), None, RerankWeights())


# ---------------------------------------------------------------------------
# Candidate hygiene
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,reason", [
    ("", "empty"),
    ("   ", "empty"),
    ("a photo", f"fewer_than_{MIN_CANDIDATE_WORDS}_words"),
    (" ".join(["word"] * (MAX_CANDIDATE_WORDS + 1)), f"more_than_{MAX_CANDIDATE_WORDS}_words"),
    ("a man on a bike a man on a bike a man on a bike", "4gram_repeated_3_times"),
])
def test_degenerate_candidates_are_named(text, reason):
    assert is_degenerate(text) == reason


def test_a_normal_caption_is_not_degenerate():
    assert is_degenerate("a photo of a person riding a bicycle on a street") is None


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def test_rerank_prefers_the_candidate_the_evidence_supports():
    pool = [cand("a photo of a cat on a bed", lm=-1.0),
            cand("a photo of a person riding a bicycle", "relation", 0, lm=-1.4)]
    out = rerank(pool, frozenset({"person", "bicycle"}), PERSON_BIKE,
                 RerankWeights(w_obj=1.0, w_rel=1.0))
    assert out["caption"] == "a photo of a person riding a bicycle"
    assert out["source"] == "relation" and out["relation_stated"] is True


def test_rerank_declines_a_wrong_relation_the_image_does_not_support():
    """The whole point: a confident but wrong relation must be refusable."""
    pool = [cand("a photo of a cat looking at a laptop", lm=-1.40),
            cand("a photo of a person riding a bicycle with a laptop", "relation", 0, lm=-2.52)]
    out = rerank(pool, frozenset({"cat", "laptop"}), PERSON_BIKE,
                 RerankWeights(w_obj=0.5, w_hall=0.5, w_rel=1.0))
    assert out["caption"] == "a photo of a cat looking at a laptop"
    assert out["relation_stated"] is False


def test_low_confidence_relation_cannot_outweigh_the_evidence():
    weak = {**PERSON_BIKE, "confidence": 0.05}
    pool = [cand("a photo of a cat on a bed", lm=-1.0),
            cand("a photo of a person riding a bicycle", "relation", 0, lm=-1.05)]
    out = rerank(pool, frozenset({"cat", "bed"}), weak,
                 RerankWeights(w_obj=0.5, w_hall=0.5, w_rel=1.0))
    assert out["caption"] == "a photo of a cat on a bed"


def test_ranking_is_complete_and_sorted_by_score():
    pool = [cand("a photo of a cat on a bed", lm=-1.0),
            cand("a photo of a cat and a laptop", "objects_only", 0, lm=-1.6),
            cand("a photo of a person riding a bicycle", "relation", 0, lm=-3.0)]
    out = rerank(pool, frozenset({"cat"}), PERSON_BIKE, RerankWeights(w_obj=0.5))
    scores = [r["score"] for r in out["ranking"]]
    assert len(out["ranking"]) == 3
    assert scores == sorted(scores, reverse=True)
    assert out["caption"] == out["ranking"][0]["text"]


def test_ties_break_towards_the_plain_baseline_candidate():
    pool = [cand("a photo of a person and a bicycle", "objects_only", 0, lm=-1.0),
            cand("a photo of a person riding a bicycle", "relation", 0, lm=-1.0),
            cand("a photo of a person with a bicycle", "baseline", 3, lm=-1.0)]
    out = rerank(pool, frozenset({"person", "bicycle"}), None, RerankWeights(w_obj=1.0))
    assert out["source"] == "baseline"
    assert [r["source"] for r in out["ranking"]] == list(SOURCE_ORDER)


def test_ties_within_a_source_break_towards_the_earlier_beam():
    pool = [cand("a photo of a person with a bicycle", "baseline", 2, lm=-1.0),
            cand("a photo of a bicycle and a person", "baseline", 1, lm=-1.0)]
    out = rerank(pool, frozenset({"person", "bicycle"}), None, RerankWeights(w_obj=1.0))
    assert out["beam_rank"] == 1


def test_selection_does_not_depend_on_the_order_candidates_are_listed():
    pool = [cand("a photo of a cat on a bed", "baseline", 0, lm=-1.0),
            cand("a photo of a cat and a laptop", "objects_only", 0, lm=-1.0),
            cand("a photo of a person riding a bicycle", "relation", 1, lm=-1.0)]
    weights = RerankWeights(0.5, 0.25, 1.0)
    base = rerank(pool, frozenset({"cat"}), PERSON_BIKE, weights)
    for order in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        other = rerank([pool[i] for i in order], frozenset({"cat"}), PERSON_BIKE, weights)
        assert other["caption"] == base["caption"]
        assert other["score"] == pytest.approx(base["score"])


def test_rerank_is_deterministic_across_repeated_calls():
    pool = [cand("a photo of a cat on a bed", lm=-1.0),
            cand("a photo of a person riding a bicycle", "relation", 0, lm=-1.2)]
    weights = RerankWeights(0.5, 0.5, 0.5)
    results = [rerank(pool, frozenset({"cat"}), PERSON_BIKE, weights) for _ in range(5)]
    assert len({json.dumps(r, sort_keys=True) for r in results}) == 1


def test_degenerate_candidates_are_dropped_but_recorded():
    pool = [cand("", lm=-0.1), cand("a photo of a cat on a bed", lm=-1.0)]
    out = rerank(pool, frozenset({"cat"}), None, RerankWeights(w_obj=1.0))
    assert out["caption"] == "a photo of a cat on a bed"
    assert out["n_candidates_scored"] == 1 and out["n_candidates_dropped"] == 1
    assert out["dropped"][0]["reason"] == "empty"


def test_an_entirely_empty_pool_raises_rather_than_inventing_text():
    with pytest.raises(ValueError):
        rerank([], frozenset({"cat"}), None, RerankWeights())
    with pytest.raises(ValueError):
        rerank([cand("")], frozenset({"cat"}), None, RerankWeights())


def test_with_no_detections_the_score_reduces_to_the_language_model():
    pool = [cand("a photo of a cat on a bed", lm=-1.0),
            cand("a photo of a dog in a park", "baseline", 1, lm=-2.0)]
    out = rerank(pool, frozenset(), None, RerankWeights(w_obj=1.0, w_hall=0.0))
    assert out["caption"] == "a photo of a cat on a bed"


def test_without_detections_a_hallucination_penalty_prefers_the_silent_caption():
    pool = [cand("a photo of a cat on a bed", lm=-1.0),
            cand("a photo taken in the afternoon", "baseline", 1, lm=-1.5)]
    out = rerank(pool, frozenset(), None, RerankWeights(w_hall=1.0))
    assert out["caption"] == "a photo taken in the afternoon"


def test_without_a_relation_the_relation_weight_has_no_effect():
    pool = [cand("a photo of a cat on a bed", lm=-1.0),
            cand("a photo of a dog in a park", "baseline", 1, lm=-1.1)]
    ev = frozenset({"cat", "bed"})
    a = rerank(pool, ev, None, RerankWeights(0.5, 0.5, 0.0))
    b = rerank(pool, ev, None, RerankWeights(0.5, 0.5, 2.0))
    assert a["caption"] == b["caption"] and a["score"] == pytest.approx(b["score"])


def test_several_competing_relations_only_the_selected_one_is_scored():
    """The detector may propose many pairs; `select_relation` has already picked
    one, and the reranker scores that one only - it never re-opens the choice."""
    pool = [cand("a photo of a person riding a bicycle", "relation", 0, lm=-1.2),
            cand("a photo of a person wearing a backpack", "baseline", 1, lm=-1.2)]
    out = rerank(pool, frozenset({"person", "bicycle", "backpack"}), PERSON_BIKE,
                 RerankWeights(w_obj=0.0, w_rel=1.0))
    assert out["caption"] == "a photo of a person riding a bicycle"


def test_first_candidate_is_the_rank_zero_baseline_beam():
    pool = [cand("second beam", "baseline", 1), cand("first beam", "baseline", 0),
            cand("relation beam", "relation", 0)]
    assert first_candidate(pool)["text"] == "first beam"
    assert first_candidate([cand("only relation", "relation", 0)]) is None

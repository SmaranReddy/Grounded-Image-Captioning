"""Grounding-aware reranking of BLIP caption candidates.

Why this module exists
----------------------
The frozen caption experiment (CAPTION_EXPERIMENTS.md) compared three arms
that differ only in the text prefix BLIP continues from. Its result:

* injecting the predicted relation ("a photo of a person riding a bicycle")
  raised CHAIR_i from 4.11% to 8.30% and CHAIR_s from 5.20% to 15.60%;
* the objects-only control ("a photo of a person and a bicycle") moved every
  metric by the same amount, so the *predicate* contributed nothing;
* CLIPScore fell (0.7115 -> 0.6733).

A prefix is a MANDATORY ASSERTION: whatever is spliced into it is stated by
the caption whether or not it is true, so every detector or relation error is
converted one-for-one into caption content. The system has no way to decline.

Reranking replaces the mandate with a choice. BLIP generates candidates from
several prefixes (baseline / objects-only / relation), and a deterministic
scoring function picks one. The relation is evidence weighted by its own
confidence, not a sentence fragment that must appear.

What may and may not be used as evidence
----------------------------------------
Evidence = what the system itself perceived: CLIP-verified YOLO detections and
the predicted relation. Human Visual Genome annotations are the EVALUATION
ground truth and are never read here - nothing in this module takes a ground
truth argument, and the rerank stage never opens the evaluation manifest.

The score
---------
For a candidate caption c of an image with verified-detection object set V and
predicted relation r = (subject, predicate, object, confidence):

    score(c) =        lm(c)                    mean per-token log P(c | image)
              + w_obj * support(c)             |mentions(c) & V|
              - w_hall * unsupported(c)        |mentions(c) - V|
              + w_rel * confidence * rel(c)    1 if c states r, else 0

`lm` carries weight 1 by definition: only the ratios between the weights
matter, so fixing it removes one redundant parameter. `mentions()` is the same
mapper the evaluation uses on both captions and ground truth, so "supported"
and "hallucinated" are measured in one vocabulary.

Candidates that are empty, too short, too long or stuck in an n-gram loop are
dropped before scoring (`is_degenerate`); these bounds are fixed sanity limits,
not tuned parameters. Ties are broken deterministically towards the plain
baseline candidate, so the reranker only departs from BLIP's own first choice
when the evidence actually prefers something else.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from utils.coco_mentions import located_mention_spans, mentions, objects_from_names, tokenize

# Candidate sources, in tie-break priority order: on an exact tie the plain
# baseline candidate wins, then the objects-only one, then the relation one.
# The reranker therefore never asserts extra content "for free".
SOURCE_ORDER: Tuple[str, ...] = ("baseline", "objects_only", "relation")
SOURCE_RANK: Dict[str, int] = {s: i for i, s in enumerate(SOURCE_ORDER)}

# Fixed sanity bounds on a candidate (pre-declared, never tuned).
MIN_CANDIDATE_WORDS = 3
MAX_CANDIDATE_WORDS = 60
REPEAT_NGRAM = 4
MAX_NGRAM_REPEATS = 3

# Scores are compared at this precision so that arithmetically equal scores
# tie deterministically instead of being ordered by float noise.
SCORE_PRECISION = 9


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RerankWeights:
    """The three free parameters of the score. `lm` is fixed at 1.0."""

    w_obj: float = 0.0
    w_hall: float = 0.0
    w_rel: float = 0.0

    def __post_init__(self) -> None:
        for name in ("w_obj", "w_hall", "w_rel"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"{name} must be a finite number, got {v!r}")
            if float(v) < 0.0:
                raise ValueError(f"{name} must be >= 0 (got {v}); the sign of each "
                                 "term is fixed by the formula, not by the weight")

    def as_dict(self) -> Dict[str, float]:
        return {"w_obj": float(self.w_obj), "w_hall": float(self.w_hall),
                "w_rel": float(self.w_rel)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "RerankWeights":
        unknown = set(d) - {"w_obj", "w_hall", "w_rel"}
        if unknown:
            raise ValueError(f"unknown weight(s): {sorted(unknown)}")
        return cls(w_obj=float(d.get("w_obj", 0.0)), w_hall=float(d.get("w_hall", 0.0)),
                   w_rel=float(d.get("w_rel", 0.0)))

    def without_relation(self) -> "RerankWeights":
        return RerankWeights(self.w_obj, self.w_hall, 0.0)

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Evidence (system perception only - never ground truth)
# ---------------------------------------------------------------------------

def evidence_objects(detections: Iterable[Mapping]) -> frozenset:
    """COCO-80 classes the system believes it saw.

    Takes the CLIP-verified YOLO detections and maps their labels through the
    SAME mapper the evaluation applies to captions, so "supported" is decided
    in one vocabulary. A detection whose label is outside COCO-80 contributes
    nothing rather than silently matching a neighbouring class.
    """
    return frozenset(objects_from_names(str(d["label"]) for d in detections))


def relation_stated(caption: str, relation: Optional[Mapping]) -> bool:
    """Does `caption` state the predicted triple, in subject-predicate-object order?

    Matching is done on the mention level, not on surface strings, so "a man
    riding a motorcycle" states (person, riding, motorcycle): `man` maps to
    `person` through the evaluation's own mapper. The predicate words must lie
    strictly between a subject mention and a later object mention, which keeps
    "a person next to a bicycle" from counting as "person riding bicycle".
    """
    if not relation:
        return False
    subject = objects_from_names([str(relation["subject"])])
    obj = objects_from_names([str(relation["object"])])
    predicate = tokenize(str(relation["predicate"]).replace("_", " "))
    if not subject or not obj or not predicate:
        return False

    tokens = tokenize(caption)
    spans = located_mention_spans(caption)
    subject_ends = [end for _, end, _, cls in spans if cls in subject]
    object_starts = [start for start, _, _, cls in spans if cls in obj]
    for a in subject_ends:
        for b in object_starts:
            if b <= a:
                continue
            window = tokens[a:b]
            n = len(predicate)
            if any(window[i:i + n] == predicate for i in range(len(window) - n + 1)):
                return True
    return False


# ---------------------------------------------------------------------------
# Candidate hygiene
# ---------------------------------------------------------------------------

def is_degenerate(text: str) -> Optional[str]:
    """Reason this candidate is unusable, or None if it is fine.

    Beam search with a 128-token budget occasionally emits an empty string or
    a phrase loop. Dropping those is hygiene, not scoring: the bounds are fixed
    constants and are never tuned on any split.
    """
    words = str(text).split()
    if not words:
        return "empty"
    if len(words) < MIN_CANDIDATE_WORDS:
        return f"fewer_than_{MIN_CANDIDATE_WORDS}_words"
    if len(words) > MAX_CANDIDATE_WORDS:
        return f"more_than_{MAX_CANDIDATE_WORDS}_words"
    lowered = [w.lower() for w in words]
    counts: Dict[Tuple[str, ...], int] = {}
    for i in range(len(lowered) - REPEAT_NGRAM + 1):
        key = tuple(lowered[i:i + REPEAT_NGRAM])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= MAX_NGRAM_REPEATS:
            return f"{REPEAT_NGRAM}gram_repeated_{MAX_NGRAM_REPEATS}_times"
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_candidate(candidate: Mapping, evidence: frozenset, relation: Optional[Mapping],
                    weights: RerankWeights) -> Dict:
    """Score one candidate. Pure function of (candidate, evidence, relation, weights)."""
    text = str(candidate["text"])
    said = mentions(text)
    support = len(said & evidence)
    unsupported = len(said - evidence)
    stated = relation_stated(text, relation)
    confidence = float(relation["confidence"]) if relation else 0.0
    lm = float(candidate["lm_score"])
    if not math.isfinite(lm):
        raise ValueError(f"candidate {text!r} has a non-finite lm_score {lm}")
    terms = {
        "lm": lm,
        "object_support": weights.w_obj * support,
        "hallucination_penalty": -weights.w_hall * unsupported,
        "relation_support": weights.w_rel * confidence * (1.0 if stated else 0.0),
    }
    return {
        "score": sum(terms.values()),
        "terms": terms,
        "n_supported": support,
        "n_unsupported": unsupported,
        "mentioned": sorted(said),
        "relation_stated": stated,
    }


def _sort_key(scored: Mapping) -> Tuple:
    return (
        -round(float(scored["score"]), SCORE_PRECISION),
        SOURCE_RANK.get(str(scored["source"]), len(SOURCE_ORDER)),
        int(scored["beam_rank"]),
        str(scored["text"]),
    )


def rerank(candidates: Sequence[Mapping], evidence: frozenset,
           relation: Optional[Mapping], weights: RerankWeights) -> Dict:
    """Pick the best-scoring candidate.

    Returns the selection, every candidate's score, and why any candidate was
    dropped. Deterministic: equal scores are broken towards the plain baseline
    candidate and then towards the earlier beam, never by dict or float order.

    Raises ValueError if no candidate survives - the caller decides what to do
    with an image BLIP produced nothing usable for, rather than this function
    inventing text.
    """
    scored: List[Dict] = []
    dropped: List[Dict] = []
    for c in candidates:
        reason = is_degenerate(c["text"])
        if reason:
            dropped.append({"text": c["text"], "source": c.get("source"),
                            "beam_rank": c.get("beam_rank"), "reason": reason})
            continue
        s = score_candidate(c, evidence, relation, weights)
        scored.append({"text": str(c["text"]), "source": str(c["source"]),
                       "beam_rank": int(c["beam_rank"]), **s})
    if not scored:
        raise ValueError("no usable candidate: "
                         + json.dumps(dropped or [{"reason": "no_candidates"}]))
    ordered = sorted(scored, key=_sort_key)
    best = ordered[0]
    return {
        "caption": best["text"],
        "source": best["source"],
        "beam_rank": best["beam_rank"],
        "score": best["score"],
        "terms": best["terms"],
        "relation_stated": best["relation_stated"],
        "n_supported": best["n_supported"],
        "n_unsupported": best["n_unsupported"],
        "n_candidates_scored": len(scored),
        "n_candidates_dropped": len(dropped),
        "dropped": dropped,
        "ranking": [{"text": s["text"], "source": s["source"], "beam_rank": s["beam_rank"],
                     "score": s["score"], "terms": s["terms"],
                     "relation_stated": s["relation_stated"],
                     "n_supported": s["n_supported"], "n_unsupported": s["n_unsupported"]}
                    for s in ordered],
    }


def first_candidate(candidates: Sequence[Mapping]) -> Optional[Mapping]:
    """BLIP's own first choice: the rank-0 beam of the plain baseline prefix.

    This is the reference the "reranker changed the caption" statistics are
    measured against, and (with the experiment's decoding parameters) it is
    byte-identical to the `baseline` arm's caption.
    """
    for c in candidates:
        if c.get("source") == "baseline" and int(c.get("beam_rank", -1)) == 0:
            return c
    return None

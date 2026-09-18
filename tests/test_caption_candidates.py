"""Multi-candidate BLIP generation and uniform candidate scoring.

BLIP is replaced by a stub whose processor and model implement exactly the
surface `utils/caption_candidates.py` uses, so these tests check the plumbing
the real run depends on - which prefixes are generated from, batch shapes, the
BOS substitution, deduplication, the shared image embedding and the fallback
to the public forward - without downloading 1 GB of weights. The equivalence
of the fast and fallback scoring paths on the real model is asserted again by
`run_caption_experiment.py smoke`.
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import utils.blip_captioner as blip  # noqa: E402
from utils.blip_captioner import BASELINE_PREFIX  # noqa: E402
from utils.caption_candidates import (  # noqa: E402
    DEFAULT_NUM_CANDIDATES,
    candidate_prefixes,
    generate_and_score,
    generate_candidates,
    score_candidates,
)

RELATION = {"subject": "person", "predicate": "riding", "object": "bicycle",
            "confidence": 0.78}

VOCAB = ["[BOS]", "[PAD]", "a", "photo", "of", "person", "riding", "bicycle", "cat",
         "bed", "and", "on", "street"]
STOI = {w: i for i, w in enumerate(VOCAB)}


class StubTokenizer:
    pad_token_id = STOI["[PAD]"]

    def __call__(self, texts, return_tensors=None, padding=True, truncation=True,
                 max_length=512):
        rows = [[STOI.get(w, 2) for w in t.split()][:max_length] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class StubBatch(dict):
    def to(self, device):
        return self


class StubProcessor:
    def __init__(self):
        self.tokenizer = StubTokenizer()
        self.generated_for = []

    def __call__(self, images=None, text=None, return_tensors=None):
        batch = StubBatch(pixel_values=torch.zeros(1, 3, 8, 8))
        if text is not None:
            batch["input_ids"] = torch.tensor(
                [[STOI["[BOS]"]] + [STOI.get(w, 2) for w in text.split()]])
        return batch

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(VOCAB[int(i)] for i in ids
                        if int(i) not in (STOI["[BOS]"], STOI["[PAD]"]))


class StubGenerated:
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.sequences_scores = torch.tensor(scores)


class StubTextDecoder:
    def __init__(self, owner):
        self.owner = owner

    def __call__(self, input_ids=None, attention_mask=None, encoder_hidden_states=None,
                 encoder_attention_mask=None):
        if self.owner.text_decoder_fails:
            raise TypeError("unexpected keyword argument")
        self.owner.decoder_calls.append({
            "input_ids": input_ids.clone(),
            "attention_mask": attention_mask.clone(),
            "encoder_shape": tuple(encoder_hidden_states.shape),
            "encoder_mask_shape": tuple(encoder_attention_mask.shape),
        })
        return _Logits(self.owner._logits(input_ids))


class _Logits:
    def __init__(self, logits):
        self.logits = logits


class StubConfig:
    class text_config:
        bos_token_id = STOI["[BOS]"]


class StubModel:
    config = StubConfig()

    def __init__(self):
        self.text_decoder = StubTextDecoder(self)
        self.text_decoder_fails = False
        self.decoder_calls = []
        self.forward_calls = []
        self.vision_calls = 0
        self._param = torch.zeros(1, dtype=torch.float32)
        # prefix -> the beams beam search "finds" for it
        self.beams = {
            BASELINE_PREFIX: ["a photo of a cat on a bed", "a photo of a cat on a street",
                              "a photo of a cat", "a photo of a cat on a bed"],
            "a photo of a person and a bicycle":
                ["a photo of a person and a bicycle", "a photo of a person and a bicycle on a street",
                 "a photo of a person and a bicycle on a bed", "a photo of a person"],
            "a photo of a person riding a bicycle":
                ["a photo of a person riding a bicycle",
                 "a photo of a person riding a bicycle on a street",
                 "a photo of a person riding a bicycle on a bed",
                 "a photo of a person riding a bicycle on a bed"],
        }

    def parameters(self):
        return iter([self._param])

    def vision_model(self, pixel_values=None):
        self.vision_calls += 1
        return (torch.ones(pixel_values.shape[0], 5, 4),)

    def _logits(self, input_ids):
        # Deterministic, token-dependent logits: token t scores highest when it
        # equals the previous token's index + 1 (mod |V|).
        b, length = input_ids.shape
        logits = torch.zeros(b, length, len(VOCAB))
        for i in range(b):
            for j in range(length):
                logits[i, j, (int(input_ids[i, j]) + 1) % len(VOCAB)] = 5.0
        return logits

    def __call__(self, pixel_values=None, input_ids=None, attention_mask=None):
        self.forward_calls.append(tuple(pixel_values.shape))
        return _Logits(self._logits(input_ids))

    def generate(self, pixel_values=None, input_ids=None, attention_mask=None,
                 num_return_sequences=1, output_scores=False, return_dict_in_generate=False,
                 **cfg):
        prefix = " ".join(VOCAB[int(i)] for i in input_ids[0] if int(i) != STOI["[BOS]"])
        beams = self.beams[prefix][:num_return_sequences]
        seqs = [torch.tensor([STOI["[BOS]"]] + [STOI[w] for w in b.split()]) for b in beams]
        return StubGenerated(seqs, [-1.0 - 0.1 * i for i in range(len(seqs))])


@pytest.fixture
def stub_blip(monkeypatch):
    processor, model = StubProcessor(), StubModel()
    monkeypatch.setattr(blip, "_processor", processor)
    monkeypatch.setattr(blip, "_model", model)
    monkeypatch.setattr(blip, "_device", torch.device("cpu"))
    return processor, model


IMAGE = object()   # the stub processor never looks at it


# ---------------------------------------------------------------------------
# Prefixes
# ---------------------------------------------------------------------------

def test_without_a_relation_only_the_baseline_prefix_is_generated_from():
    assert candidate_prefixes(None) == {"baseline": BASELINE_PREFIX}


def test_with_a_relation_all_three_prefixes_are_built_in_a_fixed_order():
    prefixes = candidate_prefixes(RELATION)
    assert list(prefixes) == ["baseline", "objects_only", "relation"]
    assert prefixes["objects_only"] == "a photo of a person and a bicycle"
    assert prefixes["relation"] == "a photo of a person riding a bicycle"


def test_a_low_confidence_relation_is_still_offered_as_a_candidate_source():
    """Confidence is scored by the reranker, not used as a generation gate."""
    prefixes = candidate_prefixes({**RELATION, "confidence": 0.01})
    assert prefixes["relation"] == "a photo of a person riding a bicycle"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def test_generate_candidates_returns_k_beams_per_prefix(stub_blip):
    out = generate_candidates(IMAGE, candidate_prefixes(RELATION), num_candidates=4)
    assert len(out) == 12
    assert [c["source"] for c in out[:4]] == ["baseline"] * 4
    assert [c["beam_rank"] for c in out[:4]] == [0, 1, 2, 3]
    assert {c["source"] for c in out} == {"baseline", "objects_only", "relation"}


def test_beam_rank_zero_of_each_prefix_is_that_prefix_single_caption(stub_blip):
    """The pool provably contains the baseline / objects-only / grounded arms."""
    pool = generate_candidates(IMAGE, candidate_prefixes(RELATION), num_candidates=4)
    single = generate_candidates(IMAGE, candidate_prefixes(RELATION), num_candidates=1)
    for source in ("baseline", "objects_only", "relation"):
        rank0 = next(c for c in pool if c["source"] == source and c["beam_rank"] == 0)
        alone = next(c for c in single if c["source"] == source)
        assert rank0["text"] == alone["text"]


def test_generate_candidates_is_deterministic(stub_blip):
    a = generate_candidates(IMAGE, candidate_prefixes(RELATION))
    b = generate_candidates(IMAGE, candidate_prefixes(RELATION))
    assert a == b


def test_num_candidates_above_num_beams_is_refused(stub_blip):
    with pytest.raises(ValueError, match="num_beams"):
        generate_candidates(IMAGE, candidate_prefixes(None), num_candidates=99)
    with pytest.raises(ValueError):
        generate_candidates(IMAGE, candidate_prefixes(None), num_candidates=0)


def test_default_num_candidates_equals_the_experiment_beam_width():
    assert DEFAULT_NUM_CANDIDATES == blip.EXPERIMENT_GENERATION_CONFIG["num_beams"]


def test_generation_uses_the_frozen_decoding_parameters(stub_blip):
    captured = {}
    _, model = stub_blip
    original = model.generate

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    model.generate = spy
    generate_candidates(IMAGE, candidate_prefixes(None))
    for key, value in blip.EXPERIMENT_GENERATION_CONFIG.items():
        assert captured[key] == value


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_score_candidates_attaches_a_finite_lm_score_to_every_candidate(stub_blip):
    pool = generate_candidates(IMAGE, candidate_prefixes(RELATION))
    scored = score_candidates(IMAGE, pool, batch_size=5)
    assert len(scored) == len(pool)
    assert all(math.isfinite(c["lm_score"]) for c in scored)
    assert all(c["lm_score"] <= 0.0 for c in scored)


def test_identical_texts_receive_identical_scores_and_are_encoded_once(stub_blip):
    _, model = stub_blip
    pool = [{"text": "a photo of a cat", "source": "baseline", "beam_rank": i}
            for i in range(3)]
    scored = score_candidates(IMAGE, pool, batch_size=8)
    assert len({c["lm_score"] for c in scored}) == 1
    assert model.decoder_calls[0]["input_ids"].shape[0] == 1


def test_the_image_is_encoded_once_per_image_not_once_per_candidate(stub_blip):
    _, model = stub_blip
    pool = generate_candidates(IMAGE, candidate_prefixes(RELATION))
    score_candidates(IMAGE, pool, batch_size=4)
    assert model.vision_calls == 1


def test_batches_have_the_expected_shapes_and_share_one_image_embedding(stub_blip):
    _, model = stub_blip
    pool = generate_candidates(IMAGE, candidate_prefixes(RELATION))
    score_candidates(IMAGE, pool, batch_size=3)
    unique = len({c["text"] for c in pool})
    assert sum(call["input_ids"].shape[0] for call in model.decoder_calls) == unique
    for call in model.decoder_calls:
        b, length = call["input_ids"].shape
        assert b <= 3
        assert call["attention_mask"].shape == (b, length)
        assert call["encoder_shape"] == (b, 5, 4)
        assert call["encoder_mask_shape"] == (b, 5)


def test_the_leading_token_is_replaced_by_the_decoder_bos(stub_blip):
    _, model = stub_blip
    score_candidates(IMAGE, [{"text": "a photo of a cat", "source": "baseline",
                              "beam_rank": 0}])
    ids = model.decoder_calls[0]["input_ids"]
    assert int(ids[0, 0]) == StubConfig.text_config.bos_token_id


def test_padding_does_not_change_a_candidate_score(stub_blip):
    short = {"text": "a photo of a cat", "source": "baseline", "beam_rank": 0}
    long = {"text": "a photo of a person riding a bicycle on a street",
            "source": "relation", "beam_rank": 0}
    alone = score_candidates(IMAGE, [short])[0]["lm_score"]
    together = score_candidates(IMAGE, [short, long])[0]["lm_score"]
    assert alone == pytest.approx(together, abs=1e-6)


def test_batch_size_does_not_change_a_candidate_score(stub_blip):
    pool = generate_candidates(IMAGE, candidate_prefixes(RELATION))
    a = {c["text"]: c["lm_score"] for c in score_candidates(IMAGE, pool, batch_size=1)}
    b = {c["text"]: c["lm_score"] for c in score_candidates(IMAGE, pool, batch_size=12)}
    assert a.keys() == b.keys()
    for text in a:
        assert a[text] == pytest.approx(b[text], abs=1e-6)


def test_scoring_falls_back_to_the_public_forward_and_agrees_with_it(stub_blip):
    """A transformers version that changes the decoder signature must degrade to
    the slower path, not to a wrong number."""
    _, model = stub_blip
    pool = generate_candidates(IMAGE, candidate_prefixes(RELATION))
    fast = {c["text"]: c["lm_score"] for c in score_candidates(IMAGE, pool, batch_size=6)}
    model.text_decoder_fails = True
    slow = {c["text"]: c["lm_score"] for c in score_candidates(IMAGE, pool, batch_size=6)}
    assert model.forward_calls, "fallback path was not exercised"
    assert all(shape[0] <= 6 for shape in model.forward_calls)
    for text in fast:
        assert fast[text] == pytest.approx(slow[text], abs=1e-6)


def test_scoring_an_empty_pool_returns_an_empty_pool(stub_blip):
    assert score_candidates(IMAGE, []) == []


def test_generate_and_score_produces_a_ready_to_rerank_pool(stub_blip):
    out = generate_and_score(IMAGE, RELATION, num_candidates=4, batch_size=6)
    assert list(out["prefixes"]) == ["baseline", "objects_only", "relation"]
    assert len(out["candidates"]) == 12
    for c in out["candidates"]:
        assert set(c) >= {"source", "prefix", "beam_rank", "text", "lm_score"}


def test_generate_and_score_without_a_relation_uses_the_baseline_prefix_only(stub_blip):
    out = generate_and_score(IMAGE, None, num_candidates=4)
    assert list(out["prefixes"]) == ["baseline"]
    assert {c["source"] for c in out["candidates"]} == {"baseline"}

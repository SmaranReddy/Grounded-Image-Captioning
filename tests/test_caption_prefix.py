"""The relation must reach BLIP, and nothing else may differ between arms.

A fake BLIP records exactly what it was conditioned on, so these tests run
offline on CPU. They pin:
  * the baseline, grounded and objects-only prefixes;
  * that the grounded BLIP input equals the baseline input plus exactly the
    relation span;
  * low-confidence / missing relations fall back to the baseline prefix, and
    no placeholder text is ever emitted as a caption;
  * generation applies no gating or relation correction;
  * the evaluator refuses a run whose grounded arm did not actually use the
    relation it was supposed to.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import utils.blip_captioner as blip  # noqa: E402
from utils.blip_captioner import (  # noqa: E402
    BASELINE_PREFIX,
    build_blip_prefix,
    build_objects_only_prefix,
    generate_blip_from_prefix,
)

REL = {"subject": "person", "predicate": "riding", "object": "bicycle", "confidence": 0.8}


# ---------------------------------------------------------------------------
# prefixes
# ---------------------------------------------------------------------------

def test_prefix_strings():
    assert BASELINE_PREFIX == "a photo of"
    assert build_blip_prefix([], [REL], min_confidence=0.5) == ("a photo of a person riding a bicycle", [REL])
    assert build_objects_only_prefix(REL) == "a photo of a person and a bicycle"
    assert build_objects_only_prefix({**REL, "object": "umbrella"}) == "a photo of a person and an umbrella"


def test_low_confidence_and_missing_relations_fall_back_to_baseline():
    assert build_blip_prefix([], [{**REL, "confidence": 0.49}], min_confidence=0.5) == (BASELINE_PREFIX, [])
    assert build_blip_prefix([], [], min_confidence=0.5) == (BASELINE_PREFIX, [])


@pytest.mark.parametrize("predicate", sorted(__import__(
    "relation_prediction.vg_dataset", fromlist=["ALLOWED_PREDICATES"]).ALLOWED_PREDICATES))
def test_every_model_predicate_can_be_injected(predicate):
    """A predicate missing from the prefix table would be dropped silently."""
    prefix, used = build_blip_prefix([], [{**REL, "predicate": predicate}], min_confidence=0.0)
    assert used and prefix != BASELINE_PREFIX and predicate in prefix


# ---------------------------------------------------------------------------
# fake BLIP
# ---------------------------------------------------------------------------

class _Batch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"[CLS]": 0, "[SEP]": 1}
        self.inv = {0: "[CLS]", 1: "[SEP]"}

    def ids(self, text):
        out = []
        for w in text.lower().split():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab)
                self.inv[self.vocab[w]] = w
            out.append(self.vocab[w])
        return out

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.ids(text)}


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.texts = []

    def __call__(self, images, text, return_tensors):
        self.texts.append(text)
        mean = sum(images.convert("L").tobytes()) / (images.width * images.height)
        return _Batch(pixel_values=torch.tensor([[mean]]),
                      input_ids=torch.tensor([[0] + self.tokenizer.ids(text) + [1]]))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.tokenizer.inv[int(i)] for i in ids if int(i) not in (0, 1))


class FakeBlip(torch.nn.Module):
    """Continues the prefix with an image-dependent phrase."""

    def __init__(self, tokenizer):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.tok = tokenizer
        self.calls = []

    def generate(self, input_ids, pixel_values, **cfg):
        self.calls.append({"input_ids": input_ids[0].tolist(), "cfg": cfg})
        cont = "near a dog" if float(pixel_values[0, 0]) > 100 else "on a street"
        return torch.tensor([input_ids[0, :-1].tolist() + self.tok.ids(cont) + [1]])


@pytest.fixture()
def fake_blip(monkeypatch):
    proc = FakeProcessor()
    model = FakeBlip(proc.tokenizer)
    monkeypatch.setattr(blip, "_processor", proc)
    monkeypatch.setattr(blip, "_model", model)
    monkeypatch.setattr(blip, "_device", torch.device("cpu"))

    def _no_gating(*a, **k):
        raise AssertionError("caption experiment must not gate or correct captions")
    monkeypatch.setattr(blip, "gate_caption", _no_gating)
    monkeypatch.setattr(blip, "correct_caption_relations", _no_gating)
    return proc, model


def test_grounded_input_is_baseline_plus_exactly_the_relation_span(fake_blip):
    proc, model = fake_blip
    img = Image.new("RGB", (32, 32), (200, 200, 200))
    base = generate_blip_from_prefix(img, BASELINE_PREFIX)
    prefix, _ = build_blip_prefix([], [REL], min_confidence=0.5)
    grd = generate_blip_from_prefix(img, prefix)
    span = proc.tokenizer(" a person riding a bicycle", add_special_tokens=False)["input_ids"]
    assert grd["input_ids"] == base["input_ids"][:-1] + span + base["input_ids"][-1:]
    assert proc.texts == [BASELINE_PREFIX, "a photo of a person riding a bicycle"]
    assert grd["caption"] == "a photo of a person riding a bicycle near a dog"
    assert base["caption"] == "a photo of near a dog"
    assert grd["prefix_echoed"] and base["prefix_echoed"]
    assert model.calls[0]["cfg"] == model.calls[1]["cfg"] == blip.EXPERIMENT_GENERATION_CONFIG


# ---------------------------------------------------------------------------
# generate -> evaluate on a synthetic run directory
# ---------------------------------------------------------------------------

def _relations_record(iid, file, selected, decision):
    eligible = [{"label": "person", "box": [0, 0, 20, 30], "score": 0.9},
                {"label": "bicycle", "box": [5, 10, 30, 30], "score": 0.8}]
    return {"image_id": iid, "file": file, "image_size": [32, 32],
            "raw_detections": eligible, "verified_detections": eligible,
            "eligible_detections": eligible if selected else eligible[:1],
            "dropped_detections": [], "pair_predictions": [],
            "selected_relation": selected, "decision": decision, "threshold": 0.5}


@pytest.fixture()
def run_dir(tmp_path):
    ids = ["101", "102", "103"]
    files = {}
    for iid, shade in zip(ids, (200, 50, 200)):
        p = tmp_path / f"{iid}.png"
        Image.new("RGB", (32, 32), (shade, shade, shade)).save(p)
        files[iid] = str(p)
    usable_sha = hashlib.sha256(",".join(ids).encode()).hexdigest()
    eval_set = {"meta": {"status": "OK", "split": "test", "usable_ids_sha256": usable_sha,
                         "final": 3},
                "usable_ids": ids,
                "images": {i: {"objects": ["person", "bicycle"], "file": files[i]} for i in ids}}
    es_path = tmp_path / "eval_set.json"
    es_path.write_text(json.dumps(eval_set))
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "run_config.json").write_text(json.dumps({
        "eval_set": {"path": str(es_path), "usable_ids_sha256": usable_sha, "split": "test"},
        "image_ids": ids, "relation_threshold": 0.5, "relation_threshold_is_default": True,
        "relation_checkpoint": {"checkpoint_dir": "x"}}))
    sel = {k: REL[k] for k in ("subject", "predicate", "object")}
    recs = [
        _relations_record("101", files["101"], {**sel, "confidence": 0.8},
                          {"use_relation": True, "fallback_reason": None}),
        _relations_record("102", files["102"], {**sel, "confidence": 0.3},
                          {"use_relation": False, "fallback_reason": "below_confidence_threshold"}),
        _relations_record("103", files["103"], None,
                          {"use_relation": False, "fallback_reason": "fewer_than_2_eligible_detections"}),
    ]
    with open(rd / "relations.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return rd


def _read(path):
    return {json.loads(l)["image_id"]: json.loads(l) for l in open(path) if l.strip()}


def test_generate_all_arms_and_evaluate(fake_blip, run_dir):
    from run_caption_experiment import run_generate
    from utils.caption_experiment_eval import evaluate_run

    for arm in ("baseline", "grounded", "objects_only"):
        run_generate(run_dir, arm, "float32")
    base = _read(run_dir / "captions_baseline.jsonl")
    grd = _read(run_dir / "captions_grounded.jsonl")
    obj = _read(run_dir / "captions_objects_only.jsonl")

    assert {r["prefix"] for r in base.values()} == {BASELINE_PREFIX}
    # used relation
    assert grd["101"]["relation_used"] and grd["101"]["prefix"] == "a photo of a person riding a bicycle"
    assert obj["101"]["prefix"] == "a photo of a person and a bicycle"
    assert grd["101"]["input_ids"] != base["101"]["input_ids"]
    # below threshold: baseline prefix, but relation caption kept for the sweep
    assert not grd["102"]["relation_used"] and grd["102"]["prefix"] == BASELINE_PREFIX
    assert grd["102"]["fallback_reason"] == "below_confidence_threshold"
    assert grd["102"]["relation_caption"].startswith("a photo of a person riding a bicycle")
    assert grd["102"]["caption"] == base["102"]["caption"]
    # no relation at all
    assert grd["103"]["relation"] is None and grd["103"]["caption"] == base["103"]["caption"]
    for r in list(base.values()) + list(grd.values()) + list(obj.values()):
        assert "cannot infer" not in r["caption"] and "the scene contains" not in r["caption"].lower()

    res = evaluate_run(run_dir, allow_small=True, n_bootstrap=50)
    assert res["validity"]["problems"] == []
    rs = res["relation_statistics"]
    assert rs["images_relation_used"] == 1 and rs["relation_usage_rate"] == pytest.approx(1 / 3)
    assert rs["fallback_reasons"] == {"below_confidence_threshold": 1,
                                      "fewer_than_2_eligible_detections": 1}
    assert res["validity"]["per_arm"]["grounded"]["caption_differs_from_baseline"] == 1
    # baseline "a photo of near a dog" hallucinates dog and mentions nothing true
    assert res["metrics_all_images"]["baseline"]["chair"]["chair_s"] == pytest.approx(2 / 3)
    assert res["exploratory_threshold_sweep"]["0.0"]["relation_usage_rate"] == pytest.approx(2 / 3)
    src = res["metrics_all_images"]["grounded"]["hallucination_source"]
    assert src == {"total_hallucinated": 2, "introduced_by_injected_prefix": 0,
                   "in_blip_continuation": 2}                       # the "dog" is BLIP's
    assert (run_dir / "caption_results.md").is_file()


def test_evaluator_refuses_a_fake_grounded_path(fake_blip, run_dir):
    from run_caption_experiment import run_generate
    from utils.caption_experiment_eval import evaluate_run

    for arm in ("baseline", "grounded"):
        run_generate(run_dir, arm, "float32")
    path = run_dir / "captions_grounded.jsonl"
    rows = [json.loads(l) for l in open(path)]
    base = _read(run_dir / "captions_baseline.jsonl")
    for r in rows:
        if r["image_id"] == "101":          # pretend the relation never reached BLIP
            r["prefix"], r["input_ids"], r["caption"] = (BASELINE_PREFIX, base["101"]["input_ids"],
                                                         base["101"]["caption"])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="not scoreable"):
        evaluate_run(run_dir, allow_small=True, n_bootstrap=50)


def test_evaluator_refuses_placeholder_captions(fake_blip, run_dir):
    from run_caption_experiment import run_generate
    from utils.caption_experiment_eval import evaluate_run

    for arm in ("baseline", "grounded"):
        run_generate(run_dir, arm, "float32")
    path = run_dir / "captions_grounded.jsonl"
    rows = [json.loads(l) for l in open(path)]
    rows[2]["caption"] = "Only one object detected — cannot infer relations."
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="placeholder"):
        evaluate_run(run_dir, allow_small=True, n_bootstrap=50)


def test_evaluator_requires_a_full_test_set(fake_blip, run_dir):
    from run_caption_experiment import run_generate
    from utils.caption_experiment_eval import evaluate_run

    for arm in ("baseline", "grounded"):
        run_generate(run_dir, arm, "float32")
    with pytest.raises(ValueError, match="too small"):
        evaluate_run(run_dir, n_bootstrap=50)


def test_generation_settings_cannot_be_mixed_within_an_arm(fake_blip, run_dir):
    from run_caption_experiment import run_generate

    run_generate(run_dir, "baseline", "float32")
    meta = json.loads((run_dir / "captions_baseline.meta.json").read_text())
    meta["blip_dtype"] = "float16"
    (run_dir / "captions_baseline.meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit):
        run_generate(run_dir, "baseline", "float32")

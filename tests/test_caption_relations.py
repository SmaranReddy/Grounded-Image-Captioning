"""The caption experiment must run the relation model that was evaluated.

* a geometry+CLIP+union checkpoint in the exact format the trainer writes loads
  at input_dim 2451 through the evaluator's loader, and anything else is refused;
* the legacy caption pipeline's width formula misreports that checkpoint as
  1669 (the regression this fixes);
* checkpoint selection reads validation metrics only;
* caption-time features are built the way the training cache was;
* prediction mirrors eval_gt_relations.py (raw argmax over valid predicates);
* selection, semantic filtering and threshold fallbacks behave as declared.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from relation_prediction.model import RelationMLP  # noqa: E402
from relation_prediction.vg_dataset import Vocab, extract_geo_features_ext  # noqa: E402
from utils.caption_relations import (  # noqa: E402
    EXPECTED_CONFIG,
    CheckpointMismatch,
    build_pair_batch,
    eligible_detections,
    load_relation_bundle,
    predict_pairs,
    relation_decision,
    select_checkpoint,
    select_relation,
    union_box,
)

LABEL_VOCAB = os.path.join(ROOT, "checkpoints_e2_geo", "label_vocab.json")
PRED_VOCAB = os.path.join(ROOT, "checkpoints_e2_geo", "pred_vocab.json")

VARIANT_DIMS = {
    "geometry": dict(geo_dim=19, geo_norm=True, clip_dim=0, union_dim=0, geo_mode="ext"),
    "geometry_clip": dict(geo_dim=19, geo_norm=True, clip_dim=768, union_dim=0, geo_mode="ext"),
    "geometry_clip_union": dict(geo_dim=19, geo_norm=True, clip_dim=768, union_dim=768, geo_mode="ext"),
    "clip_only": dict(geo_dim=0, geo_norm=False, clip_dim=768, union_dim=0, geo_mode="none"),
}


def write_checkpoint(directory, variant="geometry_clip_union", seed=42, val_acc=0.6,
                     val_macro_f1=0.4, config_overrides=None):
    """Write a checkpoint exactly as train_full_visual_semantic._save_checkpoint does."""
    os.makedirs(directory, exist_ok=True)
    lv, pv = Vocab.load(LABEL_VOCAB), Vocab.load(PRED_VOCAB)
    v = VARIANT_DIMS[variant]
    torch.manual_seed(seed)
    model = RelationMLP(len(lv), len(pv), embed_dim=64, hidden_dims=(256, 128),
                        clip_dim=v["clip_dim"], union_dim=v["union_dim"],
                        geo_dim=v["geo_dim"], geo_norm=v["geo_norm"])
    cfg = {"model_type": "mlp", "num_labels": len(lv), "num_predicates": len(pv),
           "embed_dim": 64, "clip_dim": v["clip_dim"], "pose_dim": 0,
           "union_dim": v["union_dim"], "hidden_dims": [256, 128], "geo_dim": v["geo_dim"],
           "geo_norm": v["geo_norm"], "geo_mode": v["geo_mode"], "predicate_scheme": "v1",
           "visual_filter_only": False, "require_visual": True,
           "input_dim": model.input_dim,
           "feature_blocks": [[n, w] for n, w in model.feature_blocks()]}
    cfg.update(config_overrides or {})
    torch.save({"model_state_dict": model.state_dict(), "model_config": cfg},
               os.path.join(directory, "relation_mlp.pt"))
    lv.save(os.path.join(directory, "label_vocab.json"))
    pv.save(os.path.join(directory, "pred_vocab.json"))
    with open(os.path.join(directory, "training_meta.json"), "w") as fh:
        json.dump({"epoch": 20, "val_acc": val_acc, "val_macro_f1": val_macro_f1,
                   "seed": seed, "select_metric": "top1",
                   "split": {"n_samples_val": 10482}}, fh)
    return model


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def test_union_checkpoint_loads_at_2451(tmp_path):
    write_checkpoint(tmp_path / "ck")
    b = load_relation_bundle(str(tmp_path / "ck"), device=torch.device("cpu"))
    assert b.config["input_dim"] == 2451 == EXPECTED_CONFIG["input_dim"]
    assert b.model.feature_blocks() == [("subj_label", 64), ("obj_label", 64), ("geometry", 19),
                                        ("subj_clip", 768), ("obj_clip", 768), ("union_clip", 768)]
    assert b.geo_fn is extract_geo_features_ext
    assert len(b.valid_idxs) == len(b.pred_vocab) - 2


@pytest.mark.parametrize("variant", ["geometry", "geometry_clip", "clip_only"])
def test_other_ablation_arms_are_refused(tmp_path, variant):
    write_checkpoint(tmp_path / variant, variant=variant)
    with pytest.raises(CheckpointMismatch):
        load_relation_bundle(str(tmp_path / variant), device=torch.device("cpu"))


def test_config_that_disagrees_with_weights_is_refused(tmp_path):
    write_checkpoint(tmp_path / "ck", config_overrides={"input_dim": 1669})
    with pytest.raises(SystemExit):
        load_relation_bundle(str(tmp_path / "ck"), device=torch.device("cpu"))


def test_legacy_width_formula_misreports_the_union_checkpoint(tmp_path):
    """grounded_caption_pipeline.verify_relation_model used 2*embed + 5 + 2*clip."""
    from relation_prediction import predict as rp
    write_checkpoint(tmp_path / "ck")
    rp.load_relation_model(str(tmp_path / "ck"))
    legacy = 2 * rp._model.label_emb.weight.shape[1] + 5 + 2 * rp._model_clip_dim
    assert legacy == 1669
    assert rp._model.input_dim == 2451


def test_predict_loads_clip_only_checkpoint(tmp_path):
    """geo_mode='none' used to be read back as 'basic' and refused."""
    from relation_prediction import predict as rp
    write_checkpoint(tmp_path / "ck", variant="clip_only")
    rp.load_relation_model(str(tmp_path / "ck"))
    assert rp._model_geo_mode == "none" and rp._model.input_dim == 1664


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def test_selection_uses_validation_not_test(tmp_path):
    root = tmp_path / "checkpoints_gpu"
    write_checkpoint(root / "geometry_clip_union_seed42", seed=42, val_acc=0.640)
    write_checkpoint(root / "geometry_clip_union_seed43", seed=43, val_acc=0.645)
    write_checkpoint(root / "geometry_clip_union_seed44", seed=44, val_acc=0.638)
    # A test-set result that would pick seed 44 if it were read.
    res = tmp_path / "results_gpu" / "geometry_clip_union_seed44"
    res.mkdir(parents=True)
    (res / "run.json").write_text(json.dumps({"metrics": {"top1": 0.99}}))
    sel = select_checkpoint(str(root))
    assert sel["selected"]["seed"] == 43
    assert "Test metrics are not read" in sel["rule"]


def test_selection_tie_breaks(tmp_path):
    root = tmp_path / "ck"
    write_checkpoint(root / "geometry_clip_union_seed42", seed=42, val_acc=0.64, val_macro_f1=0.40)
    write_checkpoint(root / "geometry_clip_union_seed43", seed=43, val_acc=0.64, val_macro_f1=0.41)
    write_checkpoint(root / "geometry_clip_union_seed44", seed=44, val_acc=0.64, val_macro_f1=0.41)
    assert select_checkpoint(str(root))["selected"]["seed"] == 43


def test_selection_refuses_incomplete_runs(tmp_path):
    root = tmp_path / "ck"
    write_checkpoint(root / "geometry_clip_union_seed42", seed=42)
    with pytest.raises(CheckpointMismatch):
        select_checkpoint(str(root))


def test_selection_module_never_mentions_results_gpu():
    import inspect
    import utils.caption_relations as cr
    assert "results_gpu" not in inspect.getsource(cr.select_checkpoint).split('"""')[2]


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

class RecordingEncoder:
    """Stands in for CLIPExtractor.encode_crops; returns a distinct vector per crop."""

    def __init__(self):
        self.calls = []

    def __call__(self, crops):
        self.calls.append([c.size for c in crops])
        out = torch.zeros(len(crops), 768)
        for k, c in enumerate(crops):
            out[k, 0] = c.size[0]
            out[k, 1] = c.size[1]
            out[k, 2] = sum(c.resize((2, 2)).tobytes())
        return out


DETS = [
    {"label": "person", "box": [10.0, 10.0, 60.0, 150.0], "score": 0.9},
    {"label": "bicycle", "box": [30.0, 80.0, 120.0, 160.0], "score": 0.8},
    {"label": "dog", "box": [150.0, 100.0, 199.0, 179.0], "score": 0.7},
]


def _image():
    img = Image.new("RGB", (200, 180))
    for x in range(200):
        for y in range(0, 180, 7):
            img.putpixel((x, y), (x % 256, y % 256, (x * y) % 256))
    return img


@pytest.fixture()
def bundle(tmp_path):
    write_checkpoint(tmp_path / "ck")
    return load_relation_bundle(str(tmp_path / "ck"), device=torch.device("cpu"))


def test_pair_batch_matches_feature_blocks_and_training_geometry(bundle):
    img = _image()
    enc = RecordingEncoder()
    batch = build_pair_batch(img, DETS, bundle, enc)
    n = len(DETS)
    assert len(batch["pairs"]) == n * (n - 1)
    bundle.model.check_inputs(batch["geo"], batch["subj_feat"], batch["obj_feat"],
                              batch["union_feat"])
    for row, (i, j) in enumerate(batch["pairs"]):
        expected = extract_geo_features_ext(tuple(DETS[i]["box"]), tuple(DETS[j]["box"]), 200.0, 180.0)
        assert batch["geo"][row].tolist() == pytest.approx(expected, abs=1e-6)
        assert batch["subj_idx"][row] == bundle.label_vocab[DETS[i]["label"]]


def test_each_crop_encoded_once_and_union_is_order_independent(bundle):
    img = _image()
    enc = RecordingEncoder()
    batch = build_pair_batch(img, DETS, bundle, enc)
    assert batch["n_object_crops"] == 3 and batch["n_union_crops"] == 3
    assert sum(len(c) for c in enc.calls) == 6          # not 3 * n(n-1) = 18
    rows = {pair: k for k, pair in enumerate(batch["pairs"])}
    assert torch.equal(batch["union_feat"][rows[(0, 1)]], batch["union_feat"][rows[(1, 0)]])
    assert torch.equal(batch["subj_feat"][rows[(0, 1)]], batch["obj_feat"][rows[(1, 0)]])


def test_union_and_crops_match_the_training_cache_definition(bundle):
    """build_clip_cache: union = bbox of RAW boxes, then clamp; crops clamp via _clamp_box."""
    import build_clip_cache
    from relation_prediction.clip_extractor import _clamp_box

    a, b = [-20.0, 5.0, 50.0, 60.0], [40.0, 30.0, 260.0, 90.0]
    ub = union_box(a, b)
    assert ub == (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
    for box in (a, b, list(ub)):
        assert _clamp_box(box, 200, 180) == build_clip_cache.clamp(box, 200, 180)

    img = _image()
    enc = RecordingEncoder()
    dets = [{"label": "person", "box": a, "score": 1.0}, {"label": "dog", "box": b, "score": 1.0}]
    build_pair_batch(img, dets, bundle, enc)
    object_sizes, union_sizes = enc.calls
    expect = [tuple(round(v) for v in (c[2] - c[0], c[3] - c[1]))
              for c in (_clamp_box(a, 200, 180), _clamp_box(b, 200, 180))]
    assert object_sizes == expect
    cu = _clamp_box(ub, 200, 180)
    assert union_sizes == [(round(cu[2] - cu[0]), round(cu[3] - cu[1]))]


def test_eligibility_mirrors_training_population(bundle):
    dets = DETS + [
        {"label": "tiny", "box": [0, 0, 50, 50], "score": 0.9},
        {"label": "cup", "box": [0.0, 0.0, 9.0, 40.0], "score": 0.9},        # < MIN_BOX_SIZE
        {"label": "dining_table", "box": [0.0, 0.0, 80.0, 80.0], "score": 0.9},
    ]
    kept, dropped = eligible_detections(dets, bundle.label_vocab, (200, 180))
    assert [d["label"] for d in kept] == ["person", "bicycle", "dog", "dining table"]
    assert {d["reason"] for d in dropped} == {"label_not_in_relation_vocab", "box_smaller_than_10px"}


def test_prediction_is_raw_argmax_over_valid_predicates(bundle):
    batch = build_pair_batch(_image(), DETS, bundle, RecordingEncoder())
    preds = predict_pairs(bundle, batch)
    with torch.no_grad():
        logits = bundle.model(batch["subj_idx"], batch["obj_idx"], batch["geo"],
                              subj_feat=batch["subj_feat"], obj_feat=batch["obj_feat"],
                              union_feat=batch["union_feat"])
    valid = torch.tensor(bundle.valid_idxs)
    sub = logits.index_select(1, valid)
    for row, p in enumerate(preds):
        want = bundle.pred_vocab.token(bundle.valid_idxs[int(sub[row].argmax())])
        assert p["predicate"] == want
        assert p["predicate"] not in (Vocab.PAD, Vocab.UNK)
        assert p["confidence"] == pytest.approx(float(torch.softmax(sub[row], -1).max()), abs=1e-6)


# ---------------------------------------------------------------------------
# selection policy
# ---------------------------------------------------------------------------

def _preds(*rows):
    return [{"subject_index": s, "object_index": o, "predicate": p, "confidence": c,
             "top_k": [(p, c)]} for s, o, p, c in rows]


def test_select_relation_picks_most_confident_survivor():
    dets = [{"label": "person", "box": [0, 0, 10, 10]}, {"label": "bicycle", "box": [0, 0, 10, 10]},
            {"label": "chair", "box": [0, 0, 10, 10]}]
    preds = _preds((1, 0, "holding", 0.95),     # bicycle holding person: nonsense
                   (0, 1, "riding", 0.80),
                   (0, 2, "sitting on", 0.60))
    selected, annotated = select_relation(dets, preds)
    assert (selected["subject"], selected["predicate"], selected["object"]) == ("person", "riding", "bicycle")
    assert annotated[0]["status"] == "rejected_semantic_filter"
    assert selected["predicate"] == "riding"        # never swapped for another predicate


def test_all_rejected_gives_no_relation():
    dets = [{"label": "chair", "box": [0, 0, 1, 1]}, {"label": "cup", "box": [0, 0, 1, 1]}]
    selected, _ = select_relation(dets, _preds((0, 1, "holding", 0.9)))
    assert selected is None


@pytest.mark.parametrize("selected,n_elig,n_pred,expect", [
    (None, 1, 0, (False, "fewer_than_2_eligible_detections")),
    (None, 2, 2, (False, "all_pairs_rejected_by_semantic_filter")),
    ({"confidence": 0.49}, 2, 2, (False, "below_confidence_threshold")),
    ({"confidence": 0.50}, 2, 2, (True, None)),
    ({"confidence": float("nan")}, 2, 2, (False, "invalid_confidence")),
])
def test_relation_decision(selected, n_elig, n_pred, expect):
    d = relation_decision(selected, n_elig, n_pred, 0.5)
    assert (d["use_relation"], d["fallback_reason"]) == expect


# ---------------------------------------------------------------------------
# legacy demo pipeline
# ---------------------------------------------------------------------------

def test_legacy_pipeline_reports_the_real_input_width(tmp_path, monkeypatch):
    g = pytest.importorskip("grounded_caption_pipeline")
    write_checkpoint(tmp_path / "ck")
    monkeypatch.setattr(g, "CHECKPOINT_DIR", str(tmp_path / "ck"))
    info = g.verify_relation_model()
    assert info["input_dim"] == 2451 and info["union_dim"] == 768 and info["geo_dim"] == 19

    monkeypatch.setattr(g, "CHECKPOINT_DIR", os.path.join(ROOT, "checkpoints"))
    with pytest.raises(AssertionError, match="geometry-only"):
        g.verify_relation_model()

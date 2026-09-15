"""The model's input width must come from the feature blocks a batch really carries.

Regression cover for the union arm of the visual ablation, which failed on the
GPU with

    RuntimeError: mat1 and mat2 shapes cannot be multiplied (512x1683 and 2451x256)

2451 = 64 + 64 + 19 + 768 + 768 + 768 is what a geometry + CLIP + union model
allocates. 1683 is the same minus the 768-wide union block. The 512 rows are
the validation loader (2 x batch size 256). After the final epoch,
train_full_visual_semantic.py re-scored the best model with
compute_predicate_metrics(..., has_visual) and no use_union. So union_feat came
back None, RelationMLP.forward silently skipped the block, and the first Linear
received 1683 columns.

These tests pin, for all four arms: the exact widths; that a missing block is a
named FeatureDimensionError, not a matmul error; the pre-training preflight;
the checkpoint round trip; and an end-to-end CPU run of the exact train + eval
commands the GPU runner issues, on a tiny synthetic Visual Genome.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train_full_visual_semantic as tfvs  # noqa: E402
from relation_prediction.clip_cache import ClipCache  # noqa: E402
from relation_prediction.clip_extractor import CLIPExtractor  # noqa: E402
from relation_prediction.model import FeatureDimensionError, RelationMLP  # noqa: E402
from relation_prediction.vg_dataset import (  # noqa: E402
    CLIP_DIM, GEO_DIM_EXT, UNION_FEATURE_DIM, Vocab,
)
from run_visual_experiment import VARIANTS, build_commands  # noqa: E402

VARIANT_NAMES = ["geometry", "geometry_clip", "geometry_clip_union", "clip_only"]
EMBED = 64
N_LABELS, N_PREDS = 80, 21

# Written out by hand, not derived, so a drift in the derivation cannot also
# drift the expectation.
EXPECTED_BLOCKS = {
    "geometry": [("subj_label", 64), ("obj_label", 64), ("geometry", 19)],
    "geometry_clip": [("subj_label", 64), ("obj_label", 64), ("geometry", 19),
                      ("subj_clip", 768), ("obj_clip", 768)],
    "geometry_clip_union": [("subj_label", 64), ("obj_label", 64), ("geometry", 19),
                            ("subj_clip", 768), ("obj_clip", 768),
                            ("union_clip", 768)],
    "clip_only": [("subj_label", 64), ("obj_label", 64), ("geometry", 0),
                  ("subj_clip", 768), ("obj_clip", 768)],
}
EXPECTED_INPUT_DIM = {"geometry": 147, "geometry_clip": 1683,
                      "geometry_clip_union": 2451, "clip_only": 1664}


def dims_for(variant):
    v = VARIANTS[variant]
    # Every arm runs with --use-visual (COMMON_TRAIN_ARGS); only the geometry
    # control hides CLIP from the model via --visual-filter-only.
    return tfvs.resolve_feature_dims(
        use_visual=True, visual_filter_only=v["visual_filter_only"],
        use_union=v["union"], use_pose=False, geo_mode=v["geo_mode"])


def flags_for(variant):
    return dict(has_visual=True, use_union=VARIANTS[variant]["union"], use_pose=False)


def build_model(variant):
    return RelationMLP(num_labels=N_LABELS, num_predicates=N_PREDS, embed_dim=EMBED,
                       hidden_dims=(256, 128), geo_norm=VARIANTS[variant]["geo_norm"],
                       **dims_for(variant))


class FakeVisualVG(Dataset):
    """Items laid out exactly like VGRelationshipDataset.__getitem__ with
    use_visual=True: (subj, obj, geo, pred, subj_clip, obj_clip[, union])."""

    def __init__(self, n, geo_dim, with_union, union_width=UNION_FEATURE_DIM):
        g = torch.Generator().manual_seed(0)
        self.items = []
        for i in range(n):
            item = (torch.tensor(1 + i % 5), torch.tensor(2 + i % 7),
                    torch.randn(geo_dim, generator=g), torch.tensor(2 + i % 19),
                    torch.randn(CLIP_DIM, generator=g), torch.randn(CLIP_DIM, generator=g))
            if with_union:
                item = item + (torch.randn(union_width, generator=g),)
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def fake_for(variant, n=40):
    return FakeVisualVG(n, dims_for(variant)["geo_dim"], VARIANTS[variant]["union"])


# --------------------------------------------------------------------------
# exact widths
# --------------------------------------------------------------------------

def test_the_reported_numbers_decompose_exactly():
    assert 2 * EMBED + GEO_DIM_EXT + 2 * CLIP_DIM + UNION_FEATURE_DIM == 2451
    assert 2 * EMBED + GEO_DIM_EXT + 2 * CLIP_DIM == 1683
    assert 2451 - 1683 == UNION_FEATURE_DIM == 768


@pytest.mark.parametrize("variant", VARIANT_NAMES)
def test_exact_blocks_and_input_dim_per_variant(variant):
    model = build_model(variant)
    assert model.feature_blocks() == EXPECTED_BLOCKS[variant]
    assert model.input_dim == EXPECTED_INPUT_DIM[variant]
    assert model.mlp[0].in_features == EXPECTED_INPUT_DIM[variant]
    assert sum(w for _, w in model.feature_blocks()) == model.input_dim


@pytest.mark.parametrize("variant", VARIANT_NAMES)
def test_resolved_dims_match_the_variant_table(variant):
    v, d = VARIANTS[variant], dims_for(variant)
    assert d["geo_dim"] == (0 if v["geo_mode"] == "none" else GEO_DIM_EXT)
    assert d["clip_dim"] == (0 if v["visual_filter_only"] else CLIP_DIM)
    assert d["union_dim"] == (UNION_FEATURE_DIM if v["union"] else 0)
    assert d["pose_dim"] == 0


@pytest.mark.parametrize("variant", VARIANT_NAMES)
def test_actual_batch_width_equals_model_input_dim(variant):
    """Through the real _collate and _unpack_batch, for every arm."""
    model = build_model(variant)
    ds = fake_for(variant)
    batch = tfvs._collate([ds[i] for i in range(16)])
    subj, obj, geo, _, sf, of, uf, pf = tfvs._unpack_batch(
        batch, torch.device("cpu"), **flags_for(variant))
    actual = model.check_inputs(geo, sf, of, uf, pf)
    assert sum(w for _, w in actual) == model.input_dim == EXPECTED_INPUT_DIM[variant]
    model.eval()
    assert model(subj, obj, geo, subj_feat=sf, obj_feat=of,
                 union_feat=uf, pose_feat=pf).shape == (16, N_PREDS)


# --------------------------------------------------------------------------
# a missing block is a named error, never a matmul error
# --------------------------------------------------------------------------

def _inputs(model, b=4, union=True):
    geo = torch.randn(b, model.geo_dim)
    kw = dict(subj_feat=torch.randn(b, CLIP_DIM), obj_feat=torch.randn(b, CLIP_DIM))
    if union:
        kw["union_feat"] = torch.randn(b, UNION_FEATURE_DIM)
    return torch.zeros(b, dtype=torch.long), torch.ones(b, dtype=torch.long), geo, kw


def test_union_model_without_union_feat_is_a_clear_error():
    """The exact GPU failure, now caught before concatenation."""
    model = build_model("geometry_clip_union").eval()
    s, o, geo, kw = _inputs(model, b=512, union=False)
    with pytest.raises(FeatureDimensionError) as exc:
        model(s, o, geo, **kw)
    msg = str(exc.value)
    assert "union_clip expects 768 columns but was not passed" in msg
    assert "input_dim=2451" in msg and "supplies 1683" in msg
    assert not isinstance(exc.value, RuntimeError)


@pytest.mark.parametrize("block,kwarg", [("subj_clip", "subj_feat"),
                                         ("obj_clip", "obj_feat"),
                                         ("union_clip", "union_feat")])
def test_every_enabled_block_is_required(block, kwarg):
    model = build_model("geometry_clip_union").eval()
    s, o, geo, kw = _inputs(model)
    kw[kwarg] = None
    with pytest.raises(FeatureDimensionError, match=block):
        model(s, o, geo, **kw)


def test_wrong_width_is_named():
    model = build_model("geometry_clip_union").eval()
    s, o, geo, kw = _inputs(model)
    kw["union_feat"] = torch.randn(4, 512)
    with pytest.raises(FeatureDimensionError, match="union_clip expects 768 columns, got 512"):
        model(s, o, geo, **kw)


def test_wrong_geometry_width_is_named():
    model = build_model("geometry").eval()
    with pytest.raises(FeatureDimensionError, match="geometry expects 19 columns, got 5"):
        model(torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long),
              torch.randn(4, 5))


def test_the_check_changes_no_numbers():
    """Same weights, same inputs: forward == explicit concatenation through mlp."""
    model = build_model("geometry_clip_union").eval()
    s, o, geo, kw = _inputs(model)
    with torch.no_grad():
        manual = model.mlp(torch.cat([model.label_emb(s), model.label_emb(o),
                                      model.geo_norm(geo), kw["subj_feat"],
                                      kw["obj_feat"], kw["union_feat"]], dim=-1))
        assert torch.equal(model(s, o, geo, **kw), manual)


def test_model_adds_nothing_to_the_state_dict():
    """input_dim is a plain attribute; old checkpoints must still load strictly."""
    keys = set(build_model("geometry_clip_union").state_dict())
    assert not any("input_dim" in k or "block" in k for k in keys)


# --------------------------------------------------------------------------
# batch unpacking and the post-training call that crashed
# --------------------------------------------------------------------------

def test_compute_predicate_metrics_cannot_be_called_without_feature_flags():
    model = build_model("geometry_clip_union")
    loader = DataLoader(fake_for("geometry_clip_union"), batch_size=8,
                        collate_fn=tfvs._collate)
    pv = _vocab(N_PREDS)
    with pytest.raises(TypeError):
        # the pre-fix STEP 5 call signature
        tfvs.compute_predicate_metrics(model, loader, torch.device("cpu"), pv, True)


@pytest.mark.parametrize("variant", VARIANT_NAMES)
def test_best_model_analysis_path_runs_for_every_variant(variant):
    """STEP 5 re-scores the best model on val batches of 2 x batch size."""
    model = build_model(variant)
    loader = DataLoader(fake_for(variant, n=40), batch_size=32, shuffle=False,
                        collate_fn=tfvs._collate)
    metrics, _, preds, targets = tfvs.compute_predicate_metrics(
        model, loader, torch.device("cpu"), _vocab(N_PREDS), **flags_for(variant))
    assert len(preds) == len(targets) == 40


def test_unpack_rejects_a_batch_with_an_unexpected_block():
    ds = fake_for("geometry_clip_union")
    batch = tfvs._collate([ds[i] for i in range(4)])          # carries union
    with pytest.raises(FeatureDimensionError, match="carries 7 tensors"):
        tfvs._unpack_batch(batch, torch.device("cpu"),
                           has_visual=True, use_union=False, use_pose=False)


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANT_NAMES)
def test_preflight_passes_for_every_variant(variant, capsys):
    model = build_model(variant)
    model.train()
    ds = fake_for(variant)
    tfvs.preflight_feature_dims(model, {"train": ds, "val": ds}, torch.device("cpu"),
                                **flags_for(variant))
    out = capsys.readouterr().out
    assert f"= {EXPECTED_INPUT_DIM[variant]}  OK" in out
    assert "PREFLIGHT PASSED" in out
    assert model.training, "preflight must restore train mode"


def test_preflight_catches_a_union_model_fed_no_union_before_training():
    model = build_model("geometry_clip_union")
    ds = fake_for("geometry_clip")                            # no union block
    with pytest.raises(FeatureDimensionError, match="union_clip"):
        tfvs.preflight_feature_dims(model, {"train": ds, "val": ds}, torch.device("cpu"),
                                    has_visual=True, use_union=False, use_pose=False)


def test_preflight_catches_a_mis_sized_union_cache():
    model = build_model("geometry_clip_union")
    ds = FakeVisualVG(16, GEO_DIM_EXT, with_union=True, union_width=512)
    with pytest.raises(FeatureDimensionError, match="got 512"):
        tfvs.preflight_feature_dims(model, {"train": ds, "val": ds}, torch.device("cpu"),
                                    **flags_for("geometry_clip_union"))


def test_preflight_consumes_no_rng_and_leaves_weights_untouched():
    """Seeds are part of the protocol: the check must not shift any draw."""
    model = build_model("geometry_clip_union")
    ds = fake_for("geometry_clip_union")
    torch.manual_seed(42)
    random.seed(42)
    before_torch, before_py = torch.get_rng_state(), random.getstate()
    before_state = {k: v.clone() for k, v in model.state_dict().items()}
    tfvs.preflight_feature_dims(model, {"train": ds, "val": ds}, torch.device("cpu"),
                                **flags_for("geometry_clip_union"))
    assert torch.equal(torch.get_rng_state(), before_torch)
    assert random.getstate() == before_py
    for k, v in model.state_dict().items():
        assert torch.equal(v, before_state[k]), f"{k} changed (BatchNorm stats?)"


# --------------------------------------------------------------------------
# checkpoint save -> load, for all four arms
# --------------------------------------------------------------------------

def _vocab(n):
    v = Vocab()
    i = 0
    while len(v) < n:
        v.add(f"tok{i}")
        i += 1
    return v


@pytest.mark.parametrize("variant", VARIANT_NAMES)
def test_checkpoint_round_trip_per_variant(variant, tmp_path, monkeypatch):
    from eval_gt_relations import load_checkpoint

    v = VARIANTS[variant]
    for name, value in dict(GEO_MODE=v["geo_mode"], VISUAL_FILTER_ONLY=v["visual_filter_only"],
                            USE_VISUAL=True, REQUIRE_VISUAL=True, USE_UNION=v["union"],
                            USE_POSE=False, GEO_NORM=v["geo_norm"],
                            PREDICATE_SCHEME="v1").items():
        monkeypatch.setattr(tfvs, name, value)

    model = build_model(variant)
    with torch.no_grad():                   # non-trivial BatchNorm statistics
        if model.geo_norm is not None:
            model.geo_norm.running_mean.uniform_(-1, 1)
            model.geo_norm.running_var.uniform_(0.5, 2)
    model.eval()
    tfvs._save_checkpoint(model, _vocab(N_LABELS), _vocab(N_PREDS), str(tmp_path),
                          epoch=1, val_acc=0.5)

    raw = torch.load(tmp_path / "relation_mlp.pt", weights_only=True)
    cfg = raw["model_config"]
    assert cfg["input_dim"] == EXPECTED_INPUT_DIM[variant]
    assert [tuple(b) for b in cfg["feature_blocks"]] == EXPECTED_BLOCKS[variant]
    assert cfg["union_dim"] == (768 if v["union"] else 0)
    assert cfg["visual_filter_only"] is v["visual_filter_only"]
    meta = json.loads((tmp_path / "training_meta.json").read_text())
    assert meta["input_dim"] == EXPECTED_INPUT_DIM[variant]
    assert meta["clip_dim"] == model.clip_dim

    loaded, _, _, info = load_checkpoint(str(tmp_path), torch.device("cpu"))
    icfg = info["inferred_config"]
    assert loaded.input_dim == icfg["input_dim"] == EXPECTED_INPUT_DIM[variant]
    assert loaded.feature_blocks() == EXPECTED_BLOCKS[variant]

    s, o, geo, kw = _inputs(model, b=6, union=v["union"])
    with torch.no_grad():
        assert torch.equal(model(s, o, geo, **kw), loaded(s, o, geo, **kw))


def test_evaluator_refuses_a_config_whose_input_dim_disagrees(tmp_path, monkeypatch):
    from eval_gt_relations import load_checkpoint

    monkeypatch.setattr(tfvs, "GEO_MODE", "ext")
    model = build_model("geometry_clip_union")
    tfvs._save_checkpoint(model, _vocab(N_LABELS), _vocab(N_PREDS), str(tmp_path),
                          epoch=1, val_acc=0.5)
    path = tmp_path / "relation_mlp.pt"
    raw = torch.load(path, weights_only=True)
    raw["model_config"]["input_dim"] = 1683
    torch.save(raw, path)
    with pytest.raises(SystemExit, match="input_dim=1683"):
        load_checkpoint(str(tmp_path), torch.device("cpu"))


# --------------------------------------------------------------------------
# end to end: the runner's exact commands, on a tiny synthetic VG, on CPU
# --------------------------------------------------------------------------

# (name, predicate) templates; every name normalises to a COCO label and every
# predicate is on the v1 allowlist.
_PAIRS = [("man", "riding", "horse"), ("person", "holding", "umbrella"),
          ("dog", "on", "couch"), ("woman", "sitting on", "chair"),
          ("car", "near", "bicycle"), ("person", "wearing", "backpack")]


def _write_tiny_vg(root):
    """6 images x 6 relations, a split manifest and a 100%-coverage CLIP cache."""
    os.makedirs(root / "images", exist_ok=True)
    g = torch.Generator().manual_seed(7)
    rels, meta, keys, embs = [], [], [], []

    def emb():
        e = torch.randn(CLIP_DIM, generator=g)
        return e / e.norm()

    oid = 1000
    for iid in range(1, 7):
        meta.append({"image_id": iid, "width": 640, "height": 480})
        records = []
        for j, (sname, pred, oname) in enumerate(_PAIRS):
            s_id, o_id = oid, oid + 1
            oid += 2
            subj = {"object_id": s_id, "name": sname, "x": 20 + 10 * j, "y": 30, "w": 120, "h": 200}
            obj = {"object_id": o_id, "name": oname, "x": 60 + 15 * j, "y": 90, "w": 150, "h": 160}
            records.append({"predicate": pred, "subject": subj, "object": obj})
            for k in (f"{iid}_obj_{s_id}", f"{iid}_obj_{o_id}",
                      CLIPExtractor.to_union_key(iid, s_id, o_id)):
                keys.append(k)
                embs.append(emb())
        rels.append({"image_id": iid, "relationships": records})

    (root / "relationships.json").write_text(json.dumps(rels), encoding="utf-8")
    (root / "image_data.json").write_text(json.dumps(meta), encoding="utf-8")
    ClipCache.build(keys, embs).save(str(root / "clip_cache_proper.pt"))
    manifest = {"seed": 42, "split_unit": "image_id",
                "fractions": {"train": 0.5, "val": 0.17, "test": 0.33},
                "train_ids": [1, 2, 3], "val_ids": [4], "test_ids": [5, 6]}
    split = root / "split.json"
    split.write_text(json.dumps(manifest), encoding="utf-8")
    return split


def test_runner_commands_train_and_evaluate_every_variant_end_to_end(tmp_path):
    """The seed-42 GPU runs, shrunk to CPU: 1 epoch, batch 4, synthetic data.

    Runs under a strict cp1252 stdout, i.e. the Windows log redirection that
    broke evaluation. Training has to get through STEP 5 and the final report,
    the path that crashed for the union arm.
    """
    vg = tmp_path / "vg"
    split = _write_tiny_vg(vg)
    env = {**os.environ, "PYTHONIOENCODING": "cp1252:strict", "PYTHONUTF8": "0",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    jobs = {}
    for variant in VARIANT_NAMES:
        train, evaluate = build_commands(
            variant, 42, vg, split, vg / "clip_cache_proper.pt",
            tmp_path / "ckpt" / variant, tmp_path / "res" / variant, 1, 4)
        jobs[variant] = (train, evaluate)

    def run_all(which):
        procs = {v: subprocess.Popen(cmds[which], cwd=ROOT, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                 for v, cmds in jobs.items()}
        outs = {}
        for v, p in procs.items():
            out, _ = p.communicate(timeout=600)
            outs[v] = (p.returncode, out.decode("cp1252", "replace"))
        return outs

    train_out = run_all(0)
    for variant, (rc, log) in train_out.items():
        assert rc == 0, f"{variant} training failed:\n{log[-4000:]}"
        assert f"= {EXPECTED_INPUT_DIM[variant]}  OK" in log, f"{variant}: no preflight line"
        assert "STEP 5" in log and "TRAINING COMPLETE" in log, f"{variant} stopped early"
        cfg = torch.load(tmp_path / "ckpt" / variant / "relation_mlp.pt",
                         weights_only=True)["model_config"]
        assert cfg["input_dim"] == EXPECTED_INPUT_DIM[variant]

    eval_out = run_all(1)
    n_test = set()
    for variant, (rc, log) in eval_out.items():
        assert rc == 0, f"{variant} evaluation failed:\n{log[-4000:]}"
        assert "overlap check: PASS" in log
        assert f"model {EXPECTED_INPUT_DIM[variant]}" in log
        res = json.loads((tmp_path / "res" / variant / f"{variant}_seed42.json")
                         .read_text(encoding="utf-8"))
        assert 0.0 <= res["metrics"]["top1"] <= 1.0
        assert res["checkpoint"]["inferred_config"]["input_dim"] == EXPECTED_INPUT_DIM[variant]
        n_test.add(res["split"]["n_samples_test"])
    # every arm scored on the same test population (the ablation's premise)
    assert n_test == {12}

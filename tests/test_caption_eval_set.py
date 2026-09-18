"""The caption evaluation set is fixed, test-only, human-annotated and model-free."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import build_caption_eval_set as bces  # noqa: E402


def _entity(name, oid):
    return {"object_id": oid, "names": [name], "x": 0, "y": 0, "w": 20, "h": 20}


@pytest.fixture()
def vg(tmp_path):
    """A miniature Visual Genome: 10 test, 4 train, 2 val images."""
    root = tmp_path / "vg"
    (root / "images").mkdir(parents=True)
    test_ids = list(range(100, 110))
    train_ids, val_ids = [1, 2, 3, 4], [50, 51]
    objects, rels, image_data = [], [], []
    for iid in test_ids + train_ids + val_ids:
        names = ["man", "tables", "shirt", "tree"] if iid % 2 == 0 else ["dog", "frisbee"]
        if iid == 109:
            names = ["cat"]                                   # only 1 COCO object
        objects.append({"image_id": iid, "objects": [_entity(n, k) for k, n in enumerate(names)]})
        rels.append({"image_id": iid, "relationships": [
            {"predicate": "near", "subject": _entity("person", 90), "object": _entity("bike", 91)}]
            if iid == 101 else []})
        image_data.append({"image_id": iid, "width": 40, "height": 30})
        if iid not in (103, 105):                             # two test images missing
            Image.new("RGB", (40, 30), (iid % 255, 0, 0)).save(root / "images" / f"{iid}.jpg")
    (root / "images" / "107.jpg").write_bytes(b"not a jpeg")  # corrupt
    (root / "objects.json").write_text(json.dumps(objects))
    (root / "relationships.json").write_text(json.dumps(rels))
    (root / "image_data.json").write_text(json.dumps(image_data))
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}))
    return root, manifest


def _args(vg, **kw):
    root, manifest = vg
    base = dict(vg_root=str(root), split_manifest=str(manifest), split="test",
                vg_image_dir=None, limit=6, min_objects=2, min_final=1, seed=42,
                output="unused", force=True, allow_relationships_only=False, skip_hash=False,
                expected_split_ids_sha256=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_only_test_split_images_with_enough_human_objects(vg):
    m = bces.build_manifest(_args(vg, limit=100))
    requested = set(map(int, m["images"]))
    assert requested == set(range(100, 109))                  # 109 has one object; train/val never
    assert m["meta"]["split"] == "test"
    assert m["meta"]["ground_truth_uses_model_predictions"] is False


def test_ground_truth_comes_from_objects_json_mapped_to_coco(vg):
    m = bces.build_manifest(_args(vg, limit=100))
    assert m["images"]["100"]["objects"] == ["dining table", "person"]    # man, tables
    assert m["images"]["101"]["objects"] == ["bicycle", "dog", "frisbee", "person"]
    assert m["images"]["101"]["gt_relationships_json"] == ["bicycle", "person"]
    assert m["meta"]["ground_truth_source"] == "objects.json+relationships.json"


def test_deterministic_and_seed_dependent(vg):
    a = bces.build_manifest(_args(vg))
    b = bces.build_manifest(_args(vg))
    assert list(a["images"]) == list(b["images"])
    assert a["usable_ids"] == b["usable_ids"]
    assert a["meta"]["usable_ids_sha256"] == b["meta"]["usable_ids_sha256"]
    orders = {tuple(bces.build_manifest(_args(vg, seed=s))["images"]) for s in range(6)}
    assert len(orders) > 1


def test_missing_and_corrupt_images_are_reported_not_backfilled(vg):
    m = bces.build_manifest(_args(vg, limit=100))
    meta = m["meta"]
    assert meta["requested"] == 9
    assert set(m["missing_ids"]) == {"103", "105"}
    assert m["corrupt_ids"] == ["107"]
    assert meta["usable"] == meta["final"] == 6
    assert set(m["usable_ids"]).isdisjoint({"103", "105", "107"})
    limited = bces.build_manifest(_args(vg, limit=3))
    assert limited["meta"]["requested"] == 3                  # availability never changes the request


def test_too_small_set_stops(vg, tmp_path, monkeypatch, capsys):
    out = tmp_path / "es.json"
    args = _args(vg, min_final=100, output=str(out))
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: args)
    assert bces.main() == 2
    assert json.loads(out.read_text())["meta"]["status"] == "TOO_SMALL"
    assert "STOP" in capsys.readouterr().out


def test_objects_json_is_required_by_default(vg):
    root, _ = vg
    (root / "objects.json").unlink()
    with pytest.raises(SystemExit):
        bces.build_manifest(_args(vg))
    m = bces.build_manifest(_args(vg, allow_relationships_only=True, min_objects=1))
    assert "INCOMPLETE" in m["meta"]["ground_truth_source"]


def test_frozen_split_guard(vg):
    with pytest.raises(SystemExit):
        bces.build_manifest(_args(vg, expected_split_ids_sha256=bces.FROZEN_SPLIT_IDS_SHA256))
    real = json.load(open(os.path.join(ROOT, "splits", "e0_image_split.json")))
    assert bces.split_ids_sha256(real) == bces.FROZEN_SPLIT_IDS_SHA256


def test_builder_never_imports_a_detector_or_captioner(vg):
    root, manifest = vg
    code = (
        "import sys, argparse, build_caption_eval_set as b;"
        f"b.build_manifest(argparse.Namespace(vg_root={str(root)!r}, split_manifest={str(manifest)!r},"
        "split='test', vg_image_dir=None, limit=5, min_objects=2, min_final=1, seed=42,"
        "allow_relationships_only=False, skip_hash=True, expected_split_ids_sha256=None));"
        "bad=[m for m in ('ultralytics','utils.yolo_detector','utils.detection_verifier',"
        "'utils.blip_captioner','utils.caption_relations') if m in sys.modules];"
        "print('IMPORTED', bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "IMPORTED []" in out.stdout


# ---------------------------------------------------------------------------
# The tuning set: same builder, validation split, disjoint from the test set
# ---------------------------------------------------------------------------

def test_validation_split_draws_only_validation_images(vg):
    m = bces.build_manifest(_args(vg, split="val", limit=100, min_objects=1))
    assert set(map(int, m["images"])) <= {50, 51}
    assert m["meta"]["split"] == "val"


def test_tuning_and_test_caption_sets_never_share_an_image(vg):
    """Reranking weights are chosen on the validation set and the frozen test
    set is scored once. If the two sets overlapped, that separation would be
    a formality."""
    test_set = bces.build_manifest(_args(vg, split="test", limit=100))
    val_set = bces.build_manifest(_args(vg, split="val", limit=100, min_objects=1))
    assert set(test_set["images"]).isdisjoint(set(val_set["images"]))


def test_the_real_frozen_splits_are_disjoint_so_tuning_cannot_leak():
    manifest = json.load(open(os.path.join(ROOT, "splits", "e0_image_split.json")))
    val = {int(i) for i in manifest["val_ids"]}
    test = {int(i) for i in manifest["test_ids"]}
    assert val and test and val.isdisjoint(test)


def test_shipped_caption_sets_are_disjoint_if_both_are_built():
    """Guards the real artefacts once they exist on a machine."""
    paths = [os.path.join(ROOT, "splits", "caption_eval_test_250.json"),
             os.path.join(ROOT, "splits", "caption_eval_val_200.json")]
    if not all(os.path.isfile(p) for p in paths):
        pytest.skip("caption evaluation sets are not built on this machine")
    test_set, val_set = (json.load(open(p, encoding="utf-8")) for p in paths)
    assert test_set["meta"]["split"] == "test" and val_set["meta"]["split"] == "val"
    assert set(map(str, test_set["usable_ids"])).isdisjoint(map(str, val_set["usable_ids"]))

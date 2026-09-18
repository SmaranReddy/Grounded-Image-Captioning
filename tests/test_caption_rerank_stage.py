"""The rerank stage, the weight lock and the extended scorer, on a synthetic run.

A run directory is built by hand (no models, no images) with the three prefix
arms the frozen experiment produced plus a candidate pool, so the stage, its
refusals and the five-arm evaluation can be exercised end to end in
milliseconds.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_caption_experiment as rce  # noqa: E402
from utils.blip_captioner import BASELINE_PREFIX  # noqa: E402
from utils.caption_experiment_eval import evaluate_run  # noqa: E402
from utils.caption_rerank import RerankWeights  # noqa: E402

RELATIONS = {
    "1": {"subject": "person", "predicate": "riding", "object": "bicycle",
          "confidence": 0.8, "subject_index": 0, "object_index": 1},
    "2": {"subject": "cat", "predicate": "sitting on", "object": "bed",
          "confidence": 0.3, "subject_index": 0, "object_index": 1},
    "3": None,
}
DETECTIONS = {"1": ["person", "bicycle"], "2": ["cat", "bed"], "3": []}
GT = {"1": ["bicycle", "person"], "2": ["bed", "cat"], "3": ["dog"]}
IDS = ["1", "2", "3"]

CANDIDATES = {
    "1": [("baseline", 0, "a photo of a person on a bicycle", -1.0),
          ("baseline", 1, "a photo of a person with a dog", -1.3),
          ("objects_only", 0, "a photo of a person and a bicycle", -1.2),
          ("relation", 0, "a photo of a person riding a bicycle", -1.1)],
    "2": [("baseline", 0, "a photo of a cat on a bed", -0.9),
          ("baseline", 1, "a photo of a cat and a laptop", -1.4),
          ("objects_only", 0, "a photo of a cat and a bed", -1.5),
          ("relation", 0, "a photo of a cat sitting on a bed", -1.6)],
    "3": [("baseline", 0, "a photo of a dog in a park", -1.0),
          ("baseline", 1, "a photo of a dog and a frisbee", -1.2)],
}


def _prefixes(iid):
    rel = RELATIONS[iid]
    out = {"baseline": BASELINE_PREFIX}
    if rel:
        out["objects_only"] = f"a photo of a {rel['subject']} and a {rel['object']}"
        out["relation"] = (f"a photo of a {rel['subject']} {rel['predicate']} "
                           f"a {rel['object']}")
    return out


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def build_run(tmp_path, threshold=0.5):
    """A complete three-arm run directory plus its candidate pool."""
    eval_set = {
        "meta": {"status": "OK", "split": "test", "final": len(IDS),
                 "usable_ids_sha256": hashlib.sha256(",".join(IDS).encode()).hexdigest()},
        "usable_ids": IDS,
        "images": {i: {"objects": GT[i], "file": f"{i}.jpg"} for i in IDS},
    }
    eval_path = tmp_path / "eval_set.json"
    eval_path.write_text(json.dumps(eval_set), encoding="utf-8")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(json.dumps({
        "image_ids": IDS,
        "relation_threshold": threshold,
        "relation_threshold_is_default": threshold == 0.5,
        "eval_set": {"path": str(eval_path).replace("\\", "/"),
                     "usable_ids_sha256": eval_set["meta"]["usable_ids_sha256"],
                     "split": "test", "status": "OK", "final": len(IDS)},
        "relation_checkpoint": {"checkpoint_dir": "checkpoints_gpu/fake", "weights_sha256": "0" * 64},
    }), encoding="utf-8")

    _write_jsonl(run_dir / "relations.jsonl", [{
        "image_id": i,
        "file": f"{i}.jpg",
        "raw_detections": [{"label": d} for d in DETECTIONS[i]],
        "verified_detections": [{"label": d, "box": [0, 0, 10, 10], "score": 0.9}
                                for d in DETECTIONS[i]],
        "eligible_detections": [{"label": d, "box": [0, 0, 10, 10], "score": 0.9}
                                for d in DETECTIONS[i]],
        "pair_predictions": [],
        "selected_relation": RELATIONS[i],
        "decision": ({"use_relation": RELATIONS[i]["confidence"] >= threshold,
                      "fallback_reason": None if RELATIONS[i]["confidence"] >= threshold
                      else "below_confidence_threshold"} if RELATIONS[i]
                     else {"use_relation": False,
                           "fallback_reason": "fewer_than_2_eligible_detections"}),
    } for i in IDS])

    _write_jsonl(run_dir / "candidates.jsonl", [{
        "image_id": i,
        "prefixes": _prefixes(i),
        "relation": RELATIONS[i],
        "candidates": [{"source": s, "prefix": _prefixes(i)[s], "beam_rank": r,
                        "text": t, "lm_score": lm, "beam_sequence_score": lm}
                       for s, r, t, lm in CANDIDATES[i]],
    } for i in IDS])

    base = {i: CANDIDATES[i][0][2] for i in IDS}
    for arm in ("baseline", "grounded", "objects_only"):
        rows = []
        for i in IDS:
            rel = RELATIONS[i]
            use = arm != "baseline" and bool(rel) and rel["confidence"] >= threshold
            if not use:
                prefix, caption = BASELINE_PREFIX, base[i]
            else:
                source = "relation" if arm == "grounded" else "objects_only"
                prefix = _prefixes(i)[source]
                caption = next(t for s, _, t, _ in CANDIDATES[i] if s == source)
            rows.append({"image_id": i, "arm": arm, "caption": caption, "prefix": prefix,
                         "relation_used": use, "prefix_echoed": True,
                         "input_ids": [1, 2, 3] + ([9] if use else []),
                         "relation": rel})
        _write_jsonl(run_dir / f"captions_{arm}.jsonl", rows)
    return run_dir, eval_path


def write_weights(tmp_path, obj=(0.5, 0.5), rel=1.0, name="rerank_weights.json"):
    path = tmp_path / name
    path.write_text(json.dumps({
        "objective": "test", "selected_on": {"split": "val", "n_images": 3},
        "weights": {"object_reranked": {"w_obj": obj[0], "w_hall": obj[1], "w_rel": 0.0},
                    "relation_reranked": {"w_obj": obj[0], "w_hall": obj[1], "w_rel": rel}},
    }), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------

def test_rerank_writes_one_caption_per_image_from_the_pool(tmp_path):
    run_dir, _ = build_run(tmp_path)
    weights = write_weights(tmp_path)
    rows = rce.run_rerank(run_dir, "relation_reranked", json.loads(weights.read_text()),
                          str(weights))
    assert [r["image_id"] for r in rows] == IDS
    for r in rows:
        pool = {t for _, _, t, _ in CANDIDATES[r["image_id"]]}
        assert r["caption"] in pool
        assert r["source"] in _prefixes(r["image_id"])


def test_object_arm_refuses_a_non_zero_relation_weight(tmp_path):
    run_dir, _ = build_run(tmp_path)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"weights": {
        "object_reranked": {"w_obj": 0.5, "w_hall": 0.5, "w_rel": 0.7}}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="w_rel"):
        rce.run_rerank(run_dir, "object_reranked", json.loads(path.read_text()), str(path))


def test_rerank_refuses_to_overwrite_a_run_with_different_weights(tmp_path):
    run_dir, _ = build_run(tmp_path)
    first = write_weights(tmp_path, rel=1.0)
    rce.run_rerank(run_dir, "relation_reranked", json.loads(first.read_text()), str(first))
    second = write_weights(tmp_path, rel=2.0, name="other.json")
    with pytest.raises(SystemExit, match="DIFFERENT weights"):
        rce.run_rerank(run_dir, "relation_reranked", json.loads(second.read_text()),
                       str(second))


def test_rerank_records_the_weight_provenance(tmp_path):
    run_dir, _ = build_run(tmp_path)
    weights = write_weights(tmp_path)
    rce.run_rerank(run_dir, "object_reranked", json.loads(weights.read_text()), str(weights))
    meta = json.loads((run_dir / "captions_object_reranked.meta.json").read_text())
    assert meta["weights"] == {"w_obj": 0.5, "w_hall": 0.5, "w_rel": 0.0}
    assert meta["weights_sha256"] == RerankWeights(0.5, 0.5, 0.0).sha256()
    assert meta["weights_selected_on"]["split"] == "val"
    assert len(meta["weights_file_sha256"]) == 64


def test_missing_weights_file_names_the_tuning_step(tmp_path):
    with pytest.raises(SystemExit, match="tune-rerank"):
        rce.load_weights(str(tmp_path / "nope.json"))


def test_rerank_is_reproducible(tmp_path):
    run_dir, _ = build_run(tmp_path)
    weights = write_weights(tmp_path)
    a = rce.run_rerank(run_dir, "relation_reranked", json.loads(weights.read_text()),
                       str(weights))
    (run_dir / "captions_relation_reranked.meta.json").unlink()
    b = rce.run_rerank(run_dir, "relation_reranked", json.loads(weights.read_text()),
                       str(weights))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------------
# No ground truth may reach inference
# ---------------------------------------------------------------------------

def test_rerank_output_is_unchanged_when_the_ground_truth_is_poisoned(tmp_path):
    """The evaluation manifest is the only file carrying human annotations.
    Replacing every annotation must not move a single caption - if it did, the
    system would be grounding itself in the answer key."""
    run_dir, eval_path = build_run(tmp_path)
    weights = write_weights(tmp_path)
    before = rce.run_rerank(run_dir, "relation_reranked", json.loads(weights.read_text()),
                            str(weights))

    poisoned = json.loads(eval_path.read_text())
    for i in IDS:
        poisoned["images"][i]["objects"] = ["zebra", "toaster", "giraffe"]
    eval_path.write_text(json.dumps(poisoned), encoding="utf-8")

    (run_dir / "captions_relation_reranked.meta.json").unlink()
    after = rce.run_rerank(run_dir, "relation_reranked", json.loads(weights.read_text()),
                           str(weights))
    assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)


def test_rerank_evidence_is_the_verified_detections(tmp_path):
    run_dir, _ = build_run(tmp_path)
    weights = write_weights(tmp_path)
    rows = rce.run_rerank(run_dir, "object_reranked", json.loads(weights.read_text()),
                          str(weights))
    by_id = {r["image_id"]: r for r in rows}
    assert by_id["1"]["evidence_objects"] == ["bicycle", "person"]
    assert by_id["3"]["evidence_objects"] == []


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def test_tuning_refuses_a_test_split_run_directory(tmp_path):
    run_dir, _ = build_run(tmp_path)

    class Args:
        pass

    args = Args()
    args.run_dir, args.seed, args.output, args.force = str(run_dir), 42, str(tmp_path / "w.json"), False
    with pytest.raises(SystemExit, match="refusing to tune on a TEST"):
        rce.cmd_tune_rerank(args)


def test_tuning_on_a_validation_run_locks_weights_and_records_the_grid(tmp_path):
    run_dir, _ = build_run(tmp_path)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    cfg["eval_set"]["split"] = "val"
    (run_dir / "run_config.json").write_text(json.dumps(cfg), encoding="utf-8")

    class Args:
        pass

    args = Args()
    out = tmp_path / "weights.json"
    args.run_dir, args.seed, args.output, args.force = str(run_dir), 42, str(out), False
    assert rce.cmd_tune_rerank(args) == 0

    locked = json.loads(out.read_text())
    assert locked["selected_on"]["split"] == "val"
    assert locked["weights"]["object_reranked"]["w_rel"] == 0.0
    assert len(locked["object_grid"]) == len(rce.W_OBJ_GRID) * len(rce.W_HALL_GRID)
    assert len(locked["relation_grid"]) == len(rce.W_REL_GRID)
    # the relation arm inherits the object arm's two weights
    for key in ("w_obj", "w_hall"):
        assert locked["weights"]["relation_reranked"][key] == \
            locked["weights"]["object_reranked"][key]


def test_retuning_different_weights_over_a_lock_is_refused(tmp_path):
    run_dir, _ = build_run(tmp_path)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    cfg["eval_set"]["split"] = "val"
    (run_dir / "run_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    out = tmp_path / "weights.json"
    out.write_text(json.dumps({"weights": {"object_reranked": {"w_obj": 9.0},
                                           "relation_reranked": {"w_obj": 9.0}}}),
                   encoding="utf-8")

    class Args:
        pass

    args = Args()
    args.run_dir, args.seed, args.output, args.force = str(run_dir), 42, str(out), False
    with pytest.raises(SystemExit, match="post-hoc"):
        rce.cmd_tune_rerank(args)


# ---------------------------------------------------------------------------
# Scoring the extended run
# ---------------------------------------------------------------------------

def _score(tmp_path, arms=("object_reranked", "relation_reranked")):
    run_dir, eval_path = build_run(tmp_path)
    weights = write_weights(tmp_path)
    for arm in arms:
        rce.run_rerank(run_dir, arm, json.loads(weights.read_text()), str(weights))
    return run_dir, evaluate_run(run_dir, str(eval_path), allow_small=True, n_bootstrap=50,
                                 clipscore=False)


def test_all_five_arms_are_scored_together(tmp_path):
    _, results = _score(tmp_path)
    assert set(results["arms_present"]) == {"baseline", "grounded", "objects_only",
                                            "object_reranked", "relation_reranked"}
    assert not results["validity"]["problems"]


def test_paired_comparisons_cover_the_new_arms(tmp_path):
    _, results = _score(tmp_path)
    for key in ("object_reranked_minus_baseline", "relation_reranked_minus_baseline",
                "object_reranked_minus_objects_only", "relation_reranked_minus_grounded",
                "relation_reranked_minus_object_reranked"):
        assert key in results["paired_comparisons"], key


def test_rerank_statistics_are_reported(tmp_path):
    _, results = _score(tmp_path)
    st = results["rerank_statistics"]["relation_reranked"]
    assert st["n_images"] == 3
    assert st["candidate_generation_success_rate"] == 1.0
    assert st["candidates_per_image"]["max"] == 4
    assert set(st["selected_candidate_source"]) <= set(_prefixes("1"))
    assert 0 <= st["changed_from_first_candidate_rate"] <= 1
    assert 0 <= st["changed_by_relation_term_rate"] <= 1


def test_the_report_names_the_locked_weights(tmp_path):
    run_dir, _ = _score(tmp_path)
    report = (run_dir / "caption_results.md").read_text(encoding="utf-8")
    assert "Reranking behaviour" in report
    assert "relation evidence changed the selection" in report


def test_a_reranked_caption_outside_the_pool_makes_the_run_unscoreable(tmp_path):
    run_dir, eval_path = build_run(tmp_path)
    weights = write_weights(tmp_path)
    rce.run_rerank(run_dir, "object_reranked", json.loads(weights.read_text()), str(weights))
    path = run_dir / "captions_object_reranked.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["caption"] = "a caption BLIP never produced"
    rows[0]["scores"] = []
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="not one of the"):
        evaluate_run(run_dir, str(eval_path), allow_small=True, n_bootstrap=10,
                     clipscore=False)


def test_a_reranked_arm_without_its_candidate_pool_is_refused(tmp_path):
    run_dir, eval_path = build_run(tmp_path)
    weights = write_weights(tmp_path)
    rce.run_rerank(run_dir, "object_reranked", json.loads(weights.read_text()), str(weights))
    (run_dir / "candidates.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="candidates.jsonl"):
        evaluate_run(run_dir, str(eval_path), allow_small=True, n_bootstrap=10,
                     clipscore=False)


def test_the_three_prefix_arms_still_score_without_any_reranking(tmp_path):
    """The frozen experiment must remain scoreable exactly as before."""
    run_dir, eval_path = build_run(tmp_path)
    (run_dir / "candidates.jsonl").unlink()
    results = evaluate_run(run_dir, str(eval_path), allow_small=True, n_bootstrap=50,
                           clipscore=False)
    assert results["arms_present"] == ["baseline", "grounded", "objects_only"]
    assert results["rerank_statistics"] == {}
    assert not results["validity"]["problems"]

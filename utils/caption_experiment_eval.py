"""Score a caption-experiment run directory: baseline vs relation-grounded BLIP.

Reads what run_caption_experiment.py wrote (run_config.json, relations.jsonl,
captions_<arm>.jsonl), refuses to score anything whose pairing or grounding
path is inconsistent, and writes caption_results.json / caption_results.md /
per_image.jsonl. It never writes a conclusion: the report states numbers,
intervals and the pre-registered reading rules only.

Arms
----
baseline      prefix "a photo of"                          (reference)
grounded      prefix "a photo of a <s> <predicate> a <o>"  when the selected
              relation's confidence >= threshold, else the baseline prefix
objects_only  prefix "a photo of a <s> and a <o>"          (control: same
              objects, same trigger rule, no predicate)

For an image where an arm did NOT inject anything its prefix is the baseline
prefix, so its caption IS the baseline caption. The arm regenerates it anyway
as a determinism check, but metrics use the baseline arm's caption for those
images: otherwise run-to-run floating-point noise in beam search could make
the arms differ on images the relation never touched.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from utils.caption_hallucination import (
    POPE_SETTINGS,
    aggregate_chair,
    build_pope_probes,
    caption_object_record,
    check_pairing,
    mcnemar_exact,
    paired_bootstrap,
    score_pope,
)

ARMS = ("baseline", "grounded", "objects_only")
REQUIRED_ARMS = ("baseline", "grounded")
MIN_FINAL_IMAGES = 100
EXPLORATORY_THRESHOLDS = (0.0, 0.3, 0.5, 0.7, 0.9)
LEGACY_PLACEHOLDERS = ("cannot infer relations", "no semantic interactions detected",
                       "the scene contains", "no objects detected")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_jsonl(path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def keyed(rows: Sequence[Mapping], what: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for r in rows:
        key = str(r["image_id"])
        if key in out:
            raise ValueError(f"{what}: duplicate record for image {key}")
        out[key] = dict(r)
    return out


def expected_prefix(arm: str, relation: Optional[Mapping], threshold: float) -> str:
    from utils.blip_captioner import BASELINE_PREFIX, build_blip_prefix, build_objects_only_prefix

    if arm == "baseline" or relation is None or relation["confidence"] < threshold:
        return BASELINE_PREFIX
    if arm == "grounded":
        prefix, used = build_blip_prefix([], [dict(relation)], min_confidence=threshold)
        if not used:
            raise ValueError(f"relation {relation} would be silently dropped by build_blip_prefix")
        return prefix
    if arm == "objects_only":
        return build_objects_only_prefix(relation)
    raise ValueError(arm)


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------

def validate_run(ids: Sequence[str], relations: Mapping[str, Mapping],
                 arms: Mapping[str, Mapping[str, Mapping]], threshold: float) -> Dict:
    """Check that each arm used exactly the prefix its rule implies.

    Hard problems (wrong prefix, a relation that should have been injected but
    was not, a placeholder caption) make the run unscoreable. Soft findings are
    reported: prefix not echoed, fallback regeneration differing from baseline.
    """
    from utils.blip_captioner import BASELINE_PREFIX

    problems: List[str] = []
    stats: Dict[str, Dict] = {}
    base = arms["baseline"]
    for arm, recs in arms.items():
        s = Counter()
        for iid in ids:
            rec = recs[iid]
            rel_rec = relations[iid]
            selected = rel_rec.get("selected_relation")
            decision = rel_rec.get("decision", {})
            want = expected_prefix(arm, selected, threshold)
            if rec.get("prefix") != want:
                problems.append(f"{arm}/{iid}: prefix {rec.get('prefix')!r} != expected {want!r}")
            used = bool(rec.get("relation_used"))
            should_use = arm != "baseline" and bool(decision.get("use_relation"))
            if used != should_use:
                problems.append(f"{arm}/{iid}: relation_used={used} but decision says {should_use}")
            if used and rec.get("prefix") == BASELINE_PREFIX:
                problems.append(f"{arm}/{iid}: relation marked used but baseline prefix sent to BLIP")
            if used and base[iid].get("input_ids") == rec.get("input_ids"):
                problems.append(f"{arm}/{iid}: BLIP input ids identical to baseline despite relation")
            caption = str(rec.get("caption", ""))
            if not caption.strip():
                problems.append(f"{arm}/{iid}: empty caption")
            if any(p in caption.lower() for p in LEGACY_PLACEHOLDERS):
                problems.append(f"{arm}/{iid}: placeholder/template caption {caption!r}")
            s["n"] += 1
            s["relation_used"] += int(used)
            s["prefix_echoed"] += int(bool(rec.get("prefix_echoed")))
            if arm != "baseline" and not used:
                s["fallback"] += 1
                if caption != base[iid].get("caption"):
                    s["fallback_regeneration_differs_from_baseline"] += 1
            if arm != "baseline" and used and caption != base[iid].get("caption"):
                s["caption_differs_from_baseline"] += 1
        stats[arm] = dict(s)
    return {"problems": problems, "per_arm": stats}


# ---------------------------------------------------------------------------
# Relation statistics
# ---------------------------------------------------------------------------

def relation_statistics(ids: Sequence[str], relations: Mapping[str, Mapping]) -> Dict:
    n = len(ids)
    c = Counter()
    reasons = Counter()
    used_conf, selected_conf = [], []
    predicates_used = Counter()
    for iid in ids:
        r = relations[iid]
        c["images_with_raw_detection"] += int(len(r.get("raw_detections", [])) > 0)
        c["raw_detections"] += len(r.get("raw_detections", []))
        c["verified_detections"] += len(r.get("verified_detections", []))
        c["eligible_detections"] += len(r.get("eligible_detections", []))
        c["images_with_2plus_eligible"] += int(len(r.get("eligible_detections", [])) >= 2)
        c["pairs_scored"] += len(r.get("pair_predictions", []))
        c["pairs_rejected_semantic_filter"] += sum(
            1 for p in r.get("pair_predictions", []) if p.get("status") == "rejected_semantic_filter")
        sel = r.get("selected_relation")
        if sel is not None:
            c["images_with_selected_relation"] += 1
            selected_conf.append(sel["confidence"])
        dec = r.get("decision", {})
        if dec.get("use_relation"):
            c["images_relation_used"] += 1
            used_conf.append(sel["confidence"])
            predicates_used[sel["predicate"]] += 1
        else:
            reasons[dec.get("fallback_reason")] += 1

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "n_images": n,
        **dict(c),
        "relation_usage_rate": c["images_relation_used"] / n if n else 0.0,
        "relation_fallback_rate": 1 - c["images_relation_used"] / n if n else 0.0,
        "fallback_reasons": dict(reasons),
        "mean_confidence_used_relations": _mean(used_conf),
        "mean_confidence_selected_relations": _mean(selected_conf),
        "predicates_used": dict(predicates_used.most_common()),
        "mean_raw_detections_per_image": c["raw_detections"] / n if n else 0.0,
        "mean_verified_detections_per_image": c["verified_detections"] / n if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _per_image_vectors(recs: Mapping[str, Mapping], pope_counts: Mapping[str, Mapping[str, Mapping]],
                       ids: Sequence[str]) -> Dict[str, List[List[float]]]:
    v: Dict[str, List[List[float]]] = {
        "chair_i": [], "chair_s": [], "mean_hallucinated": [], "mean_mentioned": [],
        "object_recall": [],
    }
    for s in POPE_SETTINGS:
        v[f"pope_{s}_accuracy"] = []
        v[f"pope_{s}_f1"] = []
        v[f"pope_{s}_neg_fp_rate"] = []
    for iid in ids:
        r = recs[iid]
        v["chair_i"].append([r["n_hallucinated"], r["n_mentioned"]])
        v["chair_s"].append([1.0 if r["n_hallucinated"] else 0.0])
        v["mean_hallucinated"].append([r["n_hallucinated"]])
        v["mean_mentioned"].append([r["n_mentioned"]])
        v["object_recall"].append([r["n_correct"], r["n_gt"]])
        for s in POPE_SETTINGS:
            pc = pope_counts[s][iid]
            v[f"pope_{s}_accuracy"].append([pc["tp"] + pc["tn"], pc["n_probes"]])
            v[f"pope_{s}_f1"].append([pc["tp"], pc["fp"], pc["fn"]])
            v[f"pope_{s}_neg_fp_rate"].append([pc["fp"], pc["fp"] + pc["tn"]])
    return v


def _statistic(name: str, n_images: int):
    import numpy as np

    def ratio(s):
        return np.divide(s[:, 0], s[:, 1], out=np.zeros(len(s)), where=s[:, 1] > 0)

    def mean(s):
        return s[:, 0] / n_images

    def f1(s):
        den = 2 * s[:, 0] + s[:, 1] + s[:, 2]
        return np.divide(2 * s[:, 0], den, out=np.zeros(len(s)), where=den > 0)

    if name in ("chair_s", "mean_hallucinated", "mean_mentioned", "clipscore"):
        return mean
    if name.endswith("_f1"):
        return f1
    return ratio


def _compare(vec_a, vec_b, ids_idx, n_bootstrap, seed) -> Dict:
    out = {}
    n = len(ids_idx)
    if n == 0:
        return out
    for name in vec_a:
        a = [vec_a[name][i] for i in ids_idx]
        b = [vec_b[name][i] for i in ids_idx]
        out[name] = paired_bootstrap(a, b, _statistic(name, n), n_resamples=n_bootstrap, seed=seed)
    return out


def _pope_counts_per_image(probes_by_setting, mentioned) -> Dict[str, Dict[str, Dict]]:
    return {s: {iid: score_pope({iid: p}, mentioned) for iid, p in probes.items()}
            for s, probes in probes_by_setting.items()}


def evaluate_run(run_dir, eval_set_path: Optional[str] = None, allow_small: bool = False,
                 n_bootstrap: int = 10000, seed: int = 42, clipscore: bool = False,
                 clip_device: Optional[str] = None) -> Dict:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    eval_set_path = eval_set_path or cfg["eval_set"]["path"]
    eval_set = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))
    ids = [str(i) for i in cfg["image_ids"]]
    threshold = float(cfg["relation_threshold"])

    if eval_set["meta"].get("usable_ids_sha256") != cfg["eval_set"]["usable_ids_sha256"]:
        raise ValueError("evaluation set changed since the run was generated "
                         f"({eval_set_path})")
    if not set(ids) <= set(map(str, eval_set["usable_ids"])):
        raise ValueError("run contains images that are not usable images of the eval set")
    if not allow_small:
        if eval_set["meta"].get("status") != "OK":
            raise ValueError(f"evaluation set status is {eval_set['meta'].get('status')}")
        if len(ids) < MIN_FINAL_IMAGES:
            raise ValueError(f"only {len(ids)} images (< {MIN_FINAL_IMAGES}); too small to "
                             "support a conclusion")
        if cfg["eval_set"].get("split") != "test":
            raise ValueError("the caption experiment must be scored on the frozen TEST split")

    gt = {iid: set(eval_set["images"][iid]["objects"]) for iid in ids}
    relations = keyed(read_jsonl(run_dir / "relations.jsonl"), "relations.jsonl")
    missing_rel = set(ids) - set(relations)
    if missing_rel:
        raise ValueError(f"relations.jsonl lacks {len(missing_rel)} images")

    arms: Dict[str, Dict[str, Dict]] = {}
    for arm in ARMS:
        path = run_dir / f"captions_{arm}.jsonl"
        if path.is_file():
            arms[arm] = keyed(read_jsonl(path), path.name)
        elif arm in REQUIRED_ARMS:
            raise FileNotFoundError(f"{path} missing - generate the {arm} arm first")
    check_pairing(ids, arms)

    validity = validate_run(ids, relations, arms, threshold)
    if validity["problems"]:
        raise ValueError("run is not scoreable:\n  " + "\n  ".join(validity["problems"][:25]))

    # Captions used for metrics (fallback images -> the baseline arm's caption).
    scored_caption: Dict[str, Dict[str, str]] = {}
    for arm, recs in arms.items():
        scored_caption[arm] = {
            iid: (recs[iid]["caption"] if (arm == "baseline" or recs[iid]["relation_used"])
                  else arms["baseline"][iid]["caption"])
            for iid in ids
        }

    probes = build_pope_probes({iid: gt[iid] for iid in ids}, seed=seed)
    object_recs = {arm: {iid: caption_object_record(scored_caption[arm][iid], gt[iid]) for iid in ids}
                   for arm in arms}
    mentioned = {arm: {iid: set(object_recs[arm][iid]["mentioned"]) for iid in ids} for arm in arms}

    from utils.coco_mentions import mentions as _mentions

    metrics: Dict[str, Dict] = {}
    pope_per_image: Dict[str, Dict] = {}
    for arm in arms:
        metrics[arm] = {"chair": aggregate_chair([object_recs[arm][i] for i in ids]),
                        "pope": {s: score_pope(probes[s], mentioned[arm]) for s in POPE_SETTINGS}}
        pope_per_image[arm] = _pope_counts_per_image(probes, mentioned[arm])
        # Where do this arm's hallucinations come from? An object named in the
        # injected prefix was put there by YOLO + the relation model (or is a
        # real object the VG annotators named outside COCO-80); everything else
        # is BLIP's own continuation.
        from_prefix = total = 0
        for iid in ids:
            hall = set(object_recs[arm][iid]["hallucinated"])
            total += len(hall)
            if arm != "baseline" and arms[arm][iid].get("relation_used"):
                n = len(hall & _mentions(arms[arm][iid]["prefix"]))
                object_recs[arm][iid]["n_hallucinated_from_prefix"] = n
                from_prefix += n
        metrics[arm]["hallucination_source"] = {
            "total_hallucinated": total,
            "introduced_by_injected_prefix": from_prefix,
            "in_blip_continuation": total - from_prefix,
        }

    clip_scores: Dict[str, Dict[str, float]] = {}
    if clipscore:
        from utils.clipscore import compute_clipscores
        files = {iid: eval_set["images"][iid]["file"] for iid in ids}
        clip_scores = compute_clipscores(files, {arm: scored_caption[arm] for arm in arms},
                                         device=clip_device)
        for arm in arms:
            vals = [clip_scores[arm][i] for i in ids]
            metrics[arm]["clipscore"] = {"mean": sum(vals) / len(vals), "n": len(vals)}

    vectors = {arm: _per_image_vectors(object_recs[arm], pope_per_image[arm], ids) for arm in arms}
    if clipscore:
        for arm in arms:
            vectors[arm]["clipscore"] = [[clip_scores[arm][i]] for i in ids]

    used_idx = [k for k, iid in enumerate(ids) if relations[iid]["decision"].get("use_relation")]
    all_idx = list(range(len(ids)))
    comparisons: Dict[str, Dict] = {}
    pairs = [("grounded", "baseline")]
    if "objects_only" in arms:
        pairs += [("objects_only", "baseline"), ("grounded", "objects_only")]
    for arm_b, arm_a in pairs:
        key = f"{arm_b}_minus_{arm_a}"
        comparisons[key] = {}
        for subset, idx in (("all_images", all_idx), ("relation_used_images", used_idx)):
            b_only = sum(1 for k in idx if object_recs[arm_a][ids[k]]["n_hallucinated"] > 0
                         and object_recs[arm_b][ids[k]]["n_hallucinated"] == 0)
            c_only = sum(1 for k in idx if object_recs[arm_a][ids[k]]["n_hallucinated"] == 0
                         and object_recs[arm_b][ids[k]]["n_hallucinated"] > 0)
            comparisons[key][subset] = {
                "n_images": len(idx),
                "bootstrap": _compare(vectors[arm_a], vectors[arm_b], idx, n_bootstrap, seed),
                "chair_s_discordant": {f"{arm_a}_hallucinates_only": b_only,
                                       f"{arm_b}_hallucinates_only": c_only,
                                       "mcnemar_exact_p": mcnemar_exact(b_only, c_only)},
            }

    subset_metrics = {}
    for arm in arms:
        if used_idx:
            subset_metrics[arm] = {
                "chair": aggregate_chair([object_recs[arm][ids[k]] for k in used_idx]),
                "pope": {s: score_pope({ids[k]: probes[s][ids[k]] for k in used_idx}, mentioned[arm])
                         for s in POPE_SETTINGS},
            }

    sweep = {}
    if "grounded" in arms:
        for tau in EXPLORATORY_THRESHOLDS:
            caps, n_used = {}, 0
            for iid in ids:
                sel = relations[iid].get("selected_relation")
                rc = arms["grounded"][iid].get("relation_caption")
                if sel is not None and rc and sel["confidence"] >= tau:
                    caps[iid] = rc
                    n_used += 1
                else:
                    caps[iid] = arms["baseline"][iid]["caption"]
            recs = [caption_object_record(caps[i], gt[i]) for i in ids]
            ment = {i: set(r["mentioned"]) for i, r in zip(ids, recs)}
            sweep[str(tau)] = {"relation_usage_rate": n_used / len(ids),
                               "chair": aggregate_chair(recs),
                               "pope_adversarial": score_pope(probes["adversarial"], ment)}

    results = {
        "run_dir": str(run_dir).replace("\\", "/"),
        "n_images": len(ids),
        "relation_threshold": threshold,
        "relation_threshold_is_predeclared_default": bool(cfg.get("relation_threshold_is_default")),
        "eval_set": cfg["eval_set"],
        "relation_checkpoint": cfg.get("relation_checkpoint"),
        "arms_present": list(arms),
        "validity": validity,
        "relation_statistics": relation_statistics(ids, relations),
        "metrics_all_images": metrics,
        "metrics_relation_used_images": subset_metrics,
        "paired_comparisons": comparisons,
        "exploratory_threshold_sweep": sweep,
        "notes": [
            "CHAIR/POPE ground truth: human Visual Genome annotations mapped to COCO-80; "
            "not exhaustive, so absolute hallucination rates are upper bounds for every arm.",
            "Fallback images are scored with the baseline arm's caption (same prefix).",
            "The threshold sweep reuses the grounded arm's relation captions and is "
            "exploratory; the primary result is the pre-declared threshold.",
            "CLIPScore is reference-free image-text alignment (Hessel et al., 2021), "
            "not a reference-based caption quality metric; no reference captions exist "
            "for these images, so BLEU/CIDEr/SPICE are not reported.",
        ],
    }

    with open(run_dir / "per_image.jsonl", "w", encoding="utf-8") as fh:
        for iid in ids:
            row = {"image_id": iid, "gt_objects": sorted(gt[iid]),
                   "selected_relation": relations[iid].get("selected_relation"),
                   "decision": relations[iid].get("decision"),
                   "arms": {}}
            for arm in arms:
                row["arms"][arm] = {
                    "caption": scored_caption[arm][iid],
                    "prefix": arms[arm][iid]["prefix"],
                    "relation_used": bool(arms[arm][iid].get("relation_used")),
                    **{k: object_recs[arm][iid][k] for k in
                       ("mentioned", "hallucinated", "n_mentioned", "n_hallucinated", "n_words")},
                }
                if clipscore:
                    row["arms"][arm]["clipscore"] = clip_scores[arm][iid]
            fh.write(json.dumps(row) + "\n")
    (run_dir / "caption_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "caption_results.md").write_text(render_markdown(results), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.2f}%"


def _ci(entry, scale=100.0, unit="pt") -> str:
    if not entry:
        return "n/a"
    d, (lo, hi) = entry["delta"], entry["ci95"]
    return f"{scale * d:+.2f} {unit} [{scale * lo:+.2f}, {scale * hi:+.2f}]"


def render_markdown(r: Mapping) -> str:
    L: List[str] = []
    rs = r["relation_statistics"]
    L.append("# Caption experiment: baseline vs relation-grounded BLIP\n")
    L.append(f"- images: **{r['n_images']}** (frozen test split, eval set "
             f"`{r['eval_set']['path']}`, ids sha256 `{r['eval_set']['usable_ids_sha256'][:16]}`)")
    ck = r.get("relation_checkpoint") or {}
    L.append(f"- relation checkpoint: `{ck.get('checkpoint_dir')}` (selected by validation "
             f"top-1: {ck.get('selection_rule_applied')})")
    L.append(f"- relation confidence threshold: **{r['relation_threshold']}**"
             + ("" if r["relation_threshold_is_predeclared_default"]
                else "  **(NOT the pre-declared 0.5 - deviation)**"))
    L.append(f"- arms: {', '.join(r['arms_present'])}\n")

    L.append("## Validity checks\n")
    L.append("| arm | images | relation used | fallback | prefix echoed | caption differs from baseline (used) | fallback regeneration != baseline |")
    L.append("|---|---|---|---|---|---|---|")
    for arm, s in r["validity"]["per_arm"].items():
        L.append(f"| {arm} | {s.get('n', 0)} | {s.get('relation_used', 0)} | {s.get('fallback', 0)} | "
                 f"{s.get('prefix_echoed', 0)} | {s.get('caption_differs_from_baseline', 0)} | "
                 f"{s.get('fallback_regeneration_differs_from_baseline', 0)} |")
    L.append("\nAll prefixes matched their rule; no relation was dropped before BLIP; no "
             "placeholder captions.\n")

    L.append("## Relation usage\n")
    L.append(f"- images with >=1 YOLO detection: {rs.get('images_with_raw_detection', 0)}/{rs['n_images']}")
    L.append(f"- mean detections per image: raw {rs['mean_raw_detections_per_image']:.2f}, "
             f"CLIP-verified {rs['mean_verified_detections_per_image']:.2f}")
    L.append(f"- images with >=2 eligible detections: {rs.get('images_with_2plus_eligible', 0)}")
    L.append(f"- images with a selected relation: {rs.get('images_with_selected_relation', 0)}")
    L.append(f"- **relation usage rate: {_pct(rs['relation_usage_rate'])}** "
             f"(fallback to baseline prefix: {_pct(rs['relation_fallback_rate'])})")
    L.append(f"- fallback reasons: {rs['fallback_reasons']}")
    mc = rs["mean_confidence_used_relations"]
    L.append(f"- mean confidence of used relations: {'n/a' if mc is None else f'{mc:.3f}'}")
    L.append(f"- predicates used: {rs['predicates_used']}\n")

    def table(metrics: Mapping, title: str):
        L.append(f"## {title}\n")
        arms = list(metrics)
        L.append("| metric | " + " | ".join(arms) + " |")
        L.append("|---|" + "---|" * len(arms))
        rows = [
            ("CHAIR_i (lower better)", lambda m: _pct(m["chair"]["chair_i"])),
            ("CHAIR_s (lower better)", lambda m: _pct(m["chair"]["chair_s"])),
            ("hallucinated objects / caption", lambda m: f"{m['chair']['mean_hallucinated_objects']:.3f}"),
            ("mentioned objects / caption", lambda m: f"{m['chair']['mean_mentioned_objects']:.3f}"),
            ("object recall", lambda m: _pct(m["chair"]["object_recall"])),
            ("caption length (words)", lambda m: f"{m['chair']['mean_caption_words']:.2f}"),
        ]
        for s in POPE_SETTINGS:
            rows.append((f"POPE-{s} accuracy", lambda m, s=s: _pct(m["pope"][s]["accuracy"])))
            rows.append((f"POPE-{s} F1", lambda m, s=s: _pct(m["pope"][s]["f1"])))
            rows.append((f"POPE-{s} false-positive rate on absent objects",
                         lambda m, s=s: _pct(m["pope"][s]["negative_false_positive_rate"])))
            rows.append((f"POPE-{s} yes-ratio", lambda m, s=s: _pct(m["pope"][s]["yes_ratio"])))
        for label, fn in rows:
            L.append(f"| {label} | " + " | ".join(fn(metrics[a]) for a in arms) + " |")
        if all("hallucination_source" in metrics[a] for a in arms):
            L.append("| hallucinations introduced by injected prefix | " + " | ".join(
                str(metrics[a]["hallucination_source"]["introduced_by_injected_prefix"]) for a in arms) + " |")
            L.append("| hallucinations in BLIP continuation | " + " | ".join(
                str(metrics[a]["hallucination_source"]["in_blip_continuation"]) for a in arms) + " |")
        L.append("")

    table(r["metrics_all_images"], "Hallucination metrics - all images")
    if r["metrics_relation_used_images"]:
        n_used = r["relation_statistics"].get("images_relation_used", 0)
        table(r["metrics_relation_used_images"],
              f"Hallucination metrics - images where a relation was injected (n={n_used})")

    L.append("## Paired differences (95% bootstrap CI over images)\n")
    names = [("chair_i", "CHAIR_i"), ("chair_s", "CHAIR_s"), ("mean_hallucinated", "halluc./caption"),
             ("object_recall", "object recall"), ("pope_adversarial_f1", "POPE-adv F1"),
             ("pope_adversarial_neg_fp_rate", "POPE-adv FP rate")]
    for key, comp in r["paired_comparisons"].items():
        for subset, c in comp.items():
            if not c.get("n_images"):
                continue
            L.append(f"**{key}**, {subset} (n={c['n_images']})\n")
            L.append("| metric | difference |")
            L.append("|---|---|")
            for m, label in names:
                if m == "mean_hallucinated":
                    L.append(f"| {label} | {_ci(c['bootstrap'].get(m), 1.0, 'obj')} |")
                else:
                    L.append(f"| {label} | {_ci(c['bootstrap'].get(m))} |")
            if "clipscore" in c["bootstrap"]:
                L.append(f"| CLIPScore | {_ci(c['bootstrap']['clipscore'], 1.0, '')} |")
            d = c["chair_s_discordant"]
            L.append(f"\nCHAIR_s discordant pairs: {d} \n")

    if any("clipscore" in m for m in r["metrics_all_images"].values()):
        L.append("## Caption quality (reference-free)\n")
        L.append("CLIPScore = 2.5 * max(cos(image, 'A photo depicts ' + caption), 0), "
                 "CLIP ViT-B/32. Image-text alignment, NOT a hallucination metric and not "
                 "reference-based.\n")
        L.append("| arm | CLIPScore |")
        L.append("|---|---|")
        for arm, m in r["metrics_all_images"].items():
            if "clipscore" in m:
                L.append(f"| {arm} | {m['clipscore']['mean']:.4f} |")
        L.append("")

    if r["exploratory_threshold_sweep"]:
        L.append("## EXPLORATORY threshold sweep (not the primary result)\n")
        L.append("| threshold | usage | CHAIR_i | CHAIR_s | POPE-adv F1 |")
        L.append("|---|---|---|---|---|")
        for tau, s in r["exploratory_threshold_sweep"].items():
            L.append(f"| {tau} | {_pct(s['relation_usage_rate'])} | {_pct(s['chair']['chair_i'])} | "
                     f"{_pct(s['chair']['chair_s'])} | {_pct(s['pope_adversarial']['f1'])} |")
        L.append("")

    L.append("## Notes\n")
    for n in r["notes"]:
        L.append(f"- {n}")
    L.append("\nNo conclusion is written automatically. Read the paired differences against "
             "the pre-registered rules in CAPTION_EXPERIMENTS.md.")
    return "\n".join(L) + "\n"

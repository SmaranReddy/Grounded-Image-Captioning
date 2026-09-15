"""Run one arm of the visual feature ablation, end to end, with provenance.

The question this experiment exists to answer
---------------------------------------------
The audit established that a zero-geometry lookup table over (subject class,
object class) already scores 57.4% on the frozen test set against the model's
63.9%, that training accuracy plateaus at 65.5%, and that the reported run had
use_visual=false. So the model is close to a class-prior memoriser and the one
untested lever is actual pixels. This runner measures that lever:

    geometry              label embeddings + 19-D geometry          (control)
    geometry_clip         + subject and object CLIP crops
    geometry_clip_union   + the union region spanning the pair
    clip_only             CLIP crops with NO geometry               (control)

Everything else is held fixed: the same frozen image-disjoint split, the same
predicate scheme, the same loss, the same optimiser, the same architecture, and
- critically - the same sample population. Turning visual features on drops
pairs whose crops are missing or degenerate, so the geometry control is run
with --visual-filter-only: it loads the same CLIP cache and keeps the same
samples, but is built with clip_dim=0. Without that, "geometry vs geometry+CLIP"
would differ in the features AND in the test set, and neither number would mean
anything.

Usage
-----
    python run_visual_experiment.py --variant geometry --seed 42
    python run_visual_experiment.py --variant geometry_clip --seed 42
    python run_visual_experiment.py --variant geometry_clip_union --seed 42
    python run_visual_experiment.py --variant clip_only --seed 42

    python run_visual_experiment.py --all                  # 4 variants x 3 seeds
    python run_visual_experiment.py --all --dry-run        # print the commands
    python run_visual_experiment.py --collect              # summarise results

Each run writes results_gpu/<variant>_seed<seed>/run.json recording the git
commit, the split manifest hash, the predicate scheme, the resolved feature
configuration, dataset coverage, the training configuration and the test
metrics - enough to reconstruct what produced a number without notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT))

DEFAULT_VG_ROOT = PROJ_ROOT / "data" / "visual_genome"
DEFAULT_MANIFEST = PROJ_ROOT / "splits" / "e0_image_split.json"
DEFAULT_RESULTS = PROJ_ROOT / "results_gpu"

SEEDS = (42, 43, 44)

# Shared across every arm. These are the E2 settings the 63.89% run used, minus
# the geometry choice, so the arms stay comparable to the frozen baseline.
COMMON_TRAIN_ARGS = [
    "--model", "mlp",
    "--predicate-scheme", "v1",
    "--loss", "ce",
    "--class-weight-alpha", "0",
    "--use-visual",
    "--require-visual",
]

VARIANTS: Dict[str, Dict] = {
    "geometry": {
        "geo_mode": "ext", "geo_norm": True,
        "union": False, "visual_filter_only": True,
        "desc": "label embeddings + 19-D geometry (control, visual-complete population)",
    },
    "geometry_clip": {
        "geo_mode": "ext", "geo_norm": True,
        "union": False, "visual_filter_only": False,
        "desc": "geometry + subject/object CLIP crops",
    },
    "geometry_clip_union": {
        "geo_mode": "ext", "geo_norm": True,
        "union": True, "visual_filter_only": False,
        "desc": "geometry + object CLIP + union-region CLIP",
    },
    "clip_only": {
        # geo_norm is meaningless with no geometry to normalise; this is the
        # one unavoidable asymmetry between the arms, and it is inert because
        # the block it would normalise has width 0.
        "geo_mode": "none", "geo_norm": False,
        "union": False, "visual_filter_only": False,
        "desc": "subject/object CLIP crops, NO geometry (control)",
    },
}


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def git_commit() -> Dict[str, Optional[str]]:
    def run(*a):
        try:
            return subprocess.run(a, cwd=PROJ_ROOT, capture_output=True,
                                  text=True, timeout=15).stdout.strip() or None
        except Exception:
            return None
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def file_sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def preflight(vg_root: Path, manifest: Path, clip_cache: Path,
              need_union: bool, strict: bool) -> Dict:
    """Refuse to start a run whose inputs are incomplete.

    A run that quietly trains on 60% of the intended samples produces a number
    that looks fine and means nothing, and the only place to catch that is
    before the GPU starts. Everything here is cheap.
    """
    import torch

    from build_clip_cache import enumerate_work, report_coverage
    from prepare_visual_genome import coverage_report, required_image_ids

    print("=" * 74)
    print("PREFLIGHT")
    print("=" * 74)
    problems: List[str] = []
    out: Dict = {}

    # 1. split manifest -------------------------------------------------
    if not manifest.is_file():
        raise SystemExit(f"[preflight] FATAL: split manifest missing: {manifest}")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    tr, va, te = (set(map(int, m["train_ids"])), set(map(int, m["val_ids"])),
                  set(map(int, m["test_ids"])))
    overlaps = {"train_val": len(tr & va), "train_test": len(tr & te),
                "val_test": len(va & te)}
    print(f"\n[1] split manifest  {manifest.name}")
    print(f"    images {len(tr):,}/{len(va):,}/{len(te):,}  overlaps {overlaps}")
    if any(overlaps.values()):
        problems.append(f"split manifest has overlapping images: {overlaps}")
    out["split"] = {"sha256": file_sha256(manifest), "overlaps": overlaps,
                    "n_images": {"train": len(tr), "val": len(va), "test": len(te)}}

    # 2. image coverage --------------------------------------------------
    print("\n[2] Visual Genome images")
    per_split, _ = required_image_ids(manifest)
    img_rep = coverage_report(per_split, vg_root / "images")
    out["images"] = img_rep
    if img_rep["total"]["missing"]:
        problems.append(f"{img_rep['total']['missing']:,} required images missing "
                        f"(run prepare_visual_genome.py --download)")

    # 3. CLIP cache coverage ---------------------------------------------
    print("\n[3] CLIP feature cache")
    per_image, enum_stats = enumerate_work(vg_root, "v1", need_union)
    cache_rep = report_coverage(per_image, clip_cache)
    out["clip_cache"] = cache_rep
    out["enumerate"] = enum_stats
    if not cache_rep.get("exists"):
        problems.append(f"no CLIP cache at {clip_cache} (run build_clip_cache.py)")
    else:
        if cache_rep["object"]["coverage_pct"] < 99.99:
            problems.append(f"object CLIP coverage {cache_rep['object']['coverage_pct']}% "
                            f"(< 100%)")
        if need_union and cache_rep["union"]["coverage_pct"] < 99.99:
            problems.append(f"union CLIP coverage {cache_rep['union']['coverage_pct']}% "
                            f"(< 100%); rebuild with --include-union")

    # 4. cache sanity: dimensionality, NaN / inf, zero rows ---------------
    print("\n[4] cache numerics")
    if cache_rep.get("exists"):
        from relation_prediction.clip_cache import ClipCache
        from relation_prediction.clip_extractor import CLIP_DIM
        cache = ClipCache.load(str(clip_cache))
        emb = cache.embeddings
        n_nan = int(torch.isnan(emb).any(dim=1).sum())
        n_inf = int(torch.isinf(emb).any(dim=1).sum())
        norms = emb.norm(dim=1)
        n_zero = int((norms == 0).sum())
        print(f"    shape {tuple(emb.shape)}  nan_rows {n_nan}  inf_rows {n_inf}  "
              f"zero_rows {n_zero}")
        print(f"    norm  min {norms.min():.4f}  mean {norms.mean():.4f}  "
              f"max {norms.max():.4f}   (L2-normalised, so ~1.0)")
        out["cache_numerics"] = {"shape": list(emb.shape), "nan_rows": n_nan,
                                 "inf_rows": n_inf, "zero_rows": n_zero,
                                 "norm_min": float(norms.min()),
                                 "norm_mean": float(norms.mean()),
                                 "norm_max": float(norms.max())}
        if emb.shape[1] != CLIP_DIM:
            problems.append(f"cache dim {emb.shape[1]} != expected {CLIP_DIM}")
        if n_nan or n_inf:
            problems.append(f"cache contains {n_nan} NaN and {n_inf} inf rows")
        if n_zero:
            problems.append(f"cache contains {n_zero} all-zero rows")

    # 5. device -----------------------------------------------------------
    cuda = torch.cuda.is_available()
    print(f"\n[5] device: {'cuda - ' + torch.cuda.get_device_name(0) if cuda else 'CPU ONLY'}")
    out["cuda"] = cuda
    if not cuda:
        print("    (training will run, just slowly)")

    print("\n" + "-" * 74)
    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print(f"  * {p}")
        if strict:
            raise SystemExit(1)
        print("  (--no-strict given: continuing anyway)")
    else:
        print("PREFLIGHT PASSED")
    print("-" * 74 + "\n")
    out["problems"] = problems
    return out


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------

def child_env() -> Dict[str, str]:
    """Environment for the train / eval subprocesses.

    Their stdout is redirected to train.log / eval.log, and on Windows a
    redirected stream is encoded with the ANSI code page (cp1252), which cannot
    represent every character the scripts print. UTF-8 keeps the logs lossless;
    the scripts additionally never raise on an unencodable character
    (utils/console.py), so this is belt and braces, not the fix itself.
    """
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}


def build_commands(variant: str, seed: int, vg_root: Path, manifest: Path,
                   clip_cache: Path, ckpt_dir: Path, results_dir: Path,
                   epochs: int, batch_size: int):
    v = VARIANTS[variant]
    train = [sys.executable, str(PROJ_ROOT / "train_full_visual_semantic.py"),
             *COMMON_TRAIN_ARGS,
             "--seed", str(seed),
             "--epochs", str(epochs),
             "--batch-size", str(batch_size),
             "--geo-mode", v["geo_mode"],
             "--vg-root", str(vg_root),
             "--clip-cache", str(clip_cache),
             "--split-manifest", str(manifest),
             "--checkpoint-dir", str(ckpt_dir)]
    if v["geo_norm"]:
        train.append("--geo-norm")
    if v["union"]:
        train.append("--use-union")
    if v["visual_filter_only"]:
        train.append("--visual-filter-only")

    evaluate = [sys.executable, str(PROJ_ROOT / "eval_gt_relations.py"),
                "--vg-root", str(vg_root),
                "--checkpoint-dir", str(ckpt_dir),
                "--split-manifest", str(manifest),
                "--clip-cache-path", str(clip_cache),
                "--results-dir", str(results_dir),
                "--output-name", f"{variant}_seed{seed}",
                "--force"]
    return train, evaluate


def run_one(variant: str, seed: int, args, pre: Optional[Dict]) -> Dict:
    v = VARIANTS[variant]
    tag = f"{variant}_seed{seed}"
    run_dir = Path(args.results_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_root) / tag

    train_cmd, eval_cmd = build_commands(
        variant, seed, Path(args.vg_root), Path(args.split_manifest),
        Path(args.clip_cache), ckpt_dir, run_dir, args.epochs, args.batch_size)

    print("=" * 74)
    print(f"RUN  {tag}")
    print(f"     {v['desc']}")
    print("=" * 74)
    print("  train:", " ".join(f'"{c}"' if " " in c else c for c in train_cmd))
    print("  eval :", " ".join(f'"{c}"' if " " in c else c for c in eval_cmd))
    if args.dry_run:
        return {"tag": tag, "dry_run": True}

    record: Dict = {
        "tag": tag, "variant": variant, "seed": seed,
        "description": v["desc"],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "feature_config": {k: v[k] for k in
                           ("geo_mode", "geo_norm", "union", "visual_filter_only")},
        "train_config": {"epochs": args.epochs, "batch_size": args.batch_size,
                         "seed": seed, "predicate_scheme": "v1",
                         "loss": "ce", "class_weight_alpha": 0.0,
                         "model": "mlp", "require_visual": True},
        "split_manifest": {"path": str(args.split_manifest),
                           "sha256": file_sha256(Path(args.split_manifest))},
        "clip_cache": {"path": str(args.clip_cache),
                       "sha256_skipped": "cache files are multi-GB; see coverage instead"},
        "preflight": pre,
        "commands": {"train": train_cmd, "eval": eval_cmd},
        "checkpoint_dir": str(ckpt_dir),
    }

    t0 = time.time()
    train_log = run_dir / "train.log"
    with open(train_log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(train_cmd, cwd=PROJ_ROOT, stdout=fh,
                            stderr=subprocess.STDOUT, text=True,
                            env=child_env()).returncode
    record["train_seconds"] = round(time.time() - t0, 1)
    record["train_returncode"] = rc
    print(f"  training finished rc={rc} in {record['train_seconds']}s "
          f"(log: {train_log})")
    if rc != 0:
        record["status"] = "TRAIN_FAILED"
        (run_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"  !! training failed; see {train_log}")
        return record

    t1 = time.time()
    eval_log = run_dir / "eval.log"
    with open(eval_log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(eval_cmd, cwd=PROJ_ROOT, stdout=fh,
                            stderr=subprocess.STDOUT, text=True,
                            env=child_env()).returncode
    record["eval_seconds"] = round(time.time() - t1, 1)
    record["eval_returncode"] = rc

    result_json = run_dir / f"{tag}.json"
    if rc == 0 and result_json.is_file():
        res = json.loads(result_json.read_text(encoding="utf-8"))
        record["metrics"] = res.get("metrics", {})
        record["counts"] = res.get("counts", {})
        record["model_config"] = res.get("model_config", res.get("config", {}))
        record["status"] = "OK"
        m = record["metrics"]
        print(f"  top1 {m.get('top1')}  recall@3 {m.get('recall@3')}  "
              f"macro_f1 {m.get('macro_f1')}")
    else:
        record["status"] = "EVAL_FAILED"
        print(f"  !! evaluation failed; see {eval_log}")

    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (run_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  provenance: {run_dir / 'run.json'}")
    return record


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------

def collect(results_dir: Path) -> None:
    rows = []
    for rj in sorted(results_dir.glob("*/run.json")):
        r = json.loads(rj.read_text(encoding="utf-8"))
        if r.get("status") != "OK":
            continue
        rows.append(r)
    if not rows:
        print(f"[collect] no completed runs under {results_dir}")
        return

    print("=" * 88)
    print(f"{'variant':<22}{'seed':>6}{'n_test':>9}{'top1':>9}{'R@3':>9}"
          f"{'macroF1':>10}{'train_s':>10}")
    print("-" * 88)
    by_variant: Dict[str, List[float]] = {}
    for r in rows:
        m, c = r.get("metrics", {}), r.get("counts", {})
        top1 = m.get("top1")
        by_variant.setdefault(r["variant"], []).append(top1)
        print(f"{r['variant']:<22}{r['seed']:>6}{c.get('n_samples_test', '?'):>9}"
              f"{top1:>9.4f}{m.get('recall@3', 0):>9.4f}"
              f"{m.get('macro_f1', 0):>10.4f}{r.get('train_seconds', 0):>10.0f}")

    print("-" * 88)
    print(f"{'variant':<22}{'runs':>6}{'mean top1':>12}{'std':>10}")
    for name in VARIANTS:
        vals = by_variant.get(name)
        if not vals:
            continue
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{name:<22}{len(vals):>6}{statistics.mean(vals):>12.4f}{sd:>10.4f}")
    print("=" * 88)

    # Population check: every arm must have been scored on the same test set.
    ns = {r["variant"]: r.get("counts", {}).get("n_samples_test") for r in rows}
    distinct = set(v for v in ns.values() if v is not None)
    if len(distinct) > 1:
        print("\n!! WARNING: arms were scored on DIFFERENT test-set sizes: "
              f"{ns}\n   The comparison is confounded. Check --visual-filter-only "
              "on the geometry arm.")
    else:
        print(f"\nAll arms scored on the same {distinct.pop() if distinct else '?'} "
              "test samples.")

    summary = results_dir / "summary.json"
    summary.write_text(json.dumps(
        {"runs": [{k: r.get(k) for k in
                   ("tag", "variant", "seed", "metrics", "counts",
                    "train_seconds", "git", "feature_config")} for r in rows]},
        indent=2), encoding="utf-8")
    print(f"\n[collect] wrote {summary}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one arm of the visual feature ablation.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true",
                    help=f"Run every variant for seeds {SEEDS}")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--collect", action="store_true",
                    help="Summarise the runs already in --results-dir and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands without running them")
    ap.add_argument("--vg-root", default=str(DEFAULT_VG_ROOT))
    ap.add_argument("--split-manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--clip-cache", default=None,
                    help="Default: <vg-root>/clip_cache_proper.pt")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--checkpoint-root", default=str(PROJ_ROOT / "checkpoints_gpu"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="Warn instead of aborting when preflight finds problems")
    args = ap.parse_args()

    if args.clip_cache is None:
        args.clip_cache = str(Path(args.vg_root) / "clip_cache_proper.pt")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if args.collect:
        collect(Path(args.results_dir))
        return 0

    if args.all:
        # VARIANTS insertion order, not sorted(): the controls should run in
        # the order the ablation is read (geometry -> +clip -> +union -> clip
        # only), and alphabetical order would put clip_only first.
        jobs = [(v, s) for v in VARIANTS for s in args.seeds]
    elif args.variant:
        jobs = [(args.variant, args.seed)]
    else:
        ap.error("give --variant NAME, or --all, or --collect")

    need_union = any(VARIANTS[v]["union"] for v, _ in jobs)
    pre = None
    if not args.skip_preflight and not args.dry_run:
        pre = preflight(Path(args.vg_root), Path(args.split_manifest),
                        Path(args.clip_cache), need_union, args.strict)

    print(f"\n{len(jobs)} run(s) queued: "
          + ", ".join(f"{v}/seed{s}" for v, s in jobs) + "\n")

    failures = 0
    for variant, seed in jobs:
        rec = run_one(variant, seed, args, pre)
        if rec.get("status") not in (None, "OK") and not args.dry_run:
            failures += 1
        print()

    if not args.dry_run:
        collect(Path(args.results_dir))
    return 1 if failures else 0


if __name__ == "__main__":
    from utils.console import configure_safe_stdio
    configure_safe_stdio()
    raise SystemExit(main())

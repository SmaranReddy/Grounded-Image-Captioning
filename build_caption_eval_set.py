"""
build_caption_eval_set.py — the fixed image set and human object ground truth
for the baseline-vs-grounded caption experiment.

What it produces
----------------
A JSON manifest naming N images from the frozen relation TEST split, each with
the COCO-80 objects a human annotated in it, and whether its image file is on
disk. Every arm of the caption experiment is run and scored on exactly the
`usable_ids` it lists.

Ground truth: Visual Genome human annotations, never a model
------------------------------------------------------------
* objects.json - every annotated object region's names. This is the primary
  source: it covers all annotated objects, not just related ones.
* relationships.json - subject/object names, unioned in (a subset in practice).

Both are mapped to COCO-80 with utils.coco_mentions, the SAME mapper the
evaluation applies to captions, so the ground truth and the caption side can
never disagree about what "table" or "bikes" means.

The previous version of this script read relationships.json only. That lists
an object only if an annotator related it to something, so an object a caption
correctly names would be scored as a hallucination whenever it took part in no
annotated relation. It is still available with --allow-relationships-only, and
the manifest records which source was used.

No YOLO, CLIP, BLIP or relation model is imported or run here: the image set
and its ground truth are fixed before any prediction exists.

Selection (deterministic)
-------------------------
    eligible  = frozen test-split images with >= --min-objects GT COCO objects
    order     = random.Random(seed).shuffle(sorted(eligible))
    requested = order[:limit]
    usable    = requested images whose file exists and decodes

Availability does NOT change which images are requested (no back-filling), so
two machines with the same annotation files request the same images. If fewer
than --min-final requested images are usable the manifest is written with
status TOO_SMALL and the script exits non-zero: the experiment must not run.

Usage
-----
    python download_vg.py --objects        # once: fetch objects.json
    python build_caption_eval_set.py       # 250 test images, seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from utils.coco_mentions import MENTION_MAPPER_VERSION, objects_from_names  # noqa: E402

DEFAULT_VG_ROOT = "./data/visual_genome"
DEFAULT_MANIFEST = "./splits/e0_image_split.json"
DEFAULT_OUTPUT = "./splits/caption_eval_test_250.json"
DEFAULT_LIMIT = 250
MIN_FINAL_IMAGES = 100


def stream_records(path: str) -> Iterable[Dict]:
    """Top-level array records, one at a time (VG files are hundreds of MB)."""
    from relation_prediction.vg_dataset import stream_relationship_records
    return stream_relationship_records(path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Canonical hash of the frozen E0 split: sha256 over the sorted train|val|test
# image ids. Hashing the file bytes would differ between a CRLF (Windows,
# core.autocrlf) and an LF checkout of the identical manifest.
FROZEN_SPLIT_IDS_SHA256 = "45386f40d8aa92944a13ba3b92c2baaf340794642095367dd3b5c20ddedf827b"


def split_ids_sha256(split_manifest: Dict) -> str:
    joined = "|".join(",".join(str(i) for i in sorted(int(x) for x in split_manifest[k]))
                      for k in ("train_ids", "val_ids", "test_ids"))
    return hashlib.sha256(joined.encode()).hexdigest()


def resolve_image_path(image_dir: Optional[str], image_id: int) -> Optional[str]:
    """Locate a VG image, honouring the VG_100K / VG_100K_2 layout."""
    if not image_dir:
        return None
    for subdir in ("", "VG_100K", "VG_100K_2"):
        for ext in (".jpg", ".png", ".jpeg"):
            path = (os.path.join(image_dir, subdir, f"{image_id}{ext}")
                    if subdir else os.path.join(image_dir, f"{image_id}{ext}"))
            if os.path.isfile(path):
                return path.replace("\\", "/")
    return None


def _entity_names(entity: Dict) -> List[str]:
    names = list(entity.get("names") or [])
    if entity.get("name"):
        names.append(entity["name"])
    return [str(n) for n in names]


def collect_ground_truth(vg_root: str, wanted: Set[int], use_objects: bool = True,
                         use_relationships: bool = True) -> Dict[int, Dict[str, Set[str]]]:
    """image_id -> {"objects_json": set, "relationships_json": set} of COCO classes."""
    gt: Dict[int, Dict[str, Set[str]]] = {}

    def _slot(iid: int) -> Dict[str, Set[str]]:
        return gt.setdefault(iid, {"objects_json": set(), "relationships_json": set()})

    if use_objects:
        path = os.path.join(vg_root, "objects.json")
        for record in stream_records(path):
            iid = record.get("image_id")
            if iid is None or int(iid) not in wanted:
                continue
            slot = _slot(int(iid))
            for obj in record.get("objects", []):
                slot["objects_json"] |= objects_from_names(_entity_names(obj))
    if use_relationships:
        path = os.path.join(vg_root, "relationships.json")
        for record in stream_records(path):
            iid = record.get("image_id")
            if iid is None or int(iid) not in wanted:
                continue
            slot = _slot(int(iid))
            for rel in record.get("relationships", []):
                for side in ("subject", "object"):
                    slot["relationships_json"] |= objects_from_names(
                        _entity_names(rel.get(side, {})))
    return gt


def check_image(path: Optional[str]) -> Dict:
    if path is None:
        return {"status": "missing"}
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.load()
            return {"status": "ok", "size": [im.width, im.height]}
    except Exception as exc:  # corrupt / truncated
        return {"status": "corrupt", "error": f"{type(exc).__name__}: {exc}"}


def build_manifest(args) -> Dict:
    with open(args.split_manifest, "r", encoding="utf-8") as f:
        split_manifest = json.load(f)
    split_ids = sorted({int(i) for i in split_manifest[f"{args.split}_ids"]})
    split_sha = split_ids_sha256(split_manifest)
    expected_sha = getattr(args, "expected_split_ids_sha256", FROZEN_SPLIT_IDS_SHA256)
    if expected_sha and split_sha != expected_sha:
        raise SystemExit(f"[caption-gt] {args.split_manifest} is not the frozen E0 split "
                         f"(ids sha256 {split_sha[:16]}...). Refusing to build.")
    print(f"[caption-gt] frozen manifest : {args.split_manifest}")
    print(f"[caption-gt] split           : {args.split} ({len(split_ids):,} images)")

    objects_path = os.path.join(args.vg_root, "objects.json")
    use_objects = os.path.isfile(objects_path)
    if not use_objects and not args.allow_relationships_only:
        raise SystemExit(
            f"[caption-gt] {objects_path} not found. It is the primary human object "
            "ground truth; fetch it with `python download_vg.py --objects`.\n"
            "  (--allow-relationships-only builds from relationships.json alone, "
            "which under-counts objects and inflates hallucination rates.)")
    gt_source = ("objects.json+relationships.json" if use_objects
                 else "relationships.json only (INCOMPLETE)")
    print(f"[caption-gt] ground truth    : {gt_source}")
    print("[caption-gt] scanning annotations (a few minutes) ...")
    raw_gt = collect_ground_truth(args.vg_root, set(split_ids), use_objects=use_objects)

    gt = {iid: sorted(v["objects_json"] | v["relationships_json"]) for iid, v in raw_gt.items()}
    eligible = sorted(iid for iid in split_ids if len(gt.get(iid, [])) >= args.min_objects)
    order = list(eligible)
    random.Random(args.seed).shuffle(order)
    requested = order[:args.limit]

    image_sizes: Dict[int, List[int]] = {}
    image_data = os.path.join(args.vg_root, "image_data.json")
    if os.path.isfile(image_data):
        with open(image_data, "r", encoding="utf-8") as f:
            for m in json.load(f):
                image_sizes[int(m["image_id"])] = [int(m["width"]), int(m["height"])]

    image_dir = args.vg_image_dir or os.path.join(args.vg_root, "images")
    images: Dict[str, Dict] = {}
    usable: List[str] = []
    missing: List[str] = []
    corrupt: List[str] = []
    size_mismatch: List[str] = []
    for iid in requested:
        path = resolve_image_path(image_dir, iid)
        check = check_image(path)
        entry = {
            "objects": gt[iid],
            "gt_objects_json": sorted(raw_gt[iid]["objects_json"]),
            "gt_relationships_json": sorted(raw_gt[iid]["relationships_json"]),
            "file": path,
            "file_status": check["status"],
            "image_data_size": image_sizes.get(iid),
        }
        if check["status"] == "ok":
            entry["file_size"] = check["size"]
            usable.append(str(iid))
            if image_sizes.get(iid) and image_sizes[iid] != check["size"]:
                size_mismatch.append(str(iid))
        elif check["status"] == "missing":
            missing.append(str(iid))
        else:
            entry["file_error"] = check["error"]
            corrupt.append(str(iid))
        images[str(iid)] = entry

    status = "OK" if len(usable) >= args.min_final else "TOO_SMALL"
    label_hist = Counter(o for iid in usable for o in images[iid]["objects"])
    hashes = {}
    if not args.skip_hash:
        for name in ("objects.json", "relationships.json"):
            p = os.path.join(args.vg_root, name)
            if os.path.isfile(p):
                hashes[name] = sha256_file(p)

    return {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "caption experiment: baseline vs relation-grounded BLIP",
            "status": status,
            "ground_truth_source": gt_source,
            "ground_truth_is_human_annotation": True,
            "ground_truth_uses_model_predictions": False,
            "label_space": "COCO-80",
            "mention_mapper": MENTION_MAPPER_VERSION,
            "annotation_sha256": hashes,
            "split_manifest": args.split_manifest.replace("\\", "/"),
            "split_ids_sha256": split_sha,
            "split": args.split,
            "seed": args.seed,
            "min_objects": args.min_objects,
            "selection": ("sorted(eligible test images) -> random.Random(seed).shuffle "
                          "-> first `limit`; no back-filling of missing files"),
            "n_split_images": len(split_ids),
            "n_eligible": len(eligible),
            "requested": len(requested),
            "usable": len(usable),
            "missing": len(missing),
            "corrupt": len(corrupt),
            "final": len(usable),
            "min_final": args.min_final,
            "image_data_size_mismatches": size_mismatch,
            "usable_ids_sha256": hashlib.sha256(",".join(usable).encode()).hexdigest(),
            "mean_gt_objects_per_usable_image": round(
                sum(len(images[i]["objects"]) for i in usable) / max(len(usable), 1), 3),
            "label_histogram": dict(label_hist.most_common()),
        },
        "usable_ids": usable,
        "missing_ids": missing,
        "corrupt_ids": corrupt,
        "images": images,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the fixed caption-evaluation image set with human "
                    "Visual Genome object ground truth.")
    parser.add_argument("--vg-root", default=DEFAULT_VG_ROOT)
    parser.add_argument("--split-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Frozen split to draw from (default and required for "
                             "the experiment: test)")
    parser.add_argument("--vg-image-dir", default=None,
                        help="VG image directory (default <vg-root>/images)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--min-objects", type=int, default=2,
                        help="Eligible images have at least this many distinct GT "
                             "COCO objects (default 2: a relation needs two)")
    parser.add_argument("--min-final", type=int, default=MIN_FINAL_IMAGES,
                        help="Refuse (status TOO_SMALL, exit 2) below this many "
                             "usable images (default 100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output file if it already exists")
    parser.add_argument("--allow-relationships-only", action="store_true")
    parser.add_argument("--skip-hash", action="store_true",
                        help="Do not sha256 the annotation files (faster)")
    args = parser.parse_args()

    if os.path.isfile(args.output) and not args.force:
        raise SystemExit(f"[caption-gt] Refusing to overwrite existing {args.output}. "
                         "Pass --force or choose another --output.")

    manifest = build_manifest(args)
    m = manifest["meta"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("CAPTION EVALUATION SET")
    print("=" * 60)
    print(f"  eligible test images : {m['n_eligible']:,}")
    print(f"  requested images     : {m['requested']}")
    print(f"  usable images        : {m['usable']}")
    print(f"  missing images       : {m['missing']}")
    print(f"  corrupt images       : {m['corrupt']}")
    print(f"  final count          : {m['final']}")
    print(f"  mean GT objects      : {m['mean_gt_objects_per_usable_image']}")
    print(f"  usable ids sha256    : {m['usable_ids_sha256'][:16]}...")
    print(f"  wrote                : {args.output}")
    if m["status"] != "OK":
        print(f"\nSTOP: only {m['final']} usable images (< {args.min_final}). The "
              "evaluation is too small to support a strong conclusion. Fetch the "
              "missing images (python prepare_visual_genome.py --download) and "
              "rebuild with --force.")
        return 2
    print("\nSTATUS: OK")
    return 0


if __name__ == "__main__":
    from utils.console import configure_safe_stdio
    configure_safe_stdio()
    raise SystemExit(main())

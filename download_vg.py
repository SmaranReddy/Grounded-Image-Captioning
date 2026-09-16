"""
Download and prepare Visual Genome data for relation prediction.

Downloads:
    1. relationships.json   — relation triples (subject, predicate, object)
    2. image_data.json      — image metadata (URLs, sizes)
    3. VG images (optional) — required for visual-semantic features
    4. objects.json (optional, --objects) — every human-annotated object region;
       the object ground truth for the caption hallucination experiment

Usage:
    python download_vg.py                              # JSON only
    python download_vg.py --download-images            # JSON + images (108K, ~25 GB)
    python download_vg.py --download-images --max-images 5000   # subset for testing
    python download_vg.py --verify-images              # validate existing downloads
    python download_vg.py --objects                    # + objects.json (caption GT)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request
import zipfile
from typing import Dict, List, Optional

import requests
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "visual_genome")
os.makedirs(BASE_DIR, exist_ok=True)

FILES = {
    "relationships.json": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip",
    "image_data.json":    "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
}

# Not needed by the relation experiments, so not fetched by default. The caption
# experiment's hallucination ground truth (build_caption_eval_set.py) needs it:
# relationships.json only names the objects that take part in an annotated
# relation, so an object a caption correctly mentions would be scored as a
# hallucination whenever no annotator happened to relate it to anything.
OPTIONAL_FILES = {
    "objects.json": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip",
}

USER_AGENT = "Mozilla/5.0 (VG-Downloader; relation-prediction-project)"


def download_json(include_objects: bool = False) -> None:
    """Download relationships.json and image_data.json (and objects.json) if missing."""
    files = dict(FILES, **(OPTIONAL_FILES if include_objects else {}))
    for filename, url in files.items():
        dest = os.path.join(BASE_DIR, filename)
        if os.path.exists(dest):
            print(f"[skip] {filename} already exists.")
            continue

        zip_path = dest + ".zip"
        print(f"[download] {filename} ...")
        urllib.request.urlretrieve(url, zip_path)

        print(f"[extract] {filename} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(BASE_DIR)
        os.remove(zip_path)
        print(f"[done] {filename}")

    print("\n--- JSON Verification ---")
    for filename in FILES:
        path = os.path.join(BASE_DIR, filename)
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"[{status}] {path}")


# ---------------------------------------------------------------------------
# Image download helpers
# ---------------------------------------------------------------------------

def _is_valid_jpeg(path: str) -> bool:
    """Quick-check that a .jpg file has valid JPEG magic bytes and is openable."""
    try:
        with open(path, "rb") as f:
            header = f.read(2)
            if header != b"\xff\xd8":
                return False
        # Also verify PIL can open it (catches truncated files).
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _download_with_retry(url: str, dest: str, retries: int = 3, timeout: int = 60) -> bool:
    """
    Download a single file with retry and timeout.
    Downloads to a temporary file first, then renames on success.
    Returns True on success, False if all retries fail.
    """
    headers = {"User-Agent": USER_AGENT}
    tmp_dest = dest + ".tmp"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=timeout)
            resp.raise_for_status()
            with open(tmp_dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            os.replace(tmp_dest, dest)
            return True
        except requests.RequestException as exc:
            if os.path.exists(tmp_dest):
                os.remove(tmp_dest)
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  [retry {attempt}/{retries}] {os.path.basename(dest)} failed: {exc}, retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"  [fail] {os.path.basename(dest)}: {exc}")
    return False


def download_images(max_images: int = 0, verify: bool = False, seed: int = 42) -> None:
    """
    Download VG images from URLs in image_data.json.

    VG has ~108K images across two directories:
        VG_100K/   (images 2-100000)
        VG_100K_2/ (images 1, 100001-108077)

    Args:
        max_images: Max images to download (0 = all).
        verify:     If True, re-check existing files and re-download corrupt ones.
        seed:       Random seed for reproducible subset sampling.
    """
    img_dir = os.path.join(BASE_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    json_path = os.path.join(BASE_DIR, "image_data.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"image_data.json not found. Run without --download-images first."
        )

    with open(json_path) as f:
        img_meta: List[Dict] = json.load(f)

    total = len(img_meta)

    # Determine image selection.
    if max_images > 0 and max_images < total:
        rng = random.Random(seed)
        selected = rng.sample(img_meta, max_images)
        print(f"Randomly sampled {max_images} images (seed={seed}) from {total} total.")
    else:
        selected = img_meta
        print(f"All {total} images selected for download.")

    print(f"Downloading up to {len(selected)} images to {img_dir}/ ...")

    downloaded = 0
    skipped = 0
    failed = 0

    for m in tqdm(selected, desc="VG images", unit="img"):
        iid = m["image_id"]
        dest = os.path.join(img_dir, f"{iid}.jpg")
        url = m.get("url", "")
        if not url:
            print(f"\n  [skip] image_id={iid} has no URL")
            failed += 1
            continue

        # Handle existing files.
        if os.path.exists(dest):
            if verify and not _is_valid_jpeg(dest):
                tqdm.write(f"  [corrupt] {iid}.jpg, re-downloading ...")
                os.remove(dest)
            else:
                skipped += 1
                continue

        success = _download_with_retry(url, dest)
        if success:
            if _is_valid_jpeg(dest):
                downloaded += 1
            else:
                os.remove(dest)
                tqdm.write(f"  [corrupt] downloaded {iid}.jpg has invalid JPEG data")
                failed += 1
        else:
            failed += 1

    print(f"\n--- Image Download Complete ---")
    print(f"  Success: {downloaded}  Skipped: {skipped}  Failed: {failed}")
    total_in_dir = len(os.listdir(img_dir))
    print(f"  Total files in {img_dir}/: {total_in_dir}")

    if total_in_dir < len(selected):
        print(f"\n  NOTE: Only {total_in_dir}/{len(selected)} images available.")
        print(f"  Visual-semantic training will automatically skip missing images.")
        print(f"  To retry failed downloads: python download_vg.py --verify-images")


def verify_images() -> None:
    """
    Check all downloaded images for validity.
    Reports corrupt/missing files and provides summary.
    """
    img_dir = os.path.join(BASE_DIR, "images")
    if not os.path.isdir(img_dir):
        print(f"[ERROR] Image directory not found: {img_dir}")
        print(f"  Download images first: python download_vg.py --download-images")
        return

    json_path = os.path.join(BASE_DIR, "image_data.json")
    if not os.path.exists(json_path):
        print(f"[ERROR] image_data.json not found.")
        return

    with open(json_path) as f:
        img_meta: List[Dict] = json.load(f)

    all_ids = {str(m["image_id"]) for m in img_meta}
    on_disk = {f.replace(".jpg", "") for f in os.listdir(img_dir) if f.endswith(".jpg")}

    missing = all_ids - on_disk
    extra = on_disk - all_ids

    print(f"\n--- Image Verification ---")
    print(f"  Expected images:     {len(all_ids)}")
    print(f"  Images on disk:      {len(on_disk)}")
    print(f"  Missing images:      {len(missing)}")
    if missing:
        print(f"    (sample: {sorted(missing)[:5]})")
    if extra:
        print(f"  Unknown files:       {len(extra)}")
        print(f"    (sample: {sorted(extra)[:5]})")

    # Check for corrupt files.
    corrupt = []
    for fname in os.listdir(img_dir):
        fpath = os.path.join(img_dir, fname)
        if fname.endswith(".jpg") and not _is_valid_jpeg(fpath):
            corrupt.append(fname)

    if corrupt:
        print(f"  Corrupt images:      {len(corrupt)}")
        print(f"    (sample: {corrupt[:10]})")
    else:
        print(f"  Corrupt images:      0")

    if missing or corrupt:
        print(f"\n  Run with --download-images --verify to repair:")
        print(f"    python download_vg.py --download-images --verify")
    else:
        print(f"\n  All images verified OK.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Visual Genome data for relation prediction"
    )
    parser.add_argument(
        "--download-images", action="store_true",
        help="Download VG images (~25 GB for all 108K)",
    )
    parser.add_argument(
        "--max-images", type=int, default=0,
        help="Max images to download (0 = all). Randomly sampled for subset testing.",
    )
    parser.add_argument(
        "--verify-images", action="store_true",
        help="Check existing downloads for validity without downloading",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Re-check and re-download corrupt files during --download-images",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible subset sampling (default: 42)",
    )
    parser.add_argument(
        "--objects", action="store_true",
        help="Also download objects.json (~55 MB zipped): the human object "
             "annotations used as caption hallucination ground truth",
    )
    args = parser.parse_args()

    # Always ensure JSON files are present.
    download_json(include_objects=args.objects)

    if args.verify_images:
        verify_images()
        return

    if args.download_images:
        download_images(
            max_images=args.max_images,
            verify=args.verify,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

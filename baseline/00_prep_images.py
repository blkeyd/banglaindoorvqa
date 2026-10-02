#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_prep_images.py - resize every dataset image to 768 px on the short side.

Section 5 of the plan fixes the image size at 768 px short side for all three
models. Doing it once here means no run ever re-decodes a full-size phone photo,
which is a large speedup, and it removes the OOM trap in Section 11 (an
oversized image slipping into a run).

    images/  ->  images_768/

Only downscales. An image whose short side is already below 768 px is copied
through at its original size, never upscaled.

Typical use on Windows:

    python 00_prep_images.py --src images --dst images_768 --expect 4406

Re-running is cheap: files already in the output folder are skipped unless you
pass --force. Safe to kill and restart.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover
    raise SystemExit(
        "Pillow is not installed.\n"
        "Run:  python -m pip install --upgrade pillow"
    )

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# Worker. Must stay at module level so Windows can pickle it for the pool.
# --------------------------------------------------------------------------

def resize_one(job):
    """Resize one file. Returns (status, src_name, dst_name, w, h, message)."""
    src_str, dst_str, short_side, quality, force = job
    src = Path(src_str)
    dst = Path(dst_str)

    try:
        if dst.exists() and not force:
            with Image.open(dst) as im:
                w, h = im.size
            return ("skipped", src.name, dst.name, w, h, "")

        with Image.open(src) as im:
            # Apply the EXIF rotation flag, then drop EXIF entirely on save.
            # No-op if the prepared images already had their EXIF stripped.
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                im = im.convert("RGB")

            w, h = im.size
            if min(w, h) > short_side:
                # Set the short side to exactly short_side, scale the other.
                if w <= h:
                    nw = short_side
                    nh = max(1, round(h * short_side / w))
                else:
                    nh = short_side
                    nw = max(1, round(w * short_side / h))
                im = im.resize((nw, nh), Image.LANCZOS)
            else:
                nw, nh = w, h  # already small enough, do not upscale

            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="JPEG", quality=quality, optimize=True)

        return ("done", src.name, dst.name, nw, nh, "")

    except Exception as exc:
        return ("failed", src.name, dst.name, 0, 0, type(exc).__name__ + ": " + str(exc))


# --------------------------------------------------------------------------

def list_source_images(src_dir: Path):
    files = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return files


def check_duplicate_stems(files):
    """Two files with the same name but different extensions would collide,
    because every output is written as <stem>.jpg."""
    seen = {}
    clashes = []
    for p in files:
        stem = p.stem
        if stem in seen:
            clashes.append((seen[stem].name, p.name))
        else:
            seen[stem] = p
    return clashes


def verify_outputs(dst_dir: Path, short_side: int):
    """Open every output header and report the short-side distribution."""
    out_files = sorted(p for p in dst_dir.iterdir()
                       if p.is_file() and p.suffix.lower() == ".jpg")
    exact = 0
    smaller = 0
    oversized = []
    for p in out_files:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception as exc:
            oversized.append((p.name, "unreadable: " + str(exc)))
            continue
        s = min(w, h)
        if s == short_side:
            exact += 1
        elif s < short_side:
            smaller += 1
        else:
            oversized.append((p.name, str(w) + "x" + str(h)))
    return out_files, exact, smaller, oversized


def main():
    ap = argparse.ArgumentParser(description="Resize dataset images to a fixed short side.")
    ap.add_argument("--src", default="images", help="input folder (default: images)")
    ap.add_argument("--dst", default="images_768", help="output folder (default: images_768)")
    ap.add_argument("--short-side", type=int, default=768, help="target short side in px (default: 768)")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality (default: 92)")
    ap.add_argument("--workers", type=int, default=0, help="parallel processes (default: CPU count, max 8)")
    ap.add_argument("--expect", type=int, default=4406, help="expected file count (default: 4406)")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N files (smoke test)")
    ap.add_argument("--force", action="store_true", help="redo files that already exist in the output folder")
    ap.add_argument("--no-verify", action="store_true", help="skip the final re-open pass")
    args = ap.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.is_dir():
        raise SystemExit("Source folder not found: " + str(src_dir.resolve()))
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = list_source_images(src_dir)
    print("Source folder : " + str(src_dir.resolve()))
    print("Output folder : " + str(dst_dir.resolve()))
    print("Images found  : " + str(len(files)))

    clashes = check_duplicate_stems(files)
    if clashes:
        print("")
        print("ERROR: files share a name and would overwrite each other as .jpg:")
        for a, b in clashes[:20]:
            print("  " + a + "  <->  " + b)
        raise SystemExit(1)

    if args.limit:
        files = files[:args.limit]
        print("Limit         : processing first " + str(len(files)) + " only")

    if not files:
        raise SystemExit("Nothing to do: no image files in " + str(src_dir.resolve()))

    jobs = [
        (str(p), str(dst_dir / (p.stem + ".jpg")), args.short_side, args.quality, args.force)
        for p in files
    ]

    workers = args.workers or min(8, os.cpu_count() or 1)
    workers = max(1, workers)
    print("Workers       : " + str(workers))
    print("Target        : " + str(args.short_side) + " px short side, JPEG quality " + str(args.quality))
    print("")

    counts = {"done": 0, "skipped": 0, "failed": 0}
    failures = []
    started = time.time()
    total = len(jobs)

    def report(done_n):
        elapsed = time.time() - started
        rate = done_n / elapsed if elapsed > 0 else 0.0
        remaining = (total - done_n) / rate if rate > 0 else 0.0
        print("  " + str(done_n) + "/" + str(total)
              + "   " + ("%.1f" % rate) + " img/s"
              + "   eta " + ("%.1f" % (remaining / 60.0)) + " min", flush=True)

    if workers == 1:
        for i, job in enumerate(jobs, start=1):
            status, src_name, _dst, _w, _h, msg = resize_one(job)
            counts[status] += 1
            if status == "failed":
                failures.append((src_name, msg))
            if i % 200 == 0 or i == total:
                report(i)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(resize_one, job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), start=1):
                status, src_name, _dst, _w, _h, msg = fut.result()
                counts[status] += 1
                if status == "failed":
                    failures.append((src_name, msg))
                if i % 200 == 0 or i == total:
                    report(i)

    elapsed = time.time() - started
    print("")
    print("Resized       : " + str(counts["done"]))
    print("Already there : " + str(counts["skipped"]) + "   (re-run with --force to redo)")
    print("Failed        : " + str(counts["failed"]))
    print("Elapsed       : " + ("%.1f" % (elapsed / 60.0)) + " min")

    if failures:
        print("")
        print("Failures:")
        for name, msg in failures[:20]:
            print("  " + name + "  ->  " + msg)
        if len(failures) > 20:
            print("  ... and " + str(len(failures) - 20) + " more")

    # ---------------- verification ----------------
    print("")
    print("=" * 62)
    print("VERIFY")
    print("=" * 62)
    label_w = max(len(args.src), len(args.dst))
    print("Files in " + args.src.ljust(label_w) + " : " + str(len(list_source_images(src_dir))))

    out_files = sorted(p for p in dst_dir.iterdir()
                       if p.is_file() and p.suffix.lower() == ".jpg")
    print("Files in " + args.dst.ljust(label_w) + " : " + str(len(out_files)))

    if not args.no_verify:
        out_files, exact, smaller, oversized = verify_outputs(dst_dir, args.short_side)
        print("  short side == " + str(args.short_side) + " : " + str(exact))
        print("  short side <  " + str(args.short_side) + " : " + str(smaller) + "   (small originals, not upscaled)")
        print("  short side >  " + str(args.short_side) + " : " + str(len(oversized)))
        if oversized:
            print("  OVERSIZED FILES - these will cause the CUDA OOM in Section 11:")
            for name, size in oversized[:20]:
                print("    " + name + "  " + size)

    expected = args.expect
    actual = len(out_files)
    print("")
    if args.limit:
        print("RESULT: smoke test only (--limit " + str(args.limit) + "), count check skipped.")
    elif actual == expected:
        print("RESULT: PASS - " + str(actual) + " files, matches the expected " + str(expected) + ".")
    else:
        print("RESULT: MISMATCH - " + str(actual) + " files in " + args.dst
              + ", expected " + str(expected) + " (difference " + str(actual - expected) + ").")
        print("        Do not run 01_build_items.py until this is explained.")

    if counts["failed"]:
        return 1
    if not args.limit and actual != expected:
        return 1
    return 0


if __name__ == "__main__":
    # Bangla filenames or paths would otherwise crash printing on cmd.exe.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    # The guard is required on Windows for ProcessPoolExecutor. Do not remove it.
    raise SystemExit(main())

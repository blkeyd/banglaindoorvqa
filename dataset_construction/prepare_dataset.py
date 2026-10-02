#!/usr/bin/env python3
"""
BanglaIndoorVQA — dataset preparation (single pass).

For each image under INPUT_ROOT/H_xx/ it will:
  - bake in EXIF orientation (so the image is upright everywhere)
  - convert HEIC/HEIF/PNG -> JPG (RGB)
  - STRIP all metadata, including GPS location (privacy)
  - optionally downscale the long edge (detail-safe, tool-friendly)
  - assign a stable, opaque ID:  H01_0001.jpg, H01_0002.jpg, ...
  - write a row to manifest.csv (with empty columns for human screening)

Originals are NEVER modified. Output goes to a NEW folder (OUTPUT_ROOT).

Requirements (install locally):
    pip install pillow pillow-heif

Run from the folder that contains the `dataset/` directory:
    python prepare_dataset.py

Recommended: run once, then open manifest.csv and spot-check ~20 prepared
images (especially former HEIC and portrait shots) BEFORE proceeding.
"""

import csv
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

# ---- HEIC support (optional but recommended) -------------------------------
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except ImportError:
    HEIF_OK = False

# ---- CONFIG (edit these) ---------------------------------------------------
INPUT_ROOT    = Path("dataset")            # contains H_01 ... H_45
OUTPUT_ROOT   = Path("dataset_prepared")   # created fresh; prepared images go here
MANIFEST      = Path("manifest.csv")
LONG_EDGE_CAP = 2048    # cap longest side in px; set to None to keep full resolution
JPEG_QUALITY  = 95      # high quality; 90-95 is visually lossless for photos
VALID_EXT     = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
# ----------------------------------------------------------------------------

# Pillow resampling constant (works across versions)
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def house_id_from_folder(name: str) -> str | None:
    """'H_01' -> 'H01'; tolerates 'H1', 'h_01', etc. Returns None if no number."""
    m = re.search(r"(\d+)", name)
    if not m:
        return None
    return f"H{int(m.group(1)):02d}"


def main() -> None:
    if not HEIF_OK:
        print("WARNING: pillow-heif not installed -> HEIC/HEIF files will be SKIPPED.\n"
              "         Install with:  pip install pillow-heif\n", file=sys.stderr)

    if not INPUT_ROOT.exists():
        sys.exit(f"ERROR: input folder not found: {INPUT_ROOT.resolve()}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []

    house_dirs = sorted(d for d in INPUT_ROOT.iterdir() if d.is_dir())
    for hdir in house_dirs:
        hid = house_id_from_folder(hdir.name)
        if hid is None:
            print(f"Skipping non-house folder: {hdir.name}", file=sys.stderr)
            continue

        out_hdir = OUTPUT_ROOT / hid
        out_hdir.mkdir(parents=True, exist_ok=True)

        # deterministic order within a house -> stable IDs on re-run
        imgs = sorted(
            (p for p in hdir.iterdir() if p.suffix.lower() in VALID_EXT),
            key=lambda p: p.name.lower(),
        )

        counter = 0
        for src in imgs:
            img = None
            try:
                img = Image.open(src)
                orig_w, orig_h = img.size
                orig_fmt = (img.format or src.suffix.lstrip(".")).upper()

                # 1) bake EXIF orientation so the pixels are physically upright
                img = ImageOps.exif_transpose(img)

                # 2) RGB for JPEG (handles PNG alpha, HEIC, CMYK, palette)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # 3) optional downscale (preserves aspect ratio, never upscales)
                if LONG_EDGE_CAP:
                    img.thumbnail((LONG_EDGE_CAP, LONG_EDGE_CAP), RESAMPLE)

                counter += 1
                new_id = f"{hid}_{counter:04d}"
                out_path = out_hdir / f"{new_id}.jpg"

                # 4) save WITHOUT exif -> strips GPS and all other metadata
                img.save(out_path, "JPEG", quality=JPEG_QUALITY, optimize=True)

                final_w, final_h = img.size
                rows.append({
                    "image_id": new_id,
                    "house_id": hid,
                    "original_filename": src.name,
                    "original_relpath": str(src.relative_to(INPUT_ROOT)),
                    "original_format": orig_fmt,
                    "orig_width": orig_w,
                    "orig_height": orig_h,
                    "final_width": final_w,
                    "final_height": final_h,
                    "orientation": "portrait" if final_h >= final_w else "landscape",
                    # ---- columns to be filled by humans during screening ----
                    "room_type": "",       # kitchen | bedroom | living | bathroom | NA
                    "capture_type": "",    # overview | group | closeup
                    "include_flag": "",    # 1 = keep, 0 = exclude
                    "exclude_reason": "",  # corrupt|duplicate|privacy|non_room|blurry|other
                    "has_face": "",        # 1 if a face is visible (-> blur pass)
                    "notes": "",
                })
            except (UnidentifiedImageError, OSError) as e:
                skipped.append((str(src), str(e)))
                print(f"SKIP (unreadable): {src}  -> {e}", file=sys.stderr)
            finally:
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass

    fieldnames = [
        "image_id", "house_id", "original_filename", "original_relpath",
        "original_format", "orig_width", "orig_height", "final_width",
        "final_height", "orientation", "room_type", "capture_type",
        "include_flag", "exclude_reason", "has_face", "notes",
    ]
    with MANIFEST.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} images processed, {len(skipped)} skipped.")
    print(f"Prepared images: {OUTPUT_ROOT.resolve()}/")
    print(f"Manifest:        {MANIFEST.resolve()}")
    if skipped:
        print(f"\n{len(skipped)} files were unreadable (logged above). "
              "Investigate them before proceeding — do not silently lose images.")


if __name__ == "__main__":
    main()

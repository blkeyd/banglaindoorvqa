#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_items.py - flatten banglaindoorvqa.json into one shuffled items.jsonl.

Section 4 of the plan: every QA pair becomes one line, the whole list is
shuffled once with seed 42, and every worker reads this same file. Because the
order is shuffled, any point a run stops at is already a fair random sample of
the dataset, so a partial run is still reportable.

    banglaindoorvqa.json  ->  items.jsonl   (18,276 lines)

Output line format (Section 5):

    {"rank": 0, "qa_id": "H09_0137_q3", "image_id": "H09_0137", "house_id": "H09",
     "image_file": "H09_0137.jpg", "qa_type": "spatial", "group": "A",
     "question": "...", "gold": "..."}

Text is written through unchanged apart from Unicode NFC. The question must
reach the model exactly as annotated, and the reference answer keeps its daari.
Cleaning happens at scoring time, in normalize.py, and nowhere else.

Typical use on Windows:

    python 01_build_items.py --json banglaindoorvqa.json --images-dir images_768

If the script cannot work out the JSON layout, run it with --print-schema and
it will show you what it found.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------
# The eight question types (Section 3)
# --------------------------------------------------------------------------

GROUP_A_TYPES = ["scene_id", "object", "spatial", "count", "obj_name", "obj_property"]
GROUP_B_TYPES = ["obj_function", "assistive"]
ALL_TYPES = GROUP_A_TYPES + GROUP_B_TYPES

GROUP_OF = {t: "A" for t in GROUP_A_TYPES}
GROUP_OF.update({t: "B" for t in GROUP_B_TYPES})

# Schema v1 called the object-track third slot obj_hazard. Accept it, rename it.
TYPE_ALIASES = {"obj_hazard": "obj_property"}

EXPECTED_COUNTS = {
    "scene_id": 2529,
    "object": 2529,
    "spatial": 2529,
    "count": 2529,
    "obj_name": 1877,
    "obj_property": 1877,
    "obj_function": 1877,
    "assistive": 2529,
}
EXPECTED_QA_TOTAL = 18276
EXPECTED_IMAGES = 4406
EXPECTED_SCENE_RECORDS = 2529   # 5 QA each
EXPECTED_CLOSEUP_RECORDS = 1877  # 3 QA each

# --------------------------------------------------------------------------
# Key names the loader will look for. The DiB JSON is schema v2, but the field
# names are not pinned in the plan, so each of these is tried in order.
# --------------------------------------------------------------------------

RECORD_CONTAINER_KEYS = ["records", "data", "images", "annotations", "items", "dataset", "entries"]
QA_CONTAINER_KEYS = ["qa_pairs", "qa", "qas", "qa_list", "questions", "question_answers", "pairs"]

QA_TYPE_KEYS = ["qa_type", "type", "q_type", "question_type", "slot", "slot_type"]
QUESTION_KEYS = ["question", "question_bn", "q", "prompt", "text"]
ANSWER_KEYS = ["answer", "answer_bn", "a", "gold", "gt", "reference"]
QA_ID_KEYS = ["qa_id", "qaid", "qid", "id"]

IMAGE_ID_KEYS = ["image_id", "img_id", "imageId", "image", "id"]
HOUSE_ID_KEYS = ["house_id", "houseId", "house"]
IMAGE_FILE_KEYS = ["image_file", "file_name", "filename", "image_path", "img_path",
                   "file", "path", "image_filename", "image"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def nfc(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return unicodedata.normalize("NFC", value)


def first_present(d: dict, keys):
    """Return the first key in `keys` whose value in `d` is a non-empty scalar."""
    for k in keys:
        if k in d:
            v = d[k]
            if isinstance(v, (str, int, float)) and str(v).strip() != "":
                return str(v).strip()
    return None


def canon_type(raw_type: str):
    """Map a raw type string onto one of the eight canonical types."""
    if raw_type is None:
        return None
    t = str(raw_type).strip()
    t = TYPE_ALIASES.get(t, t)
    if t in GROUP_OF:
        return t
    low = t.lower().replace("-", "_").replace(" ", "_")
    low = TYPE_ALIASES.get(low, low)
    return low if low in GROUP_OF else None


def derive_house_id(image_id: str):
    if not image_id:
        return ""
    head = image_id.split("_")[0]
    return head if head else ""


def truncate(value, width=60):
    s = str(value).replace("\n", " ")
    return s if len(s) <= width else s[:width] + " ..."


# --------------------------------------------------------------------------
# Layout detection
# --------------------------------------------------------------------------

def find_records(raw):
    """Return (list_of_records, description)."""
    if isinstance(raw, list):
        return raw, "top-level list"
    if isinstance(raw, dict):
        for key in RECORD_CONTAINER_KEYS:
            value = raw.get(key)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value, "raw['" + key + "']"
        for key, value in raw.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value, "raw['" + key + "'] (auto-detected)"
    raise SystemExit(
        "Could not find the record list in the JSON.\n"
        "Re-run with --print-schema and send me the output."
    )


def qa_from_dict(qa: dict, slot: int, image_id: str):
    """Pull one QA pair out of a dict. Returns dict or None."""
    raw_type = first_present(qa, QA_TYPE_KEYS)
    qa_type = canon_type(raw_type)
    if qa_type is None:
        return None
    question = first_present(qa, QUESTION_KEYS)
    answer = first_present(qa, ANSWER_KEYS)
    qa_id = first_present(qa, QA_ID_KEYS) or (image_id + "_q" + str(slot))
    return {
        "qa_id": qa_id,
        "qa_type": qa_type,
        "question": question or "",
        "gold": answer or "",
        "raw_type": raw_type,
    }


def extract_qas(record: dict, image_id: str):
    """Yield QA dicts from one record. Handles four possible layouts."""
    # Layout 1: a list of QA dicts under a known key.
    for key in QA_CONTAINER_KEYS:
        value = record.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            out = []
            for slot, qa in enumerate(value, start=1):
                got = qa_from_dict(qa, slot, image_id)
                if got:
                    out.append(got)
            if out:
                return out, "qa list under '" + key + "'"

    # Layout 2: a dict keyed by question type.
    for key in QA_CONTAINER_KEYS:
        value = record.get(key)
        if isinstance(value, dict) and value:
            out = []
            for slot, (raw_type, qa) in enumerate(value.items(), start=1):
                qa_type = canon_type(raw_type)
                if qa_type is None:
                    continue
                if isinstance(qa, dict):
                    question = first_present(qa, QUESTION_KEYS) or ""
                    answer = first_present(qa, ANSWER_KEYS) or ""
                    qa_id = first_present(qa, QA_ID_KEYS) or (image_id + "_q" + str(slot))
                else:
                    continue
                out.append({"qa_id": qa_id, "qa_type": qa_type, "question": question,
                            "gold": answer, "raw_type": raw_type})
            if out:
                return out, "qa dict keyed by type under '" + key + "'"

    # Layout 3: flat q_<type> / a_<type> fields on the record itself
    # (this is what the Label Studio field names look like).
    out = []
    slot = 0
    for candidate in ALL_TYPES + list(TYPE_ALIASES.keys()):
        qkey = "q_" + candidate
        akey = "a_" + candidate
        if qkey in record or akey in record:
            slot += 1
            qa_type = canon_type(candidate)
            if qa_type is None:
                continue
            out.append({
                "qa_id": record.get(qa_type + "_qa_id") or (image_id + "_q" + str(slot)),
                "qa_type": qa_type,
                "question": nfc(record.get(qkey, "")),
                "gold": nfc(record.get(akey, "")),
                "raw_type": candidate,
            })
    if out:
        return out, "flat q_<type>/a_<type> fields on the record"

    # Layout 4: the record IS a single QA pair.
    single = qa_from_dict(record, 1, image_id)
    if single:
        return [single], "one QA pair per record"

    return [], "none"


def print_schema(raw, records, description):
    print("Container     : " + description)
    print("Records       : " + str(len(records)))
    print("")
    if isinstance(raw, dict):
        print("Top-level keys: " + ", ".join(list(raw.keys())[:30]))
        print("")
    if not records:
        return
    rec = records[0]
    print("First record keys:")
    for k, v in rec.items():
        kind = type(v).__name__
        if isinstance(v, list):
            extra = " (len " + str(len(v)) + ")"
            if v and isinstance(v[0], dict):
                extra += " first element keys: " + ", ".join(v[0].keys())
        elif isinstance(v, dict):
            extra = " keys: " + ", ".join(list(v.keys())[:12])
        else:
            extra = " = " + truncate(v)
        print("  " + k.ljust(20) + kind + extra)
    print("")
    print("First record, pretty printed (values truncated):")
    print(json.dumps(rec, ensure_ascii=False, indent=2)[:2000])


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Flatten and shuffle the dataset into items.jsonl")
    ap.add_argument("--json", default="banglaindoorvqa.json", help="release JSON (default: banglaindoorvqa.json)")
    ap.add_argument("--images-dir", default="images_768", help="folder checked for the referenced images")
    ap.add_argument("--out", default="items.jsonl", help="output file (default: items.jsonl)")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed (default: 42, do not change)")
    ap.add_argument("--image-ext", default=".jpg", help="extension written into image_file (default: .jpg)")
    ap.add_argument("--expect-qa", type=int, default=EXPECTED_QA_TOTAL, help="expected line count (default: 18276)")
    ap.add_argument("--expect-images", type=int, default=EXPECTED_IMAGES, help="expected image count (default: 4406)")
    ap.add_argument("--print-schema", action="store_true", help="show the JSON layout and exit")
    ap.add_argument("--skip-image-check", action="store_true", help="do not look for the image files on disk")
    args = ap.parse_args()

    json_path = Path(args.json)
    if not json_path.is_file():
        raise SystemExit("JSON not found: " + str(json_path.resolve()))

    print("Reading       : " + str(json_path.resolve()))
    with json_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    records, container_desc = find_records(raw)

    if args.print_schema:
        print_schema(raw, records, container_desc)
        return 0

    print("Container     : " + container_desc)
    print("Records       : " + str(len(records)))

    # ---------------- flatten ----------------
    items = []
    layouts = Counter()
    qa_per_record = Counter()
    problems = []
    remapped = 0

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            problems.append("record " + str(idx) + " is not an object")
            continue

        image_id = first_present(rec, IMAGE_ID_KEYS) or ""
        if not image_id:
            problems.append("record " + str(idx) + " has no image id")
            continue

        house_id = first_present(rec, HOUSE_ID_KEYS) or derive_house_id(image_id)

        raw_file = first_present(rec, IMAGE_FILE_KEYS) or (image_id + args.image_ext)
        stem = Path(str(raw_file).replace("\\", "/")).stem or image_id
        image_file = stem + args.image_ext

        qas, layout = extract_qas(rec, image_id)
        layouts[layout] += 1
        qa_per_record[len(qas)] += 1

        if not qas:
            problems.append("record " + str(idx) + " (" + image_id + ") produced no QA pairs")
            continue

        for qa in qas:
            if qa.get("raw_type") in TYPE_ALIASES:
                remapped += 1
            question = nfc(qa["question"])
            gold = nfc(qa["gold"])
            if not question:
                problems.append(qa["qa_id"] + ": empty question")
            if not gold:
                problems.append(qa["qa_id"] + ": empty answer")
            items.append({
                "qa_id": qa["qa_id"],
                "image_id": image_id,
                "house_id": house_id,
                "image_file": image_file,
                "qa_type": qa["qa_type"],
                "group": GROUP_OF[qa["qa_type"]],
                "question": question,
                "gold": gold,
            })

    if not items:
        raise SystemExit(
            "No QA pairs were extracted. Re-run with --print-schema and send me the output."
        )

    print("Layout        : " + ", ".join(k + " x" + str(v) for k, v in layouts.most_common()))
    if remapped:
        print("Renamed       : " + str(remapped) + " obj_hazard -> obj_property (schema v1 name)")

    # ---------------- deterministic order, then shuffle ----------------
    # Sorting first means items.jsonl is byte-identical on any machine and for
    # any JSON key order. Keep this if you ever rebuild the file, otherwise
    # seed 42 no longer reproduces the same sample.
    items.sort(key=lambda it: (it["image_id"], it["qa_id"], it["qa_type"]))
    random.Random(args.seed).shuffle(items)
    for rank, item in enumerate(items):
        item["rank"] = rank

    field_order = ["rank", "qa_id", "image_id", "house_id", "image_file",
                   "qa_type", "group", "question", "gold"]

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            ordered = {k: item[k] for k in field_order}
            fh.write(json.dumps(ordered, ensure_ascii=False) + "\n")

    # ---------------- checks ----------------
    line_count = 0
    with out_path.open("r", encoding="utf-8") as fh:
        for _ in fh:
            line_count += 1

    type_counts = Counter(it["qa_type"] for it in items)
    group_counts = Counter(it["group"] for it in items)
    image_ids = {it["image_id"] for it in items}
    house_ids = {it["house_id"] for it in items}

    dup_ids = [qid for qid, n in Counter(it["qa_id"] for it in items).items() if n > 1]

    missing_images = []
    wrong_ext = []
    if not args.skip_image_check:
        img_dir = Path(args.images_dir)
        if img_dir.is_dir():
            on_disk = {}
            for p in img_dir.iterdir():
                if p.is_file():
                    on_disk.setdefault(p.stem, p.name)
            for image_id in sorted(image_ids):
                stem = Path(image_id).stem
                actual = on_disk.get(stem)
                if actual is None:
                    missing_images.append(stem + args.image_ext)
                elif not actual.lower().endswith(args.image_ext.lower()):
                    wrong_ext.append(actual)
        else:
            print("NOTE          : images folder not found (" + str(img_dir.resolve()) + "), image check skipped")

    # ---------------- report ----------------
    print("")
    print("=" * 62)
    print("VERIFY")
    print("=" * 62)
    print("Output file   : " + str(out_path.resolve()))
    print("Shuffle seed  : " + str(args.seed))
    print("")

    print("Images (distinct image_id) : " + str(len(image_ids))
          + "   expected " + str(args.expect_images)
          + "   " + ("PASS" if len(image_ids) == args.expect_images else "MISMATCH"))
    print("Lines in " + args.out + (" " * max(1, 18 - len(args.out))) + ": " + str(line_count)
          + "   expected " + str(args.expect_qa)
          + "   " + ("PASS" if line_count == args.expect_qa else "MISMATCH"))
    print("Houses                     : " + str(len(house_ids)))
    print("")

    print("QA per record  : " + ", ".join(
        str(n) + " QA x " + str(c) + " records" for n, c in sorted(qa_per_record.items())))
    print("                 expected 5 QA x " + str(EXPECTED_SCENE_RECORDS)
          + " (scene) and 3 QA x " + str(EXPECTED_CLOSEUP_RECORDS) + " (close-up)")
    print("")

    print("By question type:")
    print("  " + "type".ljust(16) + "count".rjust(8) + "expected".rjust(10) + "   group")
    all_types_ok = True
    for t in ALL_TYPES:
        got = type_counts.get(t, 0)
        exp = EXPECTED_COUNTS[t]
        ok = got == exp
        all_types_ok = all_types_ok and ok
        print("  " + t.ljust(16) + str(got).rjust(8) + str(exp).rjust(10)
              + "   " + GROUP_OF[t] + ("" if ok else "   MISMATCH"))
    extra_types = set(type_counts) - set(ALL_TYPES)
    for t in sorted(extra_types):
        all_types_ok = False
        print("  " + t.ljust(16) + str(type_counts[t]).rjust(8) + "       -   UNKNOWN TYPE")
    print("")
    print("  Group A (scored for accuracy) : " + str(group_counts.get("A", 0)) + "   expected 13870")
    print("  Group B (chrF++ only)         : " + str(group_counts.get("B", 0)) + "   expected 4406")
    print("")

    if dup_ids:
        print("DUPLICATE qa_id: " + str(len(dup_ids)) + "  e.g. " + ", ".join(dup_ids[:5]))
    else:
        print("Duplicate qa_id: none")

    if not args.skip_image_check:
        print("Missing images : " + str(len(missing_images)))
        for name in missing_images[:10]:
            print("    " + name)
        if len(missing_images) > 10:
            print("    ... and " + str(len(missing_images) - 10) + " more")
        if wrong_ext:
            print("NOTE: " + str(len(wrong_ext)) + " files on disk are not " + args.image_ext
                  + " (e.g. " + wrong_ext[0] + "). items.jsonl still says " + args.image_ext
                  + " - run 00_prep_images.py and point 02_run_model.py at images_768/.")

    if problems:
        print("")
        print("Data problems  : " + str(len(problems)))
        for p in problems[:15]:
            print("    " + p)
        if len(problems) > 15:
            print("    ... and " + str(len(problems) - 15) + " more")

    # Shard preview, for Section 6.
    print("")
    print("Shard sizes (worker k of N processes line i where i % N == k):")
    for n in (2, 3, 4, 6, 8, 9):
        sizes = [line_count // n + (1 if k < line_count % n else 0) for k in range(n)]
        print("  N=" + str(n) + "  ->  " + str(min(sizes)) + " to " + str(max(sizes)) + " items per worker")

    print("")
    print("First two lines:")
    with out_path.open("r", encoding="utf-8") as fh:
        for _ in range(2):
            line = fh.readline().rstrip("\n")
            if line:
                print("  " + line)

    ok = (line_count == args.expect_qa
          and len(image_ids) == args.expect_images
          and all_types_ok
          and not dup_ids
          and not missing_images
          and not problems)

    print("")
    if ok:
        print("RESULT: PASS - " + str(line_count) + " lines, " + str(len(image_ids))
              + " images, all type counts match. items.jsonl is ready.")
        return 0
    print("RESULT: CHECK THE MISMATCHES ABOVE before running 02_run_model.py.")
    return 1


if __name__ == "__main__":
    # Bangla would otherwise crash printing on cmd.exe.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())

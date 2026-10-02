#!/usr/bin/env python3
"""
BanglaIndoorVQA v2 - compute every number the README / datasheet / paper needs.

Reads the full release file plus splits.csv and writes stats.json (+ prints a summary).
Counts are data-driven: whatever room_type / capture_type / qa type strings appear in
the data are what get counted, so the stats can't drift from the actual labels.

Usage:
  python make_stats.py --in banglaindoorvqa_all.json --splits splits.csv --out stats.json
"""

import argparse
import collections
import csv
import json
import re
import statistics
import unicodedata

ROOM_TYPES = ["kitchen", "bedroom", "living room", "bathroom", "NA"]
CAPTURE_TYPES = ["overview", "group", "closeup"]
SPLITS = ["train", "val", "test"]

WORD_RE = re.compile(r"\S+")


def desc(values):
    """min / median / mean / max for a list of numbers."""
    if not values:
        return {}
    return {"min": min(values), "median": statistics.median(values),
            "mean": round(statistics.mean(values), 2), "max": max(values),
            "n": len(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="banglaindoorvqa_all.json")
    ap.add_argument("--splits", default="splits.csv",
                    help="house_id,annotator_id,split,n_images - used when the release "
                         "file carries no `split` field")
    ap.add_argument("--out", default="stats.json")
    ap.add_argument("--top", type=int, default=30, help="how many lexicon entries to list")
    args = ap.parse_args()

    data = json.load(open(args.inp, encoding="utf-8"))

    # ---- attach split (from the record if present, else from splits.csv) ----
    house_split = {}
    try:
        with open(args.splits, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                house_split[row["house_id"].strip().upper()] = row["split"].strip()
    except FileNotFoundError:
        print(f"NOTE: {args.splits} not found; per-split tables come from the records only.")
    for r in data:
        if "split" not in r:
            r["_split"] = house_split.get(r["house_id"], "")
        else:
            r["_split"] = r["split"]

    houses = sorted({r["house_id"] for r in data})
    annotators = sorted({r["annotator_id"] for r in data})
    per_house = collections.Counter(r["house_id"] for r in data)

    rooms = collections.Counter(r["room_type"] for r in data)
    caps = collections.Counter(r["capture_type"] for r in data)
    splits = collections.Counter(r["_split"] for r in data)
    cross = collections.Counter((r["capture_type"], r["room_type"]) for r in data)
    per_annot = collections.Counter(r["annotator_id"] for r in data)
    dims = collections.Counter(f"{r.get('image_w')}x{r.get('image_h')}" for r in data)
    orient = collections.Counter(
        "portrait" if (r.get("image_h") or 0) > (r.get("image_w") or 0) else
        "landscape" if (r.get("image_w") or 0) > (r.get("image_h") or 0) else "square"
        for r in data)

    # ---- QA ----
    qa_by_type = collections.Counter()
    qa_per_image = []
    q_chars, a_chars, q_words, a_words = [], [], [], []
    len_by_type = collections.defaultdict(lambda: {"q": [], "a": []})
    qa_ids, questions, answers = [], collections.Counter(), collections.Counter()
    non_nfc = 0

    # ---- boxes ----
    box_ids = []
    boxes_per_record = []
    named_boxes = 0
    lexicon = collections.Counter()
    with_box_by_cap = collections.Counter()
    box_area_frac = []

    for r in data:
        qas = r.get("qa_pairs", [])
        qa_per_image.append(len(qas))
        for qa in qas:
            qa_by_type[qa["type"]] += 1
            qa_ids.append(qa.get("qa_id", ""))
            q, a = qa["question"], qa["answer"]
            questions[q] += 1
            answers[a] += 1
            q_chars.append(len(q)); a_chars.append(len(a))
            q_words.append(len(WORD_RE.findall(q))); a_words.append(len(WORD_RE.findall(a)))
            len_by_type[qa["type"]]["q"].append(len(WORD_RE.findall(q)))
            len_by_type[qa["type"]]["a"].append(len(WORD_RE.findall(a)))
            if unicodedata.normalize("NFC", q + a) != q + a:
                non_nfc += 1

        objs = r.get("objects", [])
        boxes_per_record.append(len(objs))
        if objs:
            with_box_by_cap[r["capture_type"]] += 1
        for o in objs:
            box_ids.append(o.get("object_id", ""))
            name = unicodedata.normalize("NFC", o.get("bangla_name", "").strip())
            if name:
                named_boxes += 1
                lexicon[name] += 1
            bb = o.get("bbox")
            if bb and r.get("image_w") and r.get("image_h"):
                box_area_frac.append(round(bb[2] * bb[3] / (r["image_w"] * r["image_h"]), 4))

    # ---- per split ----
    per_split = {}
    for s in SPLITS:
        recs = [r for r in data if r["_split"] == s]
        if not recs:
            continue
        per_split[s] = {
            "images": len(recs),
            "houses": len({r["house_id"] for r in recs}),
            "qa_pairs": sum(len(r.get("qa_pairs", [])) for r in recs),
            "boxes": sum(len(r.get("objects", [])) for r in recs),
            "room_type": dict(collections.Counter(r["room_type"] for r in recs)),
            "capture_type": dict(collections.Counter(r["capture_type"] for r in recs)),
            "annotator_id": dict(collections.Counter(r["annotator_id"] for r in recs)),
            "qa_by_type": dict(collections.Counter(
                qa["type"] for r in recs for qa in r.get("qa_pairs", []))),
        }

    total_qa = sum(qa_by_type.values())
    total_boxes = sum(boxes_per_record)

    stats = {
        "images_total": len(data),
        "houses_total": len(houses),
        "annotators_total": len(annotators),
        "annotator_ids": annotators,
        "images_per_house": desc(list(per_house.values())),
        "images_per_house_detail": dict(sorted(per_house.items())),
        "images_per_annotator": dict(sorted(per_annot.items())),
        "room_type_counts": dict(rooms),
        "capture_type_counts": dict(caps),
        "capture_x_room": {f"{c}|{r}": n for (c, r), n in sorted(cross.items())},
        "image_size_counts": dict(dims),
        "orientation_counts": dict(orient),
        "split_counts": dict(splits),
        "per_split": per_split,

        "qa_pairs_total": total_qa,
        "qa_pairs_by_type": dict(qa_by_type),
        "qa_pairs_per_image": desc(qa_per_image),
        "question_length_chars": desc(q_chars),
        "answer_length_chars": desc(a_chars),
        "question_length_words": desc(q_words),
        "answer_length_words": desc(a_words),
        "length_words_by_qa_type": {t: {"question": desc(v["q"]), "answer": desc(v["a"])}
                                    for t, v in sorted(len_by_type.items())},
        "unique_questions": len(questions),
        "unique_answers": len(answers),
        "question_reuse_rate": round(1 - len(questions) / max(1, total_qa), 4),

        "bounding_boxes_total": total_boxes,
        "bounding_boxes_named": named_boxes,
        "records_with_box_by_capture_type": dict(with_box_by_cap),
        "boxes_per_record": desc(boxes_per_record),
        "box_area_fraction": desc(box_area_frac),

        "object_lexicon_size": len(lexicon),
        "object_lexicon_ge20": sum(1 for v in lexicon.values() if v >= 20),
        "object_lexicon_ge10": sum(1 for v in lexicon.values() if v >= 10),
        "object_lexicon_ge5": sum(1 for v in lexicon.values() if v >= 5),
        "object_lexicon_singletons": sum(1 for v in lexicon.values() if v == 1),
        "object_lexicon_top": lexicon.most_common(args.top),

        "integrity": {
            "duplicate_image_ids": len(data) - len({r["image_id"] for r in data}),
            "duplicate_qa_ids": len(qa_ids) - len(set(qa_ids)),
            "duplicate_object_ids": len(box_ids) - len(set(box_ids)),
            "blank_qa_ids": sum(1 for i in qa_ids if not i),
            "blank_object_ids": sum(1 for i in box_ids if not i),
            "qa_fields_not_nfc": non_nfc,
            "records_missing_image_dims": sum(1 for r in data
                                              if not (r.get("image_w") and r.get("image_h"))),
        },
    }
    json.dump(stats, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- summary ----
    print(f"images: {len(data)}  |  houses: {len(houses)}  |  annotators: {len(annotators)}")
    print(f"QA pairs: {total_qa}  |  boxes: {total_boxes} ({named_boxes} named)")
    print(f"images/house: {stats['images_per_house']}")
    print(f"room_type: {dict(rooms)}")
    print(f"capture_type: {dict(caps)}")
    print(f"splits: {dict(splits)}")
    print(f"QA by type: {dict(qa_by_type)}")
    print(f"object lexicon: {len(lexicon)} unique | >=20x: {stats['object_lexicon_ge20']} "
          f"| >=10x: {stats['object_lexicon_ge10']} | singletons: "
          f"{stats['object_lexicon_singletons']}")
    print(f"question words (median): {stats['question_length_words'].get('median')}  |  "
          f"answer words (median): {stats['answer_length_words'].get('median')}")
    print(f"integrity: {stats['integrity']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

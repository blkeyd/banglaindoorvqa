#!/usr/bin/env python3
"""
BanglaIndoorVQA v2 - merge & normalize Label Studio exports into the release JSON.

Changes vs v1:
  * schema v2: annotator_id (A1..A4), record-level image_w/image_h, stable qa_id and
    object_id, no `exclude`, no original_filename, no original_format.
  * Unicode NFC normalization + whitespace collapse on ALL Bangla text. This is not
    cosmetic: the raw exports contain the SAME character encoded two ways
    (য় as U+09DF vs U+09AF+U+09BC, ড় as U+09DC vs U+09A1+U+09BC), which silently
    breaks exact-match scoring and object-frequency counts.
    NOTE: ড়/ঢ়/য় are Unicode composition exclusions, so NFC produces the DECOMPOSED
    form (base + nukta U+09BC). Your lexicon and eval code must use the same form.
  * image_w/image_h taken from manifest.csv (final_width/final_height), cross-checked
    against the Label Studio region dims.
  * expanded structural validation + a normalization audit in the report.

Pipeline:
  1. Read one or more Label Studio JSON exports (per-house).
  2. Recover image_id + house_id from the filename (strips the LS upload hash).
  3. Stamp annotator_id from house_assignment.csv (house_id -> annotator_id).
  4. Regroup the flat `result` array into objects[] + qa_pairs[], converting bbox
     percentages to pixel [x, y, w, h].
  5. Normalize all text, assign deterministic qa_id / object_id.
  6. Run completeness (manifest vs annotated) + structural validation.

Usage:
  python merge_annotations.py \
      --exports JSON/*.json \
      --assignment house_assignment.csv \
      --manifest manifest.csv \
      --out banglaindoorvqa.json
"""

import argparse
import collections
import csv
import glob
import json
import re
import sys
import unicodedata
from pathlib import Path

SCHEMA_VERSION = "2.0"

# ---- closed vocabularies ----
ROOM_TYPES = ["kitchen", "bedroom", "living room", "bathroom", "NA"]
CAPTURE_TYPES = ["overview", "group", "closeup"]

# The third object-track slot holds a hazard question where a real hazard exists and a
# salient attribute (colour / material / state) otherwise, so the type string is
# obj_property rather than obj_hazard. Single point of change.
O3_TYPE = "obj_property"

# ---- mapping from Label Studio field names to our schema ----
# The slot index fixes qa_id: scene q1..q5, closeup q1..q3, regardless of which are
# present. That keeps qa_id stable even if a field is blank in some record.
SCENE_QA = [  # (slot, type, question_field, answer_field)
    (1, "scene_id",  "q_scene",     "a_scene"),
    (2, "object",    "q_object",    "a_object"),
    (3, "spatial",   "q_spatial",   "a_spatial"),
    (4, "count",     "q_count",     "a_count"),
    (5, "assistive", "q_assistive", "a_assistive"),
]
OBJECT_QA = [
    (1, "obj_name",     "q_obj_name",     "a_obj_name"),
    (2, "obj_function", "q_obj_function", "a_obj_function"),
    (3, O3_TYPE,        "q_obj_hazard",   "a_obj_hazard"),
]

IMAGE_ID_RE = re.compile(r"(H\d+_\d+)", re.IGNORECASE)
HOUSE_RE = re.compile(r"(H\d+)", re.IGNORECASE)
LATIN_RE = re.compile(r"[A-Za-z]")
ASCII_DIGIT_RE = re.compile(r"[0-9]")
STRIP_CHARS = ("\u200b", "\ufeff", "\u00a0")  # zero-width space, BOM, nbsp - removed

# ZWNJ / ZWJ only do anything in Bangla immediately after hasant (U+09CD), where they
# control conjunct formation. Elsewhere they are invisible keyboard artifacts that make
# the same visible word compare unequal. Remove the inert ones, keep the functional ones.
INERT_JOINER_RE = re.compile(r"(?<!\u09cd)[\u200c\u200d]")
FUNCTIONAL_JOINER_RE = re.compile(r"(?<=\u09cd)[\u200c\u200d]")

# audit counters
NORM = collections.Counter()
CLIP = collections.Counter()
CLIP_DETAIL = []
DROPPED_BOXES = []
ROOMFIX = []

# A box that survives clipping with almost nothing left was not an annotation of an
# object - it is a mis-drag started at the canvas edge. Such boxes are removed.
MIN_BOX_AREA_PCT = 15.0   # keep >= this % of the drawn area
MIN_BOX_SIDE_PX = 16      # and be at least this many px on both sides


def norm_text(v):
    """Label Studio TextArea value is {'text': [...]}. Join, NFC-normalize, collapse ws."""
    if not v:
        return ""
    if isinstance(v, list):
        raw = " ".join(str(x) for x in v if str(x).strip() != "")
    else:
        raw = str(v)
    if not raw:
        return ""

    NORM["fields_seen"] += 1
    s = raw

    for ch in STRIP_CHARS:
        if ch in s:
            NORM["stripped_invisible"] += 1
            s = s.replace(ch, "")
    n_func = len(FUNCTIONAL_JOINER_RE.findall(s))
    stripped = INERT_JOINER_RE.sub("", s)
    if stripped != s:
        NORM["fields_with_inert_joiner"] += 1
        NORM["inert_joiners_removed"] += len(s) - len(stripped)
        s = stripped
    if n_func:
        NORM["functional_joiners_kept"] += n_func

    nfc = unicodedata.normalize("NFC", s)
    if nfc != s:
        NORM["nfc_changed"] += 1
    s = nfc

    ws = re.sub(r"\s+", " ", s).strip()
    if ws != s:
        NORM["whitespace_changed"] += 1
    s = ws

    if s != raw:
        NORM["fields_changed"] += 1
    return s


def recover_image_id(task):
    """Get H##_#### from file_upload (preferred) or data.image."""
    candidates = []
    if task.get("file_upload"):
        candidates.append(task["file_upload"])
    img = (task.get("data") or {}).get("image", "")
    if img:
        candidates.append(img)
    for c in candidates:
        m = IMAGE_ID_RE.search(c)
        if m:
            return m.group(1).upper()
    return None


def pick_annotation(task):
    """Best annotation dict, or None if the task was skipped/empty."""
    anns = task.get("annotations") or []
    real = [a for a in anns if not a.get("was_cancelled") and a.get("result")]
    return real[0] if real else None


def load_assignment(path):
    """house_id -> annotator_id. Requires an annotator_id column (A1..A4)."""
    m = {}
    if not path:
        return m
    p = Path(path)
    if not p.exists():
        print(f"WARNING: {path} not found; annotator_id will be blank.", file=sys.stderr)
        return m
    with p.open(encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        if "annotator_id" not in (rdr.fieldnames or []):
            sys.exit(f"ERROR: {path} has no 'annotator_id' column. Found: {rdr.fieldnames}")
        for row in rdr:
            if row.get("house_id"):
                m[row["house_id"].strip().upper()] = (row.get("annotator_id") or "").strip()
    return m


def load_manifest(path):
    rows = {}
    if not path:
        return rows
    p = Path(path)
    if not p.exists():
        print(f"WARNING: manifest {path} not found; image_w/image_h fall back to "
              f"Label Studio region dims.", file=sys.stderr)
        return rows
    with p.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            iid = (r.get("image_id") or "").strip().upper()
            if iid:
                rows[iid] = r
    return rows


def as_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def box_geometry(v, ow, oh, clip=True):
    """Label Studio rectangle x,y,width,height are percentages (0-100) of the image.

    LS lets an annotator drag a rectangle past the canvas edge, so x+width can exceed
    100 (or x can go negative). Those coordinates describe area that does not exist in
    the image: naive crops silently return a smaller array and IoU maths goes wrong.
    We clip the box to the frame, which discards only the non-existent part, and record
    how much was clipped so it can be reported in the datasheet.

    Returns (bbox_px, bbox_pct, overflow_px, area_lost_pct).
    """
    x, y, w, h = float(v["x"]), float(v["y"]), float(v["width"]), float(v["height"])

    over_px = 0.0
    if ow and oh:
        over_px = max(0.0,
                      -x / 100.0 * ow,
                      -y / 100.0 * oh,
                      (x + w - 100.0) / 100.0 * ow,
                      (y + h - 100.0) / 100.0 * oh)
    out_of_frame = (x < 0 or y < 0 or x + w > 100.0 or y + h > 100.0)

    lost = 0.0
    if out_of_frame:
        CLIP["boxes_out_of_frame"] += 1
        if clip:
            nx, ny = max(0.0, x), max(0.0, y)
            nw, nh = max(0.0, min(x + w, 100.0) - nx), max(0.0, min(y + h, 100.0) - ny)
            area0 = w * h
            lost = (1.0 - (nw * nh) / area0) * 100.0 if area0 > 0 else 100.0
            x, y, w, h = nx, ny, nw, nh
            CLIP["boxes_clipped"] += 1
            if nw <= 0 or nh <= 0:
                CLIP["boxes_empty_after_clip"] += 1

    pct = [round(x, 3), round(y, 3), round(w, 3), round(h, 3)]
    px = None
    if ow and oh:
        px = [round(x / 100.0 * ow), round(y / 100.0 * oh),
              round(w / 100.0 * ow), round(h / 100.0 * oh)]
        if clip:  # rounding can still push x+w one pixel over; clamp in pixel space too
            px[0] = max(0, min(px[0], ow))
            px[1] = max(0, min(px[1], oh))
            px[2] = max(0, min(px[2], ow - px[0]))
            px[3] = max(0, min(px[3], oh - px[1]))
    return px, pct, round(over_px, 1), round(lost, 2)


def normalize_task(task, assignment, manifest, clip_bbox=True,
                   force_closeup_na=True):
    image_id = recover_image_id(task)
    if image_id is None:
        return None, "no_image_id", None

    hm = HOUSE_RE.search(image_id)
    house_id = hm.group(1).upper() if hm else ""

    ann = pick_annotation(task)
    if ann is None:
        return None, "skipped_or_empty", image_id
    result = ann.get("result", [])

    # ---- index the flat result array ----
    choices, textareas, boxes, box_names = {}, {}, {}, {}
    for r in result:
        fn, rid, typ, val = r.get("from_name"), r.get("id"), r.get("type"), r.get("value", {})
        if typ == "choices":
            choices[fn] = (val.get("choices") or [None])[0]
        elif typ == "rectanglelabels":
            boxes[rid] = {"v": val, "ow": r.get("original_width"), "oh": r.get("original_height")}
        elif typ == "textarea":
            if fn == "bbox_name":
                box_names[rid] = norm_text(val.get("text"))
            else:
                textareas[fn] = norm_text(val.get("text"))

    # ---- image dimensions: manifest is authoritative, LS regions are the cross-check ----
    man = manifest.get(image_id, {})
    img_w = as_int(man.get("final_width"))
    img_h = as_int(man.get("final_height"))
    ls_dims = {(b["ow"], b["oh"]) for b in boxes.values() if b["ow"] and b["oh"]}
    dims_conflict = ""
    if img_w and img_h and ls_dims and (img_w, img_h) not in ls_dims:
        dims_conflict = f"manifest={img_w}x{img_h} ls={sorted(ls_dims)}"
    if not (img_w and img_h) and ls_dims:
        img_w, img_h = sorted(ls_dims)[0]

    rec = {
        "image_id": image_id,
        "house_id": house_id,
        "annotator_id": assignment.get(house_id, ""),
        "image_file": f"{house_id}/{image_id}.jpg",
        "image_w": img_w,
        "image_h": img_h,
        "capture_type": choices.get("capture_type", "") or "",
        "room_type": choices.get("room_type", "") or "",
        "_room_raw": choices.get("room_type", "") or "",
        "objects": [],
        "qa_pairs": [],
        # internal only - popped before writing
        "_exclude": choices.get("exclude", "") or "",
        "_dims_conflict": dims_conflict,
    }

    # The object track carries no room label by convention: on a close-up the room is
    # usually not determinable from the image, so any room value here is unverifiable.
    if force_closeup_na and rec["capture_type"] == "closeup" and rec["room_type"] != "NA":
        ROOMFIX.append({"image_id": image_id, "was": rec["room_type"]})
        rec["room_type"] = "NA"

    # ---- objects: sorted top-to-bottom, left-to-right so object_id is deterministic ----
    tmp = []
    for rid, b in boxes.items():
        v, ow, oh = b["v"], b["ow"], b["oh"]
        px, pct, over_px, lost = box_geometry(v, ow, oh, clip=clip_bbox)
        tmp.append({
            "_sort": (round(v["y"], 4), round(v["x"], 4), rid),
            "_over_px": over_px,
            "_lost": lost,
            "bangla_name": box_names.get(rid, ""),
            "bbox": px,
            "bbox_pct": pct,
        })
    tmp.sort(key=lambda o: o["_sort"])
    for i, o in enumerate(tmp, start=1):
        o.pop("_sort")
        over_px, lost = o.pop("_over_px"), o.pop("_lost")
        oid = f"{image_id}_b{i}"
        if over_px > 0:
            CLIP_DETAIL.append({"image_id": image_id, "object_id": oid,
                                "overflow_px": over_px, "area_lost_pct": lost})
        bb = o.get("bbox")
        kept_pct = 100.0 - lost
        degenerate = bool(bb) and (kept_pct < MIN_BOX_AREA_PCT
                                   or bb[2] < MIN_BOX_SIDE_PX or bb[3] < MIN_BOX_SIDE_PX)
        if degenerate:
            CLIP["boxes_dropped_degenerate"] += 1
            DROPPED_BOXES.append({"image_id": image_id, "object_id": oid, "bbox": bb,
                                  "bangla_name": o.get("bangla_name", ""),
                                  "kept_area_pct": round(kept_pct, 2)})
            continue
        rec["objects"].append({"object_id": oid, **o})

    # ---- qa pairs, by track; slot index fixes qa_id ----
    qa_set = OBJECT_QA if rec["capture_type"] == "closeup" else SCENE_QA
    for slot, qtype, qf, af in qa_set:
        q, a = textareas.get(qf, ""), textareas.get(af, "")
        if q or a:
            rec["qa_pairs"].append({
                "qa_id": f"{image_id}_q{slot}",
                "type": qtype,
                "question": q,
                "answer": a,
            })

    return rec, None, image_id


def validate(rec):
    """List of structural problem strings for one record (empty list = clean)."""
    problems = []
    cap, room = rec["capture_type"], rec["room_type"]

    if cap not in CAPTURE_TYPES:
        problems.append(f"bad_capture_type:{cap!r}")
    if room not in ROOM_TYPES:
        problems.append(f"bad_room_type:{room!r}")
    if not rec["annotator_id"]:
        problems.append("missing_annotator_id")
    if not (rec["image_w"] and rec["image_h"]):
        problems.append("missing_image_dims")
    if rec["_dims_conflict"]:
        problems.append(f"dims_conflict({rec['_dims_conflict']})")

    is_scene = cap in ("overview", "group")
    expected_qa = 5 if is_scene else 3
    if len(rec["qa_pairs"]) != expected_qa:
        problems.append(f"qa_count={len(rec['qa_pairs'])}(expected {expected_qa})")

    seen_types = collections.Counter(qa["type"] for qa in rec["qa_pairs"])
    for t, n in seen_types.items():
        if n > 1:
            problems.append(f"duplicate_qa_type:{t}")

    for qa in rec["qa_pairs"]:
        if not qa["question"].strip():
            problems.append(f"empty_question:{qa['type']}")
        if not qa["answer"].strip():
            problems.append(f"empty_answer:{qa['type']}")
        blob = qa["question"] + qa["answer"]
        if LATIN_RE.search(blob):
            problems.append(f"latin_chars:{qa['type']}")
        if ASCII_DIGIT_RE.search(blob):
            problems.append(f"ascii_digits:{qa['type']}")

    # scene track requires exactly one box; close-up boxes are optional by design
    if is_scene and len(rec["objects"]) != 1:
        problems.append(f"scene_boxes={len(rec['objects'])}(expected 1)")

    for obj in rec["objects"]:
        if not obj.get("bangla_name", "").strip():
            problems.append("missing_bbox_name")
        if LATIN_RE.search(obj.get("bangla_name", "")):
            problems.append("latin_chars_in_bbox_name")
        bb = obj.get("bbox")
        if bb is None:
            problems.append("bbox_missing_dims")
        elif rec["image_w"] and rec["image_h"]:
            x, y, w, h = bb
            if w <= 0 or h <= 0:
                problems.append(f"bbox_nonpositive:{obj['object_id']}")
            elif x < -1 or y < -1 or x + w > rec["image_w"] + 1 or y + h > rec["image_h"] + 1:
                problems.append(f"bbox_out_of_bounds:{obj['object_id']}")
    return problems


def strip_internal(rec):
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def main():
    global MIN_BOX_AREA_PCT, MIN_BOX_SIDE_PX
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", nargs="+", required=True,
                    help="Label Studio JSON exports (globs like JSON/*.json are fine)")
    ap.add_argument("--assignment", default=None,
                    help="house_assignment.csv (house_id,annotator_name,annotator_id)")
    ap.add_argument("--manifest", default=None,
                    help="manifest.csv - needed for image_w/image_h + completeness check")
    ap.add_argument("--out", default="banglaindoorvqa.json")
    ap.add_argument("--keep-closeup-room", action="store_true",
                    help="do NOT force room_type=NA on the object track (default: force)")
    ap.add_argument("--min-box-area-pct", type=float, default=MIN_BOX_AREA_PCT,
                    help="drop a clipped box keeping less than this %% of its drawn area")
    ap.add_argument("--min-box-side-px", type=int, default=MIN_BOX_SIDE_PX,
                    help="drop a clipped box thinner than this on either side")
    ap.add_argument("--no-clip-bbox", action="store_true",
                    help="do NOT clip boxes that extend past the image border "
                         "(default: clip, and report what was clipped)")
    ap.add_argument("--keep-excluded", action="store_true",
                    help="keep privacy/exclude records in the main dataset file")
    args = ap.parse_args()

    MIN_BOX_AREA_PCT, MIN_BOX_SIDE_PX = args.min_box_area_pct, args.min_box_side_px

    export_files = []
    for pat in args.exports:
        hits = sorted(glob.glob(pat))
        export_files.extend(hits if hits else [pat])

    assignment = load_assignment(args.assignment)
    manifest = load_manifest(args.manifest)

    records, excluded = {}, {}
    skipped_ids, issues = [], []
    for ex in export_files:
        try:
            data = json.load(open(ex, encoding="utf-8"))
        except Exception as e:
            print(f"ERROR reading {ex}: {e}", file=sys.stderr)
            continue
        for task in data:
            rec, err, image_id = normalize_task(
                task, assignment, manifest,
                clip_bbox=not args.no_clip_bbox,
                force_closeup_na=not args.keep_closeup_room)
            if err:
                issues.append((ex, task.get("id"), err))
                if err == "skipped_or_empty" and image_id:
                    skipped_ids.append(image_id)
                continue
            iid = rec["image_id"]
            if iid in records or iid in excluded:
                issues.append((ex, iid, "duplicate_image_id_across_exports"))
            if rec["_exclude"] and not args.keep_excluded:
                excluded[iid] = rec
            else:
                records[iid] = rec

    # A scene-track record must carry exactly one box. If its only box was dropped as
    # degenerate the record cannot be repaired here, so it leaves the release entirely.
    # Close-ups keep their record - boxes are optional on that track by design.
    degenerate_records = {}
    for iid in [i for i, r in records.items()
                if r["capture_type"] in ("overview", "group") and not r["objects"]]:
        degenerate_records[iid] = records.pop(iid)

    kept = sorted(records.values(), key=lambda r: r["image_id"])

    # ---- validation runs on the internal form, output is written stripped ----
    val_problems = {r["image_id"]: p for r in kept if (p := validate(r))}
    out_list = [strip_internal(r) for r in kept]
    json.dump(out_list, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- reports ----
    print(f"Merged {len(out_list)} records -> {args.out}   (schema v{SCHEMA_VERSION})")
    n_scene = sum(1 for r in out_list if r["capture_type"] in ("overview", "group"))
    n_close = sum(1 for r in out_list if r["capture_type"] == "closeup")
    n_qa = sum(len(r["qa_pairs"]) for r in out_list)
    n_box = sum(len(r["objects"]) for r in out_list)
    box_by_cap = collections.Counter(r["capture_type"] for r in out_list if r["objects"])
    print(f"  scene-track: {n_scene}   object-track(closeup): {n_close}")
    print(f"  qa_pairs: {n_qa}   boxes: {n_box}")
    print(f"  records carrying >=1 box, by capture_type: {dict(box_by_cap)}")

    print("\nText normalization audit:")
    print(f"  text fields processed : {NORM['fields_seen']}")
    print(f"  changed by NFC        : {NORM['nfc_changed']}")
    print(f"  changed by whitespace : {NORM['whitespace_changed']}")
    print(f"  invisible chars removed: {NORM['stripped_invisible']}")
    print(f"  inert ZWNJ/ZWJ removed: {NORM['inert_joiners_removed']} "
          f"in {NORM['fields_with_inert_joiner']} field(s)")
    print(f"  functional ZWNJ/ZWJ kept (after hasant): {NORM['functional_joiners_kept']}")
    print(f"  total fields altered  : {NORM['fields_changed']}")

    print("\nBounding box frame audit:")
    print(f"  boxes drawn past the image border : {CLIP['boxes_out_of_frame']}")
    print(f"  boxes clipped to the frame        : {CLIP['boxes_clipped']}"
          f"{'  (CLIPPING DISABLED)' if args.no_clip_bbox else ''}")
    if CLIP_DETAIL:
        ov = sorted(d["overflow_px"] for d in CLIP_DETAIL)
        ls = sorted(d["area_lost_pct"] for d in CLIP_DETAIL)
        print(f"  overflow px  min/median/max       : {ov[0]} / {ov[len(ov)//2]} / {ov[-1]}")
        print(f"  box area lost % min/median/max    : {ls[0]} / {ls[len(ls)//2]} / {ls[-1]}")
        worst = sorted(CLIP_DETAIL, key=lambda d: -d["overflow_px"])[:10]
        print("  largest overflows:")
        for d in worst:
            print(f"    {d['object_id']:20} overflow={d['overflow_px']}px  "
                  f"area lost={d['area_lost_pct']}%")
    if CLIP["boxes_dropped_degenerate"]:
        print(f"  boxes dropped as degenerate       : {CLIP['boxes_dropped_degenerate']} "
              f"(kept <{MIN_BOX_AREA_PCT}% of drawn area or <{MIN_BOX_SIDE_PX}px on a side)")
        for d in DROPPED_BOXES:
            print(f"    {d['object_id']:20} kept={d['kept_area_pct']}%  "
                  f"bbox={d['bbox']}  name={d['bangla_name']}")
    if CLIP["boxes_empty_after_clip"]:
        print(f"  ERROR: {CLIP['boxes_empty_after_clip']} box(es) lie ENTIRELY outside "
              f"the image and are empty after clipping - these need manual review.")

    if ROOMFIX:
        was = collections.Counter(d["was"] for d in ROOMFIX)
        print(f"\nObject-track room labels normalized to NA: {len(ROOMFIX)} "
              f"(was {dict(was)})")
        print(f"  affected image_ids: {[d['image_id'] for d in ROOMFIX][:15]}"
              f"{' ...' if len(ROOMFIX) > 15 else ''}")

    cross = collections.Counter((r["capture_type"], r["room_type"]) for r in out_list)
    print("\ncapture_type x room_type:")
    for cap in CAPTURE_TYPES:
        row = {rm: cross.get((cap, rm), 0) for rm in ROOM_TYPES if cross.get((cap, rm), 0)}
        print(f"  {cap:9} {row}")

    ann_counts = collections.Counter(r["annotator_id"] for r in out_list)
    print("\nimages per annotator_id:", dict(sorted(ann_counts.items())))

    no_annot = sorted(r["image_id"] for r in out_list if not r["annotator_id"])
    if no_annot:
        print(f"  WARNING: {len(no_annot)} records have no annotator_id: {no_annot[:10]}")

    skipped = [i for i in issues if i[2] == "skipped_or_empty"]
    dups = [i for i in issues if i[2] == "duplicate_image_id_across_exports"]
    other = [i for i in issues if i[2] not in ("skipped_or_empty", "duplicate_image_id_across_exports")]
    if skipped:
        print(f"\n{len(skipped)} task(s) skipped/empty in Label Studio (not in dataset).")
    if dups:
        print(f"WARNING: {len(dups)} duplicate image_id across exports (last one kept).")
        for ex, tid, _ in dups[:20]:
            print(f"    [{Path(ex).name}] {tid}")
    if other:
        print(f"{len(other)} other issue(s):")
        for ex, tid, why in other[:50]:
            print(f"    [{Path(ex).name}] task {tid}: {why}")

    if degenerate_records:
        print(f"\nDROPPED (scene record left with no usable box): {len(degenerate_records)}")
        for iid in sorted(degenerate_records):
            print(f"  {iid}  file={degenerate_records[iid]['image_file']}")
        print("  ^ delete these image files from the release image folder too.")

    if excluded:
        print(f"\nEXCLUDED (privacy/exclude, NOT in released dataset): {len(excluded)}")
        for iid, r in sorted(excluded.items()):
            print(f"  {iid}  reason={r['_exclude']}  file={r['image_file']}")
        print("  ^ delete these image files from the release image folder too.")

    if val_problems:
        kinds = collections.Counter(p.split(":")[0].split("=")[0].split("(")[0]
                                    for ps in val_problems.values() for p in ps)
        print(f"\nVALIDATION: {len(val_problems)} record(s) need a look:")
        for k, v in kinds.most_common():
            print(f"  {k:28} {v}")
        for iid in sorted(val_problems)[:40]:
            print(f"    {iid}: {', '.join(val_problems[iid])}")
        if len(val_problems) > 40:
            print(f"    ... and {len(val_problems) - 40} more (see report file).")
    else:
        print("\nVALIDATION: all records structurally clean.")

    # ---- completeness vs manifest ----
    completeness = {}
    skip_set = set(skipped_ids)
    if manifest:
        annotated = set(records) | set(excluded) | set(degenerate_records)
        all_ids = set(manifest)
        accounted = annotated | (skip_set & all_ids)
        real_gap = sorted(all_ids - accounted)
        extra = sorted(annotated - all_ids)
        completeness = {
            "manifest_total": len(all_ids),
            "in_dataset": len(records),
            "excluded": len(excluded),
            "dropped_degenerate": len(degenerate_records),
            "intentional_skips": len(skip_set & all_ids),
            "real_gap_count": len(real_gap),
            "real_gap": real_gap,
            "extra_not_in_manifest": extra,
        }
        print(f"\nCompleteness: {len(accounted)}/{len(all_ids)} manifest images accounted for.")
        print(f"  in dataset: {len(records)} | excluded: {len(excluded)} | "
              f"intentional skips: {len(skip_set & all_ids)}")
        if real_gap:
            print(f"  REAL GAP - {len(real_gap)} image(s) neither annotated, excluded, nor "
                  f"skipped (first 20): {real_gap[:20]}")
        else:
            print("  REAL GAP: none.")
        if extra:
            print(f"  {len(extra)} annotated but NOT in manifest (first 20): {extra[:20]}")

    report = {
        "schema_version": SCHEMA_VERSION,
        "out_file": args.out,
        "counts": {"total": len(out_list), "scene": n_scene, "closeup": n_close,
                   "qa_pairs": n_qa, "boxes": n_box,
                   "records_with_box_by_capture": dict(box_by_cap),
                   "excluded": len(excluded), "skipped": len(skip_set)},
        "normalization": dict(NORM),
        "bbox_clipping": {"summary": dict(CLIP), "clipped_boxes": CLIP_DETAIL,
                          "dropped_boxes": DROPPED_BOXES,
                          "thresholds": {"min_area_pct": MIN_BOX_AREA_PCT,
                                         "min_side_px": MIN_BOX_SIDE_PX}},
        "dropped_degenerate": sorted(degenerate_records),
        "closeup_room_normalized": ROOMFIX,
        "capture_x_room": {f"{c}|{r}": n for (c, r), n in sorted(cross.items())},
        "images_per_annotator": dict(sorted(ann_counts.items())),
        "no_annotator_id": no_annot,
        "excluded": {iid: r["_exclude"] for iid, r in excluded.items()},
        "skipped_ids": sorted(skip_set),
        "validation": val_problems,
        "duplicates": [i[1] for i in dups],
        "completeness": completeness,
    }
    rep_path = Path(args.out).with_suffix(".report.json")
    json.dump(report, open(rep_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nFull report -> {rep_path}")


if __name__ == "__main__":
    main()

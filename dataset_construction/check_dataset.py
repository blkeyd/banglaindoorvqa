#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_dataset.py - one-off sanity check on the release JSON. Not part of the
six-script pipeline; run it once, read the output, then delete it if you like.

Answers two questions:

  1. Which packaging is this JSON, and do its image_file paths still point at
     house subfolders that no longer exist in the flat DiB images/ folder?
  2. Did any digit forms slip into the count answers, against the convention
     that counts are written as Bangla words?

Usage on Windows:

    python check_dataset.py --json banglaindoorvqa.json --images-dir images
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ASCII_DIGITS = set("0123456789")
BANGLA_DIGITS = set("\u09e6\u09e7\u09e8\u09e9\u09ea\u09eb\u09ec\u09ed\u09ee\u09ef")
ANY_DIGIT = ASCII_DIGITS | BANGLA_DIGITS

QA_KEYS = ["qa_pairs", "qa", "qas", "questions"]
TYPE_KEYS = ["type", "qa_type", "q_type"]


def get(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="banglaindoorvqa.json")
    ap.add_argument("--images-dir", default="images")
    ap.add_argument("--show", type=int, default=25, help="how many offenders to list")
    args = ap.parse_args()

    raw = json.load(open(args.json, encoding="utf-8"))
    records = raw if isinstance(raw, list) else get(raw, ["records", "data", "images", "items"], [])
    print("JSON          : " + str(Path(args.json).resolve()))
    print("Records       : " + str(len(records)))

    # ---------------- 1. packaging ----------------
    paths = [str(r.get("image_file", "")) for r in records]
    with_folder = [p for p in paths if "/" in p or "\\" in p]
    print("")
    print("-- packaging --")
    print("image_file with a folder prefix : " + str(len(with_folder)) + " of " + str(len(paths)))
    if paths:
        print("example                         : " + paths[0])
    if with_folder:
        print("=> this is the per-house (main paper) style path.")
        print("   Harmless for our pipeline (01_build_items.py uses the basename only),")
        print("   but a reuser who joins this path onto a flat images/ folder gets a")
        print("   missing file. Worth telling sir if this is the DiB release JSON.")
    else:
        print("=> flat paths, matches the DiB images/ folder.")

    mismatched = [p for r, p in zip(records, paths)
                  if Path(p.replace("\\", "/")).stem != str(r.get("image_id", ""))]
    print("basename != image_id            : " + str(len(mismatched)))
    for p in mismatched[:args.show]:
        print("    " + p)

    img_dir = Path(args.images_dir)
    if img_dir.is_dir():
        on_disk = {p.stem for p in img_dir.iterdir() if p.is_file()}
        referenced = {str(r.get("image_id", "")) for r in records}
        print("files in " + args.images_dir + " : " + str(len(on_disk)))
        print("referenced but missing          : " + str(len(referenced - on_disk)))
        print("on disk but not referenced      : " + str(len(on_disk - referenced)))
    else:
        print("images folder not found, skipped: " + str(img_dir.resolve()))

    # ---------------- 2. digits in answers ----------------
    rows = []
    for r in records:
        for qa in get(r, QA_KEYS, []) or []:
            rows.append((str(get(qa, TYPE_KEYS, "")),
                         str(qa.get("qa_id", "")),
                         str(qa.get("answer", "")),
                         str(qa.get("question", ""))))

    print("")
    print("-- digits in answers --")
    print("QA pairs      : " + str(len(rows)))

    count_rows = [x for x in rows if x[0] == "count"]
    count_digit = [x for x in count_rows if ANY_DIGIT & set(x[2])]
    print("count answers                   : " + str(len(count_rows)))
    print("  containing a digit form       : " + str(len(count_digit)))
    for _t, qid, ans, _q in count_digit[:args.show]:
        kind = "ASCII" if ASCII_DIGITS & set(ans) else "Bangla"
        print("    " + qid.ljust(16) + kind.ljust(8) + ans)
    if len(count_digit) > args.show:
        print("    ... and " + str(len(count_digit) - args.show) + " more")

    other_digit = [x for x in rows if x[0] != "count" and ANY_DIGIT & set(x[2])]
    print("non-count answers with a digit  : " + str(len(other_digit)))
    for t, qid, ans, _q in other_digit[:args.show]:
        print("    " + qid.ljust(16) + t.ljust(14) + ans)

    ascii_any = [x for x in rows if ASCII_DIGITS & set(x[2])]
    print("answers with an ASCII digit     : " + str(len(ascii_any))
          + "   (normalize.py maps these to Bangla digits)")

    # The distinct count vocabulary, for eyeballing.
    print("")
    print("-- distinct count answers (" + str(len({x[2] for x in count_rows})) + ") --")
    for ans, n in Counter(x[2] for x in count_rows).most_common(40):
        print("  " + str(n).rjust(6) + "  " + ans)

    # Answers that do not end with a daari, in case the convention slipped there too.
    no_daari = [x for x in rows if x[2].strip() and not x[2].strip().endswith("\u0964")]
    print("")
    print("-- answers not ending in a daari : " + str(len(no_daari)) + " --")
    for t, qid, ans, _q in no_daari[:args.show]:
        print("    " + qid.ljust(16) + t.ljust(14) + ans)

    # ---------------- 3. answer/question copy-paste slips ----------------
    def flat(s):
        return unicodedata.normalize("NFC", re.sub(r"\s+", " ", s)).strip().rstrip("\u0964?.")

    same = [x for x in rows if x[2].strip() and flat(x[2]) == flat(x[3])]
    blank = [x for x in rows if not x[2].strip() or not x[3].strip()]
    print("")
    print("-- answer copied from the question : " + str(len(same)) + " --")
    for t, qid, ans, _q in same[:args.show]:
        print("    " + qid.ljust(16) + t.ljust(14) + ans)
    if len(same) > args.show:
        print("    ... and " + str(len(same) - args.show) + " more")
    print("-- blank question or answer        : " + str(len(blank)) + " --")
    for t, qid, ans, q in blank[:args.show]:
        print("    " + qid.ljust(16) + t.ljust(14) + "q=" + repr(q[:40]) + " a=" + repr(ans[:40]))

    print("")
    print("Nothing here blocks the run. It tells you which rows in Table 2 will")
    print("be scored against a form the models are unlikely to produce.")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize.py - shared Bangla text cleaning for the BanglaIndoorVQA zero-shot runs.

Implements the cleaning rule from Section 3 of BanglaIndoorVQA_ZeroShot_Plan.md,
applied in this exact order:

    1. Unicode NFC
    2. remove ZWJ / ZWNJ that are NOT immediately after hasant (U+09CD)
    3. collapse whitespace
    4. strip the trailing daari (U+0964) and other end punctuation
    5. lowercase Latin letters
    6. map ASCII digits to Bangla digits

Every script that compares text must import from here, so the scoring rule lives
in exactly one place:

    from normalize import normalize, strict_match, relaxed_match

Self-test:
    python normalize.py
"""

from __future__ import annotations

import re
import sys
import unicodedata

# --------------------------------------------------------------------------
# Characters
# --------------------------------------------------------------------------

HASANT = "\u09cd"   # Bangla hasant / virama
ZWNJ = "\u200c"     # zero-width non-joiner
ZWJ = "\u200d"      # zero-width joiner
ZWSP = "\u200b"     # zero-width space (invisible, not Unicode whitespace)
BOM = "\ufeff"      # byte-order mark / zero-width no-break space

# End punctuation stripped from the tail of a string.
# Edit this one constant if you want a different set; nothing else changes.
#   U+0964 daari, U+0965 double daari, ASCII sentence punctuation, quotes,
#   ellipsis, and the pipe (some Bangla keyboard layouts type "|" for daari).
TERMINAL_PUNCT = "\u0964\u0965.!?,;:\u2026|\"'\u2018\u2019\u201c\u201d"

# ASCII digits -> Bangla digits
ASCII_TO_BANGLA_DIGITS = str.maketrans("0123456789", "\u09e6\u09e7\u09e8\u09e9\u09ea\u09eb\u09ec\u09ed\u09ee\u09ef")

# A ZWJ/ZWNJ that is NOT preceded by hasant is a keyboard artefact.
# One preceded by hasant is functional (e.g. the reph in ...) and must survive.
_INERT_JOINER_RE = re.compile("(?<!" + HASANT + ")[" + ZWNJ + ZWJ + "]")

_WHITESPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# The individual steps, exposed so they can be tested or reused separately
# --------------------------------------------------------------------------

def to_nfc(text: str) -> str:
    """Step 1. Unicode NFC composition."""
    return unicodedata.normalize("NFC", text)


def strip_inert_joiners(text: str) -> str:
    """Step 2. Drop ZWJ/ZWNJ unless directly after hasant (U+09CD)."""
    return _INERT_JOINER_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Step 3. Any run of whitespace becomes a single space; ends trimmed.

    ZWSP and BOM are invisible but are not Unicode whitespace, so they are
    turned into spaces first. Left in place they silently break exact match.
    """
    text = text.replace(ZWSP, " ").replace(BOM, " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_terminal_punct(text: str) -> str:
    """Step 4. Remove trailing end punctuation (and any space in front of it).

    str.rstrip removes every trailing character in the set, in any order, so
    "chair ." and "chair.." and "chair . ." all end up as "chair".
    """
    return text.rstrip(TERMINAL_PUNCT + " \t")


def lowercase_latin(text: str) -> str:
    """Step 5. Lowercase Latin letters. Bangla has no case, so it is untouched."""
    return text.lower()


def ascii_digits_to_bangla(text: str) -> str:
    """Step 6. 0-9 -> Bangla digits. Helps when a model answers "3" for tin."""
    return text.translate(ASCII_TO_BANGLA_DIGITS)


# --------------------------------------------------------------------------
# The function everything else calls
# --------------------------------------------------------------------------

def normalize(text) -> str:
    """Apply the full Section 3 cleaning rule. None and non-strings are safe."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = to_nfc(text)
    text = strip_inert_joiners(text)
    text = collapse_whitespace(text)
    text = strip_terminal_punct(text)
    text = lowercase_latin(text)
    text = ascii_digits_to_bangla(text)
    return text


def strict_match(prediction, reference) -> bool:
    """Cleaned prediction is exactly equal to the cleaned reference."""
    ref = normalize(reference)
    if not ref:
        return False
    return normalize(prediction) == ref


def relaxed_match(prediction, reference) -> bool:
    """Cleaned reference appears somewhere inside the cleaned prediction."""
    ref = normalize(reference)
    if not ref:
        return False
    return ref in normalize(prediction)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _self_test() -> int:
    cases = [
        # (raw, expected, note)
        ("\u098f\u0995\u099f\u09bf \u09ab\u09cd\u09b0\u09bf\u099c\u0964",
         "\u098f\u0995\u099f\u09bf \u09ab\u09cd\u09b0\u09bf\u099c",
         "trailing daari removed"),
        ("  \u09ab\u09cd\u09b0\u09bf\u099c   \u0964 ",
         "\u09ab\u09cd\u09b0\u09bf\u099c",
         "whitespace collapsed, spaced daari removed"),
        ("Fridge.", "fridge", "Latin lowercased, full stop removed"),
        ("3\u099f\u09bf", "\u09e9\u099f\u09bf", "ASCII digit mapped to Bangla"),
        ("\u0995" + ZWNJ + "\u0996", "\u0995\u0996", "inert ZWNJ dropped"),
        ("\u09b0" + HASANT + ZWJ + "\u09af",
         "\u09b0" + HASANT + ZWJ + "\u09af",
         "joiner after hasant kept"),
        ("\u09ab\u09cd\u09b0\u09bf\u099c" + ZWSP, "\u09ab\u09cd\u09b0\u09bf\u099c",
         "zero-width space removed"),
        (None, "", "None is safe"),
    ]

    failures = 0
    for raw, expected, note in cases:
        got = normalize(raw)
        ok = got == expected
        failures += 0 if ok else 1
        print(("  PASS  " if ok else "  FAIL  ") + note)
        if not ok:
            print("        expected: " + repr(expected))
            print("        got     : " + repr(got))

    # The example from Section 3 of the plan.
    pred = "\u098f\u099f\u09bf \u098f\u0995\u099f\u09bf \u09ab\u09cd\u09b0\u09bf\u099c\u0964"   # "It is a fridge."
    gold = "\u09ab\u09cd\u09b0\u09bf\u099c\u0964"                                               # "Fridge."
    checks = [
        (strict_match(pred, gold) is False, "strict match rejects the full sentence"),
        (relaxed_match(pred, gold) is True, "relaxed match accepts the full sentence"),
        (strict_match(gold, gold) is True, "strict match accepts the bare answer"),
        (relaxed_match("", gold) is False, "empty prediction is not a match"),
        (relaxed_match(pred, "") is False, "empty reference is never a match"),
    ]
    for ok, note in checks:
        failures += 0 if ok else 1
        print(("  PASS  " if ok else "  FAIL  ") + note)

    print("")
    print("normalize.py self-test: " + ("all checks passed" if failures == 0
                                        else str(failures) + " FAILED"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(_self_test())

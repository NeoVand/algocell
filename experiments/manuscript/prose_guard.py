#!/usr/bin/env python3
"""Compare numerical and bracketed-source multisets in two UTF-8 prose files.

Usage: prose_guard.py OLD NEW

This is a lexical guard, not a scientific or semantic audit. It checks the whole
files, including headings, tables, references and code. Numerical spelling is
preserved: ``1,000`` differs from ``1000``, ``1.0`` differs from ``1``, and ``one``
differs from ``1``. Written-out cardinal and ordinal words are case-folded but
otherwise preserved, including *every* standalone occurrence of ``one`` and
``first``. Hyphenated compounds are split into words. Signs attached to a
standalone numeral, decimal fractions, scientific notation, numeric ordinals and
``0x`` hexadecimal literals are retained as whole numerical tokens. Bare hex
words containing letters cannot be distinguished from identifiers and are not
interpreted as numbers; digit-only code bytes remain numerical tokens.

The first, prose-oriented numerical inventory excludes numeric substrings of
identifiers (C1, L32, ed8_1), commit hashes, URLs, file paths and dotted version
identifiers, rather than splitting them into arbitrary numbers. A coefficient
directly adjoining a letter (2L, 8r) is also
an identifier under this conservative lexical rule. Numeric ratios, ranges and
standalone machine names such as 6502 remain checked. No semantic quantity,
unit, status, denominator attachment or citation referent is inferred.

A second, stricter ``raw digit tokens`` inventory scans every digit sequence
outside the protected bracket spans, without identifier/path exclusions or
word boundaries. It therefore checks 2 in 2L, 16 in L16, 4 in c4 and every
numeric part of a path or hash. Decimal/scientific/grouped/hexadecimal forms
are still retained as whole tokens where matched. This intentionally flags
changes in non-scientific labels too; it does not silently dismiss them.

All balanced, unescaped square-bracket spans outside Markdown code are compared
as opaque strings, with whitespace runs collapsed to one space. This includes
source brackets, citation brackets, placeholders and mathematical indexing;
nested brackets belong to their outermost span. Their contents are excluded
from both numerical inventories. Brackets inside inline/fenced code are
ordinary code, so numerical literals and indexing digits there are checked
by the raw inventory (and by the prose inventory when its boundaries allow).
Escaped brackets (including
LaTeX display delimiters) are not source delimiters. Unmatched prose brackets
are errors. This lightweight scanner is not a full Markdown/LaTeX parser.

Exit status: 0 identical multisets; 1 differences; 2 input or bracket error.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


CARDINALS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "thirty forty fifty sixty seventy eighty ninety hundred thousand million "
    "billion"
).split()
ORDINALS = (
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth "
    "eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth "
    "eighteenth nineteenth twentieth thirtieth fortieth fiftieth sixtieth "
    "seventieth eightieth ninetieth hundredth thousandth millionth billionth"
).split()
# Fraction words are quantities too. They are intentionally not interpreted.
FRACTIONS = "half halves quarter quarters".split()
WORD_NUMBER = re.compile(
    r"(?<![\w])(?:" + "|".join(CARDINALS + ORDINALS + FRACTIONS) + r")(?![\w])",
    re.IGNORECASE,
)
NUMBER = re.compile(
    r"(?<![\w.])"
    r"[+\-−]?"
    r"(?:0[xX][0-9a-fA-F]+|"
    r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+\-−]?\d+)?(?:st|nd|rd|th)?)"
    r"(?![\w]|\.\d)",
    re.IGNORECASE,
)
RAW_NUMBER = re.compile(
    r"[+\-−]?"
    r"(?:0[xX][0-9a-fA-F]+|"
    r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+\-−]?\d+)?(?:st|nd|rd|th)?)",
    re.IGNORECASE,
)
URL = re.compile(r"(?:https?://|doi:|www\.)[^\s<>`]+", re.IGNORECASE)
PATH = re.compile(r"(?<![\w])(?:[\w.~-]+/)+[\w.~-]+(?:#[\w.-]+)?")
VERSION = re.compile(r"(?<![\w.])\d+(?:\.\d+){2,}(?![\w.])")
HEX_HASH = re.compile(r"(?<![\w])[0-9a-fA-F]{7,64}(?![\w])")
FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def blank_ranges(text: str, ranges: list[tuple[int, int]]) -> str:
    """Keep positions while removing guarded or excluded spans."""
    chars = list(text)
    for start, end in ranges:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def code_ranges(text: str) -> list[tuple[int, int]]:
    """Identify fenced code and matching backtick spans, without dependencies."""
    ranges: list[tuple[int, int]] = []
    opened: tuple[str, int, int] | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        match = FENCE.match(line)
        if match:
            run = match.group(1)
            if opened is None:
                opened = (run[0], len(run), offset)
            elif (
                run[0] == opened[0]
                and len(run) >= opened[1]
                and not line[match.end() :].strip()
            ):
                ranges.append((opened[2], offset + len(line)))
                opened = None
        offset += len(line)
    if opened is not None:
        ranges.append((opened[2], len(text)))

    outside_fences = blank_ranges(text, ranges)
    runs = list(re.finditer(r"`+", outside_fences))
    index = 0
    while index < len(runs):
        opening = runs[index]
        closing_index = index + 1
        while closing_index < len(runs):
            if len(runs[closing_index].group()) == len(opening.group()):
                ranges.append((opening.start(), runs[closing_index].end()))
                index = closing_index
                break
            closing_index += 1
        index += 1
    return sorted(ranges)


def is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def bracket_spans(text: str) -> list[tuple[int, int]]:
    code = bytearray(len(text))
    for start, end in code_ranges(text):
        code[start:end] = b"\1" * (end - start)
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    for index, char in enumerate(text):
        if code[index] or char not in "[]" or is_escaped(text, index):
            continue
        if char == "[":
            stack.append(index)
        elif not stack:
            raise ValueError(f"unmatched ] on line {text.count(chr(10), 0, index) + 1}")
        else:
            start = stack.pop()
            if not stack:
                spans.append((start, index + 1))
    if stack:
        line = text.count("\n", 0, stack[0]) + 1
        raise ValueError(f"unmatched [ on line {line}")
    return spans


def inventory(text: str) -> tuple[Counter[str], Counter[str], Counter[str]]:
    spans = bracket_spans(text)
    brackets = Counter(" ".join(text[start:end].split()) for start, end in spans)
    outside = blank_ranges(text, spans)
    # Use precisely the same protected source spans for both numeric layers.
    # Code indexing remains visible, whereas indexing in prose is a bracket span.
    raw_numbers = Counter(match.group() for match in RAW_NUMBER.finditer(outside))
    exclusions = [(match.start(), match.end()) for match in URL.finditer(outside)]
    for match in PATH.finditer(outside):
        # Preserve numeric fractions such as 1/2 and chains such as 1/2/3.
        if any(char.isalpha() or char == "_" for char in match.group()):
            exclusions.append((match.start(), match.end()))
    exclusions.extend((match.start(), match.end()) for match in VERSION.finditer(outside))
    for match in HEX_HASH.finditer(outside):
        # An all-digit count is not a commit hash. Mixed hex hashes are opaque.
        if any(char.isalpha() for char in match.group()) and any(
            char.isdigit() for char in match.group()
        ):
            exclusions.append((match.start(), match.end()))
    outside = blank_ranges(outside, exclusions)
    # ASCII two-dot and three-dot ranges are separators, not decimal points.
    outside = re.sub(r"\.{2,}", lambda match: " " * len(match.group()), outside)
    numbers = Counter(match.group() for match in NUMBER.finditer(outside))
    numbers.update(match.group().casefold() for match in WORD_NUMBER.finditer(outside))
    return numbers, raw_numbers, brackets


def report(label: str, old: Counter[str], new: Counter[str]) -> bool:
    removed = old - new
    added = new - old
    same = not removed and not added
    print(
        f"{label}: {'PASS' if same else 'FAIL'} "
        f"(old={old.total()}, new={new.total()}; "
        f"removed={removed.total()}, added={added.total()})"
    )
    for direction, changes in (("removed", removed), ("added", added)):
        for token, count in sorted(changes.items()):
            print(f"  {direction} {count} × {json.dumps(token, ensure_ascii=False)}")
    return same


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args()
    try:
        old_numbers, old_raw_numbers, old_brackets = inventory(args.old.read_text(encoding="utf-8"))
        new_numbers, new_raw_numbers, new_brackets = inventory(args.new.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"prose_guard: {args.old} -> {args.new}")
    numbers_ok = report("numbers", old_numbers, new_numbers)
    raw_numbers_ok = report("raw digit tokens", old_raw_numbers, new_raw_numbers)
    brackets_ok = report("bracketed spans", old_brackets, new_brackets)
    all_ok = numbers_ok and raw_numbers_ok and brackets_ok
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

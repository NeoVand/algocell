"""Check (and with --fix, repair) that references are numbered by order of first citation.

    .venv/bin/python manuscript/refcheck.py [--md manuscript/MAIN_nature.md] [--fix]

Citations are superscripts ^…^ holding numbers, commas and en-dash ranges (e.g. ^1–4^, ^16,22^). The order of first
appearance is taken over the whole document before the reference list, then over the part after it (Supplementary
Information). --fix renumbers every citation and reorders the reference list; uncited references are reported.
"""
import argparse
import re
import sys

CIT = re.compile(r"\^([0-9][0-9,–\-]*)\^")


def expand(tok: str) -> list[int]:
    out = []
    for part in tok.split(","):
        if "–" in part or "-" in part:
            a, b = re.split("[–-]", part)
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def compress(nums: list[int]) -> str:
    nums = sorted(set(nums))
    parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}"))
        i = j + 1
    return ",".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="manuscript/MAIN_nature.md")
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    s = open(a.md).read()
    h = s.index("\n## References")
    nxt = s.index("\n## ", h + 5)
    body, refs, tail = s[:h], s[h:nxt], s[nxt:]
    entries = re.findall(r"^(\d+)\. (.*)$", refs, flags=re.M)
    ref = {int(n): t for n, t in entries}
    order = []
    for m in CIT.finditer(body + tail):
        for n in expand(m.group(1)):
            if n not in order:
                order.append(n)
    missing = [n for n in order if n not in ref]
    uncited = [n for n in ref if n not in order]
    in_order = order == sorted(order) and order == list(range(1, len(order) + 1))
    print(f"references {len(ref)}, cited {len(order)}, in order: {in_order}; cited but not listed: {missing}; listed but not cited: {uncited}")
    if not in_order:
        print("first-citation order:", order)
    if a.fix and not in_order:
        if missing:
            sys.exit("refusing to fix: citations without a reference entry")
        new = {old: i + 1 for i, old in enumerate(order)}
        for old in uncited:
            new[old] = len(new) + 1
        fixn = lambda t: CIT.sub(lambda m: "^" + compress([new[n] for n in expand(m.group(1))]) + "^", t)  # noqa: E731
        body2, tail2 = fixn(body), fixn(tail)
        lines = [f"{new[n]}. {t}" for n, t in sorted(ref.items(), key=lambda kv: new[kv[0]])]
        head = refs[: refs.index("\n", 1) + 1]
        refs2 = head + "\n" + "\n".join(lines) + "\n"
        open(a.md, "w").write(body2 + refs2 + tail2)
        print("renumbered; mapping old->new:", {k: v for k, v in new.items() if k != v})


if __name__ == "__main__":
    main()

"""Z80 ISA model + suppression grammar, loaded from shader/isa.json (exported from
src/lib/z80-opcodes.ts) so the Python side resolves patterns exactly like the app."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

SHADER_DIR = Path(__file__).parent / "shader"
PAGES = ("base", "cb", "ed")
PAGE_INDEX = {"base": 0, "cb": 1, "ed": 2}


@lru_cache(maxsize=1)
def load() -> dict:
    with open(SHADER_DIR / "isa.json") as f:
        return json.load(f)


def instructions() -> list[dict]:
    isa = load()
    return [
        ins
        for pg in PAGES
        for ins in isa["pages"][pg]
        if ins["family"] not in ("prefix", "undefined")
    ]


def instructions_of(family: str) -> list[dict]:
    return [i for i in instructions() if i["family"] == family]


def selectable_families() -> list[dict]:
    return [f for f in load()["families"] if f["selectable"]]


_PAGED = re.compile(r"^(base|cb|ed):(?:0x)?([0-9a-f]{2})$")
_HEX = re.compile(r"^(?:0x)?([0-9a-f]{2})$")


def match_pattern(pattern: str) -> list[dict]:
    """Port of matchPattern() in z80-opcodes.ts (same grammar, same precedence)."""
    raw = pattern.strip()
    if not raw:
        return []
    lower = raw.lower()
    isa = load()
    if lower.startswith("family:"):
        fid = lower[len("family:"):].strip()
        if fid == "writes-mem":
            return [i for i in instructions() if i["writesMem"]]
        fam = next((f for f in isa["families"] if f["id"] == fid), None)
        if not fam or not fam["selectable"]:
            return []
        return instructions_of(fid)
    m = _PAGED.match(lower)
    if m:
        ins = isa["pages"][m.group(1)][int(m.group(2), 16)]
        return [] if ins["family"] in ("prefix", "undefined") else [ins]
    m = _HEX.match(lower)
    if m:
        ins = isa["pages"]["base"][int(m.group(1), 16)]
        return [] if ins["family"] == "prefix" else [ins]
    upper = re.sub(r",\s+", ",", raw.upper())
    if upper == "RST N":
        return instructions_of("rst")
    return [i for i in instructions() if upper in i["mnemonic"].upper()]


def resolve(patterns) -> dict[str, set[int]]:
    """Additions first, then '-pattern' subtractions (same as resolveSuppression in z80-opcodes.ts)."""
    sets: dict[str, set[int]] = {pg: set() for pg in PAGES}
    adds = [p for p in patterns if not p.strip().startswith("-")]
    subs = [p.strip()[1:] for p in patterns if p.strip().startswith("-")]
    for pat in adds:
        for ins in match_pattern(pat):
            sets[ins["page"]].add(ins["code"])
    for pat in subs:
        for ins in match_pattern(pat):
            sets[ins["page"]].discard(ins["code"])
    return sets


def masks(sets: dict[str, set[int]]) -> np.ndarray:
    """24 u32 words: base[0:8], cb[8:16], ed[16:24]; bit (code & 31) of word (code >> 5)."""
    m = np.zeros(24, dtype=np.uint32)
    for pg in PAGES:
        off = PAGE_INDEX[pg] * 8
        for c in sets[pg]:
            m[off + (c >> 5)] |= np.uint32(1 << (c & 31))
    return m


def count(sets: dict[str, set[int]]) -> int:
    return sum(len(s) for s in sets.values())


def parse_patterns(spec: str | None) -> list[str]:
    """';'-separated like the app's text box (commas belong to mnemonics)."""
    if not spec:
        return []
    return [p.strip() for p in spec.split(";") if p.strip()]


# ── Disassembly + mechanism classification ──────────────────────────────────

# Which write-capable families count as which replication mechanism.
MECHANISM_OF_FAMILY = {
    "stack": "stack",       # PUSH (POP does not write; filtered by writesMem)
    "ex": "stack",          # EX (SP),HL
    "call-ret": "stack",    # CALL pushes PC
    "rst": "stack",
    "block-copy": "block-copy",
    "ld8-mem": "ld-mem",
    "ld16-mem": "ld-mem",
    "incdec-mem": "rmw",
    "rotate-mem": "rmw",
    "bit-set-mem": "rmw",
    "block-io": "io",
}


def disassemble(tape: bytes, wrap: int | None = None, suppress: dict[str, set[int]] | None = None) -> list[dict]:
    """Linear disassembly of a tape from byte 0, following the Z80 decode rules
    (prefix bytes, operand lengths). With `suppress` (per-page sets, as from
    resolve()), a suppressed opcode is rendered the way the GPU core executes
    it: a NOP that consumes only the opcode (and prefix) bytes, so its operand
    bytes decode as the next instructions. Addresses wrap at `wrap` (default len)."""
    isa = load()
    sup = suppress or {}
    n = len(tape)
    wrap = wrap or n
    out: list[dict] = []
    pc = 0

    def is_sup(page: str, code: int) -> bool:
        return code in sup.get(page, ())

    while pc < n:
        start = pc
        b0 = tape[pc]
        pc += 1
        prefix = ""
        page = "base"
        op = b0
        if b0 in (0xDD, 0xFD):
            prefix = "DD" if b0 == 0xDD else "FD"
            if pc >= n:
                out.append({"offset": start, "bytes": [b0], "mnemonic": f"{prefix} prefix", "family": "prefix", "writesMem": False})
                break
            b1 = tape[pc]
            if b1 in (0xDD, 0xFD, 0xED):
                out.append({"offset": start, "bytes": [b0], "mnemonic": f"{prefix} prefix", "family": "prefix", "writesMem": False})
                continue
            pc += 1
            if b1 == 0xCB:
                # DD CB d op
                d = tape[pc] if pc < n else 0
                cbop = tape[pc + 1] if pc + 1 < n else 0
                pc += 2
                ins = isa["pages"]["cb"][cbop]
                mn = ins["mnemonic"].replace("(HL)", f"(I{'X' if prefix=='DD' else 'Y'}+d)")
                if is_sup("cb", cbop):
                    out.append({"offset": start, "bytes": list(tape[start:pc]), "mnemonic": f"({mn} suppressed)", "family": "suppressed", "writesMem": False, "page": "cb", "code": cbop})
                else:
                    out.append({"offset": start, "bytes": list(tape[start:pc]), "mnemonic": mn, "family": ins["family"], "writesMem": ins["writesMem"], "page": "cb", "code": cbop})
                continue
            op = b1
            ins = isa["pages"]["base"][op]
            mn = ins["mnemonic"].replace("(HL)", f"(I{'X' if prefix=='DD' else 'Y'}+d)").replace("HL", "IX" if prefix == "DD" else "IY")
            if is_sup("base", op):
                # prefix + opcode consumed, operands fall through
                out.append({"offset": start, "bytes": list(tape[start:pc]), "mnemonic": f"({mn} suppressed)", "family": "suppressed", "writesMem": False, "page": "base", "code": op})
                continue
            length = ins["length"]
            if "(HL)" in ins["mnemonic"] and ins["mnemonic"] not in ("JP (HL)",):
                length += 1  # displacement
            pc = start + 1 + length
            out.append({"offset": start, "bytes": list(tape[start:min(pc, n)]), "mnemonic": mn, "family": ins["family"], "writesMem": ins["writesMem"], "page": "base", "code": op})
            continue
        if b0 == 0xCB:
            cbop = tape[pc] if pc < n else 0
            pc += 1
            ins = isa["pages"]["cb"][cbop]
            page = "cb"
            op = cbop
        elif b0 == 0xED:
            edop = tape[pc] if pc < n else 0
            pc += 1
            ins = isa["pages"]["ed"][edop]
            page = "ed"
            op = edop
            if not is_sup("ed", edop):
                pc = start + ins["length"]
        else:
            ins = isa["pages"]["base"][op]
            if not is_sup("base", op):
                pc = start + ins["length"]
        if is_sup(page, op):
            out.append({"offset": start, "bytes": list(tape[start:min(pc, n)]), "mnemonic": f"({ins['mnemonic']} suppressed)", "family": "suppressed", "writesMem": False, "page": page, "code": op})
        else:
            out.append({"offset": start, "bytes": list(tape[start:min(pc, n)]), "mnemonic": ins["mnemonic"], "family": ins["family"], "writesMem": ins["writesMem"], "page": page, "code": op})
    return out


def mechanisms(tape: bytes, suppress: dict[str, set[int]] | None = None) -> list[str]:
    """Replication-relevant write mechanisms present in a tape (linear decode)."""
    found: set[str] = set()
    for ins in disassemble(tape, suppress=suppress):
        if ins["writesMem"]:
            found.add(MECHANISM_OF_FAMILY.get(ins["family"], ins["family"]))
    return sorted(found)

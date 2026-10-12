"""Read the existing period-2 lists; distinguish literals from memory copies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path


DEFAULT_8086 = Path("/private/tmp/claude-501/-Users-neo-repos-algocell/"
                    "26fc62cd-9c63-4d42-8467-b9398dac54bd/scratchpad/isa_research/census")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--8086-dir", dest="x86", type=Path, default=DEFAULT_8086)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[3]
    sources = [root / "experiments/results/review_r2/census_p2.csv",
               root / "experiments/results/census6502/control_z80_words.csv"]
    rows, provenance = [], []
    z80_literal = {"01 c5": "BC", "11 d5": "DE", "21 e5": "HL"}
    for path in sources:
        provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        with path.open() as f:
            source = [r for r in csv.DictReader(f) if r.get("period", "2") == "2"]
        assert {r["word"] for r in source} == set(z80_literal) | {"2a e5", "e5 2a"}
        for r in source:
            word = r["word"]
            x, y = bytes.fromhex(word)
            literal = word in z80_literal
            rows.append({"machine": "Z80", "source": str(path.relative_to(root)),
                         "word": word, "literal_channel": literal,
                         "mechanism": "LD rr,nn; PUSH rr" if literal else "LD HL,(nn); PUSH HL",
                         "load_opcode": f"{x:02x}" if literal else "",
                         "push_opcode": f"{y:02x}" if literal else "",
                         "literal_bytes": f"{y:02x} {x:02x}" if literal else "",
                         "alternating": True,
                         "note": "" if literal else "Memory read: outside Proposition 9"})
    path = args.x86 / "out8086_p2.txt"
    traces = []
    if path.exists():
        provenance.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "qualification": "Partial unvalidated interpreter; 9,343 unknown words"})
        for match in re.finditer(r"SELF-WRITER ([0-9a-f]{2}) ([0-9a-f]{2})", path.read_text()):
            x, y = [int(s, 16) for s in match.groups()]
            a, b = (x, y) if 0xB8 <= x <= 0xBF else (y, x)
            literal = 0xB8 <= a <= 0xBF and b == 0x50 + a - 0xB8 and a != 0xBC
            word = f"{x:02x} {y:02x}"
            if literal:
                mechanism, note = "MOV r16,imm16; PUSH r16", ""
                if x == b:
                    note = "Starts with initial-register push; steady load-push pair follows"
            elif x == 0xFF or y == 0xFF:
                mechanism, note = "Memory/indirect or arithmetic", "Not a one-byte literal channel"
            elif {x, y} == {0xAD, 0x50}:
                mechanism, note = "LODSW; PUSH AX", "Memory read"
            elif x == 0x07:
                mechanism, note = "POP ES; MOVS", "Memory copy"
            elif x == 0xC4:
                mechanism, note = "LES; PUSH DX", "Memory read"
            elif (x, y) == (0x0E, 0xEA):
                mechanism, note = "PUSH CS; far JMP", "Control transfer"
            else:
                mechanism, note = "Other arithmetic/store mechanism", "Outside load-literal/push grammar"
            rows.append({"machine": "8086", "source": str(path), "word": word,
                         "literal_channel": literal, "mechanism": mechanism,
                         "load_opcode": f"{a:02x}" if literal else "",
                         "push_opcode": f"{b:02x}" if literal else "",
                         "literal_bytes": f"{b:02x} {a:02x}" if literal else "",
                         "alternating": True, "note": note})
        # Replay only harness questions; this is not independent ISA validation.
        cpu_path = args.x86 / "m8086.py"
        spec = importlib.util.spec_from_file_location("audit_m8086", cpu_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        provenance.append({"path": str(cpu_path), "sha256": hashlib.sha256(cpu_path.read_bytes()).hexdigest()})
        for word in ("b8 50", "50 b8", "b9 51", "bc 54", "54 bc"):
            code = bytes.fromhex(word)
            memory = bytearray(code * 8 + bytes(16))
            cpu = module.CPU(memory, 16)
            error = ""
            while cpu.steps < 128 and not cpu.halted:
                ip, ax, sp = cpu.ip, cpu.r[0], cpu.r[4]
                try:
                    cpu.step()
                except module.Unknown as e:
                    error = str(e)
                    break
                traces.append({"word": word, "step": cpu.steps, "ip_before": ip,
                               "AX_before": f"{ax:04x}", "SP_before": f"{sp:04x}",
                               "AX_after": f"{cpu.r[0]:04x}", "SP_after": f"{cpu.r[4]:04x}",
                               "partner": memory[16:].hex(" ")})
            exact = any(memory[16:] == bytes(code[(i + s) % 2] for i in range(16)) for s in range(2))
            provenance.append({"replay_word": word, "exact_zero_partner": exact,
                               "unknown": error, "partner": memory[16:].hex(" ")})
    else:
        provenance.append({"path": str(path), "missing": True})
    args.out.mkdir(parents=True, exist_ok=True)
    for filename, data in [("census_audit.csv", rows), ("8086_harness_trace.csv", traces)]:
        if data:
            with (args.out / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(data)
    (args.out / "census_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"{len(rows)} source rows; {sum(r['literal_channel'] for r in rows)} literal-channel rows")


if __name__ == "__main__":
    main()

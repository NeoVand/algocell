"""Cross-check independent symbolic implementations and distinct opcode sets."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from direct_search import check, shapes
from search import replay, solve


def main():
    out = Path(__file__).parent
    rows = []
    for width in (1, 2, 3):
        for k in range(1, 17):
            direct = {}
            for word, pushed in shapes(k, width):
                nl = (k - len(pushed)) // (width + 1)
                key = nl, len(pushed)
                hit, _ = check(word, pushed, width)
                direct[key] = direct.get(key, False) or hit is not None
            for nl in range(1, (k - 1) // (width + 1) + 1):
                np = k - (width + 1) * nl
                result, witness, reason = solve(k, nl, np, width, tuple(range(width)), exact=True)
                assert not reason and result in ("sat", "unsat")
                assert (result == "sat") == direct.get((nl, np), False)
                rows.append({"machine": "single load/push", "width": width,
                             "execution_period": k, "loads": nl, "pushes": np,
                             "result": result, "agrees_direct": True})
    # A concrete abstract machine with THREE distinct load opcodes and TWO
    # distinct push opcodes, all sharing the same register (not a real ISA).
    for k in range(1, 17):
        for nl in range(1, (k - 1) // 2 + 1):
            np = k - 2 * nl
            result, witness, reason = solve(k, nl, np, 1, (0,), loads=(1, 2, 3),
                                           pushes=(0x80, 0x81), exact=True)
            assert result == "unsat" and not reason and witness is None
            rows.append({"machine": "3 loads / 2 pushes", "width": 1,
                         "execution_period": k, "loads": nl, "pushes": np,
                         "result": result, "agrees_direct": True})
    for filename in ("minimal_witnesses.json", "direct_witnesses.json"):
        for witness in json.loads((out / filename).read_text()):
            replay(witness["word"], witness["width"], tuple(range(witness["width"])))
    finite = json.loads((out / "finite_witness.json").read_text())
    word = finite["word"]
    assert next(d for d in range(1, len(word) + 1) if len(word) % d == 0 and all(
        word[i] == word[i % d] for i in range(len(word)))) == 16
    assert finite["written_ascending"][:16] == [word[(i + 5) % 16] for i in range(16)]
    assert finite["written_ascending"][16] != word[5]
    with (out / "verification.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"crosschecked_cases": sum(r["machine"] == "single load/push" for r in rows),
                      "multi_opcode_cases": sum(r["machine"] != "single load/push" for r in rows),
                      "concrete_replays": "passed", "finite_window_replay": "passed"}))


if __name__ == "__main__":
    main()

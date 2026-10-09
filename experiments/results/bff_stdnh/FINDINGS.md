# B5 — BFF as published with harmless brackets (`stdnh`: no literal, no pointer wrap, unmatched brackets are no-ops)

Pre-registered in `REVISION_PREREG.md` (B5); 12 soups, seeds 1–12, 2¹⁷ programs, 16,384 epochs, Modal batch `bff_stdnh`
(≈ 25 min each on an L40S; 5 GPU-h ≈ $10). Scored by `micro/bff_analysis.py` (`runs.csv`, `NUMBERS_BFF.md`).

| prediction | outcome |
|---|---|
| B5-1 every first replicator loop-bearing | **met**: 7 of 7 (no straight-line first replicator) |
| B5-2 transition rate ≥ 9 of 12 | **not met**: 7 of 12 (58%, against 9 of 24 for the published rule) |
| B5-3 first replicator pointer-closed (entered ≤ 0.05) in every world | **not met**: 5 closed (entered 0.00–0.03), 1 intermediate (0.23), 1 open (seed 11: enters the partner in every encounter, copies 0.81, loop-bearing) |

Unregistered observation: in 4 of the 7 soups with a transition (seeds 4, 5, 7, 9) the replicator is gone again by
16,384 epochs (final heritable fraction 0); the three that keep it (2, 11, 12) end closed and fully heritable. Without
halting, every random program runs its full 8,192 steps, and the soup appears to erase its replicators as readily as it
makes them.

Reading for the classification (no literal channel, benign tar): born closed, as predicted from the absence of a literal
channel, but not stably: life appears in 7 of 12 and persists in 3. The cell is filled; the sentence "no literal ⇒ born
closed" holds in the loop sense (Theorem 1) and almost always in the pointer sense (6 of 7).

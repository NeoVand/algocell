# Stage D — the L = 9 reversal, confirmatory (280 runs, 2026-10-07)

Design (PLAN.md Stage D, fixed before any run): L = 9, mutation 1/16, **new seeds 1001–1020**, 300k steps, no early stop, fine early sampling (steps 1–34 explicit, then every 50 to 5,000, then every 500), full snapshots and interaction census on the log schedule. Arms × {128, 512} Z80 steps: `none`; `stack-writes` (Stage B's 46-opcode arm); `stack-write-only` (PUSH, EX (SP),HL, CALL*, RST — the 22 that write through SP); `stack-read-only` (POP, RET*, EX DE,HL, EXX, EX AF,AF' — the 24 that write nothing); `push` (4); `call-rst-write` (17); plus `none` at 32 steps and at mutation 1/64. 280/280 runs completed, 0 failures, median 102 s per run. Numbers: `NUMBERS.md`, `cells.csv`, `assays.csv` in this directory.

## Pre-registered outcomes

| Test | Prediction | Result | Verdict |
|---|---|---|---|
| **D1** primary | `stack-write-only` > `none` in heritable emergence, pooled over budgets (one-sided CMH, α = 0.05) | 128 steps 12/20 vs 6/20 (one-sided Fisher p = 0.056); 512 steps 19/20 vs 5/20 (p = 5 × 10⁻⁶); CMH z = 4.47, **p = 3.9 × 10⁻⁶** | **confirmed** |
| **D1b** | `stack-writes` (46) > `none` — direct replication of Stage B | 12/20 vs 6/20 and 18/20 vs 5/20; CMH **p = 1.2 × 10⁻⁵** | **replicated** |
| **D2** (restated before the run) | zero_frac(`none`) − zero_frac(`stack-write-only`) ≥ 0.10 at step 5,000 | 0.337 − 0.194 = **+0.143** (128); 0.327 − 0.186 = **+0.141** (512); the read-only arm changes the flood by 0.000 | **confirmed**; the residual 0.19 comes from non-stack writers, as predicted |
| **D3** | write side carries ≥ 0.75 of the effect; read-only within 0.10 of `none` | ratio (write-only − none)/(stack-writes − none) = 1.0 (128) and 1.08 (512); read-only 8/20 vs 6/20 (+0.10) and 10/20 vs 5/20 (+0.25, one-sided p = 0.095; CMH p = 0.054) | first clause **confirmed**; second clause **not met at 512**: a small effect of removing the pointer routes (POP/RET/EX) is not excluded |
| **D4** | every L = 9 replicator LDIR-based with period 3 or 9 | 142 first replicators: period 3 × 119, 9 × 22, 6 × 1; every exemplar contains `ED B0` (LDIR); 133/142 labelled block-copy, the other 9 are phase artefacts of linear disassembly | **confirmed** (6 | 18 is also a divisor of the ring) |
| **D5** | `none` at 32 steps emerges more than at 128 (per-byte-hazard account) | 4/20 vs 6/20 (one-sided p = 0.86) | **not supported** |
| exploratory | `none` at mutation 1/64 | 3/20 (vs 6/20 at 1/16) | lower mutation gives fewer LDIR assemblies |

By the pre-registered occupancy event `tq_10`, `none` "emerges" in 20/20 runs at 128 steps and `stack-write-only` in 9/20: the occupancy measure inverts the result because it fires on the zero flood. Every heritable replicator in Stage D is also faithful.

## Which writer suppresses emergence

| arm removed | heritable 128 | heritable 512 | CMH one-sided p vs `none` | flood reduction at 5k (vs 0.33) |
|---|---|---|---|---|
| `call-rst-write` (CALL*, RST; 17) | 12/20 | 14/20 | 4.5 × 10⁻⁴ | −0.06 |
| `push` (PUSH ×4) | 11/20 | 8/20 | 0.033 | −0.04 |
| `stack-write-only` (both + EX (SP),HL) | 12/20 | 19/20 | 3.9 × 10⁻⁶ | −0.14 |
| `stack-read-only` (POP, RET*, EX DE,HL, EXX, EX AF,AF') | 8/20 | 10/20 | 0.054 | 0.00 |

Removing CALL and RST alone recovers most of the effect at both budgets while reducing the flood no more than removing PUSH does, so the suppression is **not explained by the zero load alone**: the return-address writers are the specific culprits (they write `addr, 00` pairs that make the sterile RST/CALL smears, e.g. `ff 41 00 41`, which spread without heredity and occupy the cells a 9-byte LDIR copier needs). The two writers are roughly additive at 512 steps (8 + 14 → 19 of 20).

## Dynamics (from the explicit early samples and the census)

- The flood forms within tens of steps in every arm: 0.05–0.07 zeros after one step under the full ISA, 0.30–0.33 by step 50, 0.33–0.34 at step 5,000; with the writing stack opcodes removed 0.015 → 0.13 → 0.19. The Stage A/B 500-step sampling never saw this.
- At step 5,000 the interaction census records 0.75–2.1 copy events per step (out of ≈ 3,950 interactions) under the full ISA against 6–35 per step once an LDIR cloud exists (stack arms removed, 512 steps); 79–87% of interactions write nothing into either program.
- Emergence, when it happens, is slow at L = 9 in every arm: KM medians 82k–252k steps or not reached; the stack writers do not slow LDIR assembly so much as make it fail within the horizon.

## Mechanism: the interaction census and the nascent-copier scan (added 2026-10-08, `census/`, `census_512/`)

The question left open above — destruction of nascent copiers or occupation of the niche by sterile smears — was put to the recorded data in two ways (`census_dynamics.py`, `nascent.py`; CPU only).

*Interaction census, pre-emergence samples ≤ 20,000 steps, medians over seeds (128 steps):* copy events per 1,000 interactions `none` 0.51, `stack-read-only` 0.75, `push` 0.50, `call-rst-write` 0.25, **`stack-write-only` 0.00**; programs destroyed (≥ half their bytes changed in one encounter) per 1,000 interactions `none` 6.8, `stack-read-only` 6.6, `push` 6.3, `call-rst-write` 2.0, **`stack-write-only` 1.3**; zero bytes written into B per interaction 0.042, 0.041, 0.035, 0.023, 0.014 in the same order. At 512 steps: copy events 0.25 / 0.26 / 0.25 / 0.25 / 0.00, destroyed 7.3 / 6.7 / 6.8 / 3.0 / 2.3. Under the full ISA the pre-emergence copy events are the return-address smears copying themselves (none of them is heritable), and the soup destroys five times more programs per encounter than without the stack writers.

*Nascent-copier scan (top-10 exemplars of every sample, LDIR-bearing = contains ED B0/B8/A0/A8), before the heritable event:*

| arm (128 steps) | runs where an LDIR tape ever enters the top-10 | first entry (median step) | episodes (present, then gone) per run, median / max |
|---|---|---|---|
| `none` | 5/20 | 118,500 | 0 / 0 |
| `stack-read-only` | 5/20 | 116,500 | 0 / 0 |
| `push` | 9/20 | 130,500 | 0 / 1 |
| `call-rst-write` | 16/20 | 27,750 | 0.5 / 4 |
| `stack-write-only` | **20/20** | **3,400** | **6.5 / 18** |

At 512 steps the same ordering holds (`none` 2/20 at 103,500; `stack-write-only` 20/20 at 325 steps, 5.5 episodes). So the suppression acts **before** a nascent copier is ever visible: with the stack writers present, LDIR tapes do not reach even 0.5% of the soup for a hundred thousand steps; without them they appear within thousands of steps, flicker in and out of the top-10 several times (a birth-death process close to its threshold) and finally take over. Removing CALL/RST alone moves the first appearance from 118,500 to 27,750 steps (16/20 runs) while removing PUSH alone barely moves it (130,500; 9/20), matching the emergence counts. The mechanism is therefore niche occupation and a high per-encounter destruction rate that keep LDIR units from assembling, not the killing of established copiers.

## What this settles and what it does not

Settled: removing instructions can raise the probability that self-replication emerges, the effect replicates with independent seeds (p ≈ 10⁻⁶), it is carried by the opcodes that write through the stack pointer, chiefly the return-address writers, and the replicators that then appear are the 3-byte and 9-byte LDIR units whose periods divide the 18-byte ring. Not settled: whether the suppression acts by destroying nascent copiers (zero writes into their tails) or by niche occupation by sterile smears; the time-resolved census in Stage C/E and the snapshots can separate these (fraction of LDIR-bearing cells overwritten per step). The per-byte-hazard prediction (D5) failed and is dropped.

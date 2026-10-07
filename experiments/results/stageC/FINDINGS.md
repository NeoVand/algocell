# Stage C — long horizons, uncensored succession, finer ablations (490 runs, 2026-10-07)

Design (PLAN.md Stage C, fixed before any run): L = 16 unless stated, mutation 1/16 unless stated, **seeds 101–110**, no early stop, explicit early samples (steps 1–34, every 50 to 5,000, every 500 after; every 1,000 after 5,000 in the 1M-step runs), full snapshots and interaction census on the log schedule. **C1** `no-copy`, `rmw-only`, `all-ld` at (128 steps, 1/16) and (32, 1/4) for **1,000,000 steps**. **C2** `none`, `block-copy`, `ld-mem` × {128, 512} steps × mutation {1/4, 1/16, 1/64}, 300,000 steps. **C3** eleven finer ablations at (128, 1/16) and (32, 1/4), 300,000 steps. **C5** `none` and `stack-writes` at L ∈ {36, 100}. 490/490 runs completed, 0 failures, median 81 s per run (max 329 s). Clock: 0.395 active interactions per cell per step in every run (0.394–0.396), so 1,000 steps ≈ 395 encounters per cell; all arms share one grid, so only steps are quoted. Numbers: `NUMBERS.md`, `stage_c/NUMBERS_C.md`, `ZOO.md`, `cells.csv`, `assays.csv`, `succession.csv` in this directory.

Ablation sets (`make_conds.ABLATIONS`): `no-copy` = the six LD families + stack + EX + block copy; `rmw-only` = every memory-writing instruction except the read-modify-write ones (INC/DEC (HL), rotates and bit-sets on memory); `all-ld` = the six LD families (block copy allowed); `ld-imm` = 8- and 16-bit immediate loads; `ld-reg` = register loads + special loads; `ld-mem` = 8- and 16-bit memory loads; `cb-page` = rotates, shifts and bit operations (RLD/RRD kept); `ed-loads` = the eight ED-page 16-bit memory loads; `push-only` = PUSH and POP; `ex-sp-only` = EX (SP),HL; `call-rst` = CALL, RET, RST; `stack-writes` = stack + EX + CALL/RET + RST (46 opcodes); `stack-write-only` = PUSH, EX (SP),HL, CALL, RST (22); `stack-read-only` = POP, RET, EX DE,HL, EXX, EX AF,AF' (24).

## Pre-registered outcomes

| Test | Prediction | Result | Verdict |
|---|---|---|---|
| **C1a** | `no-copy` and `rmw-only` stay at 0/10 at 1,000,000 steps | 0/10, 0/10, 0/10, 0/10; pooled 0/40 → 95% upper bound on the per-run emergence probability within 1M steps = 0.072 | **confirmed** |
| **C1b** | `all-ld` emergence fraction rises above 0.3 | 2/10 at both budgets (at 132k, 313k, 218k, 881k steps); all four are LDIR units with POP/EX register set-up | **not met** |
| **C3a** | `stack-write-only` reproduces the `stack-writes` delay within ×2 | 8/10 vs 8/10 heritable; KM medians 76,500 vs 32,000 steps (×2.4); seed-paired write-only vs stack-writes 5 slower / 4 faster / 1 tie; both arms 10/10 slower than `none` (sign p = 0.002) | suppression **reproduced**; the ×2 bound on the KM ratio is exceeded (×2.4) but the paired test cannot separate the two arms |
| **C3b** | `stack-read-only` indistinguishable from `none` (≤ 7/10 concordant) | 10/10 heritable, KM 600 vs 200; paired 5 slower / 5 faster | **confirmed** |
| **C3c** | `ex-sp-only` and `push-only` each delay less than `stack-write-only` | `ex-sp-only` KM 700 (×3.5; 7/2/1, p = 0.18) — yes; `push-only` 6/10 heritable, KM 198,500 (×992; 10/0, p = 0.002) — no, removing PUSH/POP alone delays **more** than removing all 22 stack writers | **half met** |
| **C3d** | `ld-imm` is the load family whose removal costs most | `ld-imm` 2/10 heritable in 300k steps (10/0 slower, p = 0.002) vs `ld-reg` 10/10 (KM 150) and `ld-mem` 10/10 (KM 200) | **confirmed**, and it is the strongest single-family suppression in the atlas at 128 steps |
| **C3e** | `cb-page` and `ed-loads` change nothing | 10/10 each; paired 6/4 (p = 0.75) and 6/3/1 (p = 0.51); KM 1,300 and 450 vs 200 | **confirmed** by the paired test; the KM ratios (×6.5, ×2.2) are within the n = 10 noise |
| **C5** | dominant tapes at L = 100 have period ≤ 8 in most seeds | first heritable replicator under `none`: period 2 in 9/10 (one period 20); final faithful dominant: 0/8 (period 10 ×7, period 9 ×1); `stack-writes` first replicators ≤ 8 in 2/10 | met for the **first** replicator under `none` only |

At 32 steps · 1/4 the picture is different: `none` 9/10 (KM 68,500) and **no arm differs by the seed-paired test** (every p ≥ 0.34; `stack-writes` 5/10, `call-rst` 6/10, all others 9–10/10). `ld-imm` is 9/10 with KM 67,500 — the immediate loads do not matter in this regime, where every first replicator is a period-4 LDIR unit (`x4 5e ed b0`: LD E,(HL) ; LDIR; 4 × 9 for `none`). The 1M-step nulls are the same at both budgets (`all-ld` 2/10, `rmw-only` 0/10, `no-copy` 0/10). Stage E (E7, 20 new seeds) confirms that in this regime removing block copy abolishes emergence entirely (0/20 vs 18/20).

## The atlas at L = 16, 128 steps, 1/16: a pusher regime with three load-bearing parts

In every arm that permits it, the first heritable replicator is the 2-byte unit **`LD rr,nn ; PUSH rr`** (`01 c5`, `11 d5`, `21 e5`: the 16-bit immediate *is* the unit, so each PUSH writes one more copy of it below the stack pointer, i.e. into the partner's tail). It is the first replicator in 10/10 runs of `none`, `block-copy` and `ld-mem` (ZOO.md), byte-identical `01 c5 ×8` in 8/10 `none` runs. It needs a 16-bit immediate load and PUSH, and nothing else. Consequently the atlas has exactly three strong entries:

- **immediate loads** (`ld-imm`): 2/10 heritable within 300k steps; the two replicators are the period-4 LDIR unit (`84 5e ed b0`, `a4 5e ed b0`) at 168,000 and 205,500 steps. The eight unresolved soups stay flooded: zero fraction 0.307 at 300k (0.013–0.422), high-order entropy 0.67 bits/byte.
- **the stack writers**: `push-only` ×992 (6/10), `stack-write-only` ×382 (8/10), `stack-writes` ×160 (8/10). Remove PUSH and emergence waits for an LDIR unit of period 4 or 8 (periods 4×3 + 8×3; 4×3 + 8×5; 4×4 + 8×4). The delay is longest when the other stack writers remain (`push-only` keeps CALL, RST and EX (SP),HL): the return-address-writer suppression that Stage D quantified at L = 9 (CALL/RST removal p = 4.5 × 10⁻⁴) reappears at L = 16 once the pusher is taken away.
- **copying itself**: `no-copy` and `rmw-only` 0/40 at 1,000,000 steps.

Everything else is within the seed-paired noise of `none` (KM ratio; paired slower/faster): `ld-reg` ×0.75 (5/5), `ld-mem` ×1.0 (4/4/2), `call-rst` ×1.25 (6/3/1), `block-copy` ×1.75 (6/4), `ed-loads` ×2.2 (6/3/1), `stack-read-only` ×3.0 (5/5), `ex-sp-only` ×3.5 (7/2/1), `cb-page` ×6.5 (6/4); no p < 0.18. Every first replicator in Stage C at L = 16 has period 2, 4 or 8. The occupancy event `tq_10` inverts the result again (`ld-imm` 10/10 by the flood, `stack-write-only` 1/10), as in Stage D.

## Succession (C2): the mutation rate selects which mechanism wins

From the census family of the dominant unit at 5k / 50k / 300k steps (`succession.csv`, families: `push` = LD rr,nn ; PUSH rr; `ex_sp` = EX (SP),HL-based; `ldir` = block copy):

- At 1/16 and 1/64 (128 and 512 steps) the pusher comes first (`push` dominant at 5k in 9–10/10 runs at 512 steps) and is replaced by the **EX (SP),HL family** by 50k steps; at 300k `ex_sp` is dominant in 8–9/10 `none` runs (`ex_sp:9, ldir:1` at 128·1/16; `ex_sp:8, ldir:2` at 128·1/64; `ex_sp:9, ldir:1` at 512·1/16). The byte-identical final dominant in 9/10 `none` runs at 128·1/16 is `ad e3 21 e3 21 c0 ad c0 ×2` (XOR L ; EX (SP),HL ; LD HL,nn ; RET NZ; period 8). LDIR takes over in 1–2/10 runs.
- At 1/4 the outcome depends on the budget: at 32 and 128 steps **LDIR displaces everything** (`ldir:9` and `ldir:8` at 300k; take-over in 9 and 8 runs with medians 64,000 and 63,750 steps; LDIR share 0.84 and 0.80), while at 512 steps the pusher holds for 300k steps (`push:10`) with a persistent zero load (0.13).
- Where LDIR is impossible (`block-copy` at 128·1/4) the soup never leaves the flooded state in 6/10 runs (`none:6, ex_sp:4` at 300k; zero fraction 0.229).
- Emergence time is nearly independent of the mutation rate under `none` at 128 steps (KM 200, 200, 650 steps at 1/64, 1/16, 1/4) but the functional fraction of the final population falls with it (random-cell replicator fraction 0.97, 0.84, 0.70 at 128 steps; 0.42, 0.77, 0.34 at 512).

## First replicator versus final dominant: succession selects for fidelity (exploratory)

The modal first tape and the modal faithful final tape of the `none` runs, assayed in isolation at 128 steps against 64 random partners (`stage_c/first_vs_final_none.csv`):

| L | first replicator | seeds byte-identical | gen2 | self bytes changed / encounter | final dominant | seeds byte-identical | gen2 | self bytes changed |
|---|---|---|---|---|---|---|---|---|
| 16 | `01 c5 ×8` (period 2) | 8/10 | 0.59 | 4.5 | `ad e3 21 e3 21 c0 ad c0 ×2` (period 8) | 9/10 | 1.00 | 0.0 |
| 36 | `01 c5 ×18` (period 2) | 7/10 | 0.64 | 8.1 | pusher with `10 f0` (DJNZ −16) inserted (period 14) | 7/10 | 0.83 | 0.0 |
| 100 | `01 c5 ×50` (period 2) | 9/10 | 0.72 | 1.8 | `c2 c5 01 c5 01 c5 01 c5 c2 c5 ×10` (JP NZ,nn in place of LD BC,nn; period 10) | 7/10 | 0.99 | 2.0 |

Both write the whole partner (15.6 vs 15.9, 34.1 vs 35.8, 92.0 vs 98.7 partner bytes changed), so the successor is not a faster copier; it is a **more heritable** one (gen2 0.59 → 1.00, 0.64 → 0.83, 0.72 → 0.99), and at L = 16 and 36 it no longer damages itself during an encounter. The successors are byte-identical across 7–9 of 10 independent seeds: the same 36-byte and 100-byte tapes evolve from different random soups. This is the same succession Stage B saw through the early stop; here it is uncensored.

## C5: L = 36 and L = 100 without censoring

`none`@36: 10/10 heritable at KM 350 (period 2 first), all faithful at 300k; final dominants period 2 ×3, 14 ×7. `none`@100: 10/10 at KM 1,200; 9/10 faithful at emergence, 8/10 faithful at 300k; final periods 10 ×7, 9 ×1 (an LDDR-based unit, `b8 16 b6 17 21 41 67 63 ed`), plus one unfaithful period-25 LDIR unit and one unfaithful period-4 unit. `stack-writes`@36: 10/10 heritable (KM 12,000), first periods 4–12 (all divide 72; 7/10 divide 36), final dominants period 4 ×6, 6 ×2, 8 ×2, all faithful. `stack-writes`@100: 10/10 heritable (KM 8,500) but 1/10 faithful; first periods 5, 8, 20 ×4, 25 ×3, 76; 9/9 tiled periods divide 200.

## Tiling and the ring

Every tiled first replicator in Stage C has a period that divides 2L (`divides_2L` = 1.00 in all 33 cells; periods 8 at L = 36 and L = 100 divide 72 and 200 but not 36 and 100). The offset form of the law (period = gcd(copy offset, 2L)) cannot be evaluated here: the modal copy offset of these tapes is 0.

## The zero flood

Under `none` at 128 steps the zero fraction is 0.281 at the occupancy crossing and 0.020 at 300k (median 0.004); `stack-write-only` and `stack-writes` sit at 0.006–0.011 throughout. Without a replicator it persists: `no-copy` 0.247 (128 steps) and 0.193 (32 steps) after 1,000,000 steps; `ld-imm` 0.307 at 300k. `rmw-only` stays at the random level (0.004 ≈ 1/256; byte entropy 7.996–7.999 bits): the soup never leaves its initial state.

## What this settles and what it does not

Settled: at L = 16 and 128 steps the dependence of emergence on the instruction set is concentrated in three places — the 16-bit immediate loads, the stack writers, and copying itself — and every other family tested (register and memory loads, the ED-page loads, the CB page, CALL/RET/RST, EX (SP),HL alone, the stack readers) is within the seed-paired noise; the write side of the stack carries the suppression and the read side carries none (C3a, C3b); `no-copy` and `rmw-only` are not slow but absent to 1M steps (upper bound 7% per run); the mutation rate decides which family holds the soup at 300k (LDIR at 1/4 and ≤ 128 steps; the EX (SP),HL family at 1/16–1/64); and succession runs toward heritability, converging on byte-identical tapes across seeds. Not settled: why removing PUSH/POP alone delays more than removing all stack writers (candidate: the residual CALL/RST suppression of Stage D; test: a `push + call-rst` arm); why the pusher is not viable at 32 steps (Stage E's isolated-unit sweep addresses this); C4 (functional fraction over time) is recorded (16 random tapes per sample, snapshots) but not yet computed. The `all-ld` prediction (C1b) failed: without the LD families LDIR copiers do assemble, but in 2/10 runs per million steps.

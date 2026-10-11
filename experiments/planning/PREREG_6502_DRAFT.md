# DRAFT — second-machine test on the NMOS 6502 (not registered)

Status: draft written 2026-10-10, before any 6502 soup exists and before the 6502 core is validated. It becomes a
registration only when copied into `REVISION_PREREG.md` with its census recomputed on the validated Zilion 6502 core
(SingleStepTests/65x02 `6502/v1`, data-only vectors) and committed before any soup is run. Anything below may change
until then; changes will be dated.

## Why this machine

The paper's theory predicts the first replicator from the substrate's short self-writers (Theorem 6, Proposition 3,
the BFF write-ratio arm). An exploratory census on an unvalidated interpreter (`planning/ISA_RESEARCH_2026-10-10.md`
§4.1) found no straight-line self-writer of period 1–4 on the NMOS 6502 at L = 16 (4.29 × 10⁹ tilings), against
5, 1 and 641 on the Z80 at periods 2, 3 and 4, and 6-byte closed stack-copy loops whose zero-page operand sets the copy
period (a Proposition 8 analogue, d − 6 transmitted sites). The 6502 is therefore the clean opposite of the Z80: the
theory predicts a loop-bearing, closed beginning, if replication begins at all.

## Planned design (to be fixed at registration)

- Machine: NMOS 6502 core of Zilion (validated per instruction against SingleStepTests 6502/v1; JAM opcodes halt;
  `BRK` faithful, with a `BRK`-halts arm as the analogue of lethal tar; NMOS decimal mode; unstable opcodes as in the
  vectors).
- World: the shared Zilion world, identical to the Z80 runs: 160 × 125 lattice, 4-neighbour pairing, 8,192 draws per
  step, pair memory a ring of 2L bytes, organism first, A = X = Y = 0, P = 0x20, PC = 0, S such that the first push
  lands on the last byte of the partner, 128 instructions per encounter (and a 512-instruction arm, since a minimal
  6502 copy loop spends about four instructions per byte), background mutation 8,192/2^k bytes per step, k = 4.
- Lengths: L = 16 and 32. Seeds: 20 per arm. Horizon: 1,000,000 steps (pilot first; see below).
- Assays: the same culture test, partner tests, traced execution (pointer confinement, inflow), single-mutant scan,
  serial and branching transfer, population classes, as frozen for the Z80 (Zilion world, same thresholds).

## Planned predictions (from the census, to be recomputed on the validated core)

- P1 (form): in at least 0.8 of worlds that produce a heritable replicator, the first one carries a loop instruction
  (branch back or `JMP`/`JSR`/`RTS`/`RTI` returning into the organism) and is confined (pointer never in the partner in
  16 of 16 traced encounters).
- P2 (no open phase): no world has an open straight-line replicator holding ≥ 0.5% of cells before its first closed one.
- P3 (heredity from the start): among heritable cells at the end, both regenerators (≤ 2 sites) and transmitters
  (≥ 5) occur across worlds, and the class of a closed core tracks its copy-offset operand (d − 6 sites).
- P4 (rarity): the first heritable replicator arrives later than on the Z80 at the same L (Kaplan–Meier median > 450
  steps at L = 16), or not at all.
- Null outcome: if fewer than 5 of 20 worlds produce a heritable replicator within the horizon, the test is reported
  as uninformative about the form of the beginning (but informative about rarity).
- Kill (theory): if at least 0.25 of first replicators are open straight-line copiers, the claim that the census
  predicts the beginning is withdrawn for the 6502.

## Before registration

1. Validated core (Zilion M2) and world identical to the Z80 world (Zilion M1 equivalence proof).
2. Census of periods 1–4 (and seeded closers) recomputed on the validated core, on GPU, with the same harness.
3. Local pilot: 4 worlds × 100,000 steps at L = 16 and 32 to size the horizon and the spend; the pilot is reported
   and its worlds excluded.
4. Spend estimate and approval before any Modal sweep.

## Predictions locked (2026-10-11, before any 6502 soup was run)

**Census basis.** The census is now recomputed on the validated core: `results/census6502/CENSUS.md`, merged at
b5fe4f7. Its period-4 BRK-halt tail ran on Modal after a cross-platform identity check (`MODAL_TAIL.md`). Findings:

- **Same harness, positive control.** On the Z80 it reproduces the paper's census word for word: 0, 5, 1 and 646 at
  periods 1–4, L = 16.
- **No straight-line self-writer on the 6502.** At periods 1–4 (L = 16) and periods 1–3 (L = 20 and 32) there are 0,
  in all four arms: BRK faithful or halting, 128 or 512 instructions.
- **Open stack copiers are weak.** They copy exactly into a minority of random partners: a median of 1 in 1,024 fresh
  partners, at best 36 (BRK faithful) or 55 (BRK halting).
- **No closed period-4 word copies exactly.** The best reach 87.5% (14 of 16 bytes).
- **Seeded closed copy loops copy exactly.** Their class follows the copy-offset operand: the d − 6 law holds in
  296 of 296 qualifying cases.

**P1–P4, the null outcome and the kill rule stand as written above.** Two clarifications fix them before any soup
exists:

- (a) An "open straight-line replicator" in P2 and the kill rule is a heritable replicator (by the frozen culture
  test) for which both hold: its traced execution never transfers control backwards into its own bytes, and its
  pointer enters the partner in at least one traced encounter. The census's open stack copiers count if they become
  heritable replicators in a soup.
- (b) These predictions were fixed from the census alone. No 6502 soup had been run when they were written.

**The pilot sizes the design; it cannot touch the predictions.**

- The pilot is 4 worlds per length (8 in all), each 100,000 steps, at L = 16 and 32, with 128 instructions.
  `pilot6502.py` runs it, on Modal.
- It sets only the horizon and the spend estimate. It cannot change P1–P4, the null outcome, the kill rule or the
  analysis.
- Its report holds only steps per second and population-concentration trajectories: the share of the most common
  tape and the quasispecies share.
- No replicator classification (open/closed, loops, heredity) is run on pilot soups.
- The pilot worlds are excluded from the registered test.

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

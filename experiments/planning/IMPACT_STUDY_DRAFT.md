# DRAFT — emulator impact study (not registered)

Status: draft written 2026-10-10, before the fixed Zilion Z80 core exists and before any comparison is run. It becomes
a registration when it is copied into `REVISION_PREREG.md` with the three cores' SHA-256s filled in and committed,
before any corpus is run under the fixed cores. Changes until then are dated here.

## Why

Every registered result ran on the Zilion 0.2.0 Z80 core. That core has now been checked against all 1,604,384
SingleStepTests/z80 vectors (zilion `test/baselines/z80-v0.2.0.md`, ledger SHA-256 `6f959567b2e6507b…`). Every
divergence falls in a documented class, and none is unexplained. Four of the classes can change a soup's bytes:

- **R after a wasted DD/FD prefix.** The core gives R + 2, where the hardware gives R + 1. R reaches a register
  through `LD A,R`, and from there it can be stored anywhere.
- **F3/F5 (X/Y) of `BIT n,(HL)`.** The hardware takes them from MEMPTR; Zilion takes them from the operand.
- **F3/F5 of `SCF`/`CCF`.**
  - On a Zilog NMOS Z80 they are ((Q xor F) or A) & 0x28; Zilion 0.2.0 uses A & 0x28.
  - The difference is vendor-specific: it is Zilog NMOS behaviour, and the design keeps it as an opt-in variant.
  - The ring difftest's reference (Fuse lineage) uses Zilion's formula, so it could not see this class.
- **F3/F5 of a repeating block instruction** (`LDIR`, `LDDR`, `CPIR`, `CPDR`, and the I/O repeats with H and P/V).
  - On the hardware, an iteration that repeats takes X/Y from PC bits 13 and 11; Zilion 0.2.0 keeps its
    non-repeating formula.
  - Normally the next iteration overwrites F, so the difference vanishes.
  - It survives when the repeat stops being executed. That happens if the instruction overwrites its own `ED B0`
    bytes mid-repeat, as a self-copying `LDIR` can. The next fetch is then another instruction, which runs with the
    iteration's F. (Note added 2026-10-10, from the Z80 fixes; an earlier version of this draft wrongly called this
    class unobservable.)

A flag difference reaches memory only through `PUSH AF`, or `EX AF,AF'` followed by `PUSH AF`. No conditional
tests F3/F5, so control flow is never directly affected.

The other classes cannot change a soup's bytes:

- IM, MEMPTR itself, Q itself and the EI/P latches are internal, and a soup has no interrupts.
- I/O is the environment's choice, not the CPU's: a soup has no device, so `IN` reads 0 in every core compared
  here. It is not compared.

## Cores (identical world; only the CPU differs)

| | Core | What differs from C0 |
|---|---|---|
| C0 | Zilion 0.2.0, as used: the `compat:'0.2'` machine, proven byte-identical to Algocell's shaders (zilion `check:bytes`, `test:algocell`) | — |
| C1 | Zilion 0.3 default Z80 core (`compat:'0.3'`, classic) | R + 1 per wasted prefix; MEMPTR, so `BIT n,(HL)` X/Y from WZ; IM; block-repeat X/Y (and H, P/V for the I/O repeats) |
| C2 | C1 with variant `zilog-nmos` | adds the Q latch: SCF/CCF X/Y = ((Q xor F) or A) & 0x28 |

The world is the Algocell world of each condition, as registered:

- segments [L, L] on a ring of P bytes;
- stack at the last byte;
- 128 instructions;
- lethal rule, dial and ablation masks as the condition had them.

Each core runs as a Zilion machine on the same GPU, with the same inputs.

## Corpora

1. **Random programs.**
   - Ring-difftest sizes: P = 32, 38, 40, 64, 100 and 128, with fresh uniform random pairs, 100,000 per P at
     seed 1.
   - Stage F's odd and padded rings, which no difftest covered: P = 33, 35, 37, 73, 75 and 79, with each L as in
     Stage F, 100,000 per P at seed 1.
2. **Archived soups.** These are the inputs that matter, because evolved tapes are not random.
   - Every registered condition with archived final soups (`*.soup_final.u8.br`): Stages A–F, the line-of-descent
     runs (lod_v6, N6) and the X1–X3 inputs.
   - 100,000 lattice-neighbour pairs per condition, drawn uniformly with replacement over its worlds and cells
     (seed 1).
   - Each pair runs as (A, B) in that condition's world.
   - Where emergence snapshots are archived, the same is done for them.
3. **Assays.** The replication assay (`assay.py`) on every tape the manuscript names, under C0, C1 and C2.

## Measures

- **Affected-pair rate.** The fraction of pairs whose final pair memory (both tapes) differs from C0, with a 95%
  Clopper–Pearson interval. It is computed per corpus and condition, separately for C1 and C2.
- **Attribution.** Each affected pair is attributed by `startMap` and a single-instruction replay to the first
  instruction where the cores' states diverge. That instruction is classed as wasted prefix + `LD A,R`,
  `BIT n,(HL)`, `SCF`/`CCF`, a repeating block instruction, or other. An "other" is a bug, and it stops the study.
- **Assay identity.** Whether every named tape's assay scores are identical under C0, C1 and C2.

## Decision rule (as in the design, §5.7)

- **C1.**
  - If the affected-pair rate is below 1e-4 in every archived-soup condition and every assay is identical, the
    results stand. The Methods then names the core, its documented deviations and the measured rates.
  - Otherwise the affected conditions are flagged. The paper then reports which claims rest on them.
- **C2.** The rates are reported. A rate above 1e-4 is expected to be possible, since SCF and CCF are common bytes.
  It is not by itself a reason to re-run, because the Q latch is one vendor's behaviour. It does flag the
  condition for the second tier.
- **Second tier, only for flagged conditions, and only with explicit spend approval.** A confirmatory replication
  of the flagged headline results under the fixed core: N6-style, 20 worlds, scored by the registered N6 criteria.

## Before registration

1. The fixed core merged into zilion `next`. The SHA-256s of C1 and C2 then go into the table above.
2. A generic Algocell runner for Zilion machines: pairs in, final pair memory and maps out. It must reproduce C0's
   outputs bit for bit on the archived pairs against Algocell's own shaders. That is the equivalence check that
   lets the comparison use one runner for all three cores.
3. A dry run of 1,000 pairs per corpus to size the run and check the attribution code. The dry run is reported,
   and its pairs are excluded.

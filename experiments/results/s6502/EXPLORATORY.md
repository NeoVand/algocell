**EXPLORATORY (NOT REGISTERED)**

# Exploratory S1 ring-position follow-up

This examination implements the exploratory hypothesis declared at the end of `experiments/REVISION_PREREG.md`. Registered verdicts are final: P1 remains not supported; P3 remains not tested. All analyses and tables here are exploratory.

Exploratory hypothesis result: **holds for 6 of 6** first replicators without a registered PC decrease. Every one uses a taken forward branch to a higher 16-bit alias of an earlier organism position. No BRK cycle is needed to explain these six.

## Exploratory method and audit

Replayed 20 first replicators × 16 encounters = 320 original pairs, with reset plus budgets 1–128 (41,280 recorded prefix states). Each prefix is an independent replay of its initial pair, as `executor.pc_trace` does, using the unchanged vendored `m6502_pair_maps_L16`. All recorded registers, memory, fetch/start maps and the opcode at the pre-instruction state are saved; terminal/halted rows have `executes_next=False`.

Read only committed `s1_worlds.csv`, `s1_cells.csv` and the scorer's committed `judged_tapes` tables in `worlds/*.json`. The ordered cache is replayed through frozen `score6502.judge_tape` from `default_rng([seed, 0, 11])` through each first tape (49 cache misses). The scorer itself consumes the assay seed and draws the uniform-byte partners. Every judgment matches its committed row; all 20 registered loop calls and all 320 confinement checks match. No raw run/soup or Modal volume file was read or fetched. The trace partners are uniform random bytes, as the scorer actually implements them.

Position mapping comes from the vendored shader's `zil_pos`: `position = (PC & 0xFFFF) % 32`. For P = 32 this equals zilion's seam mapping: 32 divides the 65,536-address space. An exploratory position return is an executed transition to a strictly lower ring position in organism A (0–15). Prefix adjacency alone after a halt is not an instruction. Categories are exclusive, prioritising (i), then (ii), then (iii); the tables also retain all mechanisms seen in each encounter.

Mnemonics and lengths were read by Node directly from `m6502.table()` in zilion `next`, `/Users/neo/repos/zilion/dist/cores/m6502/index.js`; addressing modes come from that module's `M6502_OPCODES`. No mnemonic is inferred from memory. The short disassembly below reads the initial bytes at observed instruction-start positions, including overlapping decodes, collapsed onto the 8-byte block that every tape repeats twice. An operand can come from the second block. Branch targets in that listing assume the shown initial PC. Witness targets use the actual PC and mutable memory. The full disassembly table also marks positions reached as instruction starts, including overlapping decodes.

## Exploratory classification

| Exploratory category | First replicators |
|---|---:|
| (i) Earlier organism position via registered PC decrease | 14 |
| (ii) Earlier organism position via higher alias | 6 |
| (iii) BRK vector cycle | 0 |
| (iv) No earlier-position return | 0 |
| (v) Other | 0 |

The witness is the first eligible event in encounter order, then prefix order. Counts after the witness are the encounters (out of 16) with any earlier-position return.

| Exploratory seed | Category | Responsible instruction | Witness PC → PC (ring → ring) | Encounters |
|---|---|---|---|---:|
| 65001 | (i) | BVC -38 ($FFC0, ring 0) | FFE4 → FFC0 (4 → 0) | 16 / 16 |
| 65002 | (ii) | BCS +56 ($0042, ring 2) | 0008 → 0042 (8 → 2) | 16 / 16 |
| 65003 | (i) | JMP $C8E1 | C8E7 → C8E1 (7 → 1) | 16 / 16 |
| 65004 | (ii) | BPL +88 ($0060, ring 0) | 0006 → 0060 (6 → 0) | 16 / 16 |
| 65005 | (i) | BCC -39 ($FFC0, ring 0) | FFE5 → FFC0 (5 → 0) | 16 / 16 |
| 65006 | (ii) | BPL +25 ($0020, ring 0) | 0005 → 0020 (5 → 0) | 16 / 16 |
| 65007 | (i) | JMP $C700 | C704 → C700 (4 → 0) | 16 / 16 |
| 65008 | (i) | JMP ($6FA0) | 6FA7 → 6FA0 (7 → 0) | 16 / 16 |
| 65009 | (ii) | BCC +56 ($0040, ring 0) | 0006 → 0040 (6 → 0) | 16 / 16 |
| 65010 | (i) | JMP ($68C4) | 5286 → 5280 (6 → 0) | 16 / 16 |
| 65011 | (i) | JMP ($0006) | 0004 → 0000 (4 → 0) | 16 / 16 |
| 65012 | (i) | JMP ($00E6) | 0004 → 0000 (4 → 0) | 16 / 16 |
| 65013 | (ii) | BNE +57 ($0042, ring 2) | 0007 → 0042 (7 → 2) | 16 / 16 |
| 65014 | (i) | JMP $C982 | C986 → C982 (6 → 2) | 16 / 16 |
| 65015 | (i) | JMP $0000 | 0005 → 0000 (5 → 0) | 16 / 16 |
| 65016 | (i) | JMP $D200 | D205 → D200 (5 → 0) | 16 / 16 |
| 65017 | (i) | JMP $4302 | 4306 → 4302 (6 → 2) | 16 / 16 |
| 65018 | (ii) | BPL +56 ($0040, ring 0) | 0006 → 0040 (6 → 0) | 16 / 16 |
| 65019 | (i) | JMP $5900 | 5904 → 5900 (4 → 0) | 16 / 16 |
| 65020 | (i) | JMP $FD00 | FD04 → FD00 (4 → 0) | 16 / 16 |

Exploratory detail: 0 BRK instructions executed across these 320 encounters. For seed 65002, the BCS at reset PC 0 is not taken (carry clear); SEC at position 7 sets carry, and the BCS at PC 8 then jumps to 66 (ring 2). The registered-loop comparison tapes can also have higher-alias returns; category (i) retains their registered PC-decrease classification.

### Exploratory short disassemblies of all 20 tapes

| Exploratory seed | Initial bytes at observed starts (8-byte block repeated twice) |
|---|---|
| 65001 | `00: PLA; 01: STA $48,X; 03: INX; 04: BVC -38 ($FFE0, ring 0)` |
| 65002 | `00: BCS +56 ($003A, ring 26); 02: DEX; 03: LDA $AA,X; 05: STA $02,X; 07: SEC` |
| 65003 | `00: SBC ($C8,X); 01: INY; 02: BPL +0 ($0004, ring 4); 04: PLA; 05: STA ($61),Y; 07: JMP $C8E1` |
| 65004 | `00: PLA; 01: NOP #$94; 03: STA ($00),Y; 05: INY; 06: BPL +88 ($0060, ring 0)` |
| 65005 | `00: PLA; 01: STA $02E8,X; 04: INX; 05: BCC -39 ($FFE0, ring 0)` |
| 65006 | `00: PLA; 01: STA $12C8,Y; 04: INY; 05: BPL +25 ($0020, ring 0)` |
| 65007 | `00: LDA $E7,X; 02: DEX; 03: PHA; 04: JMP $C700` |
| 65008 | `00: LDY #$6F; 02: STA $B068,X; 05: PLA; 06: TSX; 07: JMP ($6FA0)` |
| 65009 | `00: DEY; 01: LDA $0048,Y; 04: CLD; 05: PHA; 06: BCC +56 ($0040, ring 0)` |
| 65010 | `00: PLA; 01: STA ($80),Y; 03: INY; 04: NOP #$52; 06: JMP ($68C4)` |
| 65011 | `00: PHA; 01: LDA $06,X; 03: DEX; 04: JMP ($0006)` |
| 65012 | `00: PHA; 01: TSX; 02: LDA $A8,X; 04: JMP ($00E6)` |
| 65013 | `00: AND $CAD9,Y; 02: DEX; 03: LDY $6A,X; 05: STY $02,X; 07: BNE +57 ($0042, ring 2)` |
| 65014 | `00: CMP #$76; 02: PLA; 03: STA ($42),Y; 05: INY; 06: JMP $C982` |
| 65015 | `00: DEY; 01: PHA; 02: LDA $4B87,Y; 05: JMP $0000` |
| 65016 | `00: PLA; 01: STA $4C28,X; 04: INX; 05: JMP $D200` |
| 65017 | `00: SRE ($6A,X); 02: PLA; 03: STA $28,X; 05: INX; 06: JMP $4302` |
| 65018 | `00: EOR ($00),Y; 02: PLA; 03: STA ($42),Y; 05: INY; 06: BPL +56 ($0040, ring 0)` |
| 65019 | `00: PHA; 01: TSX; 02: LDA $68,X; 04: JMP $5900` |
| 65020 | `00: INX; 01: STA $A6,X; 03: PLA; 04: JMP $FD00` |

## Exploratory P3 diagnosis

The distribution below is the frozen `copy_offset` column, with a replay using the same culture partners. It does not reinterpret d or change eligibility.

| Exploratory scanned population | d | Cells | Regenerators | Transmitters | Open |
|---|---|---:|---:|---:|---:|
| all scanned heritable | 0 | 304 | 193 | 94 | 17 |
| all scanned heritable | undefined | 16 | 16 | 0 | 0 |
| confined | 0 | 287 | 193 | 94 | 0 |
| confined | undefined | 16 | 16 | 0 | 0 |
| open | 0 | 17 | 0 | 0 | 17 |
| open | undefined | 0 | 0 | 0 | 0 |

Exploratory assay replay matched all 320 registered scanned-cell offsets. The final-cell assay draws were reconstructed from `default_rng([seed, 23])`: the 64-cell index choice out of the registered 20,000 cells, then the assay seed. The committed scanned tapes share its same 32 uniform partners, so a soup snapshot is unnecessary.

`copy_offset` is defined by the unchanged assay: for each first-generation partner, choose the best cyclic shift of the organism tape; retain partners with best similarity ≥ 0.75; return the modal retained shift, or `None` if none qualify. Similarity ties keep the first shift (ascending from 0); modal ties keep the smallest shift. Thus d is a cyclic alignment of offspring bytes, not a load/store address distance or the size of a control-flow jump. Measured d = 0 means alignment at zero, and alone does not mean a whole-tape exact copy.

Of 304 cells with a defined exploratory replay offset, every modal offset is 0. Every qualifying first-generation partner also has best shift 0 under the frozen tie rule. The 16 undefined cells are from seed(s) 65004. Each has 0 qualifying first-generation partners out of 32, while its gen2 score is 0.545182–0.553906 (heritable at ≥ 0.3). Heritability uses normalised second-generation gain and therefore does not guarantee a first-generation ≥ 75% match needed to define d.

Exploratory diagnosis: P3's class-occurrence part was supported, but its offset-equality part had no eligible denominator. Every measured confined d is 0 (< 6), and the remaining confined values are undefined. The registered `sites = d − 6` check for measured d ≥ 6 could not be tested; no offset or eligibility criterion is redefined.

## Exploratory artifacts and provenance

- Exploratory opcodes: [exploratory_opcodes.csv](exploratory_opcodes.csv).
- Exploratory draws: [exploratory_draws.csv](exploratory_draws.csv).
- Exploratory partners: [exploratory_partners.csv](exploratory_partners.csv).
- Exploratory steps: [exploratory_steps.csv](exploratory_steps.csv).
- Exploratory encounters: [exploratory_encounters.csv](exploratory_encounters.csv).
- Exploratory classes: [exploratory_classification.csv](exploratory_classification.csv).
- Exploratory disassembly: [exploratory_disassembly.csv](exploratory_disassembly.csv).
- Exploratory offsets: [exploratory_offsets.csv](exploratory_offsets.csv).
- Exploratory distribution: [exploratory_offset_distribution.csv](exploratory_offset_distribution.csv).
- Exploratory provenance: [exploratory_provenance.csv](exploratory_provenance.csv).
- Exploratory zilion_branch: `next`, `183ff5c7429403439c664487980a40bf62f49b7b`.
- Exploratory adapter: `Apple M4 (Metal, IntegratedGPU)`.
- Exploratory foreground_gpu_seconds: `1.658`.

Reproduce the exploratory analysis with `experiments/explore_s1_6502.py`; `--report-only` regenerates the derived tables and this report from the saved CSVs without GPU work. The foreground replay is followed by an equal pause. Registered inputs and scorer sources are checked against git and their committed scorer hashes, and their SHA-256s are recorded.

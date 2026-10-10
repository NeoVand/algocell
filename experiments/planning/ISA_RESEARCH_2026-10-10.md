I've finished the research, but I could not write REPORT.md: the harness refuses report files from a subagent, so the full report is below for you to save as-is. I recommend the **Intel 8086/8088** as the primary machine (predicted open start) and the **NMOS 6502** as the contrast (predicted loop-bearing start).

Two things matter most:
- **The 8086 may not repeat the Z80 sequence under the standard harness.** It has period-2 "one-pass" open copiers that fill the whole partner before the pointer gets there. The paper's own BFF ×2/×3 result says such copiers keep the open form alive with no inflow and no closure. With general registers randomised (the paper's C7 arm), those copiers disappear and only Z80-like literal pushers remain. That gives two pre-registrable predictions on one machine.
- **The 6502 has no self-writer at all up to period 4.** It may also fail to start replicating within the compute budget, so it needs a pilot run first.

## Summary

- **Primary: Intel 8086/8088 (real mode, 8088 semantics).**
  - It is the only candidate whose test vectors were recorded from real silicon. That fits your rule to validate against a real reference.
  - It has the paper's literal channel: `MOV r16,imm16 ; PUSH r16` (e.g. `B9 51`), the counterpart of `01 c5`.
  - It has `LDIR`-like closers. `POP ES ; DEC CX ; REP MOVSW` (`07 49 F3 A5`, 4 bytes) and `MOV DI,d ; DEC CX ; REP MOVSB` (6 bytes, copy offset set by an operand) give a Proposition 8 analogue with d − 6 transmitted sites.
- **Contrast: NMOS 6502.**
  - Exhaustive census, periods 1–4, at L = 16, about 4.3 × 10⁹ tiles: **zero self-writers**, under both treatments of `BRK`. The Z80 has 5, 1 and 641 at periods 2, 3 and 4.
  - Shortest replicators I found: 6-byte closed stack-copy loops with a one-byte offset switch. Each copies exactly into 64 of 64 random partners, never enters the partner, and carries d − 6 sites.
  - An independent, unreviewed 6502 soup (vicgalle) finds replicators about 100× rarer than on the Z80. In it, a single niche never started replicating; a 32-niche grid started at about 61,000 epochs, with a loop replicator and no open phase.
- **Write ratio decides whether closure follows.** Write ratio means bytes written per byte executed. Every Z80 period-2 self-writer has ratio 1/2. Of the 8086's 29 period-2 self-writers, 10 have ratio 1, for example `PUSH [BX+SI]` (`FF 30`) or `LODSW ; PUSH AX`. Over 64 random partners each of those produced a single offspring (zero inflow) and an exact copy every time. By hand, the 68000, PDP-11, MSP430, Thumb and 65816 have similar one-pass copiers. So any non-8080 machine is likely to change the beginning rather than replicate it.
- **Alternatives:** the 68000 as a non-Intel open primary (period-2 `MOVE.W #imm,-(A7)`, two JSON vector sets, but 2–3× the code and no one-instruction block move). The 65816 (6502 plus `PEA`/`PEI` and `MVN`) could later serve as a same-family "add the literal" test. I advise against PDP-11 (zero is `HALT`, closed one-word self-writers exist, no vectors) and RISC-V (zero is illegal, only 4.5% of random words are valid, no straight-line self-writer; likely never starts, like SUBLEQ).
- **Effort:** a validated 6502 core is about 450–550 WGSL lines and 1–2 weeks; a validated 8086 core is about 1,500–2,000 lines and 3–5 weeks.

**Caveat on all computed results:** both interpreters are my own code. The 6502 core passed 30,000 Python-against-C cross-checks with every opcode exercised, plus hand checks, but it has not been checked against vectors. The 8086 interpreter is partial: 14% of period-2 tiles were left unevaluated.

---

# Full report: a second, undesigned instruction set for the open → closed → regenerator/transmitter test

## 1. What the theory needs, and one criterion it adds

From the manuscript (Theorem 6, Proposition 3, Lemma 7, and the BFF write-ratio arm):

- **H1 (von Neumann, default flow).** Code and data share memory, and execution runs on past the organism.
- **H3 (self-consistent literal write).** One or two instructions write their own operand bytes at an advancing pointer. Then the shortest self-writers are straight-line, hence open.
- **Tar semantics.** Lethal tar removes the open start.
- **Short closers.** Without them, closure is slow or absent.
- **Write ratio (from the paper's BFF arm).** When the literal writes two or three copies (ratio 4/3 or 2, so the copy completes in one pass), "the open form persists for 16,384 epochs in 12 of 12 soups, with zero inflow from birth."
  - All five Z80 period-2 self-writers have ratio 1/2. The pointer always executes unwritten partner bytes, which is where the inflow comes from.
  - So the Z80 sequence (open window, then closure) is predicted only where the shortest straight-line self-writers have ratio below 1. If a one-pass copier of the same length exists, the predicted outcome is a persistent open form with no inflow.

I used this criterion to rank the candidates.

## 2. Candidate survey

"Census" marks results I computed (my code); "hand" marks hand-derived results that I did not simulate.

| ISA | Encoding; random-code validity | Zero byte or word | Shortest straight-line self-writer (ratio) | One-pass open copier? | Block move or closer | Halt / illegal | Predicted start |
|---|---|---|---|---|---|---|---|
| **Z80** (reference) | byte + prefixes; about 100% execute | NOP | `01 c5`, period 2 (1/2) | none at period 2 | `LDIR`/`LDDR`; 3–4-byte closers | `HALT` 76 | open window, then closure (observed) |
| **8086/8088** | byte, 1–6 bytes + prefixes; no invalid-opcode exception, all 256 first bytes execute | `ADD [BX+SI],AL`: NOP while AL = 0, otherwise adds AL into the organism's byte at address 0 | census: 29 period-2 words, all open; literal `MOV r16,imm;PUSH r16` (1/2) | **yes, 10 words** (`FF 30`–`35`, `FF 37` = `PUSH [reg]`; `AD 50`/`50 AD`; `07 A5`); they need zeroed pointer registers | `REP MOVS`/`STOS`, `LOOP`; 4–6-byte closers (seeded) | `HLT` F4; `INT`/`INTO`/divide error vector through ring bytes 4n | open; closure only if literal pushers dominate |
| **6502 NMOS** | byte, 1–3 bytes; 256/256 execute, 12 JAM opcodes halt (4.7%) | `BRK`: pushes PC+2 and P, jumps through `$FFFE` = ring bytes 2L−2, 2L−1 | **none, periods 1–4 (census, 4.29 × 10⁹ tiles)** | none | none; 6-byte closed loops (seeded) | JAM | **loop-bearing, closed** (if it starts) |
| 65C02 (WDC) | all 256 defined; undefined opcodes are NOPs | `BRK` (also clears D) | none expected (only 1-byte writes) | none | none; 6-byte loops (`BRA`) | `STP`, `WAI` | loop-bearing |
| 65816 | byte | `BRK` | hand: `PEA` tile `F4`, period 1 (2/3) | hand: `PEI ($00)` tile `D4 00` (1) | `MVN`/`MVP` | `STP`, `WAI` | open; λ = 0 likely |
| 6809 | byte, 1–5 bytes | `NEG <$00`: negates page-0 byte 0, i.e. mutagenic tar | hand, unverified: `LDD #`/`PSHS` gives no consistent tiling | hand: `PULU D,X,Y;PSHS D,X,Y`, period 4 (1.5) | multi-register push/pull + loops | `SYNC`/`CWAI`; 14/15/CD lock up | open via copying (unverified) |
| 68000 | 16-bit words, even alignment; line-A/F words (12.5%) trap; odd-address access traps | `ORI.B #0,D0` (NOP-like) | hand: `MOVE.W #imm,-(A7)` = `3F3C`, period 2 (1/2) | hand: `MOVE.W (A0),-(A7)` `3F10` (1); `MOVE.L (A0),-(A7)` `2F10` (2) | `MOVEM` + `DBRA`; about 8–12 bytes | `STOP`, `ILLEGAL` 4AFC, exceptions vector through 0–3FF | open, likely λ = 0 |
| PDP-11 | 16-bit words, even alignment | **`HALT` (natively lethal)** | hand: `MOV #x,-(SP)` 012746 (1/2) | hand: `MOV (PC),-(SP)` 011746 (1); `MOV -(PC),-(PC)` (1, runs backward) | `SOB` loops; **closed one-word `MOV -(PC),-(Rn)` 01474n** | `HALT`; traps to 4 and 10 | closed, or a one-word monoculture |
| MSP430 | 16-bit words; 0x0xxx undefined | 0x0000 undefined (`MOVA @PC,PC` on 430X) | hand: `PUSH #imm` 0x1230 (1/2) | hand: `PUSH @R4` 0x1224 (1) | none | halts by setting `CPUOFF` | open |
| Thumb-1 (ARMv4T) | 16-bit; few undefined encodings | `LSLS r0,r0,#0` (NOP) | hand: `LDR r0,[PC,#0];PUSH {r0}`, 4 bytes (1) | yes (that one) | `LDMIA`/`STMIA` | `UDF`, `SWI` | open, λ = 0 |
| RV32I/E (+C) | 32-bit, or 16-bit with C; **4.5% of random 32-bit words are valid RV32I (computed)** | **illegal (lethal)**, as is all-ones | none: no push, no auto-increment, at most one word copied per ≥ 3 instructions | none | loops of about 16–20 bytes | illegal opcodes trap | loop-bearing, or never starts (SUBLEQ-like) |

Stack conventions:
- **6502:** the stack is fixed in page 1. Use S = 0xFF − ((0x1FF − (2L−1)) mod 2L), so the first push lands on the last byte of B (S = 0xFF when 2L divides 256). The zero page aliases the ring; zero-page indexing wraps at 8 bits before the ring modulus.
- **Word-oriented machines** (68000, PDP-11, MSP430, Thumb, RISC-V) need an even L and alignment rules; odd addresses trap.

## 3. Validation sources

| ISA | Data-only vectors | Coverage | Origin | Reference emulator (licence, language) |
|---|---|---|---|---|
| 8088 | SingleStepTests/8088 v2 | 10,000 tests per opcode (fewer for string ops, CL shifts, INC/DEC reg, flag ops); undefined-flag masks in `metadata.json`. Excludes `0F`, `9B`, `F4`, `F0`/`F1`, register forms of LEA/LES/LDS, `FE /2-7`. MIT. | **real AMD D8088 hardware** | MartyPC (MIT, Rust); 86Box (GPL-2, C/C++) |
| 8086 | SingleStepTests/8086 (also 80186, 80286, V20) | 2,000 per opcode | real Intel 80C86A | same |
| 6502 NMOS | SingleStepTests/65x02 `6502/v1` | one JSON per opcode, 10,000 tests each; listing shows 00–FB including JAM `02` and unstable `8b ab 93 9b 9c 9e 9f`. MIT. | emulator-generated, cross-validated | MAME m6502 (BSD-3, C++); chips m6502.h |
| 65C02 | same repo: wdc65c02 / rockwell65c02 / synertek65c02 | complete except `cb`/`db` (WAI/STP), which are empty files | emulator | MAME (BSD-3) |
| 6502 programs | Klaus Dormann functional test, 65C02 extended test; Bruce Clark decimal test | documented NMOS opcodes | GPL-3.0; 6502 binaries that run inside our emulator — **your call** whether that counts as third-party code | — |
| 65816 | SingleStepTests/65816 | `xx.e.json` / `xx.n.json`, 10,000 each | emulator; no licence file seen | MAME g65816 (BSD-3), ares (ISC) |
| 68000 | SingleStepTests/680x0 (Harte); m68000 (MAME microcoded core, all verified except TAS and TRAPV; stored as `.json.bin` and needs their `decode.py`) | broad | emulator | Musashi (MIT-style, C); Moira (MIT, C++) |
| ARM7TDMI | SingleStepTests/ARM7TDMI | experimental; Thumb coverage unclear | NanoBoyAdvance (GPL-3) | — |
| 6809, PDP-11, MSP430, RISC-V | **none found** | — | — | Open SIMH (MIT); Sail RISC-V (BSD-2); Spike (BSD-3); mspdebug (GPL-2); MSPSim (BSD-3) |

**Complete data-only vectors exist for:** 6502, 65C02, 65816, 8088/8086, 68000. Of these, only the 8088/8086 sets are hardware-captured. For a near-silicon 6502 reference, the options are a transistor-level netlist simulator (perfect6502/Visual6502; licence not checked) or recording vectors from a physical chip.

## 4. Computations

All code and outputs are in `scratchpad/isa_research/census/`.

### 4.1 6502 census (Proposition 3 analogue)

Method: tile period k into A (L = 16), all-zero B, 128 instructions, then also 16 random partners; open or closed means whether the pointer fetches a partner byte. Run with `BRK` faithful and with `BRK` as halt. Wall time was about 10 minutes per mode on 10 cores.

| Period | Tiles | Exact into zero partner | Exact into random partners | ≥ 75% copy in ≥ 8/16 partners |
|---|---|---|---|---|
| 1, 2, 3 | 256 / 65,280 / 16,776,960 | 0 (only the trivial `00` under halt-`BRK`) | 0 | 0 |
| 4 | 4,294,901,760 | **0** | 2 tiles, each in 1 of 16 partners (open straight-line stack copiers, `TSX;LDA $0C,X;PHA` and `LAX $8F,Y;DEY;PHA`) | 16 (faithful `BRK`) / 22 (halt): `JSR` return-address smears at exactly 75%, and closed indirect-indexed loops at up to 87.5% (e.g. `91 4c c8 b3`) |

Periods 1–3 repeated at L = 20 and L = 32 also gave 0. The reason is structural: every 6502 store writes one byte, while an immediate load costs two bytes plus a one-byte push. So no tiling is consistent, unlike `01 c5`.

### 4.2 6502 seeded closers

| Closer | Copies exact into random partners | Pointer in partner | Sites carried |
|---|---|---|---|
| `B5 0F 48 CA 50 FA` (`LDA zp,X;PHA;DEX;BVC`) | 64/64 (×3 random bodies) | 0/64 | all 16 |
| `BA B5 10 48 50 FA` (`TSX;LDA zp,X;PHA;BVC`) | 64/64 (×3 random bodies) | 0/64 | all 16 |

- The zero-page operand zz sets the copy period d = zz + 1.
  - zz = 07: sites 0–7 are carried and 8–15 erased (a regenerator transmitting 2 sites).
  - zz = 0F: a transmitter (all sites carried).
  - This is the 6502 analogue of Proposition 8, with d − 6 transmitted sites.
- Open straight-line copier tiles (e.g. `B5 0F 48 CA`) copied exactly into 0 of 64 random partners.
- **Budget caveat:** a minimal 6502 loop spends 4 instructions per copied byte. With 128 instructions it copies only 32 bytes, so complete copies need L ≤ 32 unless the budget is raised.

### 4.3 8086 partial census, period 2

65,280 tiles; 55,937 evaluated. 9,343 were left unevaluated because I did not implement IN/OUT, BCD adjust, AAM/AAD, MUL/DIV and a few undefined forms.

**29 self-writers, all open:**
- **Literal load–push:** `B9 51`, `BA 52`, `BB 53`, `BD 55`, `BE 56`, `BF 57` and their rotations, plus `50 B8`. `B8 50` fails in that phase, because the zero tar `ADD [BX+SI],AL` rewrites byte 0 once AL ≠ 0.
- **Other words:** `50 0D`, `0E EA` (`PUSH CS; JMP FAR`), `35 FF`, `C4 52`, `57 C7`, `07 A4`.
- **One-pass memory copiers:** `FF 30`–`35`, `FF 37`, `AD 50`, `50 AD`, `07 A5`.

Inflow over 64 random partners:
- **The 10 one-pass copiers:** a single offspring (λ = 0), 64/64 exact, and they never fetch an unwritten partner byte.
- **The other 19:** fetch unwritten partner bytes in every encounter, produce 7–31 distinct offspring (λ > 0), and copy exactly into 21–90% of partners — Z80-like.

**Random-register arm** (AX, BX, CX, DX, BP, SI, DI and flags random; SP = 0): only 4 words remain self-writers in at least 8 of 16 draws — `B8 50` (14), `BA 52` (12), `0E EA` (12), `B9 51` (10). All are literal, ratio 1/2.

**Seeded closers:**
| Closer | Exact into random partners | Pointer in partner | Result |
|---|---|---|---|
| `07 49 F3 A5` | 64/64 | 0/64 | transmitter |
| `BF 10 00 49 F3 A4` | 64/64 | 0/64 | transmitter |
| `BF 08 00 49 F3 A4` | 0/64 | 0/64 | regenerator: sites 0–7 kept, 8–15 erased |

## 5. Predictions for the top three

**8086 (primary).**
- The first replicator is a period-2 straight-line word, and it is open.
- Fork:
  - If a one-pass memory copier spreads, the theory predicts the open form persists with λ = 0 and no closure (BFF ×2/×3 regime). Among those copiers, `FF 3x` are d = 2 regenerators and `07 A5` copies the whole tape.
  - Under randomised registers, only literal pushers (λ > 0) remain, so the prediction is an open window, then `REP MOVS` closers, then the regenerator/transmitter split by the DI operand.
- The lethality dial applies as on the Z80.

**6502 (contrast).**
- No straight-line self-writer exists, so replication should start loop-bearing and closed, with λ = 0 from birth.
- Regenerators and transmitters should appear from the start, through the one-byte offset switch.
- It may not start at all. Pre-register a null outcome as acceptable, with a compute cap.

**68000 (alternative primary).**
- Open start through `3F3C` and its register variants.
- One-pass `2F10`/`3F10` predict a persistent open form under zeroed A0.
- Closers need `DBRA`/`MOVEM` loops of about 8–12 bytes, so closure is slower.

**Others, briefly:**
- PDP-11: a one-word closed `MOV -(PC),-(Rn)` (d = 2) monoculture, with zero `HALT`.
- 65816: open through `PEA`/`PEI`, closers through `MVN`.
- RISC-V: probably never starts.

## 6. Literature

- **Agüera y Arcas et al. 2024, arXiv 2406.19108, pp. 16–17.**
  - SUBLEQ: the smallest hand-written replicator is 60 bytes; RSUBLEQ4: 25 bytes. Neither arose from random soups.
  - Z80: stack-based replicators first, then `LDIR`/`LDDR` copiers.
  - 8080 (long tape): replicators were always `01 c5`-type, and no looping variant was ever seen.
- **Z80 only:** Cicala et al. 2026 (arXiv 2607.09211; `LDIR` copiers displace load–push ones); Jha et al. 2026 (arXiv 2609.10817). **BFF only:** Knierim et al. 2026 (arXiv 2607.01483).
- **vicgalle/coevolution-soup** (GitHub, 2026, unreviewed README): the only other-ISA soup I found, on the 6502 (summary numbers above).
- **No soup experiment found** on x86, RISC-V, ARM, MIPS, 68000, PDP-11, MSP430, WebAssembly, or with Forth or SUBLEQ beyond the 2024 paper.
- **Related but not spontaneous emergence:**
  - Darwin (1961, IBM 7090); Core War (Dewdney 1984); Rasmussen's Coreworld (1990); Pargellis's Amoeba (1996, 2001); Greenbaum & Pargellis (2017).
  - Ray 1991, p. 375: mutation in real machine code is "almost certain to produce a non-functional program".
  - SPTH 2011 (arXiv 1105.1534, native x86, seeded, brittle); Nordin's machine-code GP; Schulte et al. 2014 on mutational robustness.
  - PDP-11 `MOV -(PC),-(PC)` is folklore (comp.unix.wizards 1985; cctalk 2024).

## 7. Recommendation

**Rationale.**
- **8086/8088:** hardware-captured vectors; the Z80's ingredients present (literal push, operand-set block-move closers); and a built-in within-machine test of the write-ratio prediction through the existing random-register arm.
- **6502:** the strongest-validated loop-first machine. Its prediction already has independent support.

**Harness mapping, 8086.**
- Ring of 2L bytes; physical address = ((16·seg + off) mod 2²⁰) mod 2L.
- Registers and segments zero, IP = 0, FLAGS = F002h (IF = 0, DF = 0).
- SP = largest multiple of 2L below 65536 (0 when 2L divides 65536), so the first push writes the last two bytes of B.
- 128 instructions; each `REP` iteration counts as one, like `LDIR`; prefixes belong to their instruction (cap long prefix chains).
- `HLT` halts. Software interrupts are taken faithfully: push FLAGS, CS, IP; vector from ring bytes 4n…4n+3.
- I/O: `IN` returns FFh; `OUT` is a no-op. `ESC` decodes ModRM and does a dummy read; `WAIT` is a no-op.
- Use architectural self-modifying-code semantics (no stale prefetch-queue bytes) and say so.
- Undefined flags and undefined opcodes: reproduce the 8088 hardware, as captured in the vectors.

**Harness mapping, 6502.**
- Ring of 2L bytes; A = X = Y = 0; flags zero (P = 20h); S as in Section 2; PC = 0; 128 instructions.
- JAM halts.
- `BRK` faithful: the first `BRK` in an encounter returns to its own return address, because the vector is the address it just pushed. Add `BRK`-halt as a lethal-tar dial arm.
- NMOS decimal mode; unstable opcodes take the vectors' constants. Use L ≤ 32, or add a budget-512 arm.

**Main risks.**
- 8086:
  - The persistent-open branch may happen.
  - Referees may see the 8086 as part of the Intel/8080 family.
  - The zero tar is mutagenic once AL ≠ 0.
  - Interrupt vectors sit in the organism's own bytes.
  - Decode complexity and GPU divergence; about 1.5–2× the Z80 cost per step (my estimate).
- 6502:
  - It may never start replicating.
  - `BRK` vector aliasing depends on L.
  - The vectors are emulator-generated, not recorded from hardware.

**Effort.**
- 6502: about 450–550 WGSL lines; 1–2 weeks including about 2.56 M vectors, harness and assays. Run a short GPU pilot before any paid sweep.
- 8086: about 1,500–2,000 lines; 3–5 weeks including about 3 M hardware vectors.
- Total: about 5–7 weeks, then runs, with review and approval before any paid sweep.

Files are in `/private/tmp/claude-501/-Users-neo-repos-algocell/26fc62cd-9c63-4d42-8467-b9398dac54bd/scratchpad/isa_research/census/`:
- `census6502.c`, `cpu6502.h`, `m6502.py`, `optable.py`
- `out_k*_L16_bh*_c*.txt`, `summarize.py`
- `m8086.py`, `census8086.py`, `census8086_randreg.py`, `inflow8086.py`, `ratio8086.py`
- `seeded6502.py`, `seeded8086.py`, `rv32i_density.py`

The text of Agüera y Arcas 2024 is in `isa_research/dl_aguera/cl.txt`.

Sources:
- [SingleStepTests org](https://github.com/orgs/SingleStepTests/repositories), [8088](https://github.com/SingleStepTests/8088), [65x02](https://github.com/SingleStepTests/65x02), [65816](https://github.com/SingleStepTests/65816), [m68000](https://github.com/SingleStepTests/m68000), [680x0](https://github.com/SingleStepTests/680x0), [ARM7TDMI](https://github.com/SingleStepTests/ARM7TDMI)
- [Agüera y Arcas et al. 2024](https://arxiv.org/abs/2406.19108), [Cicala et al. 2026](https://arxiv.org/abs/2607.09211), [Knierim et al. 2026](https://arxiv.org/abs/2607.01483), [Jha et al. 2026](https://arxiv.org/abs/2609.10817)
- [vicgalle 6502 README](https://github.com/vicgalle/coevolution-soup/blob/main/README_6502.md), [cubff](https://github.com/paradigms-of-intelligence/cubff)
- [SPTH 2011](https://arxiv.org/abs/1105.1534), [Ray, Tierra](https://www.tomray.me/pubs/tierra/node4.html), [cctalk PDP-11 thread](https://classiccmp.org/mailman3/hyperkitty/list/cctalk@classiccmp.org/message/HID7TWQBD5POGRH26OQZHTQ45YH6Z3UA/)
- [righto: undocumented 8086 instructions](https://www.righto.com/2023/07/undocumented-8086-instructions.html), [OS/2 Museum: undocumented 8086 opcodes](https://www.os2museum.com/wp/undocumented-8086-opcodes-part-i/), [righto: 8088 prefetch](https://www.righto.com/2024/03/8088-prefetch-circuitry.html)
- [NESdev: unofficial opcodes](https://www.nesdev.org/wiki/CPU_unofficial_opcodes), [6502.org: 65C816 opcodes](http://www.6502.org/tutorials/65c816opcodes.html), [hoglet67: undocumented 6809](https://github.com/hoglet67/6809Decoder/wiki/Undocumented-6809-Behaviours)
- [RISC-V unprivileged spec](https://docs.riscv.org/reference/isa/v20260120/unpriv/intro.html), [MartyPC](https://github.com/dbalsom/martypc), [Open SIMH](https://github.com/open-simh/simh), [MAME licence](https://docs.mamedev.org/license.html), [Musashi](https://github.com/kstenerud/Musashi)
# Zilion as a multi-ISA substrate for artificial-life research: architecture and design audit

**Date:** 2026-10-10.
**Status:** design only. Nothing is implemented, and no file in either repository was edited.
**Audited:**
- `zilion@0.2.0`: `/Users/neo/repos/zilion`, `main` at `8557d3b`.
- `algocell`: `feat/experiments-runner` at `593ec67`.

**Conventions:** line references are `file:line` in those trees. "Core" means
`zilion/src/z80-core.wgsl.ts`. All Algocell paths are relative to its repository root.

**Structure.** The architecture comes first, per the scope update:

- §1 Architecture: layers, the plug-in contract, ISA-specific versus shared parts, the ALife-layer split.
- §2 How researchers will use it: TypeScript, Python, reproducibility, performance.
- §3 Prior art.
- §4 Inventory of what Algocell built around Zilion.
- §5 Correctness work: R, MEMPTR, and the risk to existing results.
- §6 Testing.
- §7 Staged roadmap, milestones, risks and release.
- Appendices.

---

## 0. Summary

### The thesis

Algocell's own theory already defines the right abstraction. THEORY.md §1 (`experiments/THEORY.md:7-9`):

- A **machine** is "a deterministic byte-code interpreter with a program counter and some registers".
- A **world** places A and B (length L) in a memory of P ≥ 2L bytes (addresses mod P), zeroes the
  registers, starts the PC at A[0], runs S steps, and writes back only A and B.
- The observables are E_S(A,B), the memory after S steps, and T_S(A,B), the set of addresses
  visited.

Zilion should be **the library that computes E_S and T_S for billions of (machine, world) pairs, for
many machines, with world semantics that are literally the same code for every machine.**

That last clause is the scientific point. The comparative claims (the 8080 subset, BFF switches,
lethality dials, ring sizes, and soon a second chip) are only clean if the world does not change
when the machine does.

Today it does change:

- Algocell has built the world **twice**: once around the Zilion Z80 (`src/lib/gpu/shaders.ts`, plus
  four Python generators that patch Zilion's core text) and once for BFF (`experiments/micro/bff.py`).
- The two worlds differ in memory layout, in halting-dial randomness, and in the records they keep.
- The 8080 condition is a Z80 with instructions ablated (`experiments/make_conds.py:63`), not an 8080.

### Proposed architecture (§1)

Five layers. The first three are in Zilion.

1. **L0 ISA cores.** A WGSL core per chip, a CPU twin transpiled from the *same* WGSL, and data-only
   conformance vectors.
2. **L1 World / host contract.** Memory models (flat mask, ring mod P, segments/pair, padding),
   hooks, ablation, halting and trap policies, traces, register conventions, budgets.
3. **L2 Batch engine.** Compiled machines, resident batches, many worlds per dispatch, manifests,
   the Python runtime.
4. **L3 Soup toolkit.** Pairing, mutation, step pipeline, snapshots, event logs, culture-test
   harness. This is a **separate package**, `@neovand/zilion-soup`, in the same monorepo.
5. **L4 Research definitions.** Thresholds, closure, inflow, birth classes. These stay in Algocell.

### Key decisions

| Decision | Choice | Why |
|---|---|---|
| Family or siblings | **One monorepo, two packages**: `@neovand/zilion` (L0–L2; one subpath per ISA) and `@neovand/zilion-soup` (L3) | One world contract and one version for all cores. The soup layer changes at research pace and is policy-laden. |
| Proof that the contract is not Z80-shaped | The second core is **BFF** (ring-native tape machine, about 150 lines, already exists as WGSL in Algocell); the third is the **6502** (SingleStepTests 65x02) | If the contract fits a prefix-heavy CISC and a headless tape machine, it will fit most targets |
| Test truth | Data-only per-instruction vectors (SingleStepTests where they exist; the 8086/8088 sets are hardware-generated), plus analytic world tests, plus CPU-twin cross-checks. No third-party emulator code in CI. | z80-emulator had five defects (§4.4). Memory note: "validate against a real reference". |
| Reproducibility | A "strict" determinism mode in the soup layer: counter-based RNG, order-independent pairing, race-free mutation. Legacy mode is kept for registered work. | Today a seed is one sample of the process (`experiments/PLAN.md:22`). The audit also found a mutation write race (§2.3). |
| Python | Codegen stays in TypeScript. Python gets manifests and a small runtime. | Single source of truth, as `scripts/export-sim.ts` already does |

### Prioritised work list (sizes S ≤ 1 day, M 2–4 days, L 1–2 weeks; milestones in §7)

| # | Item | Layer | Milestone | Size |
|---|---|---|---|---|
| 1 | Headless harness: Node 22 + Dawn (`webgpu` npm); Linux CI on software Vulkan; port `test/conformance.js` | test | M1 | M |
| 2 | Generic per-instruction vector runner (sparse-memory host, full-state I/O, divergence ledger); Z80 SingleStepTests adapter; baseline on 0.2.0 | L0/test | M1 | M |
| 3 | Z80 fixes: R for wasted prefixes; full MEMPTR; IM; `OUT (C),r` WZ; `z80_reset`; halt reasons; `compat:'0.2'` | L0 | M1 | M |
| 4 | Host contract v2 hooks (`on_instruction_start`, `on_fetch`, `on_decode`/`on_fetch_opcode`, `on_reset`, `zil_trap`, `io_*`), opt-in; keep Algocell's anchors byte-stable | L1 | M1 | M |
| 5 | World spec: ring/segments/padding memory models; seam semantics and helpers; analytic ring families for any P (odd, padded); wrap-mask mutants | L1/test | M1 | M |
| 6 | Equivalence proof: Algocell's 222 exported shaders regenerated from world specs, bit-identical under `compat:'0.2'`. Then the impact study of #3. | Algocell | M1 | M |
| 7 | Monorepo restructure; `IsaCore` interface; `buildMachine`; manifest; outputs (state record, fetch/start maps, write counts, steps, halt reason); per-lane budgets | L1/L2 | M1 | M |
| 8 | Batch engine: resident batches, `encode()`, chunking against device limits (fixes a latent bug), typed results, compile errors surfaced | L2 | M1 | M |
| 9 | Python runtime (manifest-driven, wgpu-py), repo-only | L2 | M1/M2 | M |
| 10 | CPU twin: WGSL-subset→JS transpiler; GPU-versus-twin fuzz | L0 | M2 | L |
| 11 | BFF core (ring-native); bit-identical to `micro/bff.py` | L0 | M2 | S–M |
| 12 | 6502 (NMOS) core + SingleStepTests/65x02 | L0 | M2 | L |
| 13 | `zilion-soup`: topologies, pairing schedulers (legacy / deterministic), race-free mutation, fused step pipeline, many soups per dispatch, event logs, snapshots, culture-test harness | L3 | M3 | L |
| 14 | Strict determinism mode, cross-backend bitwise CI; performance benchmark gate | L3/test | M3 | M |
| 15 | Further chips: 8086 (hardware-generated vectors), RISC-V RV32I (architecture tests), PDP-11 (vector gap: risk); cycle and fetch-byte budgets | L0 | M4 | L each |
| 16 | Opt-in Z80 `zilog-nmos` variant (Q latch), block-repeat flags; storage-backed and packed memory | L0/L1 | M2–M3 | S–M |

**Publishing:** nothing goes to npm or PyPI, and no tag is pushed, without the user's explicit
approval (§7.4).

---

## 1. Architecture

### 1.1 From the theory to the layers

| THEORY.md concept | Layer | Zilion object |
|---|---|---|
| Machine M (interpreter, PC, registers) | L0 | `IsaCore` (WGSL core + CPU twin + vectors) |
| World: A ‖ B in P ≥ 2L, addresses mod P, registers zeroed, PC at A[0], S steps, write back A and B, discard padding | L1 | `WorldSpec` (memory model, init convention, budget, halting/trap policy, ablation) |
| E_S(A,B), T_S(A,B) | L1 outputs | `memory`, `fetchMap` (T_S), `startMap` (instruction starts), state, write counts |
| Billions of encounters | L2 | `Machine`, `Batch` (resident buffers, many worlds per dispatch) |
| The soup: who meets whom, mutation, time | L3 | `zilion-soup` |
| Self-writer, closure, aliveness (gen2), inflow, line of descent | L4 | Algocell research code |

### 1.2 Layer diagram and responsibilities

```
 L4  research code (Algocell)       definitions, thresholds, pre-registered analyses, figures
 ───────────────────────────────────────────────────────────────────────────────────────────
 L3  @neovand/zilion-soup           topologies · pairing schedulers · mutation · step pipeline ·
                                    snapshots · event logs (genealogy) · culture-test harness
 ───────────────────────────────────────────────────────────────────────────────────────────
 L2  @neovand/zilion  engine        Machine (compiled pipeline) · Batch (resident) · chunking ·
                                    multi-world dispatch · manifest · Python runtime
 L1  @neovand/zilion  world         memory models · hooks · ablation · halting/trap policy ·
                                    traces · init conventions · budgets · counter-based RNG
 L0  @neovand/zilion/<isa>          WGSL core · CPU twin (transpiled) · tables · vector adapter
                                    z80 (now) · bff · m6502 · i8086 · rv32i · pdp11 …
```

**Dependency rule.**

- Lower layers never import higher ones.
- L0 cores talk to the world only through the hook contract (§1.3).
- L3 composes L2 machines with soup kernels. It never edits core text, which is what Algocell's
  generators do today.

### 1.3 The plug-in contract: what a new core must provide

A core is accepted when it ships all of the items in this subsection.

**(a) WGSL symbols.** All symbols carry the core's prefix; the builder emits `isa_*` aliases.

```wgsl
// state: any var<private> it needs, prefixed (e.g. m6502_a). The Z80 keeps its legacy cpu_* names.
const <ID>_ADDR_MASK: u32;        // 0xffff (Z80, 6502, PDP-11), 0xfffff (8086), 0xffffffff (RV32), 0 = ring-native (BFF)
fn <id>_reset()                   // the documented *world* reset (all-zero state, PC 0), not the chip's reset sequence
fn <id>_step()                    // one step (§1.4 "step"); calls hooks only at the canonical points
fn <id>_halted() -> u32           // ZIL_RUNNING, or a ZIL_HALT_* reason
fn <id>_pc() -> u32               // address of the next instruction (traces, assays)
fn <id>_store(o: u32)             // serialise state into zil_state_out[o ..) per the state descriptor
fn <id>_load(o: u32)              // and back (resumable runs, vector tests, random init)
```

**(b) Hooks the core calls.** The host provides them. `buildMachine` emits defaults; a bare core
never calls a disabled hook.

```wgsl
fn mem_read(addr: u32) -> u32                    // one byte; addr already masked to the ISA width
fn mem_write(addr: u32, val: u32)
fn on_fetch(addr: u32)                           // every instruction-stream byte (opcode, prefix, operand, displacement)
fn on_instruction_start(pc: u32, unit: u32) -> u32 // once per step, after the first fetch unit: ZIL_CONTINUE | ZIL_HALT | ZIL_SKIP
fn on_decode(page: u32, key: u32) -> bool        // ablation after full decode; true = skip (consumed length is ISA-defined)
fn zil_trap(cause: u32) -> u32                   // illegal op, odd address, div-by-0, ecall, JAM…: ZIL_TRAP_HALT | _VECTOR | _IGNORE
fn io_read(port: u32) -> u32                     // ISAs with port I/O only (Z80, 8080, 8086)
fn io_write(port: u32, val: u32)
```

**(c) TypeScript descriptor.**

```ts
export interface IsaCore {
  readonly id: string;                         // 'z80' | 'bff' | 'm6502' | 'i8086' | 'rv32i' | 'pdp11' | …
  readonly name: string;
  readonly coreVersion: string;
  readonly coreSha256: string;                 // provenance
  readonly addressBits: 0 | 16 | 20 | 32;      // 0 = ring-native
  readonly fetchUnitBytes: 1 | 2 | 4;          // 'unit' passed to on_instruction_start
  readonly endianness: 'little';               // every planned target is little-endian
  readonly state: readonly StateField[];       // { name, bits, randomizable, zero } → record layout + random init
  readonly decodePages: readonly { id: number; name: string; keyBits: number }[];  // ablation mask layout
  readonly variants: Readonly<Record<string, VariantSpec>>;   // e.g. z80 'classic' | 'zilog-nmos'; m6502 'nmos' | 'cmos'
  readonly conventions: IsaConventions;        // step definition, stack-init rule, illegal-opcode policy, skip semantics
  wgsl(o: CoreGenOptions): string;             // { variant?, compat?, hooks: EnabledHooks } → core text
  table(): readonly InstructionInfo[];         // mnemonics, families, lengths, writesMem
  disassemble(bytes: Uint8Array, o?: { ablation?: AblationSet }): DisasmLine[];
  readonly vectors?: VectorAdapter;            // per-instruction JSON field mapping, or a program-signature runner
}
export interface IsaConventions {
  step: string;                                // e.g. 'one instruction; one LDIR/CPIR/INIR iteration; a wasted prefix'
  stackTop(P: number, targetPos: number): Partial<Record<string, number>>;  // registers realising "stack top at targetPos"
  illegal: Readonly<Record<string, 'as-chip' | 'nop' | 'halt'>>;            // per variant
  skip: string;                                // what an ablated opcode consumes
}
```

**(d) A CPU twin.** It comes free if the core stays inside the Zilion WGSL subset (§6.1): u32/i32/bool,
`select`, `switch`, `if`/`for`, private scalars and arrays. A hand-written twin is not accepted; that
is the `src/lib/sim/z80.ts` trap (memory note `z80-not-ground-truth`).

**(e) A conformance package.**
- a vector adapter, or a program-signature runner;
- a divergence ledger;
- the shared world tests instantiated with ISA **probe encoders**: "jump to absolute address a",
  "relative branch by e", "push", "store byte at a", "load byte from a". From these the generic ring
  families (§6.3) are generated for that ISA.

**(f) A conventions document.** Step definition, reset state, stack-top rule, halting, illegal-opcode
and trap policy, skip semantics, unstable opcodes.

#### Worked example 1: 6502 (NMOS) core skeleton

```wgsl
var<private> m6502_a: u32; var<private> m6502_x: u32; var<private> m6502_y: u32;
var<private> m6502_s: u32; var<private> m6502_p: u32; var<private> m6502_pc: u32; var<private> m6502_halted: u32;
const M6502_ADDR_MASK: u32 = 0xffffu;
fn m6502_fetch() -> u32 { on_fetch(m6502_pc); let v = mem_read(m6502_pc); m6502_pc = (m6502_pc + 1u) & 0xffffu; return v; }
fn m6502_push(v: u32) { mem_write(0x100u | m6502_s, v & 0xffu); m6502_s = (m6502_s - 1u) & 0xffu; }  // stack fixed in page 1
fn m6502_reset() { m6502_a = 0u; m6502_x = 0u; m6502_y = 0u; m6502_p = 0x24u; m6502_s = 0u; m6502_pc = 0u; m6502_halted = 0u; }
fn m6502_step() {
    if (m6502_halted != 0u) { return; }
    let pc0 = m6502_pc;
    let op = m6502_fetch();
    let act = on_instruction_start(pc0, op);
    if (act == ZIL_HALT) { m6502_halted = ZIL_HALT_HOST; m6502_pc = pc0; return; }
    if (act == ZIL_SKIP || on_decode(0u, op)) { return; }        // 1-byte NOP; operand bytes run next
    switch (op) {
        case 0x02u, 0x12u, 0x22u, 0x32u, 0x42u, 0x52u, 0x62u, 0x72u, 0x92u, 0xb2u, 0xd2u, 0xf2u: {   // JAM (NMOS)
            if (zil_trap(ZIL_TRAP_ILLEGAL) != ZIL_TRAP_IGNORE) { m6502_halted = ZIL_HALT_TRAP; m6502_pc = pc0; }
        }
        // … 244 more opcodes; zero-page indexed modes wrap inside page 0; JMP ($xxFF) reads $xx00 …
        default: {}
    }
}
```

Its descriptor: `addressBits 16`, `fetchUnitBytes 1`, one decode page (`keyBits 8`), variants
`nmos` (JAM halts, unstable opcodes with a pinned magic constant) and `cmos` (undefined opcodes are
NOPs). The stack-top rule picks the largest S ≤ 0xFF with (0x100 + S) mod P = targetPos. For
P ≤ 256 some S always exists, because S spans 256 consecutive addresses.

#### Worked example 2: BFF as a ring-native core

This is a port of `experiments/micro/bff.py:33-129`.

- `BFF_ADDR_MASK = 0`: the core computes positions mod P itself, using a builder constant
  `ZIL_RING_P`.
- State: `ip`, `h0`, `h1`, `halted`.
- `on_fetch(ip)` per op. `on_decode(0u, op_id)` after the alphabet map (the density map is a core
  variant parameter).
- An unmatched bracket calls `zil_trap(ZIL_TRAP_UNMATCHED)`. The world's halting policy decides,
  which reproduces the published rule, `nohalt`, and the probability dial.
- IP leaving [0, P) is either a halt or a wrap (the `ip_wrap` variant).
- One step is one op, including a whole bracket scan (cubff semantics).
- Nothing is Z80-shaped, yet the same world (pair of 64-byte tapes, ring 128, budget 2^13, fetch
  map, write counts) applies unchanged. That is the test of the contract.

### 1.4 What is ISA-specific and what is shared

Columns: **Shared** (L1 world, L2 engine) and the ISA-specific realisations.

#### (a) Instruction decode and fetch hooks

| | Shared | Z80 | 6502 | 8086 | PDP-11 | RV32I | BFF |
|---|---|---|---|---|---|---|---|
| Fetch unit, the `unit` in `on_instruction_start` | hook semantics; `on_fetch` is always per **byte**, so fetch maps are comparable across ISAs | byte | byte | byte | 16-bit word | 16 bits (32 if C absent; manifest says which) | byte (op) |
| What one step is | budget accounting, halt test between steps | instruction; one LDIR/CPIR/INIR iteration; a wasted prefix | instruction (BRK is one step) | instruction; one REP iteration | instruction | instruction | one op, including the bracket scan |
| Decode key for ablation (`on_decode(page,key)`) | packed bitset per page (`2^keyBits` bits), uniform or storage; same pattern grammar | pages base/CB/ED/DD/FD/DDCB/FDCB, key = opcode byte | page 0, key = opcode | page 0 = opcode; group pages (80–83, D0–D3, F6/F7, FE/FF) key = ModR/M.reg | pages by format (double/single-operand, branch, …), key ≤ 10 bits | page = opcode[6:2] (32), key = funct3 \| funct7≪3 | page 0, key = op id |

#### (b) Prefixes and multi-byte opcodes

| | Shared | Z80 | 6502 | 8086 | PDP-11 | RV32I | BFF |
|---|---|---|---|---|---|---|---|
| Structure | opcode units vs operand units distinguished in the manifest | CB/ED/DD/FD prefixes, DDCB d op, wasted prefixes | 1 opcode + 0–2 operand bytes | segment, REP, LOCK prefixes (repeatable) + opcode + ModR/M + disp + imm (1–6 bytes + prefixes) | 1 word + 0–2 extension words | fixed 32-bit (16 with C) | 1 byte |
| Ablation skip consumes | rule: **opcode units only; operands run as code** (documented per ISA, because it shapes evolution) | prefix(es) + op (`core:57-64`) | 1 byte | prefixes + opcode (+ ModR/M for group pages) | the opcode word | the whole instruction (no operand bytes exist) | 1 byte |

#### (c) Stack conventions

| | Shared | Z80 | 6502 | 8086 | PDP-11 | RV32I | BFF |
|---|---|---|---|---|---|---|---|
| Stack | the world convention "stack top aliases position t" (Algocell: the last byte of B); `stackTop(P, t)` per ISA; the stack is ordinary memory mapped by the same ring function | SP 16-bit, push pre-decrements by 2; SP0 = highestAlias(t, P) (`shaders.ts:82-87`) | `0x0100 \| S`, 8-bit S, 1-byte push post-decrement, **fixed to page 1** | SS:SP, physical = SS·16 + SP; SS = 0, so SP0 as for the Z80 | R6, word-aligned (an odd SP traps); SP0 = the largest even value aliasing so that the pushed high byte lands at t | none in hardware; x2 by ABI; x2 = highestAlias over 32 bits | none (two heads) |

#### (d) Address width and aliasing onto a small ring

| | Shared | Z80 | 6502 | 8086 | PDP-11 | RV32I | BFF |
|---|---|---|---|---|---|---|---|
| Width | `pos = (addr & ADDR_MASK) % P`. Seam at `2^w mod P` (§4.3). Multi-byte accesses are **decomposed into byte accesses at consecutive ISA-masked addresses**, as the Z80 core does (`core:568, 575`), never a word read at a position. | 16 | 16 | 20 (offset wraps at 64 K inside a segment; physical wraps at 1 MB) | 16 (the I/O page is ordinary ring memory unless the world maps devices) | 32 (the seam exists but is unreachable in short runs) | ring-native |
| Address quirks inside the core | – | none beyond 16-bit wrap | zero-page indexed wrap within page 0; JMP ($xxFF) | segment arithmetic | odd-address trap on word access | misaligned-fetch trap without C | heads wrap mod P |

#### (e) Halting and illegal opcodes

| | Shared | Z80 | 6502 | 8086 | PDP-11 | RV32I | BFF |
|---|---|---|---|---|---|---|---|
| Native halt | halt reason codes; `ZIL_HALT_HOST` from `on_instruction_start` (lethal-unit dial); `zil_trap` policy `halt` / `vector` / `ignore` | HALT (76), parked on the opcode | JAM opcodes (NMOS); STP (WDC) | HLT (F4) | **HALT = 000000**: a zero word halts natively | none; ECALL/EBREAK | unmatched bracket; IP out of range |
| Illegal / undefined | the world-level `illegal` policy | none: every byte decodes (undefined ED = 2-byte NOP) | NMOS undocumented opcodes (some unstable, so pin a constant); CMOS NOPs | none on the 8086 (aliases, `POP CS`); div-by-0 → INT 0 through the IVT in ring memory | reserved-instruction and odd-address traps through vectors 10/4 (in ring memory) | illegal-instruction trap; **0x00000000 is illegal** | none (other bytes are no-ops) |

**Research note.** Algocell's "lethal tar" (a zero opcode halts) is an *intervention* on the Z80,
but a *native* property of the PDP-11 and of RISC-V. The shared trap policy lets a study impose it on
one chip or remove it from another with the same switch. Comparisons across ISAs need exactly that.

#### (f) Conformance-test format

| | Shared | Z80 | 6502 | 8086 | PDP-11 | RV32I | BFF |
|---|---|---|---|---|---|---|---|
| Source | generic per-instruction JSON runner (`{initial, final, ram:[[addr,val]]}`), sparse-memory host for any width, ledger, CI rules; plus a program-signature runner | SingleStepTests/z80 (JSMoo/Ares lineage; WZ, Q, R fields) | SingleStepTests/65x02 (6502, 65C02 variants, NES) | SingleStepTests/8086 and /8088, **hardware-generated** | **none found**: DEC diagnostics as programs, or vectors from a reference model (risk) | riscv-arch-test (signature files from the Sail reference model) | spec-derived analytic vectors + `micro/bff.py` outputs recorded as data |

The SingleStepTests organisation also covers the 80286, 80386, NEC V20, 68000, 65816, SPC700, SM83,
R3000 and SH4 [8]. Any of these would be an easy later target.

### 1.5 The shared world contract (L1)

```ts
export interface WorldSpec {
  memory: MemoryModel;                       // §1.5.1
  entry?: { position: number };              // PC starts at the ISA address that maps to this position (default 0)
  init?: InitSpec;                           // §1.5.4
  budget: BudgetSpec;                        // §1.5.5
  halting?: HaltingSpec;                     // §1.5.3
  ablation?: AblationSpec;                   // §1.5.2
  outputs?: OutputSpec;                      // §1.5.6
  hooks?: RawHooks;                          // escape hatch: raw WGSL bodies for any hook
}
export interface MachineSpec { isa: IsaCore; variant?: string; compat?: string; world: WorldSpec;
  workgroupSize?: number; mode?: 'kernel' | 'library'; bindingNames?: Record<string, string> }
export function buildMachine(s: MachineSpec): { wgsl: string; manifest: ShaderManifest };
```

`mode: 'library'` emits `zil_load(lane)`, `zil_run(lane, budget)` and `zil_store(lane)` with no
bindings or entry point. A multi-entry-point soup shader (Algocell's, or zilion-soup's) can then call
the machine against host-owned buffers such as `pair_data`.

#### 1.5.1 Memory models

```ts
export type MemoryModel =
  | { kind: 'mask'; bytes: number }                              // power of two; today's buildComputeShader
  | { kind: 'ring'; bytes: number }                              // any P ≥ 4; P compile-time (mul-shift %)
  | { kind: 'segments'; segments: readonly { bytes: number; countWrites?: boolean }[];
      ring?: number; padding?: 'zero-each-run' }                 // Algocell pair: [L, L], ring P ≥ 2L
  | { kind: 'storage'; bytes: number }                           // lane slice of a storage buffer (≥ 1 KB … 64 KB+)
  | { kind: 'sparse'; maxEntries: number };                      // vector tests, any address width
export interface MemoryOptions { packing?: 'byte-per-word' | 'packed' }  // packed = 4 B per u32 (cuts private memory 4×)
export function ringPos(addr: number, P: number, addrBits = 16): number;
export function highestAlias(pos: number, P: number, addrBits = 16): number;   // 2^w − 1 − ((2^w − 1 − pos) mod P)
export function seamOffset(P: number, addrBits = 16): number;                  // 2^w mod P
```

`segments` reproduces `PAIR_MEM_WGSL` (`src/lib/gpu/shaders.ts:99-128`) and the write counters
(`:198-202`) exactly. `mask` is today's model; `buildComputeShader` keeps its binding layout as a
thin wrapper.

#### 1.5.2 Ablation

```ts
export type AblationSpec = { expr: string }            // raw WGSL over (page, key)
                         | { table: 'uniform' | 'storage' };   // per-ISA packed bitsets; changed per run, no recompile
```

The mask layout comes from `IsaCore.decodePages`. For the Z80, the 3-page layout Algocell uses
(DD/FD follow base, DDCB/FDCB follow CB; `shaders.ts:67-74`) is the default, and a 7-page fine
layout is available.

The pattern grammar and family tables move from `src/lib/z80-opcodes.ts` into
`@neovand/zilion/z80` (`table()`, `resolve()`, `masks()`, `isa.json`).

#### 1.5.3 Halting and trap policy

```ts
export interface HaltingSpec {
  lethal?: { firstUnit?: number; decode?: { page: number; key: number }; probability?: number; salt?: number };  // zero-halts and the dial
  traps?: 'halt' | 'vector' | 'ignore';      // default: ISA 'as-chip'
  illegal?: 'as-chip' | 'nop' | 'halt';
}
```

The dial's randomness is **counter-based**: `zil_hash(seed, lane, step, salt) < T`. This follows
the BFF kernel's `hash32(pair, step, seed)` (`micro/bff.py:43-48`), not the Z80 dial's PCG stream
(`gen_dial_shader.py:31-44`), because the CPU twin and resumable runs then agree trivially.

The Z80 dial's exact PCG form remains reproducible through `hooks` for registered experiments.

#### 1.5.4 Initial-state conventions

```ts
export interface InitSpec {
  state?: 'zero' | 'random' | 'perLane';     // zero = the ISA descriptor's zero values (6502 P bit 5 = 1, RISC-V x0 …)
  random?: { fields?: readonly string[]; salt?: number };    // from IsaCore.state (randomizable); Algocell randreg/randsp
  stack?: { topAliases: number | 'lastByte' };              // world convention, realised by IsaCore.conventions.stackTop
}
```

#### 1.5.5 Budgets

```ts
export interface BudgetSpec { unit: 'steps' | 'fetchBytes' | 'cycles'; value: number | 'perLane' }
```

- `fetchBytes` is implemented once in L1 via `on_fetch`, so it is **ISA-neutral**. It is a natural
  normaliser for cross-ISA comparisons (a 6502 and a RISC-V "instruction" are not the same amount of
  work).
- `cycles` needs per-ISA tables and is deferred to M4.
- Outputs always include steps executed and the halt reason.

#### 1.5.6 Outputs

`memory` (E_S); `state` (the ISA's state record); `fetchMap` (T_S, `ceil(P/32)` words, uncapped);
`startMap`; `writeCounts` (per counted segment); `steps`; `haltReason`; `stepTrace` (per step:
selected registers and write records; first-divergence analysis in a single dispatch).

The Z80 contract details (hook ordering, anchors, `compat`) are in §5 and Appendix A.

### 1.6 Batch execution engine (L2)

```ts
export async function createEngine(o?: { device?: GPUDevice; gpu?: GPU; adapter?: GPURequestAdapterOptions;
                                        requiredLimits?: Record<string, number> }): Promise<Engine>;
interface Engine {
  machine(spec: MachineSpec): Promise<Machine>;     // compiles; surfaces getCompilationInfo() errors with hook text
  readonly adapterInfo: GPUAdapterInfo;             // recorded in provenance
}
interface Machine {
  readonly manifest: ShaderManifest;
  batch(capacity: number, o?: { worlds?: number }): Batch;  // worlds > 1: independent populations in one dispatch
  run(input: Uint8Array | Uint8Array[], o: RunOptions): Promise<RunResult>;  // convenience: upload → run → read
}
interface Batch {
  upload(data: Uint8Array, o?: { stride?: number; state?: Uint32Array; world?: number }): void;
  encode(enc: GPUCommandEncoder, o: { seed?: number; budget?: number | Uint32Array; ablation?: Uint32Array }): void;
  read(o?: { memory?: boolean; state?: boolean; maps?: boolean }): Promise<RunResult>;
  readonly buffers: Readonly<Record<string, GPUBuffer>>;   // for chaining with soup kernels
  destroy(): void;
}
```

Fixed by construction:

- Chunking against `maxComputeWorkgroupsPerDimension` and `maxStorageBufferBindingSize`. Today
  `zilion.ts:207` fails above 4,194,240 programs, and the default limits cap one binding at 128 MiB.
- Grow-only buffer reuse and one staging buffer.
- Typed results.
- A params ring for many encodes per submit, as `experiments/algocell_exp/soup.py:153,247-270` does.

The `Zilion` class of 0.2 stays as a thin compatibility facade.

**Manifest.** It lists the bindings, the params layout (byte offsets), the record layouts
(memory/segments, state fields, maps), the entry point, the workgroup size, `zilion` and
`coreVersion`, and `coreSha256` / `worldSha256` / `optionsSha256`. Generated WGSL starts with the same
hashes in a comment.

### 1.7 The ALife layer: what goes where

| Component | Home | Reason |
|---|---|---|
| ISA cores, twins, tables, vector adapters | Zilion L0 | |
| World semantics: memory models, ring/seam, segments, padding, hooks, ablation, halting/trap policy, init conventions, budgets, traces, write counters | Zilion L1 | Must be identical across ISAs; useful beyond ALife (GP, fuzzing, batch retro tooling) |
| Engine, manifest, Python runtime, counter-based RNG, keyed Feistel permutation | Zilion L1/L2 | Infrastructure every population method needs |
| Topologies (square-4 reflecting, hex-6, well-mixed); pairing schedulers (legacy greedy-atomic, deterministic min-claim, perfect matching); mutation operators (count-per-step, per-byte Bernoulli; race-free); step pipeline (clear/claim/execute/absorb/mutate, fused); snapshots (`.u8.br` format); population kernels (byte histogram, FNV cell hashes) | **`@neovand/zilion-soup`** | General methodology shared by the Z80, BFF and 6502 soups. It carries policy (rates, topologies) and changes at research pace, so a separate version keeps Zilion's conformance story clean. |
| Event capture for genealogy: fixed-slot records per active pair (step, i, j, before/after hashes, writes per segment, partner-entered flag, optional full bytes) | zilion-soup (mechanism) | `experiments/lod.py` today reads back everything every step on the host; slots make it cheap and deterministic |
| Culture-test *harness*: organism vs N partners in both orders, serial transfer, traced encounters, partners sampled from snapshots | zilion-soup | `experiments/algocell_exp/assay.py:163-200` generalised; ISA-agnostic |
| *Definitions and thresholds*: copy similarity ≥ 0.75 at the best cyclic shift, gen2 ≥ 0.3, faithfulness ≥ 0.5, closure/openness, inflow H(Y\|X=x), birth classes (copy/damage/novel/mutation), family→mechanism map (`isa.py:113-125`) | **Algocell (L4)** | Pre-registered science. A library default that changed would silently change results. They can graduate to `zilion-soup/metrics` after publication, frozen and versioned. |
| Rendering, UI, figures | Algocell | |

### 1.8 Packaging

```
zilion/ (monorepo, npm workspaces)
  packages/zilion/          @neovand/zilion
    src/world/  src/engine/  src/rng/  src/manifest/  src/cpu/ (transpiler)
    src/isa/z80/  src/isa/bff/  src/isa/m6502/ …        → subpaths '@neovand/zilion/z80', '/bff', '/m6502'
    python/zilion/          repo-only runtime (wgpu-py + numpy)
  packages/zilion-soup/     @neovand/zilion-soup
  vectors/                  fetch scripts + ledgers (vector data git-ignored, never in npm files)
```

The root of `@neovand/zilion` keeps exporting `Zilion`, `Z80_CORE_WGSL`, `buildComputeShader`,
`REG_FIELDS` and `REGS_PER_PROGRAM`, so existing users do not break.

The package version is shared. Every core carries its own `coreVersion` and `coreSha256`, and
provenance names those, so "which chip changed" is answerable even though the npm version is shared.

**Alternatives considered:**
- One package per ISA. More publishes, version skew between host and cores, and the world contract
  is duplicated or versioned separately.
- Folding the soup layer into Zilion. Couples emulator conformance to research policy.

---

## 2. How researchers will use it

### 2.1 TypeScript and WebGPU (browser and Node)

```ts
import { createEngine } from '@neovand/zilion';
import { z80 } from '@neovand/zilion/z80';
// Node: import { create, globals } from 'webgpu'; Object.assign(globalThis, globals); const gpu = create([]);
const engine = await createEngine(/* { gpu } in Node */);
const m = await engine.machine({
  isa: z80,
  world: {
    memory: { kind: 'segments', segments: [{ bytes: 16, countWrites: true }, { bytes: 16, countWrites: true }] },
    init: { stack: { topAliases: 'lastByte' } },
    budget: { unit: 'steps', value: 128 },
    halting: { lethal: { firstUnit: 0x00, probability: 0.03 } },
    ablation: { table: 'uniform' },
    outputs: { memory: true, fetchMap: true, startMap: true, writeCounts: true },
  },
});
const r = await m.run(pairs /* N × 32 bytes */, { seed: 1, ablation: z80.masks(['family:block-copy']) });
```

- **Browser:** the same API. Algocell's soup shader uses `mode: 'library'` until zilion-soup exists.
- **Node:** Dawn's `webgpu` package [5]. Metal on macOS; Vulkan on Linux, including lavapipe or
  SwiftShader for GPU-less CI. Deno's built-in WebGPU is a possible extra target; it was not verified
  in this audit.
- **No GPU at all:** the CPU twin (`@neovand/zilion/cpu`) runs the same machine spec, slowly, for
  tests, debugging and the app's hover trace.

### 2.2 Python (wgpu-py, exported WGSL)

**Codegen stays in TypeScript**, the single source of truth, as `scripts/export-sim.ts` already
arranges. Python never re-derives a layout by hand.

```
npx zilion export worlds.json --out shaders/     # → <name>.wgsl + <name>.manifest.json, plus --check mode
```

```python
from zilion import Machine                       # repo-only runtime, ~200 lines: wgpu-py + numpy
m = Machine.load("shaders/z80_pair_L16.manifest.json")         # verifies the SHA-256s in the manifest
out = m.run(pairs, budget=128, seed=1, ablation=masks, outputs=("memory", "fetchMap"))
out.memory, out.fetch_map, out.provenance        # provenance = zilion, core and world hashes + adapter info
```

What changes for Algocell:

- `worlds.json` replaces the 222 hand-enumerated exports (`scripts/export-sim.ts:26-44`) **and** the
  five `gen_*_shader.py` string patchers.
- The five copies of dispatch boilerplate (`assay.py:67-134`, `exectrace.py:18-74`,
  `optrace.py:33-85`, `conv.py:21-40`, `z80_ring_gpu.py:30-83`) collapse into `Machine.run`.
- The adapter policy (`soup.py:35-57`) moves into the runtime: refuse CPU adapters unless
  `ZILION_ALLOW_CPU_ADAPTER=1`.
- The Modal Vulkan recipe is unchanged.

### 2.3 Reproducibility

**Where Algocell stands today.**
- **Encounters are deterministic** (integer-only cores).
- **Z80 soups are not reproducible from a seed**, for two reasons:
  1. **Pair claims race.** `prepare_batch` (`src/lib/gpu/shaders.ts:255-267`) claims cells with
     `atomicCompareExchangeWeak` in scheduling order, and the weak form may also fail spuriously.
     The drawn pairs are seeded; *which ones win* is not.
  2. **New finding: mutations race.** `mutate_soup` (`shaders.ts:349-368`) does an unsynchronised
     read-modify-write of a u32 soup word. Two mutations landing in the same word in one dispatch
     can lose one update. At the registered setting (512 mutations per step, `soup.py:216-220`, over
     20,000 cells × 4 words) about 512·511/2/80,000 ≈ 1.6 same-word pairs occur per step. That puts
     an upper bound of about 0.3% on mutations dropped, non-deterministically.
- The project already treats each (condition, seed) as one sample (`experiments/PLAN.md:22`), measures
  within-seed variance (`PLAN.md:216`), and reports re-runs as statistical, not bitwise
  (`REVISION_PREREG.md:329`).
- BFF soups pair on the host with a seeded `rng.permutation` (`micro/bff_soup.py:107,149`), so they
  are likely bitwise reproducible.

**Proposed determinism levels** (zilion-soup):

| Level | Guarantee | How |
|---|---|---|
| `legacy` | Today's Z80 process, for registered experiments | greedy atomic claims; per-mutation RMW (the race should still be fixed; see below) |
| `strict` | Bitwise identical soups for a seed across runs, GPUs, backends and the CPU twin | (i) **counter-based randomness everywhere**: `zil_hash(seed, step, index, purpose)`, no state carried between invocations; (ii) **order-independent pairing**: a two-phase claim with `atomicMin` on hashed priorities (`min` commutes, so scheduling cannot matter), or a perfect matching from a keyed Feistel bijection for well-mixed soups, or a parity/direction lattice partition (Margolus-style: every cell paired, no atomics); (iii) **race-free mutation**: per-byte Bernoulli from the hash, one invocation per word; (iv) integer-only decisions |

`strict` changes the stochastic process. The active-pair fraction and the mutation count
distribution are different, so it is a **new condition**, not a drop-in replacement.

The mutation race should be fixed in `legacy` too, as a separately reported change. It is a bug, not
a modelling choice.

**Provenance per run:**
- zilion version, `coreSha256`, `worldSha256`;
- the zilion-soup version;
- adapter info (vendor, backend, driver);
- the wgpu-py/Dawn version;
- seed and run-seed derivation (SplitMix64, `algocell_exp/prng.py`).

The existing `shader_sha256_16` (`run.py:313-314`) becomes a subset of this record.

### 2.4 Performance at soup scale

**Measured today** (memory notes `modal-wgpu-recipe`, `morphogenesis-feature`):
- A Z80 soup step (20,000 cells; 8,192 pairs × 128 instructions ≈ 1.05 M instruction slots) takes
  ≈ 1.0 ms on an M4 and ≈ 0.35 ms on an H200. That is about 3 G slots/s, and it is
  **latency-bound**: five small dependent dispatches per step.
- A BFF epoch (2^17 programs) is **host-bound**, by the numpy pairing and gather/scatter.
- The morph engine runs 4,096 genomes in ≈ 120 ms and hits an occupancy cliff at 4 KB lanes,
  because of one byte per u32.

**Levers, in expected order of payoff:**

1. **Many worlds per dispatch.** R independent soups (replicate seeds or conditions) share each
   dispatch, through a world index in the batch. This amortises the fixed latency with no change to
   semantics. It is the open idea from the Modal note, and the main way to use big GPUs for sweeps.
2. **Fewer dispatches per step.** Fuse claim, execute and absorb: exclusive claims make in-place
   execution race-free. Clear only the claimed cells in the next step's prologue. That gives two
   dispatches per step.
3. **Many steps per submit.** The params ring, already in `soup.py:153,247-270`, becomes an engine
   feature for the browser too.
4. **Memory footprint.** Packed private memory (4 B per u32) above about 256 B lanes; storage-backed
   lanes for very large tapes.
5. **GPU pairing for well-mixed soups.** A Feistel permutation removes BFF's host loop.
6. **Readback discipline.** Statistics stay on the GPU (byte histogram and hashes already do);
   genealogy goes into fixed-slot event logs; snapshots follow a schedule.
7. **Control-flow divergence** is inherent in random code. Lane compaction for heterogeneous budgets
   is P3.

**Benchmark gate.** Fixed configurations are recorded per release on one Apple-silicon and one
NVIDIA machine:

- a Z80 L16 soup step;
- an assay of 64 k pairs;
- a BFF epoch of 2^17;
- a 6502 soup step, once that core exists.

A regression above 5% blocks a release. Hooks the world does not enable are not emitted, so
generality must cost nothing when unused.

---

## 3. Prior art and positioning

| System | Machine | Execution | World / interaction | Relation to Zilion |
|---|---|---|---|---|
| Core War (Dewdney, 1984) | Redcode (designed) | CPU (MARS) | Programs fight in a circular core with modular addressing | Conceptual ancestor of the ring world |
| Tierra (Ray, 1991) | Designed 32-instruction ISA, template addressing | CPU, time-sliced | One shared memory soup; reaper, slicer | Designed for evolvability; Zilion runs real chips in pairwise worlds |
| Avida (Adami, Ofria, Wilke; 1994–) | Designed head-based ISA | CPU | Organisms on a 2-D grid; CPU-time rewards for computing logic tasks (explicit fitness) | An evolution platform with fitness. Zilion is fitness-free substrate infrastructure. |
| cubff / *Computational Life* (Agüera y Arcas et al., 2024) [9] | BFF, Forth variants, SUBLEQ (no replicators found), Z80/8080 emulation | CUDA + CPU | Random pairing, concatenation, execution, split; background mutation | **Closest prior art**; Algocell replicates and extends it. Zilion adds conformance-tested real cores, one world contract across ISAs, built-in T_S / instruction-start / event instrumentation, and WebGPU portability (browser, Node, Python). |
| *BFF: simple explanations…* (arXiv 2607.01483, 2026) [9] | BFF | – | Mutation random walks; capped ancestry trees | Motivates first-class genealogy capture in zilion-soup |
| zff (Mordvintsev) | Z80 | Browser (wasm) | Lattice pairs, reflecting edges | Origin of Algocell's constants and neighbour rule (`src/lib/sim/constants.ts:1`, `shaders.ts:11-14`); the paper's Z80 runs used superzazu/z80 (`experiments/lit_review/VERIFICATION.md:222`) |
| CuLE (Dalton, Frosio, Garland; NeurIPS 2020) [10] | Atari 2600 (6507 + TIA) | CUDA, thousands of emulators per GPU | Lock-step game frames for RL; 40–190 M frames/hour | Shows GPU batch emulation of a real chip pays off. One fixed machine and workload; no researcher-defined worlds. |
| Designed chemistries (Nanopond, Amoeba, Stringmol); JAX ALife (Lenia-family, EvoJAX) | Designed / continuous | CPU / accelerators | Varied | Different substrate class |

**Positioning.** This audit found no tool that combines all of the following:
1. real-ISA cores tested per instruction against data-only vectors;
2. one world contract shared across those ISAs, with ablation, halting/trap dials and ring geometry
   as first-class parameters;
3. GPU batch execution that runs unchanged in the browser, in Node and from Python.

Algocell's `experiments/lit_review/` should confirm this before any paper claims it.

---

## 4. Inventory: what Algocell built around Zilion 0.2.0

### 4.1 Zilion 0.2.0 baseline and the gaps found

| Area | What exists | Gap / defect |
|---|---|---|
| Core | Full base/CB/ED/DD/FD/DDCB/FDCB; R per M1 (`core:35-36`); HALT re-executes (`core:797-801, 813`); prefix-aware ablation hook (`core:46-64, 428, 537, 840-843, 851`) | Wasted DD/FD adds 2 to R (`core:824-833`). `BIT n,(HL)` takes F3/F5 from the operand (`core:436-439`). No WZ, IM or Q. `OUT (C),r` falls to `default` (`core:658`). No reset function. |
| Builder | `buildComputeShader(memBytes,{fetchHook})` (`shader.ts:43-139`) | Power-of-two memory only. Registers out only (no R/I/IFF/IM/halted/steps). The hook is baked in, so each ablation set recompiles. |
| API | `Zilion.create/run/destroy` (`zilion.ts`) | Six buffers created per `run` (`161-201`); two `mapAsync` (`213-218`). Dispatch limit at 4,194,240 programs (`207`). Default 128 MiB binding cap, i.e. 524,288 × 256 B (`99-106`). One object per program (`232-249`). No compile errors surfaced. |
| Tests | 17 known-answer vectors (`test/conformance.js:13-178`), browser only | No differential, golden or ring tests in the repo, although the README claims continuous differential testing (`README.md:41,190`) |
| Docs | README, CHANGELOG | `fetchHook` missing from the options table (`README.md:100-105`); stale core header (`core:1-5`); R deviation undocumented; no `v0.2.0` git tag |

### 4.2 Capabilities Algocell implemented outside Zilion

| # | Capability | Where | What | Zilion 0.2 |
|---|---|---|---|---|
| 1 | Ring / modulo-P memory | `src/lib/gpu/shaders.ts:191-202, 978-987`; `createSimShader` `130-138` | Compile-time `MEM_LENGTH`; `mem[addr % MEM_LENGTH]`; private array sized `max(word-padded pair, P)` | mask only |
| 2 | 16-bit PC/SP aliasing, the seam, initial SP | `shaders.ts:77-88`; `src/lib/sim/constants.ts:43-54`; semantics `src/lib/dev/z80difftest_ring.ts:1-21,70-73` | §4.3 | undocumented |
| 3 | Pair / two-tape layout, ring padding | `shaders.ts:90-128`; `soup.py:117-118`; `assay.py:99-105` | Word-padded tapes concatenated; padding zeroed each run and never stored; hex tapes 19 B in 5 words | none |
| 4 | Write accounting | `shaders.ts:198-202, 322-323`; `soup.py:304-314` | writes landing in A's and B's halves | hookable |
| 5 | Fetch tracing (T_S) | `experiments/algocell_exp/gen_trace_shader.py:18-27`; `exectrace.py:36-80` | Patches `z80_fetch()`; 256-position cap (`gen_trace_shader.py:19`) | none |
| 6 | Instruction-start bitmap | `algocell_exp/optrace.py:14-21,48-85` | Patches after the first fetch in `z80_step` | none |
| 7 | Ablation masks, grammar, families | `shaders.ts:64-75,152-155`; `src/lib/z80-opcodes.ts:1-22,507-529`; Python `isa.py:1-107`; golden vectors `scripts/export-sim.ts:46-65` | Three 256-bit pages in the uniform (no recompile) | baked expression hook |
| 8 | Halting: zero halts; probability dial | `gen_lethal_shader.py:31-38`; `gen_dial_shader.py:25-44` | Raw first-of-step 0x00 halts (PC backed up, after `r_inc`, before prefix and ablation); dial = PCG draw seeded by (batch_seed, lane) | none |
| 9 | Init conventions | zero + `sp_init()` (`shaders.ts:301-311,1000-1004`); randreg/randsp (`gen_conv_shader.py:7-62`) | Hosts never reset `cpu_i`, `cpu_r`, `idx_*`; they rely on WGSL zero-initialising `var<private>` | `init[]` only, SP=0xFFFF |
| 10 | Budgets | `params.z80_steps` (`constants.ts:62`); halted lanes break (`shaders.ts:314-316`) | One step = one `z80_step()` | uniform only, no steps/halt-reason output |
| 11 | Batch / readback formats | executor registers `a,f,b,c,d,e,h,l,sp,pc,writes_a,writes_b` (`shaders.ts:1010-1013`); Params 32×u32 (`z80difftest.ts:117-130`, `soup.py:222-235`) | A second 12-word layout, incompatible with `REG_FIELDS` | `REG_FIELDS` |
| 12 | Python use of WGSL | `scripts/export-sim.ts:26-94` (222 files, `isa.json`, `meta.json` with SHA and zilion version); boilerplate ×5 | Bind and Params layouts rebuilt by hand | none |
| 13 | Headless GPU testing | `src/lib/dev/z80difftest_ring.node.ts:1-127` → spawns `experiments/z80_ring_gpu.py` | First-divergence tracing re-runs every budget 1..128 (`z80_ring_gpu.py:93,103`) | browser only |
| 14 | Differential oracle | `src/lib/dev/z80oracle.ts`; `z80difftest.ts`; `z80difftest_ring.ts:530-747` | §4.4 | none |
| 15 | CPU twin for the hover trace | `src/lib/sim/soup.ts:32-81` → AI-written `src/lib/sim/z80.ts` | Diverges from the GPU for P ≠ 32 (Appendix C) | none |
| 16 | **A second machine, the world rebuilt** | `experiments/micro/bff.py:33-129` (BFF WGSL: storage-buffer pair, `ip_wrap`, hash-based bracket dial, alphabet map, records steps/entered/max-pc/writes) | The same world concepts, implemented again with different RNG and layout | n/a |
| 17 | **8080 as a Z80 ablation subset** | `experiments/make_conds.py:63`; `check_8080_closer.py` | Removes Z80-only instructions. Keeps Z80 flag semantics (P/V overflow, H/N, DAA), Z80 timing-free R, and DD/FD/ED decoding of bytes that alias other instructions on an 8080. | not an 8080 |
| 18 | Performance tricks | §4.5 | | |

### 4.3 The 16-bit aliasing semantics

- The core forms every address as a 16-bit value (`core:79, 90-101, 568, 675`); the host maps it.
- **Mask model:** seamless.
- **Ring model:** when P ∤ 65536, 0xFFFF and 0x0000 sit at positions `65535 mod P` and 0, so any
  crossing of address 0 lands at `(65536 + t) mod P`.
- **Initial SP** = `highestAlias(pair_length − 1, P)` = `0xFFFF − ((0xFFFF − (pair_length−1)) mod P)`.
  Values: P=38 → 0xFFE7; 40 → 0xFFEF; 100 → 0xFFDB. It equals 0xFFFF iff P | 65536.
- A "toroidal" CPU whose register arithmetic wraps mod P is **not offered**: registers are also
  numbers.

### 4.4 The oracle (`z80-emulator` 2.3.0): defects and workarounds

| Defect | Workaround | Events (seed 1, all sizes; `experiments/results/emulator/RING_DIFFTEST.md:17`) |
|---|---|---|
| NOPs DD/FD opcodes without an IX/IY form | console.log intercept (`z80oracle.ts:18-33`); `oracleCannotDecode` + consume the prefix as one M1 (`z80difftest_ring.ts:605-625,669-674`) | 279,591 |
| `LD A,R` returns `R & 0xF7` | A := (R7&0x80)\|(R&0x7F), F recomputed (`:675-680`) | 492 |
| `LD R,A` leaves R7 unset | R7 := A&0x80 (`:689-693`) | 96 |
| `IN F,(C)` loses carry | F := (oldF&C)\|sz53p(0) (`:681-684`) | 181 |
| `ADC/SBC HL,rr` Z flag from the 17-bit sum | Z := (HL==0) (`:685-688`) | 656 |
| *Harness:* the browser "hex" mode compiled L=16 (the GPU wrapped mod 32, the oracle mod 38) | `makePipeline()` ignores L (`z80difftest.ts:231-243`); the ring driver fixes it (`z80difftest_ring.node.ts:203`) | – |

Zilion deviations found by the same runs:

- `bit-mem-f3f5`: 1,204 events (seed 1) and 1,368 events (seed 2).
- `wasted-prefix-r`: 3 events (seed 1) and 0 events (seed 2).

There were 0 unexplained divergences.

### 4.5 Performance tricks in use

1. A private array with one byte per u32, unpacked and packed once per lane.
2. Compile-time P, so `%` by a constant is cheap.
3. Halted lanes break early.
4. Ablation masks in the uniform.
5. One encoder per soup step; in Python, 32 steps per submit through a params ring.
6. Pipelines cached per (L, P, rule).
7. Dispatches chunked at 262,144 cases.
8. Adapter policy that never uses llvmpipe in production; scale by fanning out soups.
9. Throughput credited on `onSubmittedWorkDone` (`src/lib/gpu/engine.ts:593-596`).

---

## 5. Correctness work (Z80)

### 5.1 Z80 hook ordering and compatibility

**Order within `z80_step`:** `on_fetch(pc)` → `r_inc` → `on_instruction_start(pc, byte)` →
[DD/FD: `on_fetch(pc+1)`; a wasted prefix returns here] → `on_decode` / `on_fetch_opcode` →
[displacement: `on_fetch`] → execute.

`ZIL_HALT` reproduces Algocell's lethal hunk exactly: `cpu_halted = 2`, PC set back to `pc`, R
already incremented.

The Z80 keeps the name `on_fetch_opcode(prefix, op)` (0.2 contract) as its `on_decode` form.

**Back-compat rules:**
- New hooks are opt-in at generation time. WGSL has no default functions, so a bare
  `Z80_CORE_WGSL` must never call an undefined hook.
- The anchors in Appendix A stay byte-stable through 0.3.x.
- New symbols use the `z80_`, `cpu_` and `zil_` prefixes.

### 5.2 R register: wasted DD/FD prefix

**The bug** (`core:822-835`). On DD/FD followed by DD/FD/ED, the core counts the follower's M1
(`:825`), then backs PC up (`:831`). The next step fetches and counts the follower again. The wasted
prefix therefore costs R +2; a real Z80 costs +1.

**The patch** keeps every anchor:

```wgsl
    if (op == 0xddu || op == 0xfdu) {
        idx_mode = select(2u, 1u, op == 0xddu);
        let next = z80_fetch();
        if (next == 0xddu || next == 0xfdu || next == 0xedu) {
            // Wasted prefix: one M1 (counted above). `next` is fetched and counted again as the next step's first M1.
            cpu_pc = (cpu_pc - 1u) & 0xffffu;
            return;
        }
        r_inc(); // M1: opcode after the prefix
        op = next;
    }
```

**Why not a peek.** Algocell's traced executor marks the follower as fetched during the wasted step.
A `mem_read` peek would change `fetchMap` whenever the budget ends right after a wasted prefix.

**Tests.** `DD FD 21 34 12 ED 5F` gives A = 5. `DD DD DD 00 ED 5F` gives A = 6.
`compat:'0.2'` reproduces the old values. The ring difftest's `wasted-prefix-r` count goes to 0.

### 5.3 MEMPTR (WZ): full model

The rules come from the MEMPTR documents (boo_boo & Kladov), as checked by the redcode Z80 test suite
against real CPUs [1][2]. Zilog NMOS behaviour is assumed.

| Instruction | WZ after | Zilion site |
|---|---|---|
| `LD A,(BC)` / `LD A,(DE)` | rp + 1 | `core:702-703` |
| `LD (BC),A` / `LD (DE),A` | (A≪8) \| ((rp+1)&0xFF) | `core:694-695` |
| `LD A,(nn)` | nn + 1 | `core:705` |
| `LD (nn),A` | (A≪8) \| ((nn+1)&0xFF) | `core:697` |
| `LD HL/IX/IY,(nn)`, `LD (nn),HL/IX/IY`, ED `LD rr,(nn)` / `LD (nn),rr` | nn + 1 | `core:696,704,563-577` |
| `ADD HL/IX/IY,rr`, `ADC HL,rr`, `SBC HL,rr` | old HL/IX/IY + 1 | `core:348-356,579-604` |
| `RLD`, `RRD` | HL + 1 | `core:606-618` |
| `JR`, `JR cc` (taken), `DJNZ` (taken) | target | `core:672-684` |
| `JP nn`, `JP cc,nn` (taken **or not**); `CALL nn`, `CALL cc,nn` (taken **or not**) | nn | `core:744,747,769,774` |
| `RET`, `RET cc` (taken), `RETN`, `RETI` | popped PC | `core:724,729,552-554` |
| `RST p` | p | `core:782` |
| `EX (SP),HL/IX/IY` | new HL/IX/IY | `core:751-758` |
| any `(IX+d)`/`(IY+d)` instruction, including DDCB/FDCB | IX/IY + d | `core:855-858`, `848-852` (after the ablation hook) |
| `IN A,(n)` | ((A_before≪8)\|n) + 1 | `core:750` |
| `OUT (n),A` | (A≪8) \| ((n+1)&0xFF) | `core:749` |
| `IN r,(C)`, `IN (C)`, `OUT (C),r`, `OUT (C),0` | BC + 1 | `core:621-626`; **new** cases for ED 41/49/51/59/61/69/71/79 |
| `LDIR`/`LDDR` repeating | instruction address + 1 | `core:541-542` |
| `CPI`/`CPD` | WZ ± 1; `CPIR`/`CPDR` repeating: instruction address + 1 | `core:543-546` |
| `INI`/`IND` (+R forms) | BC_before ± 1 (repeat case to be verified against vectors) | `core:630-643` |
| `OUTI`/`OUTD` (+R forms) | BC_after ± 1 | `core:646-657` |
| `LDI`, `LDD`, `JP (HL)`, `LD SP,HL`, PUSH, POP, an ablated instruction | unchanged | – |

**The only flag change** is in `BIT n,(HL)`, where F3/F5 = `(cpu_wz >> 8) & 0x28` (`core:436-439`,
`z == 6`). `BIT n,(IX+d)` already uses the address (`core:467`). Nothing else reads WZ.

### 5.4 Other items found (scoped separately)

| Item | Zilion 0.2 | Observable in Algocell? | Decision |
|---|---|---|---|
| IM register | ignored | no (no interrupts) | model in M1 (needed by the SingleStepTests `im` field) |
| I/O ports | IN reads 0, OUT no-op | no | optional `io_*` hooks; the default is unchanged |
| **Q latch** (SCF/CCF X/Y = ((Q⊕F)\|A)&0x28, Zilog NMOS) | A-only (Fuse-like) | **yes, often**: SCF/CCF are common bytes, and F reaches memory via `PUSH AF` | **opt-in** `variant: 'zilog-nmos'`; not the default (vendor-specific) |
| Block-repeat X/Y from PC bits 13/11 (+ P/V, H for INxR/OTxR) [4] | no | only in the final F when the budget expires mid-repeat; discarded | M2; mask in the vectors until then |
| PC while halted | parked on HALT | no | keep; map in the vector adapter |

### 5.5 `compat: '0.2'`

This is a generator switch: double R on wasted prefixes, no WZ (`BIT n,(HL)` X/Y from the operand),
no IM. It is what makes the hook migration provable without mixing in the fixes.

The exact 0.2.0 text stays available from the immutable npm tarball and from Algocell's committed
`experiments/algocell_exp/shader/*.wgsl`.

A test asserts that `compat:'0.2'` reproduces outputs recorded from 0.2.0, stored as data: 100 k
random programs at P = 32, 38, 40, 64, 100 and 128.

### 5.6 Validating the fixes

Three lineages:

- **SingleStepTests:** the `wz` and `r` fields per instruction.
- **z80-emulator:** a Fuse-lineage core that models memptr (`z80oracle.ts:117`). The
  `bit-mem-f3f5` count must drop to 0.
- **Analytic probes:** R and WZ tests written from the tables above.

### 5.7 Risk to Algocell's existing results

**The user's statement holds for these two fixes.** Only two things change:

- (a) the value `LD A,R` loads after at least one wasted prefix in the same encounter. R starts at 0
  each encounter, and only `LD A,R` reads it.
- (b) F3/F5 immediately after `BIT n,(HL)`. That is MEMPTR's only architectural observable; IM is
  unobservable.

**Qualification.** Neither change is benign by construction in a self-modifying soup:

- a wrong A can be stored anywhere;
- F3/F5 reach memory through `PUSH AF`, or through `EX AF,AF'` followed by `PUSH AF`;
- no conditional ever tests F3/F5, so control flow is never directly affected.

The browser difftest's "benign F3/F5-only" class (`z80difftest.ts:379-383`) is valid per run,
because memory matched in those runs. It is not a general proof.

**Bound from data.** Only programs whose ring-difftest trace contains a Zilion event can change.
Those are 24–76 per 100,000 random programs (0.024–0.076%) at P = 32–128 in both seeds
(`ring_difftest_seed{1,2}.json`, outcome `zilion-deviation`, random column). Of them,
`wasted-prefix-r` accounts for 3 of 1.2 M programs. The subset whose memory changed is unmeasured,
because the tracer adopted the GPU's F.

**Why that is not enough.** Evolved tapes are not random. A dominant replicator containing
`BIT n,(HL)` followed by `PUSH AF`, or one using `LD A,R`, would affect every interaction it takes
part in. Runs are already non-deterministic (§2.3), so the question is distributional.

**Impact study** (M1, local GPU only, protocol pre-registered before it runs):

1. **Random programs.** The ring-difftest corpora, plus the odd and padded rings of Stage F
   (P = 33, 35, 37, 73, 75, 79; the `L_P` variants in `scripts/export-sim.ts:29-35`), which the
   ring difftest never covered. Count programs whose final memory differs between the two cores, and
   attribute each through `startMap`.
2. **Archived soups.** 100 k lattice-neighbour pairs from every registered condition's emergence and
   final snapshots, run under both cores. Report the affected-pair rate and the opcodes responsible.
3. **Assays.** The replication assay on the paper's named tapes under both cores. Identical scores
   are expected.
4. **Decision rule.**
   - If the affected-pair rate is below 1e-4 in every condition and the assays are identical, the
     results stand. Methods states "Zilion 0.2.0, two documented deviations, measured impact …".
   - Otherwise the affected conditions are flagged for re-runs. Any paid re-run needs explicit
     approval (memory note `feedback-rigor-before-spend`).

Registered experiments in progress stay on 0.2.0. New pre-registrations choose.

---

## 6. Testing (every core, every world)

### 6.1 Headless harness

| Tier | Engine | Use |
|---|---|---|
| **A (primary)** | Node 22 + Dawn (`webgpu` npm) [5] | Every GPU suite. macOS uses Metal. Linux CI uses `mesa-vulkan-drivers` (lavapipe) or SwiftShader via `VK_ICD_FILENAMES`. The cores are integer-only, so a CPU adapter must agree bit-for-bit with a GPU, and any disagreement is a compiler or driver bug worth surfacing. Tests set `ZILION_ALLOW_CPU_ADAPTER=1`. |
| **B** | CPU twin: a transpiler from the Zilion WGSL subset to JS, run on the *same* core and world text | Fast unit tests, single-stepping, GPU-less runs, the app's hover trace. Validated by bit-exact GPU-versus-twin fuzz: two independent compilers of one source. Spike `wgsl_reflect`'s `WgslExec` (an AST interpreter [6]) for a day first; it may do for tests but is likely too slow for the app. A linter enforces the subset in core sources. |
| C | wgpu-py | The Python runtime's own tests; keeps the Modal path honest |

The browser page `test/index.html` survives as a demo, not a gate.

### 6.2 Per-instruction golden vectors (data only)

The generic runner handles any core:

- Input is JSON `{name, initial:{…, ram:[[addr,val]]}, final:{…}}`, mapped through the core's
  `VectorAdapter`.
- Memory is the `sparse` model: per-lane linear-search tables. An unlisted read sets an error bit; a
  write updates or appends.
- Ports come from the vector's `ports` list through `io_read`.
- State is loaded and stored with `<id>_load` / `<id>_store`.
- One lane per test; budget 1.
- **Divergence ledger** (`vectors/<isa>/divergences.json`): entries of the form
  `{files, fields, mask, reason, source, status}`. CI fails on any unlisted mismatch *and* on any
  stale entry.
- **Why single-step vectors suffice:** if every instruction maps *complete* state correctly
  (including WZ, Q, R, halted), sequences are correct by induction. The world side is covered by §6.3
  and §6.4.

**Z80 source:** SingleStepTests/z80 (MIT) [3].
- About 1,600 files × 1,000 tests. The repository listing truncates at 1,000 files, with 604 more
  omitted.
- Fields: `a b c d e f h l i r ix iy pc sp wz af_ bc_ de_ hl_ im ei iff1 iff2 p q ram ports cycles`.
- Lineage: JSMoo, translated from Ares and bug-fixed. Independent of the Fuse lineage, but not
  hardware truth.
- A fetch script pins a commit and a SHA-256 per file. Data is git-ignored, cached in CI, and never in
  npm `files`.
- **Downloading needs the user's approval; this audit downloaded nothing.**

**Other cores:**
- 65x02 [7]: 6502, 65C02 variants, NES.
- 8086/8088: **hardware-generated** [8].
- RISC-V: riscv-arch-test, a *program-signature* runner (memory model `storage`; trap → halt; compare
  the signature region).
- PDP-11: no data-only vectors were found. DEC diagnostics could run as programs (pass = halt at a
  known address), or vectors could be generated once from a reference model. This is a risk (§7.2).

**Nightly (optional):** hardware-verified Z80 programs (Patrik Rak's z80test CRC suites, zexall).
These are programs, not emulator code. They need `storage` memory and resumable chunked runs.
Licences vary, so fetch them at CI time and never vendor them.

### 6.3 World tests (shared, instantiated per ISA)

- **Ring families**, generated from each core's probe encoders: `z80difftest_ring.ts:185-443` made
  ISA-generic. They cover relative branch back and forward across address 0, sequential run-off,
  push/pop wrap, the stack-top convention, call/ret wrap, block copies, indexed and absolute 16/20/32-bit
  operands at the top of the address space. All expectations are analytic; no reference emulator is
  involved.
- **Sizes:**
  - mask: 16, 32, 64, 128, 256;
  - ring: 32, **33, 35, 37**, 38, 40, 64, **73, 75, 79**, 100, 128;
  - padded segments: every (L, P) in `scripts/export-sim.ts:29-35`.
- The pop-init family assumes an odd SP0 (`z80difftest_ring.ts:278`), which holds only for even P. It
  needs generalising.
- **Metamorphic:** ring with P = 2^k equals mask with 2^k, bit for bit.
- **Seam coverage:** port `seamCoverage` (`z80difftest_ring.ts:762-778`).

### 6.4 Fuzz, invariants and determinism

- **Generators:** uniform bytes; prefix-heavy (rare paths); random full state; random ablation sets;
  random P; random hook configurations.
- **Invariants:**
  - registers stay within their widths;
  - memory changes only at logged writes;
  - **hook neutrality:** hooks that always return CONTINUE/false equal no hooks;
  - **resumability:** n steps = k steps + save/load + (n−k) steps;
  - chunk and workgroup-size invariance;
  - `compat:'0.2'` equals the recorded 0.2.0 outputs;
  - GPU equals the CPU twin;
  - **strict-mode soups** are bitwise identical across two runs, Metal versus Vulkan, and GPU versus
    twin.

### 6.5 Mutation tests

Port `MUTANT_DEFS` (`z80difftest_ring.node.ts:160-177`). Add mutants for:

- a dropped `r_inc`;
- WZ errors;
- the F3/F5 source;
- a misplaced hook call;
- a broken ring mapping;
- the stack-top rule.

Each mutant must be killed by some suite.

### 6.6 What happens to the z80-emulator differential

It becomes a dev-only cross-check (`npm run difftest:fuse`), not a CI gate, with the five named
corrections from §4.4. Algocell keeps its own copy as paper provenance.

### 6.7 CI matrix

`{ubuntu-latest + lavapipe, macos-14 (Metal)} × node 22`. Every run covers:

- conformance;
- a vector subset per core;
- world tests;
- 1 M-program fuzz;
- mutants;
- anchors;
- `compat`;
- the manifest schema;
- the Python runtime on wgpu-py.

Nightly adds the full vector sets, 10 M-program fuzz, the program suites and the benchmarks.

---

## 7. Staged roadmap

### 7.1 Milestones

| Milestone | Scope (work items, §0) | Exit criteria | Version |
|---|---|---|---|
| **M0 Freeze** (days) | Pin Algocell to `"@neovand/zilion": "0.2.0"` exactly (it is `^0.2.0` today). Tag `v0.2.0` at `8557d3b` (push after approval). Write `SEMANTICS-0.2.md` describing the current behaviour, deviations included. | Algocell lockfile and `package.json` pinned in one PR (memory note) | – |
| **M1 Foundations: the Z80 on the new architecture** | 1–9: harness; vector runner + Z80 ledger; R/WZ/IM fixes + `compat`; hooks; world spec and memory models; monorepo + `IsaCore` + `buildMachine` + manifest + outputs; batch engine; Python runtime | (a) Every Algocell executor and soup shader regenerates from world specs and is **bit-identical in outputs** under `compat:'0.2'` (1 M cases each, fixed seeds), so the `gen_*_shader.py` patchers can be retired. (b) Z80 ledger reviewed, nothing unexplained. (c) Fixes validated by three lineages (§5.6). (d) Impact study reported (§5.7). (e) CI green on lavapipe and Metal. (f) The Python runtime reproduces `assay.py` results bitwise. | 0.3.0 |
| **M2 The contract meets other machines** | 10–12, 16: CPU twin; BFF core; 6502 (NMOS); Z80 `zilog-nmos` variant; block-repeat flags; storage/packed memory | (a) The BFF core is bitwise identical to `micro/bff.py` on fixed seeds (steps, entered, max-pc, writes, memory; hash dial included). (b) 6502 ledger clean against 65x02 vectors. (c) **One world spec, three ISAs**: the L=16 pair, ring 32, 128 steps, fetch maps, run unchanged on Z80, BFF and 6502. (d) Every contract change those two cores forced is folded in, and **host contract v1 is frozen**. | 0.4.0 |
| **M3 Soup toolkit and scale** | 13–14: `zilion-soup` (topologies, pairing schedulers, race-free mutation, fused step pipeline, many soups per dispatch, event logs, snapshots, culture-test harness); strict determinism; benchmark gate | (a) Algocell's lattice soup runs through zilion-soup in `legacy` mode, with **pre-registered distributional equivalence** to the current engine. (b) `strict` mode is bitwise identical across Metal, Vulkan and the twin. (c) The throughput gain from many soups per dispatch is measured against a baseline taken at the start of M3 (no number guessed now). (d) Genealogy event logs reproduce `lod.py` records on a validation world. | zilion 0.5.0, zilion-soup 0.1.0 |
| **M4 More chips** | 15: 8086 (hardware vectors), RV32I (architecture tests), a real 8080 core (replacing the ablation subset), PDP-11 (only after a vector-source decision); `cycles` budgets | Each core ships `experimental` until its ledger is clean and the world tests pass; the cross-ISA world runs on every core | 0.6+ |
| **M5 1.0** | API freeze after the paper; docs site | Two consecutive minors with no contract changes | 1.0.0 |

**Why BFF before the 6502.** BFF is the smallest core (about 150 lines of existing WGSL with a
working Python host and published semantics), and it is *not bus-shaped*. It forces ring-native
addressing, alphabet remaps and trap-style halting into the contract before the 6502 forces
page-fixed stacks, JAM opcodes and decimal mode. Two very different second machines are the cheapest
protection against a Z80-shaped contract.

### 7.2 Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | The contract is Z80-shaped and has to break when the 6502, 8086 or RISC-V arrive | med | high | BFF + 6502 in M2 *before* the contract is frozen; probe-encoder world tests; `experimental` flags for new cores |
| R2 | The fixes or migration change registered results | low (bounded at 0.024–0.076% of random programs) | high | Pin 0.2.0; `compat`; bit-identical migration proof; pre-registered impact study; registered work stays on 0.2.0 |
| R3 | Vectors are not hardware truth (Z80 SST comes from JSMoo/Ares) | med | med | Triangulate (SST, Fuse lineage, analytic, hardware program suites); ledger with sources; hardware-generated 8086 sets |
| R4 | No data-only vectors for PDP-11 (and only program signatures for RISC-V) | high | med | Signature runner; DEC diagnostics; defer PDP-11 until a source is agreed; never ship it un-gated |
| R5 | GPU compiler or driver bugs (Tint, Naga, vendor drivers) | med | med | CPU twin + cross-backend bitwise CI; integer-only cores |
| R6 | Generality costs throughput | med | med | Hooks not emitted when unused; benchmark gate (>5% blocks a release) |
| R7 | Strict determinism changes the stochastic process | certain | med | A new, labelled condition; `legacy` kept; equivalence studied, not assumed |
| R8 | The ALife layer grows into a framework that absorbs research definitions | med | med | Separate package; definitions and thresholds stay in L4 until published and frozen |
| R9 | The twin's WGSL subset constrains core authors | med | low | Document and lint the subset early (M1); a small subset suffices for interpreters |
| R10 | Unstable or undocumented opcodes (6502 XAA/ANE etc.; 8086 aliases) | high | low | Pin constants as named variants; vectors decide; document |
| R11 | Maintenance load for one maintainer (cores × vectors × ledgers) | high | med | Strict acceptance checklist per core (§1.3); `experimental` tier; no core without a vector source |
| R12 | Dawn's Node bindings are experimental | med | low | Tier C (wgpu-py) as a fallback runner for every suite |

### 7.3 Versioning and back-compat policy

- **No behaviour change in any 0.2.x release**: Algocell's `^0.2.0` would pick it up silently.
- Each 0.x minor may change behaviour, but only with a CHANGELOG "Behavior changes" section and a
  `compat` switch.
- The root exports stay (`Zilion`, `Z80_CORE_WGSL`, `buildComputeShader`, `REG_FIELDS`,
  `REGS_PER_PROGRAM`). The binding layout of `buildComputeShader` is unchanged.
- Appendix A's anchors stay byte-stable through 0.3.x.
- Provenance constants are exported: `ZILION_VERSION`, and per core `coreVersion` and `coreSha256`.
- The host contract gets its own version (`HOST_CONTRACT = 2`, frozen at the end of M2).

### 7.4 Release mechanics

- **Pre-release without publishing:** `npm pack` produces a tarball; Algocell installs it from the
  file on a branch, regenerates its shaders, and runs the equivalence and impact studies. No npm
  dist-tags.
- **Gates for each minor:** all §6 suites green on both platforms; ledgers reviewed; benchmark gate;
  docs reviewed; the `npm pack --dry-run` file list inspected.
- **Docs:**
  - README (options table including `fetchHook`; scope; limits);
  - `docs/ARCHITECTURE.md` (§1);
  - `docs/SEMANTICS.md` (step, budget, halting, memory and seam, R, WZ, ablation accounting);
  - `docs/HOST_CONTRACT.md` (hooks, order, state at each call, anchors);
  - `docs/ADDING_A_CORE.md` (the §1.3 checklist);
  - `docs/CONFORMANCE.md`;
  - `docs/MIGRATION-0.3.md`;
  - `docs/REPRODUCIBILITY.md`;
  - a CHANGELOG per release, with per-core sections;
  - fix the stale core header (`core:1-5`).
- **0.3.0 CHANGELOG draft:**
  - **Behavior changes:** a wasted DD/FD prefix adds 1 to R; WZ is modelled, and `BIT n,(HL)` takes
    F3/F5 from WZ; IM is modelled. `core.compat:'0.2'` restores the old semantics.
  - **Added:** host contract v2 hooks; `z80_reset`/`z80_run` and halt reasons; world spec (ring,
    segments, padding, init, budgets, halting, ablation tables, outputs); `buildMachine` + manifest;
    batch engine; provenance constants; Python runtime (repo only).
  - **Fixed:** large-batch dispatch and binding limits; buffer churn and the double readback;
    compile errors surfaced.
  - **Tests:** a headless harness, per-instruction vectors with a ledger, world tests, fuzz, mutants.
- **Publishing:** `npm publish` (with `--provenance` from CI), pushing git tags, and any PyPI release
  of the Python runtime happen **only after the user approves each one explicitly**.

---

## Appendix A. Anchors Algocell asserts on Zilion's core text (byte-stable through 0.3.x)

| Exact text | Used by |
|---|---|
| `"    var op = z80_fetch();\n    r_inc(); // M1: opcode (or prefix) fetch\n"` | `gen_lethal_shader.py:31`, `gen_dial_shader.py:26`, `optrace.py:14` |
| `"    let val = mem_read(cpu_pc);\n    cpu_pc = (cpu_pc + 1u) & 0xffffu;\n    return val;"` | `gen_trace_shader.py:20` |
| `"var<private> cpu_halted: u32;"` + `\n` | `gen_dial_shader.py:25`, `gen_conv_shader.py:24`, `gen_trace_shader.py:19` |
| `"if (op == 0xddu || op == 0xfdu) {"` before `"if (on_fetch_opcode(pfx, op)) { return; }"` | `gen_lethal_shader.py:52` |
| `cpu_pc = (cpu_pc + 1u) & 0xffffu`, `cpu_sp = (cpu_sp ± 1u) & 0xffffu`, `cpu_pc = u32(i32(cpu_pc) + d) & 0xffffu` | mutants, `z80difftest_ring.node.ts:160-177` |

A contract test asserts that each anchor occurs exactly once in the default `Z80_CORE_WGSL`. The
commitment ends at 0.4.0, once Algocell uses hooks.

## Appendix B. Z80 divergence-ledger seed (0.2.0 → 0.3.0)

| ID | Instructions | Fields | 0.2.0 | 0.3.0 |
|---|---|---|---|---|
| R-WASTED | DD/FD followed by DD/FD/ED | r | +2 per wasted prefix | fixed |
| WZ | §5.3 | wz | not modelled | fixed |
| BIT-HL-XY | CB 46…7E | f & 0x28 | from the operand | fixed (from WZ) |
| IM | ED 46/4E/56/5E/66/6E/76/7E | im | not modelled | fixed |
| IO | IN/OUT/block I/O | regs, ram | port reads 0 | vector ports via `io_read`; default unchanged |
| Q-XY | 37 (SCF), 3F (CCF) | f & 0x28 | A only | known; opt-in `zilog-nmos` (M2) |
| REP-XY | ED B0/B1/B2/B3/B8/B9/BA/BB repeating | f | result-derived | known (M2) |
| HALT-PC | 76 | pc | parked on HALT | convention; mapped by the adapter |

## Appendix C. Out-of-scope findings in Algocell (for the caller)

1. **Mutation write race**, new (§2.3). `mutate_soup` (`src/lib/gpu/shaders.ts:349-368`) does an
   unsynchronised read-modify-write on u32 words. There are ≈1.6 same-word mutation pairs per step at
   the registered setting, so up to about 0.3% of mutations may be dropped, non-deterministically.
   Fix with per-word ownership, or with per-byte Bernoulli mutation from a counter-based hash. Report
   the fix as a separate engine change.
2. **Hover trace.** `traceCell` (`src/lib/sim/soup.ts:32-81`) runs the AI-written `z80.ts` with
   SP = 0xFFFF (`z80.ts:27`), `getTapeLength(grid)` (always 16 for square) and `getPairLength`. It
   diverges from the GPU for hex (P=38, SP0 0xFFE7) and for any square L ≠ 16.
3. **Browser difftest hex mode.** It ran a 32-byte ring (`z80difftest.ts:231-243`). Superseded, but
   the bug is still in that file.
4. **Trace cap.** The traced executors cap P at 256 (`gen_trace_shader.py:19`).
5. **Private zero-init.** Both hosts rely on WGSL zero-initialising `cpu_i`, `cpu_r` and `idx_*`
   (`shaders.ts:301-311,1000-1004`). That is unsafe for persistent-lane hosts.
6. **Ring coverage gap.** The ring difftest validated P = 2L, even, only. The odd and padded rings
   used in Stage F were not covered.
7. **The `i8080` condition** (`experiments/make_conds.py:63`) is a Z80 with instructions removed, not
   an 8080. Flag semantics (P/V as overflow, H/N, DAA) stay Z80, and bytes that alias other
   instructions on an 8080 (e.g. 0xCB = JMP, 0xD9 = RET, 0xDD/0xED/0xFD = CALL) are decoded as Z80
   prefixes. Worth one sentence in Methods; a real 8080 core is an M4 item.
8. **Version range.** Algocell depends on `^0.2.0`. Pin `0.2.0` exactly until M1 is decided.

---

## Sources (web, consulted 2026-10-10)

1. redcode Z80 wiki, *Z80 Test Suite*: https://github.com/redcode/Z80/wiki/Z80-Test-Suite
2. MEMPTR rules as implemented in an emulator project: https://git.applefritter.com/6502/CLK/commit/53f05efb2dd7d0edc49526afec92e20955c984e4
3. SingleStepTests/z80 (fields, JSMoo/Ares lineage, MIT): https://github.com/SingleStepTests/z80
4. redcode Z80 wiki, *Z80 Block Flags Test*: https://github.com/redcode/Z80/wiki/Z80-Block-Flags-Test
5. `webgpu` npm (dawn.node): https://cdn.jsdelivr.net/npm/webgpu@0.4.0/README.md; Dawn node bindings: https://dawn.googlesource.com/dawn/+show/73591422f504902c597a67beaeb8e22101f030bd/src/dawn/node/README.md
6. wgsl_reflect shader execution: https://raw.githubusercontent.com/brendan-duncan/wgsl_reflect/main/docs/shader-execution.md
7. SingleStepTests/65x02 (MIT): https://github.com/SingleStepTests/65x02
8. SingleStepTests organisation (z80, 65x02, 8086/8088/80286/80386 hardware-generated, m68000, sm83, …): https://github.com/SingleStepTests
9. Agüera y Arcas et al., *Computational Life* (2024): https://arxiv.org/pdf/2406.19108; follow-up *BFF: Simple explanations for complex phenomena* (2026): https://www.emergentmind.com/papers/2607.01483
10. Dalton, Frosio and Garland, *Accelerating Reinforcement Learning through GPU Atari Emulation* (CuLE, NeurIPS 2020): https://arxiv.org/pdf/1907.08467

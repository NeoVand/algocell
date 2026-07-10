# Developmental Computation — Technical Report & Handoff

**Purpose.** Hand this research off to a stronger collaborator. It states the thesis, inventories
every result with numbers and file pointers, gives an honest gap analysis toward a Nature Machine
Intelligence–grade paper, and a prioritized roadmap. Read §6 (gaps) and §7 (roadmap) first if you
are deciding what to do next.

Branch: `feat/morphogenesis-ca`. Code roots: `src/lib/devcomp/` (the paper's rule + demo + GPU
trainer + Z80 proof), `src/lib/morph/` (the older Exp A–I lineage + the real-Z80 tooling). Companion
docs: [`devcomp-build-plan.md`](devcomp-build-plan.md) (execution log with per-stage findings),
[`gradient-morphogenesis.md`](gradient-morphogenesis.md) (Exp A–D method history),
[`gpu-trainer-design.md`](gpu-trainer-design.md).

---

## 1. Thesis

**One local rule, learned by gradient descent through development, grows a functional, self-repairing,
position-invariant digital computer — and that computer is literally a program in a real ISA whose
exact training gradient the substrate itself carries.**

Positioning (what makes this novel):
- **vs Neural CA (Growing NCA and descendants):** they grow *images/patterns*; here the CA grows a
  *computer* — a rule whose I/O is placed arbitrarily and that keeps *computing* under damage,
  asynchrony, and live input changes.
- **vs classic Artificial Life:** those evolve discrete rules by black-box search (no gradient); here
  the rule is trained by exact gradient through the developmental rollout (BPTT), and *beat evolution
  on evolution's own hardest target* — see Exp C (the letter "F" that MAP-Elites could not make).
- **vs learned-CA-on-a-fixed-grid work:** those have no growth, no repair, and are not a program in a
  real instruction set. Here the same rule grows from a seed, heals, and **runs on a real Z80**, with
  its training gradient computable *in the same substrate* (forward-mode AD / dual numbers).

The one-sentence differentiator, now proven for the actual trained rule (§4): *programs and gradients
on real silicon.*

---

## 2. The rule (single source of truth: [`src/lib/devcomp/rule.ts`](../src/lib/devcomp/rule.ts))

Per interior cell, per step:
```
perceive = [identity, gx=(right−left)/2, gy=(down−up)/2, laplacian]  per channel   (FEAT=4)
dl       = W2 · relu(W1 · perceive + b1) + b2
state'   = tanh(state + dl)          # residual INSIDE the tanh
```
- Inputs are clamped into a signal channel every step; border cells stay 0; damage zeros a cell's
  channels then the input clamp re-applies. Param layout `[W1(HD×PERC), b1, W2(C×HD), b2]`.
- **Config-driven** (`RuleConfig`, `makeConfig`): one code path serves any grid/channel/hidden size.
- **Position invariance** uses marker channels (`IN_MARK`=ch1, `OUT_MARK`=ch2, re-stamped each step);
  the rule reads markers, never absolute coordinates → ports are draggable, grid-size agnostic.
- **Movable XOR needs separated signal channels** (input k → its own channel; ch0 = readout). A single
  shared channel blends the two bit-waves into one geometry-dependent scalar → XOR becomes unlearnable.
  This was *representational, not capacity* (bigger models did not help).
- **Persistence (hold-window) loss** makes the answer a genuine attractor (score the last `whold`
  states), fixing metastable drift.
- **Reactivity**: a mid-rollout input flip with a two-window loss trains the field to re-settle to the
  new answer without re-seeding.
- **Async / NCA robustness** (`fireRate<1`): each cell updates with probability `fireRate` per step
  (hash mask, bit-identical CPU↔WGSL); desynchronizes the CA and damps the period-3 synchronous limit
  cycle (poles near ±120° on the unit circle) into genuine fixed points.

Three implementations agree bit-for-bit and are cross-validated: the f64 reference (`rule.ts`), the
WGSL GPU kernel (`shader.ts`/`engine.ts`, headless-validated at `/devcomp/validate`), and the fixed-
point/Z80 datapath (`z80/`).

---

## 3. Results inventory (what is proven)

| # | Result | Numbers | Where |
|---|--------|---------|-------|
| E1 | XOR **gate** on 9×9, output 5 cells away | loss ~0, 4/4 cases | `params/e1_gate.json`, `morph/dev/expE.ts` |
| E2 | **Self-repair**: damage mid-compute, regrow, still XOR | heals in ~20 steps, stays healed | `params/e3_seed.json` (E3 rule is the stable universal one) |
| E3 | **Grow from a single seed** → compute → heal | genuine long-horizon attractor (600+ steps) | `params/e3_seed.json`, `morph/dev/expG.ts` |
| S6 | **1-bit full adder** (3 in → sum,carry); compute + stable + self-repair + **reactive** | compute 8/8 loss 0.0000; reactive 64/64 prior→new transitions; drift 8/8 @50/150/400 | `params/adder_{compute,stable,reactive}.json`, `morph/dev/expH.ts` |
| S6d | **Movable wire** (position-invariant routing, draggable ports) | 100% @ 11×11, **100% @ 13×13 and 17×17** (same params, never trained there) | `params/wire_invariant.json`, `morph/dev/expI.ts` |
| S6d | **Movable XOR** (position-invariant *computation*) | 100% @11, 95% @13, 68% @17; held long-horizon | `params/xor_invariant.json` |
| — | **Reactive movable XOR** (live input change at distance) | ~58% (partial; migration-at-distance unsolved) | — |
| S7 | **In-browser GPU trainer** (batch-packed WGSL reverse-mode BPTT, Adam, keep-best) | gradient matches f64 reference to **3e-5** over all 7792 params; ≈**65× CPU** throughput | `devcomp/{trainShader,gpuTrainer}.ts`, `/devcomp/traingpu` |
| — | **Async NCA updates** (fireRate) | validated CPU+GPU+demo (GRAD PASS, fireRate 0.5); damps period-3 ring | `rule.ts`, `morph/dev/spectral.ts` |
| **S9** | **Z80-substrate proof** (this session — §4) | value + gradient on a real Z80 | `devcomp/z80/` |

Live demo (`/devcomp`): grow / compute / brush-damage / heal / drag-ports / async, interactive on
WebGPU, running frozen params. Faithfulness table at `/devcomp/validate`.

Lineage (older, real-Z80 substrate primitives; `src/lib/morph/`): **Exp A** — forward-mode AD (dual
numbers) runs exactly on a real Z80 (tangent of θ² is 2θ; descent solves θ²=t). **Exp B** — that
gradient trained a single-channel developmental rule. **Exp C** — directional perception grew the
asymmetric letter "F" evolution couldn't. **Exp D** — scaled to a color emoji (🦎).

---

## 4. This session's contribution — the Z80-substrate proof (S9), in detail

**Claim closed:** the trained developmental rule executes as a real Z80 program, and that same program
hands back its exact training gradient. This is the paper's novelty anchor, proven for the *actual*
trained rule (not a toy), on the real Z80 core Zilion is conformance-tested against, run offline in
the "grid→lane" design (one lane holds the whole field + weights and sweeps cells — Zilion v0.1.2 has
no cross-lane comms, so this is an offline provenance proof, **not** the live GPU demo; do not oversell
it as running on the Zilion GPU core).

Method: build a **bit-faithful** signed fixed-point emulation of the rule first (no assembly), decide
the precision, then implement that exact datapath in Z80 assembly and validate every layer on the real
emulator. Files in [`src/lib/devcomp/z80/`](../src/lib/devcomp/z80/):

| Phase | File | What runs | Result |
|------|------|-----------|--------|
| 1 | `quantize.ts` | rule in bit-faithful fixed-point | correct XOR truth table at **Q8.8** (0 saturations); ~exact by Q8.16 |
| 1.5 | `gradfixed.ts` | forward-mode dual gradient in fixed-point | matches f64 gradient; **Q16.16 clean (1.5e-5)**, Q8.8 coarse (4e-3) |
| 2 | `z80mac.ts` | signed Q8.8 MAC (16×16→32 + 32-bit accumulate) | bit-exact vs reference on 300 random vectors + **all 48 real W1 rows** |
| 2b | `z80cell.ts` | one full cell update (perceive→MLP→tanh) | **bit-exact over 144 real cell updates**; 2e-2 vs f64 |
| 2c | `z80cell.ts` | whole gate: cell swept over grid × 24 steps | **XOR truth table computed entirely on the Z80**: 0.05 / 0.98 / 0.99 / 0.04 |
| 3 | `z80grad.ts` | dual-number cell → d(out)/dθ | matches f64 finite-diff to **Q8.8 quantum (4.3e-3)**; bit-identical to the TS dual |

Supporting: `fixed.ts` (signed Q(W.F) fixed-point + dual ops, wide MAC accumulation, tanh + derivative
LUTs). Assembler [`morph/z80asm.ts`](../src/lib/morph/z80asm.ts) gained the CB-prefixed shift/rotate
group (SRA/RR for the signed `>>1` in perceive) and a **strict JR/DJNZ range check** (an out-of-range
relative branch now fails assembly rather than silently wrapping — the bug that first broke perceive).
Backward-compatible: Exp A and the m0 CA differential suite still pass (diff=0).

Caveats to carry forward honestly: (a) offline grid→lane, not the live GPU core; (b) Q8.8 carries the
gradient only coarsely — a *precision-grade* gradient table should rebuild the datapath in Q16.16
(proven faithful in Phase 1.5); (c) proven for the E1 gate — the adder / movable rules would need the
same treatment (mechanically identical, just more weights/tape).

Reproduce: `npx tsx src/lib/devcomp/z80/{quantize,gradfixed,z80mac,z80cell,z80grad}.ts`.

---

## 5. S8 — ablations & multi-seed statistics (the rigor pass)

Harness: [`morph/dev/s8.ts`](../src/lib/morph/dev/s8.ts) (parameterized gate trainer with ablation
knobs) + [`morph/dev/s8_run.ts`](../src/lib/morph/dev/s8_run.ts) (concurrency orchestrator). Task: the
XOR gate on 9×9 trained through a distance curriculum to a displaced output (d=5), scored as "all 4
cases within 0.2 of target." Conditions: `baseline` (full perception, HD=48, ReLU), `iso` (isotropic
perception = identity+laplacian only), `id` (identity-only = **no neighbour information**), `norelu`
(linear hidden layer), and a capacity sweep HD ∈ {4,8,16,32,48,96}. 8 seeds each.

Results (8 seeds/condition; success = XOR solved at the displaced output d=5; full grid in
[`docs/s8_results.json`](s8_results.json), 942 s wall):

| condition | success | mean loss ± std | mean solved-dist /5 | reading |
|-----------|---------|-----------------|----------------------|---------|
| **baseline** (full percep, HD48, ReLU) | **8/8 (100%)** | 0.0000 ± 0.0000 | 5.00 | trains reproducibly — the headline has an error bar |
| **id** (identity-only, no neighbour info) | **0/8 (0%)** | **0.2500 ± 0.0000** | 0.00 | clean necessity: 0.25 is *exactly* the constant-output baseline → without spatial coupling it cannot beat "ignore the input" |
| **norelu** (linear hidden layer) | **2/8 (25%)** | 0.0923 ± 0.1007 | 2.38 | the hidden nonlinearity matters — success collapses, variance explodes |
| **iso** (isotropic percep: id+laplacian) | 8/8 (100%) | 0.0000 | 5.00 | directional perception *not* needed at this fixed symmetric placement (→ test on the movable task) |
| HD=4 | 5/8 (63%) | 0.0482 ± 0.0851 | 3.75 | capacity threshold: HD=4 under-capacity, unreliable |
| HD=8 | 8/8 (100%) | 0.0001 | 5.00 | ... saturates by HD=8 |
| HD=16 / 32 / 96 | 8/8 (100%) | ≤0.0002 | 5.00 | no benefit beyond HD≈8 |

**What this establishes (honestly):** (1) the gate is a *reproducible* result, not a lucky seed
(8/8); (2) a **clean necessity ablation** — spatial coupling (neighbour perception) is required, and
its absence lands exactly on the provable constant-baseline loss; (3) the hidden **nonlinearity is
important** (100%→25%); (4) a real **capacity curve** with a threshold at HD≈8. **What it does not yet
establish:** these ran on the *fixed, symmetric* gate, where directional perception is redundant
(`iso` = 100%). The sharp directional-perception ablation belongs on the *movable / position-invariant*
task; and there is still no non-developmental baseline. Both are P0 in §7.

---

## 6. Honest gap analysis — are we paper-ready? (short answer: not yet)

We have a compelling **principle** and a genuinely novel **anchor** (Z80 substrate + in-substrate
gradient). What a top venue would push on, and where we are weak:

1. **Scale / composition of computation.** The headline computation is a 1-bit adder. That proves
   "scales to arithmetic" but not "composes into non-trivial circuits." A reviewer will ask for a
   2-bit adder or a multi-gate composed circuit. **This is the biggest scientific gap.**
2. **Statistical rigor.** Most other headline numbers are still single runs, but S8 (this session) now
   gives the *gate* real rigor: 8/8 reproducibility, a clean necessity ablation (`id` lands exactly on
   the constant baseline), a nonlinearity ablation (100%→25%), and an HD capacity curve (threshold
   ≈8). Still needed: the same treatment on the **movable / position-invariant** task (where directional
   perception should bite — it's redundant on the fixed symmetric gate), per-position/size **heal
   heatmaps** for self-repair, and multi-seed stats for the **adder** and **movable XOR**.
3. **Baselines.** No non-developmental control (e.g., a single global MLP, or a hand-coded CA) to show
   the developmental route buys something. No comparison to a learned-CA-on-fixed-grid baseline.
4. **Position invariance is partial.** Movable XOR is 68% at 17×17 and reactive-at-distance caps ~58%.
   Either push these up or scope the claim carefully.
5. **Framing / related work.** Not written. The novelty is real but must be positioned precisely
   against Growing-NCA, differentiable-CA, and neural-program-synthesis literatures.
6. **The substrate proof is offline and Q8.8.** Strong as provenance; to headline it, add the Q16.16
   gradient table and (optionally) a Zilion-GPU-core run of the grid→lane sweep.

**Assessment:** the story is *"a new paradigm — developmental computation — with a unique bridge to
real hardware."* It is a strong workshop/short-paper today; for NMI it needs (1) composition and (2)
rigor most of all. None of the gaps look blocked — they are execution.

---

## 7. Prioritized roadmap for continuation

**P0 — close the science gaps that gate publication.**
1. **Composition / scale.** Train a **2-bit adder** (or half-adder → full-adder → ripple carry) as one
   developmental rule; failing that, a small composed multi-gate circuit. The GPU trainer (65×) makes
   this affordable. This is the single highest-value experiment.
2. **Rigor, done right.** Extend S8 to the **movable** task (where the ablations should bite): multi-
   seed success curves with error bars, HD sweep, isotropic-vs-directional, damage-in-training vs not,
   per-position heal heatmaps. Add a **non-developmental baseline**.

**P1 — strengthen the anchor and the artifact.**
3. **Q16.16 gradient-grade Z80 table** (rebuild `z80/` datapath at Q16.16; Phase 1.5 shows it's clean)
   → the paper's output-match + gradient table across all gate/adder cases.
4. **Hero figure** rendered from the in-substrate run: grow → compute → damage → heal → drag-ports →
   async, one rule.
5. Push **reactive movable XOR** past 58% (longer migration window, lower LR, keep-best) *or* scope the
   reactivity claim to the fixed adder (which is 64/64).

**P2 — packaging.**
6. Write the paper skeleton + related-work positioning (this disciplines which experiments are load-
   bearing). 7. Optional: Zilion-GPU-core run of the grid→lane sweep (offline, low occupancy — a figure,
   not the demo).

---

## 8. Codebase map & how to run

**Paper rule + demo + trainer** — `src/lib/devcomp/`:
- `rule.ts` — the spec, configs, forward/backward reference (f64), experiments, I/O layouts.
- `shader.ts` / `engine.ts` — WGSL kernel + WebGPU engine (demo forward).
- `trainShader.ts` / `gpuTrainer.ts` — in-browser reverse-mode BPTT trainer.
- `params/*.json` — frozen trained params (2940–7792 floats each).
- `z80/` — the substrate proof (this session): `fixed.ts`, `quantize.ts`, `gradfixed.ts`, `z80mac.ts`,
  `z80cell.ts`, `z80grad.ts`.

**Routes** (`npm run dev`, then): `/devcomp` (demo), `/devcomp/validate` (CPU↔GPU faithfulness),
`/devcomp/traingpu` (in-browser trainer + gradient validation).

**Trainers / experiments** — `src/lib/morph/dev/`: `expE` (gate), `expF` (self-repair), `expG`
(grow-from-seed), `expH` (adder), `expI` (movable wire/XOR), `s8`/`s8_run` (rigor). Real-Z80 tooling:
`morph/z80asm.ts` (assembler), `morph/dev/z80run.ts` (`runOnRealZ80`), `morph/dev/m0.ts` (CA
differential test). Zilion core: `@neovand/zilion` (external package).

**Run examples:**
```
npx tsx src/lib/morph/dev/expE.ts                 # train the gate (curriculum)
TASK=gate ITERS=800 npx tsx src/lib/morph/dev/expE.ts
npx tsx src/lib/devcomp/z80/z80cell.ts            # XOR gate on a real Z80
npx tsx src/lib/devcomp/z80/z80grad.ts            # training gradient on a real Z80
npx tsx src/lib/morph/dev/s8_run.ts               # ablation × multi-seed grid → docs/s8_results.json
```

**Validation discipline (do not regress):** the GPU gradient must match the finite-diff-checked f64
reference (`lossAndGradMarkers`) to ~3e-5; the Z80 datapath must match the bit-faithful fixed reference
exactly; the shared async fire-mask must be bit-identical CPU↔WGSL; never validate the emulator against
AI-written code — it is conformance-tested against a real reference (superzazu) and the m0 suite.

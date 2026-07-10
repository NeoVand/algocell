# Developmental Computation — Technical Report & Handoff

**Purpose.** Hand this research off to a stronger collaborator. It states the thesis (scoped to what is
actually shown), inventories every result with numbers, seeds, and file pointers, gives an honest gap
analysis toward a Nature Machine Intelligence–grade paper, and a prioritized roadmap. **Read §6 (gaps),
§7 (roadmap), and §9 (an adversarial multi-agent review of this very report) first if you are deciding
what to do next.** This report has already been through one round of adversarial review; §9 records the
verdict and the overclaims that were corrected.

Branch: `feat/morphogenesis-ca`. Code roots: `src/lib/devcomp/` (the paper's rule + demo + GPU trainer
+ Z80 proof), `src/lib/morph/` (the older Exp A–I lineage + the real-Z80 tooling). Companion docs:
[`devcomp-build-plan.md`](devcomp-build-plan.md), [`gradient-morphogenesis.md`](gradient-morphogenesis.md),
[`gpu-trainer-design.md`](gpu-trainer-design.md), [`s8_results.json`](s8_results.json).

---

## 1. Thesis (scoped)

**A single local rule — a per-cell MLP, trained by gradient descent through a developmental rollout —
can grow small, self-repairing digital circuits on a cellular automaton; and such a rule, quantized to
integer fixed-point, executes as an ordinary program in a real Z80 instruction set, with the same
fixed-point datapath carrying a matching forward-mode gradient.**

Honest scoping (what a single trained rule does *not* do, stated up front so the reader trusts the rest):
- **Different capabilities are, today, different trained rules.** The gate (E1), the self-repairing
  grow-from-seed rule (E3), the 1-bit adder, and the movable wire/XOR are *separate* parameter sets, not
  one rule that does everything. "One rule grows a whole computer" is an aspiration, not yet a result.
- **Compositional depth: now shown once (n=1).** A **2-bit ripple-carry adder** trains as one rule
  (16/16, carry internalized — `adder2.ts`, `params/adder2_2bit.json`), giving the produced-then-
  consumed internal carry the 1-bit adder lacked. But it is a **single seed**, the carry is
  **distributed** (single-cell interventions are only partial; a full carry-cell lesion gives a clean
  sum0-survives / sum1-cout-degrade dissociation), and it is not yet composed further. Multi-seed is
  the top remaining rigor item (§7, P0-2).
- **Position-invariant *computation* is only partial.** Movable *routing* (a wire) generalizes to unseen
  grid sizes (100% @ 11/13/17); movable *XOR* is ~68% at 17×17 (essentially unsolved at scale).
- **The Z80 "gradient" is a primitive, not the training gradient** — see §4. It is a single-cell,
  single-step forward-mode tangent, not ∂L/∂θ via BPTT.

What is genuinely, defensibly novel (all four §9 reviewers agree): the trained *local rule* (not a toy)
quantized to integer fixed-point (i) executes as machine code on a real Z80 ISA and (ii) the same
datapath carries a matching forward-mode tangent, cross-validated bit-for-bit across the f64 reference,
the WGSL GPU kernel, and the Z80, and against finite differences. **"A learned rule and its gradient
co-resident in one real integer instruction set" is a bridge no Growing-NCA or learned-CA paper has.**

Positioning (must be argued explicitly, not asserted — see §6.5): the nearest prior art is **not** just
Growing NCA (images). "A local CA rule that computes" is already occupied by **Self-classifying MNIST
CA** (2020), **Differentiable Logic CA** (2024), and the **Neural GPU** (2016, length-generalizing
binary arithmetic). The contribution must be pinned against these — realistically on the *ISA-execution*
and *in-substrate-gradient* axes, plus growth+repair+position-invariance combined.

---

## 2. The rule (single source of truth: [`src/lib/devcomp/rule.ts`](../src/lib/devcomp/rule.ts))

Per interior cell, per step:
```
perceive = [identity, gx=(right−left)/2, gy=(down−up)/2, laplacian]  per channel   (FEAT=4)
dl       = W2 · relu(W1 · perceive + b1) + b2
state'   = tanh(state + dl)          # residual INSIDE the tanh — a per-step nonlinearity
```
Inputs are clamped into a signal channel every step; border cells stay 0; damage zeros a cell then the
input clamp re-applies. Config-driven (`RuleConfig`) so one code path serves any grid/channel/hidden
size. Position invariance uses re-stamped marker channels (rule reads markers, not coordinates).
Movable XOR needs **separated signal channels** (each input its own channel; a shared channel blends
the bit-waves → unlearnable — representational, not capacity). Persistence (hold-window loss) makes the
answer an attractor; reactivity (mid-rollout input flip) trains re-settling; async updates (`fireRate<1`,
hash mask bit-identical CPU↔WGSL) damp the period-3 synchronous limit cycle. Three implementations agree
bit-for-bit and are cross-validated (f64 `rule.ts`; WGSL `shader.ts`/`engine.ts` at `/devcomp/validate`;
fixed-point/Z80 `z80/`).

---

## 3. Results inventory (with seed counts — n=1 means a single trained model)

| # | Result | Numbers | Seeds | Where |
|---|--------|---------|-------|-------|
| E1 | XOR **gate**, output 5 cells away | 4/4 cases; **8/8 seeds solve** (S8) | **n=8** | `params/e1_gate.json`, `morph/dev/expE.ts`, `s8.ts` |
| E2 | **Self-repair**: damage → regrow → still XOR | heals ~20 steps, stays healed | n=1 | `params/e3_seed.json` |
| E3 | **Grow from a seed** → compute → heal | attractor to 600+ steps | n=1 | `params/e3_seed.json`, `morph/dev/expG.ts` |
| S6 | **1-bit full adder** (compute+stable+repair+reactive) | compute 8/8 cases; reactive 64/64 transitions; drift 8/8 @50/150/400 | **n=1** (all one model) | `params/adder_*.json`, `morph/dev/expH.ts` |
| S6 | **2-bit ripple-carry adder** (internal carry — compositional depth) | 16/16 cases, carry internalized; carry-cell lesion → sum0 16/16, sum1/cout 10/16 (dissociation); carry distributed | **n=1** | `params/adder2_2bit.json`, `morph/dev/adder2.ts` |
| S6d | **Movable wire** (position-invariant routing) | 100% @11, 100% @13, 100% @17 (unseen sizes) | n=1; placement-count not tabulated | `params/wire_invariant.json`, `morph/dev/expI.ts` |
| S6d | **Movable XOR** (position-invariant *computation*) | 100% @11, 95% @13, **68% @17** | n=1; denominator not tabulated | `params/xor_invariant.json` |
| — | Reactive movable XOR | ~58% (near floor; unsolved) | n=1 | — |
| S7 | **In-browser GPU trainer** (WGSL reverse-mode BPTT) | gradient vs f64 ref **3e-5**; ≈**65× CPU** | — | `devcomp/{trainShader,gpuTrainer}.ts`, `/devcomp/traingpu` |
| — | **Async NCA updates** (fireRate) | validated CPU+GPU+demo | — | `rule.ts`, `morph/dev/spectral.ts` |
| **S9** | **Z80-substrate proof** (§4) | value bit-exact; gradient = single-cell JVP | — | `devcomp/z80/` |

**Caveat that governs this whole table:** only the E1 gate (row 1) has multi-seed statistics. Every
other headline is a single trained model — treat as existence proofs, not measured success rates, until
§7-P0 is done. The GPU trainer (65×) makes multi-seed re-runs affordable.

Lineage (older, real-Z80 primitives; `src/lib/morph/`): Exp A (dual-number AD runs on a real Z80: tangent
of θ² is 2θ), Exp B (that gradient trained a 1-channel rule), Exp C (directional perception grew an
asymmetric letter "F" — **note: single anecdote, no matched baseline; do not headline as "beat evolution"**),
Exp D (color emoji 🦎).

---

## 4. The Z80-substrate proof (S9) — precise claims

**Setup.** A **bit-faithful** signed fixed-point emulation of the rule (`z80/fixed.ts`) fixes the
precision; that exact datapath is then implemented in Z80 assembly and validated on
**`z80-emulator` v2.3.0** (Léon Kesteloot's TypeScript Z80, via `morph/dev/z80run.ts::runOnRealZ80`).
*This is a conformance-tested software emulator of the Z80 ISA, not physical silicon; and it is a
different emulator from `superzazu` — superzazu was the reference the WGSL/Zilion GPU core was
conformance-tested against, a separate result.* The design is "grid→lane": one lane holds the field +
weights (Zilion v0.1.2 has no cross-lane comms, so this is an **offline provenance proof**, not the live
GPU demo).

| Phase | File | What runs | Result | Precise scope |
|------|------|-----------|--------|---------------|
| 1 | `quantize.ts` | rule in bit-faithful fixed-point | XOR truth table correct at **Q8.8** | value |
| 1.5 | `gradfixed.ts` | forward-mode dual tangent d(out)/dθ | matches f64; **Q16.16 clean 1.5e-5**, Q8.8 4e-3 | *single-step cell map*, TS |
| 2 | `z80mac.ts` | signed Q8.8 MAC | bit-exact vs ref, 300 vectors + all 48 W1 rows | one dot product |
| 2b | `z80cell.ts` | one full cell update | **bit-exact, 144 real cell updates**; 2e-2 vs f64 | one cell, one step, on the emulator |
| 2c | `z80cell.ts` | whole gate: 0.05/0.98/0.99/0.04 ✓ | XOR truth table correct | **each cell update runs on the emulator; the grid sweep + 24-step rollout are orchestrated in TypeScript** (memoized) |
| 3 | `z80grad.ts` | dual-number cell → d(out)/dθ | matches finite-diff to **Q8.8 (4.3e-3)** | **single cell, single step, 6 hand-picked weights, one input case** |

**What Phase 3 is and is not.** It is a forward-mode tangent (JVP) of the *single-step cell map* for a
handful of weights, run in fixed point on the Z80, matching finite differences at Q8.8. It is **not** the
BPTT training gradient ∂L/∂θ that actually trains the rule (that is over T=24 steps × ~49 cells × 4
cases; see `s8.ts`/`expH.ts`). "Exact training gradient" and "programs+gradients on real silicon" were
**overclaims** (flagged in §9, corrected here). The honest headline: *a single cell's forward-mode
tangent runs in-substrate and matches finite differences in fixed point.* To reach the real thing:
propagate the dual tangent through the full rollout and combine with ∂L/∂output, rebuilt at Q16.16
(Phase 1.5 shows the fixed-point tangent is clean there).

**What is solid and novel:** the *value* path — the actual trained rule quantized and executed cell-by-
cell on a real Z80 ISA, bit-exact, producing the correct XOR truth table — plus the demonstrated
*mechanism* that the same integer datapath can carry a forward-mode tangent. Assembler additions:
CB-prefixed SRA/RR + a strict JR/DJNZ range check; backward-compatible (Exp A + m0 CA suite pass, diff=0).
Reproduce: `npx tsx src/lib/devcomp/z80/{quantize,gradfixed,z80mac,z80cell,z80grad}.ts`.

---

## 5. S8 — ablations & multi-seed statistics (the rigor template)

8 seeds/condition, XOR gate 9×9, distance curriculum to d=5, success = all 4 cases within 0.2 of target.
Full grid in [`s8_results.json`](s8_results.json) (942 s wall). `s8.ts` / `s8_run.ts`.

| condition | success (n=8) | mean loss ± std | reading |
|-----------|---------------|-----------------|---------|
| **baseline** (full percep, HD48, ReLU) | **8/8** | **1.0e-5 ± 8.0e-6** | reproducible — the error bar (note: *not* "0.0000"; the spread is ~1e-5) |
| **id** (identity-only, no neighbour info) | **0/8** | **0.2500 ± 6e-17** | necessity control: 0.25 is *exactly* the constant-output MSE → without spatial coupling it cannot beat "ignore the input". (Somewhat tautological: id-perception structurally disconnects the displaced output; it confirms the harness + that coupling is required, not a mechanism.) |
| **norelu** (linear hidden layer) | 2/8 | 0.092 ± 0.10 | hidden ReLU matters — but the per-step `tanh` remains, so this is "remove one of two nonlinearities", not "linear". 2/8 still solve. |
| **iso** (isotropic: id+laplacian) | 8/8 | ≤2e-4 | directional perception **redundant** on this fixed symmetric gate → this ablation belongs on the *movable* task (§7-P1) |
| HD=4 / 8 / 16 / 32 / 96 | 5/8 / 8/8 / 8/8 / 8/8 / 8/8 | — | capacity helps, but 5/8-vs-8/8 (HD4 vs HD8) is **not** statistically separated (Fisher p≈0.2); no CIs. "Threshold ≈8" is a point-estimate reading of n=8. |

**Honest summary:** S8 makes the *gate* a rigorous result (reproducibility + a necessity control + a
capacity trend) and is the template every other headline must be held to. It does **not** support the
directional-perception mechanism claim (iso=100% here) and gives no error bars beyond n=8.

---

## 6. Gap analysis — are we paper-ready? (No — strong workshop paper today)

1. **Compositional depth — first result in hand (n=1).** The **2-bit ripple-carry adder** now trains as
   one rule with a produced-then-consumed internal carry (16/16; carry-cell lesion cleanly dissociates
   carry-independent sum0 from carry-dependent sum1/cout). This directly answers the review's biggest
   lever — but it is a single seed and the carry is distributed. Remaining: multi-seed + push depth
   further (a composed 3-bit/ripple chain, or a multiplier). **§7-P0-2.**
2. **Statistical rigor beyond the gate (now the biggest gap).** Only E1 has multi-seed stats. The 1-bit
   and 2-bit adders, movable XOR/wire, E2/E3 are each n=1. **§7-P0-2.**
3. **No non-developmental baseline.** Nothing shows development buys anything vs a global MLP or a
   hand-coded CA. **§7-P0.**
4. **The Z80 anchor is overclaimed and must be made load-bearing.** Value path is solid; the "gradient"
   is a single-cell JVP, not ∂L/∂θ; "real silicon" is an emulator. Corrected in §1/§4. To headline it,
   either carry the full-rollout gradient at Q16.16 or show the ISA result is scientifically load-bearing
   (e.g., an argument/experiment that the fixed-point/ISA constraint matters), not decorative.
5. **Related-work positioning is missing and the novelty is asserted against strawmen.** Must cite and
   contrast Self-classifying MNIST CA, Differentiable Logic CA, Neural GPU, deep-thinking/algorithm-
   synthesis nets. **§7-P0.**
6. **Partial position-invariant computation** (68% @17) and **near-floor reactive movable XOR** (58%) —
   push up or scope the claim to routing.
7. **Underplayed real finding:** the position/scale-invariance *boundary* (routing generalizes to unseen
   sizes; placed multi-input computation does not; the fix is channel separation, not capacity) is a
   genuine scientific result currently buried under the Z80 narrative — promote it.

---

## 7. Prioritized roadmap

**P0 — the experiments that gate publication.**
1. **Compositional depth — DONE (n=1), extend it.** The 2-bit ripple-carry adder trains as one rule with
   an internalized carry (`adder2.ts`: expose-then-internalize curriculum → 16/16; region-lesion causal
   dissociation). Next: (a) **multi-seed** (P0-2), (b) push depth — a 3-bit ripple chain or a 2-bit
   multiplier — to show the depth generalizes, and (c) tighten the causal story (the carry is
   distributed; an interchange intervention on the carry *region* or an information-flow measure would
   make it airtight).
2. **Multi-seed everything, to the S8 bar.** Re-run `adder2.ts`, `expH.ts` (1-bit adder), and `expI.ts`
   (movable) over ≥8–16 seeds with Wilson CIs; per-position/size **heal heatmaps** for E2/E3; explicit
   placement-count denominators for movable percentages. This is now the single biggest rigor lever, and
   it is gated on the enabling engineering (P1-7: GPU-trainer N-in/M-out) to be affordable.
3. **Non-developmental baseline battery.** (A) A global MLP on the same truth tables → a table of
   {compute, self-repair, position-invariance, grid-transfer, ISA-execution} where only the developmental
   rule wins the last four. (B) Damage-in-training vs not → heal ≈0 without the damage stage (self-repair
   is *causally earned*).
4. **Related-work contrast table** (one row per prior system, columns = the capability axes) to pin the
   uniquely-ours cell (realistically: executes on a real ISA + carries an in-substrate gradient).

**P1 — strengthen the anchor & mechanism.**
5. **Movable-WIRE directional-perception ablation** (full/iso/id × relu, 8 seeds, at 11×11 + held-out
   13/17): the clean necessity cliff S8 couldn't give (predict iso/id collapse, full succeeds).
6. **Fix the gradient claim in fact, not just wording:** full-rollout tangent at Q16.16 → an in-substrate
   value+gradient table for the gate *and* adder; or restate precisely as the single-step JVP it is.
7. **Extend the GPU trainer to N-in/M-out** (widen port encoding, extend seed/score kernels; validate vs
   the finite-diff-checked CPU reference) so the 2-bit adder and a movable adder are affordable to run
   multi-seed.
8. Promote the **invariance-boundary** finding to a headline with error bars.

**P2 — packaging.** Standardize the success threshold across experiments (gate 0.2 vs adder/movable 0.3)
and report sensitivity; hero figure from the in-substrate run; paper skeleton + framing.

---

## 8. Codebase map & how to run

`src/lib/devcomp/`: `rule.ts` (spec + f64 reference), `shader.ts`/`engine.ts` (WGSL + WebGPU),
`trainShader.ts`/`gpuTrainer.ts` (in-browser BPTT trainer), `params/*.json` (frozen params),
`z80/` (substrate proof: `fixed,quantize,gradfixed,z80mac,z80cell,z80grad`).
Routes (`npm run dev`): `/devcomp` (demo), `/devcomp/validate` (CPU↔GPU faithfulness), `/devcomp/traingpu`.
`src/lib/morph/dev/`: `expE`(gate) `expF`(repair) `expG`(grow) `expH`(adder) `expI`(movable) `s8`/`s8_run`
(rigor). Real-Z80 tooling: `morph/z80asm.ts`, `morph/dev/z80run.ts` (`runOnRealZ80`, uses `z80-emulator`
v2.3.0), `morph/dev/m0.ts` (CA differential test). Zilion GPU core: `@neovand/zilion` (external).

```
npx tsx src/lib/morph/dev/expE.ts               # train the gate
npx tsx src/lib/devcomp/z80/z80cell.ts          # XOR gate value path on a real Z80
npx tsx src/lib/devcomp/z80/z80grad.ts          # single-cell forward-mode tangent on a real Z80
npx tsx src/lib/morph/dev/s8_run.ts             # ablation × multi-seed grid → docs/s8_results.json
```

Validation discipline (do not regress): GPU gradient ↔ finite-diff-checked f64 ref to ~3e-5; Z80 datapath
↔ bit-faithful fixed reference exactly; async fire-mask bit-identical CPU↔WGSL; never validate the
emulator against AI-written code (`z80-emulator` is conformance-tested; the m0 suite guards the CA).

---

## 9. External review — verdict & corrections applied

This report was reviewed by four independent adversarial critics (rigor, novelty/positioning, factual
correctness, experiment-design) plus a synthesis pass (`docs` workflow, 2026-07-10). **Consensus verdict:
strong workshop / short-paper today, not NMI-ready.** The single biggest lever is **compositional depth
(the 2-bit adder, §7-P0-1)**; reframing alone leaves the computation contribution below the baselines and
the Z80 anchor at risk of reading as a gimmick.

**Two things all four judged genuinely strong:** (1) the S8 gate ablation as NMI-grade rigor and the
template for the rest; (2) the *value*-path Z80 result — the actual trained rule executing as real-ISA
machine code — plus the demonstrated mechanism that the same integer datapath can carry a forward-mode
tangent ("a learned rule and its gradient in one real ISA" — a bridge no prior CA paper has).

**Overclaims the review caught, now corrected in this report and in the code output:**
- "the substrate carries its own **exact training gradient**" → it carries a single-cell, single-step
  forward-mode tangent for 6 weights at Q8.8 (4.3e-3), not ∂L/∂θ via BPTT. (§1, §4, and `z80grad.ts`
  output corrected.)
- "programs + gradients on **real silicon**" → a conformance-tested emulator (`z80-emulator` v2.3.0);
  each cell runs on it, the grid sweep + rollout are TypeScript-orchestrated. (§4.)
- "grows a **computer**, not an image" as the novelty → false dichotomy; Self-classifying MNIST CA,
  Differentiable Logic CA, Neural GPU already compute with local rules and must be cited. (§1, §6.5.)
- "**beat evolution** on the letter F" → single anecdote, no matched baseline; removed from headlines. (§3.)
- "one rule grows+computes+repairs+**position-invariant**" / "scales to arithmetic" → separate rules; the
  adder has no compositional depth; position-invariant *computation* is ~68%. (§1.)
- single-seed headlines presented beside the 8-seed gate → every non-gate headline now tagged **n=1**. (§3, §5.)

The corrections make the honest core stand on its own: one rigorously-measured result (the gate), one
genuinely novel bridge (rule + tangent in a real ISA), and a clear, honest map of exactly what to build
next (§7) to make it a paper.

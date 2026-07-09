# Developmental Computation on WebGPU / Zilion — build plan & status

Porting the validated "developmental computation" paradigm (grow a functional,
self-repairing computer from a seed, by gradient descent through development) to
a live in-browser GPU demo + the experiment/evidence base for a Nature Machine
Intelligence submission. Companion to [`gradient-morphogenesis.md`](gradient-morphogenesis.md)
(the method + the E1/E2/E3 results). This file is the execution roadmap.

## Architecture — two tracks

**Track 1 — native WGSL f32 field-CA.** The demo AND all experiments. The rule
is ~2940 MACs/cell/step — trivial for the GPU. Trained offline (Node,
reverse-mode AD in `src/lib/morph/dev/expE/F/G.ts`); the browser runs frozen
params. Real-time grow / compute / brush-damage / heal, interactive, at scale.
Plus (user request) an **in-browser trainer** via forward-gradient (O(1) memory,
parallel over lanes) — a paper contribution and the bridge to Track 2.

**Track 2 — the rule as real Z80 programs on Zilion.** The novelty anchor: the
learned computer is *literally a program in a real ISA*, and the same substrate
(Exp A) carries its exact training gradient. **Honest constraint:** Zilion v0.1.2
has no cross-lane communication, so per-cell halo exchange can't run on-GPU →
Track 2 is an **offline provenance proof** (grid→lane: one lane holds the whole
9×9 field + weights and sweeps cells sequentially, like `src/lib/morph/ca.ts`),
run once for a figure/table — not the live demo. Do not oversell it.

## Rule spec (single source of truth: `src/lib/devcomp/rule.ts`)

Grid 9×9, C=12 channels (ch0 = signal, 1..11 = hidden), FEAT=4, PERC=48, HD=48,
P=2940. Per interior cell per step:
- perceive = [identity, gx=(right−left)/2, gy=(down−up)/2, laplacian] per channel
- dl = W2·relu(W1·perceive + b1) + b2
- state' = **tanh(state + dl)** (residual INSIDE the tanh)
- inputs clamped (ch0) every step; border cells stay 0; damage zeros all channels
  of masked cells, then the input clamp is re-applied.

Param layout `[W1(HD×PERC), b1(HD), W2(C×HD), b2(C)]`. Frozen params committed at
`src/lib/devcomp/params/{e1_gate,e2_repair,e3_seed}.json` (2940 floats each).

## Staged sequence (status)

- **S0 — freeze + commit params + this doc.** ✅ e1/e2/e3 params in
  `src/lib/devcomp/params/`; expE gained `PARAMS_OUT`.
- **S1 — shared spec module + reference test.** ✅ `src/lib/devcomp/rule.ts`
  reproduces the trainer's E2/E3 numbers bit-for-bit (f64).
- **S2 — WGSL kernel, headless GPU validation (critical de-risk).** ✅
  `src/lib/devcomp/{shader,engine}.ts`; `/devcomp/validate` route compares GPU
  vs f64 reference: **20/20 cases, max|CPU−GPU| = 3.2e-5** (E1@24, E2/E3@50 with
  damage). Bugs found here: `params` buffer needed `COPY_SRC`; `self` is a WGSL
  reserved keyword. Keep the shader-compile-error surfacing in `engine.create()`.
- **S3 — render + demo route `/devcomp`.** ⏳ Fork `GPUEngine` render/camera
  halves (`src/lib/gpu/engine.ts`); mirror soup chrome; play/step/reset/inputs;
  live output readout.
- **S4 — damage brush + self-repair + grow-from-seed in the demo.** ⏳
- **S5 — tiling/scale spectacle** (N×N independent machines, ≥30fps). ⏳
- **S6 — new experiment: adder.** 🟡 **Compute stage DONE.** `expH.ts` — 1-bit
  FULL ADDER (11×11, C=16, HD=64, 5200 params): 3 inputs → 2 outputs
  (`sum = a⊕b⊕cin` 3-input parity, `carry = majority`), **8/8 cases correct,
  loss 0.0000** across a 6-cell transport distance (distance curriculum, 8/8 at
  every stage). Params: `src/lib/devcomp/params/adder_compute.json`. Multi-output
  backprop gradient-checked exact. *The paradigm scales to arithmetic.*
  **(b) DONE — kernel generalized + adder live in the demo.** `rule.ts` is now
  config-driven (`RuleConfig`, `makeConfig`, `EDIM`/`ADIM`); `shader.ts`/`engine.ts`
  take a cfg; the adder is a 4th tab on `/devcomp` (multi-output sum+carry readout).
  **(a) DONE — stable + self-repairing adder.** `expH.ts MODE=stabilize` (full-IC
  persist with a long 40-step hold window + damage, warm from the compute params):
  **held 8/8, +damage healed 8/8, long-horizon drift @50/@150/@400 all 8/8.**
  Params `adder_stable.json`; the adder tab now runs indefinitely (`stable: true`)
  and heals the damage brush live (verified: (1,1,0)→sum 0, carry 1 at step 994,
  correct through damage). `/devcomp/validate` passes **36/36** (incl. adder
  self-repair on GPU), max|CPU−GPU|=3.2e-5. Note: dropped the dual-IC/seed here —
  it collapsed the adder to the 0.25 constant baseline (the compute rule has no
  seed-growing to warm-start from); full-IC + a long hold window gave genuine
  long-horizon stability.
  **(a3) DONE — input-reactive adder.** `expH.ts MODE=reactive` (rollout holds the
  prior input's answer, switches input mid-run, must re-settle to the new answer;
  + a damage stage to retain self-repair). **64/64 prior→target transitions land on
  the new answer** (was 24/64), drift @50/@150/@400 still 8/8. Params
  `adder_reactive.json`; Experiment gains a `reactive` flag; the demo re-clamps
  (not re-seeds) inputs for reactive experiments, so toggling an input re-settles
  the *running* field live (verified: step count keeps climbing, output tracks).
  Inspector overhaul shipped too (Inspect/Damage toggle, click a cell to see its
  field channels, damage respects pause + step-through). `/devcomp/validate` 36/36.
  Remaining for S6: (a2) grow-from-seed adder (separate seed-IC curriculum);
  (c) 2-bit adder stretch (fallback ladder: mux → half-adder). BIG NEXT IDEA —
  positional invariance: marker channels (IN_MARK/OUT_MARK) + randomized terminal
  placement so one rule grows the right circuit wherever you drop the terminals
  (routing-to-arbitrary-output is the risky part; may need a learned output beacon).
- **S7 — in-browser forward-gradient trainer** (watch it learn) + JS reverse-mode
  Web Worker baseline. ⏳
- **S8 — ablations + multi-seed statistics** (isotropic-perception / no-hidden /
  HD-sweep / no-damage-training baselines; success over many seeds w/ error bars;
  per-position heal heatmaps). ⏳
- **S9 — Track 2 offline Z80-substrate proof** (`src/lib/devcomp/z80rule.ts` via
  `z80asm` + `runOnRealZ80` + `Zilion`; output-match table vs f32/f64; dual-number
  gradient vs finite-diff/backprop). ⏳

## Reuse map (from the codebase)

- Device + render + camera: `src/lib/gpu/engine.ts` (`init`, full-screen-quad
  viewer, `zoomAt`/`pan`/`screenToCell`, readback idiom); shaders in
  `src/lib/gpu/shaders.ts`. Fork the render half; the fragment reads `array<f32>`
  channels (replace `read_soup_byte` with a channel→color map).
- UI chrome + rAF loop + pointer painting + theme: `src/routes/soup/+page.svelte`,
  `src/routes/layout.css`.
- Track 2 primitives: `src/lib/dev/z80run.ts` (`runOnRealZ80`),
  `src/lib/morph/z80asm.ts` (`assemble`), `src/lib/morph/zilion.ts`,
  `src/lib/morph/dev/expA.ts`/`expB.ts` (dual-number Z80, field-CA rollout).

## NMI evidence map (what a defensible submission needs)

Central claim: *developmental computation* — one CA rule, learned by gradient
through development, grows a functional self-repairing digital computer from a
seed. Distinct from Neural CA (images, no computation), classic A-life (no
gradients), and learned-CA-on-fixed-grid work (no growth/repair, not a program).

Figures/experiments: (1) hero: grow→compute→heal frames; (2) E1 gate all cases;
(3) E2 heal-rate vs damage size/position **with error bars over many seeds**;
(4) E3 grow-from-seed success rate; (5) **new: adder** (paradigm scales to
arithmetic); (6) **baselines/ablations** — isotropic perception can't break
symmetry, no-hidden, HD-sweep, damage-in-training vs not, developmental vs a
non-developmental MLP; (7) Track-2 in-substrate Z80 execution + dual gradient vs
finite-diff; (8) the interactive WebGPU artifact as an impact/repro asset.

**Missing today:** the adder; tabulated ablations; multi-seed statistics/error
bars; in-substrate execution of the full MLP rule (Exp A did the primitive); the
browser artifact; related-work positioning + a non-developmental baseline.

**Biggest risks:** f32-vs-f64 drift over long rollouts (mitigated — measured 3e-5
at S2; read a thresholded/held output if needed); the 2-bit adder may not
converge monolithically (full-adder-first + fallback ladder is the insurance);
Track-2 `memBytes` occupancy for full-MLP-in-lane (accept low occupancy; offline).

## Finding (S4): long-horizon attractor stability

The demo runs the rule indefinitely, which exposed a training-horizon artifact:
**E2 (self-repair) is only *metastable*** — its "1" outputs decay to 0 past
~150 steps (`[0,1]`: 0.997@50 → 0.091@200 → −0.04@600), because persistence was
enforced only over a short window (hold 8 steps + a 5-step repair tail). **E3
(grow-from-seed) IS genuinely stable** — holds to 600+ steps in *both* the seed
and full initial conditions, and after mid-run damage it heals in ~20 steps and
**stays** healed. E3's richer dual-IC + multi-stage training produced a true
attractor. So E3 is the *universal* rule (compute + hold + self-repair, any IC),
and the live demo runs E3 for the self-repair and grow tabs (E1 gate stays
capped at tGrow). **Also note:** these rules are *developmental*, not reactive —
changing an input on a settled field does not migrate it to the new answer, so
the demo re-seeds on input change. Both point to the same next step:

**Train for long-horizon stability AND input-reactivity.** Enforce persistence
over a much longer window (100s of steps, or a stationarity penalty), and add
**input transitions** mid-rollout (change the input, require the output to
re-settle) — the same trick that gave self-repair, applied to inputs. This makes
every rule a genuinely stable, reactive circuit, strengthens the "stable
attractor" claim for the paper, and lets the demo respond to live input toggles
without re-seeding. Fold into S6 (train the adder reactive + long-stable from the
start) and backfill E1/E2.

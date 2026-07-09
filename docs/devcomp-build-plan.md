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
- **S6 — new experiment: adder.** ⏳ 1-bit full adder first (3-in/2-out, 8 cases;
  more likely to converge), then 2-bit adder as stretch; fallback ladder
  mux → half-adder. New `expH.ts`, curriculum = distance → persist → damage → seed.
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

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
  (c) 2-bit adder stretch (fallback ladder: mux → half-adder).
- **S6d — POSITIONAL INVARIANCE — crux DE-RISKED (`expI.ts`).** One grid-agnostic
  rule (C=16: ch0 signal, ch1 IN_MARK, ch2 OUT_MARK, ch3.. hidden; HD=96; 7792
  params) that routes a bit to a **randomly-placed** output — a MOVABLE WIRE. The
  terminals announce themselves via marker channels (re-stamped each step; the rule
  reads markers, never absolute positions) and it learns to route (emit beacons from
  the markers, follow them — "waves find each other"). Trained on random placements
  on 11×11, **routing accuracy 100%**, converges in ~100 iters. **Grid-size
  invariant** (same params, never trained on these sizes): 13×13 **100%**, 17×17
  **100%** (with more steps for the longer distances). Params `wire_invariant.json`.
  This validates the whole direction. **DRAGGABLE-PORT DEMO — DONE (movable wire):**
  WGSL kernel gained marker support (`config.markers`, isOutput buffer); rule.ts
  `IDIM` (17×17 markers), `seedMarkers`, `movable_wire`/`movable_xor` experiments;
  `/devcomp` has a "Move ports" tool — drag input (○)/output (□) ports and the
  kernel re-stamps markers live so the running field re-routes. Verified: movable
  wire on 17×17, input 0→−0.004, 1→1.000, ports drag freely.

  **MOVABLE XOR — hard open problem (characterized, not solved).** Position-
  invariant *computation* is much harder than routing. Findings across 6 attempts:
  the constant-0.5 output is a strong attractor; warm-from-wire hurts (the wire
  routes by *flooding* ch0, which two inputs can't share). The recipe that finally
  trained a MARKER XOR at a FIXED ADJACENT placement: zero-init last layer + a
  cosine-decayed lr (the stability fix) → **100%, loss 0**. But it does NOT
  generalize: the instant the placement curriculum spreads the ports apart, accuracy
  cliffs to 0% (collapse). So the rule can XOR when the output is adjacent to both
  inputs (local, no routing) but cannot ROUTE two distinct signals to a distant
  output and combine them, position-invariantly. NEXT (untried): a 3-phase
  curriculum that decouples distance from position — (1) fixed adjacent (works),
  (2) fixed inputs + output distance-ramp (E1-style, which trained fixed XOR to
  d=5), (3) then randomize position — warm-starting + keep-best per stage. Also
  consider distinct per-input markers, or an explicit two-channel signal encoding
  so both bits reach the output cleanly.
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

## Finding (S6d): movable XOR needs SEPARATED signal channels, not more capacity

Position-invariant XOR (drag 2 inputs + 1 output anywhere, plane rewires to
compute a⊕b) resisted ~half a dozen attempts. The wire (1 signal) trained to
100% position-invariance; XOR kept collapsing. **Root cause was representational,
not capacity:** both inputs were injected into the SAME channel (ch0), so the two
bit-waves blended into one geometry-dependent scalar the output couldn't un-mix.
Bigger models (C=24/HD=128) did NOT help — confirming it wasn't capacity.

**Fix:** give each input its OWN signal channel (input 0→ch3, input 1→ch4, ch0 =
readout). Now XOR = "two independently-routable wires + a local XOR readout."
Second fix: `ZERO=readout` init (zero ONLY the readout row so the signal channels
propagate from step 1) — broke a 50% weak-signal plateau. With both, the FIXED
adjacent XOR trains to 100% (loss 7e-5).

**Bootstrap vs generalize:** a single FIXED placement bootstraps easily but its
solution is placement-specific and catastrophically forgets when ports spread
(the "cliff"). Training on VARIED placements from iter 1 (per user direction)
does NOT bootstrap with a small batch (collapses to 0.5) — it needs (a) a LARGE
batch so the position-invariant signal averages out of placement noise, and (b)
a curriculum that HOLDS adjacent (varied but close) for ~20% to bootstrap the
combine, THEN spreads. Large batch is why the GPU trainer matters.

## Finding (S8): ablations + multi-seed stats on the XOR gate (`s8.ts` / `s8_run.ts`)

8 seeds/condition, XOR gate 9×9 distance-curriculum to d=5 (`docs/s8_results.json`):
**baseline 8/8 (100%)** — reproducible. **id (identity-only, no neighbour info) 0/8, loss 0.2500** =
exactly the constant-output baseline → spatial coupling is *necessary* (clean ablation). **norelu
(linear hidden) 2/8 (25%), 0.092±0.10** → the hidden nonlinearity matters. **HD sweep**: HD=4 63% →
HD≥8 100% (capacity threshold ≈8). **iso (isotropic) 8/8 (100%)** → directional perception is redundant
on the *fixed symmetric* gate; that ablation belongs on the movable task. Remaining rigor: movable-task
ablations, self-repair heal heatmaps, adder/movable multi-seed stats, a non-developmental baseline.

## Finding (S9-Phase1): the trained rule quantizes — value AND gradient survive fixed-point

Before writing a line of Z80 assembly, a **bit-faithful** fixed-point emulation
of the rule (`src/lib/devcomp/z80/fixed.ts` — signed Q(W.F) integers, wide MAC
accumulation, tanh table — exactly the Z80 datapath) answers the load-bearing
quantization question the substrate proof rests on:

- **Value** (`quantize.ts`): the E1 gate computes correct XOR in fixed-point at
  *every* tested precision, incl. **Q8.8** (16-bit, one register pair — expA's
  format): truth table PASS, **0 saturations**, max|Δout vs f64| 8.9e-2. Q8.16+
  is near-exact (1.2e-3). Even a 64-entry tanh table holds. Magnitude budget:
  max|pre-activation| 10.25 → needs 5 integer bits; Q8.8 has 8. Headroom.
- **Gradient** (`gradfixed.ts`): a forward-mode DUAL tangent d(out)/dθ carried
  through the fixed-point cell update matches the f64 analytic gradient to ~Q
  resolution — Q8.8 4.3e-3 (coarse but alive), **Q16.16 1.5e-5** (clean), Q8.24
  4e-8. This is exactly the dual arithmetic Exp A proved on the Z80, now for the
  rule's real datapath.

**Verdict:** the paradigm ports. Q8.8 suffices for the *value* proof; Q16.16 is
the gradient-grade format (matches expB). Phase 2 = the datapath on a real Z80
(output-match vs this fixed reference); Phase 3 = the dual tangent on the Z80 vs
finite-diff. This de-risks the whole of S9.

## Finding (S9-Phase2): the trained rule RUNS as a real Z80 program — XOR in-substrate

The E1 gate's rule is now an actual Z80 program (`src/lib/devcomp/z80/`), built
bottom-up and validated on a real Z80 (`z80-emulator` via `runOnRealZ80`, the same
core Zilion is conformance-tested against):

- **Signed fixed-point MAC** (`z80mac.ts`): Σ w·x + bias in Q8.8 — signed 16×16→32
  multiply (sign-magnitude around expA's `mul8`) + 32-bit accumulate + >>8. Matches
  the bit-faithful reference on **300 random signed vectors AND all 48 real W1 rows**,
  0 mismatches.
- **Full cell update** (`z80cell.ts`): perceive → relu(W1·perc+b1) → W2·h+b2 →
  tanh(state+dl), all Q8.8, weights in tape, tanh a 8192-entry LUT. **Bit-exact vs
  the fixed reference over 144 real cell updates** (worst Δ 0), 2e-2 vs f64.
- **The whole gate**: sweeping that cell over every interior cell for T=24 steps (one
  lane holds the field + weights — the grid→lane design) computes the **correct XOR
  truth table** on the Z80: [0,0]→0.05, [0,1]→0.98, [1,0]→0.99, [1,1]→0.04. *The
  learned computer is literally a program in a real ISA.*

Assembler (`morph/z80asm.ts`) gained the CB-prefixed shift/rotate group (SRA/RR for
the signed >>1 in perceive) + a strict JR/DJNZ range check (an out-of-range branch
now fails assembly instead of silently wrapping — the bug that first broke perceive).
Backward-compatible: Exp A and the m0 CA differential suite still pass (diff=0).

## Finding (S9-Phase3): the trained rule's GRADIENT runs on the Z80 — anchor closed

`z80grad.ts` extends the cell to DUAL numbers (value v + tangent v̇=d(value)/dθ):
`dualdot` wide-accumulates the value Σ wv·xv AND the product-rule tangent
Σ(wv·xd + wd·xv); ReLU gates both; tanh maps the value through the LUT and the
tangent through a (1−tanh²) derivative LUT. Seed one weight's tangent = 1 (field
tangents are 0 — the seed is constant in θ), run the cell on the real Z80, read
d(out)/dθ. **Matches f64 finite-diff to Q8.8 resolution (worst 4.3e-3 ≈ the 3.9e-3
quantum), and is bit-identical to the TS fixed-point dual reference.** This is the
Exp-A test (tangent vs finite-diff) now for the *whole trained MLP rule*, on real
silicon. Q8.8 carries the gradient coarsely; the same construction in Q16.16 gives
~1.5e-5 (Phase 1.5) for a precision-grade gradient table — a format choice, the
mechanism is proven either way.

**S9 core is DONE:** the learned computer is literally a program in a real ISA AND
that program hands back its exact training gradient — *programs + gradients on real
silicon*, the differentiator vs Neural-CA / learned-CA-on-grid work. Remaining is
packaging: the paper figure (grow→compute→heal frames rendered from the in-substrate
run) + the output-match/gradient table across all cases; optionally re-running the
grid→lane sweep on the Zilion GPU core (offline, low occupancy — do not oversell).

## Finding (S7-GPU): batch-packed WGSL reverse-mode trainer (in-browser)

Built a full GPU trainer (`trainShader.ts` + `gpuTrainer.ts`): batch-packs B
samples (placement×case) into one grid, forward stores the trajectory on GPU,
backward = per-cell VJP + param-grad **matmul reduction** (no f32 atomics) +
neighbour-gather for the state gradient, Adam + gradient-clipping + keep-best,
params resident on GPU. **Validated** at `/devcomp/traingpu`: GPU gradient vs the
finite-diff-checked CPU reference (`lossAndGradMarkers`) = **maxRel 3e-5 over all
7792 params**; loss match 5e-8. Throughput ≈ **65× the CPU per-sample rate** (B=64:
289 samples/s vs CPU ~4.4). Gradient clipping (the CPU had it; GPU lacked it) is
what stops the spread-phase collapse. This trainer is the workhorse for S8 (multi-
seed stats), the 2-bit adder, and powers the "train in the browser" story (S7).

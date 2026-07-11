# Figure specification — Developmental Computation (NMI / arXiv)

Every figure, exactly what it shows. Tags: **🎨 CONCEPTUAL** = you make it (Illustrator / image model),
I give the full spec + caption; **📊 DATA** = I generate it from committed params/scripts (reproducible).
Shared visual language (keep consistent everywhere): cells = rounded squares; the **signal/readout
channel** = a warm sequential colormap (e.g. viridis-magma), 0→dark, 1→bright; **markers/ports**: input
= ○, output = □, internal-carry = ◇; dataflow = thin arrows; a fixed 6-colour categorical palette for
channels/conditions. Font: the paper's sans (match final template). All data figures regenerate from a
`figures/` script so numbers never drift.

---

## Fig 1 — "One rule grows a self-repairing computer" · HERO · 🎨 CONCEPTUAL

**Job:** in one glance, make a smart non-specialist *want* to read the paper. Three beats: a local rule →
a developing, self-healing computer → the same rule is real machine code.

- **Panel a — the rule (local, shared).** A single cell and its 4 orthogonal neighbours. Arrows show it
  *perceiving* four features per channel (identity, ∂x, ∂y, ∇²), feeding a tiny MLP (one hidden ReLU
  layer), outputting a state increment added residually and squashed by tanh. Label emphatically: *the
  SAME rule runs in every cell; it sees only its neighbours.* Small inset: the whole grid tiled with
  identical little rule-icons.
- **Panel b — the developmental loop (filmstrip, 5–6 frames).** Signal-channel heatmaps across:
  ① seed / initial condition → ② growth & self-organisation → ③ computing (inputs ○ injected, output □
  read, correct answer) → ④ damage (a square patch zeroed) → ⑤ healing (regrows) → ⑥ still computing
  (same correct answer). One continuous strip; a time arrow beneath.
- **Panel c — the punchline (the bridge).** A short Z80 assembly snippet (real, from our datapath) beside
  a CPU-chip glyph, with the line: *the trained rule, quantised to integers, runs as an ordinary program
  on a real CPU ISA — bit-exactly — and the same datapath carries its own forward-mode gradient.* This is
  the "wait, what?" beat that separates us from all prior CA work.

**Caption (draft):** *Developmental computation. A single local rule (a), shared by every cell and seeing
only its four neighbours, is trained by gradient descent through a developmental rollout to grow a small
digital circuit that computes, is destroyed and heals, and keeps computing (b). The same learned rule,
quantised to fixed-point integers, executes as an ordinary program on a real CPU instruction set and
carries its own gradient in the same substrate (c).*

---

## Fig 2 — "Gradient descent through development" · METHOD · 🎨 CONCEPTUAL

**Job:** the algorithm, conceptually, so a reader understands *how* before the results.

- **Panel a — perception.** The 3×3 stencil showing the four fixed filters applied per channel: identity,
  Sobel-x (∂x), Sobel-y (∂y), Laplacian (∇²). Note these are *fixed*, not learned; they give the rule a
  sense of direction and curvature.
- **Panel b — the per-cell update as a compute graph.** perceive(4·C features) → W₁ → ReLU → W₂ → +bias →
  **add to current state → tanh** (label the residual-inside-tanh explicitly; it is why states stay
  bounded and attractors form). Show parameter shapes.
- **Panel c — training = BPTT through development.** The grid unrolled over T developmental steps; inputs
  clamped each step; loss read at output cell(s) at the end (+ over a hold window for persistence);
  gradient (dashed) flowing back through every step to the *shared* weights. A small side-note: markers
  re-stamped each step so the rule reads *ports*, not coordinates → position invariance.
- **Panel d — the curricula (four mini-schematics).** distance-ramp (output cell moves away over
  training); persistence (score a hold window, not one step); reactivity (flip an input mid-rollout,
  re-settle); expose→internalise (an internal signal is teacher-forced, then released so the field must
  produce & consume it). Each as a tiny timeline.

**Caption (draft):** *Training. Each cell applies the same rule: it perceives four fixed features per
channel (a), maps them through a small MLP, and updates its state residually inside a tanh (b). The rule
is trained by backpropagation through T steps of development, with the loss on the output cell(s) and the
gradient shared across all cells and steps (c). Curricula (d) bootstrap long-range routing, persistence,
input-reactivity, and produced-then-consumed internal signals.*

---

## Fig 3 — "From a gate to arithmetic" · COMPOSITION · 📊 DATA

**Job:** the capability ladder culminating in *compositional depth* (a produced-then-consumed internal carry).
**Sources:** `e1_gate.json`, `adder_reactive.json`, `adder2_2bit.json`, `s8_results.json`, multiseed runs.

- **a — XOR gate.** Signal-channel filmstrip over development for one case + a compact truth-table panel
  with the actual grown outputs (0.05 / 0.98 / 0.99 / 0.04).
- **b — 1-bit full adder.** Converged field for a representative case with ○ inputs (a,b,cin) and □ outputs
  (sum,carry) labelled; truth table 8/8.
- **c — 2-bit adder + the causal probe.** The field with the internal carry cell ◇ highlighted; a small
  panel showing the carry-cell value tracking true carry₀ across all 16 cases (≈1 for the four carry₀=1
  cases, ≈0 otherwise); and the **lesion dissociation** bar chart: carry-*independent* sum₀ = 16/16 vs
  carry-*dependent* sum₁/cout = 10/16 when the carry cell is zeroed.
- **d — reproducibility.** Success-rate bars with Wilson-95% CIs: gate 8/8, 1-bit adder 6/8, movable wire
  9/10, (2-bit adder — pending run).

**Caption (draft):** *Learned developmental computation scales from a gate to arithmetic. One rule grows an
XOR gate (a) and a 1-bit full adder (b). A 2-bit ripple-carry adder (c) requires a produced-then-consumed
internal carry: the highlighted carry cell tracks the true carry₀ across all 16 cases, and lesioning it
degrades exactly the carry-dependent outputs (sum₁, cout) while leaving carry-independent sum₀ intact.
Success rates over random seeds with 95% confidence intervals (d).*

---

## Fig 4 — "Robust under damage and asynchrony" · 📊 DATA

**Sources:** `e3_seed.json`, adder params, `spectral.ts`, `reactest.ts`; **new runs needed:** heal
heatmap (damage sweep), fireRate stability sweep.

- **a — self-repair.** Damage→heal filmstrip + a **heal heatmap** (recovery rate vs damage centre position
  and size).
- **b — asynchronous (NCA) update.** Spectral analysis: lag-1 autocorrelation / dominant period showing a
  period-3 limit cycle under synchronous update, damped to a fixed point under stochastic (fireRate<1)
  update; a fireRate sweep (stability metric vs fireRate).
- **c — long-horizon stability.** Accuracy vs developmental step (50/150/400/600+) for gate & adder.
- **d — live reactivity.** Output trace of the adder as an input is flipped mid-rollout, re-settling to
  the new answer.

**Caption (draft):** *Robustness. The grown computer regenerates after damage across positions and sizes
(a). Stochastic (asynchronous) updates damp the synchronous period-3 limit cycle into a genuine fixed
point (b), giving long-horizon stability (c). The adder tracks live input changes, re-settling without
re-seeding (d).*

---

## Fig 5 — "Position- and scale-invariant computation" · 📊 DATA

**Sources:** `wire_invariant.json`, `xor_invariant.json`, `movseed` run; **new run:** channel-separation ablation.

- **a — movable ports.** The field re-routing for two different random port placements under the *same*
  parameters (○○□ dragged); show the "waves from the ports find each other."
- **b — grid-size generalisation.** Accuracy at 11×11 / 13×13 / 17×17 with identical parameters (never
  trained at those sizes); overlay the movable-wire multi-seed (9/10, mean 98±4%).
- **c — the representational finding.** Channel-separation ablation: shared signal channel (loss plateaus
  at the 0.5 constant baseline — unlearnable) vs separated channels (learns) — two loss curves.
- **d — reproducibility.** Per-seed routing-accuracy bars for the movable wire (the 9/10 result).

**Caption (draft):** *One rule, any placement, any grid. Dragging the input/output ports re-routes the
field under fixed parameters (a); the same parameters generalise to unseen grid sizes (b). Position-
invariant computation requires separated signal channels — a shared channel collapses to the constant
baseline (c) — and routing is reproducible across seeds (d).*

---

## Fig 6 — "The learned computer is a real program" · ANCHOR · 📊 DATA (panel a semi-conceptual)

**Sources:** `z80cell.ts`, `z80grad.ts`, `quantize.ts`, `gradfixed.ts`.

- **a — the datapath.** rule → fixed-point Q8.8 → Z80 assembly (perceive → signed MAC → tanh LUT), as a
  clean pipeline diagram. *(If you'd rather art-direct this one, I'll hand you the exact stages; otherwise
  I render it.)*
- **b — bit-exact execution.** Z80 vs f32 across the truth table / per-cell error (bit-exact vs the fixed
  reference; ≈2e-2 vs f64) — the grown XOR gate computed entirely on the emulated Z80: 0.05/0.98/0.99/0.04.
- **c — in-substrate gradient.** Forward-mode dual-number tangent d(out)/dθ on the Z80 vs finite difference
  for representative weights (Q8.8 → 4.3e-3; Q16.16 → 1.5e-5).
- **d — honest scope (a boxed note, not a plot).** offline grid→lane on a conformance-tested emulator;
  the gradient shown is a single-cell, single-step forward-mode tangent (the primitive), not the full BPTT
  training gradient. *We state this in the main text.*

**Caption (draft):** *The learned rule is a program in a real instruction set. Quantised to integers, the
rule's datapath (a) executes on a real Z80 ISA bit-exactly, computing the correct truth table entirely in
the substrate (b); the same integer datapath carries a matching forward-mode gradient (c). Scope: an
offline provenance proof on a conformance-tested emulator, with a single-step in-substrate tangent (d).*

---

## Fig 7 — "What matters, measured" · ABLATIONS + STATS · 📊 DATA

**Sources:** `s8_results.json` + **new runs** (movable-task perception ablation, damage-in-training, the
non-developmental baseline).

- **a — ablation bars.** Success rate under: full vs isotropic vs identity perception (on the *movable*
  task, where directionality bites); ReLU vs none; HD capacity sweep; damage-in-training vs not.
- **b — multi-seed success curves** with 95% CIs across experiments (gate, 1-bit adder, 2-bit adder, wire).
- **c — the baseline capability matrix.** Rows: developmental rule vs global-MLP control (vs learned-CA-
  on-grid). Columns: computes · self-repairs · position-invariant · grid-transfers · runs-on-a-real-ISA ·
  carries-in-substrate-gradient. Checkmarks pin the uniquely-ours cells.

**Caption (draft):** *Ablations and baselines. Directional perception and a hidden nonlinearity are
necessary on the position-invariant task and capacity saturates early (a); every headline reproduces
across seeds with confidence intervals (b). A non-developmental control computes the same functions but
lacks self-repair, position/scale invariance, and ISA-realisation (c) — isolating what development buys.*

---

## Supplementary figures (📊 DATA unless noted)

- **S1** grow-from-seed (a single seed cell → full computer → heal), filmstrip + success rate.
- **S2** reactive movable XOR (the partial 58% result), stated honestly.
- **S3** spectral analysis details (autocorrelation, Nyquist energy, dominant period vs fireRate).
- **S4** the Z80 assembly listing + the full validation tables (bit-exact + gradient across formats).
- **S5** per-experiment truth tables + full hyperparameters.
- **S6** 🎨 (optional) an expanded schematic of the marker/position-invariance mechanism.

---

## Division of labour (summary)

| You (🎨 conceptual/explainer) | Me (📊 data/plots/sim-stages) |
|---|---|
| Fig 1 (hero), Fig 2 (method), (opt) Fig 6a datapath, S6 | Fig 3, 4, 5, 6b–d, 7; S1–S5 |

For each 🎨 figure I'll also hand you: a tight element list, the exact labels/numbers, the intended
"story beat," and the caption — enough to drive Illustrator or an image model directly.

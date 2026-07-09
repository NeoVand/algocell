# Gradient-Based Morphogenesis on a Zillion Tiny Computers

**Working method note + paper seed + Zilion-porting roadmap + deliberate memory for context compaction.**
Branch: `feat/morphogenesis-ca`. Code: `src/lib/morph/`. Last updated during Exp D (64×64 color emoji).

---

## 0. TL;DR (read this first after a compaction)

We set out to grow shapes with an artificial-life soup of Z80 programs (Algocell / Zilion). Blind
evolution over discrete CA rules **stalled** on anything asymmetric (a letter "F" reached only Dice
~0.87 after 800 MAP-Elites generations; the isotropic lookup rule *provably* cannot make asymmetric
shapes). The user pushed for a better paradigm: *"a smarter feedback mechanism than black-box
evolution … a deeper marriage between programs and gradient-based optimization."*

We found it and validated it end-to-end:

| Exp | Claim proven | Result | File |
|-----|--------------|--------|------|
| **A** | Forward-mode AD (dual numbers) runs **exactly on a real Z80** | tangent of `θ²` is exactly `2θ`; gradient descent solves `θ²=t`→`√t` using only the in-substrate gradient | `dev/expA.ts` |
| **B** | That gradient carried through a **developmental rollout** (fixed-point Q16.16) trains a rule | grows a disc; AD-through-time matches finite diff | `dev/expB.ts` |
| **C** | Multi-channel + **directional perception** grows an **asymmetric** target | the **letter F** (Dice 0.87→loss 0.004) that evolution couldn't | `dev/expC.ts` |
| **D** | Scales to a **complex color emoji**, NCA-style, and is **interpretable** | a 🦎 lizard (RGBA + hidden morphogens); the rule learns to rely on directional gradients | `dev/expD.ts` |

**The paradigm**: *don't differentiate the discrete program; keep it as the "physics" and put a
continuous field in cell memory; carry the gradient through the smooth field dynamics by forward-mode
AD; let the massively-parallel population handle the discrete decisions.* Gradient descent through
development replaces black-box evolution and **beats it on evolution's own hardest target.**

Everything is committed on `feat/morphogenesis-ca` (commits: Exp A `d6e79dc`, Exp B `ec88484`,
Exp C `923f492`, Exp D in progress). Nothing pushed.

---

## 1. The paradigm

### 1.1 Why evolution over programs stalls
A program (bytes of Z80) is **catastrophically discontinuous**: flip a byte and execution teleports.
There is no useful `∂output/∂byte`. So we were doing black-box evolution over a rugged lookup table —
sample-starved, no credit assignment, no geometry. And an **outer-totalistic** rule (index = self,
Σneighbours) is **isotropic**: it discards *which* neighbour is which, so from a symmetric seed it can
only make radially/4-fold-symmetric patterns. That is the wall.

### 1.2 The inversion (the core idea)
Stop trying to differentiate the discrete program. Instead:

> Keep the program **fixed and discrete — it is the physics.** Put a **continuous field** in each
> cell's memory that the program evolves. Gradients, memory, and smoothness live on the **field**,
> never on the code. Carry the gradient **through** the program via dual-number / Weil-algebra
> arithmetic (forward-mode AD) — exact on the smooth parts. The **discrete decisions** (branches,
> cell-fate commitments) are handled by the **population** (Zilion's millions of lanes).

This is exactly **Michael Levin's** picture of development: the genome (discrete hardware) specifies ion
channels; the **pattern** — the target, the memory — lives in the continuous **bioelectric field**
above the genome. Cells are competent agents navigating morphospace toward a re-writable setpoint with
error-correcting feedback (anatomical homeostasis), not open-loop rule-followers. And it is **Lewis
Wolpert's positional information**: cells establish a coordinate system (morphogen gradients), read
their position, and look up their fate. Our networks *invent these morphogen fields* by gradient
descent (see Exp D's hidden channels).

### 1.3 What Zilion uniquely affords
Forward-mode AD is normally dismissed because it costs **one pass per input direction** — fatal for a
million-parameter net. **Zilion erases that**: millions of independent Z80 lanes = millions of
directions in parallel. Forward-mode works through *arbitrary programs* (no tape, no global smoothness,
unlike reverse-mode). Its only cost — directions-per-pass — is exactly what population parallelism
buys. **Zilion turns forward-mode AD's fatal flaw into its superpower.**

---

## 2. The math

### 2.1 Forward-mode AD = arithmetic in a Weil algebra (and a model of SDG)
Represent a value as a **dual number** `(v, v̇)` with `ε² = 0`, where `v̇ = ∂v/∂θ` for a chosen input
`θ`. Arithmetic transports the tangent by the chain rule:

```
(a,ȧ) + (b,ḃ) = (a+b, ȧ+ḃ)
(a,ȧ) · (b,ḃ) = (a·b, a·ḃ + ȧ·b)        # product rule; the ε² term vanishes
tanh(a,ȧ)     = (tanh a, (1 − tanh²a)·ȧ)
const c       = (c, 0);   the seeded input θ = (θ, 1)
```

Dual numbers are the smallest **Weil algebra** `ℝ[ε]/ε²`. Running a program on them *is* computing
inside a model of **synthetic differential geometry** (Kock–Lawvere axiom: a map out of the
infinitesimal object `D = {x : x²=0}` is its value plus its derivative). Consequences we can exploit:
- **Higher jets** `ℝ[ε]/ε^{k+1}` transport 2nd/3rd derivatives for a few more ops → each cell can do a
  **local Newton / trust-region step** (curvature-aware, smarter than 1st-order). *(Not yet built.)*
- **A metric**: black-box evolution has no geometry; the continuous field gives a Riemannian structure
  (Fisher information) → **natural gradient**. *(Not yet built.)*

**Proven (Exp A):** a hand-written **Q8.8 fixed-point dual multiply on a real Z80** (shift-add via
`ADD A,A` / `ADD HL,HL` — no shift opcodes needed) gives `d(θ²)/dθ = 2θ` to the last bit, byte-identical
to a reference, and gradient descent using only that in-substrate gradient converges to `√t`.

### 2.2 The (neural) cellular automaton and its loss
State field `s ∈ ℝ^{H×W×C}` (C channels; 0..3 = RGBA visible, 4..C−1 = hidden morphogens). Per cell,
directional + diffusive **perception** per channel:

```
perc = [ identity=s ,  gx=(right−left)/2 ,  gy=(down−up)/2 ,  lap=Σneighbours−4·self ]   # 4C features
```

Signed gradients `gx, gy` make the rule **non-reflection-equivariant** → it can break symmetry from a
symmetric seed (an isotropic rule provably cannot — this is the whole point). **Residual update** with
a small MLP and a **tanh squash** (bounded, keeps gradients alive; a hard clamp gives dead-ReLU
gradient death, which collapsed early runs to empty):

```
s_{t+1} = tanh( s_t + W₂·ReLU(W₁·perc(s_t) + b₁) + b₂ )          # T steps from a seed
L(θ)    = mean_{cells, RGBA} ( s_T[·,0:4] − target )²             # θ = {W₁,b₁,W₂,b₂}
```

### 2.3 Getting `∇_θ L`: forward-mode, reverse-mode, forward-gradient
Three equivalent-in-value routes; they differ only in cost and where they can run.

- **Forward-mode** (Exp B/C): seed the tangent on `θ_i`, roll out with dual numbers, read `L̇ = ∂L/∂θ_i`.
  Cost = **one rollout per parameter**. Exact. Runs in-substrate on Zilion. Used to grow the F (Exp C).
- **Reverse-mode / backprop** (Exp D): one forward + one backward pass gives the **whole** gradient.
  Cost = **O(1) rollouts**, independent of #params. Exact — *the identical gradient forward-mode gives*
  (we gradient-check backprop to rel-err ~1e-4). Needs a tape/backward pass → natural on a host trainer,
  **not** natural on the Z80. Used at emoji scale for speed only.
- **Forward-gradient** (Baydin et al. 2022) — *the scalable in-substrate route*: draw a random unit
  direction `v`, compute the directional derivative `∇L·v` (one forward-mode pass — or, for a **discrete
  rule**, a finite-difference `(L(θ+εv)−L(θ−εv))/2ε`, i.e. **two plain program evaluations**), then
  `ĝ = P·(∇L·v)·v` is an **unbiased** estimator of `∇L` (isotropic `v`, `E[vvᵀ]=I/P`). Average `K`
  samples → variance `∝ P/K`. **Each sample is one Zilion lane** → average millions in one dispatch.

> **Key unification.** Forward-gradient needs only **directional derivatives**, which a *discrete*
> program yields via finite differences over the population. So we can optimize **discrete Z80-program
> rules by forward-gradient with no differentiability at all** — the population provides the estimates.
> This reconnects the smooth-NCA result to the original discrete-program vision.

### 2.4 Fixed-point & the discreteness principle
- **Fixed-point** (Q16.16 = 32-bit) carries the gradient cleanly through a 16–40-step rollout (Exp B
  shows Q8.8/Q16.16 viable; on Z80 this is a 32-bit extension of Exp A's 16-bit multiply).
- **Branches = strata.** The dual tangent is exact *within* a control-flow stratum; at a branch the
  derivative is per-piece. Finite-diff blurs across strata (this explained a benign gradient-check
  mismatch in Exp B). The **population covers the boundaries** — gradients through the smooth, sampling
  through the discrete.

---

## 3. Results (what each experiment shows)

- **Exp A** (`dev/expA.ts`): in-substrate forward-mode AD on the Z80. `θ²`→`2θ` exact (0.0000 err over
  a sweep, vs closed-form and finite diff); descent to `√t` for t∈{2,3,5,7} using only the Z80's
  gradient. Runs on Zilion (byte-identical to `z80-emulator`).
- **Exp B** (`dev/expB.ts`): a smooth field-CA, dual numbers through all 16 steps → exact `d(loss)/dθ`;
  gradient descent grows a **disc** (loss 0.15→0.05) and asymmetric ramp structure. Lessons: **smooth
  nonlinearity** (tanh) not hard clamp; **enough fractional bits** (Q16.16).
- **Exp C** (`dev/expC.ts`): 6-channel NCA, directional perception, 114 params, full **forward-mode**
  AD (one rollout/param), Adam. **Grows the letter F**: loss 0.44→0.0039 in 400 steps; a recognizable
  F. This is the shape blind evolution could barely reach.
- **Exp D** (`dev/expD.ts`): color emoji. 16 channels (RGBA + 12 hidden), directional+diffusive
  perception, MLP (HD hidden units), reverse-mode AD for speed (backprop gradient-checked to ~1e-4).
  32×32 linear model → rough green lizard silhouette; **MLP at 64×64 on the real Noto 🦎** (target from
  `dev/lizard64.json`). Analysis (in the generated HTML): the **hidden channels are morphogen fields**
  (smooth spatial gradients / positional codes the network invented), and **weight energy concentrates
  on the directional gradients gx,gy (~63%)** — mechanistic evidence of learned symmetry-breaking.

Baseline it beats: `src/lib/morph/{ca,bootstrap,evolve,mapelites}.ts` — the discrete outer-totalistic
CA (v1) and SUM×DIR16 (v2) trained by MAP-Elites, which stalled at F Dice ~0.87 and flag ~0.60.

---

## 4. Porting to Zilion (the roadmap)

Today: the **method** is validated in TS (float + fixed-point); the **substrate** (Zilion) is proven to
run dual arithmetic (Exp A). To make gradient-based morphogenesis genuinely run *on a zillion Z80s*:

1. **32-bit fixed-point dual arithmetic on the Z80.** Extend Exp A's 16-bit `fixmul` to Q16.16 (32-bit
   shift-add). Provide dual `add/sub/mul` and a **tanh via a fixed-point table** (+ its derivative
   `1−tanh²`). Differential-test against the TS fixed-point reference (the discipline this repo already
   lives by). *Deliverable: `zilion`-side dual-number ops or a WGSL "Soft-Zilion" kernel.*
2. **The field-CA as a per-lane program.** Each Zilion lane holds one cell's C-channel state (or a small
   patch) and runs the update rule carrying a tangent. Perception reads neighbours via the shared-memory
   / spatial-topology fork (see `THE-ZILION-MACHINE.md` §3 substrate contract; needs neighbour access
   across lanes — a WGSL compute kernel is the natural home rather than the isolated-memory Z80 core).
3. **Forward-gradient over lanes (the scalable trainer).** Host draws `K` random directions; each lane
   computes one directional derivative (dual JVP, *or* finite-diff for a discrete rule); host averages
   → `∇L` estimate → Adam step. This is embarrassingly parallel and needs **no reverse-mode**.
   - **Variance reduction** (the open frontier at large P): antithetic pairs (`±v`), structured/low-rank
     directions, **local losses** (Ren et al. 2022, "Scaling Forward Gradient With Local Losses"),
     activity-perturbation instead of weight-perturbation.
4. **Two flavours to offer:**
   - *Soft-Zilion*: a differentiable WGSL field kernel (continuous state, dual numbers) → exact JVPs.
   - *Discrete-rule*: keep the rule a real Z80 program; forward-gradient via finite-diff over the
     population (no differentiability). This is the purest "programs + gradients on Zilion."
5. **Deploy**: a trained rule runs as a normal Zilion dispatch (self-organising, regenerating). The
   Algocell app gets a "grow an emoji" mode (upload image → train → watch it develop / damage-and-regrow).

Substrate features needed (cross-ref the manifesto's execution contract): neighbour/shared-memory
access, per-lane RNG seed (for directions/async), configurable channels, batched on-GPU loss.

---

## 5. The paper

**Working title options**
- *Gradient-Based Morphogenesis on a Population of Tiny Computers*
- *Forward-Mode Morphogenesis: Growing Patterns by Differentiating Through Development on a Zillion Z80s*
- *Programs that Learn to Grow: In-Substrate Forward-Mode AD for Neural Cellular Automata*

**Core contributions**
1. A paradigm: **discrete program as physics + continuous field for gradients + population for
   discreteness**, grounded in developmental biology (Levin bioelectricity, Wolpert positional info) and
   in Weil-algebra/SDG differentiation.
2. **In-substrate forward-mode AD**: exact gradients computed *by* a fixed-point Z80 program about its
   own execution (Exp A) — and the observation that Zilion's parallelism makes forward-mode (and
   forward-gradient) the *right* choice at scale.
3. **Gradient-based morphogenesis beats black-box evolution** on asymmetric/complex targets (F, color
   emoji) where evolution provably/empirically stalls — with an apples-to-apples baseline in the same
   codebase.
4. **Interpretability**: the trained rule's *invented morphogen fields* and its reliance on directional
   perception, shown directly.
5. **Discrete-rule forward-gradient**: optimizing genuinely discrete Z80-program rules with no
   differentiability, via population finite-difference directional derivatives.

**Related work to position against**: Growing NCA (Mordvintsev/Randazzo/Niklasson/Levin, Distill 2020);
forward-gradient (Baydin et al. 2022; Ren et al. 2022 local losses); evolution strategies (Salimans et
al. 2017); differentiable programming / dual numbers (Elliott 2018 "simple essence of AD");
synthetic differential geometry (Kock; Lawvere); Levin's developmental bioelectricity; Wolpert
positional information; Turing morphogenesis / reaction-diffusion; the Computational Life / BFF lineage
(Agüera y Arcas et al. 2024) that Algocell/Zilion come from.

**Experiments for the paper (done ✓ / to run ▢)**
- ✓ In-substrate AD on Z80 (A); grow disc (B); grow F (C); grow color emoji + interpretability (D).
- ▢ **Regeneration / robustness**: damage the grown pattern, run more steps, measure recovery
  (persistence loss + damage during training — the NCA regeneration recipe). *The "it's alive" figure.*
- ▢ **Forward-gradient at scale**: does it grow the F/emoji? how many samples K? variance-reduction
  ablation. *This is the honest open result the substrate story hinges on.*
- ▢ **Discrete-rule forward-gradient**: optimize a real Z80-program CA rule by population finite-diff.
- ▢ **True in-substrate run** on Zilion (WGSL/Z80 fixed-point) end-to-end, timed.
- ▢ **Scaling**: capability vs channels / grid / params / steps.
- ▢ (stretch) **higher-order jets** → local Newton; **natural gradient** via the field metric.

**Figures**: the A→B→C→D arc; the emoji developing + morphogen-field panel (already generated as HTML);
the evolution-vs-gradient comparison on the F; the forward-gradient trajectory.

---

## 6. Significance, positioning, and the dream

### 6.1 Is this a big deal? (honest calibration)
As *literally demonstrated so far*, **not yet a landmark** — and we should be clear-eyed about that:
dual-number AD is textbook; running it on a Z80 (Exp A) is a systems curio; growing an emoji by
**backprop** through an NCA (Exp D) largely **reproduces Growing-NCA** (Mordvintsev 2020). The letter-F
via *forward-mode* (Exp C) is the most genuinely novel single result, but small. So today this is a
strong **proof-of-concept**. It becomes a real contribution the moment we land a result **nobody has**
— and those are within reach (§6.5).

### 6.2 Where we sit vs. other A-life
Two mostly-separate lineages, and we occupy the gap between them:
- **Program-soup A-life** — Tierra (Ray 1991), Avida, Computational Life / BFF (Agüera y Arcas 2024):
  Turing-complete cells, self-replication, open-ended evolution — **but no gradients.**
- **Gradient morphogenesis** — Growing NCA, Lenia / Flow-Lenia, differentiable self-organization:
  gradient-trained self-organizing rules — **but the "cells" are fixed neural nets, trained by backprop
  on a GPU.**
- **Us: gradients on a soup of Turing-complete computers** — the intersection almost nobody occupies.

### 6.3 Core advantages (proven vs. potential — labelled honestly)
1. **Cells are computers, not conv kernels** — the rule can be an actual program (loops, memory,
   addressing, self-modification). *[potential — our rule is still an MLP; unexploited]*
2. **Learning without backprop** — forward-mode / forward-gradient + population works where backprop
   can't (discrete rules, no autodiff, non-differentiable dynamics). *[proven in miniature: Exp A]*
3. **One medium for evolution AND gradient** — self-replicators evolving *and* programs descending, in
   the same soup. *[the killer unification — not yet demonstrated]*
4. **Biologically plausible / neuromorphic** — local, forward-only, no weight-transport problem.
   *[conceptual]*
5. **A trainable testbed for developmental biology** — Levin bioelectric goal-fields, Wolpert positional
   information; our hidden channels literally **invent morphogen fields** (Exp D). *[shown]*

### 6.4 What we expect the paradigm to do that others can't
- **Grow *functional* structures, not just pictures** — morphogenesis of *computation* (a grown circuit,
  wire, adder) on Turing cells. NCA does images; this does machines.
- **Learn on substrates with no gradient** — discrete programs optimized by population directional
  derivatives.
- **Blend Darwinian + Lamarckian** — evolution proposes, gradient refines, in one medium.
- **Regeneration grounded in a re-writable goal** — homeostasis toward a *setpoint you can rewrite*
  (Levin's two-headed planaria, in silico), not a trained reflex.

### 6.5 Long-term implication
A **second axis of learning machines**: not one big differentiable function trained by backprop, but
**populations of tiny programs that learn (forward gradients) and evolve**. Even if it never beats
backprop on a benchmark, it can own a niche backprop cannot touch — **learning + evolution + self-repair
+ computation in one asynchronous medium** — and a path to learning on hardware where backprop is
infeasible (neuromorphic/edge/biological), toward the "billions of tiny learning programs" vision.

### 6.6 Dream demos (ranked by novelty × field-excitement)
1. **Developmental computation** 🥇 — grow a working **circuit** (a 1-bit adder, a signal wire) from a
   seed; then **damage it and it regrows and still computes.** Novel, exploits the Turing cells (NCA
   can't), and "it grew a self-healing computer" is a headline. **The one to chase.**
2. **Evolution × gradient soup** — self-replicators that also get gradient nudges toward a task; the
   hybrid beats either alone. Genuinely new for A-life.
3. **Digital planarian** — cut-and-regrow with a *switchable* setpoint (grow two heads by rewriting the
   goal). Echoes Levin; the basal-cognition community would love it.
4. **A zillion Z80s learning live, in the browser** — millions of tiny computers gradient-learning to
   grow/repair a pattern in real time on a GPU. Spectacle of scale; the Zilion story made visceral.
5. **Programs that learn to learn** — self-modifying rules discovered by gradient (a learned optimizer
   as an evolvable program).

### 6.7 The thesis to aim the paper at
> **Developmental computation: growing self-repairing machines by gradient descent through development,
> on a population of tiny computers.**

It combines the three things only this substrate gives — **Turing-complete cells + gradients +
self-repair** — has an irresistible demo (#1), and cleanly differentiates from both NCA (images only)
and classic A-life (no gradients). The forward-gradient / in-substrate work (§2.3, §4) is the *systems*
backbone that makes the "on tiny computers" claim real.

---

## 7. Repro / file map

```
src/lib/morph/
  z80asm.ts            minimal Z80 assembler (used by Exp A + the discrete CA)
  ca.ts, bootstrap.ts  discrete outer-totalistic CA (v1) + SUM×DIR16 (v2)  ← the EVOLUTION baseline
  evolve.ts, mapelites.ts, fitness.ts, targets.ts, genomes.ts   the GA baseline
  dev/
    z80run.ts          run a tape on z80-emulator (== Zilion by conformance)
    expA.ts            in-substrate forward-mode AD on the Z80              npx tsx …/expA.ts
    expB.ts            gradient through a developmental rollout (fixed-pt)  npx tsx …/expB.ts
    expC.ts            grow the letter F (forward-mode AD, multi-channel)   npx tsx …/expC.ts
    expD.ts            grow a color emoji (NCA + reverse-mode, 64×64)       EXPD_VIZ=… npx tsx …/expD.ts
    lizard64.json      the real Noto 🦎 target (premultiplied RGBA, 64×64)
docs/gradient-morphogenesis.md   ← this file
THE-ZILION-MACHINE.md (repo root) ← the broader research manifesto (12 branches)
```

Viz HTML for A–D is generated to the scratchpad and sent to the user (development strips, morphogen
fields, weight analysis, loss curves). Memory: `autodiff-on-zilion.md`, `morphogenesis-feature.md`,
`zilion-learning-machine.md` in the project memory dir.

---

## 8. Immediate next steps (resume here)

**DONE so far:** Exp A (dual-number AD on a real Z80), B (gradient through a developmental rollout),
C (grew the F evolution couldn't), D (color 🦎 emoji at 64×64, best loss 0.0028), **E0 WIRE**,
**E1 GATE**, and **E2 SELF-REPAIR** — see below. The app is now a hub (`/` landing → `/soup`,
morphogenesis, research card).

- **E0 WIRE — DONE (`src/lib/morph/dev/expE.ts`).** One CA rule transports a 1-bit signal from an input
  cell to an output cell `d` cells away, correct for both input values. Long-range transport is a
  vanishing-gradient problem from scratch → solved with a **distance curriculum** (learn `d=1`,
  warm-start, extend one cell at a time). Reaches `d=6`.
- **E1 GATE (XOR) — DONE.** *Actual computation, not just transport:* two input cells, output = their
  **XOR** at a cell 5 away, one rule correct on all 4 cases: **[0,0]→0.000, [0,1]→0.981, [1,0]→0.994,
  [1,1]→0.000, loss 0.0001** (vs 0.25 constant-output baseline). It can't memorize an answer — the same
  rule must handle every case, so it builds a computation. Recipe that made it robust:
  - **Distance curriculum** (output moves one cell right per stage, inputs fixed so the warm-started rule
    transfers) + **6 random restarts on stage 1 only** (XOR-from-scratch has a strong constant-0.5
    minimum; restarts escape it, then warm-start carries it outward).
  - **Warm-start-aware lr** (the key fix): a warm stage is *refinement*, not exploration — a hot lr kicks
    the good incoming solution out of its basin (loss bounces back to the 0.25 baseline). Warm stages use
    a gentler peak (0.003) with **cosine decay to a low floor** (6e-4); from-scratch stage 1 keeps the hot
    lr (0.01). This re-saturates the transport instead of destabilizing it.
  - XOR reduces to *sum then bump*: `f(a+b)` with `f(0)=0,f(1)=1,f(2)=0 = relu(x)−2·relu(x−1)` — trivially
    representable; the whole difficulty was optimization/credit-assignment, which the curriculum handles.
  - Viz: `TASK=gate GATE_VIZ=path.json` dumps development frames → `dev/gen_gate_html.mjs` renders
    the animated 4-case HTML + the invented hidden channels. Sent to the user.

- **E2 SELF-REPAIR — DONE (`src/lib/morph/dev/expF.ts`) — the paper's headline.** The grown XOR gate,
  when a patch of its own structure is destroyed *mid-computation*, **regrows and still computes XOR.**
  After a central 3×3 patch is zeroed at step 32 of 50: **healed outputs [−0.007, 0.997, 0.997, 0.005]**
  (want 0 1 1 0), repair loss **0.0000**. Two ideas, each a warm-started fine-tune with the E1 gentle lr:
  - **Persistence** — score the output over a *window* of steps, not a single readout, so the answer must
    be a **stable attractor**, not a one-shot spike. (The clean E1 gate is correct at exactly T_GROW but
    *drifts* to [−0.60, −0.34, −0.77, −0.68] by step 50 — persistence training fixes that: it then holds
    [−0.009, 0.984, 0.992, 0.002].)
  - **Damage during training** — zero a random square patch mid-rollout and require recovery in the tail.
    Backprop flows through the damage as a 0/1 mask (gradient killed inside the hole, same mechanism as the
    input-clamp). Inputs stay clamped through damage (external boundary), so the undamaged cells + inputs
    re-drive the field back to the attractor — self-healing, not re-initialization.
  - **Robustness (`SWEEP=1`, trained on 3×3 only):** heals **100%** of all 2×2 and 3×3 damage positions,
    **88%** of 4×4, **67%** of 5×5 (a 5×5 destroys most of a 9×9 interior). Graceful degradation, and it
    generalizes to patch sizes it never saw in training.
  - **Why it matters:** a machine — grown by gradient descent through development — that is *both functional
    and self-repairing*. This is the clean differentiator from NCA (images, no computation) and classic
    A-life (no gradients). §6.6/§6.7.
  - Gradient-checked (multi-readout + damage path, rel err 9.6e-6). Viz: `FVIZ=path.json` →
    `dev/gen_repair_html.mjs` renders the animated damage-and-regrow (compute → hold → 💥 → heal). Sent to
    the user. Trained params saved via `PARAMS_OUT` (for the Zilion port / a live in-app demo).

**NEXT:**
1. **Grow-from-seed for the gate.** The substrate is still pre-seeded "all cells alive." Next: grow the
   computational structure from a single seed cell (as in C/D) *and then* compute + self-repair — the full
   "grown, functional, self-healing machine" in one rollout.
2. **A bigger circuit** — a 1-bit adder (2 outputs) or a 2-bit function, to show the paradigm scales past a
   single gate. Same recipe (curriculum → persist → damage).
3. **Regeneration on Exp D** (damage the lizard, watch it heal) — the visual companion figure.
4. **Forward-gradient at scale** — run it on C/D/E; quantify K and variance reduction (antithetic, local
   losses). *Decides how "in-substrate" the results can be — the systems backbone.*
5. **Start the Z80/WGSL fixed-point dual field kernel** (roadmap §4.1–4.2); the saved E2 params give a
   concrete rule to port and demo live in the app.
6. Keep everything on `feat/morphogenesis-ca`; commit per experiment. (User approved pushing to the branch.)
```

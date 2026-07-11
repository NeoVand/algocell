# Developmental Computation — Nature Machine Intelligence Production Plan

**The master plan.** Everything the paper needs, organized: thesis, honest status, ablations,
every figure, section-by-section outline, related work + citations, supplementary, the interactive
website, a prioritized roadmap, and the standard we hold ourselves to. Companion to
[`handoff-report.md`](handoff-report.md) (results inventory) and [`devcomp-build-plan.md`](devcomp-build-plan.md)
(execution log). Read §9 (mindset) if you read nothing else.

---

## 0. The pitch (one line, then one paragraph)

**One local rule, learned by gradient descent through development, grows a functional,
self-repairing, position-invariant digital circuit — and that circuit is literally a program in a
real instruction set whose gradient the same substrate carries.**

Biology grows robust computers (a brain, a regenerating flatworm) from a single cell, with no
global blueprint and no external controller, and repairs them when damaged. Can we *learn* a local
rule that does this for *computation* — not to grow a shape, but to grow a machine that computes?
We show yes: a single per-cell neural rule, trained by backpropagation through a developmental
rollout, grows small digital circuits (a gate, a 1- and 2-bit adder) that self-repair, run
asynchronously, and place their I/O anywhere; and we close the loop to real hardware by showing the
trained rule runs as an ordinary program on a real CPU ISA, gradient and all.

---

## 1. Status dashboard (honest traffic lights)

| Pillar | State | Note |
|---|---|---|
| Thesis + novelty | 🟢 strong | unique anchor (programs+gradients in a real ISA); positioning must be argued, not asserted |
| Core results (gate→adder→2-bit) | 🟢 done | 2-bit adder = compositional depth (internal carry + causal probe), n=1 |
| Robustness (repair, async, reactive, long-horizon) | 🟢 done | mostly n=1; heal heatmaps missing |
| Position/scale invariance | 🟡 partial | wire 9/10 multi-seed; XOR 68%@17 n=1 (crux) |
| Substrate bridge (Z80) | 🟢 value path / 🟡 gradient | value bit-exact; "gradient" is single-step JVP not full BPTT (fix or scope) |
| **Ablations** | 🟡 partial | gate solid; movable-task + baseline + damage-causality missing |
| **Multi-seed statistics** | 🟡 started | gate n=8, 1-bit adder 6/8, wire 9/10; adder-2bit / XOR pending |
| **Baselines** | 🔴 none | the biggest scientific hole |
| **Figures** | 🔴 none designed | §3 is the plan |
| **Writing / related work** | 🔴 none | §4, §5 |
| **Interactive website** | 🟡 demo exists | `/devcomp` is the seed; §7 |

**Verdict:** we have the scientific *spine* — a novel thesis and a ladder of real results. We are
early on *rigor completion, baselines, figures, and writing*. This is a real project (weeks of
focused work), but it is well-scoped and de-risked: nothing on the critical path is unknown-hard.

---

## 2. Ablations & statistics — the full plan

**Done:** gate perception-necessity (id→0/8), nonlinearity (norelu→2/8), capacity (HD 4→96); +
multi-seed: gate 8-seed, 1-bit adder 6/8, movable wire 9/10 (Wilson CIs). All GPU-affordable now.

**Must add (each is one focused run):**
1. **Directional-perception ablation on the MOVABLE task** — full vs isotropic vs identity perception,
   ≥8 seeds, on the movable wire/XOR (where directionality *should* be necessary; on the symmetric
   gate it is redundant). Predicted clean cliff. *This is the mechanism ablation the gate couldn't give.*
2. **Non-developmental baseline (the #1 hole)** — (a) a global MLP trained on the same truth tables →
   a capability matrix {compute · self-repair · position-invariance · grid-transfer · runs-on-ISA}
   where only the developmental rule wins the last four; (b) optionally a learned-CA-on-fixed-grid.
3. **Damage-in-training causal ablation** — train the identical rule/curriculum WITHOUT the damage
   stage → heal-rate ≈0 vs ≈100% with it (self-repair is *earned*, not incidental).
4. **Channel-separation ablation** — shared vs separated signal channels for movable XOR → learnable
   vs collapses-to-0.5 (the representational finding, tabulated).
5. **Sync vs async ablation** — fireRate=1 vs <1 → stability metric (period-3 blinker vs fixed point),
   with the spectral analysis as the mechanism.
6. **Complete multi-seed** — 2-bit adder (≥6/8), movable XOR (full recipe), self-repair *heal heatmaps*
   (recovery rate vs damage position & size), grow-from-seed success rate. Hold everything to the gate's
   8-seed + Wilson-CI bar.

**Statistics standard:** ≥8 seeds (16 for load-bearing), Wilson 95% CIs on success rates, mean±std on
continuous metrics, report the success threshold and its sensitivity (standardize 0.2 vs 0.3 across tasks).

---

## 3. Figures — the complete plan

Design principle: **Figs 1–2 are conceptual (make a biologist *and* an ML reader care in 30s);
Figs 3–7 are data.** Each panel earns its place; every number has an error bar.

| Fig | Title | Panels (a→d) | Type | Status |
|---|---|---|---|---|
| **1** | *One rule grows a self-repairing computer* (HERO) | a: cell + neighborhood + per-cell MLP schematic; b: filmstrip seed→grow→compute→damage→heal; c: the punchline — same rule runs as real Z80 code | concept | 🔴 design |
| **2** | *Gradient descent through development* (METHOD) | a: perceive stencil [id,∂x,∂y,∇²]; b: rule as compute graph (residual-in-tanh); c: BPTT through T dev-steps, loss on I/O cells, markers; d: the curricula (distance ramp, persistence, reactivity, expose→internalize carry) | concept | 🔴 design |
| **3** | *From a gate to arithmetic* (COMPOSITION) | a: XOR gate + signal-channel development; b: 1-bit adder (sum,carry); c: **2-bit adder** internal carry + causal-lesion dissociation; d: success bars w/ CIs | data | 🟡 data exists |
| **4** | *Robust under damage & asynchrony* | a: self-repair filmstrip + **heal heatmap**; b: async/NCA spectral (period-3 ring damped) + fireRate sweep; c: long-horizon accuracy vs steps; d: live input reactivity (adder) | data | 🟡 partial |
| **5** | *Position- and scale-invariant computation* | a: movable wire/XOR — drag ports, field re-routes; b: grid-size generalization 11→13→17 (same params); c: channel-separation (shared vs separated → un/learnable); d: multi-seed reproducibility | data | 🟡 partial |
| **6** | *The learned computer is a real program* (ANCHOR) | a: rule→fixed-point→Z80 datapath; b: bit-exact Z80 vs f32 across the truth table; c: forward-mode gradient in-substrate vs finite-diff; d: honest scope box (offline grid→lane; single-cell tangent) | data | 🟢 data exists |
| **7** | *What matters, measured* (ABLATIONS) | a: ablation bars (perception/nonlinearity/capacity/damage/developmental-vs-not); b: multi-seed success curves w/ CIs; c: non-developmental baseline capability matrix | data | 🟡 needs §2 runs |
| (S) | Supplementary figs | grow-from-seed, reactive-XOR, per-experiment tables, spectral details, Z80 assembly | data | 🟡 |

Production: build a `figures/` pipeline — each figure is generated from committed params + a small
script (SVG/matplotlib) so it's reproducible and regenerates when data changes. Conceptual figures
(1,2) drafted as clean vector art (consistent visual language: cells, arrows, filmstrips).

---

## 4. Paper structure — section by section

- **Title / Abstract** — the pitch (§0) in ~150 words; lead with the thesis, end with the substrate bridge.
- **Introduction** — (i) biology grows & repairs robust computers with no blueprint; (ii) the gap: Neural
  CA grow *images* not computers; classic ALife has *no gradient*; learned-CA-on-grid have *no growth/
  repair* and are *not a program*; differentiable-logic/Neural-GPU compute but not developmentally-grown,
  self-repairing, position-invariant, ISA-realized; (iii) contribution list (the ladder + the anchor).
- **Results** (each subsection = a figure):
  1. A learned local rule grows a functional gate (Fig 1,2,3a).
  2. The paradigm composes into arithmetic with a produced-then-consumed internal signal (Fig 3).
  3. The grown computer is robust to damage and asynchronous update (Fig 4).
  4. One rule computes at any placement and grid size (Fig 5).
  5. The learned rule is a program in a real ISA that carries its own gradient (Fig 6).
  6. Ablations, baselines, and statistics (Fig 7).
- **Discussion** — developmental computation as a paradigm; links to morphogenesis & basal cognition
  (Levin), robust/soft hardware, in-materia computing; **limitations, stated plainly** (small circuits;
  position-invariant *computation* partial; substrate proof offline; gradient is single-step in-substrate).
- **Methods** — rule spec; BPTT + forward/backward; curricula; ablation protocols; Z80 datapath +
  validation discipline; statistics; compute.
- **Data & code availability** — repo + frozen params + **the interactive artifact** (§7).

---

## 5. Related work & citations — the map (cite AND distinguish; no strawmen)

- **Neural Cellular Automata** — Mordvintsev, Randazzo, Niklasson, Levin, *Growing NCA*, Distill 2020;
  *Self-classifying MNIST CA* (Randazzo 2020); Texture/Isotropic/Attention-NCA follow-ups. → they grow
  *patterns/classify*; we grow a *computer* that composes, is position-invariant, and is ISA-realized.
- **Differentiable logic / algorithmic nets** — *Differentiable Logic CA* (2024); *Neural GPU* (Kaiser
  & Sutskever, ICLR 2016); Neural Turing Machine / DNC (Graves 2014; Nature 2016). → they compute/
  length-generalize but are not developmentally grown, self-repairing, position-invariant, or ISA-run.
- **Classical CA & universality** — von Neumann (self-reproduction, 1966); Conway's Life; Wolfram (NKS
  2002); Cook (Rule 110 universality, 2004). → hand-designed/evolved, not gradient-learned.
- **Morphogenesis & basal cognition** — Turing (1952); Levin (regeneration, bioelectricity, "computational
  boundary of a self," 2019). → biological motivation; we give a *learned, computational* instance.
- **Physical / in-materia computing** — Wright et al., *Deep physical neural networks*, Nature 2022;
  reservoir computing (Jaeger; Maass). → gradient in a physical substrate; ours is a *discrete ISA* + a
  program that is itself the computer.
- **AD & training** — Werbos (BPTT, 1990); forward-mode AD / dual numbers (Wengert; Griewank & Walther).
- **Developmental encodings** — Stanley (CPPN/HyperNEAT). → evolved developmental encodings vs our
  gradient-trained local rule.

Target ~35–45 references. Build `paper/refs.bib` incrementally; every claim of novelty pins a specific row.

---

## 6. Supplementary materials

- Full rule spec + all hyperparameters + curricula (reproducible).
- Every ablation table + multi-seed run with seeds + CIs.
- Z80: the assembly, the validation (bit-exact + gradient), the honest scope.
- Extra experiments: grow-from-seed, reactive adder/XOR, spectral analysis.
- Videos: grow/compute/damage/heal; async; drag-ports; live input change.
- **Code + frozen params release** (the repo, cleaned) + the interactive artifact.

---

## 7. The interactive companion website (a real NMI asset)

A Distill-style **interactive paper** built on Algocell (`/devcomp` is the seed). Sections mirror the
figures but *playable*: (1) the concept, animated; (2) **play** — grow / compute / brush-damage / heal /
drag ports / toggle async, live on WebGPU; (3) **train it yourself** — the in-browser GPU trainer watches
a gate/adder learn from scratch; (4) **the substrate bridge** — watch the trained rule run as Z80 assembly.
This is impact + reproducibility in one, and a genuine differentiator (few NMI papers ship a live,
trainable artifact). Deploy as a static site (SvelteKit adapter-static). Must run on a mid-range laptop
GPU; graceful fallback to precomputed frames without WebGPU.

---

## 8. Production roadmap (prioritized)

**Phase A — finish the science (rigor).** (1) non-developmental baseline; (2) movable-task
perception ablation; (3) damage-in-training ablation; (4) complete multi-seed (2-bit adder, movable XOR
full recipe, heal heatmaps); (5) resolve the Z80 gradient claim (full-rollout Q16.16 *or* scope precisely).
**Phase B — narrative & figures.** Lock the claim; draft Figs 1–2 (concept) first — they discipline the
story; then generate Figs 3–7 from committed data; write related work (§5). **Phase C — write.** Methods
first (easiest, forces precision), then Results around the figures, then Intro/Discussion/Abstract last.
**Phase D — artifact & polish.** Build the interactive site; clean the repo; internal red-team review
(the adversarial-critic pass we already ran once — repeat on the full draft).

Rule of thumb: **write the figure captions before the prose.** If a figure can't be captioned in 3
sentences, it isn't earning its place.

---

## 9. Mindset & principles (the standard)

1. **Honesty over hype.** We already caught ourselves overclaiming the Z80 "training gradient"
   (it's a single-step JVP) — that correction *is* the standard. Every claim is scoped to exactly what
   the evidence shows; limitations are stated in the main text, not buried.
2. **Every number has an error bar.** No headline is n=1 by the time we submit.
3. **Baselines make novelty legible.** A capability matrix vs a non-developmental control is what turns
   "cool" into "necessary."
4. **Conceptual clarity first.** Figs 1–2 must make a smart non-specialist *want* to read on. The
   algorithm, the loop, the bridge — each in one picture.
5. **Position precisely.** Cite and distinguish the real nearest neighbors (NCA, Diff-Logic-CA, Neural
   GPU, physical NNs); never a strawman.
6. **Reproducibility is a feature.** Frozen params + scripts + a live, trainable artifact.
7. **The reader's attention is earned, not assumed.** One thesis, a ladder that builds, no detours.

**Are we in a good place?** Yes — on the two hardest things to fake: a genuinely novel idea and a
ladder of real results, de-risked end to end (including the hardware bridge). What remains is not
invention but *execution*: complete the rigor (Phase A, ~a handful of runs), then design the figures
and write with discipline. We know exactly what to build and why. That is the right place to be.

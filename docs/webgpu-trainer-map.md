# WebGPU training system — code map & design notes

Written for an engineer/agent building a **WebGPU-based neural training system** who wants to read a
working implementation. Assumes no knowledge of this project. All paths are absolute, on this machine.
Repo root: `/Users/neo/repos/algocell`

---

## 1. What is being trained (30 seconds of context)

A **cellular automaton whose update rule is a small neural network**. A 2-D grid of cells; each cell
holds a vector of `C` float channels. Every step, *every cell applies the same tiny MLP*:

```
perceive = [identity, ∂x, ∂y, laplacian] per channel   (fixed filters, 4·C values)
delta    = W2 · relu(W1 · perceive + b1) + b2
state'   = tanh(state + delta)                          (residual INSIDE the tanh)
```

The rule is unrolled for `T` steps ("development"), a loss is read from a few designated output cells,
and the **same shared weights** are trained by **backpropagation through time (BPTT)**. Parameter count
is small (~3k–13k); the expensive part is the rollout over `T` steps × all cells × a batch.

**Why a GPU trainer:** BPTT over a batch of independent samples is embarrassingly parallel per (sample,
cell). The GPU version is ~65× the CPU version and runs in the browser.

---

## 2. The two files that ARE the trainer

### `/Users/neo/repos/algocell/src/lib/devcomp/trainShader.ts` (368 lines)
The **WGSL compute kernels**, emitted as a template string so grid/channel/hidden sizes are baked in as
constants. Eleven `@compute` entry points forming the whole training step:

- `fwd` — one CA step; writes into a stored trajectory buffer (needed for backprop).
- `seedGrad` — initialises dL/d(state) at the final step from the output cells.
- `bwd1` — per-cell vector-Jacobian product; writes per-cell scratch (`gp`, `gpre1`, `hAct`, `perc`, `gperc`).
- `bwdGather` — propagates the state gradient to neighbours (see §4, gather-not-scatter).
- `injectOut` — adds output-loss gradient at intermediate steps (persistence/hold-window objectives).
- `gradW1`, `gradW2`, `gradBias` — parameter-gradient reductions (see §4, no atomics).
- `zeroGrad`, `gradNormSq`, `adam` — optimizer: zero, grad-norm for clipping, Adam update in-place.

### `/Users/neo/repos/algocell/src/lib/devcomp/gpuTrainer.ts` (248 lines)
The **host-side `GPUTrainer` class**: allocates buffers, builds one bind group, compiles the pipelines,
and sequences the passes. Key methods: `create()`, `setParams()`, `setBatch()`, `trainStep(lr, it)`,
`computeGrad()` (for validation), `evalOutputs()`, `readParams()`. The per-iteration control flow lives
in `forwardBackward()`: T forward passes → seed gradient → zero grads → T backward passes (each a small
group of kernels submitted together).

---

## 3. Where it is driven from (browser entry points)

Headless WebGPU was **not** available on this machine (the Dawn node binding hangs on device init), so
all GPU training is driven from browser pages under a SvelteKit dev server (`npm run dev`, port 5173):

- `/Users/neo/repos/algocell/src/routes/devcomp/traingpu/+page.svelte` — validates the GPU gradient
  against the CPU reference, then trains a position-invariant task. **Best file to read first for usage.**
- `/Users/neo/repos/algocell/src/routes/devcomp/valfixed/+page.svelte` — validates a second I/O mode and
  trains a small adder end-to-end (shows a full training loop + curriculum).
- `/Users/neo/repos/algocell/src/routes/devcomp/multiseed/+page.svelte` — many-seed runs for statistics.
- `/Users/neo/repos/algocell/src/routes/devcomp/movseed/+page.svelte` — same, for a different task.

*Practical note:* a long synchronous training loop blocks the page. These files `await` a zero-delay
promise every ~50 iterations so the UI stays responsive and progress can be read out.

---

## 4. The design decisions that actually matter (read this before writing your own)

1. **No f32 atomics in WebGPU.** You cannot atomically accumulate float parameter gradients across
   threads. Solution: **reduction instead of scatter** — one thread per *parameter*, looping over all
   (sample, cell) entries and summing from a per-cell `scratch` buffer. See `gradW1`/`gradW2`/`gradBias`.
   This is the single biggest structural constraint.
2. **Gather, don't scatter, for the state gradient.** The perceive stencil means a cell's gradient
   contributes to its 4 neighbours. Scattering would race. Instead `bwdGather` has each cell *pull*
   from its own and its neighbours' `gperc` scratch. Transposed stencil, race-free.
3. **Ping-pong gradient buffers.** `gsA`/`gsB` alternate each backward step (a `dir` flag in the uniform)
   because a pass cannot safely read and write the same buffer.
4. **Buffer packing to fit the 8-storage-buffer/stage baseline limit.** Several logical arrays share one
   buffer at fixed offsets: `optim = [grad | m | v | ‖grad‖²]`, `portsU = [isInput | isOutput]`,
   `portsF = [inVal0 | inVal1 | tgt0 | tgt1]`, and a per-cell concatenated `scratch`.
5. **The trajectory dominates memory.** Backprop needs every intermediate state: `(T+1) × B × N × C`
   floats. This can exceed the default 128 MB binding cap → request the adapter's
   `maxStorageBufferBindingSize` / `maxBufferSize` in `requestDevice()` (done in `GPUTrainer.create`).
6. **A small uniform carries per-step control** (`t`, `dir`, `lr`, Adam bias-correction terms), rewritten
   before each pass. Cheap and avoids recompiling pipelines.
7. **Throughput is submit-bound at small batch.** Many tiny dispatches per step means CPU-side submit
   overhead dominates when the batch is small (~45 it/s at batch 8). Larger batches amortise it.

---

## 5. Correctness discipline (do not skip)

The GPU gradient is validated against a **CPU reference that is itself finite-difference-checked** —
never the other way round. The reference lives in:

`/Users/neo/repos/algocell/src/lib/devcomp/rule.ts`
- `lossAndGradMarkers` (line ~309) — reference for one I/O mode
- `lossAndGradFixed` (line ~412) — reference for fixed multi-input/multi-output layouts
- also contains the rule spec, config, and forward stepper that all implementations must agree with.

Achieved agreement: **3e-5** (max relative) and **1.97e-6** on the two paths. Any refactor that breaks
those numbers is a bug. The validation pages in §3 run these checks on load.

---

## 6. Related but NOT the trainer

- `/Users/neo/repos/algocell/src/lib/devcomp/shader.ts` (95) and
  `/Users/neo/repos/algocell/src/lib/devcomp/engine.ts` (134) — **forward-only** WebGPU (inference for a
  live demo). Much simpler; a good warm-up read for the WGSL style.
- `/Users/neo/repos/algocell/src/lib/morph/dev/*.ts` — **CPU (Node/tsx) trainers** for the same rule.
  Useful as readable, unoptimised ground truth for the same math. Not WebGPU.
- `/Users/neo/repos/algocell/src/lib/gpu/`, `/Users/neo/repos/algocell/src/lib/morph/zilion.ts` — an
  unrelated GPU subsystem (a Z80 CPU emulator in WGSL). Ignore for training purposes.
- `/Users/neo/repos/algocell/docs/gpu-trainer-design.md` — the original design doc.

---

## 7. Suggested reading order

1. `rule.ts` — understand the math being differentiated (forward + the CPU backward).
2. `shader.ts` — the same forward step in WGSL, minimal.
3. `trainShader.ts` — the backward + optimizer kernels (§4 explains the non-obvious choices).
4. `gpuTrainer.ts` — buffers, bind group, pass sequencing.
5. `routes/devcomp/traingpu/+page.svelte` — end-to-end usage: validate, then train.

**Known limitation of this implementation:** there is no damage/dropout-style perturbation *inside* the
GPU forward pass, so objectives that require corrupting the state mid-rollout are still done on CPU.
Adding it would mean zeroing a masked region in `fwd` at a chosen step and masking the gradient
correspondingly in the backward pass.

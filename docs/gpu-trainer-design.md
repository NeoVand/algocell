# GPU/browser trainer for the field-CA rule (S7-GPU)

Goal: train the developmental-computation rule **on the GPU, in the browser**, much faster
than the single-threaded JS CPU trainer — so (a) all the paper's remaining training (movable
XOR, 2-bit adder, S8 multi-seed stats, ablations) is fast, and (b) users can watch/drive
training live in the demo (S7).

The forward kernel already exists (`shader.ts`) and is GPU-validated bit-faithful to the CPU
reference (`rule.ts`). This adds the **backward pass** (reverse-mode BPTT) + optimizer on GPU.
The CPU reference `expI.ts::lossAndGrad` is finite-difference gradient-checked (≈1e-7), so it
is the **trusted oracle**: the GPU gradient MUST match it to ≈1e-4 before we trust a byte.

## Rule (per interior cell, per step)
```
perc[k]   = [identity, gx=(r−l)/2, gy=(d−u)/2, lap=r+l+u+d−4·self] per channel   (PERC = 4·C)
pre1[hh]  = b1[hh] + Σ_k W1[hh,k]·perc[k]        (HD)
h[hh]     = relu(pre1[hh])
dl[c]     = b2[c] + Σ_hh W2[c,hh]·h[hh]          (C)
s'[i][c]  = tanh(s[i][c] + dl[c])                ← residual INSIDE tanh
then STAMP: markers (ch1=IN_MARK, ch2=OUT_MARK) + per-input signal channels re-clamped.
```

## Backward (VJP of one step): gs = dL/ds'[t+1]  →  gsPrev = dL/ds[t]
Clamp first: `gs[*][IN_MARK]=0`, `gs[*][OUT_MARK]=0`, `gs[inputCell][inCh]=0` (clamped values
don't depend on the rule). Then per interior cell i (recompute perc,pre1,h from s[t]):
```
gp[c]        = gs[i][c] · (1 − s'[i][c]²)               ; gsPrev[i][c] += gp[c]   (residual)
grad_b2[c]  += gp[c] ;  grad_W2[c,hh] += gp[c]·h[hh]
gh[hh]       = Σ_c W2[c,hh]·gp[c]
gpre1[hh]    = gh[hh] · (pre1[hh] > 0)                  (relu gate)
grad_b1[hh] += gpre1[hh] ;  grad_W1[hh,k] += gpre1[hh]·perc[k]
gperc[k]     = Σ_hh W1[hh,k]·gpre1[hh]                  (= dL/dperc[i][k])
```
Perc is a linear gather from the 4-neighbourhood, so its transpose scatters gperc back. To
avoid GPU write-races we express it as a **gather** (each cell writes only its own gsPrev):
```
gsPrev[j][ch] += gperc[j][ch·4+id]·1  + gperc[j][ch·4+lap]·(−4)
              + gperc[j−1][ch·4+gx]·0.5 + gperc[j+1][ch·4+gx]·(−0.5)
              + gperc[j−SW][ch·4+gy]·0.5 + gperc[j+SW][ch·4+gy]·(−0.5)
              + (gperc[j−1]+gperc[j+1]+gperc[j−SW]+gperc[j+SW])[ch·4+lap]·1
```

## Param-grad accumulation = matmul reduction (NO f32 atomics)
Param grads are outer-product sums over all cells (batch·interior) and all steps:
```
grad_W2[c,hh]  = Σ_cell gp[cell][c]·h[cell][hh]        → GP^T · H     ([C×HD])
grad_W1[hh,k]  = Σ_cell gpre1[cell][hh]·perc[cell][k]  → GPRE1^T·PERC ([HD×PERC])
grad_b2[c]     = Σ_cell gp[cell][c]                    ; grad_b1[hh] = Σ_cell gpre1[cell][hh]
```
So `BWD1` stores per-cell `gp[C], gpre1[HD], h[HD], perc[PERC], gperc[PERC]`; then small matmul
kernels reduce over cells into the grad buffers (accumulated across steps). Avoids the
high-contention f32-CAS-atomic trap entirely, and the matmuls are tiny (≤96×64 outputs).

## Batch packing
Pack B independent samples (each = one placement × one input-case) along a batch dim: buffers
are `[B][N][C]`. The rule is sample-independent, so one kernel invocation per (b,i). Markers /
inputs are **per-sample** (`isInput[B][N]` channel+1-encoded, `inputVal[B][N]`, `isOutput[B][N]`).
Effective batch = B placements per iter at ~constant wall-time → the speedup + exactly the
"many samples, each a different port location, from iter 1" methodology.

## Buffers
- `state` ping-pong `A/B` `[B·N·C]`; **trajectory** `traj[(T+1)·B·N·C]` (stored for backward).
- `params[P]`; `grad[P]` (f32, zeroed each iter); Adam `m[P]`, `v[P]`.
- per-sample `isInput[B·N] (u32, chan+1)`, `inputVal[B·N] (f32)`, `isOutput[B·N] (u32)`.
- per-cell scratch (per step, reused): `gp[B·N·C]`, `gpre1[B·N·HD]`, `hAct[B·N·HD]`,
  `percBuf[B·N·PERC]`, `gperc[B·N·PERC]`.
- `gsA/gsB[B·N·C]` gradState ping-pong; `lossOut` (reduction of output-cell errors).

## Kernels
1. `fwd`      : traj[t] → traj[t+1] (forward + stamp), per (b,i).
2. `seedGrad` : gsA = 0; at output cells ch0 set `2·(o−tgt)/norm` (loss grad).
3. `bwd1`     : per (b,i) recompute perc/pre1/h from traj[t]; write gp,gpre1,hAct,percBuf,gperc;
               gsPrev residual (own cell). (gs clamped at markers/inputs first.)
4. `gradW1/W2/b1/b2` : matmul/colsum reductions over cells → += grad buffers.
5. `bwdGather`: gsPrev[j] += perc-transpose gather from gperc neighbours.
6. `adam`     : params -= lr·Adam(grad); zero grad.

## Orchestration (JS, all data resident on GPU)
```
per iter:
  build per-sample placements/cases (varied from iter 1) → upload isInput/inputVal/isOutput
  seed traj[0] (uniform-alive + stamped markers/signals)
  for t in 0..T-1: dispatch fwd
  dispatch seedGrad (+ read output cells for loss/acc logging — tiny readback)
  zero grad
  for t in T-1..0: clampGs; bwd1; gradW1;gradW2;gradb1;gradb2; bwdGather; swap gs
  dispatch adam
```
Only per-iter CPU↔GPU sync: upload placements (small) + optional loss readback (B·#out floats).

## Validation plan (decisive)
`/devcomp/traingpu` route:
1. **Gradient check**: fixed params + fixed batch of placements/cases; compute grad on GPU and
   with CPU `expI.lossAndGrad`; assert `max|Δ|/(|cpu|+1e-6) < 1e-3` over all P. This is the go/no-go.
2. **Loss match**: GPU forward loss == CPU loss for the same batch.
3. **Train**: run varied-from-start movable XOR with large B; report position-invariant accuracy
   (reuse CPU `evalAcc` on the trained params) on 11×11 / 13×13 / 17×17.

Only after (1) passes do we trust GPU-trained params. Params are portable (same layout P), so a
GPU-trained rule drops straight into the existing demo + `forwardMarkers` CPU validation.
```

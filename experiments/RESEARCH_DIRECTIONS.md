# Research directions — draft v0 (2026-10-07, written before the Stage C/D relaunch)

Purpose: decide what the paper is *about* before spending more GPU time. Stages A and B were an atlas; the atlas has produced three mechanistic regularities that look like laws of this system, not catalogue entries. The programme below is built around testing them to destruction, with falsifiable predictions stated before the runs, and around the instrument upgrades to Zilion that each test needs.

## 1. What the data already says (Stages A, B; 1,270 runs)

1. **Replication is cheap and robust above 16 bytes.** Unablated soups produce a heritable replicator in 10/10 seeds at every L from 16 to 100, in 500–1,500 steps, independent of L.
2. **Form is not size.** Large organisms are tiled by short motifs: period-2 Load–Push pairs under the full ISA, shift-copies whose period divides the 2L-byte pair memory when the stack is removed. Dominant-tape high-order entropy is 1–2 bits/byte at every L. No whole-tape replicator of period > 32 was seen in 640 runs.
3. **Removing instructions moves the system between regimes rather than killing it** — except for the 112-instruction no-copy set (0/250 across all L and budgets) — and in one place it does the opposite: at L = 9 the stack-writing families are a sterile zero flood, and removing them raises emergence from 4/10 and 1/10 to 9/10 and 10/10.
4. **The dominant genotype is often not the replicator.** Sterile pushed offspring outnumber their copier; return-address smears spread without heredity; a CALL chain is heritable for two generations and then dies. Occupancy-based detection (the field's default) mislabels all three.

## 2. Three candidate laws, each with a decisive test

### Law A — ring arithmetic: a shift-copier's period is gcd(offset, pair length)
Mechanism (verified with the executor, 2026-10-07): LDIR with DE − HL = d on a ring of P = 2L bytes converges to a period-gcd(d, P) pattern; offsets with gcd 1 scramble the tape (L = 36, d = 5: partner period 36, parent destroyed). Evolved offsets in Stage B are divisors of 2L in 84–100% of runs (3 | 18, 5 | 50, 7 | 98, 9 | 162, 4·8·16·32 | 128).
**Prediction A1.** Pad the pair memory to a prime length P (e.g. 2L + 1 = 33 at L = 16; 73 at L = 36): shift-copiers cannot exist, so under the stack-writes ablation replication should vanish or be replaced by exact-offset copiers (`LD DE,L ; LDIR`, period = L). Under the full ISA the Load–Push family (which does not depend on gcd) should be unaffected. A one-byte change in memory size should remove an entire replicator family. **Prediction A2.** With P = 2L + 2 (gcd 2 available), period-2 shift-copiers appear.
**Instrument:** Zilion/engine parameter `pairMemoryLength ≥ 2L` (padding bytes zero, not persisted). Small shader change (`PAIR_MEM_WGSL`), exported to the headless runner.
**Cost:** L ∈ {16, 36} × P ∈ {2L, 2L+1, 2L+2, next prime} × {none, stack-writes} × 10 seeds × 300k steps = 160 runs ≈ 4.5 GPU-h ≈ $9.

### Law B — budget arithmetic: one encounter copies at most ~`steps` bytes, so tiling makes replication piecewise
Mechanism (verified): each LDIR iteration is one Z80 step; at L = 100 and 128 steps a shift-copier converts 31/100 partner bytes per encounter, at 512 steps 100/100. A tiled tape is correct in any chunk at a consistent phase, so it can be assembled across encounters; a whole-tape copier cannot.
**Prediction B1.** At fixed L = 100, raising the budget from 128 to 2048 steps lets whole-tape replicators (period = L, carrying cargo) appear and persist; at ≤ 128 they never fix. **Prediction B2.** At fixed budget, emergence time is flat in L until `steps < L + setup` for the stack family too (pushes write 2 bytes/2 steps), i.e. a knee near L ≈ steps. Stage B saw no knee up to L = 100 at 128 steps for Load–Push — the knee should appear at L ≈ 144–196 (headless only).
**Cost:** L = 100 × steps ∈ {32, 128, 512, 2048} × {none, stack-writes} × 10 seeds + L ∈ {144, 196, 256} × 128 steps × none × 10 seeds = 110 runs; the 2048-step runs cost 16× per step — ≈ 12 GPU-h ≈ $24.

### Law C — mutation arithmetic: the maintained information is bounded by the error threshold, and the excess becomes redundancy
Mechanism: the per-cell mutation rate is constant in L (512 bytes/step over 20,000 cells → one byte per cell every 39 steps; one interaction per cell every ~2 steps). Quasispecies theory bounds the maintainable genome length by ~ ln(s)/μ. Tiling keeps the *effective* genome (one period) short while filling L.
**Prediction C1.** At L = 100, the dominant period (effective information) decreases monotonically with mutation rate k = 8 → 1, and the LDIR cloud becomes the only survivor above a critical rate (Stage A saw this at L = 16: LDIR clouds at k = 2, Load–Push at k = 6). **Prediction C2.** Whole-tape copiers from B1 lose their cargo at high mutation first, keeping the core.
**Cost:** L ∈ {16, 100} × k ∈ {1, 2, 3, 4, 5, 6, 8} × none × 10 seeds × 300k = 140 runs ≈ 4 GPU-h ≈ $8.

## 3. Two deeper questions the atlas cannot answer without new instrumentation

### D — minimal replicators and the vulnerability barrier
`DEC E ; LDIR` is a 3-byte replicator. A 4-byte copier scores 1.00 in the assay at L = 4, yet L = 4 soups produced 1 replicator in 80 runs and L = 9 is marginal. Hypothesis: small copiers are suicidal as the second program — the partner's junk executes first and the copier's own LDIR then copies that junk over itself. The assay measures copying as A and B but not *destruction suffered as B*.
**Add to the assay:** `self_preserved_as_B` (fraction of T's bytes intact after a random partner runs first). **Prediction D1.** Vulnerability, not copy fidelity, separates L = 4/9 from L ≥ 16. **Experiment:** headless L ∈ {2, 3, 5, 6, 7, 8, 10, 12} × {none, stack-writes} × 10 seeds — where does life start, and is the first replicator always the 3-byte unit at L divisible by 3? ≈ 160 runs ≈ $9.

### E — succession as measurable ecology
The soup is deterministic per pair. The pairwise interaction matrix among the top genotypes (who overwrites whom, how much) can be *measured exactly* with the executor, and the macroscopic succession (Load–Push takeover → LDIR invasion sweeping to 98% in ~15k steps) can then be predicted by replicator dynamics from micro-payoffs and compared with the observed trajectories. No external fitness exists, so this is a genuine test of whether pairwise execution payoffs explain population dynamics.
**Instrument:** provenance tagging in Zilion — one author tag per soup byte (written by A / by B / mutation), updated by the host write path. With it: true lineages (who copied whom), in-situ fitness, parasitism detection, phylogenies — and the time-resolved functional fraction without random-cell assays. Memory cost: one byte per soup byte.

## 4. Instrument upgrades to Zilion/Algocell (each justified by a test above)

| Upgrade | Needed by | Size |
|---|---|---|
| `pairMemoryLength` (ring padding) | Law A | small (shader const + engine + export) |
| Executed-instruction counter and bytes-written counter per pair | Laws B, D | small (two u32 per pair) |
| `self_preserved_as_B` and generation-3 test in the assay | D, faithfulness | small (Python) |
| Provenance tags per soup byte | E | medium (extra buffer + host writes + readback) |
| Deterministic pairing option (per-step perfect matching, no atomics) | reproducibility; a reviewer will ask | medium; changes the interaction graph, so it is an option, not a replacement |
| Periodic soup snapshots at a cadence (not only emergence/final) | E, time-resolved assays | small (runner) |
| Headless support for non-square L | D | small (export + runner already parametric) |

## 5. What this makes the paper
Working title: *The arithmetic of self-replication: memory, budget and noise decide the form of life in a soup of programs.* Claim: in a fitness-free Z80 soup the structure of emergent replicators is set by three measurable constraints — ring arithmetic (gcd), encounter budget (bytes per interaction) and mutation load (error threshold) — not by organism size; instruction ablations move the system between regimes, and at small sizes the stack is a sterile flood whose removal *creates* life. Each constraint has a prediction that a one-parameter change should confirm or kill. Figures: the ablation × budget × mutation atlas as a regime map; the period-vs-divisor law; the prime-ring kill experiment; the information-vs-mutation curve with the Eigen bound; soup snapshots coloured by family and by provenance.

## 6. Order of work
1. Finish the code review (in progress), fix, test, document — no GPU until done.
2. Relaunch Stage C (420) and D (148 remaining) only for the parts still needed under this programme: C1 (1M-step nulls), C3 (finer ablations), C5 (size without stop) and D stay; C2 (uncensored succession at L = 16) is kept because E needs the trajectories. Cost ≈ $42.
3. Implement `pairMemoryLength`, the per-pair counters and `self_preserved_as_B`; export; test; pre-register Laws A–C and D as Stage E in PLAN.md with the predictions above; run (≈ $50).
4. Provenance tagging and the ecology analysis (E) as the second half of the paper.

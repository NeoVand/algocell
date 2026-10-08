# Pre-registration: the two biology-bridge analyses (2026-10-08, written before any computation)

Both analyses use data already on disk (no paid runs; GPU use limited to partner-test assays of a few seconds).
Predictions and kill criteria are fixed here before the code is run. Outcomes are appended at the end of this file.

## A. Assembly fires on sterile order

**Claim under test.** Complexity-times-abundance measures of selection (assembly theory: Sharma et al., *Nature*
2023; Marshall et al., *Nat. Commun.* 2021) cannot separate the first replicator from the sterile order (tar) that
precedes it, because both are low-complexity periodic objects of high copy number.

**Data.** Stage G (`runs/stageG`, 80 worlds, L ∈ {16, 20, 50, 64}): per-sample records (top-10 exemplar tapes and
shares, `unique`, `hoe`) at every sampled step; soup snapshots at 500, 1,000, 2,000, 3,000, 5,000, 7,500, 10,000,
15,000, 20,000, 30,000, 50,000, 75,000, 100,000, 150,000, 200,000, 300,000 steps, at emergence and at the end;
`results/stageG/stageG/stage_g_runs.csv` (t_rep, first_tape, final_tape). Stage E (`runs/stageE`, `results/stageE/assays.csv`)
for the detector comparison across tape lengths, with the strata of `detectors.py`.

**Measures (fixed now).**
- String assembly index a(s): minimal number of join operations to build s from single bytes with reuse of built
  substrings. Computed as the Re-Pair upper bound: repeatedly replace the most frequent adjacent symbol pair by a
  new symbol (ties broken by first occurrence) until no pair repeats; a = number of rules + (length of the final
  sequence − 1). Exact for periodic tapes (e.g. `01 c5` × 8 → 4; sixteen zero bytes → 4; a 16-byte string with no
  repeated pair → 15).
- Assembly of a snapshot: A = Σ_i e^{a_i} (n_i − 1) / N_T over unique tapes i with copy number n_i ≥ 2, N_T = 20,000
  (Sharma et al. 2023, eq. 1). A_top10(t): the same sum over the ten most common classes recorded in the sample at
  step t (shares × N_T), available at every sampled step.
- Baseline: A_top10 at the first sample (step 1).

**Predictions.**
- P1. The assembly index of the first heritable replicator (first_tape) is ≤ the assembly index of the modal tape at
  the last sample before t_rep (the tar) in ≥ 80% of the 80 Stage G worlds. Kill: < 50%.
- P2. A_top10(t) exceeds 10 × baseline at some sample before t_rep in ≥ 75% of Stage G worlds (assembly fires on tar
  before life exists). Kill: < 50%.
- P3. As a detector of heredity across Stage E worlds (ground truth `event` of `detectors.py`: the run's first
  heritable replicator has appeared), the AUC of A_top10 pooled over L ≥ 25 is ≤ 0.65. Kill: ≥ 0.75 (assembly would
  then be a usable detector of heredity, and the bridge claim fails).

**Outputs.** `results/biology/assembly/`: per-world table (a of first tape, a of tar modal tape, step of first A
crossing, t_rep), per-sample A_top10 table, per-snapshot exact A table, AUC table by L, a time-course figure for the
exemplar world (L = 16, seed 2001) with A, HOE and t_rep, a strip plot of first-crossing step against t_rep for all
worlds, NUMBERS_ASSEMBLY.md and FINDINGS.md.

## B. Individuality as zero information inflow

**Claim under test.** Krakauer et al. (2020) define individuality by how much of a system's future is determined by
its own present rather than by its environment. In a deterministic world with the organism X fixed, the information
flowing from the environment E into the offspring O is I(O; E | X) = H(O | X) − H(O | X, E) = H(O | X): the entropy
of the offspring over random environments. Lemma 7 (closure ⇔ zero inflow) predicts that this quantity is positive for
the open first replicator and zero for its closed successor.

**Data and procedure.** Stage G first and final tapes (80 worlds). For each tape X: 256 uniformly random partners E_j
of length L; execute the pair (X in cells 0..L−1, E_j in cells L..2L−1) for 128 instructions with
`algocell_exp.assay.execute_pairs` (same kernel as the culture test); offspring O_j = the partner half afterwards,
X'_j = the organism half afterwards. BFF (84 soups, `results/bff/runs.csv`): the same with the BFF executor of
`micro/bff.py` for the first and final tapes of every life-producing soup, under each soup's own variant rules.

**Measures.** H(O | X): plug-in entropy in bits of the empirical distribution of O_j over the 256 partners (maximum
8 bits). H_class(O | X): the same over three outcome classes, faithful copy (best-shift similarity to X ≥ 0.95),
partial copy (≥ 0.75), other. p_modal: share of the most common offspring. H(X' | X): the entropy of the organism's
own half (self-damage).

**Predictions.**
- Q1. H(O | X_first) > 1 bit in ≥ 90% of the 80 Stage G worlds. Kill: < 60%.
- Q2. H(O | X_final) < 0.5 bit in ≥ 90% of the worlds whose final dominant carries a loop instruction. Kill: < 60%.
- Q3. H(O | X_first) − H(O | X_final) ≥ 2 bits in ≥ 75% of worlds. Kill: < 50%.
- Q4. BFF: H(O | X_first) < 0.5 bit in ≥ 90% of life-producing soups of the published variant (born closed), and
  > 1 bit in ≥ 90% of soups of the literal-push variants (lit, wraplit, wraplitnh). Kill: either < 60%.

**Outputs.** `results/biology/individuality/`: per-replicator table (world, which, H, H_class, p_modal, H_self, has_loop,
copied, damaged), a paired first → final figure per machine, NUMBERS_INDIVIDUALITY.md and FINDINGS.md.

## Outcomes

### B. Individuality as zero information inflow — outcome (2026-10-08, `results/biology/individuality/`)

| | population | statement | count | outcome |
|---|---|---|---|---|
| Q1 | 80 Stage G worlds | H(O \| X_first) > 1 bit | 80/80 | met |
| Q2 | 67 worlds whose final carries a loop instruction | H(O \| X_final) < 0.5 bit | 63/67 (0.94) | met |
| Q3 | 80 Stage G worlds | drop ≥ 2 bits | 63/80 (0.79) | met |
| Q4a | 9 life-producing soups, BFF as published | H(O \| X_first) < 0.5 bit | 6/9 (0.67) | between |
| Q4b | 36 soups of the literal-push variants | H(O \| X_first) > 1 bit | 12/36 (0.33) | **killed** |

Z80: first replicators carry a median 7.0 bits about the partner (3.9, 6.2, 7.8, 8.0 bits at L = 16, 20, 50, 64); the
loop-bearing successor carries 0 bits in 63 of 67 worlds, the four exceptions copying all but 1–3 bytes, which stay the
partner's. Q4b was killed because the clause equated "the pointer enters the partner" with "information flows into the
offspring": the literal-push variant without a wrapping pointer is at the 8-bit ceiling (12/12), but with a wrapping
pointer the all-P organism laps the ring and overwrites everything, so its offspring is the same string in nearly every
context (wraplit 0.75 bits, halted encounters only; wraplitnh 0.04 bits) although execution enters the partner in 100% of
encounters. Lemma 7(ii) had predicted exactly this for a perfect copier; the pre-registration contradicted the lemma for
that organism and the data sided with the lemma. Corrected statement: inflow comes from partial copying (unwritten
offspring positions, data read from the partner); pointer entry is the mechanism that makes the Z80 pusher's copy
partial, but it is neither necessary for inflow (1-byte tails of pointer-closed BFF replicators) nor sufficient (the
flooding organism). THEOREMS.md's reading of Lemma 7 is qualified accordingly. Analysis A: below.

### A. Assembly fires on sterile order — outcome (2026-10-08, `results/biology/assembly/`)

| | statement | result | verdict |
|---|---|---|---|
| P1 | a(first tape) ≤ a(tar modal tape before t_rep) in ≥ 80% | 80/80 (79 equal, 1 less) | met, but without power: every first tape is period-2 and its Re-Pair index (4, 5, 7, 6 at L = 16, 20, 50, 64) is the minimum any string of that length can have, so "≤" could not fail; in 59/80 worlds the modal tape before t_rep is already the first tape |
| P2 | A_top10 > 10 × step-1 baseline before t_rep in ≥ 75% | 71/80 (0.89); median lead 19× | met |
| P3 | AUC of A_top10 against `event`, Stage E, L ≥ 25, ≤ 0.65 (kill ≥ 0.75) | 0.721 (HOE on the same samples 0.749) | grey zone: neither met nor killed |

Reading: the assembly index places the first replicator and the sterile order before it at the same, minimal value, and
the assembly measure rises on the tar well before heredity exists (in the exemplar world the crossing is at step 50
against t_rep = 700, and A is higher at the tar sample than at t_rep). As a detector of heredity across tape lengths
≥ 25 it is no better than the compression biosignature. Four L = 64 worlds never cross because one accidentally
duplicated random 64-byte tape in the step-1 sample sets the baseline at e^63/20,000 ≈ 10^23: under the e^a weighting a
single coincidence outweighs every periodic class by twenty orders of magnitude, a property of the measure worth
stating. Post hoc companion (labelled in FINDINGS.md): against the modal tape before the first tape enters the top ten,
a_first ≤ a_tar in 80/80, strictly less in 48.

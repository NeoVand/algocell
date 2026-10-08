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

(appended after the analyses run)

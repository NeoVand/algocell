# Review 1 (AI, three perspectives, 2026-10-08): what it got right, what we verified, what to do

Source: `~/Downloads/paper review/output/pdf/paper-review-three-perspectives.pdf` (untrusted; read only). A second review
is coming; nothing in the manuscript has been changed yet. Verification below was done against the primary sources and
our own tables on 2026-10-08, late night.

## 1. Verified against primary sources: the novelty sentence is wrong and must go

The review says prior work already described the Load–Push first replicator, the LDIR takeover and a multigeneration
transfer detector. Checked (arXiv HTML, fetched tonight into an isolated directory):

- **Agüera y Arcas et al. 2024** (our ref. 5), Z80 section: "Early generations use stack-based copy mechanism: at
  initialization Z80 sets the stack pointer at the end of the address space, so pushing values onto stack gives tape A a
  simple mechanism of writing to tape B"; "The emergence is followed by a 'zero-poisoning' period, after which a new
  family of replicators takes over the soup."
- **Cicala et al. 2026 v2** (our ref. 12), §2.3: "The first, a simple replicator we call 'Load–Push', consists of a
  string of paired 'Load' and 'Push' instruction bytes. Lacking a loop, full replication requires a long chain of these
  pairs"; "the Load–Push replicators were the first to appear and dominate the grids, but they eventually gave way to
  the LDIR replicators"; "LDIR replicators are more robust to mutation than Load–Push replicators"; takeover also without
  tasks (slower); LDD replicators when LDIR is blocked; robustness hierarchy LDIR >> LDD >> Load–Push.
- **Knierim et al. 2026** (our ref. 37), §2: a self-replication detector that repeats five times "T1 ← T2; T2 ← Noise;
  run", i.e. serial transfer into fresh random partners with an exact-position byte-match criterion.

Our `LITERATURE.md` entry for Cicala et al. ends "To extract from the full text: their statement about the 2024
stack→LDIR observation" and that extraction was never done. The sentence "What has not been asked is what the first
replicator is ... nor has heredity been measured directly" survived because of that. This is a process failure and it is
recorded here as one.

**What remains ours** (to be stated as the contribution, with the three papers credited in the first paragraph):
(1) the open/closed characterisation and its measurement by intervention: partner-copy fraction, self-damage, and
information inflow in bits, first against final, in 80 pre-registered worlds and four lengths; (2) the mechanism
"operand = output", the exhaustive two-byte census (the five load–push words are the only two-byte self-writers, all
straight-line), and the theorems; (3) the seed-paired ablation atlas (13 families), the size axis 3–100 with the ring
dead zone and its switch, budget and mutation sweeps; (4) the second machine with the literal switch (BFF + `P`) and the
benign/lethal bracket pair; (5) lethal zeros in the Z80 (born closed, late); (6) the quantitative tar result and its
effect on detectors (zero-poisoning was named in 2024; its 24–34% zero fraction, its 10× assembly rise and the AUC
inversions are ours); (7) the well-mixed null. Cicala et al. explain the LDIR takeover by mutational robustness; we have
not tested that alternative against context independence, and the review is right that we must.

## 2. Verified against our tables: the "seven of eight bits" sentence is wrong in the way the review says, but the
replicators are not the one-random-byte case

`results/biology/individuality/per_replicator.csv` already records what the review asks for. Z80 first replicators:
every position varies in 80 of 80 (`n_var_pos` = L); the offspring is an exact copy in a median 56% of encounters at
L = 16, 29% at L = 20, 5% at L = 50, 0.8% at L = 64 (`p_modal`); 102 / 178 / 240 / 254 distinct offspring strings out of
256 (`n_unique_offspring`); mean positional identity 0.75–0.82 (`sim_mean`). Loop-bearing finals: one offspring string
in all 256 contexts in 63 of 67; the four exceptions copy all but 3, 3, 3 and 1 bytes (`n_var_pos`), the last one being
exactly the reviewer's one-random-byte case (7.1 bits). BFF `lit`: 23 of 64 positions vary, exact copy 0.8%, identity
0.94.

So: the inflow measure is right, its headline is wrong. "Most of what a child looks like is decided by the neighbour" and
"seven of eight bits" must be replaced by the exact-copy probability, the number of varying positions and the mean
identity, with H(o | x) reported as what it is (plug-in entropy over 256 partners, ceiling 8 bits, saturated at L ≥ 50).
"Exactly zero" becomes "no variation in 256 contexts (one-sided 95% bound 1.2%)". Lemma 7's monotonic gloss ("selection
for fidelity is selection against listening") is not proved by the lemma and should be dropped or stated as a reading.

## 3. Verified: no smear ever dated an emergence; the L = 16 successor is not a pusher descendant

All 80 Stage G first replicators are load–push words (`01 c5` 63, `11 d5` 11, `21 e5` 6), none with a control-flow or
block instruction, so the 246 return-address smears never triggered t_rep. The modal L = 16 final (17 of 20 worlds) is
`ad e3 21 e3 21 c0 ad c0`, which shares no bytes with the pusher: at L = 16 the paper may say *succession*, not
*descent*. At L = 50 the modal final (14 of 20) is the pusher tiling with `20 f0` (JR NZ) inserted, where descent is
plausible. The Fig. 2 legend written tonight says "a descendant carrying a conditional return": change to "a new class".
The Z80 partner test does not record pointer entry (`entered_frac` is empty for the Z80 rows); pointer confinement is
known only from exemplar traces.

## 4. The review's points, sorted by what they cost

**Prose and definitions (a day, no data).** Novelty paragraph (§1). Entropy sentences (§2). Three properties stated and
reported separately: loop-bearing (syntactic), pointer confinement (mechanism; measured in BFF, traced in the Z80),
context independence of offspring and byte invariance (measured; the principal outcomes). Theorem 1 quantified over a
partner none of whose bytes already matches; "two or more distinct values" in both the text and the SI (the text still
says three). Theorem 6(iii) concludes "loop-bearing", not "born closed". Theorem 2's actual proviso stated in the main
text. "None can" → "none found up to period 10"; "open for ever" → "open through 16,384 epochs"; Fig. 5d called a regime
map with one unrun cell, or that cell run (BFF published semantics with no-op brackets; 12 soups; local GPU is enough).
"Costs nothing to execute" → "performs no state change"; "never reads its surroundings" → "does not test the partner
before writing". "Forced by arithmetic" → the conditional statement. Title: keep "Heredity before individuality" only if
individuality is defined operationally in the first paragraph and the dissociations are presented as the result;
otherwise the reviewers' "From self-writing to robust heredity" is the honest one.

**Analyses on existing data (days, no GPU).** Report `p_modal`, `n_var_pos`, `sim_mean` beside H (§2). Definition grid
on the culture-test thresholds and the t_rep share rule (planned Stage J). World-clustered uncertainty for the AUC
comparison; a defined estimand for censored delay ratios. Lineage by proximity in the seed-2002 dense snapshots (every 5
steps to 3,000): what the first tiled pusher's neighbourhood was, and what the first `c0`-bearing tape was.

**New experiments, all local GPU, pre-register first.**
1. *Matched-genotype test of the closer* (the review's first priority; decisive and cheap). At L = 50 the ancestor and
   the successor differ by the inserted `20 f0`: run the partner test, a single-byte mutational-robustness assay and a
   head-to-head invasion (1% seed of each into a soup of the other) for both. This separates context independence from
   mutational robustness, which is Cicala et al.'s explanation of the same succession. Repeat for the L = 16 RET NZ
   design against the pusher (not matched; report as such).
2. *Pointer confinement for every Z80 tape*: one flag in the test shader (PC ≥ L at any fetch), rerun the partner test
   for the 160 first/final tapes. Makes property (b) measured, not traced.
3. *Heritable variation channel* (the optimistic review's "bigger question", and the one that could make the paper).
   For each architecture (pusher, pusher + JR, RET NZ, LDIR), mutate each position to a random byte, run the culture
   test on the mutant, and record whether the mutant still replicates and whether the new byte is transmitted. The
   number of transmissible neutral positions is the genome's capacity for inherited variation. Prediction to register:
   the pusher has none (every byte is the word); closed designs have some. If true, closure is the opening of a channel
   for variation, which is a claim about evolvability, not only about fidelity.
4. *The unrun classification cell* (BFF as published with no-op brackets).

**Conceded, no action beyond wording.** Units are imposed by the harness (fixed tapes, start address, pairing); we study
autonomy within imposed units. Changing zero to a halt changes the language, not only the by-product; the lethal-tar
conclusion is stated as a semantic switch. Finite searches do not prove impossibility.

## 5. Verdict on the review

Right on the novelty sentence, the entropy headline, the three conflated closures, the Theorem 1 quantifier and the
finite-search overclaims; right that descent is unproved at L = 16; wrong, or at least unchecked, in implying that the
seven bits could be one random byte (our table excludes that for every first replicator). Its best idea is the
variation-channel assay. Its proposed abstract is honest but flat; ours should keep the explanation and lose the
universals. The paper is a major revision, not a retreat: every number stands, several sentences do not.

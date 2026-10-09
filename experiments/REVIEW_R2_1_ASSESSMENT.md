# Round-2 review, first report (AI, skeptical + optimistic + synthesis, 2026-10-09): what we verified, what to do

Source: `~/Downloads/paper-review-round2.pdf` (untrusted; read only). It reviews revision 1 (`manuscript/MAIN_nature.md`,
reading copy, and `RESPONSE_TO_REVIEWS.md`). Verdict of all three readers: materially better, still a focused major
revision. A second round-2 review is still running.

## Verified tonight against our code and data

1. **A closed copier exists in the 8080 subset at L = 32 (the reviewer's construction).** `21 20 00 31 40 00 16 10 2b 46
   2b 4e c5 15 c2 08 00 c3 11 00` + 12 arbitrary bytes. In our executor under the `i8080` suppression: exact copy into
   256/256 random partners, pointer confined in 256/256, parent intact, eight serial transfers exact, culture test score
   1.00 and gen2 1.00, in 4 trials with random payloads (`check_8080_closer.py`, `results/review_r2/check_8080_closer.txt`).
   Our sentence "where the instruction set expresses no closer" and M4's reading are **false**: the barrier at L = 32
   is accessibility, not expressibility. The construction also shows a confined copier carrying 12 arbitrary bytes
   exactly: closure does not by itself erase variation; our evolved closers do.
2. **Model 5 states a window "of order 10¹⁹ tape-epochs"**; the true order is 10⁷ (0.6 × 2¹⁷ × 192 ≈ 1.5 × 10⁷; the whole
   run is at most 2¹⁷ × 16,384 ≈ 2.1 × 10⁹). The same paragraph keeps the old lethal-tar medians 46,250/525. Our audit
   covered the main text only; the SI was not audited. Error, ours.
3. **Theorem 1's gloss "copies into every partner, as the culture test requires"** is false: the culture test is finite,
   approximate and thresholded.
4. **The mutational scan measures transmission in first-generation copies only** (`mutscan.py`: mutant byte at the aligned
   position in ≥ 50% of ≥ 75% copies); gen2 tests function, not the allele. "Pushed again by the copy" is unmeasured.
5. **Fig. 5e labels the harmless-bracket BFF cell "born closed"** although one of seven first replicators is open and one
   intermediate (`results/bff_stdnh/FINDINGS.md`, B5-3).
6. **Prior art for copied-but-unexecuted sequence:** Cicala et al. validate an LDIR prefix followed by 28 random bytes
   "reproduced faithfully along with the functional prefix" (checked in the downloaded text). The capability is theirs;
   its emergence and fate in an open-ended soup is ours.

## Accepted without new checks (they follow from the text)

Proposition 4 counts occurrences of a two-byte substring and then speaks of a tiled self-writer present at step 0 (a full
tiling has expected abundance 20,000/256¹⁶); nucleation is not shown, so "counting fixes the order" is not proved.
Theorem 2's boundary: saturating copiers are information-closed but not confined, so they are not "closure" in the
theorem's sense. Σ log₂(1 + 255 f) is a single-mutant variation score, not a channel capacity (epistasis, thresholds,
extrapolation from 32 sampled values). "Inherits nothing", "canalised", "smaller closers" (all tapes have the same length;
we mean a smaller executed core) and "lose to" (modal-tape turnover, not lineage fate) overstate. The invasion endpoint is
post hoc motif enrichment with three inserted jumps, not a one-jump intervention. The robustness comparison with Cicala et
al. is within our assay only. "Two real instruction sets" should be "the Z80 and an 8080-like restriction of it".

## Pushback (little)

None on the facts. On emphasis: the order of events stays an empirical result, robust across 170 worlds and two
machines; we drop the claim that counting proves it, not the result.

## What the reviews converge on

The strongest result is no longer "closure is individuality" but **reproductive regeneration versus transmission**:
selection for reliable reproduction finds closers that regenerate a canonical sequence and erase variation, although
closers that carry arbitrary sequence exist (the 8080 construction; Cicala's LDIR prefix). Both reviewers propose the same
decisive, cheap tests, and the synthesis asks for no more atlas.

## Plan (all local GPU, no Modal; pre-register each before running)

1. **Corrections, no data**: the six verified items above; the accepted wording changes; an audit of the SI and Methods
   (not only the main text); summary "Counting fixes the order" becomes an empirical statement with the copying constraint
   as explanation.
2. **Serial allele retention (the central figure both readers ask for)**: for representative genotypes (pusher at L = 16
   and 50, the RET NZ closer, an LDIR tiling, the L = 50 closer with its non-identity map, the three transient genomes,
   the constructed 8080 closer), introduce single-byte alleles, follow the exact allele through several serial transfers
   into fresh partners with every encounter in the denominator, and partition outcomes into loss of function, erasure,
   and persistent transmission. Deterministic executor; minutes.
3. **Accessibility or selection**: seed the constructed payload-carrying closer at 1% into the 8080 L = 32 pusher worlds,
   and payload-carrying closers into Z80 worlds of evolved regenerative closers (both directions). If they win, evolution
   failed to find them; if they lose, erasure is selected. Uses `invasion_pair.py` with suppression; about an hour.
4. **Validate the invasion marker** against traced confinement of random cells at the invasion time points.
5. **Population-level sampling** of random cells at final snapshots for the variation score and confinement (not only the
   modal tape), for Fig. 4d and the "genomes arise and lose" claim.
6. Statistics: bootstrap intervals across worlds, threshold grid on existing data; the exact gen2 formula in Methods.

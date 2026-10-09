# Response to the second-round reviews: Heredity can precede individuality in a digital primordial soup (revision 2)

*To the reviewers. Thank you for three careful reports. We checked every factual point against our code and data before acting, and where we could check, the reports were right. They led to twenty new tests, each registered before it ran, with predictions and kill criteria in a dated record. Several overturned claims of revision 1, and several of our own registered predictions failed; all are reported. The theory was audited again, and two parts of it were wrong and are withdrawn. The reports also pointed us to the result we now consider the most important: in the commonest closing design, one byte of the genome decides whether a closed lineage erases or transmits its variation, and the environment, not the design, decides which kind prevails.*

## What changed, in brief

1. **Selection at aligned lengths (second report, required 1).** Seeded at 1% into a world filled with the pusher, the evolved return closer at L = 16 and the evolved block-copy tiling at L = 32 each took over in 5 of 5 soups (half the cells within 400–750 and 260–290 steps; 0.64–0.91 of cells at 20,000 steps, every sampled heritable cell confined); the pusher seeded into a closer's world vanished in 5 of 5 (Extended Data Fig. 9d,e). L = 50 is no longer used as evidence of selection or descent.
2. **When closure happens (second report, required 2).** The first closed replicator appears at a Kaplan–Meier median of 22,500 steps at L = 16 with benign tar against 38,000 with lethal tar, where there is no open phase (log-rank P = 0.007). The text says that these data do not decide between descent and displacement. The McNemar test is deleted.
3. **Harness conventions (second report, required 3).** With every register drawn at random before each encounter, the first replicator is a load–push word in 20 of 20 worlds; with a random stack pointer it is open in 18 of 20. Closure does depend on the convention: every evolved closer relies on empty registers. With random registers closure came in 1 of 60 worlds by 300,000 steps and in 8 of 40 by three million (a registered replication with new seeds gave 2 of 20, after the first set's 6 closures had all fallen on odd seeds), always through closers that set their own source and offset, jump over bytes they never run and copy with block-move instructions, transmitters in 6 of the 8. A 13-byte self-initialising closer seeded at 1% into a random-register pusher world displaced the pusher within 650–1,300 steps in 5 of 5. The barrier is accessibility; four bytes suffice for a closer with empty registers and six without.
4. **The copy-offset switch and the copy-period law (new).** In the four-byte block-copy core `XX 5e ed b0` the first byte is both an instruction and the copy offset d. For d < L the closer tiles its first d bytes over its own body and erases every mutation beyond them; for d = L it copies the tape verbatim. Proposition 8 proves that d − 4 sites are transmitted when d divides L, and in the population scans 2,289 of 2,343 such cells carry exactly d − 4 (Fig. 6c).
5. **The environment chooses, and the mode belongs to the parent (new).** A matched regenerator and transmitter are indistinguishable in encounters (to within 0.01 in every component) and drift without mutation; with mutation they converge from either side to an environment-set share, low at L = 32 with benign tar (0.03–0.37 transmitters) and high at L = 16 with lethal tar (0.71–0.92). At one length (L = 32) the tar alone reverses the outcome: 0.03–0.37 transmitters with benign tar against 0.55–0.87 with lethal tar, in all 20 worlds of each. Copied serially in six environments, every parent kept its mode wherever it survived (24 of 24 combinations); the environment decided survival.
6. **The lethality dial (second report, optional 3).** When a fetched zero halts with probability p, the open beginning holds in 10 of 10 worlds up to p = 0.1, in 2 of 10 at p = 0.3 and in none at p = 1, a threshold in the same interval as in BFF; the transmitter share of heritable cells rises with p (medians 0, 0, 0, 0.26, 0.56, 0.78; Spearman ρ = 0.50, P < 10⁻⁴). Closure timing is not monotone in p: a damaged open phase delays closure beyond either extreme. Our registered monotone prediction failed, and Model 5's window reading is now restricted to the contrast of the extremes.
7. **Theory corrected (second report, required 4; third report).** Theorem 1 is now a per-encounter count (at least 128 instructions, 64 changes and 64 head moves, so a confined copier loops); the uniform gloss and part (ii) were wrong and are withdrawn. Proposition 4's comparison of a two-byte word with a four-byte closer was wrong (the pusher works only as four bytes in phase) and the "exponentially likelier" claim is withdrawn; Proposition 3 now gives the census of every word of period two, three and four: 5 open self-writers at period two, 1 closed at period three, and 558 closed against 83 open at period four. Counting therefore does not explain why the open word comes first; its head start is measured. (H2) of Theorem 6 is a hypothesis, not a theorem, for BFF; Avida is outside (H1); the provisos of Theorem 2 fail under lethal tar and in budget arms shorter than the tape, as stated.
8. **Ref. 22's criterion, run (second report, required 5).** Exact self-copy into the zero partner after 1, 4 or 8 mutations scores our pushers and block-copy finals at L = 32 alike (medians 0.000–0.002), and the matched pair 0.88 (transmitter) against 0.000 (regenerator) after one mutation: the criterion measures transmission of mutations, not survival. Our registered prediction (block copiers more robust) failed.
9. **Statistics.** Fisher's exact test for well-mixed pairing (7 of 10 against 20 of 20, P = 0.03), a Cochran–Armitage trend test for closure across aligned lengths (P = 6 × 10⁻⁵), the log-rank test above, and permutation tests for the dial.

## Response to the first report

**Serial transmission.** Done with the five-byte window rule, every lineage in the denominator, the three-way partition you proposed and a chance background from unmutated lineages (Methods, Serial allele retention). Our registered prediction that the pusher's sites persist failed and is now stated in the main text as well as Methods.

**"Capacity".** Renamed the single-mutant variation score, defined as a summary of single mutants that are heritable and reach first-generation copies, explicitly not a channel capacity. "Inherits nothing", "canalised" and "smaller closers" are gone.

**Competition and the marker.** The marker was checked against traced confinement and failed; the L = 50 invasion is reported as the spread of the jump, with the confined share rising to about 0.2 and levelling off. The selection claim now rests on the aligned-length invasions.

**Robustness and Cicala et al.** Ref. 22's criterion was run on our tapes (above). Cicala et al. are credited for showing that a block-copy prefix carries 28 arbitrary bytes; the soups add which design arises, where, and what becomes of it, and the copy-offset switch explains why their criterion favours their genome-carrying seed.

**"Genomes lose to smaller closers".** Replaced by the matched-pair experiment: neither design is fitter in encounters, and the environment sets the outcome.

**The 8080 subset.** Called an 8080-like restriction of the same emulator throughout. Your counterexample runs in our executor and takes over the subset's pusher worlds when seeded; the text says that no closer emerged, not that none exists.

**Proposition 4 and "counting fixes the order".** Withdrawn (above). The order of events is reported as observed; the counts bound it without forcing it.

**Theorem 1.** Restated a second time, as a per-encounter count. Your step was not needed, but the round-1 replacement had two errors of its own: brackets test the byte under h0 and `,` copies partner bytes into the organism, so a confined program's control flow can depend on the partner (the uniform gloss fails), and bytes written into the partner ahead of the pointer are executed when it arrives (part (ii) fails). Both are withdrawn and the withdrawal is stated in the Supplementary Information.

**The BFF map and the closed-first rule.** Fig. 5e reads "born loop-bearing" with loop, pointer and inflow counts; Lemma 7 now counts 7 of the 22 first replicators whose pointer never entered the partner as carrying inflow.

**Theorem 2's boundary.** Rewritten as Lemma 7: information closure without pointer closure, with the BFF literal that writes two or three copies as the example.

**Model 5.** The BFF window is now computed from the recorded shares (8.2–8.7 × 10⁶ tape-epochs under a wrapping pointer, 2.5–4.1 × 10⁵ without); the contrast with "order 10⁵ steps from a random start" is removed because it confounded population size with clonality; the benign-against-lethal closure timing is added as the model's one tested prediction.

**Statistics.** Methods give the exact gen2 equation; Extended Data Table 3 gives bootstrap intervals and the threshold grid (the load–push word first in 170 of 170 worlds for heredity thresholds 0.2–0.4).

**Executed against operand; final dominant; code and data.** As in our previous letter. Access to the repository for the reviewers is arranged through the editor. [The authors' decision.]

## Response to the second report

**Required 1, selection at aligned lengths.** Done (above). L = 50 is described as a stable mixture in which the jump spreads and confinement does not, and it no longer appears in the summary or the Discussion as evidence of selection. The variation result leads with serial retention across lengths.

**Required 2, closure from the open phase.** Done with Kaplan–Meier curves and a log-rank test for every stage (Methods, First closure). Benign closes earlier than lethal at L = 16, as Model 5 reads it, but the text says "displace" wherever descent would be implied, and "Later a return evolves" and "Evolution's first act" are gone. The McNemar test is deleted.

**Required 3, conventions.** Done: random registers and random stack pointer, 20 worlds each, plus 20 random-register worlds to three million steps and 20 more as a replication, plus invasions of a self-initialising closer. "Need nothing but the instruction set" is replaced by a statement that both closing designs rely on registers the harness empties, and "every setting we tried" is replaced by the settings we tested. The conventions are credited to ref. 16, whose tapes were all at an aligned length.

**Required 4, theory.** Done (above). Fig. 6b is redrawn as the census of self-writers by period, and the head start is stated as measured.

**Required 5, scope.** Stage M is the Z80 restricted to 8080 opcodes, an ablation, throughout; the impossibility claim and the ref. 16 explanation are gone. Ref. 22's test was run.

**Required 6, the variation result and the BFF map.** The failed predictions are in the main text; "canalised", "inherits nothing" and the von Neumann comparison are replaced (the bytes are copied without being interpreted, half of what von Neumann's description does); alignment is defined by the five-byte window; Fig. 5e is relabelled.

**Errors and inconsistencies 1–17.** All corrected: Fig. 3f gives L = 16 medians and cites Extended Data Fig. 4; the zero-byte range cites Stage B and the emergence medians cite Methods; the L = 20 count is reconciled (five closers carry a relative jump, and Extended Data Table 1 classes one of them, seed 2003, by its block repeat, as Methods now say); Model 5 and Lemma 7 use the Kaplan–Meier medians and per-length inflow (3.9, 6.2, 7.8 and 8.0 bits); the BFF recovery is 5 of 6 and the dial range 0 < p ≤ 0.1; "closure is a return" names its three forms; closure is defined by the pointer; the tests are reported; the stage counts are corrected; the census and Extended Data Fig. 2 are explained; "no other family matters" is replaced by the measured delays (at most 6.5-fold, no sign test below P = 0.18); the padding legend marks the aligned rings; L = 81 and 100 are reported (a load–push word first in 20 of 20 and 18 of 20 worlds, not traced); Stage I is in the registered list and "heritable material" is defined; Avida and lethal tar are outside the theorems' hypotheses; ref. 23 is Knierim, C.

**Optional items.** We ran item 3, the lethality dial in the Z80 (above): the open beginning ends at a threshold between p = 0.1 and 0.3, transmitters rise with p, and closure is slowest at intermediate lethality, which refutes the monotone reading of Model 5. Item 2 is answered by the serial-retention partition and the population classes. We did not run lineage tags (item 1), a real 8080 (item 4) or a functional payload (item 5).

## Response to the third report

**Audit the heredity assay.** Methods give the gen2 estimator exactly. A new table (Supplementary, `results/review_r2/LINEAGES.md`) shows parent, offspring and granddaughter byte sequences for each kind of replicator: the pusher's offspring are damaged and vary with the partner; the regenerators' offspring are exact; a regenerator with a mutation outside its copied bytes produces exact unmutated offspring, so it is heritable without transmitting; the transmitters' offspring are exact copies including their unexecuted bytes. Serial allele retention separates self-replication from the production of a different replicator: a lineage counts as alive only if every transfer makes a ≥ 75% copy of the current tape, and an allele counts only if it is carried. The sensitivity of the first-replicator dating to the heredity threshold is in Extended Data Table 3.

**Consistent closure definitions.** Closure is defined by the pointer only; population classes are open, regenerator (confined, at most two transmissible sites), intermediate (three or four) and transmitter (at least five), used in Fig. 4 and Extended Data Fig. 9.

**Withdraw the counting proof.** Done (above); Theorem 1(ii) and the exponential claim of Proposition 4 are withdrawn, and the census replaces them.

**Rename the variation measure.** Done.

**Parent × environment serial transfer (your top priority).** Done and registered as PE: six parents at L = 16 (the pusher, two regenerators, three transmitters including the self-initialising closer) in six environments (benign or lethal rule; random partners or partners from open, closed and lethal soups; random registers). Wherever a parent's lineages survived, the regenerators carried no transmissible site and the transmitters 9–12 (24 of 24 combinations); the environment decided survival (the return closer dies under lethal tar, every zero-register closer under random registers). The pusher's lineages were lost everywhere except among partners from a closed world, where 0.36 survived without transmitting; our prediction that they would be lost everywhere failed.

**Aligned-length competition.** Done (above).

**Bound the prose and the title.** The title is now "Heredity can precede individuality in a digital primordial soup". "A theorem rather than an observation", "counting fixes the order", "the dichotomy is arithmetic" and "exponentially rarer" are gone; the Discussion lists the conventions among the limits.

## What we did not do

- Lineage tags to settle descent against displacement at L = 16 and 32.
- A real 8080 with its alternate opcodes, and a functional payload for the unexecuted bytes.
- The full mechanism by which the environment chooses between regenerators and transmitters: our registered explanation by isolated mutational flows failed. Exploratory censuses found one component, that a body of copies of the copying code is hijacked by intruding executors while a body of junk protects, but it favours transmitters under both tars and does not by itself explain the benign-tar equilibrium.

## Errors we found ourselves

- **Theorem 1, a second time.** The round-1 replacement contained the two errors above; an audit of the Supplementary Information found them.
- **Proposition 4.** The two-byte against four-byte comparison was wrong.
- **The L = 50 lineage.** It stops copying at the fourth transfer, not the third.
- **The copy-offset switch, first analysis.** Our first reading of the switch experiment counted core loss in 3 of the lethal-tar worlds; it was 7 of 20, and the text now gives fixed-time shares in worlds that kept the core.
- **A confound in our own switch experiment.** The first switch analysis compared L = 32 with benign tar against L = 16 with lethal tar, confounding tar with length; an independent audit flagged it and the matched pair at L = 32 under lethal tar removed it.
- **The commonest closing core.** A draft called `XX 5e ed b0` the commonest closing core; it is carried by 2,347 of 15,655 confined heritable cells sampled, fewer than the return closers, and the text now says that it recurs across stages.
- **Registered metrics that failed on definition.** The motif shares of the constructed closers and of the switch cores (the motifs mutate while their descendants hold the soup). Functional classes of random cells are reported instead, labelled post hoc.

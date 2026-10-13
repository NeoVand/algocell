# Full pre-submission review of manuscript v8

## Overall verdict

**Not ready.** The central result—a conditional, mechanism-specific account of which parental differences survive copying—is supported by the checked records, with appropriately bounded registered and exploratory evidence, but one relevant failed prediction is omitted and several figure labels overstate the evidence. Submission should follow the reporting and presentation corrections below, with the contribution framed around variant transmission and inherited copying geometry rather than discovery of the copier architectures or demonstration of a Darwinian transition.

## BLOCKERS

**None established that invalidate the central scientific result or require new simulations before a suitably scoped submission.** “Not ready” here means that the major reporting and interpretation issues require correction, not that the paper needs another open-ended experimental programme.

Review scope: current `MAIN_v8.md`, `SI_v8.md`, the five supplied v8 figures, registration and relevant source records at HEAD `7cab316`. Paths below are relative to `experiments/` unless explicitly absolute; image findings cite their corresponding figure-building lines. No simulations or network access were used. Author-supplied declarations/accession placeholders and the acknowledged unfinished ED artwork are not counted as defects; the ED assessment concerns the SI map only. Durable data/code access still has to be supplied at submission, as the manuscript itself acknowledges.

The abstract and main-text claim audit did **not** establish an incorrect headline empirical count. The checked records support the branching control survival of 48/64 and 0/512 qualifying pusher mutation types; the 2,347/15,655 recurring-core count and 2,342/2,343 unexecuted-score agreement; 101 Z80 settings from 22 parents with 1,067,608 direct cases; 42 eligible 6502 roles representing 41 tapes from 11 worlds; and the reported competition medians and undefined-ratio exclusion. Sources checked include `results/branching/SUPP_TABLE.md:3–9`, `results/offset/CORE_FAMILIES.md:3–15`, the cited crystal CSVs, `results/capacity/CAPACITY.md:5–26`, `results/evolved6502/RETENTION.md:4–41`, and the competition/T2 tables. The 296 constructed 6502 cases remain distinct from exact capacities. The overlap proof's ordered-copy induction is sound under its stated conditions; finite joint assays do not establish universal payload robustness, and the manuscript generally says so.

The important negative statuses also mostly survive inspection: X1-2 failed; S1 P1 is unsupported despite 20 confined beginnings because only 14 meet its loop requirement; S2 has 14 detections, two open and 12 confined, failing R1 and triggering the route kill; registered L1/L2 remain untested, with L2 support suspended and 724 unresolved certificates, including 708 at `POP ES`. The constructed 8086 controls and the exploratory 32-genotype, 2,224,640-encounter panel are not substituted for those empty registered assays. These statements match `results/s6502/CORRECTION.md:15–41`, `results/s8086/S2.md:11–44`, the law-denominator tables and the exploratory assay tables. The principal exception to this otherwise careful status discipline is M1 below.

## MAJOR

### M1. Report the failed mutation-on abundance prediction alongside the environmental success

**Location:** `manuscript/MAIN_v8.md:73–79`; `manuscript/SI_v8.md:171`, `228`, `304`.

**Problem:** The environmental account explicitly names the failed E1 prediction and successful L32 lethal endpoint, but omits E2, the registered abundance prediction for the mutation-on competitions being discussed. A reader should not have to inspect an external verdict ledger to discover that this prediction failed historically and remains failed with the corrected core. The clean L32 contrast also does not describe the L16 benign ecology: there the classified core population largely disappears.

**Evidence checked:** `REVISION_PREREG.md:399–403` states E2; `REVISION_PREREG.md:434–443` records its historical failure and the L16 classifier limitation. `results/tier2/T2_COMPONENTS.csv:12–14` records corrected-core E2 failures of 2/10 L16 benign, 0/10 L16 lethal and 5/10 L32 benign, against at least 7/10 required. `results/tier2/scored/E/REPORT.md:10` gives L16 benign median T share 0.70 but only 0.05 classified core abundance. The L32 medians 0.05/0.76 and its registered lethal success are correct.

**Exact insertion after the E1 discussion at MAIN:73:**

> The registered mutation-on prediction E2 also failed: the required fall into T shares of 0.05–0.4 occurred in only 2/10 L16 benign, 0/10 L16 lethal and 5/10 L32 benign worlds, against at least 7/10 required. At L16 under benign chemistry, the census-defined core classes fell to a median 5% of all cells. We therefore distinguish the L32 environmental contrast from the failed abundance prediction and the different L16 outcome.

Add the same E2 verdict to the SI competition/validation account. At SI:304, replace **“failed E1 and explanatory predictions”** with **“failed E1 and E2 abundance predictions, rejected explanatory predictions”**. Retain the important MAIN:175 qualification that the competitors differ in payload background as well as displacement: these experiments do not isolate selection on capacity itself.

### M2. Put the precise advance over refs 13–14 at the front of the paper

**Location:** `manuscript/MAIN_v8.md:11`, `19–23`, `53`, `69`, `99–105`.

**Problem:** The paper has a defensible advance, but the abstract gives prominent space to introducing a substrate and to Woese while leaving the new measurement and conditional result comparatively implicit. The main text does credit the prior architectures, so this is not an accusation of uncredited discovery. For an editor, however, the distinction from an already published four-byte copier with 28 faithfully copied random bytes and mutation–replication robustness tests must be immediate. The broadly important proposition that replication alone is insufficient for evolvability is established background; the new contribution is its interventional resolution in these particular spontaneous copiers, including changes in the set of transmitted alternatives.

**Evidence checked:** Ref. 13's locally cached primary text describes the `01 C5` 8080 tile and subsequent block-copy architectures. Ref. 14, Methods §4.5, explicitly specifies `1E 20 ED B0` plus 28 random bytes and repeated mutation–replication challenges; its §4.4 describes task-free controls. The local files inspected were `/private/tmp/claude-501/-Users-neo-repos-algocell/26fc62cd-9c63-4d42-8467-b9398dac54bd/scratchpad/isa_research/dl_aguera/cl.txt:1094–1106` and `/private/tmp/claude-501/-Users-neo-repos-algocell/26fc62cd-9c63-4d42-8467-b9398dac54bd/scratchpad/cicala_v2.txt:491–503`. In the latter assay the next cycle carries forward the executed first half, and final success compares the two resulting halves; that is materially different from tracing recovery of an introduced parental marker. Current MAIN:51–65 and SI Notes 1–2 supply the added conditional capacity/provenance result. No broad claim to priority over the entire digital-evolution literature has been independently established by this local review.

**Exact replacement for the transition from prior work to the question at MAIN:19–21:**

> Earlier work already demonstrated load–push and block-copy architectures, faithfully copied random tails, and robustness to repeated mutation–replication cycles [13, 14]. Here we ask which introduced parental alternatives remain recoverable through successive descendants, and whether an inherited change in the copying operation changes that set. We track introduced differences separately from lineage survival and derive a conditional count of independently variable positions for a specified family of overlapping copiers.

Keep the existing architectural credit at MAIN:53. Keep the bounded Woese comparison in the Discussion, but remove it from the abstract; no adaptive use of payload, unlimited heredity or historical Darwinian transition is demonstrated. The essential experiments for this scoped paper are already present. Demonstrating selection for useful inherited payload would be an additional scientific advance, not a wording repair that can be demanded as though the present text claimed it.

**Exact replacement abstract, 110 whitespace-delimited words:**

> Reliable reproduction need not preserve the differences on which selection can act. In machine-code populations, a representative Z80 copier sustained lineages while none of 512 tested variants met the registered carriage criterion. For a recurring class of overlapping copiers, we derive a conditional count of freely transmissible byte positions controlled by an inherited displacement byte. Registered Z80 interventions and eligible evolved 6502 copiers matched this prediction. In 8086 populations, the predicted emergence route failed and registered copying-law tests remained untested; separate exploratory interventions agreed with the rule. Environmental changes shifted copier frequencies without establishing selection for payload capacity. The results distinguish independence from partner contents from transmission of parental variation.

For general-reader flow, let Figs 2–3 carry the argument before expanding the architecture comparison. The full validation accounting can move from Fig. 1d to ED1; it establishes reliability but interrupts the explanation of the phenomenon. The equation is a conditional property of the specified copying family, not a universal new law of heredity.

### M3. Figure 1a visually asserts an evolutionary transition that the legend disclaims

**Location:** `manuscript/figures/out/v8/fig1.png`, panel a; `manuscript/figures/build_v8.py:164`; `manuscript/MAIN_v8.md:113`; `manuscript/SI_v8.md:177`.

**Problem:** “Writing code; evolving a return”, followed by numbered stages ending in “comes back”, reads as a pusher acquiring the depicted return. The legend expressly says that the examples are not a documented ancestor–descendant pair. A caption qualification cannot repair the causal story told by the picture itself.

**Evidence checked:** The supplied figure, the title in `build_v8.py:164`, and the explicit non-ancestry statements in MAIN:113 and SI:177. The figure is a mechanistic comparison; the current claim does not supply a genealogy connecting these two examples.

**Exact display replacements:** Change the panel title to **“Literal self-writing and return-based copying”**. Label the lower drawing **“Separate return-containing copier”**. Replace the numbered **“3 comes back”** annotation with unnumbered **“Return redirects execution”**, and visibly separate the two examples. Keep the legend's non-ancestry sentence. This needs a schematic correction, not a new simulation.

### M4. Figure 1d reports criterion satisfaction as literal identity

**Location:** `manuscript/figures/out/v8/fig1.png`, panel d; `manuscript/figures/build_v8.py:197–206`; `manuscript/MAIN_v8.md:113`; `manuscript/SI_v8.md:222`.

**Problem:** The grey key “Same”, column “Same / tested” and row “Ring memory stable” make 643/690 look like exact agreement. This row instead counts conditions whose upper confidence bound falls below a changed-memory rate threshold. Some of those conditions contain observed changes. The numbers are right; the displayed proposition is wrong.

**Evidence checked:** `results/impact/decisions.csv:2` classifies `random/P32_L16`, core C1, as unaffected despite two changed pairs among 100,000. Direct aggregation of decision-scope C1 rows gives 643 unaffected conditions, **78 of which have nonzero observed changes**, totalling 132 changed-pair events. SI:222 correctly defines the confidence-bound criterion. The other rows have different units and decision rules, as MAIN:113 already explains.

**Exact display replacements:** **“Same” → “Meets row criterion”**; **“Same / tested” → “Meets criterion / tested”**; **“Ring memory stable” → “Below change-rate bound”**; **“Diff.” → “Flagged / different”**. Add to the legend:

> For the ring comparison, meeting the criterion permits a small nonzero observed change rate; it does not denote identical memory in every encounter.

Keep all five denominators separate. Do not replace this panel with a pooled “accuracy” percentage.

## MINOR

### N1. Figure 2c turns failure of a site threshold into apparent marker absence

**Location:** `manuscript/figures/build_v8.py:280`, `292–299`; `manuscript/MAIN_v8.md:117`; Fig. 2c.

**Problem and evidence:** “Transfer 1, absent at 8” labels hollow circles. The code actually selects sites that score at generation one and fail the generation-eight site criterion, using the `transmissible` field. That does not establish absence of every marker copy. The main legend correctly describes scoring, and Fig. 2a deliberately retains minority carriers; the graphical key should use the same distinction.

**Exact replacement:** **“Scored at 1; below criterion at 8”**. Replace **“Marker site at 8”** with **“Scored at transfer 8”** for the filled symbol. There is no demonstrated ≥/strict-inequality error: `build_v8.py:260` and the serial/branching scorers use ≥0.5.

### N2. Figure 2a uses “closed” where the stated classification is confinement

**Location:** `manuscript/figures/build_v8.py:234–235`; `manuscript/MAIN_v8.md:43`; `manuscript/SI_v8.md:5`; Fig. 2a.

**Problem and evidence:** The group headings “Closed regenerators” and “Closed transmitter” conflict with the manuscript's deliberate separation of fetch confinement from independence of original partner contents. MAIN:43 calls these examples confined, and SI:5 defines closure as a different property. Even sampled zero inflow would not establish partner-universal closure.

**Exact replacements:** **“Confined regenerators”** and **“Confined transmitter”**. Preserve “closure” for the separately defined dependence property.

### N3. Figure 5d needs to identify its full-address classification explicitly

**Location:** `manuscript/figures/build_v8.py:652`, `666`; `manuscript/MAIN_v8.md:93`, `129`, `215`; `manuscript/SI_v8.md:256–258`; Fig. 5d.

**Problem and evidence:** Eight examples are labelled “One-pass copy” under “Mechanism”, while the post hoc analysis says all ten execute 64 byte copies at displacement four and repeatedly traverse mapped ring positions. The frozen labels are legitimate classifications based on repeated full instruction-start addresses, but they are not descriptions of different physical overlap geometries. Readers should not need the SI to resolve this apparent contradiction.

**Exact display replacements:** Header **“Frozen full-address class (16/16 encounters)”**; entries **“One-pass class”** and **“Loop class”**. Add to the legend:

> These classes use full instruction-start addresses. Post hoc traces show repeated copying of mapped ring positions at d = 4 in all ten tapes.

Do not change the frozen classifications or their registered status.

### N4. Figure 3d overstates the status of its sampled eligibility screen

**Location:** `manuscript/figures/build_v8.py:452–472`; `manuscript/MAIN_v8.md:67`, `121`; Fig. 3d.

**Problem and evidence:** The display says “in law domain; shade = capacity” and “outside domain”, whereas the prose correctly describes sampled eligibility checks and predicted payload. `prop8_eligible` selects the plotted category; the finite screen is not an independent universal certificate of arbitrary-value or serial robustness. Failure of this screen should not be read as a general impossibility result either.

**Exact replacements:** **“88 working, outside domain” → “88 working, outside sampled eligibility”**; **“11 in law domain; shade = capacity” → “11 pass sampled screen; shade = predicted payload”**; **“Law domain:” → “Sampled screen:”**. Keep the counts 157, 88 and 11.

### N5. State the overlap conditions before the introductory tiling conclusion

**Location:** `manuscript/MAIN_v8.md:49–51`; `manuscript/SI_v8.md:17–33`.

**Problem and evidence:** MAIN:49 initially sounds like a statement about every fixed-displacement copy around a ring. The later theorem correctly requires divisibility, coverage and noninterference. Without these, the claim that the first d bytes tile the whole ring in the required offspring phase need not follow. This is an exposition defect, not a refutation of the qualified theorem.

**Exact replacement for the opening four sentences of MAIN:49:**

> For the most tractable family of evolved copiers, the answer is geometric (Fig. 3a). Consider a forward copy beginning at the parental tape's first byte, with displacement d dividing tape length L and enough ordered writes to cover the ring. Each write is d bytes ahead of its read, without interfering writes. The first d reads obtain original parental bytes; later reads recover earlier writes, propagating that protected prefix around the ring.

Retain the small-versus-full displacement comparison, changing **“reproduces it entire”** to **“reproduces the entire parental tape”**.

### N6. Cite the mathematical result on which the literal-channel proof relies

**Location:** `manuscript/MAIN_v8.md:207`, `297–325`; `manuscript/SI_v8.md:81–91`, `105`.

**Problem and evidence:** The proof invokes the Fine–Wilf period lemma, but none of the fourteen bibliography entries identifies its source. This is a direct dependency of a named theorem, rather than a request for a general literature survey. The supplied SI proof and finite-bound counterexamples otherwise make its use inspectable.

**Exact text replacement:** **“Fine–Wilf requires” → “The Fine–Wilf periodicity theorem [15] requires”**, with the corresponding SI citation. Add the primary reference, verifying final bibliographic formatting before submission:

> 15. Fine, N. J. & Wilf, H. S. Uniqueness theorems for periodic functions. *Proc. Am. Math. Soc.* **16**, 109–114 (1965).

### N7. Improve legibility at publication size rather than adding more caption qualifications

**Location:** `manuscript/figures/build_v8.py:173`, `251`, `289–301`, `464–471`, `646–670`; Figs 1–5; legends at `manuscript/MAIN_v8.md:113–129`.

**Problem and evidence:** The supplied images are readable when enlarged, but key text is generated at 5–5.7 pt, including Fig. 1b's legend and the dense Fig. 2c/3d keys. Figure 4 combines trajectories, endpoint distributions and a numerical table; Figure 5 places a ten-row hexadecimal catalogue alongside the main comparison. These are demanding even for a specialist and obscure the simpler scientific sequence.

**Required production change:** Target at least 7 pt for substantive labels at the intended final width, simplify keys, and move secondary catalogues or validation detail into the mapped ED. Preserve the denominator/status labels when doing so. This is a layout change, so no replacement prose is required. Check the eventual PDFs at their actual submission width; this review does not certify the unfinished ED artwork.

## COPY

### C1. Remove the numerical-audit appendix from the scientific Methods

**Location:** `manuscript/MAIN_v8.md:251–261`; especially `253`; `manuscript/SI_v8.md:179`.

**Problem and evidence:** The “Numerical derivations” section retains a named earlier audit and explains elementary arithmetic and display bookkeeping such as 16/4 tile repeats, the number of curve keys and three bubble-size exemplars. This reads as a response to a review/build process, not a reproducible scientific protocol. The thresholds, equations and units already appear in the actual Methods and SI Note 5. Removing review-only source brackets will not remove this prose.

**Exact replacement for the entire MAIN section:**

> ### Source data and figure provenance
>
> Generated source tables provide the figure coordinates and empirical counts. Supplementary Note 5 defines the cohorts, measurement units, exclusions and display selections; Supplementary Table 15 identifies the figure source data and provenance manifests.

Move any remaining useful arithmetic checks to the release's provenance documentation. Remove the prose reference to `AUDIT_v7_RESOLUTION.md` from the submission-facing SI as well; a versioned provenance index can remain in the archive.

### C2. Make the SI independent of older manuscript versions and proposition numbering

**Location:** `manuscript/SI_v8.md:21`, `23`, `31`, `73`, `75`, `77`.

**Problem and evidence:** The current SI repeatedly points to `MAIN_nature.md` and calls the specialization “Proposition 8”, although the current supplement presents its own proof and does not number a Proposition 8. A referee should not need a previous manuscript to know which theorem is being asserted. This differs from a legitimate citation to an archived dataset or frozen protocol.

**Exact replacements:** Change **“Z80 specialization of Proposition 8”** to **“Z80 specialization of the ordered overlap theorem”**. Replace the earlier-manuscript proof/source citations at SI:21 and :31 with **“Supplementary Note 1, ordered overlap theorem and Z80 specialization”**. At SI:73, :75 and :77 remove the obsolete theorem-number pointers and retain the complete local arguments and their explicit hypotheses. If historical corrections must be indexed, use the neutral sentence:

> Historical statements and their corrections are retained in the protocol archive; the statements used here are those given in this Supplementary Note.

The review-only preamble at MAIN:7 should also be removed in the submission export, as already intended, while preserving any definitions it currently supplies elsewhere.

### C3. Move billing and dispatch minutiae out of the scientific narrative

**Location:** `manuscript/MAIN_v8.md:197`, `203`, `229–237`; `manuscript/SI_v8.md:203–216`, `272`.

**Problem and evidence:** “No paid compute”, metered charges, spending caps, provider-stopped applications, noninteractive commands and ledger-lock repairs are operational provenance. They are not biological results or essential Methods. Scientific amendments, nondeterministic restarts, exclusions, source identities and the timing of analysis decisions do matter and should remain auditable. The current SI includes both categories without prioritizing them.

**Exact deletions:** Remove **“Displacement experiment C1 used local computation with no paid compute”** at MAIN:197 and **“The retention experiment C2 used no paid compute”** at MAIN:203. At MAIN:237 replace the last sentence with:

> No new population worlds were run, and the registered verifier and verdicts were unchanged.

**Exact replacement for the billing paragraph at SI:214:**

> Source, world/checkpoint identities and constructed-control hashes were verified against the archived manifests; unchanged reporters generated the final verdicts. Failed attempts, restarts and operational amendments remain documented in the provenance archive.

Keep the scientifically relevant restart explanation at SI:212, including why the first attempt was excluded and that no trajectory was spliced. Move the provider, billing and command-level detail to the archive, not into oblivion.

### C4. Reduce the overloaded C1/C2 nomenclature in reader-facing prose

**Location:** `manuscript/MAIN_v8.md:7`, `31`, `63`, `65`, `195–203`, `243`; `manuscript/SI_v8.md:5`.

**Problem and evidence:** C1 and C2 each name both an emulator version and a registered study. The manuscript discloses this, but repeats explanations and requires readers to keep separate naming systems active. It is internally resolvable but unnecessarily hard to read, particularly next to Fig. 1's correction comparison and Fig. 3's intervention evidence.

**Exact naming rule:** On first use write **“corrected Z80 core (core C1)”**, **“Z80 displacement study (registered experiment C1)”** and **“6502 retention study (registered experiment C2)”**. Thereafter use **“corrected core”**, **“displacement study”** and **“6502 retention study”** in prose and reader-facing figure keys. Preserve original IDs in registration/verdict tables and source-data fields so the audit trail remains exact.

## Venue table

The local manuscript has approximately **202 abstract words, 3,839 main-text words, 1,234 legend words and 4,695 Methods words**, counting whitespace-delimited tokens after removing bracketed citations/source pointers and including section headings; it has **five main figures, fourteen references and nine planned ED items**. These counts are reproducible local facts, not the journal's submission-system count. The formatting numbers below are conservative planning targets based on established article formats, **not verified October 2026 policy**: current author instructions were not accessed because network use was prohibited. Do not treat a planning target as a certified hard limit, particularly for Science Advances or Artificial Life; the final selected article type's current instructions need checking before submission.

| No. | Venue and fit | Required revision / format plan | Recommendation when time to publication is the priority |
|---|---|---|---|
| V1 | **Nature:** low probability in the present scientific scope. The conditional inheritance result is interesting, but architectures and faithful tails are prior art, adaptive use is unshown, and the cross-machine route prediction fails. | Use a summary of ≤200 words; target a substantially leaner main narrative, approximately 3,000 words, with article-specific allocation checked. Five figures fit the familiar 5–6-display scale, but reduce density. Fourteen references are below the usual roughly 50-reference scale; add direct dependencies rather than padding. Keep detailed Methods separate and use ED/SI for validation/protocol accounting. M1–M4 are necessary but do not create the missing breadth of advance. | **Do not start here.** A prestige-first submission cascade adds desk-rejection risk without resolving the scope issue. |
| V2 | **Science:** low probability as a Research Article; especially poor as a short Report. Generality beyond a restricted copying family would need to be much more compelling. | For a Research Article, plan ≤125 abstract words, roughly ≤4,500 main-text words, up to about six displays and about fifty main references; verify the current category. The current main-text count fits that planning scale, but the abstract does not and the narrative needs sharpening. Put detailed Materials and Methods in the supplementary format required by the selected category. A Report would require much greater compression, ordinarily around 2,500 words/four displays. | **Do not start here.** Formatting compliance would not fix the editorial significance gap. |
| V3 | **Science Advances:** plausible only with a sharply focused general-interest argument and rigorous reporting cleanup; still a substantial editorial gamble. | Use the proposed ≤125-word abstract as a conservative target and verify whether the current format permits more. Five displays and fourteen references are modest for the customary approximately six-display/eighty-reference scale. The approximately 8,534 main-plus-Methods words should be reorganized and checked against the current length allowance; do not silently count Methods as exempt. Use Introduction, Results, Discussion and Materials and Methods, with detailed proofs/provenance in supplementary files. | **Possible, but not the fastest expected route.** The venue would likely ask why the conditional capacity result changes understanding beyond the cited digital-soup work. |
| V4 | **Nature Communications:** the strongest broad multidisciplinary candidate, provided the paper presents a substantial advance for digital evolution rather than a general theory of life's transition. | Target ≤150 abstract words and approximately ≤5,000 main-text words, with separate Methods. Five main figures and nine ED items are within the familiar scale of up to ten each, subject to current instructions and whether ED is appropriate for the selected format. Fourteen references are below the usual roughly seventy-reference guidance. Complete the ED package, correct the graphical claims and foreground the interventional contribution. | **Reasonable higher-risk alternative to Artificial Life.** Choose it only if broader visibility justifies a likely longer or unsuccessful first round. |
| V5 | **Artificial Life:** strongest disciplinary fit. Machine semantics, conditional heredity, negative architecture predictions and explicit constructions are directly relevant to its audience. | Retain the theorem and important machine-contract detail while removing build/billing history. The shortened abstract is safe as a drafting target; confirm the current abstract/page requirements rather than assuming a universal hard cap. Five figures and the present reference count are not intrinsically excessive, but expand directly relevant attribution and improve print legibility. Integrate Methods/technical sections in the article as the journal permits and supply appendices or supplementary data instead of assuming a Nature-style ED package. | **Recommended first submission after the listed fixes.** This is an assessment of fit and avoidable rejection risk, not a claim that its editorial or production turnaround is demonstrably shortest. |


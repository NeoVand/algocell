# Changes from MAIN_v7 to MAIN_v8

This revision applies the manuscript review to `MAIN_v8.md`, assembles `SI_v8.md` from current sources, supplies the numbered bibliography, and resolves the original mechanical-audit flags in `AUDIT_v7_RESOLUTION.md`. The v7 manuscript, review and audit remain unchanged. No simulation was run and no commit was made. Author declarations and release accessions remain explicit placeholders by the author’s instruction. Rendered-artwork work belongs to the parallel figure job; this document separates implemented text/caption changes from its remaining requirements.

## Review dispositions

| Review ID | Disposition in the v8 review package |
|---|---|
| B1 | Fourteen verified references, numbered in first-citation order; all author–year brackets converted. The eight original works remain cited; the six historical additions are included. Result-path brackets remain in this review copy, with an explicit note that they move to a Source Data/provenance index for submission. |
| B2 | Data/code accession, verified archive access, release-wide manifest and encounter benchmark package/table remain explicit author-supplied release placeholders. Availability distinguishes committed reports from raw records stored locally or externally; no public release or inaccessible-input verification is invented. |
| B3 | `SI_v8.md` replaces the prospective supplement plan with nine actual Supplementary Notes, numbered tables/source mappings, proofs and an Extended Data rendering inventory. `SI.md` was not reused as the scientific authority. Main theorem citations now point to the new SI. Rendering status is separate from existence of completed analyses. |
| B4 | One title, “What spontaneous replicators pass on”; author block, contributions, funding/acknowledgements and competing interests are explicit placeholders. AI use is also flagged at the top and drafted factually in Methods for author confirmation. |
| M1 | Heading “Carriage can fail despite survival”; minority carriage explicitly distinguished from complete erasure. Figure 2 caption states the <50%/≥50% partition. Ancestry erasure remains confined to the traced example. Corresponding artwork labels are specified below. |
| M2 | Methods defines ≥75% best-cyclic-shift offspring qualification against the immediate parent, uninterrupted serial survival, replacement of parents by qualifying offspring, uniform cap at 32 and nonempty branching survival. |
| M3 | Competition partial-copy R/full-tape T classes are distinguished from ≤2/≥5 site-score classes at first use and in the caption. Methods gives the exact L32 hexadecimal membership sets; Results uses “full-tape-class shares, T/(R+T)”. |
| M4 | Figure 2 caption distinguishes physical read address 4 from ancestry at original byte 0 and destination 8. Artwork replacement is specified below. |
| M5 | Instruction-stream fetch includes operands throughout the definitions and captions; the abstract’s proposed wording was tightened to “instruction-stream bytes”. Artwork replacements are specified below. |
| M6 | Global absence/priority claim removed. The question is founder-variant retention separately from lineage survival; faithful unmodified copying does not by itself establish retention of introduced variants. |
| M7 | Historical bridge credits Langton, Tierra/Ray, Avida/Ofria–Wilke, Fontana, Pargellis and C G et al., using verified references 7–12. |
| M8 | Mutation-off variation is consistent with drift, not proof of neutrality. Failed E1 stability is visible: 4/10 benign core-C1 worlds per length ended within 0.3–0.7, below ≥7/10. |
| M9 | Eligible 6502 panel described as 42 sampling roles, 41 distinct tapes from 11 worlds; nested units remain explicit. |
| M10 | Registered 8086 L1/L2 remain untested and L2 support suspended. Separate post hoc law results remain exploratory and are not plotted as registered evidence. |
| M11 | Methods now gives all four S1 arms, budgets, BRK policies, paired seeds, horizons and top-three schedule; C2 assays use budget 128 even for arm-C origins. |
| M12 | Methods explicitly distinguishes registered 64-partner final-cell wording from frozen32 execution and soup-partner trace wording from 16 uniform-byte partners. Corrected frozen32 is primary; both complete sensitivities are named and retain their reported outcomes. |
| M13 | Main and figure-index review status no longer imply a currently verified complete SVG package. At the initial revision check, `fig1.svg` was missing. Historical manifest/clearance text is explicitly scoped to the archived build. Figure job must restore/rebuild/verify final artwork. |
| M14 | `MAIN_v8.md` is the canonical caption source. `figures/legends_v8.py` now extracts it and enforces the five-caption/250-word contract; both exported `LEGENDS_V8.md` files were regenerated identically. |
| M15 | Replaced all five long captions with the proposed publication captions: 235, 246, 247, 236 and 245 whitespace-separated words including titles. Displaced protocols, source mappings, proofs and operational history are in Methods/SI. |
| m1 | Opening now requires heritable variation in reproductive success without asserting that every new difference must survive. |
| m2 | Proposed abstract A adopted, with review-copy result paths retained and the instruction-stream precision noted under M5. Nested encounter totals are absent from the abstract. |
| m3 | Fig. 1b dates culture-based detection; Fig. 1c shows illustrative sequence similarity. |
| m4 | Both final-majority counts and detection-order regrouping explicitly exploratory; detection order is not ancestry. |
| m5 | Registered mechanistic predictions failed to explain the balance; the measured small protective toxin effect is not rewritten as no effect. |
| m6 | Fig. 1 caption retains the confidence-bound meaning of ring agreement; it is not literal zero observed change. Required artwork count header and row are below. |
| m7 | Figure 2 caption uses fetch/confinement terminology. Closure remains partner-content independence. Artwork must replace “Closed” with “Confined”. |
| m8 | Preface, Results, Methods and captions distinguish core C1 from displacement experiment C1 and 6502 retention experiment C2. Standalone labels specified below. |
| m9 | Figure 3 caption states 101 settings, including 22 original-value controls, in 22 archived parents; it does not call all settings edits. |
| m10 | Figure 1 caption treats the two mechanisms as separate observations, not a demonstrated ancestor–descendant pair. Artwork sequence cues must be corrected as below. |
| m11 | The benign core-C1 T-into-R median uses four defined ratios from five worlds; zero abundance and diversity remain in the absolute summaries. |
| m12 | Methods states pseudorandom initialization, padding exclusion and distinct competition initialization. Detection definition is explicitly scoped to S2; SI gives cohort-specific sampling. |
| m13 | Generation-two culture score includes all first-generation outputs, not only close-copy successes. |
| m14 | Consolidated statistical paragraph separates worlds from nested units, deterministic enumerations from world trials, scheduled right-censoring and undefined ratios. SI retains applicable procedures. |
| m15 | Darwinian-threshold framing retained as requested, with the explicit limitation that the operational criterion is neither necessary for Darwinian evolution nor a reconstruction of the cellular transition. Adaptive use remains unshown. |
| m16 | Task-free evidence is identified; Cicala’s task-selected conditions are acknowledged separately. |
| m17 | Load–push motif credit identifies the original 8080 report and subsequent Z80 assays. |
| m18 | Prior immediate-displacement and offset-reading architectures distinguished; Cicala v2 cited. Present contribution is the conditional payload count and its tests. |
| m19 | Global caption note states that colours are defined separately in each panel; no implication that confinement equals transmission. |
| c1 | Finding-named Results headings retained and shortened; concise Methods navigation headings used. |
| c2 | Internal “Use the final…” instruction replaced with descriptive prose. Detailed operational failures, restart logic and accounting moved to SI Note 6; main Methods gives a short factual summary. |
| c3 | Figure 2 caption specifies zero-based step 6; Figure 5 uses tile outlines. |
| c4 | Hexadecimal byte grids identified; prose edits use 0x notation. Captions identify the L20 seed-6003 versus seed-6004 specimen switch. Expanded artwork labels specified below. |
| c5 | Separate Data availability and Code availability sections distinguish records/access from frozen code, dependencies, licence, hardware and runnable-instruction placeholders. |

## Audit resolution

`AUDIT_v7_RESOLUTION.md` records one decision for each originally flagged row, retaining the original N identifiers. The original mismatch N0046 is corrected to **six total scored sites, four unexecuted**, for the period-four tape with displacement eight. Unresolved values are assigned exact current sources or explicit derivations; moved caption facts retain scientific provenance in SI rather than being treated as resolved merely by deletion. Methods contains the numerical derivations and source conventions supplied by the audit reconciliation. This is a source reconciliation, not a fresh experiment or independent certification of arbitrary executable programs.

## Supplement and reference apparatus

SI Notes 1–9 cover, respectively, overlap/proof/Z80 specialization; provenance and inflow; literal-channel proofs/constructions; 6502/8086 law conditions; cohorts/measurement/figure details; registration and operational chronology; I1 and Tier-2 impact; causal census correction; and exploratory S2 analyses. Tables 1–3 locate cohort definitions, registration chronology and complete verdicts, with further tables mapping detailed generated outputs. The Extended Data inventory records actual file existence and separates older displays from current v8 targets.

Bibliographic checks preserve the published “constraints” in Vasas 2010, Santos in the 2012 author list, “C G” surname styling, Fontana’s 1991 chapter year and Cicala v2. The references are not padded with uncited optional works. All fourteen are cited and appear in first-citation order.

## Parallel figure-job handoff

The figure job owns rendered displays. This manuscript revision owns the canonical text and synchronized caption exports. Read `MAIN_v8.md` for final captions and use `figures/legends_v8.py` to export them; do not reinstate the old generated long captions. Use the `ed8_*.pdf` targets and per-item source/coverage requirements listed in `SI_v8.md`; a legacy `ed_v6_*` display is not silently relabelled as complete coverage of a new item.

| Panel/package | Required artwork or verification action |
|---|---|
| Fig. 1a | “Two copying mechanisms”; “Return-containing design”; remove or visibly separate the sequential third badge. The examples are not an observed ancestor–descendant pair. |
| Fig. 1d | Ring row “Below change-rate bound”; count header “Agreement / below bound”. |
| Fig. 2a | “Lost under copy criterion”; “Alive, <50% carrying”; “Alive, ≥50% carrying”; “Confined regenerators”; “Confined transmitter”; “Branching, generation 8 (archive = corrected core)”. |
| Fig. 2a–c | Identify “Transmitter, L=20, seed 6003” in a,b and “Transmitter, L=20, seed 6004” in c; expand “generation 8”; space “≥ half”. |
| Fig. 2c | “Instruction-stream fetched”, including operands. |
| Fig. 2d | “Read address 4; ancestry: byte 0”; “Writes 0x04 over byte 8 (0xA7)”; “Erasure at instruction step 6 (zero-based)”. |
| Fig. 3b | “Z80 exact positions: 101 settings”; “6502 exact positions: 42 sampling roles”; “Registered 8086 law test: untested”; “Frozen extractor certified no eligible copier”. Post hoc points remain absent. |
| Fig. 4a,b | “Partial-copy R” and “Full-tape T”; median full-tape-class share T/(R+T). Benign corrected T-into-R median note: “n=4 defined ratios; one of five worlds has R+T=0”. |
| Fig. 4c | “Open: partner instruction-stream fetch in ≥8/16 traces; confined: none in 16/16.” This panel’s site-score classes differ from a,b. |
| Fig. 5 | Preserve local colour keys and tile outlines; shared bytes do not certify payload. |
| Package | Restore `figures/out/v8/fig1.svg`, render the ED targets, regenerate the manifest and run technical verification on the actual final files. Earlier review/clearance hashes do not certify changed artwork. |

## Verification record

The five canonical legends and both exported copies were checked for equality and the word limit. Fourteen numbered references have sequential first appearances and no uncited entries; no author–year brackets remain. Abstract and main text contain continuous prose without bullets. Review changes preserve the failed X1 carriage prediction, unsupported S1 P1, untested S1 P3, failed E1, historical versus corrected DZ outcomes, S2 route failure/kill and untested registered S2 law verdicts. Independent recomputation confirms 42 eligible 6502 roles = 41 distinct tapes from 11 worlds and the E1 4/10 counts. Final checks passed: all 154 original exception IDs occur exactly once; the targeted cross-table evidence assertions pass; all cited existing-source paths in MAIN and the resolution ledger resolve; SI sources resolve apart from the nine explicitly prospective ED PDF targets. The original v7 manuscript hash is unchanged. The five caption exports match exactly. All 43 review IDs have dispositions in this log.

The final content check records 193 abstract words and 3,684 main-text words after removing headings and source/citation brackets. Methods is 4,431 words on the same convention, retaining explicit numerical derivations for the review audit. SI contains Notes 1–9 and Tables 1–15. The current filesystem still has no v8 ED displays and no `fig1.svg`; these artwork tasks remain assigned to the parallel figure job, not falsely certified as completed.

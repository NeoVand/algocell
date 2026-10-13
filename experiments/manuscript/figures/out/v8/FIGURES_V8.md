# Figures v8: S2 results and v7 cleanup

Technical verification passed for all five figures. Astra cleared 5/5 delivered images. Clearance requires both a no-substantive-defects verdict and an exact hash match between the reviewed and delivered PNG.

| Figure | Page (mm) | figcheck | Astra rounds | Clearance |
|---|---|---|---|---|
| [fig1](fig1.png) | 180 × 146 | clean | 3 | [CLEARED](critique/fig1/round3/critique.md) |
| [fig2](fig2.png) | 180 × 136 | clean | 3 | [CLEARED](critique/fig2/round3/critique.md) |
| [fig3](fig3.png) | 180 × 166 | clean | 3 | [CLEARED](critique/fig3/round3/critique.md) |
| [fig4](fig4.png) | 180 × 163 | clean | 3 | [CLEARED](critique/fig4/round3/critique.md) |
| [fig5](fig5.png) | 180 × 162 | clean | 3 | [CLEARED](critique/fig5/round3/critique.md) |

All figures are supplied as vector PDF, SVG with editable chart text, and PDF-rendered RGB PNG at 600 dpi. Every panel is populated. No pending 8086 law entry remains.

PDFs: [fig1](fig1.pdf) · [fig2](fig2.pdf) · [fig3](fig3.pdf) · [fig4](fig4.pdf) · [fig5](fig5.pdf). SVGs: [fig1](fig1.svg) · [fig2](fig2.svg) · [fig3](fig3.svg) · [fig4](fig4.svg) · [fig5](fig5.svg).

**Changes.** Figures 1–4 start from the v7 scripts and frozen data. Paragraph-like method notes were moved to `LEGENDS_V8.md`, preserving direct labels, category headings, inline keys and number columns. Figures 1 and 2 were compacted after removing prose; Fig. 4 retains its complete dial and paired C1/archive endpoints. Fig. 3b states that the evolved 8086 law is untested, with no eligible certified copier. No numerical v7 measurement was changed.

Fig. 5, “Different machines, different beginnings,” shows all-world route partitions, L16 Kaplan–Meier detection curves with the six coincident S2 censoring marks directly labelled, a literal-width/pusher/route matrix, and all ten physical-phase `5F A4 x y` first-copier tapes (160 bytes). The S2 surprise is shown as a teal outlined row: width 2 and the exact `B9 51` census pusher coexist with 12/14 confined first detections. The two open detections and six censored worlds remain visible. Registered verdicts are in the legends.

**Scientific limits.** Route panel a uses all 80 historical Z80 G worlds, while b uses the 20 L16 worlds. Historical Z80 route evidence, S1 and S2 classifiers, sampling schedules and horizons differ. Timing is descriptive, with no registered directional timing claim. The S2 law is untested: zero eligible evolved cases do not imply a successful or failed evolved-law assay. Tile colour shows shared byte values, not certified roots or free payload. The optional S2 exploratory report was unavailable when panel d was authored.

**Provenance and review.** Input SHA-256 and plotting-table hashes are frozen under `data/`. S2 survival coordinates were checked against the frozen Kaplan–Meier table. Every shown first-copier byte and detection update was checked against the S2 world table. Vector text, PDF page bounds, exact PNG dimensions/DPI, and clean figcheck were verified. Each Astra round attaches the old `fig1.png` and `fig4v5.png` plus the new image to the explicitly requested gpt-6-astra/high CLI invocation in an empty directory. Prompt, images, hashes, unmodified review text and command logs are archived. The first-round prompt incorrectly applied the Fig. 3b law-status requirement to every figure; rounds 2–3 scoped it to Fig. 3b. Every review is retained unchanged. At most three rounds per figure. No simulation, GPU burst or remote experiment was run. The experiment branch was merged because the local v7 branch and S2 branch diverged and could not fast-forward.

**Reproduce.**

```sh
/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/figures/prepare_v8.py
/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/figures/build_v8.py
/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/figures/legends_v8.py
/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/figures/verify_v8.py
```

Astra review: `critique_v8.py --figure N --round R`; archived rounds are never overwritten.

[Full legends](LEGENDS_V8.md) · [Technical manifest](qa_manifest.json) · [Frozen data](data/)

**Final unedited image-only assessments.**

### fig1

**Figure 1 meets or exceeds the supplied references in polish, information, and self-explanation.** I find no supported substantive defect in the attached PNG. All four panels contain substantial content, and the added measurements and validation summaries retain the references’ emphasis on direct labels and explicit comparisons.

At the stated 180 mm width, the typography appears comparable to the references’ publication scale. The smallest mechanism annotations deserve attention in a print proof, but they do not appear demonstrably less readable than the reference annotations. There is no visible clipping, consequential overlap, broken glyph, or empty panel.

- **Panel a:** The numbered sequence, byte labels, organism/partner identities, occupied-memory labels, stack annotation, and return arrow explain the mechanism at the reference’s level. The inset preserves the crucial “operand = output” explanation. The new heading gives the diagram a clearer organizing statement.
- **Panel b:** The outcome, world-update axis, tape-length encoding, and cohort size are explicit. The six curves are identifiable, including the markedly slower and incomplete emergence for **L = 9**. The logarithmic ticks communicate the time scale adequately.
- **Panel c:** The three quantities have distinct encodings and a direct key. The title supplies the organism architecture and tape length; the footer identifies the seed and contemporaneous modal-tape comparison. The sharp increase in modal similarity and accompanying decreases in the other quantities are visible. The singular “One Z80 world” appropriately limits the displayed result.
- **Panel d:** Both comparison groups are named, all five rows are populated, and the number columns expose the denominators. The displayed differences agree arithmetically with the counts. The **0.32% expected** annotation makes the otherwise nearly invisible hardware difference legible. C1 is defined within the figure.

**Optional polish, not substantive defects:**

- **Panel a:** If further clarification is desired, label the lower diagram as the evolved return-containing variant. This would make its relationship to the upper sequence more immediate. The unexplained `e3` byte could also receive a short functional label if its identity matters to the intended reading.
- **Panel b:** Increase the separation between “World update” and the archive/cohort footer slightly; they currently form a tight text block.
- **Panel c:** Replace “Cells ≥75% modal match” with “Cells matching modal tape at ≥75% of bytes,” if byte agreement is the intended definition. This would reduce the shorthand without adding methods.
- **Panel d:** Align the numerator and denominator of the hardware count more consistently with the single-line counts below. A small marker at the expected-difference segment could make the orange callout’s attachment more immediate.

The figure does not need full protocols or additional scientific claims to make its displayed results understandable.

MEETS OR EXCEEDS REFERENCES - NO SUBSTANTIVE DEFECTS

### fig2

Figure 2 meets the references’ standard in polish, information, and self-explanation. Every panel contains substantive content. At the stated 180 mm width, the body text, ticks, keys, and numerical columns appear readable relative to the supplied references. I see no substantive overlap, clipping, broken glyphs, or missing panel content.

**Substantive defects: none supported by the images.**

- **Panel a:** Category headers, three outcome colors, a proportion axis, control counts, and site counts make the comparison understandable. The denominators are visible. The site-threshold note explains how a small carried fraction can coexist with a site count of zero, including the thin teal segment in the block-copy row.
- **Panel b:** The axis explicitly identifies the numerator and denominator, and the transfer axis and series key establish the comparison. Persistent carriage in Genome 20, world 3, versus rapid loss in the other examples is clear. Marker shapes distinguish the two orange series even where their trajectories converge.
- **Panel c:** The byte tracks combine execution status, early carriage, persistent carriage, and four numerical summaries effectively. The heading explicitly explains that tracks are scaled to their respective tapes, preventing their equal displayed lengths from implying equal byte counts. The different examples used here versus panels a–b are identified by their lengths and world labels.
- **Panel d:** Both plots and the accompanying direct labels contribute essential information. The address trace locates the erasure event; the ancestry-copy trace shows its consequence. The byte identity, replacement value, instruction number, parent/partner address ranges, and reset at step 128 make the mechanism understandable without reproducing the experimental protocol.

**Optional polish, with concrete fixes:**

- **Panel a:** “Archive = C1” introduces an unexplained identifier, while panel b uses only “archive.” Either identify C1 briefly in both headings if it matters to the comparison, or move the identifier to the manuscript legend. This does not prevent interpretation of the displayed results.
- **Panels a–c:** Standardize example naming. “Genome 20, world 3” and “Genome 20, world 4” could use “Genome, L = 20, world …”; panel b could likewise use “Pusher, L = 16” and “Block-copy, L = 32.” The existing labels are decipherable, but consistent notation would reduce cross-panel effort.
- **Panel a:** Change “≥half” to “≥ half.” “At g8” could become “at generation 8” in the numerical header if space permits.
- **Panel c:** The title is unusually instructional and visually heavy. Shorten it to “Serial retention by byte site,” retaining “One box per byte; tracks scaled to tape length” as a smaller inline explanation. Preserve the rich key and number columns.
- **Panel d:** The title and parent/partner subtitle are tightly stacked compared with the surrounding spacing. Add a little vertical separation. Use typographic ranges consistently: “a–c,” “0–31,” and “32–63.”

The figure’s information density is purposeful, and its direct labels adequately explain the measurements and main outcomes. The remaining refinements are editorial.

MEETS OR EXCEEDS REFERENCES - NO SUBSTANTIVE DEFECTS

### fig3

Figure 3 **meets the supplied references in information and self-explanation, with comparable publication polish**. I find no supported substantive defect in the PNG. All four panels contain meaningful scientific content. The typography appears comparable to the references’ stated 5–7 pt body/tick standard at 180 mm width; I see no actual clipping, text collisions, broken glyphs, or unreadable essential labels.

The mechanisms and results remain understandable without bringing the full methods into the figure:

- **a:** The address ranges, two-byte box key, four-byte core, copy directions and offsets explain the mechanism. The two outcomes distinguish payload erasure from retention, and the brackets connect the diagrams to \(r=d-c\). The initially arbitrary contents of B are explicitly identified.
- **b:** Marker shapes, colors, cohort sizes, axes and units are provided. The identity line communicates the predicted agreement; the inset makes the small values inspectable. The exceptional cell is explicitly labeled with its assay and carried-position counts. The required **8086 statement correctly says “untested” and “no eligible certified copier.”** The disc-area key distinguishes cell counts from series-only rings.
- **c:** The displacement edit, hexadecimal-to-decimal conversion, modulo operation and resulting 28 retained payload positions are explicit. The core/payload boundaries, categorical key and offspring generation make the displayed comparison interpretable.
- **d:** The hexadecimal axes cover the 256 byte values. The three category counts sum to 256, and the working criterion, restricted law domain, capacity scale and highlighted example values are supplied. Its relationship to the displacement edit in c is understandable.

**Optional polish only:**

- **a — Reduce redundant bracketing.** In the \(d=4\) example, the core and retained-prefix brackets span the same two boxes and sit close together. A combined label such as “Retained prefix: d = c = 4 bytes” would simplify this without losing information.
- **a — Tighten the lower whitespace.** The equation is separated from the schematic key, followed by a substantial blank region before c. Moving the equation closer to the key would make the explanation feel more cohesive.
- **b — Shorten the inset title.** “Detail: 0–10 on both axes” would state its range more precisely than “Same points: 0-10.” Explicit inset axis titles are unnecessary here.
- **b — Consolidate the size key.** Bringing its explanatory sentence, example discs and numerical labels slightly closer together would improve grouping.
- **c — Clarify the mutation location inline.** “Edit byte 0: 0x04 → 0x60” would connect the intervention to the preceding sentence more immediately.
- **d — Make the final category label parallel.** “11 working, within law domain; shade = capacity” would remove the small inference currently needed to distinguish it from the 88 working values outside the domain.
- **Overall — Align the right-column headings.** Panel d’s letter and title begin slightly left of b’s. Matching their horizontal positions would improve the grid.

These are refinements, not missing scientific content or barriers to understanding.

MEETS OR EXCEEDS REFERENCES - NO SUBSTANTIVE DEFECTS

### fig4

Figure 4 **meets the supplied reference standard in polish, information, and self-explanation**. I find no substantive defect supported by the PNG. Its density is comparable to reference Figure 4, and the typography appears consistent with the supplied 5–7 pt publication standard at 180 mm width.

All panels contain substantive content. I see no clipped labels, broken glyphs, overlapping text, or empty panels.

- **Panel a:** The benign/lethal comparison is immediately apparent. R and T, implementation differences, starting mixtures, cohort sizes, time units, and both measured fractions are explicitly identified. The endpoint tables provide numerical comparisons without visibly obstructing the trajectories. The main result—low final transmitter share under benign conditions and high share under lethal conditions, despite high motif prevalence in both—is understandable directly.
- **Panel b:** Individual-world distributions complement the medians. Condition colors, implementation fills, category labels, and measurement denominators are present. The sample-size note explicitly accounts for the undefined transmitter ratio when R + T = 0. The core-loss annotation identifies the exceptional observation without requiring a methods paragraph.
- **Panel c:** Every probability setting has results for three outcomes. The probability definition, implementation order, counts out of ten, observation horizon for transmitter dominance, waiting-time units, and open/confined distinction are supplied. The table communicates the comparison without reproducing the experimental protocol.

**Optional polish, rather than substantive defects:**

- **Panel a:** Six trajectories become difficult to distinguish individually where they converge. Slightly strengthen the light/dark implementation contrast and use consistently spaced, sparse markers along the curves. Preserve the endpoint tables, which already resolve the numerical comparison.
- **Panel b:** The point shapes inherit their starting-mixture meaning from the top legend. Add a compact “Shapes: starting mixtures as in a” label to make this panel easier to read independently.
- **Panel b, right plot:** “Distinct T tails / world (bytes 4:31)” is the most compressed measurement label. “Distinct T-tail sequences per world” with the byte range beneath would read more naturally.
- **Panel c:** Change “at 300k” to “at 300,000 updates,” and standardize “thousand” versus “thousands” across the time labels.

These are refinements; none prevents understanding the measurements, encodings, cohorts, or principal outcomes.

MEETS OR EXCEEDS REFERENCES - NO SUBSTANTIVE DEFECTS

### fig5

Figure 5 meets the references in polish, information, and self-explanation. I find no substantive defect in the supplied PNG. All four panels contain substantive content, and the main comparisons remain understandable without reproducing the Methods.

- **Panel a:** The route definitions, segment counts, detected-world denominators, and machine/cohort labels explain the comparison directly. The 8086 bar correctly distinguishes 2 open, 12 confined, and 6 censored worlds. Its areas agree with those counts. The pooled Z80 cohort is explicitly identified by its four tape lengths.
- **Panel b:** The arrival curves have clear axes, a readable logarithmic time scale, direct cohort labels, and detection totals. The 16-byte restriction and 20-world denominators distinguish this comparison from panel a’s pooled Z80 result. The 8086 plateau at 14/20 and the “6 censored” annotation are consistent. Color identifies cohorts here; direct labels prevent confusion with panel a’s route encoding.
- **Panel c:** Literal width, census motif presence, and observed routes are clearly separated. The highlighted 8086 row makes the central comparison apparent: an exact pusher candidate is present, while most detected first copiers are confined. The 12/14 count and two open cases reconcile with panel a; the six undetected worlds are already accounted for there.
- **Panel d:** The family frequency, confinement result, opcode interpretation, byte positions, variable bytes, world identifiers, detection updates, and mechanism classifications provide substantial explanatory content. The ten displayed rows agree with the stated family count. Repeated teal opcode cells make the shared sequence easy to inspect.

At the specified 180 mm width, the dense table and byte-grid text are comparable to the references’ smaller labels. I see no supported reason to flag actual unreadability, overlap, clipping, broken glyphs, or missing panel content. The figure is more typographically assertive than the references, but remains coherent and navigable.

Optional polish:

- **Panel b:** Draw the censoring key using the same graphical terminal marker used on the teal curve. The footer’s isolated vertical stroke is less immediately recognizable than the other keys.
- **Panels c and d:** Add “detected” to “12/14” and “10/14,” respectively. The denominator is recoverable from panels a–b, so this would improve independent panel reading rather than repair missing information.
- **Panel d:** Strengthen the vertical rules after byte positions 3, 7, and 11 slightly. This would expose the four repeated units more immediately without adding text or changing the encoding.
- **Overall:** Slightly reduce the weight or size of the overarching title and longer panel headings. The current hierarchy is heavier than either reference; the information itself already supplies a strong structure.

MEETS OR EXCEEDS REFERENCES - NO SUBSTANTIVE DEFECTS

# Style notes from the reference library (2026-10-08)

Source: `experiments/library/` (gitignored), 19 open-access papers fetched from the user's list: five Nature Articles
(Sharma 2023 assembly theory; Moger-Reischer 2023 minimal cell; Matreux 2024 heat flows; Müller 2022 RNA–peptide world;
Singh 2025 thioester aminoacylation), the Cronin & Walker 2016 Science perspective (author PDF), the Computational Life
preprint (Agüera y Arcas 2024), and the Nature-portfolio ALife and theory papers (Hintze & Bohm 2026; Piñero 2026;
Adams 2017; Takeuchi 2017; Ouazan-Reboul 2023; Ameta 2021; Mizuuchi 2022; Marshall 2021; Kempes 2025; Plum 2025;
Schmickl 2016; Abbott 2026). The paywalled Nature and Science papers were not obtainable; their abstracts are on the
publisher pages. These notes are my observations, not a template to copy.

## 1. The summary paragraph is a ladder

Every Nature summary read the same way, in four rungs, 150–200 words:

1. **Settled knowledge, with references, in one or two sentences.** The field's big question stated as something
   everyone agrees on: the emergence of building blocks "is a crucial step during the origins of life"; the RNA world
   "is one of the most fundamental pillars"; a minimal cell "can reveal mechanisms … critical for the persistence and
   stability of life".
2. **The gap, in one sentence, signalled by "However", "yet" or "it remains unclear".** Matreux: all known pathways
   rely on rare pure feedstocks and manual purification. Moger-Reischer: unclear how a minimal cell responds to
   evolution. The gap is specific and the reader can see why it is hard.
3. **"Here we show/report/demonstrate" with the result and its mechanism**, one or two numbers at most ("by up to
   three orders of magnitude", "39% faster"), and the key qualifier of scope.
4. **Consequence, two or three sentences:** what the result demonstrates or implies for the field, in the future
   conditional where appropriate ("could have explored", "provides critical insights into").

What is absent: rhetorical questions, slogans, italics for emphasis, descriptions of the apparatus, lists of
numbers, hedges stacked on hedges.

## 2. The main text builds the problem before it touches the system

Three or four paragraphs (350–600 words) precede any detail of the method:

- Paragraph 1 states the broad phenomenon and why it matters, with references in every sentence.
- Paragraph 2 says what has been done and with what approach, and credits it generously.
- Paragraph 3 says what remains unknown or "elusive", and why the existing approaches cannot reach it.
- Paragraph 4 previews the approach and the finding: "To gain insights into …, we conducted …"; "In this work, we show
  that … provide an answer to this problem."

Moger-Reischer walks genome complexity → the simplest organism → the minimal cell → the open question → two
competing expectations → the experiment. Matreux walks the key moment → how pathways are studied → the purification
problem → partial solutions → "remains elusive" → "In this work, we show". The system is named only in paragraph 4,
with just what the reader needs to follow Fig. 1.

## 3. Results paragraphs: claim, evidence, interpretation, next question

- Headings name findings or topics in ≤ 40 characters: "Highest recorded mutation rate", "Recovery of fitness in a
  minimal cell", "Divergent mechanisms of adaptation", "Peptide synthesis on RNA".
- A paragraph opens with the claim, gives the evidence with the statistic in parentheses (t, P, n, ± s.e.m.) and the
  figure reference at the claim, interprets it against a named expectation ("consistent with predictions from the
  drift-barrier hypothesis"), and ends by raising the next question ("Second, with knowledge of the mutational input,
  we evaluated whether …").
- "(Methods)" defers every procedural detail; the text never explains how a measurement is done beyond a clause.
- Sentences are long, declarative and complete; transitions are spare ("Notably", "Importantly", "In contrast",
  "Taken together"); the voice is "we" without personality.
- Numbers appear where they carry the claim, not as inventories; comparisons are stated as effects ("increased by 80%,
  whereas the minimal cell remained the same").

## 4. Discussion and outlook

Two to four paragraphs. First, the finding restated plainly without numbers. Second, what it explains and what it
connects to, with named theories and named prior results. Third, implications and predictions, including for
practice ("Some degree of genome minimization will probably be a common path … It would be undesirable if …").
Fourth, limits stated as conditions ("if we assume that our findings are somewhat general"), then a confident final
sentence about what the work establishes.

The ALife paper (Hintze & Bohm) contrasts the finding with the classical picture (Langton's loop versus distributed
replicators), meets the obvious objection (crystal growth) head on, and says in one sentence what is new. The
Computational Life preprint ends with a list of open questions about which substrate properties encourage or inhibit
self-replicators; those are questions our paper answers and should cite as such.

## 5. Figures

From Sharma Fig. 1, Moger-Reischer Figs 1–3, Matreux Fig. 1, Hintze Fig. 1:

- **Few elements.** A main-text panel often holds one small chart: two lines, two groups, one comparison. A figure
  rarely has more than four panels. Nothing is dense.
- **Type.** 6–7 pt throughout, never smaller; horizontal labels; bold lowercase panel letters outside the top-left
  corner; P values at the top-right inside a panel; n and error-bar meaning in the legend.
- **Colour.** Two saturated colours for two groups (blue and red), grey for everything structural (arrows, boxes,
  shading), black for line art and text. No decorative colour, no coloured text, no five-colour categorical sets.
- **Schematics.** Illustrator-grade vector: uniform stroke width, proper arrowheads, aligned boxes, generous spacing,
  labels in the same typeface as the axes; a schematic in Fig. 1 establishes a vocabulary the later figures reuse
  (Matreux 1a/1b: the conventional workflow beside the proposed mechanism). Grid-cell drawings (Hintze) use large cells
  and short labels ("a. glider").
- **Whitespace.** Margins inside every panel; no element touches another; legends explain, panels show.
- **Legends.** A bold title sentence; then "a, …" per panel saying what is plotted, how many, what the error bars
  mean and which test gave the P value. No narrative, no methods.

## 6. Where our draft and figures fall short of this (details in REWRITE_PLAN.md)

- The summary and introduction skip rungs 1–2: two short paragraphs, then the apparatus. The reader is never told why
  the dependence of a first replicator on its surroundings matters, what question it answers, or what it explains.
- The register is a blog's: slogans ("Code equals data equals literal"), fragments ("Note what closure is not"),
  italics, bytes in the first results paragraph, numbers crowding every sentence.
- Nine results sections for 3,400 words fragment the argument; the parts that make the paper general (the theorem and
  the second machine) come last and short; the discussion neither explains nor predicts.
- The figures have five crowded panels each, 4.5–5 pt text, crude arrows, decorative colour, and schematics drawn
  without a visual reference. Legends list rather than explain.

Checklist before the next version is shown: every paragraph of the introduction maps to a rung; every heading is a
finding; every results paragraph has claim–evidence–interpretation; the discussion has all four parts; every figure has
≤ 4 panels, ≥ 6 pt text, one accent colour, no overlaps at print size; every legend has the title sentence and the
per-panel statistics.

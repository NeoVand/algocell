# Paper plan — Nature Article (decided 2026-10-08 05:45; venue: Nature; arXiv preprint planned)

## 1. Nature's constraints (official formatting guide, fetched 2026-10-08; figure guide fetched the same night)

| item | requirement | our target |
|---|---|---|
| length | physical-sciences Articles "do not normally exceed 6 pages" ≈ 2,500 words incl. summary + 4 display items; biological ≈ 8 pages ≈ 4,300 words + 5–6 display items; counts exclude title, authors, acknowledgements, references | **≤ 3,500 words main text, 5 main display items** (we sit between the categories: origin of life × computation; the editor decides the page budget, so the text must cut to 2,500 if asked) |
| summary paragraph | "ideally of no more than 200 words", fully referenced, "Here we show", no numbers/abbreviations unless essential; 2–3 sentences of basic introduction, background, main conclusions, 2–3 sentences of context | 180–200 words, one number at most |
| title | ≤ 2 lines in print = **75 characters incl. spaces**; avoid numbers, acronyms, punctuation | working title "Open replicators evolve closure in a digital primordial soup" (61 chars) |
| subheadings | ≤ **40 characters** incl. spaces | all results subheadings rewritten to ≤ 40 |
| references | typically ≤ **50** in the main text (Methods/SI refs not counted) | ≤ 45 main, rest in Methods |
| Methods | "does not typically exceed 3,000 words"; no figures/tables; short bold subheadings | ≤ 3,000 words, with Statistics and Pre-registration subsections |
| figure legends | **< 300 words** each; start with a brief title; no methods details | ≤ 250 words |
| display items | a modest item ≈ ¼ page (~270 words of budget), a composite ≈ ½ page (~600 words) | 5 composite figures ≈ 2.5 pages of the 8 |
| Extended Data | **≤ 10** figures + tables; online only; multi-panel allowed | 10 (list in §3) |
| Supplementary Information | no figures (figures go to Extended Data); text/tables/code allowed | THEOREMS, full tables, pre-registration documents, code links |
| figure files | pdf or eps preferred; vector, text editable, fonts embedded (TrueType 2/42), no outlined text; RGB; sans-serif Helvetica/Arial, **5–7 pt** text, **panel letters 8 pt bold lowercase**; axis lines + ticks; units as "quantity (unit)"; no gridlines, shadows, patterns; accessible palette (Wong 2011 = Okabe–Ito), no coloured text; widths **90 mm** (single) / **180 mm** (double), depth ≤ **170 mm** | matplotlib: `svg.fonttype = 'none'`, `pdf.fonttype = 42`, Helvetica/Arial, 6 pt body / 5 pt ticks / 8 pt bold panel letters; export SVG + PDF (SVG for us and arXiv; PDF for submission) |
| preprints | Springer Nature permits preprints on any platform before submission; one author-services page says not during review — verify on Nature's preprint policy page before posting | post to arXiv (q-bio.PE + cs.NE + nlin.AO) on the day of submission, cite the preprint in the manuscript |

To confirm at submission time: the current word/display-item budget for our subject category; whether SVG is accepted
for initial submission (PDF is safe); the preprint-during-review clause.

## 2. Shape of the paper

**Title (≤ 75 chars).** Open replicators evolve closure in a digital primordial soup.

**Summary paragraph (≤ 200 words).** Draft in MAIN.md; rewrite to Nature's four-part structure, remove most numbers.

**Main text (≤ 3,500 words), subheadings ≤ 40 chars:**
1. (no heading; 2 paragraphs of introduction)
2. *A soup with nothing to select for* — system, culture test, scale.
3. *Tar, then an open replicator* — zero flood; `01 c5`; two-byte census.
4. *Closure evolves without a membrane* — partner tests, Stage G, convergence, L-dependence.
5. *Three load-bearing instructions* — atlas; L = 9 reversal in one paragraph.
6. *No size floor, a geometry switch* — dead zone and padding; P3 geometry vs topology in one sentence.
7. *Tar fools compression detectors* — one paragraph.
8. *A literal channel switches the open phase* — BFF std / wrap / wraplit / wraplitnh / lit; the classification.
9. *Closure as a theorem* — Theorems 1, 2, 6 and Lemma 7 in prose; the window model in one paragraph.
10. Discussion — individuality as a topological property of control; relation to prelife, error threshold, Darwinian threshold, organisational closure; biosignatures; limits.

**Methods (≤ 3,000 words):** Soup; Instruction set and ablations; Culture test; Events and statistics; Pre-registration
and amendments; BFF implementation and switches; Searches and censuses; Compute and reproducibility; Code and data
availability.

## 3. Display items

**Main figures (5, all composite, double column unless noted):**
- **Fig. 1 | The soup and the order of events.** a, schematic of the pair, ring, pointer and stack (conceptual —
  *to design with the user*); b, the pusher mechanism `01 c5` (code = data = literal; conceptual — *to design*);
  c, one world's time course: zero fraction, heritable fraction, dominant family (Stage G L = 16, data);
  d, Kaplan–Meier emergence by L (data).
- **Fig. 2 | The first replicator is open, the successor closed.** a, partner-copy fraction of first vs final per world,
  four L (Stage G, data); b, self-damage likewise; c, heritable fraction of random cells vs step, four L (data);
  d, closers: the `RET NZ`, `JR NZ`, `DJNZ`, `LDIR` genomes with the cycle marked (conceptual/diagram — *to design*);
  e, byte-identical convergence counts (data).
- **Fig. 3 | What the instruction set must provide.** a, atlas forest plot (Stage C, data); b, size axis: fraction
  alive and KM median vs L with the dead zone (Stage E, data); c, the dead-zone switch by padding (Stage F4, data);
  d, L = 9 reversal (Stage D, data).
- **Fig. 4 | One instruction decides how life begins in BFF.** a, time series of heritable fraction for the five
  variants (data); b, first-replicator openness vs final (data); c, the all-`P` organism and its collapse: share and
  pointer-entered fraction vs epoch (data); d, the classification table as a 2 × 2 diagram (conceptual — *to design*).
- **Fig. 5 | Closure as a cycle: theory.** a, Theorem 2 illustrated: pointer trajectory of an open vs a closed organism
  (conceptual — *to design*); b, the closure window q∫n dt with the three substrates placed (data + model);
  c, Lemma 7: fidelity vs leakage, measured points (data).
  (Figs 1a–b, 2d, 4d, 5a are the conceptual items the user wants to design together before tokens are spent.)

**Extended Data (10):** ED1 KM curves by L and ablation (B, E); ED2 the replicator zoo with disassembly (G);
ED3 ring sweep and F1/F3 falsifications (F); ED4 budget threshold and error-threshold curve (E4, E5); ED5 detector
benchmark (AUC by L and step); ED6 census dynamics of LDIR assembly (D); ED7 invasion assays; ED8 two-byte census and
fixed points; ED9 BFF structured search and closed-replicator search; ED10 reproducibility (same-seed divergence in the
Z80; bitwise identity in BFF) and compute. ED Table 1 final-tape classes and partner tests per L (G); ED Table 2
ablation sets.

**Supplementary Information:** THEOREMS.md (proofs), THEORY.md (pre-registrations with timestamps and amendments),
PLAN.md change log, full NUMBERS tables, LITERATURE notes; code/data links (the paper repository, to be created at the
end).

## 4. Quality factors (what Nature referees and editors reward; our checklist)

1. **One sentence of news.** "The first replicators are open literals; individuality is an evolved cycle in control
   flow, and one instruction switches it on." Every section earns its place by serving that sentence.
2. **Pre-registration visible.** Say in the main text that Stages D, F, G and the BFF cells were pre-registered with kill
   criteria, and say where the predictions failed (G1(c), G2 syntax, P1(b) mechanism, e3/e4). Referees trust papers that
   report their own misses.
3. **Numbers only from generated tables**; every number in the text traceable to a file in the repository.
4. **Statistics stated once, in Methods**: seed-paired tests, exact tests, Kaplan–Meier with censoring, no p-value
   theatre; effect sizes with counts (x/20) throughout.
5. **Figures that read without the text**: titles in legends not on panels, consistent colours per concept across all
   figures (open = black open markers; closed = vermilion filled; tar = grey; the five BFF variants fixed colours),
   same L ordering everywhere, lowercase bold panel letters, 90/180 mm, 5–7 pt.
6. **Generality claims bounded**: Theorem 6's hypotheses stated; "von Neumann machines", not "any substrate";
   chemistry as an open question in one sentence.
7. **Related work fair and specific**: Agüera y Arcas et al. credited for the system and the detector; Fontana & Buss
   for organisational closure; Krakauer et al. for individuality; Cicala et al. for Z80 soups; Nowak & Ohtsuki for
   prelife; the CALL-smear and zero-flood tar linked to Benner.
8. **Reproducibility**: one command per figure from the raw summaries; seeds listed; compute cost stated; the BFF soup's
   bitwise reproducibility noted; the Z80's GPU nondeterminism disclosed.
9. **No overclaim on thermodynamics**; Landauer/England as readings only.
10. **Length discipline**: cut the detector section to a paragraph and the geometry section to three sentences if the
    editor asks for 6 pages.

## 5. Work plan (order)

1. Follow-up runs `lit` and `wraplitnh` (running) → results into FINDINGS/THEORY → Fig. 4 data complete.
2. Figure style: Nature profile in `figstyle.py`; regenerate every data figure as SVG + PDF at 90/180 mm.
3. Main text rewrite to the structure above with the word budget; Methods; legends; Extended Data list.
4. Conceptual figures: design session with the user (Figs 1a–b, 2d, 4d, 5a), then build.
5. References (≤ 50 main), formatted; verify every DOI.
6. arXiv version (same text, figures as SVG→PDF, SI appended) on submission day.
7. Repository restructuring into a clean paper repo (last).

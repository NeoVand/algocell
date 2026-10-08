# Rewrite plan for the Nature manuscript (2026-10-08, after the user's review)

The user's verdict on draft v2 (`MAIN_nature.md`) and on the first figures: messy, careless, undergrad-level; the prose
does not read like a Nature paper, with almost no motivation, no "why this matters", no "what this explains", and the
big-picture storytelling we built (STORY.md, the alignment discussion) lost. This plan records what is wrong, the
story we are telling, the new outline, and the figure rules, so the next version starts from the right standard. The
style observations behind it are in `STYLE_NOTES.md`.

## 1. What is wrong with draft v2

1. **No ladder.** The summary jumps from a one-line definition of life to the apparatus. The introduction is two
   short paragraphs. The reader never learns (i) that origin-of-life theory locates the beginning of life in the
   appearance of an individual whose persistence depends on its own organisation, (ii) that no system has let that
   step be watched and dissected, (iii) that soups of random programs produce replicators without selection but have
   never been asked what the first replicator is or why, (iv) why an answer matters beyond the simulation.
2. **Wrong register.** Aphorisms ("Code equals data equals literal"), fragments ("Note what closure is not"),
   italics for emphasis, machine bytes in the first results paragraph, inventories of numbers in prose, catchy
   headings ("Tar, then an open replicator") instead of findings.
3. **Fragmented argument.** Nine results sections for 3,400 words; the ablation atlas, the geometry switch and the
   detector results interrupt the open→closed story; the theorem and the second machine, which make the result
   general, come last and short.
4. **Thin discussion.** It does not say what the result explains (why first replicators are sloppy and only partly
   heritable; why individuality can precede a membrane; why compression biosignatures fire before life and go quiet
   during it; why the Computational Life "zero-poisoning" happens), what it predicts (which substrates begin open; what
   to measure in chemistry), or its limits in the measured way of the exemplars.
5. **Figures.** Five crowded panels per figure; text at 4.5–5 pt; crude orange arcs; five-colour categorical sets;
   schematics drawn blind in matplotlib; legends that list.

## 2. The story (recovered from STORY.md, THEORY.md §5, THEOREMS.md Part II, LITERATURE.md)

**One sentence of news.** In soups of random programs the first self-replicators are open, their fate depends on the
code they run into, and evolution's first act is to close them with a cycle in control flow that makes an organism's
future depend on itself alone; this order is forced by the architecture of any machine with a literal write channel,
is switched on and off by one instruction in a second machine, and is invisible to the compression tests used to
detect digital life.

**Why it matters, rung by rung.**
- Theories of the transition to life agree on one thing across their disagreements: at some point a self-propagating
  pattern becomes an individual, something whose persistence depends on its own operation rather than on its
  surroundings (operational closure, Maturana & Varela; closure to efficient causation, Rosen; organisational closure,
  Fontana & Buss; information-theoretic individuality, Krakauer et al.; the Darwinian threshold, Szathmáry; prelife
  vs life, Nowak & Ohtsuki). Chemistry has not yet let that step be watched, repeated and taken apart.
- Soups of random programs give self-replicators without any fitness function (Agüera y Arcas et al.; Cicala et al.;
  Fontana's Turing gas; Core War, Tierra, Avida for engineered starts). But what the first replicator is, why it is
  that and not something else, what happens to it next, and which properties of a substrate decide this are the open
  questions the field itself lists, and detection has relied on compression statistics.
- Here we show: tar, then a two-byte literal that copies itself without a loop, open; then closure by a control cycle,
  convergently, without a membrane; three load-bearing instruction families; a theorem that closure requires a cycle
  and that literal channels make life start open; a second machine where one instruction switches the open phase on,
  and the lethality of its tar decides whether the open phase ends in closure, extinction or permanence.
- Consequences: the first replicators, wherever they arise, are expected to be sloppy, environment-dependent and only
  partly heritable; individuality is a topological property of control, not a geometric property of a boundary;
  sterile order precedes and mimics life, so biosignatures must test heredity by intervention; the classification
  predicts which substrates begin open; limits and what would test them in chemistry.

## 3. New outline (Nature Article; main text ≤ 3,500 words; 4–5 display items)

**Title (≤ 75 chars):** Open replicators evolve closure in a digital primordial soup (keep, 61 chars), or "The first
replicators are open, and evolution closes them" (51).

**Summary (≤ 200 words), four rungs:** settled knowledge (individuality as the mark of the transition) → gap (never
watched; soups unexplained) → here we show (order of events; cycle; theorem; switch) → consequence (prediction for
any literal-write substrate; detection must test heredity).

**Introduction, four paragraphs (≈ 550 words).**
- P1 The question and the theories, with references; what they agree on; why chemistry cannot yet show the step.
- P2 Computational soups: what they showed, what they did not ask; the compression detector; Cicala's Z80 soups.
- P3 Our approach in one paragraph: a Z80 soup as an experiment (3,560 worlds; ablations, sizes, budgets, mutation;
  pre-registered confirmatory stages), the culture test (heredity by intervention), a second machine, proofs.
- P4 "Here we show" preview, one paragraph, no numbers.

**Results, seven sections, headings as findings (≤ 40 chars):**
1. *Sterile order precedes life* (≈ 300 w): the zero flood, its mechanism, its sterility; detectors fire on it;
   the culture test.
2. *The first replicator is a literal* (≈ 400 w): the pusher in 80/80 worlds; the census (five words, all load–push);
   mechanism in prose (operand = written bytes); no loop, no counter, no reading. Fig. 1.
3. *Openness limits the first replicator* (≈ 350 w): partner tests, damage, the single cause; population heritability
   stays low; no budget rescues it. Fig. 2a–c.
4. *Closure evolves as a control cycle* (≈ 450 w): successors, four instruction families, 1.00/0.00; convergence;
   what closure is and is not (one measured sentence each); L-dependence and the pre-registration misses. Fig. 2d–e.
5. *Three instruction families are load-bearing* (≈ 350 w): atlas; stack writers and tar at L = 9; no size floor
   and the padding switch in three sentences. Fig. 3.
6. *One instruction decides how life begins* (≈ 450 w): BFF born closed; the literal makes it open; lethal tar
   collapses the window; benign tar makes it permanent; the classification. Fig. 4.
7. *Closure requires a cycle* (≈ 350 w): Theorems 2 and 6 in prose, Lemma 7 in one sentence, the window model, what
   theorems do not decide. Fig. 5.

**Discussion, four paragraphs (≈ 550 w):** what was found, plainly; what it explains (sloppy first replicators;
individuality before membranes; zero-poisoning; biosignatures); what it predicts and how to test it (literal channels
in template chemistry; detection by intervention; closure times vs size); limits (one real instruction set and one toy;
GPU nondeterminism handled by seeds; chemistry analogies as hypotheses), then the closing sentence.

**Methods (≤ 3,000 w):** unchanged content, reorganised to the exemplars' subsections.

## 4. Display items, redesigned

Rules for every figure: ≤ 4 panels unless a panel is a single small chart; ≥ 6 pt text; black, greys and one accent
colour (vermilion) for "closed/written"; categorical colour only where unavoidable (BFF variants) with one shared
legend; byte strips ≤ 16 cells and bytes shown only when the bytes are the point; proper arrowheads, uniform stroke;
nothing touching; every panel inspected at print size before it is shown; legends with a title sentence, then per-panel
content, n, error bars and tests.

- **Fig. 1 | The first replicator is an open literal.** a, the pair as a strip with the two pointers (schematic, from
  the user's generated reference); b, the pusher mechanism (schematic); c, one world's time course; d, emergence by L.
- **Fig. 2 | Open, then closed.** a, copy success first vs final; b, self-damage; c, heritable fraction vs time;
  d, the closers' cycles (three strips at most, from the traces, drawn to the new vocabulary); e, convergence counts.
- **Fig. 3 | What the instruction set must provide.** a, atlas; b, size axis. Padding switch and the L = 9 reversal to
  Extended Data unless the editor's page budget allows them.
- **Fig. 4 | One instruction decides how life begins.** a, heritable fraction vs epoch (split into two panels: without
  and with a literal); b, first-vs-final openness; c, the all-P share under lethal and benign tar; d, the classification
  (schematic, from the generated reference).
- **Fig. 5 | Closure as a cycle.** a, the trajectories (Theorem 2); b, the closure window (to design); c, fidelity vs
  damage.

## 5. Order of work

1. The user's generated images arrive → agree the visual vocabulary → redraw the schematics to it (vector).
2. Rewrite the summary, introduction and discussion to the ladder; then the results to the seven findings; every number
   from `NUMBERS_INDEX.md`, fewer of them in prose.
3. Rewrite the legends to the exemplar structure.
4. Rebuild the data figures to the rules above; inspect each at print size.
5. Read the whole against the checklist in `STYLE_NOTES.md` §6 before showing it.

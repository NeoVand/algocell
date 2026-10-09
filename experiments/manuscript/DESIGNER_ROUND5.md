# Designer brief, round 5 (2026-10-09)

Three panels of the revised manuscript are still drawn in code and would gain from the designer's hand. The renders to
start from are in `manuscript/figures/out/` (PNG and vector PDF); the legends are in `manuscript/MAIN_nature.md`. Every
number on a panel must stay exactly as in the current render, which is drawn from the generated tables.

**House rules (unchanged from round 4).** Final width 180 mm (double column), type no smaller than 5 pt at that width,
Helvetica-like sans. Colours: organism cells pale teal, partner cells grey, the instruction or arc that closes a cycle
vermilion, the instruction pointer teal, lethal tar purple. A loop is always filled vermilion, no loop open grey.
Conceptual panels stay abstract: about a dozen elements, abbreviated strips with an ellipsis cell, no prose inside the
drawing.

1. **Fig. 3f, control flow of the first replicator and four closers** (`fig3.png`, bottom right). Five strips: the
   pusher, whose pointer runs on into the partner; `RET NZ`, which returns to addresses formed from bytes it wrote
   itself; `JR NZ` and `DJNZ`, which jump back through the 16-bit address wrap; and `LDIR`, which repeats in place.
   Show each return as a vermilion arc, and keep the medians below the strips (copies 0.68 and self-damage 0.33 for the
   first replicator; 1.00 and 0.00 for every closer).
2. **Fig. 5e, the substrate map** (`fig5v4.png`, bottom). A two-by-two table: literal-write instruction present or
   absent, against tar benign or lethal. The cell texts are final. A small icon per cell for the order of events would
   help: open then closed, open throughout, open then collapse, born closed and late, born closed. No sentence goes
   under the table; the takeaway is in the legend.
3. **Fig. 6, what is proved** (`fig6v4.png`).
   - (a) Theorem 2 for the Z80. A 128-execution budget bar and an organism bar drawn to scale beneath it (at most 100
     cells). Below them, the two ways a pass can end: open, where the pointer leaves for the partner, and closed, where
     it revisits a cell (vermilion return arc).
   - (b) Proposition 3. The two-byte word `01 c5` tiled (no jump; 5 of 65,536 two-byte words write themselves), against
     the smallest closer seen, the four-byte block copy `1e a4 ed b0` (LDIR repeats in place). Give the chance that each
     word is present in the first soup: 0.99 and 6 × 10⁻⁵.

**From the round-4 critique, still open in the composed Fig. 1.** Panel letters must sit at the same height across a row.
The Kaplan–Meier panel uses the L colour palette of Fig. 4d.

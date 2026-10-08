# Conceptual panels, abstracted (proposal, 2026-10-08 evening)

The user's point: a conceptual infographic conveys the concept, not the data. My panels so far are byte-exact listings
dressed as schematics. This proposal cuts each conceptual panel to the one idea it must carry, with a budget of at
most twelve graphic elements and about fifteen words. The byte-exact versions (built, checked, in `concept.py`) move to
Extended Data, where exactness is the point.

Style (from the user's generated reference set): teal organism, grey partner, vermilion for what the organism writes
or where it jumps, charcoal type and lines, white background, generous air; cells large, never numbered unless the
number is the message.

## Fig. 1a — the encounter

One ring of twelve cells, six teal and six grey, drawn as a ring (not a strip with a wrap arrow): the pointer as a teal
arrowhead on the first teal cell with a short teal arc showing its direction; the stack pointer as a vermilion arrowhead
on the last grey cell with a short vermilion arc the other way. Two labels: "organism" and "partner". Nothing else; the
legend says what happens in an encounter.

## Fig. 1b — the first replicator

A teal strip of four cells reading `01 c5 01 c5`; one vermilion arrow from the strip to two vermilion cells at the far
end of a short grey strip reading `c5 01`; a teal pointer arrow passing from the end of the teal strip into the grey
strip. One line of text: "the word it loads is the word it writes". The three-box inset goes.

## Fig. 2d — closure

Two strips, no bytes. Top: a teal strip with a teal path running across it and out into a grey strip, label "open".
Bottom: a teal strip with the path running across it and a vermilion arc returning to its start, label "closed".
Underneath, one line in small type: "RET NZ · JR NZ · DJNZ · LDIR: four instructions, one cycle". The five byte-exact
rows become Extended Data Fig. 2.

## Fig. 4d — the classification

The two-by-two grid with one icon per cell and a label of at most four words: "open, then closed", "open for ever",
"open, then extinct", "born closed", "not run". Row and column headings as now. Every count moves to the legend.

## Fig. 5a — Theorem 2

A data chart, so it stays a chart, but with nothing written inside the plotting area: two traces, the shaded band, the
hollow circle where the pointer leaves, the legend outside the data. The numbers (ten instructions, ten of twenty bytes)
live in the legend.

## Process rule (applies to every figure from now on)

A figure is shown to the user only after (1) `figcheck` reports clean on the built figure (no text overlaps, nothing
clipped, no text crossed by a line, no type under 5 pt, no legend on data), and (2) an independent critic that sees only
the rendered image has returned no defects of those kinds. Layout is planned on paper first: a grid, the position of
every label, and the empty space, before any code is written.

# Conceptual panels — design brief for our discussion (nothing built yet)

Five panels in the main figures carry ideas rather than data: Fig. 1a (the system), Fig. 1b (the pusher), Fig. 2d (the
closers), Fig. 4d (the classification) and Fig. 5a (closure as a cycle). Nature's rules for them: same 5–7 pt
Helvetica, no coloured text, accessible palette, vector, every symbol explained in the legend. Below, for each panel:
what it must make a reader see in five seconds, two or three design options, and my recommendation. The one visual
vocabulary should run through all five: **memory = a horizontal strip of cells; the organism's bytes = black outline;
the partner's = grey; the instruction pointer = a small black arrow above the strip; the stack/write pointer = a hollow
arrow below the strip; a written byte = filled vermilion; a control cycle = a vermilion loop arrow.**

## Fig. 1a — the pair, the ring, the pointer and the stack

Must show: two tapes become one ring; execution starts in the organism and flows toward the partner; writes come from
below, from the far end of the partner.
- Option A, *the strip*: a 32-cell strip (L = 16), left half outlined black ("organism A"), right half grey ("partner
  B"), the strip's two ends joined by a thin arc beneath to show the ring; the pointer arrow at cell 0 with a dotted
  path rightwards; the stack arrow at cell 31 with a dotted path leftwards. One line of caption inside: "128 steps, then
  both halves are written back".
- Option B, *the ring*: a circle of 32 cells, organism on the left semicircle; clearer about the wrap, less clear about
  "first/second".
- Option C, *three strips in time*: before, after 128 steps, written back; shows the outcome but needs more space.
- **Recommendation: A** (the strip vocabulary is reused by every other panel; the arc is enough for the ring).

## Fig. 1b — the pusher: code = data = literal

Must show: the two-byte word is at once the instruction, its operand and what gets written; and that the pointer runs
on into the partner (openness) without any loop.
- Option A, *annotated strip*: the organism's strip filled with `01 c5 01 c5 …`; above cell 0 a bracket "LD BC,$c501"
  spanning three cells, then a bracket "PUSH BC" on the fourth; below, an arrow from PUSH down to the two vermilion
  cells it writes at the partner's end; a second, fainter pair one push later; at the right edge the pointer arrow
  crossing into the grey partner with the label "executes the partner's bytes".
- Option B, *the tautology as an equation*: three boxes "instruction", "operand", "written bytes", each containing the
  same two bytes, joined by "=" signs; minimal, abstract, memorable; loses the spatial story.
- Option C, *A + B combined*: the strip with the equation as an inset.
- **Recommendation: C**, equation inset at the top-right; the equation is the sentence of the paper.

## Fig. 2d — the closers, with the cycle marked

Must show: four different closed genomes that share one feature, a control transfer that keeps the pointer inside the
organism; the pusher's open flow for contrast.
- Option A, *five strips*: pusher (pointer arrow exits to the right, "open"), `RET NZ` (L = 16), `JR NZ` (L = 50),
  `DJNZ` (L = 36), `LDIR` (L = 20), each a strip of its bytes with the executed path drawn above as a thin line and the
  backward jump as a vermilion loop arrow; the LDIR strip shows a tight loop on one cell ("repeats in place").
  Disassembly in 5 pt under each strip; "copies 1.00 / damages 0.00" at the right.
- Option B, *one strip per mechanism class*: jump-back, return, repeat-in-place; fewer strips, less convincing about
  convergence.
- **Recommendation: A**, five strips, because the point is that four unrelated instructions do the same thing.

## Fig. 4d — the classification

Must show: two binary properties of a substrate, four cells, what we observed in three of them, and the fourth as the
open cell; also that the axes are architectural, not chemical.
- Option A, *a 2 × 2 table-figure*: columns "tar benign" / "tar lethal", rows "literal write channel" / "none"; in each
  cell a tiny strip icon (open organism flowing into the partner; closed organism with a loop arrow; a grey strip with a
  halting mark) and the outcome in words: "open → closed (Z80: 40/40 at 300k)", "open → extinct (BFF + literal:
  12/12)", "born closed (BFF: 28/28)", "? (BFF, no literal, lethal tar — not run)" — the benign-tar cell updates with
  today's result.
- Option B, *a phase diagram*: x = lethality of tar (continuous), y = literal bandwidth β; regions shaded; our four
  systems as points. Prettier, but we have only binary knowledge of both axes; a continuous diagram would overclaim.
- **Recommendation: A**, honest and readable; keep the icons tiny and consistent with Fig. 1a.

## Fig. 5a — Theorem 2, open vs closed trajectories

Must show: the executed-address sequence of an open organism leaves its bytes; a closed one revisits an address; the
bandwidth bound (less than one byte written per byte executed) is why confinement forces a repeat.
- Option A, *two pointer traces*: x = time (instruction count), y = address along the strip (organism's bytes shaded);
  the open trace is a straight diagonal that leaves the shaded band at the organism's end and keeps going; the closed
  trace is a sawtooth that returns at the loop; a second y-axis or marks show "bytes written" accumulating; the closed
  trace writes the full L bytes only because it repeats.
- Option B, *a flow graph*: nodes = addresses, edges = executed transitions; the open graph is a path, the closed graph
  contains a cycle. Clean and formal; less tied to the strip vocabulary.
- **Recommendation: A** (traces), with the written-byte count as small ticks on the trace; it makes Theorem 2 visible.

## What I need from you

1. Agreement on the shared vocabulary (strip, arrows, vermilion = written/closed, grey = partner/tar).
2. Choice per panel (A/B/C) or your own idea.
3. Whether Fig. 4d should include the benign-tar cell's result (it will be in `results/bff/FINDINGS.md` within the hour).
Then I build them as SVG in matplotlib (same style module), 180 mm wide where they share a figure, and we iterate on the
rendered PNGs.

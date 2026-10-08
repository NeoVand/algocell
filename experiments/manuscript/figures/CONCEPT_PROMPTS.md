# Content prompts for the five conceptual panels (2026-10-08)

Each prompt describes **what the picture must contain**, in plain language, for someone (or an image model) who knows
nothing about the project. No style is specified. Paste the shared context first, then one panel prompt.

## Shared context (prepend to every prompt)

We study a computer simulation of how life could begin. There are 20,000 tiny programs. Each program is a row of 16
memory cells, and each cell holds one byte, written as two hexadecimal digits such as 01 or c5. The programs are written
for a classic 8-bit processor, the Zilog Z80. Over and over, the simulation picks two programs at random, called the
organism A and the partner B, puts them side by side in one shared memory of 32 cells, and runs the processor for 128
instructions starting at A's first cell. The 32 cells form a ring: the position after the last cell is the first cell
again. Two pointers matter. The instruction pointer marks the cell being executed and normally moves to the right, one
instruction at a time. The stack pointer marks where "push" instructions write; it starts at the last cell of B and
moves to the left by two cells with every push, so pushes fill the partner from its far end backwards. After the 128
instructions both halves are put back where they came from. Nothing is scored or selected: a program persists only if
its bytes keep getting written over other programs. At first, executing random bytes just scatters junk; after a while
some pattern starts writing copies of itself, and that is the moment life begins in the simulation.

---

## Panel 1 (Fig. 1a) — the pair, the ring, the pointer and the stack

Draw a single horizontal row of 32 equal square cells. The left 16 cells are the organism A and the right 16 are the
partner B; the two groups must be told apart at a glance (for example by how they are filled or outlined), with a
bracket above each group labelled "organism A" and "partner B". Below the row, a thin curved line joins the right end of
the row to the left end, labelled "a ring: after the last cell comes the first". Above the leftmost cell, a small arrow
points down at the cell, labelled "instruction pointer: starts here, executes one instruction at a time, moves right";
from that arrow a dotted line runs to the right along the top of the row to show its direction of travel. Below the
rightmost cell, a small arrow points up at the cell, labelled "stack pointer: starts here; each push writes two cells
and moves two cells left"; from that arrow a dotted line runs to the left along the bottom of the row. Under the whole
drawing one line of text: "128 instructions, then both halves are written back".

## Panel 2 (Fig. 1b) — the first replicator: a two-byte word that is at once instruction, data and what gets written

This panel shows the very first self-copying program that appears in these simulations, in nearly every world. Draw the
same 32-cell row as in Panel 1 (organism A on the left, partner B on the right). Fill the 16 cells of A with the bytes
01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 — the two-byte word "01 c5" repeated eight times. Fill the 16 cells of
B with arbitrary bytes (for example ff e4 22 79 f3 bd 06 83 66 a8 52 c1 bb 96 51 f3), except as described below.

Above A's first three cells (01 c5 01) draw a bracket labelled "LD BC,nn: load the next two bytes into register BC".
Above A's fourth cell (c5) draw a bracket labelled "PUSH BC: write BC's two bytes where the stack pointer is". The
processor reads the three-byte instruction, so the two bytes after the opcode, "c5 01", are its operand; the push then
writes exactly those two bytes, "c5 01", into memory. Show this with an arrow that starts at A's fourth cell, curves
down and to the right, and ends on the 30th and 31st cells of the row (the second- and third-from-last cells, which lie
in B), and make those two cells read c5 01, visibly marked as just written. The 32nd cell, the very last one, keeps its
arbitrary byte for now. A second, fainter arrow starts at A's eighth cell (the second push) and ends on the 28th and
29th cells, which also read c5 01 and are marked as just written. Because every push lands two cells further left, the
partner fills from its far end with the same alternating pattern, lined up with the organism's own.

At the boundary between A and B, draw the instruction pointer's arrow passing from A's last cell into B's first cell,
with the label "the pointer runs on into the partner and executes the partner's bytes". This is the point of the panel:
the program has no loop and never checks its surroundings, so its pointer simply keeps going into foreign code.

In the top right, an inset of three boxes side by side: a box labelled "the instruction" containing 01 c5 01; a box
labelled "its operand" containing c5 01; a box labelled "what the push writes" containing c5 01; an equals sign between
the second and third boxes. Under the inset one line: "the tape is one two-byte word repeated; the code is its own
data, and what it writes is itself".

## Panel 3 (Fig. 2d) — four unrelated instructions that close the loop, and the open original

Background for this panel: a program is called *open* if, while it runs, its instruction pointer leaves its own cells
and executes the partner's bytes (the first replicator above does this). It is called *closed* if the pointer stays
inside the program's own cells for the whole run. Staying inside is only possible if the program contains an
instruction that sends the pointer backwards to one of its own earlier cells, forming a cycle. In different simulated
worlds, four unrelated instructions ended up doing exactly this, and every one of the resulting programs copies itself
perfectly into any partner. The picture shows five programs as five horizontal rows of cells, one above the other, with
the pointer's path drawn as a thin line above each row.

Row 1, labelled "open — the first replicator (16 cells)": cells 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5. The
path line runs from the first cell straight to the right, past the last cell, and continues over a few extra cells
drawn after the row and marked "partner". Label at the right: "no backward jump: the pointer leaves the organism".

Row 2, labelled "closed with RET NZ (16 cells)": cells ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0. The two cells
reading c0 (cells 6 and 8, counting from 1) are the instruction "return if not zero", which takes a two-byte address
from the stack and jumps there; the instruction e3 (cells 2, 4, 10, 12) swaps the processor's HL register with the two
bytes at the stack, which is how this program writes its bytes out and also how it plants the address it will return
to. Draw the path running right from cell 1 and two curved arrows above the row: one from cell 6 back to cell 4, one
from cell 8 back to cell 1. Label at the right: "the bytes it writes are also the addresses it returns to".

Row 3, labelled "closed with JR NZ (50 cells)": a row of 50 cells filled with the alternating pair 21 e5, except that
the pair 20 f0 appears three times, fourteen cells apart, at cells 10–11, 24–25 and 38–39 (counting from 1). The pair
20 f0 is "jump back 16 bytes if not zero". Draw three arrows above the row: a curved arrow from cells 24–25 back to
cell 10 (the backward jump); a curved arrow from cells 10–11 forward to cell 32, drawn differently from the first (for
example dashed), because here the same backward jump underflows the processor's 16-bit address and lands forward on
the row; and a dashed arrow from cell 36 back to cell 1, where the 16-bit address wraps from 65535 to 0. The pointer
therefore cycles over cells 1–39 and never reaches cells 40–50 or the partner. Label: "a relative jump of 16 bytes
back".

Row 4, labelled "closed with DJNZ (20 cells)": cells 21 e5 21 e5 21 4e 10 e5 21 e5 21 e5 21 e5 21 4e 10 e5 21 e5. The
pair 10 e5 (cells 7–8 and 17–18) is "count down and jump back 27 bytes if the count is not yet zero". Draw a curved
arrow from cells 17–18 back to cell 8, and a dashed arrow from cell 16 back to cell 1 (the 16-bit address wrap). The
pointer cycles over cells 1–18. Label: "a counted loop".

Row 5, labelled "closed with LDIR (20 cells)": cells 1e a4 ed b0 1e a4 ed b0 1e a4 ed b0 1e a4 ed b0 1e a4 ed b0. The
pair ed b0 is a single instruction that copies a block of memory byte by byte and repeats itself in place until done.
Instead of a backward arrow, draw a small tight loop arrow sitting on the first ed b0 (cells 3–4). Label: "a hardware
loop: the instruction repeats on the spot".

To the right of rows 2–5 write "copies 100% of partners, damages itself 0%". To the right of row 1 write "copies about
two thirds of partners, damaged in a third of encounters". Below the whole panel one line: "closure is a cycle
in control flow, not a wall around the bytes".

## Panel 4 (Fig. 4d) — a two-by-two classification of worlds

Background: we ran the same kind of soup on two different machines. One is the Z80 described above. The other is a
tiny language called BFF, whose programs are 64 cells long and whose instructions are the eight symbols < > { } + - . ,
plus brackets [ ] that form loops. Two properties of a machine turned out to decide how life begins in it.

Property 1, "has a literal-write instruction": an instruction that writes the bytes of its own operand into memory
(like PUSH after LD above). With it, a program that is nothing but one such word repeated copies itself without any
loop, so life starts *open*. Without it, the smallest self-copier needs a loop, so life is *born closed*.

Property 2, "the tar is benign or lethal": "tar" is the inert junk that random execution fills memory with before life
appears. On the Z80 the junk is zero bytes, which do nothing when executed, so a program whose pointer runs into junk
keeps going: the tar is benign. In BFF the junk contains unmatched brackets, and running into one halts the program:
the tar is lethal to any open program.

Draw a table of two rows and two columns with a title "how life begins, by machine". Column headings: "tar benign" and
"tar lethal". Row headings: "has a literal-write instruction" and "has none". In each cell draw a tiny row of cells as
an icon plus a short text:

- Top left (literal, benign): two entries stacked. First: an icon of a row whose pointer arrow leaves the row to the
  right, then an icon of a row with a backward loop arrow, joined by an arrow, text "Z80 soup: open first, then closed —
  every one of 40 worlds". Second: an icon of a row whose pointer leaves to the right with a repeat symbol next to it,
  text "BFF with a literal-write instruction and harmless brackets: open for ever, no closed design exists — 12 of 12
  worlds". Under both: "which of the two happens depends on whether the machine offers a jump that the organism can
  copy along with itself".
- Top right (literal, lethal): an icon of a row whose pointer leaves to the right, followed by a bar across the row
  meaning death, text "BFF with a literal-write instruction: open first, then extinct — 12 of 12 worlds".
- Bottom right (none, lethal): an icon of a row with a backward loop arrow, text "BFF as published: born closed —
  28 of 28 worlds that produced life".
- Bottom left (none, benign): no icon, text "not run".

## Panel 5 (Fig. 5a) — why a closed self-copier must contain a cycle (a theorem, drawn)

Background: the theorem says that a program which writes fewer bytes than it executes cannot copy all of its own cells
while its pointer stays inside those cells, unless the pointer visits some cell twice. So every closed self-copier
contains a cycle, and a self-copier without a cycle must be open. The first replicator writes two bytes for every four
cells it executes, so if it stayed inside its 16 cells without repeating it could write at most 8 bytes.

Draw two charts side by side with identical axes. The horizontal axis is time, labelled "instructions executed", from 0
to about 24. The vertical axis is "memory address", from 0 at the bottom to 32 at the top, with a shaded horizontal band
from 0 to 16 labelled "the organism's own cells" and the unshaded part above labelled "the partner's cells".

Left chart, titled "open: the first replicator". One line, labelled "instruction pointer", starts at address 0 and
rises in a staircase: up 3 cells, then up 1 cell, then up 3, then up 1, and so on, one step per instruction. After 8
instructions it reaches 16, crosses out of the shaded band and keeps rising into the partner's cells. A second line,
labelled "stack pointer", starts at address 31 and descends 2 cells on every second instruction; put a small mark on it
at each descent, meaning "two bytes written". By the time the first line leaves the band, the second line has made only
4 marks (8 bytes written out of the 16 that make a copy).

Right chart, titled "closed: a program with a backward jump". The instruction-pointer line rises inside the shaded band
and, before reaching 16, drops back down to an earlier address, rises again, drops again: a sawtooth that never leaves
the band. The stack-pointer line descends with marks as before and, because the sawtooth keeps running, the marks
continue until more than 16 bytes have been written.

Under both charts one line: "fewer than one byte written per cell executed: to copy all its cells a program must either
leave them (open) or come back to one of them (a cycle)".

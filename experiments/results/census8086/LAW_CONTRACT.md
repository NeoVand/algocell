# Frozen constructed REP law controls

Frozen after the geometry traces and before the payload assays, 2026-10-12.
This is the constructed control's conditional geometry, not a population prediction.

Code: `BF dLo dHi 49 F3 A4` (MOV DI,d; DEC CX; REP MOVSB), and the separately
verified word geometry `BF dLo dHi 49 F3 A5` (REP MOVSW). Both have six core bytes.
`law_eligibility.csv` and `law_geometry_traces.csv` bind the trace checks. All
340 rows qualify: five (L,d) pairs, two operations, two budgets, 17 partners.

Eligibility is determined from code and trace: zero initial DS/ES/SS/CS and SI,
DF=TF=0; CX=0 at reset so DEC makes FFFF; 6 <= d <= L, d divides L, d even for
MOVSW; forward source and destination, constant byte displacement d; a d-periodic
baseline parent with core at entry phase 0; preserved copied core at offspring
phase 0; no intervening operation, no original partner reads and enough writes
to cover the entire partner. Budget counts every REP iteration. REP holds its
decode and runs for budget−2 copying iterations, below CX exhaustion. Mutations
are applied only after this eligibility is determined; the core is held fixed.

The first d bytes are the seed block. Writes at source+d propagate that block
through both tapes, before any later source read. Exactly positions 6..d−1
in the primary seed are free payload coordinates: d−6, independent of outcomes.
For L=16, d=8/16 gives 2/10; for L=32, d=8/16/32 gives 2/10/26. Changing only a
primary payload byte must propagate that value to every offspring coordinate
congruent to it modulo d. Initial positions d..L−1 are overwritten before read.
The offset bytes and all core bytes must be copied exactly. Offspring entry is
phase 0 because d divides L. No similarity threshold substitutes for this test.

Assays (`census8086_law.py assay`): every one of 256 values separately at every
predicted coordinate (255 alternatives plus its baseline value), 256
counterbalanced full-payload assignments and 256 fixed random joint assignments,
each against zero and the same frozen 16-partner panel at both budgets. Joint
assignments test combinations; they do not exhaust 256^(d−6) combinations.
Mutate each initial coordinate d..L−1 through all 256 values as erased-position
controls, against the same panel. Core and displacement inheritance are checked
on every descendant. All single and joint cases are serially passed to g=8,
with state reset and a fresh counter-keyed random partner at each generation.
The serial stream is independent of the frozen census panel; no partner is
initialized with payload. Every coordinate must match the expected repeated
primary seed block; a mismatch stops the assay and is diagnosed, not averaged.

Seeds and generation keys are fixed in `census8086.py` and `census8086_law.py`.
No evolved eligibility, route prediction or population trajectory is assessed.

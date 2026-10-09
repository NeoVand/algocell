# Head-to-head invasion of the matched L = 50 pair (2026-10-08 night; pre-registered in `REVISION_PREREG.md`, I)

Ancestor: the pusher tiling `01 c5` × 25. Successor: the same tiling with `20 f0` (JR NZ, −16) inserted three times (the
final dominant of 14 of 20 Stage G worlds at L = 50); the two differ at 6 of 50 positions. Soups of 20,000 cells, Stage G
dynamics (8,192 pairs, 128 instructions, mutation 1/16, square lattice). Resident = every cell one tape; invader = 1% of
cells the other tape at step 0; 5 seeds per direction; controls with no invader, 3 seeds per resident. Script
`invasion_pair.py`; table `invasion_pair.csv`; report `REPORT.md`.

## Outcomes

- **Pre-registered metric (share of cells within Hamming 4, later 13, of the invader's tape, nearest class) failed on a
  definitional flaw, not on the biology**: at L = 50 the quasispecies cloud splits between the two classes (a closed
  tape that has lost two of its three jumps is nearer the pusher), so the registered share never passes 50% (0.42 at
  20,000 steps) even when every cell descends from the invader. Reported as a miss of the metric.
- **Post hoc metric, the share of cells carrying the jump word `20 f0`** (defined after seeing the data): the closed form
  seeded at 1% into a pusher-filled world passes 50% of cells at step 110, 120, 120, 130, 130 (5 of 5) and settles at
  0.73–0.75; about 40 encounters per cell.
- **The pusher seeded into a closed world leaves no trace**: its core share at 20,000 steps (0.159–0.163) equals the
  closed world's own mutational cloud without any invader (0.160–0.166); the jump-word share stays at 0.73–0.74.
- **The closer arises on its own in a saturated open world**: in the three pusher-only controls the jump word appears in
  more than 1% of cells at steps 2,750, 3,750 and 7,500 and passes 50% at steps 5,000, 7,500 and 13,250 (3 of 3), against
  a Kaplan–Meier median of order 10⁵ steps from a random start in Stage G. This is Model 5's window with n(t) at
  carrying capacity from step 0.

## Reading

The jump itself is selected, in a world made of its ancestor's copies (kin), not only among strangers: the partner-test
advantage (1.00 against 0.69 copies; 0.00 against 0.33 self-damage) is a competitive advantage of about one doubling per
few encounters. With the mutational scan this gives the two sides of closure in one matched pair: the jump wins the
competition and loses the variation channel (23 → 1 transmissible sites).

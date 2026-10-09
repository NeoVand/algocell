# Round-2 review, second report (Synthesis, Pessimist, Optimist; 2026-10-09): assessment and plan

Source: `~/Downloads/paper-review-round-2.md` (untrusted; read only, copied to the scratchpad). It reviews revision 1, as
the first report did. Recommendations: Synthesis major (narrow), Pessimist major, Optimist minor. Most text points
coincide with the first report and are already fixed in the revision-2 draft (8080 naming and the false impossibility
claim, "counting fixes the order", "theorem rather than observation", Model 5 numbers, Theorem 1 quantifiers, "canalised",
"inherits nothing", "pushed again by the copy", the BFF cell, the boundary paragraph, variation from one tape, L = 50 in
the summary). The round-2 tests already run answer several of its requests: serial retention with the
lethal/erased/kept split (Optimist landmark 1b), population-level classes (landmark 1a, Synthesis optional 2), the
invasion marker, the 8080 counterexample.

## Verified here before acting

1. **The pusher's working unit is four bytes in phase.** `01 c5 01 c5` executes LD BC,0x01c5; PUSH BC. As a tape the
   pusher is a period-2 tiling, but the unit that does work is as long as the closer `1e a4 ed b0`; so Fig. 6b's
   0.99 against 6 × 10⁻⁵ comparison does not show a head start, and "exponentially higher probability" does not follow.
   Accepted: Fig. 6b and Proposition 4 are reframed around a census of self-writers by period (period 3 added) and the
   measured head start (removing the stack writers delays life 160- to 992-fold).
2. **The L = 16 return closer works through address folding and zero registers** (0x21e3 mod 32 = 3, mod 64 = 35;
   XOR L on zero registers sets the flag its RET NZ tests). "The return and the block copy need nothing but the
   instruction set" is false. Accepted; the convention test is now required by all three reviewers.
3. **Statistics quoted.** Mixed pairing 7/10 against 20/20: Fisher P ≈ 0.03 (to be computed); the McNemar test is
   uninformative (a loop-free first replicator can only gain a loop): deleted.
4. **Ref. 23 first author** is C. Knierim (as both reports say; to be fixed in the reference list).

## New work (registered in `REVISION_PREREG.md`, R3, before running)

- **C, conventions.** Derived shaders with random initial registers (all but PC and SP) and, separately, a random
  initial stack pointer, drawn per encounter. 20 worlds each at L = 16, Stage G conditions, 300,000 steps, local GPU.
  Partner tests of every Stage G and K first and final tape under random registers (no new soups).
- **B, invasions at aligned lengths.** L = 16 (return closer against pusher) and L = 32 (Stage K block-copy tiling
  against pusher), both directions, five soups each, three controls each, functional endpoints (classes of random cells).
- **T, first-closure timing.** From the recorded samples: the first top-3 tape (≥ 0.5% of cells) that is heritable and
  confined; Kaplan–Meier curves for benign L = 16 (Stages G, L), lethal L = 16 (I), L = 32 (K); log-rank tests.
- **P, census by period.** All 16,777,216 period-3 words tiled to 16 bytes against the zero partner: self-writers,
  classified open or closed (period 4, 4.3 × 10⁹ words, if time allows).
- **F, ref. 22's robustness test** on the Stage K tapes (1, 4 and 8 successive mutations, exact copy into a zero partner,
  512 instructions), next to our single-mutant scan of the same tapes.
- **The L = 50 mixture**: what the jump-free quarter of cells is (post hoc, descriptive).

## Not done (stated in the response)

Writer tags (lineage) at L = 16 and 32; a non-mirrored memory and true-ring re-execution (the trace already matches the
wrap arithmetic; folding is ref. 16's convention at aligned lengths); a real 8080 with its alternate opcodes; the Z80
lethality dial; a functional reward for unexecuted bytes; soup-sampled partners for the partner test (done for serial
retention only); a third-party time stamp (the dated commit history is archived instead, as the Synthesis advises).

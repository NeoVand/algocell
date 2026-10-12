# Claude, Round 2 — response to ASTRA_R1 (2026-10-11 night)

## Concessions (Astra is right)

1. **Prior art.**
   - Agüera y Arcas et al. 2024 own the Z80 stack-to-block-copy succession and `01 c5`.
   - Cicala et al. 2026 own the load–push/block-copy succession and compact copy cores.
   - "Replication precedes individuality" cannot be the headline novelty.
2. **My "three selves coincide by theorem" (R1b) is false in general.**
   - BFF dissociates pointer confinement from inflow in both directions.
   - Regenerators are confined yet carry ≤ 2 sites.
   - I retract it. Individuality becomes a motivated question, not the claim.
3. **Chronology.**
   - The census informed S1's prediction; Proposition 9 came after S1 and explains it in retrospect.
   - Literal "parity" was wrong in general: the independent check found width-3 mixtures that self-write. The line
     is width 1 against ≥ 2, and only for periodic one-register writers.
   - The 6502 is a scope and generality result, not a triumph of the theorem.
4. **The conceptual distinction is not new.** Replication against evolvability goes back to Vasas–Szathmáry–Santos 2010
   and to von Neumann. Ours are the evolved mechanism, the exact law and the ecology.

## Where I push further (bolder, and I think still true)

**A. One principle, stated sharply.** *A replicator can pass on only what its reproduction does not depend on.*
- Formally, site i is heritable for every value if and only if the reproductive run is non-interfering in A[i]
  apart from verbatim copying: same control flow, same addresses, same stopping. This is your copy-dependency
  programme (#2) in one sentence.
- Consequences:
  - Inert sites are always heritable (Lemma 10).
  - An executed site is heritable only when it is functionally neutral for every mutant value. That is rare: Stage L
    seed 6010 has 3 sites, 0 of them unexecuted, so it happens.
  - **"All-machinery" replicators**, whose every byte steers the copy (the pusher: straight-line code fetches every
    byte, Lemma 11), have **zero capacity**. X1's 0 of 512 is then explained, not just observed. The 2024 paper saw
    the pusher; it did not show, or explain, that it inherits nothing.
- The principle reaches beyond machine code. In any self-reproducing system the heritable degrees of freedom are the
  copied ones the reproductive dynamics is independent of. This is von Neumann's "copy the description
  uninterpreted", derived as a necessity rather than a design choice, and it says why autocatalytic networks without
  such coordinates lack evolvability (Vasas et al.).
- **A dynamical restatement.** Reproduction is a map R on tapes, and replicators are its attractors.
  - A regenerator or pusher is a *point* attractor: perturbations decay back to the form.
  - A transmitter has a *manifold* of fixed points, with 256^(d − c) payloads that are neutral directions.
  - The capacity for heredity is the dimension of the replicator's neutral manifold.
  - The origin of heredity is the birth of a neutral manifold in R: a point attractor becoming a line attractor.

**B. The capacity for heredity is itself a heritable byte.**
- In the dominant copier family, d (and with it the capacity d − c) is set by one byte that is copied with the core.
  So capacity mutates, is inherited, and is selected: DZ3, transmitter majority under lethal tar, is met under C1;
  and the R/T ecology.
- This is the "evolution of evolvability" debate (Wagner & Altenberg 1996 and later) in its most concrete form: a
  single byte that sets how much else can be inherited, and that the environment visibly shifts.
- **Caveat (Astra's):** we do not know *why* the environment selects d. Selection may act on a correlate.
- **Claim:** capacity is a heritable, environment-sensitive trait. **Not claimed:** "selection favours
  evolvability".

**C. The arc.** We built an instrument (Zilion). It showed a soup full of excellent copiers that remember nothing
(X1). One principle explains why: inheritance lives only in what reproduction does not depend on. That principle
predicts the capacity exactly (the overlap law). Capacity is a one-byte heritable trait that the environment shifts.
Three machines bound the reach:
- Z80: zero-capacity copiers first.
- 6502: closed copy loops from the start; capacity on evolved 6502 copiers is still unmeasured.
- 8086 (added by the user): predicted open zero-capacity pusher first, then REP MOVSB copy-offset cores.
  **Prospective test:** the overlap law with the 8086's own core length c, on evolved copiers. That is the
  prospective law test you asked for, on a new machine.

## Answers to Astra's questions (short)

1. **Strongest sentence:** *"Spontaneously evolved replicators range from copiers that inherit nothing to copiers that
   inherit exactly d − c bytes; inheritance is confined to what reproduction does not depend on, and the one byte
   that sets d is itself inherited and shifted by the environment."* Falsified by: a straight-line open copier that
   retains arbitrary alleles; a qualifying core retaining beyond d; or the 8086's evolved REP cores violating d − c.
2. **Persistence, confinement, partner-independence and retention are now kept separate.** The only implications I
   use are: no dependence ⇒ heritable (always), and straight-line open ⇒ every byte is a dependence (Lemma 11).
3. **The R/T ecology** is the opening of the next question, not the climax. The climax is the law and its prospective
   test.
4. **Your Fine–Wilf correction is correct.** It is confirmed independently (2w + 1 for odd w). For arbitrary
   mixtures: no periodic width-1 writer (proved, one register; searched to k = 64); width-3 mixtures do self-write.
5. **Biological novelty:** the evolved mechanism, the exact law, capacity as a heritable byte, and the dependence
   principle as a general, testable criterion. Not the distinction itself.
6. **The 6502** is a scope result: a prospective census-guided prediction about the route, honestly reported. It is
   not the lead.
7. **Knierim:** agreed. Our genealogies show what happened, not that encounters are better search.
8. **Remove from the main text:** the size/ablation atlas, the detector benchmarking, and the full genealogy (one
   ED figure).
9. **The payload without selected function** is still compelling as *capacity* (the channel exists and is
   inherited). Smallest experiment to show use: a neutral-tag lineage-tracing run. A functional payload is the next
   paper.
10. **The partition between local mechanism and ecological reruns:** keep it explicit. I1-T2 is now scored. All
    claims stand except DZ2 and DZ3 (not met → met) and an L2 shift, with the re-run attribution limit stated.

## For the joint charter

Center: your Architecture A, sharpened by A and B above. The 8086 becomes the prospective law test as well as a
route test. Zilion is introduced in this paper (the user's decision; no separate tool paper), with the semantic
contract as method.

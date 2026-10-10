# Night of 9–10 October 2026: results

Each item gives the pre-registration (REVISION_PREREG.md), the outcome and the source table. Nothing here has been put
into the manuscript.

## N1 — the toxic payload (rejected)
*Hypothesis:* bytes a transmitter never runs are selected to halt intruders (0x00 under lethal tar, 0x76 under both).
*Outcome:* rejected. Payload zeros are inherited scars of stack damage: they rise towards the end of the payload (to
0.32–0.38 at the second-last byte at L = 32) with an even–odd pattern, are as common under benign as under lethal tar
(L = 32: 0.067 against 0.052), and are absent without mutation. 0x76 is not enriched. Removing the halting bytes lowers a
host's survival as an intact partner by about 0.001 per encounter (registered threshold 0.02). Sources:
`results/toxin/COMPOSITION.md`, `results/toxin/CAUSAL.md`.
*What it gives the paper:* transmitters inherit the record of the damage done to them; regenerators erase it.

## N2 and N4 — why the tar chooses between regenerators and transmitters (mechanism found; both registered
hypotheses rejected)
*Registered hypotheses, both rejected:* backup cores (N2-2: a regenerator with a zero in its first core never copies
itself under either rule) and inherited zeros as a shield (N4: removing them keeps ~90% of the effect).
*What the demography shows* (`results/n2/BACKGROUND.md`; 40 runs of 2,000 steps, every encounter recorded):
- The rule, not the soup's composition, sets the direction: a benign soup moved to the lethal rule gains transmitters
  (+0.039 in 2,000 steps), a lethal soup moved to the benign rule loses them (−0.049).
- Regenerators and transmitters copy each other at exactly balanced rates. Everything happens in their encounters with
  the broken background (core-free cells, 11–16% of the soup):
  - a regenerator's body, eight copies of its core, is a trap: an intruder that runs into it is captured in the LDIR
    loop, destroys it (0.86–0.89 of entries) and is sometimes converted into a regenerator itself (≈ 3 × 10⁻⁴ per
    capita per step, under both rules);
  - a transmitter's body is a maze of junk: intruders wander, and under the lethal rule a zero halts them (destruction on
    entry 0.58 → 0.48);
  - the lethal rule therefore protects transmitters more than regenerators, and the summed terms predict the observed
    drifts (+0.031 against +0.039; −0.059 against −0.049).
*Reading:* the environment chooses through the bodies' response to broken neighbours: a regenerator's redundancy
captures intruders, a transmitter's junk loses them, and lethal by-products decide whether the lost ones survive.

## N3 — the line of descent of the first self-confined replicators (running on Modal)
Validation: exact replay of 795,934 recorded encounters (0 mismatches), accounting within the mutation count, and all
64 sampled confined copiers in a seeded world traced to the seeded closer. `runs/lod/validate/validate.json`.

## Literature check (`lit_review/VERIFICATION.md`, primary sources)
- Cicala et al. 2026 (arXiv:2607.09211, v2 2 Sep) report the Load–Push → LDIR takeover in Z80 soups, also without task
  pressure, and explain it by mutational robustness. Their v2 appendix lists our `XX 5e … ed b0` motif with byte 0 as
  the "partner offset" (never treated as a switch or counted). Their ancestry tracking follows niche labels only: whether
  LDIR replicators descend from Load–Push ones is open, which N3 answers.
- Agüera y Arcas et al. 2024: every passage cited by the reviews is confirmed (Z80 stack copiers → LDIR/LDDR, the 8080
  long-tape result, zero-poisoning, the Forth non-functional head, the "complex rewrite event").
- Knierim et al. 2026 exists; random search beats paired interaction only with tuned byte distributions.
- 8080 aliases (CB = JMP, D9 = RET, DD/ED/FD = CALL) confirmed from a die-reconstructed decoder and MAME; Intel's manual
  lists them as undefined. Our ablation should be called "8080-like" unless it implements them.
- New: Jha et al. 2026 (arXiv:2609.10817), a Z80 soup with an energy budget; no replicator taxonomy.

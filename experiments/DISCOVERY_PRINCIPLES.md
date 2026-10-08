# When may we say we have discovered something? Principles, and how far each claim is from them (2026-10-08)

Written at the user's request, to be read before any claim is strengthened. The principles are standards for us, not
for referees; a claim that fails one is not wrong, it is not yet a discovery. The audit below is deliberately harsh.

## Principles

1. **Pre-registered, then replicated.** The statement, its measure and its kill criterion were written before the data
   that test it existed, and it held in a confirmatory stage with new seeds. A finding first seen in the data is a
   hypothesis until this has happened, however many worlds show it.
2. **Measured by intervention against a ground truth that is not our own convenience.** Where the ground truth is a
   definition of ours (the culture test, t_rep, "closed"), the result must be shown to survive every reasonable change
   of the definition's parameters, and the parameters must be reported with it.
3. **Mechanism traced to the machine.** We can execute the exemplar and show the instructions that produce the effect,
   not only the statistic that summarises it.
4. **Beyond one substrate.** A second machine shows the same thing, or a theorem whose hypotheses the machines satisfy
   predicts it, with at least one boundary case where the hypothesis fails and the phenomenon changes as predicted.
5. **Rivals killed, not ignored.** For each claim the simplest alternative explanations (an artefact of the definition,
   the horizon, the sampling schedule, GPU non-determinism, spatial structure, the mutation model) were tested, and the
   tests are on record.
6. **Every number from a generated table, every figure showing the data it claims, every error logged.** The sentence
   "pairing is random and non-spatial" survived two drafts; the log of how it was caught is part of the evidence.
7. **Reproducible by strangers** from the repository: conditions, seeds, code, raw run files, and the order in which
   things were done.
8. **Stated at the strength the evidence allows**, including what it does not show.
9. **Surprising and predictive.** A discovery says something a knowledgeable person would not have guessed, and it
   predicts a further observation that has been, or can be, tested.
10. **Survives a reader who wants it to be false.** An adversarial pass has been made and its objections answered or
    conceded.

## The claims, audited

Grades: **A** meets the principle; **B** partly; **C** not yet. Distance: how far from "discovery" by these standards.

| claim | 1 pre-reg | 2 ground truth | 3 mechanism | 4 beyond one substrate | 5 rivals | 9 surprise | distance |
|---|---|---|---|---|---|---|---|
| C1 Sterile order (tar) precedes life and fools compression and assembly detectors | A (flood); B (detectors, post hoc) | C: the ground truth is our culture test | A (zero writes traced) | B (BFF shows the same flood) | B (detector AUC grey zone for assembly) | A | small: test the culture-test thresholds |
| C2 The first replicator is an open two-byte literal | A (emergence, modal tape) | B: t_rep's 0.5% share rule makes "first" a clonality criterion; analysis A showed the pusher is modal before t_rep in 59/80 worlds | A (trace) | A (BFF literal switch) | B | A | small: redo under alternative t_rep rules on existing data |
| C3 Closure evolves as a control cycle | A at L = 16, 50 (Stage G); misses at L = 20 (syntactic criterion), incomplete at L = 64 by 300k | B (our closure definition; now also inflow) | A (RET NZ, JR NZ, DJNZ, LDIR traces) | A (theorem); B (BFF never closes by control flow, consistent) | C: spatial structure untested as a driver | A | moderate: Stage H (mixing); longer horizons at L = 64 or an explicit "in progress" statement |
| C4 Three load-bearing instruction families | A (seed-paired, confirmatory) | A | A | C (Z80 only) | A | B | small; modest in importance |
| C5 Information inflow 7 → 0 bits across the transition | B (analysis pre-registered today on existing data; Q4b killed) | A (an identity, I = H under determinism; intervention) | A | B (BFF measured; flooding boundary found) | B | A | small: replicate on new seeds (Stage H serves) |
| C6 Life begins at minimum complexity; assembly fires on tar | B (P1 met but unfalsifiable as written; P2 met; P3 grey) | C (our ground truth; Re-Pair bound) | A | C | B | A | moderate: a proper detector comparison, the published assembly algorithm, alternative ground truths |
| C7 Theorem 2 and the literal channel; one instruction switches the open beginning | A (BFF cells pre-registered, with logged incidents) | A | A | A (theorem, two machines, boundary case) | B (write-ratio counting in BFF to be stated) | A | small: state the saturation route in the theorem |
| C8 Tar lethality decides how the open phase ends | B (follow-ups pre-registered after the first result) | B | A | C: lethal tar exists only in BFF | C | A | moderate: a lethal-tar Z80 variant (zero halts) |
| C9 Openness is cheap among kin; space sustains the open phase | A (Stage H, pre-registered, both predictions killed) | A | A (kin 57% vs 11%; damage 2.5% vs 34%) | C | A | A | dead as a hypothesis; the null (order independent of space) is a result; the reliability-of-closure reading is new and untested |
| C10 Heredity without clones at 64 steps; error-threshold curve | B (post hoc reading of a pre-registered cell) | B | B | C | B | A | moderate; supplementary unless replicated |

Principles 6, 7, 8 and 10 apply to the whole: 6 is met with one logged failure; 7 is met for the code and runs but the
manuscript's prose is still draft v2; 8 is met in the documents and not yet in the manuscript; 10 has been applied to
figures, not yet to the argument.

## Where we are

We have a strong, pre-registered, reproducible phenomenon in one real instruction set, a theorem with a tested
boundary, a second machine in which one instruction switches the phenomenon on, and, since today, two measurements in
biology's own currencies (information inflow; assembly). By the standards above the digital result is close to a
discovery; the biological reading is a set of predictions, and the manuscript says so. What stands between us and the
word "discovery" without qualification:

- **Definitions.** The central claims depend on t_rep, the culture-test thresholds (0.75 copy, gen2 0.3) and the
  closure criterion. Re-run the Stage G analysis under a grid of these parameters, on existing data, and report the
  envelope. Cost: a day, no GPU.
- **Space.** Tested (Stage H): the open phase does not need kin and closure is not hastened by mixing; the order of
  events is independent of spatial structure. What remains open is whether space makes closure reliable (post hoc).
- **Lethal tar in the first machine.** The classification's lethal column rests on BFF alone. A Z80 variant in which a
  zero byte halts execution would put the first machine in that column. Cost: a shader flag and ten local worlds.
- **Horizon.** Closure at L = 64 is in progress at 300,000 steps; either run to one million or say "in progress".
- **Strangers.** No one outside this project has run the code. A clean repository and a preprint are the test.

## Ambition, stated so it can fail

If Stage H shows a shorter or absent open phase under mixing, the claim becomes: *individuality is invented at the
frontier, where an open replicator meets strangers*, which links the origin of selves to spatial structure and makes a
prediction for any surface-bound chemistry. If the lethal-zero Z80 goes extinct after an open beginning, the
classification becomes a two-machine law. If the definition grid leaves the order of events untouched, the order is
robust. If any of these fails, the manuscript shrinks to what survived, and says so.

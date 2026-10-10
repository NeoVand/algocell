# Night of 9–10 October 2026: results

Each item gives the pre-registration (REVISION_PREREG.md), the outcome and the source table. Nothing here has been put
into the manuscript.

## Headline (N3, 20 worlds recorded exactly, 17 closed) — corrected after the figure critic
**The first self-confined replicator in each world is born by recombination.** On the exact line of descent, the first
confined copier of each of the 17 closed worlds was completed by a recombination (16) or a copy of the partner (1), never
by a point mutation of a copier, a median of 79 steps after its line left the open-copier lineage, with only non-copying
intermediates in between (16 of 17), from bytes of a median of 3 separate events. Its parts are one-byte variants of the
pusher's own words (`21 e3`, EX (SP),HL; `21 e0`, RET PO): point mutation made the parts, recombination joined them.
Every founder's line runs back to the open pushers (descent, with the base-rate caveat that pushers dominated the soup).
Later founders in the same worlds (15) are mostly conversions by existing closers. Withdrawn after the critic: "bytes
written mostly by non-copiers" (it matches the soup's composition, 0.88 against 0.86) and the longer chain lengths (a bug
in the chain start). Sources: `results/lod/FOUNDERS2.md`, `founders2.csv`; figure to be revised.

## Spend
Modal tonight: $52.96 through 00:00 CDT (algocell-lod), about $55 in all. With the previous night's revision runs
(≈ $93, 9 Oct UTC) the revision has used about $150 of the $300 authorised. October's metered total for all algocell
apps is $441 (it includes the 7–8 Oct atlas and BFF stages).

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

Two measurement problems found and fixed on the way (both documented in REVISION_PREREG.md):
- The registered copy rule counted damage as copying in soups of near-identical tapes, so its founding events (C*) were
  confined loops that never copy, or open copiers (tested directly: copy rate 0.00–0.11 against random partners). The
  amended rule requires that a copy make at least L/4 bytes newly match the executor.
- Even under the amended rule the oldest confined record is a transient: confinement of pusher–return hybrids depends on
  the partner. The founder is now located by genotype (every ancestor classified against random partners) at the start of
  the final confined stretch.
First amended-rule world with the full analysis (31012): every line of descent of the closed population passes through
open copiers (pushers `01 c5` × 8, `21 e5` × 8). The first confined copiers on the line, `01 ed b0 58 01 ed b0 ed b0 ed b0
c5 01 43 01 c5`, copy with an LDIR at position 7 that is absent from the sampled open ancestors (10 bytes change between
sampled ancestors, so the founding encounter is not in the sample); a later confined form, `01 ed b0 58 f9 f0 …`,
returns into the operand of its own first instruction (`01 ed b0`, LD BC,0xb0ed) and executes those two bytes as LDIR
(instruction starts at 0, 1, 3, 4, 5: the code overlaps itself). The complete record of every ancestor (`lod_v5`) is
running so that the founding encounters can be read exactly.

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

Direct check of the conversion term (`results/n2/CONVERT.md`, 65,536 encounters per row): a core-free executor that
meets a regenerator becomes a regenerator in 0.5–0.9% of encounters (1.4–3.8% of those in which it enters the host); one
that meets a transmitter becomes a transmitter in 0.1–0.3%.

Methods note: the GPU soup is not bitwise deterministic (two identical local runs of 2,000 steps from the same seed
differ, through the atomic collision claim of the pair draw), so worlds reproduce statistically, not exactly, and a
world cannot be replayed to inspect an event after the fact. The N3 recorder therefore records every step as it happens.

### N3, functional parents (lod_v6) — first world (31002), provisional
The line now follows, at each recombination, the parent that supplied the bytes the new tape executes. In world 31002
all 64 sampled lines share one founder F, the return closer `3d e3 21 e3 21 e0 3d e0` × 2, made at step 2,457, 117 steps
after the last open copier on the line (the pusher `21 e5` × 8). Between them, 32 records: the line leaves the pusher
lineage into a cell that is written over again and again by many different neighbours (11 copies, 13 damage writes, 7
recombinations, 1 mutation); its writers carry `21 e3` (EX (SP),HL) and `21 e0` (RET PO) words and zeros. F's 16 bytes
come from 7 distinct events, 15 of them written by tapes that cannot copy themselves. The first self-confined copier was
assembled in a non-replicating cell from words circulating among pusher variants, not by point mutations within one
lineage. (`results/lod/CHAIN*.md`; to be checked on all 20 worlds.)

Exploratory landscape (`results/lod/LANDSCAPE.md`): a gradual route also exists in principle. Four point mutations
(PUSH → RET PO at positions 5 and 15, PUSH → EX (SP),HL at 9 and 1) lead from the pusher to a confined copier through
copiers only, with copying fidelity rising (0.59 → 1.00) and the pointer's entry into the partner falling (1.00 → 0.62 →
0.00). Whether each step is favoured is being tested (N5).

### N3, functional parents — five founders in three worlds (provisional; `results/lod/CHAIN_v6.md`, `PARTS_v6.md`)
- Every founder of a confined lineage was completed by a recombination (5 of 5), after a median of 48 records and 253
  steps since the last open copier on its line; its 16 bytes came from a median of 5 distinct events (2–7), mostly
  written by tapes that cannot copy themselves.
- Two families of founders: return closers (`XX e3 21 e3 21 e0 XX e0` × 2, the family of the paper's evolved return
  closer) and LDIR block copiers (`… 1e b0 … ed b0 …`).
- For the return closers, the words they execute (`21 e3`, EX (SP),HL; `21 e0`, RET PO) were already in 0.2–4.6% of all
  cells before the founder appeared (10–200 times a random tape), almost all of them in non-copiers. No cell carried the
  founder's whole program before it was assembled.
- Where the words come from (direct test): a pusher with one mutated operand byte (`21 e3` instead of `21 e5`) writes the
  new word into its partners about once per encounter, as data, among its own pusher words; its offspring are the pusher
  or variants carrying the new word, many of them non-copiers. Sloppy open copiers keep a communal pool of words; the
  first confined copiers were assembled from that pool by recombination in a cell that could not copy itself.
- Exploratory landscape with heritability: in 3 of 4 cubes between the pusher and a return closer, no single-byte path
  through heritable genotypes reaches a heritable confined genotype (43–163 exist per cube); in one, a 6-step path does.

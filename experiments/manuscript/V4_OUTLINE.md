# v4 outline: the three transitions (written 2026-10-09 while Stage L and the dials run)

Target: 3,500 words of main text that cut to the chase. Six figures. Every number below is from a generated table named in
the FINDINGS files cited; nothing about "run on two machines" or other housekeeping goes in the text (Methods only).

## Summary (≤ 200 words): three sentences per transition, one for the map, one for the lemma
Open heredity first (the literal word, 100 of 100 worlds across five lengths and two instruction sets); individuality by
closure, which wins the competition in ~120 steps and inherits no variation; the genotype as copied-but-unexecuted
information, present at birth under lethal tar and rare otherwise; the substrate map (literal channel × by-product
lethality) now with four filled cells; the lemma: a copier whose written range equals its executed range transmits no
neutral variation.

## 1. Order before life (short; keep the tar, lose the detector digression to ED)
24–34% zeros before any replicator; the first replicator has the minimal assembly index; only intervention tells them
apart. (`results/biology/assembly`, `results/stageE`.)

## 2. The first replicator is open (merge old §2 + §3)
`01 c5` or a register variant first in 80/80 (G), 20/20 (K, L = 32), 40/40 (M, 8080 subset), 10/10 mixed (H); medians
325–750 steps. Exhaustive two-byte census: the five load–push words are the only two-byte self-writers, all straight-line
(Prop. 3). Open: enters the partner in 256/256 encounters (traced, 110/110 firsts); exact copy in 56% (L = 16) to 0.8%
(L = 64) of encounters; every position varies; H(o|x) 3.9 bits → the 8-bit ceiling. (`results/exectrace`,
`results/biology/individuality`.)

## 3. Closure: individuality that wins and inherits nothing (the new centre)
- Succession: loop-bearing final in 67/80 (G), 12/20 (K), 20/20 at L = 16 in M; mechanisms: return (L = 16, both
  machines, byte-identical design), block copy (L = 20, 32, 64), relative jump through the address wrap (L = 20, 50;
  25 of 67 G closers; stated as a convention).
- Measured closure: pointer-confined 89/89 loop-bearing finals; confinement and zero inflow agree 106/110; copies 1.00,
  self-damage 0.00.
- The jump is selected: seeded at 1% into a pusher world it holds half the cells by step ~120 (5/5); the pusher cannot
  invade the closed world; pusher-only worlds close de novo at 5,000–13,250 steps (3/3). (`results/invasion_pair`.)
- The cost: closers are not more mutation-robust (M1 killed: 2/20, 2/20, 0/20, 8/20) and inherit no variation
  (transmissible sites 0–1 vs 12–32 for the pusher at L ≥ 50; capacity 6–42 vs 32–325 bits); the pusher's sites are
  executed operands, the closers' only ever copied-but-unexecuted bytes. (`results/mutscan*`, `results/exectrace`.)
- Timing by heritable fraction, not first clone: lethal tar delays heritable material 30-fold (40,000 vs 1,250) and
  closed heredity arrives at ~40,000 vs ~30,000 either way (`results/stageI`, `results/stageG/c4`).

## 4. The genotype: information copied but not run (new; Stage L decides its weight)
Five G worlds and Stage I births have transmissible sites that are never executed (82/518 and 20/24); the variation
figure's position maps show it. Stage L (10M steps): does capacity recover after closure, and only in block-copy lineages
(L1–L3)? Stage M at L = 32: with no block copy the open phase lasts a million steps and keeps its ~11 sites. If L1 holds,
this is the third transition; if not, "the first individuals stay canalised for as long as we watched" and the section
shrinks to one paragraph.

## 5. Two substrate properties decide the beginning (old BFF section, tightened; two real instruction sets + one toy)
Literal channel: BFF born closed 28/28 and 7/7 (no-op brackets), open with `P` 12/12. Lethality: open then extinct (BFF
lethal brackets), open through 16,384 epochs (harmless), born closed and late (Z80 lethal zeros, 10/10), open then closed
(Z80 benign). The dials, if they land in time: lethality as a dial (D1) and the write ratio (D2) turn the map into a phase
diagram; the saturation boundary of the theorem. (`results/bff*`, `results/bff_dials`.)

## 6. What is proved (one paragraph; SI holds the rest)
Theorem 1 (BFF: ≥ 129 instructions to copy; every replicator loops), Theorem 2 with its two provisos, Prop. 3 exact at
period ≤ 2, Prop. 4 as a count plus one measured premise, the write-range lemma (new; prove it in the SI as a definitional
statement and show the data obey it), Lemma 7. Fig. 6 redrawn for the Z80 argument actually used.

## Discussion (≤ 400 words): chemistry as predictions, in the order of the three transitions
Template chemistry as a literal channel (open first); closure by circularisation / rolling circle as the return; the first
genotype as copied-but-untranslated sequence; by-product lethality as the dial; detection by intervention.

## Figures
1 encounter (designer) + emergence by L across machines; 2 one world watched; 3 open → closed (slope charts incl.
inflow, confinement agreement, heritable fraction, closers); 4 the cost of closure and the birth of the genotype (figvar:
position maps, pair invasion, capacity over time with Stage L); 5 the substrate map with two real instruction sets, the
8080 column and the dials; 6 theory as proved. ED: atlas, size axis, ring, tar/detectors, BFF searches, scans (≤ 10).

## Housekeeping to do once, at the end
References renumbered by first citation, uncited pruned or cited (Eigen, Tierra, von Neumann, Langton, Amoeba in §3–4);
ED ≤ 10; Data and Code availability; OSF time stamp; DOIs; the designer's remaining panels (3f, 5d, 6).

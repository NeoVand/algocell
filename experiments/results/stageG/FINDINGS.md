# Stage G — closure, confirmatory: findings

Pre-registered in `PLAN.md` on 2026-10-08 before any run (predictions, thresholds and kill criteria quoted
below); 80 runs on Modal L40S, seeds 2001–2020, `none` (no ablation), 128 steps per encounter, mutation 1/16;
L = 16 and L = 50 to 300,000 steps, L = 20 and L = 64 to 1,000,000 steps; 16 random tapes recorded per sample for
the heritable fraction. Launched 02:38, finished 02:46 (80 ok, 0 failed); preflight cost estimate 5.8 GPU-hours
≈ $11 at list price. Numbers below are from `stageG/NUMBERS_G.md`, `NUMBERS.md`, `ZOO.md` and `c4/NUMBERS_C4.md`
(all generated); the per-world table is `stageG/stage_g_runs.csv`.

## 1. Emergence (new seeds reproduce the atlas)

Every world produced a heritable replicator: 20/20 at each L. Kaplan–Meier median first-replicator step 450
(L = 16), 700 (L = 20), 600 (L = 50), 750 (L = 64). Quasispecies occupancy reached 10% in 79/80 worlds (19/20 at
L = 20).

## 2. The first replicator is open, in all 80 worlds

The first heritable replicator (the `t_rep` exemplar) contains **no control-flow instruction in 80/80 worlds**
(jump, relative jump, DJNZ, CALL/RET, RST by linear disassembly; LDIR/LDDR also absent). Executed as A against 256
random partners for one 128-step encounter it **copies into fewer than 90% of partners in 80/80 worlds**: median
copied 0.68 (L = 16, range 0.62–0.75), 0.66 (L = 20, 0.56–0.71), 0.68 (L = 50, 0.59–0.78), 0.73 (L = 64,
0.68–0.84). It damages itself (≥ 25% of its bytes lost) in a median 0.33 of encounters at L = 16, 0.43 at L = 20,
0.11 at L = 50 and 0.06 at L = 64. The modal first tape is the two-byte pusher `01 c5` (`LD BC,nn ; PUSH BC`) tiled
to the tape: 19/20 worlds at L = 16, 12/20 at L = 20, 14/20 at L = 50, 18/20 at L = 64 (the others are its
register variants).

## 3. At 300,000 steps the dominant is closed and byte-identical across worlds (G1)

**L = 16:** the faithful dominant carries a control-flow instruction in **20/20** worlds; paired per world, 20 gained
control flow and 0 lost it. 17/20 worlds end on the identical tape
`ad e3 21 e3 21 c0 ad c0 …` (`EX (SP),HL ; LD HL,nn ; RET NZ ; XOR L …`), the `RET NZ` closer of Stages C and E;
the other three carry `JP`, `JR Z + LDIR`, or `JP (HL) + JR NZ + LDIR`. Every final dominant copies into 1.00 of 256
random partners with self-damage 0.00 (20/20 ≥ 0.95 copied).

**L = 50:** control flow in **20/20** (all `JR NZ,d`), 14/20 worlds on the identical tape
`01 c5 01 c5 01 c5 01 c5 01 20 f0 c5 01 c5 …` — the pusher tiling with one `JR NZ,-16` inserted; copied 1.00,
damaged 0.00 in every world.

**Heritable fraction of 16 random cells (median over worlds):** at L = 16 it stays below 0.20 at every sampled step
up to 20,000 (0.03 at 1,000; 0.12 at 2,000–10,000; 0.19 at 15,000–20,000), then rises to 0.75 at 30,000 and
0.88 at 100,000 (0.91 at 300,000). At L = 50 it rises at once — 0.38 at 1,500, 0.59 at 2,000 — and then plateaus
between 0.41 and 0.62 through 300,000 (0.62 at 100,000).

**Verdicts (pre-registered thresholds):**
- G1(a), first replicator without control flow in ≥ 18/20 and copying < 90% of partners in ≥ 18/20: **met at both L
  (20/20 and 20/20).**
- G1(b), faithful control-flow dominant in ≥ 16/20, modal final tape copied ≥ 0.95 with ≤ 0.05 self-damage: **met at
  both L (20/20; 1.00 / 0.00).**
- G1(c), median heritable fraction < 0.20 at every sampled step ≤ 5,000 after emergence and > 0.70 at 100,000: **met
  at L = 16, not met at L = 50** (0.59 at 2,000; 0.62 at 100,000). The threshold had been calibrated on L = 16
  (Stage C4); at L = 50 the open pusher damages itself far less (0.11 vs 0.33) and its copies are heritable by the
  culture test even though they are partner-dependent, so the functional fraction is high from the start, and the
  closed successor lifts it only to ≈ 0.6. Why the plateau sits at 0.6 rather than 0.9 is not established (the
  exact-tape share of the dominant is not informative at L = 50: 0.031–0.036 in every world, because a 50-byte
  tape rarely stays byte-exact at mutation 1/16, against a median 0.58 at L = 16). To be checked by assaying cells
  by quasispecies class before any claim is made.
- Kill criterion (fewer than 12/20 control-flow dominants, or ≥ 6/20 first replicators with control flow): **not
  triggered at either L.**

## 4. At 1,000,000 steps closure is the rule at L = 20 and the exception at L = 64 (G2)

The pre-registered G2 criterion was syntactic: a faithful dominant *with a control-flow instruction* that copies
≥ 95% of random partners. The construct it stood for is behavioural (Definition 2 of `THEORY.md`: the organism's
bytes are invariant under execution in every context), and the syntactic proxy turned out too narrow: the
dominant family at L = 20 closes with **LDIR/LDDR**, block-repeat instructions that loop in hardware without a jump.
Both counts are reported; the pre-registered one decides the pre-registered verdict.

**L = 20:** control flow (pre-registered definition) in **6/20** → *between* the prediction (≥ 10) and the
alternative (< 5). LDIR/LDDR in 15/20. **Closed by the partner test (copied ≥ 0.95, damaged 0.00) in 19/20**, every
one of them with a loop of one kind or the other; the remaining world is still the open pusher (copied 0.69,
damaged 0.36). Modal final tape `1e 04 ed b0 …` (`LD E,n ; LDIR` ×5) in 4/20; the LDIR family is diverse.

**L = 64:** control flow in **3/20** → the *alternative*. LDIR/LDDR in 7/20. Closed by the partner test in **8/20**;
**12/20 worlds are still dominated by the open pusher at one million steps** (`11 d5` / `01 c5` tilings; copied
0.78, damaged 0.06). Stage E had found the pusher still dominant at 300,000 in 10/10 L = 64 worlds; the census
here puts the pusher family at 17/20 worlds at 300,000 and 13/20 at 1,000,000.

Reading, with the caveat that it was not pre-registered: the time to closure depends on L, and the L = 64
pusher is the least damaged open replicator in the atlas (self-damage 0.06, copy success 0.73–0.78), so the
selective pressure for closure is weakest exactly where closure is slowest. Whether closure at L = 64 is merely
slow or needs a geometric coincidence (`THEORY.md` P3: absolute-address closers only work on rings where the
target lands in the body) is open; the heritable fraction at L = 64 stays at 0.19–0.34 from 10,000 to
1,000,000 steps (median over worlds), i.e. the open regime is stable for a million steps.

## 5. What Stage G adds to the atlas

- The order of events — tar, then an open two-byte replicator, then closure by a loop — reproduces on 80 new seeds
  with every first replicator open (80/80) and every 300k dominant closed at L = 16 and L = 50 (40/40).
- Convergence to byte-identical closed tapes across independent worlds: 17/20 at L = 16, 14/20 at L = 50.
- Closure is not tied to one instruction: `RET NZ` (L = 16), `JR NZ` (L = 50), `DJNZ`, `JP`, `JP (HL)`, `RET`,
  and the hardware loops `LDIR`/`LDDR` (L = 20, 64) all close; what they share is a control-flow cycle that keeps
  the pointer inside the organism (`THEORY.md` Lemma 2).
- Closure time grows with L: by 300k at L = 16 and 50, by 1M in 19/20 at L = 20, in 8/20 at L = 64.
- Two pre-registration lessons, recorded for the paper's methods: the heritable-fraction thresholds of G1(c) do
  not transfer from L = 16 to L = 50; and a syntactic loop criterion must include block-repeat instructions — the
  behavioural partner test is the measurement that matches the definition.

Figures: `stageG/G_partner_independence.{pdf,svg,png}` (per-world partner-copy fraction of the first replicator and
the final dominant, loop-bearing tapes marked) and `stageG/G_heritable_fraction.{pdf,svg,png}` (median and IQR of the
heritable fraction of 16 random cells vs step, per L); Kaplan–Meier curves `km_L*.png`.

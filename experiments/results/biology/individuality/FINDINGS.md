# Individuality as zero information inflow — findings (analysis B, pre-registered in `BIOLOGY_PREREG.md` on 2026-10-08)

Code: `biology/individuality.py`. Numbers: `NUMBERS_INDIVIDUALITY.md` (generated; every number below is taken from it or from
`per_replicator.csv`, `summary.csv`, `predictions.csv` in this directory). Inputs: `results/stageG/stageG/stage_g_runs.csv`
(80 Z80 worlds, L ∈ {16, 20, 50, 64}) and `results/bff/runs.csv` with `runs/bff_modal/bff/<run>/cond.json` (64 life-producing
BFF soups of 84). No soup was run; the GPU did four `execute_pairs` launches of 10,240 pairs (one per L) and five BFF launches
of 4,608–9,728 pairs (one per flag set), a few seconds in all.

**The identity.** With the organism X fixed and the offspring a deterministic function O = F(X, E) of the random partner E,
H(O | X, E) = 0, so the information flowing from the environment into the offspring is I(O; E | X) = H(O | X) − H(O | X, E) =
H(O | X), the entropy of the offspring over partners; it is estimated by intervention with 256 uniformly random partners per
tape (the partner distribution of both culture tests), one encounter each (Z80: 128 instructions; BFF: the soup's own executor
flags and 2¹³ steps), as the plug-in entropy of the 256 offspring byte strings (ceiling log₂ 256 = 8 bits).

## Verdicts (pre-registered thresholds; `predictions.csv`)

| | population | statement | count | fraction | met at | kill below | outcome |
|---|---|---|---|---|---|---|---|
| Q1 | 80 Stage G worlds | H(O \| X_first) > 1 bit | 80/80 | 1.000 | 0.90 | 0.60 | **met** |
| Q2 | worlds whose final dominant carries a loop instruction (67) | H(O \| X_final) < 0.5 bit | 63/67 | 0.940 | 0.90 | 0.60 | **met** |
| Q3 | 80 Stage G worlds | H(O \| X_first) − H(O \| X_final) ≥ 2 bits | 63/80 | 0.788 | 0.75 | 0.50 | **met** |
| Q4a | life-producing soups of the published BFF (std, 9) | H(O \| X_first) < 0.5 bit | 6/9 | 0.667 | 0.90 | 0.60 | **between** (neither met nor killed) |
| Q4b | life-producing soups of the literal-push variants (lit, wraplit, wraplitnh; 36) | H(O \| X_first) > 1 bit | 12/36 | 0.333 | 0.90 | 0.60 | **KILL** |

Q4 as a whole is killed by its second clause. The kill is real by the letter of the pre-registration and is reported as such;
section 2 shows that the clause contradicted Lemma 7 for the organism it was written about, and what the measurement says instead.

## 1. Z80: the first replicator is environmentally determined, its successor is not (Q1–Q3 met)

**Q1 — every first replicator imports information (80/80).** H(O | X_first) is 3.454–8.000 bits (median 7.024, IQR 5.383–7.895),
never below 1 bit. It rises with tape length — medians 3.856 (L = 16), 6.168 (L = 20), 7.777 (L = 50), 7.984 (L = 64) — because the
exact copy becomes rare as the tape lengthens: the modal offspring (the byte-exact copy) is produced in 55.5% of encounters at
L = 16, 28.7% at L = 20, 5.1% at L = 50 and 0.8% at L = 64 (`p_modal`, medians). Every offspring position takes more than one
value across partners (`n_var_pos` median = L at every L), and 3–11% of offspring bytes are the partner's own original byte
(`partner_byte_share` medians 0.031, 0.041, 0.099, 0.106). The first replicators have no loop instruction (0/80, from
`stage_g_runs.csv`); they are the `LD rr,nn ; PUSH rr` tilings, which run off their own tape into the partner, load partner bytes
as immediates and push them — the open organism reads its context as code and writes what it read. The organism's own half is
equally partner-determined: H(X′ | X_first) median 6.113 bits, byte-intact after the encounter in only 28.7% of encounters
(`self_intact_frac`, median). At the class level (faithful ≥ 0.95 / partial ≥ 0.75 / other) the three outcomes are all populated:
H_class median 1.534 bits of a possible 1.585.

**Q2 — the loop-bearing successor has zero inflow at byte resolution in 63 of 67 worlds.** H(O | X_final) is exactly 0 (one
offspring string in 256 encounters) in 63/80 worlds; the organism's own half is also invariant (H(X′ | X_final) median 0,
byte-intact in 100%). Among the 67 finals with a loop instruction (jump, CALL/RET, RST or LDIR/LDDR), 63 have H = 0 and 4 do
not, all four for the same reason: the copy is shorter than the cell, and the uncopied positions keep the partner's bytes.
L20_s2003 (`DJNZ`+`JP NZ`+`LDDR`), L20_s2009 (`DJNZ`) and L20_s2014 (`RET`+`LDDR`) copy 17 of 20 bytes (offspring similarity 0.85
in every encounter; copied 1.00, faithful 0.00) and leave exactly 3 positions holding the partner's original byte
(`n_var_pos` = `n_partner_pos` = 3), so all 256 offspring are distinct and H sits at the 8-bit ceiling; L64_s2006 (`JP (HL)`,
a pusher with a closing jump) leaves one position (`n_var_pos` = 1), H = 7.10 bits. For these four the class entropy is 0
(every offspring is a deterministic partial copy): they are closed as programs and open only in a 1–3-byte tail that is not
part of the program. The byte-string entropy counts that tail in full — one uniformly random uncopied byte alone gives ≈ 7.1 bits
with 256 partners — which is why Q2 was pre-registered at the strict byte level and still passes at 94%.

**Q3 — the jump is ≥ 2 bits in 63/80 worlds (median drop 5.868 bits, IQR 3.603–7.761).** The 17 worlds below 2 bits are the
4 tails above (drops −2.3 to +0.9 bits) and the 13 worlds whose final dominant has no loop instruction (L20_s2005 and 12 of the 20
L = 64 worlds): there the final tape is still the open pusher at the 300,000-step horizon (H_final median 7.992, copied median
0.785, faithful 0.359), as Stage G already recorded (closure incomplete at L = 64 by 300k). In no world does a loop-free final
have H < 0.5 bit (0/13), and in no world does a loop-bearing final have H above a 1–3-byte tail: the loop instruction and zero
inflow coincide world by world (`fig_z80_slope`: filled markers at 0 bits, open markers at the ceiling).

**Reading.** In Krakauer et al.'s terms the first replicator is environmentally determined (its offspring and its own next
state carry 4–8 bits about the partner) and its successor is organismal (0 bits); the transition is the closing of information
inflow, and it happens with the arrival of the loop instruction. This is Lemma 7 measured: the open pusher has λ > 0 and is
imperfect (exact copy in 0.8–55% of contexts), the closer has λ = 0 and is perfect (one offspring string in 256 contexts).

## 2. BFF: the published variant is born with (nearly) zero inflow; the literal-push clause is killed

**Q4a — std, 6/9 below 0.5 bit (between).** The six (std_s2, s9, s10, s16, s17, s20) produce one offspring string in 254–256 of
256 encounters (H 0–0.037 bits, p_modal ≥ 0.996), with the pointer never entering the partner (`entered_frac` 0) and no partner
byte surviving (`partner_byte_share` ≈ 0). The three exceptions: std_s4 (H 7.40) and std_s11 (H 2.06) are the two first
replicators the culture test had already classed *intermediate* (copies 0.85 and 0.83 here, 0.72 and 0.86 in `runs.csv`;
std_s4's pointer enters the partner in 19.5% of encounters); std_s22 (H 7.25) is pointer-closed (entered 0) and copies 100% of
partners at ≥ 95% similarity, but leaves exactly one position holding the partner's byte (`n_var_pos` = `n_partner_pos` = 1) —
the same 1-byte tail as L64_s2006 above. By class entropy all nine are at or near 0. The wrapping-pointer variant (wrap, 19 soups,
no prediction attached) behaves the same way: 14/19 below 0.5 bit; the five exceptions are three intermediates (wrap_s3, s10, s12:
H 1.7–2.5 bits, 18–27% of partners not converted) and two 1-byte tails (wrap_s17, s23: H 7.1–7.2 bits, pointer-closed,
copied 1.00, faithful 1.00). Finals: H = 0 in 7/9 std and 18/19 wrap worlds; the remaining three finals (std_s2, std_s9,
wrap_s20) are the sterile one-symbol fills that had displaced the replicator (copied 0.00–0.02), not replicators.

**Q4b — literal-push variants, 12/36 above 1 bit: KILL.** The clause holds in the variant without a wrapping pointer and fails
in both variants with one:
- **lit (12/12 above 1 bit; H 7.977–8.000).** The encounter ends after one pass (median 54 of 8,192 steps executed; 100% end before
  the budget because the pointer leaves the pair). The all-`P` tiling pushes `P P` pairs downward through the partner from its
  top, then executes the `P`s it has just written and fills most of the rest; what survives is a seam of two to three bytes
  derived from the partner (the last `P` of the organism takes its two operands from the partner's first two bytes and writes
  them; one position beside them is left unwritten) plus, in ≈ 8% of encounters, a stray partner instruction that ends or
  diverts the pass early (mechanism from a CPU trace of one encounter with `micro.bff.reference_execute`; the tabulated facts
  are `n_var_pos` median 23, `p_modal` 0.008, `copied` 0.93, `faithful` 0.87). Every offspring is distinct: H at the ceiling.
- **wraplit (0/12 above 1 bit; H 0.587–0.988, median 0.751).** With the pointer wrapping, the pushes lap the 128-byte ring about
  forty times in 8,192 steps and overwrite the seam as well; the offspring is the string `P⁶⁴` in 89.5–94.1% of encounters
  (`p_modal`), the rest being encounters that an unmatched bracket in the partner halted before the flood was complete
  (`halted_frac` median 0.076, `exec_steps_mean` 7,570). H ≈ −p log p for p ≈ 0.92 plus 8 bits for each of the ≈ 8% distinct
  survivors: 0.6–1.0 bits, in the pre-registered gap between the two thresholds.
- **wraplitnh (0/12 above 1 bit; 12/12 below 0.5 bit; H 0–0.111, median 0.037).** Nothing halts, every encounter runs 8,192 steps,
  the flood always completes: `P⁶⁴` in 98.8–100% of encounters.

**Why the clause failed, and what the measurement says.** The pre-registration equated "open" (the pointer enters the partner;
`entered_frac` = 1.00 in all 36 literal-push first replicators) with "information flows into the offspring". For the Z80 pusher
the two coincide because its writes carry what it read (immediates loaded in the partner) and leave partner bytes unwritten.
The one-byte BFF organism reads its partner as code too, but its writes carry nothing of what it read — every push writes
`P P` — and under a wrapping pointer they erase everything the offspring could remember. `results/bff/FINDINGS.md` already
recorded that this organism copies 1.00 of random partners in wraplitnh; Lemma 7(ii) (perfect fidelity across contexts implies
λ = 0) therefore predicted H ≈ 0 for it, and the clause as written contradicted the lemma. The data confirm the lemma and kill
the clause. The corrected statement is: the open organism has inflow when its copy is *partial* (Z80 pusher, lit), not because it
is open; openness by control flow is neither necessary for inflow — the 1-byte tails of std_s22, wrap_s17, wrap_s23 and the 3-byte
tails of the three L = 20 Z80 finals have λ > 0 with the pointer never entering the partner — nor sufficient for it
(wraplit, wraplitnh). What the Z80 transition closes is the inflow itself, which the pusher has because it is an imperfect copier;
the BFF + `P` organism, perfect under a wrapping pointer, never had inflow to close, which is consistent with its never closing
by control flow (`results/bff/FINDINGS.md`, 0/12 closed classes in wraplitnh).

**A qualification to the theory text.** THEOREMS.md (reading of Lemma 7) calls the pointer-entered record "a sufficient condition
for λ(x) = 0 in these machines". At byte resolution it is not: 7 of the 28 pointer-closed BFF first replicators have H > 0,
three of them at 7.1–7.3 bits from a single unwritten offspring byte (`entered_frac` = 0 but H > 0: first 7, final 0). Pointer
closure rules out reading the context as code; inflow by omission (unwritten offspring positions) and by data reads remains.
The record is sufficient for zero inflow into the *copied* region, not into the cell. THEOREMS.md is outside this analysis's
write scope and was not changed.

## 3. Cross-checks against the input tables

- Z80: `copied_frac` (fraction of partners that became ≥ 75% copies) against `first_copied`/`final_copied` of `stage_g_runs.csv`
  (same kernel, 256 partners, a different random stream): median |Δ| 0.008, max |Δ| 0.117, within 0.10 in 158/160 (the two beyond,
  L16_s2016 first 0.734 vs 0.629 and L20_s2006 first 0.676 vs 0.559, are 2.6–2.9 binomial standard deviations of a difference of
  two 256-partner estimates at p ≈ 0.65; two of 160 is what chance gives); means 0.827 vs 0.825. `damaged_frac` against
  `*_damaged`: median |Δ| 0.006, max 0.156. `has_loop` (`has_cf | has_block`) agrees 160/160 by construction.
- BFF: `copied_frac` (forwards-or-reversed similarity, as the BFF culture test) against `first_copies`/`final_copies` of `runs.csv`
  (64 partners): median |Δ| 0.004, max |Δ| 0.129, within 0.15 in 128/128; means 0.934 vs 0.931. `damaged_frac` against
  `*_self_damage`: median |Δ| 0.000, max 0.043. `has_loop` recomputed with `micro.bff.has_loop` under each soup's alphabet agrees
  with `first_loop`/`final_loop` 128/128. The executor flags read from each run's `cond.json` matched the variant names for all
  64 runs (std: no wrap, no literal, halting; wrap: wrapping pointer; stdlit = lit: literal push without wrap; wraplit; wraplitnh:
  unmatched brackets are no-ops); all have density 1 (ASCII alphabet), 64-byte tapes and 8,192 steps.

## 4. Deviations from the pre-registration and caveats

1. **Saturation.** With 256 partners the plug-in entropy cannot exceed 8 bits; 5/80 Z80 first replicators and 5/64 BFF first
   replicators have 256 distinct offspring and sit at the ceiling, where H is a lower bound. Conversely a single uniformly random
   unwritten byte alone yields ≈ 7.1 bits (156–167 distinct values of 256), so H in bits resolves "how many offspring bytes the
   environment determines" only up to about one byte. The class entropy (0 for all 1–3-byte tails) and `p_modal` are reported
   alongside for this reason. All thresholds were met or missed by margins that this does not affect (Q1 80/80 at > 1 bit; Q2
   exceptions are at 7.1–8.0 bits, far above 0.5).
2. **Similarity for BFF** is the BFF culture test's (best cyclic shift, forwards or reversed; `micro.bff_soup.batch_best_similarity`)
   rather than the forwards-only Z80 measure named in the pre-registration, so that `copied_frac` is comparable with `runs.csv`;
   it affects only `H_class`, `copied_frac` and `faithful_frac` of the BFF rows, not H, `p_modal` or `H_self`.
3. **Partners** were drawn with numpy `default_rng` seeded per replicator (`partner_seed` column), not with Stage G's shared stream,
   so the Z80 cross-check is statistical (section 3), not bit-exact.
4. **BFF population.** The pre-registration says "84 soups"; 64 are life-producing (t_top not null) and have a first tape — std 9,
   wrap 19, lit 12, wraplit 12, wraplitnh 12 — and only these enter Q4. `runs.csv` names the lit variant "stdlit". The wrap variant
   (19 soups) carries no prediction and is reported descriptively.
5. **Not pre-registered (descriptive only):** `n_var_pos`, `n_partner_pos`, `partner_byte_share`, `entered_frac`,
   `exec_steps_mean`, `halted_frac`, used above to say where the inflow sits; the mechanism of the lit seam comes from a hand
   trace of one encounter, not from a tabulated statistic.
6. **Determinism.** The identity I(O; E | X) = H(O | X) needs O to be a deterministic function of (X, E); both executors are
   deterministic for a fixed pair and step budget (BFF is bitwise reproducible across machines per `results/bff/FINDINGS.md`;
   the Z80 culture-test kernel is the one Stage G used).
7. **Outcomes not yet appended** to `BIOLOGY_PREREG.md` (outside this analysis's write scope); the table in the Verdicts section
   is the text to append.

## 5. Files

- `per_replicator.csv` — one row per replicator (machine, world, group = L or variant, which, has_loop, H_bits, H_class_bits,
  p_modal, H_self_bits, copied_frac, damaged_frac, faithful_frac, sim_mean, self_intact_frac, n_unique_offspring, n_unique_self,
  n_var_pos, n_partner_pos, partner_byte_share, entered_frac / exec_steps_mean / halted_frac (BFF), csv_copied, csv_damaged,
  csv_has_loop, loop_kind, sim_kind, n_partners, steps, partner_seed, tape_hex).
- `summary.csv` — per machine × group × which: n, median and IQR of H, medians of H_class, p_modal, H_self, copied, damaged,
  fraction > 1 bit, fraction < 0.5 bit, fraction < 0.5 bit among loop-bearing, and the paired drop (median, IQR, fraction ≥ 2 bits).
- `predictions.csv` — Q1–Q4b with counts, fractions, thresholds and outcomes.
- `offspring_hist.json` — per replicator, the sorted multiplicities of the 256 offspring strings (the distribution H is computed from).
- `fig_z80_slope.{pdf,svg,png}` — H(O | X) first → final per world, one panel per L; filled vermilion markers where the tape has a loop
  instruction, open grey where not; dotted lines at the 8-bit ceiling and the 1-bit (Q1) and 0.5-bit (Q2) thresholds.
- `fig_bff_slope.{pdf,svg,png}` — the same per variant (std, wrap, lit, wraplit, wraplitnh); filled where `[` and `]` are both present.
- `NUMBERS_INDIVIDUALITY.md` — every number with its source columns, including per-world tables.

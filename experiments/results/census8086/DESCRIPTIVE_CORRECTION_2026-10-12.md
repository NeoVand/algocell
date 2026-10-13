# Final descriptive census correction — 2026-10-12

Prospective R2 withdrawal was committed as `39de676` before this audit and
before any S2 population data. R2a/R2b are descriptive; R0, fetch-based R1,
the route kill rule, L1 and amended L2 remain confirmatory and unchanged.
No label change stopped this complete audit.

All 1,246 candidate settings / 21,182 encounters
were rerun against the complete zero-plus-16 frozen partner panel under the
prefix-causal operand, memory and reset-value classifier. Full original lane
records, memory and archived trace bytes agree in every encounter: zero
mismatches. Raw word/output/match/coverage/access/count data are unchanged.

Exactly **1,247 labels change** relative to `2b06edb`:
- 1,175 literal writer → other.
- 72 memory-fed one-pass copier → other.

Every changed label is listed, with L, budget, period, exact word, partner,
old/new class and reason in `descriptive_correction_labels.csv`. Complete
before/after class counts are in `descriptive_correction_counts.csv` and
per-setting metadata in `descriptive_correction_provenance.json`. Every other
case (including unchanged ones) is shown in `descriptive_other_cases.csv`.
All four `mechanisms_*.csv` and the hit/candidate/mechanism tables are current.
Historical corrections below remain a chronological record; this correction
supersedes their class labels and counts.

Corrected nonzero exact-zero word counts (period two):

| L/budget | Literal | Memory-fed | Confined loop | Other |
|---|---:|---:|---:|---:|
| 16/128 | 3 | 2 | 0 | 15 |
| 16/512 | 3 | 1 | 0 | 25 |
| 32/128 | 3 | 5 | 0 | 18 |
| 32/512 | 3 | 4 | 0 | 19 |

Nonzero period-one exact-zero counts remain zero. Total exact-zero period-two
word counts remain 20, 29, 26, 26. 07 A5 remains causally verified memory-fed
one-pass in all 68 encounters, including all 64 open random-partner encounters.
B9 51 and AD 50 include failed prefix-causal claims and are reported as other
where indicated; equal values cannot override a failed intervention. The
existing two overflow cases remain other. Mechanisms classify encounters,
not probabilities of establishment or population predictions.

All claims use both 55/AA interventions; unsupported/mixed/vector/overflow or
failed claims stay other. Detailed checks are archived in
`../s8086/prelaunch/descriptive_intervention_checks.jsonl.gz`; full audit and
burst metadata are beside it. The independent 8,192-random-program prefix,
source-expression and class differential is reported in
`../s8086/prelaunch/descriptive_random_validation.json`. It checks the causal
verification of conservative hypotheses and raw/observer parity, not arbitrary
unmodeled architectural semantics. No population or paid computation occurred.

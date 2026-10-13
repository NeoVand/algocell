# V8 review fixes: Methods, back matter and Supplementary Information

Scope: fresh review items M1 (SI and Methods consistency), N6 and C1–C4. Work began on `feat/experiments-runner` at `cf9e803`. Only `MAIN_v8.md` from `## Methods` onward and `SI_v8.md` were edited, in addition to this log. The saved MAIN prefix through the end of Figure legends is byte-for-byte unchanged. No figure source, `audit_v8*` or `AUDIT_v8*` file was edited. Changes are uncommitted.

Baselines were saved before editing at `/private/tmp/algocell-v8-review-baselines/MAIN_v8.md` and `/private/tmp/algocell-v8-review-baselines/SI_v8.md`. The complete old → new ledger below reproduces both final files exactly when replayed against those baselines; there are no unlogged textual changes.

## Scientific checks

M1: `results/tier2/T2_COMPONENTS.csv` E2 rows give corrected-core counts 2/10 L16 benign, 0/10 L16 lethal and 5/10 L32 benign, all `not met`. `REVISION_PREREG.md`, E, defines the conjunction of a fall below the starting share and endpoint T/(R+T) in 0.05–0.4 at 300,000 updates in ≥ 7/10 equal mixtures; its outcome gives historical counts 3, 0 and 6 of 10. `results/tier2/scored/E/REPORT.md` gives the corrected L16 benign equal-mixture median share 0.70 and classified-core abundance 0.05. The SI now states the counts, definition, failure in both cohorts and the absolute-abundance limitation. The validation prose in both files and the SI environmental display map retain the E2 failure. No competitor-background qualification was removed.

N6: reference 15 was added to the MAIN bibliography and cited at the Methods finite-window result and both named SI Fine–Wilf applications. The main manuscript’s numbered references first appear in the order 1 through 15; reference 14 precedes reference 15. Bibliographic metadata were corroborated by the [AMS reference entry](https://www.ams.org/proc/2003-131-06/S0002-9939-03-06970-3/) and [Wilf’s publication list](https://www2.math.upenn.edu/~wilf/reprints.html). The requested DOI is https://doi.org/10.1090/S0002-9939-1965-0174934-9 . The DOI resolver itself was unavailable through the browser tool; the AMS entry confirms the DOI.

C2: the complete local overlap, Z80, copy-cost and conditional self-writer arguments remain in place. All `MAIN_nature.md` and `Proposition 8` pointers were removed from the submission-facing SI. No new proof claim was substituted.

C3: scientific amendments, pre-outcome timing, the noncanonical world 8086001 attempt, its hash identity, publication-failure restart from initialization, missing continuation state, absence of trajectory splicing, nondeterminism and scoring-only recovery remain explicit. Payment figures, spending caps, provider confirmations, dispatch timeouts, command repair and lock-retry details were removed from the narrative. One neutral sentence locates failed attempts, restarts and operational amendments in the provenance archive. Existing operational source brackets remain unchanged, including the accounting/verification bracket, so the original record remains identifiable. The 239-file provenance count and source integrity statement are retained. The source records themselves were not edited.

C4: first uses in Methods and SI introduce the corrected Z80 core (core C1), Z80 displacement study (registered experiment C1) and 6502 retention study (registered experiment C2). Subsequent prose uses plain names. `results/impact/cores.csv` identifies core C2 as variant `zilog-nmos`; its first prose use now names the vendor-specific Zilog-NMOS Z80 core. Original IDs remain in registration tables, verdict ledgers and source fields. Protected main text and legends were not edited.

## C1 definition, threshold and unit coverage

The former Numerical derivations section is replaced by the proposed Source data and figure provenance paragraph. Its full old text is retained in the old → new ledger below as provenance. The following check distinguishes scientifically meaningful definitions from duplicate arithmetic and figure bookkeeping.

| Former content | Retained location or required relocation |
|---|---|
| Paths resolve from `experiments/`; source tables and protocol/display conventions | SI opening paragraph; new Methods Source data and figure provenance; SI Note 5 and Table 15. Review-audit pointers removed. |
| K/L32/seed5006: six total scored sites, four unexecuted | Existing SI Note 5 Competition classes and figure measurements, sampled-exception paragraph. |
| Four measurements and two distinct properties | Moved verbatim in substance to the opening of SI Note 5. Individual definitions remain in Methods Culture, carriage and inflow and SI Notes 1–2. |
| Cumulative detection = 1 minus Kaplan–Meier survival | Moved to SI Note 5 Statistical procedures. |
| Culture normalization and before-match < 0.75 | Existing Methods Culture, carriage and inflow retains `(after−before)/(1−before)` and the threshold. |
| L16 modal match permits four mismatches / requires 12 of 16 | Existing SI Note 5 illustrative-world paragraph retains the 12-of-16 threshold; the equivalent subtraction is redundant arithmetic. |
| L32 ring length, parent/partner address ranges and 0x04/0x60 displacement reduction | Existing Methods Forward-overlap theorem and SI Note 5 erasure trace retain P = 2L and physical 0–31/32–63 addresses. Explicit P = 2L = 64 and 96 mod 64 = 32 moved to the Note 5 intervention-strip paragraph. |
| Four-byte fixed core and payload counts 0/28 at L32 | Existing Methods Forward-overlap theorem fixes the core and conditional count; SI Note 5 strips retain erasure/carrying of all 28 noncore coordinates. |
| Conceptual L12, c = 4, d = 4/12, zero/eight free bytes, two-byte units | Existing SI Note 5 overlap-schematic paragraph retains all these definitions. |
| Four repeats of the 16-byte `5F A4 x y` tape | Existing SI Note 5 first-family display paragraph retains four repeats and positions 0–15; 16/4 is redundant arithmetic. |
| Inclusive root interval has d coordinates; bounding prefix 1+max(K) differs from footprint cardinality | Existing Methods Provenance certificates retains R = {s,…,s+d−1}, c = 1+max(K) and exact count d−\|C_R\| rather than d−c. |
| 160 × 125 lattice / 20,000 cells; 8,192 draws; exponent 4 / 512 mutation attempts; attempts distinct from applied changes | Existing Methods Worlds and detection retains every quantity and distinction. Exponent 4 also added beside mutation attempts in SI Note 5. |
| Detection gate 0.005 = 0.5% = 100/20,000 cells | Fraction retained in Methods and SI Table 1; cell-count equivalence moved to SI Note 5 immediately below Table 1. |
| At least half of 16 means ≥ 8/16; strict majority > 0.5 | Existing Methods Registered 8086 tests retains ≥ 8/16. Explicit transmitter-majority threshold moved to SI Note 5 figure measurements. |
| Reciprocal 1% invasions, T shares 0.01/0.99 | The percentage is in SI Table 1; endpoint shares moved to SI Note 5 display definitions. The complementary R share is redundant arithmetic. |
| S2 motif observations and nested-observation units | Existing Methods Worlds and detection retains 20 worlds, 585 records/world, the five checkpoints and the 20,000-cell population; SI Note 9 retains all products, totals and repeated-observation warning. No definition or denominator was lost. |
| Complete 256 period-one / 65,536 period-two assignments | Existing Methods Causal census classification retains both census denominators and completeness. |
| Registered 512 joint assignments = 256 counterbalanced + 256 seeded; exploratory 64 joint assignments | Existing Methods Registered 8086 tests and Exploratory 8086 analyses retain the designs and units. |
| 8086 SP = 1, decrement by 2 modulo 65,536, low/high mapping to P−1/0 at registered rings | SP and byte mapping already in Methods Coherent-fetch 8086; explicit decrement and modulus moved there. The 65,535 intermediate address is redundant arithmetic. |
| Fig. 1c seed/length; Fig. 2 serial and site keys; marker lineage; Fig. 3 d settings, disc scale and lineage strips | Existing SI Note 5 retains every selection, key, unit and exemplar count. Added the missing explicit one-bar-per-design convention after the equal branching-endpoint statement. |
| Competition start keys, T_tiled control, dial fields, three arrival groups, six coincident censors | Start keys/control/dial fields/arrival groups moved to SI Note 5; its existing display paragraph retains the six coincident censoring observations. |
| Claim component versus whole claim, no extra independent units | Moved to SI Note 5 beside the display grouping definitions; SI Note 7 retains the all-components criterion. |

## Guard results and reconciliation

Both required `prose_guard.py OLD NEW` runs completed with exit 1, not a clean lexical PASS. The differences are intentional: C1 deletes duplicate numeric bookkeeping and relocates definitions; N6 adds a reference, DOI and citation brackets; C2 replaces obsolete local proof pointers; C3 removes operational payment/timeout numbers; C4 changes numeric substrings of identifiers in prose; M1 adds the omitted E2 account. No unrequested empirical value, denominator, verdict or status was changed. Both files parsed without bracket errors. The guard is lexical and cannot distinguish these authorized changes from accidental ones.

The exact commands, complete numerical/raw-digit/source-span differences and final return codes follow. The `000` and ordinal `second` removals in SI are lexical effects of deleting the old operational `9,000-second` wording; no scientific time point was removed. The retained table IDs and source fields are exempt from prose renaming as requested.

### MAIN guard

```sh
python3 experiments/manuscript/prose_guard.py /private/tmp/algocell-v8-review-baselines/MAIN_v8.md experiments/manuscript/MAIN_v8.md
```

```text
prose_guard: /private/tmp/algocell-v8-review-baselines/MAIN_v8.md -> /Users/neo/repos/algocell/experiments/manuscript/MAIN_v8.md
numbers: FAIL (old=919, new=798; removed=126, added=5)
  removed 5 × "0"
  removed 1 × "0.005"
  removed 2 × "0.01"
  removed 2 × "0.5"
  removed 1 × "0.75"
  removed 2 × "0.99"
  removed 1 × "0x04"
  removed 1 × "0x60"
  removed 10 × "1"
  removed 2 × "100"
  removed 2 × "11,700"
  removed 3 × "12"
  removed 1 × "125"
  removed 5 × "16"
  removed 1 × "160"
  removed 2 × "2"
  removed 1 × "2,000"
  removed 1 × "2,000,000"
  removed 2 × "20"
  removed 3 × "20,000"
  removed 1 × "2002"
  removed 5 × "256"
  removed 1 × "28"
  removed 1 × "3"
  removed 1 × "31"
  removed 4 × "32"
  removed 1 × "35,100"
  removed 15 × "4"
  removed 2 × "512"
  removed 1 × "585"
  removed 1 × "63"
  removed 4 × "64"
  removed 1 × "65,535"
  removed 1 × "65,536"
  removed 2 × "8"
  removed 2 × "8,192"
  removed 3 × "8086"
  removed 1 × "96"
  removed 1 × "eight"
  removed 1 × "first"
  removed 3 × "four"
  removed 1 × "half"
  removed 7 × "one"
  removed 3 × "six"
  removed 5 × "three"
  removed 7 × "two"
  removed 3 × "zero"
  added 1 × "109"
  added 1 × "114"
  added 2 × "15"
  added 1 × "1965"
raw digit tokens: FAIL (old=1003, new=876; removed=140, added=13)
  removed 5 × "0"
  removed 1 × "0.005"
  removed 2 × "0.01"
  removed 2 × "0.5"
  removed 1 × "0.75"
  removed 1 × "0.99"
  removed 1 × "04"
  removed 1 × "0x04"
  removed 1 × "0x60"
  removed 14 × "1"
  removed 2 × "100"
  removed 2 × "11,700"
  removed 4 × "12"
  removed 1 × "125"
  removed 10 × "16"
  removed 1 × "160"
  removed 11 × "2"
  removed 1 × "2,000"
  removed 1 × "2,000,000"
  removed 4 × "20"
  removed 3 × "20,000"
  removed 1 × "2002"
  removed 5 × "256"
  removed 1 × "28"
  removed 3 × "3"
  removed 1 × "31"
  removed 11 × "32"
  removed 1 × "35,100"
  removed 12 × "4"
  removed 1 × "4009"
  removed 1 × "5"
  removed 2 × "50"
  removed 1 × "5006"
  removed 2 × "512"
  removed 1 × "585"
  removed 1 × "60"
  removed 1 × "6003"
  removed 1 × "6004"
  removed 1 × "63"
  removed 4 × "64"
  removed 1 × "65,535"
  removed 1 × "65,536"
  removed 1 × "7"
  removed 2 × "8"
  removed 2 × "8,192"
  removed 1 × "8080"
  removed 3 × "8086"
  removed 1 × "96"
  removed 1 × "−0.99"
  removed 2 × "−1"
  removed 5 × "−4"
  added 1 × "-0174934"
  added 1 × "-1965"
  added 1 × "-9"
  added 1 × "-9939"
  added 1 × "0002"
  added 1 × "10.1090"
  added 1 × "109"
  added 1 × "114"
  added 2 × "15"
  added 1 × "1965"
  added 2 × "80"
bracketed spans: FAIL (old=134, new=131; removed=5, added=2)
  removed 1 × "[algocell_exp/assay.py; algocell_exp/metrics.py; manuscript/figures/out/v8/data/geometry_units.csv; manuscript/figures/out/v8/data/dial_coordinates.csv; manuscript/figures/out/v8/data/lineage_strips.csv; manuscript/figures/out/v8/data/first_copier_tiles.csv; REVISION_PREREG.md, S2 B2]"
  removed 1 × "[manuscript/figures/out/v8/data/competition_endpoints.csv; results/offset/INTRUDER.md; manuscript/figures/out/v8/data/lethality_summary.csv; manuscript/figures/out/v8/data/arrival_worlds.csv; manuscript/figures/out/v8/data/arrival_censors.csv; results/tier2/T2_CLAIMS.csv; results/tier2/T2_COMPONENTS.csv]"
  removed 1 × "[manuscript/figures/out/v8/data/world_trajectory.csv; manuscript/figures/out/v8/data/serial_summary.csv; manuscript/figures/out/v8/data/branching_endpoints.csv; manuscript/figures/out/v8/data/erasure_manifest.json; manuscript/figures/out/v8/data/erasure_ordered.csv; manuscript/figures/out/v8/data/geometry_units.csv; manuscript/figures/out/v8/data/lineage_strips.csv; manuscript/figures/prepare_v7.py; manuscript/figures/build_v8.py]"
  removed 1 × "[results/crystal/cell_exceptions.csv; manuscript/AUDIT_v7_RESOLUTION.md]"
  removed 1 × "[score8086.py; s8086.py; algocell_exp/soup.py; dial_soups.py; manuscript/figures/out/v8/data/competition_endpoints.csv; results/s8086/exploratory/motif_scan.csv.gz; results/census8086/table_main.csv; results/census8086/LAW_CONTRACT.md; results/s8086/EXPLORATORY.md; algocell_exp/machines/i8086_pair_maps_L16.wgsl]"
  added 1 × "[15]"
  added 1 × "[https://doi.org/10.1090/S0002-9939-1965-0174934-9]"
FAIL
```

Return code: 1.

### SI guard

```sh
python3 experiments/manuscript/prose_guard.py /private/tmp/algocell-v8-review-baselines/SI_v8.md experiments/manuscript/SI_v8.md
```

```text
prose_guard: /private/tmp/algocell-v8-review-baselines/SI_v8.md -> /Users/neo/repos/algocell/experiments/manuscript/SI_v8.md
numbers: FAIL (old=1006, new=1043; removed=10, added=47)
  removed 1 × "000"
  removed 1 × "13.22"
  removed 1 × "13.22378749"
  removed 1 × "77.32835455"
  removed 1 × "8"
  removed 1 × "9,000"
  removed 2 × "95"
  removed 2 × "second"
  added 2 × "0"
  added 1 × "0.005"
  added 1 × "0.01"
  added 2 × "0.05"
  added 1 × "0.4"
  added 2 × "0.5"
  added 1 × "0.70"
  added 1 × "0.99"
  added 1 × "0x60"
  added 3 × "1"
  added 5 × "10"
  added 1 × "100"
  added 1 × "2"
  added 1 × "20,000"
  added 1 × "3"
  added 1 × "300,000"
  added 1 × "32"
  added 1 × "4"
  added 2 × "5"
  added 1 × "6"
  added 2 × "64"
  added 3 × "6502"
  added 1 × "7"
  added 1 × "8086"
  added 1 × "96"
  added 1 × "first"
  added 1 × "four"
  added 2 × "one"
  added 3 × "three"
  added 2 × "two"
raw digit tokens: FAIL (old=1212, new=1245; removed=14, added=47)
  removed 6 × "1"
  removed 1 × "13.22"
  removed 1 × "13.22378749"
  removed 1 × "77.32835455"
  removed 1 × "8"
  removed 2 × "9,000"
  removed 2 × "95"
  added 2 × "0"
  added 1 × "0.005"
  added 1 × "0.01"
  added 2 × "0.05"
  added 1 × "0.4"
  added 2 × "0.5"
  added 1 × "0.70"
  added 1 × "0.99"
  added 1 × "0x60"
  added 5 × "10"
  added 1 × "100"
  added 3 × "16"
  added 2 × "2"
  added 1 × "20,000"
  added 1 × "3"
  added 1 × "300,000"
  added 4 × "32"
  added 1 × "4"
  added 2 × "5"
  added 1 × "50"
  added 1 × "6"
  added 2 × "64"
  added 3 × "6502"
  added 1 × "7"
  added 4 × "80"
  added 1 × "8086"
  added 1 × "96"
bracketed spans: FAIL (old=69, new=70; removed=7, added=8)
  removed 1 × "[Corrected reading: `manuscript/MAIN_nature.md`, Theorem 6; `results/literal_channel/CHECK.md`]"
  removed 1 × "[Current certificates: `results/capacity/candidate_eligibility.csv`, `results/capacity/edit_eligibility.csv`; historical derivation: `manuscript/MAIN_nature.md`, Proposition 8]"
  removed 1 × "[Generated coordinates: `manuscript/figures/out/v8/data/serial_summary.csv`, `manuscript/figures/out/v8/data/serial_sites.csv`, `manuscript/figures/out/v8/data/lineage_strips.csv`, `manuscript/figures/out/v8/data/erasure_ordered.csv`, `manuscript/figures/out/v8/data/law_population.csv`, `manuscript/figures/out/v8/data/dial_coordinates.csv`, `manuscript/figures/out/v8/data/arrival_censors.csv`; preparation/rendering: `manuscript/figures/prepare_v7.py`, `manuscript/figures/prepare_v8.py`, `manuscript/figures/prepare_v62.py`, `manuscript/figures/build_v8.py`; per-occurrence source index: `manuscript/AUDIT_v7_RESOLUTION.md`]"
  removed 1 × "[Generated endpoints: `manuscript/figures/out/v8/data/competition_endpoints.csv`, `manuscript/figures/out/v8/data/competition_medians.csv`; verdicts: `results/tier2/T2_COMPONENTS.csv`]"
  removed 1 × "[Original statement and proof: `manuscript/MAIN_nature.md`, Theorem 1]"
  removed 1 × "[Proof specialization: `manuscript/MAIN_nature.md`, Proposition 8; generated geometry: `results/capacity/geometry_traces.csv.gz`]"
  removed 1 × "[Scope corrections: `manuscript/MAIN_nature.md`, Theorems 1–2]"
  added 2 × "[15]"
  added 1 × "[Constructions and scope: `results/literal_channel/CHECK.md`]"
  added 1 × "[Current certificates: `results/capacity/candidate_eligibility.csv`, `results/capacity/edit_eligibility.csv`; derivation: Supplementary Note 1, ordered overlap theorem and Z80 specialization]"
  added 1 × "[Generated coordinates: `manuscript/figures/out/v8/data/serial_summary.csv`, `manuscript/figures/out/v8/data/serial_sites.csv`, `manuscript/figures/out/v8/data/lineage_strips.csv`, `manuscript/figures/out/v8/data/erasure_ordered.csv`, `manuscript/figures/out/v8/data/law_population.csv`, `manuscript/figures/out/v8/data/dial_coordinates.csv`, `manuscript/figures/out/v8/data/arrival_censors.csv`; preparation/rendering: `manuscript/figures/prepare_v7.py`, `manuscript/figures/prepare_v8.py`, `manuscript/figures/prepare_v62.py`, `manuscript/figures/build_v8.py`]"
  added 1 × "[Generated endpoints: `manuscript/figures/out/v8/data/competition_endpoints.csv`, `manuscript/figures/out/v8/data/competition_medians.csv`; verdicts: `results/tier2/T2_COMPONENTS.csv`; E2 definition and historical outcome: `REVISION_PREREG.md`, E; corrected competition report: `results/tier2/scored/E/REPORT.md`]"
  added 1 × "[Proof specialization: Supplementary Note 1, ordered overlap theorem and Z80 specialization; generated geometry: `results/capacity/geometry_traces.csv.gz`]"
  added 1 × "[manuscript/figures/out/v8/data/competition_endpoints.csv; results/offset/INTRUDER.md; manuscript/figures/out/v8/data/lethality_summary.csv; manuscript/figures/out/v8/data/arrival_worlds.csv; manuscript/figures/out/v8/data/arrival_censors.csv; results/tier2/T2_CLAIMS.csv; results/tier2/T2_COMPONENTS.csv]"
FAIL
```

Return code: 1.

## Complete old → new ledger

Each entry gives the exact replaced text, its replacement and the review reason. Entries are ordered as applied; later wording refinements are explicit. Source brackets included in a replaced passage are preserved literally here, including those removed from the submission prose.

### 1. MAIN: C4 (1 occurrence)

Reason: First Methods use identifies the registered study.

Old → new:

```text
OLD:
The 6502 retention experiment C2 used budget

NEW:
The 6502 retention study (registered experiment C2) used budget
```

### 2. MAIN: C4 (1 occurrence)

Reason: First Methods use identifies the corrected Z80 core.

Old → new:

```text
OLD:
Historical and core-C1 trajectories

NEW:
Historical and corrected Z80 core (core C1) trajectories
```

### 3. MAIN: C4 (1 occurrence)

Reason: Use the plain name after introduction.

Old → new:

```text
OLD:
Under core C1,

NEW:
Under the corrected core,
```

### 4. MAIN: C4 (1 occurrence)

Reason: First Methods use identifies the registered study.

Old → new:

```text
OLD:
Displacement experiment C1 was registered

NEW:
The Z80 displacement study (registered experiment C1) was registered
```

### 5. MAIN: C4 (1 occurrence)

Reason: Use the plain study name after introduction.

Old → new:

```text
OLD:
The 6502 retention experiment C2 shares

NEW:
The 6502 retention study shares
```

### 6. MAIN: C4 (1 occurrence)

Reason: Core C2 is the zilog-nmos Z80 variant in results/impact/cores.csv, not the retention study.

Old → new:

```text
OLD:
I1 compared historical compatibility C0 with corrected classic core C1 and vendor-specific core C2, distinct from displacement experiment C1 and retention experiment C2.

NEW:
I1 compared historical compatibility core C0 with the corrected core and the vendor-specific Zilog-NMOS Z80 core (core C2).
```

### 7. MAIN: C4 (1 occurrence)

Reason: Use the plain study name after introduction.

Old → new:

```text
OLD:
The retention experiment C2’s

NEW:
The 6502 retention study’s
```

### 8. MAIN: C4 (1 occurrence)

Reason: Use the plain core name after introduction.

Old → new:

```text
OLD:
The benign core-C1 T-into-R subgroup

NEW:
The benign corrected-core T-into-R subgroup
```

### 9. SI: C4 (1 occurrence)

Reason: Introduce the unambiguous core and study names; the core C2 identity is checked against results/impact/cores.csv.

Old → new:

```text
OLD:
Core C0/C1/C2 denote the historical compatibility, corrected classic and vendor-specific Zilog-NMOS emulator versions, respectively. Displacement experiment C1 and 6502 retention experiment C2 are separate experimental labels.

NEW:
The emulator versions are historical compatibility core C0, the corrected Z80 core (core C1) and the vendor-specific Zilog-NMOS Z80 core (core C2). The Z80 displacement study (registered experiment C1) and the 6502 retention study (registered experiment C2) are separate studies.
```

### 10. SI: C4 (1 occurrence)

Reason: Use the plain core or study name after introduction; preserve table and source identifiers.

Old → new:

```text
OLD:
interventions of experiment C2

NEW:
interventions of the 6502 retention study
```

### 11. SI: C4 (1 occurrence)

Reason: Use the plain core or study name after introduction; preserve table and source identifiers.

Old → new:

```text
OLD:
The 6502 retention experiment C2 uses

NEW:
The 6502 retention study uses
```

### 12. SI: C4 (1 occurrence)

Reason: Use the plain core or study name after introduction; preserve table and source identifiers.

Old → new:

```text
OLD:
Displacement experiment C1 contains

NEW:
The displacement study contains
```

### 13. SI: C4 (1 occurrence)

Reason: Use the plain core or study name after introduction; preserve table and source identifiers.

Old → new:

```text
OLD:
in retention experiment C2;

NEW:
in the 6502 retention study;
```

### 14. SI: C4 (1 occurrence)

Reason: Use the plain core or study name after introduction; preserve table and source identifiers.

Old → new:

```text
OLD:
Historical and core-C1 X1 endpoints

NEW:
Historical and corrected-core X1 endpoints
```

### 15. SI: C4 (1 occurrence)

Reason: Use the plain core or study name after introduction; preserve table and source identifiers.

Old → new:

```text
OLD:
The core-C1 benign

NEW:
The corrected-core benign
```

### 16. SI: C4 (1 occurrence)

Reason: Use the plain core or study name after introduction; preserve table and source identifiers.

Old → new:

```text
OLD:
Historical and core-C1 trajectories

NEW:
Historical and corrected-core trajectories
```

### 17. SI: C4 (1 occurrence)

Reason: Use the plain core or study name after introduction; preserve table and source identifiers.

Old → new:

```text
OLD:
I1 compares historical compatibility core C0 with corrected classic core C1 and vendor-specific Zilog-NMOS core C2, independently of the experiments also named C1/C2.

NEW:
I1 compares historical compatibility core C0 with the corrected core and the vendor-specific Zilog-NMOS core.
```

### 18. SI: C4 (1 occurrence)

Reason: Use the plain core or study name after introduction; preserve table and source identifiers.

Old → new:

```text
OLD:
Core C1 flags 47/690 conditions and core C2 114/690

NEW:
The corrected core flags 47/690 conditions and the vendor-specific Zilog-NMOS core 114/690
```

### 19. SI: C4 (1 occurrence)

Reason: Use the plain core or study name after introduction; preserve table and source identifiers.

Old → new:

```text
OLD:
All 47 core-C1 flags

NEW:
All 47 corrected-core flags
```

### 20. SI: C4 (1 occurrence)

Reason: Use the plain core or study name after introduction; preserve table and source identifiers.

Old → new:

```text
OLD:
under core C1 and rescored

NEW:
under the corrected core and rescored
```

### 21. SI: M1 (1 occurrence)

Reason: Restore the omitted E2 definition, historical/corrected failures and the L16 absolute-abundance qualification from registration, the component ledger and the corrected E report.

Old → new:

```text
OLD:
variation across worlds does not establish neutrality.

NEW:
variation across worlds does not establish neutrality. With mutation on, E2 required T/(R+T) to fall below its starting share and lie in 0.05–0.4 at 300,000 updates in ≥ 7/10 equal-start worlds. This criterion was met in only 2/10 L16 benign, 0/10 L16 lethal and 5/10 L32 benign worlds under the corrected core, compared with historical counts of 3, 0 and 6 of 10, respectively; E2 therefore failed in every registered condition in both cohorts. In the corrected-core L16 benign equal-start worlds, median T share was 0.70 but census-defined core abundance was only 0.05 of all cells. This loss limits interpretation of the relative share and differs from the L32 environmental contrast.
```

### 22. SI: M1 (1 occurrence)

Reason: Add the requested primary source pointers for the restored E2 account.

Old → new:

```text
OLD:
verdicts: `results/tier2/T2_COMPONENTS.csv`]

NEW:
verdicts: `results/tier2/T2_COMPONENTS.csv`; E2 definition and historical outcome: `REVISION_PREREG.md`, E; corrected competition report: `results/tier2/scored/E/REPORT.md`]
```

### 23. SI: M1 (1 occurrence)

Reason: Retain the failed E2 status in the validation account.

Old → new:

```text
OLD:
E1 remains not met.

NEW:
E1 and E2 remain not met; the historical and corrected-core E2 counts are reported in Supplementary Note 5.
```

### 24. SI: M1 (1 occurrence)

Reason: Use the requested Extended Data map wording.

Old → new:

```text
OLD:
failed E1 and explanatory predictions

NEW:
failed E1 and E2 abundance predictions, rejected explanatory predictions
```

### 25. MAIN: N6 (1 occurrence)

Reason: Cite the source of the finite-window periodicity bound after the existing citation to reference 14.

Old → new:

```text
OLD:
Fine–Wilf requires

NEW:
The Fine–Wilf periodicity theorem [15] requires
```

### 26. SI: N6 (1 occurrence)

Reason: Cite the theorem at the proof dependency.

Old → new:

```text
OLD:
The Fine–Wilf period lemma at the stated length

NEW:
The Fine–Wilf periodicity theorem [15] at the stated length
```

### 27. SI: N6 (1 occurrence)

Reason: Cite the finite-witness application of the same theorem.

Old → new:

```text
OLD:
The Fine–Wilf threshold for those periods

NEW:
The Fine–Wilf threshold [15] for those periods
```

### 28. MAIN: N6 (1 occurrence)

Reason: Add reference 15 with the requested DOI; metadata corroborated by the AMS bibliography and Wilf publication list.

Old → new:

```text
OLD:
(absent)

NEW:
15. Fine, N. J. & Wilf, H. S. Uniqueness theorems for periodic functions. *Proc. Am. Math. Soc.* **16**, 109–114 (1965). [https://doi.org/10.1090/S0002-9939-1965-0174934-9](https://doi.org/10.1090/S0002-9939-1965-0174934-9).
```

### 29. SI: C2 (1 occurrence)

Reason: Point to the complete local proof.

Old → new:

```text
OLD:
[Proof specialization: `manuscript/MAIN_nature.md`, Proposition 8; generated geometry: `results/capacity/geometry_traces.csv.gz`]

NEW:
[Proof specialization: Supplementary Note 1, ordered overlap theorem and Z80 specialization; generated geometry: `results/capacity/geometry_traces.csv.gz`]
```

### 30. SI: C2 (1 occurrence)

Reason: Remove obsolete proposition numbering.

Old → new:

```text
OLD:
### Z80 specialization of Proposition 8

NEW:
### Z80 specialization of the ordered overlap theorem
```

### 31. SI: C2 (1 occurrence)

Reason: Use the requested self-contained theorem name.

Old → new:

```text
OLD:
The overlap theorem gives ancestry

NEW:
The ordered overlap theorem (Supplementary Note 1) gives ancestry
```

### 32. SI: C2 (1 occurrence)

Reason: Point to the complete local proof without changing generated certificate paths.

Old → new:

```text
OLD:
historical derivation: `manuscript/MAIN_nature.md`, Proposition 8

NEW:
derivation: Supplementary Note 1, ordered overlap theorem and Z80 specialization
```

### 33. SI: C2 (1 occurrence)

Reason: The complete copy-cost argument is already local; remove the obsolete pointer.

Old → new:

```text
OLD:
 [Original statement and proof: `manuscript/MAIN_nature.md`, Theorem 1]

NEW:
(removed)
```

### 34. SI: C2 (1 occurrence)

Reason: Keep the local withdrawn-bound explanation and remove the obsolete pointer.

Old → new:

```text
OLD:
 [Scope corrections: `manuscript/MAIN_nature.md`, Theorems 1–2]

NEW:
(removed)
```

### 35. SI: C2 (1 occurrence)

Reason: Retain the current supporting record and full local argument.

Old → new:

```text
OLD:
[Corrected reading: `manuscript/MAIN_nature.md`, Theorem 6; `results/literal_channel/CHECK.md`]

NEW:
[Constructions and scope: `results/literal_channel/CHECK.md`]
```

### 36. SI: C1 (1 occurrence)

Reason: Move the measurement/property definitions from Numerical derivations into Supplementary Note 5.

Old → new:

```text
OLD:
### Initialization, detection and sampling



NEW:
### Initialization, detection and sampling

The four measurements are culture, execution confinement, partner inflow and transfer retention; the two properties are partner-information closure and parental-variant transmission. These definitions do not supply additional experimental denominators.


```

### 37. SI: C1 (1 occurrence)

Reason: Keep the mutation-setting definition explicitly in Note 5 as well as Methods.

Old → new:

```text
OLD:
The usual mutation setting attempts 512 uniformly located byte replacements per update;

NEW:
At noise exponent 4, the usual mutation setting attempts 512 uniformly located byte replacements per update;
```

### 38. SI: C1 (1 occurrence)

Reason: Move the missing cumulative-detection transformation to Note 5.

Old → new:

```text
OLD:
Competition summaries use world-level medians and retain undefined-ratio exclusions explicitly.

NEW:
Competition summaries use world-level medians and retain undefined-ratio exclusions explicitly. Cumulative detection is 1 minus Kaplan–Meier survival.
```

### 39. SI: C1 (1 occurrence)

Reason: Retain the physical displacement/unit conversion underlying the displayed intervention.

Old → new:

```text
OLD:
The two generation-eight intervention strips show parallel forward settings from one parent, not consecutive observations of a spontaneous mutation.

NEW:
The two generation-eight intervention strips show parallel forward settings from one parent, not consecutive observations of a spontaneous mutation. Displacement is reduced modulo the encounter-ring length P = 2L = 64 at L32; byte `0x60` gives 96 mod 64 = 32.
```

### 40. SI: C1 (1 occurrence)

Reason: Move the remaining starting-share, control, dial, arrival-group and claim-component definitions to Note 5, retaining their original source bracket.

Old → new:

```text
OLD:
### Statistical procedures



NEW:
Competition plots select three starting categories, `mix50`, `R_into_T` and `T_into_R`; reciprocal 1% invasions have initial T shares 0.01 or 0.99. The exploratory intruder comparison names two designs, R and T, and retains T_tiled as a separate control. The dial reports three outcome fields: open-first count, transmitter-majority count and confinement-time median. Transmitter majority requires more than 0.5 of heritable cells in the confined transmitter class. Arrival curves select three machine/cohort groups: Z80 G, 6502 A and 8086 S2. A claim component is one constituent registered criterion, and whole claims combine their component verdicts. These groupings do not supply extra independent experimental units. [manuscript/figures/out/v8/data/competition_endpoints.csv; results/offset/INTRUDER.md; manuscript/figures/out/v8/data/lethality_summary.csv; manuscript/figures/out/v8/data/arrival_worlds.csv; manuscript/figures/out/v8/data/arrival_censors.csv; results/tier2/T2_CLAIMS.csv; results/tier2/T2_COMPONENTS.csv]

### Statistical procedures


```

### 41. SI: C1 (1 occurrence)

Reason: Remove the submission-facing audit-file pointer; retain scientific data and preparation sources.

Old → new:

```text
OLD:
; per-occurrence source index: `manuscript/AUDIT_v7_RESOLUTION.md`

NEW:
(removed)
```

### 42. MAIN: C1 (1 occurrence)

Reason: Replace the audit appendix with the proposed section after checking every definition, threshold and unit; coverage and relocation ledger below. Duplicate arithmetic and source brackets remain in this provenance log.

Old → new:

```text
OLD:
### Numerical derivations

Result-path citations resolve from `experiments/`. Generated tables establish empirical counts; frozen contracts, scorers and figure-preparation code establish protocol thresholds, coordinate conventions and display choices. The numerical provenance index is `manuscript/AUDIT_v7_RESOLUTION.md`; detailed figure provenance is in Supplementary Note 5. The K/L32/seed5006 tape has six total scored sites, four unexecuted. The four measurements are culture, execution confinement, partner inflow and transfer retention; the two properties are partner-information closure and parental-variant transmission. These are definitions, not additional experimental denominators [results/crystal/cell_exceptions.csv; manuscript/AUDIT_v7_RESOLUTION.md].

Probability complements and normalizations use unit probability: cumulative detection is 1 minus Kaplan–Meier survival, and normalized culture gain is (after−before)/(1−before), over partners with before-match below 0.75. The illustrated L16 modal neighbourhood allows L/4 = 4 mismatches, hence requires at least 16−4 = 12 matching bytes. For L32, P = 2L = 64 and zero-based parent/partner addresses are 0–31/32–63; an edit from 0x04 to 0x60 changes displacement from 4 to 96 mod 64 = 32. With core size 4, the corresponding payload counts are 4−4 = 0 and 32−4 = 28. In the conceptual L12 diagram they are 4−4 = 0 and 12−4 = 8 bytes, drawn as two-byte units. A 16-byte `5F A4 x y` tape has 16/4 = 4 tile repeats. An inclusive source interval {s,…,s+d−1} contains d positions, and a zero-based core bounding prefix has length 1+max(K), distinct from the cardinality of a noncontiguous K [algocell_exp/assay.py; algocell_exp/metrics.py; manuscript/figures/out/v8/data/geometry_units.csv; manuscript/figures/out/v8/data/dial_coordinates.csv; manuscript/figures/out/v8/data/lineage_strips.csv; manuscript/figures/out/v8/data/first_copier_tiles.csv; REVISION_PREREG.md, S2 B2].

The lattice contains 160×125 = 20,000 cells. At noise exponent 4, the declared 8,192 pair draws give 8,192/2^4 = 512 mutation attempts per update; draws and attempts are not active encounters or applied changes. A share threshold of 0.005 is 0.5%, or 100/20,000 cells. At least half of 16 traces means at least 8/16; a strict majority means more than 0.5. Reciprocal 1% invasions have initial T shares 0.01 or 0.99; the latter leaves 1−0.99 = 0.01 for R. The S2 motif archive contains 20×585 = 11,700 scheduled top-three sample records, representing 11,700×3 = 35,100 ranked tape entries, and 20×5×20,000 = 2,000,000 checkpoint-cell observations. These repeated records are not independent worlds. Complete byte-word censuses contain 256^1 = 256 period-one and 256^2 = 65,536 period-two assignments. The registered 512 joint assignments comprise 256 counterbalanced plus 256 keyed random assignments; the exploratory panel used 64 independent joint assignments. For the 8086 reset SP = 1, a word push decrements SP by 2 modulo 65,536: the low byte maps from 65,535 to P−1 and the high byte from 0 to 0 for the declared P = 32/64 rings [score8086.py; s8086.py; algocell_exp/soup.py; dial_soups.py; manuscript/figures/out/v8/data/competition_endpoints.csv; results/s8086/exploratory/motif_scan.csv.gz; results/census8086/table_main.csv; results/census8086/LAW_CONTRACT.md; results/s8086/EXPLORATORY.md; algocell_exp/machines/i8086_pair_maps_L16.wgsl].

Display counts follow explicit selections. Fig. 1c contains one unique (seed,L) pair, (2002,16). Fig. 2b selects four keys, {pusher16, ret16, ldir32, genome20_s6003}, with one curve per key; Fig. 2c selects six, {pusher50, ret16, ldir32, lethal16_s4009, genome20_s6004, closer32_8080_p1}. Historical and corrected branching endpoints coincide in every plotted field, permitting one bar per design without merging their cohorts. Fig. 2d follows one marker lineage across its two ordered encounters. Fig. 3a contrasts the two settings d∈{4,12}; the Fig. 3b disc-area key contains three illustrative counts {1, 100, 2,000}, not additional observations. The two generation-eight strips select ldir32_v04 and ldir32_v60, each at lineage zero, as parallel interventions [manuscript/figures/out/v8/data/world_trajectory.csv; manuscript/figures/out/v8/data/serial_summary.csv; manuscript/figures/out/v8/data/branching_endpoints.csv; manuscript/figures/out/v8/data/erasure_manifest.json; manuscript/figures/out/v8/data/erasure_ordered.csv; manuscript/figures/out/v8/data/geometry_units.csv; manuscript/figures/out/v8/data/lineage_strips.csv; manuscript/figures/prepare_v7.py; manuscript/figures/build_v8.py].

Competition plots select three starting categories {mix50, R_into_T, T_into_R}; the exploratory intruder comparison names two designs {R,T} and retains T_tiled as a separate control. The dial reports three outcome fields, open-first count, transmitter-majority count and confinement-time median. Arrival curves select three machine/cohort groups {Z80 G, 6502 A, 8086 S2}; one terminal censor coordinate represents six coincident 8086 censoring records. A claim component is one constituent registered criterion, and whole claims combine their component verdicts. These set counts and grouping operations describe the displays and scorers, not extra independent experimental units [manuscript/figures/out/v8/data/competition_endpoints.csv; results/offset/INTRUDER.md; manuscript/figures/out/v8/data/lethality_summary.csv; manuscript/figures/out/v8/data/arrival_worlds.csv; manuscript/figures/out/v8/data/arrival_censors.csv; results/tier2/T2_CLAIMS.csv; results/tier2/T2_COMPONENTS.csv].



NEW:
### Source data and figure provenance

Generated source tables provide the figure coordinates and empirical counts. Supplementary Note 5 defines the cohorts, measurement units, exclusions and display selections; Supplementary Table 15 identifies the figure source data and provenance manifests.


```

### 43. MAIN: C3 (1 occurrence)

Reason: Remove the payment/computation aside; retain its source bracket.

Old → new:

```text
OLD:
 Displacement experiment C1 used local computation with no paid compute

NEW:
(removed)
```

### 44. MAIN: C3 (1 occurrence)

Reason: Remove the payment aside; retain its source bracket.

Old → new:

```text
OLD:
 The retention experiment C2 used no paid compute

NEW:
(removed)
```

### 45. MAIN: C3 (1 occurrence)

Reason: Use the proposed scientific scope statement.

Old → new:

```text
OLD:
No new world or paid remote computation was run, and the registered verifier and verdicts were unchanged

NEW:
No new population worlds were run, and the registered verifier and verdicts were unchanged
```

### 46. MAIN: C3 (1 occurrence)

Reason: Name the scientifically relevant content.

Old → new:

```text
OLD:
### Operational amendments

NEW:
### Restart and scoring continuity
```

### 47. MAIN: C3 (1 occurrence)

Reason: Retain restart, exclusion timing and scoring continuity; leave operational details in the cited archive.

Old → new:

```text
OLD:
Operational amendments changed publication, dispatch and scoring recovery without changing scientific definitions. Failed attempts, restarts and the full amendment chronology are retained in Supplementary Note 6 and the reproducibility records

NEW:
Publication failure required a restart from initialization before biological outputs were read; no trajectory was spliced. Scoring recovery did not rerun completed worlds or change scientific definitions (Supplementary Note 6)
```

### 48. MAIN: C3 (1 occurrence)

Reason: Keep the scientific continuity statement without operational narration.

Old → new:

```text
OLD:
operational recovery records identify subsequent changes without rewriting prior history

NEW:
subsequent restart and scoring recovery preserved the scientific definitions
```

### 49. MAIN: C3 (1 occurrence)

Reason: Match the revised scientific account.

Old → new:

```text
OLD:
registration chronology and operational amendments (Supplementary Note 6)

NEW:
registration chronology and restart exclusions (Supplementary Note 6)
```

### 50. MAIN: C3 (1 occurrence)

Reason: Remove operational and billing inventory prose while preserving all archive pointers.

Old → new:

```text
OLD:
source and download inventories, failed-attempt provenance, eligibility release, operational amendments and billing reconciliation

NEW:
source and download inventories, failed-attempt provenance and eligibility release
```

### 51. MAIN: C3 (1 occurrence)

Reason: Describe reproducibility without command-level framing.

Old → new:

```text
OLD:
; commands are recorded in `manuscript/figures/out/v8/FIGURES_V8.md`

NEW:
; reproduction instructions are recorded in `manuscript/figures/out/v8/FIGURES_V8.md`
```

### 52. SI: C3 (1 occurrence)

Reason: Name the scientifically relevant content.

Old → new:

```text
OLD:
## Supplementary Note 6. Registration and operational chronology

NEW:
## Supplementary Note 6. Registration chronology and restart exclusions
```

### 53. SI: C3 (1 occurrence)

Reason: Remove spending-cap detail from the chronology; retain all scientific amendments and decision timing.

Old → new:

```text
OLD:
; cap successively superseded by $95

NEW:
(removed)
```

### 54. SI: C3 (1 occurrence)

Reason: Remove command, dispatch, timeout and lock minutiae; retain scientific restart/scoring sequence.

Old → new:

```text
OLD:
Noninteractive stop, volume publication/restart, parallel dispatch/9,000-second world limit, then scoring lock recovery

NEW:
Publication failure and restart from initialization, followed by scoring recovery
```

### 55. SI: C3 (1 occurrence)

Reason: Keep the scientific follow-up scope without payment detail.

Old → new:

```text
OLD:
no new population world or paid remote compute

NEW:
no new population world
```

### 56. SI: C3 (1 occurrence)

Reason: Focus the account on restart exclusion and source identity.

Old → new:

```text
OLD:
### S2 operational record

NEW:
### S2 restart and source continuity
```

### 57. SI: C3 (1 occurrence)

Reason: Retain the failed world identity, exclusion before outcome inspection, restart reason, missing continuation state, nondeterminism and scoring-only recovery; remove command/dispatch/lock/payment minutiae.

Old → new:

```text
OLD:
A failed detached-app stop was repaired by making the command noninteractive. World 8086001 subsequently completed computation but hard-link publication failed; before reading biological output, the attempt was designated noncanonical and retained by opaque hashes. Publication changed to same-directory atomic rename, and the same seed restarted from initialization. Checkpoints lacked full continuation state, so no trajectory was spliced. A separate amendment authorised parallel dispatch and the measured 9,000-second world body limit. After all worlds completed, transient ledger lock contention falsely terminated scoring; bounded retries and diagnostic cause labels repaired the control path, and only scoring onward resumed. These changes preserved scientific source, machine, sampling, eligibility and endpoint definitions. Failed attempts and charges remain archived; nondeterministic restart does not promise the same trajectory.

NEW:
World 8086001 completed computation but failed to publish its output; before reading biological output, the attempt was designated noncanonical and retained by opaque hashes. After the publication failure was repaired, the same seed restarted from initialization. Checkpoints lacked full continuation state, so no trajectory was spliced. Nondeterministic restart does not promise the same trajectory. After all worlds completed, a scoring interruption was repaired and only scoring onward resumed, without rerunning completed worlds. These changes preserved scientific source, machine, sampling, eligibility and endpoint definitions.
```

### 58. SI: C3 (1 occurrence)

Reason: Replace billing/provider details with source and hash verification and a single neutral operational-archive sentence; retain original source bracket.

Old → new:

```text
OLD:
The final reconciliation recorded $13.22378749 in app-attributed charges, reported as $13.22, including preparation and earlier attempts. Billing may lag; this is a metered reading, not a settled invoice. The conservative retained bound was $77.32835455 against the amended $95 cap. All attributed apps were provider-confirmed stopped. Source, world/checkpoint identities and control hashes were verified; unchanged reporters generated the final verdicts. The archive preserves terminal failed ledgers rather than clearing their flags, binds the prior execution freeze to each exact completed world’s hash and records renewed scoring authority separately.

NEW:
Source, world/checkpoint identities and constructed-control hashes were verified against the archived manifests; unchanged reporters generated the final verdicts. The manifests bind the prior execution freeze to each exact completed world’s hash and retain the excluded attempt’s identity. Failed attempts, restarts and operational amendments remain documented in the provenance archive.
```

### 59. SI: C3 (1 occurrence)

Reason: Preserve biological interpretation and amendment timing without duplicating operational-record prose.

Old → new:

```text
OLD:
Operational incompleteness does not count as a biological negative. Arm reduction, R2 withdrawal and the source-root correction were scientific amendments before population data; publication, dispatch and ledger recovery were dated operational amendments. Historical launch receipts document the operational record.

NEW:
Operational incompleteness does not count as a biological negative. Arm reduction, R2 withdrawal and the source-root correction were scientific amendments before population data; subsequent restart and scoring recovery preserved those definitions.
```

### 60. SI: C3 (1 occurrence)

Reason: Retain archive count, integrity, sources and scientific scope while removing payment/accounting narration.

Old → new:

```text
OLD:
The publication archive preserves 239 historical launch and accounting files byte-for-byte or by deterministic gzip, with original/stored hashes and lengths. No new population, paid remote computation or changed registered verifier was involved in these follow-ups. Committed tables and the resource ledger document the follow-ups.

NEW:
The publication archive preserves 239 provenance files byte-for-byte or by deterministic gzip, with original/stored hashes and lengths. No new population worlds were run, and the registered verifier and verdicts were unchanged. Committed tables document the follow-ups.
```

### 61. MAIN: C4 (1 occurrence)

Reason: Clarify the grammatical attachment of the first core-name introduction.

Old → new:

```text
OLD:
Historical and corrected Z80 core (core C1) trajectories are summarized separately at world level

NEW:
Trajectories from the historical implementation and the corrected Z80 core (core C1) are summarized separately at world level
```

### 62. MAIN: C3 (1 occurrence)

Reason: Attach the retained source bracket to the sentence after removing the paid-compute aside.

Old → new:

```text
OLD:
conditional mathematics. [results/capacity/serial.csv.gz

NEW:
conditional mathematics [results/capacity/serial.csv.gz
```

### 63. MAIN: C3 (1 occurrence)

Reason: Attach the retained source bracket to the sentence after removing the paid-compute aside.

Old → new:

```text
OLD:
registered directional prediction. [evolved6502_retention.py

NEW:
registered directional prediction [evolved6502_retention.py
```

### 64. MAIN: M1 (1 occurrence)

Reason: Make the Methods validation account explicitly consistent with the restored E2 failure; existing component-ledger source bracket remains unchanged.

Old → new:

```text
OLD:
Preserved verdicts include negative outcomes.

NEW:
Preserved verdicts include the failed E1 and E2 abundance predictions (Supplementary Note 5).
```

### 65. MAIN: C1 (1 occurrence)

Reason: Retain the quantitative stack decrement and architectural modulus from Numerical derivations in the execution contract.

Old → new:

```text
OLD:
At the registered ring sizes, the first word push decrements SP and writes low then high bytes at ring positions P−1 and 0.

NEW:
A word push decrements SP by 2 modulo 65,536. At the registered ring sizes, the first word push writes low then high bytes at ring positions P−1 and 0.
```

### 66. SI: C1 (1 occurrence)

Reason: Keep the cell-count equivalent of the detection gate in Supplementary Note 5, alongside the fractional definition in its cohort table.

Old → new:

```text
OLD:
S1 specified 64 culture partners, whereas its cited frozen final-cell estimator used 32;

NEW:
The detection share threshold of 0.005 is 0.5%, or 100/20,000 cells.

S1 specified 64 culture partners, whereas its cited frozen final-cell estimator used 32;
```

### 67. SI: C1 (1 occurrence)

Reason: Retain the display-merging definition while preserving independent cohort identities.

Old → new:

```text
OLD:
Historical and corrected-core X1 endpoints agree for every plotted outcome and side-column count.

NEW:
Historical and corrected-core X1 endpoints agree for every plotted outcome and side-column count. The display therefore uses one bar per design while keeping the cohorts separate.
```

### 68. SI: M1 (1 occurrence)

Reason: Give the restored E2 account its own continuous-prose paragraph.

Old → new:

```text
OLD:
variation across worlds does not establish neutrality. With mutation on, E2

NEW:
variation across worlds does not establish neutrality.

With mutation on, E2
```

## Final verification

Exact replay of all logged replacements: PASS for both files. MAIN content before Methods, including Figure legends: byte-for-byte PASS against the saved baseline. First-citation order: 1–15, PASS. Obsolete SI manuscript/proposition pointers and review-audit pointer: absent. Payment/provider/command narration: removed. `git diff --check`: PASS. No simulations, figure rebuilds or commits were performed.

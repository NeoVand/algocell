# I1-T2 — fixed-core claim re-evaluation (DRAFT, NOT REGISTERED)

Prepared source commit: `97376d3d5148cd725d6c2fd0e5f38c62015c9d43` on `feat/tier2`, based on
`feat/experiments-runner` at `0970b82`.
No registered tier-2 world has been run. The coordinator must settle the decisions below,
edit this draft, bind the final source commit and artifact hashes, and register before dispatch.
The approved aggregate spend ceiling is **$40**, including preparation and scoring.

## Scope and frozen design

Re-run the claim-bearing experiments flagged by I1 under **C1 = Zilion Z80 compat
`0.3`, variant `classic`**, vendored from Zilion `183ff5c7429403439c664487980a40bf62f49b7b`.
Exclude flagged Stage A, Stage C, random corpora, and exploratory Stage E arms.
Use [INVENTORY.md](INVENTORY.md) for criterion wording, original runners, published outcomes,
registration sections/line references, and archive paths.

The proposed complete design is [DESIGN_FULL.json](DESIGN_FULL.json): **232 worlds,
250,400,000 registered simulation steps**. Each condition preserves the original schedule,
initialisation, lattice, collision handling, encounter stream, mutation, masks, rule and ring;
only its execution core changes. The shared soup design is square 160 × 125 = 20,000 cells,
8,192 drawn pairs per step, 128 instructions per encounter, k = 4 = 512 mutated bytes
per step, unless the original condition disables mutation.

| Experiment | Worlds and seeds | Horizon | Ring / convention / rule |
|---|---|---|---|
| DZ | 60: ten seeds 9001–9010 at each p = 0, .01, .03, .1, .3, 1 | 300,000 | L16/P32, standard registers/SP, original raw-hook lethal-fetch PCG stream |
| E | 90: L16 benign, L16 lethal, L32 benign; each ten mix50 mutation-off, ten mix50 on, five 1% invasions each direction; seed IDs 1–10 / 1–5 | off 100,000; on 300,000 | P=2L, standard registers/SP; original random transmitter tails and seed formula |
| E-L32L | 20: ten mix50 and five each invasion, mutation-on, original seed IDs | 300,000 | L32/P64, deterministic lethal fetch, standard registers/SP |
| Stage L | 20: ten at each L16 and L20, seeds 6001–6010 | 10,000,000 | P32/P40, plain, standard registers/SP; complete original snapshot arrays |
| R4 | 32: five per invasion direction/convention plus three unseeded controls per resident/convention | randreg 50,000; standard 100,000 | L16/P32; raw-PCG random registers or standard; original SI13 tails/1% starts/mutation |
| Stage F F4c | ten original seeds 101–110, `none@ring32` | 300,000 | L8/P32 padded ring, plain, standard registers/SP |

For E and R4, seed IDs are not soup seeds. Preserve the formulas in the inventory and
original runner, including E's start and mutation-off offsets. Preserve the frozen runner's
one-iteration overshoot in saved E/R4 `.npy` endpoints; score the registered horizon shares
from the journals, and final-cell assays from the saved endpoints, as published.

Re-run all **14 original serial-panel genotypes** using `serial_retention.py` and all
**five branching genotypes**, with U for all and K for pusher16/pusher64, using
`branching_transfer.py`. Keep every registered alternative-value count, replicate count,
generation, copy threshold, population cap and RNG stream. The panel tapes, masks, rules,
source-table SHA-256s, and all 80 read-only K-pool archive hashes are frozen in
[control_inputs.json](control_inputs.json). These are fixed assay inputs; do not substitute
tapes chosen after inspecting new C1 soups. All assay encounters execute C1.

## Machine and scoring freeze

Soup machines are the existing `imp_<world>_c1` entries in `worlds_impact.json`.
Assays use matching C1 machines in `worlds_tier2.json`; these preserve every execution-world
field and raw hook, adding only observational fetch/start maps. Their C0 twins were checked
against the original executors and ZSoup bridge. Select the matching world for every dial,
random-register, lethal, padded-ring and ablated assay. A plain C1 machine is not an acceptable
substitute for a dial or random-register world.

[machines.csv](machines.csv) freezes **core, world, complete WGSL and manifest SHA-256s**
for all used soup machines/twins and all vendored assay machines. The WGSL SHA-256 column
is the machine-file identity. Bind this CSV's own SHA-256 in the final registration.
[scoring_sources.json](scoring_sources.json) records original and prepared scoring-script
SHA-256s, plus unchanged AST hashes for the numerical scoring functions. Script changes are
explicit input/CLI/backend routing; the registered numeric scoring is retained.

Run the complete commands in the inventory with:

```sh
PY=/Users/neo/repos/algocell/experiments/.venv/bin/python
R=<C1-only downloaded runs root>
O=<separate C1 scoring output root>
```

On Modal use the image's Python and corresponding mounted paths, with the same frozen
sources. The Stage L/F `assay_batch.py` and `c4_functional.py`, and the DZ/E/R4 paths, reject
C0-world inputs when the tier-2 C1 scorer is selected. Log assay backend usage and preserve
all generated tables. Every long GPU scoring command belongs on Modal.

Scoring entrypoints:

- DZ: `dial_soups.py --analyse --machine c1`; keep all six levels for both rank trends and
  the original KM/permutation scoring. Score DZ1, DZ2 **and DZ3**, including every component.
- E/E-L32L: `offset_switch.py --analyse --machine c1`; the byte classifier is core-independent.
  Read registered E2's end-below-start conjunction and E-L32L's >.5 / <.4 thresholds from the
  frozen start/end CSV. `offset_encounter.py --machine tier2c1_assay` supplies ED6e's
  descriptive encounter components without inventing a binary verdict.
- L: `assay_batch.py`, `c4_functional.py`, `stage_g.py --no-verdicts`, and
  `capacity_over_time.py --machine tier2c1_assay --stages L`, then unchanged `score_L.py`.
  Include both lengths for the registered joint kill.
- Serial and branching: full panels on `tier2c1_assay`, then their unchanged `--report`
  functions. K pools remain the registered archived inputs.
- R4: `r4_analyse.py --machine c1`, paired with the runner's motif-share journals. Set
  `--seed-path-root /Users/neo/repos/algocell/experiments/runs/r4_inv/r4_inv` to preserve
  the original path-length RNG salt.
- F: `assay_batch.py --machine tier2c1_assay`, then `stage_f.py --f4c-only`; the registered
  F4c test uses the frozen `cell` KM/emergence scoring and all ten original seeds.

## Decision rule

**A claim stands under C1 if every one of its registered verdicts—met or not met—is the
same as published; otherwise report the changed verdict and affected claim.** Compare
components as well as compound/partly-met outcomes, including registered kill criteria.
Do not treat a merely similar numerical estimate as verdict equality. Do not reinterpret
smoke outcomes or use them for selection. Descriptive/no-direction arms receive their
frozen tables and a report of changes; no new pass/fail or winner criterion is invented.
Missing required worlds or budget-stopped worlds leave the relevant verdict incomplete.

## Decisions the coordinator must resolve before registration

1. Include DZ3, since ED8d depends on it; the proposed design does so.
2. Confirm inclusion of both Stage L lengths, all E arms and E-L32L, and the complete
   serial/branching panels. They preserve registered multi-arm decisions within the cap.
3. **R4 wording:** the literal C4b SI13 motif-majority component is not met (0/5 original
   worlds), while confinement and reverse pusher suppression are met (5/5).
   Published prose interprets the closed descendants as displacement; it does not justify
   changing the registered motif test to descendant majority. Freeze separate original
   motif and confinement verdicts, and state how the descendant claim is reported.
   C4c registered no direction and has no binary winner verdict to compare.
4. Preserve/report scorer omissions and wording discrepancies explicitly: DZ's printed report
   omits the KM reversal decision; E's printed E2 count omits below-start; E-L32L's threshold
   is absent from that report; Stage L's frozen L2 implementation allows ≥half sites
   unexecuted whereas the registration wording suggests all sites. The proposed policy is
   to report the frozen script result and the literal registered component separately.
   Resolve the published baseline for each before C1 data; do not change frozen formulas.
5. F4c is registered in `PLAN.md:260–277`, not `REVISION_PREREG.md`; cite the actual source.
6. **Additional claim-bearing flag:** Stage E `none@nominal_L10_st128_k4@emergence` supports
   ED2b and is distinct from experiment E. Its optional ten-world original nominal-L10
   addendum would cost approximately $0.5 on L4 plus existing overhead, leaving the design
   under $40. Decide whether to include it and freeze its original conditions/scoring first.
   It has descriptive emergence evidence; do not invent a new registered direction.
7. Bind the preparation commit and hashes below in the final registration. Review/freeze
   [PUBLISHED_VERDICTS.csv](PUBLISHED_VERDICTS.csv), the 36-component baseline ledger,
   before dispatch; the R4 literal-motif row needs the wording decision above.

## Cost and execution boundaries

Use **NVIDIA L4**, one reserved physical CPU core and 8 GiB/container, four-hour timeout
for Stage L, with no unbudgeted automatic retries. [COST.md](COST.md) and [cost.csv](cost.csv)
give the reproducible measured-rate estimate: **$34.99** for the complete 232-world design,
all scoring/control assays, startup, storage and preparation reserves. Nothing is dropped.
The L40S-only envelope is **$58.37**, above the cap; L4 changes hardware while preserving
every world's registered design. GPU collision scheduling was already nondeterministic;
the equivalence claim is execution on identical drawn/active pairs, not independently
identical whole-soup trajectories across devices.

The estimate uses the slowest fresh-state timing at every C1 world, an additional 20%
throughput reduction and 10% runner overhead; mature Stage L timing is faster and not used
to reduce the budget. Price source: [Modal pricing](https://modal.com/pricing), 2026-10-11.
Maintain an aggregate actual/forecast spend ledger. If required remaining work no longer
fits under $40, stop dispatch, retain completed outputs, and have the coordinator amend
the registration; do not truncate horizons/seeds/p levels or score an incomplete test.

The preparation app `modal_tier2.py` accepts only bounded smoke/validation/performance
actions and writes worlds solely under `tier2_smoke/`. The five C1 smoke worlds at seed
424242 and horizon 2,000, and the performance/equivalence inputs, are excluded from I1-T2.
No registered runs, registered endpoint scores, push or registration was performed here.

## Prepared artifact SHA-256 bindings

- `DESIGN_FULL.json`: `ebc567db3b43b25c60bbfdbfef6556ec0f420807009ed44ffe9f3f8f456d4ffb`
- `machines.csv`: `d17e5f9b9eda1233e638e050e453f149c4ef0ebb74664f686e1777a9f58cacad`
- `scoring_sources.json`: `375442553209e4729ea6bd4288297018b7da180a1c587b04e1a8c01b106317ed`
- `control_inputs.json`: `e122e8a664787b68ddce80165969c3f64a4e567f29c5d46562bfc05591a321cd`
- `PUBLISHED_VERDICTS.csv`: `06bf82a5994fb3ddfa6e0019c74c7bc2252de868220ab277288cf9565542c4f9`

C1 soup machine WGSL SHA-256s (core/world/manifest identities and mapped twins are in `machines.csv`):

| Machine | WGSL SHA-256 |
|---|---|
| `imp_L16_c1` | `92b11dec76fadf3fc9cb5d2d42d7023abab198bd6bc1ec1e6c8efba66593da2e` |
| `imp_L20_c1` | `9b8878fc8b104c651ff37f3bdb7db63faa30ae76fe5f7044a507d8351bdea4ee` |
| `imp_L32_c1` | `cea34e536ee41eba30757958728f36ba2f8f141081fd4903dde624b860e58ffc` |
| `imp_L8_P32_c1` | `f03caf10d734d0b1de4109d209add1c917435cf12f0e93d418cea4b3ec94fce2` |
| `imp_lethal_L16_c1` | `3a76461357c5da0aa26b6b0a79b11b4210c59184c9bf0c0f2ca69eb2e25c83c8` |
| `imp_lethal_L32_c1` | `26fa0a2b10fefa714830da9ec38af050133dc2fdd2d0a5de7afa704bf64fc383` |
| `imp_lp001_L16_c1` | `08bee6fa7b62115e4a02672251c86601e9553480d1eb0ed14177ed83a0c0921e` |
| `imp_lp003_L16_c1` | `5c751281485e70da4b1859d158efe54ad85e60172494eb3d43309c9553559283` |
| `imp_lp01_L16_c1` | `7fe23fdb528c8e69cac30af086f3c9d22595ae29f281c98a52d58ea1ffff4545` |
| `imp_lp03_L16_c1` | `46459d45f490c38a0b1ebb0e8ebf019b17c228b92bbe587bbc1877d74d9a392f` |
| `imp_lp0_L16_c1` | `c19db0e6a29eba18476af179bae2fa3cc5d482011b3dd2f6ce10543910bc47d5` |
| `imp_lp1_L16_c1` | `b3707e495f72e356d7ce155719a2677193d9f10f1aa252845d12fc76ebf9df35` |
| `imp_randreg_L16_c1` | `037cf89695de5837347a5f9a4fb59f9d465541299ac47c878a89dd1119d92560` |

The original Modal wrappers prefer L40S by default. The coordinator must explicitly override
GPU/resources to L4/one CPU core/8 GiB and disable automatic retries when wiring the
registered launcher; the costed design must not dispatch those higher-cost defaults.

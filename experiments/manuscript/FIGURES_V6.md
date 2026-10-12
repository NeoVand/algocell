# Figures v6: data build and coordinator handoff

Branch: `feat/figs-v6`, based on `feat/experiments-runner` at `b4e834e`.
Worktree: `/Users/neo/repos/algocell-wt-figs`. Nothing is pushed.

All requested data panels are exported individually and in composed figures at
`figures/out/v6/`: PDF, editable-text SVG, and PNG. Pages are exactly 180 mm wide;
PNGs are rendered from the PDFs by Poppler at 600 dpi (4,252 pixels wide).
Type is Arial, 5.5–8 pt (body predominantly 6–6.5 pt); figures contain at most
four labelled panels. Figure 4 is 170 mm deep. Teal/vermilion and structural
greys serve Figures 1–3; teal/house purple serve Figure 4. No conceptual drawing
was made. Layout decisions precede code in `figures/LAYOUT_V6.md`.

**Completion limits.** Figure 4a has the complete historical trajectories but
only C1 endpoint medians: the 40 corrected-core C1 E trajectory JSONLs are not
in the local archives. The slots state that trajectories are unavailable;
no line is interpolated between endpoints. Figure 3d has exact joint challenges
and a separate constructed eligibility screen, but the existing tables do not
contain joint controls for altered phase or inadequate forward coverage.
Those two controls remain outstanding and are disclosed in the caption.
Designer panels 1a, 1b, 3a await `DESIGNER_ROUND6.md`; evolved 8086 points and
Figure 5 wait for S2 (the prospective second-stage 8086 study). Figure 4d was
not requested. These limits keep the current files reviewable without inventing
results and must be resolved before submission.

**Independent art clearance.** Figures 1 and 2 passed the third image-only
critique. Figure 3's third critique asked for separate score/capacity labels
and a defined constructed ordinate; Figure 4's asked for operational event
definitions and registration thresholds. These were corrected after round 3.
No fourth critique was run, in accordance with the user's three-round cap.
The latest Figures 3 and 4 are technically clean but are **not claimed to have
independent final art clearance**. `out/v6/post_critique_fixes.json` records the
corrections and image hashes; the coordinator must resolve this sign-off.

The label **C1 in Figure 3** denotes the registered displacement-capacity
intervention. **C1 in Figures 1, 2 and 4** denotes the corrected-core rerun
cohort. Their source tables and units are kept separate.

## Panel inventory

| Panels | Content | Generated sources (relative to experiments/) | Status |
|---|---|---|---|
| 1a, 1b | Abstract encounter and mechanism briefs | `manuscript/DESIGNER_ROUND6.md` | Designer reserved |
| 1c | Illustrative world; separate branching, entry and inflow reference assays | archived `runs/video/video_L16_st128_k4_s2002.jsonl`; `results/branching/branching_summary.csv`; `results/exectrace/per_replicator.csv` | Built; historical, not a median history |
| 1d | Distinct instruction/encounter/assay/ecological validation units | `results/impact/outcome.csv`, `validation.csv`; `results/tier2/T2_COMPONENTS.csv`, `T2_CLAIMS.csv`; absolute generated Zilion 8086 architectural JSON baseline | Built; 8086 ecology blank |
| 2a | Controls alive, mutant populations alive and carriage; C1 coincides with historical; X1-2 prediction failed | historical and `results/tier2/scored/Branching/branching_summary.csv` | Built; registered |
| 2b | Every alternative at every position, loss/eraser/carrier, fetched ticks and rare-position counts | `results/serial_retention/sr_mutants.csv.gz`, `sr_sites.csv` | Built; historical serial, any-carrier map distinct from scored sites |
| 2c | Complete ordered erasure archive and descendant reproduction | frozen existing `capacity_common.Z80Provenance`; `figures/out/v6/data/erasure_ordered.csv`, `erasure_states.csv`, `erasure_manifest.json` | Built; exploratory |
| 2d | Full six-parent × six-environment result | `results/pe/pe_matrix.csv`; registered serial protocol | Built; lineage counts and site counts distinct |
| 3a | Abstract copy geometry brief | `manuscript/DESIGNER_ROUND6.md` | Designer reserved |
| 3b | Sampled Z80 non-core payload; Z80 world means; exact evolved 6502 and constructed controls | `results/offset/offset_cells.csv`; `results/crystal/cells.csv`, `cell_exceptions.csv`; `results/evolved6502/eligibility.csv`, `exact_cases.csv.gz`; `results/census6502/table_closers_law.csv` | Built; Z80 exploratory; evolved 8086 empty |
| 3c | C1 inherited edits; measured g8 descendant capacity 0–60; reciprocal edit; exact full descendant sequence example in caption | `results/capacity/edit_eligibility.csv`, `serial.csv.gz`, `capacity_cases.csv.gz`; exploratory `results/crystal/alleles_ldir32.csv`, `allele_encounters_ldir32.csv` | Built; 101 edits, 22 archived parents; exploratory dial separated |
| 3d | Exact all-value, joint and erased challenges; constructed eligibility count | `results/capacity/capacity_cases.csv.gz`; `results/evolved6502/exact_cases.csv.gz`; `results/census6502/table_closers_law.csv` | Built data; phase/coverage joint controls outstanding; 8086 empty |
| 4a | Historical L32 trajectories; side-by-side C1 endpoint strips at 300k | historical `runs/offset/*`; historical and `results/tier2/scored/E/offset_switch.csv` | Partial: needs C1 raw trajectories |
| 4b | Every world's ratio, absolute core abundance and nested tail diversity; undefined ratio disclosed | historical and C1 `offset_switch.csv` | Built; 80 worlds, one core-loss world retained |
| 4c | Historical/C1 open-first, transmitter-majority, KM confinement; original failures and remaining reversal | historical and `results/tier2/scored/DZ/dial_worlds.csv`; `results/tier2/T2_DZ.csv`, `T2_COMPONENTS.csv` | Built; world unit, censoring explicit |

## Build and audit

Use the specified Python from this worktree:

```sh
FIGS_PY=/Users/neo/repos/algocell/experiments/.venv/bin/python
"$FIGS_PY" experiments/manuscript/figures/archive_erasure_v6.py
"$FIGS_PY" experiments/manuscript/figures/prepare_v6.py
"$FIGS_PY" experiments/manuscript/figures/build_v6.py
"$FIGS_PY" experiments/manuscript/figures/verify_v6.py
"$FIGS_PY" experiments/manuscript/figures/write_v6_manifest.py
```

`prepare_v6.py --c1-runs
/path/to/local/C1/logs` will reconcile every raw endpoint before adding its
trajectory. The default historical archive is read-only outside the worktree.
Plotting CSVs and source hashes are committed under `figures/out/v6/data/`;
every plotted point and displayed count is derived by the committed scripts.
The full generated result tables remain their existing committed sources.
`data/sources.csv` hashes every source table/raw log and the absolute 8086
baseline. Coincident scatter coordinates are merged; counts remain in tables.
World means are grouped by world and prediction; cells are not treated as
independent worlds. No error bar or interval is implied where none is drawn.

`archive_erasure_v6.py` records all fetch/read/write events, checks final memory
and registers against the uninstrumented executor, and tracks the original
marker ancestry through the actual descendant. Local GPU only; no Modal was
invoked. Burst accounting includes compilation and readback, and enforces a
58-second alarm. The archived run's longest burst and total activity are below.

Every build invokes the existing `figcheck.py` for all panels, compositions and
rendered captions. Additional checks catch text leaving its reserved slot and
scatter points outside plot limits. `figcheck.json` contains the current reports.
The independent critic is `codex exec -m gpt-6-astra` with high reasoning in an
empty temporary directory. It receives only PNGs and is instructed not to use
tools. Each round archives its actual inputs, image hashes, prompt, review and
CLI logs in `out/v6/critique/figN/roundN/`. Rounds 1 and 2 were followed by fixes;
round 3 includes the captions as rendered PNGs. Required label/caption fixes
for Figures 3 and 4 were then applied within the three-round limit, without
claiming a further independent review. `invocation_failed/` archives
an initial CLI argument-parsing failure, not an art-editor round. There were
exactly three actual critiques per figure, no fourth. `qa_manifest.json` records
the physical-export checks, current/reviewed PNG hashes and independent-clearance
status separately. `verify_v6.py` exits 1 while art clearance remains pending,
even when all technical checks pass.

| Figure | figcheck (panels, composition, caption) | Image-only critic rounds |
|---|---|---|
| 1 | clean | [round 1](figures/out/v6/critique/fig1/round1/critique.md): SUBSTANTIVE DEFECTS; [round 2](figures/out/v6/critique/fig1/round2/critique.md): SUBSTANTIVE DEFECTS; [round 3](figures/out/v6/critique/fig1/round3/critique.md): NO SUBSTANTIVE DEFECTS |
| 2 | clean | [round 1](figures/out/v6/critique/fig2/round1/critique.md): SUBSTANTIVE DEFECTS; [round 2](figures/out/v6/critique/fig2/round2/critique.md): SUBSTANTIVE DEFECTS; [round 3](figures/out/v6/critique/fig2/round3/critique.md): NO SUBSTANTIVE DEFECTS |
| 3 | clean | [round 1](figures/out/v6/critique/fig3/round1/critique.md): SUBSTANTIVE DEFECTS; [round 2](figures/out/v6/critique/fig3/round2/critique.md): SUBSTANTIVE DEFECTS; [round 3](figures/out/v6/critique/fig3/round3/critique.md): SUBSTANTIVE DEFECTS |
| 4 | clean | [round 1](figures/out/v6/critique/fig4/round1/critique.md): SUBSTANTIVE DEFECTS; [round 2](figures/out/v6/critique/fig4/round2/critique.md): SUBSTANTIVE DEFECTS; [round 3](figures/out/v6/critique/fig4/round3/critique.md): SUBSTANTIVE DEFECTS |

Hashed source inputs: 68. Longest local GPU burst: 0.741579 s; total active: 2.019798 s; pauses recorded beside each burst.

## Caption and legend notes

These notes update the older pending wording in `MAIN_v6.md` without editing the coordinator's manuscript. Each is also supplied as a 180 mm, 600 dpi, 6.5 pt caption PNG for the art-editor review.

### Figure 1

Figure 1 | Zilion makes reproduction an experiment.

a,b, Reserved for the designer: the encounter contract and three abstract reproductive mechanisms.

c, One historical illustrative world (seed 2002; 20,000 cells, 16 bytes/cell). Teal counts cells with at least 12 of 16 bytes matching the contemporaneous modal tape, divided by 20,000 cells; grey counts zero-valued bytes, divided by 320,000 bytes. This is a sequence-similarity neighbourhood, not a spatial-neighbour count or a median world history. The separate reference assays use the pusher16 and ret16 tapes; they are not asserted to descend from this video world. Survival is generation-eight unmutated branching populations alive out of 64. Retention is thresholded positions out of the tape length. Partner entry counts encounters that fetch at least one instruction byte in partner addresses, out of 256 fresh-partner encounters. Inflow is offspring entropy across those partners (bits), a separate measure from entry. The reference pusher and return-closer traces use the archived G/L16/seed2001 first and final records.

d, Validation coverage retains its counting units: raw architectural passes among 8088 hardware vectors, historical-to-C1 unchanged encounter conditions, identical isolated assay cases, and retained registered claim verdicts/components. Expected hardware divergences are shown separately; queue timing is outside that vector contract. Instruction-vector counts are not evidence of ecological replication. 8086 population measurements await S2, the planned prospective second-stage 8086 study. No population result is inferred from this blank slot.

### Figure 2

Figure 2 | Survival is not retention.

a, Registered uniform-partner branching assays through generation eight. Black: surviving unmutated populations / 64 starting control populations. Red: surviving mutant populations / all starting mutant populations. Teal: mutant populations carrying the introduced marker / all starting mutant populations, not just survivors. There are 32 tested alternative values/position and four independently founded replicate populations/alternative; the displayed product is mutation types × four populations. Replicates are pooled by their mean across mutation types. A population carries when at least half its members contain the mutant's centred five-byte window, allowing cyclic alignment. The generation-eight site count uses the registered background-corrected retention criterion for at least half of tested alternatives at a position. C1 measurements exactly coincide with the historical trajectories: open C1 circles are drawn on historical solid lines. X1-2 was a completed experiment whose prediction failed (no scored site), not an unavailable or failed-to-run assay.

b, Complete historical serial-transfer maps, all 255 alternative values/position, 16 serial lineages/alternative, generation eight. Mutually exclusive displayed classes: teal if any of the 16 lines survives carrying its introduced five-byte marker window; otherwise red if any line survives without the marker; otherwise light grey if all lines are lost. Dark neutral cells mark unchanged parental values, which are untested. These any-carrier maps are not thresholded site maps. Black ticks mark fetched positions. The aligned summaries locate rare carrier alternatives at fetched positions: block-copy R position 0 has 16 alternative–position combinations; transmitter positions 2 and 3 have 6 and 50, respectively. These count mutation types, not sites or replicate lines.

c, One exploratory ldir32 lineage, with a parental mutation at position 8, followed through two complete 128-step encounters against zero partners. Dots show every write in executor order; the archive contains all 1,014 fetch/read/write events. The circle marks overwrite at step 6. The lower trace counts ring bytes descended from the original introduced byte; it falls from one to zero and stays zero when the actual descendant reproduces. At step 128 the descendant becomes the focal parent and processor state resets.

d, A separate six-parent L16 panel under six partner/reset environments. Each cell reports surviving unmutated serial lineages / 256 and thresholded retained sites / 16 positions; extinction means zero surviving controls. Site scoring uses 255 alternative values/position, 16 serial lines/alternative, and the background-corrected marker-retention criterion for at least half of identifiable alternatives. Thus panels a and d have different replicate units, mutation panels and parents; their site counts are not interchangeable. Soup samples supply partner backgrounds, not independent world replicates. Peach denotes measured zero sites, teal positive sites, grey extinction. No confidence interval is implied by a matrix cell.

### Figure 3

Figure 3 | An inherited displacement controls payload.

a, The designer supplies the abstract forward-copy geometry. b, Circles are sampled Z80 unexecuted-site scores (2,343 cells from 42 worlds); triangles are exact descendant capacities of 42 eligible evolved 6502 copiers from 11 worlds; squares are observed carried-payload-position counts in 296 qualifying constructed 6502 cases, not evolved cells or exhaustive descendant capacities. A constructed position is counted when its original byte appears at the aligned offspring position in all 64 encounters for each of three sampled payload bodies (192 encounters/case). The three ordinates are score, exact capacity and sampled carriage count, respectively; both plots use positions as units. The labelled gate-deficit circle is a sampled score zero, not a demonstrated payload-capacity zero. Symbols merge coincident coordinate pairs; multiplicities are preserved in the plotting tables. Grey diagonals show equality. The 0–10 detail preserves exact coordinates. The lower plot contains Z80 world-by-prediction mean residuals, score minus predicted free payload in positions; the single deficit cell has residual −4 and its five-cell world/prediction cluster averages −0.8. These are cluster means, not cell residuals or independent cell replicates. Z80 regrouping is exploratory: 53 Z80 cells have additional scored fetched positions, displayed separately from non-core payload; four source cells fail divisibility/range screening. The sole gate-related deficit retains its predicted payload in targeted transfers but fails the whole-tape population similarity gate. Motif/divisibility alone is not a full theorem certification. The empty 8086 slot awaits S2, the planned prospective second-stage 8086 study.

c, Registered C1 displacement edits: 101 edits from 22 archived parents, each followed in 16 serial lineages for eight generations, forward and reciprocal. Each symbol is a unique predicted/measured capacity pair; coincident edits are merged, with counts in the source table. A measured free coordinate requires every one of 255 alternative values to pass exact offspring and ancestry checks against all 17 partners. Generation-eight capacities range from zero to 60 positions. The 25,856 exact serial records are repeated observations across generations/directions, not independent edits. Exact ldir32 example (lineage 0): displacement byte 04 edited to 08. Forward generation-eight offspring: 08 5e ed b0 04 5e ed b0 08 5e ed b0 04 5e ed b0 08 5e ed b0 04 5e ed b0 08 5e ed b0 04 5e ed b0. Reciprocal generation-eight offspring: 04 5e ed b0 04 5e ed b0 04 5e ed b0 04 5e ed b0 04 5e ed b0 04 5e ed b0 04 5e ed b0 04 5e ed b0. These are actual full descendant sequences, not a conceptual byte diagram. In the separate exploratory ldir32 dial, 99/256 byte values pass the culture reproduction criterion; 50 of those 99 values reappear at the aligned position in at least half of first-generation offspring. This first-generation dial is distinct from serial marker retention.

d, Numerators count exact successes, denominators count intervention cases: one assignment to a descendant tested against zero plus 16 frozen random partners (all 17 must pass), including every final byte and its causal source. Z80 counts pool forward and reciprocal directions. Single-coordinate tests use all 255 alternatives/coordinate; joint tests use 64 sampled assignments/edit or eligible copier, not exhaustive combinations, and include zero-payload controls. Beyond-prefix successes mean erasure into protected-source descendants, not retention of the introduced alternative. Repeated identical descendant executions retain their lineage identities and are not independent trials. The constructed screen is a separate exploratory eligibility count: 5,848 cases excluded among 6,144 screened, leaving 296 qualifying geometries; those exclusions are not successes or failures in the joint table. Unsteady/nondividing constructed tests are single-coordinate controls. Existing tables do not provide controlled joint challenges of inadequate coverage or altered phase; those controls remain outstanding. The 8086 joint slot has no data and awaits S2.

### Figure 4

Figure 4 | Environments sort reproductive designs.

a, Mutation-on L32 competitions, historical and corrected Z80 core (C1) shown side by side. R denotes a regenerator and T a transmitter. Teal: benign; purple: lethal. Three starts are equal shares (10 worlds/chemistry/cohort), R introduced at 1% into T (five), and T introduced at 1% into R (five). Historical curves are medians across worlds at each archived sample; line styles distinguish starts. Transmitter fraction is T/(R+T), while core-bearing abundance includes all core carriers and is divided by all 20,000 cells. These denominators are distinct. Raw C1 trajectories are not locally archived, so C1 displays only generated endpoint medians in categorical start strips, all measured at 300,000 steps. Small horizontal offsets separate chemistries within a start category; they do not encode time. No interpolation substitutes for unavailable trajectories.

b, Each point is one world at 300,000 steps; circles/squares/triangles distinguish the three starts. Horizontal positions within a category avoid collisions and carry no quantitative meaning. Open: historical; filled: C1. Black bars to the right are medians of defined values. Each chemistry/cohort has 20 registered worlds. C1 benign has 19 defined T ratios and one world without core carriers; that ratio is undefined, omitted from its median and explicitly counted in the missing-ratio note. The same world remains in the abundance and tail plots with measured zeros. Tail diversity counts distinct transmitter tails within each world and is a nested observation, not an additional replicate.

c, Historical open grey markers and C1 smaller filled purple markers expose coincident values. Probability levels are categorical tested settings, equally spaced; slopes do not express change per unit probability. Open-first means the first detected heritable top-three tape, occupying at least 0.5% of the population, fetches partner instruction addresses in at least 8 of 16 traced encounters. Confinement means a detected heritable top-three tape at that abundance fetches no partner instruction address in all 16 traced encounters. The confinement event time is elapsed world steps from soup initialization to that first confined detection, not time since the first replicator. Open-first and transmitter-majority frequencies use all ten registered worlds per setting, including worlds without a detection in that denominator. Transmitter majority means more than half of sampled heritable cells. Confinement timing is the Kaplan–Meier median using all ten worlds, with nondetections censored at the 300,000-step horizon. The same registered rules apply to both cohorts. DZ1 requires at least 8/10 open-first worlds at probability 0, at most 2/10 at 1, and a falling rank trend with one-sided P<0.05. DZ2 requires at most one downward reversal across the six KM medians plus a positive Spearman correlation with permutation P<0.05; historical medians reverse twice and fail, C1 once and pass. DZ3 requires at most 3/10 transmitter-majority worlds at 0, at least 7/10 at 1, and a positive share trend with permutation P<0.05. Historical 6/10 at 1 fails; C1 8/10 passes. The C1 timing drop from probability .3 to 1 remains visible. These labels are registration verdicts, not additional data points. No mechanistic equilibrium curve is fitted.

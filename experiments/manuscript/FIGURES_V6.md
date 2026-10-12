# Figures v6.2: redesign and coordinator handoff

Worktree: `/Users/neo/repos/algocell-wt-figs`, branch `feat/figs-v6`, redesign starting at `2f6aa1f`. Nothing is pushed.

The authoritative deliverables are in `figures/out/v6/` relative to this manuscript directory. All main figures, individual data panels and six Extended Data builds have 180 mm PDF pages, editable-text SVGs and PDF-rendered 600 dpi PNGs (4,252 pixels wide). Main figure depths are 134 mm for Figure 1 and 154 mm for Figures 2–4. Arial body/axis type is 5.5–6.5 pt and panel letters are bold lowercase 8 pt. Letters share the same baseline within each row. There are at most four labelled panels per figure; Figure 4 has three. No conceptual illustration was invented: 1a, 1b and 3a are clean empty frames.

The redesign removes panel titles, prose, denominators, source notes and registration status words. Lines and groups use direct labels; the only main explanatory annotation is Figure 2c's two-word overwrite label. Figure 3c's arrow gives the exact byte edit. All removed explanations, counts, sources and registration context are in `out/v6/LEGENDS_V62.md`, with a bold title sentence and panel-by-panel legends; separate rendered legend PNGs are the critic's definition evidence.

Teal consistently means transmitter, carriage or benign tar; purple means lethal tar; vermilion erasure or loss; grey structure/context. Historical versus C1 uses open/filled markers or faint/solid opacity in the same hue. Figure 4c uses open grey historical and filled black C1 markers with dashed/solid lines, as directed.

## Main panel inventory

| Panel | Final content | Generated plotting tables in out/v6/data/ |
|---|---|---|
| 1a,b | Empty designer frames | No plotted data |
| 1c | Historical illustrative world with two direct labels | world_trajectory.csv |
| 1d | Preserved fractions at five distinct validation levels | validation_fractions.csv |
| 2a | 2 x 2 survival/carriage curves, copier-name headers only; three direct labels once | branching.csv |
| 2b | Four all-alternative maps; small fetched-core brackets below axes | retention_maps.csv, retention_sites.csv |
| 2c | Ordered write events with one overwrite annotation | erasure_ordered.csv, erasure_manifest.json |
| 2d | Colour-only six-parent x six-environment tile matrix, grey extinction crosses | parent_environment.csv; full numbers in ed_parent_environment.csv |
| 3a | Empty designer frame | No plotted data |
| 3b | One identity axis: visible light grey population discs with proportional area and a 1/100/2,000-cell key; diagonally dodged registered Z80 C1, evolved 6502 C2 and constructed controls | law_population.csv, law_z80_c1.csv, law_6502_c2.csv, law_6502_controls.csv |
| 3c | Exact generation-eight ldir32 lineage-0 position strips, 04 -> 60 edit | lineage_strips.csv |
| 3d | 16 × 16 byte-value map: non-working grey, working outside-condition teal outlines, 11 eligible teal fills for predicted payload 0–28; black outlines on 04 and 60 | dial_coordinates.csv |
| 4a | One T/(R+T) trajectory axis; C1 solid-opacity, historical faint; three mixture line styles | competition_medians.csv |
| 4b | Final world-level T share only; historical open/C1 filled, chemistry hues | endpoint_positions.csv, competition_endpoints.csv |
| 4c | Two categorical-probability axes: open-first and transmitter-majority counts | lethality_summary.csv |

## Extended Data inventory

| Build in out/v6/ (PDF, SVG, PNG) | Material moved from main figures |
|---|---|
| ed_v6_reference | Separate branching controls/carriage, partner-entry fraction and offspring inflow |
| ed_v6_erasure | Original-marker causal ancestry across both encounters |
| ed_v6_law | World/prediction residual, low-capacity magnification, forward/reciprocal exact capacities |
| ed_v6_joint_table | Exact all-value, joint and beyond-prefix challenge table; granular counts in data/ed_joint_challenges.csv |
| ed_v6_competition | Core abundance and tail-diversity endpoints, plus core-abundance trajectories |
| ed_v6_confinement | Kaplan-Meier confinement medians and the remaining p=.3 to 1 timing reversal |

The complete numerical Figure 2d table is `data/ed_parent_environment.csv` (36 rows); it includes all 256-control denominators, survival counts, 255-alternative/16-lineage protocol fields and retained-site counts. No matrix value was dropped. The six ED builds each have their own legend in LEGENDS_V62.md and a rendered legend PNG.

## Registered data and scientific limits

All 40 mutation-on L32 C1 E trajectories were found on Modal volume `algocell-atlas-runs`: benign in `tier2/E/`, lethal in `tier2/E-L32L/`, as specified in `modal_tier2_dispatch.py` and the frozen dispatch manifest. Only `modal volume get` was used, downloading JSONLs and completion markers. No Modal function or paid compute was invoked. Initial download requests encountered a volume-list rate limit; sequential retries completed. `data/c1_download_audit.json` records every remote path, download hash, dispatch digest, source commit, machine-manifest hash and verification result. Each log matches all condition fields, the registered completion/spec marker, core/world/WGSL hashes, endpoint step, T share, core abundance and tail count. The prior ledger did not contain raw-log content hashes; those hashes were recorded at download, rather than claimed as pre-existing. The generated source ledger binds those exact logs to the plotted trajectories. No trajectory is interpolated from an endpoint.

Figure 3c uses the actual registered `ldir32_v04` and `ldir32_v60` lineage-0 forward records at generation eight. Hexadecimal 60 is decimal 96; in the 64-byte ring it gives displacement 32. All 28 payload positions 4–31 are causally transmitted in that edited arm, versus all 28 erased in the original arm. The strips are parallel interventions, not a claim of spontaneous evolution between those observations.

Figure 3d replaces the nominal-arithmetic scatter with a 16 × 16 hexadecimal map of all 256 exploratory byte values. The 157 non-working values are very light grey; 88 working values outside the recorded theorem conditions are white with thin teal outlines. Only 11 working eligible values receive teal fill for predicted free payload, bounded by L − 4 = 28 positions. Eligibility uses the sampled C1 forward-copy/protected-prefix and coverage prerequisites as well as 4 ≤ d ≤ L and d dividing L. No outside-condition value receives a payload prediction. Black outlines mark 04 and 60. The continuous colourbar defines eligible payload, while the separate legend defines the three cell categories and the exploratory limits.

Z80 population retention scores, exact registered capacities and sampled constructed-carriage counts remain distinct despite sharing position units in Figure 3b. Light grey population discs are drawn before the other series, from largest to smallest, and their area is exactly proportional to coordinate multiplicity. The area key shows 1, 100 and 2,000 cells. Equal x/y display offsets in data coordinates separate the other series along the identity diagonal, so every exact agreement remains on y = x at both panel sizes. The population gate deficit remains at its true (4, 0) coordinate, highlighted by an unfilled grey ring without changing the count-scaled disc area; exact coordinates and counts remain in the source tables. The reserved 8086 symbol appears only in the key; no 8086 point or ecology outcome is plotted. S2 remains outstanding. The exact joint-control archive lacks controlled altered-phase/inadequate-forward-coverage challenges; those limitations remain in the legends. One C1 benign world has an undefined final T ratio; it is omitted from Figure 4b and retained with measured zeros in ED abundance/tails. No ecological replicate is replaced by a nested tail or cell count.

## QA and archived critique rounds

Every build runs `figcheck.py` before export for panels, compositions, ED figures and rendered legends. The current `figcheck_v62.json` has 32 clean reports. `verify_v62.py` additionally verifies physical PDF/SVG dimensions, 600 dpi PNGs, editable SVG text, current generated table hashes, all 40 verified C1 sources, the 64 exact lineage states and the 11/99 formal-prediction distinction.

Image-only Astra reviews use the requested CLI/model with high reasoning in an empty temporary directory, explicitly forbidding tools or repository inspection. Each round archives its exact PNGs and hashes, visible legend, prompt, clean figcheck snapshot, review, stdout/stderr and execution manifest under `out/v6/critique_v62/<figure>/roundN/`. There are no more than three v6.2 rounds per main or ED figure. The earlier v6 reviews remain unchanged under `out/v6/critique/`; they do not confer clearance on the redesign. Legacy `figcheck.json`, `qa_manifest.json` and `post_critique_fixes.json` describe that earlier build; the authoritative current audits have the `_v62` suffix.

The coordinator's two targeted Figure 3 fixes begin at `5cc3961`. Their image-only QA uses at most two new reviews, archived as Figure 3 rounds 2 and 3; the pre-fix round 1 remains unchanged. The critic prompt describes the byte map and population area key and specifically checks that the diagonal dodge keeps every exact point on the identity line while retaining the real (4, 0) gate deficit.

Figure 1's second round reviewed the completed numerical legend. Figure 4's first round found world-marker collisions at composed print size; the fix recalculated horizontal offsets for the 56 mm endpoint axis and reduced marker area to 10 pt². Raw ordinate values were preserved. The ED reference review prompted a consistent entropy axis label; the competition review prompted lower tail-axis margins and explicit line-style definitions. The confinement review exposed floating-point rounding at exactly half survival (0.5000000000000001), which had skipped the C1 p=.01 median of 42,500 steps. Exact rational survival now reproduces the registered C1 KM summary at every probability; no tolerance or event definition was changed. No clearance is claimed unless the latest review says NO SUBSTANTIVE DEFECTS and its input hashes match current images.

| Figure/build | figcheck | v6.2 rounds | Latest critic verdict | Current clearance |
|---|---|---|---|---|
| fig1 | clean | 2 | NO SUBSTANTIVE DEFECTS | Cleared |
| fig2 | clean | 1 | NO SUBSTANTIVE DEFECTS | Cleared |
| fig3 | clean | 3 | NO SUBSTANTIVE DEFECTS | Cleared |
| fig4 | clean | 2 | NO SUBSTANTIVE DEFECTS | Cleared |
| ed_v6_reference | clean | 2 | NO SUBSTANTIVE DEFECTS | Cleared |
| ed_v6_erasure | clean | 1 | NO SUBSTANTIVE DEFECTS | Cleared |
| ed_v6_law | clean | 1 | NO SUBSTANTIVE DEFECTS | Cleared |
| ed_v6_joint_table | clean | 1 | NO SUBSTANTIVE DEFECTS | Cleared |
| ed_v6_competition | clean | 2 | NO SUBSTANTIVE DEFECTS | Cleared |
| ed_v6_confinement | clean | 2 | NO SUBSTANTIVE DEFECTS | Cleared |

`out/v6/qa_manifest_v62.json` is the machine-readable source for this table. Optional polish alone does not block clearance; any substantive defect or changed image does.

## Reproduction

From the repository root, use the specified Python:

```sh
FIGS_PY=/Users/neo/repos/algocell/experiments/.venv/bin/python
"$FIGS_PY" experiments/manuscript/figures/fetch_c1_v62.py
"$FIGS_PY" experiments/manuscript/figures/prepare_v6.py --c1-runs /tmp/figs-v62-c1
"$FIGS_PY" experiments/manuscript/figures/prepare_v62.py
"$FIGS_PY" experiments/manuscript/figures/build_v62.py
"$FIGS_PY" experiments/manuscript/figures/verify_v62.py
"$FIGS_PY" experiments/manuscript/figures/write_v62_manifest.py
```

The committed erasure archive is reused; no new GPU work was necessary. No local simulation or GPU burst ran during this redesign. All plotted values read the generated CSVs, produced by committed scripts from the archived source tables. `data/sources.csv` records source hashes; `data/v62_generated_hashes.csv` records generated CSV hashes. Raw Modal downloads stay outside Git at `/tmp/figs-v62-c1`.

To run a new permitted critique before the three-round cap, use `critique_v6.py --version v62 --figure N --round R`, or `--extended reference|erasure|law|joint_table|competition|confinement --round R`. Existing archives are never overwritten. Verify and rewrite the handoff after a reviewed change. No fourth review is allowed for a figure in this redesign round.

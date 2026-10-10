# Numbers index — every quantitative statement in MAIN_nature.md and its generated source

Rule (PLAN.md): no number is typed from memory; each comes from a generated table or recorded run file. This index is
the audit trail for referees and for us. Paths are relative to `experiments/`.

| statement in the main text | value(s) | source table / file |
|---|---|---|
| soup size, pairs per step, active pairs, interactions per cell | 20,000; 8,192; ≈ 3,980; 0.395 | `PLAN.md` §System (measured 2026-10-07: 3,982 ± 35 active pairs) |
| zero bytes at the 10% occupancy crossing | 24–34% at every L (128 steps) | `results/stageB/FINDINGS.md` §zero flood; `results/stageG/NUMBERS.md` §3 (zero_emergence_mean 0.17–0.34) |
| modal first tape `01 c5` per L (Stage G) | 19/20, 12/20, 14/20, 18/20 at L = 16, 20, 50, 64 | `results/stageG/stageG/NUMBERS_G.md` ("modal first tape … in x/20 worlds") |
| KM median first replicator | 450 (16), 700 (20), 600 (50), 750 (64) | `results/stageG/NUMBERS.md` §1 |
| two-byte census: five fixed points, none ≥ 0.95, 81% max | 5; 0; 0.81 | `results/census2/NUMBERS_CENSUS2.md` (+ fixed-point section); `results/census2/heritable.csv` |
| 246 CALL-smear words pass gen2 without a 75% copy | 246 | `results/census2/NUMBERS_CENSUS2.md` |
| first replicator copies median 0.66–0.73, range 0.56–0.84; self-damage medians 0.06–0.43 | per L medians 0.68/0.66/0.68/0.73; ranges; damaged 0.33/0.43/0.11/0.06 | `results/stageG/stageG/NUMBERS_G.md` per-L first-replicator lines |
| pusher at 512/1,024 steps still 0.69 | 0.69 | `results/closure/NUMBERS_CLOSURE.md` (budget note) |
| 300k dominant: control flow 20/20 at L = 16; 17/20 identical `ad e3 …`; L = 50 20/20 `JR NZ`, 14/20 identical | 20/20; 17/20; 20/20; 14/20 | `results/stageG/stageG/NUMBERS_G.md`; `results/stageG/ZOO.md` |
| exploratory worlds: 92 gained / 3 lost control flow, McNemar P = 7 × 10⁻²⁴ | 92; 3; 7.2e-24 | `results/closure/NUMBERS_CLOSURE.md` §1 |
| successors copy 1.00 of 256 partners, 0.00 self-damage | 1.00; 0.00 | `results/stageG/stageG/NUMBERS_G.md` (final dominant lines); `results/closure/closure_tapes.csv` |
| heritable fraction at L = 16: 0.07–0.19 for 20k steps, then 0.75–0.94 | medians 0.03–0.19 ≤ 20k; 0.75 at 30k; 0.88 at 100k; 0.94 at 200k; 0.91 at 300k | `results/stageG/stageG/NUMBERS_G.md` L = 16 table (median_heritable) |
| closure by 1M: 19/20 at L = 20, 8/20 at L = 64, 12/20 still the pusher; L = 64 self-damage 0.06 | 19/20; 8/20; 12/20; 0.06 | `results/stageG/stageG/NUMBERS_G.md` G2 lines |
| G2 syntactic vs behavioural | 6/20 & 3/20 vs 19/20 & 8/20 | same |
| ablation atlas: ld-imm 2/10; stack writers ×160–×992; no-copy/rmw-only 0/40 at 1M | 2/10; 160/382/992; 0/40 | `results/stageC/stage_c/NUMBERS_C.md`; `results/stageC/stage_c/c3_ablations_st128_k4.csv` (km_ratio_vs_none, t_rep_n) |
| L = 100 budget: LDIR regime 1/10 faithful at 128 → 10/10 at 256; pusher 9/10 at 128 | 1/10; 10/10; 9/10 | `results/stageE/FINDINGS.md` E4 row; `results/stageE/stage_e/budget_L100.csv` |
| 64 steps: no exemplar ≥ 0.5%; 6/10 heritable populations without a clone (top ≤ 0.07%, 69–100% heritable) | 0/10; 6/10 | `results/stageE/FINDINGS.md` §Budget at L = 100 |
| L = 9 reversal: 12/20 vs 6/20 (128), 19/20 vs 5/20 (512), CMH P = 3.9 × 10⁻⁶ | as stated | `results/stageD/FINDINGS.md` table (stack-write-only row); `results/stageD/stage_d/cells_D.csv` |
| LDIR in top-10: 5/20 at 118,500 vs 20/20 within 3,400; destroyed 6.8 vs 1.3 per 1,000 | as stated | `results/stageD/FINDINGS.md` nascent-copier table and interaction census |
| dead zone: L = 5 alive (9/10), L = 8 (10/10), none at 3, 4, 7; 4/10, 4/10, 6/10 at 9, 10, 12; pusher heritability 0.20–0.32 | as stated | `results/stageE/FINDINGS.md` E-floor row and §Where life starts; `results/stageE/stage_e/size_arms.csv`, `unit_fitness_vs_L.csv` |
| padding switch: L = 12 at P = 28/36 → 10/10 at 350/600 vs P = 24 6/10 at 245,500 | as stated | `results/stageF/FINDINGS.md` F4a; `results/stageF/stage_f/rings.csv` |
| RET NZ closer 0.00 on other rings; relative-jump closer keeps control closed | 0.00; 0.12–0.73 (DJNZ writes) | `THEORY.md` P3 result paragraph (executor test 2026-10-08) |
| detectors: HOE AUC 0.07–0.34 at L = 25, 49, 64, 81 under none; robust detectors ≥ 0.84 except one L; occupancy 0.61–0.95 | as stated | `results/detectors/FINDINGS.md` tables |
| BFF structured search: 16,447,860 programs; only fills | 16.4 M; 130 pass; 8 'heritable' fills | `results/bff_search/NUMBERS_SEARCH.md` |
| BFF std: 9/24; 9/9 loop; 0/9 open; 7/9 closed | as stated | `results/bff/NUMBERS_BFF.md` std section & transition table |
| BFF wrap: 19/24 (Fisher P = 0.008); 19/19 loop; 0/19 open; 16/19 closed | as stated | `results/bff/NUMBERS_BFF.md` wrap section & tests |
| BFF wraplit: 12/12 all-P first at epoch 64 with 53–60% share; open 100%; collapse in 12/12 by ≈ 256; 0.2% minority; 0/12 closed | as stated | `results/bff/NUMBERS_BFF.md` wraplit section; `results/bff/runs.csv`; `results/bff/FINDINGS.md` §4 |
| closed-replicator search: 1,309,528 tilings, 0 closed heritable | 1.3 M; 0 | `results/bff_closed_search/NUMBERS_CLOSED_SEARCH_p10.md` |
| Theorem 1 bound: 64 writes + 63 moves + 2 changes > 128 | arithmetic | `THEOREMS.md` Theorem 1 |
| compute: 3,560 Z80 runs; 84 BFF soups; ≈ $400 | counts from run summaries; cost from preflight logs and recorded run times | `PLAN.md` change log; `runs/*/` summaries (A 630, B 640, C 490, D 280, E 1,230, F 210, G 80) |
| BFF wraplitnh: 12/12 all-P first (epoch 64); persists 12/12 (final heritable median 1.00, collapsed 0/12) vs 0/12 in wraplit; 0/12 closed; HOE ≥ 1 at epoch 0 in 12/12, ≤ 0.03 from epoch 64 | as stated | `results/bff/NUMBERS_BFF.md` wraplitnh section & readings (f1–f3); `results/bff/runs.csv`; `results/bff/FINDINGS.md` §4b |
| BFF lit: 12/12 all-P first (epoch 64, copies median 0.91); heritable ≥ 0.5 in 5/12; collapse by the letter 5/12; final heritable 0.00 in 12/12; 0/12 closed | as stated | `results/bff/NUMBERS_BFF.md` stdlit section & readings (l1–l2); `results/bff/runs.csv`; `results/bff/FINDINGS.md` §4c |
| closed-design search behind "none evolves": 1.3 M periodic programs ≤ period 10, 0 closed heritable | 1,309,528; 0 | `results/bff_closed_search/NUMBERS_CLOSED_SEARCH_p10.md` |

| Fig. 2d partner-test medians (pusher L = 16: copies 0.68, damaged 0.33; closers 1.00 / 0.00) | computed at build time | `manuscript/figures/concept.py` from `results/stageG/stageG/stage_g_runs.csv` |
| Fig. 2d cycles and Fig. 5a trajectories (pusher leaves after 10 instructions with 10 of 20 bytes written; RET NZ returns to cells 0 and 3; JR NZ 23→9, 9→31, wrap 35→0; DJNZ 16→7, wrap 15→0; LDIR in place) | generated | `trace_z80.py` → `results/concept/traces.json` |
| pusher literal byte order: bytes `01 c5 01` = `LD BC,$01c5`; first push writes `c5 01` at cells 29–30 of the 32-byte ring | GPU executor | `trace_z80.py` (sp 31 → 29 after the first push) |
| information inflow H(o \| x), Z80: first replicators median 7.0 bits (3.9, 6.2, 7.8, 8.0 by L); loop-bearing finals 0 bits in 63/67; drop ≥ 2 bits in 63/80 | 7.0; 3.9/6.2/7.8/8.0; 63/67; 63/80 | `results/biology/individuality/summary.csv`, `predictions.csv`, `NUMBERS_INDIVIDUALITY.md` |
| information inflow, BFF first replicators: published < 0.5 bit in 6/9; lit 8.0 bits; wraplit 0.75; wraplitnh 0.04 | 6/9; 7.99; 0.75; 0.037 | `results/biology/individuality/summary.csv` |
| assembly index of the first replicator equals the minimum for its length and the tar's (79/80 equal); assembly measure fires before t_rep in 71/80 (median lead 19×); detector AUC 0.72 vs HOE 0.75 at L ≥ 25 | 4, 5, 7, 6; 79/80; 71/80; 19×; 0.721/0.749 | `results/biology/assembly/per_world.csv`, `auc.csv`, `NUMBERS_ASSEMBLY.md` |
| Stage H well-mixed control: open-phase heritability 0.125 vs 0.125; closure median 50,000 vs 30,000, 7/10 vs 20/20; t_rep 125 vs 525; kin encounters 11% vs 57%; damage vs kin 0–2.5%, vs strangers 31–34% | see statement | `results/stageH/NUMBERS_H.md`, `predictions.csv`, `per_world.csv`, `KIN_CENSUS.md` |
| Stage I lethal tar: no pusher establishes (3 exemplar sightings in 6,980 samples); first heritable tape a block copier in 10/10, 1.00 copied / 0.00 damaged, no zero byte; t_rep median 46,250 (20,000–219,000) vs 525; zero fraction 0.18 at step 500 | see statement | `results/stageI/NUMBERS_I.md`, `predictions.csv`, `per_world.csv`, `stageI/stage_g_runs.csv` |

Figure data sources: Fig. 1c `runs/stageG/none@closure_L16_st128_k4_s2001.jsonl` + `results/stageG/c4/functional.csv`; Fig. 1d `results/stageE/assays.csv`
(`none@nominal`); Fig. 2a,b,e `results/stageG/stageG/stage_g_runs.csv`; Fig. 2c `results/stageG/c4/functional.csv`; Fig. 3a
`results/stageC/stage_c/c3_ablations_st128_k4.csv`; Fig. 3b `results/stageE/stage_e/size_arms.csv` + `unit_fitness_vs_L.csv`; Fig. 3c
`results/stageF/stage_f/rings.csv`; Fig. 3d `results/stageD/stage_d/cells_D.csv`; Fig. 4a–c `runs/bff_modal/bff/*/samples.jsonl`, `epochs.csv`,
`results/bff/runs.csv`; Fig. 5c `stage_g_runs.csv` + `results/bff/runs.csv`. Builder: `manuscript/figures/make_figures.py`.

Figure renumbering (2026-10-08, late night): old Fig. 1c → Fig. 1b (old 1b, the seed-2001 world, dropped); new Fig. 2 =
seven lattice frames of `runs/video/video_L16_st128_k4_s2002` (zero peak 0.373 at step 320; tq_10 840; at step 100,000:
zero fraction 0.004, dominant-tape occupancy 0.894, 2,290 distinct tapes; frame-3 window rows 72–88, cols 30–54; top
classes `21 e5` 984 tapes, `01 c5` 568); old Fig. 2a,b,c,d,e → Fig. 3a,b,d,f,e; old Fig. 5b → Fig. 3c; old Fig. 3 → Fig. 4;
old Fig. 4 → Fig. 5; old Fig. 5a → Fig. 6. ED 12 and ED 13 are now drawn by `make_figures.ed12`/`ed13` from
`results/biology/assembly/{per_sample,per_world,auc}.csv` and `results/stageI/{c4/functional.csv,stageI/stage_g_runs.csv}`
with the Stage G L = 16 tables as the benign control (medians 46,250 vs 525).

## Revision 3 (2026-10-10): numbers added to the main text and Methods

| statement | value(s) | source table / file |
|---|---|---|
| genealogy worlds recorded / closed | 20 / 17 | `results/lod/FOUNDERS2.md` (first founders: 17 in 17 worlds); `REVISION_PREREG.md` N3 outcome (lod_v6) |
| replay validation | 795,934 encounters, 0 mismatches; 64 of 64 sampled confined copiers traced to the seeded closer | `runs/lod/validate/validate.json`; `results/night/NIGHT_RESULTS_2026-10-09.md` |
| base rate (open copier among ancestors within 2,000 steps) | median 1.0 | `REVISION_PREREG.md`, outcome of N3 under the registered rule |
| first founder completed by rewrite / copy of partner / point mutation ("recombination" withdrawn: residual class) | 16 / 1 / 0 | `results/lod/FOUNDERS2.md` ("completed by: novel 16, copyA 1") |
| founding executor a non-copier; side partner / own half | 17 of 17; 13 / 4 | `results/lod/FOUNDING.md` |
| bytes of the founder in neither tape before | median 7 (1–13) | `results/lod/FOUNDING.md` |
| routes: open writer in that encounter / confined producer earlier / open producer | 11 / 5 / 1; producers 1–3 steps before F, by point mutation 3 (2 confined) and rewrite 3 | `results/lod/FOUNDING.md`, `founding.csv` |
| in-encounter writers produce F with random partners | at most 0.19 | `results/lod/founding.csv` (executor_writes_F) |
| steps since the newest open copier (median, range); records | 79 (1–360); 14 | `results/lod/FOUNDERS2.md`; `results/lod/founders2.csv` (steps_elapsed) |
| only non-copiers between | 16 of 17 | `results/lod/FOUNDERS2.md` |
| contributing events | median 3 (1–6) | `results/lod/FOUNDERS2.md` |
| executed words carried before the founder | median 0.205% of cells; baseline ≈ 0.024%; ratio ≈ 8.5 | `results/lod/PARTS_FIRST.md` |
| first founders with the return-closer motif | 15 of 17 (1 LDIR block copier, 1 other) | `results/lod/PARTS_FIRST.md` |
| later founders; with bytes from a confined copier | 15 in 10 worlds; 11 of 15 | `results/lod/FOUNDERS2.md` |
| hypercubes: no heritable single-byte path | 3 of 4 (one six-step path); k = 12 or 16 | `results/lod/LANDSCAPE.md` (section "With heritability") |
| R → T and T → R copies balance | within 0.1% (largest difference 0.0007 of R–T encounters) | `results/n2/MUTUAL.md` |
| core-free cells in the starting soups | median 13% (5–24%) | `results/n2/COMPOSITION_START.md` (the night note's "11–16%" had no generated source and was wrong) |
| regenerator destroyed if the intruder entered | 0.75–0.89 (0.7467–0.8889) | `results/n2/SCARS.md` (rows host R) |
| backup cores: regenerator with a damaged first core copies itself | 0 of 4,096 | `REVISION_PREREG.md`, outcome of N2 (N2-2) |
| zeros removed: protection kept | 88–94% | `REVISION_PREREG.md`, outcome of N4; `results/n2/SCARS.md` |
| payload zeros as scars (benign vs lethal frequency) | 0.067 vs 0.052 at L = 32 | `results/toxin/COMPOSITION.md` |
| BFF open literal: offspring carry 8 bits of the partner without wrap | 8 bits (ceiling) | Supplementary Fig. 3 (old ED Fig. 7); `results/biology/individuality/NUMBERS_INDIVIDUALITY.md` |
| `21 e3` / `21 e0` carried before first founders (where executed) | median 0.24% (0.11–0.82%) / 2.4% (0.83–4.6%) | `results/lod/PARTS_FIRST.md` |
| confined copiers already present before the founder | 2 of 17 worlds, at most 0.05% of cells | `results/lod/PARTS_FIRST.md`; `founders2.csv` (soup_confined) |
| t_rep median with well-mixed pairing | 125 | `results/stageH/NUMBERS_H.md` |

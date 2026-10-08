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

Figure data sources: Fig. 1c `runs/stageG/none@closure_L16_st128_k4_s2001.jsonl` + `results/stageG/c4/functional.csv`; Fig. 1d `results/stageE/assays.csv`
(`none@nominal`); Fig. 2a,b,e `results/stageG/stageG/stage_g_runs.csv`; Fig. 2c `results/stageG/c4/functional.csv`; Fig. 3a
`results/stageC/stage_c/c3_ablations_st128_k4.csv`; Fig. 3b `results/stageE/stage_e/size_arms.csv` + `unit_fitness_vs_L.csv`; Fig. 3c
`results/stageF/stage_f/rings.csv`; Fig. 3d `results/stageD/stage_d/cells_D.csv`; Fig. 4a–c `runs/bff_modal/bff/*/samples.jsonl`, `epochs.csv`,
`results/bff/runs.csv`; Fig. 5c `stage_g_runs.csv` + `results/bff/runs.csv`. Builder: `manuscript/figures/make_figures.py`.

# Figures and Extended Data v8

Technical verification passed for all 14 figures. Astra cleared 5/14 delivered images. Clearance requires a no-substantive-defects verdict and an exact reviewed/delivered PNG hash match.

| Figure | Page (mm) | figcheck | Review rounds | Final verdict | Clearance |
|---|---|---|---|---|---|
| [fig1](figures/out/v8/fig1.png) | 180 × 164 | clean | 3 | Substantive defects | [fixed after final review, not re-reviewed](figures/out/v8/critique_fresh/fig1/round3/critique.md) |
| [fig2](figures/out/v8/fig2.png) | 180 × 145 | clean | 3 | Substantive defects | [fixed after final review, not re-reviewed](figures/out/v8/critique_fresh/fig2/round3/critique.md) |
| [fig3](figures/out/v8/fig3.png) | 180 × 166 | clean | 2 | No substantive defects | [CLEARED](figures/out/v8/critique_fresh/fig3/round2/critique.md) |
| [fig4](figures/out/v8/fig4.png) | 180 × 180 | clean | 3 | Substantive defects | [NOT CLEARED](figures/out/v8/critique_fresh/fig4/round3/critique.md) |
| [fig5](figures/out/v8/fig5.png) | 180 × 174 | clean | 3 | No substantive defects | [CLEARED](figures/out/v8/critique_fresh/fig5/round3/critique.md) |
| [ed8_1](figures/out/v8/ed8_1.png) | 180 × 216 | clean | 5 | No substantive defects | [CLEARED](figures/out/v8/critique_review/ed8_1/round5/critique.md) |
| [ed8_2](figures/out/v8/ed8_2.png) | 180 × 285 | clean | 6 | Substantive defects | [fixed after final review, not re-reviewed](figures/out/v8/critique_review/ed8_2/round6/critique.md) |
| [ed8_3](figures/out/v8/ed8_3.png) | 180 × 211 | clean | 5 | Substantive defects | [fixed after final review, not re-reviewed](figures/out/v8/critique_review/ed8_3/round5/critique.md) |
| [ed8_4](figures/out/v8/ed8_4.png) | 180 × 260 | clean | 5 | Substantive defects | [fixed after final review, not re-reviewed](figures/out/v8/critique_review/ed8_4/round5/critique.md) |
| [ed8_5](figures/out/v8/ed8_5.png) | 180 × 324 | clean | 5 | Substantive defects | [fixed after final review, not re-reviewed](figures/out/v8/critique_review/ed8_5/round5/critique.md) |
| [ed8_6](figures/out/v8/ed8_6.png) | 180 × 224 | clean | 4 | Substantive defects | [fixed after final review, not re-reviewed](figures/out/v8/critique_review/ed8_6/round4/critique.md) |
| [ed8_7](figures/out/v8/ed8_7.png) | 180 × 228 | clean | 7 | No substantive defects | [CLEARED](figures/out/v8/critique_review/ed8_7/round7/critique.md) |
| [ed8_8](figures/out/v8/ed8_8.png) | 180 × 190 | clean | 7 | No substantive defects | [CLEARED](figures/out/v8/critique_review/ed8_8/round7/critique.md) |
| [ed8_9](figures/out/v8/ed8_9.png) | 180 × 231 | clean | 7 | Substantive defects | [fixed after final review, not re-reviewed](figures/out/v8/critique_review/ed8_9/round7/critique.md) |

PDF and SVG retain editable chart text; PNGs are rendered from PDF at 600 dpi. Figure 1 links its raster render to relative companion PNG files. No delivered SVG contains embedded raster data. All 14 PDF exports have text at least 6 pt; nominal labels are 6.5 pt or larger at 180 mm width. Extended Data heights follow the rendered content bounds. These technical checks do not override the image critic.

All nine Extended Data numbers, PDF targets and titles match the SI_v8.md Extended Data display map; no title mismatch remains. Legends, source paths and unavailable measurements are in [ED_V8.md](figures/out/v8/ED_V8.md). Pointer randomization is unmeasured and was not an S2 arm.

Final label-matched publication legends are exported by applying the exact LEGEND_UPDATES.md replacements to MAIN_v8.md in memory. MAIN_v8.md remains unedited. Every main and Extended Data legend is at most 250 words. Registered 8086 law tests remain untested; post hoc results appear only in the explicitly exploratory ED9 panels.

The prior critique archive is retained. This final bounded revision adds exactly round 3 for main Figures 1, 2, 4 and 5, and one round per Extended Data display (ED6 required a text-size repair). Figure 3 retains its prior clearance. No main round 4 was run. Both original references are attached to every review. Reviews are retained unedited. Each row reports the unedited final archived verdict and whether its reviewed PNG is identical to the delivered PNG. Post-final-review label fixes and ED2/ED3 export clipping corrections are recorded in FINAL_REVIEW_FIXES.json and are not re-reviewed. No cleared image is revised after its clearing review.

**Reproduce:** Run `prepare_v8.py`, `build_v8.py`, `build_ed_v8.py`, `legends_v8.py`, then `verify_v8.py` with `/Users/neo/repos/algocell/experiments/.venv/bin/python`. Archived final critique: `critique_v8.py --figure N --round 3` (N = 1, 2, 4, 5); Extended Data final rounds 4–7. Do not start another review round.

[Fresh review record](figures/out/v8/FRESH_REVIEW.md) · [Legend updates](figures/out/v8/LEGEND_UPDATES.md) · [Publication legends](figures/out/v8/LEGENDS_V8.md) · [QA manifest](figures/out/v8/qa_manifest.json) · [ED sources](figures/out/v8/ed_sources.json) · [ED corrections](figures/out/v8/ED_V8_REVIEW.md) · [Frozen main data](figures/out/v8/data/)

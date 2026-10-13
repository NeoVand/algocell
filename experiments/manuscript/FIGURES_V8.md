# Figures and Extended Data v8

Technical verification passed for all 14 figures. Astra cleared 11/14 delivered images. Clearance requires a no-substantive-defects verdict and an exact reviewed/delivered PNG hash match.

| Figure | Page (mm) | figcheck | Review rounds | Final verdict | Clearance |
|---|---|---|---|---|---|
| [fig1](figures/out/v8/fig1.png) | 180 × 146 | clean | 1 | No substantive defects | [CLEARED](figures/out/v8/critique_review/fig1/round1/critique.md) |
| [fig2](figures/out/v8/fig2.png) | 180 × 136 | clean | 2 | No substantive defects | [CLEARED](figures/out/v8/critique_review/fig2/round2/critique.md) |
| [fig3](figures/out/v8/fig3.png) | 180 × 166 | clean | 1 | No substantive defects | [CLEARED](figures/out/v8/critique_review/fig3/round1/critique.md) |
| [fig4](figures/out/v8/fig4.png) | 180 × 163 | clean | 1 | No substantive defects | [CLEARED](figures/out/v8/critique_review/fig4/round1/critique.md) |
| [fig5](figures/out/v8/fig5.png) | 180 × 162 | clean | 2 | No substantive defects | [CLEARED](figures/out/v8/critique_review/fig5/round2/critique.md) |
| [ed8_1](figures/out/v8/ed8_1.png) | 180 × 180 | clean | 3 | No substantive defects | [CLEARED](figures/out/v8/critique_review/ed8_1/round3/critique.md) |
| [ed8_2](figures/out/v8/ed8_2.png) | 180 × 220 | clean | 4 | No substantive defects | [CLEARED](figures/out/v8/critique_review/ed8_2/round4/critique.md) |
| [ed8_3](figures/out/v8/ed8_3.png) | 180 × 180 | clean | 3 | No substantive defects | [CLEARED](figures/out/v8/critique_review/ed8_3/round3/critique.md) |
| [ed8_4](figures/out/v8/ed8_4.png) | 180 × 220 | clean | 3 | No substantive defects | [CLEARED](figures/out/v8/critique_review/ed8_4/round3/critique.md) |
| [ed8_5](figures/out/v8/ed8_5.png) | 180 × 290 | clean | 3 | No substantive defects | [CLEARED](figures/out/v8/critique_review/ed8_5/round3/critique.md) |
| [ed8_6](figures/out/v8/ed8_6.png) | 180 × 220 | clean | 3 | No substantive defects | [CLEARED](figures/out/v8/critique_review/ed8_6/round3/critique.md) |
| [ed8_7](figures/out/v8/ed8_7.png) | 180 × 180 | clean | 4 | Substantive defects | [REVISED AFTER REVIEW; NOT CLEARED](figures/out/v8/critique_review/ed8_7/round4/critique.md) |
| [ed8_8](figures/out/v8/ed8_8.png) | 180 × 180 | clean | 4 | Substantive defects | [REVISED AFTER REVIEW; NOT CLEARED](figures/out/v8/critique_review/ed8_8/round4/critique.md) |
| [ed8_9](figures/out/v8/ed8_9.png) | 180 × 180 | clean | 4 | Substantive defects | [REVISED AFTER REVIEW; NOT CLEARED](figures/out/v8/critique_review/ed8_9/round4/critique.md) |

PDF and SVG retain editable chart text; PNGs are rendered from PDF at 600 dpi. Figure 1 links its raster render to relative companion PNG files. No delivered SVG contains embedded raster data.

All nine Extended Data numbers, PDF targets and titles match the SI_v8.md Extended Data display map; no title mismatch remains. Legends, source paths and unavailable measurements are in [ED_V8.md](figures/out/v8/ED_V8.md). Pointer randomization is unmeasured and was not an S2 arm.

The canonical publication legends match MAIN_v8.md; every main and Extended Data legend is at most 250 words. Registered 8086 law tests remain untested; post hoc results appear only in the explicitly exploratory ED9 panels.

The prior critique archive is retained. This revision uses `critique_review/`, with at most two image-only gpt-6-astra/high rounds for main figures and four for Extended Data. Both original references are attached to every review. Reviews are retained unedited. Each row reports the final archived verdict and whether its reviewed PNG is identical to the delivered PNG. No cleared image is revised after its clearing review.

**Reproduce:** Run `prepare_v8.py`, `build_v8.py`, `build_ed_v8.py`, `legends_v8.py`, then `verify_v8.py` with `/Users/neo/repos/algocell/experiments/.venv/bin/python`. Critique: `critique_v8.py --figure N --round R` (N = 1–5, R = 1–2; or ed8_1–ed8_9, R = 1–4).

[Publication legends](figures/out/v8/LEGENDS_V8.md) · [QA manifest](figures/out/v8/qa_manifest.json) · [ED sources](figures/out/v8/ed_sources.json) · [ED corrections](figures/out/v8/ED_V8_REVIEW.md) · [Frozen main data](figures/out/v8/data/)

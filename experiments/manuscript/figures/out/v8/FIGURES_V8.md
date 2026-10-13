# Figures and Extended Data v8

Technical verification passed for all 14 figures. Astra cleared 2/14 delivered images. Clearance requires a no-substantive-defects verdict and an exact reviewed/delivered PNG hash match.

| Figure | Page (mm) | figcheck | Review rounds | Final verdict | Clearance |
|---|---|---|---|---|---|
| [fig1](fig1.png) | 180 × 156 | clean | 2 | Substantive defects | [NOT CLEARED](critique_fresh/fig1/round2/critique.md) |
| [fig2](fig2.png) | 180 × 145 | clean | 2 | Substantive defects | [REVISED AFTER REVIEW; NOT CLEARED](critique_fresh/fig2/round2/critique.md) |
| [fig3](fig3.png) | 180 × 166 | clean | 2 | No substantive defects | [CLEARED](critique_fresh/fig3/round2/critique.md) |
| [fig4](fig4.png) | 180 × 180 | clean | 2 | Substantive defects | [REVISED AFTER REVIEW; NOT CLEARED](critique_fresh/fig4/round2/critique.md) |
| [fig5](fig5.png) | 180 × 174 | clean | 2 | Substantive defects | [REVISED AFTER REVIEW; NOT CLEARED](critique_fresh/fig5/round2/critique.md) |
| [ed8_1](ed8_1.png) | 180 × 180 | clean | 4 | Substantive defects | [NOT CLEARED](critique_review/ed8_1/round4/critique.md) |
| [ed8_2](ed8_2.png) | 180 × 220 | clean | 5 | Substantive defects | [NOT CLEARED](critique_review/ed8_2/round5/critique.md) |
| [ed8_3](ed8_3.png) | 180 × 180 | clean | 4 | Substantive defects | [NOT CLEARED](critique_review/ed8_3/round4/critique.md) |
| [ed8_4](ed8_4.png) | 180 × 220 | clean | 4 | Substantive defects | [NOT CLEARED](critique_review/ed8_4/round4/critique.md) |
| [ed8_5](ed8_5.png) | 180 × 290 | clean | 4 | Substantive defects | [NOT CLEARED](critique_review/ed8_5/round4/critique.md) |
| [ed8_6](ed8_6.png) | 180 × 220 | clean | 3 | No substantive defects | [CLEARED](critique_review/ed8_6/round3/critique.md) |
| [ed8_7](ed8_7.png) | 180 × 220 | clean | 6 | Substantive defects | [NOT CLEARED](critique_review/ed8_7/round6/critique.md) |
| [ed8_8](ed8_8.png) | 180 × 220 | clean | 6 | Substantive defects | [NOT CLEARED](critique_review/ed8_8/round6/critique.md) |
| [ed8_9](ed8_9.png) | 180 × 220 | clean | 6 | Substantive defects | [NOT CLEARED](critique_review/ed8_9/round6/critique.md) |

PDF and SVG retain editable chart text; PNGs are rendered from PDF at 600 dpi. Figure 1 links its raster render to relative companion PNG files. No delivered SVG contains embedded raster data. Main-figure PDF text is at least 6 pt, with nominal substantive labels of 6.5 pt or larger at 180 mm width; this technical check does not override the image critic.

All nine Extended Data numbers, PDF targets and titles match the SI_v8.md Extended Data display map; no title mismatch remains. Legends, source paths and unavailable measurements are in [ED_V8.md](ED_V8.md). Pointer randomization is unmeasured and was not an S2 arm.

The canonical publication legends are exported from MAIN_v8.md; proposed label-matching edits are in LEGEND_UPDATES.md for the manuscript editor. Every main and Extended Data legend is at most 250 words. Registered 8086 law tests remain untested; post hoc results appear only in the explicitly exploratory ED9 panels.

The prior critique archive is retained. This revision uses `critique_fresh/` for at most two new image-only gpt-6-astra/high rounds per main figure. Extended Data reviews remain in `critique_review/`, continuing through at most round 6 for confirmation and terminology fixes. Both original references are attached to every review. Reviews are retained unedited. Each row reports the final archived verdict and whether its reviewed PNG is identical to the delivered PNG. No cleared image is revised after its clearing review.

**Reproduce:** Run `prepare_v8.py`, `build_v8.py`, `build_ed_v8.py`, `legends_v8.py`, then `verify_v8.py` with `/Users/neo/repos/algocell/experiments/.venv/bin/python`. Critique: `critique_v8.py --figure N --round R` (N = 1–5, R = 1–2; or ed8_1–ed8_9, R = 1–6).

[Fresh review record](FRESH_REVIEW.md) · [Legend updates](LEGEND_UPDATES.md) · [Publication legends](LEGENDS_V8.md) · [QA manifest](qa_manifest.json) · [ED sources](ed_sources.json) · [ED corrections](ED_V8_REVIEW.md) · [Frozen main data](data/)

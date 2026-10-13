# Final bounded figure review

Main Figures 1, 2, 4 and 5 each received exactly one final image-only gpt-6-astra/high review (round 3), with both reference PNGs. Figure 3 retained its round-2 clearance. Every Extended Data display received one final review; ED6 required repair because its previous minimum was 5.1 pt. No main round 4 was run. Reviews and verdicts remain unedited.

All 14 exports pass figcheck and verify_v8.py, retain editable vector text, render at 600 dpi / 180 mm width, and contain no embedded SVG raster data. Delivered minimum text is 6.498 pt for Figure 1 and 6.5 pt for every other display. The technical minimum does not override the visual reviewer.

| Figure | Final review | Final verdict | Cleared | Minimum text (pt) |
|---|---|---|---|---|
| [fig1](fig1.png) | [round3](critique_fresh/fig1/round3/critique.md) | Substantive defects | fixed-after-final-review | 6.498 |
| [fig2](fig2.png) | [round3](critique_fresh/fig2/round3/critique.md) | Substantive defects | fixed-after-final-review | 6.500 |
| [fig3](fig3.png) | [round2](critique_fresh/fig3/round2/critique.md) | No substantive defects | yes | 6.500 |
| [fig4](fig4.png) | [round3](critique_fresh/fig4/round3/critique.md) | Substantive defects | no | 6.500 |
| [fig5](fig5.png) | [round3](critique_fresh/fig5/round3/critique.md) | No substantive defects | yes | 6.500 |
| [ed8_1](ed8_1.png) | [round5](critique_review/ed8_1/round5/critique.md) | No substantive defects | yes | 6.500 |
| [ed8_2](ed8_2.png) | [round6](critique_review/ed8_2/round6/critique.md) | Substantive defects | fixed-after-final-review | 6.500 |
| [ed8_3](ed8_3.png) | [round5](critique_review/ed8_3/round5/critique.md) | Substantive defects | fixed-after-final-review | 6.500 |
| [ed8_4](ed8_4.png) | [round5](critique_review/ed8_4/round5/critique.md) | Substantive defects | fixed-after-final-review | 6.500 |
| [ed8_5](ed8_5.png) | [round5](critique_review/ed8_5/round5/critique.md) | Substantive defects | fixed-after-final-review | 6.500 |
| [ed8_6](ed8_6.png) | [round4](critique_review/ed8_6/round4/critique.md) | Substantive defects | fixed-after-final-review | 6.500 |
| [ed8_7](ed8_7.png) | [round7](critique_review/ed8_7/round7/critique.md) | No substantive defects | yes | 6.500 |
| [ed8_8](ed8_8.png) | [round7](critique_review/ed8_8/round7/critique.md) | No substantive defects | yes | 6.500 |
| [ed8_9](ed8_9.png) | [round7](critique_review/ed8_9/round7/critique.md) | Substantive defects | fixed-after-final-review | 6.500 |

“fixed-after-final-review” means changes were verified technically and visually but not re-reviewed; it is not a replacement passing critic verdict. Strict image-review clearance remains limited to Figure 3, Figure 5, ED1, ED7 and ED8.

## Post-final-review changes

The exact changes are recorded in [FINAL_REVIEW_FIXES.json](FINAL_REVIEW_FIXES.json). Figure 1, Figure 2 and ED2–ED6/ED9 received wording or label repairs. ED2 bars/maps and ED3 swatches also needed an export correction: already-plotted artists extended below their original off-axis layout boxes and were clipped there. Their complete recorded content is now visible; no input observations, counts or data coordinates changed. No further review was run.

## Findings left unresolved within the bounded round

- Figures 1 and 2, ED3 and ED4: the final critic retains visual print-readability concerns. The exports meet the measured 6.5 pt nominal floor (Figure 1’s vector overlay measures 6.498 pt). Label findings were repaired, but no additional layout review or redesign was authorized after the final round.
- Figure 4: converging trajectory encodings and crowded table rows remain visually below standard. These require visual/layout changes beyond pure wording or label fixes.
- ED5: faint individual-world trajectories and visual hierarchy remain reviewer concerns. World-count, grouped-header and host-byte definitions were repaired after review; further styling was outside the final-review wording allowance.
- ED9: the reviewer requests separate encounter counts for nonempty perturbations and empty-assignment controls, plus further visual readability work. The existing table explicitly retains combined totals; splitting them would add a numerical breakdown beyond a pure wording fix. The comparison target and pooled partner-condition labels were repaired.
- ED2 and ED6: all supported final-review content/definition findings were repaired, but they remain unreviewed revisions and are not marked cleared.

## Legends and scientific scope

MAIN_v8.md remains unedited. [LEGEND_UPDATES.md](LEGEND_UPDATES.md) lists every exact old → exact new sentence needed for the final main figures. Applying the list in memory reproduces the delivered LEGENDS_V8.md; Figure 1–5 legend word counts are 233, 244, 250, 245 and 245. Every ED legend is ≤250 words and matches its panels. Frozen source hashes, numerical coordinates, SI display titles and evidence qualifications remain verified. No simulations, GPU compute or push were performed.

## Delivered and final-reviewed PNG hashes

| Figure | Delivered SHA-256 | Reviewed SHA-256 |
|---|---|---|
| fig1 | `f48d0b6c6ed2c5c360a831c5573dcfc1594c949e17b1cf583a133cc1722613e6` | `8d0ab3a0e9693f671b1719fee4ad8b294a46d50a3f1df85868bb486c256aa519` |
| fig2 | `15b9b99c74aa77b7898b902988ff72c7fef1c955b11e2e1f1b46cb9040b46637` | `17f6200dacc4af30621597845535ac3795524ca402ac4c1b5ad852cd1323cb79` |
| fig3 | `a64b422af7843dd586ea4a45428ead2e5c786febd6166fc0fd724fbca71d8d1d` | `a64b422af7843dd586ea4a45428ead2e5c786febd6166fc0fd724fbca71d8d1d` |
| fig4 | `ba3df4b892b459909f12508ffd0d76018dc0fba778803e930d94fb81eae06aca` | `ba3df4b892b459909f12508ffd0d76018dc0fba778803e930d94fb81eae06aca` |
| fig5 | `4b7ef068f8f1ea49df0c1739e4e027cf5c6c0d8b12d2ddf5c70d36b98217d1a4` | `4b7ef068f8f1ea49df0c1739e4e027cf5c6c0d8b12d2ddf5c70d36b98217d1a4` |
| ed8_1 | `98245b5ab4fb18f6f3fe92bd7c7c8b8430d125a304ffe244f7d8bbb71463891d` | `98245b5ab4fb18f6f3fe92bd7c7c8b8430d125a304ffe244f7d8bbb71463891d` |
| ed8_2 | `b034b00210e9e08776e1ac3c7dd50cc12c43f4946d0fd1e1d6f087c939f9f79b` | `90d18679602b1b9d4a9ebab4f8becedbfcb7fb76d2c02b5e3991eb7de232322c` |
| ed8_3 | `d95a66ec7df7c8e11744ba0ac7d994b3ff091bd833f1ada61d7c61e9264ee11b` | `6298b01280ad9c5bb034e36214c0d527e4cb5aae0ff74dca12424b945a3ab522` |
| ed8_4 | `23f9619ed8fd774787e0525f68f8509857ab7ecff91b362ed47d89342bd63379` | `73a34c37617fcc4cb1a74c81a1593e7817dc6c198669a5f0d55806d89a98b495` |
| ed8_5 | `0eece906bd80745d5b631a42c5ad769b7a26bd2a754b7f83550513d4eade562f` | `a1cde3aabf589a24935a3fc027f24d4dd17e05090665629d5204133a5121584a` |
| ed8_6 | `31cd819a43121e76b949ae60c7016102836962c92418caa53bfc55ebe69a2f30` | `fc10102a565e53948bdc6d363cc24a3ee9050db8cb0b1261b6749f14d16fafaf` |
| ed8_7 | `3fe9a374c0c68dafb6cd3fca79732ac1decfad90d43f32dcf834fc08f74fd64e` | `3fe9a374c0c68dafb6cd3fca79732ac1decfad90d43f32dcf834fc08f74fd64e` |
| ed8_8 | `7c540e6184bebc289f4649fc285689fe6a443e6fefce735d333af8f145e7a258` | `7c540e6184bebc289f4649fc285689fe6a443e6fefce735d333af8f145e7a258` |
| ed8_9 | `0e6043ed1dd3ea7b0a26854a5b51c5f250640b4dcc6c4e8c0d973398d680cf9e` | `a0b9a32f11465ea48633723b49b4238b63784796633033756298867bc3466e06` |

[QA manifest](qa_manifest.json) · [Figure package](FIGURES_V8.md) · [ED legends and sources](ED_V8.md)

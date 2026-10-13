# Algocell restored-base review

Local branch: `feat/app-legacy-base`, created from `811925e`. Baseline: detached `main` at `4c84fec`. No push or main-ref change was made. Zilion is pinned to 0.3.1; no runtime dependency was added.

## Performance gate

Apple M4, Chromium WebGPU/Metal 3. Same engine defaults on both builds: 200×200 square cells, L16/P32, 8192 pair slots, 128 steps, seed 6, mutation exponent 4, zero suppression and benign zero. The UI's viewport-dependent fit can use approximately 20,000 cells; this gate uses the 200×200 `DEFAULT_GRID_CONFIG` explicitly.

Both engines were compiled as minified production Vite bundles from their respective checkouts. One completed warmup, then reseed. Each timed burst ran for 5 seconds, draining the queue after every four updates; 2-second idle pauses separated bursts. Three repeats interleaved main and all core selections. Rendering and periodic composition readbacks are excluded from this compute gate; the rendered preset bursts below measure their own fps and completed update rate. Instruction slots are proposed pairs×budget, including inactive slots and early halts, not counted hardware cycles or executed instructions.

| Build/core          | Updates/s (median) | Instruction slots/s | Range of updates/s | Change vs main |
| ------------------- | -----------------: | ------------------: | -----------------: | -------------: |
| main · Z80 0.2      |              541.5 |              567.8M |        534.5–545.2 |       baseline |
| Z80 0.2             |              560.0 |              587.2M |        538.5–575.8 |           3.4% |
| Z80 0.3 classic     |              534.5 |              560.5M |        530.1–551.1 |          -1.3% |
| Z80 0.3 Zilog NMOS  |              531.0 |              556.8M |        528.7–541.1 |          -1.9% |
| NMOS 6502           |             1314.3 |             1378.2M |      1267.2–1333.8 |         142.7% |
| Coherent-fetch 8086 |              400.7 |              420.1M |        310.0–401.6 |         -26.0% |

**Gate passes:** Z80 compatibility 0.2 is faster in this measurement; both 0.3 variants are within 10% of main. Absolute rates vary with soup trajectory and local system conditions, especially 8086. These are short local measurements, not universal device guarantees. Raw repetitions: [benchmark.json](benchmark.json). Reproduce with [benchmark.mjs](../../scripts/benchmark.mjs) after creating the detached baseline checkout at `/tmp/algocell-main-review` and installing its lockfile.

The first integration failed the gate: the generic library loop/helpers yielded roughly 25% lower Z80 throughput. [Initial measurement](benchmark-initial.json) is retained. The final soup keeps the original tight host execution loop and dynamic byte packing/modulo; Zilion library mode owns ISA text, memory accesses, resets, decode/trap hooks and GPU fetch maps. The adapter changes only the generated memory-position helper, with an exact contract guard; it never patches ISA instructions. Unused soup write-count records are omitted. Separate culture machines retain their full records. This adapter is why the package version is pinned.

## Visual regression

Playwright screenshots use 1440×900 and 390×844, paused and reseeded to the same original random condition. **The main field, toolbar, information panel and original hover inspector are pixel-identical at both sizes:** 0 changed pixels and 0 mean channel error. The byte renderer, native square/hex visuals, original diagrams, palettes and panel typography remain.

| View                      | 1440×900 before/after                                                                       | 390×844 before/after                                                                      | Intended change                                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Main                      | [before](screenshots/main-1440-main.png) / [after](screenshots/app-1440-main.png)           | [before](screenshots/main-390-main.png) / [after](screenshots/app-390-main.png)           | None                                                                                                                                           |
| Original hover inspector  | [before](screenshots/main-1440-hover.png) / [after](screenshots/app-1440-hover.png)         | [before](screenshots/main-390-hover.png) / [after](screenshots/app-390-hover.png)         | None                                                                                                                                           |
| Preset/settings menu      | [before](screenshots/main-1440-presets.png) / [after](screenshots/app-1440-presets.png)     | [before](screenshots/main-390-presets.png) / [after](screenshots/app-390-presets.png)     | Source-bound paper selector, processor/lethality controls, compact World contract disclosure, mutation-off and fetch view controls             |
| Persistent tape inspector | [before](screenshots/main-1440-inspector.png) / [after](screenshots/app-1440-inspector.png) | [before](screenshots/main-390-inspector.png) / [after](screenshots/app-390-inspector.png) | Click now opens a persistent original-style modal with culture and expandable ISA disassembly; baseline image is the preserved hover inspector |
| About                     | [before](screenshots/main-1440-about.png) / [after](screenshots/app-1440-about.png)         | [before](screenshots/main-390-about.png) / [after](screenshots/app-390-about.png)         | Scientific paragraphs/credits updated within original modal typography; added Guide tab                                                        |
| Guide                     | [before](screenshots/main-1440-guide.png) / [after](screenshots/app-1440-guide.png)         | [before](screenshots/main-390-guide.png) / [after](screenshots/app-390-guide.png)         | Scientific guide within the original modal; baseline is the existing Keys page, which remains available on desktop                             |

Changed-view whole-image differences are expected: the persistent inspector dims the field, and the scientific text has different content/length. They are not represented as unchanged-view matches. [Quantitative differences](visual-diff.json), [desktop main diff](screenshots/diff-1440-main.png), [phone main diff](screenshots/diff-390-main.png), [capture errors](screenshots/errors.json), [capture script](../../scripts/screenshots.mjs) and [diff script](../../scripts/visual-diff.mjs) are included. Every existing SVG/Mermaid diagram remains. About/Guide tabs use the original shell; Keys remains hidden on phones as in main.

Other intended behavior changes: all runs pause after 55 seconds and on hidden tabs; pointer focus has no ring, keyboard focus uses a 1px warm outline; native selects keep their standard arrow/keyboard behavior. New checkpoints and user presets retain machine/world and planting settings. Original .acell files select Z80 compatibility 0.2 with SP=FFFF; the original format still does not guarantee a bit-identical future GPU trajectory.

## Preset runs

All 22 source presets work in their declared core. Every source condition was also tried in all five available core selections: 107 successful runs, 3 explicitly unsupported combinations, 0 caught execution/page errors. Each rendered run is 400ms at 8 updates/frame, then drained before timing; 250ms pauses separate runs. Each has one composition snapshot afterward and, when a repeated top tape exists, a 16-partner/two-generation culture. This test performs no per-frame population readback. Cross-ISA reinterpretations are exploratory, not registered cohorts. Raw rates, top shares, zero fractions and culture observations: [preset-bursts.json](preset-bursts.json); [runner](../../scripts/preset-bursts.mjs).

In each cell below: **fps / completed updates per second**. A plus sign means the post-burst top tape passed the short culture screen; it does not establish heredity or spontaneous discovery. The six seeded demonstrations passed in their declared Z80 core. No top candidate from a random condition passed in these short bursts (normally 112 updates); this is not evidence of impossibility or a reproduction of the paper's horizons.

| Preset                    |        Z80 0.2 | Z80 0.3 classic |   Z80 0.3 NMOS |         6502 |         8086 |
| ------------------------- | -------------: | --------------: | -------------: | -----------: | -----------: |
| stage-g-l16               |   32.1 / 256.8 |    31.8 / 254.0 |   31.5 / 252.3 | 31.6 / 252.5 | 31.2 / 249.2 |
| z80-corrected             |   32.4 / 259.3 |    31.7 / 253.6 |   32.2 / 257.7 | 32.4 / 258.9 | 31.3 / 250.8 |
| lethal-zero               |   32.4 / 259.2 |    32.2 / 257.4 |   32.2 / 257.3 | 32.4 / 259.6 | 31.8 / 254.7 |
| dial-0                    |   32.4 / 259.4 |    31.9 / 255.4 |   32.3 / 258.2 | 32.1 / 256.9 | 31.4 / 250.8 |
| dial-0.01                 |   32.1 / 256.8 |    31.9 / 255.5 |   32.4 / 259.4 | 32.3 / 258.5 | 32.1 / 256.5 |
| dial-0.03                 |   32.0 / 256.0 |    32.5 / 260.3 |   32.1 / 257.2 | 31.6 / 252.8 | 31.4 / 250.8 |
| dial-0.1                  |   31.5 / 251.7 |    31.7 / 254.0 |   32.2 / 257.6 | 32.0 / 255.9 | 31.5 / 251.9 |
| dial-0.3                  |   31.9 / 254.8 |    31.9 / 255.4 |   32.5 / 260.1 | 32.1 / 257.1 | 31.7 / 253.5 |
| dial-1                    |   32.4 / 258.9 |    31.8 / 254.7 |   31.7 / 253.9 | 31.9 / 255.1 | 31.7 / 253.6 |
| random-registers          |   32.3 / 258.5 |    31.6 / 252.6 |   32.2 / 257.2 | 31.9 / 255.1 | 25.9 / 207.1 |
| ring-l12-p24              |   32.0 / 256.4 |    32.1 / 256.8 |   32.6 / 260.9 | 32.3 / 258.4 | 31.7 / 253.5 |
| ring-l12-p28              |   32.1 / 257.2 |    31.9 / 255.2 |   31.2 / 249.8 | 31.6 / 252.4 | 31.9 / 254.9 |
| no-block-copy             |   32.0 / 256.3 |    31.6 / 252.7 |   31.9 / 255.0 |  unsupported | 31.6 / 252.9 |
| no-copy                   |   32.1 / 257.1 |    32.0 / 255.8 |   31.8 / 254.7 |  unsupported |  unsupported |
| m6502-s1-a                |   32.5 / 260.2 |    32.3 / 258.0 |   32.3 / 258.7 | 32.0 / 256.2 | 31.9 / 255.4 |
| i8086-s2-a                |   31.2 / 249.4 |    32.2 / 257.5 |   31.3 / 250.7 | 31.9 / 254.8 | 31.5 / 252.3 |
| offset-competition-benign | 32.0 / 255.9 + |  31.7 / 253.6 + | 32.2 / 257.4 + | 32.0 / 256.2 | 31.1 / 249.0 |
| offset-competition-lethal | 32.6 / 260.7 + |  32.3 / 258.4 + | 32.6 / 260.7 + | 32.4 / 259.1 | 31.1 / 249.1 |
| displacement-4            | 32.5 / 259.9 + |  32.4 / 259.2 + | 31.6 / 252.7 + | 32.1 / 257.1 | 31.9 / 255.2 |
| displacement-32           | 32.0 / 256.0 + |  32.4 / 259.3 + | 32.1 / 257.1 + | 32.6 / 260.7 | 32.3 / 258.5 |
| pusher                    | 32.0 / 255.7 + |  32.8 / 262.6 + | 32.3 / 258.4 + | 32.2 / 257.5 | 32.3 / 258.5 |
| return-closer             | 32.8 / 262.2 + |  32.0 / 256.0 + | 32.6 / 260.8 + | 32.4 / 259.1 | 32.2 / 257.3 |

Cannot express without changing the named intervention: 6502 has no block-copy family for the no-block-copy condition; the Z80 no-copy family's LD8/LD16/EX taxonomy has no declared equivalent on 6502 or 8086. These combinations are rejected, never silently treated as equivalent. No catalogue preset itself is withheld. BFF, hardware 8088 prefetch, full inflow/ancestry/branching assays and a marker-transfer UI are outside this request and remain unavailable. The displacement marker demonstrations are available through their planted tapes and culture offspring.

## Verification and owner's defects

- `npm run check`: zero errors/warnings.
- `npm run lint`: passes Prettier and ESLint.
- `npm run build`: static production build succeeds.
- `npm run test:e2e`: 40 tests pass, including all 22 presets in restored controls, all core selections, GPU library/standalone memory and fetch-map equality, discarded-padding confinement, pointer/keyboard focus, stable DOM nodes, culture, mobile documentation, unsupported WebGPU and .acell machine restore.
- Svelte MCP autofixer: zero issues for restored component and route. Original GPU synchronization effects remain intentional.
- Flicker: GPU handles use raw state; static controls derive from settings, not per-step counters. Tests retain the same preset select/chip nodes and enabled state while running. Culture/preset controls only change during explicit operations.
- Dropdowns: native selects, visible arrows, original panel colors and typography; keyboard operation retained.
- Focus: Chrome's native-select pointer focus quirk is overridden by input modality. Keyboard-visible focus is 1px and consistent; pointer focus has no outline or shadow.
- Layout: full-screen soup, original compact toolbar/info bar, original-width settings with closed World contract disclosure. No replacement desk or separate legacy route remains. Applying paper geometry fits the full lattice to the actual canvas aspect; a regression assertion covers every preset.
- Confinement: GPU per-lane fetch bitmaps, including operands and padding; overlay records the last initiating encounter only. Modified/mutated tapes lose their observation. Unknown is shown separately. This does not measure data dependence or heredity.
- Culture: separate Zilion machine, created on demand; no assay readback or CPU pairing is added to the soup loop. Historical CPU population service remains only as an unreachable reference-test service; the rejected UI/session have been removed.

## Local commits and manual review

1. `334e470` — GPU-resident Zilion library integration.
2. `8378044` — restored-instrument commit — original route, controls, scientific content, persistence and rejected-desk removal.
3. `6457fda` — fit paper geometry to the actual canvas aspect.
4. Verification/review commit — tests, reproduction scripts and these artifacts.

Every commit ends with the requested Claude Opus 5.5 co-author trailer. See the final coordinator message for final hashes.

Coordinator should check native dropdown keyboard interaction on a visible browser, wheel/pinch/drag and hex palette/detail modes, About/Guide long-content scrolling, mutation-off/World contract controls, and a seeded displacement culture plus saved .acell round-trip. Review the guarded Zilion memory adapter before any future dependency update. Runs are deliberately short; longer scientific cohorts, snapshot ancestry and full paper assay eligibility have not been claimed here.

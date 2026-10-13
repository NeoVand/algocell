# Evolution baseline

Frozen revision: `ee67698dd7b743e838eacd684855a3d9fcefb362` on `feat/app-legacy-base`, captured before application changes. A detached checkout is retained at `/tmp/algocell-evolution-ee67698`; neither `main` nor any remote was changed.

## Performance

Apple M4 MacBook Air, 32 GiB memory, macOS 26.3 / Darwin 25.3.0, Chrome 155.0.8059.40, WebGPU Metal 3. Results are local short-burst measurements, subject to soup trajectory, GPU scheduling and thermal variation.

All selections use exactly 200×200 square cells, L16/P32, 8192 proposed pairs/update, 128 maximum steps, seed 6, mutation exponent 4, benign zero, no ablation and fixed initial registers. Z80 uses the legacy stack and historical random dial; other cores use first-push stack and counter dial. No measurement overlay is enabled. Both baseline and candidate compile as minified production Vite engine bundles.

One completed warmup is followed by reset to seed 6. Each measurement submits four updates and waits for completed GPU work, repeatedly for five seconds. Each run closes its page, destroys its engine and pauses for two seconds before the next run. Three repeats interleave the five cores and two modes. Every active burst is below 60 seconds.

“Rendered” draws the native canvas at 1440×900 after each batch of four updates, without requestAnimationFrame throttling. It measures GPU rendering cost in addition to simulation. It does not measure Svelte interaction, assay work or population readback. Compare UI behavior separately using Playwright. Instruction-slot rate means proposed pairs × maximum steps, including inactive slots and early halts; it is not a count of executed instructions.

| Core                | Compute updates/s, median (range) | Rendered updates/s, median (range) |
| ------------------- | --------------------------------: | ---------------------------------: |
| Z80 0.2             |               785.8 (780.6–805.9) |                776.8 (756.1–780.2) |
| Z80 0.3 classic     |               785.1 (752.9–791.7) |                751.1 (749.8–788.2) |
| Z80 0.3 Zilog NMOS  |               768.8 (767.6–852.8) |                733.1 (728.8–833.2) |
| NMOS 6502           |            1745.3 (1640.8–1911.1) |             1834.9 (1425.6–1870.5) |
| Coherent-fetch 8086 |               456.5 (433.7–518.0) |                429.4 (419.7–444.8) |

The rendered 6502 median exceeds its compute median because the repetitions vary; the result does not mean rendering accelerates simulation. Marginal comparisons should use interleaved baseline/candidate runs. Raw observations and exact settings: [baseline-benchmark.json](baseline-benchmark.json).

From the working repository, reproduce this frozen baseline with:

```sh
BENCH_REPO=/tmp/algocell-evolution-ee67698 BENCH_LABEL=baseline BENCH_OUTPUT=docs/review/evolution/baseline-benchmark.json node scripts/evolution-benchmark.mjs
```

After implementation, measure the current working tree against the frozen baseline in an interleaved run:

```sh
BASELINE_REPO=/tmp/algocell-evolution-ee67698 BENCH_OUTPUT=docs/review/evolution/benchmark.json node scripts/evolution-benchmark.mjs
```

To recreate the frozen checkout on another machine, use `git worktree add --detach /tmp/algocell-evolution-ee67698 ee67698dd7b743e838eacd684855a3d9fcefb362`, install its lockfile with `npm ci`, and run `npm run prepare` there. The local captured checkout uses the existing repository's `node_modules` through a symlink. No runtime dependencies were added.

## Original instrument screenshots

All captures pause and reset the frozen build to seed 6. Desktop is 1440×900; mobile is 390×844. Original wording remains in these historical screenshots as evidence of the starting point.

| State       | Desktop                                                    | Mobile                                                   |
| ----------- | ---------------------------------------------------------- | -------------------------------------------------------- |
| Field       | [baseline-1440-main.png](baseline-1440-main.png)           | [baseline-390-main.png](baseline-390-main.png)           |
| Settings    | [baseline-1440-settings.png](baseline-1440-settings.png)   | [baseline-390-settings.png](baseline-390-settings.png)   |
| Hover probe | [baseline-1440-hover.png](baseline-1440-hover.png)         | [baseline-390-hover.png](baseline-390-hover.png)         |
| Inspector   | [baseline-1440-inspector.png](baseline-1440-inspector.png) | [baseline-390-inspector.png](baseline-390-inspector.png) |
| About       | [baseline-1440-about.png](baseline-1440-about.png)         | [baseline-390-about.png](baseline-390-about.png)         |
| Guide       | [baseline-1440-guide.png](baseline-1440-guide.png)         | [baseline-390-guide.png](baseline-390-guide.png)         |

Run `npm run build` then `npm run preview -- --host 127.0.0.1 --port 4187 --strictPort` in the frozen checkout, then `node scripts/evolution-baseline-capture.mjs` in the working repository. The capture uses the production build and needs no development-only engine hook. Capture error log: [baseline-capture-errors.json](baseline-capture-errors.json).

import { readFile, writeFile } from 'node:fs/promises';
const benchmark = JSON.parse(await readFile('docs/review/benchmark.json', 'utf8'));
const bursts = JSON.parse(await readFile('docs/review/preset-bursts.json', 'utf8'));
const visual = JSON.parse(await readFile('docs/review/visual-diff.json', 'utf8'));
if (visual.some((r) => ['main', 'hover'].includes(r.view) && r.changedPixelsAbove12 !== 0))
	throw new Error(
		'Unchanged instrument views differ; inspect screenshots before writing the review.'
	);
const cores = [
	['main', 'z80', '0.2', 'classic', 'main · Z80 0.2'],
	['app', 'z80', '0.2', 'classic', 'Z80 0.2'],
	['app', 'z80', '0.3', 'classic', 'Z80 0.3 classic'],
	['app', 'z80', '0.3', 'zilog-nmos', 'Z80 0.3 Zilog NMOS'],
	['app', 'm6502', '0.3', 'nmos', 'NMOS 6502'],
	['app', 'i8086', '0.3', 'nmos', 'Coherent-fetch 8086']
];
const median = (a) => [...a].sort((a, b) => a - b)[Math.floor(a.length / 2)];
const rate = (c) =>
	median(
		benchmark.results
			.filter((r) => [r.build, r.isa, r.compat, r.variant].join('/') === c.slice(0, 4).join('/'))
			.map((r) => r.updatesPerSecond)
	);
const reference = rate(cores[0]);
let report = `# Algocell restored-base review

Local branch: \`feat/app-legacy-base\`, created from \`811925e\`. Baseline: detached \`main\` at \`4c84fec\`. No push or main-ref change was made. Zilion is pinned to 0.3.1; no runtime dependency was added.

## Performance gate

Apple M4, Chromium WebGPU/Metal 3. Same engine defaults on both builds: 200×200 square cells, L16/P32, 8192 pair slots, 128 steps, seed 6, mutation exponent 4, zero suppression and benign zero. The UI's viewport-dependent fit can use approximately 20,000 cells; this gate uses the 200×200 \`DEFAULT_GRID_CONFIG\` explicitly.

Both engines were compiled as minified production Vite bundles from their respective checkouts. One completed warmup, then reseed. Each timed burst ran for 5 seconds, draining the queue after every four updates; 2-second idle pauses separated bursts. Three repeats interleaved main and all core selections. Rendering and periodic composition readbacks are excluded from this compute gate; the rendered preset bursts below measure their own fps and completed update rate. Instruction slots are proposed pairs×budget, including inactive slots and early halts, not counted hardware cycles or executed instructions.

| Build/core | Updates/s (median) | Instruction slots/s | Range of updates/s | Change vs main |
|---|---:|---:|---:|---:|
`;
for (const c of cores) {
	const values = benchmark.results
		.filter((r) => [r.build, r.isa, r.compat, r.variant].join('/') === c.slice(0, 4).join('/'))
		.map((r) => r.updatesPerSecond);
	const r = rate(c);
	report += `| ${c[4]} | ${r.toFixed(1)} | ${((r * 8192 * 128) / 1e6).toFixed(1)}M | ${Math.min(...values).toFixed(1)}–${Math.max(...values).toFixed(1)} | ${c[0] === 'main' ? 'baseline' : ((r / reference - 1) * 100).toFixed(1) + '%'} |\n`;
}
report += `
**Gate passes:** Z80 compatibility 0.2 is faster in this measurement; both 0.3 variants are within 10% of main. Absolute rates vary with soup trajectory and local system conditions, especially 8086. These are short local measurements, not universal device guarantees. Raw repetitions: [benchmark.json](benchmark.json). Reproduce with [benchmark.mjs](../../scripts/benchmark.mjs) after creating the detached baseline checkout at \`/tmp/algocell-main-review\` and installing its lockfile.

The first integration failed the gate: the generic library loop/helpers yielded roughly 25% lower Z80 throughput. [Initial measurement](benchmark-initial.json) is retained. The final soup keeps the original tight host execution loop and dynamic byte packing/modulo; Zilion library mode owns ISA text, memory accesses, resets, decode/trap hooks and GPU fetch maps. The adapter changes only the generated memory-position helper, with an exact contract guard; it never patches ISA instructions. Unused soup write-count records are omitted. Separate culture machines retain their full records. This adapter is why the package version is pinned.

## Visual regression

Playwright screenshots use 1440×900 and 390×844, paused and reseeded to the same original random condition. **The main field, toolbar, information panel and original hover inspector are pixel-identical at both sizes:** 0 changed pixels and 0 mean channel error. The byte renderer, native square/hex visuals, original diagrams, palettes and panel typography remain.

| View |1440×900 before/after |390×844 before/after | Intended change |
|---|---|---|---|
`;
const viewRows = [
	['main', 'Main', 'None'],
	['hover', 'Original hover inspector', 'None'],
	[
		'presets',
		'Preset/settings menu',
		'Source-bound paper selector, processor/lethality controls, compact World contract disclosure, mutation-off and fetch view controls'
	],
	[
		'inspector',
		'Persistent tape inspector',
		'Click now opens a persistent original-style modal with culture and expandable ISA disassembly; baseline image is the preserved hover inspector'
	],
	[
		'about',
		'About',
		'Scientific paragraphs/credits updated within original modal typography; added Guide tab'
	],
	[
		'guide',
		'Guide',
		'Scientific guide within the original modal; baseline is the existing Keys page, which remains available on desktop'
	]
];
for (const [file, label, intended] of viewRows)
	report += `| ${label} | [before](screenshots/main-1440-${file}.png) / [after](screenshots/app-1440-${file}.png) | [before](screenshots/main-390-${file}.png) / [after](screenshots/app-390-${file}.png) | ${intended} |\n`;
report += `
Changed-view whole-image differences are expected: the persistent inspector dims the field, and the scientific text has different content/length. They are not represented as unchanged-view matches. [Quantitative differences](visual-diff.json), [desktop main diff](screenshots/diff-1440-main.png), [phone main diff](screenshots/diff-390-main.png), [capture errors](screenshots/errors.json), [capture script](../../scripts/screenshots.mjs) and [diff script](../../scripts/visual-diff.mjs) are included. Every existing SVG/Mermaid diagram remains. About/Guide tabs use the original shell; Keys remains hidden on phones as in main.

Other intended behavior changes: all runs pause after 55 seconds and on hidden tabs; pointer focus has no ring, keyboard focus uses a 1px warm outline; native selects keep their standard arrow/keyboard behavior. New checkpoints and user presets retain machine/world and planting settings. Original .acell files select Z80 compatibility 0.2 with SP=FFFF; the original format still does not guarantee a bit-identical future GPU trajectory.

## Preset runs

All 22 source presets work in their declared core. Every source condition was also tried in all five available core selections: 107 successful runs, 3 explicitly unsupported combinations, 0 caught execution/page errors. Each rendered run is 400ms at 8 updates/frame, then drained before timing; 250ms pauses separate runs. Each has one composition snapshot afterward and, when a repeated top tape exists, a 16-partner/two-generation culture. This test performs no per-frame population readback. Cross-ISA reinterpretations are exploratory, not registered cohorts. Raw rates, top shares, zero fractions and culture observations: [preset-bursts.json](preset-bursts.json); [runner](../../scripts/preset-bursts.mjs).

In each cell below: **fps / completed updates per second**. A plus sign means the post-burst top tape passed the short culture screen; it does not establish heredity or spontaneous discovery. The six seeded demonstrations passed in their declared Z80 core. No top candidate from a random condition passed in these short bursts (normally 112 updates); this is not evidence of impossibility or a reproduction of the paper's horizons.

| Preset | Z80 0.2 | Z80 0.3 classic | Z80 0.3 NMOS |6502 |8086 |
|---|---:|---:|---:|---:|---:|
`;
for (const id of [...new Set(bursts.results.map((r) => r.id))]) {
	report += `| ${id} |`;
	for (const c of cores.slice(1)) {
		const r = bursts.results.find(
			(r) => r.id === id && [r.isa, r.compat, r.variant].join('/') === c.slice(1, 4).join('/')
		);
		report +=
			r.status === 'ok'
				? ` ${r.fps.toFixed(1)} / ${r.updatesPerSecond.toFixed(1)}${r.culture?.positive ? ' +' : ''} |`
				: ` unsupported |`;
	}
	report += '\n';
}
report += `
Cannot express without changing the named intervention: 6502 has no block-copy family for the no-block-copy condition; the Z80 no-copy family's LD8/LD16/EX taxonomy has no declared equivalent on 6502 or 8086. These combinations are rejected, never silently treated as equivalent. No catalogue preset itself is withheld. BFF, hardware 8088 prefetch, full inflow/ancestry/branching assays and a marker-transfer UI are outside this request and remain unavailable. The displacement marker demonstrations are available through their planted tapes and culture offspring.

## Verification and owner's defects

- \`npm run check\`: zero errors/warnings.
- \`npm run lint\`: passes Prettier and ESLint.
- \`npm run build\`: static production build succeeds.
- \`npm run test:e2e\`: 40 tests pass, including all 22 presets in restored controls, all core selections, GPU library/standalone memory and fetch-map equality, discarded-padding confinement, pointer/keyboard focus, stable DOM nodes, culture, mobile documentation, unsupported WebGPU and .acell machine restore.
- Svelte MCP autofixer: zero issues for restored component and route. Original GPU synchronization effects remain intentional.
- Flicker: GPU handles use raw state; static controls derive from settings, not per-step counters. Tests retain the same preset select/chip nodes and enabled state while running. Culture/preset controls only change during explicit operations.
- Dropdowns: native selects, visible arrows, original panel colors and typography; keyboard operation retained.
- Focus: Chrome's native-select pointer focus quirk is overridden by input modality. Keyboard-visible focus is 1px and consistent; pointer focus has no outline or shadow.
- Layout: full-screen soup, original compact toolbar/info bar, original-width settings with closed World contract disclosure. No replacement desk or separate legacy route remains. Applying paper geometry fits the full lattice to the actual canvas aspect; a regression assertion covers every preset.
- Confinement: GPU per-lane fetch bitmaps, including operands and padding; overlay records the last initiating encounter only. Modified/mutated tapes lose their observation. Unknown is shown separately. This does not measure data dependence or heredity.
- Culture: separate Zilion machine, created on demand; no assay readback or CPU pairing is added to the soup loop. Historical CPU population service remains only as an unreachable reference-test service; the rejected UI/session have been removed.

## Local commits and manual review

1. \`334e470\` — GPU-resident Zilion library integration.
2. \`8378044\` — restored-instrument commit — original route, controls, scientific content, persistence and rejected-desk removal.
3. \`6457fda\` — fit paper geometry to the actual canvas aspect.
4. Verification/review commit — tests, reproduction scripts and these artifacts.

Every commit ends with the requested Claude Opus 5.5 co-author trailer. See the final coordinator message for final hashes.

Coordinator should check native dropdown keyboard interaction on a visible browser, wheel/pinch/drag and hex palette/detail modes, About/Guide long-content scrolling, mutation-off/World contract controls, and a seeded displacement culture plus saved .acell round-trip. Review the guarded Zilion memory adapter before any future dependency update. Runs are deliberately short; longer scientific cohorts, snapshot ancestry and full paper assay eligibility have not been claimed here.
`;
await writeFile('docs/review/REVIEW.md', report);

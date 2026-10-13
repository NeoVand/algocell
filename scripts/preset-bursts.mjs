import { chromium } from '@playwright/test';
import { writeFile, readFile } from 'node:fs/promises';
import { build } from 'vite';
import { createServer } from 'node:http';
import path from 'node:path';
const repo = process.cwd();
const entry = '/tmp/algocell-preset-entry.mjs';
await writeFile(
	entry,
	[
		['GPUEngine', 'src/lib/gpu/engine.ts'],
		['PAPER_PRESETS', 'src/lib/lab/presets.ts'],
		['populationMetrics', 'src/lib/lab/engine.ts'],
		['ablationMasks', 'src/lib/lab/ablation.ts']
	]
		.map(([name, file]) => `export { ${name} } from ${JSON.stringify(path.join(repo, file))};`)
		.join('\n')
);
const outDir = '/tmp/algocell-preset-build';
await build({
	configFile: false,
	root: repo,
	logLevel: 'warn',
	resolve: { alias: { $lib: path.join(repo, 'src/lib') } },
	build: {
		outDir,
		emptyOutDir: true,
		lib: { entry, formats: ['es'], fileName: () => 'entry.js' },
		minify: true
	}
});
const server = createServer(async (req, res) => {
	if (req.url === '/') {
		res.end('<!doctype html>');
		return;
	}
	res.setHeader('Content-Type', 'text/javascript');
	res.end(await readFile(path.join(outDir, 'entry.js')));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const page = await browser.newPage();
await page.goto(url);
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
await page.evaluate(async () => {
	const { GPUEngine, PAPER_PRESETS, populationMetrics, ablationMasks } = await import('/entry.js');
	const canvas = document.createElement('canvas');
	canvas.width = 1440;
	canvas.height = 900;
	document.body.append(canvas);
	const engine = new GPUEngine(6);
	if (!(await engine.init(canvas))) throw new Error('No WebGPU');
	window.burstState = { engine, canvas, PAPER_PRESETS, populationMetrics, ablationMasks };
});
const ids = await page.evaluate(() => window.burstState.PAPER_PRESETS.map((p) => p.id));
const results = [];
for (const id of ids)
	for (const [isa, compat, variant] of [
		['z80', '0.2', 'classic'],
		['z80', '0.3', 'classic'],
		['z80', '0.3', 'zilog-nmos'],
		['m6502', '0.3', 'nmos'],
		['i8086', '0.3', 'nmos']
	]) {
		const result = await page.evaluate(
			async ({ id, isa, compat, variant }) => {
				const { engine, canvas, PAPER_PRESETS, populationMetrics, ablationMasks } =
					window.burstState;
				const preset = PAPER_PRESETS.find((p) => p.id === id);
				const settings = {
					...preset.config,
					isa,
					compat,
					variant,
					stack: isa === 'z80' ? 'top' : 'first-push',
					dialMode: isa === 'z80' && compat === '0.2' ? 'historical-pcg' : 'counter'
				};
				try {
					ablationMasks(isa, settings.ablation);
				} catch (e) {
					return { id, isa, compat, variant, status: 'unsupported', reason: e.message };
				}
				const canonical =
					isa === preset.config.isa &&
					compat === preset.config.compat &&
					variant === preset.config.variant;
				try {
					await engine.configure(settings, preset.initialCondition);
					let frames = 0,
						updates = 0;
					const t0 = performance.now();
					while (performance.now() - t0 < 400) {
						await new Promise((r) => requestAnimationFrame(r));
						for (let k = 0; k < 8; k++) {
							engine.simulateStep();
							updates++;
						}
						engine.render(canvas);
						frames++;
					}
					await engine.waitForIdle();
					const seconds = (performance.now() - t0) / 1000;
					const padded = await engine.readSoupData();
					const L = settings.length;
					const stride = Math.ceil(L / 4) * 4;
					const soup = new Uint8Array(settings.width * settings.height * L);
					for (let c = 0; c < settings.width * settings.height; c++)
						soup.set(padded.subarray(c * stride, c * stride + L), c * L);
					const metrics = populationMetrics(soup, settings, preset.initialCondition);
					const culture =
						metrics.topCount > 1
							? await engine.runCulture(metrics.topTape, { trials: 16, seed: 4242, generations: 2 })
							: null;
					return {
						id,
						isa,
						compat,
						variant,
						canonical,
						status: 'ok',
						updates,
						seconds,
						fps: frames / seconds,
						updatesPerSecond: updates / seconds,
						topShare: metrics.topShare,
						uniqueTapes: metrics.uniqueTapes,
						zeroFraction: metrics.zeroFraction,
						initial: preset.initialCondition.kind,
						culture: culture
							? { positive: culture.positive, gain: culture.gain, confinement: culture.confinement }
							: null,
						emergence:
							preset.initialCondition.kind !== 'random'
								? 'planted demonstration'
								: culture?.positive
									? 'abundant candidate passed short culture screen'
									: 'no culture-positive abundant candidate detected in short burst'
					};
				} catch (e) {
					return { id, isa, compat, variant, canonical, status: 'error', reason: e.message };
				}
			},
			{ id, isa, compat, variant }
		);
		results.push(result);
		console.log(JSON.stringify(result));
		await new Promise((r) => setTimeout(r, 250));
	}
await page.evaluate(() => window.burstState.engine.destroy());
await writeFile(
	'docs/review/preset-bursts.json',
	JSON.stringify(
		{
			method:
				'Each of 22 source presets tested in five core selections. A 400ms animation burst at8 updates/frame, GPU drained before rates. 250ms pauses. One post-burst CPU composition snapshot and optional16-partner/two-generation culture; no population readbacks in the loop. Cross-ISA conditions are exploratory reinterpretations, not registered cohorts.',
			errors,
			results
		},
		null,
		2
	)
);
await browser.close();
await new Promise((r) => server.close(r));

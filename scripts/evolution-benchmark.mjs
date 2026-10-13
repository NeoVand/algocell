import { chromium } from '@playwright/test';
import { build } from 'vite';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import os from 'node:os';
import path from 'node:path';

// BENCH_REPO can point at the frozen baseline or the current working tree.
// Optional BASELINE_REPO interleaves both builds to reduce thermal/time bias.
const sources = [
	...(process.env.BASELINE_REPO ? [['baseline', process.env.BASELINE_REPO]] : []),
	[process.env.BENCH_LABEL ?? 'candidate', process.env.BENCH_REPO ?? process.cwd()]
];
const output = process.env.BENCH_OUTPUT ?? 'docs/review/evolution/benchmark.json';
const burstMs = Math.min(10_000, Math.max(1000, Number(process.env.BURST_MS ?? 5000)));
const pauseMs = Math.max(2000, Number(process.env.PAUSE_MS ?? 2000));
const repeats = Number(process.env.REPEATS ?? 3);
const fixedUpdates = process.env.FIXED_UPDATES
	? Math.max(4, Math.floor(Number(process.env.FIXED_UPDATES) / 4) * 4)
	: null;
const fieldModes = (process.env.FIELD_MODES ?? 'bytes').split(',');
const modes = (process.env.MODES ?? 'compute,rendered').split(',');
const alternateOrder = process.env.BENCH_ORDER === 'alternate';
const warmupUpdates = Math.max(1, Math.min(2048, Number(process.env.WARMUP_UPDATES ?? 1)));
const drainReset = process.env.DRAIN_RESET === '1';
const base = await import('node:fs/promises').then((fs) =>
	fs.mkdtemp(path.join(os.tmpdir(), 'algocell-evolution-bench-'))
);
const revisions = {};
const bundleSha256 = {};
for (const [name, repo] of sources) {
	revisions[name] = execFileSync('git', ['-C', repo, 'rev-parse', 'HEAD'], {
		encoding: 'utf8'
	}).trim();
	await build({
		configFile: false,
		root: repo,
		logLevel: 'warn',
		resolve: { alias: { $lib: path.join(repo, 'src/lib') } },
		build: {
			outDir: path.join(base, name),
			emptyOutDir: true,
			lib: {
				entry: path.join(repo, 'src/lib/gpu/engine.ts'),
				formats: ['es'],
				fileName: () => 'engine.js'
			},
			minify: true
		}
	});
	bundleSha256[name] = createHash('sha256')
		.update(await readFile(path.join(base, name, 'engine.js')))
		.digest('hex');
}
const server = createServer(async (req, res) => {
	try {
		if (req.url === '/') {
			res.end('<!doctype html><canvas width="1440" height="900"></canvas>');
			return;
		}
		res.setHeader('Content-Type', 'text/javascript');
		res.end(await readFile(path.join(base, req.url)));
	} catch {
		res.statusCode = 404;
		res.end();
	}
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const cases = [
	['z80', '0.2', 'classic'],
	['z80', '0.3', 'classic'],
	['z80', '0.3', 'zilog-nmos'],
	['m6502', '0.3', 'nmos'],
	['i8086', '0.3', 'nmos']
].filter((entry) => !process.env.CASES || process.env.CASES.split(',').includes(entry.join(':')));
const results = [];
try {
	for (let repeat = 0; repeat < repeats; repeat++)
		for (const [isa, compat, variant] of cases)
			for (const mode of modes)
				for (const fieldMode of fieldModes)
					for (const [buildName] of alternateOrder && repeat % 2
						? [...sources].reverse()
						: sources) {
						const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
						await page.goto(`http://127.0.0.1:${server.address().port}`);
						const result = await page.evaluate(
							async ({
								buildName,
								isa,
								compat,
								variant,
								mode,
								fieldMode,
								burstMs,
								fixedUpdates,
								warmupUpdates,
								drainReset
							}) => {
								const { GPUEngine } = await import(`/${buildName}/engine.js`);
								const engine = new GPUEngine(6);
								const canvas = document.querySelector('canvas');
								if (!(await engine.init(canvas))) throw new Error('WebGPU unavailable');
								const { info } = await navigator.gpu.requestAdapter();
								await engine.configure({
									isa,
									compat,
									variant,
									length: 16,
									ring: 32,
									steps: 128,
									width: 200,
									height: 200,
									topology: 'square',
									seed: 6,
									pairAttempts: 8192,
									mutationExponent: 4,
									lethalProbability: 0,
									randomRegisters: false,
									ablation: [],
									stack: isa === 'z80' ? 'legacy-sp' : 'first-push',
									dialMode: isa === 'z80' ? 'historical-pcg' : 'counter'
								});
								if (fieldMode !== 'bytes') {
									await engine.setFieldMode(fieldMode);
									if (fieldMode === 'marker') engine.setMarker({ position: 8, value: 165 });
								}
								for (let warm = 0; warm < warmupUpdates; warm++) {
									engine.simulateStep();
									if (warm % 4 === 3) await engine.device.queue.onSubmittedWorkDone();
								}
								if (mode === 'rendered') engine.render(canvas);
								await engine.device.queue.onSubmittedWorkDone();
								engine.reset(6);
								if (drainReset) await engine.device.queue.onSubmittedWorkDone();
								const t0 = performance.now();
								let updates = 0;
								let frames = 0;
								while (fixedUpdates ? updates < fixedUpdates : performance.now() - t0 < burstMs) {
									if (performance.now() - t0 > 55_000)
										throw new Error('GPU burst exceeded the 55-second safety budget.');
									for (let k = 0; k < 4; k++) {
										engine.simulateStep();
										updates++;
									}
									if (mode === 'rendered') {
										engine.render(canvas);
										frames++;
									}
									await engine.device.queue.onSubmittedWorkDone();
								}
								const seconds = (performance.now() - t0) / 1000;
								engine.destroy();
								return {
									build: buildName,
									isa,
									compat,
									variant,
									mode,
									fieldMode,
									updates,
									frames,
									seconds,
									updatesPerSecond: updates / seconds,
									instructionSlotsPerSecond: (updates * 8192 * 128) / seconds,
									gpu: {
										vendor: info.vendor,
										architecture: info.architecture,
										device: info.device,
										description: info.description
									}
								};
							},
							{
								buildName,
								isa,
								compat,
								variant,
								mode,
								fieldMode,
								burstMs,
								fixedUpdates,
								warmupUpdates,
								drainReset
							}
						);
						results.push({ ...result, repeat });
						console.log(JSON.stringify(results.at(-1)));
						await page.close();
						await new Promise((resolve) => setTimeout(resolve, pauseMs));
					}
	await mkdir(path.dirname(output), { recursive: true });
	await writeFile(
		output,
		JSON.stringify(
			{
				capturedAt: new Date().toISOString(),
				revisions,
				bundleSha256,
				environment: {
					platform: os.platform(),
					release: os.release(),
					cpu: os.cpus()[0].model,
					memoryGiB: os.totalmem() / 2 ** 30,
					browser: browser.version()
				},
				settings: {
					grid: '200x200 square',
					length: 16,
					ring: 32,
					steps: 128,
					pairs: 8192,
					seed: 6,
					mutationExponent: 4,
					viewport: '1440x900',
					measurement: fieldModes.some((mode) => mode !== 'bytes'),
					fieldModes,
					fixedUpdates,
					alternateOrder,
					warmupUpdates,
					drainReset,
					burstMs,
					pauseMs,
					repeats
				},
				method:
					'Minified production engine bundles. settings.warmupUpdates completed warmup updates, reset to seed 6 (settings.drainReset optionally drains reset upload before timing), batches of four updates drained through onSubmittedWorkDone. Rendered mode draws once per batch, without requestAnimationFrame throttling. No population or assay readback. Runs interleaved by repeat, core, mode, field mode, build. settings.fixedUpdates selects equal completed workload instead of equal time; alternateOrder reverses build order on odd repeats. Instruction slots include inactive slots and early halts; rendered rates measure GPU canvas cost, not UI interactions.',
				results
			},
			null,
			2
		) + '\n'
	);
} finally {
	await browser.close();
	await new Promise((resolve) => server.close(resolve));
}

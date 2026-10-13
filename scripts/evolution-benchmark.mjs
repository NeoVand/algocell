import { chromium } from '@playwright/test';
import { build } from 'vite';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { execFileSync } from 'node:child_process';
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
const base = await import('node:fs/promises').then((fs) =>
	fs.mkdtemp(path.join(os.tmpdir(), 'algocell-evolution-bench-'))
);
const revisions = {};
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
];
const results = [];
try {
	for (let repeat = 0; repeat < repeats; repeat++)
		for (const [isa, compat, variant] of cases)
			for (const mode of ['compute', 'rendered'])
				for (const [buildName] of sources) {
					const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
					await page.goto(`http://127.0.0.1:${server.address().port}`);
					const result = await page.evaluate(
						async ({ buildName, isa, compat, variant, mode, burstMs }) => {
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
							engine.simulateStep();
							if (mode === 'rendered') engine.render(canvas);
							await engine.device.queue.onSubmittedWorkDone();
							engine.reset(6);
							const t0 = performance.now();
							let updates = 0;
							let frames = 0;
							while (performance.now() - t0 < burstMs) {
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
						{ buildName, isa, compat, variant, mode, burstMs }
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
					measurement: false,
					burstMs,
					pauseMs,
					repeats
				},
				method:
					'Minified production engine bundles. One completed warmup, reset to seed 6, batches of four updates drained through onSubmittedWorkDone. Rendered mode draws once per batch, without requestAnimationFrame throttling. No population or assay readback. Runs interleaved by repeat, core, mode, build. Instruction slots include inactive slots and early halts; rendered rates measure GPU canvas cost, not UI interactions.',
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

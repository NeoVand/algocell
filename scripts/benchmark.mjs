import { chromium } from '@playwright/test';
import { build } from 'vite';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
const base = '/tmp/algocell-benchmark-builds';
for (const [name, repo] of [
	['main', '/tmp/algocell-main-review'],
	['app', process.cwd()]
]) {
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
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const cases = [
	['main', 'z80', '0.2', 'classic'],
	['app', 'z80', '0.2', 'classic'],
	['app', 'z80', '0.3', 'classic'],
	['app', 'z80', '0.3', 'zilog-nmos'],
	['app', 'm6502', '0.3', 'nmos'],
	['app', 'i8086', '0.3', 'nmos']
];
const results = [];
for (let repeat = 0; repeat < Number(process.env.REPEATS ?? 3); repeat++)
	for (const [buildName, isa, compat, variant] of cases.filter(
		(c) => !process.env.CASES || process.env.CASES.split(',').includes(c.slice(0, 4).join(':'))
	)) {
		const page = await browser.newPage();
		await page.goto(url);
		const result = await page.evaluate(
			async ({ buildName, isa, compat, variant, burstMs }) => {
				const { GPUEngine } = await import(`/${buildName}/engine.js`);
				const engine = new GPUEngine(6);
				const canvas = document.querySelector('canvas');
				if (!(await engine.init(canvas))) throw new Error('WebGPU unavailable');
				const adapter = await navigator.gpu.requestAdapter();
				const info = adapter.info;
				if (buildName === 'app')
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
				// One completed warmup, then reseed: all timed runs begin with the same soup.
				engine.simulateStep();
				await engine.device.queue.onSubmittedWorkDone();
				engine.reset(6);
				const t0 = performance.now();
				let updates = 0;
				do {
					for (let k = 0; k < 4; k++) {
						engine.simulateStep();
						updates++;
					}
					await engine.device.queue.onSubmittedWorkDone();
				} while (performance.now() - t0 < burstMs);
				const seconds = (performance.now() - t0) / 1000;
				engine.destroy();
				return {
					build: buildName,
					isa,
					compat,
					variant,
					updates,
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
			{ buildName, isa, compat, variant, burstMs: Number(process.env.BURST_MS ?? 5000) }
		);
		results.push({ ...result, repeat });
		console.log(JSON.stringify(results.at(-1)));
		await page.close();
		await new Promise((r) => setTimeout(r, 2000));
	}
await mkdir('docs/review', { recursive: true });
await writeFile(
	'docs/review/benchmark.json',
	JSON.stringify(
		{
			settings: {
				grid: '200x200 square',
				length: 16,
				steps: 128,
				pairs: 8192,
				seed: 6,
				mutationExponent: 4,
				measurement: false
			},
			method:
				'Minified production Vite bundles; batches of four updates drained through onSubmittedWorkDone; one warmup then reset; each timed burst 5s, pause 2s; three repeats interleaved by core.',
			results
		},
		null,
		2
	)
);
await browser.close();
await new Promise((r) => server.close(r));

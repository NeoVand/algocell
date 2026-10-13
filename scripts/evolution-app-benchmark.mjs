import { chromium, expect } from '@playwright/test';
import { spawn, execFileSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import os from 'node:os';

// Run only with an exclusive quiet CPU/GPU window. Both full applications render
// and perform their normal UI/statistics work; no replacement animation loop.
const sources = [
	{
		name: 'baseline',
		repo: process.env.BASELINE_REPO ?? '/tmp/algocell-evolution-ee67698',
		port: 4187
	},
	{ name: 'candidate', repo: process.cwd(), port: 4188 }
];
const burstMs = Math.min(10_000, Number(process.env.BURST_MS ?? 5000));
const repeats = Number(process.env.REPEATS ?? 3);
const pauseMs = Math.max(2000, Number(process.env.PAUSE_MS ?? 3000));
const output = process.env.BENCH_OUTPUT ?? 'docs/review/evolution/app-benchmark.json';
const cores = [
	['z80', '0.2', 'classic', 'z80-0.2'],
	['z80', '0.3', 'classic', 'z80-0.3'],
	['z80', '0.3', 'zilog-nmos', 'z80-nmos'],
	['m6502', '0.3', 'nmos', 'm6502'],
	['i8086', '0.3', 'nmos', 'i8086']
];
const children = [];
const results = [];
const errors = [];
let browser;

async function startServer(source) {
	const code = `
		import { createServer } from 'vite';
		import { realpathSync } from 'node:fs';
		const server = await createServer({ server: {
			host: '127.0.0.1', port: ${source.port}, strictPort: true,
			fs: { allow: [process.cwd(), realpathSync('node_modules')] }
		} });
		await server.listen();
		console.log('ALGOCELL BENCH READY');
		process.once('SIGTERM', async () => { await server.close(); process.exit(0); });
	`;
	const child = spawn(process.execPath, ['--input-type=module', '-e', code], {
		cwd: source.repo,
		stdio: ['ignore', 'pipe', 'pipe']
	});
	children.push(child);
	await new Promise((resolve, reject) => {
		let log = '';
		const timeout = setTimeout(
			() => reject(new Error(`Server startup timed out: ${source.name}\n${log}`)),
			30000
		);
		child.stderr.on('data', (data) => {
			log += data;
		});
		child.stdout.on('data', (data) => {
			log += data;
			if (log.includes('ALGOCELL BENCH READY')) {
				clearTimeout(timeout);
				resolve();
			}
		});
		child.once('exit', (code) => {
			clearTimeout(timeout);
			reject(new Error(`Server ${source.name} exited ${code}\n${log}`));
		});
	});
}

async function prepare(source, core = cores[0]) {
	const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
	page.on('pageerror', (error) => errors.push({ build: source.name, message: error.message }));
	await page.goto(`http://127.0.0.1:${source.port}`);
	await page.waitForFunction(() => Boolean(window.__algocell?.engine));
	await page.getByTitle('Pause (Space)', { exact: true }).click();
	await page.getByTitle('Settings', { exact: true }).click();
	await page.getByLabel('Processor', { exact: true }).selectOption(core[3]);
	await expect(page.getByLabel('Processor', { exact: true })).toBeEnabled();
	await page.getByLabel('Close settings', { exact: true }).click();
	await page.evaluate(() => document.fonts.ready);
	await page.evaluate(async ([isa, compat, variant]) => {
		const engine = window.__algocell.engine;
		await engine.waitForIdle();
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
		const canvas = document.querySelector('canvas');
		engine.resetView(canvas.width, canvas.height);
		engine.simulateStep();
		engine.render(canvas);
		await engine.waitForIdle();
		engine.reset(6);
		await engine.waitForIdle();
	}, core);
	// Let the application observe the explicit condition change before timing.
	await page.waitForTimeout(650);
	const speed = (await page.getByTitle('Speed', { exact: true }).innerText()).trim();
	if (speed !== '8x') throw new Error(`Expected native 8x speed, got ${speed}`);
	return page;
}

try {
	for (const source of sources) {
		source.revision = execFileSync('git', ['-C', source.repo, 'rev-parse', 'HEAD'], {
			encoding: 'utf8'
		}).trim();
		await startServer(source);
	}
	browser = await chromium.launch({
		channel: 'chrome',
		headless: true,
		args: ['--enable-unsafe-webgpu']
	});
	// Compile/hydrate both full applications before any timed run.
	for (const source of sources) {
		const page = await prepare(source);
		await page.close();
		await new Promise((resolve) => setTimeout(resolve, pauseMs));
	}
	for (let repeat = 0; repeat < repeats; repeat++) {
		// Reverse alternate repeats to reduce always-running-second bias.
		for (const core of cores)
			for (const source of repeat % 2 ? [...sources].reverse() : sources) {
				const page = await prepare(source, core);
				const result = await page.evaluate(async (burstMs) => {
					const engine = window.__algocell.engine;
					await engine.waitForIdle();
					engine.reset(6);
					await engine.waitForIdle();
					const calls = {};
					for (const name of ['readSoupData', 'readCell', 'readCellHashes', 'readByteCounts']) {
						if (typeof engine[name] !== 'function') continue;
						const method = engine[name].bind(engine);
						calls[name] = 0;
						engine[name] = (...args) => {
							calls[name]++;
							return method(...args);
						};
					}
					const initialUpdate = engine.batchCount;
					const start = performance.now();
					let frames = 0;
					let previous = start;
					let maxFrameGapMs = 0;
					document.querySelector('button[title="Play (Space)"]').click();
					await new Promise((resolve) => {
						const observe = (now) => {
							frames++;
							maxFrameGapMs = Math.max(maxFrameGapMs, now - previous);
							previous = now;
							if (now - start >= burstMs) {
								document.querySelector('button[title="Pause (Space)"]').click();
								resolve();
							} else requestAnimationFrame(observe);
						};
						requestAnimationFrame(observe);
					});
					await engine.waitForIdle();
					const seconds = (performance.now() - start) / 1000;
					const updates = engine.batchCount - initialUpdate;
					return {
						frames,
						updates,
						seconds,
						fps: frames / seconds,
						updatesPerSecond: updates / seconds,
						instructionSlotsPerSecond: (updates * 8192 * 128) / seconds,
						maxFrameGapMs,
						calls,
						settings: engine.settings,
						view: { ...engine.view },
						canvas: { width: engine.context.canvas.width, height: engine.context.canvas.height }
					};
				}, burstMs);
				results.push({
					build: source.name,
					repeat,
					isa: core[0],
					compat: core[1],
					variant: core[2],
					...result
				});
				console.log(JSON.stringify(results.at(-1)));
				await page.close();
				await new Promise((resolve) => setTimeout(resolve, pauseMs));
			}
	}
	await mkdir('docs/review/evolution', { recursive: true });
	await writeFile(
		output,
		JSON.stringify(
			{
				capturedAt: new Date().toISOString(),
				sources: sources.map(({ name, revision }) => ({ name, revision })),
				environment: {
					cpu: os.cpus()[0].model,
					memoryGiB: os.totalmem() / 2 ** 30,
					release: os.release(),
					browser: browser.version()
				},
				method:
					'Actual full applications in matching Vite development servers. Both apps preloaded before timing. Native 8x UI loop, 1440x900 canvas, 200x200 L16/P32, seed 6, 8192 pairs, 128 steps, mutation exponent 4, benign zero. One completed warmup then reset. Five-second bursts with final GPU drain; real UI and periodic statistics remain active. requestAnimationFrame observer counts presented opportunities. Three repeats with alternating build order; pages closed before idle pauses. Normal readback calls counted without changing their implementation.',
				burstMs,
				pauseMs,
				repeats,
				errors,
				results
			},
			null,
			2
		) + '\n'
	);
} finally {
	await browser?.close();
	for (const child of children) child.kill('SIGTERM');
	await Promise.all(
		children.map((child) =>
			child.exitCode !== null
				? Promise.resolve()
				: new Promise((resolve) => child.once('exit', resolve))
		)
	);
}

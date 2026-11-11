// Headless sweep of the reversible interaction rule. Needs the dev server of this checkout:
//   npm run dev -- --host 127.0.0.1 --port 5176 --strictPort
//   node scripts/reversible-sweep.mjs --plan docs/reversible-sweep/plan.json --out docs/reversible-sweep/coarse.jsonl --budget 50
// The plan is a JSON array of runs: {interaction, dissipation, mutationExponent, boundary, seed, updates,
// checkpoints: [..], initial?: 'random' | 'pusher'}. Finished runs (same key) are skipped, and no new
// run starts after --budget seconds, so a long plan is worked through in short GPU bursts.
import { chromium } from '@playwright/test';
import { readFile, appendFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const arg = (name, fallback) => {
	const i = process.argv.indexOf(`--${name}`);
	return i < 0 ? fallback : process.argv[i + 1];
};
const planFile = arg('plan');
const outFile = arg('out');
const budget = Number(arg('budget', '50'));
const soupDir = arg('soups', '');
const url = arg('url', 'http://127.0.0.1:5176/demo/playwright');
if (!planFile || !outFile) throw new Error('Give --plan and --out.');

const keyOf = (r) =>
	[
		r.interaction,
		r.dissipation,
		r.mutationExponent,
		r.boundary ?? 'reflect',
		r.seed,
		r.updates,
		r.initial ?? 'random',
		r.width ?? '',
		r.height ?? ''
	].join('|');
const plan = JSON.parse(await readFile(planFile, 'utf8'));
const done = new Set();
if (existsSync(outFile))
	for (const line of (await readFile(outFile, 'utf8')).split('\n'))
		if (line.trim()) done.add(JSON.parse(line).key);
const todo = plan.filter((r) => !done.has(keyOf(r)));
console.log(`${plan.length} runs in plan, ${todo.length} to do`);
if (!todo.length) process.exit(0);
if (soupDir) await mkdir(soupDir, { recursive: true });

const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('page error:', e.message));
await page.goto(url);
await page.evaluate(async () => {
	const { GPUEngine } = await import('/src/lib/gpu/engine.ts');
	const { ORIGINAL_SETTINGS } = await import('/src/lib/gpu/machine.ts');
	const canvas = document.createElement('canvas');
	canvas.width = 64;
	canvas.height = 64;
	document.body.append(canvas);
	const engine = new GPUEngine(6);
	if (!(await engine.init(canvas))) throw new Error('No WebGPU');

	const gzipRatio = async (bytes) => {
		const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'));
		return (await new Response(stream).arrayBuffer()).byteLength / bytes.length;
	};
	const measure = async (X, P, prev, w, h, L) => {
		const cells = w * h;
		const n = cells * L;
		const hist = new Float64Array(256);
		for (let i = 0; i < n; i++) hist[X[i]]++;
		let entropy = 0;
		for (let v = 0; v < 256; v++) if (hist[v]) entropy -= (hist[v] / n) * Math.log2(hist[v] / n);
		const counts = new Map();
		const zeroCell = new Uint8Array(cells);
		for (let c = 0; c < cells; c++) {
			const tape = X.subarray(c * L, (c + 1) * L);
			const k = String.fromCharCode.apply(null, tape);
			counts.set(k, (counts.get(k) ?? 0) + 1);
			let z = 1;
			for (let b = 0; b < L; b++) if (tape[b]) z = 0;
			zeroCell[c] = z;
		}
		let top = 0;
		let topKey = '';
		let sumSq = 0;
		for (const [k, v] of counts) {
			sumSq += v * v;
			if (v > top) {
				top = v;
				topKey = k;
			}
		}
		// Position-wise byte agreement with the right and lower neighbour, interior pairs only, so the
		// number does not depend on the boundary rule.
		let agree = 0;
		let pairs = 0;
		let sameTape = 0;
		let tapePairs = 0;
		for (let y = 0; y < h; y++)
			for (let x = 0; x < w; x++) {
				const a = (y * w + x) * L;
				for (const [dx, dy] of [
					[1, 0],
					[0, 1]
				]) {
					if (x + dx >= w || y + dy >= h) continue;
					const b = ((y + dy) * w + x + dx) * L;
					let same = 0;
					for (let k = 0; k < L; k++) if (X[a + k] === X[b + k]) same++;
					agree += same;
					pairs += L;
					if (same === L) sameTape++;
					tapePairs++;
				}
			}
		let hot = 0;
		let restCells = 0;
		if (P) {
			for (let c = 0; c < cells; c++) {
				let d = 0;
				for (let b = 0; b < L; b++) if (X[c * L + b] !== P[c * L + b]) d++;
				hot += d;
				if (!d) restCells++;
			}
		}
		let changed = 0;
		if (prev) for (let i = 0; i < n; i++) if (X[i] !== prev[i]) changed++;
		return {
			zero: hist[0] / n,
			zeroCells: zeroCell.reduce((s, v) => s + v, 0) / cells,
			entropy,
			distinct: counts.size / cells,
			top: top / cells,
			topTape: [...topKey].map((ch) => ch.charCodeAt(0).toString(16).padStart(2, '0')).join(' '),
			simpson: sumSq / (cells * cells),
			neighbourBytes: agree / pairs,
			neighbourTapes: sameTape / tapePairs,
			hotBytes: P ? hot / n : null,
			restCells: P ? restCells / cells : null,
			activity100: prev ? changed / n : null,
			gzip: await gzipRatio(X.subarray(0, n))
		};
	};

	window.__run = async (run) => {
		const settings = {
			...ORIGINAL_SETTINGS,
			interaction: run.interaction,
			dissipation: run.dissipation,
			mutationExponent: run.mutationExponent,
			boundary: run.boundary ?? 'reflect',
			seed: run.seed,
			...(run.width ? { width: run.width, height: run.height } : {})
		};
		const t0 = performance.now();
		// 'pusher' starts from a lattice tiled with the replicator the ordinary soup settles on.
		const initial =
			run.initial === 'pusher'
				? {
						kind: 'tiled',
						bytes: Array.from({ length: settings.length }, (_, i) => (i % 2 ? 0xc5 : 0x01))
					}
				: { kind: 'random' };
		await engine.configure(settings, initial);
		const { width: w, height: h, length: L } = settings;
		if (L % 4) throw new Error('Tape length must be a multiple of four for this harness.');
		const marks = new Set();
		for (const c of run.checkpoints) {
			marks.add(c);
			if (c > 100) marks.add(c - 100);
		}
		const rows = [];
		let prev = null;
		let prevAt = -1;
		let last = null;
		for (let u = 1; u <= run.updates; u++) {
			engine.simulateStep();
			if (u % 50 === 0) await engine.waitForIdle();
			if (!marks.has(u)) continue;
			await engine.waitForIdle();
			const X = await engine.readSoupData();
			if (run.checkpoints.includes(u)) {
				const P = run.interaction === 'reversible' ? await engine.readBeforeData() : null;
				rows.push({
					update: u,
					...(await measure(X, P, prevAt === u - 100 ? prev : null, w, h, L))
				});
				last = X;
			}
			prev = X;
			prevAt = u;
		}
		let soup = null;
		if (run.keepSoup && last) {
			let s = '';
			for (let i = 0; i < last.length; i += 8192)
				s += String.fromCharCode.apply(null, last.subarray(i, i + 8192));
			soup = btoa(s);
		}
		return {
			settings,
			rows,
			soup,
			seconds: (performance.now() - t0) / 1000,
			mutationsPerUpdate: Math.floor(settings.pairAttempts / 2 ** settings.mutationExponent)
		};
	};
});

const started = Date.now();
let ran = 0;
for (const run of todo) {
	if ((Date.now() - started) / 1000 > budget) break;
	const result = await page.evaluate((r) => window.__run(r), {
		...run,
		keepSoup: Boolean(soupDir)
	});
	const key = keyOf(run);
	if (soupDir && result.soup) {
		const name = key.replaceAll('|', '_') + '.u8';
		await writeFile(path.join(soupDir, name), Buffer.from(result.soup, 'base64'));
	}
	delete result.soup;
	await appendFile(outFile, JSON.stringify({ key, run, ...result }) + '\n');
	ran++;
	const f = result.rows.at(-1);
	console.log(
		`${key}  ${result.seconds.toFixed(1)} s  zero ${f.zero.toFixed(3)} top ${f.top.toFixed(3)} distinct ${f.distinct.toFixed(3)} hot ${f.hotBytes === null ? '-' : f.hotBytes.toFixed(3)} act ${f.activity100?.toFixed(3)}`
	);
}
console.log(
	`ran ${ran}, ${todo.length - ran} left, ${((Date.now() - started) / 1000).toFixed(0)} s`
);
await browser.close();

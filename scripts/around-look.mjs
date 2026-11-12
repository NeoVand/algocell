// A look at the cell-wise worlds: run headless, save the lattice at several times with summary
// numbers. Needs the dev server on 5176.
//   node scripts/around-look.mjs <out dir> key=value ...
// Keys (defaults): rule=neighbourhood|cells gather=4 steps=128 (loop cut-off, neighbourhood only)
//   mutation=4 (exponent; 31 is off) width=96 height=96 (cells) seed=1 updates=2000
//   marks=0,500,2000 (updates at which to save) patch=0 (side, in cells, of a square of the pusher
//   tape `01 c5 ...` placed in the middle at the start)
// Writes tapes_<update>.u8 (16 bytes per cell, row-major cells), heads_<update>.u32, look.json.
import { chromium } from '@playwright/test';
import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

const [out, ...pairs] = process.argv.slice(2);
if (!out) throw new Error('Give an output directory.');
const q = {
	rule: 'neighbourhood',
	gather: 4,
	steps: 128,
	mutation: 4,
	width: 96,
	height: 96,
	seed: 1,
	updates: 2000,
	marks: '',
	patch: 0
};
for (const pair of pairs) {
	const [k, v] = pair.split('=');
	if (!(k in q)) throw new Error(`Unknown key ${k}`);
	q[k] = typeof q[k] === 'number' ? Number(v) : v;
}
q.marks = (q.marks ? q.marks.split(',').map(Number) : [0, q.updates / 4, q.updates]).map(
	Math.round
);
await mkdir(out, { recursive: true });
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('page error:', e.message));
await page.goto('http://127.0.0.1:5176/demo/playwright');
const result = await page.evaluate(async (q) => {
	const { GPUEngine } = await import('/src/lib/gpu/engine.ts');
	const { ORIGINAL_SETTINGS } = await import('/src/lib/gpu/machine.ts');
	const { HEAD_WORDS, HEAD_AGE, HEAD_WAITING, HEAD_RUNS, HEAD_LAST_RUN } =
		await import('/src/lib/gpu/free.ts');
	const canvas = document.createElement('canvas');
	canvas.width = 64;
	canvas.height = 64;
	document.body.append(canvas);
	const engine = new GPUEngine(6);
	if (!(await engine.init(canvas))) throw new Error('No WebGPU');
	const settings = {
		...ORIGINAL_SETTINGS,
		width: q.width,
		height: q.height,
		seed: q.seed,
		interaction: q.rule,
		gather: q.gather,
		steps: q.steps,
		mutationExponent: q.mutation
	};
	await engine.configure(settings, { kind: 'random' });
	const cells = q.width * q.height;
	if (q.patch) {
		const soup = await engine.readSoupData();
		const x0 = (q.width - q.patch) >> 1;
		const y0 = (q.height - q.patch) >> 1;
		for (let y = y0; y < y0 + q.patch; y++)
			for (let x = x0; x < x0 + q.patch; x++)
				for (let i = 0; i < 16; i++) soup[(y * q.width + x) * 16 + i] = i & 1 ? 0xc5 : 0x01;
		engine.writeSoupData(soup);
	}
	const b64 = (u8) => {
		let t = '';
		for (let i = 0; i < u8.length; i += 8192)
			t += String.fromCharCode.apply(null, u8.subarray(i, i + 8192));
		return btoa(t);
	};
	const rows = [];
	let t = 0;
	const t0 = performance.now();
	const chunk = q.rule === 'cells' ? 50 : 5;
	for (const m of q.marks) {
		while (t < m) {
			const n = Math.min(chunk, m - t);
			engine.simulateUpdates(n);
			t += n;
			if ((t / chunk) % 10 === 0) await engine.waitForIdle();
		}
		await engine.waitForIdle();
		const seconds = (performance.now() - t0) / 1000;
		const tapes = await engine.readSoupData();
		const heads = await engine.readHeads();
		const hist = new Float64Array(256);
		for (const v of tapes) hist[v]++;
		let entropy = 0;
		for (const c of hist) if (c) entropy -= (c / tapes.length) * Math.log2(c / tapes.length);
		const key = (c) => {
			let s = '';
			for (let i = 0; i < 16; i++) s += tapes[c * 16 + i].toString(16).padStart(2, '0');
			return s;
		};
		const keys = new Array(cells);
		const seen = new Map();
		for (let c = 0; c < cells; c++) {
			keys[c] = key(c);
			seen.set(keys[c], (seen.get(keys[c]) ?? 0) + 1);
		}
		let sameEast = 0;
		let sameSouth = 0;
		for (let y = 0; y < q.height; y++)
			for (let x = 0; x < q.width; x++) {
				const c = y * q.width + x;
				if (keys[c] === keys[y * q.width + ((x + 1) % q.width)]) sameEast++;
				if (keys[c] === keys[((y + 1) % q.height) * q.width + x]) sameSouth++;
			}
		const top = [...seen.entries()].sort((a, b) => b[1] - a[1]).slice(0, 4);
		const reasons = [0, 0, 0, 0];
		let steps = 0;
		let runs = 0;
		const last = [];
		for (let c = 0; c < cells; c++) {
			const o = c * HEAD_WORDS;
			reasons[heads[o + HEAD_WAITING] & 3]++;
			steps += heads[o + HEAD_AGE];
			runs += heads[o + HEAD_RUNS];
			if (heads[o + HEAD_RUNS]) last.push(heads[o + HEAD_LAST_RUN]);
		}
		last.sort((a, b) => a - b);
		rows.push({
			update: m,
			seconds: +seconds.toFixed(2),
			entropy: +entropy.toFixed(3),
			zero: +(hist[0] / tapes.length).toFixed(4),
			distinctTapes: seen.size,
			topTapes: top.map(([k, n]) => [k, +(n / cells).toFixed(4)]),
			sameTapeAsEast: +(sameEast / cells).toFixed(4),
			sameTapeAsSouth: +(sameSouth / cells).toFixed(4),
			topBytes: [...hist.entries()]
				.sort((a, b) => b[1] - a[1])
				.slice(0, 5)
				.map(([v, n]) => [v.toString(16).padStart(2, '0'), +(n / tapes.length).toFixed(4)]),
			...(q.rule === 'cells'
				? {
						restarts: runs,
						meanRun: runs ? +((m * cells) / runs).toFixed(1) : null,
						lastRunMedian: last.length ? last[last.length >> 1] : null
					}
				: {
						endedByHalt: +(reasons[1] / cells).toFixed(4),
						endedByFullCell: +(reasons[2] / cells).toFixed(4),
						endedByBudget: +(reasons[3] / cells).toFixed(4),
						meanSteps: +(steps / cells).toFixed(1)
					}),
			tapes: b64(tapes),
			heads: b64(new Uint8Array(heads.buffer))
		});
	}
	engine.destroy();
	return { q, rows };
}, q);
for (const r of result.rows) {
	await writeFile(path.join(out, `tapes_${r.update}.u8`), Buffer.from(r.tapes, 'base64'));
	await writeFile(path.join(out, `heads_${r.update}.u32`), Buffer.from(r.heads, 'base64'));
	delete r.tapes;
	delete r.heads;
	console.log(JSON.stringify(r));
}
await writeFile(path.join(out, 'look.json'), JSON.stringify(result, null, 1));
await browser.close();

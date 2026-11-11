// A first look at the free-running world: run it headless, save the sheet at several times.
//   node scripts/free-look.mjs <out dir> [tiles=128] [heads=1] [directions=weave] [halt=wait] [mutationExponent=10] [ticks=8000] [seed=6]
// Needs the dev server on 5176. Writes sheet_<tick>.u8 (row-major bytes) and look.json.
import { chromium } from '@playwright/test';
import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

const [out, tiles = '128', heads = '1', directions = 'weave', halt = 'wait', mutation = '10', ticks = '8000', seed = '6'] =
	process.argv.slice(2);
if (!out) throw new Error('Give an output directory.');
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu'] });
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('page error:', e.message));
await page.goto('http://127.0.0.1:5176/demo/playwright');
const result = await page.evaluate(
	async (q) => {
		const { GPUEngine } = await import('/src/lib/gpu/engine.ts');
		const { ORIGINAL_SETTINGS } = await import('/src/lib/gpu/machine.ts');
		const { HEAD_WORDS, HEAD_WAITING } = await import('/src/lib/gpu/free.ts');
		const canvas = document.createElement('canvas');
		canvas.width = 64;
		canvas.height = 64;
		document.body.append(canvas);
		const engine = new GPUEngine(6);
		if (!(await engine.init(canvas))) throw new Error('No WebGPU');
		const settings = {
			...ORIGINAL_SETTINGS,
			width: q.tiles,
			height: q.tiles,
			seed: q.seed,
			interaction: 'free',
			heads: q.heads,
			headDirections: q.directions,
			haltRule: q.halt,
			mutationExponent: q.mutation
		};
		await engine.configure(settings, { kind: 'random' });
		const w = q.tiles * 4;
		const toSheet = (soup) => {
			const sheet = new Uint8Array(w * w);
			for (let y = 0; y < w; y++)
				for (let x = 0; x < w; x++)
					sheet[y * w + x] = soup[((y >> 2) * (w >> 2) + (x >> 2)) * 16 + (y & 3) * 4 + (x & 3)];
			return sheet;
		};
		const b64 = (u8) => {
			let t = '';
			for (let i = 0; i < u8.length; i += 8192) t += String.fromCharCode.apply(null, u8.subarray(i, i + 8192));
			return btoa(t);
		};
		const marks = [0, 100, 500, 2000, q.ticks].filter((m, i, a) => m <= q.ticks && a.indexOf(m) === i);
		const rows = [];
		let t = 0;
		const t0 = performance.now();
		let previous = null;
		for (const m of marks) {
			for (; t < m; t++) {
				engine.simulateStep();
				if (t % 100 === 99) await engine.waitForIdle();
			}
			await engine.waitForIdle();
			const seconds = (performance.now() - t0) / 1000;
			const sheet = toSheet(await engine.readSoupData());
			const hd = await engine.readHeads();
			let waiting = 0;
			for (let k = 0; k < hd.length / HEAD_WORDS; k++) waiting += hd[k * HEAD_WORDS + HEAD_WAITING];
			const hist = new Float64Array(256);
			for (const v of sheet) hist[v]++;
			let entropy = 0;
			for (const c of hist) if (c) entropy -= (c / sheet.length) * Math.log2(c / sheet.length);
			let changed = 0;
			if (previous) for (let i = 0; i < sheet.length; i++) if (sheet[i] !== previous[i]) changed++;
			previous = sheet;
			rows.push({
				tick: m,
				seconds,
				zero: hist[0] / sheet.length,
				halts: hist[0x76] / sheet.length,
				entropy,
				waitingHeads: waiting / (hd.length / HEAD_WORDS),
				changedSinceLastMark: changed / sheet.length,
				sheet: b64(sheet)
			});
		}
		return { settings, heads: (await engine.readHeads()).length / HEAD_WORDS, rows };
	},
	{ tiles: +tiles, heads: +heads, directions, halt, mutation: +mutation, ticks: +ticks, seed: +seed }
);
for (const r of result.rows) {
	await writeFile(path.join(out, `sheet_${r.tick}.u8`), Buffer.from(r.sheet, 'base64'));
	delete r.sheet;
	console.log(JSON.stringify(r));
}
await writeFile(path.join(out, 'look.json'), JSON.stringify(result, null, 1));
await browser.close();

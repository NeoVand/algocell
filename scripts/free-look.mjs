// A look at the free-running world: run it headless, save the sheet at several times with summary
// numbers. Needs the dev server on 5176.
//   node scripts/free-look.mjs <out dir> key=value ...
// Keys (defaults): tiles=128 (side, in 16-byte tiles) heads=0.25 (per tile) halt=restart turnover=0
//   facing=any|written|bytes|rows|two-axes|same (any: a landing processor faces one of eight ways at
//   random; written: the way the byte it lands on was written; bytes: towards its largest neighbour;
//   same: all read rightwards; rows: rightwards or leftwards, stack above or below; two-axes:
//   rightwards or downwards)
//   reach=256 (side of the square a processor can address: 16, 32, 64, 128 or 256)
//   mutation=4 (exponent; 31 is off) ticks=20000 seed=6 start=random|halt-with-pusher|zero
//   marks=0,1000,5000,20000 (ticks at which to save)
//   start=pattern fill=01c5 block=c501c50118f8 : the sheet tiled with `fill` (hex), and a 32 x 16-byte
//   block of `block` in the middle. Each row is tiled from its left end.
//   start=checker : a chessboard of 01 and c5 (every row and every column reads `01 c5 01 c5 ...`).
//   patch=checker|rows side=32 : after any start, a square in the middle holding the chessboard, or
//   rows of `01 c5` all in step. With start=random this is an invasion test of the waiting state.
// Writes sheet_<tick>.u8 (row-major bytes), grain_<tick>.u8 (the facing of each byte's last writer),
// heads_<tick>.u32 and look.json.
import { chromium } from '@playwright/test';
import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

const [out, ...pairs] = process.argv.slice(2);
if (!out) throw new Error('Give an output directory.');
const q = {
	tiles: 128,
	heads: 0.25,
	halt: 'restart',
	turnover: 0,
	facing: 'any',
	reach: 256,
	mutation: 4,
	ticks: 20000,
	seed: 6,
	start: 'random',
	marks: '',
	fill: '76',
	block: '',
	patch: '',
	side: 32
};
for (const pair of pairs) {
	const [k, v] = pair.split('=');
	if (!(k in q)) throw new Error(`Unknown key ${k}`);
	q[k] = typeof q[k] === 'number' ? Number(v) : v;
}
q.marks = (q.marks ? q.marks.split(',').map(Number) : [0, q.ticks / 20, q.ticks / 4, q.ticks]).map(
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
	const { HEAD_WORDS, HEAD_WAITING, HEAD_AGE, HEAD_RUNS, HEAD_LAST_RUN, HEAD_FACING } =
		await import('/src/lib/gpu/free.ts');
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
		haltRule: q.halt,
		turnover: q.turnover,
		facing: q.facing,
		reach: q.reach,
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
	const fromSheet = (sheet) => {
		const soup = new Uint8Array(w * w);
		for (let y = 0; y < w; y++)
			for (let x = 0; x < w; x++)
				soup[((y >> 2) * (w >> 2) + (x >> 2)) * 16 + (y & 3) * 4 + (x & 3)] = sheet[y * w + x];
		return soup;
	};
	if (q.start === 'pattern') {
		const bytes = (hex) => hex.match(/../g).map((b) => parseInt(b, 16));
		const fill = bytes(q.fill);
		const block = q.block ? bytes(q.block) : [];
		const sheet = new Uint8Array(w * w);
		for (let y = 0; y < w; y++)
			for (let x = 0; x < w; x++) sheet[y * w + x] = fill[x % fill.length];
		const c = w >> 1;
		if (block.length)
			for (let y = c - 8; y < c + 8; y++)
				for (let x = c - 16; x < c + 16; x++) sheet[y * w + x] = block[(x - c + 16) % block.length];
		engine.writeSoupData(fromSheet(sheet));
	} else if (q.start === 'checker') {
		const sheet = new Uint8Array(w * w);
		for (let y = 0; y < w; y++)
			for (let x = 0; x < w; x++) sheet[y * w + x] = (x + y) & 1 ? 0xc5 : 0x01;
		engine.writeSoupData(fromSheet(sheet));
	} else if (q.start !== 'random') {
		const sheet = new Uint8Array(w * w).fill(q.start === 'zero' ? 0 : 0x76);
		if (q.start === 'halt-with-pusher') {
			const c = w >> 1;
			for (let y = c - 8; y < c + 8; y++)
				for (let x = c - 16; x < c + 16; x++) sheet[y * w + x] = x & 1 ? 0xc5 : 0x01;
		}
		engine.writeSoupData(fromSheet(sheet));
	}
	if (q.patch) {
		const sheet = toSheet(await engine.readSoupData());
		const c = w >> 1;
		const half = q.side >> 1;
		for (let y = c - half; y < c + half; y++)
			for (let x = c - half; x < c + half; x++)
				sheet[y * w + x] = (q.patch === 'checker' ? x + y : x) & 1 ? 0xc5 : 0x01;
		engine.writeSoupData(fromSheet(sheet));
		// The patch counts as written by processors reading rightwards with the stack above.
		const grain = new Uint32Array(await engine.readGrain());
		for (let y = c - half; y < c + half; y++)
			for (let x = c - half; x < c + half; x++) grain[y * w + x] = 0;
		engine.writeGrain(grain);
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
	let previous = null;
	for (const m of q.marks) {
		while (t < m) {
			const n = Math.min(50, m - t);
			engine.simulateTicks(n);
			t += n;
			if (t % 500 < 50) await engine.waitForIdle();
		}
		await engine.waitForIdle();
		const seconds = (performance.now() - t0) / 1000;
		const sheet = toSheet(await engine.readSoupData());
		const hd = await engine.readHeads();
		const count = hd.length / HEAD_WORDS;
		let waiting = 0;
		let runs = 0;
		const ages = [];
		const last = [];
		const facings = new Array(8).fill(0);
		for (let k = 0; k < count; k++) {
			facings[hd[k * HEAD_WORDS + HEAD_FACING]]++;
			waiting += hd[k * HEAD_WORDS + HEAD_WAITING];
			runs += hd[k * HEAD_WORDS + HEAD_RUNS];
			ages.push(hd[k * HEAD_WORDS + HEAD_AGE]);
			if (hd[k * HEAD_WORDS + HEAD_RUNS]) last.push(hd[k * HEAD_WORDS + HEAD_LAST_RUN]);
		}
		ages.sort((a, b) => a - b);
		last.sort((a, b) => a - b);
		const at = (a, f) => (a.length ? a[Math.min(a.length - 1, Math.floor(a.length * f))] : null);
		const hist = new Float64Array(256);
		for (const v of sheet) hist[v]++;
		let entropy = 0;
		for (const c of hist) if (c) entropy -= (c / sheet.length) * Math.log2(c / sheet.length);
		// How often a byte equals the byte k places to its right, and k rows below.
		const same = (dx, dy) => {
			let n = 0;
			for (let y = 0; y < w; y++)
				for (let x = 0; x < w; x++)
					if (sheet[y * w + x] === sheet[((y + dy) % w) * w + ((x + dx) % w)]) n++;
			return +(n / sheet.length).toFixed(4);
		};
		const chance = hist.reduce((s, c) => s + (c / sheet.length) ** 2, 0);
		const right = [];
		for (let k = 1; k <= 32; k++) right.push(same(k, 0));
		// The same at equal distance along the two diagonals (k steps of one row and one column).
		const diagonal = [];
		for (let k = 1; k <= 8; k++) diagonal.push(+((same(k, k) + same(w - k, k)) / 2).toFixed(4));
		const grain = await engine.readGrain();
		const grainBytes = Uint8Array.from(grain);
		const grainCounts = new Array(8).fill(0);
		for (const g of grainBytes) grainCounts[g]++;
		const down = [];
		for (let k = 1; k <= 32; k++) down.push(same(0, k));
		const pairsSeen = new Map();
		for (let i = 0; i + 1 < sheet.length; i++) {
			const key = (sheet[i] << 8) | sheet[i + 1];
			pairsSeen.set(key, (pairsSeen.get(key) ?? 0) + 1);
		}
		const topPairs = [...pairsSeen.entries()]
			.sort((a, b) => b[1] - a[1])
			.slice(0, 6)
			.map(([key, n]) => [key.toString(16).padStart(4, '0'), +(n / sheet.length).toFixed(4)]);
		const topBytes = [...hist.entries()]
			.sort((a, b) => b[1] - a[1])
			.slice(0, 6)
			.map(([v, n]) => [v.toString(16).padStart(2, '0'), +(n / sheet.length).toFixed(4)]);
		let changed = 0;
		if (previous) for (let i = 0; i < sheet.length; i++) if (sheet[i] !== previous[i]) changed++;
		previous = sheet;
		rows.push({
			tick: m,
			seconds: +seconds.toFixed(2),
			zero: +(hist[0] / sheet.length).toFixed(4),
			haltBytes: +(hist[0x76] / sheet.length).toFixed(4),
			entropy: +entropy.toFixed(3),
			waitingHeads: +(waiting / count).toFixed(4),
			runsFinished: runs,
			meanRunLength: runs ? +((m * count) / runs).toFixed(1) : null,
			ageMedian: at(ages, 0.5),
			ageNinetieth: at(ages, 0.9),
			headsNeverRestarted: +(ages.filter((a) => a === m).length / count).toFixed(4),
			lastRunMedian: at(last, 0.5),
			lastRunNinetieth: at(last, 0.9),
			sameByChance: +chance.toFixed(4),
			sameRight: right,
			sameDown: down,
			facings,
			sameDiagonal: diagonal,
			grainShares: grainCounts.map((n) => +(n / grainBytes.length).toFixed(4)),
			topBytes,
			share: Object.fromEntries(
				['01', 'c5', '11', 'd5', '18', 'f8'].map((b) => [
					b,
					+(hist[parseInt(b, 16)] / sheet.length).toFixed(4)
				])
			),
			topPairs,
			changedSinceLastMark: +(changed / sheet.length).toFixed(4),
			sheet: b64(sheet),
			grain: b64(grainBytes),
			heads: b64(new Uint8Array(hd.buffer))
		});
	}
	const heads = (await engine.readHeads()).length / HEAD_WORDS;
	engine.destroy();
	return { q, sheetSide: w, heads, rows };
}, q);
for (const r of result.rows) {
	await writeFile(path.join(out, `sheet_${r.tick}.u8`), Buffer.from(r.sheet, 'base64'));
	await writeFile(path.join(out, `grain_${r.tick}.u8`), Buffer.from(r.grain, 'base64'));
	delete r.grain;
	await writeFile(path.join(out, `heads_${r.tick}.u32`), Buffer.from(r.heads, 'base64'));
	delete r.sheet;
	delete r.heads;
	console.log(JSON.stringify(r));
}
await writeFile(path.join(out, 'look.json'), JSON.stringify(result, null, 1));
await browser.close();

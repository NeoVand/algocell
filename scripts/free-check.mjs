// Checks of the free-running world on the real GPU pipeline. Needs the dev server:
//   npm run dev -- --host 127.0.0.1 --port 5176 --strictPort
//   node scripts/free-check.mjs
// 1. The GPU world matches the TypeScript reference (src/lib/dev/freeReference.ts, with the
//    third-party z80-emulator executing instructions) tick for tick: sheet bytes, registers and
//    landing sites, on small crowded sheets where heads often want the same byte. All three HALT
//    rules, with and without the restart chance and mutation, on a sheet narrower than a head's
//    window and on one wider than it whose sides are not powers of two.
// 2. Two GPU runs of one seed give the same sheet, with thousands of heads and mutation on.
// 3. A known replicator of the pair worlds (the pusher, `01 c5`) spreads on the sheet.
import { chromium } from '@playwright/test';

const url = process.argv[2] ?? 'http://127.0.0.1:5176/demo/playwright';
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('page error:', e.message));
await page.goto(url);
const report = await page.evaluate(async () => {
	const { GPUEngine } = await import('/src/lib/gpu/engine.ts');
	const { ORIGINAL_SETTINGS } = await import('/src/lib/gpu/machine.ts');
	const { freeLayout, mutationChance, HEAD_WORDS, HEAD_ORIGIN_X, HEAD_ORIGIN_Y, HEAD_RUNS } =
		await import('/src/lib/gpu/free.ts');
	const { createReference } = await import('/src/lib/dev/freeReference.ts');
	const canvas = document.createElement('canvas');
	canvas.width = 64;
	canvas.height = 64;
	document.body.append(canvas);
	const engine = new GPUEngine(6);
	if (!(await engine.init(canvas))) throw new Error('No WebGPU');
	const out = [];
	const check = (name, pass, detail) => out.push({ name, pass, detail });
	const base = {
		...ORIGINAL_SETTINGS,
		compat: '0.3',
		variant: 'zilog-nmos',
		stack: 'first-push',
		dialMode: 'counter',
		interaction: 'free',
		boundary: 'reflect'
	};
	// Lattice storage (tiles of 4 x 4 bytes) to sheet order (row-major bytes) and back.
	const toSheet = (soup, w, h) => {
		const sheet = new Uint8Array(w * h);
		for (let y = 0; y < h; y++)
			for (let x = 0; x < w; x++)
				sheet[y * w + x] = soup[((y >> 2) * (w >> 2) + (x >> 2)) * 16 + (y & 3) * 4 + (x & 3)];
		return sheet;
	};
	const fromSheet = (sheet, w, h) => {
		const soup = new Uint8Array(w * h);
		for (let y = 0; y < h; y++)
			for (let x = 0; x < w; x++)
				soup[((y >> 2) * (w >> 2) + (x >> 2)) * 16 + (y & 3) * 4 + (x & 3)] = sheet[y * w + x];
		return soup;
	};

	const cases = [];
	for (const haltRule of ['restart', 'wait', 'pass'])
		for (const seed of [3, 4, 5])
			cases.push({ haltRule, seed, tilesW: 8, tilesH: 8, heads: 1, turnover: 0, exponent: 31 });
	// Restart chance and mutation on; a sheet wider than a head's window (280 bytes) and not a power of two.
	for (const seed of [6, 7, 8])
		cases.push({
			haltRule: 'restart',
			seed,
			tilesW: 70,
			tilesH: 9,
			heads: 0.1,
			turnover: 1 / 64,
			exponent: 0
		});
	for (const c of cases) {
		const settings = {
			...base,
			width: c.tilesW,
			height: c.tilesH,
			seed: c.seed,
			heads: c.heads,
			haltRule: c.haltRule,
			turnover: c.turnover,
			mutationExponent: c.exponent
		};
		const w = c.tilesW * 4;
		const h = c.tilesH * 4;
		await engine.configure(settings, { kind: 'random' });
		// Start from the engine's random sheet with the index prefixes taken out (see the reference).
		const start = toSheet(await engine.readSoupData(), w, h);
		for (let i = 0; i < start.length; i++) if (start[i] === 0xdd || start[i] === 0xfd) start[i] = 0;
		engine.writeSoupData(fromSheet(start, w, h));
		const heads = await engine.readHeads();
		const count = heads.length / HEAD_WORDS;
		const layout = freeLayout(settings);
		const reference = createReference(w, h, start, heads, {
			haltRule: c.haltRule,
			turnover: c.turnover,
			mutation: c.exponent >= 31 ? 0 : mutationChance(c.exponent)
		});
		const ticks = 400;
		const chunk = 5;
		let agreed = 0;
		let mismatch = null;
		for (let t = chunk; t <= ticks && !mismatch; t += chunk) {
			engine.simulateTicks(chunk);
			for (let i = 0; i < chunk; i++) reference.tick();
			if (reference.firstIndexPrefixTick !== null) break;
			await engine.waitForIdle();
			const sheet = toSheet(await engine.readSoupData(), w, h);
			const gpuHeads = await engine.readHeads();
			let bytes = 0;
			for (let i = 0; i < sheet.length; i++) if (sheet[i] !== reference.sheet[i]) bytes++;
			let registers = 0;
			let example = null;
			for (let k = 0; k < count; k++) {
				const want = {
					...reference.registers(k),
					ox: reference.origin(k).x,
					oy: reference.origin(k).y
				};
				const got = {
					...Object.fromEntries(
						['a', 'f', 'b', 'c', 'd', 'e', 'h', 'l', 'ix', 'iy', 'sp', 'pc'].map((n) => [
							n,
							gpuHeads[k * HEAD_WORDS + layout.state[n]]
						])
					),
					ox: gpuHeads[k * HEAD_WORDS + HEAD_ORIGIN_X],
					oy: gpuHeads[k * HEAD_WORDS + HEAD_ORIGIN_Y]
				};
				// Bits 3 and 5 of F are undocumented and the two cores model them differently (Zilion's
				// zilog-nmos variant carries the Q register, z80-emulator does not), so F is compared on its
				// documented bits. A difference that mattered would also show in the sheet.
				want.f &= 0xd7;
				got.f &= 0xd7;
				for (const n of Object.keys(want))
					if (want[n] !== got[n]) {
						registers++;
						example ??= { head: k, register: n, want: want[n], got: got[n] };
					}
			}
			let runs = 0;
			for (let k = 0; k < count; k++) runs += gpuHeads[k * HEAD_WORDS + HEAD_RUNS];
			if (runs !== reference.restarts)
				mismatch = { tick: t, restartsWanted: reference.restarts, restartsGot: runs };
			else if (bytes || registers) mismatch = { tick: t, bytes, registers, example };
			else agreed = t;
		}
		let changed = 0;
		for (let i = 0; i < start.length; i++) if (start[i] !== reference.sheet[i]) changed++;
		const exercised =
			reference.contested > 0 &&
			(c.haltRule !== 'restart' || reference.restarts > 0) &&
			(c.exponent >= 31 || reference.mutations > 0);
		check(
			`GPU world equals the reference: HALT ${c.haltRule}, ${w} x ${h} bytes, restart chance ${c.turnover}, mutation ${c.exponent >= 31 ? 'off' : 'on'}, seed ${c.seed}`,
			!mismatch && agreed >= 40 && exercised,
			{
				heads: count,
				ticksAgreed: agreed,
				stoppedAtIndexPrefixTick: reference.firstIndexPrefixTick,
				sheetBytesChanged: changed,
				contestedBytes: reference.contested,
				haltsExecuted: reference.halts,
				restarts: reference.restarts,
				mutations: reference.mutations,
				mismatch
			}
		);
	}

	// 2. Repeatability with many heads, the restart chance and mutation on.
	const big = {
		...base,
		width: 64,
		height: 64,
		seed: 9,
		heads: 1,
		haltRule: 'restart',
		turnover: 1 / 256,
		mutationExponent: 2
	};
	const run = async () => {
		await engine.configure(big, { kind: 'random' });
		for (let t = 0; t < 10; t++) engine.simulateTicks(50);
		await engine.waitForIdle();
		return engine.readSoupData();
	};
	const first = await engine.configure(big, { kind: 'random' }).then(() => engine.readSoupData());
	const a = await run();
	const b = await run();
	let differ = 0;
	let moved = 0;
	for (let i = 0; i < a.length; i++) {
		if (a[i] !== b[i]) differ++;
		if (a[i] !== first[i]) moved++;
	}
	check(
		'two runs of one seed give the same sheet (4,096 heads, 500 ticks, restart chance and mutation on)',
		differ === 0 && moved > 1000,
		{ differingBytes: differ, bytesChangedByTheRun: moved }
	);

	// 3. The pusher. A sheet of HALT bytes (a landing head lifts off again at once) with one block of
	// `01 c5`: LD BC,0x01c5 then PUSH BC, which writes `c5 01` leftwards along the row above.
	{
		const tiles = 64;
		const w = tiles * 4;
		const settings = {
			...base,
			width: tiles,
			height: tiles,
			seed: 11,
			heads: 0.25,
			haltRule: 'restart',
			turnover: 0,
			mutationExponent: 31
		};
		await engine.configure(settings, { kind: 'random' });
		const sheet = new Uint8Array(w * w).fill(0x76);
		for (let y = 120; y < 136; y++)
			for (let x = 112; x < 144; x++) sheet[y * w + x] = x & 1 ? 0xc5 : 0x01;
		engine.writeSoupData(fromSheet(sheet, w, w));
		const pusherBytes = (s) => {
			let n = 0;
			for (let i = 0; i < s.length; i++) if (s[i] === 0x01 || s[i] === 0xc5) n++;
			return n;
		};
		const before = pusherBytes(sheet);
		const series = [];
		for (let k = 0; k < 4; k++) {
			for (let t = 0; t < 10; t++) engine.simulateTicks(50);
			await engine.waitForIdle();
			series.push(pusherBytes(toSheet(await engine.readSoupData(), w, w)));
		}
		check(
			'the pusher `01 c5` spreads from one block on a sheet of HALT bytes (no mutation)',
			series[3] > 4 * before && series.every((n, i) => n > (i ? series[i - 1] : before)),
			{ sheetBytes: w * w, pusherBytesAtStart: before, after500_1000_1500_2000Ticks: series }
		);
	}
	engine.destroy();
	return out;
});
let failed = 0;
for (const r of report) {
	if (!r.pass) failed++;
	console.log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.name}  ${JSON.stringify(r.detail)}`);
}
await browser.close();
process.exit(failed ? 1 : 0);

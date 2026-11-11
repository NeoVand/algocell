// Checks of the free-running world on the real GPU pipeline. Needs the dev server:
//   npm run dev -- --host 127.0.0.1 --port 5176 --strictPort
//   node scripts/free-check.mjs
// 1. The GPU world matches the TypeScript reference (src/lib/dev/freeReference.ts, with the
//    third-party z80-emulator executing instructions) tick for tick: sheet bytes, registers and
//    landing sites, on small crowded sheets where heads often want the same byte. All three HALT
//    rules, with and without the restart chance and mutation, on a sheet narrower than a head's
//    window and on one wider than it whose sides are not powers of two.
//    Heads face any of the eight ways; two runs keep every head in the same frame. A comparison ends
//    at the first tick on which the two Z80s are known to differ (DD/FD prefix chains; H and P/V on a
//    repeating block I/O step).
// 2. Two GPU runs of one seed give the same sheet, with thousands of heads and mutation on.
// 3. The rule has no preferred direction: turning or mirroring the whole world (sheet, landing sites
//    and facings) by any of the seven non-trivial symmetries of the square and running it gives the
//    turned or mirrored result, byte for byte.
// 4. Landing heads face the eight ways equally often; the `bytes` rule faces the largest neighbour
//    and favours no direction in a tie.
// 5. A known replicator of the pair worlds (the pusher, `01 c5`) spreads on the sheet.
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
	const {
		freeLayout,
		mutationChance,
		faceOffset,
		facingFromNeighbours,
		FACING_UNDECIDED,
		HEAD_WORDS,
		HEAD_ORIGIN_X,
		HEAD_ORIGIN_Y,
		HEAD_RUNS,
		HEAD_FACING
	} = await import('/src/lib/gpu/free.ts');
	const { createReference, mix32 } = await import('/src/lib/dev/freeReference.ts');
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
	// Every head in the same frame (reading rightwards, stack on the row above).
	for (const seed of [3, 4])
		cases.push({
			haltRule: 'restart',
			facing: 'same',
			seed,
			tilesW: 8,
			tilesH: 8,
			heads: 1,
			turnover: 0,
			exponent: 31
		});
	// The sheet sets the facing: the grain of the landing byte (`written`), or the neighbouring bytes
	// (`bytes`). Heads start undecided and take their facing from the rule.
	for (const facing of ['written', 'bytes'])
		for (const seed of [3, 4])
			cases.push({
				haltRule: 'restart',
				facing,
				seed,
				tilesW: 8,
				tilesH: 8,
				heads: 1,
				turnover: 0,
				exponent: 31
			});
	// Restart chance and mutation on; a sheet wider than a head's window (280 bytes) and not a power of two.
	for (const [facing, seeds] of [
		['any', [6, 7, 8]],
		['written', [7, 8]],
		['bytes', [7, 8]]
	])
		for (const seed of seeds)
			cases.push({
				haltRule: 'restart',
				facing,
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
			facing: c.facing ?? 'any',
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
		const reference = createReference(
			w,
			h,
			start,
			heads,
			{
				haltRule: c.haltRule,
				turnover: c.turnover,
				mutation: c.exponent >= 31 ? 0 : mutationChance(c.exponent),
				facing: settings.facing
			},
			await engine.readGrain()
		);
		const facingsAtStart = new Set();
		for (let k = 0; k < count; k++) facingsAtStart.add(heads[k * HEAD_WORDS + HEAD_FACING]);
		const ticks = 400;
		const chunk = 5;
		let agreed = 0;
		let mismatch = null;
		for (let t = chunk; t <= ticks && !mismatch; t += chunk) {
			engine.simulateTicks(chunk);
			for (let i = 0; i < chunk; i++) reference.tick();
			if (reference.firstIndexPrefixTick !== null || reference.firstRepeatIoTick !== null) break;
			await engine.waitForIdle();
			const sheet = toSheet(await engine.readSoupData(), w, h);
			const gpuHeads = await engine.readHeads();
			const gpuGrain = await engine.readGrain();
			let bytes = 0;
			let grains = 0;
			for (let i = 0; i < sheet.length; i++) {
				if (sheet[i] !== reference.sheet[i]) bytes++;
				if (gpuGrain[i] !== reference.grain[i]) grains++;
			}
			let registers = 0;
			let example = null;
			for (let k = 0; k < count; k++) {
				const want = {
					...reference.registers(k),
					ox: reference.origin(k).x,
					oy: reference.origin(k).y,
					facing: reference.origin(k).facing
				};
				const got = {
					...Object.fromEntries(
						['a', 'f', 'b', 'c', 'd', 'e', 'h', 'l', 'ix', 'iy', 'sp', 'pc'].map((n) => [
							n,
							gpuHeads[k * HEAD_WORDS + layout.state[n]]
						])
					),
					ox: gpuHeads[k * HEAD_WORDS + HEAD_ORIGIN_X],
					oy: gpuHeads[k * HEAD_WORDS + HEAD_ORIGIN_Y],
					facing: gpuHeads[k * HEAD_WORDS + HEAD_FACING]
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
			else if (bytes || grains || registers)
				mismatch = { tick: t, bytes, grains, registers, example };
			else agreed = t;
		}
		let changed = 0;
		for (let i = 0; i < start.length; i++) if (start[i] !== reference.sheet[i]) changed++;
		const exercised =
			reference.contested > 0 &&
			(c.haltRule !== 'restart' || reference.restarts > 0) &&
			(c.exponent >= 31 || reference.mutations > 0) &&
			(settings.facing === 'any'
				? facingsAtStart.size === 8
				: settings.facing === 'same'
					? facingsAtStart.size === 1
					: facingsAtStart.has(FACING_UNDECIDED) && facingsAtStart.size === 1);
		check(
			`GPU world equals the reference: facing ${settings.facing}, HALT ${c.haltRule}, ${w} x ${h} bytes, restart chance ${c.turnover}, mutation ${c.exponent >= 31 ? 'off' : 'on'}, seed ${c.seed}`,
			!mismatch && agreed >= 10,
			{
				heads: count,
				facingsAtStart: facingsAtStart.size,
				ticksAgreed: agreed,
				stoppedAtIndexPrefixTick: reference.firstIndexPrefixTick,
				stoppedAtRepeatingBlockIoTick: reference.firstRepeatIoTick,
				everyFeatureExercised: exercised,
				sheetBytesChanged: changed,
				contestedBytes: reference.contested,
				haltsExecuted: reference.halts,
				restarts: reference.restarts,
				mutations: reference.mutations,
				mismatch
			}
		);
	}

	// A comparison ends where the two Z80s are known to differ (see the reference), which can be early.
	// Besides "no mismatch in any run", ask that enough runs were long and exercised every feature.
	const long = out.filter(
		(r) => r.pass && r.detail.ticksAgreed >= 40 && r.detail.everyFeatureExercised
	).length;
	check(
		`at least two thirds of the ${cases.length} comparisons lasted 40 ticks or more and exercised contested bytes, restarts and mutation where switched on`,
		long >= (2 * cases.length) / 3,
		{ longComparisons: long, ticksAgreedInAll: out.reduce((s, r) => s + r.detail.ticksAgreed, 0) }
	);

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

	// 3. No preferred direction. With the `pass` rule and no restart chance or mutation a tick uses no
	// random numbers, so the statement is exact: for a symmetry g of the square, running the world g
	// (sheet, landing sites and facings all turned or mirrored) gives g of the result of running the
	// world itself. Byte values and register contents are not touched by g.
	{
		const tiles = 24;
		const w = tiles * 4; // 96 x 96 bytes: smaller than a head's window, not a power of two
		const settings = {
			...base,
			width: tiles,
			height: tiles,
			seed: 21,
			heads: 1,
			haltRule: 'pass',
			turnover: 0,
			facing: 'any',
			mutationExponent: 31
		};
		const ticks = 600;
		const wrap = (v) => ((v % w) + w) % w;
		const moved = (g, x, y) => {
			const [dx, dy] = faceOffset(g, x, y);
			return [wrap(dx), wrap(dy)];
		};
		// The facing f' with g(f(v)) = f'(v) for every offset v.
		const compose = (g, f) => {
			const want = [faceOffset(g, ...faceOffset(f, 1, 0)), faceOffset(g, ...faceOffset(f, 0, 1))];
			for (let k = 0; k < 8; k++) {
				const got = [faceOffset(k, 1, 0), faceOffset(k, 0, 1)];
				if (String(got) === String(want)) return k;
			}
			throw new Error('facings do not close under composition');
		};
		await engine.configure(settings, { kind: 'random' });
		const startSheet = toSheet(await engine.readSoupData(), w, w);
		const startHeads = await engine.readHeads();
		const startGrain = await engine.readGrain();
		const count = startHeads.length / HEAD_WORDS;
		const runFrom = async (sheet, heads, grain) => {
			await engine.configure(settings, { kind: 'random' });
			engine.writeSoupData(fromSheet(sheet, w, w));
			engine.writeHeads(heads);
			engine.writeGrain(grain);
			for (let t = 0; t < ticks; t += 50) engine.simulateTicks(50);
			await engine.waitForIdle();
			return {
				sheet: toSheet(await engine.readSoupData(), w, w),
				heads: await engine.readHeads(),
				grain: await engine.readGrain()
			};
		};
		const plain = await runFrom(startSheet, startHeads, startGrain);
		let grainChanged = 0;
		for (let i = 0; i < startGrain.length; i++)
			if (startGrain[i] !== plain.grain[i]) grainChanged++;
		let changed = 0;
		for (let i = 0; i < startSheet.length; i++) if (startSheet[i] !== plain.sheet[i]) changed++;
		const stateWords = freeLayout(settings).stateWords;
		const results = [];
		for (let g = 1; g < 8; g++) {
			const sheet = new Uint8Array(w * w);
			const grain = new Uint32Array(w * w);
			for (let y = 0; y < w; y++)
				for (let x = 0; x < w; x++) {
					const [nx, ny] = moved(g, x, y);
					sheet[ny * w + nx] = startSheet[y * w + x];
					grain[ny * w + nx] = compose(g, startGrain[y * w + x]);
				}
			const heads = new Uint32Array(startHeads);
			for (let k = 0; k < count; k++) {
				const o = k * HEAD_WORDS;
				const [nx, ny] = moved(g, startHeads[o + HEAD_ORIGIN_X], startHeads[o + HEAD_ORIGIN_Y]);
				heads[o + HEAD_ORIGIN_X] = nx;
				heads[o + HEAD_ORIGIN_Y] = ny;
				heads[o + HEAD_FACING] = compose(g, startHeads[o + HEAD_FACING]);
			}
			const turned = await runFrom(sheet, heads, grain);
			let bytes = 0;
			let grains = 0;
			for (let y = 0; y < w; y++)
				for (let x = 0; x < w; x++) {
					const [nx, ny] = moved(g, x, y);
					if (turned.sheet[ny * w + nx] !== plain.sheet[y * w + x]) bytes++;
					if (turned.grain[ny * w + nx] !== compose(g, plain.grain[y * w + x])) grains++;
				}
			let registers = 0;
			for (let k = 0; k < count; k++)
				for (let i = 0; i < stateWords; i++)
					if (turned.heads[k * HEAD_WORDS + i] !== plain.heads[k * HEAD_WORDS + i]) registers++;
			results.push({
				symmetry: g,
				differingBytes: bytes,
				differingGrains: grains,
				differingRegisterWords: registers
			});
		}
		check(
			`turning or mirroring the world commutes with running it (${w} x ${w} bytes, ${count} heads, ${ticks} ticks, all seven symmetries)`,
			changed > 1000 &&
				grainChanged > 1000 &&
				results.every(
					(r) => r.differingBytes === 0 && r.differingGrains === 0 && r.differingRegisterWords === 0
				),
			{ bytesChangedByTheRun: changed, grainsChangedByTheRun: grainChanged, results }
		);
	}

	// 4. Landing heads face the eight ways equally often: after many restarts on a sheet of HALT
	// bytes (every head lifts off every tick), count the facings.
	{
		const tiles = 32;
		const w = tiles * 4;
		const settings = {
			...base,
			width: tiles,
			height: tiles,
			seed: 31,
			heads: 4,
			haltRule: 'restart',
			turnover: 0,
			facing: 'any',
			mutationExponent: 31
		};
		await engine.configure(settings, { kind: 'random' });
		engine.writeSoupData(fromSheet(new Uint8Array(w * w).fill(0x76), w, w));
		const counts = new Array(8).fill(0);
		const columns = new Array(4).fill(0);
		const rows = new Array(4).fill(0);
		let total = 0;
		for (let round = 0; round < 25; round++) {
			engine.simulateTicks(7);
			await engine.waitForIdle();
			const heads = await engine.readHeads();
			for (let k = 0; k < heads.length / HEAD_WORDS; k++) {
				counts[heads[k * HEAD_WORDS + HEAD_FACING]]++;
				columns[Math.floor((heads[k * HEAD_WORDS + HEAD_ORIGIN_X] * 4) / w)]++;
				rows[Math.floor((heads[k * HEAD_WORDS + HEAD_ORIGIN_Y] * 4) / w)]++;
				total++;
			}
		}
		// Chi-squared against equal shares; 7 and 3 degrees of freedom. 0.1% points: 24.3 and 16.3.
		const chi = (c) => c.reduce((s, n) => s + (n - total / c.length) ** 2 / (total / c.length), 0);
		check(
			'landing heads face the eight ways, and land in the four quarters of each axis, equally often',
			chi(counts) < 24.3 && chi(columns) < 16.3 && chi(rows) < 16.3,
			{
				landings: total,
				facings: counts,
				chiSquaredFacings: +chi(counts).toFixed(2),
				chiSquaredColumns: +chi(columns).toFixed(2),
				chiSquaredRows: +chi(rows).toFixed(2)
			}
		);
	}

	// The `bytes` rule by itself: forward points at the largest neighbour whenever there is one, and
	// when all four are equal the eight facings come out equally often.
	{
		const unit = [
			[1, 0],
			[-1, 0],
			[0, 1],
			[0, -1]
		]; // east, west, south, north
		let wrongForward = 0;
		let draws = 0;
		const tied = new Array(8).fill(0);
		let state = 12345;
		const next = () => (state = mix32((state + 0x9e3779b9) >>> 0));
		for (let i = 0; i < 80000; i++) {
			const v = [next() & 0xff, next() & 0xff, next() & 0xff, next() & 0xff];
			const top = Math.max(...v);
			const rnd = next();
			if (v.filter((x) => x === top).length === 1) {
				const facing = facingFromNeighbours(v[0], v[1], v[2], v[3], rnd, mix32);
				const [dx, dy] = faceOffset(facing, 1, 0);
				const want = unit[v.indexOf(top)];
				if (dx !== want[0] || dy !== want[1]) wrongForward++;
				draws++;
			}
			tied[facingFromNeighbours(7, 7, 7, 7, rnd, mix32)]++;
		}
		const chi = tied.reduce((s, n) => s + (n - 10000) ** 2 / 10000, 0);
		check(
			'the `bytes` rule faces the largest neighbouring byte, and with four equal neighbours the eight facings are equally likely',
			wrongForward === 0 && draws > 70000 && chi < 24.3,
			{
				drawsWithOneLargest: draws,
				wrongForward,
				facingsWhenTied: tied,
				chiSquared: +chi.toFixed(2)
			}
		);
	}

	// 5. The pusher. A sheet of HALT bytes (a landing head lifts off again at once) with one block of
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

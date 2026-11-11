// Checks of the free-running world on the real GPU pipeline. Needs the dev server:
//   npm run dev -- --host 127.0.0.1 --port 5176 --strictPort
//   node scripts/free-check.mjs
// 1. The GPU world matches the TypeScript reference (src/lib/dev/freeReference.ts, with the
//    third-party z80-emulator executing instructions) tick for tick: sheet bytes and registers, on a
//    small crowded sheet where heads often want the same byte. Both HALT rules.
// 2. Two GPU runs of one seed give the same sheet, with thousands of heads and mutation on.
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
	const { freeLayout, HEAD_WORDS } = await import('/src/lib/gpu/free.ts');
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

	for (const haltRule of ['wait', 'pass']) {
		for (const seed of [3, 4, 5, 6]) {
			const settings = {
				...base,
				width: 8,
				height: 8,
				seed,
				heads: 1,
				headDirections: 'all',
				haltRule,
				mutationExponent: 31
			};
			const w = 32;
			const h = 32;
			await engine.configure(settings, { kind: 'random' });
			// Start from the engine's random sheet with the index prefixes taken out (see the reference).
			const start = toSheet(await engine.readSoupData(), w, h);
			for (let i = 0; i < start.length; i++)
				if (start[i] === 0xdd || start[i] === 0xfd) start[i] = 0;
			engine.writeSoupData(fromSheet(start, w, h));
			const heads = await engine.readHeads();
			const layout = freeLayout(settings);
			const reference = createReference(w, h, start, heads, haltRule);
			const ticks = 400;
			let agreed = 0;
			let mismatch = null;
			for (let t = 1; t <= ticks && !mismatch; t++) {
				engine.simulateStep();
				reference.tick();
				if (reference.firstIndexPrefixTick !== null) break;
				if (t % 10 && t !== ticks) continue;
				await engine.waitForIdle();
				const sheet = toSheet(await engine.readSoupData(), w, h);
				const gpuHeads = await engine.readHeads();
				let bytes = 0;
				for (let i = 0; i < sheet.length; i++) if (sheet[i] !== reference.sheet[i]) bytes++;
				let registers = 0;
				let example = null;
				for (let k = 0; k < heads.length / HEAD_WORDS; k++) {
					const want = reference.registers(k);
					const got = {
						...Object.fromEntries(
							['a', 'f', 'b', 'c', 'd', 'e', 'h', 'l', 'ix', 'iy', 'sp', 'pc'].map((n) => [
								n,
								gpuHeads[k * HEAD_WORDS + layout.state[n]]
							])
						)
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
				if (bytes || registers) mismatch = { tick: t, bytes, registers, example };
				else agreed = t;
			}
			let changed = 0;
			for (let i = 0; i < start.length; i++) if (start[i] !== reference.sheet[i]) changed++;
			check(
				`GPU world equals the reference, HALT rule ${haltRule}, seed ${seed}`,
				!mismatch && agreed >= 50 && reference.contested > 0,
				{
					heads: heads.length / HEAD_WORDS,
					ticksAgreed: agreed,
					stoppedAtIndexPrefixTick: reference.firstIndexPrefixTick,
					sheetBytesChanged: changed,
					contestedBytes: reference.contested,
					headTicksOnHalt: reference.halts,
					mismatch
				}
			);
		}
	}

	// 2. Repeatability with many heads and mutation on.
	const big = {
		...base,
		width: 64,
		height: 64,
		seed: 9,
		heads: 1,
		headDirections: 'weave',
		haltRule: 'wait',
		mutationExponent: 6
	};
	const run = async () => {
		await engine.configure(big, { kind: 'random' });
		for (let t = 0; t < 500; t++) engine.simulateStep();
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
		'two runs of one seed give the same sheet (4,096 heads, 500 ticks, mutation on)',
		differ === 0 && moved > 1000,
		{
			differingBytes: differ,
			bytesChangedByTheRun: moved
		}
	);
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

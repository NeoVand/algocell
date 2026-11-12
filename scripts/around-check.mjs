// Checks of the two cell-wise worlds (`neighbourhood` and `cells`) on the real GPU pipeline. Needs
// the dev server on 5176.
//   node scripts/around-check.mjs
// 1. One GPU update equals one update of the TypeScript reference (src/lib/dev/aroundReference.ts,
//    with the third-party z80-emulator), cell by cell, starting each comparison from the GPU's own
//    state. Cells whose run fetched an instruction on which the two Z80s are known to differ are
//    counted and left out.
//    The same under the two-layer law (now' = out xor now xor before) at dissipation 0.5 and 0,
//    where the remembered tapes are compared as well.
// 2. Two runs of one seed give the same lattice.
// 4. A closed world (dissipation 0, mutation off) run forwards and then backwards for the same
//    number of updates is back where it started, byte for byte; with dissipation 0.5 it is not.
// 3. Processors meet their neighbours in each of the eight orders equally often.
import { chromium } from '@playwright/test';

const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('page error:', e.message));
await page.goto(process.argv[2] ?? 'http://127.0.0.1:5176/demo/playwright');
const report = await page.evaluate(async () => {
	const { GPUEngine } = await import('/src/lib/gpu/engine.ts');
	const { ORIGINAL_SETTINGS } = await import('/src/lib/gpu/machine.ts');
	const { aroundLayout, aroundMutationChance } = await import('/src/lib/gpu/around.ts');
	const { HEAD_WORDS, HEAD_AGE, HEAD_WAITING, HEAD_FACING, HEAD_RUNS } =
		await import('/src/lib/gpu/free.ts');
	const { referenceUpdate } = await import('/src/lib/dev/aroundReference.ts');
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
		boundary: 'reflect',
		width: 16,
		height: 12,
		steps: 128
	};
	const cases = [];
	for (const rule of ['neighbourhood', 'cells'])
		for (const gather of [1, 2, 4])
			for (const exponent of [31, 0]) cases.push({ rule, gather, exponent, dissipation: 1 });
	for (const dissipation of [0.5, 0])
		for (const gather of [1, 4])
			for (const exponent of [31, 0])
				cases.push({ rule: 'neighbourhood', gather, exponent, dissipation });
	{
		{
			for (const { rule, gather, exponent, dissipation } of cases) {
				const settings = {
					...base,
					interaction: rule,
					gather,
					dissipation,
					seed: 40 + gather,
					mutationExponent: exponent
				};
				await engine.configure(settings, { kind: 'random' });
				const state = aroundLayout(settings).state;
				const cells = settings.width * settings.height;
				const updates = rule === 'cells' ? 400 : 40;
				let compared = 0;
				let flagged = 0;
				let mismatches = 0;
				let example = null;
				let halts = 0;
				let full = 0;
				let budget = 0;
				let changed = 0;
				for (let u = 0; u < updates; u++) {
					const before = await engine.readSoupData();
					const remembered = await engine.readBeforeData();
					const heads = await engine.readHeads();
					engine.simulateUpdates(1);
					await engine.waitForIdle();
					const after = await engine.readSoupData();
					const rememberedAfter = await engine.readBeforeData();
					const headsAfter = await engine.readHeads();
					const want = referenceUpdate(
						settings.width,
						settings.height,
						before,
						heads,
						state,
						{
							rule,
							gather,
							budget: settings.steps,
							mutation: exponent >= 31 ? 0 : aroundMutationChance(rule, exponent),
							dissipation
						},
						remembered
					);
					for (let c = 0; c < cells; c++) {
						const w = want[c];
						if (w.flagged) {
							flagged++;
							continue;
						}
						compared++;
						const o = c * HEAD_WORDS;
						let bad = null;
						for (let i = 0; i < 16; i++) {
							if (after[c * 16 + i] !== w.tape[i])
								bad ??= { byte: i, want: w.tape[i], got: after[c * 16 + i] };
							if (after[c * 16 + i] !== before[c * 16 + i]) changed++;
							if (rule === 'neighbourhood' && rememberedAfter[c * 16 + i] !== w.before[i])
								bad ??= { rememberedByte: i, want: w.before[i], got: rememberedAfter[c * 16 + i] };
						}
						if (rule === 'neighbourhood') {
							if (headsAfter[o + HEAD_AGE] !== w.steps || headsAfter[o + HEAD_WAITING] !== w.reason)
								bad ??= {
									steps: [w.steps, headsAfter[o + HEAD_AGE]],
									reason: [w.reason, headsAfter[o + HEAD_WAITING]]
								};
							if (w.reason === 1) halts++;
							else if (w.reason === 2) full++;
							else budget++;
						} else {
							for (const n of ['a', 'f', 'b', 'c', 'd', 'e', 'h', 'l', 'sp', 'pc']) {
								const mask = n === 'f' ? 0xd7 : 0xffff;
								if ((headsAfter[o + state[n]] & mask) !== (w.registers[n] & mask))
									bad ??= { register: n, want: w.registers[n], got: headsAfter[o + state[n]] };
							}
							if ((headsAfter[o + HEAD_FACING] & 7) !== w.order)
								bad ??= { order: [w.order, headsAfter[o + HEAD_FACING]] };
							if (w.restarted) halts++;
						}
						if (bad) {
							mismatches++;
							example ??= {
								update: u,
								cell: c,
								...bad,
								fetched: w.fetched,
								registersBefore: Object.fromEntries(
									['a', 'f', 'b', 'c', 'h', 'l', 'i', 'r', 'sp', 'pc'].map((n) => [
										n,
										heads[c * HEAD_WORDS + state[n]].toString(16)
									])
								)
							};
						}
					}
				}
				check(
					`GPU equals the reference: ${rule}, ${gather} neighbour${gather > 1 ? 's' : ''} gathered, dissipation ${dissipation}, mutation ${exponent >= 31 ? 'off' : 'on'}`,
					mismatches === 0 && compared > 1000 && changed > 500 && halts > 0,
					{
						updates,
						cellUpdatesCompared: compared,
						leftOutAsNotComparable: flagged,
						bytesChanged: changed,
						mismatches,
						example,
						...(rule === 'neighbourhood'
							? { endedByHalt: halts, endedByFullCell: full, endedByBudget: budget }
							: { restarts: halts })
					}
				);
			}
		}
	}

	// 4. Forwards then backwards. Time is turned round by exchanging the two layers; the random
	// words are then drawn in reverse.
	for (const dissipation of [0, 0.5]) {
		const settings = {
			...base,
			width: 64,
			height: 48,
			interaction: 'neighbourhood',
			gather: 4,
			dissipation,
			seed: 12,
			mutationExponent: 31
		};
		await engine.configure(settings, { kind: 'random' });
		const startNow = await engine.readSoupData();
		const startBefore = await engine.readBeforeData();
		const startHeads = await engine.readHeads();
		const updates = 300;
		for (let t = 0; t < updates; t += 25) engine.simulateUpdates(25);
		await engine.waitForIdle();
		const endNow = await engine.readSoupData();
		let changed = 0;
		for (let i = 0; i < endNow.length; i++) if (endNow[i] !== startNow[i]) changed++;
		engine.reverseTime();
		for (let t = 0; t < updates; t += 25) engine.simulateUpdates(25);
		await engine.waitForIdle();
		const countAtTheStart = engine.batchCount;
		engine.simulateUpdates(25); // nothing lies before update 0
		await engine.waitForIdle();
		engine.reverseTime();
		await engine.waitForIdle();
		const backNow = await engine.readSoupData();
		const backBefore = await engine.readBeforeData();
		const backHeads = await engine.readHeads();
		let notRestored = 0;
		for (let i = 0; i < backNow.length; i++) {
			if (backNow[i] !== startNow[i]) notRestored++;
			if (backBefore[i] !== startBefore[i]) notRestored++;
		}
		let counters = 0;
		for (let c = 0; c < settings.width * settings.height; c++)
			if (backHeads[c * HEAD_WORDS + 38] !== startHeads[c * HEAD_WORDS + 38]) counters++;
		check(
			dissipation === 0
				? `a closed world run ${updates} updates forwards and ${updates} backwards is back at its start, both layers`
				: `with dissipation ${dissipation} the same procedure does not restore the lattice`,
			changed > 10000 &&
				countAtTheStart === 0 &&
				engine.batchCount === 0 &&
				(dissipation === 0 ? notRestored === 0 && counters === 0 : notRestored > 10000),
			{
				bytesChangedForwards: changed,
				bytesNotRestored: notRestored,
				randomCountersNotRestored: counters,
				updateCountAfterGoingBack: countAtTheStart
			}
		);
	}

	// 2. Repeatability.
	for (const rule of ['neighbourhood', 'cells']) {
		const settings = {
			...base,
			width: 64,
			height: 48,
			interaction: rule,
			gather: 4,
			seed: 7,
			mutationExponent: 2
		};
		const run = async () => {
			await engine.configure(settings, { kind: 'random' });
			for (let t = 0; t < 10; t++) engine.simulateUpdates(rule === 'cells' ? 50 : 5);
			await engine.waitForIdle();
			return engine.readSoupData();
		};
		const first = await engine
			.configure(settings, { kind: 'random' })
			.then(() => engine.readSoupData());
		const a = await run();
		const b = await run();
		let differ = 0;
		let moved = 0;
		for (let i = 0; i < a.length; i++) {
			if (a[i] !== b[i]) differ++;
			if (a[i] !== first[i]) moved++;
		}
		check(`two runs of one seed give the same lattice: ${rule}`, differ === 0 && moved > 1000, {
			differingBytes: differ,
			bytesChangedByTheRun: moved
		});
	}

	// 3. The eight orders of meeting the neighbours are equally likely (neighbourhood: drawn every
	// update and left in the head for inspection).
	{
		const settings = {
			...base,
			width: 64,
			height: 48,
			interaction: 'neighbourhood',
			gather: 4,
			seed: 9,
			mutationExponent: 31
		};
		await engine.configure(settings, { kind: 'random' });
		const counts = new Array(8).fill(0);
		let total = 0;
		for (let u = 0; u < 20; u++) {
			engine.simulateUpdates(1);
			await engine.waitForIdle();
			const heads = await engine.readHeads();
			for (let c = 0; c < settings.width * settings.height; c++) {
				counts[heads[c * HEAD_WORDS + HEAD_FACING] & 7]++;
				total++;
			}
		}
		const chi = counts.reduce((s, n) => s + (n - total / 8) ** 2 / (total / 8), 0);
		check('the eight orders of meeting the neighbours are equally likely', chi < 24.3, {
			draws: total,
			counts,
			chiSquared: +chi.toFixed(2)
		});
	}
	void HEAD_RUNS;
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

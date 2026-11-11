// Checks of the reversible interaction rule on the real GPU pipeline. Needs the dev server:
//   npm run dev -- --host 127.0.0.1 --port 5176 --strictPort
//   node scripts/reversible-check.mjs
// The lattice update resolves colliding pairs with a GPU race, so two runs of many pairs per update
// differ even for one seed. Every exact check below therefore uses one pair per update, where there
// is nothing to race over.
// 1. Precondition: with one pair per update the overwrite rule repeats itself bit for bit.
// 2. Reversible at full dissipation equals overwrite bit for bit, with mutation on.
// 3. The law: on a lattice that is not at rest, each closed event gives
//        now' = out ^ now ^ before,   before' = now
//    where `out` is what the programs write (read off by replaying the same draw at full dissipation).
// 4. Undo: run closed events forward, swap the two layers, replay the same draws in reverse order,
//    swap again; the starting lattice must come back bit for bit. With dissipation it must not.
// 5. Every boundary rule runs and the rules give different lattices.
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
	const canvas = document.createElement('canvas');
	canvas.width = 64;
	canvas.height = 64;
	document.body.append(canvas);
	const engine = new GPUEngine(6);
	if (!(await engine.init(canvas))) throw new Error('No WebGPU');
	const base = { ...ORIGINAL_SETTINGS, width: 48, height: 40, seed: 11 };
	const L = base.length;
	const cells = base.width * base.height;
	const differing = (a, b) => {
		let n = 0;
		for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++;
		return n;
	};
	const step = async () => {
		engine.simulateStep();
		await engine.waitForIdle();
	};
	const read = async () => ({ X: await engine.readSoupData(), P: await engine.readBeforeData() });
	// Replace the engine's stream of per-update draws with a list we control.
	const feed = (seeds) => {
		let k = 0;
		engine.cpuRng = { nextU32: () => seeds[k++] };
	};
	const seedsFor = (n, salt) => {
		const seeds = new Uint32Array(n);
		let s = (0x9e3779b9 ^ salt) >>> 0;
		for (let i = 0; i < n; i++) {
			s = (Math.imul(s ^ (s >>> 15), 0x2c1b3c6d) + 0x297a2d39) >>> 0;
			seeds[i] = s;
		}
		return seeds;
	};
	const out = [];
	const check = (name, pass, detail) => out.push({ name, pass, detail });

	// 1 and 2: one pair and one mutation per update (mutationExponent 0).
	const single = { pairAttempts: 1, mutationExponent: 0 };
	const runSingle = async (extra, updates) => {
		await engine.configure({ ...base, ...single, ...extra }, { kind: 'random' });
		feed(seedsFor(updates, 1));
		for (let u = 0; u < updates; u++) engine.simulateStep();
		await engine.waitForIdle();
		return engine.readSoupData();
	};
	const start = await runSingle({ interaction: 'overwrite' }, 0);
	const a = await runSingle({ interaction: 'overwrite' }, 1500);
	const b = await runSingle({ interaction: 'overwrite' }, 1500);
	const c = await runSingle({ interaction: 'reversible', dissipation: 1 }, 1500);
	check(
		'overwrite repeats itself (one pair per update, 1,500 updates)',
		differing(a, b) === 0 && differing(start, a) > 500,
		{
			differingBytes: differing(a, b),
			bytesChangedByTheRun: differing(start, a)
		}
	);
	check(
		'reversible at full dissipation equals overwrite bit for bit, mutation on',
		differing(a, c) === 0,
		{
			differingBytes: differing(a, c)
		}
	);

	// A lattice that is not at rest: 40 closed updates with many pairs, then one pair per update.
	const closed = { interaction: 'reversible', dissipation: 0, mutationExponent: 31 };
	const heat = async () => {
		await engine.configure({ ...base, ...closed, pairAttempts: 512 }, { kind: 'random' });
		for (let u = 0; u < 40; u++) engine.simulateStep();
		await engine.waitForIdle();
		await engine.setMachine({ ...engine.settings, pairAttempts: 1 });
	};

	// 3. The law, event by event.
	await heat();
	const events = 300;
	const lawSeeds = seedsFor(events, 2);
	let cellsChecked = 0;
	let cellsHotBefore = 0;
	let lawBroken = 0;
	let idleBroken = 0;
	for (let e = 0; e < events; e++) {
		const before = await read();
		engine.setDissipation(1);
		feed([lawSeeds[e]]);
		await step();
		const F = await engine.readSoupData();
		engine.writeSoupData(before.X);
		engine.writeBeforeData(before.P);
		engine.setDissipation(0);
		feed([lawSeeds[e]]);
		await step();
		const after = await read();
		for (let cell = 0; cell < cells; cell++) {
			let outChanged = false;
			let law = true;
			let untouched = true;
			let swapped = true;
			let hot = false;
			for (let i = cell * L; i < (cell + 1) * L; i++) {
				if (F[i] !== before.X[i]) outChanged = true;
				if (before.X[i] !== before.P[i]) hot = true;
				if (after.X[i] !== (F[i] ^ before.X[i] ^ before.P[i]) || after.P[i] !== before.X[i])
					law = false;
				if (after.X[i] !== before.X[i] || after.P[i] !== before.P[i]) untouched = false;
				if (after.X[i] !== before.P[i] || after.P[i] !== before.X[i]) swapped = false;
			}
			if (outChanged) {
				cellsChecked++;
				if (hot) cellsHotBefore++;
				if (!law) lawBroken++;
			} else if (!untouched && !swapped) idleBroken++;
		}
	}
	check(
		'each closed event obeys now′ = out ⊕ now ⊕ before, before′ = now',
		cellsChecked > 100 && cellsHotBefore > 20 && lawBroken === 0 && idleBroken === 0,
		{
			events,
			cellsWhoseOutputChanged: cellsChecked,
			ofThoseNotAtRest: cellsHotBefore,
			lawBroken,
			idleBroken
		}
	);

	// 4. Undo.
	const undo = async (dissipation, n) => {
		await heat();
		engine.setDissipation(dissipation);
		const s0 = await read();
		const seeds = seedsFor(n, 3);
		feed(seeds);
		for (let u = 0; u < n; u++) engine.simulateStep();
		await engine.waitForIdle();
		const s1 = await read();
		engine.writeSoupData(s1.P);
		engine.writeBeforeData(s1.X);
		feed(Uint32Array.from(seeds).reverse());
		for (let u = 0; u < n; u++) engine.simulateStep();
		await engine.waitForIdle();
		const s2 = await read();
		return {
			bytesChangedForward: differing(s0.X, s1.X) + differing(s0.P, s1.P),
			bytesNotRestored: differing(s2.P, s0.X) + differing(s2.X, s0.P)
		};
	};
	const closedUndo = await undo(0, 2000);
	check(
		'2,000 closed events run backwards restore both layers bit for bit',
		closedUndo.bytesChangedForward > 1000 && closedUndo.bytesNotRestored === 0,
		closedUndo
	);
	const leakyUndo = await undo(0.5, 2000);
	check(
		'with dissipation 0.5 the same procedure does not restore the lattice',
		leakyUndo.bytesNotRestored > 0,
		leakyUndo
	);

	// 5. Boundary rules.
	const finals = {};
	for (const boundary of ['reflect', 'cylinder', 'torus', 'mobius', 'klein', 'projective']) {
		await engine.configure({ ...base, interaction: 'overwrite', boundary }, { kind: 'random' });
		for (let u = 0; u < 40; u++) engine.simulateStep();
		await engine.waitForIdle();
		finals[boundary] = await engine.readSoupData();
	}
	const names = Object.keys(finals);
	let least = Infinity;
	for (let i = 0; i < names.length; i++)
		for (let j = i + 1; j < names.length; j++)
			least = Math.min(least, differing(finals[names[i]], finals[names[j]]));
	check('six boundary rules compile and run', least > 0, {
		smallestDifferenceBetweenTwoRules: least
	});
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

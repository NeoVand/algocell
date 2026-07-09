// Substrate-v2 (SUM×DIR16) verification — Step A of the plan (DEV, headless).
//
//   npx tsx src/lib/morph/dev/v2verify.ts
//
// Gates every GPU result: the v2 Z80 bootstrap must be byte-exact vs the v2 TS
// reference (stepCAv2). Three checks:
//   1. Random-genome sweep — exercises all 832 entries and all 16 dir patterns.
//   2. Bit-assignment trap genome[index] = index&15 (== dir) — makes the output
//      equal the directional code, so any swapped/mis-weighted direction bit in
//      the Z80 shows as a diff. (The top correctness hazard flagged in review.)
//   3. Warm-start equivalence — lift a v1 genome into all 16 dir slots; the v2
//      rule must reproduce the v1 rule exactly (never regress below v1).

import {
	type MorphParams,
	computeMemoryMapV2,
	buildTape,
	readFront,
	runCAv2,
	runCA,
	centerSeed,
	asymmetricSeed,
	asciiGrid,
	genomeLengthV2,
	liftV1toV2,
	kOf
} from '../ca';
import { assembleBootstrapV2 } from '../bootstrap';
import { growthGenome, randomGenome, gridDiff, mulberry32 } from '../genomes';
import { runOnRealZ80 } from './z80run';

function randomGenomeV2(S: number, seed: number): Uint8Array {
	const rng = mulberry32(seed);
	const g = new Uint8Array(genomeLengthV2(S));
	for (let i = 0; i < g.length; i++) g[i] = Math.floor(rng() * S);
	return g;
}

/** Assemble v2, run on a real Z80, diff against the v2 reference. */
function diffV2(p: MorphParams, genome: Uint8Array, seed: Uint8Array, label: string, show = false): boolean {
	const map = computeMemoryMapV2(p);
	const code = assembleBootstrapV2(p, map);
	const tape = buildTape(code, genome, seed, map, p);
	const cap = p.T * p.W * p.H * 300 + 300000;
	const { mem, steps, halted } = runOnRealZ80(tape, map.memBytes, cap);
	const z80Grid = readFront(mem, map, p);
	const refGrid = runCAv2(seed, genome, p);
	const diff = gridDiff(z80Grid, refGrid);
	const ok = halted && diff === 0;
	console.log(
		`[${ok ? 'PASS' : 'FAIL'}] ${label}: ${p.W}x${p.H} S=${p.S} T=${p.T} | code=${code.length}B mem=${map.memBytes} genome=${map.genomeBytes} | ${steps} instr halted=${halted} | diff=${diff}`
	);
	if (show) console.log(asciiGrid(z80Grid, p));
	return ok;
}

function main() {
	console.log('=== Substrate-v2 (SUM×DIR16) verification ===\n');
	let ok = true;

	console.log('--- 1. random-genome sweep (all 832 entries, all 16 dir patterns) ---');
	const sweep: MorphParams[] = [
		{ W: 16, H: 16, S: 4, T: 8 },
		{ W: 24, H: 24, S: 4, T: 10 },
		{ W: 12, H: 20, S: 5, T: 6 },
		{ W: 16, H: 16, S: 4, T: 1 }
	];
	for (let i = 0; i < sweep.length; i++) {
		const p = sweep[i];
		ok = diffV2(p, randomGenomeV2(p.S, 0xa5 + i * 613), asymmetricSeed(p), `rand#${i}`) && ok;
	}

	console.log('\n--- 2. bit-assignment trap: genome[index] = index & 15 (= dir), T=1 ---');
	{
		// T=1 so the dir-code output (0..15, out of S range) is read directly and
		// never re-fed as a state (which would produce out-of-range indices). Any
		// swapped/mis-weighted direction bit in the Z80 shows as a diff here.
		const p: MorphParams = { W: 16, H: 16, S: 4, T: 1 };
		const g = new Uint8Array(genomeLengthV2(p.S));
		for (let i = 0; i < g.length; i++) g[i] = i & 15; // output == the dir code
		ok = diffV2(p, g, asymmetricSeed(p), 'trap-dir') && ok;
	}

	console.log('\n--- 3. warm-start equivalence: lift(v1) on v2 == v1 ---');
	{
		const p: MorphParams = { W: 16, H: 16, S: 4, T: 12 };
		const seed = centerSeed(p, 1);
		for (const [name, v1g] of [
			['growth', growthGenome(p.S)],
			['random', randomGenome(p.S, 999)]
		] as const) {
			const v2g = liftV1toV2(v1g, p.S);
			// (a) pure-TS: the lifted v2 rule must equal the v1 rule exactly.
			const v2ref = runCAv2(seed, v2g, p);
			const v1ref = runCA(seed, v1g, p);
			const tsDiff = gridDiff(v2ref, v1ref);
			// (b) on a real Z80: v2 bootstrap with the lifted genome == v2 reference.
			const z80ok = diffV2(p, v2g, seed, `warmstart-${name}-z80`);
			const tsOk = tsDiff === 0;
			console.log(`[${tsOk ? 'PASS' : 'FAIL'}] warmstart-${name}-ts: v2(lift(v1)) vs v1 reference diff=${tsDiff}`);
			ok = z80ok && tsOk && ok;
		}
	}

	console.log('');
	if (!ok) {
		console.error('FAIL: substrate-v2 did not verify.');
		process.exit(1);
	}
	console.log('ALL PASS: v2 Z80 bootstrap is byte-exact vs the reference, and warm-starts to v1.');
	void kOf;
}

main();

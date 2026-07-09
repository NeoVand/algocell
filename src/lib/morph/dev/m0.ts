// M0 differential test (DEV, headless — run with `npx tsx`).
//
// Proves the hand-written Z80 CA bootstrap is correct by running it on a real
// Z80 (`z80-emulator`, the same reference the repo's Z80 oracle uses) and
// asserting its output grid matches the pure-TS reference of the identical
// rule. If these agree, the Z80 program is correct; since Zilion's GPU core is
// itself conformance-tested against this emulator, it will run identically.
//
//   npx tsx src/lib/morph/dev/m0.ts

import {
	type MorphParams,
	computeMemoryMap,
	buildTape,
	readFront,
	runCA,
	centerSeed,
	asciiGrid
} from '../ca';
import { assembleBootstrap } from '../bootstrap';
import { growthGenome, randomGenome, gridDiff } from '../genomes';
import { runOnRealZ80 } from './z80run';

/** Assemble, run on a real Z80, and diff against the reference. Returns pass/fail. */
function runConfig(p: MorphParams, genome: Uint8Array, label: string, show: boolean): boolean {
	const map = computeMemoryMap(p);
	const code = assembleBootstrap(p, map);
	const seed = centerSeed(p, 1);
	const tape = buildTape(code, genome, seed, map, p);

	const cap = p.T * p.W * p.H * 200 + 200000;
	const { mem, steps, halted } = runOnRealZ80(tape, map.memBytes, cap);
	const z80Grid = readFront(mem, map, p);
	const refGrid = runCA(seed, genome, p);
	const diff = gridDiff(z80Grid, refGrid);
	const ok = halted && diff === 0;

	console.log(
		`[${ok ? 'PASS' : 'FAIL'}] ${label}: ${p.W}x${p.H} S=${p.S} T=${p.T} | ` +
			`code=${code.length}B mem=${map.memBytes} | ${steps} instr halted=${halted} | diff=${diff}`
	);
	if (show) {
		console.log('\n--- reference ---\n' + asciiGrid(refGrid, p));
		console.log('\n--- Z80 ---\n' + asciiGrid(z80Grid, p));
	}
	return ok;
}

function main() {
	console.log('=== M0: Z80 CA differential test (real Z80 vs TS reference) ===\n');

	// Canonical growth demo — shown so we can eyeball actual growth.
	let allPass = runConfig({ W: 16, H: 16, S: 4, T: 12 }, growthGenome(4), 'growth', true);

	console.log('\n--- generality sweep (random genomes exercise every rule entry) ---');
	const sweep: MorphParams[] = [
		{ W: 16, H: 16, S: 4, T: 8 },
		{ W: 16, H: 16, S: 6, T: 8 },
		{ W: 24, H: 24, S: 4, T: 10 },
		{ W: 12, H: 20, S: 5, T: 6 }, // non-square
		{ W: 16, H: 16, S: 4, T: 1 } // single step
	];
	for (let i = 0; i < sweep.length; i++) {
		const p = sweep[i];
		allPass = runConfig(p, randomGenome(p.S, 0x1234 + i * 977), `rand#${i}`, false) && allPass;
	}

	console.log('');
	if (!allPass) {
		console.error('FAIL: at least one configuration diverged.');
		process.exit(1);
	}
	console.log('ALL PASS: Z80 CA matches the reference across every configuration.');
}

main();

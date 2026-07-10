// SPECTRAL / TIME analysis of the movable-XOR rule's activations.
// Goal: detect the oscillatory ("ringing") modes the user noticed — the tell-tale of a
// SYNCHRONOUS cellular automaton sitting near a pole at −1 (period-2 limit cycle). We roll
// the held rule out at a fixed placement, hold the input, and after the field settles we
// measure, per (cell, channel) probe: std (is it moving at all?), lag-1 autocorrelation
// (≈ −1 → period-2 alternation; ≈ +1 → slow drift), and the fraction of AC energy at the
// Nyquist frequency (period-2 energy). Also runs an input-flip and measures ring during the
// transient. Pure CPU reference (rule.ts), f64.
//
//   npx tsx src/lib/morph/dev/spectral.ts

import { readFileSync } from 'node:fs';
import { IDIM, forwardMarkers, loadParams, inputChannels, type Experiment } from '../../devcomp/rule';

const cfg = IDIM; // C=16, HD=96, 17×17
const C = cfg.C, SW = cfg.SW;
const MOV = (r: number, c: number) => r * SW + c;
const inCh = [3, 4];

const par = loadParams(cfg, JSON.parse(readFileSync('src/lib/devcomp/params/xor_invariant.json', 'utf8')) as number[]);

// a moderate placement (like the demo default): 2 inputs left, output right
const ins = [MOV(6, 4), MOV(10, 4)], out = MOV(8, 12);

// probe cells: output, an input, and a few interior cells on the signal path
const probeCells: [string, number][] = [
	['out', out], ['inA', ins[0]], ['mid', MOV(8, 8)], ['nearOut', MOV(8, 11)], ['center', MOV(8, 8)]
];
const probeChans = [0, 3, 4, 6, 9, 12]; // readout, signalA, signalB, three hidden

function analyze(series: number[]): { std: number; ac1: number; nyq: number; period: number } {
	const n = series.length;
	const mean = series.reduce((a, b) => a + b, 0) / n;
	const x = series.map((v) => v - mean);
	const var0 = x.reduce((a, b) => a + b * b, 0) / n;
	const std = Math.sqrt(var0);
	// lag-1 autocorrelation
	let c1 = 0; for (let i = 1; i < n; i++) c1 += x[i] * x[i - 1];
	const ac1 = var0 > 1e-12 ? c1 / (n - 1) / var0 : 0;
	// DFT magnitude (skip DC); find dominant freq and Nyquist (period-2) energy fraction
	let total = 0, nyqE = 0, bestK = 0, bestMag = 0;
	for (let k = 1; k <= n / 2; k++) {
		let re = 0, im = 0;
		for (let i = 0; i < n; i++) { const a = (-2 * Math.PI * k * i) / n; re += x[i] * Math.cos(a); im += x[i] * Math.sin(a); }
		const mag = re * re + im * im;
		total += mag;
		if (k === Math.floor(n / 2)) nyqE = mag; // Nyquist bin = period-2
		if (mag > bestMag) { bestMag = mag; bestK = k; }
	}
	const nyq = total > 1e-9 ? nyqE / total : 0;
	const period = bestK > 0 ? n / bestK : 0;
	return { std, ac1, nyq, period };
}

function run(bits: number[], steps: number, label: string, settle = 80) {
	const states = forwardMarkers(cfg, par, ins, [out], bits, inCh, { steps });
	console.log(`\n== ${label}: input [${bits}] held ${steps} steps (analysis window = last ${steps - settle}) ==`);
	console.log(`  probe            ch   std      ac(lag1)  nyq%(period-2)  domPeriod   ringing?`);
	for (const [name, cell] of probeCells) {
		for (const ch of probeChans) {
			const series: number[] = [];
			for (let t = settle; t <= steps; t++) series.push(states[t][cell * C + ch]);
			const { std, ac1, nyq, period } = analyze(series);
			const ringing = std > 0.01 && (ac1 < -0.3 || nyq > 0.3);
			if (std > 0.003) // only show channels that actually move after settling
				console.log(`  ${name.padEnd(10)}   ${String(ch).padStart(4)}   ${std.toFixed(4)}   ${ac1.toFixed(3).padStart(7)}   ${(nyq * 100).toFixed(1).padStart(6)}%        ${period.toFixed(1).padStart(5)}     ${ringing ? 'OSCILLATES (~period-2)' : ''}`);
		}
	}
}

function main() {
	console.log('spectral analysis of movable XOR (held rule) — synchronous-CA ring test');
	run([0, 0], 250, 'HOLD 0⊕0=0');
	run([1, 0], 250, 'HOLD 1⊕0=1');
	// transient after an input flip (reactivity path): measure ring during migration
	const st = forwardMarkers(cfg, par, ins, [out], [0, 0], inCh, { steps: 250, switchAt: 120, bits2: [1, 0] });
	const tail = [];
	for (let t = 121; t <= 250; t++) tail.push(st[t][out * C + 0]);
	const a = analyze(tail);
	console.log(`\n== after input flip 0⊕0→1⊕0 at step 120 (output ch0, steps 121..250) ==`);
	console.log(`  output: std ${a.std.toFixed(4)}  ac(lag1) ${a.ac1.toFixed(3)}  nyq%(period-2) ${(a.nyq * 100).toFixed(1)}%  domPeriod ${a.period.toFixed(1)}  final ${st[250][out * C + 0].toFixed(3)}`);
}

main();

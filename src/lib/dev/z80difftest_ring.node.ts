// Headless driver for the ring-size Z80 differential test (DEV ONLY — never imported by the app).
//
// Run from the repository root (installs nothing, builds nothing for the app, needs no browser):
//
//   node_modules/.bin/esbuild src/lib/dev/z80difftest_ring.node.ts --bundle --platform=node \
//     --format=esm --tsconfig=tsconfig.json --outfile="$TMPDIR/z80ring.mjs" --log-level=warning
//   node "$TMPDIR/z80ring.mjs" [--sizes 32,38,40,64,100,128] [--random 100000] [--targeted 200]
//        [--steps 128] [--seed 1] [--out DIR] [--python experiments/.venv/bin/python] [--mutants]
//
// For each ring size P (tape L = P/2; P = 38 is the hex layout) it
//   1. builds the executor WGSL with createZ80TestShader(L, P) — the host bits the soup shader uses
//      (MEM_LENGTH = P, sp_init, pair layout) around the zilion core — and checks its provenance: identical
//      to the exported experiments/algocell_exp/shader/z80_test_L{L}.wgsl, and the soup shader for that ring
//      carries the same core and the same memory model;
//   2. generates random and targeted cases (./z80difftest_ring.ts);
//   3. runs them on the GPU via wgpu-py (experiments/z80_ring_gpu.py) and on the reference (./z80oracle.ts,
//      unchanged), comparing the final pair memory and registers (registers unmasked, all flag bits);
//   4. checks every analytic expectation of the targeted probes against both sides independently;
//   5. re-compares every program that differs, instruction by instruction to the end of the run, with the
//      reference's known defects corrected where they occur (each correction checked against the GPU there)
//      and zilion's known deviations named; anything else is reported as unexplained;
//   6. measures how many random programs moved a 16-bit register across the 0xFFFF ↔ 0x0000 seam;
//   7. with --mutants, re-runs the targeted cases (and 20,000 random ones) on shaders with one 16-bit wrap
//      mask deliberately removed, to show which ring sizes can detect such a bug.

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { Z80_CORE_WGSL } from '@neovand/zilion';
import { createSimShader, createZ80TestShader } from '$lib/gpu/shaders';
import { spInit } from '$lib/sim/constants';
import { runOracle, type OracleResult } from './z80oracle';
import {
	RING_SIZES,
	REGS_PER_CASE,
	checkExpectation,
	diffState,
	gpuRegs,
	hex,
	isBenign,
	oracleRegs,
	randomCases,
	mirrorFinal,
	seamCoverage,
	targetedCases,
	traceAgainstReference,
	type DivergenceEvent,
	type RingCase,
	type TraceResult
} from './z80difftest_ring';

// ── CLI ───────────────────────────────────────────────────────────────────────

function arg(name: string, dflt: string): string {
	const i = process.argv.indexOf(`--${name}`);
	return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : dflt;
}
const REPO = resolve(arg('repo', process.cwd()));
const SIZES = arg('sizes', RING_SIZES.join(',')).split(',').map(Number);
const N_RANDOM = Number(arg('random', '100000'));
const PER_VARIANT = Number(arg('targeted', '200'));
const STEPS = Number(arg('steps', '128'));
const SEED = Number(arg('seed', '1'));
const OUT = resolve(arg('out', join(tmpdir(), 'z80difftest_ring')));
const PYTHON = resolve(REPO, arg('python', 'experiments/.venv/bin/python'));
const GPU_SCRIPT = join(REPO, 'experiments/z80_ring_gpu.py');
const SHADER_DIR = join(REPO, 'experiments/algocell_exp/shader');
const MUTANTS = process.argv.includes('--mutants');
const TRACE_CHUNK = 2000; // programs per stepwise GPU trace run

for (const P of SIZES)
	if (!(P >= 8 && P % 2 === 0)) throw new Error(`ring size ${P} must be even (P = 2L)`);
if (!existsSync(GPU_SCRIPT)) throw new Error(`run from the repo root (missing ${GPU_SCRIPT})`);
if (!existsSync(PYTHON)) throw new Error(`no python at ${PYTHON} (pass --python)`);
mkdirSync(OUT, { recursive: true });

const sha = (s: string) => createHash('sha256').update(s).digest('hex').slice(0, 16);

// ── GPU (wgpu-py) ─────────────────────────────────────────────────────────────

interface GpuBudget {
	mem: Uint8Array; // N × P
	regs: Uint32Array; // N × REGS_PER_CASE
}

function gpuRun(
	shaderPath: string,
	P: number,
	inputs: Uint8Array[],
	trace: boolean,
	tag: string
): GpuBudget[] {
	const N = inputs.length;
	const flat = new Uint8Array(N * P);
	inputs.forEach((b, i) => flat.set(b, i * P));
	const inPath = join(OUT, `cases_${tag}.bin`);
	const outPath = join(OUT, `gpu_${tag}.bin`);
	writeFileSync(inPath, flat);
	const args = [
		GPU_SCRIPT,
		'--shader',
		shaderPath,
		'--tape',
		String(P / 2),
		'--steps',
		String(STEPS)
	];
	args.push('--inp', inPath, '--out', outPath);
	if (trace) args.push('--trace');
	const r = spawnSync(PYTHON, args, { stdio: ['ignore', 'inherit', 'inherit'] });
	if (r.status !== 0) throw new Error(`GPU run failed (${tag}): exit ${r.status} ${r.error ?? ''}`);
	const buf = readFileSync(outPath);
	const budgets = trace ? STEPS : 1;
	const per = N * P + N * REGS_PER_CASE * 4;
	if (buf.length !== budgets * per)
		throw new Error(`GPU output size ${buf.length} != ${budgets * per}`);
	const out: GpuBudget[] = [];
	for (let k = 0; k < budgets; k++) {
		const off = buf.byteOffset + k * per;
		const mem = new Uint8Array(buf.buffer.slice(off, off + N * P));
		const regs = new Uint32Array(buf.buffer.slice(off + N * P, off + per));
		out.push({ mem, regs });
	}
	return out;
}

// ── Shader provenance ─────────────────────────────────────────────────────────

function memoryModelOk(src: string, P: number): boolean {
	return (
		src.includes(Z80_CORE_WGSL) &&
		src.includes(`const MEM_LENGTH: u32 = ${P}u;`) &&
		src.includes('mem[addr % MEM_LENGTH]') &&
		src.includes('let a = addr % MEM_LENGTH;') &&
		src.includes('return 0xffffu - ((0xffffu - want) % MEM_LENGTH);')
	);
}

function provenance(P: number, shader: string) {
	const L = P / 2;
	const hexLayout = P === 38;
	const execFile = join(SHADER_DIR, `z80_test_L${L}.wgsl`);
	const simFile = join(SHADER_DIR, hexLayout ? 'sim_hex.wgsl' : `sim_square_L${L}.wgsl`);
	const sim = hexLayout ? createSimShader('hex') : createSimShader('square', L);
	return {
		layout: hexLayout ? 'hex (19-byte tapes)' : `square (${L}-byte tapes)`,
		executorSha: sha(shader),
		executorMemoryModel: memoryModelOk(shader, P),
		exportedExecutor:
			!hexLayout && existsSync(execFile) ? readFileSync(execFile, 'utf8') === shader : null,
		soupShaderMemoryModel: memoryModelOk(sim, P),
		exportedSoupShader: existsSync(simFile) ? readFileSync(simFile, 'utf8') === sim : null
	};
}

// ── Mutants: one 16-bit wrap mask removed from the zilion core inside the executor ──────────

const MUTANT_DEFS = [
	{
		name: 'pc-increment-unmasked',
		from: 'cpu_pc = (cpu_pc + 1u) & 0xffffu',
		to: 'cpu_pc = cpu_pc + 1u'
	},
	{ name: 'pop-sp-unmasked', from: 'cpu_sp = (cpu_sp + 1u) & 0xffffu', to: 'cpu_sp = cpu_sp + 1u' },
	{
		name: 'push-sp-unmasked',
		from: 'cpu_sp = (cpu_sp - 1u) & 0xffffu',
		to: 'cpu_sp = cpu_sp - 1u'
	},
	{
		name: 'rel-jump-unmasked',
		from: 'cpu_pc = u32(i32(cpu_pc) + d) & 0xffffu',
		to: 'cpu_pc = u32(i32(cpu_pc) + d)'
	}
];

// ── Main ──────────────────────────────────────────────────────────────────────

interface FamilyStats {
	n: number;
	real: number;
	benign: number;
	expectChecked: number;
	oracleFails: number;
	gpuFails: number;
}

const report: Record<string, unknown> = {
	reference:
		'z80-emulator 2.3.0 (Lawrence Kesteloot, github.com/lkesteloot/trs80, MIT) via src/lib/dev/z80oracle.ts',
	core: 'createZ80TestShader(L, P) from src/lib/gpu/shaders.ts around @neovand/zilion',
	steps: STEPS,
	seed: SEED,
	sizes: {}
};
const t0 = Date.now();

for (const P of SIZES) {
	const L = P / 2;
	const tSize = Date.now();
	const shader = createZ80TestShader(L, P);
	const shaderPath = join(OUT, `z80_test_P${P}.wgsl`);
	writeFileSync(shaderPath, shader);
	const prov = provenance(P, shader);

	const cases: RingCase[] = [
		...randomCases(P, N_RANDOM, (SEED * 0x10001 + P) >>> 0),
		...targetedCases(P, PER_VARIANT, SEED)
	];
	const g = gpuRun(
		shaderPath,
		P,
		cases.map((c) => c.input),
		false,
		`P${P}`
	)[0];

	const oracle: OracleResult[] = [];
	const fam: Record<string, FamilyStats> = {};
	const mism: number[] = [];
	const expectFailures: unknown[] = [];
	let real = 0;
	let realMem = 0;
	let benign = 0;
	for (let i = 0; i < cases.length; i++) {
		const c = cases[i];
		const o = runOracle(c.input, STEPS, P);
		oracle.push(o);
		const gm = g.mem.subarray(i * P, (i + 1) * P);
		const gr = gpuRegs(g.regs, i);
		const or = oracleRegs(o);
		const key = c.family === 'random' ? 'random' : `${c.family}/${c.variant}`;
		const f = (fam[key] ??= {
			n: 0,
			real: 0,
			benign: 0,
			expectChecked: 0,
			oracleFails: 0,
			gpuFails: 0
		});
		f.n++;
		const d = diffState(gm, gr, o.mem, or, true);
		if (d.memDiffBytes.length || d.regDiffs.length) {
			mism.push(i);
			if (isBenign(d)) {
				benign++;
				f.benign++;
			} else {
				real++;
				f.real++;
				if (d.memDiffBytes.length) realMem++;
			}
		}
		if (c.expect) {
			f.expectChecked++;
			const ob = checkExpectation(c.expect, o.mem, or);
			const gb = checkExpectation(c.expect, gm, gr);
			if (ob.length) f.oracleFails++;
			if (gb.length) f.gpuFails++;
			if ((ob.length || gb.length) && expectFailures.length < 20)
				expectFailures.push({ key, input: Array.from(c.input), oracle: ob, gpu: gb });
		}
	}

	// Every program that differs from the plain reference (benign or not) is re-compared instruction by
	// instruction against the reference with its two known defects corrected where they occur.
	const traces: Array<TraceResult & { idx: number }> = [];
	for (let s0 = 0; s0 < mism.length; s0 += TRACE_CHUNK) {
		const chunk = mism.slice(s0, s0 + TRACE_CHUNK);
		const tr = gpuRun(
			shaderPath,
			P,
			chunk.map((i) => cases[i].input),
			true,
			`P${P}_trace`
		);
		chunk.forEach((ci, j) => {
			const gpuAt = (k: number) => ({
				mem: tr[k - 1].mem.subarray(j * P, (j + 1) * P),
				regs: gpuRegs(tr[k - 1].regs, j)
			});
			const t = traceAgainstReference(cases[ci].input, P, gpuAt, STEPS);
			if (t.outcome !== 'unexplained' && t.events.length === 0)
				throw new Error(
					`case ${ci}: differs from runOracle but matches the uncorrected mirror (mirror bug)`
				);
			traces.push({ ...t, idx: ci });
		});
	}
	const outcomes: Record<string, { random: number; targeted: number }> = {};
	const events: Record<string, { n: number; verified: number }> = {};
	const examples: Record<string, unknown> = {};
	const describe = (ev: DivergenceEvent, ci: number) => ({
		step: ev.step,
		family: `${cases[ci].family}${cases[ci].family === 'random' ? '' : '/' + cases[ci].variant}`,
		pc: hex(ev.pc, 4),
		instruction: `${ev.bytes.map((x) => hex(x)).join(' ')}  ${ev.mnemonic}`,
		verified: ev.verified,
		diff: ev.diff ?? null,
		program: Array.from(cases[ci].input, (x) => hex(x)).join(' ')
	});
	for (const t of traces) {
		const o = (outcomes[t.outcome] ??= { random: 0, targeted: 0 });
		if (cases[t.idx].family === 'random') o.random++;
		else o.targeted++;
		for (const ev of t.events) {
			const e = (events[ev.cause] ??= { n: 0, verified: 0 });
			e.n++;
			if (ev.verified) e.verified++;
			const prev = examples[ev.cause] as { step: number } | undefined;
			if (!prev || ev.step < prev.step) examples[ev.cause] = describe(ev, t.idx);
		}
		if (t.stop) {
			const key =
				t.stop.cause === 'unexplained'
					? 'stop:unexplained'
					: `stop:unexplained:${t.stop.cause}-not-matched`;
			const prev = examples[key] as { step: number } | undefined;
			if (!prev || t.stop.step < prev.step) examples[key] = describe(t.stop, t.idx);
		}
	}

	// The single-stepping mirror must reproduce runOracle exactly (checked on the first 2,000 cases).
	for (let i = 0; i < Math.min(2000, cases.length); i++) {
		const m = mirrorFinal(cases[i].input, P, STEPS);
		const d = diffState(m.mem, m.regs, oracle[i].mem, oracleRegs(oracle[i]));
		if (d.memDiffBytes.length || d.regDiffs.length)
			throw new Error(`mirror != runOracle on case ${i}`);
	}

	// Seam coverage of the random programs.
	const cov = { any: 0, pc: 0, sp: 0, ptr: 0 };
	for (const c of cases) {
		if (c.family !== 'random') continue;
		const s = seamCoverage(c.input, P, STEPS);
		if (s.any) cov.any++;
		if (s.pc) cov.pc++;
		if (s.sp) cov.sp++;
		if (s.ptr) cov.ptr++;
	}

	// Mutants.
	const mutants: Record<string, unknown> = {};
	if (MUTANTS) {
		const sub = cases.map((c, i) => i).filter((i) => cases[i].family !== 'random' || i < 20000);
		for (const m of MUTANT_DEFS) {
			const n = shader.split(m.from).length - 1;
			if (n === 0) throw new Error(`mutant ${m.name}: pattern not found`);
			const mPath = join(OUT, `z80_test_P${P}_${m.name}.wgsl`);
			writeFileSync(mPath, shader.split(m.from).join(m.to));
			const gmut = gpuRun(
				mPath,
				P,
				sub.map((i) => cases[i].input),
				false,
				`P${P}_${m.name}`
			)[0];
			let anyDet = 0;
			let memDet = 0;
			let targetedMemDet = 0;
			sub.forEach((ci, j) => {
				const base = diffState(
					g.mem.subarray(ci * P, (ci + 1) * P),
					gpuRegs(g.regs, ci),
					oracle[ci].mem,
					oracleRegs(oracle[ci])
				);
				if (base.memDiffBytes.length || base.regDiffs.length) return; // only cases the real core passes
				const d = diffState(
					gmut.mem.subarray(j * P, (j + 1) * P),
					gpuRegs(gmut.regs, j),
					oracle[ci].mem,
					oracleRegs(oracle[ci])
				);
				if (d.memDiffBytes.length || d.regDiffs.length) anyDet++;
				if (d.memDiffBytes.length) {
					memDet++;
					if (cases[ci].family !== 'random') targetedMemDet++;
				}
			});
			mutants[m.name] = {
				sites: n,
				cases: sub.length,
				detected: anyDet,
				detectedInMemory: memDet,
				detectedInMemoryByTargeted: targetedMemDet
			};
		}
	}

	const nRandom = cases.filter((c) => c.family === 'random').length;
	const nTargeted = cases.length - nRandom;
	const targetedStats = Object.entries(fam).filter(([k]) => k !== 'random');
	const expChecked = targetedStats.reduce((s, [, f]) => s + f.expectChecked, 0);
	const expOracleFails = targetedStats.reduce((s, [, f]) => s + f.oracleFails, 0);
	const expGpuFails = targetedStats.reduce((s, [, f]) => s + f.gpuFails, 0);
	const targetedReal = targetedStats.reduce((s, [, f]) => s + f.real, 0);

	const entry = {
		P,
		L,
		spInit: hex(spInit(P), 4),
		seamOffset: 65536 % P, // position of address 0x10000 if it were not wrapped: 0 iff P divides 65,536
		provenance: prov,
		random: nRandom,
		targeted: nTargeted,
		real,
		realMem,
		benign,
		targetedReal,
		expectations: {
			checked: expChecked,
			oracleFails: expOracleFails,
			gpuFails: expGpuFails,
			failures: expectFailures
		},
		outcomes,
		events,
		examples,
		seamCoverageRandom: cov,
		families: fam,
		mutants,
		seconds: (Date.now() - tSize) / 1000
	};
	(report.sizes as Record<string, unknown>)[P] = entry;

	const pct = (x: number, n: number) => `${((100 * x) / n).toFixed(2)}%`;
	console.log(
		`\nP=${P} (L=${L}, ${prov.layout}; SP0=0x${entry.spInit}; 65536 mod P = ${entry.seamOffset})` +
			`\n  executor ${prov.executorSha}: memory model ${prov.executorMemoryModel ? 'ok' : 'MISSING'}; ` +
			`= exported z80_test_L${L}.wgsl: ${prov.exportedExecutor ?? 'n/a'}; soup shader same core+memory model: ${prov.soupShaderMemoryModel}; ` +
			`exported soup shader current: ${prov.exportedSoupShader}` +
			`\n  random ${nRandom}: non-benign mismatches ${real - targetedReal} | targeted ${nTargeted}: non-benign mismatches ${targetedReal} | ` +
			`benign (F3/F5 only) ${benign} | memory differs in ${realMem}` +
			`\n  analytic expectations: ${expChecked} checked; reference fails ${expOracleFails}; GPU fails ${expGpuFails}` +
			`\n  differing programs re-compared instruction by instruction: ${traces.length}; outcomes ${JSON.stringify(outcomes)}` +
			`\n  events (reference quirks corrected / zilion deviations; verified = GPU equals the corrected state there): ${JSON.stringify(events)}` +
			`\n  random programs crossing the 0xFFFF<->0x0000 seam: ${pct(cov.any, nRandom)} (PC ${pct(cov.pc, nRandom)}, SP ${pct(cov.sp, nRandom)}, HL/DE/BC/IX/IY ${pct(cov.ptr, nRandom)})` +
			(MUTANTS ? `\n  mutants: ${JSON.stringify(mutants)}` : '') +
			`\n  ${entry.seconds.toFixed(1)} s`
	);
}

report.seconds = (Date.now() - t0) / 1000;
writeFileSync(join(OUT, 'report.json'), JSON.stringify(report, null, 1));
console.log(`\nreport: ${join(OUT, 'report.json')}`);

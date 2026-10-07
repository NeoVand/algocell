// Differential test harness (DEV ONLY — never imported by the app runtime).
//
// Runs thousands of random programs through BOTH the GPU Z80 core (the exact
// shader code the simulation ships: the zilion core wrapped in Algocell's
// memory model, pair layout and suppression hook) and a real-Z80 reference
// emulator (./z80oracle.ts), then compares final memory + registers.
//
// Modes: 'square' (two 16-byte tapes = 32-byte pair) or 'hex' (two 19-byte
// tapes, word-padded to 5 words each = 38-byte address space), each with or
// without an instruction-suppression set. Together these cover the ISA, the
// prefix-aware ablation path, and the hex pair layout.

import { createZ80TestShader } from '$lib/gpu/shaders';
import { runOracle } from './z80oracle';
import { disassemble } from '$lib/z80-disasm';
import {
	INSTRUCTIONS,
	emptySuppression,
	suppressionMasks,
	countSuppressed,
	type SuppressSets
} from '$lib/z80-opcodes';

const REGS_PER_CASE = 12; // a,f,b,c,d,e,h,l,sp,pc,writes_a,writes_b
const REG_NAMES = ['a', 'f', 'b', 'c', 'd', 'e', 'h', 'l', 'sp', 'pc'] as const;
const PARAM_WORDS = 32; // Params: 8 scalars + 6 × vec4<u32> suppression masks

export type LayoutMode = 'square' | 'hex';

export function tapeLengthOf(mode: LayoutMode): number {
	return mode === 'hex' ? 19 : 16;
}

export interface DiffMismatch {
	caseIdx: number;
	input: number[]; // pair bytes (32 or 38)
	memDiffByte: number; // -1 if memory matched
	regDiffs: { name: string; gpu: number; cpu: number }[];
	benign: boolean; // true = only undocumented F3/F5 flag bits differ
	oracleQuirk: boolean; // true = oracle relied on its DD/FD-NOP quirk (not a Zilion bug)
	disasm: string[];
}

export interface DiffReport {
	total: number;
	steps: number;
	seed: number;
	mode: LayoutMode;
	pairLength: number;
	suppressed: number; // instructions in the suppression set (0 = none)
	real: number; // mismatches that matter (memory or documented register/flag)
	realMem: number; // subset of `real` where final MEMORY differs (affects the sim)
	realRegOnly: number; // subset of `real` where only registers differ (sim discards these)
	realTrue: number; // subset of `real` NOT caused by the oracle's DD/FD-NOP quirk (genuine)
	oracleQuirk: number; // subset of `real` where the oracle NOPed a DD/FD opcode a real Z80 executes
	benign: number; // only undocumented flag bits (F3/F5) differ
	mismatches: DiffMismatch[]; // capped sample
	durationMs: number;
}

export interface DiffOptions {
	count?: number;
	steps?: number;
	seed?: number;
	mode?: LayoutMode;
	/** Instruction-suppression set applied to both sides (null/undefined = none). */
	suppress?: SuppressSets | null;
}

function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/**
 * Deterministic pseudo-random ablation set: every real instruction on every
 * page is included with probability `p`. Dense enough that random programs
 * hit suppressed base, CB, ED, IX/IY and DDCB instructions constantly.
 */
export function randomSuppression(seed: number, p = 0.25): SuppressSets {
	const rng = mulberry32((seed ^ 0x5bd1e995) >>> 0);
	const s = emptySuppression();
	for (const ins of INSTRUCTIONS) if (rng() < p) s[ins.page].add(ins.code);
	return s;
}

// ── Pair packing (must match PAIR_MEM_WGSL in shaders.ts) ───────────────────
// A case is a flat pair (tape A bytes then tape B bytes). On the GPU side each
// tape is stored word-padded: wpc = ceil(tape/4) u32 per tape.

function packCase(pair: Uint8Array, tape: number, out: Uint32Array, off: number): void {
	const wpc = Math.ceil(tape / 4);
	for (let t = 0; t < 2; t++) {
		for (let b = 0; b < tape; b++) {
			out[off + t * wpc + (b >> 2)] |= (pair[t * tape + b] & 0xff) << ((b & 3) * 8);
		}
	}
}

function unpackCase(words: Uint32Array, off: number, tape: number): Uint8Array {
	const wpc = Math.ceil(tape / 4);
	const pair = new Uint8Array(tape * 2);
	for (let t = 0; t < 2; t++) {
		for (let b = 0; b < tape; b++) {
			pair[t * tape + b] = (words[off + t * wpc + (b >> 2)] >>> ((b & 3) * 8)) & 0xff;
		}
	}
	return pair;
}

function buildParams(
	tape: number,
	count: number,
	steps: number,
	suppress: SuppressSets | null | undefined
): Uint32Array<ArrayBuffer> {
	const params = new Uint32Array(PARAM_WORDS);
	params[2] = tape; // tape_length
	params[3] = tape * 2; // pair_length
	params[4] = count; // pair_count (reused as case count)
	params[6] = steps; // z80_steps
	params.set(suppressionMasks(suppress ?? emptySuppression()), 8);
	return params;
}

interface GpuRun {
	mems: Uint8Array[]; // per case, flat pair bytes
	regs: Uint32Array; // REGS_PER_CASE per case
}

async function runGpu(
	device: GPUDevice,
	pipeline: GPUComputePipeline,
	inputs: Uint8Array[],
	tape: number,
	steps: number,
	suppress: SuppressSets | null | undefined
): Promise<GpuRun> {
	const N = inputs.length;
	const wpc = Math.ceil(tape / 4);
	const wordsPerCase = wpc * 2;
	const ioData = new Uint32Array(N * wordsPerCase);
	for (let c = 0; c < N; c++) packCase(inputs[c], tape, ioData, c * wordsPerCase);

	const ioBuf = device.createBuffer({
		size: ioData.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
	});
	device.queue.writeBuffer(ioBuf, 0, ioData);
	const regsBytes = N * REGS_PER_CASE * 4;
	const regsBuf = device.createBuffer({
		size: regsBytes,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});
	const params = buildParams(tape, N, steps, suppress);
	const paramsBuf = device.createBuffer({
		size: params.byteLength,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
	});
	device.queue.writeBuffer(paramsBuf, 0, params);
	const bind = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{ binding: 0, resource: { buffer: paramsBuf } },
			{ binding: 1, resource: { buffer: ioBuf } },
			{ binding: 2, resource: { buffer: regsBuf } }
		]
	});
	const ioStage = device.createBuffer({
		size: ioData.byteLength,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
	});
	const regsStage = device.createBuffer({
		size: regsBytes,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
	});
	const enc = device.createCommandEncoder();
	const pass = enc.beginComputePass();
	pass.setPipeline(pipeline);
	pass.setBindGroup(0, bind);
	pass.dispatchWorkgroups(Math.ceil(N / 64));
	pass.end();
	enc.copyBufferToBuffer(ioBuf, 0, ioStage, 0, ioData.byteLength);
	enc.copyBufferToBuffer(regsBuf, 0, regsStage, 0, regsBytes);
	device.queue.submit([enc.finish()]);

	await ioStage.mapAsync(GPUMapMode.READ);
	const io = new Uint32Array(ioStage.getMappedRange().slice(0));
	ioStage.unmap();
	await regsStage.mapAsync(GPUMapMode.READ);
	const regs = new Uint32Array(regsStage.getMappedRange().slice(0));
	regsStage.unmap();
	ioBuf.destroy();
	regsBuf.destroy();
	paramsBuf.destroy();
	ioStage.destroy();
	regsStage.destroy();

	const mems: Uint8Array[] = [];
	for (let c = 0; c < N; c++) mems.push(unpackCase(io, c * wordsPerCase, tape));
	return { mems, regs };
}

function rank(m: DiffMismatch): number {
	if (m.benign) return 2;
	if (m.oracleQuirk) return 1;
	return 0;
}

function regsOf(g: Uint32Array, off: number): Record<string, number> {
	return {
		a: g[off],
		f: g[off + 1],
		b: g[off + 2],
		c: g[off + 3],
		d: g[off + 4],
		e: g[off + 5],
		h: g[off + 6],
		l: g[off + 7],
		sp: g[off + 8] & 0xffff,
		pc: g[off + 9] & 0xffff
	};
}

async function makePipeline(): Promise<{ device: GPUDevice; pipeline: GPUComputePipeline }> {
	if (!navigator.gpu) throw new Error('WebGPU not available');
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) throw new Error('No WebGPU adapter');
	const device = await adapter.requestDevice();
	const pipeline = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module: device.createShaderModule({ code: createZ80TestShader() }),
			entryPoint: 'z80_test'
		}
	});
	return { device, pipeline };
}

// Diagnostic: for a single program, find the first execution step at which the
// GPU core and the reference diverge, and report the instruction that ran there.
// Used to root-cause mismatches surfaced by runZ80DiffTest.
export async function traceFirstDivergence(
	bytes: number[],
	maxSteps = 128,
	strictFlags = false,
	mode: LayoutMode = 'square',
	suppress: SuppressSets | null = null
): Promise<{
	divergeStep: number; // -1 if no divergence
	opcodeBytes: number[]; // bytes at reference pc entering the diverging step
	mnemonic: string;
	cpu: Record<string, number>;
	gpu: Record<string, number>;
}> {
	const tape = tapeLengthOf(mode);
	const PAIR = tape * 2;
	const { device, pipeline } = await makePipeline();
	const input = new Uint8Array(PAIR);
	for (let i = 0; i < PAIR; i++) input[i] = bytes[i] ?? 0;

	try {
		for (let k = 1; k <= maxSteps; k++) {
			const g = await runGpu(device, pipeline, [input], tape, k, suppress);
			const o = runOracle(input, k, PAIR, suppress ?? undefined);
			const gpu = regsOf(g.regs, 0);
			const cpu: Record<string, number> = {
				a: o.a,
				f: o.f,
				b: o.b,
				c: o.c,
				d: o.d,
				e: o.e,
				h: o.h,
				l: o.l,
				sp: o.sp,
				pc: o.pc
			};
			let differs = false;
			for (let i = 0; i < PAIR; i++) if (g.mems[0][i] !== o.mem[i]) differs = true;
			for (const n of REG_NAMES) {
				if (n === 'f') {
					const mask = strictFlags ? 0xff : ~0x28; // strict = include undocumented F3/F5
					if (((gpu.f ^ cpu.f) & mask) !== 0) differs = true;
				} else if (gpu[n] !== cpu[n]) differs = true;
			}
			if (differs) {
				const pc = k > 1 ? runOracle(input, k - 1, PAIR, suppress ?? undefined).pc : 0;
				const opBytes = [input[pc % PAIR], input[(pc + 1) % PAIR], input[(pc + 2) % PAIR]];
				return {
					divergeStep: k,
					opcodeBytes: opBytes,
					mnemonic: disassemble(new Uint8Array(opBytes))[0]?.mnemonic ?? '',
					cpu,
					gpu
				};
			}
		}
		return { divergeStep: -1, opcodeBytes: [], mnemonic: '', cpu: {}, gpu: {} };
	} finally {
		device.destroy();
	}
}

export async function runZ80DiffTest(opts: DiffOptions = {}): Promise<DiffReport> {
	const N = opts.count ?? 5000;
	const STEPS = opts.steps ?? 128;
	const seed = opts.seed ?? 1;
	const mode = opts.mode ?? 'square';
	const suppress = opts.suppress ?? null;
	const tape = tapeLengthOf(mode);
	const PAIR = tape * 2;
	const t0 = performance.now();

	// --- Generate deterministic random programs ---
	const rng = mulberry32(seed);
	const inputs: Uint8Array[] = [];
	for (let c = 0; c < N; c++) {
		const b = new Uint8Array(PAIR);
		for (let i = 0; i < PAIR; i++) b[i] = (rng() * 256) | 0;
		inputs.push(b);
	}

	// --- GPU run (exact shipping core + host bits) ---
	const { device, pipeline } = await makePipeline();
	let gpu: GpuRun;
	try {
		gpu = await runGpu(device, pipeline, inputs, tape, STEPS, suppress);
	} finally {
		device.destroy();
	}

	// --- Reference run on the same programs ---
	const mismatches: DiffMismatch[] = [];
	let real = 0;
	let realMem = 0;
	let realRegOnly = 0;
	let realTrue = 0;
	let oracleQuirk = 0;
	let benign = 0;
	const F3F5 = 0x08 | 0x20; // undocumented flag bits

	for (let c = 0; c < N; c++) {
		const o = runOracle(inputs[c], STEPS, PAIR, suppress ?? undefined);

		let memDiffByte = -1;
		for (let i = 0; i < PAIR; i++) {
			if (gpu.mems[c][i] !== o.mem[i]) {
				memDiffByte = i;
				break;
			}
		}

		const cpuRegs: Record<string, number> = {
			a: o.a,
			f: o.f,
			b: o.b,
			c: o.c,
			d: o.d,
			e: o.e,
			h: o.h,
			l: o.l,
			sp: o.sp,
			pc: o.pc
		};
		const gpuRegs = regsOf(gpu.regs, c * REGS_PER_CASE);
		const regDiffs: { name: string; gpu: number; cpu: number }[] = [];
		for (const k of REG_NAMES) {
			if (cpuRegs[k] !== gpuRegs[k]) regDiffs.push({ name: k, gpu: gpuRegs[k], cpu: cpuRegs[k] });
		}

		if (memDiffByte >= 0 || regDiffs.length > 0) {
			// Benign = the ONLY difference is in the undocumented F3/F5 bits of F
			// (these never affect control flow, so they don't change dynamics).
			const onlyF = memDiffByte < 0 && regDiffs.length === 1 && regDiffs[0].name === 'f';
			const benignFlag = onlyF && ((regDiffs[0].gpu ^ regDiffs[0].cpu) & ~F3F5) === 0;
			if (benignFlag) benign++;
			else {
				real++;
				if (memDiffByte >= 0) realMem++;
				else realRegOnly++;
				if (o.ddQuirk) oracleQuirk++;
				else realTrue++;
			}
			if (mismatches.length < 40 || (!benignFlag && !o.ddQuirk)) {
				mismatches.push({
					caseIdx: c,
					input: Array.from(inputs[c]),
					memDiffByte,
					regDiffs,
					benign: benignFlag,
					oracleQuirk: o.ddQuirk,
					disasm: disassemble(inputs[c])
						.slice(0, 8)
						.map((l) => l.mnemonic)
				});
			}
		}
	}

	// Genuine divergences first (they are what matters), then oracle quirks, then benign.
	mismatches.sort((x, y) => rank(x) - rank(y));

	return {
		total: N,
		steps: STEPS,
		seed,
		mode,
		pairLength: PAIR,
		suppressed: suppress ? countSuppressed(suppress) : 0,
		real,
		realMem,
		realRegOnly,
		realTrue,
		oracleQuirk,
		benign,
		mismatches,
		durationMs: performance.now() - t0
	};
}

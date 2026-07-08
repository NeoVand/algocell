// Differential test harness (DEV ONLY — never imported by the app runtime).
//
// Runs thousands of random 32-byte programs through BOTH the GPU Z80 core
// (the exact code the simulation ships, sliced into a standalone test shader)
// and the CPU-side Z80 twin (src/lib/sim/z80.ts, used by the trace path), then
// compares final memory + registers. The two are independent transcriptions of
// the same intended subset, so agreement over many random programs is strong
// evidence neither has a transcription bug, and any divergence is a real defect
// to fix (or a known, documented subset difference).

import { createZ80TestShader } from '$lib/gpu/shaders';
import { runOracle } from './z80oracle';
import { disassemble } from '$lib/z80-disasm';

const PAIR = 32; // bytes per test program (two 16-byte cells)
const WORDS = PAIR / 4; // 8 u32 per program
const REGS_PER_CASE = 12; // a,f,b,c,d,e,h,l,sp,pc,writes_a,writes_b
const REG_NAMES = ['a', 'f', 'b', 'c', 'd', 'e', 'h', 'l', 'sp', 'pc'] as const;

export interface DiffMismatch {
	caseIdx: number;
	input: number[]; // 32 bytes
	memDiffByte: number; // -1 if memory matched
	regDiffs: { name: string; gpu: number; cpu: number }[];
	benign: boolean; // true = only undocumented F3/F5 flag bits differ
	disasm: string[];
}

export interface DiffReport {
	total: number;
	steps: number;
	seed: number;
	real: number; // mismatches that matter (memory or documented register/flag)
	realMem: number; // subset of `real` where final MEMORY differs (affects the sim)
	realRegOnly: number; // subset of `real` where only registers differ (sim discards these)
	benign: number; // only undocumented flag bits (F3/F5) differ
	mismatches: DiffMismatch[]; // capped sample
	durationMs: number;
}

// Diagnostic: for a single program, find the first execution step at which the
// GPU core and the CPU twin diverge, and report the instruction that ran there.
// Used to root-cause mismatches surfaced by runZ80DiffTest.
export async function traceFirstDivergence(
	bytes: number[],
	maxSteps = 128,
	strictFlags = false
): Promise<{
	divergeStep: number; // -1 if no divergence
	opcodeBytes: number[]; // bytes at CPU pc entering the diverging step
	mnemonic: string;
	cpu: Record<string, number>;
	gpu: Record<string, number>;
}> {
	if (!navigator.gpu) throw new Error('WebGPU not available');
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) throw new Error('No WebGPU adapter');
	const device = await adapter.requestDevice();
	const pipeline = device.createComputePipeline({
		layout: 'auto',
		compute: { module: device.createShaderModule({ code: createZ80TestShader() }), entryPoint: 'z80_test' }
	});

	const input = new Uint8Array(32);
	for (let i = 0; i < 32; i++) input[i] = bytes[i] ?? 0;

	async function gpuAt(steps: number): Promise<{ mem: Uint8Array; regs: Uint32Array }> {
		const ioData = new Uint32Array(WORDS);
		for (let w = 0; w < WORDS; w++)
			ioData[w] =
				(input[w * 4] | (input[w * 4 + 1] << 8) | (input[w * 4 + 2] << 16) | (input[w * 4 + 3] << 24)) >>> 0;
		const ioBuf = device.createBuffer({
			size: ioData.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(ioBuf, 0, ioData);
		const regsBuf = device.createBuffer({
			size: REGS_PER_CASE * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
		});
		const params = new Uint32Array(16);
		params[2] = 16;
		params[3] = PAIR;
		params[4] = 1;
		params[6] = steps;
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
			size: REGS_PER_CASE * 4,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		});
		const enc = device.createCommandEncoder();
		const pass = enc.beginComputePass();
		pass.setPipeline(pipeline);
		pass.setBindGroup(0, bind);
		pass.dispatchWorkgroups(1);
		pass.end();
		enc.copyBufferToBuffer(ioBuf, 0, ioStage, 0, ioData.byteLength);
		enc.copyBufferToBuffer(regsBuf, 0, regsStage, 0, REGS_PER_CASE * 4);
		device.queue.submit([enc.finish()]);
		await ioStage.mapAsync(GPUMapMode.READ);
		const io = new Uint32Array(ioStage.getMappedRange().slice(0));
		ioStage.unmap();
		await regsStage.mapAsync(GPUMapMode.READ);
		const regs = new Uint32Array(regsStage.getMappedRange().slice(0));
		regsStage.unmap();
		const mem = new Uint8Array(PAIR);
		for (let i = 0; i < PAIR; i++) mem[i] = (io[i >> 2] >>> ((i & 3) * 8)) & 0xff;
		ioBuf.destroy();
		regsBuf.destroy();
		paramsBuf.destroy();
		ioStage.destroy();
		regsStage.destroy();
		return { mem, regs };
	}

	function cpuAt(steps: number): { mem: Uint8Array; regs: Record<string, number>; pcBefore: number } {
		const o = runOracle(input, steps, PAIR);
		const pcBefore = steps > 0 ? runOracle(input, steps - 1, PAIR).pc : 0;
		return {
			mem: o.mem,
			regs: { a: o.a, f: o.f, b: o.b, c: o.c, d: o.d, e: o.e, h: o.h, l: o.l, sp: o.sp, pc: o.pc },
			pcBefore
		};
	}

	const names = REG_NAMES;
	for (let k = 1; k <= maxSteps; k++) {
		const g = await gpuAt(k);
		const c = cpuAt(k);
		const gpuRegs: Record<string, number> = {
			a: g.regs[0], f: g.regs[1], b: g.regs[2], c: g.regs[3], d: g.regs[4],
			e: g.regs[5], h: g.regs[6], l: g.regs[7], sp: g.regs[8] & 0xffff, pc: g.regs[9] & 0xffff
		};
		let differs = false;
		for (let i = 0; i < PAIR; i++) if (g.mem[i] !== c.mem[i]) differs = true;
		for (const n of names) {
			if (n === 'f') {
				const mask = strictFlags ? 0xff : ~0x28; // strict = include undocumented F3/F5
				if (((gpuRegs.f ^ c.regs.f) & mask) !== 0) differs = true;
			} else if (gpuRegs[n] !== c.regs[n]) differs = true;
		}
		if (differs) {
			const pc = c.pcBefore; // pc entering the diverging step k
			const opBytes = [input[pc % PAIR], input[(pc + 1) % PAIR], input[(pc + 2) % PAIR]];
			device.destroy();
			return {
				divergeStep: k,
				opcodeBytes: opBytes,
				mnemonic: disassemble(new Uint8Array(opBytes))[0]?.mnemonic ?? '',
				cpu: c.regs,
				gpu: gpuRegs
			};
		}
	}
	device.destroy();
	return { divergeStep: -1, opcodeBytes: [], mnemonic: '', cpu: {}, gpu: {} };
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

export async function runZ80DiffTest(
	opts: { count?: number; steps?: number; seed?: number; nowMs?: number } = {}
): Promise<DiffReport> {
	const N = opts.count ?? 5000;
	const STEPS = opts.steps ?? 128;
	const seed = opts.seed ?? 1;
	const t0 = opts.nowMs ?? 0;

	if (!navigator.gpu) throw new Error('WebGPU not available');

	// --- Generate deterministic random programs ---
	const rng = mulberry32(seed);
	const inputs: Uint8Array[] = [];
	for (let c = 0; c < N; c++) {
		const b = new Uint8Array(PAIR);
		for (let i = 0; i < PAIR; i++) b[i] = (rng() * 256) | 0;
		inputs.push(b);
	}

	// --- GPU run ---
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) throw new Error('No WebGPU adapter');
	const device = await adapter.requestDevice();

	const module = device.createShaderModule({ code: createZ80TestShader() });
	const pipeline = device.createComputePipeline({
		layout: 'auto',
		compute: { module, entryPoint: 'z80_test' }
	});

	const ioData = new Uint32Array(N * WORDS);
	for (let c = 0; c < N; c++) {
		for (let w = 0; w < WORDS; w++) {
			const o = w * 4;
			ioData[c * WORDS + w] =
				(inputs[c][o] |
					(inputs[c][o + 1] << 8) |
					(inputs[c][o + 2] << 16) |
					(inputs[c][o + 3] << 24)) >>>
				0;
		}
	}

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

	// Params layout matches the Params struct in createZ80TestShader().
	const params = new Uint32Array(16);
	params[2] = 16; // tape_length
	params[3] = PAIR; // pair_length
	params[4] = N; // pair_count (reused as case count)
	params[6] = STEPS; // z80_steps
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
	const gpuIo = new Uint32Array(ioStage.getMappedRange().slice(0));
	ioStage.unmap();
	await regsStage.mapAsync(GPUMapMode.READ);
	const gpuRegs = new Uint32Array(regsStage.getMappedRange().slice(0));
	regsStage.unmap();

	// --- CPU run (z80.ts) on the same programs ---
	const mismatches: DiffMismatch[] = [];
	let real = 0;
	let realMem = 0;
	let realRegOnly = 0;
	let benign = 0;
	const F3F5 = 0x08 | 0x20; // undocumented flag bits

	for (let c = 0; c < N; c++) {
		const o = runOracle(inputs[c], STEPS, PAIR); // real Z80 ground truth
		const mem = o.mem;

		// Compare memory
		let memDiffByte = -1;
		for (let i = 0; i < PAIR; i++) {
			const g = (gpuIo[c * WORDS + (i >> 2)] >>> ((i & 3) * 8)) & 0xff;
			if (g !== mem[i]) {
				memDiffByte = i;
				break;
			}
		}

		// Compare registers
		const g = gpuRegs.subarray(c * REGS_PER_CASE, c * REGS_PER_CASE + REGS_PER_CASE);
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
		const gpu: Record<string, number> = {
			a: g[0],
			f: g[1],
			b: g[2],
			c: g[3],
			d: g[4],
			e: g[5],
			h: g[6],
			l: g[7],
			sp: g[8] & 0xffff,
			pc: g[9] & 0xffff
		};
		const regDiffs: { name: string; gpu: number; cpu: number }[] = [];
		for (const k of REG_NAMES) {
			if (cpu[k] !== gpu[k]) regDiffs.push({ name: k, gpu: gpu[k], cpu: cpu[k] });
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
			}
			if (mismatches.length < 40) {
				mismatches.push({
					caseIdx: c,
					input: Array.from(inputs[c]),
					memDiffByte,
					regDiffs,
					benign: benignFlag,
					disasm: disassemble(inputs[c])
						.slice(0, 8)
						.map((l) => l.mnemonic)
				});
			}
		}
	}

	device.destroy();
	return { total: N, steps: STEPS, seed, real, realMem, realRegOnly, benign, mismatches, durationMs: t0 };
}

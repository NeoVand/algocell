// GPUTrainer — WebGPU reverse-mode BPTT trainer for the field-CA rule (see trainShader.ts).
// Trains in the browser, much faster than the JS CPU trainer, by batch-packing B samples
// (placement × case) and keeping params/grads/trajectory resident on the GPU. The only
// per-iter CPU work is building the per-sample port layout + seed (small). Validated against
// the finite-diff-checked CPU reference in /devcomp/traingpu.

import type { RuleConfig } from './rule';
import { trainShaderWGSL, type TrainDims } from './trainShader';

// A training sample: N input ports (each a bit into channel inCh[k]) and M output ports
// (each scored to a target). Single-output marker rules pass outPort/target; fixed multi-IO
// rules (gate/adder) pass outPorts/targets. Reactive: bits2/target(s)2 are the post-switch values.
export interface Sample {
	inPorts: number[]; inCh: number[]; bits: number[]; bits2?: number[];
	outPort?: number; target?: number; target2?: number;      // single-output form
	outPorts?: number[]; targets?: number[]; targets2?: number[]; // multi-output form
}

const KERNELS = ['fwd', 'seedGrad', 'bwd1', 'bwdGather', 'injectOut', 'gradW1', 'gradW2', 'gradBias', 'zeroGrad', 'gradNormSq', 'adam'] as const;
type Kernel = (typeof KERNELS)[number];

export class GPUTrainer {
	readonly cfg: RuleConfig;
	readonly B: number;
	readonly T: number;
	readonly aliveFrom: number;
	readonly whold: number;
	private device: GPUDevice;
	private pipelines!: Record<Kernel, GPUComputePipeline>;
	private bind!: GPUBindGroup;
	private buf: Record<string, GPUBuffer> = {};
	private ctrlHost = new ArrayBuffer(32);
	private ctrlU = new Uint32Array(this.ctrlHost);
	private ctrlF = new Float32Array(this.ctrlHost);
	private readback: GPUBuffer;
	private readonly BN: number;
	private readonly SCR: number;

	private constructor(device: GPUDevice, cfg: RuleConfig, dims: TrainDims) {
		this.device = device; this.cfg = cfg;
		this.B = dims.B; this.T = dims.T; this.aliveFrom = dims.aliveFrom; this.whold = dims.whold ?? 1;
		this.BN = cfg.N * this.B;
		this.SCR = cfg.C + 2 * cfg.HD + 2 * cfg.PERC;
		const d = device;
		const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
		const mk = (name: string, floats: number, usage: number) => (this.buf[name] = d.createBuffer({ size: floats * 4, usage }));
		mk('params', cfg.P, S | CD | CS);
		mk('optim', cfg.P * 3 + 4, S | CD | CS); // grad(P)|m(P)|v(P)|‖grad‖² scalar (+pad)
		mk('traj', (this.T + 1) * this.BN * cfg.C, S | CD | CS);
		mk('portsU', 2 * this.BN, S | CD);
		mk('portsF', 4 * this.BN, S | CD); // [inVal0|inVal1|tgt0|tgt1]
		mk('gsA', this.BN * cfg.C, S);
		mk('gsB', this.BN * cfg.C, S);
		mk('scratch', this.BN * this.SCR, S);
		this.buf.ctrl = d.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | CD });
		this.readback = d.createBuffer({ size: Math.max(cfg.P, this.BN * cfg.C) * 4, usage: GPUBufferUsage.MAP_READ | CD });

		const ro = 'read-only-storage' as const, st = 'storage' as const;
		const types: GPUBufferBindingType[] = [st, st, st, ro, ro, st, st, st]; // params (0) is written by Adam
		const entries: GPUBindGroupLayoutEntry[] = types.map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } }));
		entries.push({ binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
		const layout = d.createBindGroupLayout({ entries });
		const pl = d.createPipelineLayout({ bindGroupLayouts: [layout] });
		const module = d.createShaderModule({ code: trainShaderWGSL(cfg, dims) });
		this.pipelines = {} as Record<Kernel, GPUComputePipeline>;
		for (const k of KERNELS) this.pipelines[k] = d.createComputePipeline({ layout: pl, compute: { module, entryPoint: k } });
		const names = ['params', 'optim', 'traj', 'portsU', 'portsF', 'gsA', 'gsB', 'scratch', 'ctrl'];
		this.bind = d.createBindGroup({ layout, entries: names.map((n, binding) => ({ binding, resource: { buffer: this.buf[n] } })) });
	}

	static async create(cfg: RuleConfig, dims: TrainDims): Promise<GPUTrainer> {
		if (!navigator.gpu) throw new Error('WebGPU not available');
		const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
		if (!adapter) throw new Error('no GPU adapter');
		// the trajectory buffer can exceed the default 128MB binding cap → request the adapter's max
		const device = await adapter.requestDevice({
			requiredLimits: {
				maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
				maxBufferSize: adapter.limits.maxBufferSize
			}
		});
		device.addEventListener('uncapturederror', (e) => console.error('[gpu-train error]', (e as GPUUncapturedErrorEvent).error.message));
		const module = device.createShaderModule({ code: trainShaderWGSL(cfg, dims) });
		const info = await module.getCompilationInfo();
		const errs = info.messages.filter((m) => m.type === 'error');
		for (const m of errs) console.error(`[wgsl-train error] line ${m.lineNum}:${m.linePos} — ${m.message}`);
		if (errs.length) throw new Error(`WGSL compile: line ${errs[0].lineNum} — ${errs[0].message}`);
		device.pushErrorScope('validation');
		const t = new GPUTrainer(device, cfg, dims);
		const err = await device.popErrorScope();
		if (err) { console.error('[gpu-train create]', err.message); throw new Error('pipeline: ' + err.message); }
		return t;
	}

	/** Upload params; reset Adam moments (optim m/v regions) to 0. */
	setParams(par: Float32Array): void {
		if (par.length !== this.cfg.P) throw new Error(`params length ${par.length} !== ${this.cfg.P}`);
		this.device.queue.writeBuffer(this.buf.params, 0, par);
		this.device.queue.writeBuffer(this.buf.optim, 0, new Float32Array(this.cfg.P * 3 + 4)); // grad+m+v+norm = 0
	}

	/** Build per-sample port layout + seed traj[0] and upload. Samples vary freely per iter.
	 *  `tSwitch>0` makes the batch REACTIVE: input flips from bits→bits2 at that state, and the
	 *  output must re-settle from target→target2. tSwitch is shared across the batch (a uniform);
	 *  per-sample bits2/target2 default to bits/target (non-reactive) when absent. */
	setBatch(samples: Sample[], tSwitch = 0): void {
		if (samples.length !== this.B) throw new Error(`batch ${samples.length} !== ${this.B}`);
		const { N, C, markers } = this.cfg;
		const portsU = new Uint32Array(2 * this.BN), portsF = new Float32Array(4 * this.BN);
		const traj0 = new Float32Array(this.BN * C);
		for (let b = 0; b < this.B; b++) {
			const s = samples[b];
			const bits2 = s.bits2 ?? s.bits;
			const outPorts = s.outPorts ?? [s.outPort!];
			const targets = s.targets ?? [s.target!];
			const targets2 = s.targets2 ?? (s.target2 != null ? [s.target2] : targets);
			// uniform-alive interior (channels aliveFrom..C-1); signal/readout channels start at 0
			for (let y = 1; y < this.cfg.SH - 1; y++) for (let x = 1; x < this.cfg.SW - 1; x++) {
				const cell = b * N + (y * this.cfg.SW + x);
				for (let c = this.aliveFrom; c < C; c++) traj0[cell * C + c] = 1;
			}
			s.inPorts.forEach((p, k) => {
				const cell = b * N + p;
				portsU[cell] = s.inCh[k] + 1;             // channel+1 encoding
				portsF[cell] = s.bits[k];                 // inVal0
				portsF[this.BN + cell] = bits2[k];        // inVal1 (post-switch)
				if (markers) traj0[cell * C + 1] = 1;     // IN_MARK (markers rules only)
				traj0[cell * C + s.inCh[k]] = s.bits[k];  // inject the bit into its channel
			});
			outPorts.forEach((op, k) => {
				const oc = b * N + op;
				portsU[this.BN + oc] = 1;                 // isOutput
				portsF[2 * this.BN + oc] = targets[k];    // tgt0 (pre-switch answer)
				portsF[3 * this.BN + oc] = targets2[k];   // tgt1 (post-switch answer)
				if (markers) traj0[oc * C + 2] = 1;       // OUT_MARK (markers rules only)
			});
		}
		this.curTSwitch = tSwitch;
		this.device.queue.writeBuffer(this.buf.portsU, 0, portsU);
		this.device.queue.writeBuffer(this.buf.portsF, 0, portsF);
		this.device.queue.writeBuffer(this.buf.traj, 0, traj0);
	}

	private curTSwitch = 0;
	private curSeed = 0; // stochastic-update seed (varies per iter; fixed within a rollout so fwd/bwd masks match)
	private writeCtrl(t: number, dir: number, lr = 0, b1c = 1, b2c = 1): void {
		this.ctrlU[0] = t; this.ctrlU[1] = dir; this.ctrlU[2] = this.curTSwitch; this.ctrlU[3] = this.curSeed;
		this.ctrlF[4] = lr; this.ctrlF[5] = b1c; this.ctrlF[6] = b2c;
		this.device.queue.writeBuffer(this.buf.ctrl, 0, this.ctrlHost);
	}

	private pass(k: Kernel, threads: number): void {
		const enc = this.device.createCommandEncoder();
		const p = enc.beginComputePass();
		p.setPipeline(this.pipelines[k]);
		p.setBindGroup(0, this.bind);
		p.dispatchWorkgroups(Math.ceil(threads / 64));
		p.end();
		this.device.queue.submit([enc.finish()]);
	}
	// several kernels in one submit (ordered, memory-visible between passes) — all share ctrl
	private passes(list: [Kernel, number][]): void {
		const enc = this.device.createCommandEncoder();
		for (const [k, threads] of list) {
			const p = enc.beginComputePass();
			p.setPipeline(this.pipelines[k]);
			p.setBindGroup(0, this.bind);
			p.dispatchWorkgroups(Math.ceil(threads / 64));
			p.end();
		}
		this.device.queue.submit([enc.finish()]);
	}

	/** Forward rollout (fills traj) + seed grad + full backward (accumulates grad). No optimizer. */
	private forwardBackward(): void {
		const { HD, PERC, C, P } = this.cfg;
		for (let t = 0; t < this.T; t++) { this.writeCtrl(t, 0); this.pass('fwd', this.BN); }
		this.writeCtrl(0, 0); this.pass('seedGrad', this.BN);
		this.pass('zeroGrad', P);
		const wStartB = this.T - this.whold + 1, sw = this.curTSwitch;
		for (let t = this.T - 1; t >= 0; t--) {
			const dir = (this.T - 1 - t) % 2;
			this.writeCtrl(t, dir);
			const list: [Kernel, number][] = [['bwd1', this.BN], ['gradW1', HD * PERC], ['gradW2', C * HD], ['gradBias', HD + C], ['bwdGather', this.BN]];
			if (t >= wStartB || (sw > 0 && t >= sw - this.whold && t < sw)) list.push(['injectOut', this.BN]);
			this.passes(list);
		}
	}

	private forwardOnly(): void {
		for (let t = 0; t < this.T; t++) { this.writeCtrl(t, 0); this.pass('fwd', this.BN); }
	}

	/** Forward-only rollout on a batch → output-cell readouts (for held-out accuracy eval).
	 *  With tSwitch>0 the input flips mid-rollout, so the final readout tests reactive migration. */
	async evalOutputs(samples: Sample[], tSwitch = 0, seed = 0): Promise<number[]> {
		this.curSeed = seed >>> 0;
		this.setBatch(samples, tSwitch);
		this.forwardOnly();
		return this.readFinalOutputs(samples);
	}

	/** One optimizer step: forward+backward, grad-norm (for clipping), then Adam. `seed` drives the
	 *  stochastic-update masks (defaults to `it` so masks vary per iter). */
	trainStep(lr: number, it: number, seed = it): void {
		this.curSeed = seed >>> 0;
		this.forwardBackward();
		this.pass('gradNormSq', 1);
		const b1c = 1 - Math.pow(0.9, it), b2c = 1 - Math.pow(0.999, it);
		this.writeCtrl(0, 0, lr, b1c, b2c);
		this.pass('adam', this.cfg.P);
	}

	/** Forward+backward only, then read gradient (for validation vs the CPU reference). */
	async computeGrad(seed = 0): Promise<Float32Array> {
		this.curSeed = seed >>> 0;
		this.forwardBackward();
		return this.readFloats('optim', 0, this.cfg.P);
	}

	/** Read the FIRST output-cell readout for each sample (single-output convenience). */
	async readFinalOutputs(samples: Sample[]): Promise<number[]> {
		const { N, C } = this.cfg;
		const final = await this.readFloats('traj', this.T * this.BN * C, this.BN * C);
		return samples.map((s, b) => final[(b * N + (s.outPorts?.[0] ?? s.outPort!)) * C + 0]);
	}

	/** Read ALL output-cell readouts for each sample (multi-output eval). */
	async readFinalOutputsMulti(samples: Sample[]): Promise<number[][]> {
		const { N, C } = this.cfg;
		const final = await this.readFloats('traj', this.T * this.BN * C, this.BN * C);
		return samples.map((s, b) => (s.outPorts ?? [s.outPort!]).map((op) => final[(b * N + op) * C + 0]));
	}

	async readParams(): Promise<Float32Array> { return this.readFloats('params', 0, this.cfg.P); }

	private async readFloats(name: string, offsetFloats: number, count: number): Promise<Float32Array> {
		const enc = this.device.createCommandEncoder();
		enc.copyBufferToBuffer(this.buf[name], offsetFloats * 4, this.readback, 0, count * 4);
		this.device.queue.submit([enc.finish()]);
		await this.readback.mapAsync(GPUMapMode.READ, 0, count * 4);
		const out = new Float32Array(this.readback.getMappedRange(0, count * 4).slice(0));
		this.readback.unmap();
		return out;
	}

	destroy(): void { for (const b of Object.values(this.buf)) b.destroy(); this.readback.destroy(); }
}

// FieldCAEngine — WebGPU compute for the developmental-computation field CA.
// Ping-pong f32 state, one compute pass per step. Dimensions come from a
// RuleConfig, so one engine class serves any grid/channel size. Rendering is
// added elsewhere; this class owns the simulation + readback so it can be
// validated headlessly against the CPU reference in `rule.ts`.

import type { RuleConfig } from './rule';
import { fieldShaderWGSL } from './shader';

export class FieldCAEngine {
	device: GPUDevice;
	cfg: RuleConfig;
	private stateA: GPUBuffer;
	private stateB: GPUBuffer;
	private params: GPUBuffer;
	private isInput: GPUBuffer;
	private inputVal: GPUBuffer;
	private damageKeep: GPUBuffer;
	private ctrl: GPUBuffer;
	private pipeline: GPUComputePipeline;
	private bindAB: GPUBindGroup;
	private bindBA: GPUBindGroup;
	private readbackBuf: GPUBuffer;
	private cur: 0 | 1 = 0;
	private readonly stateBytes: number;
	private zeroCell: Float32Array;

	private constructor(device: GPUDevice, cfg: RuleConfig) {
		this.device = device;
		this.cfg = cfg;
		this.stateBytes = cfg.N * cfg.C * 4;
		this.zeroCell = new Float32Array(cfg.C);
		const d = device;
		const mk = (bytes: number, usage: number) => d.createBuffer({ size: bytes, usage });
		const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
		this.stateA = mk(this.stateBytes, ST);
		this.stateB = mk(this.stateBytes, ST);
		this.params = mk(cfg.P * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
		this.isInput = mk(cfg.N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
		this.inputVal = mk(cfg.N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
		this.damageKeep = mk(cfg.N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
		this.ctrl = mk(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
		this.readbackBuf = mk(this.stateBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

		const module = d.createShaderModule({ code: fieldShaderWGSL(cfg) });
		this.pipeline = d.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'step' } });
		const layout = this.pipeline.getBindGroupLayout(0);
		const bind = (inBuf: GPUBuffer, outBuf: GPUBuffer): GPUBindGroup =>
			d.createBindGroup({
				layout,
				entries: [
					{ binding: 0, resource: { buffer: inBuf } },
					{ binding: 1, resource: { buffer: outBuf } },
					{ binding: 2, resource: { buffer: this.params } },
					{ binding: 3, resource: { buffer: this.isInput } },
					{ binding: 4, resource: { buffer: this.inputVal } },
					{ binding: 5, resource: { buffer: this.damageKeep } },
					{ binding: 6, resource: { buffer: this.ctrl } }
				]
			});
		this.bindAB = bind(this.stateA, this.stateB);
		this.bindBA = bind(this.stateB, this.stateA);
	}

	static async create(cfg: RuleConfig): Promise<FieldCAEngine> {
		if (!navigator.gpu) throw new Error('WebGPU not available');
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) throw new Error('no GPU adapter');
		const device = await adapter.requestDevice();
		device.addEventListener('uncapturederror', (e) => console.error('[gpu error]', (e as GPUUncapturedErrorEvent).error.message));
		const module = device.createShaderModule({ code: fieldShaderWGSL(cfg) });
		const info = await module.getCompilationInfo();
		const errs = info.messages.filter((m) => m.type === 'error');
		for (const m of errs) console.error(`[wgsl error] line ${m.lineNum}:${m.linePos} — ${m.message}`);
		if (errs.length) throw new Error(`WGSL compile: line ${errs[0].lineNum} — ${errs[0].message}`);
		device.pushErrorScope('validation');
		const engine = new FieldCAEngine(device, cfg);
		const err = await device.popErrorScope();
		if (err) { console.error('[gpu create error]', err.message); throw new Error('pipeline: ' + err.message); }
		return engine;
	}

	setParams(par: Float32Array): void { this.device.queue.writeBuffer(this.params, 0, par); }
	setInputs(isInput: Uint32Array, inputVal: Float32Array): void {
		this.device.queue.writeBuffer(this.isInput, 0, isInput);
		this.device.queue.writeBuffer(this.inputVal, 0, inputVal);
	}
	setDamageKeep(keep: Uint32Array): void { this.device.queue.writeBuffer(this.damageKeep, 0, keep); }

	seed(state: Float32Array): void {
		this.device.queue.writeBuffer(this.stateA, 0, state);
		this.device.queue.writeBuffer(this.stateB, 0, new Float32Array(this.cfg.N * this.cfg.C));
		this.cur = 0;
	}

	step(applyDamage = false): void {
		this.device.queue.writeBuffer(this.ctrl, 0, new Uint32Array([applyDamage ? 1 : 0, 0, 0, 0]));
		const enc = this.device.createCommandEncoder();
		const pass = enc.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.cur === 0 ? this.bindAB : this.bindBA);
		pass.dispatchWorkgroups(Math.ceil(this.cfg.SW / 8), Math.ceil(this.cfg.SH / 8));
		pass.end();
		this.device.queue.submit([enc.finish()]);
		this.cur = this.cur === 0 ? 1 : 0;
	}

	private currentBuffer(): GPUBuffer { return this.cur === 0 ? this.stateA : this.stateB; }

	/** Destroy a cell in the LIVE field (the damage brush); the rule regrows it. */
	damageCell(cell: number): void {
		this.device.queue.writeBuffer(this.currentBuffer(), cell * this.cfg.C * 4, this.zeroCell);
	}

	async readState(): Promise<Float32Array> {
		const enc = this.device.createCommandEncoder();
		enc.copyBufferToBuffer(this.currentBuffer(), 0, this.readbackBuf, 0, this.stateBytes);
		this.device.queue.submit([enc.finish()]);
		await this.readbackBuf.mapAsync(GPUMapMode.READ);
		const out = new Float32Array(this.readbackBuf.getMappedRange().slice(0));
		this.readbackBuf.unmap();
		return out;
	}

	destroy(): void {
		for (const b of [this.stateA, this.stateB, this.params, this.isInput, this.inputVal, this.damageKeep, this.ctrl, this.readbackBuf]) b.destroy();
	}
}

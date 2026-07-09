// FieldCAEngine — WebGPU compute for the developmental-computation field CA.
// Ping-pong f32 state, one compute pass per step. Rendering is added in a later
// stage; this class owns the simulation + readback so it can be validated
// headlessly against the CPU reference in `rule.ts`.

import { N, C, SW, SH, P } from './rule';
import { fieldShaderWGSL } from './shader';

export class FieldCAEngine {
	device: GPUDevice;
	private stateA: GPUBuffer;
	private stateB: GPUBuffer;
	private params: GPUBuffer;
	private isInput: GPUBuffer;
	private inputVal: GPUBuffer;
	private damageKeep: GPUBuffer;
	private ctrl: GPUBuffer;
	private module!: GPUShaderModule;
	private pipeline: GPUComputePipeline;
	private bindAB: GPUBindGroup; // in=A, out=B
	private bindBA: GPUBindGroup; // in=B, out=A
	private readback: GPUBuffer;
	private cur: 0 | 1 = 0; // which buffer holds the latest state (0=A, 1=B)
	private readonly stateBytes = N * C * 4;

	private constructor(device: GPUDevice) {
		this.device = device;
		const d = device;
		const mk = (bytes: number, usage: number) => d.createBuffer({ size: bytes, usage });
		const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
		this.stateA = mk(this.stateBytes, ST);
		this.stateB = mk(this.stateBytes, ST);
		this.params = mk(P * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
		this.isInput = mk(N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
		this.inputVal = mk(N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
		this.damageKeep = mk(N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
		this.ctrl = mk(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
		this.readback = mk(this.stateBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

		this.module = d.createShaderModule({ code: fieldShaderWGSL() });
		this.pipeline = d.createComputePipeline({ layout: 'auto', compute: { module: this.module, entryPoint: 'step' } });
		const layout = this.pipeline.getBindGroupLayout(0);
		const common = (inBuf: GPUBuffer, outBuf: GPUBuffer): GPUBindGroup =>
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
		this.bindAB = common(this.stateA, this.stateB);
		this.bindBA = common(this.stateB, this.stateA);
	}

	static async create(): Promise<FieldCAEngine> {
		if (!navigator.gpu) throw new Error('WebGPU not available');
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) throw new Error('no GPU adapter');
		const device = await adapter.requestDevice();
		device.addEventListener('uncapturederror', (e) => console.error('[gpu error]', (e as GPUUncapturedErrorEvent).error.message));
		const module = device.createShaderModule({ code: fieldShaderWGSL() });
		const info = await module.getCompilationInfo();
		const errs = info.messages.filter((m) => m.type === 'error');
		for (const m of errs) console.error(`[wgsl error] line ${m.lineNum}:${m.linePos} — ${m.message}`);
		if (errs.length) throw new Error(`WGSL compile: line ${errs[0].lineNum} — ${errs[0].message}`);
		device.pushErrorScope('validation');
		const engine = new FieldCAEngine(device);
		const err = await device.popErrorScope();
		if (err) { console.error('[gpu create error]', err.message); throw new Error('pipeline: ' + err.message); }
		return engine;
	}

	setParams(par: Float32Array): void {
		this.device.queue.writeBuffer(this.params, 0, par);
	}

	/** Per-cell input mask (1 if input) + clamped ch0 value. */
	setInputs(isInput: Uint32Array, inputVal: Float32Array): void {
		this.device.queue.writeBuffer(this.isInput, 0, isInput);
		this.device.queue.writeBuffer(this.inputVal, 0, inputVal);
	}

	/** Per-cell keep-mask (1 keep, 0 destroy) used on the next damage step. */
	setDamageKeep(keep: Uint32Array): void {
		this.device.queue.writeBuffer(this.damageKeep, 0, keep);
	}

	/** Load an initial field into A (the latest buffer) and clear B. */
	seed(state: Float32Array): void {
		this.device.queue.writeBuffer(this.stateA, 0, state);
		this.device.queue.writeBuffer(this.stateB, 0, new Float32Array(N * C));
		this.cur = 0;
	}

	/** Advance one CA step. `applyDamage` zeroes damageKeep==0 cells this step. */
	step(applyDamage = false): void {
		this.device.queue.writeBuffer(this.ctrl, 0, new Uint32Array([applyDamage ? 1 : 0, 0, 0, 0]));
		const enc = this.device.createCommandEncoder();
		const pass = enc.beginComputePass();
		pass.setPipeline(this.pipeline);
		pass.setBindGroup(0, this.cur === 0 ? this.bindAB : this.bindBA);
		pass.dispatchWorkgroups(Math.ceil(SW / 8), Math.ceil(SH / 8));
		pass.end();
		this.device.queue.submit([enc.finish()]);
		this.cur = this.cur === 0 ? 1 : 0;
	}

	private currentBuffer(): GPUBuffer {
		return this.cur === 0 ? this.stateA : this.stateB;
	}

	/** Read the latest field back to the CPU as a Float32Array (length N*C). */
	async readState(): Promise<Float32Array> {
		const enc = this.device.createCommandEncoder();
		enc.copyBufferToBuffer(this.currentBuffer(), 0, this.readback, 0, this.stateBytes);
		this.device.queue.submit([enc.finish()]);
		await this.readback.mapAsync(GPUMapMode.READ);
		const out = new Float32Array(this.readback.getMappedRange().slice(0));
		this.readback.unmap();
		return out;
	}

	/** Debug: read the params buffer back (to verify upload). */
	async debugReadParams(): Promise<Float32Array> {
		const rb = this.device.createBuffer({ size: P * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
		const enc = this.device.createCommandEncoder();
		enc.copyBufferToBuffer(this.params, 0, rb, 0, P * 4);
		this.device.queue.submit([enc.finish()]);
		await rb.mapAsync(GPUMapMode.READ);
		const out = new Float32Array(rb.getMappedRange().slice(0));
		rb.unmap(); rb.destroy();
		return out;
	}

	destroy(): void {
		for (const b of [this.stateA, this.stateB, this.params, this.isInput, this.inputVal, this.damageKeep, this.ctrl, this.readback]) b.destroy();
	}
}

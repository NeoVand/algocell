// WebGPU Engine: orchestrates all GPU compute and rendering
// All simulation runs entirely on GPU via compute shaders

import {
	SOUP_WIDTH,
	SOUP_HEIGHT,
	TAPE_LENGTH,
	PAIR_LENGTH,
	SOUP_SIZE,
	CELL_COUNT,
	MAX_BATCH_PAIR_N,
	Z80_STEPS
} from '$lib/sim/constants';
import { SplitMix64 } from '$lib/sim/prng';
import { createColormap } from '$lib/colormap';
import { simShader, renderShader } from './shaders';

interface SimParams {
	soup_width: number;
	soup_height: number;
	tape_length: number;
	pair_length: number;
	pair_count: number;
	mutation_count: number;
	z80_steps: number;
	batch_seed: number;
}

export class GPUEngine {
	private device!: GPUDevice;
	private context!: GPUCanvasContext;
	private format!: GPUTextureFormat;

	// Storage buffers
	private soupBuffer!: GPUBuffer;
	private pairsBuffer!: GPUBuffer;
	private pairDataBuffer!: GPUBuffer;
	private writeCountsBuffer!: GPUBuffer;
	private rngStatesBuffer!: GPUBuffer;
	private paramsBuffer!: GPUBuffer;
	private pairActiveBuffer!: GPUBuffer;
	private byteCountsBuffer!: GPUBuffer;
	private collisionMaskBuffer!: GPUBuffer;

	// Render buffers
	private colormapBuffer!: GPUBuffer;
	private renderParamsBuffer!: GPUBuffer;
	private traceImageBuffer!: GPUBuffer;

	// Staging buffer for readback
	private byteCountsStagingBuffer!: GPUBuffer;
	private soupStagingBuffer!: GPUBuffer;

	// Compute pipelines
	private clearCollisionPipeline!: GPUComputePipeline;
	private prepareBatchPipeline!: GPUComputePipeline;
	private z80ExecutePipeline!: GPUComputePipeline;
	private absorbPipeline!: GPUComputePipeline;
	private mutatePipeline!: GPUComputePipeline;
	private countBytesPipeline!: GPUComputePipeline;
	private clearByteCountsPipeline!: GPUComputePipeline;

	// Render pipeline
	private renderPipeline!: GPURenderPipeline;

	// Bind groups
	private simBindGroup!: GPUBindGroup;
	private renderBindGroup!: GPUBindGroup;

	// View state (zoom/pan)
	view = { zoom: SOUP_WIDTH, offsetX: 0, offsetY: 0 };

	// State
	private batchIndex = 0;
	private cpuRng: SplitMix64;
	private _noiseExp = 4; // 1/2^4 = 1/16
	private _pairCount = MAX_BATCH_PAIR_N;
	private _z80Steps = Z80_STEPS;
	private hoverCell = -1;
	private showAverage = 0;
	private colormap: Uint32Array;

	// Stats (read back from GPU periodically)
	byteCounts = new Uint32Array(256);
	opsPerSec = 0;
	private lastStatsTime = 0;
	private statsOpsAccum = 0;

	constructor(private seed: number) {
		this.cpuRng = new SplitMix64(seed);
		this.colormap = createColormap('rainbow');
	}

	get noiseExp(): number {
		return this._noiseExp;
	}
	set noiseExp(v: number) {
		this._noiseExp = v;
	}

	get pairCount(): number {
		return this._pairCount;
	}
	set pairCount(v: number) {
		this._pairCount = Math.min(v, MAX_BATCH_PAIR_N);
	}

	get z80Steps(): number {
		return this._z80Steps;
	}
	set z80Steps(v: number) {
		this._z80Steps = Math.max(16, Math.min(1024, v));
	}

	get batchCount(): number {
		return this.batchIndex;
	}

	async init(canvas: HTMLCanvasElement): Promise<boolean> {
		if (!navigator.gpu) {
			console.error('WebGPU not supported');
			return false;
		}

		const adapter = await navigator.gpu.requestAdapter({
			powerPreference: 'high-performance'
		});
		if (!adapter) {
			console.error('No GPU adapter found');
			return false;
		}

		this.device = await adapter.requestDevice({
			requiredLimits: {
				maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
				maxBufferSize: adapter.limits.maxBufferSize
			}
		});

		this.context = canvas.getContext('webgpu')!;
		this.format = navigator.gpu.getPreferredCanvasFormat();
		this.context.configure({
			device: this.device,
			format: this.format,
			alphaMode: 'premultiplied'
		});

		this.createBuffers();
		this.createPipelines();
		this.initSoup();
		this.uploadColormap();

		return true;
	}

	private createBuffers(): void {
		const dev = this.device;

		// Soup: 640,000 bytes = 160,000 u32s
		const soupWords = CELL_COUNT * (TAPE_LENGTH / 4);
		this.soupBuffer = dev.createBuffer({
			size: soupWords * 4,
			usage:
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});

		// Pairs: 2 u32s per pair
		this.pairsBuffer = dev.createBuffer({
			size: MAX_BATCH_PAIR_N * 2 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		// Pair data: 8 u32s (32 bytes) per pair
		this.pairDataBuffer = dev.createBuffer({
			size: MAX_BATCH_PAIR_N * 8 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		// Write counts: 2 u32s per pair
		this.writeCountsBuffer = dev.createBuffer({
			size: MAX_BATCH_PAIR_N * 2 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		// RNG states: 1 u32 per pair
		this.rngStatesBuffer = dev.createBuffer({
			size: MAX_BATCH_PAIR_N * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		// Params uniform
		this.paramsBuffer = dev.createBuffer({
			size: 32, // 8 u32s
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		// Pair active flags
		this.pairActiveBuffer = dev.createBuffer({
			size: MAX_BATCH_PAIR_N * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		// Byte counts: 256 atomic u32s
		this.byteCountsBuffer = dev.createBuffer({
			size: 256 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});

		// Collision mask: CELL_COUNT atomic u32s
		this.collisionMaskBuffer = dev.createBuffer({
			size: CELL_COUNT * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		// Staging buffers for readback
		this.byteCountsStagingBuffer = dev.createBuffer({
			size: 256 * 4,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		});

		this.soupStagingBuffer = dev.createBuffer({
			size: soupWords * 4,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		});

		// Render buffers
		this.colormapBuffer = dev.createBuffer({
			size: 256 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		this.renderParamsBuffer = dev.createBuffer({
			size: 48, // RenderParams struct (12 fields)
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		this.traceImageBuffer = dev.createBuffer({
			size: PAIR_LENGTH * Z80_STEPS * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
	}

	private createPipelines(): void {
		const dev = this.device;

		// Simulation shader module
		const simModule = dev.createShaderModule({ code: simShader });

		// Simulation bind group layout
		const simBindGroupLayout = dev.createBindGroupLayout({
			entries: [
				{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{
					binding: 5,
					visibility: GPUShaderStage.COMPUTE,
					buffer: { type: 'uniform' }
				},
				{ binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
			]
		});

		const simPipelineLayout = dev.createPipelineLayout({
			bindGroupLayouts: [simBindGroupLayout]
		});

		this.simBindGroup = dev.createBindGroup({
			layout: simBindGroupLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.soupBuffer } },
				{ binding: 1, resource: { buffer: this.pairsBuffer } },
				{ binding: 2, resource: { buffer: this.pairDataBuffer } },
				{ binding: 3, resource: { buffer: this.writeCountsBuffer } },
				{ binding: 4, resource: { buffer: this.rngStatesBuffer } },
				{ binding: 5, resource: { buffer: this.paramsBuffer } },
				{ binding: 6, resource: { buffer: this.pairActiveBuffer } },
				{ binding: 7, resource: { buffer: this.byteCountsBuffer } },
				{ binding: 8, resource: { buffer: this.collisionMaskBuffer } }
			]
		});

		const makeComputePipeline = (entryPoint: string) =>
			dev.createComputePipeline({
				layout: simPipelineLayout,
				compute: { module: simModule, entryPoint }
			});

		this.clearCollisionPipeline = makeComputePipeline('clear_collision');
		this.prepareBatchPipeline = makeComputePipeline('prepare_batch');
		this.z80ExecutePipeline = makeComputePipeline('z80_execute_batch');
		this.absorbPipeline = makeComputePipeline('absorb_results');
		this.mutatePipeline = makeComputePipeline('mutate_soup');
		this.countBytesPipeline = makeComputePipeline('count_bytes');
		this.clearByteCountsPipeline = makeComputePipeline('clear_byte_counts');

		// Render pipeline
		const renderModule = dev.createShaderModule({ code: renderShader });

		const renderBindGroupLayout = dev.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					buffer: { type: 'read-only-storage' }
				},
				{
					binding: 1,
					visibility: GPUShaderStage.FRAGMENT,
					buffer: { type: 'read-only-storage' }
				},
				{
					binding: 2,
					visibility: GPUShaderStage.FRAGMENT,
					buffer: { type: 'uniform' }
				},
				{
					binding: 3,
					visibility: GPUShaderStage.FRAGMENT,
					buffer: { type: 'read-only-storage' }
				}
			]
		});

		this.renderPipeline = dev.createRenderPipeline({
			layout: dev.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
			vertex: { module: renderModule, entryPoint: 'vs_main' },
			fragment: {
				module: renderModule,
				entryPoint: 'fs_main',
				targets: [{ format: this.format }]
			},
			primitive: { topology: 'triangle-list' }
		});

		this.renderBindGroup = dev.createBindGroup({
			layout: renderBindGroupLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.soupBuffer } },
				{ binding: 1, resource: { buffer: this.colormapBuffer } },
				{ binding: 2, resource: { buffer: this.renderParamsBuffer } },
				{ binding: 3, resource: { buffer: this.traceImageBuffer } }
			]
		});
	}

	private initSoup(): void {
		// Generate random soup data on CPU and upload
		const soupData = new Uint8Array(SOUP_SIZE);
		const rng = new SplitMix64(this.seed);
		for (let i = 0; i < SOUP_SIZE; i += 4) {
			const r = rng.nextU32();
			soupData[i] = r & 0xff;
			if (i + 1 < SOUP_SIZE) soupData[i + 1] = (r >> 8) & 0xff;
			if (i + 2 < SOUP_SIZE) soupData[i + 2] = (r >> 16) & 0xff;
			if (i + 3 < SOUP_SIZE) soupData[i + 3] = (r >> 24) & 0xff;
		}
		this.device.queue.writeBuffer(this.soupBuffer, 0, soupData);
		this.batchIndex = 0;
	}

	private uploadColormap(): void {
		this.device.queue.writeBuffer(this.colormapBuffer, 0, this.colormap.buffer);
	}

	updateColormap(colormap: Uint32Array): void {
		this.colormap = colormap;
		this.uploadColormap();
	}

	reset(seed: number): void {
		this.seed = seed;
		this.cpuRng = new SplitMix64(seed);
		this.initSoup();
		this.batchIndex = 0;
		this.opsPerSec = 0;
		this.lastStatsTime = 0;
		this.statsOpsAccum = 0;
		this.byteCounts.fill(0);
	}

	setHoverCell(cell: number): void {
		this.hoverCell = cell;
	}

	setShowAverage(show: boolean): void {
		this.showAverage = show ? 1 : 0;
	}

	// Run one simulation step (all on GPU)
	simulateStep(): void {
		const noiseCoef = 1 / Math.pow(2, this._noiseExp);
		const mutationCount = Math.floor(this._pairCount * noiseCoef);

		// Update params
		const paramsData = new Uint32Array([
			SOUP_WIDTH,
			SOUP_HEIGHT,
			TAPE_LENGTH,
			PAIR_LENGTH,
			this._pairCount,
			mutationCount,
			this._z80Steps,
			this.cpuRng.nextU32() // batch_seed - different each batch
		]);
		this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

		const encoder = this.device.createCommandEncoder();

		// 1. Clear collision mask
		{
			const pass = encoder.beginComputePass();
			pass.setPipeline(this.clearCollisionPipeline);
			pass.setBindGroup(0, this.simBindGroup);
			pass.dispatchWorkgroups(Math.ceil(CELL_COUNT / 256));
			pass.end();
		}

		// 2. Prepare batch (generate pairs, claim cells, copy data)
		{
			const pass = encoder.beginComputePass();
			pass.setPipeline(this.prepareBatchPipeline);
			pass.setBindGroup(0, this.simBindGroup);
			pass.dispatchWorkgroups(Math.ceil(this._pairCount / 64));
			pass.end();
		}

		// 3. Z80 execute
		{
			const pass = encoder.beginComputePass();
			pass.setPipeline(this.z80ExecutePipeline);
			pass.setBindGroup(0, this.simBindGroup);
			pass.dispatchWorkgroups(Math.ceil(this._pairCount / 64));
			pass.end();
		}

		// 4. Absorb results
		{
			const pass = encoder.beginComputePass();
			pass.setPipeline(this.absorbPipeline);
			pass.setBindGroup(0, this.simBindGroup);
			pass.dispatchWorkgroups(Math.ceil(this._pairCount / 64));
			pass.end();
		}

		// 5. Mutate
		if (mutationCount > 0) {
			const pass = encoder.beginComputePass();
			pass.setPipeline(this.mutatePipeline);
			pass.setBindGroup(0, this.simBindGroup);
			pass.dispatchWorkgroups(Math.ceil(mutationCount / 64));
			pass.end();
		}

		this.device.queue.submit([encoder.finish()]);
		this.batchIndex++;

		// Update ops/sec
		const now = performance.now();
		if (this.lastStatsTime > 0) {
			const dt = (now - this.lastStatsTime) / 1000;
			if (dt > 0) {
				const ops = this._pairCount * this._z80Steps;
				this.statsOpsAccum = this.statsOpsAccum * 0.9 + (ops / dt) * 0.1;
				this.opsPerSec = this.statsOpsAccum;
			}
		}
		this.lastStatsTime = now;
	}

	// Read byte counts from GPU (async)
	async readStats(): Promise<Uint32Array> {
		const encoder = this.device.createCommandEncoder();

		// Clear byte counts
		{
			const pass = encoder.beginComputePass();
			pass.setPipeline(this.clearByteCountsPipeline);
			pass.setBindGroup(0, this.simBindGroup);
			pass.dispatchWorkgroups(1);
			pass.end();
		}

		// Count bytes
		{
			const soupWords = CELL_COUNT * (TAPE_LENGTH / 4);
			const pass = encoder.beginComputePass();
			pass.setPipeline(this.countBytesPipeline);
			pass.setBindGroup(0, this.simBindGroup);
			pass.dispatchWorkgroups(Math.ceil(soupWords / 256));
			pass.end();
		}

		// Copy to staging
		encoder.copyBufferToBuffer(this.byteCountsBuffer, 0, this.byteCountsStagingBuffer, 0, 256 * 4);

		this.device.queue.submit([encoder.finish()]);

		// Read back
		await this.byteCountsStagingBuffer.mapAsync(GPUMapMode.READ);
		const data = new Uint32Array(this.byteCountsStagingBuffer.getMappedRange().slice(0));
		this.byteCountsStagingBuffer.unmap();

		this.byteCounts.set(data);
		return data;
	}

	// Read soup data from GPU for CPU-side trace/disassembly
	private _soupMapPending = false;
	async readSoupData(): Promise<Uint8Array> {
		if (this._soupMapPending) return new Uint8Array(0);
		this._soupMapPending = true;
		try {
			const soupBytes = CELL_COUNT * TAPE_LENGTH;
			const encoder = this.device.createCommandEncoder();
			encoder.copyBufferToBuffer(this.soupBuffer, 0, this.soupStagingBuffer, 0, soupBytes);
			this.device.queue.submit([encoder.finish()]);

			await this.soupStagingBuffer.mapAsync(GPUMapMode.READ);
			const data = new Uint8Array(this.soupStagingBuffer.getMappedRange().slice(0));
			this.soupStagingBuffer.unmap();
			return data;
		} finally {
			this._soupMapPending = false;
		}
	}

	zoomAt(screenX: number, screenY: number, canvasW: number, canvasH: number, factor: number): void {
		const aspect = canvasW / canvasH;
		const cellsX = this.view.zoom;
		const cellsY = this.view.zoom / aspect;

		// Grid position under cursor before zoom
		const gridX = (screenX / canvasW) * cellsX + this.view.offsetX;
		const gridY = (screenY / canvasH) * cellsY + this.view.offsetY;

		const maxZoom = Math.max(SOUP_WIDTH, SOUP_HEIGHT) * 2;
		const newZoom = Math.max(4, Math.min(maxZoom, this.view.zoom * factor));
		const newCellsX = newZoom;
		const newCellsY = newZoom / aspect;

		// Keep grid position under cursor at same screen position
		this.view.zoom = newZoom;
		this.view.offsetX = gridX - (screenX / canvasW) * newCellsX;
		this.view.offsetY = gridY - (screenY / canvasH) * newCellsY;
	}

	pan(deltaX: number, deltaY: number, canvasW: number, canvasH: number): void {
		const aspect = canvasW / canvasH;
		const cellsX = this.view.zoom;
		const cellsY = this.view.zoom / aspect;
		this.view.offsetX -= (deltaX / canvasW) * cellsX;
		this.view.offsetY -= (deltaY / canvasH) * cellsY;
	}

	resetView(): void {
		this.view = { zoom: SOUP_WIDTH, offsetX: 0, offsetY: 0 };
	}

	// Convert screen position to cell index
	screenToCell(screenX: number, screenY: number, canvasW: number, canvasH: number): number {
		const aspect = canvasW / canvasH;
		const cellsX = this.view.zoom;
		const cellsY = this.view.zoom / aspect;
		const gridX = (screenX / canvasW) * cellsX + this.view.offsetX;
		const gridY = (screenY / canvasH) * cellsY + this.view.offsetY;
		const cx = Math.floor(gridX);
		const cy = Math.floor(gridY);
		if (cx < 0 || cx >= SOUP_WIDTH || cy < 0 || cy >= SOUP_HEIGHT) return -1;
		return cy * SOUP_WIDTH + cx;
	}

	// Render the soup to the canvas
	render(canvas: HTMLCanvasElement): void {
		const tileSize = 4;
		const renderParams = new ArrayBuffer(48); // 12 fields * 4 bytes
		const view = new DataView(renderParams);
		view.setUint32(0, SOUP_WIDTH, true);
		view.setUint32(4, SOUP_HEIGHT, true);
		view.setUint32(8, tileSize, true);
		view.setFloat32(12, canvas.width, true);
		view.setFloat32(16, canvas.height, true);
		view.setUint32(20, this.showAverage, true);
		view.setInt32(24, this.hoverCell, true);
		view.setFloat32(28, this.view.zoom, true);
		view.setFloat32(32, this.view.offsetX, true);
		view.setFloat32(36, this.view.offsetY, true);
		view.setUint32(40, 0, true); // pad1
		view.setUint32(44, 0, true); // pad2

		this.device.queue.writeBuffer(this.renderParamsBuffer, 0, renderParams);

		const encoder = this.device.createCommandEncoder();
		const textureView = this.context.getCurrentTexture().createView();

		const pass = encoder.beginRenderPass({
			colorAttachments: [
				{
					view: textureView,
					clearValue: { r: 0.03, g: 0.03, b: 0.05, a: 1 },
					loadOp: 'clear',
					storeOp: 'store'
				}
			]
		});

		pass.setPipeline(this.renderPipeline);
		pass.setBindGroup(0, this.renderBindGroup);
		pass.draw(6); // Full-screen quad (2 triangles)
		pass.end();

		this.device.queue.submit([encoder.finish()]);
	}

	destroy(): void {
		this.soupBuffer?.destroy();
		this.pairsBuffer?.destroy();
		this.pairDataBuffer?.destroy();
		this.writeCountsBuffer?.destroy();
		this.rngStatesBuffer?.destroy();
		this.paramsBuffer?.destroy();
		this.pairActiveBuffer?.destroy();
		this.byteCountsBuffer?.destroy();
		this.collisionMaskBuffer?.destroy();
		this.colormapBuffer?.destroy();
		this.renderParamsBuffer?.destroy();
		this.traceImageBuffer?.destroy();
		this.byteCountsStagingBuffer?.destroy();
		this.soupStagingBuffer?.destroy();
	}
}

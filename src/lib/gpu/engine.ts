// WebGPU Engine: orchestrates all GPU compute and rendering
// All simulation runs entirely on GPU via compute shaders

import {
	SOUP_WIDTH,
	MAX_BATCH_PAIR_N,
	Z80_STEPS,
	HEX_HEIGHT_RATIO,
	DEFAULT_GRID_CONFIG,
	getTapeLength,
	getPairLength,
	type GridConfig,
	type GridType
} from '$lib/sim/constants';
import { SplitMix64 } from '$lib/sim/prng';
import { createColormap } from '$lib/colormap';
import { createSimShader, createRenderShader, createFieldShader } from './shaders';
import {
	chanceThreshold,
	createFreeShader,
	createHeads,
	FACING_MASKS,
	HALT_RULES,
	headCount,
	HEAD_WORDS,
	LOG_WORDS,
	mutationChance,
	NO_CLAIM
} from './free';
import { suppressionMasks, emptySuppression, type SuppressSets } from '$lib/z80-opcodes';
import { ORIGINAL_SETTINGS, validateSettings, createInitialSoup } from './machine';
import { ablationMasks } from '../lab/ablation';
import { runCulture } from './culture';
import { runAssays, type AssayResult, type AssayOptions } from './assays';
import {
	measureRegion,
	normalizeRegion,
	type FieldMode,
	type Marker,
	type CellRegion,
	type RegionMeasurement
} from '../observation';
import type { LabSettings, InitialCondition, CultureResult } from '../lab/types';
import { withInteractionDefaults } from '../lab/types';

let nextEngineId = 0;
let nextDeviceId = 0;

export interface EngineDiagnostics {
	engineId: number;
	deviceId: number;
	deviceInitializations: number;
	resetCount: number;
	bufferRevision: number;
	pipelineRevision: number;
}

export class GPUEngine {
	private readonly lifecycle: EngineDiagnostics = {
		engineId: ++nextEngineId,
		deviceId: 0,
		deviceInitializations: 0,
		resetCount: 0,
		bufferRevision: 0,
		pipelineRevision: 0
	};
	private configurationQueue: Promise<void> = Promise.resolve();
	private readbackBarrier: Promise<void> | null = null;
	private pendingReadbacks = 0;
	private readbackWaiters: (() => void)[] = [];
	private markerPipeline: GPUComputePipeline | null = null;
	private device!: GPUDevice;
	private context!: GPUCanvasContext;
	private format!: GPUTextureFormat;

	// Storage buffers
	private soupBuffer!: GPUBuffer;
	/** Reversible rule: each cell's tape before its last event. Equal to the soup when at rest. */
	private beforeBuffer!: GPUBuffer;
	// Free-running world (interaction `free`): persistent processors on the sheet of bytes.
	private freeModule: GPUShaderModule | null = null;
	private freePipelines: { step: GPUComputePipeline; apply: GPUComputePipeline } | null = null;
	private freeBindGroup: GPUBindGroup | null = null;
	private headsBuffer: GPUBuffer | null = null;
	private headLogBuffer: GPUBuffer | null = null;
	private claimsBuffer: GPUBuffer | null = null;
	private freeParamsBuffer: GPUBuffer | null = null;
	private freeHeads = 0;
	private freeSignature = '';
	private freeParams = new Uint32Array(8);
	/**
	 * Free-running world: ticks queued by one `simulateStep`. A tick is only a few thousand
	 * instructions, so the app's frame loop asks for them in bunches. It changes nothing about the
	 * world, only how often the screen is redrawn per tick.
	 */
	freeTicksPerStep = 32;
	private pairsBuffer!: GPUBuffer;
	private pairDataBuffer!: GPUBuffer;
	private writeCountsBuffer!: GPUBuffer;
	private rngStatesBuffer!: GPUBuffer;
	private paramsBuffer!: GPUBuffer;
	private pairActiveBuffer!: GPUBuffer;
	private byteCountsBuffer!: GPUBuffer;
	private collisionMaskBuffer!: GPUBuffer;
	private cellHashBuffer!: GPUBuffer;
	private fetchStatusBuffer!: GPUBuffer;
	private fieldBuffer!: GPUBuffer;
	private simModule!: GPUShaderModule;
	private visualModules: GPUShaderModule[] = [];
	private machineSettings: LabSettings = structuredClone(ORIGINAL_SETTINGS);
	private measureFetch = false;
	private fieldMode: FieldMode = 'bytes';
	private marker: Marker | null = null;
	private fieldDirty = true;
	private fieldUpdate = -1;
	private masks = new Uint32Array(24);
	// 8 scalars, 24 suppression words, then the reversible rule's dissipation threshold and padding.
	private paramsData = new Uint32Array(36);
	private dissipationThreshold = 0xffffffff;

	// Render buffers
	private colormapBuffer!: GPUBuffer;
	private renderParamsBuffer!: GPUBuffer;
	private traceImageBuffer!: GPUBuffer;

	// Staging buffer for readback
	private byteCountsStagingBuffer!: GPUBuffer;
	private soupStagingBuffer!: GPUBuffer;
	private cellHashStagingBuffer!: GPUBuffer;
	private cellStagingBuffer!: GPUBuffer;
	private cellReadQueue: Promise<void> = Promise.resolve();

	// Compute pipelines
	private clearCollisionPipeline!: GPUComputePipeline;
	private prepareBatchPipeline!: GPUComputePipeline;
	private z80ExecutePipeline!: GPUComputePipeline;
	private absorbPipeline!: GPUComputePipeline;
	private mutatePipeline!: GPUComputePipeline;
	private countBytesPipeline!: GPUComputePipeline;
	private clearByteCountsPipeline!: GPUComputePipeline;
	private hashCellsPipeline!: GPUComputePipeline;

	// Render pipeline
	private renderPipeline!: GPURenderPipeline;
	private fieldPipeline!: GPUComputePipeline;

	// Bind groups
	private simBindGroup!: GPUBindGroup;
	private renderBindGroup!: GPUBindGroup;
	private fieldBindGroup!: GPUBindGroup;

	// Grid configuration
	private gridConfig: GridConfig;

	// View state (zoom/pan) - initialized in constructor
	view = { zoom: SOUP_WIDTH, offsetX: 0, offsetY: 0 }; // default, reset in constructor

	// State
	private batchIndex = 0;
	private cpuRng: SplitMix64;
	private _noiseExp = 4; // 1/2^4 = 1/16
	private _pairCount = MAX_BATCH_PAIR_N;
	private _z80Steps = Z80_STEPS;
	private hoverCell = -1;
	private showAverage = 0;
	private brightness = 0; // -1..1
	private contrast = 1; // 0..2
	private saturation = 1; // 0..2
	private showGrid = 1; // 1 = show grid lines, 0 = hide
	private colormap: Uint32Array;
	// Instruction suppression (three per-page sets, see $lib/z80-opcodes).
	private suppression: SuppressSets = emptySuppression();

	// Stats (read back from GPU periodically)
	byteCounts = new Uint32Array(256);
	// Throughput: instruction slots (pairs × max steps) per second of *completed*
	// GPU work, measured from onSubmittedWorkDone rather than submit time.
	opsPerSec = 0;
	private rateAccumOps = 0;
	private rateT0 = 0;

	// Derived dimensions (from gridConfig)
	private get soupWidth(): number {
		return this.gridConfig.width;
	}
	private get soupHeight(): number {
		return this.gridConfig.height;
	}
	private get cellCount(): number {
		return this.gridConfig.width * this.gridConfig.height;
	}
	get tapeLength(): number {
		return this.gridConfig.tapeLength ?? getTapeLength(this.gridType);
	}
	private get pairLength(): number {
		return (
			this.gridConfig.pairLength ??
			(this.gridConfig.tapeLength ? this.tapeLength * 2 : getPairLength(this.gridType))
		);
	}
	private get wordsPerCell(): number {
		return Math.ceil(this.tapeLength / 4);
	}
	private get soupSize(): number {
		return this.cellCount * this.wordsPerCell * 4;
	}
	private get gridType(): GridType {
		return this.gridConfig.gridType;
	}

	constructor(
		private seed: number,
		config: GridConfig = DEFAULT_GRID_CONFIG
	) {
		this.gridConfig = { ...config };
		this.view.zoom = config.width; // rough default, resetView corrects after init
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
		this._z80Steps = Math.max(1, Math.min(4096, v));
	}

	get batchCount(): number {
		return this.batchIndex;
	}

	/** Lifecycle-only counters: reading diagnostics never touches the GPU or simulation loop. */
	get diagnostics(): Readonly<EngineDiagnostics> {
		return Object.freeze({ ...this.lifecycle });
	}

	setSuppression(sets: SuppressSets): void {
		this.suppression = { base: new Set(sets.base), cb: new Set(sets.cb), ed: new Set(sets.ed) };
		this.masks.set(suppressionMasks(this.suppression));
		this.machineSettings.ablation = (['base', 'cb', 'ed'] as const).flatMap((page) =>
			[...sets[page]].map((code) => `${page}:${code.toString(16).padStart(2, '0')}`)
		);
		// Immediately write the masks to the GPU so they take effect even
		// before the next simulateStep() call
		this.flushSuppressMask();
	}

	private flushSuppressMask(): void {
		// Masks start at byte offset 32 (after the 8 scalar params): 24 u32 words.
		this.paramsData.set(this.masks, 8);
		this.device.queue.writeBuffer(this.paramsBuffer, 32, this.masks.buffer);
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
				maxBufferSize: adapter.limits.maxBufferSize,
				maxStorageBuffersPerShaderStage: Math.min(
					10,
					adapter.limits.maxStorageBuffersPerShaderStage
				)
			}
		});
		this.lifecycle.deviceId = ++nextDeviceId;
		this.lifecycle.deviceInitializations++;

		this.context = canvas.getContext('webgpu')!;
		this.format = navigator.gpu.getPreferredCanvasFormat();
		this.context.configure({
			device: this.device,
			format: this.format,
			alphaMode: 'premultiplied'
		});

		this.createBuffers();
		this.createPipelines();
		await this.checkShader();
		this.initSoup();
		this.uploadColormap();
		this.resetView(canvas.width, canvas.height);

		return true;
	}

	private createBuffers(): void {
		this.lifecycle.bufferRevision++;
		const dev = this.device;

		const soupWords = this.cellCount * this.wordsPerCell;
		this.soupBuffer = dev.createBuffer({
			size: soupWords * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});
		this.beforeBuffer = dev.createBuffer({
			size: soupWords * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});

		// Pairs: 2 u32s per pair
		this.pairsBuffer = dev.createBuffer({
			size: MAX_BATCH_PAIR_N * 2 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		// Pair data: wordsPerCell*2 u32s per pair
		const wordsPerPair = this.wordsPerCell * 2;
		this.pairDataBuffer = dev.createBuffer({
			size: MAX_BATCH_PAIR_N * wordsPerPair * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		// Write counts: 2 u32s per pair
		this.writeCountsBuffer = dev.createBuffer({
			size: MAX_BATCH_PAIR_N * 10 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		this.fetchStatusBuffer = dev.createBuffer({
			size: this.cellCount * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		this.fieldBuffer = dev.createBuffer({
			size: this.cellCount * 4,
			usage: GPUBufferUsage.STORAGE
		});

		// RNG states: 1 u32 per pair
		this.rngStatesBuffer = dev.createBuffer({
			size: MAX_BATCH_PAIR_N * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		// Params uniform: 8 scalars + 6 × vec4<u32> suppression masks = 128 bytes, then one vec4-sized
		// slot the reversible rule reads (the overwrite shader's struct ends at 128 and ignores it).
		this.paramsBuffer = dev.createBuffer({
			size: 144,
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

		// Collision mask: cellCount atomic u32s
		this.collisionMaskBuffer = dev.createBuffer({
			size: this.cellCount * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		// Cell hashes: one u32 per cell for species tracking
		this.cellHashBuffer = dev.createBuffer({
			size: this.cellCount * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});

		// Staging buffers for readback
		this.byteCountsStagingBuffer = dev.createBuffer({
			size: 256 * 4,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		});

		const soupStagingWords = this.cellCount * this.wordsPerCell;
		this.soupStagingBuffer = dev.createBuffer({
			size: soupStagingWords * 4,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		});

		this.cellHashStagingBuffer = dev.createBuffer({
			size: this.cellCount * 4,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		});
		this.cellStagingBuffer = dev.createBuffer({
			size: 128,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		});

		// Render buffers
		this.colormapBuffer = dev.createBuffer({
			size: 256 * 4,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});

		this.renderParamsBuffer = dev.createBuffer({
			size: 80, // RenderParams struct (20 fields)
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
		});

		this.traceImageBuffer = dev.createBuffer({
			size: this.pairLength * 4096 * 4, // max Z80 steps
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
	}

	private createPipelines(): void {
		this.lifecycle.pipelineRevision++;
		const dev = this.device;

		// Simulation shader module
		const simModule = dev.createShaderModule({
			code: createSimShader(this.gridType, this.settings, this.measureFetch)
		});
		this.simModule = simModule;

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
				{ binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
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
				// Slot 4 is unused by the overwrite shader; the reversible rule keeps its second layer there.
				{
					binding: 4,
					resource: {
						buffer:
							this.machineSettings.interaction === 'reversible'
								? this.beforeBuffer
								: this.rngStatesBuffer
					}
				},
				{ binding: 5, resource: { buffer: this.paramsBuffer } },
				{ binding: 6, resource: { buffer: this.pairActiveBuffer } },
				{ binding: 7, resource: { buffer: this.byteCountsBuffer } },
				{ binding: 8, resource: { buffer: this.collisionMaskBuffer } },
				{ binding: 9, resource: { buffer: this.cellHashBuffer } },
				{ binding: 10, resource: { buffer: this.fetchStatusBuffer } }
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
		this.hashCellsPipeline = makeComputePipeline('hash_cells');

		// Render pipeline
		const renderModule = dev.createShaderModule({ code: createRenderShader(this.gridType) });
		const fieldModule = dev.createShaderModule({ code: createFieldShader() });
		this.visualModules = [renderModule, fieldModule];

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
				},
				{ binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
				{ binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } }
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
				{ binding: 3, resource: { buffer: this.traceImageBuffer } },
				{ binding: 4, resource: { buffer: this.fetchStatusBuffer } },
				{ binding: 5, resource: { buffer: this.fieldBuffer } }
			]
		});
		const fieldLayout = dev.createBindGroupLayout({
			entries: [
				{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
				{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
			]
		});
		this.fieldPipeline = dev.createComputePipeline({
			layout: dev.createPipelineLayout({ bindGroupLayouts: [fieldLayout] }),
			compute: { module: fieldModule, entryPoint: 'measure_field' }
		});
		this.fieldBindGroup = dev.createBindGroup({
			layout: fieldLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.soupBuffer } },
				{ binding: 1, resource: { buffer: this.renderParamsBuffer } },
				{ binding: 2, resource: { buffer: this.fieldBuffer } }
			]
		});
		this.setupFree();
	}

	/** Instructions per counted update: every head once (a tick), or every pair for its full budget. */
	get opsPerUpdate(): number {
		return this.free ? this.freeHeads : this._pairCount * this._z80Steps;
	}

	private get free(): boolean {
		return this.machineSettings.interaction === 'free';
	}

	private dropFree(): void {
		this.headsBuffer?.destroy();
		this.headLogBuffer?.destroy();
		this.claimsBuffer?.destroy();
		this.freeParamsBuffer?.destroy();
		this.headsBuffer = this.headLogBuffer = this.claimsBuffer = this.freeParamsBuffer = null;
		this.freePipelines = null;
		this.freeBindGroup = null;
		this.freeModule = null;
		this.freeSignature = '';
	}

	/**
	 * Build the free-running world (interaction `free`): its three passes, the heads, their write log
	 * and the per-byte bids. Heads survive a rebuild unless something that defines them changed.
	 */
	private setupFree(): void {
		if (!this.free) {
			this.dropFree();
			return;
		}
		const dev = this.device;
		const settings = this.settings;
		const signature = [
			this.soupWidth,
			this.soupHeight,
			settings.heads,
			settings.facing,
			settings.compat,
			settings.variant,
			this.lifecycle.bufferRevision
		].join('|');
		const keepHeads = signature === this.freeSignature && this.headsBuffer !== null;
		if (!keepHeads) {
			this.dropFree();
			this.freeHeads = headCount(settings);
			this.headsBuffer = dev.createBuffer({
				size: this.freeHeads * HEAD_WORDS * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
			});
			this.headLogBuffer = dev.createBuffer({
				size: this.freeHeads * LOG_WORDS * 4,
				usage: GPUBufferUsage.STORAGE
			});
			this.claimsBuffer = dev.createBuffer({
				size: this.cellCount * 16 * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
			});
			this.freeParamsBuffer = dev.createBuffer({
				size: 32,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
			});
			this.freeSignature = signature;
		}
		const module = dev.createShaderModule({ code: createFreeShader(settings) });
		this.freeModule = module;
		const layout = dev.createBindGroupLayout({
			entries: [
				{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
				{ binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
			]
		});
		const pipelineLayout = dev.createPipelineLayout({ bindGroupLayouts: [layout] });
		const pipeline = (entryPoint: string) =>
			dev.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
		this.freePipelines = {
			step: pipeline('free_step'),
			apply: pipeline('free_apply')
		};
		this.freeBindGroup = dev.createBindGroup({
			layout,
			entries: [
				{ binding: 0, resource: { buffer: this.soupBuffer } },
				{ binding: 1, resource: { buffer: this.headsBuffer! } },
				{ binding: 2, resource: { buffer: this.headLogBuffer! } },
				{ binding: 3, resource: { buffer: this.claimsBuffer! } },
				{ binding: 4, resource: { buffer: this.freeParamsBuffer! } }
			]
		});
		if (!keepHeads) this.resetHeads();
	}

	/** Put every head back in its starting state and clear the bids. */
	private resetHeads(): void {
		if (!this.headsBuffer || !this.claimsBuffer) return;
		this.device.queue.writeBuffer(this.headsBuffer, 0, createHeads(this.settings));
		this.device.queue.writeBuffer(
			this.claimsBuffer,
			0,
			new Uint32Array(this.cellCount * 16).fill(NO_CLAIM)
		);
	}

	/**
	 * Free-running world: queue `ticks` ticks. One tick is one instruction for every head. All the
	 * randomness of a tick comes from counters the heads carry, so the ticks need nothing from here
	 * between them.
	 */
	simulateTicks(ticks: number): void {
		if (!this.free || !this.freePipelines) throw new Error('This world has no heads.');
		this.freeParams.set([
			this.soupWidth * 4,
			this.soupHeight * 4,
			this.freeHeads,
			HALT_RULES[this.machineSettings.haltRule],
			chanceThreshold(this.machineSettings.turnover),
			chanceThreshold(mutationChance(this._noiseExp)),
			FACING_MASKS[this.machineSettings.facing],
			0
		]);
		this.device.queue.writeBuffer(this.freeParamsBuffer!, 0, this.freeParams);
		const encoder = this.device.createCommandEncoder();
		const groups = Math.ceil(this.freeHeads / 64);
		for (let t = 0; t < ticks; t++) {
			for (const pipeline of [this.freePipelines.step, this.freePipelines.apply]) {
				const pass = encoder.beginComputePass();
				pass.setPipeline(pipeline);
				pass.setBindGroup(0, this.freeBindGroup!);
				pass.dispatchWorkgroups(groups);
				pass.end();
			}
		}
		this.device.queue.submit([encoder.finish()]);
		this.batchIndex += ticks;
		const ops = this.freeHeads * ticks;
		this.device.queue.onSubmittedWorkDone().then(() => this.onStepDone(ops));
	}

	/** Free-running world: every head's processor state, landing site and counters (HEAD_WORDS each). */
	async readHeads(): Promise<Uint32Array> {
		if (!this.headsBuffer) throw new Error('This world has no heads.');
		const size = this.headsBuffer.size;
		const staging = this.device.createBuffer({
			size,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		});
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.headsBuffer, 0, staging, 0, size);
		this.device.queue.submit([encoder.finish()]);
		await staging.mapAsync(GPUMapMode.READ);
		const data = new Uint32Array(staging.getMappedRange().slice(0));
		staging.unmap();
		staging.destroy();
		return data;
	}

	/** Free-running world: replace every head (for tests and restored states). */
	writeHeads(data: Uint32Array<ArrayBuffer>): void {
		if (!this.headsBuffer || data.byteLength !== this.headsBuffer.size)
			throw new Error('Head data does not match this world.');
		this.device.queue.writeBuffer(this.headsBuffer, 0, data);
	}

	private initSoup(): void {
		this.fieldDirty = true;
		// Generate random soup data on CPU and upload
		const soupData = new Uint8Array(this.soupSize);
		const rng = new SplitMix64(this.seed);
		for (let i = 0; i < this.soupSize; i += 4) {
			const r = rng.nextU32();
			soupData[i] = r & 0xff;
			if (i + 1 < this.soupSize) soupData[i + 1] = (r >> 8) & 0xff;
			if (i + 2 < this.soupSize) soupData[i + 2] = (r >> 16) & 0xff;
			if (i + 3 < this.soupSize) soupData[i + 3] = (r >> 24) & 0xff;
		}
		// Padding bytes (the last word of a 19-byte hex tape) are not part of the
		// tape: keep them zero so they never show up in statistics.
		const stride = this.wordsPerCell * 4;
		if (stride !== this.tapeLength) {
			for (let c = 0; c < this.cellCount; c++) {
				for (let b = this.tapeLength; b < stride; b++) soupData[c * stride + b] = 0;
			}
		}
		this.device.queue.writeBuffer(this.soupBuffer, 0, soupData);
		this.device.queue.writeBuffer(this.fetchStatusBuffer, 0, new Uint32Array(this.cellCount));
		this.batchIndex = 0;
		this.restBefore();
		this.resetHeads();
	}

	/** Put every cell at rest: its stored past becomes its present tape (reversible rule). */
	private restBefore(): void {
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(this.soupBuffer, 0, this.beforeBuffer, 0, this.soupBuffer.size);
		this.device.queue.submit([encoder.finish()]);
	}

	private static thresholdOf(dissipation: number): number {
		return Math.min(0xffffffff, Math.floor(dissipation * 4294967296));
	}

	/** Change the reversible rule's dissipation without rebuilding the pipelines. */
	setDissipation(value: number): void {
		if (!Number.isFinite(value) || value < 0 || value > 1)
			throw new Error('Dissipation must be between zero and one.');
		this.machineSettings.dissipation = value;
		this.dissipationThreshold = GPUEngine.thresholdOf(value);
	}

	private uploadColormap(): void {
		this.device.queue.writeBuffer(this.colormapBuffer, 0, this.colormap.buffer);
	}

	updateColormap(colormap: Uint32Array): void {
		this.colormap = colormap;
		this.uploadColormap();
	}

	reset(seed: number): void {
		this.lifecycle.resetCount++;
		this.seed = seed;
		this.cpuRng = new SplitMix64(seed);
		this.initSoup();
		this.batchIndex = 0;
		this.resetRate();
		this.byteCounts.fill(0);
		this.flushSuppressMask();
	}

	private resetRate(): void {
		this.opsPerSec = 0;
		this.rateAccumOps = 0;
		this.rateT0 = 0;
	}

	// Called when a submitted step has actually finished on the GPU.
	private onStepDone(ops: number): void {
		const now = performance.now();
		if (this.rateT0 === 0) {
			this.rateT0 = now;
			return;
		}
		this.rateAccumOps += ops;
		const dt = now - this.rateT0;
		if (dt >= 500) {
			const inst = this.rateAccumOps / (dt / 1000);
			this.opsPerSec = this.opsPerSec === 0 ? inst : this.opsPerSec * 0.7 + inst * 0.3;
			this.rateAccumOps = 0;
			this.rateT0 = now;
		}
	}

	get settings(): LabSettings {
		return {
			...this.machineSettings,
			width: this.soupWidth,
			height: this.soupHeight,
			topology: this.gridType,
			length: this.tapeLength,
			ring: this.pairLength,
			seed: this.seed,
			steps: this._z80Steps,
			pairAttempts: this._pairCount,
			mutationExponent: this._noiseExp,
			ablation: [...this.machineSettings.ablation]
		};
	}

	private async checkShader(): Promise<void> {
		const infos = await Promise.all(
			[this.simModule, ...this.visualModules, ...(this.freeModule ? [this.freeModule] : [])].map(
				(module) => module.getCompilationInfo()
			)
		);
		const errors = infos.flatMap((info) =>
			info.messages.filter((message) => message.type === 'error')
		);
		if (errors.length)
			throw new Error(errors.map((message) => `${message.lineNum}: ${message.message}`).join('\n'));
	}

	async waitForIdle(): Promise<void> {
		await this.device.queue.onSubmittedWorkDone();
	}

	private queueConfiguration(operation: () => Promise<void>): Promise<void> {
		const task = this.configurationQueue.then(operation);
		this.configurationQueue = task.catch(() => {});
		return task;
	}

	private finishReadback(): void {
		this.pendingReadbacks--;
		if (this.pendingReadbacks === 0) {
			for (const resolve of this.readbackWaiters) resolve();
			this.readbackWaiters = [];
		}
	}

	private waitForReadbacks(): Promise<void> {
		return this.pendingReadbacks === 0
			? Promise.resolve()
			: new Promise((resolve) => this.readbackWaiters.push(resolve));
	}

	/** World/ISA edits are explicit events; no host memory round-trip between updates. */
	setMachine(settings: LabSettings): Promise<void> {
		const next = withInteractionDefaults(structuredClone(settings));
		return this.queueConfiguration(() => this.applyMachine(next));
	}

	private async applyMachine(settings: LabSettings): Promise<void> {
		validateSettings(settings);
		if (settings.pairAttempts > MAX_BATCH_PAIR_N)
			throw new Error('This instrument supports up to 8192 pair slots per update.');
		const geometryChanged =
			settings.width !== this.soupWidth ||
			settings.height !== this.soupHeight ||
			settings.topology !== this.gridType ||
			settings.length !== this.tapeLength ||
			settings.ring !== this.pairLength;
		let releaseReads: (() => void) | undefined;
		if (geometryChanged && !this.readbackBarrier)
			this.readbackBarrier = new Promise((resolve) => {
				releaseReads = resolve;
			});
		try {
			await this.waitForIdle();
			if (geometryChanged) await this.waitForReadbacks();
			const enteringReversible =
				settings.interaction === 'reversible' && this.machineSettings.interaction !== 'reversible';
			this.machineSettings = structuredClone(settings);
			this.dissipationThreshold = GPUEngine.thresholdOf(settings.dissipation);
			this.z80Steps = settings.steps;
			this.pairCount = settings.pairAttempts;
			this.noiseExp = settings.mutationExponent;
			this.masks.fill(0);
			this.masks.set(ablationMasks(settings.isa, settings.ablation));
			if (geometryChanged) {
				this.changeGridConfig(
					{
						width: settings.width,
						height: settings.height,
						gridType: settings.topology,
						tapeLength: settings.length,
						pairLength: settings.ring
					},
					this.context.canvas.width,
					this.context.canvas.height
				);
			} else {
				this.createPipelines();
				this.flushSuppressMask();
				this.resetRate();
				// A world switched to the reversible rule starts with every cell at rest.
				if (enteringReversible) this.restBefore();
			}
			await this.checkShader();
		} finally {
			if (releaseReads) {
				this.readbackBarrier = null;
				releaseReads();
			}
		}
	}

	configure(settings: LabSettings, initial: InitialCondition = { kind: 'random' }): Promise<void> {
		const next = withInteractionDefaults(structuredClone(settings));
		const condition = structuredClone(initial);
		return this.queueConfiguration(async () => {
			// Validate initial bytes before changing the live condition.
			const bytes = createInitialSoup(next, condition);
			let releaseReads!: () => void;
			this.readbackBarrier = new Promise((resolve) => {
				releaseReads = resolve;
			});
			try {
				await this.waitForReadbacks();
				await this.applyMachine(next);
				this.reset(next.seed);
				if (condition.kind !== 'random') {
					const padded = new Uint8Array(this.soupSize);
					for (let c = 0; c < this.cellCount; c++)
						padded.set(
							bytes.subarray(c * this.tapeLength, (c + 1) * this.tapeLength),
							c * this.wordsPerCell * 4
						);
					this.device.queue.writeBuffer(this.soupBuffer, 0, padded);
					this.restBefore();
				}
			} finally {
				this.readbackBarrier = null;
				releaseReads();
			}
		});
	}

	setAblation(patterns: string[]): void {
		const masks = ablationMasks(this.machineSettings.isa, patterns);
		this.machineSettings.ablation = [...patterns];
		this.masks.fill(0);
		this.masks.set(masks);
		this.flushSuppressMask();
	}

	async setShowConfinement(show: boolean): Promise<void> {
		await this.setFieldMode(show ? 'confinement' : 'bytes');
	}

	setFieldMode(mode: FieldMode): Promise<void> {
		return this.queueConfiguration(() => this.applyFieldMode(mode));
	}

	private async applyFieldMode(mode: FieldMode): Promise<void> {
		if (!['bytes', 'identity', 'confinement', 'zeros', 'diversity', 'marker'].includes(mode))
			throw new Error('Unknown observation mode.');
		this.fieldMode = mode;
		this.fieldDirty = true;
		const show = mode === 'confinement';
		if (show === this.measureFetch) return;
		await this.waitForIdle();
		this.measureFetch = show;
		this.device.queue.writeBuffer(this.fetchStatusBuffer, 0, new Uint32Array(this.cellCount));
		this.createPipelines();
		await this.checkShader();
	}

	setMarker(marker: Marker | null): void {
		if (
			marker &&
			(!Number.isInteger(marker.position) ||
				marker.position < 0 ||
				marker.position >= this.tapeLength ||
				!Number.isInteger(marker.value) ||
				marker.value < 0 ||
				marker.value > 255)
		)
			throw new Error('Marker position or byte value is out of range.');
		this.marker = marker ? { ...marker } : null;
		this.fieldDirty = true;
	}

	runCulture(
		tape: number[] | Uint8Array,
		options: { trials?: number; seed?: number; generations?: number } = {}
	): Promise<CultureResult> {
		return runCulture(this.device, this.settings, tape, options);
	}

	runAssays(tape: number[] | Uint8Array, options: AssayOptions = {}): Promise<AssayResult> {
		return runAssays(this.device, this.settings, tape, options);
	}

	get config(): GridConfig {
		return { ...this.gridConfig };
	}

	changeGridConfig(config: GridConfig, canvasW?: number, canvasH?: number): void {
		this.gridConfig = { ...config };
		if (this.marker && this.marker.position >= this.tapeLength) this.marker = null;
		// Preserve suppression across buffer recreation
		const savedSuppression = this.suppression;
		// Destroy old buffers
		this.destroy();
		// Recreate everything with new dimensions
		this.createBuffers();
		this.createPipelines();
		this.initSoup();
		this.uploadColormap();
		this.resetView(canvasW, canvasH);
		this.batchIndex = 0;
		// Restore and flush suppression masks to the new paramsBuffer
		this.suppression = savedSuppression;
		this.flushSuppressMask();
		this.resetRate();
		this.byteCounts.fill(0);
	}

	setHoverCell(cell: number): void {
		this.hoverCell = cell;
	}

	setShowAverage(show: boolean): void {
		this.showAverage = show ? 1 : 0;
	}

	setBCS(brightness: number, contrast: number, saturation: number): void {
		this.brightness = brightness;
		this.contrast = contrast;
		this.saturation = saturation;
	}

	setShowGrid(show: boolean): void {
		this.showGrid = show ? 1 : 0;
	}

	// Run one simulation step (all on GPU)
	simulateStep(): void {
		if (this.free && this.freePipelines) {
			this.simulateTicks(this.freeTicksPerStep);
			return;
		}
		const noiseCoef = 1 / Math.pow(2, this._noiseExp);
		const mutationCount = Math.floor(this._pairCount * noiseCoef);

		// Update params (8 scalars + 24 suppression mask words = 32 u32s)
		const paramsData = this.paramsData;
		paramsData.set([
			this.soupWidth,
			this.soupHeight,
			this.tapeLength,
			this.pairLength,
			this._pairCount,
			mutationCount,
			this._z80Steps,
			this.cpuRng.nextU32() // batch_seed - different each batch
		]);
		paramsData.set(this.masks, 8);
		paramsData[32] = this.dissipationThreshold;
		this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

		const encoder = this.device.createCommandEncoder();

		// 1. Clear collision mask
		{
			const pass = encoder.beginComputePass();
			pass.setPipeline(this.clearCollisionPipeline);
			pass.setBindGroup(0, this.simBindGroup);
			pass.dispatchWorkgroups(Math.ceil(this.cellCount / 256));
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

		// Throughput is credited when the GPU finishes this step, not when it is
		// submitted (submitting 8 steps back-to-back takes ~0.1 ms each).
		const ops = this._pairCount * this._z80Steps;
		this.device.queue.onSubmittedWorkDone().then(() => this.onStepDone(ops));
	}

	// Read byte counts from GPU (async)
	// Serialised like the other readbacks: a second caller while a map is
	// pending gets the last counts instead of a mapAsync OperationError.
	private _statsMapPending = false;
	async readStats(): Promise<Uint32Array> {
		if (this.readbackBarrier) return this.readbackBarrier.then(() => this.readStats());
		if (this._statsMapPending) return this.byteCounts;
		this._statsMapPending = true;
		this.pendingReadbacks++;
		try {
			return await this.readStatsNow();
		} finally {
			this._statsMapPending = false;
			this.finishReadback();
		}
	}

	private async readStatsNow(): Promise<Uint32Array> {
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
			const soupWords = this.cellCount * this.wordsPerCell;
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

	// Read cell hashes from GPU for species tracking
	private _hashMapPending = false;
	async readCellHashes(): Promise<Uint32Array> {
		if (this.readbackBarrier) return this.readbackBarrier.then(() => this.readCellHashes());
		if (this._hashMapPending) return new Uint32Array(0);
		this._hashMapPending = true;
		this.pendingReadbacks++;
		try {
			const encoder = this.device.createCommandEncoder();

			// Dispatch hash_cells
			const pass = encoder.beginComputePass();
			pass.setPipeline(this.hashCellsPipeline);
			pass.setBindGroup(0, this.simBindGroup);
			pass.dispatchWorkgroups(Math.ceil(this.cellCount / 256));
			pass.end();

			// Copy to staging
			encoder.copyBufferToBuffer(
				this.cellHashBuffer,
				0,
				this.cellHashStagingBuffer,
				0,
				this.cellCount * 4
			);
			this.device.queue.submit([encoder.finish()]);

			await this.cellHashStagingBuffer.mapAsync(GPUMapMode.READ);
			const data = new Uint32Array(this.cellHashStagingBuffer.getMappedRange().slice(0));
			this.cellHashStagingBuffer.unmap();
			return data;
		} finally {
			this._hashMapPending = false;
			this.finishReadback();
		}
	}

	// Write soup data back to GPU (for checkpoint restore)
	writeSoupData(data: Uint8Array): void {
		this.fieldDirty = true;
		this.device.queue.writeBuffer(
			this.soupBuffer,
			0,
			data.buffer,
			data.byteOffset,
			data.byteLength
		);
		this.restBefore();
	}

	/**
	 * Load the hidden layer of the reversible rule (each cell's tape before its last event). Call it
	 * after `writeSoupData`, which puts every cell at rest; together they restore a two-layer state.
	 */
	writeBeforeData(data: Uint8Array): void {
		if (data.byteLength !== this.beforeBuffer.size)
			throw new Error('The hidden layer must be the size of the lattice.');
		this.device.queue.writeBuffer(
			this.beforeBuffer,
			0,
			data.buffer,
			data.byteOffset,
			data.byteLength
		);
	}

	// Read soup data from GPU for CPU-side trace/disassembly
	private _soupMapPending = false;
	async readSoupData(): Promise<Uint8Array> {
		if (this.readbackBarrier) return this.readbackBarrier.then(() => this.readSoupData());
		if (this._soupMapPending) return new Uint8Array(0);
		this._soupMapPending = true;
		this.pendingReadbacks++;
		try {
			const soupBytes = this.cellCount * this.wordsPerCell * 4;
			const encoder = this.device.createCommandEncoder();
			encoder.copyBufferToBuffer(this.soupBuffer, 0, this.soupStagingBuffer, 0, soupBytes);
			this.device.queue.submit([encoder.finish()]);

			await this.soupStagingBuffer.mapAsync(GPUMapMode.READ);
			const data = new Uint8Array(this.soupStagingBuffer.getMappedRange().slice(0));
			this.soupStagingBuffer.unmap();
			return data;
		} finally {
			this._soupMapPending = false;
			this.finishReadback();
		}
	}

	/**
	 * Read the hidden layer of the reversible rule: each cell's tape before its last event. Under the
	 * overwrite rule the layer is not maintained and the bytes mean nothing.
	 */
	async readBeforeData(): Promise<Uint8Array> {
		if (this.readbackBarrier) return this.readbackBarrier.then(() => this.readBeforeData());
		while (this._soupMapPending) await new Promise((resolve) => setTimeout(resolve, 0));
		this._soupMapPending = true;
		this.pendingReadbacks++;
		try {
			const soupBytes = this.cellCount * this.wordsPerCell * 4;
			const encoder = this.device.createCommandEncoder();
			encoder.copyBufferToBuffer(this.beforeBuffer, 0, this.soupStagingBuffer, 0, soupBytes);
			this.device.queue.submit([encoder.finish()]);
			await this.soupStagingBuffer.mapAsync(GPUMapMode.READ);
			const data = new Uint8Array(this.soupStagingBuffer.getMappedRange().slice(0));
			this.soupStagingBuffer.unmap();
			return data;
		} finally {
			this._soupMapPending = false;
			this.finishReadback();
		}
	}

	/** Read only the selected tape; probing never downloads the population. */
	readCell(cell: number): Promise<Uint8Array> {
		if (this.readbackBarrier) return this.readbackBarrier.then(() => this.readCell(cell));
		if (!Number.isInteger(cell) || cell < 0 || cell >= this.cellCount)
			return Promise.reject(new Error('Cell index is outside the lattice.'));
		this.pendingReadbacks++;
		const task = this.cellReadQueue.then(async () => {
			try {
				const length = this.tapeLength;
				const size = this.wordsPerCell * 4;
				const staging = this.cellStagingBuffer;
				const encoder = this.device.createCommandEncoder();
				encoder.copyBufferToBuffer(this.soupBuffer, cell * size, staging, 0, size);
				this.device.queue.submit([encoder.finish()]);
				await staging.mapAsync(GPUMapMode.READ, 0, size);
				try {
					return new Uint8Array(staging.getMappedRange(0, size).slice(0, length));
				} finally {
					staging.unmap();
				}
			} finally {
				this.finishReadback();
			}
		});
		this.cellReadQueue = task.then(
			() => {},
			() => {}
		);
		return task;
	}

	/** Queue-ordered snapshot; later population updates may continue while it maps. */
	async readRegion(region: CellRegion): Promise<RegionMeasurement> {
		if (this.readbackBarrier) return this.readbackBarrier.then(() => this.readRegion(region));
		const selection = normalizeRegion(region, this.soupWidth, this.soupHeight);
		const { width, height, length } = this.settings;
		const marker = this.marker;
		const update = this.batchIndex;
		const data = await this.readSoupData();
		if (!data.length) throw new Error('Another population sample is pending. Try again shortly.');
		return measureRegion(data, width, height, length, selection, update, marker);
	}

	/** One queue-ordered GPU patch preserves neighbouring bytes even during live updates. */
	async plantMarker(cell: number, position: number, value: number): Promise<void> {
		if (this.readbackBarrier)
			return this.readbackBarrier.then(() => this.plantMarker(cell, position, value));
		if (!Number.isInteger(cell) || cell < 0 || cell >= this.cellCount)
			throw new Error('Cell index is outside the lattice.');
		this.setMarker({ position, value });
		if (!this.markerPipeline)
			this.markerPipeline = this.device.createComputePipeline({
				layout: 'auto',
				compute: {
					module: this.device.createShaderModule({
						code: `
struct Patch { word: u32, shift: u32, value: u32, cell: u32 }
@group(0) @binding(0) var<storage, read_write> soup: array<u32>;
@group(0) @binding(1) var<storage, read_write> fetched: array<u32>;
@group(0) @binding(2) var<uniform> marker_patch: Patch;
@compute @workgroup_size(1) fn plant_marker() {
  let mask = 255u << marker_patch.shift;
  soup[marker_patch.word] = (soup[marker_patch.word] & ~mask) | (marker_patch.value << marker_patch.shift);
  fetched[marker_patch.cell] = 0u;
}`
					}),
					entryPoint: 'plant_marker'
				}
			});
		const params = this.device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM,
			mappedAtCreation: true
		});
		new Uint32Array(params.getMappedRange()).set([
			cell * this.wordsPerCell + (position >> 2),
			(position & 3) * 8,
			value,
			cell
		]);
		params.unmap();
		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(this.markerPipeline);
		pass.setBindGroup(
			0,
			this.device.createBindGroup({
				layout: this.markerPipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: this.soupBuffer } },
					{ binding: 1, resource: { buffer: this.fetchStatusBuffer } },
					{ binding: 2, resource: { buffer: params } }
				]
			})
		);
		pass.dispatchWorkgroups(1);
		pass.end();
		this.device.queue.submit([encoder.finish()]);
		this.fieldDirty = true;
		try {
			await this.waitForIdle();
		} finally {
			params.destroy();
		}
	}

	/** Canvas-pixel bounds of a cell; hex returns its enclosing visual box. */
	cellBounds(
		cell: number,
		canvasW: number,
		canvasH: number
	): { x: number; y: number; width: number; height: number } {
		const x = cell % this.soupWidth;
		const y = Math.floor(cell / this.soupWidth);
		const scale = canvasW / this.view.zoom;
		if (this.gridType === 'hex') {
			const row = y * 5;
			const centerX = x * 5 + (y & 1) * 2 + 0.5 + (row & 1) * 0.5;
			const centerY = (row + 0.5) * HEX_HEIGHT_RATIO;
			return {
				x: (centerX - 2.6 - this.view.offsetX) * scale,
				y: (centerY - 2.6 * HEX_HEIGHT_RATIO - this.view.offsetY) * scale,
				width: 5.2 * scale,
				height: 5.2 * HEX_HEIGHT_RATIO * scale
			};
		}
		// Both axes have the same pixel scale because the renderer preserves aspect.
		void canvasH;
		return {
			x: (x - this.view.offsetX) * scale,
			y: (y - this.view.offsetY) * scale,
			width: scale,
			height: scale
		};
	}

	zoomAt(screenX: number, screenY: number, canvasW: number, canvasH: number, factor: number): void {
		const aspect = canvasW / canvasH;
		const cellsX = this.view.zoom;
		const cellsY = this.view.zoom / aspect;

		// Grid position under cursor before zoom
		const gridX = (screenX / canvasW) * cellsX + this.view.offsetX;
		const gridY = (screenY / canvasH) * cellsY + this.view.offsetY;

		const zoomScale = this.gridType === 'hex' ? 10 : 2;
		const maxZoom = Math.max(this.soupWidth, this.soupHeight) * zoomScale;
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

	resetView(canvasW?: number, canvasH?: number): void {
		const aspect = canvasW && canvasH ? canvasW / canvasH : 1;
		if (this.gridType === 'hex') {
			// Cells on offset hex grid: cell (cx, cy) center at offset (cx*5 + (cy&1)*2, cy*5)
			const W = this.soupWidth;
			const H = this.soupHeight;
			const CELL_SPACING = 5;
			const ODD_SHIFT = 2;
			const corners = [
				[0, 0],
				[W - 1, 0],
				[0, H - 1],
				[W - 1, H - 1]
			];
			let minPx = Infinity,
				maxPx = -Infinity;
			let minPy = Infinity,
				maxPy = -Infinity;

			for (const [cx, cy] of corners) {
				const centerCol = cx * CELL_SPACING + (cy & 1) * ODD_SHIFT;
				const centerRow = cy * CELL_SPACING;
				// Cell center in axial
				const qCell = centerCol - Math.floor((centerRow - (centerRow & 1)) / 2);
				const rCell = centerRow;
				// Check byte positions at hex distance ≤ 2 from center
				for (let dq = -2; dq <= 2; dq++) {
					for (let dr = -2; dr <= 2; dr++) {
						if (Math.max(Math.abs(dq), Math.abs(dr), Math.abs(dq + dr)) > 2) continue;
						const bq = qCell + dq;
						const br = rCell + dr;
						// Axial to offset coords
						const col = bq + Math.floor((br - (br & 1)) / 2);
						const row = br;
						const isOdd = (row & 1) !== 0;
						const px = col + 0.5 + (isOdd ? 0.5 : 0.0);
						const py = (row + 0.5) * HEX_HEIGHT_RATIO;
						minPx = Math.min(minPx, px - 0.6);
						maxPx = Math.max(maxPx, px + 0.6);
						minPy = Math.min(minPy, py - 0.6);
						maxPy = Math.max(maxPy, py + 0.6);
					}
				}
			}

			const margin = 1;
			const extentX = maxPx - minPx + 2 * margin;
			const extentY = maxPy - minPy + 2 * margin;
			// Fit to canvas: zoom = cellsX, cellsY = zoom / aspect
			// Need cellsX >= extentX and cellsY >= extentY
			// So zoom >= extentX AND zoom >= extentY * aspect
			this.view.zoom = Math.max(extentX, extentY * aspect);
			// Center the content
			const cellsX = this.view.zoom;
			const cellsY = this.view.zoom / aspect;
			this.view.offsetX = minPx - margin - (cellsX - extentX) / 2;
			this.view.offsetY = minPy - margin - (cellsY - extentY) / 2;
		} else {
			// Square mode: grid is soupWidth × soupHeight in cell units
			const W = this.soupWidth;
			const H = this.soupHeight;
			// Fit to canvas aspect: zoom = cellsX, cellsY = zoom / aspect
			const zoom = Math.max(W, H * aspect);
			const cellsX = zoom;
			const cellsY = zoom / aspect;
			this.view.zoom = zoom;
			this.view.offsetX = -(cellsX - W) / 2;
			this.view.offsetY = -(cellsY - H) / 2;
		}
	}

	// Convert screen position to cell index
	screenToCell(screenX: number, screenY: number, canvasW: number, canvasH: number): number {
		const aspect = canvasW / canvasH;
		const cellsX = this.view.zoom;
		const cellsY = this.view.zoom / aspect;
		const gridX = (screenX / canvasW) * cellsX + this.view.offsetX;
		const gridY = (screenY / canvasH) * cellsY + this.view.offsetY;

		if (this.gridType === 'hex') {
			return this.hexScreenToCell(gridX, gridY);
		}

		const cx = Math.floor(gridX);
		const cy = Math.floor(gridY);
		if (cx < 0 || cx >= this.soupWidth || cy < 0 || cy >= this.soupHeight) return -1;
		return cy * this.soupWidth + cx;
	}

	private hexScreenToCell(pixelX: number, pixelY: number): number {
		// pixelX, pixelY are in hex-byte pixel space (same as shader)

		// Find nearest hex byte
		const rowEst = pixelY / HEX_HEIGHT_RATIO - 0.5;
		const rowGuess = Math.round(rowEst);

		let bestDist = Infinity;
		let bestHx = 0;
		let bestHy = 0;

		for (let dr = -1; dr <= 1; dr++) {
			const hy = rowGuess + dr;
			const isOdd = (hy & 1) !== 0;
			const xOff = isOdd ? 0.5 : 0.0;
			const centerY = (hy + 0.5) * HEX_HEIGHT_RATIO;
			const colEst = pixelX - xOff - 0.5;
			const colGuess = Math.round(colEst);

			for (let dc = -1; dc <= 1; dc++) {
				const hx = colGuess + dc;
				const centerX = hx + 0.5 + xOff;
				const dx = pixelX - centerX;
				const dy = pixelY - centerY;
				const dist = dx * dx + dy * dy;
				if (dist < bestDist) {
					bestDist = dist;
					bestHx = hx;
					bestHy = hy;
				}
			}
		}

		// Find cell using offset hex grid (same as shader find_cell)
		const col = bestHx;
		const row = bestHy;
		const CELL_SPACING = 5;
		const ODD_SHIFT = 2;
		const cy0 = Math.floor(row / CELL_SPACING);

		let bestCellDist = 100;
		let cellA = 0;
		let cellB = 0;

		for (let dcy = 0; dcy <= 1; dcy++) {
			const cy = cy0 + dcy;
			const centerRow = cy * CELL_SPACING;
			const shift = (cy & 1) * ODD_SHIFT;
			const adjCol = col - shift;
			const cx0 = Math.floor(adjCol / CELL_SPACING);

			for (let dcx = 0; dcx <= 1; dcx++) {
				const cx = cx0 + dcx;
				const centerCol = cx * CELL_SPACING + shift;

				// Convert both to axial for hex distance
				const qByte = col - Math.floor((row - (row & 1)) / 2);
				const qCell = centerCol - Math.floor((centerRow - (centerRow & 1)) / 2);
				const dq = qByte - qCell;
				const dr = row - centerRow;
				const dist = Math.max(Math.abs(dq), Math.abs(dr), Math.abs(dq + dr));

				if (dist < bestCellDist) {
					bestCellDist = dist;
					cellA = cx;
					cellB = cy;
				}
			}
		}

		if (cellA < 0 || cellA >= this.soupWidth || cellB < 0 || cellB >= this.soupHeight) return -1;
		return cellB * this.soupWidth + cellA;
	}

	// Render the soup to the canvas
	render(canvas: HTMLCanvasElement): void {
		const tileSize = Math.ceil(Math.sqrt(this.tapeLength));
		const renderParams = new ArrayBuffer(80); // 20 fields * 4 bytes
		const view = new DataView(renderParams);
		view.setUint32(0, this.soupWidth, true);
		view.setUint32(4, this.soupHeight, true);
		view.setUint32(8, tileSize, true);
		view.setFloat32(12, canvas.width, true);
		view.setFloat32(16, canvas.height, true);
		view.setUint32(20, this.showAverage, true);
		view.setInt32(24, this.hoverCell, true);
		view.setFloat32(28, this.view.zoom, true);
		view.setFloat32(32, this.view.offsetX, true);
		view.setFloat32(36, this.view.offsetY, true);
		view.setUint32(40, this.tapeLength, true);
		view.setFloat32(44, this.brightness, true);
		view.setFloat32(48, this.contrast, true);
		view.setFloat32(52, this.saturation, true);
		view.setUint32(56, this.showGrid, true);
		view.setUint32(60, this.measureFetch ? 1 : 0, true);
		view.setUint32(
			64,
			['bytes', 'identity', 'confinement', 'zeros', 'diversity', 'marker'].indexOf(this.fieldMode),
			true
		);
		view.setUint32(68, this.marker?.position ?? 0, true);
		view.setUint32(72, this.marker?.value ?? 0, true);
		view.setUint32(76, this.marker ? 1 : 0, true);

		this.device.queue.writeBuffer(this.renderParamsBuffer, 0, renderParams);

		const encoder = this.device.createCommandEncoder();
		if (
			this.fieldMode !== 'bytes' &&
			this.fieldMode !== 'confinement' &&
			(this.fieldDirty || this.fieldUpdate !== this.batchIndex)
		) {
			const fieldPass = encoder.beginComputePass();
			fieldPass.setPipeline(this.fieldPipeline);
			fieldPass.setBindGroup(0, this.fieldBindGroup);
			fieldPass.dispatchWorkgroups(Math.ceil(this.cellCount / 64));
			fieldPass.end();
			this.fieldDirty = false;
			this.fieldUpdate = this.batchIndex;
		}
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
		this.beforeBuffer?.destroy();
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
		this.dropFree();
		this.soupStagingBuffer?.destroy();
		this.cellHashBuffer?.destroy();
		this.fetchStatusBuffer?.destroy();
		this.fieldBuffer?.destroy();
		this.cellHashStagingBuffer?.destroy();
		this.cellStagingBuffer?.destroy();
	}
}

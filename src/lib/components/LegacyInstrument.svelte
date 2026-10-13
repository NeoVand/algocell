<script lang="ts">
	import { GPUEngine } from '$lib/gpu/engine';
	import {
		DEFAULT_SEED,
		DEFAULT_NOISE_EXP,
		MAX_BATCH_PAIR_N,
		Z80_STEPS,
		type GridType,
		type GridConfig
	} from '$lib/sim/constants';
	import {
		disassemble as disassembleZ80,
		byteToMnemonic as z80ByteToMnemonic
	} from '$lib/z80-disasm';
	import { disassembleTape } from '$lib/lab/disassemble';
	import { ablationInstructions, ablationMasks } from '$lib/lab/ablation';
	import { PAPER_PRESETS, type PaperPreset } from '$lib/lab/presets';
	import { ABOUT_SECTIONS, GUIDE_SECTIONS, type DocumentationSection } from '$lib/lab/content';
	import type { LabSettings, LabIsa, CultureResult, InitialCondition } from '$lib/lab/types';
	import { ORIGINAL_SETTINGS } from '$lib/gpu/machine';
	import {
		resolveSuppression,
		countSuppressed,
		mnemonicOf,
		describeSuppression
	} from '$lib/z80-opcodes';
	import { unpackRGBA, createColormap, COLORMAP_NAMES } from '$lib/colormap';
	import type { ColormapName } from '$lib/colormap';
	import { untrack, onMount } from 'svelte';
	import Mermaid from '$lib/components/Mermaid.svelte';
	import { saveCheckpoint, loadCheckpoint, type CheckpointMetadata } from '$lib/checkpoint';
	import {
		type ApplyMode,
		type SettingKey,
		type SettingValues,
		type Preset,
		loadPresets,
		savePresets,
		makePresetId,
		valuesEqual
	} from '$lib/presets';
	import katex from 'katex';
	import 'katex/dist/katex.min.css';

	let colormapName: ColormapName = $state('rainbow');
	let colormap = $state(createColormap('rainbow'));
	let simpleView = $state(false);
	let showGridLines = $state(true);
	let showColorAdj = $state(false);
	let brightness = $state(0);
	let contrast = $state(1);
	let saturation = $state(1);

	let canvas: HTMLCanvasElement;
	let canvasContainer: HTMLDivElement;
	let engine = $state.raw<GPUEngine | null>(null);
	let gpuError = $state<string | null>(null);

	// Controls
	let seed = $state(DEFAULT_SEED);
	let noiseExp = $state(DEFAULT_NOISE_EXP);
	let pairCount = $state(MAX_BATCH_PAIR_N);
	let z80Steps = $state(Z80_STEPS);
	let playing = $state(true);
	let runDeadline = 0;
	let speed = $state(8); // steps per frame; 8x is the default
	let gridType = $state<GridType>('square');
	type MachineChoice = 'z80-0.2' | 'z80-0.3' | 'z80-nmos' | 'm6502' | 'i8086';
	let machine = $state<MachineChoice>('z80-0.2');
	let tapeLength = $state(16);
	let pairLength = $state(32);
	let lethalProbability = $state(0);
	let randomRegisters = $state(false);
	let stackConvention = $state<LabSettings['stack']>('legacy-sp');
	let dialMode = $state<LabSettings['dialMode']>('historical-pcg');
	let showConfinement = $state(false);
	let machineBusy = $state(false);
	let machineError = $state<string | null>(null);
	let selectedPaperId = $state('');
	let initialCondition = $state.raw<InitialCondition>({ kind: 'random' });
	const selectedPaper = $derived(PAPER_PRESETS.find((preset) => preset.id === selectedPaperId));
	const isa = $derived<LabIsa>(
		machine === 'm6502' ? 'm6502' : machine === 'i8086' ? 'i8086' : 'z80'
	);
	const instructionTable = $derived(ablationInstructions(isa));
	const isaSuppressionDescription = $derived(
		isa === 'm6502'
			? 'Use a mnemonic (STA, PHA), family (family:stack, family:writes-mem) or exact opcode (opcode:8D). A suppressed 6502 operation consumes only its opcode byte; operand bytes can execute next as code. The byte values remain in the soup.'
			: 'Use a mnemonic (MOV, PUSH), family (family:block-copy, family:stack) or exact opcode (opcode:A4). Suppression consumes prefixes and the opcode only; ModRM, displacement and immediate bytes can execute next as code. Groups are suppressed at opcode level, including every suboperation sharing the selected opcode. The byte values remain in the soup.'
	);
	let inspectedTape = $state.raw<Uint8Array | null>(null);
	let inspectedCell = $state(-1);
	let cultureResult = $state.raw<CultureResult | null>(null);
	let cultureBusy = $state(false);
	let cultureError = $state<string | null>(null);
	let inspectReadPending = false;
	let pointerDownX = 0;
	let pointerDownY = 0;
	let pointerMoved = false;
	let inspectClickTimer: ReturnType<typeof setTimeout> | undefined;

	function settings(): LabSettings {
		return {
			isa,
			compat: machine === 'z80-0.2' ? '0.2' : '0.3',
			variant: machine === 'z80-nmos' ? 'zilog-nmos' : isa === 'z80' ? 'classic' : 'nmos',
			length: tapeLength,
			ring: pairLength,
			steps: z80Steps,
			width: gridWidth,
			height: gridHeight,
			topology: gridType,
			seed,
			pairAttempts: pairCount,
			mutationExponent: noiseExp,
			lethalProbability,
			randomRegisters,
			ablation: [...suppressPatterns],
			stack: stackConvention,
			dialMode
		};
	}

	function disassemble(data: Uint8Array) {
		// Keep historical Z80 labels and use the added ISA decoders for the other cores.
		return isa === 'z80' ? disassembleZ80(data) : disassembleTape(isa, data);
	}

	function byteToMnemonic(byte: number): string {
		return isa === 'z80'
			? z80ByteToMnemonic(byte)
			: (instructionTable.find((row) => row.code === byte)?.mnemonic ?? hexByte(byte));
	}

	function getCellData(soup: Uint8Array, index: number): Uint8Array {
		const stride = Math.ceil(tapeLength / 4) * 4;
		return soup.slice(index * stride, index * stride + tapeLength);
	}

	function resolveIsaSuppression(patterns: string[]) {
		if (isa === 'z80') return resolveSuppression(patterns);
		const masks = ablationMasks(isa, patterns);
		// These sets are rebuilt only when a control changes, not on soup updates.
		// eslint-disable-next-line svelte/prefer-svelte-reactivity
		const base = new Set<number>();
		for (let byte = 0; byte < 256; byte++)
			if ((masks[byte >> 5] & (1 << (byte & 31))) !== 0) base.add(byte);
		return { base, cb: new Set<number>(), ed: new Set<number>() };
	}

	function suppressionPreview(input: string): number {
		try {
			return countSuppressed(
				resolveIsaSuppression(
					input
						.split(';')
						.map((p) => p.trim())
						.filter(Boolean)
				)
			);
		} catch {
			return 0;
		}
	}

	function assignMachineSettings(next: LabSettings) {
		machine =
			next.isa === 'z80'
				? next.compat === '0.2'
					? 'z80-0.2'
					: next.variant === 'zilog-nmos'
						? 'z80-nmos'
						: 'z80-0.3'
				: next.isa;
		tapeLength = next.length;
		pairLength = next.ring;
		lethalProbability = next.lethalProbability;
		randomRegisters = next.randomRegisters;
		stackConvention = next.stack;
		dialMode = next.dialMode;
		gridWidth = next.width;
		gridHeight = next.height;
		gridType = next.topology;
		seed = next.seed;
		pairCount = next.pairAttempts;
		z80Steps = next.steps;
		noiseExp = next.mutationExponent;
		suppressPatterns = [...next.ablation];
	}

	async function updateMachine(reset = false) {
		if (!engine || machineBusy) return;
		playing = false;
		machineBusy = true;
		machineError = null;
		try {
			if (reset) await engine.configure(settings(), initialCondition);
			else await engine.setMachine(settings());
			clearMeasurements();
		} catch (error) {
			machineError = error instanceof Error ? error.message : String(error);
		} finally {
			machineBusy = false;
		}
	}

	async function chooseMachine(event: Event) {
		const next = (event.target as HTMLSelectElement).value as MachineChoice;
		machine = next;
		suppressPatterns = [];
		selectedPaperId = '';
		stackConvention = next === 'm6502' || next === 'i8086' ? 'first-push' : 'top';
		dialMode = next === 'z80-0.2' ? 'historical-pcg' : 'counter';
		initialCondition = { kind: 'random' };
		await updateMachine(true);
	}

	async function applyPaperPreset(preset: PaperPreset) {
		if (machineBusy) return;
		assignMachineSettings(preset.config);
		selectedPaperId = preset.id;
		initialCondition = preset.initialCondition;
		await updateMachine(true);
	}

	async function changeConfinement(event: Event) {
		if (!engine || machineBusy) return;
		const next = (event.target as HTMLInputElement).checked;
		playing = false;
		machineBusy = true;
		machineError = null;
		try {
			await engine.setShowConfinement(next);
			showConfinement = next;
		} catch (error) {
			machineError = error instanceof Error ? error.message : String(error);
		} finally {
			machineBusy = false;
		}
	}

	async function inspectCell(index: number) {
		if (!engine || index < 0 || inspectReadPending) return;
		inspectReadPending = true;
		try {
			const soup = await engine.readSoupData();
			if (!soup.length) return;
			inspectedTape = getCellData(soup, index);
			inspectedCell = index;
			cultureResult = null;
			cultureError = null;
		} finally {
			inspectReadPending = false;
		}
	}

	async function runCulture(tape: Uint8Array) {
		if (!engine || cultureBusy) return;
		playing = false;
		cultureBusy = true;
		cultureError = null;
		cultureResult = null;
		try {
			cultureResult = await engine.runCulture(tape, { trials: 32, seed, generations: 2 });
		} catch (error) {
			cultureError = error instanceof Error ? error.message : String(error);
		} finally {
			cultureBusy = false;
		}
	}

	function scientificText(text: string): string {
		return text
			.replaceAll('new host-scheduled worlds', 'new GPU-scheduled worlds')
			.replaceAll('host scheduler', 'GPU scheduler')
			.replaceAll(
				'the update counter measures completed work, not seconds',
				'ops/s measures instruction slots of completed GPU work'
			)
			.replaceAll('host stream', 'GPU random stream')
			.replaceAll('pale zero-rich', 'zero-rich')
			.replaceAll(
				'instruction-stream reading enters the partner',
				'instruction-stream fetching leaves the focal tape (B or padding)'
			)
			.replaceAll(
				'“open” when it enters its partner',
				'“open” when it fetches outside that tape (B or padding)'
			)
			.replaceAll(
				'instruction fetching enters the partner',
				'instruction fetching leaves the focal tape (B or padding)'
			)
			.replaceAll(
				'no partner entry in all 16',
				'no fetch outside the focal tape (B or padding) in all 16'
			)
			.replaceAll('entry in at least eight is open', 'such a fetch in at least eight is open')
			.replaceAll(
				'Confinement describes instruction-stream addresses.',
				'Confinement describes whether instruction-stream fetching crosses the focal tape boundary into B or discarded padding.'
			)
			.replaceAll('new host’s per-encounter seeds', 'GPU kernel’s per-encounter seeds')
			.replaceAll(
				'0.3 engine and world API',
				'0.3.1 library core inside Algocell’s GPU soup kernel'
			)
			.replace(
				'The legacy instrument remains available for its original rendering, controls and checkpoints.',
				'The original Algocell rendering, controls and checkpoints remain the base of this instrument.'
			)
			.replace(
				'The interface’s visual language follows the manuscript’s figure plan: a white field, tangible byte blocks, quiet structural lines and separate encodings for byte value, execution and transmission.',
				'The interface keeps Algocell’s original dark instrument panels, tangible byte blocks and familiar color maps.'
			)
			.replace(
				'Press Run to advance the world, Pause to keep its current state, or Step for one population update.',
				'Press Play to advance the world or Pause to keep its current state.'
			)
			.replace(
				'Select a tape on the lattice or by cell index.',
				'Click a tape on the lattice or inspect an abundant species.'
			)
			.replace('Hexagonal legacy worlds', 'Hexagonal worlds');
	}

	const guideSections: DocumentationSection[] = GUIDE_SECTIONS.filter(
		(section) => section.id !== 'legacy'
	).map((section) => {
		if (section.id === 'lattice')
			return {
				...section,
				paragraphs: [
					'Byte view keeps Algocell’s original color maps. Each tape occupies a block at its actual lattice position; simple view averages its byte colors. A color or a visible texture is not a test of reproduction. Square and hexagonal neighborhoods preserve their original layouts.',
					'Colors continue to identify the same byte values when the processor changes. Their historical Z80 category colors do not imply equivalent 6502 or 8086 instruction meanings. Read the selected processor’s disassembly.',
					'The fetch layer records the last initiating encounter only. “Outside focal tape” means instruction-stream fetching reached any ring position at or beyond L, including B or discarded padding; “no observed exit” means it stayed in A during that encounter. Unknown means no current measurement. Mutated or modified cells can lose their measurement. This is not a lifetime confinement classification, and it does not measure partner-data reads, partner-information independence or heredity.'
				]
			};
		if (section.id === 'measurements')
			return {
				...section,
				title: 'Interpret the original charts',
				paragraphs: [
					'The opcode chart counts inherited byte values, even when a value is an operand or data rather than an executed instruction. The concentration factor compares a byte’s fraction with a uniform 1/256 baseline. Padding is excluded.',
					'The species and diversity tabs group full tapes by a GPU content hash. This fast composition view can have hash collisions; it is not an exact ancestry census. Shannon entropy, Simpson diversity and richness describe how those groups are distributed, not fitness, heredity or progress.',
					'An abundant exact-looking tape is a candidate to inspect and culture. A transmitting population can contain many different payloads, so a single top-tape share can hide it. Short visual runs do not reproduce the paper’s cohort evidence or emergence-time distributions.'
				]
			};
		if (section.id === 'exports')
			return {
				...section,
				paragraphs: [
					'A preset describes how to begin; a checkpoint stores the current population. Algocell keeps its original .acell format: a PNG preview with local metadata and the complete word-aligned soup attached. New checkpoints include the processor, ring geometry, reset, chemistry, suppression and planted condition. Original files restore historical Z80 compatibility 0.2 with SP = FFFF.',
					'Checkpoint populations can be inspected and continued, but the original format does not preserve every GPU random lane’s continuation state. Reloading a population is not a promise of the same subsequent trajectory. Keep the complete condition with any comparison; equal seeds alone do not identify an archived paper run.',
					'Files remain on your computer. No checkpoint is uploaded to a remote service. Source paths in the paper preset notes identify the companion manuscript’s registration, machine world and results.'
				]
			};
		return section;
	});

	// Compute grid dimensions from viewport aspect ratio
	// ~20K cells: enough for emergence, large enough to see patterns clearly
	// For hex grids, cells are visually compressed vertically by sqrt(3)/2 ≈ 0.866
	// so we can fit more rows in the same vertical space
	const HEX_Y_RATIO = 0.866025404; // sqrt(3)/2
	function computeGridDimensions(
		vw: number,
		vh: number,
		type: GridType = 'square'
	): { w: number; h: number } {
		const aspect = vw / vh;
		const TARGET_CELLS = 20_000;
		if (type === 'hex') {
			// Hex cells are compressed vertically: effective aspect = aspect / HEX_Y_RATIO
			// This means we need more rows to fill the same visual height
			const effectiveAspect = aspect * HEX_Y_RATIO;
			const h = Math.round(Math.sqrt(TARGET_CELLS / effectiveAspect));
			const w = Math.round(h * effectiveAspect);
			return {
				w: Math.max(40, Math.min(400, w)),
				h: Math.max(40, Math.min(400, h))
			};
		}
		const h = Math.round(Math.sqrt(TARGET_CELLS / aspect));
		const w = Math.round(h * aspect);
		return {
			w: Math.max(40, Math.min(400, w)),
			h: Math.max(40, Math.min(400, h))
		};
	}

	// Initial dimensions — will be recomputed on mount from actual viewport
	let gridWidth = $state(200);
	let gridHeight = $state(200);
	let gridInitialized = false;

	// Stats
	let opsPerSec = $state(0);
	let showHelp = $state(false);
	let helpTab = $state<
		'overview' | 'visuals' | 'z80' | 'params' | 'keys' | 'analysis' | 'about' | 'guide'
	>('overview');
	let showSpeedMenu = $state(false);
	let showSettings = $state(false);
	let toolbarCollapsed = $state(false);
	let openTip = $state<string | null>(null);
	let keyboardModality = $state(false);
	let showInfoChart = $state(true);
	let suppressPatterns: string[] = $state([]);
	let suppressInput = $state('');

	// Resolve patterns → per-page suppression sets (base / CB / ED), see $lib/z80-opcodes.
	let suppression = $derived(resolveIsaSuppression(suppressPatterns));
	// Base-page set: what the frequency tiles (which show base-page bytes) can mark.
	let suppressedOpcodes = $derived(suppression.base);
	let suppressedTotal = $derived(countSuppressed(suppression));

	// Frequency chart: track top N bytes normalized over time
	// Total bytes counted by shader = cells * wordsPerCell * 4 (word-aligned)
	let TOTAL_BYTES = $derived(gridWidth * gridHeight * tapeLength);
	const MAX_TRACKED = 10;
	const MAX_HISTORY = 300;

	// Efficient ring-buffer chart storage: Float32Array(256) per point for O(1) byte lookup
	// Non-reactive — we only trigger Svelte updates via chartVersion counter
	const chartRing: Float32Array[] = []; // ring buffer of Float32Array(256), each indexed by byte
	let chartLen = $state(0); // current number of valid entries
	let chartVersion = $state(0); // bump to trigger reactive re-derive

	// Tracked bytes: stable set of bytes we draw lines for, with colormap-derived colors
	let trackedBytes: { byte: number; mnemonic: string }[] = $state([]);

	// Chart hover
	let chartHoveredByte = $state(-1);

	// Info widget tab: opcodes (byte frequency) or species (block identity)
	let infoTab = $state<'opcodes' | 'species' | 'diversity'>('opcodes');

	// === Species tracking ===
	interface SpeciesEntry {
		hash: number;
		count: number;
		fraction: number;
		color: string;
		textColor: string;
		exemplar: Uint8Array | null; // cell bytes for this species
		exemplarCells: GenomeCell[]; // disassembled genome grid
	}
	interface SpeciesSnapshot {
		species: SpeciesEntry[];
		richness: number; // unique hashes above threshold
		shannon: number; // Shannon entropy H
		simpson: number; // Simpson's index D = 1 - sum(p_i^2)
		totalUnique: number;
	}
	const MAX_SPECIES_DISPLAY = 10;
	const MAX_SPECIES_TRACKED = 20;
	const SPECIES_THRESHOLD = 0.001; // 0.1% of cells

	// Species ring buffer: stores fraction for each tracked species hash over time
	// speciesHashOrder[i] = hash for index i (stable across frames when possible)
	const speciesRing: Float32Array[] = [];
	let speciesHashOrder: number[] = $state([]); // maps index -> hash
	/* eslint-disable svelte/prefer-svelte-reactivity -- not reactive, only read inside derived */
	let speciesExemplars: Map<number, Uint8Array> = new Map(); // hash -> exemplar bytes
	/* eslint-enable svelte/prefer-svelte-reactivity */
	let speciesLen = $state(0);
	let speciesVersion = $state(0);
	let currentSpecies = $state<SpeciesSnapshot | null>(null);
	let speciesHoveredIdx = $state(-1);
	let speciesModal = $state<SpeciesEntry | null>(null);
	let speciesModalTip = $state<{ idx: number; byteVal: number; x: number; y: number } | null>(null);
	let speciesLoading = false;

	// Diversity metrics time series ring buffer
	// Each slot: [shannon, simpson, richness, totalUnique]
	const diversityRing: Float32Array[] = [];
	let diversityLen = $state(0);
	let diversityVersion = $state(0);

	function speciesAvgColor(exemplar: Uint8Array | null): { color: string; textColor: string } {
		if (!exemplar || exemplar.length === 0) {
			return { color: 'rgba(128,128,128,0.5)', textColor: 'rgba(255,255,255,0.9)' };
		}
		let rSum = 0,
			gSum = 0,
			bSum = 0;
		for (let i = 0; i < exemplar.length; i++) {
			const [r, g, b] = unpackRGBA(colormap[exemplar[i]]);
			rSum += r;
			gSum += g;
			bSum += b;
		}
		const n = exemplar.length;
		const r = Math.round(rSum / n);
		const g = Math.round(gSum / n);
		const b = Math.round(bSum / n);
		const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
		return {
			color: `rgb(${r},${g},${b})`,
			textColor: lum > 0.4 ? 'rgba(0,0,0,0.85)' : 'rgba(255,255,255,0.9)'
		};
	}

	/** Brightened species color for chart lines */
	function speciesChartColor(exemplar: Uint8Array | null): string {
		const { color } = speciesAvgColor(exemplar);
		// Parse and brighten if too dark
		const m = color.match(/rgb\((\d+),(\d+),(\d+)\)/);
		if (!m) return color;
		let [r, g, b] = [+m[1], +m[2], +m[3]];
		const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
		if (lum < 0.2) {
			const boost = 0.35;
			r = Math.round(r + (255 - r) * boost);
			g = Math.round(g + (255 - g) * boost);
			b = Math.round(b + (255 - b) * boost);
		}
		return `rgb(${r},${g},${b})`;
	}

	// Hover / genome tooltip
	let hoveredCell = $state(-1);
	let cellData = $state.raw<Uint8Array | null>(null);
	let disasmLines: ReturnType<typeof disassemble> = $state.raw([]);
	let mouseX = $state(0);
	let mouseY = $state(0);
	let canvasW = $state(800);
	let canvasH = $state(800);
	let tooltipRefreshTimer: ReturnType<typeof setInterval> | undefined;

	// Panning state
	let isPanning = $state(false);
	let panStartX = 0;
	let panStartY = 0;

	let frameCount = 0;
	let animFrameId: number | undefined;
	let statsLoading = false;

	// Build frequency lines using log-scaled concentration factor
	// concentration = frac * 256 (uniform = 1.0, a byte at 50% = 128.0)
	// Log scale ensures both dominant and minor bytes are visible
	let freqLines = $derived.by(() => {
		// Touch chartVersion to subscribe to updates
		void chartVersion;
		const len = chartLen;
		if (len < 2) return [];

		const chartBytes = displayTiles.filter((t) => t.type === 'byte');
		const allSeries = chartBytes.map((tb) => {
			const values = new Float64Array(len);
			for (let i = 0; i < len; i++) {
				const frac = chartRing[i][tb.byte];
				const conc = frac * 256; // concentration factor
				values[i] = conc > 0 ? Math.log2(conc + 1) : 0;
			}
			return {
				byte: tb.byte,
				mnemonic: tb.mnemonic,
				color: byteChartColor(tb.byte),
				values
			};
		});
		// Shared max across all lines
		let globalMax = Math.log2(2.5);
		for (const s of allSeries) {
			for (let i = 0; i < s.values.length; i++) {
				if (s.values[i] > globalMax) globalMax = s.values[i];
			}
		}
		return allSeries.map((s) => ({
			byte: s.byte,
			mnemonic: s.mnemonic,
			color: s.color,
			path: buildSparklinePathTyped(s.values, 200, 120, globalMax)
		}));
	});

	let tooltipStyle = $derived(computeTooltipStyle(mouseX, mouseY));

	let genomeCells = $derived(
		cellData && disasmLines.length > 0 ? buildGenomeGrid(cellData, disasmLines) : []
	);

	function computeTooltipStyle(mx: number, my: number): string {
		const isHex = gridType === 'hex' && tapeLength === 19;
		const tooltipW = isHex ? 280 : 196;
		const tooltipH = isHex ? 260 : Math.ceil(tapeLength / 4) * 48 + 4;
		const vw = window.innerWidth;
		const vh = window.innerHeight;
		const gap = 12;

		// Center tooltip horizontally on the cursor
		let x = mx - tooltipW / 2;
		let y: number;

		// Try placing above the cursor first
		if (my - gap - tooltipH >= 4) {
			y = my - gap - tooltipH;
		} else {
			// Place below
			y = my + gap;
		}

		// Keep within viewport
		if (x + tooltipW > vw - 4) x = vw - 4 - tooltipW;
		if (x < 4) x = 4;
		if (y + tooltipH > vh - 4) y = vh - 4 - tooltipH;
		if (y < 4) y = 4;
		return `left:${x}px;top:${y}px`;
	}

	function buildSparklinePathTyped(
		values: Float64Array,
		w: number,
		h: number,
		max: number
	): string {
		if (values.length < 2) return '';
		const step = w / (values.length - 1);
		const parts: string[] = new Array(values.length);
		for (let i = 0; i < values.length; i++) {
			const x = i * step;
			const y = h - (values[i] / max) * (h - 4) - 2;
			parts[i] = `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
		}
		return parts.join(' ');
	}

	onMount(() => {
		if (!canvas) return;

		// Compute grid dimensions from actual viewport on first mount
		if (!gridInitialized) {
			const initialGridType = untrack(() => gridType);
			const dims = computeGridDimensions(window.innerWidth, window.innerHeight, initialGridType);
			gridWidth = dims.w;
			gridHeight = dims.h;
			gridInitialized = true;
		}

		const initialSeed = untrack(() => seed);
		const initialConfig: GridConfig = untrack(() => ({
			width: gridWidth,
			height: gridHeight,
			gridType: gridType,
			tapeLength,
			pairLength
		}));
		// Set canvas pixel dimensions before init so resetView gets correct aspect
		const dpr = window.devicePixelRatio || 1;
		canvas.width = canvasW = Math.round(canvas.clientWidth * dpr);
		canvas.height = canvasH = Math.round(canvas.clientHeight * dpr);

		const eng = new GPUEngine(initialSeed, initialConfig);
		let disposed = false;

		eng
			.init(canvas)
			.then((ok) => {
				if (disposed) {
					eng.destroy();
					return;
				}
				if (!ok) {
					gpuError = 'WebGPU not available. Please use Chrome 113+ or Edge 113+.';
					return;
				}
				engine = eng;
				runDeadline = performance.now() + 55_000;
				startLoop();
				if (import.meta.env.DEV) {
					// Dev-only handle for headless experiments and debugging from the console.
					(window as unknown as { __algocell?: unknown }).__algocell = {
						engine: eng,
						resolveSuppression
					};
				}
			})
			.catch((error) => {
				if (!disposed) gpuError = String(error);
			});

		return () => {
			disposed = true;
			clearInterval(tooltipRefreshTimer);
			clearTimeout(inspectClickTimer);
			if (animFrameId !== undefined) {
				cancelAnimationFrame(animFrameId);
				animFrameId = undefined;
			}
			eng.destroy();
		};
	});

	// ResizeObserver for canvas
	onMount(() => {
		if (!canvasContainer) return;
		const ro = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const { width, height } = entry.contentRect;
				canvasW = Math.round(width * devicePixelRatio);
				canvasH = Math.round(height * devicePixelRatio);
				if (canvas) {
					canvas.width = canvasW;
					canvas.height = canvasH;
				}
			}
		});
		ro.observe(canvasContainer);
		return () => ro.disconnect();
	});

	function startLoop() {
		function tick() {
			animFrameId = requestAnimationFrame(tick);
			if (!engine) return;

			if (playing && (document.hidden || performance.now() >= runDeadline)) playing = false;
			if (playing && !machineBusy && !cultureBusy) {
				for (let i = 0; i < speed; i++) {
					engine.simulateStep();
				}
			}

			if (!machineBusy) {
				engine.render(canvas);
				flushCellRefresh();
			}
			frameCount++;

			// Update stats: immediately on first frame, then every 15 frames
			if (!machineBusy && playing && (frameCount === 1 || frameCount % 15 === 0) && !statsLoading) {
				opsPerSec = engine.opsPerSec;
				updateTopBytes();
				if (infoTab === 'species' || infoTab === 'diversity') {
					updateSpecies();
				}
			}
		}
		tick();
	}

	async function updateTopBytes() {
		if (!engine || statsLoading) return;
		statsLoading = true;
		try {
			const counts = await engine.readStats();
			const entries: { byte: number; count: number; mnemonic: string }[] = [];
			for (let i = 0; i < 256; i++) {
				if (counts[i] > 0) {
					entries.push({ byte: i, count: counts[i], mnemonic: byteToMnemonic(i) });
				}
			}
			entries.sort((a, b) => b.count - a.count);

			// Track every non-zero byte, ranked by frequency. displayTiles filters
			// out suppressed opcodes and shows the top non-suppressed ones, so we
			// must keep the full ranked list here — otherwise broad suppression
			// (e.g. "LD" matches 100+ opcodes) can consume every tracked slot and
			// leave the top-10 empty. Keeping all non-zero bytes lets the freed
			// slots refill with the next-most-common non-suppressed opcodes.
			trackedBytes = entries.map((e) => ({ byte: e.byte, mnemonic: e.mnemonic }));

			// Only record chart history when chart is visible
			if (showInfoChart) {
				// Reuse or allocate a Float32Array(256) for O(1) byte lookup
				let slot: Float32Array;
				if (chartLen < MAX_HISTORY) {
					slot = new Float32Array(256);
					chartRing[chartLen] = slot;
					chartLen++;
				} else {
					// Shift ring: move slot 0 to end, shift everything left
					slot = chartRing[0];
					for (let i = 1; i < MAX_HISTORY; i++) chartRing[i - 1] = chartRing[i];
					chartRing[MAX_HISTORY - 1] = slot;
					slot.fill(0);
				}
				const total = TOTAL_BYTES;
				for (const e of entries) {
					slot[e.byte] = e.count / total;
				}
				chartVersion++;
			}
		} catch {
			// stats readback failed silently
		} finally {
			statsLoading = false;
		}
	}

	async function updateSpecies() {
		if (!engine || speciesLoading) return;
		speciesLoading = true;
		try {
			const hashes = await engine.readCellHashes();
			if (hashes.length === 0) return;

			// Count unique hashes and find first exemplar cell index for each
			/* eslint-disable svelte/prefer-svelte-reactivity -- local function variables, not reactive */
			const counts = new Map<number, number>();
			const exemplarIdx = new Map<number, number>(); // hash -> first cell index
			/* eslint-enable svelte/prefer-svelte-reactivity */
			for (let i = 0; i < hashes.length; i++) {
				const h = hashes[i];
				counts.set(h, (counts.get(h) || 0) + 1);
				if (!exemplarIdx.has(h)) exemplarIdx.set(h, i);
			}

			const total = hashes.length;
			const threshold = Math.max(1, Math.floor(total * SPECIES_THRESHOLD));

			// Compute diversity metrics
			let richness = 0;
			let shannonH = 0;
			let simpsonSum = 0;
			for (const [, count] of counts) {
				if (count >= threshold) richness++;
				const p = count / total;
				if (p > 0) {
					shannonH -= p * Math.log(p);
					simpsonSum += p * p;
				}
			}

			// Read soup data to extract exemplar cell bytes
			const soupData = await engine.readSoupData();

			// Sort by frequency, take top species
			const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
			const topSpecies: SpeciesEntry[] = sorted
				.slice(0, MAX_SPECIES_TRACKED)
				.map(([hash, count]) => {
					const cellIdx = exemplarIdx.get(hash) ?? 0;
					const exemplar = soupData.length > 0 ? getCellData(soupData, cellIdx) : null;
					const { color, textColor } = speciesAvgColor(exemplar);
					const exemplarCells = exemplar ? buildGenomeGrid(exemplar, disassemble(exemplar)) : [];
					return {
						hash,
						count,
						fraction: count / total,
						color,
						textColor,
						exemplar,
						exemplarCells
					};
				});

			currentSpecies = {
				species: topSpecies,
				richness,
				shannon: shannonH,
				simpson: 1 - simpsonSum,
				totalUnique: counts.size
			};

			// Record diversity metrics time series
			if (showInfoChart) {
				let slot: Float32Array;
				if (diversityLen < MAX_HISTORY) {
					slot = new Float32Array(4);
					diversityRing[diversityLen] = slot;
					diversityLen++;
				} else {
					slot = diversityRing[0];
					for (let i = 1; i < MAX_HISTORY; i++) diversityRing[i - 1] = diversityRing[i];
					diversityRing[MAX_HISTORY - 1] = slot;
				}
				slot[0] = shannonH;
				slot[1] = 1 - simpsonSum;
				slot[2] = richness;
				slot[3] = counts.size;
				diversityVersion++;
			}

			// Cache exemplar data for chart colors
			for (const sp of topSpecies) {
				if (sp.exemplar) speciesExemplars.set(sp.hash, sp.exemplar);
			}

			// Update ring buffer for sparklines
			if (showInfoChart) {
				// Build or update hash order (keep stable identity when possible)
				const newHashes = topSpecies.map((s) => s.hash);
				if (speciesHashOrder.length === 0) {
					speciesHashOrder = newHashes;
				} else {
					// Add any new hashes not yet tracked
					for (const h of newHashes) {
						if (!speciesHashOrder.includes(h)) {
							if (speciesHashOrder.length < MAX_SPECIES_TRACKED) {
								speciesHashOrder.push(h);
							}
						}
					}
				}

				// Record fractions for all tracked hashes
				let slot: Float32Array;
				if (speciesLen < MAX_HISTORY) {
					slot = new Float32Array(MAX_SPECIES_TRACKED);
					speciesRing[speciesLen] = slot;
					speciesLen++;
				} else {
					slot = speciesRing[0];
					for (let i = 1; i < MAX_HISTORY; i++) speciesRing[i - 1] = speciesRing[i];
					speciesRing[MAX_HISTORY - 1] = slot;
					slot.fill(0);
				}
				for (let idx = 0; idx < speciesHashOrder.length; idx++) {
					const c = counts.get(speciesHashOrder[idx]);
					slot[idx] = c ? c / total : 0;
				}
				speciesVersion++;
			}
		} catch {
			// silent fail
		} finally {
			speciesLoading = false;
		}
	}

	// Species sparklines (parallel to freqLines)
	let speciesLines = $derived.by(() => {
		void speciesVersion;
		const len = speciesLen;
		if (len < 2 || speciesHashOrder.length === 0) return [];

		const lines: { hash: number; color: string; path: string }[] = [];
		let globalMax = 0.1;

		// Pre-compute all series values to find globalMax
		const allValues: Float64Array[] = [];
		for (let idx = 0; idx < Math.min(speciesHashOrder.length, MAX_SPECIES_DISPLAY); idx++) {
			const values = new Float64Array(len);
			for (let i = 0; i < len; i++) {
				const frac = speciesRing[i][idx];
				values[i] = frac > 0 ? Math.log2(frac * 100 + 1) : 0;
			}
			allValues.push(values);
			for (let i = 0; i < values.length; i++) {
				if (values[i] > globalMax) globalMax = values[i];
			}
		}

		for (let idx = 0; idx < allValues.length; idx++) {
			const hash = speciesHashOrder[idx];
			const exemplar = speciesExemplars.get(hash) ?? null;
			lines.push({
				hash,
				color: speciesChartColor(exemplar),
				path: buildSparklinePathTyped(allValues[idx], 200, 120, globalMax)
			});
		}
		return lines;
	});

	// Diversity sparklines: each metric gets its own line with independent scaling
	const DIVERSITY_METRICS = [
		{ key: 0, label: 'H', name: 'Shannon entropy', color: 'rgb(255,180,50)' },
		{ key: 1, label: 'D', name: 'Simpson diversity', color: 'rgb(100,200,255)' },
		{ key: 2, label: 'spp', name: 'Species richness', color: 'rgb(120,255,120)' },
		{ key: 3, label: 'uniq', name: 'Unique blocks', color: 'rgb(255,130,180)' }
	] as const;

	let diversityHoveredIdx = $state(-1);

	let diversityLines = $derived.by(() => {
		void diversityVersion;
		const len = diversityLen;
		if (len < 2) return [];

		return DIVERSITY_METRICS.map((m, idx) => {
			const values = new Float64Array(len);
			let maxVal = 0;
			for (let i = 0; i < len; i++) {
				values[i] = diversityRing[i][m.key];
				if (values[i] > maxVal) maxVal = values[i];
			}
			// Each metric is normalized to its own max
			const normMax = maxVal > 0 ? maxVal : 1;
			return {
				idx,
				label: m.label,
				name: m.name,
				color: m.color,
				path: buildSparklinePathTyped(values, 200, 120, normMax),
				current: len > 0 ? diversityRing[len - 1][m.key] : 0
			};
		});
	});

	function handleCanvasMouseMove(e: MouseEvent) {
		if (touchHandled) return; // ignore synthetic mouse events from touch
		if (!engine || !canvas) return;
		const rect = canvas.getBoundingClientRect();
		const sx = e.clientX - rect.left;
		const sy = e.clientY - rect.top;
		mouseX = e.clientX;
		mouseY = e.clientY;

		if (isPanning) {
			if (Math.hypot(e.clientX - pointerDownX, e.clientY - pointerDownY) > 4) pointerMoved = true;
			const dx = e.clientX - panStartX;
			const dy = e.clientY - panStartY;
			engine.pan(dx, dy, rect.width, rect.height);
			panStartX = e.clientX;
			panStartY = e.clientY;
			return;
		}

		const scaleX = canvasW / rect.width;
		const scaleY = canvasH / rect.height;
		const cell = engine.screenToCell(sx * scaleX, sy * scaleY, canvasW, canvasH);

		if (cell >= 0) {
			if (cell !== hoveredCell) {
				hoveredCell = cell;
				engine.setHoverCell(cell);
				clearInterval(tooltipRefreshTimer);
				tooltipRefreshTimer = setInterval(() => refreshCellData(cell), 500);
			}
			refreshCellData(cell);
		} else if (cell < 0) {
			hoveredCell = -1;
			cellData = null;
			disasmLines = [];
			engine.setHoverCell(-1);
			clearInterval(tooltipRefreshTimer);
		}
	}

	let pendingCellRefresh = -1;
	let cellRefreshPending = false;
	let lastCellRefresh = 0;

	function refreshCellData(cell: number) {
		// Mark for readback after next render (so data matches what's on screen)
		pendingCellRefresh = cell;
	}

	function flushCellRefresh() {
		if (
			pendingCellRefresh < 0 ||
			!engine ||
			cellRefreshPending ||
			performance.now() - lastCellRefresh < 500
		)
			return;
		const cell = pendingCellRefresh;
		pendingCellRefresh = -1;
		cellRefreshPending = true;
		lastCellRefresh = performance.now();
		engine
			.readSoupData()
			.then((soupData) => {
				if (soupData.length === 0 || hoveredCell !== cell) return;
				const data = getCellData(soupData, cell);
				cellData = data;
				disasmLines = disassemble(data);
			})
			.finally(() => {
				cellRefreshPending = false;
			});
	}

	// Track if touch just handled an interaction, to suppress synthetic mouse events
	let touchHandled = false;

	function handleCanvasMouseDown(e: MouseEvent) {
		if (touchHandled) return; // ignore synthetic mouse events from touch
		if (e.button === 0 || e.button === 1) {
			e.preventDefault();
			isPanning = true;
			pointerMoved = false;
			pointerDownX = e.clientX;
			pointerDownY = e.clientY;
			panStartX = e.clientX;
			panStartY = e.clientY;
			hoveredCell = -1;
			cellData = null;
			disasmLines = [];
			engine?.setHoverCell(-1);
		}
	}

	function handleCanvasMouseUp(e: MouseEvent) {
		if (touchHandled) return;
		if (e.button === 0 || e.button === 1) {
			isPanning = false;
			if (e.button === 0 && !pointerMoved && engine && canvas) {
				const rect = canvas.getBoundingClientRect();
				const cell = engine.screenToCell(
					((e.clientX - rect.left) * canvasW) / rect.width,
					((e.clientY - rect.top) * canvasH) / rect.height,
					canvasW,
					canvasH
				);
				clearTimeout(inspectClickTimer);
				inspectClickTimer = setTimeout(() => {
					void inspectCell(cell);
				}, 220);
			}
		}
	}

	function handleCanvasWheel(e: WheelEvent) {
		if (!engine || !canvas) return;
		e.preventDefault();
		const rect = canvas.getBoundingClientRect();
		const sx = (e.clientX - rect.left) * (canvasW / rect.width);
		const sy = (e.clientY - rect.top) * (canvasH / rect.height);
		// Proportional to scroll amount, clamped for consistency between trackpad/mouse
		const normalizedDelta = Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 100);
		const zoomSpeed = 0.002;
		const factor = Math.exp(normalizedDelta * zoomSpeed);
		engine.zoomAt(sx, sy, canvasW, canvasH, factor);
	}

	// --- Touch events for mobile pinch-zoom, pan & tap-to-inspect ---
	let lastTouchDist = 0;
	let lastTouchX = 0;
	let lastTouchY = 0;
	let touchPanning = false;
	let touchStartX = 0;
	let touchStartY = 0;
	let touchMoved = false;
	const TAP_THRESHOLD = 8; // pixels of movement allowed for a tap

	function handleTouchStart(e: TouchEvent) {
		if (!engine || !canvas) return;
		e.preventDefault();
		// Suppress synthetic mouse events that fire after touch
		touchHandled = true;
		if (e.touches.length === 2) {
			// Pinch start
			const dx = e.touches[1].clientX - e.touches[0].clientX;
			const dy = e.touches[1].clientY - e.touches[0].clientY;
			lastTouchDist = Math.sqrt(dx * dx + dy * dy);
			lastTouchX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
			lastTouchY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
			touchPanning = false;
			touchMoved = true; // multi-touch is never a tap
		} else if (e.touches.length === 1) {
			// Pan start — also track for potential tap
			touchPanning = true;
			touchMoved = false;
			touchStartX = panStartX = e.touches[0].clientX;
			touchStartY = panStartY = e.touches[0].clientY;
		}
	}

	function handleTouchMove(e: TouchEvent) {
		if (!engine || !canvas) return;
		e.preventDefault();
		const rect = canvas.getBoundingClientRect();
		if (e.touches.length === 2) {
			// Pinch zoom
			const dx = e.touches[1].clientX - e.touches[0].clientX;
			const dy = e.touches[1].clientY - e.touches[0].clientY;
			const dist = Math.sqrt(dx * dx + dy * dy);
			if (lastTouchDist > 0) {
				const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
				const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
				const sx = (midX - rect.left) * (canvasW / rect.width);
				const sy = (midY - rect.top) * (canvasH / rect.height);
				const factor = lastTouchDist / dist; // invert: pinch out = zoom in
				engine.zoomAt(sx, sy, canvasW, canvasH, factor);
				// Also pan with pinch movement
				const panDx = midX - lastTouchX;
				const panDy = midY - lastTouchY;
				engine.pan(panDx, panDy, rect.width, rect.height);
				lastTouchX = midX;
				lastTouchY = midY;
			}
			lastTouchDist = dist;
			touchPanning = false;
		} else if (e.touches.length === 1 && touchPanning) {
			const dx = e.touches[0].clientX - panStartX;
			const dy = e.touches[0].clientY - panStartY;
			// Check if finger moved enough to count as pan (not tap)
			const totalDx = e.touches[0].clientX - touchStartX;
			const totalDy = e.touches[0].clientY - touchStartY;
			if (Math.abs(totalDx) > TAP_THRESHOLD || Math.abs(totalDy) > TAP_THRESHOLD) {
				touchMoved = true;
			}
			if (touchMoved) {
				// Dismiss tooltip when panning starts
				if (hoveredCell >= 0) {
					hoveredCell = -1;
					cellData = null;
					disasmLines = [];
					engine.setHoverCell(-1);
					clearInterval(tooltipRefreshTimer);
				}
				engine.pan(dx, dy, rect.width, rect.height);
			}
			panStartX = e.touches[0].clientX;
			panStartY = e.touches[0].clientY;
		}
	}

	function handleTouchEnd(e: TouchEvent) {
		// Reset touchHandled after synthetic events have fired (~300ms)
		setTimeout(() => {
			touchHandled = false;
		}, 400);

		if (e.touches.length === 0) {
			// All fingers lifted — check for tap
			if (!touchMoved && engine && canvas) {
				const rect = canvas.getBoundingClientRect();
				const sx = (touchStartX - rect.left) * (canvasW / rect.width);
				const sy = (touchStartY - rect.top) * (canvasH / rect.height);
				const cell = engine.screenToCell(sx, sy, canvasW, canvasH);

				if (cell >= 0 && cell !== hoveredCell) {
					// Tap on a cell — preserve the hover view and open an inspector.
					void inspectCell(cell);
					hoveredCell = cell;
					mouseX = touchStartX;
					mouseY = touchStartY;
					engine.setHoverCell(cell);
					clearInterval(tooltipRefreshTimer);
					tooltipRefreshTimer = setInterval(() => refreshCellData(cell), 500);
					refreshCellData(cell);
				} else {
					// Tap on same cell or empty space — dismiss tooltip
					hoveredCell = -1;
					cellData = null;
					disasmLines = [];
					engine.setHoverCell(-1);
					clearInterval(tooltipRefreshTimer);
				}
			}
			lastTouchDist = 0;
			touchPanning = false;
			touchMoved = false;
		} else if (e.touches.length === 1) {
			// Went from pinch to single finger — start pan
			lastTouchDist = 0;
			touchPanning = true;
			touchMoved = true; // already moved (was pinching)
			panStartX = e.touches[0].clientX;
			panStartY = e.touches[0].clientY;
		}
	}

	function handleCanvasMouseLeave() {
		hoveredCell = -1;
		cellData = null;
		disasmLines = [];
		isPanning = false;
		engine?.setHoverCell(-1);
		clearInterval(tooltipRefreshTimer);
	}

	function handleCanvasDblClick() {
		clearTimeout(inspectClickTimer);
		inspectedTape = null;
		engine?.resetView(canvasW, canvasH);
	}

	function handleReset() {
		if (!engine || machineBusy) return;
		if (initialCondition.kind !== 'random') {
			void updateMachine(true);
			return;
		}
		engine.reset(seed);
		clearMeasurements();
	}

	function clearMeasurements() {
		inspectedTape = null;
		cultureResult = null;
		cultureError = null;
		opsPerSec = 0;
		chartRing.length = 0;
		chartLen = 0;
		chartVersion++;
		trackedBytes = [];
		speciesRing.length = 0;
		speciesHashOrder = [];
		speciesLen = 0;
		speciesVersion++;
		currentSpecies = null;
		speciesExemplars = new Map();
		speciesModal = null;
		diversityRing.length = 0;
		diversityLen = 0;
		diversityVersion++;
	}

	// Checkpoint save/load
	let fileInput: HTMLInputElement;

	async function handleSave() {
		if (!engine || !canvas) return;
		const blob = await new Promise<Blob>((resolve) => {
			canvas.toBlob((b) => resolve(b!), 'image/png');
		});
		const soupData = await engine.readSoupData();
		const metadata: CheckpointMetadata = {
			version: 1,
			machineSettings: engine.settings,
			initialCondition,
			gridWidth,
			gridHeight,
			gridType,
			seed,
			noiseExp,
			z80Steps: engine.z80Steps,
			pairCount: engine.pairCount,
			suppressPatterns: [...suppressPatterns],
			batchCount: engine.batchCount,
			timestamp: Date.now()
		};
		const checkpoint = await saveCheckpoint(blob, metadata, soupData);
		const url = URL.createObjectURL(checkpoint);
		const a = document.createElement('a');
		a.href = url;
		const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
		a.download = `algocell-${ts}.acell`;
		a.click();
		URL.revokeObjectURL(url);
	}

	async function handleLoadFile(file: File) {
		if (!engine || machineBusy) return;
		playing = false;
		machineBusy = true;
		machineError = null;
		try {
			const result = await loadCheckpoint(new Uint8Array(await file.arrayBuffer()));
			if (!result) throw new Error('This file does not contain an Algocell checkpoint.');
			const { metadata, soupData } = result;
			const restored: LabSettings = metadata.machineSettings ?? {
				...ORIGINAL_SETTINGS,
				width: metadata.gridWidth,
				height: metadata.gridHeight,
				topology: metadata.gridType,
				length: metadata.gridType === 'hex' ? 19 : 16,
				ring: metadata.gridType === 'hex' ? 38 : 32,
				seed: metadata.seed,
				mutationExponent: metadata.noiseExp,
				steps: metadata.z80Steps,
				pairAttempts: metadata.pairCount,
				ablation: [...metadata.suppressPatterns]
			};
			assignMachineSettings(restored);
			initialCondition = metadata.initialCondition ?? { kind: 'random' };
			selectedPaperId = '';
			await engine.configure(restored, initialCondition);
			engine.writeSoupData(soupData);
			clearMeasurements();
		} catch (error) {
			machineError = error instanceof Error ? error.message : String(error);
		} finally {
			machineBusy = false;
		}
	}

	function handleFileInputChange(e: Event) {
		const input = e.target as HTMLInputElement;
		if (input.files?.[0]) {
			handleLoadFile(input.files[0]);
			input.value = '';
		}
	}

	function handleDrop(e: DragEvent) {
		e.preventDefault();
		const file = e.dataTransfer?.files[0];
		if (file) handleLoadFile(file);
	}

	function handleSeedChange(e: Event) {
		const val = parseInt((e.target as HTMLInputElement).value);
		if (!isNaN(val) && val >= 0) {
			seed = val;
		}
	}

	function handlePairCountChange(e: Event) {
		const val = parseInt((e.target as HTMLInputElement).value);
		if (!isNaN(val) && engine) {
			pairCount = val;
			engine.pairCount = val;
		}
	}

	function handleColormapChange(name: ColormapName) {
		colormapName = name;
		colormap = createColormap(name);
		engine?.updateColormap(colormap);
	}

	function handleGridTypeChange(type: GridType) {
		gridType = type;
		tapeLength = type === 'hex' ? 19 : 16;
		pairLength = tapeLength * 2;
		selectedPaperId = '';
		initialCondition = { kind: 'random' };
		// Recompute grid dimensions for the new grid type
		const dims = computeGridDimensions(window.innerWidth, window.innerHeight, type);
		gridWidth = dims.w;
		gridHeight = dims.h;
		applyGridConfig();
	}

	function applyGridConfig() {
		void updateMachine(true);
	}

	function addSuppressPattern(pattern: string) {
		// Support multiple patterns separated by semicolons
		// (commas are NOT delimiters because Z80 mnemonics contain commas, e.g. LD A,B)
		const parts = pattern
			.split(';')
			.map((s) => s.trim().toUpperCase())
			.filter(Boolean);
		let added = false;
		for (const p of parts) {
			if (!p || suppressPatterns.includes(p)) continue;
			try {
				ablationMasks(isa, [p]);
			} catch (error) {
				machineError = error instanceof Error ? error.message : String(error);
				continue;
			}
			suppressPatterns.push(p);
			added = true;
		}
		if (added) {
			// Svelte needs identity change for array reactivity
			suppressPatterns = [...suppressPatterns];
		}
	}

	function removeSuppressPattern(pattern: string) {
		suppressPatterns = suppressPatterns.filter((p) => p !== pattern);
	}

	function toggleSuppressOpcode(opcode: number) {
		tileTooltip = null;
		chartHoveredByte = -1;
		const mnemonic = isa === 'z80' ? mnemonicOf('base', opcode) : `opcode:${hexByte(opcode)}`;
		if (suppressedOpcodes.has(opcode)) {
			// Find and remove patterns that match this opcode
			// If the exact mnemonic is a pattern, remove it; otherwise remove patterns that only match this one
			const upper = mnemonic.toUpperCase();
			if (suppressPatterns.includes(upper)) {
				removeSuppressPattern(upper);
			}
		} else {
			addSuppressPattern(mnemonic);
		}
	}

	function clearSuppressedOpcodes() {
		suppressPatterns = [];
	}

	// ── Presets ────────────────────────────────────────────────────────────
	// Declarative schema: one entry per tunable setting. `mode` says how
	// disruptive a change is (see presets.ts). `get`/`set` wire the setting to
	// this component's state + engine. To add a new tunable setting later, add
	// one row here and one key to SETTING_KEYS in presets.ts — nothing else.
	interface SettingDef {
		key: SettingKey;
		mode: ApplyMode;
		get: () => unknown;
		set: (v: unknown) => void;
	}
	const SETTINGS: SettingDef[] = [
		{
			key: 'machine',
			mode: 'grid',
			get: () => machine,
			set: (v) => (machine = v as MachineChoice)
		},
		{
			key: 'tapeLength',
			mode: 'grid',
			get: () => tapeLength,
			set: (v) => (tapeLength = v as number)
		},
		{
			key: 'pairLength',
			mode: 'grid',
			get: () => pairLength,
			set: (v) => (pairLength = v as number)
		},
		{
			key: 'lethalProbability',
			mode: 'live',
			get: () => lethalProbability,
			set: (v) => (lethalProbability = v as number)
		},
		{
			key: 'randomRegisters',
			mode: 'live',
			get: () => randomRegisters,
			set: (v) => (randomRegisters = v as boolean)
		},
		{
			key: 'stackConvention',
			mode: 'grid',
			get: () => stackConvention,
			set: (v) => (stackConvention = v as LabSettings['stack'])
		},
		{
			key: 'dialMode',
			mode: 'live',
			get: () => dialMode,
			set: (v) => (dialMode = v as LabSettings['dialMode'])
		},
		{
			key: 'initialCondition',
			mode: 'grid',
			get: () => JSON.stringify(initialCondition),
			set: (v) => {
				initialCondition = JSON.parse(v as string) as InitialCondition;
			}
		},
		{ key: 'gridType', mode: 'grid', get: () => gridType, set: (v) => (gridType = v as GridType) },
		{ key: 'gridWidth', mode: 'grid', get: () => gridWidth, set: (v) => (gridWidth = v as number) },
		{
			key: 'gridHeight',
			mode: 'grid',
			get: () => gridHeight,
			set: (v) => (gridHeight = v as number)
		},
		{ key: 'seed', mode: 'reset', get: () => seed, set: (v) => (seed = v as number) },
		{
			key: 'noiseExp',
			mode: 'live',
			get: () => noiseExp,
			set: (v) => {
				noiseExp = v as number;
				if (engine) engine.noiseExp = noiseExp;
			}
		},
		{
			key: 'pairCount',
			mode: 'live',
			get: () => pairCount,
			set: (v) => {
				pairCount = v as number;
				if (engine) engine.pairCount = pairCount;
			}
		},
		{
			key: 'z80Steps',
			mode: 'live',
			get: () => z80Steps,
			set: (v) => {
				z80Steps = v as number;
				if (engine) engine.z80Steps = z80Steps;
			}
		},
		{
			key: 'suppressPatterns',
			mode: 'live',
			get: () => [...suppressPatterns],
			// suppression (derived) -> $effect pushes the masks to the engine
			set: (v) => (suppressPatterns = [...((v as string[]) ?? [])])
		},
		{
			key: 'colormapName',
			mode: 'live',
			get: () => colormapName,
			set: (v) => handleColormapChange(v as ColormapName)
		},
		{
			key: 'brightness',
			mode: 'live',
			get: () => brightness,
			set: (v) => (brightness = v as number)
		},
		{ key: 'contrast', mode: 'live', get: () => contrast, set: (v) => (contrast = v as number) },
		{
			key: 'saturation',
			mode: 'live',
			get: () => saturation,
			set: (v) => (saturation = v as number)
		},
		{
			key: 'showGridLines',
			mode: 'live',
			get: () => showGridLines,
			set: (v) => (showGridLines = v as boolean)
		},
		{
			key: 'simpleView',
			mode: 'live',
			get: () => simpleView,
			set: (v) => (simpleView = v as boolean)
		}
	];

	let presets = $state<Preset[]>(loadPresets());
	let newPresetName = $state('');
	let showPresetSave = $state(false);
	// When saving would collide with an existing name, hold that preset here to
	// prompt the user to confirm overwrite (which is how saved presets are edited).
	let overwriteTarget = $state<Preset | null>(null);

	function captureSettings(): SettingValues {
		const values: SettingValues = {};
		for (const def of SETTINGS) values[def.key] = def.get();
		return values;
	}

	// Which stored preset (if any) exactly matches the current settings.
	let activePresetId = $derived.by(() => {
		// Re-read current values reactively.
		const current: SettingValues = {};
		for (const def of SETTINGS) current[def.key] = def.get();
		for (const p of presets) {
			let match = true;
			for (const def of SETTINGS) {
				if (!(def.key in p.values)) continue; // partial presets only constrain the keys they carry
				if (!valuesEqual(current[def.key], p.values[def.key])) {
					match = false;
					break;
				}
			}
			if (match) return p.id;
		}
		return null;
	});

	async function applyPreset(preset: Preset) {
		if (machineBusy) return;
		let disruption: ApplyMode | 'none' = 'none';
		const rank: Record<ApplyMode | 'none', number> = { none: 0, live: 0, reset: 1, grid: 2 };
		if (!('machine' in preset.values)) {
			// Original .acell-era presets retain their original processor and reset contract.
			if (
				machine !== 'z80-0.2' ||
				lethalProbability !== 0 ||
				randomRegisters ||
				stackConvention !== 'legacy-sp'
			)
				disruption = 'grid';
			machine = 'z80-0.2';
			lethalProbability = 0;
			randomRegisters = false;
			stackConvention = 'legacy-sp';
			dialMode = 'historical-pcg';
			if (preset.id === 'builtin:classic' || preset.id === 'builtin:hex-organic') {
				const topology = preset.values.gridType === 'hex' ? 'hex' : 'square';
				if (tapeLength !== (topology === 'hex' ? 19 : 16)) disruption = 'grid';
				tapeLength = topology === 'hex' ? 19 : 16;
				pairLength = tapeLength * 2;
				initialCondition = { kind: 'random' };
			}
		}
		for (const def of SETTINGS) {
			if (!(def.key in preset.values)) continue;
			const nextVal = preset.values[def.key];
			if (valuesEqual(def.get(), nextVal)) continue;
			def.set(nextVal);
			if (rank[def.mode] > rank[disruption]) disruption = def.mode;
		}
		selectedPaperId = '';
		if (disruption === 'grid' || disruption === 'reset') await updateMachine(true);
		else if (engine && JSON.stringify(engine.settings) !== JSON.stringify(settings()))
			await updateMachine();
	}

	// Case-insensitive name match so "Classic" and "classic" are the same preset.
	function findPresetByName(name: string): Preset | undefined {
		const key = name.trim().toLowerCase();
		return presets.find((p) => p.name.toLowerCase() === key);
	}

	function requestSavePreset(name: string) {
		const trimmed = name.trim();
		if (!trimmed) return;
		const existing = findPresetByName(trimmed);
		if (existing) {
			// Name taken — confirm overwrite (this is how a saved preset is edited).
			overwriteTarget = existing;
		} else {
			commitSavePreset(trimmed, null);
		}
	}

	function commitSavePreset(name: string, existing: Preset | null) {
		const values = captureSettings();
		if (existing) {
			// Overwrite only replaces the saved settings; the existing name is kept
			// (so re-saving "Classic" as "classic" doesn't silently rename it).
			existing.values = values;
			presets = [...presets];
		} else {
			presets = [...presets, { id: makePresetId(presets), name: name.trim(), values }];
		}
		savePresets(presets);
		newPresetName = '';
		showPresetSave = false;
		overwriteTarget = null;
	}

	function confirmOverwrite() {
		if (overwriteTarget) commitSavePreset(overwriteTarget.name, overwriteTarget);
	}

	function deletePreset(id: string) {
		presets = presets.filter((p) => p.id !== id);
		savePresets(presets);
	}

	// Delete whichever preset currently matches the settings (the highlighted one).
	function deleteActivePreset() {
		if (activePresetId) deletePreset(activePresetId);
	}

	// Sync the derived suppression sets to the GPU engine
	$effect(() => {
		if (!engine || machineBusy) return;
		engine.setAblation(suppressPatterns);
	});

	// Sync simple view mode to GPU engine
	$effect(() => {
		if (!engine) return;
		engine.setShowAverage(simpleView);
	});

	// Sync brightness/contrast/saturation to GPU
	$effect(() => {
		if (!engine) return;
		engine.setBCS(brightness, contrast, saturation);
	});

	// Sync grid lines visibility to GPU
	$effect(() => {
		if (!engine) return;
		engine.setShowGrid(showGridLines);
	});

	// Build display tiles: collapse all suppressed into one card, fill remaining with non-suppressed
	interface DisplayTile {
		type: 'byte' | 'suppressed';
		byte: number; // for 'byte' type
		mnemonic: string;
		count: number; // for 'suppressed': how many suppressed bytes in top rankings
	}
	let displayTiles = $derived.by(() => {
		const tiles: DisplayTile[] = [];
		const hasSuppression = suppressedOpcodes.size > 0;
		// Reserve one slot for the collapsed "suppressed" card when suppression
		// is active, so the ranked non-suppressed opcodes always fill the rest.
		const byteLimit = hasSuppression ? MAX_TRACKED - 1 : MAX_TRACKED;
		let suppressedInTop = 0;

		for (const tb of trackedBytes) {
			if (hasSuppression && suppressedOpcodes.has(tb.byte)) {
				suppressedInTop++;
				continue;
			}
			if (tiles.length < byteLimit) {
				tiles.push({ type: 'byte', byte: tb.byte, mnemonic: tb.mnemonic, count: 0 });
			}
		}

		// Append the collapsed card summarizing suppressed opcodes present in the soup.
		if (hasSuppression && suppressedInTop > 0) {
			tiles.push({
				type: 'suppressed',
				byte: -1,
				mnemonic: `⊘ ${suppressedTotal}`,
				count: suppressedInTop
			});
		}

		return tiles;
	});

	function togglePlay() {
		if (!engine || machineBusy || cultureBusy) return;
		playing = !playing;
		if (playing) runDeadline = performance.now() + 55_000;
	}

	function setSpeed(s: number) {
		speed = s;
		showSpeedMenu = false;
	}

	function formatNumber(n: number): string {
		// Fixed 5-char output: "XXXX" + suffix or padded digits
		// Always produces exactly 5 visible characters to prevent layout shift
		if (n >= 1_000_000_000) {
			const v = (n / 1_000_000_000).toFixed(1);
			return v.padStart(4, '\u2007') + 'B'; // e.g. " 1.2B", "12.3B"
		}
		if (n >= 1_000_000) {
			const v = (n / 1_000_000).toFixed(1);
			return v.padStart(4, '\u2007') + 'M';
		}
		if (n >= 1_000) {
			const v = (n / 1_000).toFixed(1);
			return v.padStart(4, '\u2007') + 'K';
		}
		return n.toFixed(0).padStart(5, '\u2007');
	}

	function hexByte(b: number): string {
		return b.toString(16).toUpperCase().padStart(2, '0');
	}

	function tex(expr: string): string {
		return katex.renderToString(expr, { throwOnError: false, displayMode: false });
	}

	function texBlock(expr: string): string {
		return katex.renderToString(expr, { throwOnError: false, displayMode: true });
	}

	// Z80 opcode descriptions — plain-English decoder for non-experts
	// Register glossary:
	//   A = main working value, HL = address pointer (2 bytes), SP = stack pointer (read/write position)
	//   BC/DE = auxiliary 2-byte registers, AF = working value + status flags
	function opcodeDescription(op: number): string {
		const x = (op >> 6) & 3;
		const y = (op >> 3) & 7;
		const z = op & 7;
		const p = y >> 1;
		const q = y & 1;
		const R_DESC = [
			'counter B',
			'counter C',
			'register D',
			'register E',
			'high byte of address pointer',
			'low byte of address pointer',
			'the byte in memory that the address pointer points to',
			'the working value'
		];
		const RP_DESC = [
			'the BC counter pair',
			'the DE register pair',
			'the address pointer',
			'the read/write position (stack pointer)'
		];
		const RP2_DESC = [
			'the BC counter pair',
			'the DE register pair',
			'the address pointer',
			'the working value + status flags'
		];
		const CC_DESC = [
			'result was not zero',
			'result was zero',
			'no carry occurred',
			'a carry occurred',
			'parity is odd',
			'parity is even',
			'result is positive',
			'result is negative'
		];
		const ALUDESC = [
			'Add {0} to the working value',
			'Add {0} to the working value (with carry)',
			'Subtract {0} from the working value',
			'Subtract {0} from the working value (with borrow)',
			'Bitwise AND the working value with {0}',
			'Bitwise XOR the working value with {0}',
			'Bitwise OR the working value with {0}',
			'Compare the working value with {0} (updates status flags)'
		];
		if (x === 0) {
			if (z === 0) {
				if (y === 0) return 'No operation — does nothing, wastes one cycle';
				if (y === 1) return 'Swap the working value and status flags with a hidden backup copy';
				if (y === 2)
					return 'Subtract 1 from counter B; if B ≠ 0, jump by an offset (creates a tight loop)';
				if (y === 3) return 'Jump forward or backward by a short offset (unconditional)';
				return `Jump by a short offset if ${CC_DESC[y - 4]}`;
			}
			if (z === 1)
				return q === 0
					? `Load a 2-byte number from the code into ${RP_DESC[p]}`
					: `Add ${RP_DESC[p]} to the address pointer (shifts where it points)`;
			if (z === 2) {
				if (q === 0) {
					if (p === 0) return 'Write the working value into memory at the address in BC';
					if (p === 1) return 'Write the working value into memory at the address in DE';
					if (p === 2)
						return 'Write the address pointer (2 bytes) into memory at a specific address';
					return 'Write the working value into memory at a specific address';
				}
				if (p === 0) return 'Read a byte from memory (at the address in BC) into the working value';
				if (p === 1) return 'Read a byte from memory (at the address in DE) into the working value';
				if (p === 2) return 'Read 2 bytes from a specific memory address into the address pointer';
				return 'Read a byte from a specific memory address into the working value';
			}
			if (z === 3) return q === 0 ? `Add 1 to ${RP_DESC[p]}` : `Subtract 1 from ${RP_DESC[p]}`;
			if (z === 4)
				return y === 6
					? 'Add 1 to the byte in memory that the address pointer points to'
					: `Add 1 to ${R_DESC[y]}`;
			if (z === 5)
				return y === 6
					? 'Subtract 1 from the byte in memory that the address pointer points to'
					: `Subtract 1 from ${R_DESC[y]}`;
			if (z === 6)
				return y === 6
					? 'Write a byte from the code into the memory location the address pointer points to'
					: `Load a byte from the code into ${R_DESC[y]}`;
			// z === 7
			const rot = [
				'Rotate the working value left one bit (fast bit shift)',
				'Rotate the working value right one bit (fast bit shift)',
				'Rotate the working value left through the carry flag',
				'Rotate the working value right through the carry flag',
				'Adjust the working value for decimal (BCD) arithmetic',
				'Flip every bit of the working value (0↔1)',
				'Set the carry flag to 1',
				'Toggle the carry flag (0↔1)'
			];
			return rot[y];
		}
		if (x === 1) {
			if (y === 6 && z === 6) return 'Halt — stop executing until reset (cell goes dormant)';
			if (z === 6) return `Read a byte from memory (at the address pointer) into ${R_DESC[y]}`;
			if (y === 6) return `Write ${R_DESC[z]} into memory at the address pointer`;
			if (y === z) return `Copy ${R_DESC[y]} to itself (does nothing)`;
			return `Copy ${R_DESC[z]} into ${R_DESC[y]}`;
		}
		if (x === 2) {
			const target = z === 6 ? 'the byte at the address pointer' : R_DESC[z];
			return ALUDESC[y].replace('{0}', target);
		}
		// x === 3
		if (z === 0) return `Return from subroutine if ${CC_DESC[y]}`;
		if (z === 1) {
			if (q === 0) return `Read 2 bytes from memory (at the stack pointer) into ${RP2_DESC[p]}`;
			if (p === 0) return 'Return from subroutine (jump back to where we came from)';
			if (p === 1) return 'Swap BC, DE, and the address pointer with hidden backup copies';
			if (p === 2) return 'Jump to the memory address stored in the address pointer';
			return 'Set the stack pointer (read/write position) to the address pointer value';
		}
		if (z === 2) return `Jump to a specific address if ${CC_DESC[y]}`;
		if (z === 3) {
			if (y === 0) return 'Jump to a specific address (unconditional)';
			if (y === 1) return 'Prefix for extended bit/rotate instructions';
			if (y === 2)
				return 'Output to an I/O port — no device is attached, so this has no effect (a 2-byte no-op that skips the port number)';
			if (y === 3)
				return 'Input from an I/O port — no device is attached, so it always reads 0 into the working value (and skips the port-number byte)';
			if (y === 4)
				return "Swap the address pointer with the 2 bytes at the stack pointer. The stack starts at the end of the address space (inside the neighbor cell), so this is a key self-replication instruction — it moves data into the neighbor cell's memory";
			if (y === 5) return 'Swap the address pointer with the DE register pair';
			if (y === 6) return 'Disable interrupts (no interrupts exist here, so this has no effect)';
			return 'Enable interrupts (no interrupts exist here, so this has no effect)';
		}
		if (z === 4) return `Call a subroutine at a specific address if ${CC_DESC[y]}`;
		if (z === 5) {
			if (q === 0)
				return `Push ${RP2_DESC[p]} (2 bytes) onto the stack. The stack starts at the end of the address space (inside the neighbor cell), so pushing writes into the neighbor's memory — a building block of self-replication`;
			if (p === 0)
				return 'Call a subroutine at a specific address (pushes the return point onto the stack)';
			if (p === 1)
				return 'Index-register (IX) prefix — not implemented here, so it is skipped and the next instruction runs on the address pointer (HL)';
			if (p === 2) return 'Prefix for extended instructions (block copy, 16-bit memory moves)';
			return 'Index-register (IY) prefix — not implemented here, so it is skipped and the next instruction runs on the address pointer (HL)';
		}
		if (z === 6) {
			return ALUDESC[y].replace('{0}', 'a byte from the code');
		}
		// z === 7
		return `Save the return position on the stack and jump to the fixed low address ${(y * 8).toString(16).toUpperCase().padStart(2, '0')}h within this buffer (a compact call)`;
	}

	// Freq tile tooltip state
	let tileTooltip = $state<{
		byte: number;
		mnemonic: string;
		rank: number;
		x: number;
		y: number;
	} | null>(null);

	let tileTooltipSuppressUntil = 0;

	function showTileTooltip(e: MouseEvent, byte: number, mnemonic: string, rank: number) {
		if (Date.now() < tileTooltipSuppressUntil) return;
		const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
		tileTooltip = {
			byte,
			mnemonic,
			rank,
			x: rect.right + 6,
			y: rect.top - 6
		};
	}

	function hideTileTooltip() {
		tileTooltip = null;
	}

	function toggleChart() {
		showInfoChart = !showInfoChart;
		tileTooltip = null;
		tileTooltipSuppressUntil = Date.now() + 300;
	}

	function byteColor(b: number): string {
		const [r, g, bl] = unpackRGBA(colormap[b]);
		return `rgb(${r},${g},${bl})`;
	}

	function byteLuminance(b: number): number {
		const [r, g, bl] = unpackRGBA(colormap[b]);
		return (0.299 * r + 0.587 * g + 0.114 * bl) / 255;
	}

	/** Brightened color for chart lines — ensures visibility on dark backgrounds */
	function byteChartColor(b: number): string {
		const [r, g, bl] = unpackRGBA(colormap[b]);
		const lum = (0.299 * r + 0.587 * g + 0.114 * bl) / 255;
		if (lum < 0.2) {
			// Brighten dark colors: lift toward a lighter version
			const boost = 0.35;
			return `rgb(${Math.round(r + (255 - r) * boost)},${Math.round(g + (255 - g) * boost)},${Math.round(bl + (255 - bl) * boost)})`;
		}
		return `rgb(${r},${g},${bl})`;
	}

	/** Text color for cells — white on dark, black on light */
	function cellTextColor(b: number): string {
		const lum = byteLuminance(b);
		return lum > 0.4 ? 'rgba(0,0,0,0.85)' : 'rgba(255,255,255,0.9)';
	}

	interface GenomeCell {
		byteVal: number;
		label: string;
		isOpcode: boolean;
	}

	function buildGenomeGrid(data: Uint8Array, disasm: ReturnType<typeof disassemble>): GenomeCell[] {
		const cells: GenomeCell[] = [];
		// eslint-disable-next-line svelte/prefer-svelte-reactivity -- local variable, not reactive state
		const byteInfo = new Map<number, { label: string; isOpcode: boolean }>();

		for (const line of disasm) {
			byteInfo.set(line.offset, { label: line.mnemonic, isOpcode: true });
			for (let i = 1; i < line.length && line.offset + i < data.length; i++) {
				byteInfo.set(line.offset + i, {
					label: hexByte(data[line.offset + i]),
					isOpcode: false
				});
			}
		}

		for (let i = 0; i < data.length; i++) {
			const info = byteInfo.get(i);
			cells.push({
				byteVal: data[i],
				label: info ? info.label : hexByte(data[i]),
				isOpcode: info ? info.isOpcode : false
			});
		}
		return cells;
	}
</script>

<svelte:window
	onpointerdown={() => {
		keyboardModality = false;
	}}
	onkeydown={(e) => {
		keyboardModality = true;
		if (e.code === 'Escape') {
			showHelp = false;
			showSettings = false;
			speciesModal = null;
			inspectedTape = null;
			overwriteTarget = null;
		}
		if (e.target !== document.body) return;
		// Ctrl/Cmd shortcuts
		if (e.metaKey || e.ctrlKey) {
			if (e.code === 'KeyS') {
				e.preventDefault();
				handleSave();
				return;
			}
			if (e.code === 'KeyO') {
				e.preventDefault();
				fileInput?.click();
				return;
			}
		}
		switch (e.code) {
			case 'Space':
				e.preventDefault();
				togglePlay();
				break;
			case 'KeyR':
				handleReset();
				break;
			case 'KeyH':
				showHelp = !showHelp;
				break;
			case 'KeyS':
				toggleChart();
				break;
			case 'KeyF':
				engine?.resetView(canvasW, canvasH);
				break;
			case 'Escape':
				if (overwriteTarget) {
					overwriteTarget = null;
					break;
				}
				showHelp = false;
				showSpeedMenu = false;
				showSettings = false;
				speciesModal = null;
				break;
		}
	}}
/>

<svelte:head>
	<title>Algocell · Digital primordial soup</title>
</svelte:head>

<svelte:document
	onvisibilitychange={() => {
		if (document.hidden) playing = false;
	}}
/>
<div class="legacy-scope" class:pointer-focus={!keyboardModality}>
	<!-- Hidden file input for checkpoint load -->
	<input
		bind:this={fileInput}
		type="file"
		accept=".acell,.png"
		style="display:none"
		onchange={handleFileInputChange}
	/>

	<!-- Full-screen canvas -->
	<div
		bind:this={canvasContainer}
		role="region"
		aria-label="Soup viewport"
		class="fixed inset-0"
		style="-webkit-user-select:none;user-select:none;-webkit-touch-callout:none"
		ondragover={(e) => e.preventDefault()}
		ondrop={handleDrop}
	>
		{#if gpuError}
			<div
				class="absolute inset-0 z-50 flex items-center justify-center"
				style="background:var(--bg-panel)"
			>
				<p class="max-w-md px-8 text-center text-red-400">{gpuError}</p>
			</div>
		{/if}
		<canvas
			bind:this={canvas}
			class="block h-full w-full"
			style="cursor:{isPanning ? 'grabbing' : 'crosshair'};touch-action:none"
			onmousemove={handleCanvasMouseMove}
			onmouseleave={handleCanvasMouseLeave}
			onmousedown={handleCanvasMouseDown}
			onmouseup={handleCanvasMouseUp}
			onwheel={handleCanvasWheel}
			ondblclick={handleCanvasDblClick}
			ontouchstart={handleTouchStart}
			ontouchmove={handleTouchMove}
			ontouchend={handleTouchEnd}
			oncontextmenu={(e) => e.preventDefault()}
		></canvas>
	</div>

	<!-- Toolbar -->
	<div class="toolbar" class:collapsed={toolbarCollapsed}>
		<div class="toolbar-buttons" class:hidden={toolbarCollapsed}>
			<!-- Play/Pause -->
			<button
				class="tb"
				class:active={!playing}
				style="color:var(--accent)"
				title="{playing ? 'Pause' : 'Play'} (Space)"
				onclick={togglePlay}
			>
				{#if playing}
					<svg width="16" height="16" viewBox="0 0 14 14" fill="currentColor">
						<rect x="2" y="1" width="3.5" height="12" rx="1" />
						<rect x="8.5" y="1" width="3.5" height="12" rx="1" />
					</svg>
				{:else}
					<svg width="16" height="16" viewBox="0 0 14 14" fill="currentColor">
						<path d="M3 1.5v11l9-5.5z" />
					</svg>
				{/if}
			</button>

			<!-- Speed -->
			<div class="relative">
				<button
					class="tb mono"
					style="color:var(--text-secondary);min-width:32px"
					title="Speed"
					onclick={() => (showSpeedMenu = !showSpeedMenu)}
				>
					{speed}x
				</button>
				{#if showSpeedMenu}
					<div class="speed-menu">
						{#each [1, 2, 4, 8] as s (s)}
							<button class="speed-opt" class:active={speed === s} onclick={() => setSpeed(s)}
								>{s}x</button
							>
						{/each}
					</div>
				{/if}
			</div>

			<div class="tb-sep"></div>

			<!-- Reset -->
			<button class="tb" style="color:var(--accent-amber)" title="Reset (R)" onclick={handleReset}>
				<svg
					width="14"
					height="14"
					viewBox="0 0 14 14"
					fill="none"
					stroke="currentColor"
					stroke-width="1.5"
				>
					<path d="M2 7a5 5 0 1 1 1 3" stroke-linecap="round" />
					<path d="M2 3v4h4" stroke-linecap="round" stroke-linejoin="round" />
				</svg>
			</button>

			<div class="tb-sep"></div>

			<!-- Save checkpoint -->
			<button
				class="tb"
				style="color:var(--text-subtle)"
				title="Save checkpoint (Cmd+S)"
				onclick={handleSave}
			>
				<svg
					width="14"
					height="14"
					viewBox="0 0 14 14"
					fill="none"
					stroke="currentColor"
					stroke-width="1.4"
				>
					<rect x="2" y="1" width="10" height="12" rx="1" />
					<path d="M4 1v4h6V1" />
					<path d="M4 8h6M4 10h4" stroke-linecap="round" />
				</svg>
			</button>

			<!-- Load checkpoint -->
			<button
				class="tb"
				style="color:var(--text-subtle)"
				title="Load checkpoint (Cmd+O)"
				onclick={() => fileInput?.click()}
			>
				<svg
					width="14"
					height="14"
					viewBox="0 0 14 14"
					fill="none"
					stroke="currentColor"
					stroke-width="1.4"
				>
					<path
						d="M2 6V3a1 1 0 0 1 1-1h3l1.5 1.5H11a1 1 0 0 1 1 1V11a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V6z"
					/>
					<path d="M7 6v4M5 8l2-2 2 2" stroke-linecap="round" stroke-linejoin="round" />
				</svg>
			</button>

			<div class="tb-sep"></div>

			<!-- Settings -->
			<button
				class="tb"
				class:active={showSettings}
				style="color:{showSettings ? 'var(--accent)' : 'var(--text-subtle)'}"
				title="Settings"
				onclick={() => (showSettings = !showSettings)}
			>
				<svg
					width="16"
					height="16"
					viewBox="0 0 14 14"
					fill="none"
					stroke="currentColor"
					stroke-width="1.4"
				>
					<path d="M2 3.5h10M2 7h10M2 10.5h10" stroke-linecap="round" />
					<circle cx="5" cy="3.5" r="1.3" fill="currentColor" stroke="none" />
					<circle cx="9" cy="7" r="1.3" fill="currentColor" stroke="none" />
					<circle cx="6" cy="10.5" r="1.3" fill="currentColor" stroke="none" />
				</svg>
			</button>

			<!-- Help -->
			<button
				class="tb"
				style="color:var(--text-subtle)"
				title="Help (H)"
				onclick={() => (showHelp = !showHelp)}
			>
				<svg
					width="14"
					height="14"
					viewBox="0 0 14 14"
					fill="none"
					stroke="currentColor"
					stroke-width="1.5"
				>
					<circle cx="7" cy="7" r="5.5" />
					<path d="M5.5 5.5a1.75 1.75 0 0 1 3.25 1c0 1-1.5 1.25-1.5 2.25" stroke-linecap="round" />
					<circle cx="7" cy="10.5" r="0.5" fill="currentColor" stroke="none" />
				</svg>
			</button>
		</div>

		<!-- Collapse/expand toggle: shows X when expanded, + when collapsed -->
		<button
			class="tb collapse-btn"
			class:collapsed={toolbarCollapsed}
			title={toolbarCollapsed ? 'Expand toolbar' : 'Collapse toolbar'}
			onclick={() => {
				toolbarCollapsed = !toolbarCollapsed;
				if (toolbarCollapsed) {
					showSpeedMenu = false;
					showSettings = false;
				}
			}}
		>
			<svg width="12" height="12" viewBox="0 0 12 12" stroke="currentColor" stroke-width="1.5">
				<path d="M3 3l6 6M9 3l-6 6" stroke-linecap="round" />
			</svg>
		</button>
	</div>

	<!-- Bottom-left info bar -->
	<div class="info-bar">
		<div class="info-row">
			<span
				class="info-item"
				title="Instruction-slot budget per second, including unused slots in halted encounters"
			>
				<span class="info-label">ops/s</span>
				<span class="info-value">{formatNumber(opsPerSec)}</span>
			</span>
			<span class="info-sep"></span>
			<span class="info-item">
				<span class="info-label">mut</span>
				<span class="info-value"
					>{#if noiseExp >= 31}off{:else}1/2<sup>{noiseExp}</sup>{/if}</span
				>
			</span>
			<button class="info-chart-toggle" onclick={toggleChart} title="Toggle chart">
				<svg
					width="10"
					height="10"
					viewBox="0 0 10 10"
					fill="currentColor"
					style="opacity:{showInfoChart ? 1 : 0.4}"
				>
					<polyline
						points="0,8 3,4 6,6 10,1"
						fill="none"
						stroke="currentColor"
						stroke-width="1.5"
					/>
				</svg>
			</button>
		</div>
		{#if showInfoChart}
			<div class="info-tabs">
				<button
					class="info-tab"
					class:active={infoTab === 'opcodes'}
					onclick={() => (infoTab = 'opcodes')}
					title="Opcode frequency — most common byte values for the selected processor over time"
					>Opcodes</button
				>
				<button
					class="info-tab"
					class:active={infoTab === 'species'}
					onclick={() => (infoTab = 'species')}
					title="Species — distinct cell genomes (by content hash) and their populations"
					>Species</button
				>
				<button
					class="info-tab"
					class:active={infoTab === 'diversity'}
					onclick={() => (infoTab = 'diversity')}
					title="Diversity — richness and Shannon/Simpson indices of the species distribution"
					>Diversity</button
				>
			</div>
		{/if}
		{#if showInfoChart && infoTab === 'opcodes' && chartVersion > 0 && chartLen > 1}
			<div class="info-chart">
				<svg
					viewBox="0 0 200 120"
					class="info-chart-svg"
					role="img"
					aria-label="Historical measurement trend"
					preserveAspectRatio="none"
					onmouseleave={() => {
						chartHoveredByte = -1;
					}}
				>
					{#each freqLines as fl (fl.byte)}
						{#if fl.path}
							<polyline
								points={fl.path.replace(/[ML]/g, (m) => (m === 'M' ? '' : ' ')).trim()}
								fill="none"
								stroke={fl.color}
								stroke-width={chartHoveredByte === fl.byte ? '4' : '2.5'}
								opacity={chartHoveredByte >= 0
									? chartHoveredByte === fl.byte
										? '1'
										: '0.15'
									: '0.8'}
							/>
						{/if}
					{/each}
				</svg>
				<div class="freq-grid">
					{#each displayTiles as tile, i (tile.type === 'suppressed' ? 'suppressed' : tile.byte)}
						{#if tile.type === 'suppressed'}
							<div class="freq-cell freq-cell-suppressed-group" role="button" tabindex="0">
								<span class="freq-label">{tile.mnemonic}</span>
							</div>
						{:else}
							<div
								class="freq-cell"
								class:dimmed={chartHoveredByte >= 0 && chartHoveredByte !== tile.byte}
								class:highlighted={chartHoveredByte === tile.byte}
								style="background:{byteColor(tile.byte)};color:{cellTextColor(tile.byte)}"
								role="button"
								tabindex="0"
								onclick={() => toggleSuppressOpcode(tile.byte)}
								onkeydown={(e) => {
									if (e.key === 'Enter' || e.key === ' ') {
										e.preventDefault();
										toggleSuppressOpcode(tile.byte);
									}
								}}
								onmouseenter={(e) => {
									chartHoveredByte = tile.byte;
									showTileTooltip(e, tile.byte, tile.mnemonic, i + 1);
								}}
								onmouseleave={() => {
									chartHoveredByte = -1;
									hideTileTooltip();
								}}
							>
								<span class="freq-rank">{i + 1}</span>
								<span class="freq-label">{tile.mnemonic || hexByte(tile.byte)}</span>
							</div>
						{/if}
					{/each}
				</div>
				{#if suppressPatterns.length > 0}
					<div class="freq-suppressed-row">
						{#each suppressPatterns as pat (pat)}
							<button
								class="freq-suppress-chip"
								onclick={() => removeSuppressPattern(pat)}
								title="Remove pattern: {pat}"
							>
								{pat}
								<svg
									width="7"
									height="7"
									viewBox="0 0 8 8"
									stroke="currentColor"
									stroke-width="1.5"
								>
									<path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke-linecap="round" />
								</svg>
							</button>
						{/each}
						<span class="freq-suppress-count" title={describeSuppression(suppression)}
							>⊘ {suppressedTotal}</span
						>
					</div>
				{/if}
			</div>
		{:else if showInfoChart && infoTab === 'species'}
			<div class="info-chart">
				{#if speciesLen > 1 && speciesLines.length > 0}
					<svg
						viewBox="0 0 200 120"
						class="info-chart-svg"
						role="img"
						aria-label="Historical measurement trend"
						preserveAspectRatio="none"
						onmouseleave={() => {
							speciesHoveredIdx = -1;
						}}
					>
						{#each speciesLines as sl, idx (sl.hash)}
							{#if sl.path}
								<!-- Thick invisible hit area for hover -->
								<polyline
									points={sl.path.replace(/[ML]/g, (m) => (m === 'M' ? '' : ' ')).trim()}
									fill="none"
									stroke="transparent"
									role="presentation"
									stroke-width="12"
									onmouseenter={() => {
										speciesHoveredIdx = idx;
									}}
								/>
								<polyline
									points={sl.path.replace(/[ML]/g, (m) => (m === 'M' ? '' : ' ')).trim()}
									fill="none"
									stroke={sl.color}
									stroke-width={speciesHoveredIdx === idx ? '4' : '2.5'}
									opacity={speciesHoveredIdx >= 0
										? speciesHoveredIdx === idx
											? '1'
											: '0.15'
										: '0.8'}
									style="pointer-events:none"
								/>
							{/if}
						{/each}
					</svg>
				{:else}
					<div class="species-waiting">Collecting species data...</div>
				{/if}
				{#if currentSpecies}
					<div class="freq-grid">
						{#each currentSpecies.species.slice(0, MAX_SPECIES_DISPLAY) as sp, i (sp.hash)}
							{@const chartIdx = speciesHashOrder.indexOf(sp.hash)}
							<div
								class="freq-cell"
								class:dimmed={speciesHoveredIdx >= 0 && speciesHoveredIdx !== chartIdx}
								class:highlighted={speciesHoveredIdx === chartIdx}
								style="background:{sp.color};color:{sp.textColor}"
								role="button"
								tabindex="0"
								title="Click to inspect"
								onmouseenter={() => {
									speciesHoveredIdx = chartIdx;
								}}
								onmouseleave={() => {
									speciesHoveredIdx = -1;
								}}
								onclick={() => {
									speciesModal = sp;
								}}
								onkeydown={(e) => {
									if (e.key === 'Enter' || e.key === ' ') {
										e.preventDefault();
										speciesModal = sp;
									}
								}}
							>
								<span class="freq-rank">{i + 1}</span>
								<span class="freq-label">{(sp.fraction * 100).toFixed(1)}%</span>
							</div>
						{/each}
					</div>
				{/if}
				{#if suppressPatterns.length > 0}
					<div class="freq-suppressed-row">
						{#each suppressPatterns as pat (pat)}
							<button
								class="freq-suppress-chip"
								onclick={() => removeSuppressPattern(pat)}
								title="Remove pattern: {pat}"
							>
								{pat}
								<svg
									width="7"
									height="7"
									viewBox="0 0 8 8"
									stroke="currentColor"
									stroke-width="1.5"
								>
									<path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke-linecap="round" />
								</svg>
							</button>
						{/each}
						<span class="freq-suppress-count" title={describeSuppression(suppression)}
							>⊘ {suppressedTotal}</span
						>
					</div>
				{/if}
			</div>
		{:else if showInfoChart && infoTab === 'diversity'}
			<div class="info-chart">
				{#if diversityLen > 1 && diversityLines.length > 0}
					<svg
						viewBox="0 0 200 120"
						class="info-chart-svg"
						role="img"
						aria-label="Historical measurement trend"
						preserveAspectRatio="none"
						onmouseleave={() => {
							diversityHoveredIdx = -1;
						}}
					>
						{#each diversityLines as dl (dl.idx)}
							{#if dl.path}
								<polyline
									points={dl.path.replace(/[ML]/g, (m) => (m === 'M' ? '' : ' ')).trim()}
									fill="none"
									stroke="transparent"
									role="presentation"
									stroke-width="12"
									onmouseenter={() => {
										diversityHoveredIdx = dl.idx;
									}}
								/>
								<polyline
									points={dl.path.replace(/[ML]/g, (m) => (m === 'M' ? '' : ' ')).trim()}
									fill="none"
									stroke={dl.color}
									stroke-width={diversityHoveredIdx === dl.idx ? '4' : '2'}
									opacity={diversityHoveredIdx >= 0
										? diversityHoveredIdx === dl.idx
											? '1'
											: '0.15'
										: '0.8'}
									style="pointer-events:none"
								/>
							{/if}
						{/each}
					</svg>
				{:else}
					<div class="species-waiting">Collecting diversity data...</div>
				{/if}
				<div class="diversity-grid">
					{#each diversityLines as dl (dl.idx)}
						<div
							class="diversity-card"
							role="group"
							class:dimmed={diversityHoveredIdx >= 0 && diversityHoveredIdx !== dl.idx}
							class:highlighted={diversityHoveredIdx === dl.idx}
							onmouseenter={() => {
								diversityHoveredIdx = dl.idx;
							}}
							onmouseleave={() => {
								diversityHoveredIdx = -1;
							}}
						>
							<span class="diversity-swatch" style="background:{dl.color}"></span>
							<span class="diversity-label">{dl.label}</span>
							<span class="diversity-value"
								>{dl.current < 100
									? dl.current.toFixed(dl.current < 10 ? 2 : 1)
									: formatNumber(dl.current)}</span
							>
						</div>
					{/each}
				</div>
				{#if suppressPatterns.length > 0}
					<div class="freq-suppressed-row">
						{#each suppressPatterns as pat (pat)}
							<button
								class="freq-suppress-chip"
								onclick={() => removeSuppressPattern(pat)}
								title="Remove pattern: {pat}"
							>
								{pat}
								<svg
									width="7"
									height="7"
									viewBox="0 0 8 8"
									stroke="currentColor"
									stroke-width="1.5"
								>
									<path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke-linecap="round" />
								</svg>
							</button>
						{/each}
						<span class="freq-suppress-count" title={describeSuppression(suppression)}
							>⊘ {suppressedTotal}</span
						>
					</div>
				{/if}
			</div>
		{/if}
	</div>

	{#snippet disassemblyPanel(tape: Uint8Array)}
		<details class="inspector-disassembly">
			<summary>Disassembly</summary>
			{#each disassemble(tape) as line (line.offset)}<div class="disassembly-line">
					<span>{line.offset.toString(16).toUpperCase().padStart(2, '0')}</span><span
						>{line.bytes.map(hexByte).join(' ')}</span
					><span>{line.mnemonic}</span>
				</div>{/each}
		</details>
	{/snippet}

	{#snippet culturePanel(tape: Uint8Array)}
		<div class="culture-panel">
			<button
				class="preset-chip culture-button"
				disabled={cultureBusy || machineBusy || !engine}
				onclick={() => {
					void runCulture(tape);
				}}>Run culture test</button
			>
			<span class="instrument-note">32 fresh partners · 2 generations · mutation off</span>
			{#if cultureBusy}<p class="instrument-note" role="status">Testing the selected tape…</p>{/if}
			{#if cultureError}<p class="instrument-error" role="alert">{cultureError}</p>{/if}
			{#if cultureResult}
				<p class="culture-verdict">
					{cultureResult.positive === null
						? 'Culture screen unknown'
						: cultureResult.positive
							? 'Positive culture screen'
							: 'Below the culture screen'} · {cultureResult.confinement}
				</p>
				<table class="culture-table">
					<thead
						><tr><th>Generation</th><th>Gain</th><th>Exact</th><th>Cyclic</th><th>Entries</th></tr
						></thead
					><tbody>
						{#each cultureResult.generations as generation (generation.generation)}<tr
								><td>{generation.generation}</td><td
									>{generation.gain === null ? 'unknown' : generation.gain.toFixed(3)}</td
								><td>{generation.exactCopies}/{cultureResult.trials}</td><td
									>{generation.cyclicCopies}/{cultureResult.trials}</td
								><td>{generation.partnerEntries}/{cultureResult.trials}</td></tr
							>{/each}
					</tbody>
				</table>
				<p class="instrument-note">
					Continuing production is measured separately from retention of an introduced variant. Seed {cultureResult.seed};
					{cultureResult.settings.steps} instruction slots.
				</p>
				<details class="culture-example">
					<summary>Example encounter · fetch map &amp; offspring</summary>
					<p class="instrument-note">
						Fetched addresses (including operands). Underline marks an instruction start.
					</p>
					<div class="culture-bytes">
						{#each cultureResult.example.before as byte, index (index)}<span
								class="culture-byte"
								class:fetched={cultureResult.example.fetched.includes(index)}
								class:instruction-start={cultureResult.example.starts.includes(index)}
								title="Address {index}"
								style="--byte-color:{byteColor(byte)};color:{cultureResult.example.fetched.includes(
									index
								)
									? cellTextColor(byte)
									: '#888'}">{hexByte(byte)}</span
							>{/each}
					</div>
					<p class="instrument-note">Partner after the encounter</p>
					<div class="culture-bytes">
						{#each cultureResult.example.after.slice(tape.length, tape.length * 2) as byte, index (index)}<span
								class="culture-byte fetched"
								style="--byte-color:{byteColor(byte)};color:{cellTextColor(byte)}"
								title="Partner byte {index}">{hexByte(byte)}</span
							>{/each}
					</div>
					<p class="instrument-note">
						{cultureResult.example.steps} executed steps; halt {cultureResult.example.halt}; trap {cultureResult
							.example.trap}.
					</p>
				</details>
			{/if}
		</div>
	{/snippet}

	{#if inspectedTape}
		{@const tape = inspectedTape}
		{@const cells = buildGenomeGrid(tape, disassemble(tape))}
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<div
			class="species-modal-backdrop"
			onclick={() => {
				inspectedTape = null;
			}}
			role="presentation"
		>
			<div
				class="species-modal tape-inspector"
				onclick={(event) => event.stopPropagation()}
				role="dialog"
				aria-modal="true"
				aria-label="Tape inspector"
				tabindex="-1"
			>
				<div class="species-modal-header">
					<span>Cell {inspectedCell} · {isa.toUpperCase()} · {tape.length} bytes</span><button
						class="species-modal-close"
						aria-label="Close tape inspector"
						onclick={() => {
							inspectedTape = null;
						}}>×</button
					>
				</div>
				<div class="species-modal-body">
					<div class="tip-grid species-modal-grid">
						{#each cells as cell, index (index)}<div
								class="tip-cell"
								class:operand={!cell.isOpcode}
								style="background:{byteColor(cell.byteVal)};color:{cellTextColor(cell.byteVal)}"
								title="Byte {index} · {hexByte(cell.byteVal)} · {cell.label}"
							>
								{cell.label}
							</div>{/each}
					</div>
					{@render disassemblyPanel(tape)}
					{@render culturePanel(tape)}
				</div>
			</div>
		</div>
	{/if}

	<!-- Species inspect modal -->
	{#if speciesModal}
		{@const sp = speciesModal}
		{@const cells = sp.exemplarCells}
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<div
			class="species-modal-backdrop"
			onclick={() => {
				speciesModal = null;
				speciesModalTip = null;
			}}
			role="presentation"
		>
			<div
				class="species-modal"
				onclick={(e) => e.stopPropagation()}
				role="dialog"
				aria-modal="true"
				aria-label="Algocell details"
				tabindex="-1"
			>
				<div class="species-modal-header" style="background:{sp.color};color:{sp.textColor}">
					<span>{(sp.fraction * 100).toFixed(1)}% &mdash; {sp.count} cells</span>
					<button
						class="species-modal-close"
						aria-label="Close tape inspector"
						onclick={() => (speciesModal = null)}
					>
						<svg
							width="10"
							height="10"
							viewBox="0 0 10 10"
							stroke="currentColor"
							stroke-width="1.5"
						>
							<path d="M2 2l6 6M8 2l-6 6" stroke-linecap="round" />
						</svg>
					</button>
				</div>
				{#if cells.length > 0}
					<div class="species-modal-body">
						{#if gridType === 'hex' && tapeLength === 19}
							<div
								class="hex-byte-grid species-modal-hex"
								role="group"
								onmouseleave={() => {
									speciesModalTip = null;
								}}
							>
								{#each cells as cell, i (i)}
									<div
										class="hex-byte-cell hex-byte-pos-{i}"
										class:operand={!cell.isOpcode}
										class:modal-cell-hovered={speciesModalTip?.idx === i}
										role="group"
										onmouseenter={(e) => {
											const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
											speciesModalTip = {
												idx: i,
												byteVal: cell.byteVal,
												x: rect.left + rect.width / 2,
												y: rect.top
											};
										}}
									>
										<div
											class="hex-byte-inner"
											style="background:{byteColor(cell.byteVal)};color:{byteLuminance(
												cell.byteVal
											) > 0.4
												? 'rgba(0,0,0,0.85)'
												: 'rgba(255,255,255,0.9)'}"
										>
											{cell.label}
										</div>
									</div>
								{/each}
							</div>
						{:else}
							<div
								class="tip-grid species-modal-grid"
								role="group"
								onmouseleave={() => {
									speciesModalTip = null;
								}}
							>
								{#each cells as cell, i (i)}
									<div
										class="tip-cell"
										class:operand={!cell.isOpcode}
										class:modal-cell-hovered={speciesModalTip?.idx === i}
										role="group"
										style="background:{byteColor(cell.byteVal)};color:{byteLuminance(cell.byteVal) >
										0.4
											? 'rgba(0,0,0,0.85)'
											: 'rgba(255,255,255,0.9)'}"
										onmouseenter={(e) => {
											const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
											speciesModalTip = {
												idx: i,
												byteVal: cell.byteVal,
												x: rect.left + rect.width / 2,
												y: rect.top
											};
										}}
									>
										{cell.label}
									</div>
								{/each}
							</div>
						{/if}
						{#if sp.exemplar}{@render disassemblyPanel(sp.exemplar)}{@render culturePanel(
								sp.exemplar
							)}{/if}
						<div class="species-modal-hash">
							{(sp.hash >>> 0).toString(16).padStart(8, '0')}
						</div>
					</div>
				{/if}
			</div>
		</div>
	{/if}

	<!-- Species byte tooltip (floating above modal) -->
	{#if speciesModalTip}
		{@const sb = speciesModalTip.byteVal}
		{@const sdesc = opcodeDescription(sb)}
		{@const sbg = byteColor(sb)}
		{@const slum = byteLuminance(sb)}
		<div
			class="tile-tip"
			style="left:{speciesModalTip.x}px;top:{speciesModalTip.y}px;transform:translate(-50%,-100%) translateY(-8px);z-index:70"
		>
			<div
				class="tile-tip-header"
				style="background:{sbg};color:{slum > 0.4 ? 'rgba(0,0,0,0.85)' : 'rgba(255,255,255,0.95)'}"
			>
				<span class="tile-tip-mnemonic">{byteToMnemonic(sb) || hexByte(sb)}</span>
				<span class="tile-tip-hex">0x{hexByte(sb)}</span>
			</div>
			<div class="tile-tip-body">
				<p class="tile-tip-desc">{sdesc}</p>
				{#if suppressedOpcodes.has(sb)}
					<span class="tile-tip-status suppressed">⊘ Suppressed</span>
				{/if}
			</div>
		</div>
	{/if}

	<!-- Tile tooltip -->
	{#if tileTooltip}
		{@const tt = tileTooltip}
		{@const desc = opcodeDescription(tt.byte)}
		{@const bg = byteColor(tt.byte)}
		{@const lum = byteLuminance(tt.byte)}
		<div class="tile-tip" style="left:{tt.x}px;top:{tt.y}px">
			<div
				class="tile-tip-header"
				style="background:{bg};color:{lum > 0.4 ? 'rgba(0,0,0,0.85)' : 'rgba(255,255,255,0.95)'}"
			>
				<span class="tile-tip-mnemonic">{tt.mnemonic}</span>
				<span class="tile-tip-hex">0x{hexByte(tt.byte)}</span>
			</div>
			<div class="tile-tip-body">
				<p class="tile-tip-desc">{desc}</p>
				{#if suppressedOpcodes.has(tt.byte)}
					<span class="tile-tip-status suppressed">⊘ Suppressed</span>
				{/if}
				<span class="tile-tip-hint"
					>Click to {suppressedOpcodes.has(tt.byte) ? 'enable' : 'suppress'}</span
				>
			</div>
		</div>
	{/if}

	<!-- Settings panel -->
	{#if showSettings}
		<div class="panel settings-panel">
			<div class="panel-header">
				<span class="panel-title"
					><svg
						width="12"
						height="12"
						viewBox="0 0 14 14"
						fill="none"
						stroke="var(--accent)"
						stroke-width="1.4"
						><path d="M2 3.5h10M2 7h10M2 10.5h10" stroke-linecap="round" /><circle
							cx="5"
							cy="3.5"
							r="1.3"
							fill="var(--accent)"
							stroke="none"
						/><circle cx="9" cy="7" r="1.3" fill="var(--accent)" stroke="none" /><circle
							cx="6"
							cy="10.5"
							r="1.3"
							fill="var(--accent)"
							stroke="none"
						/></svg
					>Parameters</span
				>
				<button
					class="panel-close"
					aria-label="Close settings"
					onclick={() => (showSettings = false)}
				>
					<svg width="10" height="10" viewBox="0 0 10 10" stroke="currentColor" stroke-width="1.5">
						<path d="M2 2l6 6M8 2l-6 6" stroke-linecap="round" />
					</svg>
				</button>
			</div>

			<!-- Presets -->
			<div class="param">
				<div class="param-head">
					<span class="param-label"
						><svg
							width="12"
							height="12"
							viewBox="0 0 24 24"
							fill="none"
							stroke="var(--accent)"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><path d="M4 4h7v7H4zM13 4h7v4h-7zM13 11h7v9h-7zM4 14h7v6H4z" /></svg
						> Presets</span
					>
					<span class="param-info-wrap" class:show-tip={openTip === 'presets'}>
						<button
							class="param-info"
							onmouseenter={() => {
								openTip = 'presets';
							}}
							onmouseleave={() => {
								openTip = null;
							}}
							onclick={() => {
								openTip = openTip === 'presets' ? null : 'presets';
							}}>?</button
						>
						<span class="param-tip"
							>Save and switch between full setting configurations. Applying a preset that only
							changes live parameters (mutation, pairs, steps, suppression, appearance) steers the
							running simulation without resetting it. Grid or seed changes reset the soup.</span
						>
					</span>
					<div class="preset-header-actions">
						{#if activePresetId}
							<button
								class="preset-delete-active"
								onclick={deleteActivePreset}
								title="Delete selected preset"
								aria-label="Delete selected preset"
							>
								<svg
									width="12"
									height="12"
									viewBox="0 0 14 14"
									fill="none"
									stroke="currentColor"
									stroke-width="1.4"
									stroke-linecap="round"
									stroke-linejoin="round"
								>
									<path d="M2.5 3.5h9M5.5 3.5V2.5h3v1M4 3.5l.5 8h5l.5-8M6 5.5v4M8 5.5v4" />
								</svg>
							</button>
						{/if}
						<button
							class="preset-save-toggle"
							class:active={showPresetSave}
							onclick={() => {
								showPresetSave = !showPresetSave;
								if (showPresetSave) newPresetName = '';
							}}
							title="Save current settings as a preset"
						>
							<svg
								width="11"
								height="11"
								viewBox="0 0 12 12"
								fill="none"
								stroke="currentColor"
								stroke-width="1.4"
							>
								<path d="M6 2v8M2 6h8" stroke-linecap="round" />
							</svg>
						</button>
					</div>
				</div>
				{#if showPresetSave}
					<div class="preset-save-row">
						<input
							class="preset-name-input"
							type="text"
							placeholder="Preset name…"
							bind:value={newPresetName}
							onkeydown={(e) => {
								if (e.key === 'Enter' && newPresetName.trim()) {
									requestSavePreset(newPresetName);
								} else if (e.key === 'Escape') {
									showPresetSave = false;
									newPresetName = '';
								}
							}}
						/>
						<button
							class="preset-name-save"
							disabled={!newPresetName.trim()}
							onclick={() => requestSavePreset(newPresetName)}
							title="Save preset">Save</button
						>
					</div>
				{/if}
				<label class="paper-preset-label" for="paper-preset">Paper conditions</label>
				<select
					id="paper-preset"
					class="instrument-select"
					value={selectedPaperId}
					disabled={machineBusy || !engine}
					onchange={(event) => {
						const preset = PAPER_PRESETS.find((entry) => entry.id === event.currentTarget.value);
						if (preset) void applyPaperPreset(preset);
					}}
				>
					<option value="">Choose a paper condition…</option>
					{#each ['discovery', 'intervention', 'demonstration'] as category (category)}
						<optgroup label={category.charAt(0).toUpperCase() + category.slice(1)}>
							{#each PAPER_PRESETS.filter((entry) => entry.category === category) as preset (preset.id)}
								<option value={preset.id}>{preset.name}</option>
							{/each}
						</optgroup>
					{/each}
				</select>
				{#if selectedPaper}
					<div class="paper-condition">
						<p>{scientificText(selectedPaper.description)}</p>
						<p><strong>Look for</strong> {scientificText(selectedPaper.whatToLookFor)}</p>
						<details>
							<summary>Finding, limits &amp; sources</summary>
							<p>{selectedPaper.finding}</p>
							<p>{scientificText(selectedPaper.limits)}</p>
							<p>
								<strong>Registered condition</strong>
								{selectedPaper.published.width ?? 'isolated'} × {selectedPaper.published.height ??
									'culture'}, {selectedPaper.published.pairAttempts ?? 'assay'} pair attempts; horizon
								{selectedPaper.published.horizon.toLocaleString()}
								{selectedPaper.published.horizonUnit}. {scientificText(
									selectedPaper.published.note
								)}
							</p>
							<p>{selectedPaper.published.reset}</p>
							<p class="source-path">{selectedPaper.source.registration}</p>
							{#if selectedPaper.source.world}<p class="source-path">
									{selectedPaper.source.world}
								</p>{/if}
							{#each selectedPaper.source.results as source (source)}<p class="source-path">
									{source}
								</p>{/each}
							<p>
								Sources are paths in the companion algocell-paper repository. This GPU run is a
								fresh realization; equal seeds do not promise the archived scheduling trajectory.
							</p>
						</details>
					</div>
				{/if}
				<div class="preset-chips">
					{#each presets as preset (preset.id)}
						<button
							class="preset-chip"
							class:active={activePresetId === preset.id}
							class:builtin={preset.builtin}
							onclick={() => applyPreset(preset)}
							title={preset.builtin
								? `Built-in preset: ${preset.name}`
								: `Apply preset: ${preset.name}`}
						>
							{#if activePresetId === preset.id}
								<svg
									class="preset-chip-check"
									width="9"
									height="9"
									viewBox="0 0 12 12"
									fill="none"
									stroke="currentColor"
									stroke-width="2"
									stroke-linecap="round"
									stroke-linejoin="round"><path d="M2 6l3 3 5-5" /></svg
								>
							{/if}
							{preset.name}
						</button>
					{/each}
				</div>
			</div>

			<div class="param machine-controls">
				<div class="param-head">
					<label class="param-label" for="machine-select">Processor</label>
				</div>
				<select
					id="machine-select"
					class="instrument-select"
					value={machine}
					onchange={chooseMachine}
					disabled={machineBusy || !engine}
				>
					<option value="z80-0.2">Z80 · compat 0.2</option>
					<option value="z80-0.3">Z80 · 0.3 classic</option>
					<option value="z80-nmos">Z80 · 0.3 Zilog NMOS</option>
					<option value="m6502">NMOS 6502</option>
					<option value="i8086">8086 · coherent fetch</option>
				</select>
				<details class="world-contract">
					<summary
						>World contract <span class="contract-shape">L{tapeLength} / P{pairLength}</span
						></summary
					>
					<div class="world-geometry-row">
						<label for="tape-length">Tape L</label><input
							id="tape-length"
							class="seed-input"
							type="number"
							min="4"
							max="32"
							value={tapeLength}
							disabled={machineBusy}
							onchange={(event) => {
								tapeLength = Math.max(4, Math.min(32, Number(event.currentTarget.value)));
								pairLength = Math.max(pairLength, tapeLength * 2);
								initialCondition = { kind: 'random' };
								selectedPaperId = '';
								void updateMachine(true);
							}}
						/>
						<label for="pair-length">Ring P</label><input
							id="pair-length"
							class="seed-input"
							type="number"
							min={tapeLength * 2}
							max="128"
							value={pairLength}
							disabled={machineBusy}
							onchange={(event) => {
								pairLength = Math.max(
									tapeLength * 2,
									Math.min(128, Number(event.currentTarget.value))
								);
								void updateMachine(true);
							}}
						/>
					</div>
					<label class="world-control-label" for="stack-convention">Stack placement</label>
					<select
						id="stack-convention"
						class="instrument-select"
						bind:value={stackConvention}
						disabled={machineBusy}
						onchange={() => {
							void updateMachine(true);
						}}
					>
						<option value="top">Top at B’s last byte</option><option value="first-push"
							>First push at B’s last byte</option
						>{#if isa === 'z80'}<option value="legacy-sp">Original SP = FFFF</option>{/if}
					</select>
					<label class="instrument-check"
						><input
							type="checkbox"
							bind:checked={randomRegisters}
							disabled={machineBusy}
							onchange={() => {
								void updateMachine();
							}}
						/> Random registers</label
					>
					<label class="world-control-label" for="dial-mode">Dial stream</label>
					<select
						id="dial-mode"
						class="instrument-select"
						bind:value={dialMode}
						disabled={machineBusy}
						onchange={() => {
							void updateMachine();
						}}
						><option value="historical-pcg">Historical PCG</option><option value="counter"
							>Counter</option
						></select
					>
				</details>
				<div class="param-head chemistry-head">
					<label class="param-label" for="lethal-probability">Lethal zero</label><span
						class="param-val"
						>{(lethalProbability * 100).toFixed(
							lethalProbability && lethalProbability < 0.01 ? 1 : 0
						)}%</span
					>
				</div>
				<input
					id="lethal-probability"
					class="slider"
					type="range"
					min="0"
					max="1"
					step="0.01"
					bind:value={lethalProbability}
					disabled={machineBusy}
					onchange={() => {
						void updateMachine();
					}}
				/>
				{#if machineBusy}<p class="instrument-note" role="status">Preparing the processor…</p>{/if}
				{#if machineError}<p class="instrument-error" role="alert">{machineError}</p>{/if}
			</div>

			<div class="param">
				<div class="param-head">
					<span class="param-label"
						><svg
							width="12"
							height="12"
							viewBox="0 0 24 24"
							fill="none"
							stroke="var(--accent)"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><rect x="3" y="3" width="8" height="8" /><rect
								x="13"
								y="3"
								width="8"
								height="8"
							/><rect x="3" y="13" width="8" height="8" /><rect
								x="13"
								y="13"
								width="8"
								height="8"
							/></svg
						> Grid</span
					>
					<span class="param-info-wrap" class:show-tip={openTip === 'grid'}>
						<button
							class="param-info"
							onmouseenter={() => {
								openTip = 'grid';
							}}
							onmouseleave={() => {
								openTip = null;
							}}
							onclick={() => {
								openTip = openTip === 'grid' ? null : 'grid';
							}}>?</button
						>
						<span class="param-tip"
							>Grid topology and dimensions. Hex grids produce more organic patterns. Changing
							resets simulation.</span
						>
					</span>
					<div class="grid-type-toggle">
						<button
							class="grid-type-btn"
							class:active={gridType === 'square'}
							onclick={() => handleGridTypeChange('square')}
							title="Square grid — 4 neighbors"
							><svg
								width="10"
								height="10"
								viewBox="0 0 10 10"
								fill="none"
								stroke="currentColor"
								stroke-width="1.2"
								><rect x="1" y="1" width="3.5" height="3.5" /><rect
									x="5.5"
									y="1"
									width="3.5"
									height="3.5"
								/><rect x="1" y="5.5" width="3.5" height="3.5" /><rect
									x="5.5"
									y="5.5"
									width="3.5"
									height="3.5"
								/></svg
							></button
						>
						<button
							class="grid-type-btn"
							class:active={gridType === 'hex'}
							onclick={() => handleGridTypeChange('hex')}
							title="Hex grid — 6 neighbors"
							><svg
								width="10"
								height="10"
								viewBox="0 0 10 10"
								fill="none"
								stroke="currentColor"
								stroke-width="1.2"><path d="M5 1L8.5 3V7L5 9L1.5 7V3Z" /></svg
							></button
						>
					</div>
				</div>
				<div class="grid-size-row">
					<label class="grid-size-label">
						W
						<input
							type="number"
							class="grid-size-input"
							bind:value={gridWidth}
							min="50"
							max="500"
							step="10"
						/>
					</label>
					<span class="grid-size-x">&times;</span>
					<label class="grid-size-label">
						H
						<input
							type="number"
							class="grid-size-input"
							bind:value={gridHeight}
							min="50"
							max="500"
							step="10"
						/>
					</label>
					<button class="seed-apply" onclick={applyGridConfig} title="Apply & Reset">
						<svg
							width="12"
							height="12"
							viewBox="0 0 14 14"
							fill="none"
							stroke="currentColor"
							stroke-width="1.5"
						>
							<path d="M2 7a5 5 0 1 1 1 3" stroke-linecap="round" />
							<path d="M2 3v4h4" stroke-linecap="round" stroke-linejoin="round" />
						</svg>
					</button>
				</div>
			</div>

			<div class="param">
				<div class="param-head">
					<label class="param-label" for="seed-input"
						><svg
							width="12"
							height="12"
							viewBox="0 0 24 24"
							fill="none"
							stroke="var(--accent)"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><rect x="2" y="2" width="20" height="20" rx="3" /><circle
								cx="8"
								cy="8"
								r="1.5"
								fill="currentColor"
								stroke="none"
							/><circle cx="16" cy="8" r="1.5" fill="currentColor" stroke="none" /><circle
								cx="8"
								cy="16"
								r="1.5"
								fill="currentColor"
								stroke="none"
							/><circle cx="16" cy="16" r="1.5" fill="currentColor" stroke="none" /><circle
								cx="12"
								cy="12"
								r="1.5"
								fill="currentColor"
								stroke="none"
							/></svg
						> Seed</label
					>
					<span class="param-info-wrap" class:show-tip={openTip === 'seed'}>
						<button
							class="param-info"
							onmouseenter={() => {
								openTip = 'seed';
							}}
							onmouseleave={() => {
								openTip = null;
							}}
							onclick={() => {
								openTip = openTip === 'seed' ? null : 'seed';
							}}>?</button
						>
						<span class="param-tip"
							>Random seed for initial soup generation. Changing seed requires reset.</span
						>
					</span>
				</div>
				<div class="seed-row">
					<input
						id="seed-input"
						type="number"
						class="seed-input"
						value={seed}
						min="0"
						onchange={handleSeedChange}
					/>
					<button class="seed-apply" aria-label="Reset with seed" onclick={handleReset}>
						<svg
							width="12"
							height="12"
							viewBox="0 0 14 14"
							fill="none"
							stroke="currentColor"
							stroke-width="1.5"
						>
							<path d="M2 7a5 5 0 1 1 1 3" stroke-linecap="round" />
							<path d="M2 3v4h4" stroke-linecap="round" stroke-linejoin="round" />
						</svg>
					</button>
				</div>
			</div>

			<div class="param">
				<div class="param-head">
					<label class="param-label" for="noise-input"
						><svg
							width="12"
							height="12"
							viewBox="0 0 24 24"
							fill="none"
							stroke="var(--accent)"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><path
								d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"
							/></svg
						> Mutation rate</label
					>
					<span class="param-info-wrap" class:show-tip={openTip === 'mutation'}>
						<button
							class="param-info"
							onmouseenter={() => {
								openTip = 'mutation';
							}}
							onmouseleave={() => {
								openTip = null;
							}}
							onclick={() => {
								openTip = openTip === 'mutation' ? null : 'mutation';
							}}>?</button
						>
						<span class="param-tip"
							>Random byte flips per batch. Higher = more mutations = faster evolution. Applied
							live.</span
						>
					</span>
					<span class="param-val"
						>{#if noiseExp >= 31}off{:else}1/2<sup>{noiseExp}</sup>{/if}</span
					>
				</div>
				<div class="slider-track-wrap">
					<input
						id="noise-input"
						type="range"
						class="slider"
						value={noiseExp >= 31 ? 1 : 13 - noiseExp}
						min="1"
						max="12"
						step="1"
						oninput={(e) => {
							const val = 13 - parseInt((e.target as HTMLInputElement).value);
							if (!isNaN(val) && engine) {
								noiseExp = val;
								engine.noiseExp = val;
							}
						}}
					/>
				</div>
				<label class="instrument-check"
					><input
						type="checkbox"
						checked={noiseExp >= 31}
						onchange={(event) => {
							noiseExp = event.currentTarget.checked ? 31 : 4;
							if (engine) engine.noiseExp = noiseExp;
						}}
					/> Mutation off</label
				>
			</div>

			<div class="param">
				<div class="param-head">
					<label class="param-label" for="pairs-input"
						><svg
							width="12"
							height="12"
							viewBox="0 0 24 24"
							fill="none"
							stroke="var(--accent)"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><rect x="3" y="3" width="7" height="7" /><rect
								x="14"
								y="3"
								width="7"
								height="7"
							/><rect x="3" y="14" width="7" height="7" /><rect
								x="14"
								y="14"
								width="7"
								height="7"
							/></svg
						> Pairs/batch</label
					>
					<span class="param-info-wrap" class:show-tip={openTip === 'pairs'}>
						<button
							class="param-info"
							onmouseenter={() => {
								openTip = 'pairs';
							}}
							onmouseleave={() => {
								openTip = null;
							}}
							onclick={() => {
								openTip = openTip === 'pairs' ? null : 'pairs';
							}}>?</button
						>
						<span class="param-tip"
							>Cell pairs evaluated per step. More = faster evolution, higher GPU load. Applied
							live.</span
						>
					</span>
					<span class="param-val">{pairCount}</span>
				</div>
				<div class="slider-track-wrap">
					<input
						id="pairs-input"
						type="range"
						class="slider"
						bind:value={pairCount}
						min={256}
						max={MAX_BATCH_PAIR_N}
						step={256}
						oninput={handlePairCountChange}
					/>
				</div>
			</div>

			<div class="param">
				<div class="param-head">
					<label class="param-label" for="steps-input"
						><svg
							width="12"
							height="12"
							viewBox="0 0 24 24"
							fill="none"
							stroke="var(--accent)"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" /></svg
						>
						{isa === 'z80' ? 'Z80 steps' : 'Execution steps'}</label
					>
					<span class="param-info-wrap" class:show-tip={openTip === 'steps'}>
						<button
							class="param-info"
							onmouseenter={() => {
								openTip = 'steps';
							}}
							onmouseleave={() => {
								openTip = null;
							}}
							onclick={() => {
								openTip = openTip === 'steps' ? null : 'steps';
							}}>?</button
						>
						<span class="param-tip"
							>Instruction slots per pair execution. More steps = longer programs can run, slower
							throughput. Applied live.</span
						>
					</span>
					<span class="param-val">{z80Steps}</span>
				</div>
				<div class="slider-track-wrap">
					<input
						id="steps-input"
						type="range"
						class="slider"
						bind:value={z80Steps}
						min={16}
						max={1024}
						step={16}
						oninput={(e) => {
							const val = parseInt((e.target as HTMLInputElement).value);
							if (!isNaN(val) && engine) {
								z80Steps = val;
								engine.z80Steps = val;
							}
						}}
					/>
				</div>
			</div>

			<div class="param">
				<div class="param-head">
					<span class="param-label"
						><svg
							width="12"
							height="12"
							viewBox="0 0 24 24"
							fill="none"
							stroke="var(--accent)"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><circle cx="12" cy="12" r="10" /><line
								x1="4.93"
								y1="4.93"
								x2="19.07"
								y2="19.07"
							/></svg
						> Suppress opcodes</span
					>
					<span class="param-info-wrap" class:show-tip={openTip === 'suppress'}>
						<button
							class="param-info"
							onmouseenter={() => {
								openTip = 'suppress';
							}}
							onmouseleave={() => {
								openTip = null;
							}}
							onclick={() => {
								openTip = openTip === 'suppress' ? null : 'suppress';
							}}>?</button
						>
						<span class="param-tip"
							>{#if isa === 'z80'}Remove instructions from the CPU. A pattern is a mnemonic
								substring (LDIR, LD (HL), PUSH), a whole family (family:block-copy, family:stack,
								family:writes-mem) or one opcode (ed:B0, cb:FF, 3E). Prefixed ED/CB instructions are
								reachable individually. A suppressed opcode executes as a NOP; its operand bytes run
								as the next instruction. Use it to take the usual replication machinery away and see
								what else evolves.{:else}{isaSuppressionDescription}{/if}</span
						>
					</span>
					{#if suppressPatterns.length > 0}
						<button class="suppress-clear" onclick={clearSuppressedOpcodes} title="Clear all">
							<svg
								width="10"
								height="10"
								viewBox="0 0 10 10"
								stroke="currentColor"
								stroke-width="1.5"
							>
								<path d="M2 2l6 6M8 2l-6 6" stroke-linecap="round" />
							</svg>
						</button>
					{/if}
				</div>
				<div class="suppress-input-wrap">
					<input
						class="suppress-input"
						type="text"
						placeholder={isa === 'z80'
							? 'family:block-copy; LDIR; PUSH; ed:B0'
							: isa === 'm6502'
								? 'family:stack; STA; opcode:8D'
								: 'family:block-copy; MOV; opcode:A4'}
						bind:value={suppressInput}
						onkeydown={(e) => {
							if (e.key === 'Enter' && suppressInput.trim()) {
								addSuppressPattern(suppressInput);
								suppressInput = '';
							}
						}}
					/>
					{#if suppressInput.trim()}
						{@const preview = suppressionPreview(suppressInput)}
						<span class="suppress-preview"
							>↵ to suppress {preview} instruction{preview !== 1 ? 's' : ''}</span
						>
					{/if}
				</div>
				{#if suppressPatterns.length > 0}
					<div class="suppress-chips">
						{#each suppressPatterns as pat (pat)}
							<button
								class="suppress-chip"
								onclick={() => removeSuppressPattern(pat)}
								title="Remove pattern: {pat} (click to remove)"
							>
								{pat}
								<svg
									width="8"
									height="8"
									viewBox="0 0 8 8"
									stroke="currentColor"
									stroke-width="1.5"
								>
									<path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke-linecap="round" />
								</svg>
							</button>
						{/each}
						<span class="suppress-count" title={describeSuppression(suppression)}
							>{suppressedTotal} instruction{suppressedTotal !== 1 ? 's' : ''}</span
						>
					</div>
				{/if}
			</div>

			<div class="param">
				<div class="param-head">
					<span class="param-label"
						><svg
							width="12"
							height="12"
							viewBox="0 0 24 24"
							fill="none"
							stroke="var(--accent)"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><circle cx="12" cy="12" r="10" /><path
								d="M12 2a7 7 0 0 0 0 20 4 4 0 0 1 0-8 4 4 0 0 0 0-8z"
							/><circle cx="12" cy="9" r="1" fill="var(--accent)" /></svg
						> Appearance</span
					>
					<span class="param-info-wrap" class:show-tip={openTip === 'cmap'}>
						<button
							class="param-info"
							onmouseenter={() => {
								openTip = 'cmap';
							}}
							onmouseleave={() => {
								openTip = null;
							}}
							onclick={() => {
								openTip = openTip === 'cmap' ? null : 'cmap';
							}}>?</button
						>
						<span class="param-tip"
							>Color scheme for byte values. Each Z80 opcode category gets a distinct color family.</span
						>
					</span>
					<button
						class="color-adj-toggle"
						class:active={showColorAdj}
						onclick={() => (showColorAdj = !showColorAdj)}
						title="Color adjustments"
					>
						<svg
							width="10"
							height="10"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
						>
							<path
								d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"
							/><circle cx="12" cy="12" r="3" />
						</svg>
					</button>
				</div>
				<div class="cmap-row">
					{#each COLORMAP_NAMES as name (name)}
						<button
							class="cmap-btn"
							class:active={colormapName === name}
							onclick={() => handleColormapChange(name)}
						>
							<span class="cmap-preview cmap-preview-{name}"></span>
							{name}
						</button>
					{/each}
				</div>
				{#if showColorAdj}
					<div class="color-adj-panel">
						<div class="color-adj-row">
							<span class="color-adj-label">Brightness</span>
							<input
								type="range"
								class="slider color-adj-slider"
								min="-0.5"
								max="0.5"
								step="0.01"
								bind:value={brightness}
							/>
						</div>
						<div class="color-adj-row">
							<span class="color-adj-label">Contrast</span>
							<input
								type="range"
								class="slider color-adj-slider"
								min="0.3"
								max="2.5"
								step="0.01"
								bind:value={contrast}
							/>
						</div>
						<div class="color-adj-row">
							<span class="color-adj-label">Saturation</span>
							<input
								type="range"
								class="slider color-adj-slider"
								min="0"
								max="3"
								step="0.01"
								bind:value={saturation}
							/>
						</div>
						<button
							class="color-adj-reset"
							onclick={() => {
								brightness = 0;
								contrast = 1;
								saturation = 1;
							}}>Reset</button
						>
					</div>
				{/if}
				<label class="grid-lines-toggle">
					<input type="checkbox" bind:checked={showGridLines} class="sr-only" />
					<span class="check-box" class:checked={showGridLines}>
						{#if showGridLines}
							<svg
								width="8"
								height="8"
								viewBox="0 0 12 12"
								fill="none"
								stroke="var(--accent)"
								stroke-width="2"
								stroke-linecap="round"
								stroke-linejoin="round"><path d="M2 6l3 3 5-5" /></svg
							>
						{/if}
					</span>
					<span>Grid lines</span>
				</label>
			</div>

			<div class="param">
				<div class="param-head">
					<span class="param-label"
						><svg
							width="12"
							height="12"
							viewBox="0 0 24 24"
							fill="none"
							stroke="var(--accent)"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><rect x="3" y="3" width="18" height="18" rx="2" /><line
								x1="3"
								y1="12"
								x2="21"
								y2="12"
							/><line x1="12" y1="3" x2="12" y2="21" /></svg
						> Detail</span
					>
					<span class="param-info-wrap" class:show-tip={openTip === 'detail'}>
						<button
							class="param-info"
							onmouseenter={() => {
								openTip = 'detail';
							}}
							onmouseleave={() => {
								openTip = null;
							}}
							onclick={() => {
								openTip = openTip === 'detail' ? null : 'detail';
							}}>?</button
						>
						<span class="param-tip"
							>Toggle between detailed view (showing individual bytes within each cell) and simple
							view (one averaged color per cell). Hover tooltip always shows full detail.</span
						>
					</span>
					<div class="detail-toggle">
						<button
							class="grid-type-btn"
							class:active={!simpleView}
							onclick={() => (simpleView = false)}
							title="Detailed — show individual bytes"
							><svg
								width="10"
								height="10"
								viewBox="0 0 10 10"
								fill="none"
								stroke="currentColor"
								stroke-width="1"
								><rect x="0.5" y="0.5" width="4" height="4" /><rect
									x="5.5"
									y="0.5"
									width="4"
									height="4"
								/><rect x="0.5" y="5.5" width="4" height="4" /><rect
									x="5.5"
									y="5.5"
									width="4"
									height="4"
								/></svg
							></button
						>
						<button
							class="grid-type-btn"
							class:active={simpleView}
							onclick={() => (simpleView = true)}
							title="Simple — one color per cell"
							><svg
								width="10"
								height="10"
								viewBox="0 0 10 10"
								fill="none"
								stroke="currentColor"
								stroke-width="1.2"><rect x="1" y="1" width="8" height="8" rx="1" /></svg
							></button
						>
					</div>
				</div>
				<label class="instrument-check"
					><input
						type="checkbox"
						checked={showConfinement}
						disabled={machineBusy || !engine}
						onchange={changeConfinement}
					/> Open / confined fetch view</label
				>
				{#if showConfinement}<p class="instrument-note">
						<span class="fetch-key confined" style="background:{byteColor(48)}"></span> No observed
						exit from focal tape
						<span class="fetch-key open" style="background:{byteColor(192)}"></span>
						Outside focal tape (B or padding) <span class="fetch-key unknown"></span> Unknown. Last initiating
						encounter only; not lifetime confinement or heredity.
					</p>{/if}
			</div>
		</div>
	{/if}

	<!-- Preset overwrite confirmation -->
	{#if overwriteTarget}
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<div class="confirm-backdrop" onclick={() => (overwriteTarget = null)} role="presentation">
			<div
				class="confirm-dialog"
				onclick={(e) => e.stopPropagation()}
				role="dialog"
				aria-modal="true"
				aria-label="Algocell details"
				tabindex="-1"
			>
				<div class="confirm-title">Overwrite preset?</div>
				<div class="confirm-body">
					A preset named <strong>{overwriteTarget.name}</strong> already exists. Replace its saved settings
					with the current configuration?
				</div>
				<div class="confirm-actions">
					<button class="confirm-btn confirm-cancel" onclick={() => (overwriteTarget = null)}
						>Cancel</button
					>
					<button class="confirm-btn confirm-ok" onclick={confirmOverwrite}>Overwrite</button>
				</div>
			</div>
		</div>
	{/if}

	<!-- Genome tooltip -->
	{#if hoveredCell >= 0 && cellData && !isPanning}
		{#if gridType === 'hex' && tapeLength === 19}
			<!-- Hex byte tooltip: 19 hex-shaped bytes in 3-4-5-4-3 hexagonal arrangement -->
			<div class="genome-tip hex-tip" style={tooltipStyle}>
				<div class="hex-byte-grid">
					{#each genomeCells as cell, i (i)}
						<div class="hex-byte-cell hex-byte-pos-{i}" class:operand={!cell.isOpcode}>
							<div
								class="hex-byte-inner"
								style="background:{byteColor(cell.byteVal)};color:{byteLuminance(cell.byteVal) > 0.4
									? 'rgba(0,0,0,0.85)'
									: 'rgba(255,255,255,0.9)'}"
							>
								{cell.label}
							</div>
						</div>
					{/each}
				</div>
			</div>
		{:else}
			<!-- Square grid tooltip -->
			<div class="genome-tip" style={tooltipStyle}>
				<div class="tip-grid">
					{#each genomeCells as cell, i (i)}
						<div
							class="tip-cell"
							class:operand={!cell.isOpcode}
							style="background:{byteColor(cell.byteVal)};color:{byteLuminance(cell.byteVal) > 0.4
								? 'rgba(0,0,0,0.85)'
								: 'rgba(255,255,255,0.9)'}"
						>
							{cell.label}
						</div>
					{/each}
				</div>
			</div>
		{/if}
	{/if}

	<!-- Help modal -->
	{#if showHelp}
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<div class="modal-backdrop" onclick={() => (showHelp = false)} role="presentation">
			<div
				class="modal"
				onclick={(e) => e.stopPropagation()}
				role="dialog"
				aria-modal="true"
				aria-label="Algocell details"
				tabindex="-1"
			>
				<div class="modal-header">
					<div class="modal-tabs">
						<button
							class="modal-tab"
							aria-label="Overview"
							class:active={helpTab === 'overview'}
							onclick={() => (helpTab = 'overview')}
						>
							<svg
								width="14"
								height="14"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								stroke-width="2"
								stroke-linecap="round"
								stroke-linejoin="round"
								><circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line
									x1="12"
									y1="8"
									x2="12.01"
									y2="8"
								/></svg
							>
							Overview
						</button>
						<button
							class="modal-tab"
							aria-label="Guide"
							class:active={helpTab === 'guide'}
							onclick={() => (helpTab = 'guide')}>Guide</button
						>
						<button
							class="modal-tab"
							aria-label="Visuals"
							class:active={helpTab === 'visuals'}
							onclick={() => (helpTab = 'visuals')}
						>
							<svg
								width="14"
								height="14"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								stroke-width="2"
								stroke-linecap="round"
								stroke-linejoin="round"
								><circle cx="13.5" cy="6.5" r="2.5" /><circle cx="19" cy="17" r="2.5" /><circle
									cx="6"
									cy="18"
									r="2.5"
								/></svg
							>
							Visuals
						</button>
						<button
							class="modal-tab"
							aria-label="Z80"
							class:active={helpTab === 'z80'}
							onclick={() => (helpTab = 'z80')}
						>
							<svg
								width="14"
								height="14"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								stroke-width="2"
								stroke-linecap="round"
								stroke-linejoin="round"
								><rect x="4" y="4" width="16" height="16" rx="2" /><rect
									x="9"
									y="9"
									width="6"
									height="6"
								/><line x1="9" y1="1" x2="9" y2="4" /><line x1="15" y1="1" x2="15" y2="4" /><line
									x1="9"
									y1="20"
									x2="9"
									y2="23"
								/><line x1="15" y1="20" x2="15" y2="23" /><line
									x1="20"
									y1="9"
									x2="23"
									y2="9"
								/><line x1="20" y1="14" x2="23" y2="14" /><line x1="1" y1="9" x2="4" y2="9" /><line
									x1="1"
									y1="14"
									x2="4"
									y2="14"
								/></svg
							>
							Z80
						</button>
						<button
							class="modal-tab"
							aria-label="Parameters"
							class:active={helpTab === 'params'}
							onclick={() => (helpTab = 'params')}
						>
							<svg
								width="14"
								height="14"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								stroke-width="2"
								stroke-linecap="round"
								stroke-linejoin="round"
								><line x1="4" y1="21" x2="4" y2="14" /><line x1="4" y1="10" x2="4" y2="3" /><line
									x1="12"
									y1="21"
									x2="12"
									y2="12"
								/><line x1="12" y1="8" x2="12" y2="3" /><line
									x1="20"
									y1="21"
									x2="20"
									y2="16"
								/><line x1="20" y1="12" x2="20" y2="3" /><line x1="1" y1="14" x2="7" y2="14" /><line
									x1="9"
									y1="8"
									x2="15"
									y2="8"
								/><line x1="17" y1="16" x2="23" y2="16" /></svg
							>
							Params
						</button>
						<button
							class="modal-tab"
							aria-label="Analysis"
							class:active={helpTab === 'analysis'}
							onclick={() => (helpTab = 'analysis')}
						>
							<svg
								width="14"
								height="14"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								stroke-width="2"
								stroke-linecap="round"
								stroke-linejoin="round"
							>
								<polyline points="22,12 18,12 15,21 9,3 6,12 2,12" />
							</svg>
							Analysis
						</button>
						<button
							class="modal-tab tab-keys"
							aria-label="Keys"
							class:active={helpTab === 'keys'}
							onclick={() => (helpTab = 'keys')}
						>
							<svg
								width="14"
								height="14"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								stroke-width="2"
								stroke-linecap="round"
								stroke-linejoin="round"
								><rect x="2" y="4" width="20" height="16" rx="2" /><line
									x1="6"
									y1="8"
									x2="6.01"
									y2="8"
								/><line x1="10" y1="8" x2="10.01" y2="8" /><line
									x1="14"
									y1="8"
									x2="14.01"
									y2="8"
								/><line x1="18" y1="8" x2="18.01" y2="8" /><line
									x1="6"
									y1="12"
									x2="6.01"
									y2="12"
								/><line x1="10" y1="12" x2="10.01" y2="12" /><line
									x1="14"
									y1="12"
									x2="14.01"
									y2="12"
								/><line x1="18" y1="12" x2="18.01" y2="12" /><line
									x1="8"
									y1="16"
									x2="16"
									y2="16"
								/></svg
							>
							Keys
						</button>
						<button
							class="modal-tab"
							aria-label="About"
							class:active={helpTab === 'about'}
							onclick={() => (helpTab = 'about')}
						>
							<svg width="14" height="14" viewBox="0 0 32 32"
								><ellipse cx="16" cy="20" rx="14" ry="14" fill="var(--accent)" /><line
									x1="16"
									y1="8"
									x2="16"
									y2="32"
									stroke="#1a1210"
									stroke-width="1.5"
								/><circle cx="9" cy="16" r="2.5" fill="#1a1210" /><circle
									cx="10.5"
									cy="24"
									r="2.2"
									fill="#1a1210"
								/><circle cx="23" cy="16" r="2.5" fill="#1a1210" /><circle
									cx="21.5"
									cy="24"
									r="2.2"
									fill="#1a1210"
								/><circle cx="16" cy="7" r="7" fill="#1a1210" /><circle
									cx="12.8"
									cy="5.8"
									r="1.5"
									fill="#e0cfc0"
								/><circle cx="19.2" cy="5.8" r="1.5" fill="#e0cfc0" /><path
									d="M12 1.5 Q9 -2 5 0"
									fill="none"
									stroke="#1a1210"
									stroke-width="1.8"
									stroke-linecap="round"
								/><path
									d="M20 1.5 Q23 -2 27 0"
									fill="none"
									stroke="#1a1210"
									stroke-width="1.8"
									stroke-linecap="round"
								/><circle cx="5" cy="0" r="1.8" fill="#1a1210" /><circle
									cx="27"
									cy="0"
									r="1.8"
									fill="#1a1210"
								/></svg
							>
							About
						</button>
					</div>
					<button class="panel-close" aria-label="Close help" onclick={() => (showHelp = false)}>
						<svg
							width="10"
							height="10"
							viewBox="0 0 10 10"
							stroke="currentColor"
							stroke-width="1.5"
						>
							<path d="M2 2l6 6M8 2l-6 6" stroke-linecap="round" />
						</svg>
					</button>
				</div>
				<div class="modal-body">
					{#if helpTab === 'guide'}
						{#each guideSections as section (section.id)}<h4>{section.title}</h4>
							{#each section.paragraphs as paragraph (paragraph)}<p>
									{scientificText(paragraph)}
								</p>{/each}{#if section.steps}<ol class="guide-steps">
									{#each section.steps as step (step)}<li>{scientificText(step)}</li>{/each}
								</ol>{/if}{/each}
					{:else if helpTab === 'overview'}
						<p class="cmap-note">
							Based on <a
								href="https://arxiv.org/abs/2406.19108"
								target="_blank"
								rel="noopener"
								class="help-link">Agüera y Arcas et al. (2024)</a
							>. Re-implemented in WebGPU + SvelteKit from the
							<a href="https://github.com/znah/zff" target="_blank" rel="noopener" class="help-link"
								>original JavaScript/WASM implementation</a
							>.
						</p>

						<h4>Cells &amp; Tapes</h4>
						<p>
							A grid of roughly 20,000 cells, fitted to the viewport, begins with random bytes. Each
							cell holds a short <strong>tape</strong> interpreted by the selected processor. The original
							layouts below show the default 16-byte square and 19-byte hex tapes; paper conditions can
							use other tape and ring lengths:
						</p>

						<!-- Tape layout visuals -->
						<div class="tape-layouts">
							<div class="tape-layout-item">
								<div class="tape-layout-label">Square &mdash; 16 bytes (4&times;4)</div>
								<svg viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg" class="tape-svg">
									{#each Array(16).keys() as i (i)}
										{@const col = i % 4}
										{@const row = Math.floor(i / 4)}
										<rect
											x={col * 40}
											y={row * 40}
											width="38"
											height="38"
											rx="3"
											fill="hsl({i * 22}, 45%, 32%)"
											stroke="rgba(255,255,255,0.12)"
											stroke-width="1"
										/>
										<text
											x={col * 40 + 19}
											y={row * 40 + 23}
											text-anchor="middle"
											fill="rgba(255,255,255,0.7)"
											font-size="11"
											font-family="monospace">{i.toString(16).toUpperCase().padStart(2, '0')}</text
										>
									{/each}
								</svg>
								<div class="tape-layout-note">4 neighbors (up, down, left, right)</div>
							</div>
							<div class="tape-layout-item">
								<div class="tape-layout-label">Hexagonal &mdash; 19 bytes (3-4-5-4-3)</div>
								<svg viewBox="0 -2 160 148" xmlns="http://www.w3.org/2000/svg" class="tape-svg">
									{#each [{ x: 49, y: 16, i: 0 }, { x: 80, y: 16, i: 1 }, { x: 111, y: 16, i: 2 }, { x: 33, y: 44, i: 3 }, { x: 64, y: 44, i: 4 }, { x: 95, y: 44, i: 5 }, { x: 126, y: 44, i: 6 }, { x: 18, y: 72, i: 7 }, { x: 49, y: 72, i: 8 }, { x: 80, y: 72, i: 9 }, { x: 111, y: 72, i: 10 }, { x: 142, y: 72, i: 11 }, { x: 33, y: 100, i: 12 }, { x: 64, y: 100, i: 13 }, { x: 95, y: 100, i: 14 }, { x: 126, y: 100, i: 15 }, { x: 49, y: 128, i: 16 }, { x: 80, y: 128, i: 17 }, { x: 111, y: 128, i: 18 }] as h (h.i)}
										<polygon
											points={Array.from({ length: 6 }, (_, k) => {
												const a = (Math.PI / 180) * (60 * k - 30);
												return `${h.x + 15 * Math.cos(a)},${h.y + 15 * Math.sin(a)}`;
											}).join(' ')}
											fill="hsl({h.i * 19}, 45%, 32%)"
											stroke="rgba(255,255,255,0.12)"
											stroke-width="1"
										/>
										<text
											x={h.x}
											y={h.y + 4}
											text-anchor="middle"
											fill="rgba(255,255,255,0.7)"
											font-size="9"
											font-family="monospace"
											>{h.i.toString(16).toUpperCase().padStart(2, '0')}</text
										>
									{/each}
								</svg>
								<div class="tape-layout-note">6 neighbors per cell</div>
							</div>
						</div>

						<h4>What Happens Each Step</h4>
						<p>
							Every simulation step, the following cycle runs thousands of times in parallel on the
							GPU:
						</p>
						<ol class="help-steps">
							<li><strong>Pick a pair.</strong> Choose a random cell and one of its neighbors.</li>
							<li>
								<strong>Combine their tapes.</strong> Both cells' bytes are copied into a single contiguous
								buffer &mdash; Cell&nbsp;A's tape first, then Cell&nbsp;B's. The buffer wraps around,
								so in the default 32-byte ring, address 31 is followed by address 0. Extra ring padding
								starts at zero and is discarded on writeback.
							</li>
						</ol>

						<!-- Full-cycle tape combination visual -->
						{@const subs = [
							'₀',
							'₁',
							'₂',
							'₃',
							'₄',
							'₅',
							'₆',
							'₇',
							'₈',
							'₉',
							'₁₀',
							'₁₁',
							'₁₂',
							'₁₃',
							'₁₄',
							'₁₅'
						]}
						{@const sz = 18}
						{@const gap = 2}
						{@const step = sz + gap}
						{@const gridW = 4 * step - gap}
						{@const axL = 160 - gridW - 20}
						{@const bxL = 160 + 20}
						{@const bufY = 140}
						{@const bufCellW = 9}
						{@const bufGap = 0.5}
						{@const bufTotalW = 32 * (bufCellW + bufGap) - bufGap}
						{@const bufXStart = (320 - bufTotalW) / 2}
						{@const z80Y = bufY + 34}
						{@const wbArrowY = z80Y + 22}
						{@const wbCellY = wbArrowY + 32}
						<svg viewBox="0 0 320 325" xmlns="http://www.w3.org/2000/svg" class="tape-combine-svg">
							<!-- ── Row 1: Source cells ── -->
							<text
								x={axL + gridW / 2}
								y="10"
								text-anchor="middle"
								fill="#c8a060"
								font-size="9.5"
								font-family="-apple-system, sans-serif"
								font-weight="600">Cell A</text
							>
							{#each Array(16).keys() as i (i)}
								{@const col = i % 4}
								{@const row = Math.floor(i / 4)}
								<rect
									x={axL + col * step}
									y={14 + row * step}
									width={sz}
									height={sz}
									rx="2"
									fill="hsl({i * 22}, 45%, 32%)"
									stroke="rgba(255,255,255,0.12)"
									stroke-width="0.5"
								/>
								<text
									x={axL + col * step + sz / 2}
									y={14 + row * step + sz / 2 + 3}
									text-anchor="middle"
									fill="rgba(255,255,255,0.85)"
									font-size="7"
									font-family="monospace">a{subs[i]}</text
								>
							{/each}

							<text
								x="160"
								y="58"
								text-anchor="middle"
								fill="rgba(255,255,255,0.35)"
								font-size="16"
								font-family="monospace"
								font-weight="300">+</text
							>

							<text
								x={bxL + gridW / 2}
								y="10"
								text-anchor="middle"
								fill="#60a0c8"
								font-size="9.5"
								font-family="-apple-system, sans-serif"
								font-weight="600">Cell B</text
							>
							{#each Array(16).keys() as i (i)}
								{@const col = i % 4}
								{@const row = Math.floor(i / 4)}
								<rect
									x={bxL + col * step}
									y={14 + row * step}
									width={sz}
									height={sz}
									rx="2"
									fill="hsl({i * 19 + 200}, 40%, 32%)"
									stroke="rgba(255,255,255,0.12)"
									stroke-width="0.5"
								/>
								<text
									x={bxL + col * step + sz / 2}
									y={14 + row * step + sz / 2 + 3}
									text-anchor="middle"
									fill="rgba(255,255,255,0.85)"
									font-size="7"
									font-family="monospace">b{subs[i]}</text
								>
							{/each}

							<!-- ── Row 2: Arrow down ── -->
							<line
								x1="160"
								y1="100"
								x2="160"
								y2="116"
								stroke="rgba(255,255,255,0.3)"
								stroke-width="1.2"
							/>
							<polygon points="160,122 155,114 165,114" fill="rgba(255,255,255,0.3)" />
							<text
								x="160"
								y="134"
								text-anchor="middle"
								fill="rgba(255,255,255,0.45)"
								font-size="8"
								font-family="-apple-system, sans-serif">Shared buffer (wrapping)</text
							>

							<!-- ── Row 3: Combined buffer ── -->
							{#each Array(32).keys() as i (i)}
								{@const isA = i < 16}
								{@const bx = bufXStart + i * (bufCellW + bufGap)}
								<rect
									x={bx}
									y={bufY}
									width={bufCellW}
									height="14"
									rx="1"
									fill={isA ? `hsl(${i * 22}, 45%, 32%)` : `hsl(${(i - 16) * 19 + 200}, 40%, 32%)`}
									stroke="rgba(255,255,255,0.08)"
									stroke-width="0.5"
								/>
								{#if i % 4 === 0}
									<text
										x={bx + bufCellW / 2}
										y={bufY + 10}
										text-anchor="middle"
										fill="rgba(255,255,255,0.7)"
										font-size="5"
										font-family="monospace">{isA ? 'a' : 'b'}{subs[isA ? i : i - 16]}</text
									>
								{/if}
							{/each}
							<!-- Labels under buffer -->
							<text
								x={bufXStart + (16 * (bufCellW + bufGap)) / 2 - bufGap / 2}
								y={bufY + 24}
								text-anchor="middle"
								fill="#c8a060"
								font-size="7"
								font-family="-apple-system, sans-serif">a₀ – a₁₅</text
							>
							<text
								x={bufXStart +
									16 * (bufCellW + bufGap) +
									(16 * (bufCellW + bufGap)) / 2 -
									bufGap / 2}
								y={bufY + 24}
								text-anchor="middle"
								fill="#60a0c8"
								font-size="7"
								font-family="-apple-system, sans-serif">b₀ – b₁₅</text
							>
							<!-- Wrap indicator -->
							<path
								d="M {bufXStart + bufTotalW + 2},{bufY + 7} C {bufXStart + bufTotalW + 8},{bufY +
									7} {bufXStart + bufTotalW + 8},{bufY - 4} {bufXStart + bufTotalW + 2},{bufY -
									4} L {bufXStart - 2},{bufY - 4} C {bufXStart - 8},{bufY - 4} {bufXStart -
									8},{bufY + 7} {bufXStart - 2},{bufY + 7}"
								fill="none"
								stroke="rgba(255,255,255,0.12)"
								stroke-width="0.7"
								stroke-dasharray="2,2"
							/>
							<text
								x={bufXStart + bufTotalW + 6}
								y={bufY + 2}
								fill="rgba(255,255,255,0.2)"
								font-size="6"
								font-family="-apple-system, sans-serif">↻</text
							>

							<!-- ── Z80 execution arrow ── -->
							<line
								x1={bufXStart}
								y1={z80Y}
								x2={bufXStart + bufTotalW - 6}
								y2={z80Y}
								stroke="var(--accent)"
								stroke-width="1"
								opacity="0.5"
							/>
							<polygon
								points="{bufXStart + bufTotalW},{z80Y} {bufXStart + bufTotalW - 6},{z80Y -
									3} {bufXStart + bufTotalW - 6},{z80Y + 3}"
								fill="var(--accent)"
								opacity="0.5"
							/>
							<text
								x="160"
								y={z80Y + 12}
								text-anchor="middle"
								fill="var(--accent)"
								font-size="7.5"
								opacity="0.7"
								font-family="-apple-system, sans-serif">⚡ Z80 executes (PC=0, regs=0)</text
							>

							<!-- ── Row 4: Write-back ── -->
							<!-- Split arrows -->
							<line
								x1="160"
								y1={wbArrowY}
								x2="160"
								y2={wbArrowY + 12}
								stroke="rgba(255,255,255,0.25)"
								stroke-width="1"
							/>
							<line
								x1="160"
								y1={wbArrowY + 12}
								x2={axL + gridW / 2}
								y2={wbArrowY + 24}
								stroke="rgba(255,255,255,0.25)"
								stroke-width="1"
							/>
							<line
								x1="160"
								y1={wbArrowY + 12}
								x2={bxL + gridW / 2}
								y2={wbArrowY + 24}
								stroke="rgba(255,255,255,0.25)"
								stroke-width="1"
							/>
							<polygon
								points="{axL + gridW / 2},{wbArrowY + 28} {axL + gridW / 2 - 4},{wbArrowY +
									21} {axL + gridW / 2 + 4},{wbArrowY + 21}"
								fill="rgba(255,255,255,0.25)"
							/>
							<polygon
								points="{bxL + gridW / 2},{wbArrowY + 28} {bxL + gridW / 2 - 4},{wbArrowY +
									21} {bxL + gridW / 2 + 4},{wbArrowY + 21}"
								fill="rgba(255,255,255,0.25)"
							/>

							<!-- Write-back label -->
							<text
								x="160"
								y={wbArrowY + 8}
								text-anchor="middle"
								fill="rgba(255,255,255,0.35)"
								font-size="7"
								font-family="-apple-system, sans-serif">write back</text
							>

							<!-- Modified Cell A -->
							<text
								x={axL + gridW / 2}
								y={wbCellY}
								text-anchor="middle"
								fill="#c8a060"
								font-size="8"
								font-family="-apple-system, sans-serif"
								opacity="0.7">Cell A′</text
							>
							{#each Array(16).keys() as i (i)}
								{@const col = i % 4}
								{@const row = Math.floor(i / 4)}
								{@const swapped = i === 2 || i === 7 || i === 11}
								<rect
									x={axL + col * step}
									y={wbCellY + 4 + row * step}
									width={sz}
									height={sz}
									rx="2"
									fill={swapped ? `hsl(${i * 19 + 200}, 40%, 32%)` : `hsl(${i * 22}, 45%, 32%)`}
									stroke={swapped ? 'rgba(96,160,200,0.5)' : 'rgba(255,255,255,0.12)'}
									stroke-width={swapped ? 1 : 0.5}
								/>
								<text
									x={axL + col * step + sz / 2}
									y={wbCellY + 4 + row * step + sz / 2 + 3}
									text-anchor="middle"
									fill="rgba(255,255,255,0.85)"
									font-size="7"
									font-family="monospace">{swapped ? `b${subs[i]}` : `a${subs[i]}`}</text
								>
							{/each}

							<!-- Modified Cell B -->
							<text
								x={bxL + gridW / 2}
								y={wbCellY}
								text-anchor="middle"
								fill="#60a0c8"
								font-size="8"
								font-family="-apple-system, sans-serif"
								opacity="0.7">Cell B′</text
							>
							{#each Array(16).keys() as i (i)}
								{@const col = i % 4}
								{@const row = Math.floor(i / 4)}
								{@const swapped = i === 1 || i === 5 || i === 13}
								<rect
									x={bxL + col * step}
									y={wbCellY + 4 + row * step}
									width={sz}
									height={sz}
									rx="2"
									fill={swapped ? `hsl(${i * 22}, 45%, 32%)` : `hsl(${i * 19 + 200}, 40%, 32%)`}
									stroke={swapped ? 'rgba(200,160,96,0.5)' : 'rgba(255,255,255,0.12)'}
									stroke-width={swapped ? 1 : 0.5}
								/>
								<text
									x={bxL + col * step + sz / 2}
									y={wbCellY + 4 + row * step + sz / 2 + 3}
									text-anchor="middle"
									fill="rgba(255,255,255,0.85)"
									font-size="7"
									font-family="monospace">{swapped ? `a${subs[i]}` : `b${subs[i]}`}</text
								>
							{/each}

							<!-- Mutate annotation -->
							<text
								x="160"
								y={wbCellY + 4 * step + 12}
								text-anchor="middle"
								fill="rgba(255,255,255,0.3)"
								font-size="7.5"
								font-family="-apple-system, sans-serif">~ random mutations applied ~</text
							>
						</svg>

						<ol class="help-steps" start="3">
							<li>
								<strong>Run the processor.</strong> The chosen machine starts at byte 0 with its declared
								reset. The original Z80 starts with zero registers except SP = FFFF. Instructions can
								read and write the shared ring, including the partner’s bytes. The diagrams illustrate
								the default Z80 world.
							</li>
							<li>
								<strong>Write back.</strong> The modified buffer is split and written back to both cells
								in the grid. Note some bytes have been cross-contaminated (highlighted).
							</li>
							<li>
								<strong>Mutate.</strong> Random byte-replacement attempts are applied across the grid.
							</li>
						</ol>
						<p>
							After all pairs are processed, the next update begins. Copiers can emerge and spread
							in this world, but a short run need not find one. Culture a candidate before
							interpreting visible order as reproduction.
						</p>

						<h4>Order, copying and evidence</h4>
						<p>
							In the default benign Z80 reset, A, H and L begin at zero. Instructions such as
							<code>LD (HL),A</code> (store&nbsp;A at address&nbsp;HL) write zeros &mdash; since A, H,
							and L are all zero. This makes NOP (0x00) accumulate rapidly. Random code acts as a &ldquo;zero
							pump,&rdquo; flooding the grid with NOPs.
						</p>
						<p>
							Some compact load–push or exchange/stack loops can copy material into a neighbor.
							Whether a candidate keeps reproducing, enters its partner during instruction fetching,
							or preserves an introduced difference are separate measurements. The schematic below
							illustrates possible composition changes; it is not a guaranteed phase sequence.
						</p>
						<svg
							viewBox="0 0 320 180"
							xmlns="http://www.w3.org/2000/svg"
							style="width:100%;margin:8px 0;display:block;border-radius:8px;background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.04);padding:12px 8px;"
						>
							<!-- axes -->
							<line
								x1="40"
								y1="20"
								x2="40"
								y2="140"
								stroke="rgba(255,255,255,0.15)"
								stroke-width="1"
							/>
							<line
								x1="40"
								y1="140"
								x2="300"
								y2="140"
								stroke="rgba(255,255,255,0.15)"
								stroke-width="1"
							/>
							<!-- axis labels -->
							<text
								x="170"
								y="168"
								text-anchor="middle"
								fill="#8a7a6a"
								font-size="9"
								font-family="-apple-system, BlinkMacSystemFont, sans-serif">Time</text
							>
							<text
								x="14"
								y="80"
								text-anchor="middle"
								fill="#8a7a6a"
								font-size="9"
								font-family="-apple-system, BlinkMacSystemFont, sans-serif"
								transform="rotate(-90,14,80)">Concentration</text
							>
							<!-- NOP curve (white/gray, starts low, rises to moderate peak, then declines) -->
							<path
								d="M 40,125 C 50,95 60,68 80,62 S 120,64 150,78 Q 200,108 300,122"
								fill="none"
								stroke="rgba(220,215,210,0.7)"
								stroke-width="2"
							/>
							<text
								x="68"
								y="54"
								fill="rgba(220,215,210,0.8)"
								font-size="10"
								font-family="-apple-system, BlinkMacSystemFont, sans-serif">NOP</text
							>
							<!-- Self-replicating curve (orange, stays low then rises smoothly) -->
							<path
								d="M 40,130 C 100,130 150,120 186,88 C 220,58 260,42 300,42"
								fill="none"
								stroke="var(--accent)"
								stroke-width="2"
							/>
							<text
								x="195"
								y="36"
								fill="var(--accent)"
								font-size="10"
								font-family="-apple-system, BlinkMacSystemFont, sans-serif"
								>Self-Replicating Bytes</text
							>
							<!-- Phase transition dashed line at crossing (x≈180) -->
							<line
								x1="180"
								y1="22"
								x2="180"
								y2="140"
								stroke="rgba(120,160,220,0.5)"
								stroke-width="1"
								stroke-dasharray="4,3"
							/>
							<!-- Phase transition label -->
							<text
								x="180"
								y="155"
								text-anchor="middle"
								fill="rgba(120,160,220,0.7)"
								font-size="9"
								font-family="-apple-system, BlinkMacSystemFont, sans-serif">Phase Transition</text
							>
						</svg>
					{:else if helpTab === 'visuals'}
						<p>
							Each cell is colored by its first byte. The colormap assigns distinct colors to Z80
							opcode categories. Other bytes get a muted hue from a continuous sweep.
						</p>

						<div class="cmap-section">
							<div class="cmap-label">Special</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x00)}"></span><span
										class="cmap-hex">00</span
									> NOP
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x76)}"></span><span
										class="cmap-hex">76</span
									> HALT
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">16-bit loads</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x01)}"></span><span
										class="cmap-hex">01</span
									> LD BC,nn
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x11)}"></span><span
										class="cmap-hex">11</span
									> LD DE,nn
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x21)}"></span><span
										class="cmap-hex">21</span
									> LD HL,nn
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x31)}"></span><span
										class="cmap-hex">31</span
									> LD SP,nn
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">Memory access</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x2a)}"></span><span
										class="cmap-hex">2A</span
									> LD HL,(nn)
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x3a)}"></span><span
										class="cmap-hex">3A</span
									> LD A,(nn)
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x22)}"></span><span
										class="cmap-hex">22</span
									> LD (nn),HL
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x32)}"></span><span
										class="cmap-hex">32</span
									> LD (nn),A
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">Block transfer</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xed)}"></span><span
										class="cmap-hex">ED</span
									> ED prefix
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xb0)}"></span><span
										class="cmap-hex">B0</span
									> LDIR
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xb8)}"></span><span
										class="cmap-hex">B8</span
									> LDDR
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xa0)}"></span><span
										class="cmap-hex">A0</span
									> LDI
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xa8)}"></span><span
										class="cmap-hex">A8</span
									> LDD
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">Stack</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xc5)}"></span><span
										class="cmap-hex">C5</span
									> PUSH BC
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xd5)}"></span><span
										class="cmap-hex">D5</span
									> PUSH DE
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xe5)}"></span><span
										class="cmap-hex">E5</span
									> PUSH HL
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xf5)}"></span><span
										class="cmap-hex">F5</span
									> PUSH AF
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xc1)}"></span><span
										class="cmap-hex">C1</span
									> POP BC
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xd1)}"></span><span
										class="cmap-hex">D1</span
									> POP DE
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xe1)}"></span><span
										class="cmap-hex">E1</span
									> POP HL
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xf1)}"></span><span
										class="cmap-hex">F1</span
									> POP AF
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">Flow control</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xc3)}"></span><span
										class="cmap-hex">C3</span
									> JP nn
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xcd)}"></span><span
										class="cmap-hex">CD</span
									> CALL nn
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xc9)}"></span><span
										class="cmap-hex">C9</span
									> RET
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x18)}"></span><span
										class="cmap-hex">18</span
									> JR
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x10)}"></span><span
										class="cmap-hex">10</span
									> DJNZ
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">Prefixes &amp; exchange</div>
							<div class="cmap-grid">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xcb)}"></span><span
										class="cmap-hex">CB</span
									> Bit ops
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xdd)}"></span><span
										class="cmap-hex">DD</span
									> IX ops
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xfd)}"></span><span
										class="cmap-hex">FD</span
									> IY ops
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xe3)}"></span><span
										class="cmap-hex">E3</span
									> EX (SP),HL
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xeb)}"></span><span
										class="cmap-hex">EB</span
									> EX DE,HL
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x08)}"></span><span
										class="cmap-hex">08</span
									> EX AF,AF'
								</div>
							</div>
						</div>

						<div class="cmap-section">
							<div class="cmap-label">Ranges</div>
							<div class="cmap-grid wide">
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0x60)}"></span><span
										class="cmap-hex">40-7F</span
									> Register LD
								</div>
								<div class="cmap-entry">
									<span class="help-swatch" style="background:{byteColor(0xa0)}"></span><span
										class="cmap-hex">80-BF</span
									> ALU ops
								</div>
							</div>
						</div>

						<p class="cmap-note">
							All other bytes get a muted tone from a continuous hue sweep. Switch colormaps in
							Settings.
						</p>

						<h4>Cell Tooltips</h4>
						<p>
							Hover any cell to see its bytes disassembled as Z80 instructions. Square mode shows a
							4&times;4 grid of 16 bytes; hex mode shows a 3-4-5-4-3 hexagonal cluster of 19 bytes
							matching the cell's honeycomb shape.
						</p>
						<ul class="help-list">
							<li>
								<strong>Opcode bytes</strong> show the instruction mnemonic (e.g. <code>NOP</code>,
								<code>POP HL</code>, <code>LD B,C</code>)
							</li>
							<li>
								<strong>Operand bytes</strong> show the raw hex value (e.g. <code>00</code>,
								<code>3F</code>) and appear dimmed &mdash; these are data consumed by the preceding
								instruction
							</li>
						</ul>
						<p>
							For example, byte 0x00 appears as <code>NOP</code> when executed as an instruction,
							but as <code>00</code> when it is an operand of a multi-byte instruction like
							<code>LD BC,nn</code>.
						</p>

						<h4>Frequency Chart</h4>
						<p>
							The bottom-left widget shows the most common byte values in real time. When
							suppression is active, suppressed bytes are collapsed into a single card so you always
							see the top non-suppressed opcodes. Hover any tile for a plain-English description;
							click to suppress or un-suppress it. Toggle the chart sparkline with <kbd>S</kbd>.
						</p>

						<h4>Instruction Suppression (ablation)</h4>
						{#if isa === 'z80'}
							<p>
								Type a pattern into the Suppress field in Settings and press Enter to remove the
								matching instructions from the CPU. The Z80 then executes a suppressed opcode as a
								NOP. Every instruction on the base, <code>CB</code> and <code>ED</code> pages is
								addressable, so block copies like <code>LDIR</code> can be removed on their own.
							</p>
							<ul class="help-list">
								<li>
									<strong>Families</strong> &mdash; <code>family:block-copy</code>
									(LDI/LDD/LDIR/LDDR),
									<code>family:stack</code> (PUSH/POP), <code>family:ld8-mem</code>,
									<code>family:call-ret</code>, &hellip; and the pseudo-family
									<code>family:writes-mem</code> (every instruction that writes memory)
								</li>
								<li>
									<strong>Exact opcodes</strong> &mdash; <code>ed:B0</code> (LDIR),
									<code>cb:FF</code>
									(SET 7,A), <code>3E</code> or <code>base:3E</code> (LD A,n)
								</li>
								<li>
									<strong>Mnemonic substrings</strong> &mdash; <code>LDIR</code>,
									<code>LD (HL)</code>,
									<code>PUSH</code>; <code>LD</code> matches every load on every page
								</li>
								<li>
									<strong>Click-to-suppress</strong> &mdash; click any tile in the frequency widget to
									add its exact mnemonic as a pattern
								</li>
								<li>
									<strong>Remove patterns</strong> &mdash; click the &times; on any chip to remove it,
									or use the clear button to remove all
								</li>
							</ul>
							<p>
								Semantics: a suppressed base-page opcode is a 1-byte NOP, so its operand bytes run
								as the next instruction (as if the opcode had been removed from the ISA). Suppressed
								<code>CB</code>/<code>ED</code> instructions are 2-byte NOPs. IX/IY forms follow
								their base opcode. Suppression prevents <em>execution</em>, not <em>existence</em>:
								a suppressed byte value can still be written into memory or carried as data.
							</p>
						{:else}
							<p>
								Type a pattern into the Suppress field in Settings and press Enter. {isaSuppressionDescription}
							</p>
							<p>
								Click an opcode frequency tile to add its exact opcode as a pattern. Remove a chip
								to restore its operations, or use the clear button to restore all operations. Static
								disassembly still shows the ordinary instruction decoding; measured fetches show the
								execution path under suppression.
							</p>
						{/if}
					{:else if helpTab === 'z80'}
						<h4>Registers</h4>
						<p>The Z80 has 8-bit and 16-bit registers used as operands:</p>
						<div class="z80-reg-table">
							<div class="z80-reg-row head">
								<span>Name</span><span>Size</span><span>Role</span>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">A</span><span>8-bit</span><span
									>Accumulator &mdash; main arithmetic register</span
								>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">F</span><span>8-bit</span><span
									>Flags (zero, carry, sign, parity)</span
								>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">B, C</span><span>8-bit</span><span
									>General purpose; B is loop counter for DJNZ</span
								>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">D, E</span><span>8-bit</span><span
									>General purpose; DE = destination for block ops</span
								>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">H, L</span><span>8-bit</span><span
									>General purpose; HL = primary memory pointer</span
								>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">SP</span><span>16-bit</span><span>Stack pointer</span>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">BC, DE, HL</span><span>16-bit</span><span
									>Register pairs (B+C, D+E, H+L)</span
								>
							</div>
							<div class="z80-reg-row">
								<span class="z80-reg">AF</span><span>16-bit</span><span
									>Accumulator + flags pair</span
								>
							</div>
						</div>

						<h4>Notation</h4>
						<div class="z80-notation-table">
							<div class="z80-notation-row head"><span>Syntax</span><span>Meaning</span></div>
							<div class="z80-notation-row">
								<span class="z80-reg">nn</span><span>16-bit immediate value (e.g. $1234)</span>
							</div>
							<div class="z80-notation-row">
								<span class="z80-reg">n</span><span>8-bit immediate value (e.g. $FF)</span>
							</div>
							<div class="z80-notation-row">
								<span class="z80-reg">d</span><span>Signed offset for relative jumps</span>
							</div>
							<div class="z80-notation-row">
								<span class="z80-reg">(HL)</span><span>Memory at address in HL</span>
							</div>
							<div class="z80-notation-row">
								<span class="z80-reg">(nn)</span><span>Memory at absolute address nn</span>
							</div>
							<div class="z80-notation-row">
								<span class="z80-reg">(SP)</span><span>Memory at top of stack</span>
							</div>
						</div>

						<h4>Instruction Reference</h4>

						<div class="z80-cat">
							<div class="z80-cat-label">Data Movement</div>
							<div class="z80-instr-grid">
								<div class="z80-instr"><span class="z80-op">LD x,y</span> Copy y into x</div>
								<div class="z80-instr">
									<span class="z80-op">PUSH rr</span> Push 16-bit pair onto stack
								</div>
								<div class="z80-instr">
									<span class="z80-op">POP rr</span> Pop 16 bits from stack into pair
								</div>
								<div class="z80-instr">
									<span class="z80-op">EX x,y</span> Exchange (swap) x and y
								</div>
							</div>
						</div>

						<div class="z80-cat">
							<div class="z80-cat-label">Arithmetic &amp; Logic</div>
							<div class="z80-instr-grid">
								<div class="z80-instr"><span class="z80-op">ADD A,x</span> A = A + x</div>
								<div class="z80-instr"><span class="z80-op">ADC A,x</span> A = A + x + carry</div>
								<div class="z80-instr"><span class="z80-op">SUB x</span> A = A &minus; x</div>
								<div class="z80-instr">
									<span class="z80-op">SBC A,x</span> A = A &minus; x &minus; carry
								</div>
								<div class="z80-instr"><span class="z80-op">AND x</span> A = A &amp; x</div>
								<div class="z80-instr"><span class="z80-op">OR x</span> A = A | x</div>
								<div class="z80-instr"><span class="z80-op">XOR x</span> A = A ^ x</div>
								<div class="z80-instr">
									<span class="z80-op">CP x</span> Compare A with x (sets flags)
								</div>
								<div class="z80-instr"><span class="z80-op">INC x</span> x = x + 1</div>
								<div class="z80-instr"><span class="z80-op">DEC x</span> x = x &minus; 1</div>
							</div>
						</div>

						<div class="z80-cat">
							<div class="z80-cat-label">Flow Control</div>
							<div class="z80-instr-grid">
								<div class="z80-instr"><span class="z80-op">JP nn</span> Jump to address</div>
								<div class="z80-instr">
									<span class="z80-op">JR d</span> Jump relative by offset d
								</div>
								<div class="z80-instr">
									<span class="z80-op">DJNZ d</span> Decrement B, jump if B &ne; 0
								</div>
								<div class="z80-instr">
									<span class="z80-op">CALL nn</span> Push PC, jump to address
								</div>
								<div class="z80-instr">
									<span class="z80-op">RET</span> Pop PC (return from call)
								</div>
								<div class="z80-instr">
									<span class="z80-op">NOP</span> No operation (do nothing)
								</div>
								<div class="z80-instr"><span class="z80-op">HALT</span> Stop execution</div>
							</div>
						</div>

						<div class="z80-cat">
							<div class="z80-cat-label">Block Operations</div>
							<div class="z80-instr-grid">
								<div class="z80-instr">
									<span class="z80-op">LDI</span> Copy (HL)&rarr;(DE), inc both, dec BC
								</div>
								<div class="z80-instr">
									<span class="z80-op">LDD</span> Copy (HL)&rarr;(DE), dec both, dec BC
								</div>
								<div class="z80-instr">
									<span class="z80-op">LDIR</span> LDI repeated until BC = 0
								</div>
								<div class="z80-instr">
									<span class="z80-op">LDDR</span> LDD repeated until BC = 0
								</div>
							</div>
						</div>

						<div class="z80-cat">
							<div class="z80-cat-label">Bit &amp; Rotate</div>
							<div class="z80-instr-grid">
								<div class="z80-instr"><span class="z80-op">BIT n,x</span> Test bit n of x</div>
								<div class="z80-instr"><span class="z80-op">SET n,x</span> Set bit n of x</div>
								<div class="z80-instr"><span class="z80-op">RES n,x</span> Reset bit n of x</div>
								<div class="z80-instr">
									<span class="z80-op">RL x</span> Rotate left through carry
								</div>
								<div class="z80-instr">
									<span class="z80-op">RR x</span> Rotate right through carry
								</div>
								<div class="z80-instr"><span class="z80-op">SLA x</span> Shift left arithmetic</div>
								<div class="z80-instr"><span class="z80-op">SRL x</span> Shift right logical</div>
							</div>
						</div>

						<h4>An exchange/stack copying route</h4>
						<p>
							<span class="z80-op">POP HL</span> reads two bytes and advances the stack pointer;
							<span class="z80-op">EX (SP),HL</span> exchanges that word with the next stack position.
							A working loop can move its machinery into the partner under a suitable ring and reset contract.
							This is one available route, not a claim that every occurrence copies or that it always
							dominates. The paper’s historical L16 Stage G first exemplars were usually load–push designs.
						</p>

						<Mermaid
							chart={`
graph TD
    A("POP HL\nread 2 bytes from memory") -->|"HL now holds data"| B("EX (SP),HL\nwrite 2 bytes forward")
    B -->|"repeat"| A
`}
						/>
					{:else if helpTab === 'params'}
						<h4>Grid Type</h4>
						<p>
							Switch between <strong>Square</strong> and <strong>Hex</strong> topologies. Square cells
							hold 16 bytes (4&times;4) with 4 neighbors. Hex cells hold 19 bytes (3-4-5-4-3 honeycomb)
							with 6 neighbors. Changing resets the simulation.
						</p>

						<h4>Grid Size (W &times; H)</h4>
						<p>
							Width and height of the grid in cells. The initial grid fits roughly 20,000 cells to
							the viewport. Larger worlds provide more encounters and use more GPU memory and
							compute. Changing resets the simulation.
						</p>

						<h4>Seed</h4>
						<p>
							The random seed used to initialize the grid. Same seed produces the same starting
							state. Change it and press Reset to try different initial conditions.
						</p>

						<h4>Mutation Rate</h4>
						<p>
							Mutation exponent k sets floor(pair attempts / 2ᵏ) random byte-replacement attempts
							per population update. With 8,192 pair slots and k = 4, that is 512 attempts.
							Replacement can repeat the old value; this is not a fraction of all population bytes.
							The off checkbox disables mutation.
						</p>

						<h4>Pairs / Batch</h4>
						<p>
							The number of proposed neighboring pair slots per update. Collision claims can reject
							a proposal, so the number of accepted encounters is lower. More pair attempts usually
							require more GPU time; an update is not a synchronized generation.
						</p>

						<h4>{isa === 'z80' ? 'Z80 Steps' : 'Execution steps'}</h4>
						<p>
							Maximum instruction slots per encounter, with default 128. These are execution steps
							rather than hardware clock cycles; Z80 block repeats and 8086 REP consume one slot per
							iteration. Larger budgets permit longer execution and usually reduce update
							throughput. A halted lane does not execute its unused slots.
						</p>

						<h4>Suppress Opcodes</h4>
						{#if isa === 'z80'}
							<p>
								Remove instructions from the CPU: a mnemonic substring (<code>LDIR</code>,
								<code>PUSH</code>), a family (<code>family:block-copy</code>,
								<code>family:writes-mem</code>) or an exact opcode (<code>ed:B0</code>). The Z80
								skips them as NOPs. Use this to take the usual replication machinery away and
								observe which alternative strategies evolve. Patterns appear as removable chips; the
								count shows how many instructions are affected across the base, CB and ED pages.
							</p>
						{:else}<p>
								{isaSuppressionDescription} Patterns appear as removable chips; the count shows the affected
								opcodes.
							</p>{/if}

						<h4>Colormap</h4>
						<p>
							Choose between three visual themes: Rainbow (warm opcode-aware), Ocean (cool blues),
							and Thermal (heat map).
						</p>
					{:else if helpTab === 'analysis'}
						<h4>Species Tracking</h4>
						<p>
							The <strong>Species</strong> tab identifies distinct cell types by hashing each cell's full
							byte content (FNV-1a). Cells with identical bytes share a hash and are counted as one species.
							The top 10 are shown as colored tiles whose color matches the cell's average appearance
							on the grid. Click any species tile to inspect its genome (the 4x4 byte grid with Z80 disassembly).
							Hover over individual bytes in the modal to see what each instruction does.
						</p>

						<h4>Diversity Metrics</h4>
						<p>
							The <strong>Diversity</strong> tab tracks four metrics over time, each independently scaled:
						</p>
						<!-- eslint-disable svelte/no-at-html-tags -- KaTeX output is safe (static strings) -->
						<div class="help-metric-list">
							<div class="help-metric-item">
								<span class="help-metric-swatch" style="background:rgb(255,180,50)"></span>
								<div>
									<strong>H</strong> (Shannon entropy) &mdash; measures evenness of species
									distribution. High H = many species at similar frequency. Low H = monoculture.
									<div class="help-metric-eq">{@html texBlock('H = -\\sum_{i} p_i \\ln p_i')}</div>
								</div>
							</div>
							<div class="help-metric-item">
								<span class="help-metric-swatch" style="background:rgb(100,200,255)"></span>
								<div>
									<strong>D</strong> (Simpson's diversity) &mdash; probability that two random cells
									are different species. {@html tex('D \\to 1')} = diverse, {@html tex('D \\to 0')} =
									monoculture.
									<div class="help-metric-eq">{@html texBlock('D = 1 - \\sum_{i} p_i^{\\,2}')}</div>
								</div>
							</div>
							<div class="help-metric-item">
								<span class="help-metric-swatch" style="background:rgb(120,255,120)"></span>
								<div>
									<strong>SPP</strong> (species richness) &mdash; count of species with {@html tex(
										'p_i > 0.1\\%'
									)}. Tracks how many distinct genomes are meaningfully present.
								</div>
							</div>
							<div class="help-metric-item">
								<span class="help-metric-swatch" style="background:rgb(255,130,180)"></span>
								<div>
									<strong>UNIQ</strong> (total unique blocks) &mdash; every distinct 16-byte sequence
									in the grid. Raw genetic diversity before any abundance threshold.
								</div>
							</div>
						</div>
						<!-- eslint-enable svelte/no-at-html-tags -->

						<h4>Checkpoints</h4>
						<p>
							<kbd>{navigator.platform.includes('Mac') ? 'Cmd' : 'Ctrl'}+S</kbd> saves a checkpoint
							as an <code>.acell</code> file. This is a valid PNG image (viewable in any image app)
							with the full simulation state appended after the image data. Load checkpoints with
							<kbd>{navigator.platform.includes('Mac') ? 'Cmd' : 'Ctrl'}+O</kbd> or drag &amp; drop onto
							the canvas. The file contains the complete grid, all parameters, and suppression patterns
							&mdash; everything needed to resume from these bytes. Legacy files do not store the complete
							random stream, so they do not reproduce the exact historical continuation.
						</p>

						<h4>Opcode Suppression</h4>
						<p>
							Suppressing opcodes (e.g. LD, EX, ADD) forces the system to find replication
							strategies using only the remaining instructions. This can push the system past
							trivial self-replication into richer emergent behavior like spatial speciation,
							error-correcting patterns, and division of labor between cell types.
						</p>
					{:else if helpTab === 'keys'}
						<div class="modal-shortcuts">
							<div class="shortcut"><kbd>Space</kbd> Play / Pause</div>
							<div class="shortcut"><kbd>R</kbd> Reset</div>
							<div class="shortcut"><kbd>S</kbd> Toggle chart</div>
							<div class="shortcut"><kbd>H</kbd> Toggle help</div>
							<div class="shortcut"><kbd>F</kbd> Fit view</div>
							<div class="shortcut">
								<kbd>{navigator.platform.includes('Mac') ? 'Cmd' : 'Ctrl'}+S</kbd> Save checkpoint
							</div>
							<div class="shortcut">
								<kbd>{navigator.platform.includes('Mac') ? 'Cmd' : 'Ctrl'}+O</kbd> Load checkpoint
							</div>
							<div class="shortcut"><kbd>Scroll</kbd> Zoom</div>
							<div class="shortcut"><kbd>Drag</kbd> Pan</div>
							<div class="shortcut"><kbd>Dbl-click</kbd> Reset view</div>
						</div>
					{:else if helpTab === 'about'}
						{#each ABOUT_SECTIONS as section (section.id)}
							<h4>{section.title}</h4>
							{#each section.paragraphs as paragraph (paragraph)}<p>
									{scientificText(paragraph)}
								</p>{/each}
							{#if section.links}<p class="cmap-note">
									{#each section.links as link, index (link.href)}{#if index > 0}
											·
										{/if}
										<!-- eslint-disable svelte/no-navigation-without-resolve -- Curated scientific references are absolute external URLs. -->
										<a class="help-link" href={link.href} target="_blank" rel="noopener noreferrer"
											>{link.label}</a
										>
										<!-- eslint-enable svelte/no-navigation-without-resolve -->
									{/each}
								</p>{/if}
						{/each}
						<p class="cmap-note">
							Original WebGPU instrument inspired by Alexander Mordvintsev’s <a
								class="help-link"
								href="https://github.com/znah/zff"
								target="_blank"
								rel="noopener noreferrer">zff implementation</a
							>. Developed with Claude Code. SvelteKit · TypeScript · WebGPU.
						</p>
					{/if}
				</div>
			</div>
		</div>
	{/if}
</div>

<style>
	.legacy-scope {
		color: var(--text-primary);
	}
	.legacy-scope :global(button),
	.legacy-scope :global(input),
	.legacy-scope :global(select) {
		min-height: 0;
	}

	.legacy-scope :global(input:focus),
	.legacy-scope :global(select:focus),
	.legacy-scope :global(textarea:focus) {
		outline: none;
		box-shadow: none;
		border-color: var(--border-default);
	}
	.legacy-scope :global(input:focus-visible),
	.legacy-scope :global(select:focus-visible),
	.legacy-scope :global(textarea:focus-visible) {
		outline: 1px solid var(--accent);
		outline-offset: 2px;
	}

	/* Native selects can match :focus-visible after a pointer click in Chrome. */
	.legacy-scope.pointer-focus :global(:focus) {
		outline: none;
		box-shadow: none;
	}

	/* Prevent mobile zoom/scroll escaping on all UI overlays */
	:global(html),
	:global(body) {
		touch-action: manipulation;
		overscroll-behavior: none;
		-webkit-text-size-adjust: 100%;
	}

	/* Preserve visible keyboard focus in the historical interface. */
	button:focus-visible {
		outline: 1px solid var(--accent);
		outline-offset: 2px;
	}

	/* ── Toolbar ── */
	.toolbar {
		position: fixed;
		top: 12px;
		right: 12px;
		z-index: 40;
		display: flex;
		align-items: center;
		box-sizing: border-box;
		gap: 2px;
		padding: 3px;
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
		border-radius: 20px;
		backdrop-filter: blur(16px);
		box-shadow: 0 2px 20px rgba(0, 0, 0, 0.4);
		transition:
			border-radius 0.2s ease,
			width 0.2s ease,
			height 0.2s ease,
			gap 0.2s ease,
			padding 0.2s ease;
	}
	.toolbar.collapsed {
		/* Fixed square + centered content guarantees a perfect circle regardless
		   of the (zero-width) hidden button row still in the flex flow. */
		width: 40px;
		height: 40px;
		padding: 0;
		gap: 0;
		justify-content: center;
		overflow: hidden;
		border-radius: 50%;
		border-color: var(--border-muted);
	}
	.toolbar-buttons {
		display: flex;
		align-items: center;
		gap: 2px;
		max-width: 400px;
		opacity: 1;
		transition:
			max-width 0.2s ease,
			opacity 0.15s ease,
			gap 0.2s ease;
	}
	.toolbar-buttons.hidden {
		max-width: 0;
		opacity: 0;
		gap: 0;
		overflow: hidden;
		pointer-events: none;
	}
	.tb {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 34px;
		height: 34px;
		border-radius: 17px;
		border: none;
		background: transparent;
		cursor: pointer;
		transition: background 0.15s;
		flex-shrink: 0;
		color: var(--text-subtle);
	}
	.tb:hover {
		background: var(--bg-hover);
	}
	.tb.mono {
		font-size: 13px;
		font-family: monospace;
	}
	.tb-sep {
		width: 1px;
		height: 16px;
		background: var(--border-subtle);
		margin: 0 2px;
		flex-shrink: 0;
	}
	.collapse-btn {
		color: var(--text-subtle);
		transition:
			transform 0.2s ease,
			background 0.15s;
	}
	.collapse-btn.collapsed {
		transform: rotate(45deg);
	}

	/* Speed menu */
	.speed-menu {
		position: absolute;
		top: 100%;
		left: 50%;
		transform: translateX(-50%);
		margin-top: 6px;
		display: flex;
		flex-direction: column;
		background: var(--bg-elevated);
		border: 1px solid var(--border-muted);
		border-radius: 8px;
		overflow: hidden;
		min-width: 48px;
	}
	.speed-opt {
		padding: 7px 14px;
		font-size: 13px;
		font-family: monospace;
		color: var(--text-secondary);
		background: transparent;
		border: none;
		cursor: pointer;
		text-align: center;
	}
	.speed-opt:hover {
		background: var(--bg-hover);
	}
	.speed-opt.active {
		color: var(--accent);
	}

	/* ── Info bar ── */
	.info-bar {
		position: fixed;
		bottom: 12px;
		left: 12px;
		z-index: 40;
		display: flex;
		flex-direction: column;
		gap: 0;
		padding: 8px;
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
		border-radius: 10px;
		backdrop-filter: blur(16px);
		box-shadow: 0 2px 20px rgba(0, 0, 0, 0.4);
		font-size: 11px;
		font-family: monospace;
		width: 252px;
		overflow: visible;
	}
	.info-row {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.info-item {
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.info-label {
		color: var(--text-subtle);
		text-transform: uppercase;
		font-size: 10px;
		letter-spacing: 0.05em;
	}
	.info-value {
		color: var(--text-secondary);
		font-variant-numeric: tabular-nums;
		width: 5ch;
		text-align: right;
		display: inline-block;
		white-space: pre;
	}
	.info-sep {
		width: 1px;
		height: 10px;
		background: var(--border-subtle);
	}
	.info-chart-toggle {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 22px;
		height: 22px;
		border: none;
		background: transparent;
		cursor: pointer;
		color: var(--text-subtle);
		margin-left: auto;
		margin-right: 4px;
		border-radius: 4px;
		padding: 4px;
	}
	.info-chart-toggle:hover {
		background: var(--bg-hover);
	}
	.info-chart {
		margin-top: 6px;
		border-top: 1px solid var(--border-subtle);
		padding-top: 6px;
	}
	.info-chart-svg {
		width: 100%;
		height: 90px;
		background: var(--bg-subtle);
		border-radius: 4px;
	}
	.freq-grid {
		display: grid;
		grid-template-columns: repeat(5, 44px);
		gap: 4px;
		margin-top: 6px;
	}
	.freq-cell {
		position: relative;
		display: flex;
		align-items: center;
		justify-content: center;
		width: 44px;
		height: 44px;
		border: none;
		border-radius: 6px;
		cursor: pointer;
		font-size: 8.5px;
		font-family: monospace;
		font-weight: 600;
		text-align: center;
		line-height: 1.15;
		padding: 2px;
		overflow: hidden;
		transition: opacity 0.1s;
	}
	.freq-cell.highlighted {
		outline: 2px solid rgba(255, 255, 255, 0.7);
		z-index: 1;
	}
	.freq-cell.dimmed {
		opacity: 0.35;
	}
	.freq-cell-suppressed-group {
		background: rgba(255, 255, 255, 0.06);
		border: 1px dashed var(--border-muted);
		color: var(--text-muted);
		cursor: default;
		font-size: 9px;
	}
	.freq-cell.suppressed::after {
		content: '';
		position: absolute;
		inset: 0;
		background: repeating-linear-gradient(
			-45deg,
			transparent,
			transparent 3px,
			rgba(0, 0, 0, 0.4) 3px,
			rgba(0, 0, 0, 0.4) 5px
		);
		border-radius: 6px;
		pointer-events: none;
	}
	.freq-rank {
		position: absolute;
		top: 2px;
		left: 3px;
		font-size: 6px;
		opacity: 0.5;
		font-weight: 400;
	}
	.freq-label {
		overflow: hidden;
		word-break: break-all;
		max-height: 100%;
		display: -webkit-box;
		-webkit-line-clamp: 2;
		line-clamp: 2;
		-webkit-box-orient: vertical;
	}
	.freq-suppressed-row {
		display: flex;
		flex-wrap: wrap;
		gap: 3px;
		margin-top: 4px;
		padding-top: 4px;
		border-top: 1px solid var(--border-subtle);
		align-items: center;
	}
	.freq-suppress-chip {
		display: inline-flex;
		align-items: center;
		gap: 3px;
		padding: 1px 6px;
		font-size: 9px;
		font-family: var(--font-mono, monospace);
		font-weight: 600;
		color: var(--text-secondary);
		background: rgba(255, 255, 255, 0.08);
		border: 1px solid var(--border-muted);
		border-radius: 3px;
		cursor: pointer;
		line-height: 1.4;
		transition: all 0.12s;
	}
	.freq-suppress-chip:hover {
		background: rgba(220, 80, 60, 0.25);
		border-color: rgba(220, 80, 60, 0.5);
	}
	.freq-suppress-chip svg {
		opacity: 0.4;
		flex-shrink: 0;
	}
	.freq-suppress-chip:hover svg {
		opacity: 1;
	}
	.freq-suppress-count {
		font-size: 8.5px;
		color: var(--text-muted);
		margin-left: 2px;
	}

	/* ── Info tabs ── */
	.info-tabs {
		display: flex;
		gap: 0;
		margin-top: 6px;
		border-bottom: 1px solid var(--border-subtle);
	}
	.info-tab {
		background: none;
		border: none;
		padding: 4px 10px;
		font-size: 9px;
		color: var(--text-muted);
		cursor: pointer;
		border-bottom: 2px solid transparent;
		margin-bottom: -1px;
		font-family: monospace;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		transition:
			color 0.15s,
			border-color 0.15s;
	}
	.info-tab:hover {
		color: var(--text-secondary);
	}
	.info-tab.active {
		color: var(--accent);
		border-bottom-color: var(--accent);
	}

	.species-waiting {
		height: 90px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 9px;
		color: var(--text-muted);
		background: var(--bg-subtle);
		border-radius: 4px;
	}

	/* ── Diversity tab ── */
	.diversity-grid {
		display: grid;
		grid-template-columns: repeat(2, 1fr);
		gap: 4px;
		margin-top: 6px;
	}
	.diversity-card {
		display: flex;
		align-items: center;
		gap: 5px;
		padding: 4px 6px;
		border-radius: 5px;
		background: var(--bg-subtle);
		cursor: default;
		transition: opacity 0.1s;
		font-size: 10px;
		font-family: monospace;
	}
	.diversity-card.dimmed {
		opacity: 0.35;
	}
	.diversity-card.highlighted {
		outline: 1px solid rgba(255, 255, 255, 0.3);
	}
	.diversity-swatch {
		width: 8px;
		height: 8px;
		border-radius: 2px;
		flex-shrink: 0;
	}
	.diversity-label {
		color: var(--text-muted);
		text-transform: uppercase;
		font-size: 8px;
		letter-spacing: 0.05em;
		min-width: 24px;
	}
	.diversity-value {
		color: var(--text-primary);
		font-weight: 600;
		margin-left: auto;
	}

	/* ── Species inspect modal ── */
	.species-modal-backdrop {
		position: fixed;
		inset: 0;
		z-index: 60;
		background: rgba(0, 0, 0, 0.5);
		display: flex;
		align-items: center;
		justify-content: center;
	}
	.species-modal {
		background: var(--bg-elevated);
		border: 1px solid var(--border-muted);
		border-radius: 10px;
		overflow: hidden;
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
		min-width: 200px;
	}
	.species-modal-header {
		padding: 8px 12px;
		font-size: 11px;
		font-family: monospace;
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
	}
	.species-modal-close {
		background: none;
		border: none;
		color: inherit;
		cursor: pointer;
		padding: 2px;
		opacity: 0.6;
	}
	.species-modal-close:hover {
		opacity: 1;
	}
	.species-modal-body {
		padding: 8px;
	}
	.species-modal-grid {
		margin: 0;
	}
	.species-modal-hex {
		margin: 0 auto;
	}
	.species-modal-hex .hex-byte-cell.modal-cell-hovered .hex-byte-inner {
		outline: 2px solid rgba(255, 255, 255, 0.8);
		z-index: 1;
	}
	.species-modal-grid .tip-cell {
		cursor: default;
		transition: outline 0.1s;
	}
	.species-modal-grid .tip-cell.modal-cell-hovered {
		outline: 2px solid rgba(255, 255, 255, 0.8);
		z-index: 1;
	}
	.species-modal-hash {
		text-align: center;
		font-size: 8px;
		font-family: monospace;
		color: var(--text-muted);
		margin-top: 6px;
		letter-spacing: 0.1em;
	}

	/* ── Tile tooltip ── */
	.tile-tip {
		position: fixed;
		z-index: 55;
		pointer-events: none;
		min-width: 140px;
		max-width: 180px;
		border-radius: 8px;
		overflow: hidden;
		filter: drop-shadow(0 4px 16px rgba(0, 0, 0, 0.6));
		font-family:
			system-ui,
			-apple-system,
			sans-serif;
		transform: translateY(-100%);
	}
	.tile-tip-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 6px 10px;
		gap: 8px;
	}
	.tile-tip-mnemonic {
		font-family: monospace;
		font-weight: 700;
		font-size: 13px;
		letter-spacing: 0.02em;
	}
	.tile-tip-hex {
		font-family: monospace;
		font-size: 11px;
		opacity: 0.7;
	}
	.tile-tip-body {
		background: var(--bg-elevated);
		padding: 8px 10px;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.tile-tip-desc {
		margin: 0;
		font-size: 11.5px;
		line-height: 1.4;
		color: var(--text-secondary, #ccc);
	}
	.tile-tip-status.suppressed {
		font-size: 10px;
		color: var(--accent-rose);
		font-weight: 600;
	}
	.tile-tip-hint {
		font-size: 9.5px;
		color: var(--text-subtle, #666);
		font-style: italic;
		margin-top: 2px;
	}

	/* ── Panels (shared) ── */
	.panel {
		position: fixed;
		z-index: 35;
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
		border-radius: 12px;
		backdrop-filter: blur(16px);
		box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
		padding: 10px;
	}
	.panel-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 8px;
	}
	.panel-title {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-subtle);
		font-weight: 600;
	}
	.panel-close {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 22px;
		height: 22px;
		border-radius: 6px;
		border: none;
		background: transparent;
		color: var(--text-subtle);
		cursor: pointer;
	}
	.panel-close:hover {
		background: var(--bg-hover);
		color: var(--text-secondary);
	}

	/* ── Settings panel ── */
	.settings-panel {
		top: 60px;
		right: 12px;
		width: 220px;
		max-height: calc(100dvh - 72px);
		overflow-y: auto;
	}
	.param {
		margin-bottom: 8px;
	}
	.param:last-child {
		margin-bottom: 0;
	}
	.param-head {
		display: flex;
		align-items: center;
		gap: 4px;
		margin-bottom: 3px;
	}
	.param-label {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--text-muted);
		display: inline-flex;
		align-items: center;
		gap: 4px;
	}
	.param-info-wrap {
		position: relative;
		display: inline-flex;
		flex-shrink: 0;
	}
	.param-info {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 14px;
		height: 14px;
		border-radius: 50%;
		background: var(--bg-muted);
		border: none;
		color: var(--text-subtle);
		font-size: 9px;
		font-family: Georgia, serif;
		font-style: italic;
		cursor: pointer;
		flex-shrink: 0;
		transition: background 0.15s;
	}
	.param-info:hover {
		background: var(--bg-hover);
		color: var(--text-secondary);
	}
	.param-tip {
		display: none;
		position: absolute;
		top: calc(100% + 6px);
		right: 0;
		width: 200px;
		padding: 8px 10px;
		background: var(--bg-elevated);
		border: 1px solid var(--border-muted);
		border-radius: 6px;
		font-size: 11px;
		font-style: normal;
		font-family: -apple-system, sans-serif;
		line-height: 1.4;
		color: var(--text-muted);
		z-index: 10;
	}
	.param-info-wrap.show-tip .param-tip {
		display: block;
	}
	.param-val {
		margin-left: auto;
		font-size: 11px;
		font-family: monospace;
		color: var(--accent);
		font-variant-numeric: tabular-nums;
	}

	/* Seed input */
	.seed-row {
		display: flex;
		gap: 6px;
	}
	.seed-input {
		flex: 1;
		min-width: 0;
		background: rgba(255, 255, 255, 0.03);
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-radius: 4px;
		padding: 0 7px;
		height: 22px;
		font-size: 11px;
		font-family: monospace;
		color: var(--text-primary);
		outline: none;
		transition: border-color 0.15s;
		-moz-appearance: textfield;
		appearance: textfield;
	}
	.seed-input::-webkit-inner-spin-button,
	.seed-input::-webkit-outer-spin-button {
		-webkit-appearance: none;
		appearance: none;
		margin: 0;
	}
	.seed-input:focus-visible {
		border-color: rgba(255, 255, 255, 0.15);
	}
	.seed-apply {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 22px;
		height: 22px;
		background: rgba(255, 255, 255, 0.04);
		border: 1px solid var(--border-muted);
		border-radius: 5px;
		color: var(--accent);
		cursor: pointer;
		transition: all 0.15s;
		flex-shrink: 0;
	}
	.seed-apply:hover {
		background: var(--bg-hover);
		border-color: var(--accent);
	}

	/* Grid type toggle — inline with param label */
	.grid-type-toggle {
		display: flex;
		gap: 2px;
		margin-left: 6px;
	}
	.grid-type-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 24px;
		height: 20px;
		background: rgba(255, 255, 255, 0.04);
		border: 1px solid var(--border-muted);
		border-radius: 4px;
		color: var(--text-subtle);
		cursor: pointer;
		transition: all 0.15s;
	}
	.grid-type-btn.active {
		background: rgba(200, 135, 90, 0.15);
		border-color: var(--accent);
		color: var(--accent);
	}
	.grid-type-btn:hover:not(.active) {
		background: var(--bg-hover);
	}
	.detail-toggle {
		display: flex;
		gap: 2px;
		margin-left: 6px;
	}
	.grid-size-row {
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.grid-size-row .seed-apply {
		margin-left: auto;
	}
	.grid-size-label {
		display: flex;
		align-items: center;
		gap: 3px;
		font-size: 10px;
		color: var(--text-subtle);
		text-transform: uppercase;
	}
	.grid-size-input {
		width: 48px;
		background: rgba(255, 255, 255, 0.03);
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-radius: 4px;
		padding: 0 6px;
		height: 20px;
		font-size: 11px;
		font-family: monospace;
		color: var(--text-primary);
		outline: none;
		transition: border-color 0.15s;
		-moz-appearance: textfield;
		appearance: textfield;
	}
	.grid-size-input::-webkit-inner-spin-button,
	.grid-size-input::-webkit-outer-spin-button {
		-webkit-appearance: none;
		appearance: none;
		margin: 0;
	}
	.grid-size-x {
		color: var(--text-subtle);
		font-size: 12px;
	}

	/* Custom slider */
	.slider-track-wrap {
		position: relative;
		padding: 2px 0;
	}
	.slider {
		width: 100%;
		height: 3px;
		-webkit-appearance: none;
		appearance: none;
		background: rgba(255, 255, 255, 0.12);
		border-radius: 2px;
		outline: none;
		cursor: pointer;
	}
	.slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 10px;
		height: 10px;
		border-radius: 50%;
		background: var(--accent);
		border: none;
		box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
		cursor: pointer;
		transition: transform 0.1s;
	}
	.slider::-webkit-slider-thumb:hover {
		transform: scale(1.15);
	}
	.slider::-moz-range-thumb {
		width: 10px;
		height: 10px;
		border-radius: 50%;
		background: var(--accent);
		border: none;
		box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
		cursor: pointer;
	}
	.slider::-moz-range-track {
		height: 3px;
		background: rgba(255, 255, 255, 0.12);
		border-radius: 2px;
		border: none;
	}

	/* Suppress opcodes */
	.suppress-clear {
		background: none;
		border: none;
		color: var(--text-muted);
		cursor: pointer;
		padding: 2px;
		opacity: 0.6;
		margin-left: auto;
	}
	.suppress-clear:hover {
		opacity: 1;
		color: var(--accent);
	}
	.suppress-input-wrap {
		position: relative;
	}
	.suppress-input {
		width: 100%;
		box-sizing: border-box;
		padding: 0 7px;
		height: 22px;
		background: rgba(255, 255, 255, 0.03);
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-radius: 4px;
		color: var(--text-primary);
		font-size: 10px;
		font-family: monospace;
		outline: none;
		transition: border-color 0.15s;
	}
	.suppress-input::placeholder {
		color: var(--text-subtle);
		font-size: 9.5px;
		opacity: 1;
	}
	.suppress-input:focus-visible {
		border-color: rgba(255, 255, 255, 0.15);
		outline: none;
	}
	.suppress-preview {
		display: block;
		font-size: 9.5px;
		color: var(--text-muted);
		margin-top: 3px;
		padding-left: 2px;
	}
	.suppress-chips {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin-top: 6px;
		align-items: center;
	}
	.suppress-chip {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		padding: 2px 6px;
		font-size: 9px;
		font-family: var(--font-mono, monospace);
		font-weight: 600;
		color: var(--text-primary);
		background: rgba(255, 255, 255, 0.1);
		border: 1px solid var(--border-muted);
		border-radius: 3px;
		cursor: pointer;
		transition: all 0.12s;
		line-height: 1.4;
	}
	.suppress-chip:hover {
		background: rgba(220, 80, 60, 0.25);
		border-color: rgba(220, 80, 60, 0.5);
	}
	.suppress-chip svg {
		opacity: 0.5;
		flex-shrink: 0;
	}
	.suppress-chip:hover svg {
		opacity: 1;
	}
	.suppress-count {
		font-size: 9px;
		color: var(--text-muted);
		margin-left: 2px;
	}

	/* Presets */
	.preset-header-actions {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-left: auto;
	}
	.preset-delete-active {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 18px;
		height: 18px;
		padding: 0;
		background: none;
		border: none;
		color: rgb(220, 90, 70);
		cursor: pointer;
		opacity: 0.85;
		transition: opacity 0.12s;
	}
	.preset-delete-active:hover {
		opacity: 1;
	}
	.preset-save-toggle {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 18px;
		height: 18px;
		padding: 0;
		background: none;
		border: none;
		border-radius: 4px;
		color: var(--text-muted);
		cursor: pointer;
		transition: all 0.12s;
	}
	.preset-save-toggle:hover {
		color: var(--accent);
		background: var(--bg-hover);
	}
	.preset-save-toggle.active {
		color: var(--accent);
		transform: rotate(45deg);
	}
	.preset-save-row {
		display: flex;
		gap: 4px;
		margin-top: 6px;
	}
	.preset-name-input {
		flex: 1;
		min-width: 0;
		box-sizing: border-box;
		padding: 0 7px;
		height: 22px;
		background: rgba(255, 255, 255, 0.03);
		border: 1px solid rgba(255, 255, 255, 0.1);
		border-radius: 4px;
		color: var(--text-primary);
		font-size: 10px;
		outline: none;
		transition: border-color 0.15s;
	}
	.preset-name-input::placeholder {
		color: var(--text-subtle);
		font-size: 9.5px;
		opacity: 1;
	}
	.preset-name-input:focus-visible {
		border-color: var(--accent);
	}
	.preset-name-save {
		padding: 0 10px;
		height: 22px;
		font-size: 10px;
		color: var(--accent);
		background: rgba(255, 255, 255, 0.04);
		border: 1px solid var(--border-muted);
		border-radius: 4px;
		cursor: pointer;
		transition: all 0.12s;
	}
	.preset-name-save:hover:not(:disabled) {
		background: var(--bg-hover);
		border-color: var(--accent);
	}
	.preset-name-save:disabled {
		opacity: 0.4;
		cursor: default;
	}
	.preset-chips {
		display: flex;
		flex-wrap: wrap;
		gap: 4px;
		margin-top: 6px;
	}
	.preset-chip {
		display: inline-flex;
		align-items: center;
		gap: 3px;
		padding: 3px 8px;
		font-size: 9.5px;
		font-weight: 600;
		color: var(--text-secondary);
		background: rgba(255, 255, 255, 0.05);
		border: 1px solid var(--border-subtle);
		border-radius: 4px;
		cursor: pointer;
		transition: all 0.12s;
		line-height: 1.4;
		white-space: nowrap;
	}
	.preset-chip:hover {
		background: var(--bg-hover);
		border-color: var(--border-muted);
		color: var(--text-primary);
	}
	.preset-chip.builtin {
		font-style: italic;
	}
	.preset-chip.active {
		color: var(--accent);
		background: color-mix(in srgb, var(--accent) 14%, transparent);
		border-color: var(--accent);
	}
	.preset-chip-check {
		flex-shrink: 0;
	}

	/* Preset overwrite confirmation dialog */
	.confirm-backdrop {
		position: fixed;
		inset: 0;
		z-index: 60;
		display: flex;
		align-items: center;
		justify-content: center;
		background: rgba(0, 0, 0, 0.5);
		backdrop-filter: blur(2px);
	}
	.confirm-dialog {
		width: min(320px, calc(100vw - 32px));
		padding: 16px;
		background: var(--bg-elevated, var(--bg-panel));
		border: 1px solid var(--border-muted);
		border-radius: 10px;
		box-shadow: 0 8px 40px rgba(0, 0, 0, 0.5);
	}
	.confirm-title {
		font-size: 13px;
		font-weight: 600;
		color: var(--text-primary);
		margin-bottom: 8px;
	}
	.confirm-body {
		font-size: 11.5px;
		line-height: 1.5;
		color: var(--text-secondary);
		margin-bottom: 16px;
	}
	.confirm-body strong {
		color: var(--text-primary);
	}
	.confirm-actions {
		display: flex;
		justify-content: flex-end;
		gap: 8px;
	}
	.confirm-btn {
		padding: 5px 14px;
		font-size: 11px;
		border-radius: 6px;
		cursor: pointer;
		transition: all 0.12s;
	}
	.confirm-cancel {
		color: var(--text-secondary);
		background: transparent;
		border: 1px solid var(--border-muted);
	}
	.confirm-cancel:hover {
		background: var(--bg-hover);
		color: var(--text-primary);
	}
	.confirm-ok {
		color: #fff;
		background: rgb(200, 70, 55);
		border: 1px solid rgb(200, 70, 55);
	}
	.confirm-ok:hover {
		background: rgb(215, 80, 65);
	}

	/* Colormap selector */
	.cmap-row {
		display: flex;
		gap: 4px;
	}
	.cmap-btn {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 2px;
		padding: 3px 0 2px;
		font-size: 8.5px;
		font-family: monospace;
		text-transform: capitalize;
		color: var(--text-muted);
		background: rgba(255, 255, 255, 0.04);
		border: 1px solid var(--border-subtle);
		border-radius: 4px;
		cursor: pointer;
		transition: all 0.15s;
	}
	.cmap-preview {
		width: 80%;
		height: 4px;
		border-radius: 2px;
	}
	.cmap-preview-rainbow {
		background: linear-gradient(
			90deg,
			#1a0404,
			#c85a5a,
			#c8a05a,
			#5ac85a,
			#5a8ac8,
			#8a5ac8,
			#c85a8a
		);
	}
	.cmap-preview-ocean {
		background: linear-gradient(
			90deg,
			#0a1218,
			#2a4a5a,
			#3a7a8a,
			#4a9aaa,
			#5ab0c0,
			#6ac0d0,
			#8ad0e0
		);
	}
	.cmap-preview-thermal {
		background: linear-gradient(
			90deg,
			#000000,
			#4a0a2a,
			#8a1a1a,
			#c84a0a,
			#e88a2a,
			#f0c050,
			#f8f0c0
		);
	}
	.cmap-btn:hover {
		background: var(--bg-hover);
		color: var(--text-secondary);
	}
	.cmap-btn.active {
		background: rgba(200, 135, 90, 0.12);
		border-color: var(--accent);
		color: var(--accent);
	}

	/* Color adjustments */
	.color-adj-toggle {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 20px;
		height: 20px;
		margin-left: auto;
		background: none;
		border: 1px solid var(--border-muted);
		border-radius: 4px;
		color: var(--text-subtle);
		cursor: pointer;
		transition: all 0.15s;
	}
	.color-adj-toggle:hover,
	.color-adj-toggle.active {
		color: var(--accent);
		border-color: var(--accent);
	}
	.color-adj-panel {
		margin-top: 6px;
		padding: 8px;
		background: rgba(255, 255, 255, 0.03);
		border: 1px solid rgba(255, 255, 255, 0.06);
		border-radius: 6px;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.color-adj-row {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.color-adj-label {
		font-size: 9.5px;
		color: var(--text-muted);
		width: 58px;
		flex-shrink: 0;
	}
	.color-adj-slider {
		flex: 1;
		min-width: 0;
	}
	.color-adj-reset {
		align-self: flex-end;
		padding: 2px 8px;
		font-size: 9px;
		color: var(--text-subtle);
		background: none;
		border: 1px solid var(--border-muted);
		border-radius: 4px;
		cursor: pointer;
		transition: all 0.15s;
	}
	.color-adj-reset:hover {
		color: var(--text-secondary);
		border-color: var(--text-subtle);
	}

	.grid-lines-toggle {
		display: flex;
		align-items: center;
		gap: 6px;
		margin-top: 4px;
		font-size: 10px;
		color: var(--text-muted);
		cursor: pointer;
		user-select: none;
	}
	.grid-lines-toggle .sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		border: 0;
	}
	.grid-lines-toggle .check-box {
		width: 12px;
		height: 12px;
		border-radius: 3px;
		border: 1px solid var(--border-default);
		background: rgba(255, 255, 255, 0.03);
		display: flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		transition:
			border-color 0.15s,
			background 0.15s;
	}
	.grid-lines-toggle .check-box.checked {
		border-color: var(--accent);
		background: rgba(200, 135, 90, 0.15);
	}
	.grid-lines-toggle:hover .check-box {
		border-color: var(--text-subtle);
	}
	.grid-lines-toggle:hover .check-box.checked {
		border-color: var(--accent-warm);
	}

	/* ── Genome tooltip ── */
	.genome-tip {
		position: fixed;
		z-index: 45;
		pointer-events: none;
		overflow: hidden;
		background: #0a0a0e;
		padding: 4px;
		border-radius: 3px;
		filter: drop-shadow(0 0 12px rgba(0, 0, 0, 0.6));
	}
	.tip-grid {
		display: grid;
		grid-template-columns: repeat(4, 46px);
		gap: 2px;
	}
	.tip-cell {
		width: 46px;
		height: 46px;
		display: flex;
		align-items: center;
		justify-content: center;
		overflow: hidden;
		font-size: 8.5px;
		font-family: monospace;
		text-align: center;
		line-height: 1.15;
		font-weight: 600;
		padding: 2px;
	}
	.tip-cell.operand {
		opacity: 0.75;
		font-weight: 400;
		font-size: 10px;
	}

	/* ── Hex byte tooltip: 19 hex cells in 3-4-5-4-3 hexagonal arrangement ── */
	.hex-tip {
		background: none;
		padding: 2px !important;
		overflow: visible;
		filter: drop-shadow(0 0 12px rgba(0, 0, 0, 0.6));
	}
	.hex-byte-cell::before {
		content: '';
		position: absolute;
		inset: -4px;
		background: #0a0a0e;
		clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
		z-index: -1;
	}
	.hex-byte-grid {
		--hex-w: 48px;
		--hex-gap: 2px;
		--hex-dx: calc(var(--hex-w) + var(--hex-gap));
		--hex-dy: calc(var(--hex-dx) * 0.866);
		--hex-h: calc(var(--hex-w) * 1.1547);
		display: block;
		position: relative;
		width: calc(var(--hex-dx) * 4 + var(--hex-w));
		height: calc(var(--hex-dy) * 4 + var(--hex-h));
	}
	.hex-byte-cell {
		position: absolute;
		width: var(--hex-w);
		height: var(--hex-h);
	}
	.hex-byte-inner {
		width: 100%;
		height: 100%;
		clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
		display: flex;
		align-items: center;
		justify-content: center;
		overflow: hidden;
		font-size: 8px;
		font-family: monospace;
		font-weight: 600;
		text-align: center;
		line-height: 1.1;
		padding: 2px;
	}
	.hex-byte-cell.operand .hex-byte-inner {
		opacity: 0.75;
		font-weight: 400;
		font-size: 9px;
	}
	/* Byte positions in 3-4-5-4-3 hexagonal arrangement */
	/* Byte 0: center (row 2, col 2) */
	.hex-byte-pos-0 {
		left: calc(var(--hex-dx) * 2);
		top: calc(var(--hex-dy) * 2);
	}
	/* Ring 1: bytes 1-6 (clockwise from N) */
	.hex-byte-pos-1 {
		left: calc(var(--hex-dx) * 2.5);
		top: calc(var(--hex-dy) * 1);
	}
	.hex-byte-pos-2 {
		left: calc(var(--hex-dx) * 3);
		top: calc(var(--hex-dy) * 2);
	}
	.hex-byte-pos-3 {
		left: calc(var(--hex-dx) * 2.5);
		top: calc(var(--hex-dy) * 3);
	}
	.hex-byte-pos-4 {
		left: calc(var(--hex-dx) * 1.5);
		top: calc(var(--hex-dy) * 3);
	}
	.hex-byte-pos-5 {
		left: calc(var(--hex-dx) * 1);
		top: calc(var(--hex-dy) * 2);
	}
	.hex-byte-pos-6 {
		left: calc(var(--hex-dx) * 1.5);
		top: calc(var(--hex-dy) * 1);
	}
	/* Ring 2: bytes 7-18 (clockwise from NE) */
	.hex-byte-pos-7 {
		left: calc(var(--hex-dx) * 3.5);
		top: calc(var(--hex-dy) * 1);
	}
	.hex-byte-pos-8 {
		left: calc(var(--hex-dx) * 4);
		top: calc(var(--hex-dy) * 2);
	}
	.hex-byte-pos-9 {
		left: calc(var(--hex-dx) * 3.5);
		top: calc(var(--hex-dy) * 3);
	}
	.hex-byte-pos-10 {
		left: calc(var(--hex-dx) * 3);
		top: calc(var(--hex-dy) * 4);
	}
	.hex-byte-pos-11 {
		left: calc(var(--hex-dx) * 2);
		top: calc(var(--hex-dy) * 4);
	}
	.hex-byte-pos-12 {
		left: calc(var(--hex-dx) * 1);
		top: calc(var(--hex-dy) * 4);
	}
	.hex-byte-pos-13 {
		left: calc(var(--hex-dx) * 0.5);
		top: calc(var(--hex-dy) * 3);
	}
	.hex-byte-pos-14 {
		left: calc(var(--hex-dx) * 0);
		top: calc(var(--hex-dy) * 2);
	}
	.hex-byte-pos-15 {
		left: calc(var(--hex-dx) * 0.5);
		top: calc(var(--hex-dy) * 1);
	}
	.hex-byte-pos-16 {
		left: calc(var(--hex-dx) * 1);
		top: calc(var(--hex-dy) * 0);
	}
	.hex-byte-pos-17 {
		left: calc(var(--hex-dx) * 2);
		top: calc(var(--hex-dy) * 0);
	}
	.hex-byte-pos-18 {
		left: calc(var(--hex-dx) * 3);
		top: calc(var(--hex-dy) * 0);
	}

	/* ── Help modal ── */
	.modal-backdrop {
		position: fixed;
		inset: 0;
		z-index: 50;
		display: flex;
		align-items: center;
		justify-content: center;
		background: rgba(0, 0, 0, 0.4);
	}
	.modal {
		background: var(--bg-elevated);
		border: 1px solid var(--border-muted);
		border-radius: 14px;
		max-width: 480px;
		width: calc(100% - 32px);
		overflow: hidden;
	}
	.modal-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 6px 10px 0 10px;
		border-bottom: 1px solid var(--border-subtle);
	}
	.modal-tabs {
		display: flex;
		gap: 0;
		overflow-x: auto;
		min-width: 0;
	}
	.modal-tab {
		background: none;
		border: none;
		padding: 7px 8px;
		font-size: 10.5px;
		color: var(--text-muted);
		cursor: pointer;
		border-bottom: 2px solid transparent;
		margin-bottom: -1px;
		display: flex;
		align-items: center;
		gap: 3px;
		transition:
			color 0.15s,
			border-color 0.15s;
		white-space: nowrap;
	}
	.modal-tab svg {
		width: 12px;
		height: 12px;
		flex-shrink: 0;
	}
	.modal-tab:hover {
		color: var(--text-secondary);
	}
	.modal-tab.active {
		color: var(--accent);
		border-bottom-color: var(--accent);
	}
	.modal-body {
		padding: 16px;
		font-size: 13px;
		line-height: 1.6;
		color: var(--text-muted);
		max-height: 65vh;
		overflow-y: auto;
	}
	.modal-body h4 {
		color: var(--text-secondary);
		font-size: 13px;
		font-weight: 600;
		margin: 14px 0 4px;
	}
	.modal-body h4:first-of-type {
		margin-top: 2px;
	}
	.modal-body p {
		margin-bottom: 10px;
	}
	.cmap-section {
		margin-bottom: 10px;
	}
	.cmap-label {
		font-size: 10px;
		font-weight: 600;
		color: var(--text-subtle);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 4px;
	}
	.cmap-grid {
		display: grid;
		grid-template-columns: 1fr 1fr 1fr;
		gap: 1px 8px;
	}
	.cmap-entry {
		display: flex;
		align-items: center;
		gap: 5px;
		font-size: 11px;
		line-height: 1.8;
	}
	.help-swatch {
		width: 12px;
		height: 12px;
		border-radius: 2px;
		flex-shrink: 0;
		border: 1px solid rgba(255, 255, 255, 0.08);
	}
	.cmap-hex {
		font-family: monospace;
		font-size: 10px;
		color: var(--text-subtle);
		min-width: 22px;
	}
	.cmap-note {
		font-size: 11px;
		color: var(--text-subtle);
		margin: 8px 0 12px;
	}
	.help-steps {
		margin: 8px 0;
		padding-left: 20px;
	}
	.help-steps li {
		margin-bottom: 6px;
		line-height: 1.45;
	}
	.tape-layouts {
		display: flex;
		gap: 16px;
		margin: 10px 0;
		flex-wrap: wrap;
		align-items: flex-start;
		justify-content: center;
	}
	.tape-layout-item {
		flex: 1;
		min-width: 140px;
		max-width: 200px;
		text-align: center;
	}
	.tape-layout-label {
		font-size: 11px;
		font-weight: 600;
		color: var(--text-secondary);
		margin-bottom: 6px;
	}
	.tape-layout-note {
		font-size: 10px;
		color: var(--text-muted);
		margin-top: 4px;
	}
	.tape-svg {
		width: 100%;
		max-width: 110px;
		display: block;
		margin: 0 auto;
	}
	.tape-combine-svg {
		width: 100%;
		margin: 8px 0;
		display: block;
		border-radius: 8px;
		background: rgba(0, 0, 0, 0.2);
		border: 1px solid rgba(255, 255, 255, 0.04);
		padding: 8px;
	}
	.help-link {
		color: var(--accent);
		text-decoration: none;
	}
	.help-link:hover {
		text-decoration: underline;
	}
	/* ── Analysis help tab ── */
	.help-metric-list {
		display: flex;
		flex-direction: column;
		gap: 6px;
		margin: 8px 0;
	}
	.help-metric-item {
		display: flex;
		align-items: baseline;
		gap: 6px;
		font-size: 12px;
		line-height: 1.4;
	}
	.help-metric-swatch {
		width: 8px;
		height: 8px;
		border-radius: 2px;
		flex-shrink: 0;
		margin-top: 5px;
	}
	.help-metric-eq {
		margin: 4px 0 2px;
	}
	.help-metric-eq :global(.katex) {
		font-size: 1em;
	}
	/* ── Z80 reference tab ── */
	.z80-reg-table {
		margin-bottom: 10px;
	}
	.z80-reg-row {
		display: grid;
		grid-template-columns: 80px 48px 1fr;
		gap: 6px;
		font-size: 11px;
		line-height: 1.9;
		color: var(--text-muted);
		border-bottom: 1px solid rgba(255, 255, 255, 0.03);
	}
	.z80-reg-row.head {
		font-weight: 600;
		color: var(--text-subtle);
		text-transform: uppercase;
		font-size: 10px;
		letter-spacing: 0.5px;
		border-bottom: 1px solid var(--border-subtle);
	}
	.z80-reg {
		font-family: monospace;
		color: var(--accent);
		font-weight: 600;
	}
	.z80-notation-table {
		margin-bottom: 10px;
	}
	.z80-notation-row {
		display: grid;
		grid-template-columns: 56px 1fr;
		gap: 8px;
		font-size: 11px;
		line-height: 1.9;
		color: var(--text-muted);
		border-bottom: 1px solid rgba(255, 255, 255, 0.03);
	}
	.z80-notation-row.head {
		font-weight: 600;
		color: var(--text-subtle);
		text-transform: uppercase;
		font-size: 10px;
		letter-spacing: 0.5px;
		border-bottom: 1px solid var(--border-subtle);
	}
	.z80-cat {
		margin-bottom: 10px;
	}
	.z80-cat-label {
		font-size: 10px;
		font-weight: 600;
		color: var(--text-subtle);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 3px;
	}
	.z80-instr-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1px 12px;
	}
	.z80-instr {
		font-size: 11px;
		line-height: 1.8;
		color: var(--text-muted);
	}
	.z80-op {
		font-family: monospace;
		color: var(--accent);
		font-weight: 600;
		font-size: 11px;
	}

	.modal-shortcuts {
		margin-top: 14px;
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 4px 16px;
	}
	.shortcut {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 11px;
		color: var(--text-muted);
	}
	.shortcut kbd {
		display: inline-block;
		padding: 2px 6px;
		background: var(--bg-muted);
		border: 1px solid var(--border-muted);
		border-radius: 4px;
		font-size: 10px;
		font-family: monospace;
		color: var(--text-secondary);
		min-width: 20px;
		text-align: center;
	}

	/* ── Mobile ── */
	@media (max-width: 768px) {
		/* Toolbar: horizontal pill at bottom center */
		.toolbar {
			top: auto;
			bottom: 8px;
			right: auto;
			left: 50%;
			transform: translateX(-50%);
			flex-direction: row;
			width: auto;
		}
		.toolbar.collapsed {
			left: auto;
			right: 8px;
			transform: none;
		}
		.toolbar-buttons {
			flex-direction: row;
			max-width: 400px;
		}
		.toolbar-buttons.hidden {
			max-height: unset;
			max-width: 0;
			opacity: 0;
			overflow: hidden;
			pointer-events: none;
		}
		.tb-sep {
			width: 1px;
			height: 14px;
			margin: 0 1px;
		}
		.speed-menu {
			top: auto;
			bottom: calc(100% + 6px);
			left: 50%;
			right: auto;
			transform: translateX(-50%);
		}

		/* Info bar: compact at top-left, scaled down */
		.info-bar {
			bottom: auto;
			top: 8px;
			left: 8px;
			width: auto;
			min-width: 0;
			max-width: min(210px, calc(100vw - 16px));
			padding: 6px 8px;
			font-size: 10px;
			border-radius: 8px;
		}
		.info-row {
			gap: 5px;
		}
		.info-label {
			font-size: 8px;
		}
		.info-value {
			font-size: 10px;
		}
		.info-chart-svg {
			height: 68px;
		}
		.freq-grid {
			grid-template-columns: repeat(5, 34px);
			gap: 3px;
		}
		.freq-cell {
			width: 34px;
			height: 34px;
			font-size: 8px;
			border-radius: 4px;
			padding: 1px;
		}
		.freq-rank {
			font-size: 5px;
			top: 1px;
			left: 2px;
		}

		/* Settings panel: compact, anchored bottom-right above toolbar */
		.settings-panel {
			top: auto;
			bottom: 52px;
			right: 8px;
			left: auto;
			width: 286px;
			max-height: calc(100vh - 120px);
			overflow-y: auto;
			font-size: 10px;
		}
		/* Bigger slider thumbs for touch (track stays thin) */
		.slider::-webkit-slider-thumb {
			width: 22px;
			height: 22px;
		}
		.slider::-moz-range-thumb {
			width: 22px;
			height: 22px;
		}
		.slider-track-wrap {
			padding: 6px 0;
		}

		/* Help modal: full-screen on mobile */
		.modal {
			max-width: 100%;
			width: 100%;
			height: 100%;
			max-height: 100%;
			border-radius: 0;
			display: flex;
			flex-direction: column;
		}
		.modal-header {
			padding: 4px 8px 0 8px;
			flex-shrink: 0;
		}
		.modal-tabs {
			flex-wrap: nowrap;
			overflow-x: auto;
			-webkit-overflow-scrolling: touch;
			scrollbar-width: none;
			gap: 0;
		}
		.modal-tabs::-webkit-scrollbar {
			display: none;
		}
		.modal-tab {
			padding: 6px 8px;
			font-size: 10px;
			white-space: nowrap;
			flex-shrink: 0;
		}
		.modal-tab svg {
			display: none;
		}
		.modal-body {
			max-height: none;
			flex: 1;
			overflow-y: auto;
			padding: 12px;
			font-size: 12px;
		}
		.modal-backdrop {
			align-items: stretch;
			justify-content: stretch;
		}

		/* Hide keys tab on mobile (no keyboard) */
		.tab-keys {
			display: none;
		}

		/* Prevent iOS zoom on input focus — font-size must be ≥ 16px */
		.seed-input,
		.suppress-input {
			font-size: 16px;
			height: 28px;
		}
		.grid-size-input {
			font-size: 16px;
			height: 26px;
			width: 54px;
		}

		/* Genome tooltip: smaller on mobile */
		.genome-tip {
			transform: scale(0.8);
			transform-origin: top center;
		}
	}
	.instrument-select {
		width: 100%;
		min-height: 26px;
		border: 1px solid var(--border-muted);
		border-radius: 4px;
		padding: 3px 6px;
		background: #1b1b1b;
		background-image: none;
		color: var(--text-secondary);
		font-size: 11px;
		line-height: 1.35;
		appearance: auto;
		color-scheme: dark;
		box-shadow: none;
	}
	.instrument-select:disabled {
		opacity: 0.55;
	}
	.paper-preset-label,
	.world-control-label {
		display: block;
		font-size: 10px;
		color: var(--text-subtle);
		margin: 5px 0 3px;
	}
	.paper-condition {
		margin-top: 6px;
		font-size: 10px;
		line-height: 1.45;
		color: var(--text-muted);
	}
	.paper-condition p {
		margin: 0 0 6px;
	}
	.paper-condition strong {
		color: var(--text-secondary);
		font-weight: 500;
	}
	.paper-condition summary,
	.culture-example summary {
		cursor: pointer;
		color: var(--accent);
		font-size: 10px;
	}
	.paper-condition details p:first-of-type {
		margin-top: 6px;
	}
	.source-path {
		overflow-wrap: anywhere;
		font-family: monospace;
		font-size: 9px;
	}
	.world-geometry-row {
		display: grid;
		grid-template-columns: auto 1fr auto 1fr;
		gap: 5px;
		align-items: center;
		margin-top: 6px;
	}
	.world-geometry-row label {
		font-size: 10px;
		color: var(--text-muted);
	}
	.world-geometry-row input {
		width: 100%;
	}
	.instrument-check {
		display: flex;
		gap: 6px;
		align-items: center;
		margin-top: 6px;
		font-size: 10px;
		color: var(--text-muted);
		cursor: pointer;
	}
	.instrument-check input {
		width: 11px;
		height: 11px;
		color: var(--accent);
		background: var(--bg-muted);
		border: 1px solid var(--border-default);
		border-radius: 2px;
	}
	.chemistry-head {
		margin-top: 8px;
	}
	.instrument-note {
		color: var(--text-subtle);
		font-size: 10px;
		line-height: 1.45;
		margin-top: 5px;
	}
	.instrument-error {
		color: var(--accent-rose);
		font-size: 10px;
		overflow-wrap: anywhere;
		margin-top: 5px;
	}
	.fetch-key {
		display: inline-block;
		width: 6px;
		height: 6px;
		border-radius: 2px;
		margin-right: 2px;
	}
	.fetch-key.confined {
		background: var(--accent-emerald);
	}
	.fetch-key.open {
		background: var(--accent-rose);
	}
	.fetch-key.unknown {
		background: #555;
	}
	.tape-inspector,
	.species-modal {
		max-width: calc(100vw - 28px);
		max-height: 85dvh;
		overflow-y: auto;
	}
	.tape-inspector .species-modal-header {
		color: var(--text-secondary);
		border-bottom: 1px solid var(--border-subtle);
	}
	.culture-panel {
		margin-top: 8px;
		padding-top: 8px;
		border-top: 1px solid var(--border-subtle);
		max-width: 340px;
	}
	.culture-button {
		margin: 0 0 3px;
	}
	.culture-panel > .instrument-note {
		display: block;
	}
	.culture-verdict {
		font-size: 11px;
		color: var(--accent);
		margin-top: 8px;
	}
	.culture-table {
		width: 100%;
		font-size: 10px;
		color: var(--text-muted);
		font-variant-numeric: tabular-nums;
		border-collapse: collapse;
	}
	.culture-table th {
		font-weight: 500;
		color: var(--text-subtle);
	}
	.culture-table th,
	.culture-table td {
		padding: 3px 5px;
		text-align: right;
		border-bottom: 1px solid var(--border-subtle);
	}
	.culture-table th:first-child,
	.culture-table td:first-child {
		text-align: left;
	}
	.culture-example {
		margin-top: 8px;
	}
	.culture-bytes {
		display: grid;
		grid-template-columns: repeat(8, 1fr);
		gap: 2px;
		margin-top: 3px;
	}
	.culture-byte {
		padding: 4px;
		font: 9px monospace;
		text-align: center;
		color: #888;
		background: #292929;
		border: 1px solid transparent;
	}
	.culture-byte.fetched {
		background: var(--byte-color);
		border-color: var(--border-muted);
		color: #222;
	}
	.culture-byte.instruction-start {
		text-decoration: underline;
		text-underline-offset: 3px;
	}
	.guide-steps {
		list-style: decimal;
		padding-left: 20px;
		margin: 0 0 12px;
	}
	.guide-steps li {
		margin-bottom: 6px;
	}
	.inspector-disassembly {
		max-width: 340px;
		margin-top: 7px;
	}
	.inspector-disassembly summary {
		color: var(--accent);
		font-size: 10px;
		cursor: pointer;
	}
	.disassembly-line {
		display: grid;
		grid-template-columns: 18px 66px minmax(0, 1fr);
		gap: 5px;
		padding-top: 4px;
		font: 9px/1.4 monospace;
		color: var(--text-muted);
		overflow-wrap: anywhere;
	}
	.world-contract {
		margin-top: 6px;
	}
	.world-contract summary {
		color: var(--text-muted);
		font-size: 10px;
		cursor: pointer;
	}
	.contract-shape {
		color: var(--text-subtle);
		font: 9px monospace;
		margin-left: 5px;
	}
</style>

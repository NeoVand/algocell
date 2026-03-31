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
	import { disassemble, byteToMnemonic } from '$lib/z80-disasm';
	import { getCellData } from '$lib/sim/soup';
	import { unpackRGBA, createColormap, COLORMAP_NAMES } from '$lib/colormap';
	import type { ColormapName } from '$lib/colormap';
	import { untrack } from 'svelte';
	import Mermaid from '$lib/components/Mermaid.svelte';

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
	let engine = $state<GPUEngine | null>(null);
	let gpuError = $state<string | null>(null);

	// Controls
	let seed = $state(DEFAULT_SEED);
	let noiseExp = $state(DEFAULT_NOISE_EXP);
	let pairCount = $state(MAX_BATCH_PAIR_N);
	let z80Steps = $state(Z80_STEPS);
	let playing = $state(true);
	let speed = $state(1);
	let gridType = $state<GridType>('square');

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
	let helpTab = $state<'overview' | 'visuals' | 'z80' | 'params' | 'keys' | 'about'>('overview');
	let showSpeedMenu = $state(false);
	let showSettings = $state(false);
	let toolbarCollapsed = $state(false);
	let openTip = $state<string | null>(null);
	let showInfoChart = $state(true);
	let suppressPatterns: string[] = $state([]);
	let suppressInput = $state('');

	// Derive the actual set of suppressed opcodes from substring patterns
	/* eslint-disable svelte/prefer-svelte-reactivity -- derived recreates the Set each time */
	let suppressedOpcodes = $derived.by(() => {
		const set = new Set<number>();
		if (suppressPatterns.length === 0) return set;
		for (let i = 0; i < 256; i++) {
			const mnemonic = (byteToMnemonic(i) || '').toUpperCase();
			const hex = i.toString(16).toUpperCase().padStart(2, '0');
			for (const pat of suppressPatterns) {
				const p = pat.toUpperCase();
				if (mnemonic.includes(p) || hex.includes(p) || ('0X' + hex).includes(p)) {
					set.add(i);
					break;
				}
			}
		}
		return set;
	});
	/* eslint-enable svelte/prefer-svelte-reactivity */

	// Frequency chart: track top N bytes normalized over time
	// Total bytes counted by shader = cells * wordsPerCell * 4 (word-aligned)
	let TOTAL_BYTES = $derived(
		gridWidth * gridHeight * Math.ceil((gridType === 'hex' ? 19 : 16) / 4) * 4
	);
	const MAX_TRACKED = 10;
	const MAX_TRACKED_RAW = 30; // fetch more so we can fill 10 slots after filtering suppressed
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

	// Hover / genome tooltip
	let hoveredCell = $state(-1);
	let cellData = $state<Uint8Array | null>(null);
	let disasmLines: ReturnType<typeof disassemble> = $state([]);
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
		const isHex = gridType === 'hex';
		const tooltipW = isHex ? 280 : 196;
		const tooltipH = isHex ? 260 : 196;
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

	function buildSparklinePathTyped(values: Float64Array, w: number, h: number, max: number): string {
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

	$effect(() => {
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
			gridType: gridType
		}));
		// Set canvas pixel dimensions before init so resetView gets correct aspect
		const dpr = window.devicePixelRatio || 1;
		canvas.width = canvasW = Math.round(canvas.clientWidth * dpr);
		canvas.height = canvasH = Math.round(canvas.clientHeight * dpr);

		const eng = new GPUEngine(initialSeed, initialConfig);

		eng.init(canvas).then((ok) => {
			if (!ok) {
				gpuError = 'WebGPU not available. Please use Chrome 113+ or Edge 113+.';
				return;
			}
			engine = eng;
			startLoop();
		});

		return () => {
			if (animFrameId !== undefined) {
				cancelAnimationFrame(animFrameId);
				animFrameId = undefined;
			}
			eng.destroy();
		};
	});

	// ResizeObserver for canvas
	$effect(() => {
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

			if (playing) {
				for (let i = 0; i < speed; i++) {
					engine.simulateStep();
				}
			}

			engine.render(canvas);
			flushCellRefresh();
			frameCount++;

			// Update stats: immediately on first frame, then every 15 frames
			if (playing && (frameCount === 1 || frameCount % 15 === 0) && !statsLoading) {
				opsPerSec = engine.opsPerSec;
				updateTopBytes();
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

			// Update tracked bytes — fetch extra so we can fill slots after collapsing suppressed
			const topN = entries.slice(0, MAX_TRACKED_RAW);
			trackedBytes = topN.map((e) => ({ byte: e.byte, mnemonic: e.mnemonic }));

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

	function handleCanvasMouseMove(e: MouseEvent) {
		if (touchHandled) return; // ignore synthetic mouse events from touch
		if (!engine || !canvas) return;
		const rect = canvas.getBoundingClientRect();
		const sx = e.clientX - rect.left;
		const sy = e.clientY - rect.top;
		mouseX = e.clientX;
		mouseY = e.clientY;

		if (isPanning) {
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

	function refreshCellData(cell: number) {
		// Mark for readback after next render (so data matches what's on screen)
		pendingCellRefresh = cell;
	}

	function flushCellRefresh() {
		if (pendingCellRefresh < 0 || !engine) return;
		const cell = pendingCellRefresh;
		pendingCellRefresh = -1;
		engine.readSoupData().then((soupData) => {
			if (soupData.length === 0 || hoveredCell !== cell) return;
			const data = getCellData(soupData, cell, gridType);
			cellData = data;
			disasmLines = disassemble(data);
		});
	}

	// Track if touch just handled an interaction, to suppress synthetic mouse events
	let touchHandled = false;

	function handleCanvasMouseDown(e: MouseEvent) {
		if (touchHandled) return; // ignore synthetic mouse events from touch
		if (e.button === 0 || e.button === 1) {
			e.preventDefault();
			isPanning = true;
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
					// Tap on a cell — show tooltip
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
		engine?.resetView(canvasW, canvasH);
	}

	function handleReset() {
		if (!engine) return;
		engine.reset(seed);
		opsPerSec = 0;
		chartRing.length = 0;
		chartLen = 0;
		chartVersion++;
		trackedBytes = [];
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
		// Recompute grid dimensions for the new grid type
		const dims = computeGridDimensions(window.innerWidth, window.innerHeight, type);
		gridWidth = dims.w;
		gridHeight = dims.h;
		applyGridConfig();
	}

	function applyGridConfig() {
		if (!engine) return;
		const config: GridConfig = { width: gridWidth, height: gridHeight, gridType: gridType };
		engine.changeGridConfig(config, canvasW, canvasH);
		// Re-apply suppress mask after buffer recreation
		engine.setSuppressedOpcodes(suppressedOpcodes);
		// Reset stats
		opsPerSec = 0;
		chartRing.length = 0;
		chartLen = 0;
		chartVersion++;
		trackedBytes = [];
	}

	function addSuppressPattern(pattern: string) {
		// Support multiple patterns separated by semicolons
		// (commas are NOT delimiters because Z80 mnemonics contain commas, e.g. LD A,B)
		const parts = pattern.split(';').map((s) => s.trim().toUpperCase()).filter(Boolean);
		let added = false;
		for (const p of parts) {
			if (!p || suppressPatterns.includes(p)) continue;
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
		const mnemonic = byteToMnemonic(opcode) || hexByte(opcode);
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

	// Sync derived suppressedOpcodes to the GPU engine
	$effect(() => {
		if (!engine) return;
		engine.setSuppressedOpcodes(suppressedOpcodes);
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
		let suppressedCount = 0;
		const hasSuppression = suppressedOpcodes.size > 0;

		for (const tb of trackedBytes) {
			if (hasSuppression && suppressedOpcodes.has(tb.byte)) {
				suppressedCount++;
			} else {
				if (tiles.length < (hasSuppression && suppressedCount > 0 ? MAX_TRACKED - 1 : MAX_TRACKED)) {
					tiles.push({ type: 'byte', byte: tb.byte, mnemonic: tb.mnemonic, count: 0 });
				}
			}
		}

		// If there are suppressed bytes, insert the collapsed card at the end
		if (suppressedCount > 0) {
			// Make room if we filled all slots
			while (tiles.length >= MAX_TRACKED) tiles.pop();
			tiles.push({
				type: 'suppressed',
				byte: -1,
				mnemonic: `⊘ ${suppressedOpcodes.size}`,
				count: suppressedCount
			});
		}

		return tiles;
	});

	function togglePlay() {
		playing = !playing;
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
			if (z === 3)
				return q === 0
					? `Add 1 to ${RP_DESC[p]}`
					: `Subtract 1 from ${RP_DESC[p]}`;
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
			if (q === 0)
				return `Read 2 bytes from memory (at the stack pointer) into ${RP2_DESC[p]}`;
			if (p === 0) return 'Return from subroutine (jump back to where we came from)';
			if (p === 1) return 'Swap BC, DE, and the address pointer with hidden backup copies';
			if (p === 2) return 'Jump to the memory address stored in the address pointer';
			return 'Set the stack pointer (read/write position) to the address pointer value';
		}
		if (z === 2) return `Jump to a specific address if ${CC_DESC[y]}`;
		if (z === 3) {
			if (y === 0) return 'Jump to a specific address (unconditional)';
			if (y === 1) return 'Prefix for extended bit/rotate instructions';
			if (y === 2) return 'Send the working value to a hardware port (I/O output)';
			if (y === 3) return 'Read a byte from a hardware port into the working value (I/O input)';
			if (y === 4)
				return 'Swap the address pointer with the 2 bytes at the stack pointer — this is the key self-replication instruction: it writes data into a neighbor cell\'s memory';
			if (y === 5)
				return 'Swap the address pointer with the DE register pair';
			if (y === 6) return 'Disable interrupts';
			return 'Enable interrupts';
		}
		if (z === 4) return `Call a subroutine at a specific address if ${CC_DESC[y]}`;
		if (z === 5) {
			if (q === 0)
				return `Write ${RP2_DESC[p]} (2 bytes) to memory at the stack pointer`;
			if (p === 0) return 'Call a subroutine at a specific address (saves return point)';
			if (p === 1) return 'Prefix for index register instructions (IX)';
			if (p === 2) return 'Prefix for extended instructions (block copy, I/O)';
			return 'Prefix for index register instructions (IY)';
		}
		if (z === 6) {
			return ALUDESC[y].replace('{0}', 'a byte from the code');
		}
		// z === 7
		return `Call a built-in routine at address ${(y * 8).toString(16).toUpperCase().padStart(2, '0')}h (fast subroutine call)`;
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
	onkeydown={(e) => {
		if (e.target !== document.body) return;
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
				showHelp = false;
				showSpeedMenu = false;
				showSettings = false;
				break;
		}
	}}
/>

<svelte:head>
	<title>Algocell</title>
</svelte:head>

<!-- Full-screen canvas -->
<div
	bind:this={canvasContainer}
	class="fixed inset-0"
	style="-webkit-user-select:none;user-select:none;-webkit-touch-callout:none"
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
		<span class="info-item">
			<span class="info-label">ops/s</span>
			<span class="info-value">{formatNumber(opsPerSec)}</span>
		</span>
		<span class="info-sep"></span>
		<span class="info-item">
			<span class="info-label">mut</span>
			<span class="info-value">1/2<sup>{noiseExp}</sup></span>
		</span>
		<button
			class="info-chart-toggle"
			onclick={toggleChart}
			title="Toggle chart"
		>
			<svg
				width="10"
				height="10"
				viewBox="0 0 10 10"
				fill="currentColor"
				style="opacity:{showInfoChart ? 1 : 0.4}"
			>
				<polyline points="0,8 3,4 6,6 10,1" fill="none" stroke="currentColor" stroke-width="1.5" />
			</svg>
		</button>
	</div>
	{#if showInfoChart && chartVersion > 0 && chartLen > 1}
		<div class="info-chart">
			<svg
				viewBox="0 0 200 120"
				class="info-chart-svg"
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
						<div
							class="freq-cell freq-cell-suppressed-group"
							role="button"
							tabindex="0"
						>
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
							<svg width="7" height="7" viewBox="0 0 8 8" stroke="currentColor" stroke-width="1.5">
								<path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke-linecap="round" />
							</svg>
						</button>
					{/each}
					<span class="freq-suppress-count">⊘ {suppressedOpcodes.size}</span>
				</div>
			{/if}
		</div>
	{/if}
</div>

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

		<div class="param">
			<div class="param-head">
				<label class="param-label"
					><svg
						width="12"
						height="12"
						viewBox="0 0 24 24"
						fill="none"
						stroke="var(--accent)"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
						><rect x="3" y="3" width="8" height="8" /><rect x="13" y="3" width="8" height="8" /><rect x="3" y="13" width="8" height="8" /><rect x="13" y="13" width="8" height="8" /></svg
					> Grid</label
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
						>Grid topology and dimensions. Hex grids produce more organic patterns. Changing resets
						simulation.</span
					>
				</span>
				<div class="grid-type-toggle">
					<button
						class="grid-type-btn"
						class:active={gridType === 'square'}
						onclick={() => handleGridTypeChange('square')}
						title="Square grid — 4 neighbors"
					><svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.2"><rect x="1" y="1" width="3.5" height="3.5" /><rect x="5.5" y="1" width="3.5" height="3.5" /><rect x="1" y="5.5" width="3.5" height="3.5" /><rect x="5.5" y="5.5" width="3.5" height="3.5" /></svg></button
					>
					<button
						class="grid-type-btn"
						class:active={gridType === 'hex'}
						onclick={() => handleGridTypeChange('hex')}
						title="Hex grid — 6 neighbors"
					><svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M5 1L8.5 3V7L5 9L1.5 7V3Z" /></svg></button
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
				<button class="seed-apply" onclick={handleReset}>
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
						>Random byte flips per batch. Higher = more mutations = faster evolution. Applied live.</span
					>
				</span>
				<span class="param-val">1/2<sup>{noiseExp}</sup></span>
			</div>
			<div class="slider-track-wrap">
				<input
					id="noise-input"
					type="range"
					class="slider"
					value={13 - noiseExp}
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
						>Cell pairs evaluated per step. More = faster evolution, higher GPU load. Applied live.</span
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
					> Z80 steps</label
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
						>CPU cycles per pair execution. More steps = longer programs can run, slower throughput.
						Applied live.</span
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
				<label class="param-label"
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
					> Suppress opcodes</label
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
						>Type a substring pattern (e.g. LD, POP, EX) and press Enter to disable all matching
						opcodes. Suppressed instructions are skipped as NOPs. Use to block the dominant
						self-replicator and see if other strategies emerge.</span
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
					placeholder="LD; POP; EX; PUSH; ADD"
					bind:value={suppressInput}
					onkeydown={(e) => {
						if (e.key === 'Enter' && suppressInput.trim()) {
							addSuppressPattern(suppressInput);
							suppressInput = '';
						}
					}}
				/>
				{#if suppressInput.trim()}
					{@const preview = (() => {
						const parts = suppressInput.split(';').map(s => s.trim().toUpperCase()).filter(Boolean);
						if (parts.length === 0) return 0;
						let count = 0;
						for (let i = 0; i < 256; i++) {
							const m = (byteToMnemonic(i) || '').toUpperCase();
							const h = i.toString(16).toUpperCase().padStart(2, '0');
							for (const p of parts) {
								if (m.includes(p) || h.includes(p) || ('0X' + h).includes(p)) { count++; break; }
							}
						}
						return count;
					})()}
					<span class="suppress-preview">↵ to suppress {preview} opcode{preview !== 1 ? 's' : ''}</span>
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
							<svg width="8" height="8" viewBox="0 0 8 8" stroke="currentColor" stroke-width="1.5">
								<path d="M1.5 1.5l5 5M6.5 1.5l-5 5" stroke-linecap="round" />
							</svg>
						</button>
					{/each}
					<span class="suppress-count">{suppressedOpcodes.size} opcode{suppressedOpcodes.size !== 1 ? 's' : ''}</span>
				</div>
			{/if}
		</div>

		<div class="param">
			<div class="param-head">
				<label class="param-label"
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
					> Appearance</label
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
					<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
						<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>
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
						<input type="range" class="slider color-adj-slider" min="-0.5" max="0.5" step="0.01" bind:value={brightness} />
					</div>
					<div class="color-adj-row">
						<span class="color-adj-label">Contrast</span>
						<input type="range" class="slider color-adj-slider" min="0.3" max="2.5" step="0.01" bind:value={contrast} />
					</div>
					<div class="color-adj-row">
						<span class="color-adj-label">Saturation</span>
						<input type="range" class="slider color-adj-slider" min="0" max="3" step="0.01" bind:value={saturation} />
					</div>
					<button class="color-adj-reset" onclick={() => { brightness = 0; contrast = 1; saturation = 1; }}>Reset</button>
				</div>
			{/if}
			<label class="grid-lines-toggle">
				<input type="checkbox" bind:checked={showGridLines} class="sr-only" />
				<span class="check-box" class:checked={showGridLines}>
					{#if showGridLines}
						<svg width="8" height="8" viewBox="0 0 12 12" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 6l3 3 5-5" /></svg>
					{/if}
				</span>
				<span>Grid lines</span>
			</label>
		</div>

		<div class="param">
			<div class="param-head">
				<label class="param-label"
					><svg
						width="12"
						height="12"
						viewBox="0 0 24 24"
						fill="none"
						stroke="var(--accent)"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
						><rect x="3" y="3" width="18" height="18" rx="2" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="12" y1="3" x2="12" y2="21" /></svg
					> Detail</label
				>
				<span class="param-info-wrap" class:show-tip={openTip === 'detail'}>
					<button
						class="param-info"
						onmouseenter={() => { openTip = 'detail'; }}
						onmouseleave={() => { openTip = null; }}
						onclick={() => { openTip = openTip === 'detail' ? null : 'detail'; }}
					>?</button>
					<span class="param-tip"
						>Toggle between detailed view (showing individual bytes within each cell) and simple view
						(one averaged color per cell). Hover tooltip always shows full detail.</span
					>
				</span>
				<div class="detail-toggle">
					<button
						class="grid-type-btn"
						class:active={!simpleView}
						onclick={() => (simpleView = false)}
						title="Detailed — show individual bytes"
					><svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1"><rect x="0.5" y="0.5" width="4" height="4" /><rect x="5.5" y="0.5" width="4" height="4" /><rect x="0.5" y="5.5" width="4" height="4" /><rect x="5.5" y="5.5" width="4" height="4" /></svg></button
					>
					<button
						class="grid-type-btn"
						class:active={simpleView}
						onclick={() => (simpleView = true)}
						title="Simple — one color per cell"
					><svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.2"><rect x="1" y="1" width="8" height="8" rx="1" /></svg></button
					>
				</div>
			</div>
		</div>
	</div>
{/if}

<!-- Genome tooltip -->
{#if hoveredCell >= 0 && cellData && !isPanning}
	{#if gridType === 'hex'}
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
		<div class="modal" onclick={(e) => e.stopPropagation()} role="dialog" tabindex="-1">
			<div class="modal-header">
				<div class="modal-tabs">
					<button
						class="modal-tab"
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
							/><line x1="15" y1="20" x2="15" y2="23" /><line x1="20" y1="9" x2="23" y2="9" /><line
								x1="20"
								y1="14"
								x2="23"
								y2="14"
							/><line x1="1" y1="9" x2="4" y2="9" /><line x1="1" y1="14" x2="4" y2="14" /></svg
						>
						Z80
					</button>
					<button
						class="modal-tab"
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
							/><line x1="12" y1="8" x2="12" y2="3" /><line x1="20" y1="21" x2="20" y2="16" /><line
								x1="20"
								y1="12"
								x2="20"
								y2="3"
							/><line x1="1" y1="14" x2="7" y2="14" /><line x1="9" y1="8" x2="15" y2="8" /><line
								x1="17"
								y1="16"
								x2="23"
								y2="16"
							/></svg
						>
						Params
					</button>
					<button
						class="modal-tab tab-keys"
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
					<svg width="10" height="10" viewBox="0 0 10 10" stroke="currentColor" stroke-width="1.5">
						<path d="M2 2l6 6M8 2l-6 6" stroke-linecap="round" />
					</svg>
				</button>
			</div>
			<div class="modal-body">
				{#if helpTab === 'overview'}
					<p>
						A configurable grid of cells (default 200&times;200), each containing random bytes
						interpreted as Z80 machine code. Every step, random adjacent pairs are selected, their
						bytes concatenated and executed as a Z80 program, and the modified memory is written
						back. Self-replicating programs spontaneously emerge.
					</p>
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

					<h4>Grid Topologies</h4>
					<p>
						<strong>Square</strong> &mdash; Each cell is a 4&times;4 block of 16 bytes. Cells interact
						with 4 cardinal neighbors.
					</p>
					<p>
						<strong>Hexagonal</strong> &mdash; Each cell holds 19 bytes in a 3-4-5-4-3 hexagonal arrangement.
						Cells interact with 6 neighbors, producing more organic emergent structures. Uses odd-r offset
						coordinates (odd rows shift right).
					</p>

					<!-- Simulation cycle diagram -->
					<Mermaid
						chart={`
graph TD
    GRID("Cell Grid\nconfigurable size · square or hex") -->|"pick random pair"| PAIR("Cell A + Cell B\nbytes combined")
    PAIR -->|"execute as Z80"| CPU("Z80 CPU · configurable steps")
    CPU -->|"write back + mutate"| GRID
`}
					/>

					<h4>The Phase Transition</h4>
					<p>
						Initially all 256 byte values are uniformly distributed. But the Z80 CPU starts every
						execution with all registers set to zero, so instructions like
						<code>LD (HL),A</code> or <code>PUSH BC</code> tend to write zeros into memory. This makes
						NOP (0x00) accumulate rapidly &mdash; random code acts as a &ldquo;zero pump.&rdquo;
					</p>
					<p>
						Once self-replicators emerge (typically <code>POP HL</code> +
						<code>EX (SP),HL</code> loops), they actively copy their own bytes forward, displacing the
						NOPs. Watch the frequency chart to see this happen in real time.
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
						For example, byte 0x00 appears as <code>NOP</code> when executed as an instruction, but
						as <code>00</code> when it is an operand of a multi-byte instruction like
						<code>LD BC,nn</code>.
					</p>

					<h4>Frequency Chart</h4>
					<p>
						The bottom-left widget shows the most common byte values in real time. When suppression
						is active, suppressed bytes are collapsed into a single card so you always see the top
						non-suppressed opcodes. Hover any tile for a plain-English description; click to
						suppress or un-suppress it. Toggle the chart sparkline with <kbd>S</kbd>.
					</p>

					<h4>Opcode Suppression</h4>
					<p>
						Type a substring pattern (e.g. <code>LD</code>, <code>POP</code>, <code>EX</code>)
						into the Suppress field in Settings and press Enter. All opcodes whose mnemonic contains
						that substring are disabled &mdash; the Z80 CPU skips them as if they were NOPs.
					</p>
					<ul class="help-list">
						<li>
							<strong>Pattern matching</strong> &mdash; <code>LD</code> suppresses all load
							instructions (~100 opcodes), <code>POP</code> suppresses all pop instructions,
							<code>E1</code> matches by hex code
						</li>
						<li>
							<strong>Click-to-suppress</strong> &mdash; click any tile in the frequency widget
							to add its exact mnemonic as a pattern
						</li>
						<li>
							<strong>Remove patterns</strong> &mdash; click the &times; on any chip to remove
							it, or use the clear button to remove all
						</li>
					</ul>
					<p>
						Note: suppression prevents <em>execution</em>, not <em>existence</em>. A suppressed
						byte value can still appear in memory if other instructions write it or replicators
						carry it as data.
					</p>
				{:else if helpTab === 'z80'}
					<h4>Registers</h4>
					<p>The Z80 has 8-bit and 16-bit registers used as operands:</p>
					<div class="z80-reg-table">
						<div class="z80-reg-row head"><span>Name</span><span>Size</span><span>Role</span></div>
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
							<span class="z80-reg">AF</span><span>16-bit</span><span>Accumulator + flags pair</span
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
							<div class="z80-instr"><span class="z80-op">RET</span> Pop PC (return from call)</div>
							<div class="z80-instr"><span class="z80-op">NOP</span> No operation (do nothing)</div>
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

					<h4>Why POP HL &amp; EX (SP),HL Win</h4>
					<p>
						<span class="z80-op">POP HL</span> reads 2 bytes from memory via the stack pointer.
						<span class="z80-op">EX (SP),HL</span> writes them forward to the next position. Repeating
						this copies the cell's own bytes into its neighbor &mdash; a minimal self-replicating loop.
						Once one cell contains this pattern, it spreads exponentially.
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
						Switch between <strong>Square</strong> and <strong>Hex</strong> topologies. Square cells hold
						16 bytes (4&times;4) with 4 neighbors. Hex cells hold 19 bytes (3-4-5-4-3 honeycomb) with
						6 neighbors. Changing resets the simulation.
					</p>

					<h4>Grid Size (W &times; H)</h4>
					<p>
						Width and height of the grid in cells. Default is 200&times;200. Larger grids give more
						space for diverse species to evolve, but use more GPU memory and compute. Changing
						resets the simulation.
					</p>

					<h4>Seed</h4>
					<p>
						The random seed used to initialize the grid. Same seed produces the same starting state.
						Change it and press Reset to try different initial conditions.
					</p>

					<h4>Mutation Rate</h4>
					<p>
						Controls the probability of random byte flips after each execution step (1/2<sup>n</sup
						>). Higher slider values mean more mutations. Too low and replicators can't emerge; too
						high and they can't survive. Sweet spot is usually 3&ndash;5.
					</p>

					<h4>Pairs / Batch</h4>
					<p>
						How many cell pairs are selected and executed each simulation step. Higher values speed
						up evolution but use more GPU time. At max, roughly a quarter of all cells are updated
						per step.
					</p>

					<h4>Z80 Steps</h4>
					<p>
						Maximum CPU cycles per pair execution (16&ndash;1024). Lower values make the simulation
						faster but limit program complexity &mdash; simple replicators emerge quickly. Higher
						values allow more complex programs to develop but slow down the simulation. Default
						(128) balances speed and complexity.
					</p>

					<h4>Suppress Opcodes</h4>
					<p>
						Type a substring (e.g. <code>LD</code>, <code>POP</code>, <code>EX</code>) and press
						Enter to suppress all matching opcodes. The Z80 CPU will skip these instructions as if
						they were NOPs. Use this to block the dominant self-replicator pattern and observe
						whether alternative replication strategies evolve. Patterns appear as removable chips
						showing how many opcodes are affected.
					</p>

					<h4>Colormap</h4>
					<p>
						Choose between three visual themes: Rainbow (warm opcode-aware), Ocean (cool blues), and
						Thermal (heat map).
					</p>
				{:else if helpTab === 'keys'}
					<div class="modal-shortcuts">
						<div class="shortcut"><kbd>Space</kbd> Play / Pause</div>
						<div class="shortcut"><kbd>R</kbd> Reset</div>
						<div class="shortcut"><kbd>S</kbd> Toggle chart</div>
						<div class="shortcut"><kbd>H</kbd> Toggle help</div>
						<div class="shortcut"><kbd>F</kbd> Fit view</div>
						<div class="shortcut"><kbd>Scroll</kbd> Zoom</div>
						<div class="shortcut"><kbd>Drag</kbd> Pan</div>
						<div class="shortcut"><kbd>Dbl-click</kbd> Reset view</div>
					</div>
				{:else if helpTab === 'about'}
					<p>
						<strong>Algocell</strong> is a WebGPU-accelerated artificial life simulator that runs Z80
						machine code on a configurable grid of cells. Choose between square and hexagonal topologies,
						adjust grid size, Z80 execution depth, mutation rate, and more. Self-replicating programs
						spontaneously emerge from random noise.
					</p>
					<p>
						Based on the paper by
						<a
							href="https://arxiv.org/abs/2406.19108"
							target="_blank"
							rel="noopener"
							class="help-link">Agüera y Arcas et al. (2024)</a
						>
						and the
						<a href="https://github.com/znah/zff" target="_blank" rel="noopener" class="help-link"
							>original implementation</a
						>
						by Alexander Mordvintsev. This version uses WebGPU compute shaders instead of the original
						JavaScript/WASM approach, enabling massively parallel execution on the GPU. The hexagonal grid
						mode produces more organic emergent structures thanks to 6-neighbor interactions.
					</p>
					<p>
						Developed by <strong>Neo Mohsenvand</strong> with the help of
						<a href="https://claude.ai" target="_blank" rel="noopener" class="help-link"
							>Claude Code</a
						>.
					</p>
					<p class="cmap-note" style="margin-top: 12px;">
						<a
							href="https://github.com/NeoVand/algocell"
							target="_blank"
							rel="noopener"
							class="help-link">GitHub</a
						>
						&middot; SvelteKit &middot; TypeScript &middot; WebGPU &middot; Tailwind CSS
					</p>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	/* Prevent mobile zoom/scroll escaping on all UI overlays */
	:global(html),
	:global(body) {
		touch-action: manipulation;
		overscroll-behavior: none;
		-webkit-text-size-adjust: 100%;
	}

	/* Remove focus outlines on all buttons */
	button:focus,
	button:focus-visible {
		outline: none;
	}

	/* ── Toolbar ── */
	.toolbar {
		position: fixed;
		top: 12px;
		right: 12px;
		z-index: 40;
		display: flex;
		align-items: center;
		gap: 2px;
		padding: 3px;
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
		border-radius: 20px;
		backdrop-filter: blur(16px);
		box-shadow: 0 2px 20px rgba(0, 0, 0, 0.4);
		transition:
			border-radius 0.2s ease,
			gap 0.2s ease,
			padding 0.2s ease;
	}
	.toolbar.collapsed {
		border-radius: 50%;
		gap: 0;
		padding: 3px;
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
		padding: 12px;
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
		font-size: 10.5px;
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
		width: 229px;
	}
	.param {
		margin-bottom: 10px;
	}
	.param:last-child {
		margin-bottom: 0;
	}
	.param-head {
		display: flex;
		align-items: center;
		gap: 4px;
		margin-bottom: 4px;
	}
	.param-label {
		font-size: 10.5px;
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
		width: 16px;
		height: 16px;
		border-radius: 50%;
		background: var(--bg-muted);
		border: none;
		color: var(--text-subtle);
		font-size: 10px;
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
		border: 1px solid rgba(255, 255, 255, 0.10);
		border-radius: 5px;
		padding: 0 7px;
		height: 24px;
		font-size: 11px;
		font-family: monospace;
		color: var(--text-primary);
		outline: none;
		-moz-appearance: textfield;
		appearance: textfield;
	}
	.seed-input::-webkit-inner-spin-button,
	.seed-input::-webkit-outer-spin-button {
		-webkit-appearance: none;
		appearance: none;
		margin: 0;
	}
	.seed-input:focus {
		border-color: rgba(255, 255, 255, 0.15);
	}
	.seed-apply {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 24px;
		height: 24px;
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
		border: 1px solid rgba(255, 255, 255, 0.10);
		border-radius: 5px;
		padding: 0 6px;
		height: 22px;
		font-size: 11px;
		font-family: monospace;
		color: var(--text-primary);
		outline: none;
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
		height: 4px;
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
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--accent);
		border: none;
		box-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
		cursor: pointer;
		transition: transform 0.1s;
	}
	.slider::-webkit-slider-thumb:hover {
		transform: scale(1.15);
	}
	.slider::-moz-range-thumb {
		width: 12px;
		height: 12px;
		border-radius: 50%;
		background: var(--accent);
		border: none;
		box-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
		cursor: pointer;
	}
	.slider::-moz-range-track {
		height: 4px;
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
		height: 24px;
		background: rgba(255, 255, 255, 0.03);
		border: 1px solid rgba(255, 255, 255, 0.10);
		border-radius: 5px;
		color: var(--text-primary);
		font-size: 11px;
		font-family: monospace;
		outline: none;
	}
	.suppress-input::placeholder {
		color: var(--text-subtle);
		opacity: 1;
	}
	.suppress-input:focus {
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
		padding: 2px 7px;
		font-size: 10px;
		font-family: var(--font-mono, monospace);
		font-weight: 600;
		color: var(--text-primary);
		background: rgba(255, 255, 255, 0.1);
		border: 1px solid var(--border-muted);
		border-radius: 4px;
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
		padding: 4px 0 3px;
		font-size: 9px;
		font-family: monospace;
		text-transform: capitalize;
		color: var(--text-muted);
		background: rgba(255, 255, 255, 0.04);
		border: 1px solid var(--border-subtle);
		border-radius: 5px;
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
		margin-top: 6px;
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
		transition: border-color 0.15s, background 0.15s;
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
	}
	.modal-tab {
		background: none;
		border: none;
		padding: 8px 10px;
		font-size: 11px;
		color: var(--text-muted);
		cursor: pointer;
		border-bottom: 2px solid transparent;
		margin-bottom: -1px;
		display: flex;
		align-items: center;
		gap: 4px;
		transition:
			color 0.15s,
			border-color 0.15s;
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
	.help-link {
		color: var(--accent);
		text-decoration: none;
	}
	.help-link:hover {
		text-decoration: underline;
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
</style>

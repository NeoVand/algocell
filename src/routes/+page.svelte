<script lang="ts">
	import { GPUEngine } from '$lib/gpu/engine';
	import {
		SOUP_WIDTH,
		SOUP_HEIGHT,
		DEFAULT_SEED,
		DEFAULT_NOISE_EXP,
		MAX_BATCH_PAIR_N,
		Z80_STEPS
	} from '$lib/sim/constants';
	import { disassemble, byteToMnemonic } from '$lib/z80-disasm';
	import { getCellData } from '$lib/sim/soup';
	import { unpackRGBA, createColormap, COLORMAP_NAMES } from '$lib/colormap';
	import type { ColormapName } from '$lib/colormap';
	import { untrack } from 'svelte';
	import Mermaid from '$lib/components/Mermaid.svelte';

	let colormapName: ColormapName = $state('rainbow');
	let colormap = $state(createColormap('rainbow'));

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

	// Stats
	let batchCount = $state(0);
	let opsPerSec = $state(0);
	let topBytes: { byte: number; count: number; mnemonic: string }[] = $state([]);
	let showHelp = $state(false);
	let helpTab = $state<'overview' | 'visuals' | 'z80' | 'params' | 'keys' | 'about'>('overview');
	let showSpeedMenu = $state(false);
	let showSettings = $state(false);
	let toolbarCollapsed = $state(false);
	let openTip = $state<string | null>(null);
	let showInfoChart = $state(true);

	// Frequency chart: track top N bytes normalized over time
	const TOTAL_CELLS = SOUP_WIDTH * SOUP_HEIGHT;
	const MAX_TRACKED = 10;
	const MAX_HISTORY = 500;

	interface StatsPoint {
		batch: number;
		freqs: { byte: number; frac: number }[]; // fraction of total cells (0..1)
	}
	let statsHistory: StatsPoint[] = $state([]);

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

	// Derived
	let currentZoom = $derived(engine != null ? engine.view.zoom : SOUP_WIDTH);
	let zoomPercent = $derived(Math.round((SOUP_WIDTH / currentZoom) * 100));

	// Build frequency lines using concentration factor (frac * 256)
	// Uniform distribution = 1.0, a byte at 50% = 128.0
	// This makes the early random phase show natural jitter instead of flatness
	let freqLines = $derived.by(() => {
		const allSeries = trackedBytes.map((tb) => ({
			byte: tb.byte,
			mnemonic: tb.mnemonic,
			color: byteChartColor(tb.byte),
			values: statsHistory.map((s) => {
				const found = s.freqs.find((f) => f.byte === tb.byte);
				return found ? found.frac * 256 : 0; // concentration factor
			})
		}));
		// Shared max across all lines
		let globalMax = 1.5; // at least 1.5x uniform so early jitter is visible
		for (const s of allSeries) {
			for (const v of s.values) {
				if (v > globalMax) globalMax = v;
			}
		}
		return allSeries.map((s) => ({
			...s,
			path: buildSparklinePathShared(s.values, 200, 120, globalMax)
		}));
	});

	let tooltipStyle = $derived(computeTooltipStyle(mouseX, mouseY));

	let cellCoords = $derived(
		hoveredCell >= 0
			? { x: hoveredCell % SOUP_WIDTH, y: Math.floor(hoveredCell / SOUP_WIDTH) }
			: null
	);

	let genomeCells = $derived(
		cellData && disasmLines.length > 0 ? buildGenomeGrid(cellData, disasmLines) : []
	);

	function computeTooltipStyle(mx: number, my: number): string {
		const tooltipW = 196;
		const tooltipH = 196;
		const vw = window.innerWidth;
		const vh = window.innerHeight;
		let x = mx + 16;
		let y = my + 16;
		if (x + tooltipW > vw) x = mx - tooltipW - 16;
		if (y + tooltipH > vh) y = my - tooltipH - 16;
		if (x < 8) x = 8;
		if (y < 8) y = 8;
		return `left:${x}px;top:${y}px`;
	}

	function buildSparklinePathShared(values: number[], w: number, h: number, max: number): string {
		if (values.length < 2) return '';
		const step = w / (values.length - 1);
		return values
			.map((v, i) => {
				const x = i * step;
				const y = h - (v / max) * (h - 4) - 2;
				return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
			})
			.join(' ');
	}

	$effect(() => {
		if (!canvas) return;

		const initialSeed = untrack(() => seed);
		const eng = new GPUEngine(initialSeed);

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
			frameCount++;

			if (playing && frameCount % 30 === 0 && !statsLoading) {
				batchCount = engine.batchCount;
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
			topBytes = entries.slice(0, 20);

			// Update tracked bytes to current top N
			const topN = entries.slice(0, MAX_TRACKED);
			trackedBytes = topN.map((e) => ({ byte: e.byte, mnemonic: e.mnemonic }));

			// Record normalized frequencies for all 256 bytes (so history is complete)
			const freqs = entries.map((e) => ({
				byte: e.byte,
				frac: e.count / TOTAL_CELLS
			}));

			statsHistory = [
				...statsHistory.slice(-(MAX_HISTORY - 1)),
				{ batch: batchCount, freqs }
			];
		} catch {
			// stats readback failed silently
		} finally {
			statsLoading = false;
		}
	}

	function handleCanvasMouseMove(e: MouseEvent) {
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

	function refreshCellData(cell: number) {
		if (!engine) return;
		engine.readSoupData().then((soupData) => {
			if (soupData.length === 0 || hoveredCell !== cell) return;
			const data = getCellData(soupData, cell);
			cellData = data;
			disasmLines = disassemble(data);
		});
	}

	function handleCanvasMouseDown(e: MouseEvent) {
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
		const factor = e.deltaY > 0 ? 1.12 : 1 / 1.12;
		engine.zoomAt(sx, sy, canvasW, canvasH, factor);
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
		engine?.resetView();
	}

	function handleReset() {
		if (!engine) return;
		engine.reset(seed);
		batchCount = 0;
		opsPerSec = 0;
		topBytes = [];
		statsHistory = [];
		trackedBytes = [];
	}

	function handleSeedChange(e: Event) {
		const val = parseInt((e.target as HTMLInputElement).value);
		if (!isNaN(val) && val >= 0) {
			seed = val;
		}
	}

	function handleNoiseExpChange(e: Event) {
		const val = parseInt((e.target as HTMLInputElement).value);
		if (!isNaN(val) && engine) {
			noiseExp = val;
			engine.noiseExp = val;
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

	function togglePlay() {
		playing = !playing;
	}

	function setSpeed(s: number) {
		speed = s;
		showSpeedMenu = false;
	}

	function formatNumber(n: number): string {
		if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(1) + 'B';
		if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
		if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
		return n.toString();
	}

	function hexByte(b: number): string {
		return b.toString(16).toUpperCase().padStart(2, '0');
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

	function buildGenomeGrid(
		data: Uint8Array,
		disasm: ReturnType<typeof disassemble>
	): GenomeCell[] {
		const cells: GenomeCell[] = [];
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
				showInfoChart = !showInfoChart;
				break;
			case 'KeyF':
				engine?.resetView();
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
<div bind:this={canvasContainer} class="fixed inset-0">
	{#if gpuError}
		<div
			class="absolute inset-0 z-50 flex items-center justify-center"
			style="background:var(--bg-panel)"
		>
			<p class="text-red-400 text-center max-w-md px-8">{gpuError}</p>
		</div>
	{/if}
	<canvas
		bind:this={canvas}
		class="block w-full h-full"
		style="cursor:{isPanning ? 'grabbing' : 'crosshair'}"
		onmousemove={handleCanvasMouseMove}
		onmouseleave={handleCanvasMouseLeave}
		onmousedown={handleCanvasMouseDown}
		onmouseup={handleCanvasMouseUp}
		onwheel={handleCanvasWheel}
		ondblclick={handleCanvasDblClick}
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
						<button
							class="speed-opt"
							class:active={speed === s}
							onclick={() => setSpeed(s)}>{s}x</button
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
			<svg width="16" height="16" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.4">
				<path d="M2 3.5h10M2 7h10M2 10.5h10" stroke-linecap="round"/>
				<circle cx="5" cy="3.5" r="1.3" fill="currentColor" stroke="none"/>
				<circle cx="9" cy="7" r="1.3" fill="currentColor" stroke="none"/>
				<circle cx="6" cy="10.5" r="1.3" fill="currentColor" stroke="none"/>
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
				<path
					d="M5.5 5.5a1.75 1.75 0 0 1 3.25 1c0 1-1.5 1.25-1.5 2.25"
					stroke-linecap="round"
				/>
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
			<span class="info-label">batch</span>
			<span class="info-value">{formatNumber(batchCount)}</span>
		</span>
		<span class="info-sep"></span>
		<span class="info-item">
			<span class="info-label">ops/s</span>
			<span class="info-value">{formatNumber(opsPerSec)}</span>
		</span>
		<span class="info-sep"></span>
		<span class="info-item">
			<span class="info-label">mut</span>
			<span class="info-value">1/2<sup>{noiseExp}</sup></span>
		</span>
		<button class="info-chart-toggle" onclick={() => (showInfoChart = !showInfoChart)} title="Toggle chart">
			<svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor" style="opacity:{showInfoChart ? 1 : 0.4}">
				<polyline points="0,8 3,4 6,6 10,1" fill="none" stroke="currentColor" stroke-width="1.5"/>
			</svg>
		</button>
	</div>
	{#if showInfoChart && statsHistory.length > 1}
		<div class="info-chart">
			<svg viewBox="0 0 200 120" class="info-chart-svg" preserveAspectRatio="none"
				onmouseleave={() => { chartHoveredByte = -1; }}
			>
				{#each freqLines as fl (fl.byte)}
					{#if fl.path}
						<polyline
							points={fl.path.replace(/[ML]/g, (m) => (m === 'M' ? '' : ' ')).trim()}
							fill="none"
							stroke={fl.color}
							stroke-width={chartHoveredByte === fl.byte ? '4' : '2.5'}
							opacity={chartHoveredByte >= 0 ? (chartHoveredByte === fl.byte ? '1' : '0.15') : '0.8'}
						/>
					{/if}
				{/each}
			</svg>
			<div class="freq-grid">
				{#each trackedBytes as tb, i (tb.byte)}
					<div
						class="freq-cell"
						class:dimmed={chartHoveredByte >= 0 && chartHoveredByte !== tb.byte}
						class:highlighted={chartHoveredByte === tb.byte}
						style="background:{byteColor(tb.byte)};color:{cellTextColor(tb.byte)}"
						role="button"
						tabindex="0"
						onmouseenter={() => { chartHoveredByte = tb.byte; }}
						onmouseleave={() => { chartHoveredByte = -1; }}
					>
						<span class="freq-rank">{i + 1}</span>
						<span class="freq-label">{tb.mnemonic || hexByte(tb.byte)}</span>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>

<!-- Settings panel -->
{#if showSettings}
	<div class="panel settings-panel">
		<div class="panel-header">
			<span class="panel-title"><svg width="12" height="12" viewBox="0 0 14 14" fill="none" stroke="var(--accent)" stroke-width="1.4"><path d="M2 3.5h10M2 7h10M2 10.5h10" stroke-linecap="round"/><circle cx="5" cy="3.5" r="1.3" fill="var(--accent)" stroke="none"/><circle cx="9" cy="7" r="1.3" fill="var(--accent)" stroke="none"/><circle cx="6" cy="10.5" r="1.3" fill="var(--accent)" stroke="none"/></svg>Parameters</span>
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
				<label class="param-label" for="seed-input"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="3"/><circle cx="8" cy="8" r="1.5" fill="currentColor" stroke="none"/><circle cx="16" cy="8" r="1.5" fill="currentColor" stroke="none"/><circle cx="8" cy="16" r="1.5" fill="currentColor" stroke="none"/><circle cx="16" cy="16" r="1.5" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none"/></svg> Seed</label>
				<span class="param-info-wrap" class:show-tip={openTip === 'seed'}>
					<button class="param-info" onmouseenter={() => { openTip = 'seed'; }} onmouseleave={() => { openTip = null; }} onclick={() => { openTip = openTip === 'seed' ? null : 'seed'; }}>?</button>
					<span class="param-tip">Random seed for initial soup generation. Changing seed requires reset.</span>
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
					<svg width="12" height="12" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5">
						<path d="M2 7a5 5 0 1 1 1 3" stroke-linecap="round" />
						<path d="M2 3v4h4" stroke-linecap="round" stroke-linejoin="round" />
					</svg>
				</button>
			</div>
		</div>

		<div class="param">
			<div class="param-head">
				<label class="param-label" for="noise-input"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/></svg> Mutation rate</label>
				<span class="param-info-wrap" class:show-tip={openTip === 'mutation'}>
					<button class="param-info" onmouseenter={() => { openTip = 'mutation'; }} onmouseleave={() => { openTip = null; }} onclick={() => { openTip = openTip === 'mutation' ? null : 'mutation'; }}>?</button>
					<span class="param-tip">Random byte flips per batch. Higher = more mutations = faster evolution. Applied live.</span>
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
					oninput={(e) => { const val = 13 - parseInt((e.target as HTMLInputElement).value); if (!isNaN(val) && engine) { noiseExp = val; engine.noiseExp = val; } }}
				/>
			</div>
		</div>

		<div class="param">
			<div class="param-head">
				<label class="param-label" for="pairs-input"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg> Pairs/batch</label>
				<span class="param-info-wrap" class:show-tip={openTip === 'pairs'}>
					<button class="param-info" onmouseenter={() => { openTip = 'pairs'; }} onmouseleave={() => { openTip = null; }} onclick={() => { openTip = openTip === 'pairs' ? null : 'pairs'; }}>?</button>
					<span class="param-tip">Cell pairs evaluated per step. More = faster evolution, higher GPU load. Applied live.</span>
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
				<label class="param-label" for="steps-input"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg> Z80 steps</label>
				<span class="param-info-wrap" class:show-tip={openTip === 'steps'}>
					<button class="param-info" onmouseenter={() => { openTip = 'steps'; }} onmouseleave={() => { openTip = null; }} onclick={() => { openTip = openTip === 'steps' ? null : 'steps'; }}>?</button>
					<span class="param-tip">CPU cycles per pair execution. More steps = longer programs can run, slower throughput. Applied live.</span>
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
					oninput={(e) => { const val = parseInt((e.target as HTMLInputElement).value); if (!isNaN(val) && engine) { z80Steps = val; engine.z80Steps = val; } }}
				/>
			</div>
		</div>

		<div class="param">
			<div class="param-head">
				<label class="param-label"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 2a7 7 0 0 0 0 20 4 4 0 0 1 0-8 4 4 0 0 0 0-8z"/><circle cx="12" cy="9" r="1" fill="var(--accent)"/></svg> Colormap</label>
				<span class="param-info-wrap" class:show-tip={openTip === 'cmap'}>
					<button class="param-info" onmouseenter={() => { openTip = 'cmap'; }} onmouseleave={() => { openTip = null; }} onclick={() => { openTip = openTip === 'cmap' ? null : 'cmap'; }}>?</button>
					<span class="param-tip">Color scheme for byte values. Each Z80 opcode category gets a distinct color family.</span>
				</span>
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
		</div>
	</div>
{/if}

<!-- Genome tooltip -->
{#if hoveredCell >= 0 && cellData && !isPanning}
	<div class="genome-tip" style={tooltipStyle}>
		<div class="tip-grid">
			{#each genomeCells as cell, i (i)}
				<div
					class="tip-cell"
					class:operand={!cell.isOpcode}
					style="background:{byteColor(cell.byteVal)};color:{byteLuminance(cell.byteVal) > 0.4 ? 'rgba(0,0,0,0.85)' : 'rgba(255,255,255,0.9)'}"
				>
					{cell.label}
				</div>
			{/each}
		</div>
	</div>
{/if}

<!-- Help modal -->
{#if showHelp}
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<div class="modal-backdrop" onclick={() => (showHelp = false)} role="presentation">
		<div class="modal" onclick={(e) => e.stopPropagation()} role="dialog" tabindex="-1">
			<div class="modal-header">
				<div class="modal-tabs">
					<button class="modal-tab" class:active={helpTab === 'overview'} onclick={() => (helpTab = 'overview')}>
						<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
						Overview
					</button>
					<button class="modal-tab" class:active={helpTab === 'visuals'} onclick={() => (helpTab = 'visuals')}>
						<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="13.5" cy="6.5" r="2.5"/><circle cx="19" cy="17" r="2.5"/><circle cx="6" cy="18" r="2.5"/></svg>
						Visuals
					</button>
					<button class="modal-tab" class:active={helpTab === 'z80'} onclick={() => (helpTab = 'z80')}>
						<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>
						Z80
					</button>
					<button class="modal-tab" class:active={helpTab === 'params'} onclick={() => (helpTab = 'params')}>
						<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
						Params
					</button>
					<button class="modal-tab" class:active={helpTab === 'keys'} onclick={() => (helpTab = 'keys')}>
						<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><line x1="6" y1="8" x2="6.01" y2="8"/><line x1="10" y1="8" x2="10.01" y2="8"/><line x1="14" y1="8" x2="14.01" y2="8"/><line x1="18" y1="8" x2="18.01" y2="8"/><line x1="6" y1="12" x2="6.01" y2="12"/><line x1="10" y1="12" x2="10.01" y2="12"/><line x1="14" y1="12" x2="14.01" y2="12"/><line x1="18" y1="12" x2="18.01" y2="12"/><line x1="8" y1="16" x2="16" y2="16"/></svg>
						Keys
					</button>
					<button class="modal-tab" class:active={helpTab === 'about'} onclick={() => (helpTab = 'about')}>
						<svg width="14" height="14" viewBox="0 0 32 32"><ellipse cx="16" cy="20" rx="14" ry="14" fill="#c8875a"/><line x1="16" y1="8" x2="16" y2="32" stroke="#1a1210" stroke-width="1.5"/><circle cx="9" cy="16" r="2.5" fill="#1a1210"/><circle cx="10.5" cy="24" r="2.2" fill="#1a1210"/><circle cx="23" cy="16" r="2.5" fill="#1a1210"/><circle cx="21.5" cy="24" r="2.2" fill="#1a1210"/><circle cx="16" cy="7" r="7" fill="#1a1210"/><circle cx="12.8" cy="5.8" r="1.5" fill="#e0cfc0"/><circle cx="19.2" cy="5.8" r="1.5" fill="#e0cfc0"/><path d="M12 1.5 Q9 -2 5 0" fill="none" stroke="#1a1210" stroke-width="1.8" stroke-linecap="round"/><path d="M20 1.5 Q23 -2 27 0" fill="none" stroke="#1a1210" stroke-width="1.8" stroke-linecap="round"/><circle cx="5" cy="0" r="1.8" fill="#1a1210"/><circle cx="27" cy="0" r="1.8" fill="#1a1210"/></svg>
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
				{#if helpTab === 'overview'}
					<p>
						A 200&times;200 grid of cells, each containing 16 random bytes interpreted as Z80
						machine code. Every step, random adjacent pairs are selected, their 32 bytes
						concatenated and executed as a Z80 program (128 steps), and the modified
						memory is written back. Self-replicating programs spontaneously emerge.
					</p>
					<p class="cmap-note">
						Based on <a href="https://arxiv.org/abs/2406.19108" target="_blank" rel="noopener" class="help-link">Hartley &amp; Colton (2024)</a>.
						Re-implemented in WebGPU + SvelteKit from the
						<a href="https://github.com/znah/zff" target="_blank" rel="noopener" class="help-link">original code</a>.
					</p>

					<!-- Simulation cycle diagram -->
					<Mermaid chart={`
graph TD
    GRID("200×200 Grid\n40,000 cells · 16 bytes each") -->|"pick random pair"| PAIR("Cell A + Cell B\n32 bytes combined")
    PAIR -->|"execute as Z80"| CPU("Z80 CPU · 128 steps")
    CPU -->|"write back + mutate"| GRID
`} />

					<h4>The Phase Transition</h4>
					<p>
						Initially all 256 byte values are uniformly distributed. But the Z80 CPU
						starts every execution with all registers set to zero, so instructions like
						<code>LD (HL),A</code> or <code>PUSH BC</code> tend to write zeros into
						memory. This makes NOP (0x00) accumulate rapidly &mdash; random
						code acts as a &ldquo;zero pump.&rdquo;
					</p>
					<p>
						Once self-replicators emerge (typically <code>POP HL</code> +
						<code>EX (SP),HL</code> loops), they actively copy their own bytes forward,
						displacing the NOPs. Watch the frequency chart to see this happen in real
						time.
					</p>
					<svg viewBox="0 0 320 180" xmlns="http://www.w3.org/2000/svg" style="width:100%;margin:8px 0;display:block;border-radius:8px;background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.04);padding:12px 8px;">
						<!-- axes -->
						<line x1="40" y1="20" x2="40" y2="140" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>
						<line x1="40" y1="140" x2="300" y2="140" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>
						<!-- axis labels -->
						<text x="170" y="168" text-anchor="middle" fill="#8a7a6a" font-size="9" font-family="-apple-system, BlinkMacSystemFont, sans-serif">Time</text>
						<text x="14" y="80" text-anchor="middle" fill="#8a7a6a" font-size="9" font-family="-apple-system, BlinkMacSystemFont, sans-serif" transform="rotate(-90,14,80)">Concentration</text>
						<!-- NOP curve (white/gray, starts low, rises to moderate peak, then declines) -->
						<path d="M 40,125 C 50,95 60,68 80,62 S 120,64 150,78 Q 200,108 300,122" fill="none" stroke="rgba(220,215,210,0.7)" stroke-width="2"/>
						<text x="68" y="54" fill="rgba(220,215,210,0.8)" font-size="10" font-family="-apple-system, BlinkMacSystemFont, sans-serif">NOP</text>
						<!-- Self-replicating curve (orange, stays low then rises smoothly) -->
						<path d="M 40,130 C 100,130 150,120 186,88 C 220,58 260,42 300,42" fill="none" stroke="#c8875a" stroke-width="2"/>
						<text x="195" y="36" fill="#c8875a" font-size="10" font-family="-apple-system, BlinkMacSystemFont, sans-serif">Self-Replicating Bytes</text>
						<!-- Phase transition dashed line at crossing (x≈180) -->
						<line x1="180" y1="22" x2="180" y2="140" stroke="rgba(120,160,220,0.5)" stroke-width="1" stroke-dasharray="4,3"/>
						<!-- Phase transition label -->
						<text x="180" y="155" text-anchor="middle" fill="rgba(120,160,220,0.7)" font-size="9" font-family="-apple-system, BlinkMacSystemFont, sans-serif">Phase Transition</text>
					</svg>
				{:else if helpTab === 'visuals'}
					<p>Each cell is colored by its first byte. The colormap assigns distinct colors to Z80 opcode categories. Other bytes get a muted hue from a continuous sweep.</p>

					<div class="cmap-section">
						<div class="cmap-label">Special</div>
						<div class="cmap-grid">
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x00)}"></span><span class="cmap-hex">00</span> NOP</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x76)}"></span><span class="cmap-hex">76</span> HALT</div>
						</div>
					</div>

					<div class="cmap-section">
						<div class="cmap-label">16-bit loads</div>
						<div class="cmap-grid">
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x01)}"></span><span class="cmap-hex">01</span> LD BC,nn</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x11)}"></span><span class="cmap-hex">11</span> LD DE,nn</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x21)}"></span><span class="cmap-hex">21</span> LD HL,nn</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x31)}"></span><span class="cmap-hex">31</span> LD SP,nn</div>
						</div>
					</div>

					<div class="cmap-section">
						<div class="cmap-label">Memory access</div>
						<div class="cmap-grid">
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x2A)}"></span><span class="cmap-hex">2A</span> LD HL,(nn)</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x3A)}"></span><span class="cmap-hex">3A</span> LD A,(nn)</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x22)}"></span><span class="cmap-hex">22</span> LD (nn),HL</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x32)}"></span><span class="cmap-hex">32</span> LD (nn),A</div>
						</div>
					</div>

					<div class="cmap-section">
						<div class="cmap-label">Block transfer</div>
						<div class="cmap-grid">
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xED)}"></span><span class="cmap-hex">ED</span> ED prefix</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xB0)}"></span><span class="cmap-hex">B0</span> LDIR</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xB8)}"></span><span class="cmap-hex">B8</span> LDDR</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xA0)}"></span><span class="cmap-hex">A0</span> LDI</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xA8)}"></span><span class="cmap-hex">A8</span> LDD</div>
						</div>
					</div>

					<div class="cmap-section">
						<div class="cmap-label">Stack</div>
						<div class="cmap-grid">
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xC5)}"></span><span class="cmap-hex">C5</span> PUSH BC</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xD5)}"></span><span class="cmap-hex">D5</span> PUSH DE</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xE5)}"></span><span class="cmap-hex">E5</span> PUSH HL</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xF5)}"></span><span class="cmap-hex">F5</span> PUSH AF</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xC1)}"></span><span class="cmap-hex">C1</span> POP BC</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xD1)}"></span><span class="cmap-hex">D1</span> POP DE</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xE1)}"></span><span class="cmap-hex">E1</span> POP HL</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xF1)}"></span><span class="cmap-hex">F1</span> POP AF</div>
						</div>
					</div>

					<div class="cmap-section">
						<div class="cmap-label">Flow control</div>
						<div class="cmap-grid">
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xC3)}"></span><span class="cmap-hex">C3</span> JP nn</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xCD)}"></span><span class="cmap-hex">CD</span> CALL nn</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xC9)}"></span><span class="cmap-hex">C9</span> RET</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x18)}"></span><span class="cmap-hex">18</span> JR</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x10)}"></span><span class="cmap-hex">10</span> DJNZ</div>
						</div>
					</div>

					<div class="cmap-section">
						<div class="cmap-label">Prefixes &amp; exchange</div>
						<div class="cmap-grid">
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xCB)}"></span><span class="cmap-hex">CB</span> Bit ops</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xDD)}"></span><span class="cmap-hex">DD</span> IX ops</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xFD)}"></span><span class="cmap-hex">FD</span> IY ops</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xE3)}"></span><span class="cmap-hex">E3</span> EX (SP),HL</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xEB)}"></span><span class="cmap-hex">EB</span> EX DE,HL</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x08)}"></span><span class="cmap-hex">08</span> EX AF,AF'</div>
						</div>
					</div>

					<div class="cmap-section">
						<div class="cmap-label">Ranges</div>
						<div class="cmap-grid wide">
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0x60)}"></span><span class="cmap-hex">40-7F</span> Register LD</div>
							<div class="cmap-entry"><span class="help-swatch" style="background:{byteColor(0xA0)}"></span><span class="cmap-hex">80-BF</span> ALU ops</div>
						</div>
					</div>

					<p class="cmap-note">All other bytes get a muted tone from a continuous hue sweep. Switch colormaps in Settings.</p>

					<h4>Cell Tooltips</h4>
					<p>Hover any cell to see a 4x4 grid of its 16 bytes, disassembled as Z80 instructions.</p>
					<ul class="help-list">
						<li><strong>Opcode bytes</strong> show the instruction mnemonic (e.g. <code>NOP</code>, <code>POP HL</code>, <code>LD B,C</code>)</li>
						<li><strong>Operand bytes</strong> show the raw hex value (e.g. <code>00</code>, <code>3F</code>) and appear dimmed &mdash; these are data consumed by the preceding instruction</li>
					</ul>
					<p>For example, byte 0x00 appears as <code>NOP</code> when executed as an instruction, but as <code>00</code> when it is an operand of a multi-byte instruction like <code>LD BC,nn</code>.</p>

					<h4>Frequency Chart</h4>
					<p>Tracks the top 10 most common bytes over time. Y-axis is concentration factor (fraction &times; 256). Value of 1.0 = uniform distribution.</p>
				{:else if helpTab === 'z80'}
					<h4>Registers</h4>
					<p>The Z80 has 8-bit and 16-bit registers used as operands:</p>
					<div class="z80-reg-table">
						<div class="z80-reg-row head"><span>Name</span><span>Size</span><span>Role</span></div>
						<div class="z80-reg-row"><span class="z80-reg">A</span><span>8-bit</span><span>Accumulator &mdash; main arithmetic register</span></div>
						<div class="z80-reg-row"><span class="z80-reg">F</span><span>8-bit</span><span>Flags (zero, carry, sign, parity)</span></div>
						<div class="z80-reg-row"><span class="z80-reg">B, C</span><span>8-bit</span><span>General purpose; B is loop counter for DJNZ</span></div>
						<div class="z80-reg-row"><span class="z80-reg">D, E</span><span>8-bit</span><span>General purpose; DE = destination for block ops</span></div>
						<div class="z80-reg-row"><span class="z80-reg">H, L</span><span>8-bit</span><span>General purpose; HL = primary memory pointer</span></div>
						<div class="z80-reg-row"><span class="z80-reg">SP</span><span>16-bit</span><span>Stack pointer</span></div>
						<div class="z80-reg-row"><span class="z80-reg">BC, DE, HL</span><span>16-bit</span><span>Register pairs (B+C, D+E, H+L)</span></div>
						<div class="z80-reg-row"><span class="z80-reg">AF</span><span>16-bit</span><span>Accumulator + flags pair</span></div>
					</div>

					<h4>Notation</h4>
					<div class="z80-notation-table">
						<div class="z80-notation-row head"><span>Syntax</span><span>Meaning</span></div>
						<div class="z80-notation-row"><span class="z80-reg">nn</span><span>16-bit immediate value (e.g. $1234)</span></div>
						<div class="z80-notation-row"><span class="z80-reg">n</span><span>8-bit immediate value (e.g. $FF)</span></div>
						<div class="z80-notation-row"><span class="z80-reg">d</span><span>Signed offset for relative jumps</span></div>
						<div class="z80-notation-row"><span class="z80-reg">(HL)</span><span>Memory at address in HL</span></div>
						<div class="z80-notation-row"><span class="z80-reg">(nn)</span><span>Memory at absolute address nn</span></div>
						<div class="z80-notation-row"><span class="z80-reg">(SP)</span><span>Memory at top of stack</span></div>
					</div>

					<h4>Instruction Reference</h4>

					<div class="z80-cat">
						<div class="z80-cat-label">Data Movement</div>
						<div class="z80-instr-grid">
							<div class="z80-instr"><span class="z80-op">LD x,y</span> Copy y into x</div>
							<div class="z80-instr"><span class="z80-op">PUSH rr</span> Push 16-bit pair onto stack</div>
							<div class="z80-instr"><span class="z80-op">POP rr</span> Pop 16 bits from stack into pair</div>
							<div class="z80-instr"><span class="z80-op">EX x,y</span> Exchange (swap) x and y</div>
						</div>
					</div>

					<div class="z80-cat">
						<div class="z80-cat-label">Arithmetic &amp; Logic</div>
						<div class="z80-instr-grid">
							<div class="z80-instr"><span class="z80-op">ADD A,x</span> A = A + x</div>
							<div class="z80-instr"><span class="z80-op">ADC A,x</span> A = A + x + carry</div>
							<div class="z80-instr"><span class="z80-op">SUB x</span> A = A &minus; x</div>
							<div class="z80-instr"><span class="z80-op">SBC A,x</span> A = A &minus; x &minus; carry</div>
							<div class="z80-instr"><span class="z80-op">AND x</span> A = A &amp; x</div>
							<div class="z80-instr"><span class="z80-op">OR x</span> A = A | x</div>
							<div class="z80-instr"><span class="z80-op">XOR x</span> A = A ^ x</div>
							<div class="z80-instr"><span class="z80-op">CP x</span> Compare A with x (sets flags)</div>
							<div class="z80-instr"><span class="z80-op">INC x</span> x = x + 1</div>
							<div class="z80-instr"><span class="z80-op">DEC x</span> x = x &minus; 1</div>
						</div>
					</div>

					<div class="z80-cat">
						<div class="z80-cat-label">Flow Control</div>
						<div class="z80-instr-grid">
							<div class="z80-instr"><span class="z80-op">JP nn</span> Jump to address</div>
							<div class="z80-instr"><span class="z80-op">JR d</span> Jump relative by offset d</div>
							<div class="z80-instr"><span class="z80-op">DJNZ d</span> Decrement B, jump if B &ne; 0</div>
							<div class="z80-instr"><span class="z80-op">CALL nn</span> Push PC, jump to address</div>
							<div class="z80-instr"><span class="z80-op">RET</span> Pop PC (return from call)</div>
							<div class="z80-instr"><span class="z80-op">NOP</span> No operation (do nothing)</div>
							<div class="z80-instr"><span class="z80-op">HALT</span> Stop execution</div>
						</div>
					</div>

					<div class="z80-cat">
						<div class="z80-cat-label">Block Operations</div>
						<div class="z80-instr-grid">
							<div class="z80-instr"><span class="z80-op">LDI</span> Copy (HL)&rarr;(DE), inc both, dec BC</div>
							<div class="z80-instr"><span class="z80-op">LDD</span> Copy (HL)&rarr;(DE), dec both, dec BC</div>
							<div class="z80-instr"><span class="z80-op">LDIR</span> LDI repeated until BC = 0</div>
							<div class="z80-instr"><span class="z80-op">LDDR</span> LDD repeated until BC = 0</div>
						</div>
					</div>

					<div class="z80-cat">
						<div class="z80-cat-label">Bit &amp; Rotate</div>
						<div class="z80-instr-grid">
							<div class="z80-instr"><span class="z80-op">BIT n,x</span> Test bit n of x</div>
							<div class="z80-instr"><span class="z80-op">SET n,x</span> Set bit n of x</div>
							<div class="z80-instr"><span class="z80-op">RES n,x</span> Reset bit n of x</div>
							<div class="z80-instr"><span class="z80-op">RL x</span> Rotate left through carry</div>
							<div class="z80-instr"><span class="z80-op">RR x</span> Rotate right through carry</div>
							<div class="z80-instr"><span class="z80-op">SLA x</span> Shift left arithmetic</div>
							<div class="z80-instr"><span class="z80-op">SRL x</span> Shift right logical</div>
						</div>
					</div>

					<h4>Why POP HL &amp; EX (SP),HL Win</h4>
					<p>
						<span class="z80-op">POP HL</span> reads 2 bytes from memory via the stack
						pointer. <span class="z80-op">EX (SP),HL</span> writes them forward to the
						next position. Repeating this copies the cell's own bytes into its neighbor
						&mdash; a minimal self-replicating loop. Once one cell contains this pattern,
						it spreads exponentially.
					</p>

					<Mermaid chart={`
graph TD
    A("POP HL\nread 2 bytes from memory") -->|"HL now holds data"| B("EX (SP),HL\nwrite 2 bytes forward")
    B -->|"repeat"| A
`} />
				{:else if helpTab === 'params'}
					<h4>Seed</h4>
					<p>
						The random seed used to initialize the grid. Same seed produces the same
						starting state. Change it and press Reset to try different initial conditions.
					</p>

					<h4>Mutation Rate</h4>
					<p>
						Controls the probability of random byte flips after each execution step.
						Higher slider values mean more mutations. Too low and replicators can't
						emerge; too high and they can't survive. Sweet spot is usually 3&ndash;5.
					</p>

					<h4>Pairs / Batch</h4>
					<p>
						How many cell pairs are selected and executed each simulation step. Higher
						values speed up evolution but use more GPU time. At 5000, roughly 10,000 of
						the 40,000 cells are updated per step.
					</p>

					<h4>Colormap</h4>
					<p>
						Choose between four visual themes: Default (warm opcode-aware), Ocean (cool
						blues), and Thermal (heat map).
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
						<strong>Algocell</strong> is a WebGPU-accelerated artificial life simulator
						that runs Z80 machine code on a 200&times;200 grid of cells. Self-replicating
						programs spontaneously emerge from random noise.
					</p>
					<p>
						Based on the paper by
						<a href="https://arxiv.org/abs/2406.19108" target="_blank" rel="noopener" class="help-link">Hartley &amp; Colton (2024)</a>
						and the
						<a href="https://github.com/znah/zff" target="_blank" rel="noopener" class="help-link">original implementation</a>
						by Alexander Mordvintsev. This browser version uses WebGPU compute shaders
						instead of Python/JAX, so it runs anywhere with no setup required.
					</p>
					<p>
						Developed by <strong>Neo Mohsenvand</strong> with the help of
						<a href="https://claude.ai" target="_blank" rel="noopener" class="help-link">Claude Code</a>.
					</p>
					<p class="cmap-note" style="margin-top: 12px;">
						<a href="https://github.com/NeoVand/algocell" target="_blank" rel="noopener" class="help-link">GitHub</a>
						&middot; SvelteKit &middot; TypeScript &middot; WebGPU &middot; Tailwind CSS
					</p>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
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
		backdrop-filter: blur(12px);
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
		backdrop-filter: blur(12px);
		font-size: 11px;
		font-family: monospace;
		width: 252px;
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
		width: 18px;
		height: 18px;
		border: none;
		background: transparent;
		cursor: pointer;
		color: var(--text-subtle);
		margin-left: auto;
		border-radius: 4px;
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
		background: rgba(255, 235, 210, 0.04);
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
		-webkit-box-orient: vertical;
	}

	/* ── Panels (shared) ── */
	.panel {
		position: fixed;
		z-index: 35;
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
		border-radius: 12px;
		backdrop-filter: blur(12px);
		padding: 14px;
	}
	.panel-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 12px;
	}
	.panel-title {
		display: inline-flex;
		align-items: center;
		gap: 4px;
		font-size: 11px;
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
		margin-bottom: 14px;
	}
	.param:last-child {
		margin-bottom: 0;
	}
	.param-head {
		display: flex;
		align-items: center;
		gap: 4px;
		margin-bottom: 6px;
	}
	.param-label {
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-subtle);
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
		font-size: 13px;
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
		background: rgba(255, 255, 255, 0.04);
		border: 1px solid var(--border-muted);
		border-radius: 6px;
		padding: 0 8px;
		height: 26px;
		font-size: 12px;
		font-family: monospace;
		color: var(--accent);
		outline: none;
		transition: border-color 0.15s;
		-moz-appearance: textfield;
	}
	.seed-input::-webkit-inner-spin-button,
	.seed-input::-webkit-outer-spin-button {
		-webkit-appearance: none;
		margin: 0;
	}
	.seed-input:focus {
		border-color: var(--border-muted);
	}
	.seed-apply {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 26px;
		height: 26px;
		background: rgba(255, 255, 255, 0.04);
		border: 1px solid var(--border-muted);
		border-radius: 6px;
		color: var(--accent);
		cursor: pointer;
		transition: all 0.15s;
		flex-shrink: 0;
	}
	.seed-apply:hover {
		background: var(--bg-hover);
		border-color: var(--accent);
	}

	/* Custom slider */
	.slider-track-wrap {
		position: relative;
		padding: 4px 0;
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
		gap: 3px;
		padding: 5px 0 4px;
		font-size: 10px;
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
		background: linear-gradient(90deg, #1a0404, #c85a5a, #c8a05a, #5ac85a, #5a8ac8, #8a5ac8, #c85a8a);
	}
	.cmap-preview-ocean {
		background: linear-gradient(90deg, #0a1218, #2a4a5a, #3a7a8a, #4a9aaa, #5ab0c0, #6ac0d0, #8ad0e0);
	}
	.cmap-preview-thermal {
		background: linear-gradient(90deg, #000000, #4a0a2a, #8a1a1a, #c84a0a, #e88a2a, #f0c050, #f8f0c0);
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

	/* ── Genome tooltip ── */
	.genome-tip {
		position: fixed;
		z-index: 45;
		pointer-events: none;
		overflow: hidden;
		background: #0a0a0e;
		padding: 2px;
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
		opacity: 0.5;
		font-weight: 400;
		font-size: 10px;
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
		transition: color 0.15s, border-color 0.15s;
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
		border: 1px solid rgba(255,255,255,0.08);
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
		border-bottom: 1px solid rgba(255,255,255,0.03);
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
		border-bottom: 1px solid rgba(255,255,255,0.03);
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
		.toolbar {
			top: auto;
			bottom: 12px;
			right: 12px;
			flex-direction: column;
			width: auto;
		}
		.toolbar-buttons {
			flex-direction: column;
			max-width: unset;
		}
		.toolbar-buttons.hidden {
			max-height: 0;
			max-width: unset;
			opacity: 0;
			overflow: hidden;
			pointer-events: none;
		}
		.tb-sep {
			width: 14px;
			height: 1px;
			margin: 1px 0;
		}
		.speed-menu {
			top: 50%;
			left: auto;
			right: calc(100% + 6px);
			transform: translateY(-50%);
		}
		.info-bar {
			bottom: auto;
			top: 12px;
			left: 12px;
			min-width: 200px;
		}
		.settings-panel {
			top: auto;
			bottom: 56px;
			right: 12px;
			width: 220px;
		}
	}
</style>

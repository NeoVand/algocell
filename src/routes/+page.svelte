<script lang="ts">
	import { GPUEngine } from '$lib/gpu/engine';
	import { SOUP_WIDTH, SOUP_HEIGHT, DEFAULT_SEED, DEFAULT_NOISE_EXP, MAX_BATCH_PAIR_N } from '$lib/sim/constants';
	import { disassemble, byteToMnemonic } from '$lib/z80-disasm';
	import { getCellData } from '$lib/sim/soup';
	import { unpackRGBA } from '$lib/colormap';
	import { createDefaultColormap } from '$lib/colormap';

	const colormap = createDefaultColormap();

	let canvas: HTMLCanvasElement;
	let canvasContainer: HTMLDivElement;
	let engine = $state<GPUEngine | null>(null);
	let gpuError = $state<string | null>(null);

	// Controls
	let seed = $state(DEFAULT_SEED);
	let noiseExp = $state(DEFAULT_NOISE_EXP);
	let playing = $state(true);
	let speed = $state(1);

	// Stats
	let batchCount = $state(0);
	let opsPerSec = $state(0);
	let topBytes: { byte: number; count: number; mnemonic: string }[] = $state([]);
	let showStats = $state(false);
	let showHelp = $state(false);
	let showSpeedMenu = $state(false);
	let showSettings = $state(false);

	// Stats history for sparklines
	let statsHistory: { batch: number; nop: number; topReplicator: number }[] = $state([]);
	const MAX_HISTORY = 500;

	// Hover / genome tooltip
	let hoveredCell = $state(-1);
	let cellData = $state<Uint8Array | null>(null);
	let disasmLines: ReturnType<typeof disassemble> = $state([]);
	let mouseX = $state(0);
	let mouseY = $state(0);
	let canvasW = $state(800);
	let canvasH = $state(800);

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

	let nopLine = $derived(buildSparklinePath(statsHistory.map((s) => s.nop), 200, 60));
	let replicatorLine = $derived(buildSparklinePath(statsHistory.map((s) => s.topReplicator), 200, 60));

	let tooltipStyle = $derived(computeTooltipStyle(mouseX, mouseY, canvasW, canvasH));

	let cellCoords = $derived(
		hoveredCell >= 0
			? { x: hoveredCell % SOUP_WIDTH, y: Math.floor(hoveredCell / SOUP_WIDTH) }
			: null
	);

	let genomeCells = $derived(
		cellData && disasmLines.length > 0 ? buildGenomeGrid(cellData, disasmLines) : []
	);

	function computeTooltipStyle(mx: number, my: number, _cw: number, _ch: number): string {
		const tooltipW = 280;
		const tooltipH = 180;
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

	function buildSparklinePath(values: number[], w: number, h: number): string {
		if (values.length < 2) return '';
		const max = Math.max(...values, 1);
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

		const eng = new GPUEngine(seed);
		eng.noiseExp = noiseExp;

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

			if (frameCount % 30 === 0 && !statsLoading) {
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

			// Update sparkline history
			const nopCount = counts[0] || 0;
			let topReplicatorCount = 0;
			for (let i = 1; i < 256; i++) {
				if (counts[i] > topReplicatorCount) topReplicatorCount = counts[i];
			}
			statsHistory = [
				...statsHistory.slice(-(MAX_HISTORY - 1)),
				{ batch: batchCount, nop: nopCount, topReplicator: topReplicatorCount }
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

		if (cell >= 0 && cell !== hoveredCell) {
			hoveredCell = cell;
			engine.setHoverCell(cell);
			engine.readSoupData().then((soupData) => {
				const data = getCellData(soupData, cell);
				cellData = data;
				disasmLines = disassemble(data);
			});
		} else if (cell < 0) {
			hoveredCell = -1;
			cellData = null;
			disasmLines = [];
			engine.setHoverCell(-1);
		}
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
			engine.pairCount = val;
		}
	}

	function togglePlay() {
		playing = !playing;
	}

	function setSpeed(s: number) {
		speed = s;
		showSpeedMenu = false;
	}

	function formatNumber(n: number): string {
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

	interface GenomeCell {
		byteVal: number;
		label: string;
		isOpcode: boolean;
	}

	function buildGenomeGrid(data: Uint8Array, disasm: ReturnType<typeof disassemble>): GenomeCell[] {
		const cells: GenomeCell[] = [];
		// Map each byte index to its role
		const byteInfo = new Map<number, { label: string; isOpcode: boolean }>();

		for (const line of disasm) {
			// First byte of instruction is the opcode
			byteInfo.set(line.offset, { label: line.mnemonic, isOpcode: true });
			// Remaining bytes are operands
			for (let i = 1; i < line.length && line.offset + i < data.length; i++) {
				byteInfo.set(line.offset + i, { label: hexByte(data[line.offset + i]), isOpcode: false });
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
				showStats = !showStats;
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
<div
	bind:this={canvasContainer}
	class="fixed inset-0"
>
	{#if gpuError}
		<div class="absolute inset-0 z-50 flex items-center justify-center" style="background:var(--bg-panel)">
			<p class="text-red-400 text-center max-w-md px-8">{gpuError}</p>
		</div>
	{/if}
	<canvas
		bind:this={canvas}
		class="block w-full h-full"
		style="cursor:{isPanning ? 'grabbing' : 'grab'}"
		onmousemove={handleCanvasMouseMove}
		onmouseleave={handleCanvasMouseLeave}
		onmousedown={handleCanvasMouseDown}
		onmouseup={handleCanvasMouseUp}
		onwheel={handleCanvasWheel}
		ondblclick={handleCanvasDblClick}
		oncontextmenu={(e) => e.preventDefault()}
	></canvas>
</div>

<!-- Top-right toolbar -->
<div class="toolbar">
	<!-- Play/Pause -->
	<button
		class="toolbar-btn"
		style="color:var(--accent-cyan)"
		title="{playing ? 'Pause' : 'Play'} (Space)"
		onclick={togglePlay}
	>
		{#if playing}
			<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
				<rect x="2" y="1" width="3.5" height="12" rx="1"/>
				<rect x="8.5" y="1" width="3.5" height="12" rx="1"/>
			</svg>
		{:else}
			<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
				<path d="M3 1.5v11l9-5.5z"/>
			</svg>
		{/if}
	</button>

	<!-- Speed -->
	<div class="relative">
		<button
			class="toolbar-btn text-xs font-mono"
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
						class="speed-option"
						class:active={speed === s}
						onclick={() => setSpeed(s)}
					>{s}x</button>
				{/each}
			</div>
		{/if}
	</div>

	<div class="toolbar-sep"></div>

	<!-- Reset -->
	<button
		class="toolbar-btn"
		style="color:var(--accent-amber)"
		title="Reset (R)"
		onclick={handleReset}
	>
		<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5">
			<path d="M2 7a5 5 0 1 1 1 3" stroke-linecap="round"/>
			<path d="M2 3v4h4" stroke-linecap="round" stroke-linejoin="round"/>
		</svg>
	</button>

	<!-- Stats toggle -->
	<button
		class="toolbar-btn"
		style="color:{showStats ? 'var(--accent)' : 'var(--text-subtle)'}"
		title="Stats (S)"
		onclick={() => (showStats = !showStats)}
	>
		<svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
			<rect x="1" y="8" width="2.5" height="5" rx="0.5"/>
			<rect x="5.5" y="4" width="2.5" height="9" rx="0.5"/>
			<rect x="10" y="1" width="2.5" height="12" rx="0.5"/>
		</svg>
	</button>

	<!-- Settings -->
	<button
		class="toolbar-btn"
		style="color:{showSettings ? 'var(--accent)' : 'var(--text-subtle)'}"
		title="Settings"
		onclick={() => (showSettings = !showSettings)}
	>
		<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.3">
			<circle cx="7" cy="7" r="2.2"/>
			<path d="M7 1.5v1.2M7 11.3v1.2M1.5 7h1.2M11.3 7h1.2M3.1 3.1l.85.85M10.05 10.05l.85.85M3.1 10.9l.85-.85M10.05 3.95l.85-.85" stroke-linecap="round"/>
		</svg>
	</button>

	<!-- Help -->
	<button
		class="toolbar-btn"
		style="color:var(--text-subtle)"
		title="Help (H)"
		onclick={() => (showHelp = !showHelp)}
	>
		<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5">
			<circle cx="7" cy="7" r="5.5"/>
			<path d="M5.5 5.5a1.75 1.75 0 0 1 3.25 1c0 1-1.5 1.25-1.5 2.25" stroke-linecap="round"/>
			<circle cx="7" cy="10.5" r="0.5" fill="currentColor" stroke="none"/>
		</svg>
	</button>
</div>

<!-- Bottom-left info bar -->
<div class="info-bar">
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
		<span class="info-label">zoom</span>
		<span class="info-value">{zoomPercent}%</span>
	</span>
</div>

<!-- Stats panel (right side, toggleable) -->
{#if showStats}
	<div class="stats-panel">
		<div class="stats-header">
			<span class="stats-title">Statistics</span>
			<button class="stats-close" aria-label="Close stats" onclick={() => (showStats = false)}>
				<svg width="10" height="10" viewBox="0 0 10 10" stroke="currentColor" stroke-width="1.5">
					<path d="M2 2l6 6M8 2l-6 6" stroke-linecap="round"/>
				</svg>
			</button>
		</div>

		<!-- Sparkline chart -->
		{#if statsHistory.length > 1}
			<div class="sparkline-container">
				<div class="sparkline-labels">
					<span style="color:var(--text-subtle)">NOP</span>
					<span style="color:var(--accent-cyan)">Top replicator</span>
				</div>
				<svg viewBox="0 0 200 60" class="sparkline-svg">
					{#if nopLine}
						<polyline points={nopLine.replace(/[ML]/g, (m) => m === 'M' ? '' : ' ').trim()} fill="none" stroke="var(--text-subtle)" stroke-width="1" opacity="0.5"/>
					{/if}
					{#if replicatorLine}
						<polyline points={replicatorLine.replace(/[ML]/g, (m) => m === 'M' ? '' : ' ').trim()} fill="none" stroke="var(--accent-cyan)" stroke-width="1.5" opacity="0.8"/>
					{/if}
				</svg>
			</div>
		{/if}

		<!-- Top bytes -->
		<div class="stats-section-title">Top byte frequencies</div>
		<div class="byte-list">
			{#each topBytes as entry (entry.byte)}
				<div class="byte-row">
					<span class="byte-dot" style="background:{byteColor(entry.byte)}"></span>
					<span class="byte-hex">{hexByte(entry.byte)}</span>
					<span class="byte-count">{formatNumber(entry.count)}</span>
					<span class="byte-mnemonic">{entry.mnemonic || '-'}</span>
				</div>
			{/each}
			{#if topBytes.length === 0}
				<div class="byte-empty">No data yet</div>
			{/if}
		</div>
	</div>
{/if}

<!-- Settings panel -->
{#if showSettings}
	<div class="settings-panel">
		<div class="stats-header">
			<span class="stats-title">Parameters</span>
			<button class="stats-close" aria-label="Close settings" onclick={() => (showSettings = false)}>
				<svg width="10" height="10" viewBox="0 0 10 10" stroke="currentColor" stroke-width="1.5">
					<path d="M2 2l6 6M8 2l-6 6" stroke-linecap="round"/>
				</svg>
			</button>
		</div>

		<div class="setting-row">
			<label class="setting-label" for="seed-input">Seed</label>
			<div class="setting-control">
				<input
					id="seed-input"
					type="number"
					class="setting-input"
					value={seed}
					min="0"
					onchange={handleSeedChange}
				/>
				<button class="setting-apply" title="Apply seed (resets simulation)" onclick={handleReset}>Apply</button>
			</div>
		</div>

		<div class="setting-row">
			<label class="setting-label" for="noise-input">
				Mutation rate
				<span class="setting-hint">1/2^{noiseExp} = {(1 / Math.pow(2, noiseExp)).toFixed(4)}</span>
			</label>
			<input
				id="noise-input"
				type="range"
				class="setting-range"
				value={noiseExp}
				min="1"
				max="10"
				step="1"
				oninput={handleNoiseExpChange}
			/>
		</div>

		<div class="setting-row">
			<label class="setting-label" for="pairs-input">
				Pairs per batch
				<span class="setting-hint">{engine?.pairCount ?? MAX_BATCH_PAIR_N}</span>
			</label>
			<input
				id="pairs-input"
				type="range"
				class="setting-range"
				value={engine?.pairCount ?? MAX_BATCH_PAIR_N}
				min="256"
				max={MAX_BATCH_PAIR_N}
				step="256"
				oninput={handlePairCountChange}
			/>
		</div>
	</div>
{/if}

<!-- Genome tooltip -->
{#if hoveredCell >= 0 && cellData && !isPanning}
	<div class="genome-tooltip" style={tooltipStyle}>
		{#if cellCoords}
			<div class="genome-coords">({cellCoords.x}, {cellCoords.y})</div>
		{/if}

		<!-- 4x4 genome grid -->
		<div class="genome-grid">
			{#each genomeCells as cell, i (i)}
				<div
					class="genome-cell"
					class:is-operand={!cell.isOpcode}
					style="background:{byteColor(cell.byteVal)}"
				>
					<span class="genome-cell-text">{cell.label}</span>
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
				<span>About this experiment</span>
				<button class="stats-close" aria-label="Close help" onclick={() => (showHelp = false)}>
					<svg width="10" height="10" viewBox="0 0 10 10" stroke="currentColor" stroke-width="1.5">
						<path d="M2 2l6 6M8 2l-6 6" stroke-linecap="round"/>
					</svg>
				</button>
			</div>
			<div class="modal-body">
				<p>
					A 200x200 grid of cells, each containing 16 random bytes interpreted as Z80 machine code.
					Every simulation step, random pairs of adjacent cells are selected. The 32-byte pair is
					executed as a Z80 program for 128 steps, then the result is written back.
				</p>
				<p>
					With a small mutation rate (random byte flips), self-replicating programs spontaneously
					emerge and compete for space -- a computational analogue to the origin of life.
				</p>
				<p>
					Watch for the phase transition: initially all bytes are uniformly distributed (NOP count
					is ~1/256 of total). When replicators emerge, certain byte patterns dominate and the
					distribution becomes highly skewed.
				</p>
				<div class="modal-shortcuts">
					<div class="shortcut"><kbd>Space</kbd> Play / Pause</div>
					<div class="shortcut"><kbd>R</kbd> Reset</div>
					<div class="shortcut"><kbd>S</kbd> Toggle stats</div>
					<div class="shortcut"><kbd>H</kbd> Toggle help</div>
					<div class="shortcut"><kbd>F</kbd> Fit view</div>
					<div class="shortcut"><kbd>Scroll</kbd> Zoom</div>
					<div class="shortcut"><kbd>Drag</kbd> Pan</div>
					<div class="shortcut"><kbd>Dbl-click</kbd> Reset view</div>
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	/* Toolbar */
	.toolbar {
		position: fixed;
		top: 12px;
		right: 12px;
		z-index: 40;
		display: flex;
		align-items: center;
		gap: 2px;
		padding: 4px;
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
		border-radius: 20px;
		backdrop-filter: blur(12px);
	}
	.toolbar-btn {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 30px;
		height: 30px;
		border-radius: 14px;
		border: none;
		background: transparent;
		cursor: pointer;
		transition: background 0.15s;
	}
	.toolbar-btn:hover {
		background: var(--bg-hover);
	}
	.toolbar-sep {
		width: 1px;
		height: 16px;
		background: var(--border-subtle);
		margin: 0 2px;
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
	.speed-option {
		padding: 6px 12px;
		font-size: 11px;
		font-family: monospace;
		color: var(--text-secondary);
		background: transparent;
		border: none;
		cursor: pointer;
		text-align: center;
	}
	.speed-option:hover {
		background: var(--bg-hover);
	}
	.speed-option.active {
		color: var(--accent-cyan);
	}

	/* Info bar */
	.info-bar {
		position: fixed;
		bottom: 12px;
		left: 12px;
		z-index: 40;
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 6px 12px;
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
		border-radius: 14px;
		backdrop-filter: blur(12px);
		font-size: 11px;
		font-family: monospace;
	}
	.info-item {
		display: flex;
		align-items: center;
		gap: 4px;
	}
	.info-label {
		color: var(--text-subtle);
		text-transform: uppercase;
		font-size: 9px;
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

	/* Stats panel */
	.stats-panel {
		position: fixed;
		top: 12px;
		right: 56px;
		z-index: 35;
		width: 280px;
		max-height: calc(100vh - 24px);
		overflow-y: auto;
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
		border-radius: 12px;
		backdrop-filter: blur(12px);
		padding: 14px;
	}
	.stats-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 12px;
	}
	.stats-title {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-subtle);
		font-weight: 600;
	}
	.stats-close {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 20px;
		height: 20px;
		border-radius: 6px;
		border: none;
		background: transparent;
		color: var(--text-subtle);
		cursor: pointer;
	}
	.stats-close:hover {
		background: var(--bg-hover);
		color: var(--text-secondary);
	}

	/* Sparkline */
	.sparkline-container {
		margin-bottom: 14px;
		padding: 8px;
		background: var(--bg-subtle);
		border-radius: 8px;
	}
	.sparkline-labels {
		display: flex;
		justify-content: space-between;
		font-size: 9px;
		margin-bottom: 4px;
	}
	.sparkline-svg {
		width: 100%;
		height: 60px;
	}

	/* Byte list */
	.stats-section-title {
		font-size: 9px;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-subtle);
		margin-bottom: 6px;
	}
	.byte-list {
		display: flex;
		flex-direction: column;
		gap: 1px;
	}
	.byte-row {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 3px 6px;
		border-radius: 4px;
		font-size: 11px;
		font-family: monospace;
	}
	.byte-row:hover {
		background: var(--bg-hover);
	}
	.byte-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		flex-shrink: 0;
	}
	.byte-hex {
		color: var(--accent-cyan);
		width: 20px;
	}
	.byte-count {
		color: var(--text-secondary);
		width: 48px;
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
	.byte-mnemonic {
		color: var(--text-subtle);
		flex: 1;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.byte-empty {
		color: var(--text-subtle);
		font-size: 11px;
		font-style: italic;
		padding: 4px 6px;
	}

	/* Settings panel */
	.settings-panel {
		position: fixed;
		top: 56px;
		right: 12px;
		z-index: 38;
		width: 260px;
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
		border-radius: 12px;
		backdrop-filter: blur(12px);
		padding: 14px;
	}
	.setting-row {
		margin-bottom: 12px;
	}
	.setting-row:last-child {
		margin-bottom: 0;
	}
	.setting-label {
		display: block;
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-subtle);
		margin-bottom: 4px;
	}
	.setting-hint {
		text-transform: none;
		letter-spacing: 0;
		color: var(--text-muted);
		font-family: monospace;
		font-size: 10px;
		margin-left: 4px;
	}
	.setting-control {
		display: flex;
		gap: 4px;
	}
	.setting-input {
		flex: 1;
		background: var(--bg-subtle);
		border: 1px solid var(--border-muted);
		border-radius: 6px;
		padding: 4px 8px;
		font-size: 12px;
		font-family: monospace;
		color: var(--text-primary);
		outline: none;
		width: 80px;
	}
	.setting-input:focus {
		border-color: var(--accent);
	}
	.setting-apply {
		padding: 4px 10px;
		background: var(--bg-muted);
		border: 1px solid var(--border-muted);
		border-radius: 6px;
		font-size: 10px;
		color: var(--accent-cyan);
		cursor: pointer;
	}
	.setting-apply:hover {
		background: var(--bg-hover);
	}
	.setting-range {
		width: 100%;
		accent-color: var(--accent);
		height: 4px;
	}

	/* Genome tooltip */
	.genome-tooltip {
		position: fixed;
		z-index: 45;
		background: var(--bg-elevated);
		border: 1px solid var(--border-muted);
		border-radius: 10px;
		padding: 10px;
		pointer-events: none;
		backdrop-filter: blur(16px);
		width: 280px;
	}
	.genome-coords {
		font-size: 9px;
		color: var(--text-subtle);
		font-family: monospace;
		margin-bottom: 6px;
	}
	.genome-grid {
		display: grid;
		grid-template-columns: repeat(4, 1fr);
		gap: 3px;
	}
	.genome-cell {
		border-radius: 4px;
		padding: 4px 3px;
		display: flex;
		align-items: center;
		justify-content: center;
		min-height: 28px;
	}
	.genome-cell-text {
		font-size: 8px;
		font-family: monospace;
		color: var(--text-primary);
		text-align: center;
		line-height: 1.15;
		word-break: break-all;
		text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
	}
	.genome-cell.is-operand .genome-cell-text {
		color: var(--text-subtle);
		opacity: 0.6;
		font-size: 9px;
	}

	/* Help modal */
	.modal-backdrop {
		position: fixed;
		inset: 0;
		z-index: 50;
		display: flex;
		align-items: center;
		justify-content: center;
		background: rgba(0, 0, 0, 0.6);
		backdrop-filter: blur(4px);
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
		padding: 14px 16px;
		border-bottom: 1px solid var(--border-subtle);
		font-size: 13px;
		font-weight: 600;
		color: var(--text-primary);
	}
	.modal-body {
		padding: 16px;
		font-size: 12px;
		line-height: 1.6;
		color: var(--text-muted);
	}
	.modal-body p {
		margin-bottom: 10px;
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
</style>

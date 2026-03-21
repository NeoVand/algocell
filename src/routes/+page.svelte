<script lang="ts">
	import { GPUEngine } from '$lib/gpu/engine';
	import { SOUP_WIDTH, SOUP_HEIGHT, TILE_SIZE, DEFAULT_SEED, DEFAULT_NOISE_EXP } from '$lib/sim/constants';
	import { disassemble, byteToMnemonic } from '$lib/z80-disasm';
	import { getCellData } from '$lib/sim/soup';

	const CANVAS_SIZE = SOUP_WIDTH * TILE_SIZE; // 800

	let canvas: HTMLCanvasElement;
	let engine: GPUEngine | null = $state(null);
	let gpuError: string | null = $state(null);

	// Controls
	let seed = $state(DEFAULT_SEED);
	let noiseExp = $state(DEFAULT_NOISE_EXP);
	let playing = $state(true);
	let speed = $state(1); // steps per frame

	// Stats
	let batchCount = $state(0);
	let opsPerSec = $state(0);
	let topBytes: { byte: number; count: number; mnemonic: string }[] = $state([]);

	// Hover / disassembly
	let hoveredCell = $state(-1);
	let cellData: Uint8Array | null = $state(null);
	let disasmLines: ReturnType<typeof disassemble> = $state([]);

	let frameCount = 0;
	let animFrameId: number | undefined;
	let statsLoading = false;

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

			// Read stats every 30 frames (avoid race on staging buffer)
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
		} catch (e) {
			console.warn('Stats readback failed:', e);
		} finally {
			statsLoading = false;
		}
	}

	function handleMouseMove(e: MouseEvent) {
		const rect = canvas.getBoundingClientRect();
		const scaleX = CANVAS_SIZE / rect.width;
		const scaleY = CANVAS_SIZE / rect.height;
		const px = (e.clientX - rect.left) * scaleX;
		const py = (e.clientY - rect.top) * scaleY;
		const cellX = Math.floor(px / TILE_SIZE);
		const cellY = Math.floor(py / TILE_SIZE);

		if (cellX >= 0 && cellX < SOUP_WIDTH && cellY >= 0 && cellY < SOUP_HEIGHT) {
			const cell = cellY * SOUP_WIDTH + cellX;
			hoveredCell = cell;
			engine?.setHoverCell(cell);

			engine?.readSoupData().then((soupData) => {
				const data = getCellData(soupData, cell);
				cellData = data;
				disasmLines = disassemble(data);
			});
		}
	}

	function handleMouseLeave() {
		hoveredCell = -1;
		cellData = null;
		disasmLines = [];
		engine?.setHoverCell(-1);
	}

	function handleReset() {
		if (!engine) return;
		engine.reset(seed);
		batchCount = 0;
		opsPerSec = 0;
		topBytes = [];
	}

	function handleNoiseChange(value: number) {
		noiseExp = value;
		if (engine) {
			engine.noiseExp = value;
		}
	}

	function togglePlay() {
		playing = !playing;
	}

	function formatNumber(n: number): string {
		if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
		if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
		return n.toString();
	}

	function hexByte(b: number): string {
		return b.toString(16).toUpperCase().padStart(2, '0');
	}
</script>

<svelte:window
	onkeydown={(e) => {
		if (e.code === 'Space' && e.target === document.body) {
			e.preventDefault();
			togglePlay();
		}
	}}
/>

<svelte:head>
	<title>Computational Life: Z80 Primordial Soup</title>
</svelte:head>

<div class="min-h-screen bg-gray-950 text-gray-100 flex flex-col">
	<!-- Header -->
	<header class="px-6 py-4 border-b border-gray-800">
		<h1 class="text-2xl font-bold tracking-tight text-gray-50">
			Computational Life: Z80 Primordial Soup
		</h1>
		<p class="text-sm text-gray-500 mt-1">
			WebGPU-powered real-time simulation of self-replicating Z80 programs emerging from random bytes
		</p>
	</header>

	<div class="flex flex-1 gap-4 p-4 lg:p-6 max-w-[1400px] mx-auto w-full">
		<!-- Left: Canvas + Controls -->
		<div class="flex flex-col gap-4 min-w-0 flex-1">
			<!-- Canvas -->
			<div class="relative rounded-lg overflow-hidden border border-gray-800 bg-black aspect-square max-w-[800px]">
				{#if gpuError}
					<div class="absolute inset-0 flex items-center justify-center bg-red-950/50 p-8">
						<p class="text-red-400 text-center">{gpuError}</p>
					</div>
				{/if}
				<canvas
					bind:this={canvas}
					width={CANVAS_SIZE}
					height={CANVAS_SIZE}
					class="block cursor-crosshair w-full h-full"
					onmousemove={handleMouseMove}
					onmouseleave={handleMouseLeave}
				></canvas>
			</div>

			<!-- Controls -->
			<div class="flex flex-wrap items-center gap-4 bg-gray-900 rounded-lg p-4 border border-gray-800">
				<!-- Seed -->
				<div class="flex items-center gap-2">
					<label for="seed" class="text-xs text-gray-400 uppercase tracking-wider">Seed</label>
					<input
						id="seed"
						type="number"
						bind:value={seed}
						class="w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
					/>
				</div>

				<!-- Noise -->
				<div class="flex items-center gap-2">
					<label for="noise" class="text-xs text-gray-400 uppercase tracking-wider">Noise</label>
					<input
						id="noise"
						type="range"
						min="0"
						max="10"
						step="1"
						value={noiseExp}
						oninput={(e) => handleNoiseChange(parseInt(e.currentTarget.value))}
						class="w-24 accent-blue-500"
					/>
					<span class="text-xs text-gray-500 w-14 tabular-nums">1/2^{noiseExp}</span>
				</div>

				<!-- Speed -->
				<div class="flex items-center gap-2">
					<label for="speed" class="text-xs text-gray-400 uppercase tracking-wider">Speed</label>
					<select
						id="speed"
						bind:value={speed}
						class="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
					>
						<option value={1}>1x</option>
						<option value={2}>2x</option>
						<option value={4}>4x</option>
						<option value={8}>8x</option>
					</select>
				</div>

				<!-- Play/Pause -->
				<button
					onclick={togglePlay}
					class="px-4 py-1.5 rounded text-sm font-medium transition-colors {playing
						? 'bg-amber-600 hover:bg-amber-500 text-white'
						: 'bg-emerald-600 hover:bg-emerald-500 text-white'}"
				>
					{playing ? 'Pause' : 'Play'}
				</button>

				<!-- Reset -->
				<button
					onclick={handleReset}
					class="px-4 py-1.5 rounded text-sm font-medium bg-gray-700 hover:bg-gray-600 text-gray-200 transition-colors"
				>
					Reset
				</button>
			</div>
		</div>

		<!-- Right: Stats + Disassembly -->
		<div class="flex flex-col gap-4 w-72 lg:w-80 shrink-0">
			<!-- Stats Panel -->
			<div class="bg-gray-900 rounded-lg border border-gray-800 p-4">
				<h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">Statistics</h2>

				<div class="grid grid-cols-2 gap-3 mb-4">
					<div class="bg-gray-800/50 rounded p-2">
						<div class="text-xs text-gray-500">Batches</div>
						<div class="text-lg font-mono text-gray-200 tabular-nums">{formatNumber(batchCount)}</div>
					</div>
					<div class="bg-gray-800/50 rounded p-2">
						<div class="text-xs text-gray-500">Ops/sec</div>
						<div class="text-lg font-mono text-gray-200 tabular-nums">{formatNumber(opsPerSec)}</div>
					</div>
				</div>

				<!-- Top bytes -->
				<h3 class="text-xs text-gray-500 uppercase tracking-wider mb-2">Top Byte Values</h3>
				<div class="space-y-px max-h-[400px] overflow-y-auto">
					{#each topBytes as entry, i (entry.byte)}
						<div class="flex items-center gap-2 text-xs py-1 px-2 rounded {i % 2 === 0 ? 'bg-gray-800/30' : ''}">
							<span class="font-mono text-blue-400 w-6">{hexByte(entry.byte)}</span>
							<span class="font-mono text-gray-300 tabular-nums w-16 text-right">{formatNumber(entry.count)}</span>
							<span class="text-gray-500 truncate flex-1">{entry.mnemonic || '-'}</span>
						</div>
					{/each}
					{#if topBytes.length === 0}
						<p class="text-gray-600 text-xs italic">No data yet</p>
					{/if}
				</div>
			</div>

			<!-- Disassembly Panel -->
			<div class="bg-gray-900 rounded-lg border border-gray-800 p-4 flex-1">
				<h2 class="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
					Disassembly
					{#if hoveredCell >= 0}
						<span class="text-gray-600 font-normal normal-case">
							cell {hoveredCell} ({hoveredCell % SOUP_WIDTH}, {Math.floor(hoveredCell / SOUP_WIDTH)})
						</span>
					{/if}
				</h2>

				{#if cellData && disasmLines.length > 0}
					<!-- Raw bytes -->
					<div class="mb-3 font-mono text-xs text-gray-500 flex flex-wrap gap-1">
						{#each Array.from(cellData) as b, i (i)}
							<span class="bg-gray-800 px-1 rounded">{hexByte(b)}</span>
						{/each}
					</div>

					<!-- Disassembled instructions -->
					<div class="space-y-px">
						{#each disasmLines as line, i (i)}
							<div class="flex items-center gap-3 text-xs py-1 px-2 rounded hover:bg-gray-800/50 font-mono">
								<span class="text-gray-600 w-4">{hexByte(line.offset)}</span>
								<span class="text-gray-500 w-20">
									{line.bytes.map((b) => hexByte(b)).join(' ')}
								</span>
								<span class="text-emerald-400">{line.mnemonic}</span>
							</div>
						{/each}
					</div>
				{:else}
					<p class="text-gray-600 text-xs italic">Hover over a cell to inspect</p>
				{/if}
			</div>
		</div>
	</div>
</div>

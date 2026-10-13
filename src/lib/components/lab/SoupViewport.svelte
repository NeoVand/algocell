<script lang="ts">
	import type { LabSession } from '$lib/lab/session.svelte';
	let { session }: { session: LabSession } = $props();
	let canvas: HTMLCanvasElement | undefined;
	let width = 800;
	let height = 560;
	let zoom = 1;
	let panX = 0;
	let panY = 0;
	let pointer: { x: number; y: number; startX: number; startY: number; moved: boolean } | null =
		null;
	let version = $state(0);
	const palette = Array.from({ length: 256 }, (_, byte) =>
		byte === 0 ? '#f8f9fb' : `hsl(${(byte * 137.508) % 360} 49% ${42 + (byte % 19)}%)`
	);
	let previousSize = '';

	function draw() {
		if (!canvas) return;
		const context = canvas.getContext('2d');
		if (!context) return;
		const { width: columns, height: rows, length, topology } = session.settings;
		const layoutKey = `${columns}:${rows}:${length}`;
		if (layoutKey !== previousSize) {
			zoom = 1;
			panX = panY = 0;
			previousSize = layoutKey;
		}
		const dpr = Math.min(window.devicePixelRatio || 1, 2);
		canvas.width = Math.round(width * dpr);
		canvas.height = Math.round(height * dpr);
		context.setTransform(dpr, 0, 0, dpr, 0, 0);
		context.fillStyle = '#e6e8eb';
		context.fillRect(0, 0, width, height);
		const cellSize = Math.min((width - 28) / columns, (height - 28) / rows) * zoom;
		const left = (width - columns * cellSize) / 2 + panX;
		const top = (height - rows * cellSize) / 2 + panY;
		if (!session.bytes.length) return;
		const byteCols = Math.ceil(Math.sqrt(length));
		const byteRows = Math.ceil(length / byteCols);
		for (let y = 0; y < rows; y++) {
			const py = top + y * cellSize;
			if (py + cellSize < 0 || py > height) continue;
			for (let x = 0; x < columns; x++) {
				const px = left + x * cellSize + (topology === 'hex' && y % 2 ? cellSize * 0.5 : 0);
				if (px + cellSize < 0 || px > width) continue;
				const cell = y * columns + x;
				const offset = cell * length;
				if (session.layer === 'fetch') {
					context.fillStyle = ['#e6e8eb', '#25313a', '#83909a'][session.fetchStatus[cell] ?? 0];
					context.fillRect(px, py, Math.ceil(cellSize), Math.ceil(cellSize));
				} else if (session.layer === 'identity') {
					let hash = 2166136261;
					for (let b = 0; b < length; b++)
						hash = Math.imul(hash ^ session.bytes[offset + b], 16777619);
					context.fillStyle = `hsl(${(hash >>> 0) % 360} 42% ${38 + ((hash >>> 8) & 31)}%)`;
					context.fillRect(px, py, Math.ceil(cellSize), Math.ceil(cellSize));
				} else if (session.detailed && cellSize > 5) {
					for (let b = 0; b < length; b++) {
						context.fillStyle = palette[session.bytes[offset + b]];
						context.fillRect(
							px + ((b % byteCols) * cellSize) / byteCols,
							py + (Math.floor(b / byteCols) * cellSize) / byteRows,
							Math.ceil(cellSize / byteCols),
							Math.ceil(cellSize / byteRows)
						);
					}
				} else {
					let nonzero = 0;
					let total = 0;
					for (let b = 0; b < length; b++) {
						const value = session.bytes[offset + b];
						total += value;
						if (value) nonzero++;
					}
					context.fillStyle =
						nonzero === 0
							? palette[0]
							: `hsl(${((total / length) * 4.73) % 360} ${25 + (nonzero / length) * 24}% ${80 - (nonzero / length) * 38}%)`;
					context.fillRect(px, py, Math.ceil(cellSize), Math.ceil(cellSize));
				}
			}
		}
		const selectedX = session.selectedCell % columns;
		const selectedY = Math.floor(session.selectedCell / columns);
		const px =
			left + selectedX * cellSize + (topology === 'hex' && selectedY % 2 ? cellSize * 0.5 : 0);
		const py = top + selectedY * cellSize;
		context.strokeStyle = '#fff';
		context.lineWidth = 3;
		context.strokeRect(px - 1, py - 1, cellSize + 2, cellSize + 2);
		context.strokeStyle = '#25313a';
		context.lineWidth = 1;
		context.strokeRect(px - 2, py - 2, cellSize + 4, cellSize + 4);
	}

	function fit() {
		zoom = 1;
		panX = panY = 0;
		version++;
	}
	function magnify(factor: number) {
		zoom = Math.max(1, Math.min(30, zoom * factor));
		version++;
	}
	function setup(element: HTMLCanvasElement) {
		canvas = element;
		const observer = new ResizeObserver(([entry]) => {
			width = entry.contentRect.width;
			height = entry.contentRect.height;
			draw();
		});
		observer.observe(element);
		const wheel = (event: WheelEvent) => {
			event.preventDefault();
			magnify(Math.exp(-Math.sign(event.deltaY) * 0.15));
		};
		element.addEventListener('wheel', wheel, { passive: false });
		draw();
		return () => {
			observer.disconnect();
			element.removeEventListener('wheel', wheel);
			canvas = undefined;
		};
	}
	function pointerDown(event: PointerEvent) {
		(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
		pointer = {
			x: event.clientX,
			y: event.clientY,
			startX: event.clientX,
			startY: event.clientY,
			moved: false
		};
	}
	function pointerMove(event: PointerEvent) {
		if (!pointer) return;
		if (Math.hypot(event.clientX - pointer.startX, event.clientY - pointer.startY) > 5)
			pointer.moved = true;
		if (pointer.moved) {
			panX += event.clientX - pointer.x;
			panY += event.clientY - pointer.y;
			version++;
		}
		pointer.x = event.clientX;
		pointer.y = event.clientY;
	}
	function pointerUp(event: PointerEvent) {
		if (pointer && !pointer.moved && canvas) {
			const rect = canvas.getBoundingClientRect();
			const config = session.settings;
			const size = Math.min((width - 28) / config.width, (height - 28) / config.height) * zoom;
			const y = Math.floor(
				(event.clientY - rect.top - (height - config.height * size) / 2 - panY) / size
			);
			const x = Math.floor(
				(event.clientX - rect.left - (width - config.width * size) / 2 - panX) / size -
					(config.topology === 'hex' && y % 2 ? 0.5 : 0)
			);
			if (x >= 0 && x < config.width && y >= 0 && y < config.height)
				session.select(y * config.width + x);
		}
		pointer = null;
	}
	$effect(() => {
		void version;
		void session.bytes;
		void session.fetchStatus;
		void session.layer;
		void session.detailed;
		void session.selectedCell;
		draw();
	});
</script>

<section class="viewport" aria-label="Soup lattice">
	<div class="viewport-header">
		<div>
			<h2>The soup</h2>
			<span
				>{session.settings.width} × {session.settings.height} tapes · {session.settings.length} bytes
				each</span
			>
		</div>
		<div class="view-tools" aria-label="Lattice view">
			<select aria-label="Lattice layer" bind:value={session.layer}
				><option value="bytes">Byte colors</option><option value="identity">Exact tapes</option
				><option value="fetch">Fetch confinement</option></select
			>
			<button class="icon-button" onclick={() => magnify(1.5)} aria-label="Zoom in">+</button>
			<button class="icon-button" onclick={() => magnify(1 / 1.5)} aria-label="Zoom out">−</button>
			<button class="subtle-button" onclick={fit}>Fit</button>
		</div>
	</div>
	<div class="canvas-wrap">
		<canvas
			{@attach setup}
			aria-label="Interactive soup. Click a tape to inspect. Use arrow keys to move selection, plus and minus to zoom."
			tabindex="0"
			onpointerdown={pointerDown}
			onpointermove={pointerMove}
			onpointerup={pointerUp}
			onpointercancel={() => (pointer = null)}
			onkeydown={(event) => {
				const moves: Record<string, number> = {
					ArrowLeft: -1,
					ArrowRight: 1,
					ArrowUp: -session.settings.width,
					ArrowDown: session.settings.width
				};
				if (event.key in moves) {
					event.preventDefault();
					session.select(session.selectedCell + moves[event.key]);
				}
				if (event.key === '+') magnify(1.5);
				if (event.key === '-') magnify(1 / 1.5);
			}}
			>A spatial lattice of machine-code tapes. Select a numbered cell in the inspector for its
			complete bytes.</canvas
		>
		{#if !session.bytes.length}<div class="canvas-empty">
				<strong
					>{session.status === 'loading' ? 'Preparing the world' : 'Execution needs WebGPU'}</strong
				><span>Explore the experiments and guide while the instrument gets ready.</span>
			</div>{/if}
	</div>
	<div class="viewport-footer">
		{#if session.layer === 'fetch'}<div class="legend">
				<span><i class="unknown"></i>Unmeasured</span><span><i class="confined"></i>Confined</span
				><span><i class="open"></i>Open</span>
			</div>
		{:else}<span
				>{session.layer === 'identity'
					? 'Equal full tapes share a color.'
					: 'Pale cells contain more zero bytes.'} Color does not establish heredity.</span
			>{/if}
		{#if session.layer === 'bytes'}<label
				><input type="checkbox" bind:checked={session.detailed} /> Byte detail</label
			>{/if}
	</div>
</section>

<style>
	.viewport {
		min-width: 0;
		background: #fff;
		border: 1px solid var(--line);
		border-radius: 10px;
		overflow: hidden;
	}
	.viewport-header {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		align-items: center;
		padding: 16px 18px;
	}
	h2 {
		font-size: 17px;
		margin: 0 0 3px;
	}
	.viewport-header span {
		font-size: 12px;
		color: var(--muted);
	}
	.view-tools {
		display: flex;
		gap: 5px;
		align-items: center;
	}
	select {
		width: auto;
		max-width: 164px;
		font-size: 12px;
		min-height: 36px;
	}
	.icon-button {
		width: 34px;
		min-height: 36px;
		font-size: 20px;
		padding: 0;
	}
	.canvas-wrap {
		position: relative;
		background: #e6e8eb;
	}
	canvas {
		width: 100%;
		height: clamp(350px, 54vh, 680px);
		display: block;
		cursor: crosshair;
		touch-action: none;
	}
	.canvas-empty {
		position: absolute;
		inset: 0;
		display: flex;
		justify-content: center;
		align-items: center;
		flex-direction: column;
		gap: 10px;
		padding: 32px;
		text-align: center;
		pointer-events: none;
		background: #e6e8be55;
	}
	.canvas-empty span {
		color: var(--muted);
		max-width: 32ch;
	}
	.viewport-footer {
		padding: 12px 18px;
		display: flex;
		gap: 14px;
		justify-content: space-between;
		align-items: center;
		color: var(--muted);
		font-size: 12px;
		min-height: 49px;
	}
	.viewport-footer label {
		display: flex;
		align-items: center;
		gap: 6px;
		white-space: nowrap;
	}
	.legend {
		display: flex;
		gap: 16px;
		flex-wrap: wrap;
	}
	.legend span {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	i {
		width: 10px;
		height: 10px;
		border-radius: 2px;
		display: inline-block;
	}
	.unknown {
		background: #e6e8eb;
		border: 1px solid #83909a;
	}
	.confined {
		background: #25313a;
	}
	.open {
		background: #83909a;
	}
	@media (max-width: 700px) {
		.viewport-header {
			flex-wrap: wrap;
			padding: 14px;
		}
		.view-tools {
			width: 100%;
		}
		.view-tools select {
			flex: 1;
			max-width: none;
		}
		canvas {
			height: 350px;
		}
		.viewport-footer {
			align-items: flex-start;
		}
	}
</style>

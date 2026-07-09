<script lang="ts">
	// DEV-ONLY page: verifies the morphogenesis CA on Zilion's GPU Z80 core.
	// Not linked from the app. Visit /dev/morph to run.
	//
	// Confirms three things end-to-end in the browser: (1) the growth genome
	// grows a pattern that matches the pure-TS reference exactly on the GPU;
	// (2) a batch of random genomes all match (strong correctness check); and
	// (3) a rough throughput number (candidates/sec) to seed the calibration
	// that decides interactive-vs-background training.
	import { onMount } from 'svelte';
	import { createMorphEngine, type MorphEngine } from '$lib/morph/zilion';
	import { runCA, asciiGrid, type MorphParams } from '$lib/morph/ca';
	import { growthGenome, randomGenome, gridDiff } from '$lib/morph/genomes';

	const p: MorphParams = { W: 16, H: 16, S: 4, T: 12 };

	// A small state palette (state 0 = background).
	const PALETTE = ['#0b0b0f', '#3b82f6', '#f59e0b', '#ef4444', '#10b981', '#a855f7', '#e5e7eb', '#14b8a6'];

	let status = $state('starting…');
	let error = $state<string | null>(null);
	let results = $state<{ label: string; detail: string; ok: boolean }[]>([]);
	let timing = $state('');
	let ascii = $state('');
	let canvasEl: HTMLCanvasElement | undefined = $state();
	let engine: MorphEngine | null = null;

	function draw(grid: Uint8Array) {
		const c = canvasEl;
		if (!c) return;
		const cell = 20;
		c.width = p.W * cell;
		c.height = p.H * cell;
		const ctx = c.getContext('2d');
		if (!ctx) return;
		ctx.fillStyle = PALETTE[0];
		ctx.fillRect(0, 0, c.width, c.height);
		for (let y = 0; y < p.H; y++) {
			for (let x = 0; x < p.W; x++) {
				ctx.fillStyle = PALETTE[grid[y * p.W + x] % PALETTE.length];
				ctx.fillRect(x * cell, y * cell, cell - 1, cell - 1);
			}
		}
	}

	async function run() {
		error = null;
		results = [];
		timing = '';
		try {
			if (!('gpu' in navigator)) {
				throw new Error('WebGPU is not available in this browser. Use Chrome/Edge 113+, Safari 18+, or Firefox 141+.');
			}
			status = 'creating Zilion instance…';
			engine?.destroy();
			engine = await createMorphEngine(p);

			// 1) growth genome — grow on GPU, compare to reference, render.
			status = 'growing (growth genome)…';
			const gg = growthGenome(p.S);
			const [gpuGrid] = await engine.grow([gg]);
			const refGrid = runCA(engine.seed, gg, p);
			const gd = gridDiff(gpuGrid, refGrid);
			results = [{ label: 'growth genome', detail: `${gd} / ${p.W * p.H} cells differ from reference`, ok: gd === 0 }];
			ascii = asciiGrid(gpuGrid, p);
			draw(gpuGrid);

			// 2) random batch — every table entry exercised; all must match.
			status = 'checking random batch…';
			const N = 64;
			const gens = Array.from({ length: N }, (_, i) => randomGenome(p.S, 1000 + i));
			const grids = await engine.grow(gens);
			let bad = 0;
			for (let i = 0; i < N; i++) if (gridDiff(grids[i], runCA(engine.seed, gens[i], p)) !== 0) bad++;
			results = [...results, { label: `random × ${N}`, detail: `${bad} genomes diverged from reference`, ok: bad === 0 }];

			// 3) rough throughput — candidates/sec at this grid size.
			status = 'measuring throughput…';
			const P = 4096;
			const big = Array.from({ length: P }, (_, i) => randomGenome(p.S, 7 + i));
			const t0 = performance.now();
			await engine.grow(big);
			const dt = performance.now() - t0;
			timing = `${P.toLocaleString()} genomes (${p.W}×${p.H}, T=${p.T}, ${engine.steps} step budget) in ${dt.toFixed(0)} ms → ${((P / dt) * 1000).toFixed(0)} candidates/s · memBytes=${engine.memBytes}`;

			status = 'done';
			(window as unknown as { __morph: unknown }).__morph = { engine, run };
		} catch (e) {
			error = e instanceof Error ? `${e.message}\n${e.stack ?? ''}` : String(e);
			status = 'error';
		}
	}

	onMount(() => {
		run();
		return () => engine?.destroy();
	});
</script>

<div class="wrap">
	<h1>Morphogenesis CA — Zilion GPU verification</h1>
	<p class="sub">Grows a {p.W}×{p.H} outer-totalistic CA on Zilion's GPU Z80 core and diffs it against the pure-TS reference.</p>

	<div class="status">status: <strong>{status}</strong></div>

	{#if error}
		<pre class="error">{error}</pre>
	{/if}

	{#each results as r (r.label)}
		<div class="row" class:pass={r.ok} class:fail={!r.ok}>
			<strong>{r.ok ? 'PASS' : 'FAIL'}</strong> — {r.label}: {r.detail}
		</div>
	{/each}

	{#if timing}
		<div class="timing">⏱ {timing}</div>
	{/if}

	<div class="grids">
		<div>
			<div class="cap">GPU-grown grid</div>
			<canvas bind:this={canvasEl}></canvas>
		</div>
		{#if ascii}
			<div>
				<div class="cap">ASCII</div>
				<pre class="ascii">{ascii}</pre>
			</div>
		{/if}
	</div>

	<button onclick={run}>Re-run</button>
</div>

<style>
	.wrap {
		font-family: ui-monospace, monospace;
		color: #e5e7eb;
		background: #0b0b0f;
		min-height: 100vh;
		padding: 24px;
		box-sizing: border-box;
	}
	h1 {
		font-size: 18px;
		margin: 0 0 4px;
	}
	.sub {
		color: #9ca3af;
		margin: 0 0 16px;
		font-size: 13px;
	}
	.status {
		margin-bottom: 12px;
		font-size: 13px;
	}
	.row {
		padding: 8px 12px;
		border-radius: 6px;
		margin-bottom: 6px;
		font-size: 13px;
	}
	.pass {
		background: #052e1a;
		color: #6ee7b7;
	}
	.fail {
		background: #3b0a0a;
		color: #fca5a5;
	}
	.timing {
		margin: 10px 0;
		color: #fcd34d;
		font-size: 13px;
	}
	.grids {
		display: flex;
		gap: 24px;
		margin: 16px 0;
		flex-wrap: wrap;
	}
	.cap {
		color: #9ca3af;
		font-size: 12px;
		margin-bottom: 6px;
	}
	canvas {
		image-rendering: pixelated;
		border: 1px solid #1f2937;
	}
	.ascii {
		background: #111827;
		padding: 8px;
		border-radius: 6px;
		font-size: 11px;
		line-height: 1.1;
	}
	.error {
		background: #3b0a0a;
		color: #fca5a5;
		padding: 12px;
		border-radius: 6px;
		white-space: pre-wrap;
		font-size: 12px;
	}
	button {
		margin-top: 12px;
		background: #1f2937;
		color: #e5e7eb;
		border: 1px solid #374151;
		border-radius: 6px;
		padding: 8px 16px;
		cursor: pointer;
		font-family: inherit;
	}
</style>

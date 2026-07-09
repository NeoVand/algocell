<script lang="ts">
	// DEV-ONLY: live evolution of a CA genome toward a target shape, on the GPU.
	// Visit /dev/morph/evolve. Watch a population evolve (scored on Zilion) until
	// a rule grows the target from a single seed.
	import { onMount } from 'svelte';
	import { createMorphEngine, type MorphEngine } from '$lib/morph/zilion';
	import { genomeLength, type MorphParams } from '$lib/morph/ca';
	import { makeDiamond, makeDisc, makeCross, makeRing, makeBullseye, makeCoreShell, makeFrenchFlag } from '$lib/morph/targets';
	import { Evolver, type FitnessMode } from '$lib/morph/evolve';

	const p: MorphParams = { W: 16, H: 16, S: 4, T: 14 };
	const PALETTE = ['#0b0b0f', '#3b82f6', '#f59e0b', '#ef4444', '#10b981', '#a855f7', '#e5e7eb', '#14b8a6'];
	const MAX_GEN = 400;

	// Multi-color targets need 'exact' mode to score colours; single-colour shapes
	// use 'binary' (alive/dead). 'flag' is the anisotropic case that cannot be
	// reached by an isotropic rule — included to show the wall.
	const TARGETS: Record<string, { make: () => Uint8Array; mode: FitnessMode }> = {
		diamond: { make: () => makeDiamond(p, 5, 1), mode: 'binary' },
		disc: { make: () => makeDisc(p, 5, 1), mode: 'binary' },
		cross: { make: () => makeCross(p, 1, 1), mode: 'binary' },
		ring: { make: () => makeRing(p, 6, 3, 1), mode: 'binary' },
		bullseye: { make: () => makeBullseye(p, [3, 2, 1], 2), mode: 'exact' },
		'core-shell': { make: () => makeCoreShell(p, 2, 5, 2, 1), mode: 'exact' },
		'flag (unreachable)': { make: () => makeFrenchFlag(p, [1, 2, 3]), mode: 'exact' }
	};

	let targetName = $state('diamond');
	let status = $state('starting…');
	let error = $state<string | null>(null);
	let generation = $state(0);
	let bestFitness = $state(0);
	let meanFitness = $state(0);
	let genPerSec = $state(0);
	let running = $state(false);

	let bestCanvas: HTMLCanvasElement | undefined = $state();
	let targetCanvas: HTMLCanvasElement | undefined = $state();

	let mode = $state<FitnessMode>('binary');
	let engine: MorphEngine | null = null;
	let evolver: Evolver | null = null;
	let target = $state<Uint8Array>(TARGETS.diamond.make());
	let runToken = 0; // guards against overlapping loops on restart/resume

	function draw(canvas: HTMLCanvasElement | undefined, grid: Uint8Array) {
		if (!canvas) return;
		const cell = 16;
		canvas.width = p.W * cell;
		canvas.height = p.H * cell;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		ctx.fillStyle = PALETTE[0];
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		for (let y = 0; y < p.H; y++)
			for (let x = 0; x < p.W; x++) {
				ctx.fillStyle = PALETTE[grid[y * p.W + x] % PALETTE.length];
				ctx.fillRect(x * cell, y * cell, cell - 1, cell - 1);
			}
	}

	function newEvolver() {
		evolver = new Evolver(
			{
				params: p,
				target,
				genomeLength: genomeLength(p.S),
				states: p.S,
				popSize: 512,
				eliteFrac: 0.08,
				mutationsPerChild: 2,
				fgWeight: mode === 'exact' ? 2 : 3,
				mode,
				seed: 42
			},
			(genomes) => engine!.grow(genomes)
		);
		generation = 0;
		bestFitness = 0;
		meanFitness = 0;
	}

	async function loop() {
		if (!evolver) return;
		const my = ++runToken;
		running = true;
		status = 'evolving…';
		let frames = 0;
		let t0 = performance.now();
		while (runToken === my && evolver.generation < MAX_GEN && bestFitness < 1) {
			const s = await evolver.step();
			if (runToken !== my) return; // superseded while awaiting the GPU
			generation = s.generation;
			bestFitness = s.bestFitness;
			meanFitness = s.meanFitness;
			draw(bestCanvas, s.bestGrid);
			frames++;
			const now = performance.now();
			if (now - t0 > 500) {
				genPerSec = (frames * 1000) / (now - t0);
				frames = 0;
				t0 = now;
			}
			await new Promise((r) => requestAnimationFrame(r));
			if (runToken !== my) return;
		}
		running = false;
		status = bestFitness >= 1 ? `solved in ${generation} generations` : 'stopped';
	}

	function restart() {
		runToken++; // stop any running loop
		running = false;
		const entry = TARGETS[targetName];
		mode = entry.mode; // multi-color targets auto-switch to exact-state scoring
		target = entry.make();
		draw(targetCanvas, target);
		newEvolver();
		loop();
	}

	function toggle() {
		if (running) {
			runToken++;
			running = false;
		} else {
			loop();
		}
	}

	onMount(() => {
		(async () => {
			try {
				if (!('gpu' in navigator)) throw new Error('WebGPU not available in this browser.');
				status = 'creating Zilion instance…';
				engine = await createMorphEngine(p);
				draw(targetCanvas, target);
				newEvolver();
				loop();
			} catch (e) {
				error = e instanceof Error ? `${e.message}\n${e.stack ?? ''}` : String(e);
				status = 'error';
			}
		})();
		return () => {
			runToken++;
			running = false;
			engine?.destroy();
			engine = null;
		};
	});
</script>

<div class="wrap">
	<h1>Morphogenesis — live evolution on the GPU</h1>
	<p class="sub">A population of {512} CA genomes, scored on Zilion each generation, evolving a rule that grows the target from one seed.</p>

	<div class="controls">
		<label>
			target
			<select
				value={targetName}
				onchange={(e) => {
					targetName = e.currentTarget.value;
					restart();
				}}
			>
				{#each Object.keys(TARGETS) as t (t)}<option value={t}>{t}</option>{/each}
			</select>
		</label>
		<button onclick={toggle}>{running ? 'Pause' : 'Resume'}</button>
		<button onclick={restart}>Restart</button>
	</div>

	<div class="status">
		<strong>{status}</strong> · {mode} · gen {generation} · best <strong>{(bestFitness * 100).toFixed(1)}%</strong>
		· mean {(meanFitness * 100).toFixed(1)}% · {genPerSec.toFixed(0)} gen/s
	</div>

	{#if error}<pre class="error">{error}</pre>{/if}

	<div class="grids">
		<div><div class="cap">evolving (best of generation)</div><canvas bind:this={bestCanvas}></canvas></div>
		<div><div class="cap">target: {targetName}</div><canvas bind:this={targetCanvas}></canvas></div>
	</div>
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
	.controls {
		display: flex;
		gap: 12px;
		align-items: center;
		margin-bottom: 12px;
		font-size: 13px;
	}
	select,
	button {
		background: #1f2937;
		color: #e5e7eb;
		border: 1px solid #374151;
		border-radius: 6px;
		padding: 6px 12px;
		cursor: pointer;
		font-family: inherit;
	}
	.status {
		font-size: 13px;
		margin-bottom: 14px;
		color: #fcd34d;
	}
	.grids {
		display: flex;
		gap: 24px;
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
	.error {
		background: #3b0a0a;
		color: #fca5a5;
		padding: 12px;
		border-radius: 6px;
		white-space: pre-wrap;
		font-size: 12px;
	}
</style>

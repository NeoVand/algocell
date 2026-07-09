<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import {
		EXPERIMENTS, experimentById, loadParams, seedGrid, readOutputs, N, C, SW, SH
	} from '$lib/devcomp/rule';
	import { FieldCAEngine } from '$lib/devcomp/engine';
	import e1 from '$lib/devcomp/params/e1_gate.json';
	import e3 from '$lib/devcomp/params/e3_seed.json';

	// The E3 rule is the universal, long-term-stable rule: it computes XOR, holds
	// the answer indefinitely, and self-repairs — in BOTH the full and seed initial
	// conditions. The dedicated E2 self-repair stage is only *metastable* (its "1"
	// outputs decay to 0 past ~150 steps), so the demo's live-running self-repair
	// and grow tabs use E3. (e2_repair.json is kept as the training-stage artifact.)
	const PARAMS: Record<string, number[]> = { e1_gate: e1, e2_repair: e3, e3_seed: e3 };
	const CELL = 40;

	let canvas: HTMLCanvasElement;
	let engine: FieldCAEngine | null = null;

	let expId = $state('e3_seed');
	let inputs = $state([0, 0]);
	let playing = $state(true);
	let stepCount = $state(0);
	let output = $state<number[]>([0]);
	let msg = $state('starting the GPU…');

	const exp = $derived(experimentById(expId)!);
	const persistent = $derived(expId !== 'e1_gate');
	const maxSteps = $derived(persistent ? Infinity : exp.tGrow);
	const expected = $derived(exp.cases.find((c) => c.in.every((v, i) => v === inputs[i]))?.out ?? []);
	const correct = $derived(expected.every((t, k) => Math.abs((output[k] ?? 0) - t) < 0.3));

	function applyInputs() {
		if (!engine) return;
		const isIn = new Uint32Array(N), val = new Float32Array(N);
		exp.inputCells.forEach((cell, k) => { isIn[cell] = 1; val[cell] = inputs[k]; });
		engine.setInputs(isIn, val);
	}

	function load() {
		if (!engine) return;
		engine.setParams(new Float32Array(loadParams(PARAMS[expId])));
		applyInputs();
		engine.setDamageKeep(new Uint32Array(N).fill(1));
		engine.seed(new Float32Array(seedGrid(exp, inputs)));
		stepCount = 0;
	}

	function selectExp(id: string) { expId = id; load(); playing = true; }
	// These are developmental rules: they grow a machine that computes the answer
	// for the input present during growth. Changing the input therefore re-seeds
	// and regrows (you watch it recompute) rather than mutating a settled field.
	function toggleInput(k: number) { inputs = inputs.map((v, i) => (i === k ? v ^ 1 : v)); load(); playing = true; }
	function reset() { load(); playing = true; }
	function doStep() { if (engine && stepCount < maxSteps) { engine.step(false); stepCount++; } }

	// Interactive damage brush — destroy a 3×3 patch under the pointer; the running
	// rule regrows it. Most striking on self-repair / grow-from-seed (they persist).
	let painting = false;
	function paintDamage(e: PointerEvent) {
		if (!engine) return;
		const rect = canvas.getBoundingClientRect();
		const cx = Math.floor(((e.clientX - rect.left) / rect.width) * SW);
		const cy = Math.floor(((e.clientY - rect.top) / rect.height) * SH);
		for (let dy = -1; dy <= 1; dy++)
			for (let dx = -1; dx <= 1; dx++) {
				const x = cx + dx, y = cy + dy;
				if (x >= 1 && x < SW - 1 && y >= 1 && y < SH - 1) engine.damageCell(y * SW + x);
			}
		playing = true; // so it steps and heals
	}
	function pointerDown(e: PointerEvent) { painting = true; try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ } paintDamage(e); }
	function pointerMove(e: PointerEvent) { if (painting) paintDamage(e); }
	function pointerUp() { painting = false; }

	function color(v: number): string {
		const t = Math.max(-1, Math.min(1, v));
		const bg = [10, 15, 22], pos = [255, 138, 61], neg = [43, 108, 255];
		const c = t >= 0 ? pos : neg, a = Math.abs(t);
		return `rgb(${Math.round(bg[0] + (c[0] - bg[0]) * a)},${Math.round(bg[1] + (c[1] - bg[1]) * a)},${Math.round(bg[2] + (c[2] - bg[2]) * a)})`;
	}

	function draw(st: Float32Array) {
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		for (let y = 0; y < SH; y++)
			for (let x = 0; x < SW; x++) {
				ctx.fillStyle = color(st[(y * SW + x) * C + 0]);
				ctx.fillRect(x * CELL, y * CELL, CELL - 1, CELL - 1);
			}
		const seedCell = exp.seedCell;
		if (seedCell !== undefined) {
			const sx = seedCell % SW, sy = (seedCell / SW) | 0;
			ctx.fillStyle = '#34d399';
			ctx.beginPath(); ctx.arc(sx * CELL + CELL / 2, sy * CELL + CELL / 2, 4, 0, 7); ctx.fill();
		}
		for (const cell of exp.inputCells) {
			const cxp = cell % SW, cyp = (cell / SW) | 0;
			ctx.strokeStyle = '#e6edf3'; ctx.lineWidth = 2;
			ctx.beginPath(); ctx.arc(cxp * CELL + CELL / 2, cyp * CELL + CELL / 2, CELL / 2 - 6, 0, 7); ctx.stroke();
		}
		for (const cell of exp.outputCells) {
			const cxp = cell % SW, cyp = (cell / SW) | 0;
			ctx.strokeStyle = '#2dd4bf'; ctx.lineWidth = 3;
			ctx.strokeRect(cxp * CELL + 3, cyp * CELL + 3, CELL - 7, CELL - 7);
		}
	}

	onMount(() => {
		let raf = 0, disposed = false;
		(async () => {
			try { engine = await FieldCAEngine.create(); }
			catch (e) { msg = 'WebGPU unavailable: ' + (e as Error).message; return; }
			canvas.width = SW * CELL; canvas.height = SH * CELL;
			load();
			msg = '';
			const loop = async () => {
				if (disposed || !engine) return;
				if (playing && stepCount < maxSteps) { engine.step(false); stepCount++; }
				const st = await engine.readState();
				draw(st);
				output = readOutputs(st, exp);
				raf = requestAnimationFrame(loop);
			};
			loop();
		})();
		return () => { disposed = true; cancelAnimationFrame(raf); engine?.destroy(); };
	});
</script>

<svelte:head><title>Developmental computation — live on the GPU</title></svelte:head>

<main>
	<header>
		<a class="back" href={resolve('/')}>← Algocell</a>
		<h1>A computer, grown and running on your GPU</h1>
		<p class="sub">One learned cellular-automaton rule, executed live in a WebGPU compute shader. Pick an experiment, flip the inputs, watch it compute.</p>
	</header>

	<div class="tabs">
		{#each EXPERIMENTS as e (e.id)}
			<button class="tab" class:active={expId === e.id} onclick={() => selectExp(e.id)}>{e.name}</button>
		{/each}
	</div>

	<div class="stage">
		<canvas
			bind:this={canvas}
			onpointerdown={pointerDown}
			onpointermove={pointerMove}
			onpointerup={pointerUp}
			onpointerleave={pointerUp}
		></canvas>
		<div class="side">
			<p class="blurb">{exp.blurb}</p>
			<p class="hint">✎ Drag on the grid to damage it — watch it regrow and keep computing.</p>

			<div class="inputs">
				{#each inputs as bit, k (k)}
					<button class="chip" class:on={bit === 1} onclick={() => toggleInput(k)}>
						in {String.fromCharCode(65 + k)} = {bit}
					</button>
				{/each}
			</div>

			<div class="readout">
				<span class="lbl">output</span>
				<span class="val" class:ok={correct} class:bad={!correct}>{(output[0] ?? 0).toFixed(3)}</span>
				<span class="want">want {expected.join(',')} {correct ? '✓' : ''}</span>
			</div>

			<div class="controls">
				<button onclick={() => (playing = !playing)}>{playing ? '❚❚ Pause' : '▶ Play'}</button>
				<button onclick={doStep}>Step</button>
				<button onclick={reset}>Reset</button>
			</div>
			<p class="meta">step {stepCount}{persistent ? '' : ` / ${exp.tGrow}`} · ○ input · □ output{exp.seedCell !== undefined ? ' · 🌱 seed' : ''}</p>
			{#if msg}<p class="msg">{msg}</p>{/if}
		</div>
	</div>
</main>

<style>
	main {
		min-height: 100vh; box-sizing: border-box; padding: clamp(20px, 4vw, 44px);
		background: radial-gradient(1100px 520px at 50% -8%, rgba(45, 212, 191, 0.08), transparent 60%), #0a0d12;
		color: #e6edf3; font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
	}
	.back { color: #9aa7b4; text-decoration: none; font-size: 13px; }
	.back:hover { color: #e6edf3; }
	h1 { margin: 10px 0 4px; font-size: clamp(22px, 4vw, 32px); letter-spacing: -0.02em;
		background: linear-gradient(120deg, #e6edf3, #7dd3c8); -webkit-background-clip: text; background-clip: text; color: transparent; }
	.sub { margin: 0 0 20px; color: #9aa7b4; max-width: 64ch; }
	.tabs { display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap; }
	.tab { background: rgba(255, 255, 255, 0.03); color: #cbd5e1; border: 1px solid rgba(255, 255, 255, 0.08);
		border-radius: 999px; padding: 7px 16px; font-size: 13.5px; font-weight: 600; cursor: pointer; }
	.tab.active { background: color-mix(in srgb, #2dd4bf 16%, transparent); border-color: #2dd4bf; color: #e6edf3; }
	.stage { display: flex; gap: 28px; flex-wrap: wrap; align-items: flex-start; }
	canvas { border-radius: 12px; background: #05070a; image-rendering: pixelated; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4); cursor: crosshair; touch-action: none; }
	.hint { margin: 0; color: #7dd3c8; font-size: 13px; }
	.side { flex: 1; min-width: 240px; max-width: 380px; display: flex; flex-direction: column; gap: 16px; }
	.blurb { margin: 0; color: #9aa7b4; font-size: 14px; line-height: 1.5; }
	.inputs { display: flex; gap: 10px; }
	.chip { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); color: #9aa7b4;
		border-radius: 9px; padding: 8px 14px; font-size: 14px; font-weight: 600; cursor: pointer; font-variant-numeric: tabular-nums; }
	.chip.on { background: color-mix(in srgb, #ff8a3d 20%, transparent); border-color: #ff8a3d; color: #fff; }
	.readout { display: flex; align-items: baseline; gap: 10px; }
	.readout .lbl { color: #6b7785; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
	.readout .val { font-size: 26px; font-weight: 700; font-variant-numeric: tabular-nums; }
	.readout .val.ok { color: #34d399; } .readout .val.bad { color: #f87171; }
	.readout .want { color: #9aa7b4; font-size: 13px; }
	.controls { display: flex; gap: 10px; }
	.controls button { background: #14324a; color: #cdeee7; border: 1px solid #23566f; border-radius: 9px;
		padding: 8px 16px; font-size: 14px; font-weight: 600; cursor: pointer; }
	.controls button:hover { background: #1a415f; }
	.meta { margin: 0; color: #6b7785; font-size: 12.5px; }
	.msg { margin: 0; color: #fbbf24; font-size: 13px; }
</style>

<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import { EXPERIMENTS, experimentById, loadParams, seedGrid, readOutputs, type RuleConfig } from '$lib/devcomp/rule';
	import { FieldCAEngine } from '$lib/devcomp/engine';
	import e1 from '$lib/devcomp/params/e1_gate.json';
	import e3 from '$lib/devcomp/params/e3_seed.json';
	import adderStable from '$lib/devcomp/params/adder_stable.json';

	// Params by file. e2_repair + e3_seed share the stable E3 rule; the adder uses
	// its long-horizon-stable + self-repairing rule. e1 is compute-only (capped at
	// tGrow so it doesn't drift). See rule.ts `stable`.
	const PARAM_FILES: Record<string, number[]> = { 'e1_gate.json': e1, 'e3_seed.json': e3, 'adder_stable.json': adderStable };

	let canvas: HTMLCanvasElement;
	let engine: FieldCAEngine | null = null;
	let ready = false;
	let cellPx = 40;

	let expId = $state('adder');
	let inputs = $state<number[]>([0, 0, 0]);
	let playing = $state(true);
	let stepCount = $state(0);
	let output = $state<number[]>([0]);
	let msg = $state('starting the GPU…');

	const exp = $derived(experimentById(expId)!);
	const maxSteps = $derived(exp.stable ? Infinity : exp.tGrow);
	const expected = $derived(exp.cases.find((c) => c.in.every((v, i) => v === inputs[i]))?.out ?? []);
	const correct = $derived(expected.length > 0 && expected.every((t, k) => Math.abs((output[k] ?? 0) - t) < 0.3));

	async function ensureEngine(cfg: RuleConfig) {
		if (engine && engine.cfg === cfg) return;
		engine?.destroy();
		engine = await FieldCAEngine.create(cfg);
	}

	function applyInputs() {
		if (!engine) return;
		const cfg = exp.cfg;
		const isIn = new Uint32Array(cfg.N), val = new Float32Array(cfg.N);
		exp.inputCells.forEach((cell, k) => { isIn[cell] = 1; val[cell] = inputs[k]; });
		engine.setInputs(isIn, val);
	}

	async function load() {
		ready = false;
		const cfg = exp.cfg;
		await ensureEngine(cfg);
		if (!engine) return;
		cellPx = Math.max(24, Math.floor(400 / cfg.SW));
		canvas.width = cfg.SW * cellPx; canvas.height = cfg.SH * cellPx;
		engine.setParams(new Float32Array(loadParams(cfg, PARAM_FILES[exp.paramsUrl])));
		applyInputs();
		engine.setDamageKeep(new Uint32Array(cfg.N).fill(1));
		engine.seed(new Float32Array(seedGrid(cfg, exp, inputs)));
		stepCount = 0;
		ready = true;
	}

	async function selectExp(id: string) {
		expId = id;
		inputs = new Array(exp.inputCells.length).fill(0);
		await load();
		playing = true;
	}
	// Developmental rules compute the answer for the input present during growth,
	// so changing an input re-seeds and regrows (you watch it recompute).
	async function toggleInput(k: number) { inputs = inputs.map((v, i) => (i === k ? v ^ 1 : v)); await load(); playing = true; }
	async function reset() { await load(); playing = true; }
	function doStep() { if (engine && ready && stepCount < maxSteps) { engine.step(false); stepCount++; } }

	// Damage brush — destroy a 3×3 patch under the pointer; the rule regrows it.
	let painting = false;
	function paintDamage(e: PointerEvent) {
		if (!engine || !ready) return;
		const cfg = exp.cfg, rect = canvas.getBoundingClientRect();
		const cx = Math.floor(((e.clientX - rect.left) / rect.width) * cfg.SW);
		const cy = Math.floor(((e.clientY - rect.top) / rect.height) * cfg.SH);
		for (let dy = -1; dy <= 1; dy++)
			for (let dx = -1; dx <= 1; dx++) {
				const x = cx + dx, y = cy + dy;
				if (x >= 1 && x < cfg.SW - 1 && y >= 1 && y < cfg.SH - 1) engine.damageCell(y * cfg.SW + x);
			}
		playing = true;
	}
	function pointerDown(e: PointerEvent) { painting = true; try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic */ } paintDamage(e); }
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
		const cfg = exp.cfg, cp = cellPx;
		for (let y = 0; y < cfg.SH; y++)
			for (let x = 0; x < cfg.SW; x++) {
				ctx.fillStyle = color(st[(y * cfg.SW + x) * cfg.C + 0]);
				ctx.fillRect(x * cp, y * cp, cp - 1, cp - 1);
			}
		if (exp.seedCell !== undefined && exp.ic === 'seed') {
			const sx = exp.seedCell % cfg.SW, sy = (exp.seedCell / cfg.SW) | 0;
			ctx.fillStyle = '#34d399';
			ctx.beginPath(); ctx.arc(sx * cp + cp / 2, sy * cp + cp / 2, 4, 0, 7); ctx.fill();
		}
		for (const cell of exp.inputCells) {
			const cxp = cell % cfg.SW, cyp = (cell / cfg.SW) | 0;
			ctx.strokeStyle = '#e6edf3'; ctx.lineWidth = 2;
			ctx.beginPath(); ctx.arc(cxp * cp + cp / 2, cyp * cp + cp / 2, cp / 2 - 5, 0, 7); ctx.stroke();
		}
		for (const cell of exp.outputCells) {
			const cxp = cell % cfg.SW, cyp = (cell / cfg.SW) | 0;
			ctx.strokeStyle = '#2dd4bf'; ctx.lineWidth = 3;
			ctx.strokeRect(cxp * cp + 3, cyp * cp + 3, cp - 7, cp - 7);
		}
	}

	onMount(() => {
		let raf = 0, disposed = false;
		(async () => {
			try { await load(); } catch (e) { msg = 'WebGPU unavailable: ' + (e as Error).message; return; }
			msg = '';
			const loop = async () => {
				if (disposed) return;
				if (ready && engine) {
					if (playing && stepCount < maxSteps) { engine.step(false); stepCount++; }
					const st = await engine.readState();
					draw(st);
					output = readOutputs(exp.cfg, st, exp);
				}
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
		<p class="sub">One learned cellular-automaton rule per experiment, executed live in a WebGPU compute shader. Pick an experiment, flip the inputs, watch it compute — and drag on the grid to damage it.</p>
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
						{exp.inputCells.length <= 2 ? 'in ' + String.fromCharCode(65 + k) : ['a', 'b', 'cin'][k] ?? 'in' + k} = {bit}
					</button>
				{/each}
			</div>

			<div class="outputs">
				{#each exp.outputCells as _cell, k (k)}
					{@const val = output[k] ?? 0}
					{@const tgt = expected[k]}
					{@const ok = tgt !== undefined && Math.abs(val - tgt) < 0.3}
					<div class="readout">
						<span class="lbl">{exp.outputLabels?.[k] ?? 'output'}</span>
						<span class="val" class:ok class:bad={!ok}>{val.toFixed(3)}</span>
						<span class="want">want {tgt ?? '?'} {ok ? '✓' : ''}</span>
					</div>
				{/each}
			</div>

			<div class="controls">
				<button onclick={() => (playing = !playing)}>{playing ? '❚❚ Pause' : '▶ Play'}</button>
				<button onclick={doStep}>Step</button>
				<button onclick={reset}>Reset</button>
			</div>
			<p class="meta">step {stepCount}{exp.stable ? '' : ` / ${exp.tGrow}`} · ○ input · □ output{exp.ic === 'seed' ? ' · 🌱 seed' : ''}</p>
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
	.sub { margin: 0 0 20px; color: #9aa7b4; max-width: 66ch; }
	.tabs { display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap; }
	.tab { background: rgba(255, 255, 255, 0.03); color: #cbd5e1; border: 1px solid rgba(255, 255, 255, 0.08);
		border-radius: 999px; padding: 7px 16px; font-size: 13.5px; font-weight: 600; cursor: pointer; }
	.tab.active { background: color-mix(in srgb, #2dd4bf 16%, transparent); border-color: #2dd4bf; color: #e6edf3; }
	.stage { display: flex; gap: 28px; flex-wrap: wrap; align-items: flex-start; }
	canvas { border-radius: 12px; background: #05070a; image-rendering: pixelated; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4); cursor: crosshair; touch-action: none; }
	.side { flex: 1; min-width: 240px; max-width: 380px; display: flex; flex-direction: column; gap: 14px; }
	.blurb { margin: 0; color: #9aa7b4; font-size: 14px; line-height: 1.5; }
	.hint { margin: 0; color: #7dd3c8; font-size: 13px; }
	.inputs { display: flex; gap: 10px; flex-wrap: wrap; }
	.chip { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); color: #9aa7b4;
		border-radius: 9px; padding: 8px 14px; font-size: 14px; font-weight: 600; cursor: pointer; font-variant-numeric: tabular-nums; }
	.chip.on { background: color-mix(in srgb, #ff8a3d 20%, transparent); border-color: #ff8a3d; color: #fff; }
	.outputs { display: flex; flex-direction: column; gap: 6px; }
	.readout { display: flex; align-items: baseline; gap: 10px; }
	.readout .lbl { color: #6b7785; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; min-width: 46px; }
	.readout .val { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
	.readout .val.ok { color: #34d399; } .readout .val.bad { color: #f87171; }
	.readout .want { color: #9aa7b4; font-size: 13px; }
	.controls { display: flex; gap: 10px; }
	.controls button { background: #14324a; color: #cdeee7; border: 1px solid #23566f; border-radius: 9px;
		padding: 8px 16px; font-size: 14px; font-weight: 600; cursor: pointer; }
	.controls button:hover { background: #1a415f; }
	.meta { margin: 0; color: #6b7785; font-size: 12.5px; }
	.msg { margin: 0; color: #fbbf24; font-size: 13px; }
</style>

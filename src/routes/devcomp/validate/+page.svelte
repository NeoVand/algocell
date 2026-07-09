<script lang="ts">
	import { onMount } from 'svelte';
	import {
		experimentById, loadParams, forward, readOutputs, damageMask, seedGrid,
		N, C, SW, SH
	} from '$lib/devcomp/rule';
	import { FieldCAEngine } from '$lib/devcomp/engine';
	import e1 from '$lib/devcomp/params/e1_gate.json';
	import e2 from '$lib/devcomp/params/e2_repair.json';
	import e3 from '$lib/devcomp/params/e3_seed.json';

	interface Row { label: string; cpu: string; gpu: string; maxDiff: number; ok: boolean; }
	let rows = $state<Row[]>([]);
	let status = $state('running…');

	const DMG_AT = 32;
	const paramsFor: Record<string, number[]> = { e1_gate: e1, e2_repair: e2, e3_seed: e3 };
	const iy = SH >> 1, cx = Math.round((2 + (SW - 2)) / 2);
	// Each experiment is read at the step where its rule is meant to be read:
	// E1 computes at tGrow (it does not persist); E2/E3 hold + heal to step 50.
	const CONFIGS = [
		{ id: 'e1_gate', steps: 24, damage: false },
		{ id: 'e2_repair', steps: 50, damage: false },
		{ id: 'e2_repair', steps: 50, damage: true },
		{ id: 'e3_seed', steps: 50, damage: false },
		{ id: 'e3_seed', steps: 50, damage: true }
	];

	function perCellInputs(exp: ReturnType<typeof experimentById>, inputs: number[]) {
		const isInput = new Uint32Array(N), inputVal = new Float32Array(N);
		exp!.inputCells.forEach((cell, k) => { isInput[cell] = 1; inputVal[cell] = inputs[k]; });
		return { isInput, inputVal };
	}

	async function validate() {
		let engine: FieldCAEngine;
		try {
			engine = await FieldCAEngine.create();
		} catch (e) {
			status = 'WebGPU unavailable: ' + (e as Error).message;
			return;
		}
		const out: Row[] = [];
		for (const cfg of CONFIGS) {
			const exp = experimentById(cfg.id)!;
			const par64 = loadParams(paramsFor[cfg.id]);
			engine.setParams(new Float32Array(par64));
			for (const cse of exp.cases) {
				const mask = damageMask(cx, iy, 3);
				const cpuFinal = forward(par64, exp, cse.in,
					cfg.damage ? { steps: cfg.steps, damage: { at: DMG_AT, mask } } : { steps: cfg.steps })[cfg.steps];
				const { isInput, inputVal } = perCellInputs(exp, cse.in);
				engine.setInputs(isInput, inputVal);
				engine.setDamageKeep(Uint32Array.from(mask));
				engine.seed(new Float32Array(seedGrid(exp, cse.in)));
				for (let t = 0; t < cfg.steps; t++) engine.step(cfg.damage && t + 1 === DMG_AT);
				const gpuFinal = await engine.readState();
				let maxDiff = 0;
				for (let i = 0; i < N * C; i++) maxDiff = Math.max(maxDiff, Math.abs(cpuFinal[i] - gpuFinal[i]));
				const cpuOut = readOutputs(cpuFinal, exp);
				const gpuOut = exp.outputCells.map((cell) => gpuFinal[cell * C + 0]);
				const outOk = cse.out.every((tgt, k) => Math.abs(gpuOut[k] - tgt) < 0.2 && Math.abs(cpuOut[k] - tgt) < 0.2);
				const ok = maxDiff < 2e-3 && outOk;
				out.push({
					label: `${cfg.id} [${cse.in.join(',')}]${cfg.damage ? ' +dmg' : ''} @${cfg.steps}`,
					cpu: cpuOut.map((x) => x.toFixed(3)).join(' '),
					gpu: gpuOut.map((x) => x.toFixed(3)).join(' '),
					maxDiff, ok
				});
			}
		}
		rows = out;
		const allOk = out.every((r) => r.ok);
		const maxAll = Math.max(...out.map((r) => r.maxDiff));
		status = (allOk ? 'PASS' : 'FAIL') + ` — ${out.filter((r) => r.ok).length}/${out.length} cases, max|CPU−GPU|=${maxAll.toExponential(2)}`;
		console.log('[devcomp validate]', status);
		engine.destroy();
	}

	onMount(validate);
</script>

<svelte:head><title>devcomp — WGSL validation</title></svelte:head>

<main>
	<h1>WGSL field-CA validation</h1>
	<p class="status" class:pass={status.startsWith('PASS')} class:fail={status.startsWith('FAIL')}>{status}</p>
	<table>
		<thead><tr><th>case</th><th>CPU out</th><th>GPU out</th><th>max|Δ|</th><th></th></tr></thead>
		<tbody>
			{#each rows as r (r.label)}
				<tr class:bad={!r.ok}>
					<td>{r.label}</td><td>{r.cpu}</td><td>{r.gpu}</td>
					<td>{r.maxDiff.toExponential(2)}</td><td>{r.ok ? '✓' : '✗'}</td>
				</tr>
			{/each}
		</tbody>
	</table>
</main>

<style>
	main { font-family: ui-monospace, monospace; padding: 28px; color: #e6edf3; background: #0a0d12; min-height: 100vh; }
	h1 { font-size: 20px; }
	.status { font-size: 16px; font-weight: 700; }
	.status.pass { color: #34d399; } .status.fail { color: #f87171; }
	table { border-collapse: collapse; margin-top: 14px; font-size: 13px; }
	th, td { border: 1px solid #23303d; padding: 4px 10px; text-align: left; }
	tr.bad { background: rgba(248, 113, 113, 0.12); }
</style>

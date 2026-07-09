<script lang="ts">
	import { onMount } from 'svelte';
	import { experimentById, loadParams, forward, readOutputs, damageMask, seedGrid, type RuleConfig } from '$lib/devcomp/rule';
	import { FieldCAEngine } from '$lib/devcomp/engine';
	import e1 from '$lib/devcomp/params/e1_gate.json';
	import e3 from '$lib/devcomp/params/e3_seed.json';
	import adder from '$lib/devcomp/params/adder_compute.json';

	interface Row { label: string; cpu: string; gpu: string; maxDiff: number; ok: boolean; }
	let rows = $state<Row[]>([]);
	let status = $state('running…');

	const PARAM_FILES: Record<string, number[]> = { 'e1_gate.json': e1, 'e3_seed.json': e3, 'adder_compute.json': adder };
	const DMG_AT = 32;
	// Each experiment is read where its rule is meant to be read: unstable rules at
	// tGrow, stable ones at 50 (with damage to exercise self-repair).
	const CONFIGS = [
		{ id: 'e1_gate', steps: 24, damage: false },
		{ id: 'e2_repair', steps: 50, damage: false },
		{ id: 'e2_repair', steps: 50, damage: true },
		{ id: 'e3_seed', steps: 50, damage: false },
		{ id: 'e3_seed', steps: 50, damage: true },
		{ id: 'adder', steps: 30, damage: false }
	];

	async function validate() {
		const engines = new Map<RuleConfig, FieldCAEngine>();
		const out: Row[] = [];
		try {
			for (const cfg of CONFIGS) {
				const exp = experimentById(cfg.id)!;
				const rc = exp.cfg;
				let engine = engines.get(rc);
				if (!engine) { engine = await FieldCAEngine.create(rc); engines.set(rc, engine); }
				const par64 = loadParams(rc, PARAM_FILES[exp.paramsUrl]);
				engine.setParams(new Float32Array(par64));
				const cx = rc.SW >> 1, cy = rc.SH >> 1;
				for (const cse of exp.cases) {
					const mask = damageMask(rc, cx, cy, 3);
					const cpuFinal = forward(rc, par64, exp, cse.in,
						cfg.damage ? { steps: cfg.steps, damage: { at: DMG_AT, mask } } : { steps: cfg.steps })[cfg.steps];
					const isInput = new Uint32Array(rc.N), inputVal = new Float32Array(rc.N);
					exp.inputCells.forEach((cell, k) => { isInput[cell] = 1; inputVal[cell] = cse.in[k]; });
					engine.setInputs(isInput, inputVal);
					engine.setDamageKeep(Uint32Array.from(mask));
					engine.seed(new Float32Array(seedGrid(rc, exp, cse.in)));
					for (let t = 0; t < cfg.steps; t++) engine.step(cfg.damage && t + 1 === DMG_AT);
					const gpuFinal = await engine.readState();
					let maxDiff = 0;
					for (let i = 0; i < rc.N * rc.C; i++) maxDiff = Math.max(maxDiff, Math.abs(cpuFinal[i] - gpuFinal[i]));
					const cpuOut = readOutputs(rc, cpuFinal, exp);
					const gpuOut = exp.outputCells.map((cell) => gpuFinal[cell * rc.C + 0]);
					const outOk = cse.out.every((tgt, k) => Math.abs(gpuOut[k] - tgt) < 0.2 && Math.abs(cpuOut[k] - tgt) < 0.2);
					const ok = maxDiff < 2e-3 && outOk;
					out.push({
						label: `${cfg.id} [${cse.in.join(',')}]${cfg.damage ? ' +dmg' : ''} @${cfg.steps}`,
						cpu: cpuOut.map((x) => x.toFixed(2)).join(' '),
						gpu: gpuOut.map((x) => x.toFixed(2)).join(' '),
						maxDiff, ok
					});
				}
			}
		} catch (e) { status = 'error: ' + (e as Error).message; return; }
		rows = out;
		const allOk = out.every((r) => r.ok);
		const maxAll = Math.max(...out.map((r) => r.maxDiff));
		status = (allOk ? 'PASS' : 'FAIL') + ` — ${out.filter((r) => r.ok).length}/${out.length} cases, max|CPU−GPU|=${maxAll.toExponential(2)}`;
		console.log('[devcomp validate]', status);
		for (const e of engines.values()) e.destroy();
	}

	onMount(validate);
</script>

<svelte:head><title>devcomp — WGSL validation</title></svelte:head>

<main>
	<h1>WGSL field-CA validation (E-series + adder)</h1>
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

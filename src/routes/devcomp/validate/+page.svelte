<script lang="ts">
	import { onMount } from 'svelte';
	import { experimentById, loadParams, forward, forwardMarkers, readOutputs, damageMask, seedGrid, seedMarkers, inputChannels, type RuleConfig } from '$lib/devcomp/rule';
	import { FieldCAEngine } from '$lib/devcomp/engine';
	import e1 from '$lib/devcomp/params/e1_gate.json';
	import e3 from '$lib/devcomp/params/e3_seed.json';
	import adderReactive from '$lib/devcomp/params/adder_reactive.json';
	import wireInvariant from '$lib/devcomp/params/wire_invariant.json';
	import xorInvariant from '$lib/devcomp/params/xor_invariant.json';

	interface Row { label: string; cpu: string; gpu: string; maxDiff: number; ok: boolean; }
	let rows = $state<Row[]>([]);
	let status = $state('running…');

	const PARAM_FILES: Record<string, number[]> = {
		'e1_gate.json': e1, 'e3_seed.json': e3, 'adder_reactive.json': adderReactive,
		'wire_invariant.json': wireInvariant, 'xor_invariant.json': xorInvariant
	};
	const DMG_AT = 32;
	// Each experiment is read where its rule is meant to be read: unstable rules at
	// tGrow, stable ones later (with damage to exercise self-repair).
	const CONFIGS = [
		{ id: 'e1_gate', steps: 24, damage: false },
		{ id: 'e2_repair', steps: 50, damage: false },
		{ id: 'e2_repair', steps: 50, damage: true },
		{ id: 'e3_seed', steps: 50, damage: false },
		{ id: 'e3_seed', steps: 50, damage: true },
		{ id: 'adder', steps: 60, damage: false },
		{ id: 'adder', steps: 60, damage: true }
	];

	// Movable (markers) rules: validated at SEVERAL port placements — the CPU ground
	// truth is forwardMarkers (re-stamps markers every step, like the kernel). Cells are
	// (row·SW + col) on the rule's grid (17×17). xor_invariant is added once trained.
	// NOTE: these rules are trained by expI.ts for POSITION-INVARIANT COMPUTATION only,
	// with no damage in the objective — so this gate certifies routing correctness across
	// placements + GPU-faithfulness, NOT self-repair (that is E2/E3/adder's story; adding
	// damage to the movable objective is tracked as follow-up in the build plan).
	const MOV = (r: number, c: number) => r * 17 + c;
	interface MovCfg { id: string; steps: number; damage: boolean; places: { ins: number[]; out: number }[]; }
	const MOVABLE: MovCfg[] = [
		{ id: 'movable_wire', steps: 60, damage: false, places: [
			{ ins: [MOV(8, 4)], out: MOV(8, 12) }, { ins: [MOV(4, 4)], out: MOV(12, 12) }, { ins: [MOV(3, 8)], out: MOV(13, 8) }
		] },
		// movable XOR: 2 inputs + 1 output. Certifies GPU-faithfulness of the (GPU-trained) rule
		// across placements — the rule is ~68% correct at full-random 17×17, so cases where CPU≈GPU
		// but the output is wrong are the rule's accuracy, NOT a kernel mismatch (maxDiff is the gate).
		{ id: 'movable_xor', steps: 50, damage: false, places: [
			{ ins: [MOV(6, 4), MOV(10, 4)], out: MOV(8, 12) }, { ins: [MOV(4, 5), MOV(4, 11)], out: MOV(12, 8) }
		] }
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
			// --- movable (position-invariant) rules: CPU=forwardMarkers vs GPU, per placement ---
			for (const mc of MOVABLE) {
				const exp = experimentById(mc.id);
				if (!exp || !PARAM_FILES[exp.paramsUrl]) continue; // skip if params not shipped yet
				const rc = exp.cfg;
				let engine = engines.get(rc);
				if (!engine) { engine = await FieldCAEngine.create(rc); engines.set(rc, engine); }
				const par64 = loadParams(rc, PARAM_FILES[exp.paramsUrl]);
				engine.setParams(new Float32Array(par64));
				const inCh = inputChannels(exp);
				const cx = rc.SW >> 1, cy = rc.SH >> 1;
				for (let pi = 0; pi < mc.places.length; pi++) {
					const pl = mc.places[pi];
					for (const cse of exp.cases) {
						const mask = damageMask(rc, cx, cy, 3);
						const cpuFinal = forwardMarkers(rc, par64, pl.ins, [pl.out], cse.in, inCh,
							mc.damage ? { steps: mc.steps, damage: { at: DMG_AT, mask } } : { steps: mc.steps })[mc.steps];
						const isInput = new Uint32Array(rc.N), inputVal = new Float32Array(rc.N);
						pl.ins.forEach((cell, k) => { isInput[cell] = inCh[k] + 1; inputVal[cell] = cse.in[k]; });
						const isOut = new Uint32Array(rc.N); isOut[pl.out] = 1;
						engine.setInputs(isInput, inputVal);
						engine.setOutputs(isOut);
						engine.setDamageKeep(Uint32Array.from(mask));
						engine.seed(new Float32Array(seedMarkers(rc, pl.ins, [pl.out], cse.in, inCh)));
						for (let t = 0; t < mc.steps; t++) engine.step(mc.damage && t + 1 === DMG_AT);
						const gpuFinal = await engine.readState();
						let maxDiff = 0;
						for (let i = 0; i < rc.N * rc.C; i++) maxDiff = Math.max(maxDiff, Math.abs(cpuFinal[i] - gpuFinal[i]));
						const cpuOut = cpuFinal[pl.out * rc.C + 0], gpuOut = gpuFinal[pl.out * rc.C + 0], tgt = cse.out[0];
						const ruleOk = Math.abs(gpuOut - tgt) < 0.2; // rule accuracy (informational; XOR is ~68%)
						// This harness gates on CPU==GPU FAITHFULNESS (maxDiff). Where the rule is wrong,
						// CPU and GPU agree it's wrong → still faithful. Rule accuracy is measured elsewhere.
						out.push({
							label: `${mc.id} P${pi} [${cse.in.join(',')}]${mc.damage ? ' +dmg' : ''}${ruleOk ? '' : ' (rule✗)'}`,
							cpu: cpuOut.toFixed(2), gpu: gpuOut.toFixed(2), maxDiff, ok: maxDiff < 2e-3
						});
					}
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

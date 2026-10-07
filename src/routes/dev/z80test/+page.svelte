<script lang="ts">
	// DEV-ONLY page: differential test of the shipping GPU Z80 core (zilion +
	// Algocell host bits) against a real-Z80 reference emulator, in four
	// configurations: square/hex pair layout × with/without instruction
	// suppression. Not linked from the app. Visit /dev/z80test to run.
	import {
		runZ80DiffTest,
		traceFirstDivergence,
		randomSuppression,
		type DiffReport,
		type LayoutMode
	} from '$lib/dev/z80difftest';

	interface Config {
		label: string;
		mode: LayoutMode;
		suppress: boolean;
	}
	const CONFIGS: Config[] = [
		{ label: 'square · no suppression', mode: 'square', suppress: false },
		{ label: 'square · random 25% ablation', mode: 'square', suppress: true },
		{ label: 'hex (38-byte pair) · no suppression', mode: 'hex', suppress: false },
		{ label: 'hex · random 25% ablation', mode: 'hex', suppress: true }
	];

	let reports = $state<(DiffReport | null)[]>(CONFIGS.map(() => null));
	let running = $state(false);
	let error = $state<string | null>(null);
	let count = $state(5000);
	let steps = $state(128);
	let seed = $state(1);

	async function run() {
		running = true;
		error = null;
		reports = CONFIGS.map(() => null);
		try {
			for (let i = 0; i < CONFIGS.length; i++) {
				const cfg = CONFIGS[i];
				const r = await runZ80DiffTest({
					count,
					steps,
					seed,
					mode: cfg.mode,
					suppress: cfg.suppress ? randomSuppression(seed) : null
				});
				reports[i] = r;
				reports = [...reports];
			}
			// expose for headless inspection
			(window as unknown as { __z80reports: (DiffReport | null)[] }).__z80reports = reports;
		} catch (e) {
			error = e instanceof Error ? e.message + '\n' + e.stack : String(e);
		} finally {
			running = false;
		}
	}

	$effect(() => {
		// expose for headless sweeps during debugging
		(window as unknown as { __runZ80DiffTest: typeof runZ80DiffTest }).__runZ80DiffTest =
			runZ80DiffTest;
		(window as unknown as { __traceDiverge: typeof traceFirstDivergence }).__traceDiverge =
			traceFirstDivergence;
		(window as unknown as { __randomSuppression: typeof randomSuppression }).__randomSuppression =
			randomSuppression;
		run();
	});

	function hex(n: number, w = 2): string {
		return n.toString(16).toUpperCase().padStart(w, '0');
	}
	// "Genuine" = divergences not explained by the reference emulator's known
	// DD/FD-NOP quirk. That is the number that must be zero.
	function verdict(r: DiffReport): 'pass' | 'quirk' | 'fail' {
		if (r.realTrue > 0) return 'fail';
		if (r.real > 0) return 'quirk';
		return 'pass';
	}
</script>

<div class="wrap">
	<h1>Z80 differential test — GPU core vs real-Z80 reference</h1>
	<div class="controls">
		<label
			>cases <input type="number" bind:value={count} min="100" max="200000" step="1000" /></label
		>
		<label>steps <input type="number" bind:value={steps} min="1" max="1024" step="16" /></label>
		<label>seed <input type="number" bind:value={seed} min="1" max="999999" step="1" /></label>
		<button onclick={run} disabled={running}>{running ? 'Running…' : 'Run all'}</button>
	</div>

	{#if error}
		<pre class="error">{error}</pre>
	{/if}

	<table class="matrix">
		<thead>
			<tr>
				<th>configuration</th>
				<th>verdict</th>
				<th>genuine</th>
				<th>oracle quirk</th>
				<th>mem / reg-only</th>
				<th>benign F3/F5</th>
				<th>suppressed</th>
				<th>time</th>
			</tr>
		</thead>
		<tbody>
			{#each CONFIGS as cfg, i (cfg.label)}
				{@const r = reports[i]}
				<tr class={r ? verdict(r) : running ? 'pending' : ''}>
					<td>{cfg.label}</td>
					{#if r}
						<td><strong>{verdict(r).toUpperCase()}</strong></td>
						<td>{r.realTrue}</td>
						<td>{r.oracleQuirk}</td>
						<td>{r.realMem} / {r.realRegOnly}</td>
						<td>{r.benign}</td>
						<td>{r.suppressed}</td>
						<td>{(r.durationMs / 1000).toFixed(1)} s</td>
					{:else}
						<td colspan="7">{running ? 'running…' : '—'}</td>
					{/if}
				</tr>
			{/each}
		</tbody>
	</table>
	<p class="note">
		{count.toLocaleString()} random programs × {steps} steps per configuration. "Genuine" counts divergences
		in memory or documented registers not caused by the reference emulator's known DD/FD-NOP quirk; it
		must be 0. Ablation sets include each instruction on the base, CB and ED pages with probability 0.25,
		applied identically to both sides.
	</p>

	{#each CONFIGS as cfg, i (cfg.label)}
		{@const r = reports[i]}
		{#if r && r.mismatches.length > 0}
			<h2>{cfg.label} — mismatch samples ({r.mismatches.length} shown)</h2>
			{#each r.mismatches.slice(0, 12) as m (m.caseIdx)}
				<div class="mm" class:benign={m.benign} class:quirk={m.oracleQuirk && !m.benign}>
					<div class="mm-head">
						case #{m.caseIdx}
						{#if m.benign}<span class="tag">benign</span>{/if}
						{#if m.oracleQuirk}<span class="tag">oracle quirk</span>{/if}
						{#if m.memDiffByte >= 0}<span class="tag red">mem@{m.memDiffByte}</span>{/if}
						{#each m.regDiffs as rd (rd.name)}
							<span class="tag red"
								>{rd.name}: gpu={hex(rd.gpu, rd.name === 'sp' || rd.name === 'pc' ? 4 : 2)} cpu={hex(
									rd.cpu,
									rd.name === 'sp' || rd.name === 'pc' ? 4 : 2
								)}</span
							>
						{/each}
					</div>
					<div class="mm-bytes">{m.input.map((b) => hex(b)).join(' ')}</div>
					<div class="mm-asm">{m.disasm.join('  ·  ')}</div>
				</div>
			{/each}
		{/if}
	{/each}
</div>

<style>
	.wrap {
		max-width: 960px;
		margin: 0 auto;
		padding: 24px;
		font-family: system-ui, sans-serif;
		color: #ddd;
		background: #14141a;
		min-height: 100vh;
		height: 100vh;
		overflow-y: auto; /* the app layout sets body overflow:hidden */
		box-sizing: border-box;
	}
	.matrix {
		width: 100%;
		border-collapse: collapse;
		font-size: 13px;
		margin-bottom: 10px;
	}
	.matrix th,
	.matrix td {
		text-align: left;
		padding: 6px 10px;
		border-bottom: 1px solid #2a2a33;
	}
	.matrix th {
		color: #99a;
		font-weight: 500;
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.matrix tr.pass td {
		background: #12351c;
	}
	.matrix tr.quirk td {
		background: #2e2a12;
	}
	.matrix tr.fail td {
		background: #3a1414;
	}
	.matrix tr.pending td {
		color: #888;
	}
	.note {
		font-size: 12px;
		color: #99a;
		margin-bottom: 20px;
	}
	.mm.quirk {
		border-color: #875;
		background: #1d1a14;
	}
	h1 {
		font-size: 18px;
		margin-bottom: 12px;
	}
	.controls {
		display: flex;
		gap: 12px;
		align-items: center;
		margin-bottom: 16px;
	}
	.controls input {
		width: 90px;
		background: #222;
		color: #ddd;
		border: 1px solid #444;
		border-radius: 4px;
		padding: 3px 6px;
	}
	button {
		padding: 5px 16px;
		background: #2a4;
		color: #000;
		border: none;
		border-radius: 5px;
		cursor: pointer;
		font-weight: 600;
	}
	button:disabled {
		opacity: 0.5;
	}
	.error {
		color: #f77;
		white-space: pre-wrap;
		font-size: 12px;
	}
	h2 {
		font-size: 14px;
		margin: 16px 0 8px;
	}
	.mm {
		border: 1px solid #a33;
		border-radius: 5px;
		padding: 8px 10px;
		margin-bottom: 8px;
		background: #1c1416;
	}
	.mm.benign {
		border-color: #665;
		background: #1a1a14;
	}
	.mm-head {
		font-size: 12px;
		margin-bottom: 4px;
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		align-items: center;
	}
	.tag {
		font-size: 11px;
		padding: 1px 6px;
		border-radius: 3px;
		background: #333;
	}
	.tag.red {
		background: #522;
		color: #fbb;
	}
	.mm-bytes {
		font-family: monospace;
		font-size: 11px;
		color: #9ab;
		word-break: break-all;
	}
	.mm-asm {
		font-family: monospace;
		font-size: 11px;
		color: #8a8;
		margin-top: 3px;
	}
</style>

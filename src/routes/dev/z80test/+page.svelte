<script lang="ts">
	// DEV-ONLY page: differential test of the GPU Z80 core vs the CPU Z80 twin.
	// Not linked from the app. Visit /dev/z80test to run.
	import { runZ80DiffTest, traceFirstDivergence, type DiffReport } from '$lib/dev/z80difftest';

	let report = $state<DiffReport | null>(null);
	let running = $state(false);
	let error = $state<string | null>(null);
	let count = $state(5000);
	let steps = $state(128);

	async function run() {
		running = true;
		error = null;
		report = null;
		try {
			const r = await runZ80DiffTest({ count, steps, seed: 1 });
			report = r;
			// expose for headless inspection
			(window as unknown as { __z80report: DiffReport }).__z80report = r;
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
		run();
	});

	function hex(n: number, w = 2): string {
		return n.toString(16).toUpperCase().padStart(w, '0');
	}
</script>

<div class="wrap">
	<h1>Z80 differential test — GPU core vs CPU twin</h1>
	<div class="controls">
		<label>cases <input type="number" bind:value={count} min="100" max="200000" step="1000" /></label>
		<label>steps <input type="number" bind:value={steps} min="1" max="1024" step="16" /></label>
		<button onclick={run} disabled={running}>{running ? 'Running…' : 'Run'}</button>
	</div>

	{#if error}
		<pre class="error">{error}</pre>
	{/if}

	{#if report}
		<div class="summary" class:pass={report.real === 0} class:fail={report.real > 0}>
			<strong>{report.real === 0 ? 'PASS' : 'FAIL'}</strong>
			— {report.total.toLocaleString()} random programs × {report.steps} steps
			· real mismatches: <strong>{report.real}</strong>
			· benign (F3/F5 only): {report.benign}
		</div>

		{#if report.mismatches.length > 0}
			<h2>Mismatch samples ({report.mismatches.length} shown)</h2>
			{#each report.mismatches as m (m.caseIdx)}
				<div class="mm" class:benign={m.benign}>
					<div class="mm-head">
						case #{m.caseIdx}
						{#if m.benign}<span class="tag">benign</span>{/if}
						{#if m.memDiffByte >= 0}<span class="tag red">mem@{m.memDiffByte}</span>{/if}
						{#each m.regDiffs as rd (rd.name)}
							<span class="tag red">{rd.name}: gpu={hex(rd.gpu, rd.name === 'sp' || rd.name === 'pc' ? 4 : 2)} cpu={hex(rd.cpu, rd.name === 'sp' || rd.name === 'pc' ? 4 : 2)}</span>
						{/each}
					</div>
					<div class="mm-bytes">{m.input.map((b) => hex(b)).join(' ')}</div>
					<div class="mm-asm">{m.disasm.join('  ·  ')}</div>
				</div>
			{/each}
		{/if}
	{:else if running}
		<p>Running {count.toLocaleString()} programs on GPU + CPU…</p>
	{/if}
</div>

<style>
	.wrap {
		max-width: 900px;
		margin: 0 auto;
		padding: 24px;
		font-family: system-ui, sans-serif;
		color: #ddd;
		background: #14141a;
		min-height: 100vh;
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
	.summary {
		padding: 12px 16px;
		border-radius: 6px;
		margin-bottom: 16px;
		font-size: 14px;
	}
	.summary.pass {
		background: #12351c;
		border: 1px solid #2a4;
	}
	.summary.fail {
		background: #3a1414;
		border: 1px solid #a33;
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

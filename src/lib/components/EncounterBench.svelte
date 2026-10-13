<script lang="ts">
	import { untrack } from 'svelte';
	import type { GPUEngine } from '$lib/gpu/engine';
	import type { CultureResult, EncounterObservation } from '$lib/lab/types';
	import { stoppingReason, type AssayResult } from '$lib/gpu/assays';
	let { engine, tape, seed, onpause, onbusy, byteColor } = $props<{
		engine: GPUEngine;
		tape: Uint8Array;
		seed: number;
		onpause: () => void;
		onbusy: (busy: boolean) => void;
		byteColor: (byte: number) => string;
	}>();
	const inspectedCondition = untrack(() => JSON.stringify(engine.settings));
	type Test = 'culture' | 'self' | 'inflow' | 'marker';
	type Lens = 'fetch' | 'starts' | 'writes';
	const tabs: { id: Test; name: string }[] = [
		{ id: 'culture', name: 'Culture' },
		{ id: 'self', name: 'Self-pair' },
		{ id: 'inflow', name: 'Partner sensitivity' },
		{ id: 'marker', name: 'Marker transfer' }
	];
	let active = $state<Test>('culture');
	let lens = $state<Lens>('fetch');
	let culture = $state.raw<CultureResult | null>(null);
	let assays = $state.raw<AssayResult | null>(null);
	let busy = $state(false);
	let error = $state('');
	let trials = $state(16);
	let position = $state(8);
	let value = $state(126);
	let variant = $state<'baseline' | 'perturbed'>('baseline');
	const hex = (byte: number) => byte.toString(16).toUpperCase().padStart(2, '0');
	const pct = (fraction: number) => `${(fraction * 100).toFixed(1)}%`;
	const record = $derived<EncounterObservation | null>(
		active === 'culture'
			? (culture?.example ?? null)
			: active === 'self'
				? (assays?.selfPair ?? null)
				: active === 'inflow'
					? (assays?.partnerSensitivity.example[variant] ?? null)
					: (assays?.marker.mutant.example ?? null)
	);
	const evidenceSettings = $derived(active === 'culture' ? culture?.settings : assays?.settings);
	const L = $derived(evidenceSettings?.length ?? tape.length);
	const ring = $derived(evidenceSettings?.ring ?? engine.settings.ring);
	const laneNames = $derived(
		ring > L * 2
			? ['A · selected tape', 'B · partner', 'Discarded padding']
			: ['A · selected tape', 'B · partner']
	);
	const writes = $derived(record ? Object.entries(record.writes) : []);
	const writeMax = $derived(Math.max(1, ...writes.map(([, count]) => count)));
	async function run() {
		if (busy) return;
		if (JSON.stringify(engine.settings) !== inspectedCondition) {
			error = 'The condition changed. Close the bench and inspect a tape again before testing.';
			return;
		}
		onpause();
		busy = true;
		onbusy(true);
		error = '';
		try {
			if (active === 'culture')
				culture = await engine.runCulture(tape, { trials, seed, generations: 2 });
			else
				assays = await engine.runAssays(tape, {
					seed,
					trials,
					position: Math.min(position, tape.length - 1),
					value:
						active !== 'marker' && tape[Math.min(position, tape.length - 1)] === value
							? (value + 1) % 256
							: value
				});
		} catch (cause) {
			error = cause instanceof Error ? cause.message : String(cause);
		} finally {
			busy = false;
			onbusy(false);
		}
	}
	function saveEvidence() {
		const result = active === 'culture' ? culture : assays;
		if (!result) return;
		const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
		const url = URL.createObjectURL(blob),
			a = document.createElement('a');
		a.href = url;
		a.download = `algocell-${active}-seed-${seed}.json`;
		a.click();
		setTimeout(() => URL.revokeObjectURL(url), 1000);
	}
</script>

<section class="bench" aria-label="Encounter bench">
	<nav class="test-tabs" aria-label="Encounter tests">
		{#each tabs as tab (tab.id)}<button
				class:active={active === tab.id}
				aria-pressed={active === tab.id}
				onclick={() => (active = tab.id)}>{tab.name}</button
			>{/each}
	</nav>
	<div class="test-question">
		{active === 'culture'
			? 'Does this tape make more copies across fresh partners?'
			: active === 'self'
				? 'What happens when two copies of this tape meet?'
				: active === 'inflow'
					? 'Does changing the partner change the output?'
					: 'Does a changed byte persist through fresh partners?'}
	</div>
	<div class="test-controls">
		<label
			>Partners <select aria-label="Test partners" bind:value={trials} disabled={busy}
				><option value={16}>16</option><option value={32}>32</option><option value={64}>64</option
				></select
			></label
		><span>Seed {seed} · mutation off</span><button
			class="primary"
			disabled={busy}
			onclick={() => void run()}
			>{busy
				? 'Testing…'
				: active === 'culture'
					? 'Run culture test'
					: 'Run comparison tests'}</button
		>
	</div>
	{#if active !== 'culture'}<details class="test-options" open={active === 'marker'}>
			<summary>Marker intervention</summary>
			<div class="marker-inputs">
				<label
					>Byte position <input
						aria-label="Assay marker position"
						type="number"
						min="0"
						max={tape.length - 1}
						bind:value={position}
					/></label
				><label
					>New value <input
						aria-label="Assay marker value"
						type="number"
						min="0"
						max="255"
						bind:value
					/></label
				><span>{hex(tape[Math.min(position, tape.length - 1)] ?? 0)} → {hex(value)}</span>
			</div>
			<p class="note">
				Comparison tests include a self-pair, matched partner perturbations and two marker
				transfers. They use a separate machine.
			</p>
		</details>{/if}
	{#if error}<p class="error" role="alert">{error}</p>{/if}
	{#if busy}<p class="note" role="status">
			Testing the selected tape. The population is paused.
		</p>{/if}
	{#if active === 'culture' && culture}
		<p class="verdict">
			{culture.positive === null
				? 'No informative partners'
				: culture.positive
					? 'Continuing production in this screen'
					: 'Below the culture screen'}<span>{culture.confinement} · {culture.trials} partners</span
			>
		</p>
		<table>
			<thead
				><tr><th>Transfer</th><th>Gain</th><th>Exact</th><th>Rotated</th><th>Outside A</th></tr
				></thead
			><tbody
				>{#each culture.generations as gen (gen.generation)}<tr
						><td>{gen.generation}</td><td>{gen.gain === null ? 'unknown' : gen.gain.toFixed(3)}</td
						><td>{gen.exactCopies}/{culture.trials}</td><td>{gen.cyclicCopies}/{culture.trials}</td
						><td>{gen.partnerEntries}/{culture.trials}</td></tr
					>{/each}</tbody
			>
		</table>
		<p class="note">
			Gain compares similarity with the starting partner. A second-transfer gain ≥ 0.3 passes this
			screen; it does not establish heredity.
		</p>
	{:else if active === 'self' && assays}
		<div class="result-pair">
			<div>
				<strong
					>{assays.selfPair.after.slice(0, L).filter((b, i) => b !== assays!.parent[i])
						.length}/{L}</strong
				><span>changed positions in A</span>
			</div>
			<div>
				<strong
					>{assays.selfPair.after.slice(L, L * 2).filter((b, i) => b !== assays!.parent[i])
						.length}/{L}</strong
				><span>changed positions in B</span>
			</div>
		</div>
		<p class="note">
			Both tapes began identical. Unchanged output alone cannot distinguish active copying from
			doing nothing; inspect the measured writes.
		</p>
	{:else if active === 'inflow' && assays}
		<div class="result-pair">
			<div>
				<strong
					>{assays.partnerSensitivity.changedOffspring}/{assays.partnerSensitivity.trials}</strong
				><span>partners with changed output</span>
			</div>
			<div>
				<strong>{pct(assays.partnerSensitivity.meanChangedFraction)}</strong><span
					>mean changed B positions</span
				>
			</div>
		</div>
		<p class="note">
			A changed in {assays.partnerSensitivity.changedParent}/{assays.partnerSensitivity.trials} trials.
			{assays.partnerSensitivity.perturbation}
		</p>
		<div class="variant-row">
			<button class:active={variant === 'baseline'} onclick={() => (variant = 'baseline')}
				>Original partner</button
			><button class:active={variant === 'perturbed'} onclick={() => (variant = 'perturbed')}
				>Changed partner</button
			>
		</div>
	{:else if active === 'marker' && assays}
		<div class="marker-heading">
			Byte {assays.marker.position} = {hex(assays.marker.value)}
			<span>Same-position retention</span>
		</div>
		<div class="retention-key">
			<span><i></i>Changed parent</span><span><i class="control"></i>Original control</span><span
				>0–100%</span
			>
		</div>
		{#each assays.marker.retention as item (item.generation)}<div class="retention">
				<span>Transfer {item.generation}</span>
				<div>
					<div class="bar"><i style:width="{item.mutant * 100}%"></i></div>
					<div class="bar"><i class="control" style:width="{item.wildType * 100}%"></i></div>
				</div>
				<strong>{pct(item.mutant)}<small>{pct(item.wildType)}</small></strong>
			</div>{/each}
		<p class="note">{assays.marker.note}</p>
	{/if}
	{#if record}
		<div class="encounter-header">
			<h3>Example encounter</h3>
			<span>{record.steps} steps · {stoppingReason(record)}</span>
		</div>
		<nav class="lens-tabs" aria-label="Encounter view">
			<button class:active={lens === 'fetch'} onclick={() => (lens = 'fetch')}>Fetched bytes</button
			><button class:active={lens === 'starts'} onclick={() => (lens = 'starts')}
				>Instruction starts</button
			><button class:active={lens === 'writes'} onclick={() => (lens = 'writes')}
				>Write activity</button
			>
		</nav>
		{#if lens === 'writes'}
			<div class="writes">
				{#each writes as [name, count] (name)}<div class="write-row">
						<span>{name}</span>
						<div class="bar"><i style:width="{(count / writeMax) * 100}%"></i></div>
						<strong>{count}</strong>
					</div>{/each}
			</div>
			<p class="note">
				Measured stores by memory counter, including writes that keep the same value. Bar scale: 0–{writeMax}
				writes.
			</p>
		{:else}
			<div class="lane-key">
				<span><i></i>{lens === 'fetch' ? 'Fetched, including operands' : 'Instruction start'}</span
				><span>Underline = start</span>
			</div>
			{#each laneNames as name, lane (name)}
				{@const offset = lane * L}
				{@const count = lane === 2 ? ring - 2 * L : L}
				<div class="lane-label">{name}<span>{offset}–{offset + count - 1}</span></div>
				<div class="lane" style:--lane-columns={Math.min(count, 16)}>
					{#each Array.from({ length: count }, (_, i) => offset + i) as address (address)}<span
							class:lit={(lens === 'fetch' ? record.fetched : record.starts).includes(address)}
							class:started={record.starts.includes(address)}
							style:--byte-color={byteColor(record.before[address] ?? 0)}
							title="Address {address} · before {hex(
								record.before[address] ?? 0
							)} · after {address < record.after.length
								? hex(record.after[address])
								: 'not retained'}"><small>{address}</small>{hex(record.before[address] ?? 0)}</span
						>{/each}
				</div>
			{/each}
		{/if}
		<details class="after-tape">
			<summary>Partner after encounter</summary>
			<div class="lane" style:--lane-columns={Math.min(L, 16)}>
				{#each record.after.slice(L, L * 2) as byte, index (index)}<span
						class:changed={byte !== record.before[L + index]}
						style:--byte-color={byteColor(byte)}><small>{index}</small>{hex(byte)}</span
					>{/each}
			</div>
			<p class="note">
				Copper border marks a changed value. Changes are distinct from measured stores.
			</p>
		</details>
		<div class="evidence-footer">
			<span
				>{evidenceSettings?.isa.toUpperCase()} · L{L}/P{ring} · budget {evidenceSettings?.steps}</span
			><button onclick={saveEvidence}>Save evidence JSON</button>
		</div>
	{:else if !busy}<div class="empty">Run this test to see measured execution and output.</div>{/if}
</section>

<style>
	.bench {
		margin-top: 13px;
		padding-top: 11px;
		border-top: 1px solid var(--border-muted);
		color: var(--text-secondary);
		font-size: 12px;
	}
	.test-tabs,
	.lens-tabs,
	.variant-row {
		display: flex;
		gap: 4px;
		flex-wrap: wrap;
	}
	button,
	select,
	input {
		font: inherit;
		color: inherit;
	}
	button {
		background: var(--bg-subtle);
		border: 1px solid var(--border-muted);
		border-radius: 5px;
		padding: 6px 8px;
		cursor: pointer;
		white-space: nowrap;
	}
	button:hover {
		background: var(--bg-hover);
	}
	button.active,
	button.primary {
		color: var(--accent-warm);
		border-color: #c8875a66;
		background: #c8875a1a;
	}
	button:disabled {
		opacity: 0.45;
		cursor: default;
	}
	.test-tabs button {
		flex: 1;
		font-size: 11px;
	}
	.test-question {
		font-size: 13px;
		line-height: 1.5;
		margin: 13px 0 9px;
		color: var(--text-primary);
	}
	.test-controls {
		display: flex;
		gap: 9px;
		align-items: center;
		flex-wrap: wrap;
	}
	.test-controls label {
		font-size: 11px;
		color: var(--text-muted);
	}
	select {
		appearance: auto;
		color-scheme: dark;
		background: #1b1b1b;
		background-image: none;
		border: 1px solid var(--border-muted);
		border-radius: 4px;
		padding: 3px 5px;
		margin-left: 5px;
	}
	.test-controls > span {
		font: 10px monospace;
		color: var(--text-subtle);
		flex: 1;
	}
	.test-controls > button {
		font-size: 11px;
	}
	.test-options {
		margin: 11px 0;
		color: var(--text-muted);
	}
	summary {
		font-size: 11px;
		cursor: pointer;
	}
	.marker-inputs {
		display: flex;
		gap: 12px;
		margin: 9px 0;
		align-items: center;
	}
	.marker-inputs label {
		display: flex;
		gap: 6px;
		align-items: center;
		font-size: 11px;
	}
	.marker-inputs input {
		width: 55px;
		background: var(--bg-subtle);
		border: 1px solid var(--border-muted);
		border-radius: 4px;
		padding: 4px;
		font: 12px monospace;
	}
	.marker-inputs > span {
		font: 11px monospace;
		color: var(--accent-warm);
	}
	.note {
		font-size: 11px;
		line-height: 1.55;
		margin: 9px 0;
		color: var(--text-muted);
	}
	.error {
		font-size: 12px;
		color: var(--accent-rose);
	}
	.verdict {
		font-size: 12px;
		color: var(--accent-emerald);
		margin: 14px 0 9px;
	}
	.verdict span {
		display: block;
		font: 10px monospace;
		margin-top: 5px;
		color: var(--text-muted);
	}
	table {
		width: 100%;
		border-collapse: collapse;
		font: 11px monospace;
		margin: 9px 0;
	}
	th,
	td {
		text-align: right;
		padding: 7px 4px;
		border-bottom: 1px solid var(--border-subtle);
		font-weight: 400;
	}
	th {
		color: var(--text-subtle);
		font-size: 10px;
	}
	td:first-child,
	th:first-child {
		text-align: left;
	}
	.result-pair {
		display: flex;
		gap: 18px;
		border-top: 1px solid var(--border-subtle);
		border-bottom: 1px solid var(--border-subtle);
		margin: 14px 0;
		padding: 12px 0;
	}
	.result-pair div {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}
	.result-pair strong {
		font: 18px monospace;
		color: var(--text-primary);
	}
	.result-pair span {
		font-size: 10px;
		color: var(--text-muted);
	}
	.variant-row button {
		font-size: 11px;
	}
	.marker-heading {
		font: 12px monospace;
		margin: 15px 0 8px;
	}
	.marker-heading span {
		display: block;
		font: 10px sans-serif;
		color: var(--text-subtle);
		margin-top: 4px;
	}
	.retention-key {
		display: flex;
		align-items: center;
		gap: 12px;
		font-size: 9px;
		color: var(--text-muted);
		margin: 9px 0;
	}
	.retention-key span {
		display: flex;
		gap: 4px;
		align-items: center;
	}
	.retention-key i {
		width: 7px;
		height: 7px;
		background: var(--accent);
		border-radius: 2px;
	}
	.retention-key i.control {
		background: #668c9b;
	}
	.retention {
		display: flex;
		gap: 10px;
		align-items: center;
		margin: 9px 0;
	}
	.retention > span {
		font-size: 10px;
		width: 55px;
		color: var(--text-muted);
	}
	.retention > div {
		flex: 1;
		display: flex;
		gap: 3px;
		flex-direction: column;
	}
	.bar {
		background: var(--bg-hover);
		height: 7px;
		border-radius: 2px;
		overflow: hidden;
	}
	.bar i {
		height: 100%;
		background: var(--accent);
		display: block;
	}
	.bar i.control {
		background: #668c9b;
	}
	.retention strong {
		font: 10px monospace;
		width: 43px;
		text-align: right;
	}
	.retention small {
		display: block;
		color: #7b9da9;
		font: 10px monospace;
		margin-top: 2px;
	}
	.encounter-header {
		display: flex;
		align-items: center;
		gap: 8px;
		justify-content: space-between;
		border-top: 1px solid var(--border-muted);
		padding-top: 12px;
		margin-top: 13px;
	}
	.encounter-header h3 {
		font-size: 12px;
		font-weight: 500;
		margin: 0;
	}
	.encounter-header > span {
		font: 10px monospace;
		color: var(--text-muted);
	}
	.lens-tabs {
		margin: 10px 0;
	}
	.lens-tabs button {
		font-size: 10px;
		flex: 1;
	}
	.lane-key {
		display: flex;
		justify-content: space-between;
		gap: 10px;
		font-size: 9px;
		color: var(--text-subtle);
		margin-bottom: 10px;
	}
	.lane-key span {
		display: flex;
		gap: 4px;
		align-items: center;
	}
	.lane-key i {
		background: #c8875a66;
		border: 1px solid var(--accent);
		width: 8px;
		height: 8px;
		border-radius: 2px;
	}
	.lane-label {
		display: flex;
		justify-content: space-between;
		font-size: 10px;
		color: var(--text-muted);
		margin: 8px 0 5px;
	}
	.lane-label span {
		font: 9px monospace;
		color: var(--text-subtle);
	}
	.lane {
		display: grid;
		grid-template-columns: repeat(var(--lane-columns), minmax(0, 1fr));
		gap: 3px;
	}
	.lane > span {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 3px;
		padding: 4px 1px;
		font: 11px monospace;
		color: var(--text-subtle);
		background: var(--bg-subtle);
		border: 1px solid transparent;
		border-radius: 3px;
	}
	.lane small {
		font: 8px monospace;
		color: var(--text-subtle);
	}
	.lane > span.lit {
		background: color-mix(in srgb, var(--byte-color) 38%, #171717);
		color: var(--text-primary);
		border-color: #c8875a55;
	}
	.lane > span.started {
		text-decoration: underline;
		text-underline-offset: 3px;
		text-decoration-color: var(--accent-warm);
	}
	.lane > span.changed {
		border-color: var(--accent);
		color: var(--text-primary);
	}
	.after-tape {
		margin-top: 13px;
	}
	.after-tape .lane {
		margin-top: 8px;
	}
	.writes {
		margin: 12px 0;
	}
	.write-row {
		display: flex;
		gap: 10px;
		align-items: center;
		margin: 11px 0;
	}
	.write-row > span {
		font: 11px monospace;
		width: 70px;
	}
	.write-row > .bar {
		flex: 1;
	}
	.write-row > strong {
		font: 12px monospace;
		width: 40px;
		text-align: right;
	}
	.evidence-footer {
		display: flex;
		gap: 10px;
		align-items: center;
		justify-content: space-between;
		border-top: 1px solid var(--border-muted);
		margin-top: 12px;
		padding-top: 10px;
	}
	.evidence-footer > span {
		font: 9px monospace;
		color: var(--text-subtle);
	}
	.evidence-footer button {
		font-size: 10px;
	}
	.empty {
		margin: 13px 0;
		padding: 18px 10px;
		background: var(--bg-subtle);
		color: var(--text-subtle);
		text-align: center;
		border-radius: 5px;
		font-size: 11px;
	}
	@media (max-width: 768px) {
		.test-tabs button {
			font-size: 10px;
			min-height: 36px;
			padding: 5px 6px;
		}
		.test-controls > span {
			font-size: 9px;
		}
		.marker-inputs {
			gap: 8px;
			flex-wrap: wrap;
		}
		.marker-inputs input {
			font-size: 16px;
		}
		.lane {
			gap: 2px;
		}
		.lane > span {
			font-size: 9px;
			padding: 4px 0;
		}
		.lane small {
			font-size: 7px;
		}
		.encounter-header {
			flex-wrap: wrap;
		}
		.lens-tabs button {
			min-height: 34px;
			font-size: 10px;
		}
		.evidence-footer {
			flex-wrap: wrap;
		}
		.test-controls > button {
			min-height: 36px;
		}
		.test-controls select {
			min-height: 32px;
		}
	}
</style>

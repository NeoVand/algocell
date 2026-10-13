<script lang="ts">
	import { PAPER_PRESETS, clonePresetSettings, HOST_CONTRACT_NOTE } from '$lib/lab/presets';
	import { untrack } from 'svelte';
	import { ablationInstructions } from '$lib/lab/ablation';
	import type { LabSession } from '$lib/lab/session.svelte';
	import type { LabSettings, LabIsa } from '$lib/lab/types';
	let { session }: { session: LabSession } = $props();
	let draft = $state<LabSettings>(
		untrack(() => ({ ...session.settings, ablation: [...session.settings.ablation] }))
	);
	let ablationText = $state(untrack(() => session.settings.ablation.join('; ')));
	let opcode = $state('');
	let selectedPreset = $state(untrack(() => session.presetId || 'stage-g-l16'));
	const preset = $derived(
		PAPER_PRESETS.find((item) => item.id === selectedPreset) ?? PAPER_PRESETS[0]
	);
	const instructions = $derived(ablationInstructions(draft.isa));
	function changeIsa(isa: LabIsa) {
		draft.isa = isa;
		draft.compat = '0.3';
		draft.variant = isa === 'z80' ? 'classic' : 'nmos';
		draft.stack = isa === 'z80' ? 'top' : 'first-push';
		draft.ablation = [];
		ablationText = '';
	}
	async function apply() {
		await session.initialize({
			...draft,
			ablation: ablationText
				.split(';')
				.map((value) => value.trim())
				.filter(Boolean)
		});
	}
	function addOpcode() {
		if (!opcode) return;
		const values = ablationText
			.split(';')
			.map((value) => value.trim())
			.filter(Boolean);
		if (!values.includes(opcode)) values.push(opcode);
		ablationText = values.join('; ');
		opcode = '';
	}
</script>

<aside class="experiments" aria-labelledby="experiment-title">
	<h2 id="experiment-title">Start with a question</h2>
	<p class="rail-intro">Choose a world from the paper, or inspect a constructed mechanism.</p>
	<label class="field-label" for="paper-preset">Experiment</label>
	<select id="paper-preset" bind:value={selectedPreset}
		>{#each PAPER_PRESETS as item (item.id)}<option value={item.id}>{item.name}</option
			>{/each}</select
	>
	<div class="preset-story">
		<h3>{preset.name}</h3>
		<p>{preset.description}</p>
		<div class="look-for">
			<strong>What to look for</strong>
			<p>{preset.whatToLookFor}</p>
		</div>
		<button
			class="primary-button"
			disabled={session.locked}
			onclick={() =>
				session.initialize(clonePresetSettings(preset), preset.initialCondition, preset.id)}
			>Start this experiment</button
		>
	</div>
	<details class="source-detail">
		<summary>Finding & source</summary>
		<p>{preset.finding}</p>
		<p>{preset.limits}</p>
		{#if preset.published.horizonUnit === 'transfers'}
			<p>
				Registered isolated-assay horizon: {preset.published.horizon.toLocaleString()} transfers.
			</p>
			<p>
				Browser population demonstration: {preset.config.width} × {preset.config.height} cells; {preset.config.pairAttempts.toLocaleString()}
				pair attempts per update; mutation disabled.
			</p>
		{:else}
			<p>
				{#if preset.published.width !== null && preset.published.height !== null}{preset.published
						.width} × {preset.published.height} cells;
				{/if}
				{#if preset.published.pairAttempts !== null}{preset.published.pairAttempts.toLocaleString()} pair
					attempts;
				{/if}
				{#if preset.published.mutationAttempts !== null}{preset.published.mutationAttempts.toLocaleString()}
					mutation attempts;
				{/if}
				{preset.published.horizon.toLocaleString()}
				{preset.published.horizonUnit}.
			</p>
		{/if}
		<p>Reset: {preset.published.reset}</p>
		{#if preset.published.seeds.length}<p>
				Registered seeds: {preset.published.seeds.join(', ')}.
			</p>{/if}
		<p>{preset.published.note}</p>
		<code>{preset.source.registration}</code>{#if preset.source.world}<p>
				<code>{preset.source.world}</code>
			</p>{/if}{#each preset.source.results as result (result)}<p><code>{result}</code></p>{/each}
		<p>{HOST_CONTRACT_NOTE}</p>
	</details>
	<div class="rule-heading">
		<h3>World settings</h3>
		<span>Apply starts a new soup</span>
	</div>
	<div class="setting-grid">
		<label
			>Processor<select
				aria-label="Processor"
				value={draft.isa}
				onchange={(event) => changeIsa(event.currentTarget.value as LabIsa)}
				><option value="z80">Z80</option><option value="m6502">NMOS 6502</option><option
					value="i8086">8086 · coherent</option
				></select
			></label
		>
		{#if draft.isa === 'z80'}<label
				>Compatibility<select aria-label="Compatibility" bind:value={draft.compat}
					><option value="0.2" disabled={draft.variant === 'zilog-nmos'}>0.2 · historical</option
					><option value="0.3">0.3 · corrected</option></select
				></label
			><label
				>Variant<select aria-label="Variant" bind:value={draft.variant}
					><option value="classic">Classic</option><option
						value="zilog-nmos"
						disabled={draft.compat === '0.2'}>Zilog NMOS</option
					></select
				></label
			>{/if}
		<label>Seed<input type="number" min="0" max="4294967295" bind:value={draft.seed} /></label>
	</div>
	<div class="chemistry">
		<label for="lethal-probability"
			>Lethal zero <span>{Math.round(draft.lethalProbability * 100)}%</span></label
		><input
			id="lethal-probability"
			type="range"
			min="0"
			max="1"
			step="0.01"
			bind:value={draft.lethalProbability}
		/>
		<p>Chance that 00 halts a new instruction. Operands are unaffected.</p>
	</div>
	<details class="advanced">
		<summary>Geometry, mutation & budget</summary>
		<div class="setting-grid paired">
			<label
				>Tape bytes<input
					type="number"
					min="4"
					max="64"
					bind:value={draft.length}
					onchange={() => (draft.ring = Math.max(draft.ring, draft.length * 2))}
				/></label
			><label
				>Ring bytes<input
					type="number"
					min={draft.length * 2}
					max="256"
					bind:value={draft.ring}
				/></label
			>
			<label>Width<input type="number" min="2" max="200" bind:value={draft.width} /></label><label
				>Height<input type="number" min="2" max="200" bind:value={draft.height} /></label
			>
			<label
				>Pair attempts<input
					type="number"
					min="1"
					max="8192"
					bind:value={draft.pairAttempts}
				/></label
			><label>Step budget<input type="number" min="1" max="1024" bind:value={draft.steps} /></label>
			<label
				>Mutation k<input
					type="number"
					min="0"
					max="31"
					bind:value={draft.mutationExponent}
				/></label
			><label
				>Topology<select aria-label="Topology" bind:value={draft.topology}
					><option value="square">Square</option><option value="hex">Hex neighbors</option></select
				></label
			>
		</div>
		<p class="note">
			Replacement attempts per update: floor(pair attempts / 2ᵏ). k = 31 switches mutation off.
		</p>
		<label class="stack-label"
			>Stack convention<select aria-label="Stack convention" bind:value={draft.stack}
				><option value="top">Top aliases last byte</option><option value="first-push"
					>First push at last byte</option
				>{#if draft.isa === 'z80'}<option value="legacy-sp">Literal SP = 65535</option>{/if}</select
			></label
		>
		<label
			>Dial random draws<select aria-label="Dial random draws" bind:value={draft.dialMode}
				><option value="historical-pcg">Historical PCG</option><option value="counter"
					>Counter-based</option
				></select
			></label
		>
		<label class="check-label"
			><input type="checkbox" bind:checked={draft.randomRegisters} />Random registers at reset</label
		>
	</details>
	<details class="advanced">
		<summary>Instruction ablation</summary>
		<p class="note">
			Skip decoded instructions using the selected processor's opcode table. Bytes remain in memory.
		</p>
		<label
			>Choose an instruction<select aria-label="Choose an instruction" bind:value={opcode}
				><option value="">Select opcode…</option
				>{#each instructions as instruction (instruction.key)}<option value={instruction.key}
						>{instruction.key} · {instruction.mnemonic}</option
					>{/each}</select
			></label
		><button class="subtle-button add-opcode" disabled={!opcode} onclick={addOpcode}
			>Add instruction</button
		><label
			>Ablation patterns<textarea
				rows="3"
				bind:value={ablationText}
				placeholder="Separate patterns with semicolons"
			></textarea></label
		>
	</details>
	<button class="apply-button" disabled={session.locked} onclick={apply}
		>Apply settings & reset</button
	>
	<p class="browser-note">
		A local demonstration using deterministic host pairing. One short run does not reproduce a
		registered cohort.
	</p>
</aside>

<style>
	.experiments {
		min-width: 0;
	}
	h2 {
		font-size: 19px;
		line-height: 1.3;
		margin: 0;
		letter-spacing: -0.3px;
	}
	.rail-intro {
		font-size: 13px;
		line-height: 1.55;
		color: var(--muted);
		margin: 10px 0 20px;
	}
	.field-label {
		display: block;
		font-size: 12px;
		font-weight: 600;
		margin-bottom: 7px;
	}
	select {
		width: 100%;
	}
	.preset-story h3 {
		font-size: 16px;
		margin: 19px 0 8px;
		line-height: 1.35;
	}
	p {
		font-size: 12px;
		line-height: 1.65;
		color: var(--muted);
		margin: 0 0 14px;
	}
	.look-for {
		border-left: 2px solid var(--teal);
		padding-left: 12px;
		margin: 18px 0;
	}
	.look-for strong {
		font-size: 12px;
	}
	.look-for p {
		margin: 5px 0 0;
	}
	.primary-button {
		width: 100%;
		font-size: 13px;
	}
	details {
		margin: 16px 0;
		font-size: 12px;
	}
	summary {
		cursor: pointer;
		min-height: 30px;
		display: list-item;
		line-height: 1.5;
		font-weight: 600;
	}
	details p {
		margin: 10px 0;
	}
	code {
		font:
			12px ui-monospace,
			monospace;
		color: var(--muted);
		overflow-wrap: anywhere;
	}
	.rule-heading {
		border-top: 1px solid var(--line);
		padding-top: 20px;
		margin-top: 24px;
	}
	.rule-heading h3 {
		font-size: 15px;
		margin: 0 0 4px;
	}
	.rule-heading span {
		color: var(--muted);
		font-size: 12px;
	}
	.setting-grid {
		display: grid;
		gap: 12px;
		margin-top: 16px;
	}
	.paired {
		grid-template-columns: repeat(2, minmax(0, 1fr));
	}
	label {
		font-size: 12px;
		color: var(--ink);
		display: block;
	}
	label input:not([type='checkbox']),
	label select,
	label textarea {
		width: 100%;
		margin-top: 6px;
	}
	.setting-grid input {
		min-width: 0;
	}
	.chemistry {
		margin-top: 20px;
	}
	.chemistry label {
		display: flex;
		justify-content: space-between;
		font-weight: 600;
	}
	.chemistry label span {
		color: var(--violet);
		font-variant-numeric: tabular-nums;
	}
	input[type='range'] {
		width: 100%;
		margin: 12px 0 6px;
		accent-color: var(--violet);
	}
	.chemistry p {
		font-size: 12px;
	}
	.advanced {
		border-top: 1px solid var(--line);
		padding-top: 15px;
	}
	.advanced label {
		margin-top: 10px;
	}
	.note {
		font-size: 12px;
	}
	.check-label {
		display: flex;
		align-items: center;
		gap: 8px;
		line-height: 1.5;
	}
	.apply-button {
		width: 100%;
		background: #fff;
		border: 1px solid #aab5bd;
		font-size: 13px;
	}
	.browser-note {
		font-size: 12px;
		margin-top: 16px;
	}
	.add-opcode {
		margin: 8px 0 4px;
	}
	textarea {
		min-height: 64px;
		resize: vertical;
	}
	@media (max-width: 900px) {
		.setting-grid:not(.paired) {
			grid-template-columns: repeat(2, minmax(0, 1fr));
		}
	}
</style>

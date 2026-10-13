<script lang="ts">
	import type { LabSession } from '$lib/lab/session.svelte';
	import { disassembleTape } from '$lib/lab/disassemble';
	let { session }: { session: LabSession } = $props();
	let markerPosition = $state(8);
	let markerValue = $state(165);
	let tab = $state<'tape' | 'culture' | 'marker'>('tape');
	let clipboardMessage = $state('');
	const bytes = $derived(session.selectedTape);
	const lines = $derived(disassembleTape(session.settings.isa, bytes));
	const hex = (value: number) => value.toString(16).padStart(2, '0').toUpperCase();
	const fetchLabel = $derived(
		['Unmeasured', 'Confined', 'Open'][session.fetchStatus[session.selectedCell] ?? 0]
	);
	async function copyBytes() {
		try {
			if (!navigator.clipboard) throw new Error('Clipboard unavailable');
			await navigator.clipboard.writeText(Array.from(bytes, hex).join(' '));
			clipboardMessage = 'Tape bytes copied.';
		} catch {
			clipboardMessage = 'Clipboard unavailable. Select and copy the bytes below.';
		}
	}
</script>

<aside class="inspector" aria-labelledby="inspector-title">
	<div class="inspector-heading">
		<h2 id="inspector-title">One tape, up close</h2>
		<span class="index-badge">#{session.selectedCell}</span>
	</div>
	<p class="intro">Select a cell in the soup, or enter its number.</p>
	<div class="cell-input">
		<label for="selected-cell">Cell</label><input
			id="selected-cell"
			type="number"
			min="0"
			max={session.settings.width * session.settings.height - 1}
			value={session.selectedCell}
			onchange={(event) => {
				session.select(Number(event.currentTarget.value));
				// A clamped entry can leave selectedCell unchanged, so the one-way value would not resync.
				event.currentTarget.value = String(session.selectedCell);
			}}
		/><span
			>{session.settings.isa === 'm6502'
				? '6502'
				: session.settings.isa === 'i8086'
					? '8086'
					: 'Z80'}</span
		>
	</div>
	<div class="tabs" role="group" aria-label="Inspector views">
		{#each ['tape', 'culture', 'marker'] as view (view)}<button
				aria-pressed={tab === view}
				class:active={tab === view}
				onclick={() => (tab = view as typeof tab)}
				>{view === 'tape' ? 'Bytes' : view === 'culture' ? 'Culture' : 'Marker'}</button
			>{/each}
	</div>
	{#if tab === 'tape'}
		<div class="tape-strip" role="group" aria-label="Selected tape bytes">
			{#each Array.from(bytes) as byte, index (index)}<span
					class:zero={byte === 0}
					title={`Byte ${index}: ${hex(byte)}`}><small>{index}</small>{hex(byte)}</span
				>{/each}
		</div>
		<div class="tape-actions">
			<span class="fetch-label"
				><i class:measured={fetchLabel !== 'Unmeasured'}></i>{fetchLabel} fetch</span
			><button class="text-button" onclick={copyBytes} disabled={!bytes.length}>Copy hex</button>
		</div>
		{#if clipboardMessage}<p class="note" role="status">{clipboardMessage}</p>{/if}
		<p class="note">
			Last initiating encounter. Confinement does not establish copying or heredity.
		</p>
		<h3>Linear disassembly</h3>
		<div class="disassembly" role="region" aria-label="Disassembly">
			<table>
				<thead><tr><th>At</th><th>Bytes</th><th>Instruction</th></tr></thead><tbody
					>{#each lines as line (line.offset)}<tr
							><td>{hex(line.offset)}</td><td>{line.bytes.map(hex).join(' ')}</td><td
								>{line.mnemonic}{line.truncated ? ' …' : ''}</td
							></tr
						>{/each}</tbody
				>
			</table>
		</div>
		<p class="note">
			A reading of the current bytes, not a trace of execution. Instructions can rewrite the tape.
		</p>
	{:else if tab === 'culture'}
		<h3>Can this tape reproduce?</h3>
		<p>
			Place it against 16 fresh random partners, then transfer offspring to fresh partners for a
			second generation.
		</p>
		<button
			class="primary-button assay-button"
			onclick={() => session.assay()}
			disabled={!session.available || session.locked}
			>{session.status === 'working' ? 'Testing…' : 'Run culture test'}</button
		>
		{#if session.culture}
			<div class="assay-result">
				<strong
					>{session.culture.positive === null
						? 'Insufficient information'
						: session.culture.positive
							? 'Positive short screen'
							: 'Below the short-screen threshold'}</strong
				><span>{session.culture.confinement} fetch · seed {session.culture.seed}</span>
			</div>
			{#each session.culture.generations as generation (generation.generation)}<div
					class="generation-result"
				>
					<h4>Generation {generation.generation}</h4>
					<dl>
						<div>
							<dt>Exact copies</dt>
							<dd>{generation.exactCopies} / {session.culture.trials}</dd>
						</div>
						<div>
							<dt>Mean best cyclic similarity</dt>
							<dd>{(generation.meanSimilarity * 100).toFixed(1)}%</dd>
						</div>
						<div>
							<dt>Copy gain</dt>
							<dd>{generation.gain === null ? '—' : generation.gain.toFixed(3)}</dd>
						</div>
						<div>
							<dt>Fetched outside parent</dt>
							<dd>{generation.partnerEntries} / {session.culture.trials}</dd>
						</div>
					</dl>
				</div>{/each}
			<details>
				<summary>Example encounter</summary>
				<p>
					Steps: {session.culture.example.steps}<br />Halt: {session.culture.example.halt}<br
					/>Trap: {session.culture.example.trap}
				</p>
				<h4>A · initiating tape after encounter</h4>
				<p class="mono wrap">
					{session.culture.example.after.slice(0, session.settings.length).map(hex).join(' ')}
				</p>
				<h4>B · partner after encounter</h4>
				<p class="mono wrap">
					{session.culture.example.after
						.slice(session.settings.length, session.settings.length * 2)
						.map(hex)
						.join(' ')}
				</p>
				{#if session.culture.example.after.length > session.settings.length * 2}
					<h4>Remaining ring bytes</h4>
					<p class="mono wrap">
						{session.culture.example.after
							.slice(session.settings.length * 2)
							.map(hex)
							.join(' ')}
					</p>
				{/if}
				<h4>Fetch & instruction maps</h4>
				<p class="note">
					Byte offsets in the encounter ring. A starts at 0; B starts at {session.settings.length}.
				</p>
				<p class="mono wrap">
					Fetched: {session.culture.example.fetched.join(', ') || 'none'}<br />Instruction starts: {session.culture.example.starts.join(
						', '
					) || 'none'}
				</p>
				<h4>Write counts</h4>
				<dl>
					{#each Object.entries(session.culture.example.writes) as [segment, count] (segment)}<div>
							<dt>{segment}</dt>
							<dd>{count}</dd>
						</div>{/each}
				</dl>
				<h4>Final machine state</h4>
				<dl>
					{#each Object.entries(session.culture.example.state) as [register, value] (register)}<div>
							<dt>{register}</dt>
							<dd class="mono">{value} · 0x{hex(value)}</dd>
						</div>{/each}
				</dl>
			</details>
		{/if}
		<p class="note">
			A small diagnostic under this world's rules. This is not the full registered culture protocol,
			and a negative result does not prove copying is impossible.
		</p>
	{:else}
		<h3>Does a change travel?</h3>
		<p>Change one byte in an isolated copy and compare its offspring with the unedited tape.</p>
		<div class="marker-fields">
			<label
				>Position<input
					type="number"
					min="0"
					max={session.settings.length - 1}
					bind:value={markerPosition}
				/></label
			><label>Byte value<input type="number" min="0" max="255" bind:value={markerValue} /></label>
		</div>
		<button
			class="primary-button assay-button"
			onclick={() =>
				session.marker(Math.min(markerPosition, session.settings.length - 1), markerValue)}
			disabled={!session.available || session.locked}
			>{session.status === 'working' ? 'Testing…' : 'Test marker transfer'}</button
		>
		{#if session.markerResult}<div class="assay-result">
				<strong
					>Marker {hex(session.markerResult.value)} at byte {session.markerResult.position}</strong
				><span>Edited tape versus original control</span>
			</div>
			{#each session.markerResult.retention as row (row.generation)}<div class="generation-result">
					<h4>Generation {row.generation}</h4>
					<dl>
						<div>
							<dt>Edited marker carriage</dt>
							<dd>{(row.mutant * 100).toFixed(1)}%</dd>
						</div>
						<div>
							<dt>Original control</dt>
							<dd>{(row.wildType * 100).toFixed(1)}%</dd>
						</div>
					</dl>
				</div>{/each}
			<p class="note">{session.markerResult.note}</p>{/if}
		<p class="note">Equal byte values show carriage, not ancestry. The soup itself is unchanged.</p>
	{/if}
</aside>

<style>
	.inspector {
		padding: 20px;
		background: #fff;
		border: 1px solid var(--line);
		border-radius: 10px;
		min-width: 0;
		align-self: start;
	}
	.inspector-heading {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
	}
	h2 {
		font-size: 17px;
		margin: 0;
		line-height: 1.3;
	}
	.index-badge {
		color: var(--muted);
		background: #f2f4f6;
		padding: 3px 6px;
		border-radius: 4px;
		font-size: 12px;
		font-variant-numeric: tabular-nums;
	}
	p {
		font-size: 13px;
		line-height: 1.55;
		color: var(--muted);
		margin: 10px 0 14px;
	}
	.intro {
		font-size: 12px;
		margin: 8px 0 16px;
	}
	.cell-input {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-bottom: 16px;
	}
	.cell-input input {
		flex: 1;
		width: 60px;
	}
	.cell-input span {
		font-size: 12px;
		font-weight: 600;
	}
	label {
		font-size: 13px;
	}
	.tabs {
		display: flex;
		border-bottom: 1px solid var(--line);
		margin: 0 -4px 16px;
		gap: 12px;
	}
	.tabs button {
		padding: 8px 4px;
		border: 0;
		border-bottom: 2px solid transparent;
		border-radius: 0;
		font-size: 13px;
		background: none;
		min-height: 40px;
		color: var(--muted);
	}
	.tabs button.active {
		border-bottom-color: var(--teal);
		color: var(--ink);
		font-weight: 600;
	}
	.tape-strip {
		display: grid;
		grid-template-columns: repeat(4, minmax(0, 1fr));
		gap: 4px;
	}
	.tape-strip span {
		display: flex;
		flex-direction: column;
		font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
		font-size: 15px;
		line-height: 1.4;
		text-align: center;
		padding: 6px 0;
		background: #eaf1f2;
		border-radius: 4px;
	}
	.tape-strip span.zero {
		background: #f1f2f4;
		color: #6b7680;
	}
	small {
		font:
			12px Arial,
			sans-serif;
		color: #72818b;
	}
	.tape-actions {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-top: 12px;
		gap: 8px;
	}
	.fetch-label {
		font-size: 12px;
		display: flex;
		align-items: center;
		gap: 5px;
	}
	i {
		width: 7px;
		height: 7px;
		background: #e6e8eb;
		border: 1px solid #83909a;
		border-radius: 50%;
	}
	i.measured {
		background: #25313a;
	}
	.note {
		font-size: 12px;
		line-height: 1.55;
		margin: 10px 0 0;
	}
	h3 {
		font-size: 13px;
		margin: 20px 0 10px;
	}
	.disassembly {
		overflow: auto;
		max-height: 320px;
	}
	table {
		font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
		font-size: 12px;
		border-collapse: collapse;
		width: 100%;
	}
	th {
		font:
			12px Arial,
			sans-serif;
		color: var(--muted);
		text-align: left;
		padding: 6px 6px 8px 0;
		border-bottom: 1px solid var(--line);
	}
	td {
		vertical-align: top;
		padding: 7px 8px 7px 0;
		border-bottom: 1px solid #eef0f2;
	}
	td:first-child {
		color: var(--muted);
	}
	td:nth-child(2) {
		color: #667780;
		font-size: 12px;
		min-width: 48px;
	}
	td:last-child {
		min-width: 86px;
	}
	.assay-button {
		width: 100%;
	}
	.assay-result {
		border-left: 3px solid var(--teal);
		padding: 10px 12px;
		margin: 18px 0 12px;
		background: #f0f7f7;
	}
	.assay-result strong {
		display: block;
		font-size: 13px;
		line-height: 1.45;
	}
	.assay-result span {
		display: block;
		font-size: 12px;
		color: var(--muted);
		margin-top: 5px;
	}
	h4 {
		font-size: 12px;
		margin: 16px 0 8px;
	}
	dl {
		margin: 0;
		font-size: 12px;
	}
	dl div {
		display: flex;
		gap: 10px;
		justify-content: space-between;
		margin: 6px 0;
	}
	dt {
		color: var(--muted);
	}
	dd {
		margin: 0;
		font-variant-numeric: tabular-nums;
	}
	details {
		font-size: 12px;
		margin-top: 16px;
	}
	summary {
		cursor: pointer;
	}
	.wrap {
		overflow-wrap: anywhere;
	}
	.marker-fields {
		display: flex;
		gap: 12px;
		margin: 16px 0;
	}
	.marker-fields label {
		flex: 1;
		min-width: 0;
	}
	.marker-fields input {
		margin-top: 5px;
		width: 100%;
	}
	@media (max-width: 1200px) and (min-width: 701px) {
		.tape-strip {
			grid-template-columns: repeat(8, minmax(0, 1fr));
		}
	}
</style>

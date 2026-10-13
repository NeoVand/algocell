<script lang="ts">
	import { onMount } from 'svelte';
	import { resolve } from '$app/paths';
	import { LabSession } from '$lib/lab/session.svelte';
	import { PAPER_PRESETS, DEFAULT_PRESET_ID, clonePresetSettings } from '$lib/lab/presets';
	import { parseCheckpoint, serializeCheckpoint } from '$lib/lab/checkpoint';
	import ExperimentControls from '$lib/components/lab/ExperimentControls.svelte';
	import SoupViewport from '$lib/components/lab/SoupViewport.svelte';
	import TapeInspector from '$lib/components/lab/TapeInspector.svelte';
	import Measurements from '$lib/components/lab/Measurements.svelte';
	import Documentation from '$lib/components/lab/Documentation.svelte';

	const session = new LabSession();
	let documentView = $state<'about' | 'guide' | null>(null);
	let importInput: HTMLInputElement | undefined;
	function attachImport(element: HTMLInputElement) {
		importInput = element;
		return () => {
			importInput = undefined;
		};
	}
	const machineName = $derived(
		session.settings.isa === 'z80'
			? `Z80 ${session.settings.compat}`
			: session.settings.isa === 'm6502'
				? 'NMOS 6502'
				: '8086 · coherent fetch'
	);
	const statusLabel = $derived(
		{
			loading: 'Preparing world',
			ready: 'Paused',
			running: 'Running',
			working: 'Testing tape',
			unavailable: 'WebGPU unavailable'
		}[session.status]
	);
	const activePreset = $derived(PAPER_PRESETS.find((preset) => preset.id === session.presetId));

	onMount(() => {
		const preset =
			PAPER_PRESETS.find((preset) => preset.id === DEFAULT_PRESET_ID) ?? PAPER_PRESETS[0];
		void session.initialize(clonePresetSettings(preset), preset.initialCondition, preset.id);
		return () => session.dispose();
	});
	function download(text: string, filename: string, type = 'application/json') {
		const url = URL.createObjectURL(new Blob([text], { type }));
		const link = document.createElement('a');
		link.href = url;
		link.download = filename;
		link.click();
		setTimeout(() => URL.revokeObjectURL(url), 1000);
	}
	function exportCheckpoint() {
		session.pause();
		try {
			if (session.engine)
				download(
					serializeCheckpoint(session.engine.snapshot()),
					`algocell-update-${session.metrics?.update ?? 0}.json`
				);
		} catch (error) {
			session.error = error instanceof Error ? error.message : String(error);
		}
	}
	function exportMeasurements() {
		const rows = [
			'update,encounters,zero_fraction,top_tape_share,diversity,unique_tapes',
			...session.history.map((row) =>
				[
					row.update,
					row.encounters,
					row.zeroFraction,
					row.topShare,
					row.diversity,
					row.uniqueTapes
				].join(',')
			)
		];
		download(rows.join('\n'), 'algocell-measurements.csv', 'text/csv');
	}
	async function importFile(file?: File) {
		if (!file) return;
		try {
			const checkpoint = parseCheckpoint(await file.text());
			await session.restore(checkpoint);
		} catch (error) {
			session.error = error instanceof Error ? error.message : String(error);
		}
	}
	function shortcut(event: KeyboardEvent) {
		if (
			documentView ||
			event.target instanceof HTMLInputElement ||
			event.target instanceof HTMLTextAreaElement ||
			event.target instanceof HTMLSelectElement ||
			(event.target instanceof HTMLElement && event.target.isContentEditable)
		)
			return;
		if (event.code === 'Space' && event.target === document.body) {
			event.preventDefault();
			if (session.status === 'running') session.pause();
			else session.run();
		}
	}
</script>

<svelte:head
	><title>Algocell · A machine-code research instrument</title><meta
		name="description"
		content="Explore machine-code soups, test reproduction and follow variation. A local research instrument powered by Zilion."
	/></svelte:head
>
<svelte:window onkeydown={shortcut} />
<svelte:document
	onvisibilitychange={() => {
		if (document.hidden) session.pause();
	}}
/>

<div class="research-desk">
	<a class="skip-link" href="#soup-workspace">Skip to the soup</a>
	<header class="app-header">
		<a class="brand" href={resolve('/')} aria-label="Algocell home"
			><svg width="31" height="31" viewBox="0 0 31 31" aria-hidden="true"
				><rect x="1" y="1" width="8" height="8" rx="1" fill="#25313a" /><rect
					x="12"
					y="1"
					width="8"
					height="8"
					rx="1"
					fill="#83909a"
				/><rect x="23" y="1" width="7" height="8" rx="1" fill="#e6e8eb" /><rect
					x="1"
					y="12"
					width="8"
					height="8"
					rx="1"
					fill="#83909a"
				/><rect x="12" y="12" width="8" height="8" rx="1" fill="#168a8d" /><rect
					x="23"
					y="12"
					width="7"
					height="8"
					rx="1"
					fill="#83909a"
				/><rect x="1" y="23" width="8" height="7" rx="1" fill="#e6e8eb" /><rect
					x="12"
					y="23"
					width="8"
					height="7"
					rx="1"
					fill="#83909a"
				/><rect x="23" y="23" width="7" height="7" rx="1" fill="#25313a" /></svg
			>
			<div>
				<h1>Algocell</h1>
				<span>Copying is not yet inheriting.</span>
			</div></a
		>
		<nav aria-label="Main navigation">
			<button onclick={() => (documentView = 'guide')}>Guide</button><button
				onclick={() => (documentView = 'about')}>About</button
			><a href={resolve('/legacy')}>Legacy instrument</a>
		</nav>
	</header>
	<div class="desk-introduction">
		<p>A soup of short programs. Observe a pattern, isolate a tape, and test what survives.</p>
		<span>Zilion 0.3.1 · Runs locally</span>
	</div>
	<nav class="mobile-jumps" aria-label="Workspace sections">
		<a href="#experiment-title">Experiments</a><a href="#soup-workspace">Soup</a><a
			href="#inspector-title">Inspect a tape</a
		>
	</nav>
	<main class="desk-grid">
		<div class="experiment-column">
			{#key session.settings}<ExperimentControls {session} />{/key}
		</div>
		<div class="workspace" id="soup-workspace">
			<div class="run-bar">
				<div class="run-actions">
					<button
						class="primary-button run-button"
						disabled={!session.available || (session.locked && session.status !== 'running')}
						onclick={() => (session.status === 'running' ? session.pause() : session.run())}
						><span aria-hidden="true">{session.status === 'running' ? 'Ⅱ' : '▶'}</span
						>{session.status === 'running' ? 'Pause' : 'Run'}</button
					><button disabled={!session.available || session.locked} onclick={() => session.step()}
						>Step</button
					><button class="subtle-button" disabled={session.locked} onclick={session.reset}
						>Reset</button
					>
				</div>
				<div class="run-status" role="status">
					<span class="status-dot" class:running={session.status === 'running'}></span><span
						>{statusLabel}</span
					><strong>{session.metrics?.update.toLocaleString() ?? 0}</strong><span>updates</span>
				</div>
			</div>
			<div class="world-caption">
				<strong>{activePreset?.name ?? 'Custom world'}</strong><span
					>{machineName} · L{session.settings.length} / P{session.settings.ring} · {session.settings
						.steps} steps</span
				>
			</div>
			{#if session.error}<div class="error-message" role="alert">
					<strong>Execution needs attention</strong>
					<p>{session.error}</p>
					<span>The experiments and documentation remain available.</span>
				</div>{/if}
			{#if session.notice}<p class="status-notice" role="status">{session.notice}</p>{/if}
			<SoupViewport {session} />
			<div class="run-footnote">
				<span>Click to inspect · Drag to pan · Scroll to zoom</span><span
					>Run pauses after a 55-second burst</span
				>
			</div>
			<Measurements history={session.history} current={session.metrics} />
			<details class="data-tools">
				<summary>Data & reproducibility</summary>
				<p>
					Keep the complete condition with your observations. Checkpoints use this instrument's
					deterministic scheduler; legacy .acell snapshots open in the legacy instrument.
				</p>
				<div class="export-buttons">
					<button disabled={!session.available || session.locked} onclick={exportCheckpoint}
						>Save checkpoint</button
					><button disabled={session.locked} onclick={() => importInput?.click()}
						>Import checkpoint</button
					><button
						disabled={!session.manifest}
						onclick={() =>
							download(JSON.stringify(session.manifest, null, 2), 'algocell-manifest.json')}
						>Export manifest</button
					><button disabled={!session.history.length} onclick={exportMeasurements}
						>Export measurements</button
					>
				</div>
				<p class="manifest-note">
					{session.metrics?.activePairs.toLocaleString() ?? 0} active encounters in the last update. {session.metrics?.uniqueTapes.toLocaleString() ??
						0} distinct exact tapes.
				</p>
			</details>
		</div>
		<TapeInspector {session} />
	</main>
	<footer class="app-footer">
		<p>A pattern is an observation. Reproduction and inherited variation need their own tests.</p>
		<div class="footer-links">
			<button class="text-button" onclick={() => (documentView = 'about')}
				>The science behind Algocell</button
			><a href={resolve('/legacy')}>Legacy instrument</a>
		</div>
	</footer>
</div>
<input
	class="visually-hidden"
	type="file"
	accept=".json,application/json"
	aria-label="Import checkpoint file"
	{@attach attachImport}
	onchange={(event) => {
		void importFile(event.currentTarget.files?.[0]);
		event.currentTarget.value = '';
	}}
/>
{#if documentView}<Documentation kind={documentView} onclose={() => (documentView = null)} />{/if}

<style>
	.mobile-jumps {
		display: none;
	}
	.footer-links {
		display: flex;
		align-items: center;
		gap: 18px;
		flex-wrap: wrap;
	}
	.footer-links a {
		font-size: 12px;
	}

	.research-desk {
		max-width: 1800px;
		margin: 0 auto;
		padding: 0 32px;
	}
	.app-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 24px;
		padding: 26px 0 23px;
		border-bottom: 1px solid var(--line);
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 13px;
		text-decoration: none;
		color: var(--ink);
	}
	h1 {
		font-size: 27px;
		line-height: 1;
		margin: 0 0 5px;
		letter-spacing: -1px;
	}
	.brand span {
		font-size: 12px;
		color: var(--muted);
	}
	nav {
		display: flex;
		gap: 6px;
		align-items: center;
	}
	nav button,
	nav a {
		font-size: 13px;
		color: var(--ink);
		text-decoration: none;
		background: transparent;
		border: 0;
		padding: 10px 12px;
		border-radius: 5px;
	}
	nav a {
		color: var(--muted);
		border-left: 1px solid var(--line);
		margin-left: 5px;
		border-radius: 0;
	}
	.desk-introduction {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 20px;
		padding: 18px 0 26px;
	}
	.desk-introduction p {
		margin: 0;
		color: #56636d;
		font-size: 14px;
	}
	.desk-introduction span {
		white-space: nowrap;
		font-size: 12px;
		color: var(--muted);
	}
	.desk-grid {
		display: grid;
		grid-template-columns: 235px minmax(0, 1fr) 280px;
		gap: 26px;
		align-items: start;
	}
	.experiment-column {
		padding: 2px 20px 0 0;
		border-right: 1px solid var(--line);
	}
	.workspace {
		min-width: 0;
		scroll-margin: 20px;
	}
	.run-bar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 12px;
		margin-bottom: 17px;
	}
	.run-actions {
		display: flex;
		gap: 7px;
	}
	.run-actions button {
		font-size: 13px;
	}
	.run-button {
		min-width: 90px;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 9px;
	}
	.run-button span {
		font-size: 12px;
	}
	.run-status {
		display: flex;
		gap: 6px;
		align-items: center;
		color: var(--muted);
		font-size: 12px;
		white-space: nowrap;
	}
	.run-status strong {
		font-variant-numeric: tabular-nums;
		color: var(--ink);
		font-weight: normal;
		margin-left: 5px;
	}
	.status-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: #83909a;
	}
	.status-dot.running {
		background: var(--teal);
	}
	.world-caption {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 10px;
		margin-bottom: 12px;
	}
	.world-caption strong {
		font-size: 13px;
		font-weight: 600;
	}
	.world-caption span {
		font-size: 12px;
		color: var(--muted);
	}
	.run-footnote {
		display: flex;
		justify-content: space-between;
		gap: 12px;
		font-size: 12px;
		color: var(--muted);
		margin-top: 9px;
	}
	.error-message {
		background: #fcf2ee;
		border: 1px solid #eccabe;
		padding: 14px;
		margin-bottom: 16px;
		border-radius: 6px;
		font-size: 12px;
	}
	.error-message p {
		margin: 8px 0;
		overflow-wrap: anywhere;
	}
	.error-message span {
		color: var(--muted);
	}
	.status-notice {
		font-size: 12px;
		padding: 10px;
		background: #edf5f5;
		color: #196b6e;
		border-radius: 4px;
	}
	.data-tools {
		border-top: 1px solid var(--line);
		margin-top: 20px;
		padding-top: 15px;
		font-size: 12px;
	}
	summary {
		cursor: pointer;
		min-height: 30px;
	}
	.data-tools p {
		line-height: 1.6;
		color: var(--muted);
		max-width: 75ch;
	}
	.export-buttons {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	.export-buttons button {
		font-size: 12px;
	}
	.manifest-note {
		font-size: 12px;
	}
	.app-footer {
		margin-top: 36px;
		padding: 20px 0 26px;
		border-top: 1px solid var(--line);
		display: flex;
		justify-content: space-between;
		gap: 16px;
		align-items: center;
	}
	.app-footer p {
		font-size: 12px;
		color: var(--muted);
		margin: 0;
	}
	@media (min-width: 1600px) {
		.desk-grid {
			grid-template-columns: 260px minmax(0, 1fr) 310px;
			gap: 30px;
		}
	}
	@media (max-width: 1200px) {
		.research-desk {
			padding: 0 24px;
		}
		.desk-grid {
			grid-template-columns: 220px minmax(0, 1fr);
			gap: 24px;
		}
		.desk-grid :global(.inspector) {
			grid-column: 2;
		}
		.experiment-column {
			grid-row: span 2;
		}
		.world-caption {
			flex-wrap: wrap;
			gap: 5px;
		}
		.desk-introduction span {
			display: none;
		}
	}
	@media (max-width: 900px) {
		.mobile-jumps {
			display: flex;
			margin: 0 0 20px;
			gap: 8px;
			flex-wrap: wrap;
		}
		.mobile-jumps a {
			border: 1px solid var(--line);
			border-radius: 5px;
			background: #fff;
			padding: 9px 12px;
			font-size: 12px;
			margin: 0;
			min-height: 40px;
		}
		.research-desk {
			padding: 0 18px;
		}
		.desk-grid {
			grid-template-columns: 1fr;
			gap: 24px;
		}
		.workspace {
			grid-row: 1;
		}
		.experiment-column {
			grid-row: auto;
			padding: 24px 0 0;
			border-right: 0;
			border-top: 1px solid var(--line);
		}
		.desk-grid :global(.inspector) {
			grid-column: 1;
			grid-row: 2;
		}
		.app-header {
			padding: 20px 0;
		}
		nav {
			gap: 0;
		}
		.app-header nav a {
			display: none;
		}
		.desk-introduction {
			padding: 16px 0 22px;
		}
		.app-footer {
			align-items: flex-start;
		}
	}
	@media (max-width: 520px) {
		.research-desk {
			padding: 0 14px;
		}
		.brand {
			gap: 9px;
		}
		.brand svg {
			width: 25px;
			height: 25px;
		}
		h1 {
			font-size: 24px;
		}
		.brand span {
			font-size: 12px;
		}
		nav button {
			padding: 8px 9px;
			font-size: 12px;
		}
		.desk-introduction p {
			font-size: 13px;
			line-height: 1.5;
		}
		.run-bar {
			flex-wrap: wrap;
			gap: 10px;
		}
		.run-status {
			margin-left: auto;
		}
		.run-footnote {
			display: block;
			line-height: 1.6;
		}
		.run-footnote span {
			display: block;
		}
		.app-footer {
			display: block;
		}
		.app-footer button {
			margin-top: 10px;
		}
	}
</style>

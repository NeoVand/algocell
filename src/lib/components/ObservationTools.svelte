<script lang="ts">
	import type { GPUEngine } from '$lib/gpu/engine';
	import type { FieldMode, RegionMeasurement } from '$lib/observation';
	import type { LabSettings } from '$lib/lab/types';
	import { onMount, untrack } from 'svelte';

	let {
		engine,
		canvas,
		playing,
		externalBusy = false,
		benchOpen = false,
		onbusy,
		condition = '',
		onpause,
		oninspect,
		byteColor
	} = $props<{
		engine: GPUEngine;
		canvas: HTMLCanvasElement;
		playing: boolean;
		externalBusy?: boolean;
		benchOpen?: boolean;
		onbusy: (value: boolean) => void;
		condition?: string;
		onpause: () => void;
		oninspect: (index: number) => void;
		byteColor: (byte: number) => string;
	}>();
	let settings = $state.raw<LabSettings>(untrack(() => engine.settings));
	let observationBar: HTMLDivElement;
	let observationDock: HTMLElement | undefined;
	type Tool = 'navigate' | 'probe' | 'region' | 'history' | 'compare' | 'record';
	type Frame = {
		id: number;
		url: string;
		update: number;
		mode: FieldMode;
		camera: string;
		condition: string;
		isa: string;
		seed: number;
		image: HTMLCanvasElement;
	};
	const modes: { id: FieldMode; name: string; description: string }[] = [
		{ id: 'bytes', name: 'Bytes', description: 'Inherited byte values · zoom to see each tape' },
		{
			id: 'identity',
			name: 'Tape identity',
			description: 'Complete-tape fingerprints · colour collisions are possible'
		},
		{
			id: 'confinement',
			name: 'Open / confined',
			description: 'Last initiating encounter · fetching outside A includes padding'
		},
		{
			id: 'zeros',
			name: 'Zero density',
			description: 'Zero bytes per tape · composition, not a test of inactivity'
		},
		{
			id: 'diversity',
			name: 'Local diversity',
			description: 'Distinct tapes / valid cells in 3 × 3 coordinates; edges use fewer cells'
		},
		{
			id: 'marker',
			name: 'Marker matches',
			description: 'Same-position value matches · not a record of ancestry'
		}
	];
	const tools: { id: Tool; name: string }[] = [
		{ id: 'navigate', name: 'Navigate' },
		{ id: 'probe', name: 'Probe' },
		{ id: 'region', name: 'Region' },
		{ id: 'history', name: 'History' },
		{ id: 'compare', name: 'Compare' },
		{ id: 'record', name: 'Record' }
	];
	let mode = $state<FieldMode>('bytes');
	let tool = $state<Tool>('navigate');
	let busy = $state(false);
	let error = $state('');
	let notice = $state('');
	let update = $state(0);
	let selected = $state(-1);
	let cellInput = $state(0);
	let tape = $state.raw<Uint8Array | null>(null);
	let probeUpdate = $state(0);
	let probeBusy = false;
	let x0 = $state(0),
		y0 = $state(0),
		x1 = $state(9),
		y1 = $state(9);
	let region = $state.raw<RegionMeasurement | null>(null);
	let dragging = $state(false);
	let overlay = $state({ x: 0, y: 0, width: 0, height: 0 });
	let markerPosition = $state(0);
	let markerValue = $state(165);
	let recording = $state(false);
	let frames = $state.raw<Frame[]>([]);
	let frameIndex = $state(-1);
	let replaying = $state(false);
	let nextFrameId = 1;
	let regionSeries = $state.raw<{ id: number; update: number; zero: number; top: number }[]>([]);
	let regionKey = '';
	let nextSampleId = 1;
	let currentCamera = $state('');
	let reference = $state.raw<Frame | null>(null);
	let comparing = $state(false);
	let divider = $state(50);
	let exporting = $state(false);
	let mediaSupported = $state(false);
	let reducedMotion = false;
	const currentMode = $derived(modes.find((entry) => entry.id === mode)!);
	const currentFrame = $derived(frameIndex >= 0 ? frames[frameIndex] : null);
	const zeroFraction = $derived(
		tape ? Array.from(tape).filter((byte) => byte === 0).length / tape.length : 0
	);
	const cells = $derived(settings.width * settings.height);
	const hex = (value: number) => value.toString(16).toUpperCase().padStart(2, '0');
	const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
	const regionPaths = $derived.by(() => {
		const first = regionSeries[0]?.update ?? 0,
			last = regionSeries.at(-1)?.update ?? first;
		const span = Math.max(1, last - first);
		return ['zero', 'top'].map((key) =>
			regionSeries
				.map(
					(sample) =>
						`${28 + ((sample.update - first) / span) * 260},${66 - sample[key as 'zero' | 'top'] * 56}`
				)
				.join(' ')
		);
	});
	const cameraStamp = () =>
		`${engine.view.zoom.toFixed(2)} / ${engine.view.offsetX.toFixed(2)}, ${engine.view.offsetY.toFixed(2)}`;
	export function isNavigating() {
		return tool === 'navigate';
	}

	async function chooseMode(next: FieldMode) {
		if (busy || externalBusy) return;
		busy = true;
		onbusy(true);
		error = '';
		try {
			if (next === 'confinement') onpause();
			if (next === 'marker') engine.setMarker({ position: markerPosition, value: markerValue });
			await engine.setFieldMode(next);
			mode = next;
			frameIndex = -1;
			replaying = false;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : String(cause);
		} finally {
			busy = false;
			onbusy(false);
		}
	}
	function chooseTool(next: Tool) {
		if (externalBusy || busy) return;
		syncSettings();
		tool = next;
		notice = '';
		error = '';
		if (next === 'region') {
			x1 = Math.min(x1, settings.width - 1);
			y1 = Math.min(y1, settings.height - 1);
			onpause();
			updateOverlay();
		}
		if (next === 'probe' && selected < 0) void selectCell(cellInput);
		if (next !== 'history') {
			frameIndex = -1;
			replaying = false;
		}
		if (next !== 'compare') comparing = false;
	}
	function syncSettings() {
		const applied = engine.settings;
		if (JSON.stringify(applied) !== JSON.stringify(settings)) {
			settings = applied;
			resetSelection();
		}
	}
	export function resetSelection() {
		settings = engine.settings;
		selected = -1;
		tape = null;
		regionSeries = [];
		regionKey = '';
		region = null;
		frameIndex = -1;
		frames = [];
		replaying = false;
		if (tool === 'probe' || tool === 'region') tool = 'navigate';
		x0 = y0 = 0;
		x1 = Math.min(9, settings.width - 1);
		y1 = Math.min(9, settings.height - 1);
		markerPosition = Math.min(markerPosition, settings.length - 1);
		if (mode === 'marker') engine.setMarker({ position: markerPosition, value: markerValue });
	}
	export async function selectCell(index: number) {
		if (externalBusy || busy) return;
		syncSettings();
		selected = Math.max(0, Math.min(cells - 1, Math.trunc(index)));
		cellInput = selected;
		tool = 'probe';
		engine.setHoverCell(selected);
		await refreshProbe();
		updateOverlay();
	}
	async function refreshProbe() {
		if (selected < 0 || probeBusy || externalBusy) return;
		probeBusy = true;
		const index = selected;
		try {
			const bytes = await engine.readCell(index);
			if (selected === index) {
				tape = bytes;
				probeUpdate = engine.batchCount;
			}
		} catch (cause) {
			error = cause instanceof Error ? cause.message : String(cause);
		} finally {
			probeBusy = false;
			if (selected !== index && selected >= 0) void refreshProbe();
		}
	}
	function zoom(factor: number) {
		if (externalBusy) return;
		engine.zoomAt(canvas.width / 2, canvas.height / 2, canvas.width, canvas.height, factor);
		updateOverlay();
	}
	function fit() {
		if (externalBusy) return;
		engine.resetView(canvas.width, canvas.height);
		updateOverlay();
	}
	function frameSelection() {
		if (busy || externalBusy || !observationDock) return;
		if (tool === 'probe' && selected < 0) return;
		if (tool === 'region' && ![x0, y0, x1, y1].every(Number.isFinite)) return;
		updateOverlay();
		const rect = canvas.getBoundingClientRect();
		const bar = observationBar.getBoundingClientRect();
		const dock = observationDock.getBoundingClientRect();
		// Measure the usable field only on this explicit camera action. The dock
		// occupies the bottom on phones and the right side on larger screens.
		const left = rect.left + 16;
		const top = Math.max(rect.top + 16, bar.bottom + 16);
		const mobileDock = dock.width > rect.width * 0.6;
		const right = mobileDock ? rect.right - 16 : Math.min(rect.right - 16, dock.left - 16);
		const bottom = mobileDock ? Math.min(rect.bottom - 16, dock.top - 16) : rect.bottom - 16;
		if (right <= left || bottom <= top) return;
		const scaleX = canvas.width / rect.width;
		const scaleY = canvas.height / rect.height;
		if (tool === 'region') {
			const factor = Math.max(
				overlay.width / ((right - left) * 0.84),
				overlay.height / ((bottom - top) * 0.78)
			);
			engine.zoomAt(
				(overlay.x + overlay.width / 2 - rect.left) * scaleX,
				(overlay.y + overlay.height / 2 - rect.top) * scaleY,
				canvas.width,
				canvas.height,
				factor
			);
			updateOverlay();
		}
		engine.pan(
			((left + right) / 2 - overlay.x - overlay.width / 2) * scaleX,
			((top + bottom) / 2 - overlay.y - overlay.height / 2) * scaleY,
			canvas.width,
			canvas.height
		);
		updateOverlay();
		currentCamera = cameraStamp();
	}
	function updateOverlay() {
		const rect = canvas.getBoundingClientRect();
		const scaleX = rect.width / canvas.width,
			scaleY = rect.height / canvas.height;
		let bounds;
		if (tool === 'probe' && selected >= 0)
			bounds = engine.cellBounds(selected, canvas.width, canvas.height);
		else if (tool === 'region') {
			const left = Math.min(x0, x1),
				right = Math.max(x0, x1);
			const top = Math.min(y0, y1),
				bottom = Math.max(y0, y1);
			const indices = [
				top * settings.width + left,
				top * settings.width + right,
				bottom * settings.width + left,
				bottom * settings.width + right
			];
			const corners = indices.map((index) => engine.cellBounds(index, canvas.width, canvas.height));
			const x = Math.min(...corners.map((b) => b.x)),
				y = Math.min(...corners.map((b) => b.y));
			bounds = {
				x,
				y,
				width: Math.max(...corners.map((b) => b.x + b.width)) - x,
				height: Math.max(...corners.map((b) => b.y + b.height)) - y
			};
		}
		if (bounds)
			overlay = {
				x: rect.left + bounds.x * scaleX,
				y: rect.top + bounds.y * scaleY,
				width: bounds.width * scaleX,
				height: bounds.height * scaleY
			};
	}
	function cellAt(clientX: number, clientY: number) {
		const rect = canvas.getBoundingClientRect();
		return engine.screenToCell(
			((clientX - rect.left) * canvas.width) / rect.width,
			((clientY - rect.top) * canvas.height) / rect.height,
			canvas.width,
			canvas.height
		);
	}
	function startRegion(index: number) {
		if (externalBusy || busy) return;
		if (index < 0) return;
		onpause();
		dragging = true;
		region = null;
		x0 = x1 = index % settings.width;
		y0 = y1 = Math.floor(index / settings.width);
		updateOverlay();
	}
	function moveRegion(index: number) {
		if (index < 0 || !dragging) return;
		x1 = index % settings.width;
		y1 = Math.floor(index / settings.width);
		updateOverlay();
	}
	async function sampleRegion() {
		if (busy || externalBusy) return;
		onpause();
		busy = true;
		onbusy(true);
		error = '';
		x0 = Math.max(0, Math.min(settings.width - 1, Math.trunc(x0)));
		x1 = Math.max(0, Math.min(settings.width - 1, Math.trunc(x1)));
		y0 = Math.max(0, Math.min(settings.height - 1, Math.trunc(y0)));
		y1 = Math.max(0, Math.min(settings.height - 1, Math.trunc(y1)));
		try {
			region = await engine.readRegion({ x0, y0, x1, y1 });
			const sample = region!;
			const key = JSON.stringify(sample.region);
			if (key !== regionKey) {
				regionSeries = [];
				regionKey = key;
			}
			regionSeries = [
				...regionSeries.slice(-23),
				{
					id: nextSampleId++,
					update: sample.update,
					zero: sample.zeroFraction,
					top: sample.topShare
				}
			];
			updateOverlay();
		} catch (cause) {
			error = cause instanceof Error ? cause.message : String(cause);
		} finally {
			busy = false;
			onbusy(false);
		}
	}
	function applyMarker() {
		if (externalBusy) return;
		markerPosition = Math.max(0, Math.min(settings.length - 1, Math.trunc(markerPosition)));
		markerValue = Math.max(0, Math.min(255, Math.trunc(markerValue)));
		engine.setMarker({ position: markerPosition, value: markerValue });
	}
	async function plantMarker() {
		if (selected < 0 || busy || externalBusy) return;
		onpause();
		busy = true;
		onbusy(true);
		error = '';
		try {
			applyMarker();
			await engine.plantMarker(selected, markerPosition, markerValue);
			await refreshProbe();
			await engine.setFieldMode('marker');
			mode = 'marker';
			notice = `Planted ${hex(markerValue)} at byte ${markerPosition} in cell ${selected}.`;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : String(cause);
		} finally {
			busy = false;
			onbusy(false);
		}
	}
	function captureFrame(fullResolution = false): Frame {
		engine.render(canvas);
		const image = document.createElement('canvas');
		const scale = fullResolution ? 1 : Math.min(1, 640 / canvas.width, 400 / canvas.height);
		image.width = Math.max(1, Math.round(canvas.width * scale));
		image.height = Math.max(1, Math.round(canvas.height * scale));
		const context = image.getContext('2d')!;
		context.imageSmoothingEnabled = false;
		context.drawImage(canvas, 0, 0, image.width, image.height);
		return {
			id: nextFrameId++,
			condition,
			isa: settings.isa,
			seed: settings.seed,
			image,
			url: image.toDataURL('image/webp', 0.9),
			update: engine.batchCount,
			mode,
			camera: `${engine.view.zoom.toFixed(2)} / ${engine.view.offsetX.toFixed(2)}, ${engine.view.offsetY.toFixed(2)}`
		};
	}
	function captureHistory() {
		if (frameIndex >= 0 || busy || externalBusy) return;
		frames = [...frames.slice(-23), captureFrame()];
	}
	function setRecording(value: boolean) {
		recording = value;
		if (value && !frames.length) captureHistory();
	}
	function scrub(index: number) {
		onpause();
		replaying = false;
		frameIndex = index;
	}
	function returnLive() {
		frameIndex = -1;
		replaying = false;
		notice = 'Live population; the simulation has not been rewound.';
	}
	function pinReference() {
		if (externalBusy || busy) return;
		reference = captureFrame(true);
		comparing = true;
		notice = 'Reference pinned at the current view.';
	}
	function download(blob: Blob, name: string) {
		const url = URL.createObjectURL(blob),
			link = document.createElement('a');
		link.href = url;
		link.download = name;
		link.click();
		setTimeout(() => URL.revokeObjectURL(url), 1000);
	}
	function saveFrame() {
		if (!currentFrame) return;
		const frame = currentFrame;
		frame.image.toBlob((blob) => {
			if (blob) download(blob, `algocell-recorded-${frame.mode}-${frame.update}.png`);
		}, 'image/png');
	}
	function snapshot() {
		if (externalBusy || busy) return;
		const capturedMode = mode,
			capturedUpdate = engine.batchCount;
		engine.render(canvas);
		canvas.toBlob((blob: Blob | null) => {
			if (blob) {
				download(blob, `algocell-${capturedMode}-${capturedUpdate}.png`);
				notice = 'PNG snapshot saved. Use Save checkpoint to keep the population.';
			}
		}, 'image/png');
	}
	async function exportTimelapse() {
		if (frames.length < 2 || exporting || !mediaSupported || externalBusy) return;
		const clip = [...frames];
		let stream: MediaStream | undefined;
		onpause();
		exporting = true;
		error = '';
		try {
			const output = document.createElement('canvas');
			output.width = clip[0].image.width;
			output.height = clip[0].image.height;
			const context = output.getContext('2d')!;
			stream = output.captureStream(10);
			const type = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((mime) =>
				MediaRecorder.isTypeSupported(mime)
			);
			if (!type) throw new Error('This browser cannot export WebM. PNG snapshots are available.');
			const recorder = new MediaRecorder(stream, { mimeType: type });
			const chunks: BlobPart[] = [];
			recorder.ondataavailable = (event) => {
				if (event.data.size) chunks.push(event.data);
			};
			const done = new Promise<Blob>((resolve, reject) => {
				recorder.onstop = () => resolve(new Blob(chunks, { type }));
				recorder.onerror = () => reject(new Error('Timelapse recording failed.'));
			});
			recorder.start();
			for (const frame of clip) {
				context.drawImage(frame.image, 0, 0, output.width, output.height);
				context.fillStyle = 'rgba(14,14,14,.88)';
				context.fillRect(0, output.height - 24, output.width, 24);
				context.fillStyle = '#e8e4e0';
				context.font = '12px monospace';
				context.fillText(`${frame.mode} · update ${frame.update}`, 8, output.height - 8);
				await new Promise((resolve) => setTimeout(resolve, 200));
			}
			recorder.stop();
			const blob = await done;
			download(blob, `algocell-timelapse-${clip[0].update}-${clip.at(-1)!.update}.webm`);
			notice = `${clip.length} sampled frames exported at 5 frames/s.`;
		} catch (cause) {
			error = cause instanceof Error ? cause.message : String(cause);
		} finally {
			stream?.getTracks().forEach((track) => track.stop());
			exporting = false;
		}
	}
	function keyboard(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			tool = 'navigate';
			frameIndex = -1;
			comparing = false;
			replaying = false;
		}
		if (
			tool !== 'probe' ||
			event.target instanceof HTMLInputElement ||
			event.target instanceof HTMLSelectElement ||
			event.target instanceof HTMLTextAreaElement
		)
			return;
		const delta =
			event.key === 'ArrowRight'
				? 1
				: event.key === 'ArrowLeft'
					? -1
					: event.key === 'ArrowDown'
						? settings.width
						: event.key === 'ArrowUp'
							? -settings.width
							: 0;
		if (delta) {
			event.preventDefault();
			void selectCell(Math.max(0, selected) + delta);
		}
	}
	onMount(() => {
		mediaSupported =
			typeof MediaRecorder !== 'undefined' && typeof canvas.captureStream === 'function';
		reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
		const interval = setInterval(() => {
			if (externalBusy || busy) return;
			syncSettings();
			update = engine.batchCount;
			currentCamera = cameraStamp();
			updateOverlay();
			if (tool === 'probe' && playing) void refreshProbe();
		}, 500);
		const historyInterval = setInterval(() => {
			if (recording && playing) captureHistory();
		}, 1000);
		const replayInterval = setInterval(() => {
			if (replaying && frames.length) frameIndex = (frameIndex + 1) % frames.length;
		}, 400);
		const down = (event: MouseEvent) => {
			if (tool !== 'region' && tool !== 'probe') return;
			event.preventDefault();
			event.stopImmediatePropagation();
			const index = cellAt(event.clientX, event.clientY);
			if (tool === 'region') startRegion(index);
			else if (index >= 0) void selectCell(index);
		};
		const move = (event: MouseEvent) => {
			if (dragging) {
				event.stopImmediatePropagation();
				moveRegion(cellAt(event.clientX, event.clientY));
			}
			updateOverlay();
		};
		const up = (event: MouseEvent) => {
			if (tool === 'probe' || tool === 'region') event.stopImmediatePropagation();
			if (dragging) {
				moveRegion(cellAt(event.clientX, event.clientY));
				dragging = false;
				void sampleRegion();
			}
		};
		const touchStart = (event: TouchEvent) => {
			if (tool !== 'region' || event.touches.length !== 1) return;
			event.preventDefault();
			event.stopImmediatePropagation();
			startRegion(cellAt(event.touches[0].clientX, event.touches[0].clientY));
		};
		const touchMove = (event: TouchEvent) => {
			if (dragging && event.touches.length === 1) {
				event.preventDefault();
				event.stopImmediatePropagation();
				moveRegion(cellAt(event.touches[0].clientX, event.touches[0].clientY));
			}
		};
		const touchEnd = (event: TouchEvent) => {
			if (dragging) {
				event.preventDefault();
				event.stopImmediatePropagation();
				dragging = false;
				void sampleRegion();
			}
		};
		canvas.addEventListener('mousedown', down, true);
		canvas.addEventListener('mousemove', move, true);
		canvas.addEventListener('mouseup', up, true);
		canvas.addEventListener('touchstart', touchStart, { capture: true, passive: false });
		canvas.addEventListener('touchmove', touchMove, { capture: true, passive: false });
		canvas.addEventListener('touchend', touchEnd, { capture: true, passive: false });
		return () => {
			clearInterval(interval);
			clearInterval(historyInterval);
			clearInterval(replayInterval);
			canvas.removeEventListener('mousedown', down, true);
			canvas.removeEventListener('mousemove', move, true);
			canvas.removeEventListener('mouseup', up, true);
			canvas.removeEventListener('touchstart', touchStart, true);
			canvas.removeEventListener('touchmove', touchMove, true);
			canvas.removeEventListener('touchend', touchEnd, true);
		};
	});
</script>

<svelte:window onkeydown={keyboard} />

{#if currentFrame}
	<div class="recorded-field">
		<img src={currentFrame.url} alt="Recorded lattice at update {currentFrame.update}" /><span
			class="frame-label"
			>Recorded · update {currentFrame.update} · {modes.find(
				(entry) => entry.id === currentFrame.mode
			)?.name}</span
		>
	</div>
{/if}
{#if comparing && reference}
	<div class="reference-field" style:clip-path="inset(0 {100 - divider}% 0 0)">
		<img src={reference.url} alt="Reference lattice at update {reference.update}" />
	</div>
	<div class="compare-edge" style:left="{divider}%">
		<span>Reference {reference.update} │ Live {update}</span>
	</div>
{/if}
{#if (tool === 'probe' && selected >= 0) || tool === 'region'}
	<div
		class="selection"
		class:region-selection={tool === 'region'}
		style:left="{overlay.x}px"
		style:top="{overlay.y}px"
		style:width="{overlay.width}px"
		style:height="{overlay.height}px"
	>
		<span>{tool === 'probe' ? `Cell ${selected}` : 'Selected region'}</span>
	</div>
{/if}

<div
	class="observation-bar"
	aria-label="Observation controls"
	{@attach (element) => {
		observationBar = element;
	}}
>
	<div class="observation-heading">
		<span class="wordmark">Algocell</span>
		<select
			aria-label="Field view"
			value={mode}
			disabled={busy || externalBusy}
			onchange={(event) => void chooseMode(event.currentTarget.value as FieldMode)}
		>
			{#each modes as entry (entry.id)}<option value={entry.id}>{entry.name}</option>{/each}
		</select>
		<span class="transport-state"
			><i class:running={playing}></i>{playing ? 'Running' : 'Paused'}
			<span class="update-count">{update.toLocaleString()}</span></span
		>
		<div class="zoom-tools">
			<button aria-label="Zoom out" title="Zoom out" onclick={() => zoom(1.5)}>−</button><button
				aria-label="Zoom in"
				title="Zoom in"
				onclick={() => zoom(1 / 1.5)}>+</button
			><button onclick={fit} title="Fit lattice (F)">Fit</button>
		</div>
	</div>
	<nav class="observation-tabs" aria-label="Observation tools">
		{#each tools as entry (entry.id)}<button
				class:active={tool === entry.id}
				aria-pressed={tool === entry.id}
				onclick={() => chooseTool(entry.id)}
				>{entry.name}{#if entry.id === 'history' && recording}<span class="record-dot"
					></span>{/if}</button
			>{/each}
	</nav>
	<div class="mobile-tool-row">
		<select
			aria-label="Observation tool"
			value={tool}
			onchange={(event) => chooseTool(event.currentTarget.value as Tool)}
			>{#each tools as entry (entry.id)}<option value={entry.id}>{entry.name}</option
				>{/each}</select
		><span>{condition || `${settings.isa.toUpperCase()} · ${settings.length} bytes`}</span>
	</div>
	<div class="legend" aria-label="Field legend">
		{#if mode === 'confinement'}<div class="legend-scale">
				<span><i style="background:rgb(12% 12% 15%)"></i>Unknown</span><span
					><i style="background:rgb(25% 52% 72%)"></i>No exit</span
				><span><i style="background:rgb(90% 58% 28%)"></i>Outside A</span>
			</div>
		{:else if mode === 'zeros' || mode === 'diversity'}<div class="continuous-scale">
				<span>0%</span><i class:diversity-scale={mode === 'diversity'}></i><span>100%</span>
			</div>
		{:else if mode === 'marker'}<div class="legend-scale">
				<span
					><i style="background:rgb(98% 68% 35%)"></i>Byte {markerPosition} = {hex(
						markerValue
					)}</span
				><span><i style="background:rgb(12% 13% 15%)"></i>Other</span>
			</div>
		{:else if mode === 'identity'}<div class="fingerprint-scale">
				<i></i><span>One fingerprint per complete tape</span>
			</div>
		{/if}
		<p>{currentMode.description}</p>
	</div>
</div>

{#if !benchOpen && (tool !== 'navigate' || mode === 'marker')}
	<section
		class="observation-dock"
		{@attach (element) => {
			observationDock = element;
			return () => {
				observationDock = undefined;
			};
		}}
		class:region-tool={tool === 'region'}
		aria-label="{tool === 'navigate'
			? 'Marker'
			: tools.find((entry) => entry.id === tool)?.name} tool"
	>
		<header>
			<h2>
				{tool === 'navigate' ? 'Marker value' : tools.find((entry) => entry.id === tool)?.name}
			</h2>
			{#if tool === 'probe' || tool === 'region'}
				<button
					class="frame-selection"
					disabled={busy || externalBusy || (tool === 'probe' && selected < 0)}
					title={tool === 'region'
						? 'Zoom and pan the selected region into the visible field'
						: 'Pan the selected cell into the visible field without changing zoom'}
					onclick={frameSelection}>{tool === 'region' ? 'Frame region' : 'Center probe'}</button
				>
			{/if}
			<button
				class="close"
				aria-label="Close observation tool"
				onclick={() => {
					tool = 'navigate';
					comparing = false;
					frameIndex = -1;
					replaying = false;
					if (mode === 'marker') void chooseMode('bytes');
				}}>×</button
			>
		</header>
		{#if tool === 'probe'}
			<div class="input-row">
				<label for="probe-cell">Cell</label><input
					id="probe-cell"
					type="number"
					min="0"
					max={cells - 1}
					bind:value={cellInput}
					onchange={() => void selectCell(cellInput)}
				/><button onclick={() => void selectCell(cellInput)}>Probe</button><button
					title="Previous cell"
					aria-label="Previous cell"
					onclick={() => void selectCell(selected - 1)}>‹</button
				><button
					title="Next cell"
					aria-label="Next cell"
					onclick={() => void selectCell(selected + 1)}>›</button
				>
			</div>
			{#if tape}
				<div class="scope-line">
					{selected % settings.width}, {Math.floor(selected / settings.width)} · {settings.isa.toUpperCase()}
					· update {probeUpdate}
				</div>
				<div class="byte-tape">
					{#each Array.from(tape) as byte, index (index)}<span
							style:--byte-color={byteColor(byte)}
							title="Byte {index}: {hex(byte)}"><small>{index}</small>{hex(byte)}</span
						>{/each}
				</div>
				<div class="readout-row">
					<span>Zero fraction <strong>{percent(zeroFraction)}</strong></span><span
						>Length <strong>{tape.length}</strong></span
					>
				</div>
				<button
					class="primary wide"
					onclick={() => {
						onpause();
						oninspect(selected);
					}}>Open encounter bench</button
				>
				<p class="hint">
					Arrow keys move the probe. Readout refreshes twice a second while running.
				</p>
			{/if}
		{:else if tool === 'region'}
			<p class="hint">
				Drag two corners on the lattice, or enter coordinates. Measurements pause the population.
			</p>
			<div class="coordinate-grid">
				<span></span><span>x</span><span>y</span><span>From</span><input
					aria-label="Region from x"
					type="number"
					min="0"
					max={settings.width - 1}
					bind:value={x0}
					oninput={() => {
						region = null;
					}}
				/><input
					aria-label="Region from y"
					type="number"
					min="0"
					max={settings.height - 1}
					bind:value={y0}
					oninput={() => {
						region = null;
					}}
				/><span>To</span><input
					aria-label="Region to x"
					type="number"
					min="0"
					max={settings.width - 1}
					bind:value={x1}
					oninput={() => {
						region = null;
					}}
				/><input
					aria-label="Region to y"
					type="number"
					min="0"
					max={settings.height - 1}
					bind:value={y1}
					oninput={() => {
						region = null;
					}}
				/>
			</div>
			<button
				class="primary wide"
				disabled={busy || externalBusy}
				onclick={() => void sampleRegion()}>{busy ? 'Sampling…' : 'Measure region'}</button
			>
			{#if region}<div class="scope-line">Sample held at update {region.update}</div>
				<dl class="measurement-list">
					<div>
						<dt>Cells</dt>
						<dd>{region.cellCount.toLocaleString()}</dd>
					</div>
					<div>
						<dt>Exact unique tapes</dt>
						<dd>{region.uniqueTapes.toLocaleString()}</dd>
					</div>
					<div>
						<dt>Largest tape share</dt>
						<dd>{percent(region.topShare)}</dd>
					</div>
					<div>
						<dt>Zero fraction</dt>
						<dd>{percent(region.zeroFraction)}</dd>
					</div>
					{#if region.markerFraction !== null}<div>
							<dt>Marker value matches</dt>
							<dd>{percent(region.markerFraction)}</dd>
						</div>{/if}
				</dl>
				<div class="region-trend">
					<svg
						viewBox="0 0 300 86"
						role="img"
						aria-label="Region samples: zero fraction and largest tape share from 0 to 100 percent"
					>
						<line x1="28" y1="10" x2="288" y2="10" /><line x1="28" y1="66" x2="288" y2="66" />
						<text x="0" y="13">100%</text><text x="9" y="69">0%</text>
						{#each regionPaths as points, index (index)}<polyline
								{points}
								class:top-share={index === 1}
							/>{/each}
						{#each regionSeries as sample (sample.id)}
							{@const sampleX =
								28 +
								((sample.update - regionSeries[0].update) /
									Math.max(1, regionSeries.at(-1)!.update - regionSeries[0].update)) *
									260}
							<circle cx={sampleX} cy={66 - sample.zero * 56} r="2.5" />
							<circle class="top-share" cx={sampleX} cy={66 - sample.top * 56} r="2.5" />
						{/each}
						<text x="28" y="82">{regionSeries[0]?.update ?? region.update}</text><text
							x="288"
							y="82"
							text-anchor="end">{regionSeries.at(-1)?.update ?? region.update} updates</text
						>
					</svg>
					<div class="trend-key"><span>Zero fraction</span><span>Largest tape share</span></div>
				</div>
				<p class="hint held-note">
					Explicit samples of this region. Measure again to add a point.
				</p>{/if}
		{:else if tool === 'history'}
			<label class="check-row"
				><input
					type="checkbox"
					checked={recording}
					onchange={(event) => setRecording(event.currentTarget.checked)}
				/>
				Record recent frames <span>{frames.length}/24</span></label
			>
			<p class="hint">
				One image per second while running. Scrubbing pauses the world; images do not rewind it.
			</p>
			{#if frames.length}<div class="history-preview">
					<img src={(currentFrame ?? frames.at(-1)!).url} alt="History preview" />
				</div>
				<input
					class="scrubber"
					type="range"
					aria-label="History frame"
					min="0"
					max={frames.length - 1}
					value={frameIndex >= 0 ? frameIndex : frames.length - 1}
					oninput={(event) => scrub(Number(event.currentTarget.value))}
				/>
				<div class="readout-row">
					<span>{frames[0].update}</span><strong
						>{currentFrame ? `Recorded ${currentFrame.update}` : 'Live'}</strong
					><span>{frames.at(-1)!.update}</span>
				</div>
				<div class="button-row">
					<button
						disabled={frames.length < 2}
						onclick={() => {
							if (reducedMotion) {
								notice =
									'Automatic replay follows your reduced-motion preference. Use the scrubber.';
								return;
							}
							onpause();
							replaying = !replaying;
							if (frameIndex < 0) frameIndex = 0;
						}}>{replaying ? 'Stop replay' : 'Replay'}</button
					><button class="primary" onclick={returnLive}>Return live</button>
					{#if currentFrame}<button onclick={saveFrame}>Save frame</button>{/if}<button
						onclick={() => {
							frames = [];
							frameIndex = -1;
							replaying = false;
						}}>Clear</button
					>
				</div>{:else}<div class="empty-state">
					Record a short run to compare the field over time.
				</div>{/if}
		{:else if tool === 'compare'}
			<p class="hint">
				Pin the current view, then compare it with the live population. Keep the same camera to
				compare positions.
			</p>
			<button class="primary wide" onclick={pinReference}
				>{reference ? 'Replace reference' : 'Pin reference'}</button
			>
			{#if reference}<div class="scope-line">
					Reference: {modes.find((entry) => entry.id === reference!.mode)?.name} · update {reference.update}
				</div>
				<div class="scope-line">
					{reference.condition || 'Custom condition'} · {reference.isa.toUpperCase()} · seed {reference.seed}
				</div>
				{#if reference.camera !== currentCamera}<p class="hint camera-note">
						Camera differs from the reference. Align the view before comparing positions.
					</p>{/if}
				<label class="check-row"
					><input type="checkbox" bind:checked={comparing} /> Show comparison</label
				><label class="range-label" for="comparison-divider"
					>Reveal reference <span>{divider}%</span></label
				><input
					id="comparison-divider"
					aria-label="Reveal reference"
					type="range"
					min="5"
					max="95"
					bind:value={divider}
				/>
				<div class="comparison-key"><span>Reference</span><span>Live</span></div>
				<p class="hint">A visual reference, not a second simulation.</p>{/if}
		{:else if tool === 'record'}
			<p class="hint">Keep an image of this view or export a short sampled timelapse.</p>
			<button class="primary wide" onclick={snapshot}>Save PNG snapshot</button>
			<button
				class="wide"
				disabled={!mediaSupported || frames.length < 2 || exporting}
				onclick={() => void exportTimelapse()}
				>{exporting ? 'Encoding timelapse…' : `Export WebM · ${frames.length} frames`}</button
			>
			{#if !frames.length}<button
					class="text-button"
					onclick={() => {
						chooseTool('history');
						setRecording(true);
					}}>Start recording history</button
				>{/if}
			<p class="hint">
				{mediaSupported
					? 'Timelapse exports at 5 frames/s. History stores at most 24 small images.'
					: 'WebM is unavailable in this browser; PNG export is supported.'}
			</p>
			<div class="record-note">
				To save the actual bytes and condition, use <strong>Save checkpoint</strong> in the transport
				bar.
			</div>
		{/if}
		{#if mode === 'marker' || tool === 'probe'}
			<details class="marker-settings" open={mode === 'marker'}>
				<summary>Follow a byte value</summary>
				<div class="input-row">
					<label for="marker-position">Position</label><input
						id="marker-position"
						type="number"
						min="0"
						max={settings.length - 1}
						bind:value={markerPosition}
						onchange={applyMarker}
					/><label for="marker-value">Value</label><input
						id="marker-value"
						type="number"
						min="0"
						max="255"
						bind:value={markerValue}
						onchange={applyMarker}
					/>
				</div>
				<div class="button-row">
					<button
						onclick={() => {
							applyMarker();
							void chooseMode('marker');
						}}>Show matches</button
					><button disabled={selected < 0 || busy} onclick={() => void plantMarker()}
						>Plant in cell {selected >= 0 ? selected : '…'}</button
					>
				</div>
				<p class="hint">
					{hex(markerValue)} at position {markerPosition}. Matching values can occur independently.
					Test transfer in the encounter bench.
				</p>
			</details>
		{/if}
		{#if error}<p class="error" role="alert">{error}</p>{/if}
		{#if notice}<p class="notice" role="status">{notice}</p>{/if}
	</section>
{/if}

<style>
	.observation-bar,
	.observation-dock {
		position: fixed;
		z-index: 42;
		background: var(--bg-panel);
		border: 1px solid var(--border-muted);
		border-radius: 11px;
		box-shadow: 0 3px 20px #0006;
		backdrop-filter: blur(16px);
		color: var(--text-secondary);
		font-size: 12px;
	}
	.observation-bar {
		top: 12px;
		left: 12px;
		width: 475px;
		padding: 8px 10px 7px;
	}
	.observation-heading {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.wordmark {
		font-size: 14px;
		font-weight: 600;
		letter-spacing: -0.4px;
		color: var(--text-primary);
	}
	select,
	input,
	button {
		font: inherit;
		color: inherit;
	}
	select {
		appearance: auto;
		color-scheme: dark;
		background: #1b1b1b;
		background-image: none;
		border: 1px solid var(--border-muted);
		border-radius: 5px;
		min-height: 29px;
		padding: 3px 6px;
		box-shadow: none;
	}
	select:disabled {
		opacity: 0.55;
	}
	.observation-heading select {
		width: 148px;
	}
	button {
		background: var(--bg-subtle);
		border: 1px solid var(--border-muted);
		border-radius: 5px;
		cursor: pointer;
		padding: 6px 9px;
		line-height: 1.3;
		white-space: nowrap;
	}
	button:hover {
		background: var(--bg-hover);
		color: var(--text-primary);
	}
	button:disabled {
		opacity: 0.4;
		cursor: default;
	}
	.zoom-tools {
		display: flex;
		gap: 3px;
		margin-left: auto;
	}
	.zoom-tools button {
		padding: 3px 7px;
		min-width: 25px;
		min-height: 29px;
	}
	.transport-state {
		font: 10px monospace;
		color: var(--text-muted);
		display: flex;
		gap: 5px;
		align-items: center;
		white-space: nowrap;
	}
	.transport-state i {
		height: 5px;
		width: 5px;
		border-radius: 50%;
		background: var(--text-subtle);
	}
	.transport-state i.running {
		background: var(--accent-emerald);
	}
	.update-count {
		display: inline;
		font-variant-numeric: tabular-nums;
	}
	.observation-tabs {
		display: flex;
		margin: 8px -1px 6px;
		gap: 3px;
	}
	.observation-tabs button {
		border-color: transparent;
		background: transparent;
		flex: 1;
		padding: 5px 6px;
		font-size: 11px;
		color: var(--text-muted);
	}
	.observation-tabs button.active {
		background: #c8875a19;
		color: var(--accent-warm);
		border-color: #c8875a50;
	}
	.record-dot {
		display: inline-block;
		height: 5px;
		width: 5px;
		margin-left: 4px;
		background: var(--accent-rose);
		border-radius: 50%;
	}
	.legend {
		border-top: 1px solid var(--border-subtle);
		padding-top: 6px;
	}
	.legend p {
		font-size: 10px;
		line-height: 1.45;
		color: var(--text-subtle);
		margin: 0;
	}
	.legend-scale {
		display: flex;
		gap: 14px;
		margin-bottom: 4px;
		font-size: 10px;
	}
	.legend-scale span {
		display: flex;
		align-items: center;
		gap: 5px;
	}
	.legend-scale i {
		height: 7px;
		width: 7px;
		border-radius: 2px;
	}
	.continuous-scale {
		display: flex;
		gap: 9px;
		align-items: center;
		font: 10px monospace;
		margin-bottom: 5px;
	}
	.continuous-scale i {
		height: 5px;
		flex: 1;
		max-width: 180px;
		border-radius: 2px;
		background: linear-gradient(90deg in srgb, rgb(9% 16% 23%), rgb(97% 87% 66%));
	}
	.continuous-scale i.diversity-scale {
		background: linear-gradient(90deg in srgb, rgb(15% 31% 43%), rgb(91% 59% 35%));
	}
	.fingerprint-scale {
		display: flex;
		gap: 8px;
		align-items: center;
		font-size: 10px;
		margin-bottom: 4px;
	}
	.fingerprint-scale i {
		width: 40px;
		height: 5px;
		border-radius: 2px;
		background: linear-gradient(
			90deg in srgb,
			rgb(79.725% 17.975% 17.975%),
			rgb(79.725% 79.725% 17.975%),
			rgb(17.975% 79.725% 17.975%),
			rgb(17.975% 79.725% 79.725%),
			rgb(17.975% 17.975% 79.725%),
			rgb(79.725% 17.975% 79.725%),
			rgb(79.725% 17.975% 17.975%)
		);
	}
	.mobile-tool-row {
		display: none;
	}
	.observation-dock {
		right: 12px;
		bottom: 12px;
		width: 324px;
		padding: 12px;
		max-height: calc(100dvh - 86px);
		overflow: auto;
	}
	.observation-dock header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 10px;
	}
	.observation-dock h2 {
		font-size: 13px;
		margin: 0 auto 0 0;
		font-weight: 600;
	}
	.frame-selection {
		margin-right: 8px;
		padding: 4px 7px;
		font-size: 10px;
	}
	.close {
		border: 0;
		background: none;
		font-size: 20px;
		line-height: 1;
		padding: 0 3px;
		color: var(--text-subtle);
	}
	.input-row {
		display: flex;
		gap: 6px;
		align-items: center;
		margin: 8px 0;
	}
	.input-row label {
		font-size: 11px;
		color: var(--text-muted);
	}
	input[type='number'] {
		width: 65px;
		min-width: 0;
		padding: 5px 6px;
		border: 1px solid var(--border-muted);
		border-radius: 4px;
		background: #ffffff06;
		font: 12px monospace;
		box-shadow: none;
	}
	.input-row #probe-cell {
		width: 90px;
		flex: 1;
	}
	.input-row button {
		padding: 5px 9px;
	}
	.scope-line {
		font: 10px monospace;
		line-height: 1.5;
		color: var(--text-subtle);
		margin: 8px 0;
	}
	.byte-tape {
		display: grid;
		grid-template-columns: repeat(8, 1fr);
		gap: 3px;
		margin: 10px 0;
	}
	.byte-tape span {
		display: flex;
		flex-direction: column;
		align-items: center;
		padding: 5px 0;
		border-radius: 3px;
		font: 12px monospace;
		color: var(--text-primary);
		background: color-mix(in srgb, var(--byte-color) 44%, #141414);
		border-bottom: 2px solid var(--byte-color);
	}
	.byte-tape small {
		font-size: 8px;
		color: #ffffff78;
		margin-bottom: 3px;
	}
	.readout-row {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		font: 10px monospace;
		margin: 9px 0;
		color: var(--text-muted);
	}
	strong {
		font-weight: 500;
		color: var(--text-primary);
	}
	.wide {
		display: block;
		width: 100%;
		margin: 8px 0;
		min-height: 32px;
	}
	.primary {
		border-color: #c8875a66;
		background: #c8875a19;
		color: var(--accent-warm);
	}
	.hint {
		font-size: 11px;
		color: var(--text-muted);
		line-height: 1.55;
		margin: 8px 0;
	}
	.coordinate-grid {
		display: grid;
		grid-template-columns: 1fr 1fr 1fr;
		align-items: center;
		gap: 6px;
		margin: 10px 0;
		font-size: 11px;
		color: var(--text-muted);
	}
	.coordinate-grid input {
		width: 100%;
	}
	.measurement-list {
		margin: 12px 0;
	}
	.measurement-list div {
		display: flex;
		justify-content: space-between;
		gap: 10px;
		padding: 5px 0;
		border-bottom: 1px solid var(--border-subtle);
	}
	.measurement-list dt {
		color: var(--text-muted);
		font-size: 11px;
	}
	.measurement-list dd {
		margin: 0;
		font: 12px monospace;
		color: var(--text-primary);
	}
	.check-row {
		display: flex;
		gap: 7px;
		align-items: center;
		font-size: 12px;
		margin: 9px 0;
	}
	.check-row span {
		margin-left: auto;
		font: 10px monospace;
		color: var(--text-subtle);
	}
	input[type='checkbox'] {
		appearance: auto;
		accent-color: var(--accent);
		width: 13px;
		height: 13px;
		background: none;
		padding: 0;
	}
	.history-preview {
		height: 108px;
		overflow: hidden;
		border: 1px solid var(--border-muted);
		border-radius: 5px;
		margin: 10px 0;
	}
	.history-preview img {
		width: 100%;
		height: 100%;
		object-fit: cover;
		image-rendering: pixelated;
	}
	input[type='range'] {
		appearance: auto;
		accent-color: var(--accent);
		height: 20px;
		width: 100%;
		background: none;
		box-shadow: none;
		margin: 5px 0;
		padding: 0;
	}
	.button-row {
		display: flex;
		gap: 6px;
		flex-wrap: wrap;
		margin: 8px 0;
	}
	.button-row button {
		flex: 1;
	}
	.empty-state {
		padding: 22px 12px;
		text-align: center;
		font-size: 12px;
		line-height: 1.6;
		color: var(--text-subtle);
		background: var(--bg-subtle);
		border-radius: 6px;
		margin: 12px 0;
	}
	.range-label {
		display: flex;
		justify-content: space-between;
		font-size: 11px;
		margin-top: 12px;
	}
	.comparison-key {
		display: flex;
		justify-content: space-between;
		font: 10px monospace;
		color: var(--text-subtle);
	}
	.text-button {
		background: none;
		border: 0;
		padding-left: 0;
		color: var(--accent-warm);
	}
	.record-note {
		font-size: 11px;
		line-height: 1.6;
		border-top: 1px solid var(--border-muted);
		padding-top: 10px;
		color: var(--text-muted);
	}
	.marker-settings {
		border-top: 1px solid var(--border-muted);
		padding-top: 10px;
		margin-top: 12px;
	}
	.marker-settings summary {
		cursor: pointer;
		color: var(--text-muted);
		font-size: 11px;
	}
	.marker-settings .input-row input {
		flex: 1;
	}
	.error {
		color: var(--accent-rose);
		font-size: 11px;
		line-height: 1.5;
	}
	.notice {
		color: var(--accent-emerald);
		font-size: 11px;
		line-height: 1.5;
	}
	.region-trend {
		border-top: 1px solid var(--border-subtle);
		padding-top: 8px;
		margin: 10px 0;
	}
	.region-trend svg {
		display: block;
		width: 100%;
		height: 88px;
		overflow: visible;
	}
	.region-trend line {
		stroke: var(--border-muted);
		stroke-width: 1;
	}
	.region-trend text {
		fill: var(--text-subtle);
		font: 8px monospace;
	}
	.region-trend polyline {
		fill: none;
		stroke: #7dabb7;
		stroke-width: 2;
	}
	.region-trend circle {
		fill: #7dabb7;
	}
	.region-trend circle.top-share {
		fill: var(--accent-warm);
	}
	.region-trend polyline.top-share {
		stroke: var(--accent-warm);
	}
	.trend-key {
		display: flex;
		gap: 14px;
		font-size: 9px;
		color: #7dabb7;
	}
	.trend-key span:last-child {
		color: var(--accent-warm);
	}
	.camera-note {
		color: var(--accent-amber);
	}
	.selection {
		position: fixed;
		z-index: 15;
		pointer-events: none;
		border: 1px solid #f3cc92;
		box-shadow: 0 0 0 1px #0009;
		min-width: 3px;
		min-height: 3px;
	}
	.selection span {
		position: absolute;
		left: -1px;
		bottom: calc(100% + 3px);
		background: var(--bg-panel);
		color: #f3cc92;
		padding: 3px 5px;
		white-space: nowrap;
		border-radius: 3px;
		font: 10px monospace;
	}
	.region-selection {
		background: #c8875a0a;
	}
	.recorded-field,
	.reference-field {
		position: fixed;
		inset: 0;
		z-index: 10;
		pointer-events: none;
		background: var(--bg-base);
	}
	.recorded-field img,
	.reference-field img {
		width: 100%;
		height: 100%;
		object-fit: fill;
		image-rendering: pixelated;
	}
	.frame-label {
		position: absolute;
		top: 148px;
		left: 12px;
		background: var(--bg-panel);
		border: 1px solid var(--border-muted);
		color: var(--accent-warm);
		border-radius: 5px;
		padding: 6px 9px;
		font: 11px monospace;
	}
	.compare-edge {
		position: fixed;
		top: 0;
		bottom: 0;
		width: 1px;
		background: #e8e4e0aa;
		z-index: 14;
		pointer-events: none;
	}
	.compare-edge span {
		position: absolute;
		top: 160px;
		left: 8px;
		background: var(--bg-panel);
		padding: 5px 7px;
		border-radius: 3px;
		font: 10px monospace;
		white-space: nowrap;
		color: var(--text-secondary);
	}
	@media (max-width: 900px) {
		.observation-bar {
			width: 430px;
		}
		.observation-heading {
			gap: 7px;
		}
		.observation-heading select {
			width: 139px;
		}
		.transport-state {
			display: none;
		}
	}
	@media (max-width: 768px) {
		.observation-bar {
			top: 8px;
			left: 8px;
			right: 8px;
			width: auto;
			padding: 7px 9px;
			border-radius: 9px;
		}
		.observation-heading {
			gap: 8px;
		}
		.wordmark {
			font-size: 14px;
		}
		.observation-heading select {
			flex: 1;
			width: auto;
			min-height: 34px;
			font-size: 12px;
		}
		.zoom-tools button {
			min-width: 31px;
			min-height: 34px;
		}
		.observation-tabs {
			display: none;
		}
		.mobile-tool-row {
			display: flex;
			gap: 10px;
			align-items: center;
			margin: 6px 0;
		}
		.mobile-tool-row select {
			width: 111px;
			min-height: 34px;
		}
		.mobile-tool-row span {
			overflow: hidden;
			text-overflow: ellipsis;
			white-space: nowrap;
			font-size: 10px;
			color: var(--text-muted);
		}
		.legend p {
			font-size: 10px;
		}
		.legend-scale {
			font-size: 10px;
			gap: 10px;
		}
		.observation-dock {
			bottom: 88px;
			left: 8px;
			right: 8px;
			width: auto;
			max-height: calc(100dvh - 260px);
			padding: 11px;
			border-radius: 10px;
		}
		.observation-dock.region-tool {
			max-height: 49dvh;
		}
		.region-tool > .hint:first-of-type {
			display: none;
		}
		.region-tool .coordinate-grid {
			margin: 4px 0;
			gap: 4px 6px;
		}
		.region-tool .coordinate-grid input {
			min-height: 30px;
			padding: 3px 6px;
		}
		.region-tool .measurement-list {
			display: grid;
			grid-template-columns: 1fr 1fr;
			gap: 7px 14px;
			margin: 9px 0;
		}
		.region-tool .measurement-list div {
			flex-direction: column;
			gap: 4px;
			padding: 3px 0 6px;
		}
		.region-tool .measurement-list dd {
			font-size: 15px;
		}
		.region-tool .region-trend {
			margin: 8px 0;
			padding-top: 4px;
		}
		.region-tool .region-trend svg {
			height: 63px;
		}
		.region-tool .held-note {
			font-size: 10px;
			margin: 5px 0 0;
		}

		.observation-dock button:not(.close) {
			min-height: 36px;
		}
		.observation-dock input[type='number'] {
			font-size: 16px;
			min-height: 34px;
		}
		.observation-dock h2 {
			font-size: 14px;
		}
		.hint {
			font-size: 11px;
		}
		.history-preview {
			height: 90px;
		}
		.frame-label {
			top: 158px;
			left: 8px;
			right: 8px;
			font-size: 10px;
		}
		.compare-edge span {
			top: 175px;
			font-size: 9px;
			left: 4px;
		}
		.byte-tape span {
			padding: 4px 0;
		}
		.observation-dock input[type='range'] {
			min-height: 28px;
		}
	}
</style>

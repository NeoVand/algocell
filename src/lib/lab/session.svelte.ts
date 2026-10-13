import { LabEngine, createInitialSoup, populationMetrics } from './engine';
import {
	DEFAULT_SETTINGS,
	type LabSettings,
	type InitialCondition,
	type LabMetrics,
	type CultureResult,
	type MarkerResult,
	type LabSnapshot
} from './types';

export type ViewLayer = 'bytes' | 'identity' | 'fetch';

/** One session belongs to one mounted desk. GPU handles and large arrays are never proxied. */
export class LabSession {
	settings = $state<LabSettings>({ ...DEFAULT_SETTINGS, ablation: [] });
	initialCondition = $state.raw<InitialCondition>({ kind: 'random' });
	engine = $state.raw<LabEngine | null>(null);
	bytes = $state.raw<Uint8Array>(new Uint8Array(0));
	fetchStatus = $state.raw<Uint8Array>(new Uint8Array(0));
	history = $state.raw<LabMetrics[]>([]);
	metrics = $state.raw<LabMetrics | null>(null);
	selectedCell = $state(0);
	layer = $state<ViewLayer>('bytes');
	detailed = $state(false);
	status = $state<'loading' | 'ready' | 'running' | 'working' | 'unavailable'>('loading');
	error = $state('');
	notice = $state('');
	culture = $state.raw<CultureResult | null>(null);
	markerResult = $state.raw<MarkerResult | null>(null);
	presetId = $state('');
	private generation = 0;
	private running = false;
	private disposed = false;
	private deadline = 0;
	private timer: ReturnType<typeof setTimeout> | undefined;
	private busy = $state(false);
	private task: Promise<void> | null = null;

	get selectedTape(): Uint8Array {
		const start = this.selectedCell * this.settings.length;
		return this.bytes.slice(start, start + this.settings.length);
	}
	get available(): boolean {
		return this.engine !== null;
	}
	get locked(): boolean {
		return this.status === 'loading' || this.status === 'working' || this.busy;
	}
	get manifest() {
		return this.engine?.manifest ?? null;
	}

	async initialize(
		settings: LabSettings = this.settings,
		initial: InitialCondition = { kind: 'random' },
		presetId = '',
		snapshot?: LabSnapshot
	) {
		this.pause();
		const token = ++this.generation;
		this.status = 'loading';
		this.error = '';
		this.notice = '';
		const config = { ...settings, ablation: [...settings.ablation] };
		try {
			if (this.task) await this.task;
			if (this.disposed || token !== this.generation) return;
			if (!this.engine) {
				const preview = createInitialSoup(config, initial);
				this.settings = config;
				this.initialCondition = initial;
				this.presetId = presetId;
				this.bytes = preview;
				this.fetchStatus = new Uint8Array(config.width * config.height);
				this.metrics = populationMetrics(preview, config, initial);
				this.history = [this.metrics];
			}
			const next = await LabEngine.create(config, initial);
			if (this.disposed || token !== this.generation) {
				next.destroy();
				return;
			}
			try {
				if (snapshot) next.restore(snapshot);
			} catch (error) {
				next.destroy();
				throw error;
			}
			this.engine?.destroy();
			this.engine = next;
			this.settings = config;
			this.initialCondition = initial;
			this.presetId = presetId;
			this.selectedCell = 0;
			this.culture = null;
			this.markerResult = null;
			this.history = [];
			this.refresh();
			this.status = 'ready';
		} catch (error) {
			if (this.disposed || token !== this.generation) return;
			this.error = error instanceof Error ? error.message : String(error);
			this.status = this.engine ? 'ready' : 'unavailable';
		}
	}

	private refresh() {
		if (!this.engine) return;
		this.bytes = this.engine.soup.slice();
		this.fetchStatus = this.engine.fetchStatus.slice();
		this.metrics = { ...this.engine.metrics };
		this.history = [...this.history.slice(-179), this.metrics];
	}

	pause = () => {
		this.running = false;
		clearTimeout(this.timer);
		if (this.status === 'running') this.status = 'ready';
	};

	async step() {
		if (!this.engine || this.busy || this.locked) return;
		this.pause();
		this.task = this.performStep();
		await this.task;
		this.task = null;
	}

	private async performStep() {
		if (!this.engine || this.busy) return;
		this.busy = true;
		this.culture = null;
		this.markerResult = null;
		const token = this.generation;
		try {
			await this.engine.step();
			if (!this.disposed && token === this.generation) this.refresh();
		} catch (error) {
			if (!this.disposed && token === this.generation) {
				this.pause();
				this.error = error instanceof Error ? error.message : String(error);
			}
		} finally {
			this.busy = false;
		}
	}

	run = () => {
		if (!this.engine || this.locked || this.running) return;
		this.running = true;
		this.status = 'running';
		this.deadline = performance.now() + 55_000;
		this.notice = '';
		const loop = async () => {
			if (!this.running || this.disposed) return;
			if (document.hidden || performance.now() >= this.deadline) {
				this.pause();
				this.notice = document.hidden
					? 'Paused while the tab is hidden.'
					: 'Burst complete. Run again to continue.';
				return;
			}
			this.task = this.performStep();
			await this.task;
			this.task = null;
			if (this.running) this.timer = setTimeout(loop, 16);
		};
		void loop();
	};

	reset = () => this.initialize(this.settings, this.initialCondition, this.presetId);

	select = (cell: number) => {
		this.selectedCell = Math.max(
			0,
			Math.min(this.settings.width * this.settings.height - 1, Math.floor(cell))
		);
		this.culture = null;
		this.markerResult = null;
	};

	async assay() {
		if (!this.engine || this.busy || this.locked) return;
		this.pause();
		this.status = 'working';
		this.error = '';
		const token = this.generation;
		const cell = this.selectedCell;
		try {
			const result = await this.engine.assay(this.selectedTape, {
				trials: 16,
				seed: 731,
				generations: 2
			});
			if (!this.disposed && token === this.generation && cell === this.selectedCell)
				this.culture = result;
		} catch (error) {
			if (!this.disposed && token === this.generation)
				this.error = error instanceof Error ? error.message : String(error);
		} finally {
			if (!this.disposed && token === this.generation) this.status = 'ready';
		}
	}

	async marker(position: number, value: number) {
		if (!this.engine || this.busy || this.locked) return;
		this.pause();
		this.status = 'working';
		this.error = '';
		const token = this.generation;
		const cell = this.selectedCell;
		try {
			const result = await this.engine.marker(this.selectedTape, {
				position,
				value,
				trials: 16,
				seed: 731,
				generations: 2
			});
			if (!this.disposed && token === this.generation && cell === this.selectedCell)
				this.markerResult = result;
		} catch (error) {
			if (!this.disposed && token === this.generation)
				this.error = error instanceof Error ? error.message : String(error);
		} finally {
			if (!this.disposed && token === this.generation) this.status = 'ready';
		}
	}

	async restore(snapshot: LabSnapshot) {
		await this.initialize(snapshot.settings, snapshot.initialCondition, '', snapshot);
		if (!this.error) this.notice = 'Checkpoint restored. The world is paused.';
	}

	dispose() {
		this.disposed = true;
		this.generation++;
		this.pause();
		this.engine?.destroy();
		this.engine = null;
	}
}

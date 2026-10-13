import { chromium } from '@playwright/test';
import { createServer } from 'vite';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

// Reproducible user paths. A native listbox is clicked once then selected with
// the keyboard; text/range controls are clicked once then edited. Pointer travel
// is Euclidean, starting at the viewport center. Setup and screenshots are free.
const phase = process.env.UNIFIED_PHASE ?? 'before';
const round = process.env.UNIFIED_ROUND ?? '';
const workspace = process.cwd();
const output = resolve(workspace, `docs/review/unified/${phase}${round ? `/round-${round}` : ''}`);
const root = process.env.UNIFIED_ROOT ?? process.cwd();
const baseUrl = process.env.UNIFIED_URL;
const width = 1440;
const height = 900;
await mkdir(output, { recursive: true });
// SvelteKit resolves its app root from cwd, overriding Vite's root option.
process.chdir(root);
const server = baseUrl
	? null
	: await createServer({
			root,
			configFile: resolve(root, 'vite.config.ts'),
			server: { host: '127.0.0.1', port: 4185, strictPort: true, fs: { allow: [root, workspace] } }
		});
if (server) await server.listen();
const url = baseUrl ?? 'http://127.0.0.1:4185';
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const captures = [];
const errors = [];
const results = [];
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function pageReady(viewport = { width, height }, preset = null) {
	const page = await browser.newPage({ viewport });
	page.setDefaultTimeout(12000);
	page.on('pageerror', (error) => errors.push({ type: 'pageerror', message: error.message }));
	await page.goto(url);
	await page.waitForFunction(() => Boolean(window.__algocell?.engine));
	await pause(page);
	if (preset) {
		await page.getByTitle('Settings', { exact: true }).click();
		await page.getByLabel('Experiments', { exact: true }).selectOption(preset);
		await page.waitForFunction(() => !document.querySelector('#experiment-preset').disabled);
		await page.getByLabel('Close settings', { exact: true }).click();
	}
	await page.evaluate(() => document.fonts.ready);
	return page;
}
async function pause(page) {
	const button = page.getByTitle('Pause (Space)', { exact: true });
	if (await button.isVisible()) await button.click();
}
async function play(page) {
	const button = page.getByTitle('Play (Space)', { exact: true });
	if (await button.isVisible()) await button.click();
}
async function shot(page, name) {
	const path = `${output}/${name}.png`;
	await page.screenshot({ path, animations: 'disabled' });
	captures.push({
		name,
		path,
		viewport: page.viewportSize(),
		overflow: await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
	});
}
async function instrument(page) {
	await page.evaluate(() => {
		const stats = {
			pauses: 0,
			resets: 0,
			engineRecreations: 0,
			machineConfigurationCalls: 0,
			pipelineRebuilds: 0,
			populationBufferRecreations: 0,
			events: []
		};
		let engine = window.__algocell.engine;
		let playing = Boolean(document.querySelector('[title="Pause (Space)"]'));
		const hooked = new WeakSet();
		const attach = (engine) => {
			if (hooked.has(engine)) return;
			hooked.add(engine);
			for (const [method, key] of [
				['reset', 'resets'],
				['setMachine', 'machineConfigurationCalls'],
				['createPipelines', 'pipelineRebuilds'],
				['createBuffers', 'populationBufferRecreations']
			]) {
				const original = engine[method].bind(engine);
				engine[method] = (...args) => {
					stats[key]++;
					stats.events.push({ event: method, at: performance.now() });
					return original(...args);
				};
			}
		};
		attach(engine);
		const sample = () => {
			const nextEngine = window.__algocell?.engine;
			if (nextEngine && engine !== nextEngine) {
				stats.engineRecreations++;
				engine = nextEngine;
				attach(engine);
			}
			const nextPlaying = Boolean(document.querySelector('[title="Pause (Space)"]'));
			if (playing && !nextPlaying) {
				stats.pauses++;
				stats.events.push({ event: 'pause', at: performance.now() });
			}
			playing = nextPlaying;
		};
		const observer = new MutationObserver(sample);
		observer.observe(document.body, { subtree: true, attributes: true, childList: true });
		window.__taskMetrics = {
			stats,
			finish: () => {
				sample();
				observer.disconnect();
				return { ...stats, playingAtEnd: playing };
			}
		};
	});
}
class Path {
	constructor(page) {
		this.page = page;
		this.clicks = 0;
		this.travel = 0;
		this.pointer = { x: width / 2, y: height / 2 };
		this.surfaces = new Set();
		this.path = [];
	}
	button(name, exact = true) {
		return this.page.getByRole('button', { name, exact }).first();
	}
	async target(locator, surface, action) {
		await locator.scrollIntoViewIfNeeded();
		const box = await locator.boundingBox();
		if (!box) throw new Error(`No bounds: ${action}`);
		const point = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
		this.travel += Math.hypot(point.x - this.pointer.x, point.y - this.pointer.y);
		this.pointer = point;
		this.clicks++;
		this.surfaces.add(surface);
		this.path.push(action);
		await locator.click();
	}
	async click(locator, surface, action = '') {
		await this.target(
			locator,
			surface,
			action || (await locator.getAttribute('title')) || (await locator.innerText())
		);
	}
	async choose(locator, value, surface) {
		await this.target(locator, surface, `select ${value}`);
		await locator.selectOption(value);
		await this.page.waitForTimeout(80);
	}
	async fill(locator, value, surface) {
		await this.target(locator, surface, `set ${value}`);
		await locator.fill(String(value));
		await locator.press('Tab');
	}
	async range(locator, value, surface) {
		await this.target(locator, surface, `range ${value}`);
		await locator.fill(String(value));
		await locator.dispatchEvent('change');
	}
	async tool(name) {
		await this.click(
			this.page
				.getByRole('navigation', { name: 'Observation tools' })
				.getByRole('button', { name, exact: true }),
			'observation strip',
			name
		);
	}
	async settings() {
		await this.click(
			this.page.getByTitle('Settings', { exact: true }),
			'floating widget',
			'open settings'
		);
	}
	async closeSettings() {
		await this.click(
			this.page.getByLabel('Close settings', { exact: true }),
			'settings panel',
			'close settings'
		);
	}
	async probe() {
		await this.tool('Probe');
		await this.fill(this.page.locator('#probe-cell'), 0, 'observation dock');
		await this.click(
			this.page
				.getByRole('region', { name: 'Probe tool', exact: true })
				.getByRole('button', { name: 'Probe', exact: true }),
			'observation dock',
			'probe cell 0'
		);
		await this.page.getByRole('button', { name: 'Open encounter bench', exact: true }).waitFor();
	}
}
const tasks = [
	{
		id: '01-first-visit',
		persona: 'Visitor',
		frequency: 'Once on arrival; guide revisited occasionally',
		task: 'Read the guide, return to the lattice and identify the run control',
		run: async (p) => {
			await p.click(p.page.getByTitle('Help (H)', { exact: true }), 'floating widget', 'open help');
			await p.click(p.button('Guide'), 'help dialog', 'Guide');
			await shot(p.page, 'task-01-guide');
			await p.page.keyboard.press('Escape');
			p.path.push('Escape');
			await p.click(
				p.page.getByTitle('Pause (Space)', { exact: true }),
				'floating widget',
				'pause'
			);
			await p.click(
				p.page.getByTitle('Play (Space)', { exact: true }),
				'floating widget',
				'resume'
			);
		}
	},
	{
		id: '02-look-and-frame',
		persona: 'Visitor / teacher',
		frequency: '10–30 times per session',
		task: 'Compare identity, confinement and bytes, zoom in, then fit the field',
		run: async (p) => {
			for (const mode of ['identity', 'confinement', 'bytes']) {
				await p.choose(p.page.getByLabel('Field view', { exact: true }), mode, 'observation strip');
				await shot(p.page, `task-02-${mode}`);
			}
			await p.click(p.button('Zoom in'), 'observation strip');
			await p.click(p.button('Fit'), 'observation strip');
		}
	},
	{
		id: '03-change-core',
		persona: 'Researcher',
		frequency: '1–3 times per study',
		task: 'Switch to the corrected Z80 core at the same tape length',
		run: async (p) => {
			await p.settings();
			await p.choose(p.page.getByLabel('Processor', { exact: true }), 'z80-0.3', 'settings panel');
			await p.page.waitForFunction(() => !document.querySelector('#machine-select').disabled);
			await shot(p.page, 'task-03-core');
			await p.closeSettings();
		}
	},
	{
		id: '04-reproduce-preset',
		persona: 'Researcher',
		frequency: '1–6 times per session',
		task: 'Load the published short-prefix copier experiment',
		run: async (p) => {
			await p.settings();
			await p.choose(
				p.page.getByLabel('Experiments', { exact: true }),
				'displacement-32',
				'settings panel'
			);
			await p.page.waitForFunction(() => !document.querySelector('#experiment-preset').disabled);
			await shot(p.page, 'task-04-preset');
			await p.closeSettings();
		}
	},
	{
		id: '05-live-controls',
		persona: 'Teacher',
		frequency: '5–20 times per demonstration',
		task: 'Set speed to 2×, adjust mutation and zero chemistry while demonstrating',
		run: async (p) => {
			await p.click(p.page.getByTitle('Speed', { exact: true }), 'floating widget', 'open speed');
			await p.click(p.button('2x'), 'speed popover', '2×');
			await p.settings();
			await p.range(p.page.locator('#noise-input'), 5, 'settings panel');
			await p.range(p.page.locator('#lethal-probability'), 0.1, 'settings panel');
			await shot(p.page, 'task-05-live-settings');
			await p.closeSettings();
		}
	},
	{
		id: '06-follow-probe',
		persona: 'Power user',
		frequency: '20–100 times per session',
		task: 'Probe a cell, follow its neighbor, and center the selected tape',
		run: async (p) => {
			await p.probe();
			await p.click(p.button('Next cell'), 'observation dock');
			await p.click(p.button('Center probe'), 'observation dock');
			await shot(p.page, 'task-06-probe');
		}
	},
	{
		id: '07-encounter',
		persona: 'Power user',
		frequency: '5–20 times per session',
		task: 'Probe and open an encounter; run the self-pair comparison and inspect writes',
		preset: 'displacement-32',
		run: async (p) => {
			await p.probe();
			await p.click(p.button('Open encounter bench'), 'observation dock');
			await p.click(p.button('Self-pair'), 'encounter dialog');
			await p.click(p.button('Run comparison tests'), 'encounter dialog');
			await p.page.waitForFunction(() =>
				[...document.querySelectorAll('button')].some(
					(n) => n.textContent.trim() === 'Save evidence JSON' && !n.disabled
				)
			);
			await p.click(p.button('Write activity'), 'encounter dialog');
			await shot(p.page, 'task-07-encounter');
		}
	},
	{
		id: '08-region-measure',
		persona: 'Researcher / teacher',
		frequency: '5–15 times per session',
		task: 'Measure a 4-cell region and frame it on the canvas',
		run: async (p) => {
			await p.tool('Region');
			for (const [label, value] of [
				['Region from x', 0],
				['Region from y', 0],
				['Region to x', 1],
				['Region to y', 1]
			])
				await p.fill(p.page.getByLabel(label, { exact: true }), value, 'observation dock');
			await p.click(p.button('Measure region'), 'observation dock');
			await p.page.locator('dl.measurement-list').waitFor();
			await p.click(p.button('Frame region'), 'observation dock');
			await shot(p.page, 'task-08-region');
		}
	},
	{
		id: '09-follow-marker',
		persona: 'Power user',
		frequency: '2–10 times per session',
		task: 'Plant a neutral marker in a selected tape and view matching descendants',
		preset: 'displacement-32',
		run: async (p) => {
			await p.probe();
			await p.choose(
				p.page.getByLabel('Field view', { exact: true }),
				'marker',
				'observation strip'
			);
			await p.fill(p.page.locator('#marker-position'), 8, 'observation dock');
			await p.fill(p.page.locator('#marker-value'), 165, 'observation dock');
			await p.click(p.button('Show matches'), 'observation dock');
			await p.click(p.button('Plant in cell 0'), 'observation dock');
			await shot(p.page, 'task-09-marker');
		}
	},
	{
		id: '10-history',
		persona: 'Power user / teacher',
		frequency: '3–10 times per session',
		task: 'Record recent frames, scrub to an earlier sample, then return live',
		run: async (p) => {
			await p.tool('History');
			await p.click(p.page.getByLabel('Record recent frames'), 'observation dock');
			await p.page.waitForFunction(
				() => Number(document.querySelector('[aria-label="History frame"]')?.max) >= 1
			);
			const range = p.page.getByLabel('History frame', { exact: true });
			await p.click(range, 'observation dock');
			await range.press('Home');
			await shot(p.page, 'task-10-history');
			await p.click(p.button('Return live'), 'observation dock');
			await p.click(p.page.getByLabel('Record recent frames'), 'observation dock');
		}
	},
	{
		id: '11-compare',
		persona: 'Researcher / teacher',
		frequency: '3–10 times per session',
		task: 'Pin a reference, reveal the comparison, and adjust the split',
		run: async (p) => {
			await p.tool('Compare');
			await p.click(p.button('Pin reference'), 'observation dock');
			await p.click(p.page.getByLabel('Show comparison'), 'observation dock');
			await p.range(p.page.getByLabel('Reveal reference', { exact: true }), 65, 'observation dock');
			await shot(p.page, 'task-11-compare');
		}
	},
	{
		id: '12-export',
		persona: 'Researcher / teacher',
		frequency: '1–5 times per session',
		task: 'Save a PNG snapshot and a resumable world checkpoint',
		run: async (p) => {
			await p.tool('Record');
			const png = p.page.waitForEvent('download');
			await p.click(p.button('Save PNG snapshot'), 'observation dock');
			await png;
			const checkpoint = p.page.waitForEvent('download');
			await p.click(
				p.page.getByTitle('Save checkpoint (Cmd+S)', { exact: true }),
				'floating widget'
			);
			await checkpoint;
			await shot(p.page, 'task-12-export');
		}
	}
];

async function atlas() {
	for (const viewport of [
		{ width: 1280, height: 800 },
		{ width: 1440, height: 900 },
		{ width: 1920, height: 1080 }
	]) {
		const page = await pageReady(viewport);
		try {
			await shot(page, `layout-${viewport.width}`);
		} finally {
			await page.close();
			await wait(500);
		}
	}
	const page = await pageReady();
	const capture = (name) => shot(page, `state-${name}`);
	try {
		await page.getByTitle('Speed', { exact: true }).click();
		await capture('speed');
		await page.keyboard.press('Escape');
		await page.getByTitle('Collapse toolbar', { exact: true }).click();
		await capture('widget-collapsed');
		await page.reload();
		await page.waitForFunction(() => Boolean(window.__algocell?.engine));
		await pause(page);
		for (const [name, title] of [
			['opcodes', 'Opcode frequency'],
			['species', 'Species —'],
			['diversity', 'Diversity —']
		]) {
			await page.getByTitle(new RegExp(`^${title}`)).click();
			await capture(`chart-${name}`);
		}
		await page.getByTitle('Toggle chart', { exact: true }).click();
		await capture('chart-collapsed');
		await page.getByTitle('Toggle chart', { exact: true }).click();
		await page.getByTitle('Settings', { exact: true }).click();
		await capture('settings');
		await page.getByTitle('Color adjustments', { exact: true }).click();
		await capture('color-adjustments');
		await page.getByTitle('Color adjustments', { exact: true }).click();
		await page.getByText('World contract L16 / P32', { exact: true }).click();
		await capture('world-contract');
		await page.getByLabel('Close settings', { exact: true }).click();
		for (const mode of ['bytes', 'identity', 'confinement', 'zeros', 'diversity', 'marker']) {
			await page.getByLabel('Field view', { exact: true }).selectOption(mode);
			await capture(`mode-${mode}`);
		}
		for (const name of ['Navigate', 'Probe', 'Region', 'History', 'Compare', 'Record']) {
			await page
				.getByRole('navigation', { name: 'Observation tools' })
				.getByRole('button', { name, exact: true })
				.click();
			await capture(`tool-${name.toLowerCase()}-empty`);
		}
		await page.getByTitle('Help (H)', { exact: true }).click();
		for (const name of [
			'About',
			'Guide',
			'Overview',
			'Visuals',
			'Z80',
			'Parameters',
			'Analysis',
			'Keys'
		]) {
			await page.getByRole('button', { name, exact: true }).click();
			await capture(name.toLowerCase());
		}
		await page.keyboard.press('Escape');
	} catch (error) {
		errors.push({ case: 'atlas', message: error.message });
		await capture('atlas-error');
	} finally {
		await page.close();
		await wait(500);
	}
	const bench = await pageReady({ width, height }, 'displacement-32');
	try {
		const p = new Path(bench);
		await p.probe();
		await p.click(p.button('Open encounter bench'), 'observation dock');
		await shot(bench, 'state-bench-initial');
		for (const assay of ['Culture', 'Self-pair', 'Partner sensitivity', 'Marker transfer']) {
			await p.button(assay).click();
			await p.button(assay === 'Culture' ? 'Run culture test' : 'Run comparison tests').click();
			await bench.waitForFunction(() =>
				[...document.querySelectorAll('button')].some(
					(n) => n.textContent.trim() === 'Save evidence JSON' && !n.disabled
				)
			);
			await shot(bench, `state-bench-${assay.toLowerCase().replaceAll(' ', '-')}`);
		}
		for (const lens of ['Fetched bytes', 'Instruction starts', 'Write activity']) {
			await p.button(lens).click();
			await shot(bench, `state-bench-${lens.toLowerCase().replaceAll(' ', '-')}`);
		}
		await bench.getByText('Disassembly', { exact: true }).click();
		await bench.getByText('Disassembly', { exact: true }).scrollIntoViewIfNeeded();
		await shot(bench, 'state-bench-disassembly');
	} catch (error) {
		errors.push({ case: 'bench-atlas', message: error.message });
		await shot(bench, 'state-bench-atlas-error');
	} finally {
		await bench.close();
		await wait(500);
	}
}
try {
	if (process.env.UNIFIED_ATLAS !== 'only') {
		for (const task of tasks.filter(
			(t) => !process.env.UNIFIED_TASKS || process.env.UNIFIED_TASKS.split(',').includes(t.id)
		)) {
			const page = await pageReady({ width, height }, task.preset);
			try {
				await play(page);
				await instrument(page);
				const p = new Path(page);
				await task.run(p);
				const metrics = await page.evaluate(() => window.__taskMetrics.finish());
				results.push({
					id: task.id,
					persona: task.persona,
					frequency: task.frequency,
					task: task.task,
					clicks: p.clicks,
					pointerTravelPx: Math.round(p.travel),
					surfaces: [...p.surfaces],
					surfaceCount: p.surfaces.size,
					...metrics,
					path: p.path
				});
				console.log(task.id, JSON.stringify(results.at(-1)));
			} catch (error) {
				console.error(task.id, error.message);
				errors.push({ case: task.id, message: error.message });
				await shot(page, `error-${task.id}`);
			} finally {
				await pause(page);
				await page.close();
				await wait(1000);
			}
		}
	}
	if (process.env.UNIFIED_ATLAS !== 'none') await atlas();
} finally {
	await browser.close();
	await server?.close();
	const previous = JSON.parse(await readFile(`${output}/metrics.json`, 'utf8').catch(() => '{}'));
	await writeFile(
		`${output}/metrics.json`,
		JSON.stringify(
			{
				phase,
				round,
				source: root,
				viewport: { width, height },
				method:
					'Pointer starts at viewport center. Every clicked control and number/range edit counts once; a native select uses one click plus keyboard option choice. Setup excluded. Surfaces are independently positioned interaction containers. Pauses include requested and incidental transitions. Resets count engine.reset calls; setMachine counts configuration requests; createPipelines and createBuffers count real pipeline and population-buffer rebuilds separately from engine object replacements.',
				tasks: [
					...(previous.tasks ?? []).filter((old) => !results.some((next) => next.id === old.id)),
					...results
				],
				captures: [
					...(previous.captures ?? []).filter(
						(old) => !captures.some((next) => next.name === old.name)
					),
					...captures
				],
				errors
			},
			null,
			2
		) + '\n'
	);
}
if (errors.length) {
	console.error(errors);
	process.exitCode = 1;
}

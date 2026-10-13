import { chromium, expect } from '@playwright/test';
import { createServer } from 'vite';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

// Reproducible user paths. A native listbox is clicked once then selected with
// the keyboard; text/range controls are clicked once then edited. Pointer travel
// is Euclidean, starting at the viewport center. Setup and screenshots are free.
const phase = process.env.UNIFIED_PHASE ?? 'before';
const unified = phase === 'after';
const round = process.env.UNIFIED_ROUND ?? '';
const workspace = process.cwd();
const output = resolve(workspace, `docs/review/unified/${phase}${round ? `/round-${round}` : ''}`);
const reportName =
	!unified && process.env.UNIFIED_ATLAS?.startsWith('extras')
		? 'additional-states.json'
		: 'metrics.json';
const root = process.env.UNIFIED_ROOT ?? process.cwd();
const baseUrl = process.env.UNIFIED_URL;
const sourceSnapshot = JSON.parse(
	await readFile(resolve(root, '.unified-source.json'), 'utf8').catch(() => 'null')
);
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
		if (unified) {
			const p = new Path(page);
			await p.tab('Experiments');
			await p.choose(
				page.getByRole('combobox', { name: 'Experiments', exact: true }),
				preset,
				'sidebar'
			);
			await p.applyWorld();
			await p.tab('Measure');
		} else {
			await page.getByTitle('Settings', { exact: true }).click();
			await page.getByLabel('Experiments', { exact: true }).selectOption(preset);
			await page.waitForFunction(() => !document.querySelector('#experiment-preset').disabled);
			await page.getByLabel('Close settings', { exact: true }).click();
		}
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
		path: path.replace(`${workspace}/`, ''),
		viewport: page.viewportSize(),
		overflow: await page.evaluate(() => document.documentElement.scrollWidth > innerWidth),
		forbiddenUiWord: await page.evaluate(() => /\bpaper\b/i.test(document.body.innerText))
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
				if (typeof engine[method] !== 'function') continue;
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
		if ((await locator.evaluate((node) => node.tagName)) === 'SELECT') {
			await locator.selectOption(value);
		} else {
			const choices = this.page.getByRole('option');
			await choices.first().waitFor();
			const values = await choices.evaluateAll((nodes) =>
				nodes.map((node) => node.getAttribute('data-value'))
			);
			const index = values.indexOf(value);
			if (index < 0) throw new Error(`Option ${value} missing from ${values.join(', ')}`);
			await this.page.keyboard.press('Home');
			for (let next = 0; next < index; next++) await this.page.keyboard.press('ArrowDown');
			await this.page.keyboard.press('Enter');
			await expect(locator).toHaveAttribute('data-value', value, { timeout: 12000 });
		}
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
	async tab(name) {
		const tab = this.page.getByRole('tab', { name, exact: true });
		if (
			(await tab.getAttribute('aria-selected')) === 'true' &&
			!(await this.page.getByLabel('Expand sidebar', { exact: true }).isVisible())
		)
			return;
		await this.click(tab, 'sidebar', `open ${name}`);
	}
	async applyWorld() {
		const apply = this.page.getByRole('button', {
			name: 'Start new world with these settings',
			exact: true
		});
		if (!(await apply.isVisible())) await this.tab('World');
		await this.click(apply, 'sidebar', 'explicitly start staged world');
		await this.page.waitForFunction(() =>
			[...document.querySelectorAll('button')]
				.filter((button) => button.textContent.trim() === 'Start new world with these settings')
				.every((button) => !button.disabled)
		);
	}
	async section(name) {
		const control = this.page
			.getByRole('navigation', { name: 'Measure sections', exact: true })
			.getByRole('button', { name, exact: true });
		if (!(await control.isVisible())) await this.tab('Measure');
		await this.click(control, 'sidebar', `open ${name}`);
	}
	async tool(name) {
		if (unified) {
			if (['History', 'Record'].includes(name))
				return this.section(name === 'Record' ? 'Export' : name);
			const control = this.page
				.getByRole('navigation', { name: 'Lattice tools', exact: true })
				.getByRole('button', { name: name === 'Navigate' ? 'Select / pan' : name, exact: true });
			await this.click(control, 'tool rail', name);
			return;
		}
		await this.click(
			this.page
				.getByRole('navigation', { name: 'Observation tools' })
				.getByRole('button', { name, exact: true }),
			'observation strip',
			name
		);
	}
	async settings() {
		if (unified) return this.tab('World');
		await this.click(
			this.page.getByTitle('Settings', { exact: true }),
			'floating widget',
			'open settings'
		);
	}
	async closeSettings() {
		if (unified) return;
		await this.click(
			this.page.getByLabel('Close settings', { exact: true }),
			'settings panel',
			'close settings'
		);
	}
	async probe() {
		await this.tool('Probe');
		await this.fill(this.page.locator('#probe-cell'), 0, unified ? 'sidebar' : 'observation dock');
		await this.click(
			this.page
				.getByRole('region', { name: 'Probe tool', exact: true })
				.getByRole('button', { name: 'Probe', exact: true }),
			unified ? 'sidebar' : 'observation dock',
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
		task: 'Load the published full-tape copier experiment',
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
		task: 'Plant a marker in a selected tape and view matching cells',
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

// These paths reach the same outcomes as the baseline while using the final
// information architecture. Staying in the sidebar is free only when no action
// is required; every actual tab, listbox, field, and button activation is counted.
const afterTasks = {
	'01-first-visit': async (p) => {
		await p.tab('Guide');
		await shot(p.page, 'task-01-guide');
		await p.click(p.page.getByTitle('Pause (Space)', { exact: true }), 'status bar', 'pause');
		await p.click(p.page.getByTitle('Play (Space)', { exact: true }), 'status bar', 'resume');
	},
	'02-look-and-frame': async (p) => {
		for (const mode of ['identity', 'confinement', 'bytes']) {
			await p.choose(
				p.page.getByRole('combobox', { name: 'Field view', exact: true }),
				mode,
				'header'
			);
			await shot(p.page, `task-02-${mode}`);
		}
		await p.click(p.button('Zoom in'), 'header');
		await p.click(p.page.getByTitle('Fit lattice (F)', { exact: true }), 'header');
	},
	'03-change-core': async (p) => {
		await p.tab('World');
		await p.choose(
			p.page.getByRole('combobox', { name: 'Processor', exact: true }),
			'z80-0.3',
			'sidebar'
		);
		await shot(p.page, 'task-03-core');
	},
	'04-reproduce-preset': async (p) => {
		await p.tab('Experiments');
		await p.choose(
			p.page.getByRole('combobox', { name: 'Experiments', exact: true }),
			'displacement-32',
			'sidebar'
		);
		await shot(p.page, 'task-04-preset-staged');
		await p.applyWorld();
		await shot(p.page, 'task-04-preset');
	},
	'05-live-controls': async (p) => {
		await p.choose(p.page.getByRole('combobox', { name: 'Speed', exact: true }), '2', 'status bar');
		await p.tab('World');
		await p.range(p.page.locator('#noise-input'), 5, 'sidebar');
		await p.range(p.page.locator('#lethal-probability'), 0.1, 'sidebar');
		await shot(p.page, 'task-05-live-settings');
	},
	'06-follow-probe': async (p) => {
		await p.probe();
		await p.click(p.button('Next cell'), 'sidebar');
		await p.click(p.button('Center probe'), 'sidebar');
		await shot(p.page, 'task-06-probe');
	},
	'07-encounter': async (p) => {
		await p.probe();
		await p.click(p.button('Open encounter bench'), 'sidebar');
		await p.click(p.button('Self-pair'), 'sidebar');
		await p.click(p.button('Run comparison tests'), 'sidebar');
		await p.page.waitForFunction(() =>
			[...document.querySelectorAll('button')].some(
				(n) => n.textContent.trim() === 'Save evidence JSON' && !n.disabled
			)
		);
		await p.click(p.button('Write activity'), 'sidebar');
		await shot(p.page, 'task-07-encounter');
	},
	'08-region-measure': async (p) => {
		await p.tool('Region');
		for (const [label, value] of [
			['Region from x', 0],
			['Region from y', 0],
			['Region to x', 1],
			['Region to y', 1]
		])
			await p.fill(p.page.getByLabel(label, { exact: true }), value, 'sidebar');
		await p.click(p.button('Measure region'), 'sidebar');
		await p.page.locator('dl.measurement-list').waitFor();
		await p.click(p.button('Frame region'), 'sidebar');
		await shot(p.page, 'task-08-region');
	},
	'09-follow-marker': async (p) => {
		await p.probe();
		await p.tool('Marker');
		await p.fill(p.page.locator('#marker-position'), 8, 'sidebar');
		await p.fill(p.page.locator('#marker-value'), 165, 'sidebar');
		await p.click(p.button('Show matches'), 'sidebar');
		await p.click(p.button('Plant in cell 0'), 'sidebar');
		await shot(p.page, 'task-09-marker');
	},
	'10-history': async (p) => {
		await p.tool('History');
		await p.click(p.page.getByLabel('Record recent frames'), 'sidebar');
		await p.page.waitForFunction(
			() => Number(document.querySelector('[aria-label="History frame"]')?.max) >= 1
		);
		const range = p.page.getByLabel('History frame', { exact: true });
		await p.click(range, 'sidebar');
		await range.press('Home');
		await shot(p.page, 'task-10-history');
		await p.click(p.button('Return live'), 'sidebar');
		await p.click(p.page.getByLabel('Record recent frames'), 'sidebar');
	},
	'11-compare': async (p) => {
		await p.tool('Compare');
		await p.click(p.button('Pin reference'), 'sidebar');
		await p.click(p.page.getByLabel('Show comparison'), 'sidebar');
		await p.range(p.page.getByLabel('Reveal reference', { exact: true }), 65, 'sidebar');
		await shot(p.page, 'task-11-compare');
	},
	'12-export': async (p) => {
		// Round 2 consolidates all exports behind the permanent footer action.
		// Round 1 retains its original route because that footer only held checkpoints.
		if (Number(round) >= 2) await p.click(p.button('Export & checkpoints'), 'sidebar');
		else await p.tool('Record');
		const png = p.page.waitForEvent('download');
		await p.click(p.button('Save PNG snapshot'), 'sidebar');
		await png;
		if (!(await p.button('Save checkpoint').isVisible()))
			await p.click(p.button('Export & checkpoints'), 'sidebar');
		const checkpoint = p.page.waitForEvent('download');
		await p.click(p.button('Save checkpoint'), 'sidebar');
		await checkpoint;
		await shot(p.page, 'task-12-export');
	}
};

async function atlasAfter() {
	const viewports = [
		{ width: 1280, height: 800 },
		{ width: 1920, height: 1080 },
		{ width: 1440, height: 900 }
	];
	for (const viewport of viewports) {
		const page = await pageReady(viewport, 'displacement-32');
		const p = new Path(page);
		const state = async (name, prepare) => {
			const id = `state-${name}-${viewport.width}`;
			try {
				await prepare();
				await shot(page, id);
			} catch (error) {
				errors.push({ case: id, message: error.message });
				await shot(page, `error-${id}`);
			}
		};
		try {
			await state('main', async () => p.tab('Measure'));
			for (const tab of ['World', 'Measure', 'Experiments', 'Guide', 'About'])
				await state(`tab-${tab.toLowerCase()}`, async () => p.tab(tab));
			await state('probe', async () => {
				await p.probe();
				await p.button('Center probe').click();
			});
			await state('region', async () => {
				await p.tool('Region');
				for (const [label, value] of [
					['Region from x', 0],
					['Region from y', 0],
					['Region to x', 1],
					['Region to y', 1]
				])
					await page.getByLabel(label, { exact: true }).fill(String(value));
				await p.button('Measure region').click();
				await page.locator('dl.measurement-list').waitFor();
			});
			await state('history-empty', async () => p.tool('History'));
			await state('encounter', async () => {
				await p.probe();
				await p.button('Open encounter bench').click();
			});
			await page.getByLabel('Close tape inspector', { exact: true }).click();
			await state('species-inspector', async () => {
				await p.section('Charts');
				await page.getByTitle(/^Species —/).click();
				await play(page);
				await page.locator('.freq-cell[title="Click to inspect"]').first().waitFor();
				await pause(page);
				await page.locator('.freq-cell[title="Click to inspect"]').first().click();
			});
			await page
				.getByRole('button', { name: /^Close.*inspector$/ })
				.first()
				.click();
			await state('sidebar-collapsed', async () =>
				page.getByLabel('Collapse sidebar', { exact: true }).click()
			);
			await page.getByLabel('Expand sidebar', { exact: true }).click();
			if (viewport.width !== 1440) continue;
			await page.getByTitle('Fit lattice (F)', { exact: true }).click();
			for (const mode of ['bytes', 'identity', 'confinement', 'zeros', 'diversity', 'marker'])
				await state(`mode-${mode}`, async () =>
					p.choose(page.getByRole('combobox', { name: 'Field view', exact: true }), mode, 'header')
				);
			await state('field-listbox', async () =>
				page.getByRole('combobox', { name: 'Field view', exact: true }).click()
			);
			await page.keyboard.press('Escape');
			await state('speed-listbox', async () =>
				page.getByRole('combobox', { name: 'Speed', exact: true }).click()
			);
			await page.keyboard.press('Escape');
			await state('world-core-listbox', async () => {
				await p.tab('World');
				await page.getByRole('combobox', { name: 'Processor', exact: true }).click();
			});
			await page.keyboard.press('Escape');
			for (const core of ['z80-0.2', 'z80-0.3', 'z80-nmos', 'm6502', 'i8086'])
				await state(`world-core-${core}`, async () =>
					p.choose(page.getByRole('combobox', { name: 'Processor', exact: true }), core, 'sidebar')
				);
			if (Number(round) >= 3)
				await state('core-change-notice', async () => {
					const processor = page.getByRole('combobox', { name: 'Processor', exact: true });
					await p.choose(processor, 'z80-0.2', 'sidebar');
					await p.choose(processor, 'm6502', 'sidebar');
					await page.locator('.core-notice[role="status"]').waitFor();
					await processor.scrollIntoViewIfNeeded();
				});
			await state('world-staged', async () => {
				const tapeLength = page.getByLabel('Tape length', { exact: true });
				await tapeLength.fill((await tapeLength.inputValue()) === '32' ? '16' : '32');
				await page.getByLabel('Tape length', { exact: true }).press('Tab');
			});
			await state('world-discarded', async () => p.button('Discard staged changes').click());
			await state('appearance', async () => {
				await p.choose(
					page.getByRole('combobox', { name: 'Field view', exact: true }),
					'bytes',
					'header'
				);
				await page.getByRole('button', { name: 'Look', exact: true }).click();
			});
			for (const palette of ['Rainbow', 'Ocean', 'Thermal'])
				await state(`palette-${palette.toLowerCase()}`, async () =>
					p.button(new RegExp(`^${palette}$`, 'i'), false).click()
				);
			await state('layer-grid', async () => page.getByLabel('Grid lines', { exact: true }).check());
			await state('layer-average', async () => p.button('Average colour').click());
			await state('layer-individual-bytes', async () => p.button('Individual bytes').click());
			await state('checkpoints', async () => p.button('Export & checkpoints').click());
			await state('checkpoint-import-staged', async () => {
				const saved = page.waitForEvent('download');
				await p.button('Save checkpoint').click();
				const checkpoint = await saved;
				const chooser = page.waitForEvent('filechooser');
				await p.button('Import checkpoint').click();
				await (await chooser).setFiles(await checkpoint.path());
				await p.button('Restore checkpoint').waitFor();
			});
			await state('checkpoint-restored', async () => {
				await p.button('Restore checkpoint').click();
				await p.button('Restore checkpoint').waitFor({ state: 'hidden' });
			});
			await state('experiment-listbox', async () => {
				await p.tab('Experiments');
				await page.getByRole('combobox', { name: 'Experiments', exact: true }).click();
			});
			await page.keyboard.press('Escape');
			const experiments = page.getByRole('combobox', { name: 'Experiments', exact: true });
			await experiments.click();
			const presetOptions = await page.getByRole('option').evaluateAll((nodes) =>
				nodes
					.map((node) => ({
						value: node.getAttribute('data-value'),
						label: node.textContent.trim()
					}))
					.filter((option) => option.value)
			);
			await page.keyboard.press('Escape');
			for (const preset of presetOptions)
				await state(`experiment-${preset.value}`, async () =>
					p.choose(experiments, preset.value, 'sidebar')
				);
			if (Number(round) >= 3) {
				await state('world-preset-summary', async () => {
					await p.choose(experiments, 'displacement-32', 'sidebar');
					await p.tab('World');
					await p.button('Condition').click();
					const summary = page.getByRole('group', { name: 'Next world summary', exact: true });
					await summary.waitFor();
					await summary.scrollIntoViewIfNeeded();
				});
				await p.tab('Experiments');
			}
			for (const name of ['Classic', 'Hex organic', 'No block copy', 'No-copy'])
				await state(`saved-preset-${name.toLowerCase().replaceAll(' ', '-')}`, async () =>
					page.getByTitle(`Stage preset: ${name}`, { exact: true }).click()
				);
			await state('personal-preset-save-form', async () => {
				await page.getByTitle('Save current settings as a preset', { exact: true }).click();
				await page.getByLabel('Preset name', { exact: true }).fill('Unified capture');
			});
			await state('personal-preset-saved', async () => p.button('Save').click());
			await state('personal-preset-overwrite', async () => {
				await page.getByTitle('Save current settings as a preset', { exact: true }).click();
				await page.getByLabel('Preset name', { exact: true }).fill('Unified capture');
				await p.button('Save').click();
			});
			await state('personal-preset-overwritten', async () => p.button('Overwrite').click());
			await state('personal-preset-deleted', async () =>
				page.getByTitle('Delete selected preset', { exact: true }).click()
			);
			await p.tab('World');
			await p.button('Condition').click();
			await p.button('Discard staged changes').click();
			await p.choose(
				page.getByRole('combobox', { name: 'Processor', exact: true }),
				'z80-0.3',
				'sidebar'
			);
			await state('ablation-preview', async () => {
				await page.getByText('Budgets & instruction ablation', { exact: true }).click();
				await page.getByLabel('Suppress instructions', { exact: true }).fill('LDIR');
			});
			await state('ablation-active', async () =>
				page.getByLabel('Suppress instructions', { exact: true }).press('Enter')
			);
			await p.tab('Measure');
			await p.section('Charts');
			await play(page);
			await page.waitForTimeout(1200);
			await pause(page);
			for (const [name, title] of [
				['opcodes', 'Opcode frequency'],
				['species', 'Species —'],
				['diversity', 'Diversity —']
			])
				await state(`chart-${name}`, async () => page.getByTitle(new RegExp(`^${title}`)).click());
			await page.getByTitle(/^Species —/).click();
			await play(page);
			await page.locator('.freq-cell[title="Click to inspect"]').first().waitFor();
			await pause(page);
			await state('species-inspector-detail', async () =>
				page.locator('.freq-cell[title="Click to inspect"]').first().click()
			);
			await state('species-disassembly', async () => {
				await page.getByText('Disassembly', { exact: true }).click();
				await page.getByText('Disassembly', { exact: true }).scrollIntoViewIfNeeded();
			});
			await page
				.getByRole('button', { name: /^Close.*inspector$/ })
				.first()
				.click();
			await p.tab('Guide');
			const guide = page.getByRole('combobox', { name: 'Guide section', exact: true });
			await guide.click();
			const sections = await page
				.getByRole('option')
				.evaluateAll((nodes) =>
					nodes.map((n) => ({ value: n.getAttribute('data-value'), label: n.textContent.trim() }))
				);
			await page.keyboard.press('Escape');
			for (const section of sections)
				await state(
					`guide-${section.value.toLowerCase().replaceAll(/[^a-z0-9]+/g, '-')}`,
					async () => p.choose(guide, section.value, 'sidebar')
				);
			if (Number(round) >= 3)
				await state('guide-preserved-after-about', async () => {
					const visual = sections.find((section) => /Visual explanations/i.test(section.label));
					if (!visual) throw new Error('Missing visual explanations Guide section');
					await p.choose(guide, visual.value, 'sidebar');
					await p.tab('About');
					await p.tab('Guide');
					await expect(guide).toHaveAttribute('data-value', visual.value);
				});
			await state('shortcuts', async () => page.keyboard.press('?'));
			await page.keyboard.press('Escape');
			await state('marker-empty', async () => p.tool('Marker'));
			await state('compare', async () => {
				await p.tool('Compare');
				await p.button('Pin reference').click();
				await page.getByLabel('Show comparison').check();
			});
			await state('history-scrub', async () => {
				await p.tool('History');
				await page.getByLabel('Record recent frames').check();
				await play(page);
				await page.waitForFunction(
					() => Number(document.querySelector('[aria-label="History frame"]')?.max) >= 2
				);
				await pause(page);
				await page.getByLabel('Record recent frames').uncheck();
				await page.getByLabel('History frame', { exact: true }).focus();
				await page.keyboard.press('Home');
			});
			await state('history-replay', async () => {
				await p.button('Replay').click();
				await page.waitForTimeout(150);
			});
			await p.button('Return live').click();
			await state('frame-export', async () => p.tool('Record'));
			await p.probe();
			await p.button('Open encounter bench').click();
			for (const assay of ['Culture', 'Self-pair', 'Partner sensitivity', 'Marker transfer'])
				await state(`bench-${assay.toLowerCase().replaceAll(' ', '-')}`, async () => {
					await p.button(assay).click();
					await p.button(assay === 'Culture' ? 'Run culture test' : 'Run comparison tests').click();
					await page.waitForFunction(() =>
						[...document.querySelectorAll('button')].some(
							(n) => n.textContent.trim() === 'Save evidence JSON' && !n.disabled
						)
					);
				});
			for (const lens of ['Fetched bytes', 'Instruction starts', 'Write activity'])
				await state(`bench-${lens.toLowerCase().replaceAll(' ', '-')}`, async () =>
					p.button(lens).click()
				);
			await state('bench-disassembly', async () => {
				await page.getByText('Disassembly', { exact: true }).click();
				await page.getByText('Disassembly', { exact: true }).scrollIntoViewIfNeeded();
			});
		} finally {
			await pause(page);
			await page.close();
			await wait(1000);
		}
	}
}

async function atlasMobile() {
	const page = await pageReady({ width: 390, height: 844 });
	const p = new Path(page);
	const state = async (name, prepare) => {
		const id = `state-mobile-${name}-390`;
		try {
			await prepare();
			await shot(page, id);
		} catch (error) {
			errors.push({ case: id, message: error.message });
			await shot(page, `error-${id}`);
		}
	};
	try {
		await state('collapsed', async () => {});
		await state('world', async () => p.tab('World'));
		await state('probe', async () => p.probe());
		await state('region', async () => {
			await p.tool('Region');
			await p.button('Measure region').click();
			await page.locator('dl.measurement-list').waitFor();
		});
		await state('encounter', async () => {
			await p.probe();
			await p.button('Open encounter bench').click();
		});
	} finally {
		await pause(page);
		await page.close();
		await wait(1000);
	}
}

async function atlasUnavailable() {
	const page = await browser.newPage({ viewport: { width, height } });
	page.setDefaultTimeout(12000);
	try {
		await page.addInitScript(() =>
			Object.defineProperty(navigator, 'gpu', { configurable: true, get: () => undefined })
		);
		await page.goto(url);
		await page.waitForFunction(
			() =>
				/WebGPU/.test(document.body.innerText) &&
				/not supported|not available|unavailable|requires/i.test(document.body.innerText)
		);
		await shot(page, 'state-webgpu-unavailable-1440');
		if (unified) await page.getByRole('button', { name: 'Open guide', exact: true }).click();
		else {
			await page.getByTitle('Help (H)', { exact: true }).click();
			await page.getByRole('button', { name: 'Guide', exact: true }).click();
		}
		await shot(page, 'state-webgpu-guide-1440');
	} catch (error) {
		errors.push({ case: 'webgpu-unavailable', message: error.message });
	} finally {
		await page.close();
	}
}

async function atlasBeforeSpecies() {
	const page = await pageReady();
	try {
		await page.getByTitle(/^Species —/).click();
		await play(page);
		await page.locator('.freq-cell[title="Click to inspect"]').first().waitFor();
		await pause(page);
		await page.locator('.freq-cell[title="Click to inspect"]').first().click();
		await page
			.locator(
				'.species-modal .species-modal-grid .tip-cell, .species-modal .hex-byte-grid .hex-byte-cell'
			)
			.first()
			.hover();
		await shot(page, 'extra-species-byte-hover');
	} catch (error) {
		errors.push({ case: 'extra-species-byte-hover', message: error.message });
	} finally {
		await pause(page);
		await page.close();
	}
}

async function atlasBeforeExtra() {
	const page = await pageReady();
	const state = async (name, prepare) => {
		try {
			await prepare();
			await shot(page, `extra-${name}`);
		} catch (error) {
			errors.push({ case: `extra-${name}`, message: error.message });
			await shot(page, `error-extra-${name}`);
		}
	};
	try {
		await page.getByTitle('Settings', { exact: true }).click();
		const core = page.getByLabel('Processor', { exact: true });
		for (const value of ['z80-0.2', 'z80-0.3', 'z80-nmos', 'm6502', 'i8086'])
			await state(`core-${value}`, async () => {
				await core.selectOption(value);
				await page.waitForFunction(() => !document.querySelector('#machine-select').disabled);
			});
		const presets = await page
			.getByLabel('Experiments', { exact: true })
			.locator('option')
			.evaluateAll((nodes) =>
				nodes
					.map((node) => ({ value: node.value, label: node.textContent.trim() }))
					.filter((option) => option.value)
			);
		for (const preset of presets)
			await state(`experiment-${preset.value}`, async () => {
				await page.getByLabel('Experiments', { exact: true }).selectOption(preset.value);
				await page.waitForFunction(() => !document.querySelector('#experiment-preset').disabled);
			});
		for (const name of ['Classic', 'Hex organic', 'No block copy', 'No-copy'])
			await state(`preset-${name.toLowerCase().replaceAll(' ', '-')}`, async () =>
				page.getByTitle(`Built-in preset: ${name}`, { exact: true }).click()
			);
		await state('preset-save-form', async () => {
			await page.getByTitle('Save current settings as a preset', { exact: true }).click();
			await page.locator('.preset-name-input').fill('Unified capture');
		});
		await state('preset-saved', async () =>
			page.getByTitle('Save preset', { exact: true }).click()
		);
		await state('preset-overwrite', async () => {
			await page.getByTitle('Save current settings as a preset', { exact: true }).click();
			await page.locator('.preset-name-input').fill('Unified capture');
			await page.getByTitle('Save preset', { exact: true }).click();
		});
		await state('preset-overwritten', async () =>
			page.getByRole('button', { name: 'Overwrite', exact: true }).click()
		);
		await state('preset-deleted', async () =>
			page.getByLabel('Delete selected preset', { exact: true }).click()
		);
		await state('suppression-preview', async () => page.locator('.suppress-input').fill('LDIR'));
		await state('suppression-active', async () => page.locator('.suppress-input').press('Enter'));
		const tips = page.locator('.param-info');
		for (let index = 0; index < (await tips.count()); index++)
			await state(`parameter-help-${index + 1}`, async () => {
				await tips.nth(index).scrollIntoViewIfNeeded();
				await tips.nth(index).hover();
			});
		await page.getByLabel('Close settings', { exact: true }).click();
		await state('canvas-hover', async () => {
			await page.mouse.move(width / 2, height / 2);
			await page.waitForTimeout(600);
		});
		await state('species', async () => {
			await page.getByTitle(/^Species —/).click();
			await play(page);
			await page.locator('.freq-cell[title="Click to inspect"]').first().waitFor();
			await pause(page);
		});
		await state('species-inspector', async () =>
			page.locator('.freq-cell[title="Click to inspect"]').first().click()
		);
		await state('species-disassembly', async () => {
			await page.getByText('Disassembly', { exact: true }).click();
			await page.getByText('Disassembly', { exact: true }).scrollIntoViewIfNeeded();
		});
		await state('species-byte-hover', async () => {
			const bytes = page.locator(
				'.species-modal .species-modal-grid .tip-cell, .species-modal .hex-byte-grid .hex-byte-cell'
			);
			await bytes.first().hover();
		});
		await page.keyboard.press('Escape');
		const p = new Path(page);
		await p.tool('History');
		await state('history-recording', async () => {
			await page.getByLabel('Record recent frames').check();
			await play(page);
			await page.waitForFunction(
				() => Number(document.querySelector('[aria-label="History frame"]')?.max) >= 2
			);
			await pause(page);
			await page.getByLabel('Record recent frames').uncheck();
		});
		await state('history-scrub', async () => {
			await page.getByLabel('History frame', { exact: true }).focus();
			await page.keyboard.press('Home');
		});
		await state('history-replay', async () => {
			await p.button('Replay').click();
			await page.waitForTimeout(150);
		});
		await state('history-live', async () => p.button('Return live').click());
		await state('comparison-split', async () => {
			await p.tool('Compare');
			await p.button('Pin reference').click();
			await page.getByLabel('Show comparison').check();
			await page.getByLabel('Reveal reference', { exact: true }).fill('50');
		});
	} finally {
		await pause(page);
		await page.close();
		await wait(1000);
	}
}

async function atlasBefore() {
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
	if (process.env.UNIFIED_ATLAS !== 'only' && !process.env.UNIFIED_ATLAS?.startsWith('extras')) {
		for (const task of tasks.filter(
			(t) => !process.env.UNIFIED_TASKS || process.env.UNIFIED_TASKS.split(',').includes(t.id)
		)) {
			let page;
			try {
				page = await pageReady({ width, height }, task.preset);
				await play(page);
				await instrument(page);
				const p = new Path(page);
				await (unified ? afterTasks[task.id] : task.run)(p);
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
					intentionalPauses: task.id === '01-first-visit' ? Math.min(1, metrics.pauses) : 0,
					incidentalPauses:
						metrics.pauses - (task.id === '01-first-visit' ? Math.min(1, metrics.pauses) : 0),
					intentionalResets: task.id === '04-reproduce-preset' ? metrics.resets : 0,
					incidentalResets: task.id === '04-reproduce-preset' ? 0 : metrics.resets,
					path: p.path
				});
				console.log(task.id, JSON.stringify(results.at(-1)));
			} catch (error) {
				console.error(task.id, error.message);
				errors.push({ case: task.id, message: error.message });
				if (page) await shot(page, `error-${task.id}`);
			} finally {
				if (page) {
					await pause(page);
					await page.close();
				}
				await wait(1000);
			}
		}
	}
	if (process.env.UNIFIED_ATLAS !== 'none') {
		if (unified) {
			if (Number(round) >= 4) await atlasMobile();
			await atlasAfter();
			if (Number(round) === 3) await atlasMobile();
			await atlasUnavailable();
		} else {
			if (!process.env.UNIFIED_ATLAS?.startsWith('extras')) await atlasBefore();
			if (process.env.UNIFIED_ATLAS === 'extras-species') await atlasBeforeSpecies();
			else if (process.env.UNIFIED_ATLAS === 'extras-unavailable') await atlasUnavailable();
			else await atlasBeforeExtra();
		}
	}
} finally {
	await browser.close();
	await server?.close();
	const previous = JSON.parse(await readFile(`${output}/${reportName}`, 'utf8').catch(() => '{}'));
	await writeFile(
		`${output}/${reportName}`,
		JSON.stringify(
			{
				phase,
				round,
				source: root,
				sourceSnapshot,
				viewport: { width, height },
				method:
					'Pointer starts at viewport center. Every clicked control and number/range edit counts once; a native select uses one click plus keyboard option choice. Setup excluded. Surfaces are independently positioned interaction containers. Pauses include requested and incidental transitions. Resets count engine.reset calls; setMachine counts configuration requests; createPipelines and createBuffers count real pipeline and population-buffer rebuilds separately from engine object replacements.',
				tasks: [
					...(previous.tasks ?? []).filter((old) => !results.some((next) => next.id === old.id)),
					...results
				],
				captures: [
					...(previous.captures ?? []).filter(
						(old) =>
							!results.some((result) => old.name === `error-${result.id}`) &&
							!captures.some(
								(next) =>
									next.name === old.name ||
									(old.name.startsWith('error-') && next.name === old.name.slice(6))
							)
					),
					...captures
				],
				alternativeTaskPaths: previous.alternativeTaskPaths ?? [],
				errors
			},
			null,
			2
		) + '\n'
	);
}
if (unified) {
	const before = JSON.parse(
		await readFile(resolve(workspace, 'docs/review/unified/before/metrics.json'), 'utf8')
	);
	const after = JSON.parse(await readFile(`${output}/${reportName}`, 'utf8'));
	if (after.tasks.length === 12 && !errors.length) {
		const lines = [
			'<!-- UNIFIED_RESULTS_START -->',
			`## Before / after task metrics — review round ${round || 'current'}`,
			'',
			'Each entry is baseline → unified. Pointer travel is in pixels at 1440 × 900. Pauses and resets include the requested transport pause and requested preset restart, with incidental disruptions listed separately below.',
			'',
			'| Task | Clicks | Pointer px | Surfaces | Pauses | Resets |',
			'| --- | ---: | ---: | ---: | ---: | ---: |'
		];
		const fields = ['clicks', 'pointerTravelPx', 'surfaceCount', 'pauses', 'resets'];
		const totals = { before: {}, after: {} };
		for (const field of fields) {
			totals.before[field] = 0;
			totals.after[field] = 0;
		}
		for (const task of tasks) {
			const b = before.tasks.find((item) => item.id === task.id);
			const a = after.tasks.find((item) => item.id === task.id);
			lines.push(
				`| ${task.id} | ${fields.map((field) => `${b[field]} → ${a[field]}`).join(' | ')} |`
			);
			for (const field of fields) {
				totals.before[field] += b[field];
				totals.after[field] += a[field];
			}
		}
		lines.push(
			`| Total | ${fields.map((field) => `${totals.before[field]} → ${totals.after[field]}`).join(' | ')} |`
		);
		const incidentalPausesBefore = before.tasks.reduce(
			(sum, item) => sum + item.pauses - (item.id === '01-first-visit' ? 1 : 0),
			0
		);
		const incidentalPausesAfter = after.tasks.reduce((sum, item) => sum + item.incidentalPauses, 0);
		const incidentalResetsBefore = before.tasks
			.filter((item) => item.id !== '04-reproduce-preset')
			.reduce((sum, item) => sum + item.resets, 0);
		const incidentalResetsAfter = after.tasks.reduce((sum, item) => sum + item.incidentalResets, 0);
		lines.push(
			'',
			`Incidental pauses: ${incidentalPausesBefore} → ${incidentalPausesAfter}. Incidental resets: ${incidentalResetsBefore} → ${incidentalResetsAfter}. World confirmation remains a counted, intentional action in task 04.`,
			'',
			'<!-- UNIFIED_RESULTS_END -->'
		);
		const path = resolve(workspace, 'docs/review/unified/TASKS.md');
		const current = await readFile(path, 'utf8');
		const report = lines.join('\n');
		const next = current.includes('<!-- UNIFIED_RESULTS_START -->')
			? current.replace(
					/<!-- UNIFIED_RESULTS_START -->[\s\S]*?<!-- UNIFIED_RESULTS_END -->/,
					report
				)
			: `${current}\n${report}\n`;
		await writeFile(path, next);
	}
}
if (errors.length) {
	console.error(errors);
	process.exitCode = 1;
}

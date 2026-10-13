import { chromium } from '@playwright/test';
import { createServer } from 'vite';
import { mkdir, readFile, writeFile } from 'node:fs/promises';

const directory = 'docs/review/evolution';
await mkdir(directory, { recursive: true });
const server = process.env.EVOLUTION_URL
	? null
	: await createServer({ server: { host: '127.0.0.1', port: 0 } });
if (server) await server.listen();
const url = process.env.EVOLUTION_URL ?? server.resolvedUrls.local[0];
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const errors = [];
const captures = [];
const priorReport =
	process.env.CAPTURE_CASES || process.env.CAPTURE_WIDTHS
		? JSON.parse(
				await readFile(`${directory}/capture-report.json`, 'utf8').catch(
					() => '{"captures":[],"errors":[]}'
				)
			)
		: { captures: [], errors: [] };

async function tool(page, name) {
	const selector = page.getByLabel('Observation tool', { exact: true });
	if (await selector.isVisible()) await selector.selectOption(name.toLowerCase());
	else await page.getByRole('button', { name, exact: true }).first().click();
}

async function ready(page, preset = 'stage-g-l16') {
	await page.goto(url);
	await page.waitForFunction(() => Boolean(window.__algocell?.engine));
	await page.getByTitle('Pause (Space)', { exact: true }).click();
	await page.getByTitle('Settings', { exact: true }).click();
	await page.getByLabel('Experiments', { exact: true }).selectOption(preset);
	await page.getByLabel('Experiments', { exact: true }).waitFor({ state: 'visible' });
	await page.waitForFunction(
		() => !document.querySelector('#experiment-preset').matches(':disabled')
	);
	await page.getByLabel('Close settings', { exact: true }).click();
	await page.evaluate(() => document.fonts.ready);
}

// Same real seeded population and short fixed update horizon for field modes.
async function advance(page, count = 100) {
	await page.evaluate(async (count) => {
		const engine = window.__algocell.engine;
		for (let i = 0; i < count; i += 4) {
			for (let j = 0; j < Math.min(4, count - i); j++) engine.simulateStep();
			await engine.waitForIdle();
		}
		engine.render(document.querySelector('canvas'));
		await engine.waitForIdle();
	}, count);
	// The live readout samples the update clock twice a second.
	await page.waitForTimeout(600);
}

async function centerProbe(page) {
	const center = await page.evaluate(() => {
		const { width, height } = window.__algocell.engine.config;
		return Math.floor(height / 2) * width + Math.floor(width / 2);
	});
	await tool(page, 'Probe');
	await page.locator('#probe-cell').fill(String(center));
	await page
		.getByRole('region', { name: 'Probe tool', exact: true })
		.getByRole('button', { name: 'Probe', exact: true })
		.click();
	await page.getByRole('button', { name: 'Open encounter bench', exact: true }).waitFor();
	await page.getByRole('button', { name: 'Center probe', exact: true }).click();
	return center;
}

async function marker(page) {
	const center = await centerProbe(page);
	const value = await page.evaluate(
		async (center) => (await window.__algocell.engine.readCell(center))[8] ^ 255,
		center
	);
	await page.getByLabel('Field view', { exact: true }).selectOption('marker');
	await page.locator('#marker-position').fill('8');
	await page.locator('#marker-value').fill(String(value));
	await page.getByRole('button', { name: 'Show matches', exact: true }).click();
	await page.getByRole('button', { name: `Plant in cell ${center}`, exact: true }).click();
}

async function history(page) {
	await tool(page, 'History');
	await page.getByLabel('Record recent frames').check();
	await page.getByTitle('Play (Space)', { exact: true }).click();
	await page.waitForFunction(() => {
		const slider = document.querySelector('[aria-label="History frame"]');
		return slider && Number(slider.max) >= 2;
	});
	await page.getByTitle('Pause (Space)', { exact: true }).click();
	await page.getByLabel('Record recent frames').uncheck();
}

async function bench(page, assay = 'Culture', lens = null) {
	await centerProbe(page);
	await page.getByRole('button', { name: 'Open encounter bench', exact: true }).click();
	const dialog = page.getByRole('dialog', { name: 'Tape inspector', exact: true });
	await dialog.getByRole('button', { name: assay, exact: true }).click();
	await dialog
		.getByRole('button', {
			name: assay === 'Culture' ? 'Run culture test' : 'Run comparison tests',
			exact: true
		})
		.click();
	await dialog.getByRole('button', { name: 'Save evidence JSON', exact: true }).waitFor();
	await page.waitForFunction(() => {
		const button = [...document.querySelectorAll('button')].find(
			(node) => node.textContent.trim() === 'Save evidence JSON'
		);
		return button && !button.disabled;
	});
	if (!lens)
		await dialog
			.locator('.test-tabs')
			.evaluate((element) => element.scrollIntoView({ block: 'start' }));
	if (lens) {
		await dialog.getByRole('button', { name: lens, exact: true }).click();
		await dialog
			.locator('.encounter-header')
			.evaluate((element) => element.scrollIntoView({ block: 'start' }));
	}
}

const scenarios = [
	...['bytes', 'identity', 'confinement', 'zeros', 'diversity', 'marker'].map((mode) => ({
		name: `mode-${mode}`,
		prepare: async (page) => {
			await page.getByLabel('Field view', { exact: true }).selectOption(mode);
			await advance(page, 1000);
			if (mode === 'marker') await marker(page);
		}
	})),
	{
		name: 'tool-fit',
		prepare: async (page) => {
			await advance(page, 1000);
			await page.getByRole('button', { name: 'Fit', exact: true }).click();
		}
	},
	{
		name: 'tool-navigate',
		prepare: async (page) => {
			await advance(page);
			await tool(page, 'Navigate');
			await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
			await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
			const canvas = await page.locator('canvas').boundingBox();
			await page.mouse.move(canvas.width * 0.55, canvas.height * 0.5);
			await page.mouse.down();
			await page.mouse.move(canvas.width * 0.65, canvas.height * 0.56, { steps: 8 });
			await page.mouse.up();
		}
	},
	{
		name: 'tool-probe',
		preset: 'displacement-32',
		prepare: async (page) => {
			await centerProbe(page);
		}
	},
	{
		name: 'tool-marker',
		preset: 'displacement-32',
		prepare: marker
	},
	{
		name: 'tool-region',
		prepare: async (page, viewportWidth) => {
			await advance(page);
			await tool(page, 'Region');
			const { width, height } = await page.evaluate(() => window.__algocell.engine.config);
			for (const [name, value] of [
				['Region from x', Math.floor(width * (viewportWidth < 500 ? 0.46 : 0.4))],
				['Region from y', Math.floor(height * (viewportWidth < 500 ? 0.46 : 0.35))],
				['Region to x', Math.floor(width * (viewportWidth < 500 ? 0.54 : 0.6))],
				['Region to y', Math.floor(height * (viewportWidth < 500 ? 0.54 : 0.65))]
			])
				await page.getByLabel(name, { exact: true }).fill(String(value));
			await page.getByRole('button', { name: 'Measure region', exact: true }).click();
			await page.locator('dl.measurement-list').waitFor();
			for (let sample = 0; sample < 2; sample++) {
				await advance(page);
				await page.getByRole('button', { name: 'Measure region', exact: true }).click();
			}
			await page.getByRole('button', { name: 'Frame region', exact: true }).click();
		}
	},
	...['scrub', 'replay'].map((action) => ({
		name: `tool-history-${action}`,
		prepare: async (page) => {
			await history(page);
			await page.getByLabel('History frame', { exact: true }).focus();
			await page.keyboard.press('Home');
			if (action === 'replay') {
				await page.getByRole('button', { name: 'Replay', exact: true }).click();
				await page.waitForTimeout(300);
			}
		}
	})),
	{
		name: 'tool-compare',
		prepare: async (page) => {
			await advance(page);
			await tool(page, 'Compare');
			await page.getByRole('button', { name: 'Pin reference', exact: true }).click();
			await advance(page);
			await page.getByLabel('Show comparison', { exact: true }).check();
		}
	},
	{
		name: 'tool-snapshot',
		prepare: async (page, width) => {
			await advance(page);
			await tool(page, 'Record');
			const event = page.waitForEvent('download');
			await page.getByRole('button', { name: 'Save PNG snapshot', exact: true }).click();
			await (await event).saveAs(`${directory}/export-${width}-snapshot.png`);
		}
	},
	{
		name: 'tool-timelapse',
		prepare: async (page, width) => {
			await history(page);
			await tool(page, 'Record');
			const event = page.waitForEvent('download');
			await page.getByRole('button', { name: /Export WebM/ }).click();
			await (await event).saveAs(`${directory}/export-${width}-timelapse.webm`);
		}
	},
	...[
		['culture', 'Culture', null],
		['self-pair', 'Self-pair', null],
		['partner-sensitivity', 'Partner sensitivity', null],
		['marker-transfer', 'Marker transfer', null],
		['fetch', 'Self-pair', 'Fetched bytes'],
		['starts', 'Self-pair', 'Instruction starts'],
		['writes', 'Self-pair', 'Write activity']
	].map(([name, assay, lens]) => ({
		name: `bench-${name}`,
		preset: 'displacement-32',
		prepare: async (page) => {
			await bench(page, assay, lens);
		}
	})),
	{
		name: 'bench-disassembly',
		preset: 'displacement-32',
		prepare: async (page) => {
			await centerProbe(page);
			await page.getByRole('button', { name: 'Open encounter bench', exact: true }).click();
			await page.getByText('Disassembly', { exact: true }).click();
			await page
				.getByText('Disassembly', { exact: true })
				.evaluate((element) => element.scrollIntoView({ block: 'start' }));
		}
	},
	{
		name: 'experiments',
		prepare: async (page) => {
			await page.getByTitle('Settings', { exact: true }).click();
		}
	},
	...['About', 'Guide'].map((name) => ({
		name: name.toLowerCase(),
		prepare: async (page) => {
			await page.getByTitle('Help (H)', { exact: true }).click();
			await page.getByRole('button', { name, exact: true }).click();
		}
	}))
];

const captureScenarios = [
	...scenarios,
	{ ...scenarios.find((scenario) => scenario.name === 'tool-probe'), name: 'tool-center-probe' },
	{ ...scenarios.find((scenario) => scenario.name === 'tool-region'), name: 'tool-frame-region' }
];

try {
	for (const [width, height] of [
		[1440, 900],
		[390, 844]
	].filter(
		([width]) =>
			!process.env.CAPTURE_WIDTHS || process.env.CAPTURE_WIDTHS.split(',').includes(String(width))
	)) {
		for (const scenario of captureScenarios.filter(
			(scenario) =>
				!process.env.CAPTURE_CASES || process.env.CAPTURE_CASES.split(',').includes(scenario.name)
		)) {
			const page = await browser.newPage({ viewport: { width, height } });
			page.setDefaultTimeout(20_000);
			page.on('pageerror', (error) =>
				errors.push({ case: scenario.name, width, message: error.message })
			);
			try {
				await ready(page, scenario.preset);
				if (
					width < 500 &&
					(scenario.name.startsWith('mode-') || scenario.name.startsWith('tool-'))
				) {
					for (let zoom = 0; zoom < 6; zoom++) {
						const worldHeight = await page.evaluate(() => {
							const engine = window.__algocell.engine;
							return (engine.config.height * innerWidth) / engine.view.zoom;
						});
						if (worldHeight >= height) break;
						await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
					}
					if (
						scenario.name.startsWith('tool-') &&
						!['tool-navigate', 'tool-fit'].includes(scenario.name)
					) {
						await page.mouse.move(width / 2, height / 2);
						await page.mouse.down();
						await page.mouse.move(width / 2, height / 2 - 150, { steps: 8 });
						await page.mouse.up();
					}
				}
				await scenario.prepare(page, width);
				await page.mouse.move(25, 25);
				await page.waitForTimeout(200);
				const path = `${directory}/${scenario.name}-${width}.png`;
				await page.screenshot({ path, animations: 'disabled' });
				captures.push({
					case: scenario.name,
					width,
					height,
					path,
					overflow: await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
				});
				console.log(path);
			} catch (error) {
				errors.push({ case: scenario.name, width, message: error.message });
				await page.screenshot({ path: `${directory}/error-${scenario.name}-${width}.png` });
			} finally {
				// Closing also stops paused rendering; the idle pause is real GPU rest.
				await page.close();
				await new Promise((resolve) => setTimeout(resolve, 2000));
			}
		}
	}
	await writeFile(
		`${directory}/capture-report.json`,
		JSON.stringify(
			{
				captures: [
					...priorReport.captures.filter(
						(old) => !captures.some((next) => old.case === next.case && old.width === next.width)
					),
					...captures
				],
				errors: [
					...priorReport.errors.filter(
						(old) => !captures.some((next) => old.case === next.case && old.width === next.width)
					),
					...errors
				]
			},
			null,
			2
		) + '\n'
	);
	if (errors.length) process.exitCode = 1;
} finally {
	await browser.close();
	await server?.close();
}

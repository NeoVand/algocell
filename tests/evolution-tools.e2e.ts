import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import type { GPUEngine } from '../src/lib/gpu/engine';

type TestWindow = Window & {
	__algocell: { engine: GPUEngine };
	observedSelect?: Element | null;
};

const modes = ['bytes', 'identity', 'confinement', 'zeros', 'diversity', 'marker'];

async function ready(page: Page) {
	await page.goto('/');
	await page.waitForFunction(() => Boolean((window as unknown as TestWindow).__algocell?.engine));
	await page.getByTitle('Pause (Space)', { exact: true }).click();
	await page.getByTitle('Settings', { exact: true }).click();
	await page.getByLabel('Experiments', { exact: true }).selectOption('displacement-32');
	await expect(page.getByLabel('Experiments', { exact: true })).toBeEnabled();
	await page.getByLabel('Close settings', { exact: true }).click();
	await expect(page.getByLabel('Field view', { exact: true })).toBeVisible();
	await page.evaluate(() => (window as unknown as TestWindow).__algocell.engine.waitForIdle());
}

async function tool(page: Page, name: string) {
	const selector = page.getByLabel('Observation tool', { exact: true });
	if (await selector.isVisible()) await selector.selectOption(name.toLowerCase());
	else await page.getByRole('button', { name, exact: true }).first().click();
}

async function probeCell(page: Page, index = 0) {
	await tool(page, 'Probe');
	await page.locator('#probe-cell').fill(String(index));
	await page
		.getByRole('region', { name: 'Probe tool', exact: true })
		.getByRole('button', {
			name: 'Probe',
			exact: true
		})
		.click();
	await expect(
		page.getByRole('button', { name: 'Open encounter bench', exact: true })
	).toBeEnabled();
}

async function focusSelection(page: Page, control: string, cells: number[]) {
	const before = await page.evaluate(async (cells) => {
		const engine = (window as unknown as TestWindow).__algocell.engine;
		return {
			update: engine.batchCount,
			tapes: await Promise.all(cells.map(async (cell) => [...(await engine.readCell(cell))]))
		};
	}, cells);
	await page.getByRole('button', { name: control, exact: true }).click();
	const after = await page.evaluate(async (cells) => {
		const engine = (window as unknown as TestWindow).__algocell.engine;
		return {
			update: engine.batchCount,
			tapes: await Promise.all(cells.map(async (cell) => [...(await engine.readCell(cell))]))
		};
	}, cells);
	expect(after).toEqual(before);
	const bounds = await page.evaluate((cells) => {
		const engine = (window as unknown as TestWindow).__algocell.engine;
		const canvas = document.querySelector('canvas')!.getBoundingClientRect();
		return cells.map((cell) => engine.cellBounds(cell, canvas.width, canvas.height));
	}, cells);
	const viewport = page.viewportSize()!;
	const panel = await page
		.getByRole('region', {
			name: control === 'Center probe' ? 'Probe tool' : 'Region tool',
			exact: true
		})
		.boundingBox();
	for (const box of bounds) {
		expect(box.x).toBeGreaterThanOrEqual(0);
		expect(box.y).toBeGreaterThanOrEqual(0);
		expect(box.x + box.width).toBeLessThanOrEqual(viewport.width + 1);
		expect(box.y + box.height).toBeLessThanOrEqual(
			(viewport.width < 500 ? panel!.y : viewport.height) + 1
		);
	}
}

async function recordFrames(page: Page) {
	await tool(page, 'History');
	await page.getByLabel('Record recent frames').check();
	await page.getByTitle('Play (Space)', { exact: true }).click();
	await page.waitForFunction(() => {
		const slider = document.querySelector<HTMLInputElement>('[aria-label="History frame"]');
		return slider && Number(slider.max) >= 1;
	});
	await page.getByTitle('Pause (Space)', { exact: true }).click();
	await page.getByLabel('Record recent frames').uncheck();
}

test.afterEach(async ({ page }) => {
	await page.close();
	await new Promise((resolve) => setTimeout(resolve, 2000));
});

for (const viewport of [
	{ width: 1440, height: 900 },
	{ width: 390, height: 844 }
]) {
	test.describe(`observation tools at ${viewport.width}×${viewport.height}`, () => {
		test.use({ viewport });

		test('all six field modes retain the paused population and native selector', async ({
			page
		}) => {
			const errors: string[] = [];
			page.on('pageerror', (error) => errors.push(error.message));
			await ready(page);
			const before = await page.evaluate(async () => {
				const engine = (window as unknown as TestWindow).__algocell.engine;
				(window as unknown as TestWindow).observedSelect = document.querySelector(
					'[aria-label="Field view"]'
				);
				return {
					update: engine.batchCount,
					tape: [...(await engine.readCell(0))],
					view: { ...engine.view }
				};
			});
			const select = page.getByLabel('Field view', { exact: true });
			expect(await select.evaluate((node) => node.tagName)).toBe('SELECT');
			expect(await select.evaluate((node) => getComputedStyle(node).appearance)).not.toBe('none');
			for (const mode of modes) {
				await select.selectOption(mode);
				await expect(select).toBeEnabled();
				await expect(select).toHaveValue(mode);
				await page.evaluate(() => new Promise(requestAnimationFrame));
				expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
					viewport.width
				);
				await expect(page.locator('body')).not.toContainText(/\bpaper\b/i);
			}
			const after = await page.evaluate(async () => {
				const engine = (window as unknown as TestWindow).__algocell.engine;
				return {
					update: engine.batchCount,
					tape: [...(await engine.readCell(0))],
					view: { ...engine.view }
				};
			});
			expect(after).toEqual(before);
			expect(
				await page.evaluate(
					() =>
						(window as unknown as TestWindow).observedSelect ===
						document.querySelector('[aria-label="Field view"]')
				)
			).toBe(true);
			expect(errors).toEqual([]);
		});

		test('probe and marker tools change only the explicitly selected byte', async ({ page }) => {
			await ready(page);
			await probeCell(page);
			await focusSelection(page, 'Center probe', [0]);
			const before = await page.evaluate(async () => {
				const engine = (window as unknown as TestWindow).__algocell.engine;
				return { tape: [...(await engine.readCell(0))], neighbor: [...(await engine.readCell(1))] };
			});
			await expect(page.getByRole('region', { name: 'Probe tool', exact: true })).toContainText(
				'0, 0'
			);
			await page.getByLabel('Field view', { exact: true }).selectOption('marker');
			await page.locator('#marker-position').fill('8');
			await page.locator('#marker-value').fill('165');
			await page.getByRole('button', { name: 'Show matches', exact: true }).click();
			await page.getByRole('button', { name: 'Plant in cell 0', exact: true }).click();
			await expect
				.poll(async () =>
					page.evaluate(
						async () => (await (window as unknown as TestWindow).__algocell.engine.readCell(0))[8]
					)
				)
				.toBe(0xa5);
			const after = await page.evaluate(async () => {
				const engine = (window as unknown as TestWindow).__algocell.engine;
				return { tape: [...(await engine.readCell(0))], neighbor: [...(await engine.readCell(1))] };
			});
			const expected = before.tape.slice();
			expected[8] = 0xa5;
			expect(after.tape).toEqual(expected);
			expect(after.neighbor).toEqual(before.neighbor);
			await tool(page, 'Probe');
			await page.getByRole('button', { name: 'Next cell', exact: true }).click();
			await expect(page.locator('#probe-cell')).toHaveValue('1');
			await page.getByRole('button', { name: 'Previous cell', exact: true }).focus();
			await page.keyboard.press('ArrowDown');
			const width = await page.evaluate(
				() => (window as unknown as TestWindow).__algocell.engine.config.width
			);
			await expect(page.locator('#probe-cell')).toHaveValue(String(width + 1));
		});

		test('rapid probe selection keeps the last tape when an earlier read finishes late', async ({
			page
		}) => {
			await ready(page);
			await probeCell(page);
			await page.evaluate(async () => {
				const engine = (window as unknown as TestWindow).__algocell.engine;
				await engine.plantMarker(1, 8, 0x11);
				await engine.plantMarker(2, 8, 0x22);
				const readCell = engine.readCell.bind(engine);
				engine.readCell = async (cell: number) => {
					const bytes = await readCell(cell);
					if (cell === 1) await new Promise((resolve) => setTimeout(resolve, 80));
					return bytes;
				};
			});
			const next = page.getByRole('button', { name: 'Next cell', exact: true });
			await next.click();
			await next.click();
			await expect(page.locator('#probe-cell')).toHaveValue('2');
			await page.waitForTimeout(150);
			const probe = page.getByRole('region', { name: 'Probe tool', exact: true });
			await expect(probe.locator('.scope-line')).toContainText('2, 0');
			await expect(probe.locator('.byte-tape span').nth(8)).toContainText('22');
		});

		test('a numeric region sample reports exact composition and its update', async ({ page }) => {
			await ready(page);
			await tool(page, 'Region');
			for (const [label, value] of [
				['Region from x', '0'],
				['Region from y', '0'],
				['Region to x', '1'],
				['Region to y', '1']
			])
				await page.getByLabel(label, { exact: true }).fill(value);
			await page.getByRole('button', { name: 'Measure region', exact: true }).click();
			const region = page.getByRole('region', { name: 'Region tool', exact: true });
			await expect(region.locator('dl.measurement-list')).toBeVisible();
			const measure = (name: string) =>
				region
					.locator('dl.measurement-list > div')
					.filter({
						has: page.getByText(name, { exact: true })
					})
					.locator('dd');
			await expect(measure('Exact unique tapes')).toHaveText('1');
			await expect(measure('Largest tape share')).toHaveText(/100(?:\.0)?%/);
			await expect(measure('Cells')).toHaveText('4');
			await expect(region).toContainText(/update\s+0/i);
			await expect(page.getByTitle('Play (Space)', { exact: true })).toBeVisible();
			expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(viewport.width);
			await page.getByLabel('Region to x', { exact: true }).fill('2');
			await expect(region.locator('dl.measurement-list')).not.toBeVisible();
			await page.getByRole('button', { name: 'Measure region', exact: true }).click();
			await expect(measure('Cells')).toHaveText('6');
			await expect(measure('Exact unique tapes')).toHaveText('1');
			const width = await page.evaluate(
				() => (window as unknown as TestWindow).__algocell.engine.config.width
			);
			await focusSelection(page, 'Frame region', [0, width + 2]);
		});

		test('history scrubbing and visual comparison leave simulation state intact', async ({
			page
		}) => {
			await ready(page);
			await recordFrames(page);
			const update = await page.evaluate(
				() => (window as unknown as TestWindow).__algocell.engine.batchCount
			);
			const range = page.getByLabel('History frame', { exact: true });
			await range.focus();
			await page.keyboard.press('Home');
			await expect(page.getByAltText(/Recorded lattice at update/)).toBeVisible();
			await page.getByRole('button', { name: 'Replay', exact: true }).click();
			await expect(page.getByTitle('Play (Space)', { exact: true })).toBeVisible();
			await page.getByRole('button', { name: 'Return live', exact: true }).click();
			expect(
				await page.evaluate(() => (window as unknown as TestWindow).__algocell.engine.batchCount)
			).toBe(update);
			await tool(page, 'Compare');
			await page.getByRole('button', { name: 'Pin reference', exact: true }).click();
			await page.getByLabel('Show comparison', { exact: true }).check();
			await expect(page.getByAltText(/Reference lattice at update/)).toBeVisible();
			const reveal = page.getByLabel('Reveal reference', { exact: true });
			await reveal.focus();
			await page.keyboard.press('Home');
			await page.keyboard.press('ArrowRight');
			expect(Number(await reveal.inputValue())).toBeGreaterThan(0);
			expect(
				await page.evaluate(() => (window as unknown as TestWindow).__algocell.engine.batchCount)
			).toBe(update);
		});

		test('PNG and sampled WebM exports download real images and video', async ({ page }) => {
			await ready(page);
			await recordFrames(page);
			await tool(page, 'Record');
			const pngEvent = page.waitForEvent('download');
			await page.getByRole('button', { name: 'Save PNG snapshot', exact: true }).click();
			const png = await pngEvent;
			const image = await readFile((await png.path())!);
			expect(image.subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');
			expect(png.suggestedFilename()).toMatch(/\.png$/);
			const exportVideo = page.getByRole('button', { name: /Export WebM/ });
			await expect(exportVideo).toBeEnabled();
			const webmEvent = page.waitForEvent('download');
			await exportVideo.click();
			const webm = await webmEvent;
			const video = await readFile((await webm.path())!);
			expect(webm.suggestedFilename()).toMatch(/\.webm$/);
			expect(video.subarray(0, 4).toString('hex')).toBe('1a45dfa3');
			expect(video.length).toBeGreaterThan(100);
		});

		test('the encounter bench exposes measured lanes and reproducible evidence', async ({
			page
		}) => {
			await ready(page);
			await probeCell(page);
			await page.getByRole('button', { name: 'Open encounter bench', exact: true }).click();
			const bench = page.getByRole('dialog', { name: 'Tape inspector', exact: true });
			await bench.getByRole('button', { name: 'Run culture test', exact: true }).click();
			await expect(bench.locator('.verdict')).toContainText(
				/Continuing production in this screen/i
			);
			await bench.getByRole('button', { name: 'Self-pair', exact: true }).click();
			await bench.getByRole('button', { name: 'Run comparison tests', exact: true }).click();
			await expect(
				bench.getByRole('button', { name: 'Run comparison tests', exact: true })
			).toBeEnabled();
			await expect(
				bench.getByRole('button', { name: 'Save evidence JSON', exact: true })
			).toBeEnabled();
			for (const name of ['Fetched bytes', 'Instruction starts', 'Write activity']) {
				await bench.getByRole('button', { name, exact: true }).click();
				await expect(bench).not.toContainText(/\bpaper\b/i);
			}
			for (const name of ['Partner sensitivity', 'Marker transfer']) {
				await bench.getByRole('button', { name, exact: true }).click();
				await expect(bench).not.toContainText(/\bpaper\b/i);
			}
			const event = page.waitForEvent('download');
			await bench.getByRole('button', { name: 'Save evidence JSON', exact: true }).click();
			const download = await event;
			const evidence = JSON.parse(await readFile((await download.path())!, 'utf8'));
			expect(evidence.schema).toBe('algocell.encounter-bench/1');
			expect(evidence.settings.isa).toBe('z80');
			expect(evidence.manifest.schema).toBe('zilion.manifest/2');
			expect(evidence.marker.retention).toHaveLength(2);
			expect(evidence.partnerSensitivity.trials).toBe(16);
			expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(viewport.width);
		});
	});
}

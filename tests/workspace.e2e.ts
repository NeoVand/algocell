import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { PAPER_PRESETS } from '../src/lib/lab/presets';
import { choose, experiment, probe, ready, tab, type AppWindow } from './helpers/unified';

test.afterEach(async ({ page }) => {
	await page.close();
	await new Promise((resolve) => setTimeout(resolve, 2000));
});

test('the unified shell keeps controls stable while running', async ({ page }) => {
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await ready(page);
	await tab(page, 'World');
	const processor = page.getByRole('combobox', { name: 'Processor', exact: true });
	await expect(processor).toHaveAttribute('data-value', 'z80-0.2');
	await processor.evaluate((element) => {
		(window as unknown as { observedControl: Element }).observedControl = element;
	});
	const before = await page.evaluate(
		() => (window as unknown as AppWindow).__algocell.engine.batchCount
	);
	await page.getByTitle('Play (Space)', { exact: true }).click();
	await expect
		.poll(() => page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.batchCount))
		.toBeGreaterThan(before);
	expect(
		await processor.evaluate(
			(element) => element === (window as unknown as { observedControl: Element }).observedControl
		)
	).toBe(true);
	await page.getByTitle('Pause (Space)', { exact: true }).click();
	await processor.click();
	expect(await processor.evaluate((element) => getComputedStyle(element).outlineStyle)).toBe(
		'none'
	);
	await page.keyboard.press('Escape');
	await processor.focus();
	await page.keyboard.press('Tab');
	expect(await page.evaluate(() => getComputedStyle(document.activeElement!).outlineWidth)).toBe(
		'1px'
	);
	await expect(page.locator('select')).toHaveCount(0);
	expect(errors).toEqual([]);
});

for (const preset of PAPER_PRESETS) {
	test(`guided experiment stages and runs: ${preset.id}`, async ({ page }) => {
		const errors: string[] = [];
		page.on('pageerror', (error) => errors.push(error.message));
		await ready(page);
		const before = await page.evaluate(() => ({
			settings: (window as unknown as AppWindow).__algocell.engine.settings,
			update: (window as unknown as AppWindow).__algocell.engine.batchCount
		}));
		await tab(page, 'Experiments');
		await choose(page, 'Experiments', preset.id);
		expect(
			await page.evaluate(() => ({
				settings: (window as unknown as AppWindow).__algocell.engine.settings,
				update: (window as unknown as AppWindow).__algocell.engine.batchCount
			}))
		).toEqual(before);
		await page
			.getByRole('button', { name: 'Start new world with these settings', exact: true })
			.click();
		await expect(
			page.getByRole('button', { name: 'Start new world with these settings', exact: true })
		).toBeEnabled();
		await expect
			.poll(() =>
				page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.settings.isa)
			)
			.toBe(preset.config.isa);
		const applied = await page.evaluate(
			() => (window as unknown as AppWindow).__algocell.engine.settings
		);
		expect(applied).toMatchObject({ ...preset.config });
		await expect(page.locator('.experiment-condition')).toContainText(
			preset.whatToLookFor
				.replace('pale zero-rich', 'zero-rich')
				.replace(
					'instruction fetching enters the partner',
					'instruction fetching leaves the focal tape (B or padding)'
				)
		);
		await page.getByTitle('Play (Space)', { exact: true }).click();
		await expect
			.poll(() =>
				page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.batchCount)
			)
			.toBeGreaterThan(0);
		await page.getByTitle('Pause (Space)', { exact: true }).click();
		await expect(page.getByRole('alert')).toHaveCount(0);
		await expect(page.locator('body')).not.toContainText(/\bpaper\b/i);
		expect(errors).toEqual([]);
	});
}

test('ISA changes, confinement and separate culture stay in one shell', async ({ page }) => {
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	await ready(page);
	await experiment(page, 'no-copy');
	await tab(page, 'World');
	for (const machine of ['m6502', 'i8086', 'z80-0.3', 'z80-nmos', 'z80-0.2']) {
		await choose(page, 'Processor', machine);
		await expect(page.getByRole('alert')).toHaveCount(0);
	}
	await experiment(page, 'displacement-32');
	await choose(page, 'Field view', 'confinement');
	await probe(page);
	await page.getByRole('button', { name: 'Open encounter bench', exact: true }).click();
	const inspector = page.getByRole('region', { name: 'Tape inspector', exact: true });
	await expect(inspector).toBeVisible();
	await inspector.getByRole('button', { name: 'Run culture test', exact: true }).click();
	await expect(inspector.locator('.bench table')).toBeVisible();
	await expect(inspector.locator('.verdict')).toContainText('Continuing production in this screen');
	await expect(inspector.locator('.bench table tbody tr')).toHaveCount(2);
	await expect(page.getByRole('dialog')).toHaveCount(0);
	expect(errors).toEqual([]);
});

test('About and illustrated Guide occupy the sidebar on a narrow screen', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 844 });
	await ready(page);
	for (const name of ['About', 'Guide', 'World', 'Measure', 'Experiments']) {
		await tab(page, name);
		await expect(page.locator('.app-sidebar')).toBeVisible();
		expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
		await expect(page.getByRole('dialog')).toHaveCount(0);
		await expect(page.locator('body')).not.toContainText(/\bpaper\b/i);
	}
	await page.goto('/legacy');
	await expect(page.locator('body')).toContainText('404');
});

test('without WebGPU the Guide and About remain accessible', async ({ page }) => {
	await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
	await page.goto('/');
	await expect(page.locator('body')).toContainText('WebGPU');
	await tab(page, 'About');
	await expect(page.locator('.app-sidebar')).toContainText('Copying is not inheriting');
});

test('acell checkpoint import restores the selected ISA and population', async ({ page }) => {
	await ready(page);
	await experiment(page, 'm6502-s1-a');
	const before = await page.evaluate(async () => [
		...(await (window as unknown as AppWindow).__algocell.engine.readCell(0))
	]);
	const download = page.waitForEvent('download');
	await page.keyboard.press('ControlOrMeta+s');
	const saved = await download;
	const file = await readFile((await saved.path())!);
	await tab(page, 'World');
	await choose(page, 'Processor', 'i8086');
	await page
		.locator('input[type=file]')
		.setInputFiles({ name: 'saved.acell', mimeType: 'application/octet-stream', buffer: file });
	expect(
		await page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.settings.isa)
	).toBe('i8086');
	await page.getByRole('button', { name: 'Restore checkpoint', exact: true }).click();
	await expect
		.poll(() =>
			page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.settings.isa)
		)
		.toBe('m6502');
	expect(
		await page.evaluate(async () => [
			...(await (window as unknown as AppWindow).__algocell.engine.readCell(0))
		])
	).toEqual(before);
	await expect(page.getByRole('alert')).toHaveCount(0);
});

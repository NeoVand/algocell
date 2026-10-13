import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { PAPER_PRESETS } from '../src/lib/lab/presets';

test.afterEach(async ({ page }) => {
	await page.close();
	await new Promise((resolve) => setTimeout(resolve, 2000));
});

async function ready(page: import('@playwright/test').Page) {
	await page.goto('/');
	await page.waitForFunction(() =>
		Boolean((window as unknown as { __algocell?: { engine?: unknown } }).__algocell?.engine)
	);
	await expect(page.getByTitle('Pause (Space)', { exact: true })).toBeVisible();
	await page.getByTitle('Pause (Space)', { exact: true }).click();
	await page.getByTitle('Settings', { exact: true }).click();
	await expect(page.getByLabel('Experiments')).toBeEnabled();
}

test('restores the original full-screen instrument and stable controls while running', async ({
	page
}) => {
	const errors: string[] = [];
	page.on('pageerror', (e) => errors.push(e.message));
	await ready(page);
	expect(await page.getByLabel('Processor', { exact: true }).inputValue()).toBe('z80-0.2');
	await page.evaluate(() => {
		const select = document.querySelector('#experiment-preset')!;
		const chip = document.querySelector('.preset-chip')!;
		(window as unknown as { controlCheck: () => boolean }).controlCheck = () =>
			select === document.querySelector('#experiment-preset') &&
			chip === document.querySelector('.preset-chip');
	});
	await page.getByTitle('Play (Space)', { exact: true }).click();
	await expect.poll(async () => page.locator('.info-value').first().innerText()).not.toBe('0');
	await expect(page.getByLabel('Experiments')).toBeEnabled();
	expect(
		await page.evaluate(() => (window as unknown as { controlCheck: () => boolean }).controlCheck())
	).toBe(true);
	await page.getByTitle('Pause (Space)', { exact: true }).click();
	await page.getByLabel('Processor', { exact: true }).click();
	expect(
		await page
			.getByLabel('Processor', { exact: true })
			.evaluate((el) => getComputedStyle(el).outlineStyle)
	).toBe('none');
	await page.getByLabel('Processor', { exact: true }).focus();
	await page.keyboard.press('Tab');
	expect(await page.evaluate(() => getComputedStyle(document.activeElement!).outlineWidth)).toBe(
		'1px'
	);
	expect(errors).toEqual([]);
});

for (const preset of PAPER_PRESETS)
	test(`experiment preset runs in original controls: ${preset.id}`, async ({ page }) => {
		const errors: string[] = [];
		page.on('pageerror', (e) => errors.push(e.message));
		await ready(page);
		await page.getByLabel('Experiments').selectOption(preset.id);
		await expect(page.getByLabel('Experiments')).toBeEnabled();
		expect(
			await page.evaluate(() => {
				const host = (
					window as unknown as {
						__algocell: {
							engine: { config: { width: number; height: number }; view: { zoom: number } };
						};
					}
				).__algocell.engine;
				const canvas = document.querySelector('canvas')!;
				return (
					host.view.zoom >= host.config.width &&
					host.view.zoom / (canvas.width / canvas.height) >= host.config.height - 0.001
				);
			})
		).toBe(true);
		await expect(page.locator('.experiment-condition')).toContainText(
			preset.whatToLookFor
				.replace('pale zero-rich', 'zero-rich')
				.replace(
					'instruction fetching enters the partner',
					'instruction fetching leaves the focal tape (B or padding)'
				)
		);
		await page.getByTitle('Play (Space)', { exact: true }).click();
		await expect.poll(async () => page.locator('.info-value').first().innerText()).not.toBe('0');
		await page.getByTitle('Pause (Space)', { exact: true }).click();
		await expect(page.getByRole('alert')).toHaveCount(0);
		await expect(page.locator('body')).not.toContainText(/\bpaper\b/i);
		expect(errors).toEqual([]);
	});

test('ISA changes, GPU fetch overlay, tape disassembly and separate culture run', async ({
	page
}) => {
	const errors: string[] = [];
	page.on('pageerror', (e) => errors.push(e.message));
	await ready(page);
	await page.getByLabel('Experiments').selectOption('no-copy');
	await expect(page.getByLabel('Processor', { exact: true })).toBeEnabled();
	for (const machine of ['m6502', 'i8086', 'z80-0.3', 'z80-nmos', 'z80-0.2']) {
		await page.getByLabel('Processor', { exact: true }).selectOption(machine);
		await expect(page.getByLabel('Processor', { exact: true })).toBeEnabled();
		await expect(page.getByRole('alert')).toHaveCount(0);
	}
	await page.getByLabel('Experiments').selectOption('displacement-32');
	await expect(page.getByLabel('Experiments')).toBeEnabled();
	await page.getByLabel('Close settings', { exact: true }).click();
	await page.getByLabel('Field view', { exact: true }).selectOption('confinement');
	await expect(page.getByLabel('Field view', { exact: true })).toBeEnabled();
	await page
		.locator('canvas')
		.first()
		.click({ position: { x: 650, y: 450 } });
	await page.getByRole('button', { name: 'Open encounter bench', exact: true }).click();
	const inspector = page.getByRole('dialog', { name: 'Tape inspector', exact: true });
	await expect(inspector).toBeVisible();
	await inspector.getByRole('button', { name: 'Run culture test', exact: true }).click();
	await expect(inspector.locator('.bench table')).toBeVisible();
	await expect(inspector.locator('.bench .verdict')).toContainText(
		'Continuing production in this screen'
	);
	await expect(inspector.locator('.bench table tbody tr')).toHaveCount(2);
	await inspector.getByRole('button', { name: 'Close tape inspector', exact: true }).click();
	expect(errors).toEqual([]);
});

test('About and Guide use the original modal and mobile stays within the viewport', async ({
	page
}) => {
	await page.setViewportSize({ width: 390, height: 844 });
	await ready(page);
	await page.getByLabel('Close settings', { exact: true }).click();
	expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
	await page.getByTitle('Help (H)', { exact: true }).click();
	for (const tab of ['About', 'Guide', 'Overview', 'Visuals', 'Z80', 'Parameters', 'Analysis']) {
		await page.getByRole('button', { name: tab, exact: true }).click();
		await expect(page.getByRole('dialog', { name: 'Algocell details', exact: true })).toBeVisible();
		expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(390);
		await expect(page.locator('body')).not.toContainText(/\bpaper\b/i);
	}
	await page.keyboard.press('Escape');
	await expect(page.getByRole('dialog')).toHaveCount(0);
	await page.goto('/legacy');
	expect(page.url()).toContain('/legacy');
	await expect(page.locator('body')).toContainText('404');
});

test('without WebGPU documentation remains readable', async ({ page }) => {
	await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
	await page.goto('/');
	await expect(page.locator('body')).toContainText('WebGPU');
	await page.getByTitle('Help (H)', { exact: true }).click();
	await page.getByRole('button', { name: 'About', exact: true }).click();
	await expect(page.getByRole('dialog')).toContainText('Copying is not inheriting');
});

test('acell checkpoints retain the selected ISA and world when restored', async ({ page }) => {
	await ready(page);
	await page.getByLabel('Experiments').selectOption('m6502-s1-a');
	await expect(page.getByLabel('Experiments')).toBeEnabled();
	await page.getByLabel('Close settings', { exact: true }).click();
	const download = page.waitForEvent('download');
	await page.getByTitle('Save checkpoint (Cmd+S)', { exact: true }).click();
	const saved = await download;
	const file = await readFile((await saved.path())!);
	await page.getByTitle('Settings', { exact: true }).click();
	await page.getByLabel('Processor', { exact: true }).selectOption('i8086');
	await expect(page.getByLabel('Processor', { exact: true })).toBeEnabled();
	await page
		.locator('input[type=file]')
		.setInputFiles({ name: 'saved.acell', mimeType: 'application/octet-stream', buffer: file });
	await expect(page.getByLabel('Processor', { exact: true })).toHaveValue('m6502');
	await expect(page.getByLabel('Processor', { exact: true })).toBeEnabled();
	await expect(page.getByRole('alert')).toHaveCount(0);
});

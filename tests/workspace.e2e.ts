import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { PAPER_PRESETS } from '../src/lib/lab/presets';

test('loads locally, runs, inspects a tape, and opens accessible documentation', async ({
	page
}) => {
	const errors: string[] = [];
	const external: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	page.on('request', (request) => {
		if (/^https?:/.test(request.url()) && new URL(request.url()).hostname !== '127.0.0.1')
			external.push(request.url());
	});
	await page.goto('/');
	await expect(page.getByRole('heading', { name: 'Algocell', exact: true })).toBeVisible();
	await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
	await page.getByRole('button', { name: 'Run', exact: true }).click();
	await expect
		.poll(async () =>
			Number((await page.locator('.run-status strong').innerText()).replaceAll(',', ''))
		)
		.toBeGreaterThan(0);
	await page.getByRole('button', { name: 'Pause', exact: true }).click();
	await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeVisible();
	await page.getByLabel('Cell', { exact: true }).fill('12');
	await page.getByLabel('Cell', { exact: true }).press('Tab');
	await expect(page.getByRole('heading', { name: 'Linear disassembly' })).toBeVisible();
	await expect(page.locator('.inspector tbody tr')).not.toHaveCount(0);
	await page.getByLabel('Lattice layer').selectOption('fetch');
	await expect(page.getByText(/Unknown|Unmeasured/).first()).toBeVisible();
	for (const name of ['About', 'Guide']) {
		await page.getByRole('button', { name, exact: true }).click();
		await expect(page.getByRole('dialog')).toBeVisible();
		await page.keyboard.press('Escape');
		await expect(page.getByRole('dialog')).not.toBeVisible();
	}
	expect(errors).toEqual([]);
	expect(external).toEqual([]);
});

for (const preset of PAPER_PRESETS) {
	test(`paper preset runs: ${preset.id}`, async ({ page }) => {
		await page.goto('/');
		await expect(page.getByRole('button', { name: 'Step', exact: true })).toBeEnabled();
		await page.getByLabel('Experiment', { exact: true }).selectOption(preset.id);
		await page.getByRole('button', { name: 'Start this experiment', exact: true }).click();
		await expect(page.locator('.world-caption strong')).toHaveText(preset.name);
		await expect(page.getByRole('button', { name: 'Step', exact: true })).toBeEnabled();
		await page.getByRole('button', { name: 'Step', exact: true }).click();
		await expect(page.locator('.run-status strong')).toHaveText('1');
		await expect(page.getByRole('alert')).toHaveCount(0);
	});
}

test('switches processor and Z80 variant through world controls', async ({ page }) => {
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Step', exact: true })).toBeEnabled();
	for (const isa of ['m6502', 'i8086', 'z80']) {
		await page.getByLabel('Processor', { exact: true }).selectOption(isa);
		if (isa === 'z80') {
			await page.getByLabel('Compatibility', { exact: true }).selectOption('0.3');
			await page.getByLabel('Variant', { exact: true }).selectOption('zilog-nmos');
		}
		await page.getByRole('button', { name: 'Apply settings & reset', exact: true }).click();
		await expect(page.getByRole('button', { name: 'Step', exact: true })).toBeEnabled();
		await expect(page.getByRole('alert')).toHaveCount(0);
		await page.getByRole('button', { name: 'Step', exact: true }).click();
		await expect(page.locator('.run-status strong')).toHaveText('1');
	}
});

test('cultures a selected tape and compares marker transfer with a control', async ({ page }) => {
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Step', exact: true })).toBeEnabled();
	await page.getByLabel('Experiment', { exact: true }).selectOption('displacement-32');
	await page.getByRole('button', { name: 'Start this experiment', exact: true }).click();
	await expect(page.locator('.world-caption strong')).toHaveText('Displacement · full tape');
	await page.getByRole('button', { name: 'Culture', exact: true }).click();
	await page.getByRole('button', { name: 'Run culture test', exact: true }).click();
	await expect(page.getByRole('button', { name: 'Run culture test', exact: true })).toBeEnabled();
	await expect(page.locator('.inspector')).toContainText('Generation');
	await page.getByRole('button', { name: 'Marker', exact: true }).click();
	await page.getByRole('button', { name: 'Test marker transfer', exact: true }).click();
	await expect(
		page.getByRole('button', { name: 'Test marker transfer', exact: true })
	).toBeEnabled();
	await expect(page.locator('.inspector')).toContainText('control');
	await expect(page.getByRole('alert')).toHaveCount(0);
});

test('mobile workspace fits the viewport and documents remain navigable', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 844 });
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeEnabled();
	expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
	await expect(page.getByRole('link', { name: 'Experiments', exact: true })).toBeVisible();
	await expect(page.getByRole('link', { name: 'Inspect a tape', exact: true })).toBeVisible();
	await page.getByRole('button', { name: 'Guide', exact: true }).click();
	await expect(page.getByRole('dialog')).toBeVisible();
	expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
	await page.getByRole('button', { name: 'Close guide', exact: true }).click();
	await expect(page.getByRole('button', { name: 'Guide', exact: true })).toBeFocused();
});

test('checkpoint files resume exactly and an invalid import preserves the current run', async ({
	page
}) => {
	await page.goto('/');
	const step = page.getByRole('button', { name: 'Step', exact: true });
	await expect(step).toBeEnabled();
	await step.click();
	await expect(page.locator('.run-status strong')).toHaveText('1');
	await page.getByText('Data & reproducibility', { exact: true }).click();
	const save = async () => {
		const pending = page.waitForEvent('download');
		await page.getByRole('button', { name: 'Save checkpoint', exact: true }).click();
		const download = await pending;
		const path = await download.path();
		expect(path).toBeTruthy();
		return readFile(path!);
	};
	const checkpoint = await save();
	await step.click();
	await expect(page.locator('.run-status strong')).toHaveText('2');
	const expected = JSON.parse((await save()).toString());
	const importer = page.getByLabel('Import checkpoint file', { exact: true });
	await importer.setInputFiles({
		name: 'saved.json',
		mimeType: 'application/json',
		buffer: checkpoint
	});
	await expect(page.locator('.run-status strong')).toHaveText('1');
	await expect(step).toBeEnabled();
	await step.click();
	await expect(page.locator('.run-status strong')).toHaveText('2');
	const actual = JSON.parse((await save()).toString());
	expect(actual.bytes).toEqual(expected.bytes);
	expect(actual.rngState).toBe(expected.rngState);
	expect(actual.encounters).toBe(expected.encounters);
	await importer.setInputFiles({
		name: 'invalid.json',
		mimeType: 'application/json',
		buffer: Buffer.from('{"schema":"wrong"}')
	});
	await expect(page.getByRole('alert')).toContainText('not a supported');
	await expect(page.locator('.run-status strong')).toHaveText('2');
	await expect(step).toBeEnabled();
	const preserved = JSON.parse((await save()).toString());
	expect(preserved.bytes).toEqual(actual.bytes);
	expect(preserved.rngState).toBe(actual.rngState);
});

test('without WebGPU the preview and documentation stay readable and execution is disabled', async ({
	page
}) => {
	await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined }));
	await page.goto('/');
	await expect(page.getByRole('alert')).toContainText('WebGPU');
	await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeDisabled();
	await expect(page.getByRole('button', { name: 'Step', exact: true })).toBeDisabled();
	await expect(page.getByLabel('Experiment', { exact: true })).toBeEnabled();
	await page.getByRole('button', { name: 'About', exact: true }).click();
	await expect(page.getByRole('dialog')).toBeVisible();
	await page.keyboard.press('Escape');
	await expect(page.getByRole('button', { name: 'About', exact: true })).toBeFocused();
});

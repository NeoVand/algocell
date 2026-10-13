import { expect, type Page } from '@playwright/test';
import type { GPUEngine } from '../../src/lib/gpu/engine';

export type AppWindow = Window & {
	__algocell: { engine: GPUEngine; playing?: boolean };
};

export async function choose(page: Page, label: string, value: string) {
	const control = page.getByRole('combobox', { name: label, exact: true });
	await control.click();
	const option = page.locator(`[role="option"][data-value="${value}"]`);
	await option.click();
	await expect(control).toHaveAttribute('data-value', value);
	await expect(control).toBeEnabled();
}

export async function tab(page: Page, name: string) {
	await page.getByRole('tab', { name, exact: true }).click();
}

export async function ready(page: Page, pause = true) {
	await page.goto('/');
	await page.waitForFunction(() => Boolean((window as unknown as AppWindow).__algocell?.engine));
	await expect(page.getByTitle('Pause (Space)', { exact: true })).toBeVisible();
	if (pause) await page.getByTitle('Pause (Space)', { exact: true }).click();
}

export async function experiment(page: Page, id: string) {
	await tab(page, 'Experiments');
	await choose(page, 'Experiments', id);
	await page
		.getByRole('button', { name: 'Start new world with these settings', exact: true })
		.click();
	await expect(
		page.getByRole('button', { name: 'Start new world with these settings', exact: true })
	).toBeEnabled();
	await page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.waitForIdle());
}

export async function tool(page: Page, name: string) {
	if (['History', 'Charts', 'Record'].includes(name)) {
		await tab(page, 'Measure');
		await page
			.getByRole('button', { name: name === 'Record' ? 'Export' : name, exact: true })
			.first()
			.click();
	} else {
		await page
			.getByRole('navigation', { name: 'Lattice tools', exact: true })
			.getByRole('button', { name, exact: true })
			.click();
	}
}

export async function probe(page: Page, cell = 0) {
	await tool(page, 'Probe');
	await page.locator('#probe-cell').fill(String(cell));
	await page
		.getByRole('region', { name: 'Probe tool', exact: true })
		.getByRole('button', { name: 'Probe', exact: true })
		.click();
	await expect(
		page.getByRole('button', { name: 'Open encounter bench', exact: true })
	).toBeEnabled();
}

import { expect, test } from '@playwright/test';
import { choose, experiment, probe, ready, tab, tool, type AppWindow } from './helpers/unified';

test.afterEach(async ({ page }) => {
	await page.close();
	await new Promise((resolve) => setTimeout(resolve, 2000));
});

test('Guide remembers its selected section after visiting About', async ({ page }) => {
	await ready(page);
	await tab(page, 'Guide');
	for (const section of ['analysis', 'params']) {
		await choose(page, 'Guide section', section);
		await tab(page, 'About');
		await expect(page.getByRole('tab', { name: 'About', exact: true })).toHaveAttribute(
			'aria-selected',
			'true'
		);
		await tab(page, 'Guide');
		await expect(
			page.getByRole('combobox', { name: 'Guide section', exact: true })
		).toHaveAttribute('data-value', section);
	}
});

test('a live core swap explains the reset conventions and suppression it changes', async ({
	page
}) => {
	await ready(page);
	await tab(page, 'World');
	await page.getByText('Budgets & instruction ablation', { exact: true }).click();
	const ablation = page.getByLabel('Suppress instructions', { exact: true });
	await ablation.fill('LDIR');
	await ablation.press('Enter');
	await expect
		.poll(() =>
			page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.settings.ablation)
		)
		.toContain('LDIR');
	await choose(page, 'Processor', 'm6502');
	const notice = page.locator('.world-panel .inline-notice');
	await expect(notice).toContainText('Core changed on the existing lattice.');
	await expect(notice).toContainText('Stack: first push at B’s last byte');
	await expect(notice).toContainText('dial: counter');
	await expect(notice).toContainText('instruction suppression cleared');
	expect(
		await page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.settings)
	).toMatchObject({ isa: 'm6502', stack: 'first-push', dialMode: 'counter', ablation: [] });
});

test('both start actions describe the full staged condition rather than the live world', async ({
	page
}) => {
	await ready(page);
	const applied = await page.evaluate(
		() => (window as unknown as AppWindow).__algocell.engine.settings
	);
	await tab(page, 'Experiments');
	await choose(page, 'Experiments', 'm6502-s1-a');
	const summary = page.getByRole('group', { name: 'Next world summary', exact: true });
	await expect(summary).toBeVisible();
	for (const detail of [
		'NMOS 6502 · compat 0.3',
		'Mutation 1 in 16',
		'Zero 0% lethal',
		'8,192 pairs/update',
		'128 steps/encounter',
		'160 × 125 tapes',
		'L16/P32',
		'seed 65001'
	]) {
		await expect(summary).toContainText(detail);
	}
	const description = await summary.innerText();
	await tab(page, 'World');
	await expect(summary).toHaveText(description, { useInnerText: true });
	await page.getByLabel('Seed', { exact: true }).fill('12345');
	await expect(summary).toContainText('seed 12345');
	await tab(page, 'Experiments');
	await expect(summary).toContainText('seed 12345');
	await expect(summary).not.toContainText('seed 65001');
	expect(
		await page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.settings)
	).toEqual(applied);
});

test('starting history from Export keeps navigation and playback in sync', async ({ page }) => {
	await ready(page, false);
	await tool(page, 'Record');
	await page.getByRole('button', { name: 'Start recording history', exact: true }).click();
	const sections = page.getByRole('navigation', { name: 'Measure sections', exact: true });
	await expect(sections.getByRole('button', { name: 'History', exact: true })).toHaveClass(
		/\bactive\b/
	);
	await expect(sections.getByRole('button', { name: 'Export', exact: true })).not.toHaveClass(
		/\bactive\b/
	);
	await expect(page.getByLabel('Record recent frames')).toBeChecked();
	await expect(page.getByTitle('Pause (Space)', { exact: true })).toBeVisible();
	await tab(page, 'Guide');

	// Observe several 500 ms child state publications. A single immediate
	// assertion would miss the old periodic callback pulling the user back.
	const samples = await page.evaluate(async () => {
		const states: { tab: string | null; playing: boolean | undefined; update: number }[] = [];
		for (let index = 0; index < 6; index++) {
			await new Promise((resolve) => setTimeout(resolve, 250));
			const app = (window as unknown as AppWindow).__algocell;
			states.push({
				tab:
					document
						.querySelector('.sidebar-tabs [aria-selected="true"]')
						?.getAttribute('aria-label') ?? null,
				playing: app.playing,
				update: app.engine.batchCount
			});
		}
		return states;
	});
	expect(samples.every((state) => state.tab === 'Guide' && state.playing)).toBe(true);
	expect(samples.at(-1)!.update).toBeGreaterThan(samples[0].update);
	await tab(page, 'Measure');
	await expect(sections.getByRole('button', { name: 'History', exact: true })).toHaveClass(
		/\bactive\b/
	);
	await expect(page.getByLabel('Record recent frames')).toBeChecked();
	await page.getByLabel('Record recent frames').uncheck();
	await page.getByTitle('Pause (Space)', { exact: true }).click();
});

test('all three population chart modes remain usable at 390 by 844', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 844 });
	await ready(page);
	await tool(page, 'Charts');
	const charts = page.getByRole('region', { name: 'Population charts', exact: true });
	for (const name of ['Opcodes', 'Species', 'Diversity']) {
		const control = charts.getByRole('button', { name, exact: true });
		await expect(control).toBeVisible();
		await expect(control).toBeInViewport({ ratio: 1 });
		await control.click();
		await expect(control).toHaveClass(/\bactive\b/);
		await expect(charts.locator('.info-tab.active')).toHaveCount(1);
		if (name === 'Species') await expect(charts.locator('.info-chart')).toBeVisible();
		if (name === 'Diversity')
			await expect(charts.getByText('Collecting diversity data...', { exact: true })).toBeVisible();
	}
	const width = await page.evaluate(() => ({
		content: document.documentElement.scrollWidth,
		viewport: window.innerWidth
	}));
	expect(width.content).toBeLessThanOrEqual(width.viewport);
	await expect(page.getByTitle('Play (Space)', { exact: true })).toBeVisible();
});

test('the encounter return path and cell identity stay visible through long results', async ({
	page
}) => {
	await page.setViewportSize({ width: 1280, height: 800 });
	await ready(page);
	await experiment(page, 'displacement-32');
	await probe(page, 7);
	await page.getByRole('button', { name: 'Open encounter bench', exact: true }).click();
	const inspector = page.getByRole('region', { name: 'Tape inspector', exact: true });
	await inspector.getByRole('button', { name: 'Self-pair', exact: true }).click();
	await inspector.getByRole('button', { name: 'Run comparison tests', exact: true }).click();
	await expect(inspector.locator('.evidence-source')).toContainText(/cell 7/i);
	const sample = await inspector.locator('.evidence-source').innerText();
	await inspector.locator('.after-tape > summary').click();
	await page.locator('.sidebar-scroll').evaluate((element) => {
		element.scrollTop = element.scrollHeight;
	});

	const context = inspector.locator('.inspector-return');
	await expect(context).toBeInViewport({ ratio: 1 });
	await expect(
		context.getByRole('button', { name: 'Close tape inspector', exact: true })
	).toBeInViewport({ ratio: 1 });
	await expect(context).toContainText('Encounter bench');
	await expect(context).toContainText('Cell 7');
	const geometry = await context.evaluate((element) => {
		const scroller = element.closest('.sidebar-scroll')!;
		const bounds = element.getBoundingClientRect();
		const scrollBounds = scroller.getBoundingClientRect();
		const atCenter = document.elementFromPoint(
			bounds.left + bounds.width / 2,
			bounds.top + bounds.height / 2
		);
		return {
			scrollTop: scroller.scrollTop,
			top: bounds.top,
			bottom: bounds.bottom,
			viewportTop: scrollBounds.top,
			viewportBottom: scrollBounds.bottom,
			uncovered: atCenter !== null && element.contains(atCenter)
		};
	});
	expect(geometry.scrollTop).toBeGreaterThan(200);
	expect(geometry.top).toBeGreaterThanOrEqual(geometry.viewportTop - 1);
	expect(geometry.bottom).toBeLessThanOrEqual(geometry.viewportBottom + 1);
	expect(geometry.uncovered).toBe(true);
	await expect(inspector.locator('.evidence-source')).toHaveText(sample, { useInnerText: true });
	await context.getByRole('button', { name: 'Close tape inspector', exact: true }).click();
	await expect(page.getByRole('region', { name: 'Probe tool', exact: true })).toBeVisible();
	await expect(page.locator('#probe-cell')).toHaveValue('7');
});

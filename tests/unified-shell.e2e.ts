import { expect, test, type Page } from '@playwright/test';
import { choose, experiment, probe, ready, tab, tool, type AppWindow } from './helpers/unified';

const modes = ['bytes', 'identity', 'confinement', 'zeros', 'diversity', 'marker'];
const tabs = ['World', 'Measure', 'Experiments', 'Guide', 'About'];

async function identity(page: Page) {
	return page.evaluate(() => {
		const session = window as unknown as AppWindow & { continuityDocumentId?: string };
		session.continuityDocumentId ??= crypto.randomUUID();
		const engine = session.__algocell.engine;
		const { engineId, deviceId, deviceInitializations, resetCount, bufferRevision } =
			engine.diagnostics;
		return {
			documentId: session.continuityDocumentId,
			engineId,
			deviceId,
			deviceInitializations,
			resetCount,
			bufferRevision
		};
	});
}
async function keepsAdvancing(page: Page, previous: Awaited<ReturnType<typeof identity>>) {
	await expect(page.getByTitle('Pause (Space)', { exact: true })).toBeVisible();
	expect(await identity(page)).toEqual(previous);
	const update = await page.evaluate(
		() => (window as unknown as AppWindow).__algocell.engine.batchCount
	);
	await expect
		.poll(() => page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.batchCount))
		.toBeGreaterThan(update);
}

test.afterEach(async ({ page }) => {
	await page.close();
	await new Promise((resolve) => setTimeout(resolve, 2000));
});

test('playing survives every tab, field view and direct tool without replacing the world', async ({
	page
}) => {
	await ready(page, false);
	const before = await identity(page);
	for (const name of tabs) {
		await tab(page, name);
		await keepsAdvancing(page, before);
	}
	for (const mode of modes) {
		await choose(page, 'Field view', mode);
		await keepsAdvancing(page, before);
	}
	for (const name of ['Probe', 'Region', 'Marker', 'Compare', 'Select / pan']) {
		await tool(page, name);
		await keepsAdvancing(page, before);
	}
	for (const key of [
		'1',
		'2',
		'3',
		'4',
		'5',
		'?',
		'Escape',
		'v',
		'p',
		'r',
		'm',
		'c',
		'f',
		'+',
		'-'
	]) {
		await page
			.locator('canvas')
			.first()
			.click({ position: { x: 60, y: 60 } });
		await page.keyboard.press(key);
		await keepsAdvancing(page, before);
	}
});

test('world drafts stay unapplied through navigation and apply in one explicit action', async ({
	page
}) => {
	await ready(page);
	await tab(page, 'World');
	const before = await identity(page);
	const applied = await page.evaluate(
		() => (window as unknown as AppWindow).__algocell.engine.settings
	);
	await page.getByLabel('Seed', { exact: true }).fill('12345');
	await page.getByLabel('Lattice width', { exact: true }).fill('64');
	await page.getByLabel('Lattice height', { exact: true }).fill('48');
	await page.getByLabel('Tape length', { exact: true }).fill('24');
	await page.getByLabel('Ring length', { exact: true }).fill('48');
	await expect(page.getByLabel('Ring length', { exact: true })).toHaveValue('48');
	await tab(page, 'Guide');
	await choose(page, 'Field view', 'zeros');
	await tab(page, 'World');
	expect(await identity(page)).toEqual(before);
	expect(
		await page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.settings)
	).toEqual(applied);
	await expect(page.getByLabel('Seed', { exact: true })).toHaveValue('12345');
	await page.getByTitle('Play (Space)', { exact: true }).click();
	await page
		.getByRole('button', { name: 'Start new world with these settings', exact: true })
		.click();
	await expect
		.poll(() =>
			page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.settings.seed)
		)
		.toBe(12345);
	const after = await identity(page);
	expect(after.resetCount).toBe(before.resetCount + 1);
	expect(after.engineId).toBe(before.engineId);
	expect(after.deviceId).toBe(before.deviceId);
	expect(
		await page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.settings)
	).toMatchObject({ seed: 12345, width: 64, height: 48, length: 24, ring: 48 });
	await keepsAdvancing(page, after);
});

test('live core and chemistry swaps preserve paused tape bytes and staged seed', async ({
	page
}) => {
	await ready(page);
	await tab(page, 'World');
	const before = await identity(page);
	const tapes = await page.evaluate(async () => {
		const e = (window as unknown as AppWindow).__algocell.engine;
		return {
			a: [...(await e.readCell(0))],
			b: [...(await e.readCell(10))],
			update: e.batchCount,
			view: { ...e.view },
			seed: e.settings.seed
		};
	});
	await page.getByLabel('Seed', { exact: true }).fill('999');
	for (const machine of ['z80-0.3', 'm6502', 'i8086', 'z80-nmos', 'z80-0.2']) {
		await choose(page, 'Processor', machine);
		expect(await identity(page)).toEqual(before);
		expect(
			await page.evaluate(async () => {
				const e = (window as unknown as AppWindow).__algocell.engine;
				return {
					a: [...(await e.readCell(0))],
					b: [...(await e.readCell(10))],
					update: e.batchCount,
					view: { ...e.view },
					seed: e.settings.seed
				};
			})
		).toEqual(tapes);
		await expect(page.getByTitle('Play (Space)', { exact: true })).toBeVisible();
	}
	await expect(page.getByLabel('Seed', { exact: true })).toHaveValue('999');
});

test('probe, encounter assay and region sampling leave the live world advancing', async ({
	page
}) => {
	await ready(page);
	await experiment(page, 'displacement-32');
	await page.getByTitle('Play (Space)', { exact: true }).click();
	const before = await identity(page);
	await probe(page, 100);
	await keepsAdvancing(page, before);
	await page.getByRole('button', { name: 'Open encounter bench', exact: true }).click();
	const bench = page.getByRole('region', { name: 'Tape inspector', exact: true });
	await expect(bench).toBeVisible();
	expect(await bench.evaluate((element) => Boolean(element.closest('.app-sidebar')))).toBe(true);
	await bench.getByRole('button', { name: 'Run culture test', exact: true }).click();
	await expect(bench.locator('.verdict')).toBeVisible();
	await keepsAdvancing(page, before);
	await tool(page, 'Region');
	await page.getByRole('button', { name: 'Measure region', exact: true }).click();
	await expect(page.locator('.measurement-list')).toBeVisible();
	await keepsAdvancing(page, before);
});

test('history browsing and checkpoint export preserve playback', async ({ page }) => {
	await ready(page, false);
	const before = await identity(page);
	await tool(page, 'History');
	await page.getByLabel('Record recent frames').check();
	await expect
		.poll(async () => Number(await page.getByLabel('History frame').getAttribute('max')))
		.toBeGreaterThan(0);
	await page.getByLabel('History frame').focus();
	await page.keyboard.press('Home');
	await expect(page.getByAltText(/Recorded lattice at update/)).toBeVisible();
	await keepsAdvancing(page, before);
	await page.getByRole('button', { name: 'Return live', exact: true }).click();
	await page.getByLabel('Record recent frames').uncheck();
	const saved = page.waitForEvent('download');
	await page
		.locator('canvas')
		.first()
		.click({ position: { x: 50, y: 50 } });
	await page.keyboard.press('ControlOrMeta+s');
	await saved;
	await keepsAdvancing(page, before);
});

test('encounter evidence retains its source across sections and clears for a new sample', async ({
	page
}) => {
	await ready(page);
	await experiment(page, 'displacement-32');
	await probe(page, 0);
	await page.getByRole('button', { name: 'Open encounter bench', exact: true }).click();
	const bench = page.getByRole('region', { name: 'Tape inspector', exact: true });
	await bench.getByRole('button', { name: 'Self-pair', exact: true }).click();
	await bench.getByRole('button', { name: 'Run comparison tests', exact: true }).click();
	await expect(bench.locator('.evidence-source')).toContainText(/cell 0/i);
	await expect(
		bench.getByRole('heading', { name: 'Example encounter', exact: true })
	).toBeVisible();
	const source = await bench.locator('.evidence-source').innerText();
	await tab(page, 'Guide');
	await tool(page, 'History');
	await page.getByRole('button', { name: 'Encounter bench', exact: true }).click();
	await expect(bench.locator('.evidence-source')).toHaveText(source, { useInnerText: true });
	await expect(
		bench.getByRole('heading', { name: 'Example encounter', exact: true })
	).toBeVisible();
	await probe(page, 1);
	await page.getByRole('button', { name: 'Open encounter bench', exact: true }).click();
	await expect(bench.locator('.evidence-source')).toHaveCount(0);
	await expect(bench.getByRole('heading', { name: 'Example encounter', exact: true })).toHaveCount(
		0
	);
	await bench.getByRole('button', { name: 'Run comparison tests', exact: true }).click();
	await expect(bench.locator('.evidence-source')).toContainText(/cell 1/i);
});

test('transport shortcuts step exactly once and leave focused fields editable', async ({
	page
}) => {
	await ready(page);
	const before = await identity(page);
	const update = await page.evaluate(
		() => (window as unknown as AppWindow).__algocell.engine.batchCount
	);
	await page
		.locator('canvas')
		.first()
		.click({ position: { x: 60, y: 60 } });
	await page.keyboard.press('.');
	await expect
		.poll(() => page.evaluate(() => (window as unknown as AppWindow).__algocell.engine.batchCount))
		.toBe(update + 1);
	await expect(page.getByTitle('Play (Space)', { exact: true })).toBeVisible();
	expect(await identity(page)).toEqual(before);
	await page.keyboard.press('Space');
	await keepsAdvancing(page, before);
	await page.keyboard.press('Space');
	await expect(page.getByTitle('Play (Space)', { exact: true })).toBeVisible();
	await tab(page, 'World');
	const seed = page.getByLabel('Seed', { exact: true });
	await seed.fill('12345');
	await seed.press('1');
	await expect(page.getByRole('tab', { name: 'World', exact: true })).toHaveAttribute(
		'aria-selected',
		'true'
	);
	await expect(page.getByTitle('Play (Space)', { exact: true })).toBeVisible();
	expect(await identity(page)).toEqual(before);
	await seed.press('Space');
	await expect(page.getByTitle('Play (Space)', { exact: true })).toBeVisible();
	await page
		.locator('canvas')
		.first()
		.click({ position: { x: 60, y: 60 } });
	await expect(page.locator('canvas').first()).toBeFocused();
	await page.keyboard.press('Space');
	await keepsAdvancing(page, before);
});

for (const viewport of [
	{ width: 1280, height: 800 },
	{ width: 1440, height: 900 },
	{ width: 1920, height: 1080 }
]) {
	test(`shell has no overlaps or horizontal scroll at ${viewport.width}×${viewport.height}`, async ({
		page
	}) => {
		await page.setViewportSize(viewport);
		await ready(page);
		for (const name of tabs) {
			await tab(page, name);
			const layout = await page.evaluate(() => {
				const rect = (selector: string) => {
					const r = document.querySelector(selector)!.getBoundingClientRect();
					return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width };
				};
				return {
					canvas: rect('.canvas-viewport'),
					sidebar: rect('.app-sidebar'),
					rail: rect('.tool-rail'),
					header: rect('.app-header'),
					status: rect('.status-bar'),
					scrollWidth: document.documentElement.scrollWidth
				};
			});
			expect(layout.scrollWidth).toBe(viewport.width);
			expect(layout.canvas.right).toBeLessThanOrEqual(layout.rail.left + 1);
			expect(layout.rail.right).toBeLessThanOrEqual(layout.sidebar.left + 1);
			expect(layout.canvas.top).toBeGreaterThanOrEqual(layout.header.bottom - 1);
			expect(layout.canvas.bottom).toBeLessThanOrEqual(layout.status.top + 1);
			expect(layout.sidebar.right).toBeLessThanOrEqual(viewport.width + 1);
			expect(layout.sidebar.width).toBeLessThanOrEqual(393);
			await expect(page.getByRole('dialog')).toHaveCount(0);
		}
	});
}

test('playback changes numeric readouts without rebuilding or mutating controls', async ({
	page
}) => {
	await ready(page);
	await tab(page, 'World');
	await page.getByTitle('Play (Space)', { exact: true }).click();
	const counts = await page.evaluate(async () => {
		const roots = ['.app-header', '.tool-rail', '.app-sidebar', '.status-bar'].map(
			(selector) => document.querySelector(selector)!
		);
		const controls = roots.flatMap((root) => [
			...root.querySelectorAll('button,input,[role=combobox]')
		]);
		let unexpected = 0,
			numeric = 0;
		const details: string[] = [];
		const observer = new MutationObserver((records) => {
			for (const record of records) {
				const el = record.target instanceof Element ? record.target : record.target.parentElement;
				if (el?.closest('[data-readout],[data-numeric],output,.numeric-readout')) {
					numeric++;
					continue;
				}
				unexpected++;
				if (details.length < 12) details.push(`${record.type}: ${el?.outerHTML.slice(0, 180)}`);
			}
		});
		for (const root of roots)
			observer.observe(root, {
				subtree: true,
				attributes: true,
				childList: true,
				characterData: true
			});
		await new Promise((resolve) => setTimeout(resolve, 1800));
		observer.disconnect();
		return {
			unexpected,
			numeric,
			details,
			stable: controls.every((control) => control.isConnected)
		};
	});
	console.log('Control mutation sample:', JSON.stringify(counts));
	await test.info().attach('control-mutations', {
		body: JSON.stringify(counts, null, 2),
		contentType: 'application/json'
	});
	expect(counts.stable).toBe(true);
	expect(counts.unexpected, JSON.stringify(counts.details)).toBeLessThanOrEqual(2);
	expect(counts.numeric).toBeGreaterThan(0);
});

test('all shell surfaces omit prohibited UI wording and use keyboard accessible listboxes', async ({
	page
}) => {
	await ready(page);
	for (const name of tabs) {
		await tab(page, name);
		const copy = await page.evaluate(
			() =>
				document.body.innerText +
				[...document.querySelectorAll('[aria-label],[title],[alt]')]
					.map((el) =>
						[el.getAttribute('aria-label'), el.getAttribute('title'), el.getAttribute('alt')].join(
							' '
						)
					)
					.join(' ')
		);
		expect(copy).not.toMatch(/\bpaper\b/i);
	}
	const view = page.getByRole('combobox', { name: 'Field view', exact: true });
	await view.focus();
	await page.keyboard.press('ArrowDown');
	await expect(page.getByRole('listbox', { name: 'Field view', exact: true })).toBeVisible();
	await page.keyboard.press('ArrowDown');
	await page.keyboard.press('Enter');
	await expect(view).toHaveAttribute('data-value', 'identity');
	await expect(view).toBeFocused();
	await expect(page.getByTitle('Play (Space)', { exact: true })).toBeVisible();
});

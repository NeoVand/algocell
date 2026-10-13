import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
await mkdir('docs/review/screenshots', { recursive: true });
const errors = [];
for (const [build, url] of [
	['main', 'http://127.0.0.1:4184'],
	['app', 'http://127.0.0.1:4185']
])
	for (const [width, height] of [
		[1440, 900],
		[390, 844]
	]) {
		const page = await browser.newPage({ viewport: { width, height } });
		page.on('pageerror', (e) => errors.push({ build, width, message: e.message }));
		await page.goto(url);
		await page.waitForFunction(() => Boolean(window.__algocell?.engine));
		await page.locator('button[title="Pause (Space)"]').waitFor();
		await page.locator('button[title="Pause (Space)"]').click();
		await page.locator('button[title="Play (Space)"]').waitFor();
		await page.evaluate(async () => {
			await window.__algocell.engine.device.queue.onSubmittedWorkDone();
			await new Promise(requestAnimationFrame);
		});
		await page.waitForTimeout(200);
		await page.locator('button[title="Reset (R)"]').click();
		await page.evaluate(() => document.fonts.ready);
		await page.waitForTimeout(300);
		await page.screenshot({ path: `docs/review/screenshots/${build}-${width}-main.png` });
		await page.locator('button[title="Settings"]').click();
		await page.waitForTimeout(100);
		await page.screenshot({ path: `docs/review/screenshots/${build}-${width}-presets.png` });
		await page.getByLabel('Close settings', { exact: true }).click();
		await page.mouse.move(width / 2, height / 2);
		await page.waitForTimeout(500);
		await page.screenshot({ path: `docs/review/screenshots/${build}-${width}-hover.png` });
		if (build === 'app') {
			await page.mouse.click(width / 2, height / 2);
			await page.waitForTimeout(300);
		}
		await page.screenshot({ path: `docs/review/screenshots/${build}-${width}-inspector.png` });
		await page.keyboard.press('Escape');
		await page.locator('button[title="Help (H)"]').click();
		await page.getByRole('button', { name: 'About', exact: true }).click();
		await page.waitForTimeout(100);
		await page.screenshot({ path: `docs/review/screenshots/${build}-${width}-about.png` });
		const tabs = await page.locator('.modal-tabs button').allTextContents();
		console.log({ build, width, tabs });
		if (build === 'main')
			await page
				.locator('.modal-tabs button')
				.nth(5)
				.evaluate((button) => button.click());
		else await page.locator('.modal-tabs button').filter({ hasText: 'Guide' }).click();
		await page.waitForTimeout(100);
		await page.screenshot({ path: `docs/review/screenshots/${build}-${width}-guide.png` });
		await page.close();
	}
await writeFile('docs/review/screenshots/errors.json', JSON.stringify(errors, null, '\t') + '\n');
await browser.close();

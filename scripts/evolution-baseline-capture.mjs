import { chromium } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';

// Serve the frozen ee67698 baseline at this URL before capturing.
const url = process.env.BASELINE_URL ?? 'http://127.0.0.1:4187';
const directory = 'docs/review/evolution';
await mkdir(directory, { recursive: true });
const errors = [];
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
try {
	for (const [width, height] of [
		[1440, 900],
		[390, 844]
	]) {
		const page = await browser.newPage({ viewport: { width, height } });
		page.on('pageerror', (error) => errors.push({ width, message: error.message }));
		await page.goto(url);
		await page.waitForFunction(
			() => Number.parseFloat(document.querySelector('.info-value')?.textContent ?? '0') > 0
		);
		await page.getByTitle('Pause (Space)', { exact: true }).click();
		await page.getByTitle('Play (Space)', { exact: true }).waitFor();
		await page.evaluate(() => document.fonts.ready);
		await page.getByTitle('Reset (R)', { exact: true }).click();
		await page.waitForTimeout(300);
		await page.screenshot({ path: `${directory}/baseline-${width}-main.png` });
		await page.getByTitle('Settings', { exact: true }).click();
		await page.screenshot({ path: `${directory}/baseline-${width}-settings.png` });
		await page.getByLabel('Close settings', { exact: true }).click();
		await page.mouse.move(width / 2, height / 2);
		await page.waitForTimeout(500);
		await page.screenshot({ path: `${directory}/baseline-${width}-hover.png` });
		await page.mouse.click(width / 2, height / 2);
		await page.getByRole('dialog', { name: 'Tape inspector', exact: true }).waitFor();
		await page.screenshot({ path: `${directory}/baseline-${width}-inspector.png` });
		await page.keyboard.press('Escape');
		await page.getByTitle('Help (H)', { exact: true }).click();
		for (const tab of ['About', 'Guide']) {
			await page.getByRole('button', { name: tab, exact: true }).click();
			await page.screenshot({ path: `${directory}/baseline-${width}-${tab.toLowerCase()}.png` });
		}
		await page.close();
		await new Promise((resolve) => setTimeout(resolve, 2000));
	}
	await writeFile(
		`${directory}/baseline-capture-errors.json`,
		JSON.stringify(errors, null, 2) + '\n'
	);
} finally {
	await browser.close();
}

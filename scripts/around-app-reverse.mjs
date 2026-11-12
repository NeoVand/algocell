// In the app: choose the neighbourhood rule, close the world (dissipation 0, mutation off), run,
// turn time round with the button, and check that the lattice comes back to where the rule began.
//   node scripts/around-app-reverse.mjs <app url> <screenshot prefix>
import { chromium } from '@playwright/test';
const [url, out] = process.argv.slice(2);
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
await page.goto(url);
await page.waitForSelector('canvas');
await page.waitForTimeout(2000);
await page.getByTitle('Pause (Space)', { exact: true }).click();
await page.locator('#interaction-rule').click();
await page.getByText('Neighbourhood (each cell rewritten from around it)', { exact: true }).click();
await page.waitForTimeout(800);
const slider = page.locator('#dissipation');
await slider.scrollIntoViewIfNeeded();
await slider.fill('0');
await slider.dispatchEvent('change');
const mutationOff = page.getByLabel('Mutation off');
await mutationOff.scrollIntoViewIfNeeded();
if (!(await mutationOff.isChecked())) await mutationOff.check();
await page.waitForTimeout(800);
const engine = () => page.evaluate(() => window.__algocell.engine.batchCount);
const lattice = () =>
	page.evaluate(async () => {
		const e = window.__algocell.engine;
		const now = await e.readSoupData();
		const before = await e.readBeforeData();
		let h = 2166136261;
		for (const a of [now, before]) for (const v of a) h = Math.imul(h ^ v, 16777619) >>> 0;
		return h;
	});
const startCount = await engine();
const startHash = await lattice();
await page.getByTitle('Play (Space)', { exact: true }).click();
await page.waitForTimeout(3000);
await page.getByTitle('Pause (Space)', { exact: true }).click();
await page.waitForTimeout(500);
const midCount = await engine();
const midHash = await lattice();
await page.screenshot({ path: `${out}_forward.png` });
await page.locator('#reverse-time').click();
await page.getByTitle('Play (Space)', { exact: true }).click();
await page.waitForTimeout(6000);
await page.getByTitle('Pause (Space)', { exact: true }).click();
await page.waitForTimeout(500);
const backCount = await engine();
await page.screenshot({ path: `${out}_back.png` });
const label = (await page.locator('#reverse-time').innerText()).trim();
await page.locator('#reverse-time').click();
await page.waitForTimeout(500);
const backHash = await lattice();
console.log(
	JSON.stringify({
		updateWhereTheRuleBegan: startCount,
		updateAfterRunningForwards: midCount,
		updateAfterRunningBackwards: backCount,
		buttonWhileBackwards: label,
		latticeChangedForwards: midHash !== startHash,
		latticeRestored: backHash === startHash,
		errors
	})
);
await browser.close();

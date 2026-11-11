// Open the app, choose the free-running world, and take three screenshots (after 1.5 s, 6.5 s and 18.5 s).
//   node scripts/free-app-look.mjs <app url> <output path prefix>
import { chromium } from '@playwright/test';
const [url, out] = process.argv.slice(2);
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
page.on('console', (m) => {
	if (m.type() === 'error') errors.push('console: ' + m.text());
});
await page.goto(url);
await page.waitForSelector('canvas');
await page.waitForTimeout(2500);
const control = page.locator('#interaction-rule');
if (!(await control.isVisible().catch(() => false))) {
	const tabs = page.getByRole('tab');
	const n = await tabs.count();
	for (let i = 0; i < n && !(await control.isVisible().catch(() => false)); i++)
		await tabs
			.nth(i)
			.click()
			.catch(() => {});
}
console.log('interaction control visible:', await control.isVisible());
await control.click();
await page.getByText('Free-running (no cells, no step budget)').click();
await page.waitForTimeout(1500);
const text = async () => (await page.locator('body').innerText()).replace(/\s+/g, ' ');
const grab = async (label) => {
	const t = await text();
	const m = t.match(/free-running/i);
	console.log(
		label,
		'| footer has free-running:',
		!!m,
		'| heads ctl:',
		await page.locator('#head-density').count(),
		'| halt ctl:',
		await page.locator('#halt-rule').count(),
		'| restart ctl:',
		await page.locator('#restart-chance').count(),
		'| lines ctl:',
		await page.locator('#head-directions').count()
	);
	await page.screenshot({ path: `${out}_${label}.png` });
};
await grab('a');
await page.waitForTimeout(5000);
await grab('b');
await page.waitForTimeout(12000);
await grab('c');
const stats = await page.evaluate(() => {
	const t = document.body.innerText;
	return t
		.split('\n')
		.filter((l) => /update|tick|world started|needs|ops|\/s/i.test(l))
		.slice(0, 12);
});
console.log(JSON.stringify(stats));
console.log('errors:', JSON.stringify(errors.slice(0, 8)));
await browser.close();

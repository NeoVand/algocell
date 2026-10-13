import { chromium } from '@playwright/test';
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const page = await browser.newPage();
page.on('console', (m) => {
	if (m.type() === 'error') console.error(m.text());
});
page.on('pageerror', (e) => console.error(e));
await page.goto('http://127.0.0.1:4185/demo/playwright');
console.log(
	await page.evaluate(async () => {
		const { GPUEngine } = await import('/src/lib/gpu/engine.ts');
		const { ORIGINAL_SETTINGS } = await import('/src/lib/gpu/machine.ts');
		const canvas = document.createElement('canvas');
		canvas.width = 300;
		canvas.height = 300;
		document.body.append(canvas);
		const engine = new GPUEngine(6);
		if (!(await engine.init(canvas))) throw new Error('No GPU');
		const results = [];
		for (const [isa, compat, variant] of [
			['z80', '0.2', 'classic'],
			['z80', '0.3', 'classic'],
			['z80', '0.3', 'zilog-nmos'],
			['m6502', '0.3', 'nmos'],
			['i8086', '0.3', 'nmos']
		]) {
			await engine.configure({
				...ORIGINAL_SETTINGS,
				isa,
				compat,
				variant,
				stack: isa === 'z80' ? 'legacy-sp' : 'first-push'
			});
			engine.simulateStep();
			await engine.waitForIdle();
			engine.render(canvas);
			await engine.setShowConfinement(true);
			engine.simulateStep();
			await engine.waitForIdle();
			engine.render(canvas);
			await engine.setShowConfinement(false);
			const culture = await engine.runCulture(new Uint8Array(16), { trials: 2, generations: 2 });
			results.push({ isa, compat, variant, updates: engine.batchCount, cultureGain: culture.gain });
		}
		engine.destroy();
		return results;
	})
);
await browser.close();

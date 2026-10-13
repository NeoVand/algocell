import { createRequire } from 'node:module';
import { readFile, writeFile } from 'node:fs/promises';
const require = createRequire(import.meta.url);
// PNG codec already bundled with the existing Playwright dev dependency.
const { PNG } = require('../node_modules/playwright-core/lib/utilsBundle.js');
const results = [];
for (const width of [1440, 390])
	for (const view of ['main', 'hover', 'presets', 'inspector', 'about', 'guide']) {
		const before = PNG.sync.read(
			await readFile(`docs/review/screenshots/main-${width}-${view}.png`)
		);
		const after = PNG.sync.read(await readFile(`docs/review/screenshots/app-${width}-${view}.png`));
		if (before.width !== after.width || before.height !== after.height)
			throw new Error('Screenshot dimensions differ');
		const diff = new PNG({ width: before.width, height: before.height });
		let changed = 0,
			error = 0;
		for (let i = 0; i < before.data.length; i += 4) {
			let largest = 0;
			for (let c = 0; c < 3; c++) {
				const d = Math.abs(before.data[i + c] - after.data[i + c]);
				error += d;
				largest = Math.max(largest, d);
				diff.data[i + c] = d;
			}
			diff.data[i + 3] = 255;
			if (largest > 12) changed++;
		}
		const pixels = before.width * before.height;
		results.push({
			width,
			view,
			changedPixelsAbove12: changed / pixels,
			meanChannelError: error / (pixels * 3)
		});
		if (view === 'main')
			await writeFile(`docs/review/screenshots/diff-${width}-main.png`, PNG.sync.write(diff));
	}
await writeFile('docs/review/visual-diff.json', JSON.stringify(results, null, 2));
console.log(results);

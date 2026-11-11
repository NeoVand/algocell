// Cold start for the reversible rule: a blank lattice at rest with one disc of random tapes in the middle.
//   node scripts/reversible-coldstart.mjs <out dir> [dissipation=0] [mutationExponent=31] [radius=12]
// Writes the lattice (now layer) at several times plus a JSON summary. Needs the dev server on 5176.
import { chromium } from '@playwright/test';
import { writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

const out = process.argv[2];
const dissipation = Number(process.argv[3] ?? 0);
const mutationExponent = Number(process.argv[4] ?? 31);
const radius = Number(process.argv[5] ?? 12);
if (!out) throw new Error('Give an output directory.');
await mkdir(out, { recursive: true });
const browser = await chromium.launch({
	channel: 'chrome',
	headless: true,
	args: ['--enable-unsafe-webgpu']
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.error('page error:', e.message));
await page.goto('http://127.0.0.1:5176/demo/playwright');
const result = await page.evaluate(
	async ({ dissipation, mutationExponent, radius }) => {
		const { GPUEngine } = await import('/src/lib/gpu/engine.ts');
		const { ORIGINAL_SETTINGS } = await import('/src/lib/gpu/machine.ts');
		const canvas = document.createElement('canvas');
		canvas.width = 64;
		canvas.height = 64;
		document.body.append(canvas);
		const engine = new GPUEngine(6);
		if (!(await engine.init(canvas))) throw new Error('No WebGPU');
		const w = 148,
			h = 135,
			L = 16;
		const settings = {
			...ORIGINAL_SETTINGS,
			width: w,
			height: h,
			seed: 77,
			interaction: 'reversible',
			dissipation,
			mutationExponent,
			boundary: 'torus'
		};
		await engine.configure(settings, { kind: 'random' });
		const start = new Uint8Array(w * h * L);
		let s = 12345;
		const rnd = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) >>> 24;
		for (let y = 0; y < h; y++)
			for (let x = 0; x < w; x++)
				if ((x - w / 2) ** 2 + (y - h / 2) ** 2 <= radius * radius)
					for (let b = 0; b < L; b++) start[(y * w + x) * L + b] = rnd();
		engine.writeSoupData(start); // also puts every cell at rest
		const b64 = (u8) => {
			let t = '';
			for (let i = 0; i < u8.length; i += 8192)
				t += String.fromCharCode.apply(null, u8.subarray(i, i + 8192));
			return btoa(t);
		};
		const marks = [0, 250, 1000, 4000, 12000];
		const rows = [];
		let u = 0;
		for (const m of marks) {
			for (; u < m; u++) {
				engine.simulateStep();
				if (u % 50 === 49) await engine.waitForIdle();
			}
			await engine.waitForIdle();
			const X = await engine.readSoupData();
			const P = await engine.readBeforeData();
			let zero = 0,
				hot = 0,
				blankCells = 0,
				restCells = 0;
			for (let c = 0; c < w * h; c++) {
				let nz = 0,
					d = 0;
				for (let b = 0; b < L; b++) {
					if (X[c * L + b]) nz++;
					else zero++;
					if (X[c * L + b] !== P[c * L + b]) d++;
				}
				hot += d;
				if (!nz) blankCells++;
				if (!d) restCells++;
			}
			rows.push({
				update: m,
				zeroBytes: zero / (w * h * L),
				hotBytes: hot / (w * h * L),
				blankCells: blankCells / (w * h),
				restCells: restCells / (w * h),
				soup: b64(X)
			});
		}
		return { settings, radius, rows };
	},
	{ dissipation, mutationExponent, radius }
);
for (const r of result.rows) {
	await writeFile(
		path.join(out, `cold_g${dissipation}_u${r.update}.u8`),
		Buffer.from(r.soup, 'base64')
	);
	delete r.soup;
	console.log(JSON.stringify(r));
}
await writeFile(path.join(out, `cold_g${dissipation}.json`), JSON.stringify(result, null, 1));
await browser.close();

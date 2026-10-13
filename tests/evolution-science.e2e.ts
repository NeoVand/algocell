import { expect, test } from '@playwright/test';

// These tests use real Zilion GPU machines. Each is a short, isolated workload.
test.beforeEach(async ({ page }) => {
	await page.goto('/demo/playwright');
});
test.afterEach(async ({ page }) => {
	await page.waitForTimeout(2000);
});

test('exact region measurements exclude padding and use inclusive normalized endpoints', async ({
	page
}) => {
	const result = await page.evaluate(async () => {
		const path = '/src/lib/observation.ts';
		const { measureRegion } = (await import(path)) as typeof import('../src/lib/observation');
		const tapes = [
			[0, 1, 2, 3, 4],
			[0, 1, 2, 3, 4],
			[0, 0, 0, 0, 0],
			[5, 6, 7, 8, 9]
		];
		const data = new Uint8Array(4 * 8).fill(255);
		tapes.forEach((tape, i) => data.set(tape, i * 8));
		const whole = measureRegion(data, 2, 2, 5, { x0: 1, y0: 1, x1: 0, y1: 0 }, 42, {
			position: 0,
			value: 0
		});
		const firstRow = measureRegion(data, 2, 2, 5, { x0: 0, y0: 0, x1: 1, y1: 0 }, 42);
		let rejected = false;
		try {
			measureRegion(data, 2, 2, 5, { x0: 0, y0: 0, x1: 2, y1: 1 }, 42);
		} catch {
			rejected = true;
		}
		return { whole, firstRow, rejected };
	});
	expect(result.whole.cellCount).toBe(4);
	expect(result.whole.uniqueTapes).toBe(3);
	expect(result.whole.topShare).toBe(0.5);
	expect(result.whole.zeroFraction).toBe(7 / 20);
	expect(result.whole.markerMatches).toBe(3);
	expect(result.whole.update).toBe(42);
	expect(result.firstRow.diversity).toBe(0.5);
	expect(result.firstRow.markerFraction).toBeNull();
	expect(result.rejected).toBe(true);
});

test('matched marker transfers distinguish short-prefix copying from full-tape retention', async ({
	page
}) => {
	const result = await page.evaluate(async () => {
		const assaysPath = '/src/lib/gpu/assays.ts',
			machinePath = '/src/lib/gpu/machine.ts';
		const { runAssays } = (await import(assaysPath)) as typeof import('../src/lib/gpu/assays');
		const { ORIGINAL_SETTINGS } = (await import(
			machinePath
		)) as typeof import('../src/lib/gpu/machine');
		const adapter = await navigator.gpu.requestAdapter();
		const device = await adapter!.requestDevice();
		const settings = {
			...ORIGINAL_SETTINGS,
			length: 32,
			ring: 64,
			stack: 'top' as const,
			mutationExponent: 31
		};
		try {
			const short = Uint8Array.from({ length: 32 }, (_, i) => [4, 0x5e, 0xed, 0xb0][i % 4]);
			const full = short.slice();
			full[0] = 0x60;
			const options = { position: 8, value: 0x7e, trials: 16, seed: 731 };
			const a = await runAssays(device, settings, short, options);
			const b = await runAssays(device, settings, full, options);
			return { a, b, full: [...full] };
		} finally {
			device.destroy();
		}
	});
	expect(result.a.marker.mutant.positive).toBe(true);
	expect(result.b.marker.mutant.positive).toBe(true);
	expect(result.a.marker.retention.map((x) => x.mutant)).toEqual([0, 0]);
	expect(result.b.marker.retention.map((x) => x.mutant)).toEqual([1, 1]);
	expect(result.b.marker.retention.map((x) => x.wildType)).toEqual([0, 0]);
	expect(result.b.selfPair.after.slice(32)).toEqual(result.full);
	expect(result.b.partnerSensitivity.changedOffspring).toBe(0);
	expect(result.b.marker.mutant.example.writes.B).toBeGreaterThan(0);
	expect(result.b.marker.mutant.example.starts.length).toBeGreaterThan(0);
	expect(result.b.manifest).toBeTruthy();
	expect(result.b.settings.length).toBe(32);
});

test('partner sensitivity reports untouched partner differences without inferring execution', async ({
	page
}) => {
	const results = await page.evaluate(async () => {
		const assaysPath = '/src/lib/gpu/assays.ts',
			machinePath = '/src/lib/gpu/machine.ts';
		const { runAssays } = (await import(assaysPath)) as typeof import('../src/lib/gpu/assays');
		const { ORIGINAL_SETTINGS } = (await import(
			machinePath
		)) as typeof import('../src/lib/gpu/machine');
		const device = await (await navigator.gpu.requestAdapter())!.requestDevice();
		try {
			const results = [];
			for (const isa of ['m6502', 'i8086'] as const) {
				const settings = {
					...ORIGINAL_SETTINGS,
					isa,
					compat: '0.3' as const,
					variant: 'nmos' as const,
					stack: 'first-push' as const,
					dialMode: 'counter' as const
				};
				const tape = new Uint8Array(16);
				tape[0] = isa === 'm6502' ? 0x02 : 0xf4;
				const result = await runAssays(device, settings, tape, {
					position: 8,
					value: 127,
					trials: 8
				});
				results.push({ sensitivity: result.partnerSensitivity, self: result.selfPair });
			}
			return results;
		} finally {
			device.destroy();
		}
	});
	for (const result of results) {
		expect(result.sensitivity.changedOffspring).toBe(8);
		expect(result.sensitivity.changedParent).toBe(0);
		expect(result.sensitivity.meanChangedFraction).toBe(1 / 16);
		expect(result.self.writes).toEqual({ A: 0, B: 0 });
		expect(result.self.fetched).toEqual([0]);
	}
});

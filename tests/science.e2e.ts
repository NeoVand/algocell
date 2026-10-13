import { expect, test } from '@playwright/test';

// Local GPU tests run in short bursts with an idle interval between them.
test.afterEach(async ({ page }) => {
	await page.waitForTimeout(2000);
});

// These run the actual package and app service in Chromium, with no CPU simulation mock.
test.beforeEach(async ({ page }) => {
	await page.goto('/demo/playwright');
	const supported = await page.evaluate(async () => !!(await navigator.gpu?.requestAdapter()));
	expect(supported, 'Scientific checks require a real WebGPU adapter').toBe(true);
});

test('historical pusher and displacement fixtures separate copy from confinement', async ({
	page
}) => {
	const result = await page.evaluate(async () => {
		const enginePath = '/src/lib/lab/engine.ts';
		const typesPath = '/src/lib/lab/types.ts';
		const { LabEngine } = (await import(enginePath)) as typeof import('../src/lib/lab/engine');
		const { DEFAULT_SETTINGS } = (await import(typesPath)) as typeof import('../src/lib/lab/types');
		const engine = await LabEngine.create({
			...DEFAULT_SETTINGS,
			compat: '0.2',
			width: 4,
			height: 4,
			pairAttempts: 8,
			mutationExponent: 31
		});
		try {
			const partner = new Uint8Array(16).fill(0x76);
			const pusher = Uint8Array.from({ length: 16 }, (_, i) => (i % 2 ? 0xc5 : 0x01));
			const short = Uint8Array.from([0x04, 0x5e, 0xed, 0xb0, ...Array(12).fill(0xaa)]);
			const full = Uint8Array.from([0x50, 0x5e, 0xed, 0xb0, ...Array(12).fill(0xaa)]);
			const p = await engine.encounter(pusher, partner);
			const r = await engine.encounter(short, partner);
			const t = await engine.encounter(full, partner);
			const culture = await engine.assay(full, { trials: 16, seed: 731, generations: 2 });
			return { p, r, t, pusher: [...pusher], full: [...full], culture };
		} finally {
			engine.destroy();
		}
	});
	expect(result.p.after.slice(23, 31)).toEqual([0xc5, 1, 0xc5, 1, 0xc5, 1, 0xc5, 1]);
	expect(result.p.after.slice(16, 23)).toEqual(Array(7).fill(0x76));
	expect(result.p.fetched.some((p) => p >= 16)).toBe(true);
	expect(result.r.after.slice(16)).toEqual(
		Array.from({ length: 16 }, (_, i) => [4, 0x5e, 0xed, 0xb0][i % 4])
	);
	expect(result.r.fetched.every((p) => p < 16)).toBe(true);
	expect(result.t.after.slice(16)).toEqual(result.full);
	expect(result.t.fetched.every((p) => p < 16)).toBe(true);
	expect(result.culture.generations).toHaveLength(2);
	expect(result.culture.generations[1].exactCopies).toBe(16);
	expect(result.culture.positive).toBe(true);
	expect(result.culture.confinement).toBe('confined');
});

test('checkpoint continuation preserves host RNG, soup and measured counts', async ({ page }) => {
	const result = await page.evaluate(async () => {
		const enginePath = '/src/lib/lab/engine.ts';
		const typesPath = '/src/lib/lab/types.ts';
		const { LabEngine } = (await import(enginePath)) as typeof import('../src/lib/lab/engine');
		const { DEFAULT_SETTINGS } = (await import(typesPath)) as typeof import('../src/lib/lab/types');
		const settings = { ...DEFAULT_SETTINGS, width: 8, height: 8, pairAttempts: 32, seed: 0 };
		const engine = await LabEngine.create(settings);
		const restored = await LabEngine.create(settings);
		try {
			await engine.step();
			const checkpoint = engine.snapshot();
			await engine.step();
			const expected = engine.snapshot();
			let mismatchRejected = false;
			const beforeRejectedRestore = restored.snapshot();
			try {
				restored.restore({ ...checkpoint, initialCondition: { kind: 'tiled', bytes: [0] } });
			} catch {
				mismatchRejected = true;
			}
			const afterRejectedRestore = restored.snapshot();
			restored.restore(checkpoint);
			await restored.step();
			const actual = restored.snapshot();
			return {
				mismatchRejected,
				rejectedPreservedState:
					JSON.stringify(beforeRejectedRestore.bytes) ===
						JSON.stringify(afterRejectedRestore.bytes) &&
					beforeRejectedRestore.rngState === afterRejectedRestore.rngState &&
					beforeRejectedRestore.update === afterRejectedRestore.update,
				expected: {
					bytes: expected.bytes,
					rng: expected.rngState,
					update: expected.update,
					encounters: expected.encounters
				},
				actual: {
					bytes: actual.bytes,
					rng: actual.rngState,
					update: actual.update,
					encounters: actual.encounters
				},
				metrics: restored.metrics
			};
		} finally {
			engine.destroy();
			restored.destroy();
		}
	});
	expect(result.actual).toEqual(result.expected);
	expect(result.mismatchRejected).toBe(true);
	expect(result.rejectedPreservedState).toBe(true);
	expect(result.metrics.update).toBe(2);
	expect(result.metrics.activePairs).toBeGreaterThan(0);
	expect(result.metrics.zeroFraction).toBeGreaterThanOrEqual(0);
	expect(result.metrics.diversity).toBeLessThanOrEqual(1);
});

test('6502 and coherent 8086 execute their own instructions and expose state', async ({ page }) => {
	const results = await page.evaluate(async () => {
		const enginePath = '/src/lib/lab/engine.ts';
		const typesPath = '/src/lib/lab/types.ts';
		const { LabEngine } = (await import(enginePath)) as typeof import('../src/lib/lab/engine');
		const { DEFAULT_SETTINGS } = (await import(typesPath)) as typeof import('../src/lib/lab/types');
		const output = [];
		for (const isa of ['m6502', 'i8086'] as const) {
			const engine = await LabEngine.create({
				...DEFAULT_SETTINGS,
				isa,
				variant: 'nmos',
				stack: 'first-push',
				width: 4,
				height: 4,
				pairAttempts: 8,
				steps: 1
			});
			try {
				const tape = new Uint8Array(16);
				tape.set(isa === 'm6502' ? [0xa9, 0x2a] : [0xb8, 0x34, 0x12]);
				output.push({ isa, result: await engine.encounter(tape, new Uint8Array(16)) });
			} finally {
				engine.destroy();
			}
		}
		return output;
	});
	expect(results[0].result.state.a).toBe(42);
	expect(results[0].result.starts).toEqual([0]);
	expect(results[0].result.fetched).toEqual([0, 1]);
	expect(results[1].result.state.ax).toBe(0x1234);
	expect(results[1].result.state.sp).toBe(1);
	expect(results[1].result.fetched).toEqual([0, 1, 2]);
});

test('compat 0.2 matches frozen facade and chunked worlds preserve lane results', async ({
	page
}) => {
	const result = await page.evaluate(async () => {
		const packagePath = '/node_modules/@neovand/zilion/dist/index.js';
		const servicePath = '/src/lib/lab/engine.ts';
		const { createEngine, Zilion } = (await import(
			packagePath
		)) as typeof import('@neovand/zilion');
		const { machineSpec, DEFAULT_SETTINGS } = (await import(
			servicePath
		)) as typeof import('../src/lib/lab/engine');
		const engine = await createEngine();
		const modern = await engine.machine(machineSpec({ ...DEFAULT_SETTINGS, compat: '0.2' }));
		const legacy = await Zilion.create({ device: engine.device, memBytes: 32 });
		try {
			const programs = Array.from({ length: 32 }, (_, lane) =>
				Uint8Array.from({ length: 32 }, (_, byte) => (lane * 29 + byte * 17 + lane * byte) & 255)
			);
			const oldResult = await legacy.run(programs, { steps: 128 });
			const current = await modern.run(programs, { seed: 991 });
			const chunked = await modern.run(programs, { seed: 991, maxLanesPerChunk: 3 });
			return {
				oldBytes: [...oldResult.memory],
				bytes: [...current.memory!],
				chunkBytes: [...chunked.memory!],
				record: [...current.record],
				chunkRecord: [...chunked.record],
				oldRegisters: oldResult.registers.map((r) => [
					r.af,
					r.bc,
					r.de,
					r.hl,
					r.ix,
					r.iy,
					r.sp,
					r.pc
				]),
				registers: programs.map((_, i) => {
					const r = current.stateOf(i);
					return [
						(r.a << 8) | r.f,
						(r.b << 8) | r.c,
						(r.d << 8) | r.e,
						(r.h << 8) | r.l,
						r.ix,
						r.iy,
						r.sp,
						r.pc
					];
				})
			};
		} finally {
			legacy.destroy();
			modern.destroy();
			engine.destroy();
		}
	});
	expect(result.bytes).toEqual(result.oldBytes);
	expect(result.registers).toEqual(result.oldRegisters);
	expect(result.chunkBytes).toEqual(result.bytes);
	expect(result.chunkRecord).toEqual(result.record);
});

test('lethal chemistry acts on instruction starts and ablation keeps its page identity', async ({
	page
}) => {
	const result = await page.evaluate(async () => {
		const path = '/src/lib/lab/engine.ts';
		const { LabEngine, DEFAULT_SETTINGS } = (await import(
			path
		)) as typeof import('../src/lib/lab/engine');
		const settings = {
			...DEFAULT_SETTINGS,
			width: 4,
			height: 4,
			pairAttempts: 8,
			lethalProbability: 1
		};
		const engine = await LabEngine.create(settings);
		try {
			const operand = new Uint8Array(16);
			operand.set([0x3e, 0, 0x76]);
			const zero = await engine.encounter(new Uint8Array(16), new Uint8Array(16));
			const data = await engine.encounter(operand, new Uint8Array(16));
			const ablationPath = '/src/lib/lab/ablation.ts';
			const { ablationMasks } = (await import(
				ablationPath
			)) as typeof import('../src/lib/lab/ablation');
			const mask = ablationMasks('z80', ['family:block-copy']);
			return { zero, data, words: [...mask] };
		} finally {
			engine.destroy();
		}
	});
	// Zilion counts the halted instruction attempt as one step.
	expect(result.zero.steps).toBe(1);
	expect(result.zero.writes).toEqual({ A: 0, B: 0 });
	expect(result.zero.starts).toEqual([0]);
	expect(result.data.state.a).toBe(0);
	expect(result.data.starts).toEqual([0, 2]);
	expect(result.words.slice(0, 16).every((word) => word === 0)).toBe(true);
	for (const code of [0xa0, 0xa8, 0xb0, 0xb8])
		expect((result.words[16 + (code >> 5)] >>> (code & 31)) & 1).toBe(1);
});

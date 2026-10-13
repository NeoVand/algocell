import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
	await page.route('**/favicon.ico', (route) => route.fulfill({ status: 204 }));
	await page.goto('/demo/playwright');
});

test.afterEach(async ({ page }) => {
	await page.close();
	await new Promise((resolve) => setTimeout(resolve, 2000));
});

test('live core, chemistry and field changes preserve the resident world and camera', async ({
	page
}) => {
	const errors: string[] = [];
	page.on('console', (message) => {
		if (message.type() === 'error' || /Error while parsing WGSL|is invalid/.test(message.text()))
			errors.push(message.text());
	});
	const result = await page.evaluate(async () => {
		const { GPUEngine } = (await import(
			String('/src/lib/gpu/engine.ts')
		)) as typeof import('../src/lib/gpu/engine');
		const canvas = document.createElement('canvas');
		canvas.width = 480;
		canvas.height = 300;
		const engine = new GPUEngine(6, { width: 8, height: 8, gridType: 'square' });
		try {
			if (!(await engine.init(canvas))) throw new Error('WebGPU unavailable');
			await engine.configure({ ...engine.settings, pairAttempts: 16, steps: 32 });
			engine.simulateStep();
			await engine.waitForIdle();
			engine.zoomAt(240, 150, 480, 300, 0.7);
			engine.pan(12, -8, 480, 300);
			const host = engine as unknown as {
				device: GPUDevice;
				soupBuffer: GPUBuffer;
				cpuRng: { state: bigint };
			};
			const device = host.device,
				buffer = host.soupBuffer,
				rng = host.cpuRng;
			const before = {
				bytes: Array.from(await engine.readSoupData()),
				camera: { ...engine.view },
				update: engine.batchCount,
				rngState: rng.state.toString(),
				diagnostics: engine.diagnostics
			};
			const base = engine.settings;
			const contracts = [
				{ isa: 'z80', compat: '0.3', variant: 'classic' },
				{ isa: 'z80', compat: '0.3', variant: 'zilog-nmos' },
				{ isa: 'm6502', compat: '0.3', variant: 'nmos' },
				{ isa: 'i8086', compat: '0.3', variant: 'nmos' },
				{ isa: 'z80', compat: '0.2', variant: 'classic' }
			] as const;
			for (const contract of contracts)
				await engine.setMachine({
					...base,
					...contract,
					stack: contract.isa === 'z80' ? 'legacy-sp' : 'first-push',
					dialMode: contract.compat === '0.2' ? 'historical-pcg' : 'counter'
				});
			// Concurrent requests retain event order without overlapping pipeline commits.
			await Promise.all([
				engine.setMachine({ ...engine.settings, lethalProbability: 0.2 }),
				engine.setMachine({ ...engine.settings, lethalProbability: 0.7 }),
				engine.setFieldMode('confinement'),
				engine.setFieldMode('identity')
			]);
			for (const mode of ['bytes', 'zeros', 'diversity', 'marker'] as const)
				await engine.setFieldMode(mode);
			const after = {
				bytes: Array.from(await engine.readSoupData()),
				camera: { ...engine.view },
				update: engine.batchCount,
				rngState: host.cpuRng.state.toString(),
				diagnostics: engine.diagnostics
			};
			engine.noiseExp = 8;
			engine.z80Steps = 16;
			engine.pairCount = 8;
			const liveParameters = engine.settings;
			engine.simulateStep();
			await engine.waitForIdle();
			return {
				before,
				after,
				identitiesPreserved:
					host.device === device && host.soupBuffer === buffer && host.cpuRng === rng,
				chemistry: engine.settings.lethalProbability,
				liveParameters,
				continuedUpdate: engine.batchCount
			};
		} finally {
			engine.destroy();
		}
	});
	expect(errors).toEqual([]);
	expect(result.identitiesPreserved).toBe(true);
	expect(result.after.bytes).toEqual(result.before.bytes);
	expect(result.after.camera).toEqual(result.before.camera);
	expect(result.after.rngState).toBe(result.before.rngState);
	expect(result.after.update).toBe(result.before.update);
	for (const key of [
		'engineId',
		'deviceId',
		'deviceInitializations',
		'resetCount',
		'bufferRevision'
	] as const)
		expect(result.after.diagnostics[key], key).toBe(result.before.diagnostics[key]);
	expect(result.after.diagnostics.pipelineRevision).toBeGreaterThan(
		result.before.diagnostics.pipelineRevision
	);
	expect(result.chemistry).toBe(0.7);
	expect(result.liveParameters).toMatchObject({
		mutationExponent: 8,
		steps: 16,
		pairAttempts: 8
	});
	expect(result.continuedUpdate).toBe(result.before.update + 1);
});

test('live marker patches preserve adjacent bytes without population or tape readback', async ({
	page
}) => {
	const errors: string[] = [];
	page.on('console', (message) => {
		if (message.type() === 'error' || /Error while parsing WGSL|is invalid/.test(message.text()))
			errors.push(message.text());
	});
	const result = await page.evaluate(async () => {
		const { GPUEngine } = (await import(
			String('/src/lib/gpu/engine.ts')
		)) as typeof import('../src/lib/gpu/engine');
		const canvas = document.createElement('canvas');
		canvas.width = canvas.height = 300;
		const engine = new GPUEngine(6, {
			width: 8,
			height: 8,
			gridType: 'hex',
			tapeLength: 19,
			pairLength: 38
		});
		try {
			if (!(await engine.init(canvas))) throw new Error('WebGPU unavailable');
			await engine.configure(
				{ ...engine.settings, pairAttempts: 16, steps: 8, mutationExponent: 31 },
				{ kind: 'tiled', bytes: [0x76] }
			);
			let reads = 0;
			const readCell = engine.readCell.bind(engine);
			engine.readCell = (cell) => {
				reads++;
				return readCell(cell);
			};
			const before = Array.from(await engine.readSoupData());
			const diagnostics = engine.diagnostics;
			const pending = [];
			for (let position = 1; position < 4; position++) {
				engine.simulateStep();
				pending.push(engine.plantMarker(7, position, position * 17));
			}
			pending.push(engine.plantMarker(7, 18, 255));
			await Promise.all(pending);
			const after = Array.from(await engine.readSoupData());
			const readbacksDuringPlant = reads;
			return {
				before,
				after,
				tape: Array.from(await readCell(7)),
				readbacksDuringPlant,
				update: engine.batchCount,
				diagnostics,
				afterDiagnostics: engine.diagnostics
			};
		} finally {
			engine.destroy();
		}
	});
	const expected = result.before.slice();
	for (let position = 1; position < 4; position++) expected[7 * 20 + position] = position * 17;
	expected[7 * 20 + 18] = 255;
	expect(result.after).toEqual(expected);
	expect(result.tape).toHaveLength(19);
	expect(result.after[7 * 20 + 19]).toBe(0);
	expect(result.readbacksDuringPlant).toBe(0);
	expect(result.update).toBe(3);
	expect(result.afterDiagnostics).toEqual(result.diagnostics);
	expect(errors).toEqual([]);
});

test('explicit new world drains existing readbacks and releases later reads onto the new condition', async ({
	page
}) => {
	const errors: string[] = [];
	page.on('console', (message) => {
		if (message.type() === 'error' || /Error while parsing WGSL|is invalid/.test(message.text()))
			errors.push(message.text());
	});
	const result = await page.evaluate(async () => {
		const { GPUEngine } = (await import(
			String('/src/lib/gpu/engine.ts')
		)) as typeof import('../src/lib/gpu/engine');
		const canvas = document.createElement('canvas');
		canvas.width = canvas.height = 300;
		const engine = new GPUEngine(6, { width: 8, height: 8, gridType: 'square' });
		try {
			if (!(await engine.init(canvas))) throw new Error('WebGPU unavailable');
			await engine.configure(
				{ ...engine.settings, pairAttempts: 16, steps: 8, mutationExponent: 31 },
				{ kind: 'tiled', bytes: [0x76] }
			);
			engine.simulateStep();
			const before = engine.diagnostics;
			const earlier = Promise.all([
				engine.readStats(),
				engine.readCellHashes(),
				engine.readSoupData(),
				engine.readCell(7)
			]);
			const applying = engine.configure(
				{ ...engine.settings, width: 9, height: 7, length: 19, ring: 38, seed: 17 },
				{ kind: 'tiled', bytes: [9, 8, 7] }
			);
			// Let the configuration transaction install its readback barrier.
			await Promise.resolve();
			const later = engine.readCell(7);
			const [stats, hashes, soup, oldCell] = await earlier;
			await applying;
			return {
				before,
				after: engine.diagnostics,
				statsTotal: stats.reduce((sum, n) => sum + n, 0),
				hashes: hashes.length,
				soup: soup.length,
				oldCell: Array.from(oldCell),
				newCell: Array.from(await later),
				settings: engine.settings,
				update: engine.batchCount
			};
		} finally {
			engine.destroy();
		}
	});
	expect(result.statsTotal).toBe(8 * 8 * 16);
	expect(result.hashes).toBe(8 * 8);
	expect(result.soup).toBe(8 * 8 * 16);
	expect(result.oldCell).toEqual(Array(16).fill(0x76));
	expect(result.newCell).toEqual(Array.from({ length: 19 }, (_, i) => [9, 8, 7][i % 3]));
	expect(result.after.engineId).toBe(result.before.engineId);
	expect(result.after.deviceId).toBe(result.before.deviceId);
	expect(result.after.resetCount).toBe(result.before.resetCount + 1);
	expect(result.after.bufferRevision).toBe(result.before.bufferRevision + 1);
	expect(result.update).toBe(0);
	expect(result.settings).toMatchObject({ width: 9, height: 7, length: 19, ring: 38, seed: 17 });
	expect(errors).toEqual([]);
});

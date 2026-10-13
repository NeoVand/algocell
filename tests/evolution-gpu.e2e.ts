import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
	await page.route('**/favicon.ico', (route) => route.fulfill({ status: 204 }));
	await page.goto('/demo/playwright');
});
test.afterEach(async ({ page }) => {
	await page.waitForTimeout(2000);
});

test('GPU field reductions measure exact local tapes and exclude word padding', async ({
	page
}) => {
	const errors: string[] = [];
	page.on('console', (message) => {
		if (message.type() === 'error') errors.push(message.text());
	});
	const result = await page.evaluate(async () => {
		const { createFieldShader } = (await import(
			String('/src/lib/gpu/shaders.ts')
		)) as typeof import('../src/lib/gpu/shaders');
		const device = await (await navigator.gpu.requestAdapter())!.requestDevice();
		const buffers: GPUBuffer[] = [];
		const buffer = (size: number, usage: number) => {
			const value = device.createBuffer({ size, usage });
			buffers.push(value);
			return value;
		};
		try {
			const module = device.createShaderModule({ code: createFieldShader() });
			const compilation = await module.getCompilationInfo();
			const pipeline = await device.createComputePipelineAsync({
				layout: 'auto',
				compute: { module, entryPoint: 'measure_field' }
			});
			const soup = buffer(9 * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
			const params = buffer(80, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
			const field = buffer(9 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
			const staging = buffer(9 * 4, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
			const bytes = new Uint8Array(9 * 8);
			for (let cell = 0; cell < 9; cell++) {
				bytes.set(cell === 8 ? [7, 0, 0, 0, 0] : [0, 2, 3, 4, 5], cell * 8);
				// Deliberately different padding must never create additional identities.
				bytes.fill(cell + 17, cell * 8 + 5, cell * 8 + 8);
			}
			device.queue.writeBuffer(soup, 0, bytes);
			const bind = device.createBindGroup({
				layout: pipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: soup } },
					{ binding: 1, resource: { buffer: params } },
					{ binding: 2, resource: { buffer: field } }
				]
			});
			const values = new Uint32Array(20);
			values[0] = values[1] = 3;
			values[10] = 5;
			values[18] = 7;
			values[19] = 1;
			const outputs: Record<number, number[]> = {};
			for (const mode of [1, 3, 4, 5]) {
				values[16] = mode;
				device.queue.writeBuffer(params, 0, values);
				const encoder = device.createCommandEncoder();
				const pass = encoder.beginComputePass();
				pass.setPipeline(pipeline);
				pass.setBindGroup(0, bind);
				pass.dispatchWorkgroups(1);
				pass.end();
				encoder.copyBufferToBuffer(field, 0, staging, 0, 9 * 4);
				device.queue.submit([encoder.finish()]);
				await staging.mapAsync(GPUMapMode.READ);
				const raw = staging.getMappedRange().slice(0);
				outputs[mode] = Array.from(
					mode === 3 || mode === 4 ? new Float32Array(raw) : new Uint32Array(raw)
				);
				staging.unmap();
			}
			return {
				outputs,
				compilationErrors: compilation.messages
					.filter((message) => message.type === 'error')
					.map((message) => message.message)
			};
		} finally {
			for (const buffer of buffers) buffer.destroy();
			device.destroy();
		}
	});
	expect(result.compilationErrors).toEqual([]);
	expect(errors).toEqual([]);
	expect(new Set(result.outputs[1].slice(0, 8)).size).toBe(1);
	expect(result.outputs[1][8]).not.toBe(result.outputs[1][0]);
	expect(result.outputs[3][0]).toBeCloseTo(0.2);
	expect(result.outputs[3][8]).toBeCloseTo(0.8);
	expect(result.outputs[4][0]).toBeCloseTo(1 / 4);
	expect(result.outputs[4][4]).toBeCloseTo(2 / 9);
	expect(result.outputs[4][8]).toBeCloseTo(2 / 4);
	expect(result.outputs[5]).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 1]);
});

test('field switching preserves population and camera; probe and marker sample one tape', async ({
	page
}) => {
	const errors: string[] = [];
	page.on('pageerror', (error) => errors.push(error.message));
	page.on('console', (message) => {
		if (message.type() === 'error') errors.push(message.text());
	});
	const result = await page.evaluate(async () => {
		const { GPUEngine } = (await import(
			String('/src/lib/gpu/engine.ts')
		)) as typeof import('../src/lib/gpu/engine');
		const canvas = document.createElement('canvas');
		canvas.width = 480;
		canvas.height = 300;
		document.body.append(canvas);
		const engine = new GPUEngine(6, {
			width: 12,
			height: 10,
			gridType: 'square',
			tapeLength: 16,
			pairLength: 32
		});
		try {
			if (!(await engine.init(canvas))) throw new Error('WebGPU unavailable');
			await engine.configure(
				{ ...engine.settings, pairAttempts: 64, mutationExponent: 31 },
				{ kind: 'tiled', bytes: [1, 2, 3, 4] }
			);
			engine.zoomAt(240, 150, 480, 300, 0.7);
			const before = Array.from(await engine.readSoupData());
			const camera = { ...engine.view };
			const modes = ['bytes', 'identity', 'confinement', 'zeros', 'diversity', 'marker'] as const;
			for (const mode of modes) {
				await engine.setFieldMode(mode);
				engine.render(canvas);
				await engine.waitForIdle();
			}
			const afterModes = Array.from(await engine.readSoupData());
			const sampled = Array.from(await engine.readCell(3));
			await engine.plantMarker(3, 4, 127);
			const marked = Array.from(await engine.readCell(3));
			const region = await engine.readRegion({ x0: 0, y0: 0, x1: 11, y1: 9 });
			const bounds = engine.cellBounds(3, 480, 300);
			const projected = engine.screenToCell(
				bounds.x + bounds.width / 2,
				bounds.y + bounds.height / 2,
				480,
				300
			);
			const square = {
				sameBytes:
					before.length === afterModes.length &&
					before.every((byte, index) => byte === afterModes[index]),
				camera,
				afterCamera: { ...engine.view },
				update: engine.batchCount,
				sampled,
				marked,
				region,
				projected
			};
			await engine.configure({ ...engine.settings, topology: 'hex', length: 19, ring: 38 });
			for (const mode of modes) {
				await engine.setFieldMode(mode);
				engine.render(canvas);
				await engine.waitForIdle();
			}
			const hex = await engine.readCell(3);
			const hexBounds = engine.cellBounds(3, 480, 300);
			return {
				square,
				hexLength: hex.length,
				hexProjected: engine.screenToCell(
					hexBounds.x + hexBounds.width / 2,
					hexBounds.y + hexBounds.height / 2,
					480,
					300
				)
			};
		} finally {
			engine.destroy();
			canvas.remove();
		}
	});
	expect(errors).toEqual([]);
	expect(result.square.sameBytes).toBe(true);
	expect(result.square.afterCamera).toEqual(result.square.camera);
	expect(result.square.update).toBe(0);
	expect(result.square.sampled).toHaveLength(16);
	expect(result.square.marked[4]).toBe(127);
	expect(result.square.marked.filter((_, index) => index !== 4)).toEqual(
		result.square.sampled.filter((_, index) => index !== 4)
	);
	expect(result.square.region.uniqueTapes).toBe(2);
	expect(result.square.region.markerMatches).toBe(1);
	expect(result.square.projected).toBe(3);
	expect(result.hexLength).toBe(19);
	expect(result.hexProjected).toBe(3);
});

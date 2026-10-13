import { expect, test } from '@playwright/test';

test('embedded soup libraries match standalone machines across ISAs and world interventions', async ({
	page
}) => {
	await page.goto('/');
	await expect(page.getByTitle('Pause (Space)', { exact: true })).toBeVisible();
	await page.getByTitle('Pause (Space)', { exact: true }).click();
	await page.goto('/demo/playwright');
	const results = await page.evaluate(async () => {
		const { createSimShader } = (await import(
			String('/src/lib/gpu/shaders.ts')
		)) as typeof import('../src/lib/gpu/shaders');
		const { ORIGINAL_SETTINGS, machineSpec } = (await import(
			String('/src/lib/gpu/machine.ts')
		)) as typeof import('../src/lib/gpu/machine');
		const { ablationMasks } = (await import(
			String('/src/lib/lab/ablation.ts')
		)) as typeof import('../src/lib/lab/ablation');
		const { createEngine } = (await import(
			String('/node_modules/@neovand/zilion/dist/index.js')
		)) as typeof import('@neovand/zilion');
		const runner = await createEngine();
		const device = runner.device;
		const cases = [
			{},
			{ compat: '0.3' as const },
			{ compat: '0.3' as const, variant: 'zilog-nmos' as const },
			{
				isa: 'm6502' as const,
				compat: '0.3' as const,
				variant: 'nmos' as const,
				stack: 'first-push' as const,
				dialMode: 'counter' as const
			},
			{
				isa: 'i8086' as const,
				compat: '0.3' as const,
				variant: 'nmos' as const,
				stack: 'first-push' as const,
				dialMode: 'counter' as const
			},
			{ length: 12, ring: 28, stack: 'top' as const },
			{ lethalProbability: 1 },
			{ lethalProbability: 0.03 },
			{ randomRegisters: true },
			{ ablation: ['family:block-copy'] },
			{
				isa: 'm6502' as const,
				compat: '0.3' as const,
				variant: 'nmos' as const,
				stack: 'first-push' as const,
				dialMode: 'counter' as const,
				lethalProbability: 0.3,
				ablation: ['opcode:A9']
			},
			{
				isa: 'i8086' as const,
				compat: '0.3' as const,
				variant: 'nmos' as const,
				stack: 'first-push' as const,
				dialMode: 'counter' as const,
				ablation: ['opcode:A4']
			}
		];
		const output = [];
		try {
			for (const overrides of cases) {
				const settings = { ...ORIGINAL_SETTINGS, ...overrides };
				const L = settings.length;
				const n = 64;
				const lanes = Array.from({ length: n }, (_, i) =>
					Uint8Array.from({ length: 2 * L }, (_, b) => (i * 29 + b * 17 + i * b) & 255)
				);
				const wordsPerTape = Math.ceil(L / 4);
				const bytes = new Uint8Array(n * wordsPerTape * 8);
				for (let i = 0; i < n; i++)
					for (let side = 0; side < 2; side++)
						bytes.set(
							lanes[i].subarray(side * L, (side + 1) * L),
							(i * 2 + side) * wordsPerTape * 4
						);
				const buffers: GPUBuffer[] = [];
				const buffer = (size: number, usage: number) => {
					const b = device.createBuffer({ size, usage });
					buffers.push(b);
					return b;
				};
				const data = buffer(
					bytes.length,
					GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
				);
				const mapWords = Math.ceil(settings.ring / 32);
				const stride = 2 + mapWords;
				const maps = buffer(n * stride * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
				const active = buffer(n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
				const params = buffer(128, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
				const values = new Uint32Array(32);
				values.set([200, 200, L, settings.ring, n, 0, 128, 991]);
				values.set(ablationMasks(settings.isa, settings.ablation), 8);
				device.queue.writeBuffer(data, 0, bytes);
				device.queue.writeBuffer(params, 0, values);
				device.queue.writeBuffer(active, 0, new Uint32Array(n).fill(1));
				const module = device.createShaderModule({
					code: createSimShader('square', settings, true)
				});
				const pipeline = await device.createComputePipelineAsync({
					layout: 'auto',
					compute: { module, entryPoint: 'z80_execute_batch' }
				});
				const bind = device.createBindGroup({
					layout: pipeline.getBindGroupLayout(0),
					entries: [
						{ binding: 2, resource: { buffer: data } },
						{ binding: 3, resource: { buffer: maps } },
						{ binding: 5, resource: { buffer: params } },
						{ binding: 6, resource: { buffer: active } }
					]
				});
				const read = buffer(
					bytes.length + n * stride * 4,
					GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
				);
				const encoder = device.createCommandEncoder();
				const pass = encoder.beginComputePass();
				pass.setPipeline(pipeline);
				pass.setBindGroup(0, bind);
				pass.dispatchWorkgroups(1);
				pass.end();
				encoder.copyBufferToBuffer(data, 0, read, 0, bytes.length);
				encoder.copyBufferToBuffer(maps, 0, read, bytes.length, n * stride * 4);
				device.queue.submit([encoder.finish()]);
				await read.mapAsync(GPUMapMode.READ);
				const raw = read.getMappedRange();
				const actual = new Uint8Array(raw.slice(0, bytes.length));
				const fetched = new Uint32Array(raw.slice(bytes.length));
				read.unmap();
				const reference = await runner.machine(machineSpec(settings));
				const expected = await reference.run(lanes, {
					seed: 991,
					ablation: ablationMasks(settings.isa, settings.ablation)
				});
				let memoryEqual = true,
					mapsEqual = true;
				for (let i = 0; i < n; i++) {
					const memory = expected.memoryOf(i);
					for (let side = 0; side < 2; side++)
						for (let b = 0; b < L; b++)
							if (actual[(i * 2 + side) * wordsPerTape * 4 + b] !== memory[side * L + b])
								memoryEqual = false;
					const bits = new Uint32Array(mapWords);
					for (const pos of expected.fetchedOf(i)) bits[pos >> 5] |= 1 << (pos & 31);
					for (let f = 0; f < mapWords; f++)
						if (fetched[i * stride + 2 + f] !== bits[f]) mapsEqual = false;
				}
				output.push({ settings, memoryEqual, mapsEqual });
				reference.destroy();
				for (const b of buffers) b.destroy();
			}
		} finally {
			runner.destroy();
		}
		return output;
	});
	for (const result of results) {
		expect(result.memoryEqual, JSON.stringify(result.settings)).toBe(true);
		expect(result.mapsEqual, JSON.stringify(result.settings)).toBe(true);
	}
});

test('GPU confinement treats volatile padding as outside the focal tape', async ({ page }) => {
	await page.goto('/');
	await expect(page.getByTitle('Pause (Space)', { exact: true })).toBeVisible();
	await page.getByTitle('Pause (Space)', { exact: true }).click();
	await page.goto('/demo/playwright');
	const statuses = await page.evaluate(async () => {
		const { GPUEngine } = (await import(
			String('/src/lib/gpu/engine.ts')
		)) as typeof import('../src/lib/gpu/engine');
		const { ORIGINAL_SETTINGS } = (await import(
			String('/src/lib/gpu/machine.ts')
		)) as typeof import('../src/lib/gpu/machine');
		const canvas = document.createElement('canvas');
		canvas.width = 300;
		canvas.height = 300;
		const engine = new GPUEngine(6);
		await engine.init(canvas);
		try {
			// JP24 executes four padding NOPs before wrapping to JP again. No B fetch.
			await engine.configure(
				{
					...ORIGINAL_SETTINGS,
					width: 32,
					height: 32,
					length: 12,
					ring: 28,
					pairAttempts: 512,
					mutationExponent: 31
				},
				{ kind: 'tiled', bytes: [0xc3, 24, 0, ...Array(9).fill(0x76)] }
			);
			await engine.setShowConfinement(true);
			engine.simulateStep();
			await engine.waitForIdle();
			const host = engine as unknown as { device: GPUDevice; fetchStatusBuffer: GPUBuffer };
			const staging = host.device.createBuffer({
				size: 32 * 32 * 4,
				usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
			});
			// Read through a GPU pass: the live view buffer intentionally has no COPY_SRC usage.
			const output = host.device.createBuffer({
				size: 32 * 32 * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			});
			const pipeline = await host.device.createComputePipelineAsync({
				layout: 'auto',
				compute: {
					module: host.device.createShaderModule({
						code: `
          @group(0) @binding(0) var<storage, read> status: array<u32>;
          @group(0) @binding(1) var<storage, read_write> result: array<u32>;
          @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
            if (id.x < arrayLength(&status)) { result[id.x] = status[id.x]; }
          }`
					}),
					entryPoint: 'main'
				}
			});
			const bind = host.device.createBindGroup({
				layout: pipeline.getBindGroupLayout(0),
				entries: [
					{ binding: 0, resource: { buffer: host.fetchStatusBuffer } },
					{ binding: 1, resource: { buffer: output } }
				]
			});
			const encoder = host.device.createCommandEncoder();
			const pass = encoder.beginComputePass();
			pass.setPipeline(pipeline);
			pass.setBindGroup(0, bind);
			pass.dispatchWorkgroups(16);
			pass.end();
			encoder.copyBufferToBuffer(output, 0, staging, 0, 32 * 32 * 4);
			host.device.queue.submit([encoder.finish()]);
			await staging.mapAsync(GPUMapMode.READ);
			const values = [...new Set(new Uint32Array(staging.getMappedRange()))];
			staging.unmap();
			staging.destroy();
			output.destroy();
			return values;
		} finally {
			engine.destroy();
		}
	});
	expect(statuses).toContain(2);
	expect(statuses).not.toContain(1);
});

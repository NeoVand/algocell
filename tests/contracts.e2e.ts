import { expect, test } from '@playwright/test';

// These inspect the actual browser modules and generated manifests without acquiring a GPU.
test.beforeEach(async ({ page }) => {
	await page.goto('/demo/playwright');
});

test('paper catalogue preserves source geometry, reset choices and labelled demonstrations', async ({
	page
}) => {
	const entries = await page.evaluate(async () => {
		const path = '/src/lib/lab/presets.ts';
		const { PAPER_PRESETS } = (await import(path)) as typeof import('../src/lib/lab/presets');
		return PAPER_PRESETS;
	});
	const byId = Object.fromEntries(entries.map((entry) => [entry.id, entry]));
	expect(entries).toHaveLength(22);
	expect(new Set(entries.map((entry) => entry.id)).size).toBe(entries.length);
	for (const entry of entries) {
		expect(entry.config.width, entry.id).toBe(160);
		expect(entry.config.height, entry.id).toBe(125);
		expect(entry.config.pairAttempts, entry.id).toBe(8192);
		expect(entry.config.steps, entry.id).toBe(128);
		expect(entry.config.topology, entry.id).toBe('square');
		expect(entry.config.ring, entry.id).toBeGreaterThanOrEqual(2 * entry.config.length);
		if (entry.published.horizonUnit === 'transfers') {
			expect(entry.published, entry.id).toMatchObject({
				width: null,
				height: null,
				pairAttempts: null,
				mutationAttempts: null,
				horizon: 8
			});
		} else {
			expect(entry.published, entry.id).toMatchObject({
				width: 160,
				height: 125,
				pairAttempts: 8192,
				mutationAttempts: 512
			});
		}
		expect(entry.source.registration, entry.id).toContain('registrations/originals/');
		expect(entry.source.results.length, entry.id).toBeGreaterThan(0);
		for (const text of [entry.description, entry.whatToLookFor, entry.finding, entry.limits]) {
			expect(text.length, entry.id).toBeGreaterThan(30);
		}
	}
	expect(byId['stage-g-l16'].config).toMatchObject({
		isa: 'z80',
		compat: '0.2',
		length: 16,
		ring: 32,
		seed: 2001,
		stack: 'top',
		mutationExponent: 4,
		lethalProbability: 0,
		randomRegisters: false
	});
	expect(byId['stage-g-l16'].published.seeds).toEqual(
		Array.from({ length: 20 }, (_, i) => 2001 + i)
	);
	expect(byId['lethal-zero'].config).toMatchObject({ seed: 4001, lethalProbability: 1 });
	for (const probability of [0, 0.01, 0.03, 0.1, 0.3, 1]) {
		expect(byId[`dial-${probability}`].config).toMatchObject({
			seed: 9001,
			lethalProbability: probability,
			dialMode: 'historical-pcg',
			compat: '0.2'
		});
		expect(byId[`dial-${probability}`].published.seeds).toEqual(
			Array.from({ length: 10 }, (_, i) => 9001 + i)
		);
	}
	expect(byId['random-registers'].config).toMatchObject({
		seed: 8001,
		randomRegisters: true,
		stack: 'top'
	});
	for (const ring of [24, 28]) {
		expect(byId[`ring-l12-p${ring}`].config).toMatchObject({
			length: 12,
			ring,
			seed: 101,
			stack: 'top'
		});
	}
	expect(byId['no-copy'].published.horizon).toBe(1_000_000);
	expect(byId['m6502-s1-a'].config).toMatchObject({
		isa: 'm6502',
		compat: '0.3',
		variant: 'nmos',
		seed: 65001,
		stack: 'first-push'
	});
	expect(byId['m6502-s1-a'].published.horizon).toBe(10_000_000);
	expect(byId['i8086-s2-a'].config).toMatchObject({
		isa: 'i8086',
		compat: '0.3',
		variant: 'nmos',
		seed: 8086001,
		stack: 'first-push'
	});
	expect(byId['i8086-s2-a'].published.horizon).toBe(3_000_000);
	for (const chemistry of ['benign', 'lethal']) {
		expect(byId[`offset-competition-${chemistry}`].initialCondition).toEqual({
			kind: 'competition',
			regenerator: [4, 0x5e, 0xed, 0xb0],
			transmitterCore: [0xa0, 0x5e, 0xed, 0xb0],
			share: 0.5,
			randomPayload: true
		});
		expect(byId[`offset-competition-${chemistry}`].config).toMatchObject({
			length: 32,
			ring: 64,
			compat: '0.3',
			lethalProbability: chemistry === 'lethal' ? 1 : 0
		});
	}
	for (const id of ['pusher', 'return-closer', 'displacement-4', 'displacement-32']) {
		expect(byId[id].category).toBe('demonstration');
		expect(byId[id].config.mutationExponent).toBe(31);
		expect(byId[id].published.horizonUnit).toBe('transfers');
		expect(byId[id].published.note).toContain('Mutation is disabled in the browser demonstration.');
	}
	const short = byId['displacement-4'].initialCondition;
	const full = byId['displacement-32'].initialCondition;
	expect(short.kind).toBe('tiled');
	expect(full.kind).toBe('tiled');
	if (short.kind === 'tiled' && full.kind === 'tiled') {
		expect(short.bytes).toHaveLength(32);
		expect(full.bytes).toHaveLength(32);
		expect(short.bytes[0]).toBe(0x04);
		expect(full.bytes[0]).toBe(0x60);
		expect(short.bytes[8]).toBe(0x7f);
		expect(full.bytes[8]).toBe(0x7f);
		expect(short.bytes.slice(1)).toEqual(full.bytes.slice(1));
	}
});

test('world manifests pin paper stack phase, explicit resets and ISA contracts', async ({
	page
}) => {
	const result = await page.evaluate(async () => {
		const enginePath = '/src/lib/lab/engine.ts';
		const presetPath = '/src/lib/lab/presets.ts';
		const packagePath = '/node_modules/@neovand/zilion/dist/index.js';
		const { machineSpec } = (await import(enginePath)) as typeof import('../src/lib/lab/engine');
		const { getPaperPreset } = (await import(
			presetPath
		)) as typeof import('../src/lib/lab/presets');
		const { buildMachine } = (await import(packagePath)) as typeof import('@neovand/zilion');
		const built = [
			'stage-g-l16',
			'ring-l12-p28',
			'm6502-s1-a',
			'i8086-s2-a',
			'random-registers',
			'dial-0.03'
		].map((id) => {
			const spec = machineSpec(getPaperPreset(id).config);
			const { manifest } = buildMachine(spec);
			return { id, manifest, set: spec.world.init?.set };
		});
		return Object.fromEntries(built.map((entry) => [entry.id, entry]));
	});
	expect(result['stage-g-l16'].manifest.core.compat).toBe('0.2');
	expect(result['stage-g-l16'].manifest.init.stack.convention).toMatchObject({
		kind: 'topAliases',
		position: 31,
		values: { sp: 65535 }
	});
	const padded = result['ring-l12-p28'].manifest.init.stack.convention;
	expect(padded?.position).toBe(23);
	expect(padded!.values.sp % 28).toBe(23);
	expect(result['m6502-s1-a'].set).toMatchObject({ a: 0, x: 0, y: 0, p: 0x20 });
	expect(result['m6502-s1-a'].manifest.init.stack.convention).toMatchObject({
		kind: 'firstPushAt',
		position: 31,
		values: { s: 255 }
	});
	expect(result['i8086-s2-a'].set).toMatchObject({
		ax: 0,
		cx: 0,
		dx: 0,
		bx: 0,
		bp: 0,
		si: 0,
		di: 0,
		cs: 0,
		ds: 0,
		es: 0,
		ss: 0,
		ip: 0,
		flags: 0xf002
	});
	expect(result['i8086-s2-a'].manifest.core.params).toMatchObject({
		prefetch: 'coherent',
		hidden_latch: 'zero_ea'
	});
	expect(result['i8086-s2-a'].manifest.init.stack.convention).toMatchObject({
		kind: 'firstPushAt',
		position: 31,
		values: { sp: 1 }
	});
	expect(result['random-registers'].manifest.resumable).toBe(false);
	expect(result['dial-0.03'].manifest.resumable).toBe(false);
	for (const { manifest } of Object.values(result)) {
		expect(manifest.schema).toBe('zilion.manifest/2');
		for (const hash of Object.values(manifest.sha256)) expect(hash).toMatch(/^[0-9a-f]{64}$/);
	}
});

test('historical no-copy masks match the frozen paper table without disabling all writes', async ({
	page
}) => {
	const result = await page.evaluate(async () => {
		const ablationPath = '/src/lib/lab/ablation.ts';
		const presetPath = '/src/lib/lab/presets.ts';
		const { ablationMasks } = (await import(
			ablationPath
		)) as typeof import('../src/lib/lab/ablation');
		const { NO_COPY_ABLATION } = (await import(
			presetPath
		)) as typeof import('../src/lib/lab/presets');
		return {
			noCopy: [...ablationMasks('z80', NO_COPY_ABLATION)],
			block: [...ablationMasks('z80', ['family:block-copy'])],
			mos: [...ablationMasks('m6502', ['opcode:A9'])],
			intel: [...ablationMasks('i8086', ['opcode:A4'])]
		};
	});
	// Frozen paper src/algocell_exp/shader/isa.json SHA-256:
	// e4dd8f3ac25f11454a8ce7c950b1b14456252a8170b4940b22e7d94f3ca46061.
	// Derived independently from its nine registered family IDs, not the live app resolver.
	expect(result.noCopy).toEqual([
		0x44464546, 0x44464446, 0xffffffff, 0xffbfffff, 0, 0, 0x02220022, 0x0222082a, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0x88888888, 0x08080808, 0, 0x01010101, 0, 0
	]);
	const blocked = (words: number[], code: number, page = 0) =>
		((words[page * 8 + (code >> 5)] >>> (code & 31)) & 1) === 1;
	for (const code of [0xcd, 0xc7, 0x34, 0x35]) expect(blocked(result.noCopy, code)).toBe(false);
	for (const code of [0x01, 0x3e, 0xc5, 0xc1, 0xe3])
		expect(blocked(result.noCopy, code)).toBe(true);
	const edBlocked = Array.from({ length: 256 }, (_, code) => code).filter((code) =>
		blocked(result.block, code, 2)
	);
	expect(edBlocked).toEqual([0xa0, 0xa8, 0xb0, 0xb8]);
	expect(result.block.slice(0, 16).every((word) => word === 0)).toBe(true);
	expect(result.mos).toHaveLength(8);
	expect(blocked(result.mos, 0xa9)).toBe(true);
	expect(blocked(result.mos, 0xa4)).toBe(false);
	expect(blocked(result.intel, 0xa4)).toBe(true);
	expect(blocked(result.intel, 0xa9)).toBe(false);
});

test('ISA disassembly preserves operands, relative addresses, prefixes and truncation', async ({
	page
}) => {
	const result = await page.evaluate(async () => {
		const path = '/src/lib/lab/disassemble.ts';
		const { disassembleTape } = (await import(path)) as typeof import('../src/lib/lab/disassemble');
		return {
			mos: disassembleTape('m6502', [0xa9, 0x2a, 0x8d, 0x34, 0x12, 0xd0, 0xf9, 0x00, 0x00]),
			mosRelative: disassembleTape('m6502', [0xd0, 0xfe], 0x100),
			mosCut: disassembleTape('m6502', [0x4c, 0x34]),
			brkCut: disassembleTape('m6502', [0]),
			intel: disassembleTape('i8086', [0xb9, 0x34, 0x12, 0x51, 0xf3, 0xa4, 0x8b, 0x46, 0xfe]),
			intelAlias: disassembleTape('i8086', [0xc8, 1, 0]),
			intelRelative: disassembleTape('i8086', [0xeb, 0xfe]),
			intelFar: disassembleTape('i8086', [0xea, 0x34, 0x12, 0x78, 0x56]),
			intelCut: disassembleTape('i8086', [0xb8, 0x34]),
			prefixCut: disassembleTape('i8086', [0xf3]),
			z80: disassembleTape('z80', [0xdd, 0x66, 2, 0xdd, 0xcb, 1, 0]),
			z80Cut: disassembleTape('z80', [0xfd])
		};
	});
	expect(result.mos.map((line) => line.length)).toEqual([2, 3, 2, 2]);
	expect(result.mos.map((line) => line.mnemonic)).toEqual([
		'LDA #$2A',
		'STA $1234',
		'BNE $0000',
		'BRK signature $00'
	]);
	expect(result.mosRelative[0].mnemonic).toBe('BNE $0100');
	expect(result.intel.map((line) => line.length)).toEqual([3, 1, 2, 3]);
	expect(result.intel.map((line) => line.mnemonic)).toEqual([
		'MOV CX,$1234',
		'PUSH CX',
		'REP MOVSB',
		'MOV AX,word [BP-$02]'
	]);
	expect(result.intelAlias[0].mnemonic).toBe('RETF $0001 (8086 alias)');
	expect(result.intelRelative[0].mnemonic).toBe('JMP $0000');
	expect(result.intelFar[0].mnemonic).toBe('JMP FAR $5678:$1234');
	expect(result.z80.map((line) => line.mnemonic)).toEqual(['LD H,(IX+$02)', 'RLC (IX+$01),B']);
	expect(result.z80.map((line) => line.length)).toEqual([3, 4]);
	for (const lines of [
		result.mosCut,
		result.brkCut,
		result.intelCut,
		result.prefixCut,
		result.z80Cut
	]) {
		expect(lines).toHaveLength(1);
		expect(lines[0].truncated).toBe(true);
		expect(lines[0].mnemonic).toContain('incomplete');
		expect(lines[0].bytes.length).toBeLessThan(lines[0].length);
	}
	expect(result.intelCut[0].mnemonic).toContain('????');
	expect(result.mosCut[0].mnemonic).toContain('????');
});

test('checkpoint validation rejects incompatible contracts and malformed data before restore', async ({
	page
}) => {
	const result = await page.evaluate(async () => {
		const enginePath = '/src/lib/lab/engine.ts';
		const checkpointPath = '/src/lib/lab/checkpoint.ts';
		const packagePath = '/node_modules/@neovand/zilion/dist/index.js';
		const { machineSpec, DEFAULT_SETTINGS } = (await import(
			enginePath
		)) as typeof import('../src/lib/lab/engine');
		const { parseCheckpoint, serializeCheckpoint } = (await import(
			checkpointPath
		)) as typeof import('../src/lib/lab/checkpoint');
		const { buildMachine } = (await import(packagePath)) as typeof import('@neovand/zilion');
		const settings = { ...DEFAULT_SETTINGS, width: 2, height: 2, pairAttempts: 1, seed: 0 };
		const snapshot: import('../src/lib/lab/types').LabSnapshot = {
			schema: 'algocell.lab/1',
			scheduler: 'serial-reflected-pcg/1',
			settings,
			initialCondition: { kind: 'random' },
			update: 0,
			encounters: 0,
			executedSteps: 0,
			rngState: 0,
			bytes: Array.from({ length: 64 }, (_, i) => i),
			manifest: buildMachine(machineSpec(settings)).manifest,
			createdAt: '2026-10-13T00:00:00.000Z'
		};
		const mutations: [string, (value: Record<string, unknown>) => void][] = [
			[
				'schema',
				(value) => {
					value.schema = 'algocell.lab/99';
				}
			],
			[
				'scheduler',
				(value) => {
					value.scheduler = 'unknown/1';
				}
			],
			[
				'short bytes',
				(value) => {
					value.bytes = [1, 2];
				}
			],
			[
				'invalid byte',
				(value) => {
					(value.bytes as number[])[0] = 256;
				}
			],
			[
				'negative counter',
				(value) => {
					value.update = -1;
				}
			],
			[
				'RNG overflow',
				(value) => {
					value.rngState = 4294967296;
				}
			],
			[
				'bad hash',
				(value) => {
					(value.manifest as { sha256: { world: string } }).sha256.world = 'x';
				}
			],
			[
				'bad ISA',
				(value) => {
					(value.settings as { isa: string }).isa = 'bff';
				}
			],
			[
				'short ring',
				(value) => {
					(value.settings as { ring: number }).ring = 16;
				}
			],
			[
				'invalid initial tape',
				(value) => {
					value.initialCondition = { kind: 'tiled', bytes: [999] };
				}
			],
			[
				'invalid timestamp',
				(value) => {
					value.createdAt = 'never';
				}
			]
		];
		const failures = mutations.map(([name, mutate]) => {
			const copy = structuredClone(snapshot) as unknown as Record<string, unknown>;
			mutate(copy);
			try {
				parseCheckpoint(JSON.stringify(copy));
				return { name, rejected: false };
			} catch {
				return { name, rejected: true };
			}
		});
		return { roundtrip: parseCheckpoint(serializeCheckpoint(snapshot)), failures };
	});
	expect(result.roundtrip.settings.seed).toBe(0);
	expect(result.roundtrip.rngState).toBe(0);
	expect(result.roundtrip.bytes).toEqual(Array.from({ length: 64 }, (_, i) => i));
	for (const failure of result.failures) expect(failure.rejected, failure.name).toBe(true);
});

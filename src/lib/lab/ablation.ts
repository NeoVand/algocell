import { M6502_DECODE, M6502_OPCODES } from '@neovand/zilion';
import { INSTRUCTIONS, matchPattern, suppressionMasks, resolveSuppression } from '../z80-opcodes';
import { disassembleTape } from './disassemble';
import type { LabIsa } from './types';

export interface AblationInstruction {
	key: string;
	mnemonic: string;
	family: string;
	code: number;
	page: number;
}

const HEX = (n: number) => n.toString(16).padStart(2, '0').toUpperCase();
const INTEL_PREFIXES = new Set([0x26, 0x2e, 0x36, 0x3e, 0xf0, 0xf1, 0xf2, 0xf3]);

/** Opcode-level families intentionally do not claim to resolve 8086 ModRM suboperations. */
function intelFamily(op: number): string {
	if (op === 0xa4 || op === 0xa5) return 'block-copy';
	if (op >= 0xa6 && op <= 0xaf) return 'string';
	if (
		(op >= 0x50 && op <= 0x5f) ||
		[0x06, 0x07, 0x0e, 0x0f, 0x16, 0x17, 0x1e, 0x1f, 0x9c, 0x9d].includes(op)
	)
		return 'stack';
	if (
		(op >= 0x88 && op <= 0x8e) ||
		(op >= 0xb0 && op <= 0xbf) ||
		[0xa0, 0xa1, 0xa2, 0xa3, 0xc4, 0xc5, 0xc6, 0xc7].includes(op)
	)
		return 'load';
	if (INTEL_PREFIXES.has(op)) return 'prefix';
	if (
		(op >= 0x60 && op <= 0x7f) ||
		(op >= 0xe0 && op <= 0xe3) ||
		[0xe8, 0xe9, 0xea, 0xeb].includes(op)
	)
		return 'jump';
	return 'opcode';
}

export function ablationInstructions(isa: LabIsa): AblationInstruction[] {
	if (isa === 'z80')
		return INSTRUCTIONS.map((i) => ({
			key: `${i.page}:${HEX(i.code)}`,
			mnemonic: i.mnemonic,
			family: i.family,
			code: i.code,
			page: { base: 0, cb: 1, ed: 2 }[i.page]
		}));
	if (isa === 'm6502')
		return M6502_OPCODES.map((i, code) => ({
			key: `opcode:${HEX(code)}`,
			mnemonic: `${i.mnemonic} ${i.mode}`,
			family:
				i.mnemonic === 'JAM'
					? 'jam'
					: i.mnemonic.startsWith('LD')
						? 'load'
						: ['PHA', 'PHP', 'PLA', 'PLP'].includes(i.mnemonic)
							? 'stack'
							: M6502_DECODE[code] & ((1 << 5) | (1 << 7)) || ['BRK', 'JSR'].includes(i.mnemonic)
								? 'writes-mem'
								: 'opcode',
			code,
			page: 0
		}));
	const groups: Record<number, string> = {
		0x80: 'ALU r/m8,imm8',
		0x81: 'ALU r/m16,imm16',
		0x82: 'ALU r/m8,imm8 (alias)',
		0x83: 'ALU r/m16,imm8',
		0xd0: 'Rotate/shift r/m8,1',
		0xd1: 'Rotate/shift r/m16,1',
		0xd2: 'Rotate/shift r/m8,CL',
		0xd3: 'Rotate/shift r/m16,CL',
		0xf6: 'TEST/NOT/NEG/MUL/IMUL/DIV/IDIV r/m8',
		0xf7: 'TEST/NOT/NEG/MUL/IMUL/DIV/IDIV r/m16',
		0xfe: 'INC/DEC and byte aliases r/m8',
		0xff: 'INC/DEC/CALL/JMP/PUSH r/m16'
	};
	return Array.from({ length: 256 }, (_, code) => ({
		key: `opcode:${HEX(code)}`,
		mnemonic:
			groups[code] ??
			disassembleTape('i8086', [code, 0, 0, 0, 0, 0, 0])[0]
				.mnemonic.replace(/\$[0-9A-F]{4}/g, 'nn')
				.replace(/\$[0-9A-F]{2}/g, 'n'),
		family: intelFamily(code),
		code,
		page: 0
	})).filter((i) => i.family !== 'prefix');
}

export function ablationMasks(isa: LabIsa, patterns: readonly string[]): Uint32Array {
	if (isa === 'z80') {
		for (const pattern of patterns)
			if (pattern.trim() && !matchPattern(pattern).length)
				throw new Error(`Unknown Z80 ablation: ${pattern}`);
		return suppressionMasks(resolveSuppression(patterns));
	}
	const masks = new Uint32Array(8);
	const table = ablationInstructions(isa);
	for (const input of patterns) {
		const pattern = input.trim().toLowerCase();
		if (!pattern) continue;
		const exact = /^(?:(?:opcode|base):)?(?:0x)?([0-9a-f]{2})$/.exec(pattern);
		const matched = table.filter((i) =>
			exact
				? i.code === parseInt(exact[1], 16)
				: pattern.startsWith('family:')
					? i.family === pattern.slice(7)
					: i.mnemonic.toLowerCase().includes(pattern)
		);
		if (!matched.length) throw new Error(`Unknown ${isa} ablation: ${input}`);
		for (const i of matched) masks[i.code >> 5] |= 1 << (i.code & 31);
	}
	return masks;
}

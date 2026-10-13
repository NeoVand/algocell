import { M6502_OPCODES, type M6502Mode } from '@neovand/zilion';
import { PAGE_TABLES } from '../z80-opcodes';
import type { DisassemblyLine, LabIsa } from './types';

const hex = (n: number, digits = 2) =>
	`$${(n >>> 0).toString(16).toUpperCase().padStart(digits, '0')}`;
const signed8 = (n: number) => (n & 128 ? n - 256 : n);
const signed16 = (n: number) => (n & 32768 ? n - 65536 : n);

/** Static decoding of the selected bytes. It never invents bytes from the unknown partner. */
export function disassembleTape(
	isa: LabIsa,
	data: Uint8Array | number[],
	startOffset = 0
): DisassemblyLine[] {
	const bytes = data instanceof Uint8Array ? data : Uint8Array.from(data);
	if (isa === 'm6502') return disassemble6502(bytes, startOffset);
	if (isa === 'i8086') return disassemble8086(bytes, startOffset);
	return disassembleZ80(bytes, startOffset);
}

function line(
	data: Uint8Array,
	from: number,
	to: number,
	mnemonic: string,
	offset: number
): DisassemblyLine {
	const truncated = to > data.length;
	return {
		offset: offset + from,
		bytes: Array.from(data.subarray(from, Math.min(to, data.length))),
		mnemonic: truncated ? `${mnemonic} · incomplete` : mnemonic,
		length: to - from,
		...(truncated ? { truncated: true } : {})
	};
}

export function disassembleZ80(data: Uint8Array, base = 0): DisassemblyLine[] {
	const lines: DisassemblyLine[] = [];
	let p = 0;
	while (p < data.length) {
		const start = p;
		let op = data[p++],
			index = '';
		const byte = (at: number) => (at < data.length ? hex(data[at]) : '??');
		const word = (at: number) =>
			at + 1 < data.length ? hex(data[at] | (data[at + 1] << 8), 4) : '????';
		if (op === 0xdd || op === 0xfd) {
			index = op === 0xdd ? 'IX' : 'IY';
			if (p === data.length) {
				lines.push(line(data, start, p + 1, `${hex(op)} prefix`, base));
				break;
			}
			if ([0xdd, 0xfd, 0xed].includes(data[p])) {
				lines.push(line(data, start, p, `${hex(op)} prefix (wasted step)`, base));
				continue;
			}
			op = data[p++];
		}
		if (op === 0xcb) {
			if (index) {
				const disp = p++,
					code = data[p++] ?? 0;
				const operand =
					disp < data.length
						? `(${index}${signed8(data[disp]) < 0 ? '-' : '+'}${hex(Math.abs(signed8(data[disp])))})`
						: `(${index}+??)`;
				const x = code >> 6,
					y = (code >> 3) & 7,
					z = code & 7;
				const name =
					x === 0
						? ['RLC', 'RRC', 'RL', 'RR', 'SLA', 'SRA', 'SLL', 'SRL'][y]
						: x === 1
							? `BIT ${y},`
							: `${x === 2 ? 'RES' : 'SET'} ${y},`;
				const destination = ['B', 'C', 'D', 'E', 'H', 'L', '', 'A'][z];
				lines.push(
					line(
						data,
						start,
						p,
						p > data.length
							? 'Indexed CB instruction'
							: `${name}${x === 0 ? ' ' : ''}${operand}${x !== 1 && destination ? `,${destination}` : ''}`,
						base
					)
				);
			} else {
				const code = data[p++];
				lines.push(
					line(data, start, p, code === undefined ? 'CB ??' : PAGE_TABLES.cb[code].mnemonic, base)
				);
			}
			continue;
		}
		if (op === 0xed) {
			const code = data[p++];
			if (code === undefined) {
				lines.push(line(data, start, p, 'ED ??', base));
				break;
			}
			const info = PAGE_TABLES.ed[code];
			const mnemonic = info.mnemonic.replace(/\bnn\b/g, word(p));
			p += info.length - 2;
			lines.push(line(data, start, p, mnemonic, base));
			continue;
		}
		const info = PAGE_TABLES.base[op];
		let mnemonic = info.mnemonic;
		const indexedMemory = !!index && mnemonic.includes('(HL)') && op !== 0xe9;
		if (index && op !== 0xeb) {
			if (indexedMemory) {
				const disp = p++;
				const address =
					disp < data.length
						? `(${index}${signed8(data[disp]) < 0 ? '-' : '+'}${hex(Math.abs(signed8(data[disp])))})`
						: `(${index}+??)`;
				mnemonic = mnemonic.replace('(HL)', address);
			} else
				mnemonic = mnemonic
					.replace(/\bHL\b/g, index)
					.replace(/\bH\b/g, `${index}H`)
					.replace(/\bL\b/g, `${index}L`);
		}
		const operands = info.length - 1;
		if (mnemonic.includes('nn')) mnemonic = mnemonic.replace(/\bnn\b/g, word(p));
		else if (/\bn\b/.test(mnemonic)) mnemonic = mnemonic.replace(/\bn\b/g, byte(p));
		else if (/\bd\b/.test(mnemonic))
			mnemonic = mnemonic.replace(
				/\bd\b/g,
				p < data.length ? hex((base + p + 1 + signed8(data[p])) & 0xffff, 4) : '????'
			);
		p += operands;
		lines.push(line(data, start, p, mnemonic, base));
	}
	return lines;
}

const LENGTH6502: Record<M6502Mode, number> = {
	imp: 1,
	acc: 1,
	imm: 2,
	rel: 2,
	zp: 2,
	zpx: 2,
	zpy: 2,
	izx: 2,
	izy: 2,
	abs: 3,
	abx: 3,
	aby: 3,
	ind: 3
};

export function disassemble6502(data: Uint8Array, base = 0): DisassemblyLine[] {
	const lines: DisassemblyLine[] = [];
	for (let p = 0; p < data.length; ) {
		const start = p,
			{ mnemonic, mode } = M6502_OPCODES[data[p]],
			size = LENGTH6502[mode];
		const b = p + 1 < data.length ? hex(data[p + 1]) : '??';
		const w = p + 2 < data.length ? hex(data[p + 1] | (data[p + 2] << 8), 4) : '????';
		const operand: Record<M6502Mode, string> = {
			imp: '',
			acc: 'A',
			imm: mnemonic === 'BRK' ? `signature ${b}` : `#${b}`,
			rel: p + 1 < data.length ? hex((base + p + 2 + signed8(data[p + 1])) & 65535, 4) : '????',
			zp: b,
			zpx: `${b},X`,
			zpy: `${b},Y`,
			abs: w,
			abx: `${w},X`,
			aby: `${w},Y`,
			ind: `(${w})`,
			izx: `(${b},X)`,
			izy: `(${b}),Y`
		};
		p += size;
		lines.push(
			line(data, start, p, `${mnemonic}${operand[mode] ? ` ${operand[mode]}` : ''}`, base)
		);
	}
	return lines;
}

const R8 = ['AL', 'CL', 'DL', 'BL', 'AH', 'CH', 'DH', 'BH'];
const R16 = ['AX', 'CX', 'DX', 'BX', 'SP', 'BP', 'SI', 'DI'];
const SEG = ['ES', 'CS', 'SS', 'DS'];
const EA = ['BX+SI', 'BX+DI', 'BP+SI', 'BP+DI', 'SI', 'DI', 'BP', 'BX'];
const ALU = ['ADD', 'OR', 'ADC', 'SBB', 'AND', 'SUB', 'XOR', 'CMP'];
const COND = [
	'JO',
	'JNO',
	'JB',
	'JAE',
	'JE',
	'JNE',
	'JBE',
	'JA',
	'JS',
	'JNS',
	'JP',
	'JNP',
	'JL',
	'JGE',
	'JLE',
	'JG'
];

/** 8086 forms including NMOS aliases. No 80186+/32-bit interpretation is introduced. */
export function disassemble8086(data: Uint8Array, base = 0): DisassemblyLine[] {
	const lines: DisassemblyLine[] = [];
	let p = 0;
	while (p < data.length) {
		const start = p;
		let segment = '',
			repeat = '',
			lock = false;
		const get = () => data[p++] ?? 0;
		const imm8 = () => {
			const at = p;
			const value = get();
			return at < data.length ? hex(value) : '??';
		};
		const imm16 = () => {
			const at = p,
				value = get() | (get() << 8);
			return at + 1 < data.length ? hex(value, 4) : '????';
		};
		const relative = (word: boolean) => {
			const at = p,
				value = word ? signed16(get() | (get() << 8)) : signed8(get());
			return at + (word ? 1 : 0) < data.length ? hex((base + p + value) & 65535, 4) : '????';
		};
		let op = get();
		while ([0x26, 0x2e, 0x36, 0x3e, 0xf0, 0xf1, 0xf2, 0xf3].includes(op)) {
			if ([0x26, 0x2e, 0x36, 0x3e].includes(op)) segment = SEG[(op >> 3) & 3];
			else if (op === 0xf0 || op === 0xf1) lock = true;
			else repeat = op === 0xf2 ? 'REPNE' : 'REP';
			if (p >= data.length) break;
			op = get();
		}
		if ([0x26, 0x2e, 0x36, 0x3e, 0xf0, 0xf1, 0xf2, 0xf3].includes(op)) {
			lines.push(
				line(
					data,
					start,
					p + 1,
					[segment && `${segment}:`, lock && 'LOCK', repeat, 'prefix'].filter(Boolean).join(' '),
					base
				)
			);
			break;
		}
		const modrm = (word: boolean) => {
			const at = p,
				m = get(),
				mod = m >> 6,
				reg = (m >> 3) & 7,
				rm = m & 7;
			let operand: string;
			if (at >= data.length) operand = '??';
			else if (mod === 3) operand = (word ? R16 : R8)[rm];
			else {
				let address = mod === 0 && rm === 6 ? imm16() : EA[rm];
				if (mod === 1) {
					const at = p,
						d = signed8(get());
					address += at < data.length ? `${d < 0 ? '-' : '+'}${hex(Math.abs(d))}` : '+??';
				}
				if (mod === 2) {
					const at = p,
						d = signed16(get() | (get() << 8));
					address += at + 1 < data.length ? `${d < 0 ? '-' : '+'}${hex(Math.abs(d), 4)}` : '+????';
				}
				operand = `${word ? 'word' : 'byte'} [${segment ? `${segment}:` : ''}${address}]`;
			}
			return { reg, operand, register: (word ? R16 : R8)[reg], mod };
		};
		let name = '';
		if (op < 0x40 && (op & 7) <= 5) {
			const operation = ALU[op >> 3],
				form = op & 7;
			if (form < 4) {
				const m = modrm(!!(op & 1));
				name = `${operation} ${op & 2 ? `${m.register},${m.operand}` : `${m.operand},${m.register}`}`;
			} else name = `${operation} ${op & 1 ? 'AX' : 'AL'},${op & 1 ? imm16() : imm8()}`;
		} else if (op < 0x20 && (op & 7) >= 6) name = `${op & 1 ? 'POP' : 'PUSH'} ${SEG[op >> 3]}`;
		else if ([0x27, 0x2f, 0x37, 0x3f].includes(op))
			name = ['DAA', 'DAS', 'AAA', 'AAS'][(op - 0x27) >> 3];
		else if (op < 0x60 && op >= 0x40)
			name = `${['INC', 'DEC', 'PUSH', 'POP'][(op - 0x40) >> 3]} ${R16[op & 7]}`;
		else if (op >= 0x60 && op <= 0x7f)
			name = `${COND[op & 15]} ${relative(false)}${op < 0x70 ? ' (8086 alias)' : ''}`;
		else if (op >= 0x80 && op <= 0x83) {
			const m = modrm(!!(op & 1));
			name = `${ALU[m.reg]} ${m.operand},${op === 0x81 ? imm16() : imm8()}${op === 0x83 ? ' (sign-extended)' : ''}`;
		} else if (op >= 0x84 && op <= 0x8b) {
			const m = modrm(!!(op & 1));
			name = `${op < 0x86 ? 'TEST' : op < 0x88 ? 'XCHG' : 'MOV'} ${op >= 0x8a ? `${m.register},${m.operand}` : `${m.operand},${m.register}`}`;
		} else if (op === 0x8c || op === 0x8e) {
			const m = modrm(true);
			name = `MOV ${op === 0x8c ? `${m.operand},${SEG[m.reg & 3]}` : `${SEG[m.reg & 3]},${m.operand}`}`;
		} else if (op === 0x8d) {
			const m = modrm(true);
			name = `LEA ${m.register},${m.operand}${m.mod === 3 ? ' (unimplemented form)' : ''}`;
		} else if (op === 0x8f) {
			const m = modrm(true);
			name = `POP ${m.operand}`;
		} else if (op >= 0x90 && op <= 0x97) name = op === 0x90 ? 'NOP' : `XCHG AX,${R16[op & 7]}`;
		else if (op === 0x9a || op === 0xea) {
			const offset = imm16(),
				seg = imm16();
			name = `${op === 0x9a ? 'CALL' : 'JMP'} FAR ${seg}:${offset}`;
		} else if (op >= 0xa0 && op <= 0xa3) {
			const at = imm16(),
				reg = op & 1 ? 'AX' : 'AL',
				mem = `[${segment ? `${segment}:` : ''}${at}]`;
			name = `MOV ${op & 2 ? `${mem},${reg}` : `${reg},${mem}`}`;
		} else if (op >= 0xa4 && op <= 0xaf && op !== 0xa8 && op !== 0xa9)
			name = (
				{
					0xa4: 'MOVSB',
					0xa5: 'MOVSW',
					0xa6: 'CMPSB',
					0xa7: 'CMPSW',
					0xaa: 'STOSB',
					0xab: 'STOSW',
					0xac: 'LODSB',
					0xad: 'LODSW',
					0xae: 'SCASB',
					0xaf: 'SCASW'
				} as Record<number, string>
			)[op];
		else if (op === 0xa8 || op === 0xa9)
			name = `TEST ${op & 1 ? 'AX' : 'AL'},${op & 1 ? imm16() : imm8()}`;
		else if (op >= 0xb0 && op <= 0xbf)
			name = `MOV ${(op & 8 ? R16 : R8)[op & 7]},${op & 8 ? imm16() : imm8()}`;
		else if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc8, 0xc9, 0xca, 0xcb].includes(op))
			name = `${op & 8 ? 'RETF' : 'RET'}${op & 1 ? '' : ` ${imm16()}`}${[0xc0, 0xc1, 0xc8, 0xc9].includes(op) ? ' (8086 alias)' : ''}`;
		else if (op === 0xc4 || op === 0xc5) {
			const m = modrm(true);
			name = `${op === 0xc4 ? 'LES' : 'LDS'} ${m.register},${m.operand}${m.mod === 3 ? ' (unimplemented form)' : ''}`;
		} else if (op === 0xc6 || op === 0xc7) {
			const m = modrm(!!(op & 1));
			name = `MOV ${m.operand},${op & 1 ? imm16() : imm8()}`;
		} else if (op === 0xcd) name = `INT ${imm8()}`;
		else if (op >= 0xd0 && op <= 0xd3) {
			const m = modrm(!!(op & 1));
			name = `${['ROL', 'ROR', 'RCL', 'RCR', 'SHL', 'SHR', op & 2 ? 'SETMOC' : 'SETMO', 'SAR'][m.reg]} ${m.operand},${op & 2 ? 'CL' : '1'}`;
		} else if (op === 0xd4 || op === 0xd5) name = `${op === 0xd4 ? 'AAM' : 'AAD'} ${imm8()}`;
		else if (op >= 0xd8 && op <= 0xdf) {
			const m = modrm(true);
			name = `ESC ${hex((op & 7) * 8 + m.reg)},${m.operand}`;
		} else if (op >= 0xe0 && op <= 0xe3)
			name = `${['LOOPNE', 'LOOPE', 'LOOP', 'JCXZ'][op & 3]} ${relative(false)}`;
		else if ((op >= 0xe4 && op <= 0xe7) || (op >= 0xec && op <= 0xef)) {
			const port = op & 8 ? 'DX' : imm8(),
				reg = op & 1 ? 'AX' : 'AL';
			name = op & 2 ? `OUT ${port},${reg}` : `IN ${reg},${port}`;
		} else if (op === 0xe8 || op === 0xe9 || op === 0xeb)
			name = `${op === 0xe8 ? 'CALL' : 'JMP'} ${relative(op !== 0xeb)}`;
		else if (op === 0xf6 || op === 0xf7) {
			const m = modrm(!!(op & 1));
			name = `${['TEST', 'TEST', 'NOT', 'NEG', 'MUL', 'IMUL', 'DIV', 'IDIV'][m.reg]} ${m.operand}${m.reg < 2 ? `,${op & 1 ? imm16() : imm8()}` : ''}`;
		} else if (op === 0xfe || op === 0xff) {
			const m = modrm(!!(op & 1));
			name = `${['INC', 'DEC', 'CALL', 'CALL FAR', 'JMP', 'JMP FAR', 'PUSH', 'PUSH'][m.reg]} ${m.operand}${op === 0xfe && m.reg > 1 ? ' (8086 byte alias)' : ''}`;
		} else
			name =
				(
					{
						0x98: 'CBW',
						0x99: 'CWD',
						0x9b: 'WAIT',
						0x9c: 'PUSHF',
						0x9d: 'POPF',
						0x9e: 'SAHF',
						0x9f: 'LAHF',
						0xcc: 'INT 3',
						0xce: 'INTO',
						0xcf: 'IRET',
						0xd6: 'SALC',
						0xd7: 'XLAT',
						0xf4: 'HLT',
						0xf5: 'CMC',
						0xf8: 'CLC',
						0xf9: 'STC',
						0xfa: 'CLI',
						0xfb: 'STI',
						0xfc: 'CLD',
						0xfd: 'STD'
					} as Record<number, string>
				)[op] ?? `DB ${hex(op)}`;
		const prefixes = [lock && 'LOCK', repeat, segment && `${segment}:`].filter(Boolean).join(' ');
		lines.push(line(data, start, p, `${prefixes ? `${prefixes} ` : ''}${name}`, base));
	}
	return lines;
}

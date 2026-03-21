// Compact Z80 disassembler for displaying cell contents

const r8 = ['B', 'C', 'D', 'E', 'H', 'L', '(HL)', 'A'];
const r16 = ['BC', 'DE', 'HL', 'SP'];
const r16af = ['BC', 'DE', 'HL', 'AF'];
const cc = ['NZ', 'Z', 'NC', 'C', 'PO', 'PE', 'P', 'M'];
const alu = ['ADD A,', 'ADC A,', 'SUB', 'SBC A,', 'AND', 'XOR', 'OR', 'CP'];
const rot = ['RLC', 'RRC', 'RL', 'RR', 'SLA', 'SRA', 'SLL', 'SRL'];

export interface DisasmLine {
	offset: number;
	bytes: number[];
	mnemonic: string;
	length: number;
}

export function disassemble(data: Uint8Array, startOffset: number = 0): DisasmLine[] {
	const lines: DisasmLine[] = [];
	let pos = 0;

	while (pos < data.length) {
		const start = pos;
		const op = data[pos++];
		let mnemonic = '';

		const x = (op >> 6) & 3;
		const y = (op >> 3) & 7;
		const z = op & 7;
		const p = (y >> 1) & 3;
		const q = y & 1;

		if (x === 1) {
			if (y === 6 && z === 6) {
				mnemonic = 'HALT';
			} else {
				mnemonic = `LD ${r8[y]},${r8[z]}`;
			}
		} else if (x === 2) {
			mnemonic = `${alu[y]} ${r8[z]}`;
		} else if (x === 0) {
			switch (z) {
				case 0:
					if (y === 0) mnemonic = 'NOP';
					else if (y === 1) mnemonic = "EX AF,AF'";
					else if (y === 2) {
						const d = pos < data.length ? signedByte(data[pos++]) : 0;
						mnemonic = `DJNZ ${formatRel(d)}`;
					} else if (y === 3) {
						const d = pos < data.length ? signedByte(data[pos++]) : 0;
						mnemonic = `JR ${formatRel(d)}`;
					} else {
						const d = pos < data.length ? signedByte(data[pos++]) : 0;
						mnemonic = `JR ${cc[y - 4]},${formatRel(d)}`;
					}
					break;
				case 1:
					if (q === 0) {
						const nn = readWord(data, pos);
						pos += 2;
						mnemonic = `LD ${r16[p]},${hex16(nn)}`;
					} else {
						mnemonic = `ADD HL,${r16[p]}`;
					}
					break;
				case 2:
					if (q === 0) {
						if (p === 0) mnemonic = 'LD (BC),A';
						else if (p === 1) mnemonic = 'LD (DE),A';
						else if (p === 2) {
							const nn = readWord(data, pos);
							pos += 2;
							mnemonic = `LD (${hex16(nn)}),HL`;
						} else {
							const nn = readWord(data, pos);
							pos += 2;
							mnemonic = `LD (${hex16(nn)}),A`;
						}
					} else {
						if (p === 0) mnemonic = 'LD A,(BC)';
						else if (p === 1) mnemonic = 'LD A,(DE)';
						else if (p === 2) {
							const nn = readWord(data, pos);
							pos += 2;
							mnemonic = `LD HL,(${hex16(nn)})`;
						} else {
							const nn = readWord(data, pos);
							pos += 2;
							mnemonic = `LD A,(${hex16(nn)})`;
						}
					}
					break;
				case 3:
					mnemonic = q === 0 ? `INC ${r16[p]}` : `DEC ${r16[p]}`;
					break;
				case 4:
					mnemonic = `INC ${r8[y]}`;
					break;
				case 5:
					mnemonic = `DEC ${r8[y]}`;
					break;
				case 6: {
					const n = pos < data.length ? data[pos++] : 0;
					mnemonic = `LD ${r8[y]},${hex8(n)}`;
					break;
				}
				case 7:
					mnemonic = ['RLCA', 'RRCA', 'RLA', 'RRA', 'DAA', 'CPL', 'SCF', 'CCF'][y];
					break;
			}
		} else if (x === 3) {
			switch (z) {
				case 0:
					mnemonic = `RET ${cc[y]}`;
					break;
				case 1:
					if (q === 0) {
						mnemonic = `POP ${r16af[p]}`;
					} else {
						mnemonic = ['RET', 'EXX', 'JP (HL)', 'LD SP,HL'][p];
					}
					break;
				case 2: {
					const nn = readWord(data, pos);
					pos += 2;
					mnemonic = `JP ${cc[y]},${hex16(nn)}`;
					break;
				}
				case 3:
					if (y === 0) {
						const nn = readWord(data, pos);
						pos += 2;
						mnemonic = `JP ${hex16(nn)}`;
					} else if (y === 1) {
						// CB prefix
						if (pos < data.length) {
							const cb = data[pos++];
							const cx = (cb >> 6) & 3;
							const cy = (cb >> 3) & 7;
							const cz = cb & 7;
							if (cx === 0) mnemonic = `${rot[cy]} ${r8[cz]}`;
							else if (cx === 1) mnemonic = `BIT ${cy},${r8[cz]}`;
							else if (cx === 2) mnemonic = `RES ${cy},${r8[cz]}`;
							else mnemonic = `SET ${cy},${r8[cz]}`;
						} else {
							mnemonic = 'CB ??';
						}
					} else if (y === 2) {
						const n = pos < data.length ? data[pos++] : 0;
						mnemonic = `OUT (${hex8(n)}),A`;
					} else if (y === 3) {
						const n = pos < data.length ? data[pos++] : 0;
						mnemonic = `IN A,(${hex8(n)})`;
					} else if (y === 4) {
						mnemonic = 'EX (SP),HL';
					} else if (y === 5) {
						mnemonic = 'EX DE,HL';
					} else if (y === 6) {
						mnemonic = 'DI';
					} else {
						mnemonic = 'EI';
					}
					break;
				case 4: {
					const nn = readWord(data, pos);
					pos += 2;
					mnemonic = `CALL ${cc[y]},${hex16(nn)}`;
					break;
				}
				case 5:
					if (q === 0) {
						mnemonic = `PUSH ${r16af[p]}`;
					} else if (p === 0) {
						const nn = readWord(data, pos);
						pos += 2;
						mnemonic = `CALL ${hex16(nn)}`;
					} else if (p === 2) {
						// ED prefix
						if (pos < data.length) {
							const ed = data[pos++];
							mnemonic = disasmED(ed, data, pos);
							pos += edExtraBytes(ed);
						} else {
							mnemonic = 'ED ??';
						}
					} else {
						mnemonic = p === 1 ? 'DD ..' : 'FD ..';
					}
					break;
				case 6: {
					const n = pos < data.length ? data[pos++] : 0;
					mnemonic = `${alu[y]} ${hex8(n)}`;
					break;
				}
				case 7:
					mnemonic = `RST ${hex8(y * 8)}`;
					break;
			}
		}

		if (!mnemonic) mnemonic = `DB ${hex8(op)}`;

		const bytes = Array.from(data.slice(start, Math.min(pos, data.length)));
		lines.push({
			offset: startOffset + start,
			bytes,
			mnemonic,
			length: pos - start
		});
	}

	return lines;
}

function disasmED(op: number, _data: Uint8Array, _pos: number): string {
	const edMap: Record<number, string> = {
		0xa0: 'LDI',
		0xa8: 'LDD',
		0xb0: 'LDIR',
		0xb8: 'LDDR',
		0xa1: 'CPI',
		0xa9: 'CPD',
		0xb1: 'CPIR',
		0xb9: 'CPDR',
		0x44: 'NEG',
		0x45: 'RETN',
		0x4d: 'RETI',
		0x47: 'LD I,A',
		0x4f: 'LD R,A',
		0x57: 'LD A,I',
		0x5f: 'LD A,R',
		0x67: 'RRD',
		0x6f: 'RLD'
	};
	if (edMap[op]) return edMap[op];
	return `ED ${hex8(op)}`;
}

function edExtraBytes(op: number): number {
	if ([0x43, 0x53, 0x63, 0x73, 0x4b, 0x5b, 0x6b, 0x7b].includes(op)) return 2;
	return 0;
}

function signedByte(b: number): number {
	return b > 127 ? b - 256 : b;
}

function readWord(data: Uint8Array, pos: number): number {
	if (pos + 1 >= data.length) return 0;
	return data[pos] | (data[pos + 1] << 8);
}

function hex8(n: number): string {
	return '$' + (n & 0xff).toString(16).toUpperCase().padStart(2, '0');
}

function hex16(n: number): string {
	return '$' + (n & 0xffff).toString(16).toUpperCase().padStart(4, '0');
}

function formatRel(d: number): string {
	return d >= 0 ? `+${d}` : `${d}`;
}

// Get the mnemonic name for a single byte (used for byte frequency display)
export function byteToMnemonic(byte: number): string {
	const simple: Record<number, string> = {
		0x00: 'NOP',
		0x76: 'HALT',
		0x01: 'LD BC,nn',
		0x11: 'LD DE,nn',
		0x21: 'LD HL,nn',
		0x31: 'LD SP,nn',
		0xc3: 'JP nn',
		0xcd: 'CALL nn',
		0xc9: 'RET',
		0xed: 'ED prefix',
		0xcb: 'CB prefix',
		0xdd: 'DD prefix',
		0xfd: 'FD prefix',
		0xb0: 'LDIR*',
		0xb8: 'LDDR*',
		0xa0: 'LDI*',
		0xa8: 'LDD*',
		0xc5: 'PUSH BC',
		0xd5: 'PUSH DE',
		0xe5: 'PUSH HL',
		0xf5: 'PUSH AF',
		0xc1: 'POP BC',
		0xd1: 'POP DE',
		0xe1: 'POP HL',
		0xf1: 'POP AF',
		0x18: 'JR d',
		0x10: 'DJNZ d'
	};
	return simple[byte] || '';
}

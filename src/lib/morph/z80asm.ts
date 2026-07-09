// Minimal Z80 assembler (the subset the morphogenesis CA bootstrap needs).
//
// This is intentionally small and strict: it supports exactly the instruction
// forms used by `bootstrap.ts`, resolves code labels (for jumps) and caller
// -supplied symbols (for fixed data addresses), and throws loudly on anything
// it does not understand — a wrong opcode should fail assembly, not silently
// emit garbage that the differential test then has to hunt down.
//
// Two passes: pass 1 sizes every instruction and records label addresses;
// pass 2 emits bytes with a real resolver. Instruction *size* never depends on
// operand values, so pass 1 encodes with a dummy resolver that returns 0.

export type SymbolMap = Record<string, number>;

const REG8: Record<string, number> = { B: 0, C: 1, D: 2, E: 3, H: 4, L: 5, A: 7 };
const REG16: Record<string, number> = { BC: 0, DE: 1, HL: 2, SP: 3 };
const CC: Record<string, number> = { NZ: 0, Z: 1, NC: 2, C: 3, PO: 4, PE: 5, P: 6, M: 7 };
const JR_CC: Record<string, number> = { NZ: 0x20, Z: 0x28, NC: 0x30, C: 0x38 };

interface Insn {
	label?: string;
	mnem: string;
	ops: string[];
}

function isIndirect(s: string): boolean {
	return s.startsWith('(') && s.endsWith(')');
}
function inner(s: string): string {
	return s.slice(1, -1).trim();
}
function reg8(tok: string): number | null {
	const c = REG8[tok.toUpperCase()];
	return c === undefined ? null : c;
}
function reg16(tok: string): number | null {
	const c = REG16[tok.toUpperCase()];
	return c === undefined ? null : c;
}
function u16(n: number): [number, number] {
	n &= 0xffff;
	return [n & 0xff, (n >> 8) & 0xff];
}

/** Encode one instruction to bytes. `resolve` maps a value-operand to a number. */
function encode(mnem: string, ops: string[], addr: number, resolve: (op: string) => number): number[] {
	const M = mnem.toUpperCase();

	switch (M) {
		case 'NOP':
			return [0x00];
		case 'HALT':
			return [0x76];
		case 'LDIR':
			return [0xed, 0xb0];
		case 'LDDR':
			return [0xed, 0xb8];
		case 'EX':
			// only EX DE,HL is supported
			if (ops[0]?.toUpperCase() === 'DE' && ops[1]?.toUpperCase() === 'HL') return [0xeb];
			throw new Error(`EX form not supported: ${ops.join(',')}`);
		case 'RET':
			if (ops.length === 0) return [0xc9];
			return [0xc0 | (CC[ops[0].toUpperCase()] << 3)];
		case 'DB':
			return ops.map((o) => resolve(o) & 0xff);
		case 'LD':
			return encodeLD(ops, resolve);
		case 'ADD':
			return encodeADD(ops, resolve);
		case 'ADC':
			return encodeAcc(ops, resolve, 0x88, 0xce);
		case 'SBC':
			return encodeAcc(ops, resolve, 0x98, 0xde);
		case 'SUB':
			return encodeAImplicit(ops, resolve, 0x90, 0xd6);
		case 'AND':
			return encodeAImplicit(ops, resolve, 0xa0, 0xe6);
		case 'XOR':
			return encodeAImplicit(ops, resolve, 0xa8, 0xee);
		case 'OR':
			return encodeAImplicit(ops, resolve, 0xb0, 0xf6);
		case 'CP':
			return encodeAImplicit(ops, resolve, 0xb8, 0xfe);
		case 'INC':
			return encodeIncDec(ops, true);
		case 'DEC':
			return encodeIncDec(ops, false);
		case 'JP':
			return encodeJP(ops, resolve);
		case 'JR':
			return encodeJR(ops, resolve, addr);
		case 'DJNZ':
			return [0x10, (resolve(ops[0]) - (addr + 2)) & 0xff];
		case 'CALL':
			return encodeCALL(ops, resolve);
		default:
			throw new Error(`unknown mnemonic: ${M}`);
	}
}

function encodeLD(ops: string[], R: (op: string) => number): number[] {
	if (ops.length !== 2) throw new Error(`LD needs 2 operands: ${ops.join(',')}`);
	const [a, b] = ops;
	const A = a.toUpperCase();
	const B = b.toUpperCase();

	// A <- indirect
	if (A === 'A' && isIndirect(b)) {
		const x = inner(b).toUpperCase();
		if (x === 'HL') return [0x7e];
		if (x === 'DE') return [0x1a];
		if (x === 'BC') return [0x0a];
		return [0x3a, ...u16(R(inner(b)))]; // LD A,(nn)
	}
	// indirect <- A
	if (isIndirect(a) && B === 'A') {
		const x = inner(a).toUpperCase();
		if (x === 'HL') return [0x77];
		if (x === 'DE') return [0x12];
		if (x === 'BC') return [0x02];
		return [0x32, ...u16(R(inner(a)))]; // LD (nn),A
	}
	// (nn) <- HL / HL <- (nn)
	if (isIndirect(a) && B === 'HL') return [0x22, ...u16(R(inner(a)))];
	if (A === 'HL' && isIndirect(b)) return [0x2a, ...u16(R(inner(b)))];
	// (nn) <- DE / DE <- (nn)  (ED-prefixed)
	if (isIndirect(a) && B === 'DE') return [0xed, 0x53, ...u16(R(inner(a)))];
	if (A === 'DE' && isIndirect(b)) return [0xed, 0x5b, ...u16(R(inner(b)))];
	// SP <- HL
	if (A === 'SP' && B === 'HL') return [0xf9];

	// 16-bit immediate load: LD rr,nn
	const rr = reg16(A);
	if (rr !== null && reg8(B) === null && !isIndirect(b) && reg16(B) === null) {
		return [0x01 | (rr << 4), ...u16(R(b))];
	}

	// 8-bit destination (register or (HL))
	const dst = A === '(HL)' ? 6 : reg8(A);
	if (dst !== null) {
		// LD r,r'  (or LD r,(HL) / LD (HL),r)
		const src = B === '(HL)' ? 6 : reg8(B);
		if (src !== null) {
			if (dst === 6 && src === 6) throw new Error('LD (HL),(HL) is invalid');
			return [0x40 | (dst << 3) | src];
		}
		// LD r,n  (LD (HL),n => 0x36)
		return [(0x06 | (dst << 3)) & 0xff, R(b) & 0xff];
	}

	throw new Error(`LD form not supported: ${ops.join(',')}`);
}

function encodeADD(ops: string[], R: (op: string) => number): number[] {
	const A = ops[0].toUpperCase();
	if (A === 'HL') {
		const rr = reg16(ops[1]);
		if (rr === null) throw new Error(`ADD HL,rr bad reg: ${ops[1]}`);
		return [0x09 | (rr << 4)];
	}
	if (A === 'A') return encodeAccSrc(ops[1], R, 0x80, 0xc6);
	throw new Error(`ADD form not supported: ${ops.join(',')}`);
}

// ADC A,x / SBC A,x  (always "A" explicit)
function encodeAcc(ops: string[], R: (op: string) => number, regBase: number, immOp: number): number[] {
	if (ops[0].toUpperCase() !== 'A') throw new Error(`expected A as first operand: ${ops.join(',')}`);
	return encodeAccSrc(ops[1], R, regBase, immOp);
}

// SUB/AND/XOR/OR/CP x  (A implicit; tolerate an explicit leading "A,")
function encodeAImplicit(ops: string[], R: (op: string) => number, regBase: number, immOp: number): number[] {
	const src = ops.length === 2 && ops[0].toUpperCase() === 'A' ? ops[1] : ops[0];
	return encodeAccSrc(src, R, regBase, immOp);
}

function encodeAccSrc(src: string, R: (op: string) => number, regBase: number, immOp: number): number[] {
	if (src.toUpperCase() === '(HL)') return [regBase | 6];
	const r = reg8(src);
	if (r !== null) return [regBase | r];
	return [immOp, R(src) & 0xff];
}

function encodeIncDec(ops: string[], isInc: boolean): number[] {
	const t = ops[0].toUpperCase();
	const rr = reg16(t);
	if (rr !== null) return [(isInc ? 0x03 : 0x0b) | (rr << 4)];
	const r = t === '(HL)' ? 6 : reg8(t);
	if (r === null) throw new Error(`${isInc ? 'INC' : 'DEC'} bad operand: ${ops[0]}`);
	return [(isInc ? 0x04 : 0x05) | (r << 3)];
}

function encodeJP(ops: string[], R: (op: string) => number): number[] {
	if (ops.length === 2) {
		const cc = CC[ops[0].toUpperCase()];
		if (cc === undefined) throw new Error(`JP bad condition: ${ops[0]}`);
		return [0xc2 | (cc << 3), ...u16(R(ops[1]))];
	}
	return [0xc3, ...u16(R(ops[0]))];
}

function encodeJR(ops: string[], R: (op: string) => number, addr: number): number[] {
	if (ops.length === 2) {
		const op = JR_CC[ops[0].toUpperCase()];
		if (op === undefined) throw new Error(`JR bad condition: ${ops[0]}`);
		return [op, (R(ops[1]) - (addr + 2)) & 0xff];
	}
	return [0x18, (R(ops[0]) - (addr + 2)) & 0xff];
}

function encodeCALL(ops: string[], R: (op: string) => number): number[] {
	if (ops.length === 2) {
		const cc = CC[ops[0].toUpperCase()];
		if (cc === undefined) throw new Error(`CALL bad condition: ${ops[0]}`);
		return [0xc4 | (cc << 3), ...u16(R(ops[1]))];
	}
	return [0xcd, ...u16(R(ops[0]))];
}

/** Parse one source line into an instruction (or null for blank/comment/label-only). */
function parseLine(raw: string): Insn | null {
	const noComment = raw.replace(/;.*$/, '').trim();
	if (!noComment) return null;
	let rest = noComment;
	let label: string | undefined;
	const colon = rest.indexOf(':');
	if (colon >= 0 && !isIndirect(rest.split(/\s+/)[0])) {
		label = rest.slice(0, colon).trim();
		rest = rest.slice(colon + 1).trim();
	}
	if (!rest) return label ? { label, mnem: '', ops: [] } : null;
	const sp = rest.indexOf(' ');
	const mnem = sp < 0 ? rest : rest.slice(0, sp);
	const opStr = sp < 0 ? '' : rest.slice(sp + 1).trim();
	const ops = opStr ? opStr.split(',').map((o) => o.trim()) : [];
	return { label, mnem, ops };
}

/**
 * Assemble source lines to bytes. `symbols` supplies fixed addresses/constants
 * (e.g. GENOME, FRONT, W). Code labels are resolved automatically. Assembly
 * starts at address 0 (Zilion copies the program to the base of the tape).
 */
export function assemble(lines: string[], symbols: SymbolMap = {}): Uint8Array {
	const insns: Insn[] = [];
	for (const l of lines) {
		const p = parseLine(l);
		if (p) insns.push(p);
	}

	// Uppercase symbol keys for case-insensitive lookup.
	const sym: SymbolMap = {};
	for (const [k, v] of Object.entries(symbols)) sym[k.toUpperCase()] = v;

	// Pass 1: label addresses + sizes.
	const labels: SymbolMap = {};
	const sizes: number[] = [];
	let addr = 0;
	for (const ins of insns) {
		if (ins.label) {
			const key = ins.label.toUpperCase();
			if (key in labels) throw new Error(`duplicate label: ${ins.label}`);
			labels[key] = addr;
		}
		if (!ins.mnem) {
			sizes.push(0);
			continue;
		}
		const size = encode(ins.mnem, ins.ops, addr, () => 0).length;
		sizes.push(size);
		addr += size;
	}

	const resolve = (op: string): number => {
		const t = op.trim();
		if (/^0x[0-9a-fA-F]+$/.test(t)) return parseInt(t, 16);
		if (/^-?\d+$/.test(t)) return parseInt(t, 10);
		const key = t.toUpperCase();
		if (key in labels) return labels[key];
		if (key in sym) return sym[key];
		throw new Error(`unresolved symbol/label: ${op}`);
	};

	// Pass 2: emit.
	const out: number[] = [];
	addr = 0;
	for (let i = 0; i < insns.length; i++) {
		const ins = insns[i];
		if (!ins.mnem) continue;
		const bytes = encode(ins.mnem, ins.ops, addr, resolve);
		if (bytes.length !== sizes[i]) {
			throw new Error(`size mismatch on "${ins.mnem} ${ins.ops.join(',')}": ${sizes[i]} vs ${bytes.length}`);
		}
		for (const b of bytes) out.push(b & 0xff);
		addr += bytes.length;
	}
	return Uint8Array.from(out);
}

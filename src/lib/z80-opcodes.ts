// Z80 instruction-set model for Algocell.
//
// Every executable instruction across the base, CB and ED pages, decoded from
// the standard Z80 opcode structure (x/y/z/p/q fields), with a semantic
// *family* label and a memory-write flag. This is the vocabulary in which
// instruction ablation ("suppression") is expressed and resolved:
//
//   pattern            →  instructions
//   ─────────────────────────────────────────────────────────────────────────
//   family:block-copy  →  LDI, LDD, LDIR, LDDR                (whole family)
//   family:writes-mem  →  every instruction that writes memory (pseudo-family)
//   ed:B0 / cb:FF / base:3E  →  one exact instruction on a given page
//   3E / 0x3E          →  one exact base-page opcode
//   LDIR, LD (HL),n …  →  legacy: every mnemonic containing the substring
//   -<pattern>         →  subtract: remove that pattern's instructions from the set
//                         (applied after all additions, e.g. family:writes-mem; -family:incdec-mem)
//
// The resolved result is three 256-bit sets (one per page) that map 1:1 onto
// the GPU core's prefix-aware hook: IX/IY forms of a base opcode follow the
// base set (so blocking LD (HL),n also blocks LD (IX+d),n), and DDCB/FDCB
// follow the CB set. Prefix bytes (CB, ED, DD, FD) are not instructions and
// can never be suppressed.

export type Page = 'base' | 'cb' | 'ed';

export type FamilyId =
	| 'ld8'
	| 'ld8-imm'
	| 'ld8-mem'
	| 'ld16-imm'
	| 'ld16-mem'
	| 'ld-special'
	| 'stack'
	| 'ex'
	| 'block-copy'
	| 'block-cmp'
	| 'alu8'
	| 'alu16'
	| 'incdec8'
	| 'incdec16'
	| 'incdec-mem'
	| 'acc-misc'
	| 'rotate-a'
	| 'rotate'
	| 'rotate-mem'
	| 'bit'
	| 'bit-set'
	| 'bit-set-mem'
	| 'jump'
	| 'jump-rel'
	| 'call-ret'
	| 'rst'
	| 'io'
	| 'block-io'
	| 'control'
	| 'undefined'
	| 'prefix';

export interface Family {
	id: FamilyId;
	name: string;
	description: string;
	/** Shown in the UI and selectable in patterns. */
	selectable: boolean;
}

export const FAMILIES: readonly Family[] = [
	{ id: 'ld8', name: 'LD r,r′', description: '8-bit register-to-register loads', selectable: true },
	{ id: 'ld8-imm', name: 'LD r,n', description: '8-bit immediate loads', selectable: true },
	{
		id: 'ld8-mem',
		name: 'LD 8-bit ↔ memory',
		description: 'LD r,(HL) · LD (HL),r · LD (HL),n · LD A,(BC/DE/nn) · LD (BC/DE/nn),A',
		selectable: true
	},
	{ id: 'ld16-imm', name: 'LD rr,nn', description: '16-bit immediate loads', selectable: true },
	{
		id: 'ld16-mem',
		name: 'LD 16-bit ↔ memory',
		description: 'LD (nn),HL · LD HL,(nn) · LD (nn),rr · LD rr,(nn)',
		selectable: true
	},
	{
		id: 'ld-special',
		name: 'LD special',
		description: 'LD SP,HL · LD I,A · LD R,A · LD A,I · LD A,R',
		selectable: true
	},
	{ id: 'stack', name: 'PUSH / POP', description: 'Stack pushes and pops', selectable: true },
	{
		id: 'ex',
		name: 'EX / EXX',
		description: 'EX AF,AF′ · EXX · EX DE,HL · EX (SP),HL',
		selectable: true
	},
	{
		id: 'block-copy',
		name: 'Block copy',
		description: 'LDI · LDD · LDIR · LDDR',
		selectable: true
	},
	{
		id: 'block-cmp',
		name: 'Block compare',
		description: 'CPI · CPD · CPIR · CPDR',
		selectable: true
	},
	{
		id: 'alu8',
		name: '8-bit ALU',
		description: 'ADD/ADC/SUB/SBC/AND/XOR/OR/CP · NEG',
		selectable: true
	},
	{
		id: 'alu16',
		name: '16-bit ALU',
		description: 'ADD HL,rr · ADC HL,rr · SBC HL,rr',
		selectable: true
	},
	{
		id: 'incdec8',
		name: 'INC/DEC r',
		description: '8-bit register increment/decrement',
		selectable: true
	},
	{
		id: 'incdec16',
		name: 'INC/DEC rr',
		description: '16-bit register increment/decrement',
		selectable: true
	},
	{
		id: 'incdec-mem',
		name: 'INC/DEC (HL)',
		description: 'Read-modify-write of a memory byte',
		selectable: true
	},
	{
		id: 'acc-misc',
		name: 'DAA/CPL/SCF/CCF',
		description: 'Accumulator and carry adjustments',
		selectable: true
	},
	{
		id: 'rotate-a',
		name: 'RLCA/RRCA/RLA/RRA',
		description: 'Fast accumulator rotates',
		selectable: true
	},
	{
		id: 'rotate',
		name: 'Rotate/shift r',
		description: 'CB-page rotates and shifts on registers',
		selectable: true
	},
	{
		id: 'rotate-mem',
		name: 'Rotate/shift (HL)',
		description: 'CB-page rotates/shifts on memory · RLD · RRD',
		selectable: true
	},
	{ id: 'bit', name: 'BIT', description: 'Bit tests (no write)', selectable: true },
	{
		id: 'bit-set',
		name: 'SET/RES r',
		description: 'Set or clear a register bit',
		selectable: true
	},
	{
		id: 'bit-set-mem',
		name: 'SET/RES (HL)',
		description: 'Set or clear a memory bit',
		selectable: true
	},
	{
		id: 'jump',
		name: 'JP',
		description: 'Absolute jumps (JP nn · JP cc,nn · JP (HL))',
		selectable: true
	},
	{
		id: 'jump-rel',
		name: 'JR / DJNZ',
		description: 'Relative jumps and the DJNZ loop',
		selectable: true
	},
	{
		id: 'call-ret',
		name: 'CALL / RET',
		description: 'Subroutine calls and returns (CALL writes the stack)',
		selectable: true
	},
	{
		id: 'rst',
		name: 'RST',
		description: 'One-byte calls to fixed addresses (write the stack)',
		selectable: true
	},
	{
		id: 'io',
		name: 'IN / OUT',
		description: 'Port I/O (ports read 0 in the soup)',
		selectable: true
	},
	{
		id: 'block-io',
		name: 'Block I/O',
		description: 'INI/IND/INIR/INDR (write memory) · OUTI/OUTD/OTIR/OTDR',
		selectable: true
	},
	{ id: 'control', name: 'NOP/HALT/DI/EI/IM', description: 'CPU control', selectable: true },
	{
		id: 'undefined',
		name: 'Undefined ED',
		description: 'ED-page holes (execute as 2-byte NOPs)',
		selectable: false
	},
	{
		id: 'prefix',
		name: 'Prefix byte',
		description: 'CB · ED · DD · FD — not an instruction',
		selectable: false
	}
];

/** Pseudo-families resolved from instruction properties rather than a label. */
export const PSEUDO_FAMILIES = {
	/** Every instruction that can write memory (the strictest "no copy"). */
	'writes-mem': 'Every instruction that writes memory'
} as const;

export interface Instruction {
	page: Page;
	code: number;
	/** Generic mnemonic: immediates shown as n/nn/d, e.g. "LD (HL),n", "JR NZ,d". */
	mnemonic: string;
	family: FamilyId;
	/** Instruction length in bytes, including prefix. */
	length: number;
	/** True if executing it can write memory. */
	writesMem: boolean;
}

const R8 = ['B', 'C', 'D', 'E', 'H', 'L', '(HL)', 'A'];
const RP = ['BC', 'DE', 'HL', 'SP'];
const RP2 = ['BC', 'DE', 'HL', 'AF'];
const CC = ['NZ', 'Z', 'NC', 'C', 'PO', 'PE', 'P', 'M'];
const ALU = ['ADD A,', 'ADC A,', 'SUB ', 'SBC A,', 'AND ', 'XOR ', 'OR ', 'CP '];
const ROT = ['RLC', 'RRC', 'RL', 'RR', 'SLA', 'SRA', 'SLL', 'SRL'];
const ACC = ['RLCA', 'RRCA', 'RLA', 'RRA', 'DAA', 'CPL', 'SCF', 'CCF'];
const BLI = [
	['LDI', 'CPI', 'INI', 'OUTI'],
	['LDD', 'CPD', 'IND', 'OUTD'],
	['LDIR', 'CPIR', 'INIR', 'OTIR'],
	['LDDR', 'CPDR', 'INDR', 'OTDR']
];
const BLI_FAMILY: FamilyId[] = ['block-copy', 'block-cmp', 'block-io', 'block-io'];

function decodeBase(op: number): Instruction {
	const x = (op >> 6) & 3;
	const y = (op >> 3) & 7;
	const z = op & 7;
	const p = y >> 1;
	const q = y & 1;
	const I = (mnemonic: string, family: FamilyId, length = 1, writesMem = false): Instruction => ({
		page: 'base',
		code: op,
		mnemonic,
		family,
		length,
		writesMem
	});

	if (x === 0) {
		switch (z) {
			case 0:
				if (y === 0) return I('NOP', 'control');
				if (y === 1) return I("EX AF,AF'", 'ex');
				if (y === 2) return I('DJNZ d', 'jump-rel', 2);
				if (y === 3) return I('JR d', 'jump-rel', 2);
				return I(`JR ${CC[y - 4]},d`, 'jump-rel', 2);
			case 1:
				return q === 0 ? I(`LD ${RP[p]},nn`, 'ld16-imm', 3) : I(`ADD HL,${RP[p]}`, 'alu16');
			case 2:
				if (q === 0) {
					if (p === 0) return I('LD (BC),A', 'ld8-mem', 1, true);
					if (p === 1) return I('LD (DE),A', 'ld8-mem', 1, true);
					if (p === 2) return I('LD (nn),HL', 'ld16-mem', 3, true);
					return I('LD (nn),A', 'ld8-mem', 3, true);
				}
				if (p === 0) return I('LD A,(BC)', 'ld8-mem');
				if (p === 1) return I('LD A,(DE)', 'ld8-mem');
				if (p === 2) return I('LD HL,(nn)', 'ld16-mem', 3);
				return I('LD A,(nn)', 'ld8-mem', 3);
			case 3:
				return I(`${q === 0 ? 'INC' : 'DEC'} ${RP[p]}`, 'incdec16');
			case 4:
				return y === 6 ? I('INC (HL)', 'incdec-mem', 1, true) : I(`INC ${R8[y]}`, 'incdec8');
			case 5:
				return y === 6 ? I('DEC (HL)', 'incdec-mem', 1, true) : I(`DEC ${R8[y]}`, 'incdec8');
			case 6:
				return y === 6 ? I('LD (HL),n', 'ld8-mem', 2, true) : I(`LD ${R8[y]},n`, 'ld8-imm', 2);
			default:
				return I(ACC[y], y < 4 ? 'rotate-a' : 'acc-misc');
		}
	}
	if (x === 1) {
		if (y === 6 && z === 6) return I('HALT', 'control');
		if (y === 6) return I(`LD (HL),${R8[z]}`, 'ld8-mem', 1, true);
		if (z === 6) return I(`LD ${R8[y]},(HL)`, 'ld8-mem');
		return I(`LD ${R8[y]},${R8[z]}`, 'ld8');
	}
	if (x === 2) return I(`${ALU[y]}${R8[z]}`, 'alu8');
	// x === 3
	switch (z) {
		case 0:
			return I(`RET ${CC[y]}`, 'call-ret');
		case 1:
			if (q === 0) return I(`POP ${RP2[p]}`, 'stack');
			if (p === 0) return I('RET', 'call-ret');
			if (p === 1) return I('EXX', 'ex');
			if (p === 2) return I('JP (HL)', 'jump');
			return I('LD SP,HL', 'ld-special');
		case 2:
			return I(`JP ${CC[y]},nn`, 'jump', 3);
		case 3:
			if (y === 0) return I('JP nn', 'jump', 3);
			if (y === 1) return I('CB prefix', 'prefix');
			if (y === 2) return I('OUT (n),A', 'io', 2);
			if (y === 3) return I('IN A,(n)', 'io', 2);
			if (y === 4) return I('EX (SP),HL', 'ex', 1, true);
			if (y === 5) return I('EX DE,HL', 'ex');
			return I(y === 6 ? 'DI' : 'EI', 'control');
		case 4:
			return I(`CALL ${CC[y]},nn`, 'call-ret', 3, true);
		case 5:
			if (q === 0) return I(`PUSH ${RP2[p]}`, 'stack', 1, true);
			if (p === 0) return I('CALL nn', 'call-ret', 3, true);
			if (p === 1) return I('DD prefix', 'prefix');
			if (p === 2) return I('ED prefix', 'prefix');
			return I('FD prefix', 'prefix');
		case 6:
			return I(`${ALU[y]}n`, 'alu8', 2);
		default:
			return I(`RST ${(y * 8).toString(16).toUpperCase().padStart(2, '0')}`, 'rst', 1, true);
	}
}

function decodeCB(op: number): Instruction {
	const x = (op >> 6) & 3;
	const y = (op >> 3) & 7;
	const z = op & 7;
	const mem = z === 6;
	let mnemonic: string;
	let family: FamilyId;
	let writesMem = false;
	if (x === 0) {
		mnemonic = `${ROT[y]} ${R8[z]}`;
		family = mem ? 'rotate-mem' : 'rotate';
		writesMem = mem;
	} else if (x === 1) {
		mnemonic = `BIT ${y},${R8[z]}`;
		family = 'bit';
	} else {
		mnemonic = `${x === 2 ? 'RES' : 'SET'} ${y},${R8[z]}`;
		family = mem ? 'bit-set-mem' : 'bit-set';
		writesMem = mem;
	}
	return { page: 'cb', code: op, mnemonic, family, length: 2, writesMem };
}

function decodeED(op: number): Instruction {
	const x = (op >> 6) & 3;
	const y = (op >> 3) & 7;
	const z = op & 7;
	const p = y >> 1;
	const q = y & 1;
	const I = (mnemonic: string, family: FamilyId, length = 2, writesMem = false): Instruction => ({
		page: 'ed',
		code: op,
		mnemonic,
		family,
		length,
		writesMem
	});
	if (x === 1) {
		switch (z) {
			case 0:
				return I(y === 6 ? 'IN (C)' : `IN ${R8[y]},(C)`, 'io');
			case 1:
				return I(y === 6 ? 'OUT (C),0' : `OUT (C),${R8[y]}`, 'io');
			case 2:
				return I(`${q === 0 ? 'SBC' : 'ADC'} HL,${RP[p]}`, 'alu16');
			case 3:
				return q === 0
					? I(`LD (nn),${RP[p]}`, 'ld16-mem', 4, true)
					: I(`LD ${RP[p]},(nn)`, 'ld16-mem', 4);
			case 4:
				return I('NEG', 'alu8');
			case 5:
				return I(y === 1 ? 'RETI' : 'RETN', 'call-ret');
			case 6:
				return I(`IM ${[0, 0, 1, 2, 0, 0, 1, 2][y]}`, 'control');
			default:
				if (y === 0) return I('LD I,A', 'ld-special');
				if (y === 1) return I('LD R,A', 'ld-special');
				if (y === 2) return I('LD A,I', 'ld-special');
				if (y === 3) return I('LD A,R', 'ld-special');
				if (y === 4) return I('RRD', 'rotate-mem', 2, true);
				if (y === 5) return I('RLD', 'rotate-mem', 2, true);
				return I('NOP (ED)', 'undefined');
		}
	}
	if (x === 2 && z <= 3 && y >= 4) {
		// INI/IND/INIR/INDR write memory (from a port that reads 0); OUT* only read it.
		return I(BLI[y - 4][z], BLI_FAMILY[z], 2, z === 0 || z === 2);
	}
	return I('NOP (ED)', 'undefined');
}

/** All 768 page entries, indexed as PAGE_TABLES[page][code]. */
export const PAGE_TABLES: Readonly<Record<Page, readonly Instruction[]>> = {
	base: Array.from({ length: 256 }, (_, i) => decodeBase(i)),
	cb: Array.from({ length: 256 }, (_, i) => decodeCB(i)),
	ed: Array.from({ length: 256 }, (_, i) => decodeED(i))
};

/** Every real instruction (no prefix bytes, no ED holes). */
export const INSTRUCTIONS: readonly Instruction[] = (['base', 'cb', 'ed'] as Page[])
	.flatMap((pg) => PAGE_TABLES[pg])
	.filter((i) => i.family !== 'prefix' && i.family !== 'undefined');

export function instruction(page: Page, code: number): Instruction {
	return PAGE_TABLES[page][code & 0xff];
}

export function mnemonicOf(page: Page, code: number): string {
	return PAGE_TABLES[page][code & 0xff].mnemonic;
}

export function familyInfo(id: FamilyId): Family {
	return FAMILIES.find((f) => f.id === id)!;
}

export function instructionsOf(family: FamilyId): Instruction[] {
	return INSTRUCTIONS.filter((i) => i.family === family);
}

// ── Suppression resolution ──────────────────────────────────────────────────

export interface SuppressSets {
	base: Set<number>;
	cb: Set<number>;
	ed: Set<number>;
}

export function emptySuppression(): SuppressSets {
	return { base: new Set(), cb: new Set(), ed: new Set() };
}

export function countSuppressed(s: SuppressSets): number {
	return s.base.size + s.cb.size + s.ed.size;
}

export function suppressionEquals(a: SuppressSets, b: SuppressSets): boolean {
	for (const pg of ['base', 'cb', 'ed'] as Page[]) {
		if (a[pg].size !== b[pg].size) return false;
		for (const c of a[pg]) if (!b[pg].has(c)) return false;
	}
	return true;
}

/** Instructions a single pattern resolves to (see the grammar at the top of this file). */
export function matchPattern(pattern: string): Instruction[] {
	const raw = pattern.trim();
	if (!raw) return [];
	const lower = raw.toLowerCase();

	if (lower.startsWith('family:')) {
		const id = lower.slice('family:'.length).trim();
		if (id === 'writes-mem') return INSTRUCTIONS.filter((i) => i.writesMem);
		const fam = FAMILIES.find((f) => f.id === id);
		if (!fam || !fam.selectable) return [];
		return instructionsOf(fam.id);
	}

	const paged = /^(base|cb|ed):(?:0x)?([0-9a-f]{2})$/.exec(lower);
	if (paged) {
		const ins = PAGE_TABLES[paged[1] as Page][parseInt(paged[2], 16)];
		return ins.family === 'prefix' || ins.family === 'undefined' ? [] : [ins];
	}

	const hex = /^(?:0x)?([0-9a-f]{2})$/.exec(lower);
	if (hex) {
		const ins = PAGE_TABLES.base[parseInt(hex[1], 16)];
		return ins.family === 'prefix' ? [] : [ins];
	}

	// Legacy: case-insensitive substring of the generic mnemonic, across all
	// pages. Spaces after commas are ignored ("ADD A, B" == "ADD A,B") and the
	// old disassembler's "RST n" still means every RST.
	const upper = raw.toUpperCase().replace(/,\s+/g, ',');
	if (upper === 'RST N') return instructionsOf('rst');
	return INSTRUCTIONS.filter((i) => i.mnemonic.toUpperCase().includes(upper));
}

export function resolveSuppression(patterns: readonly string[]): SuppressSets {
	const s = emptySuppression();
	const adds = patterns.filter((p) => !p.trim().startsWith('-'));
	const subs = patterns.filter((p) => p.trim().startsWith('-')).map((p) => p.trim().slice(1));
	for (const pat of adds) for (const ins of matchPattern(pat)) s[ins.page].add(ins.code);
	for (const pat of subs) for (const ins of matchPattern(pat)) s[ins.page].delete(ins.code);
	return s;
}

/**
 * Pack the three sets into the GPU mask layout: 24 u32 words — base[0..8),
 * cb[8..16), ed[16..24); bit (code & 31) of word (code >> 5) within each page.
 */
export function suppressionMasks(s: SuppressSets): Uint32Array<ArrayBuffer> {
	const m = new Uint32Array(24);
	const put = (set: Set<number>, off: number) => {
		for (const c of set) m[off + (c >> 5)] |= 1 << (c & 31);
	};
	put(s.base, 0);
	put(s.cb, 8);
	put(s.ed, 16);
	return m;
}

/** Human summary such as "96 base · 12 ED · 0 CB". */
export function describeSuppression(s: SuppressSets): string {
	const parts: string[] = [];
	if (s.base.size) parts.push(`${s.base.size} base`);
	if (s.ed.size) parts.push(`${s.ed.size} ED`);
	if (s.cb.size) parts.push(`${s.cb.size} CB`);
	return parts.length ? parts.join(' · ') : 'none';
}

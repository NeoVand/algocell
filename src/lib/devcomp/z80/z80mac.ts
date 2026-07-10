// PHASE 2 core primitive — signed fixed-point MAC on a REAL Z80.
//
// The rule's whole datapath is one operation repeated: a signed Q8.8 dot product
// Σ w[k]·x[k] + bias, wide-accumulated in 32 bits then reduced to Q8.8. If THAT
// runs identically on a real Z80 (vs the bit-faithful `dotQ` reference), the
// full cell update is just looping it. So we build it bottom-up and validate
// each layer on the actual emulator:
//   umul16 : unsigned 16×16 → 32          (four 8×8 partials, expA's mul8)
//   smul16 : signed   16×16 → 32          (sign-magnitude around umul16)
//   dot    : Σ w·x + bias → Q8.8          (32-bit accumulate, >>8, low-16)
//
// The assembler (`z80asm.ts`) has no shift/CPL/16-bit-SBC — negation is
// complement (XOR 0xFF) + INC, shifts are ADD A,A / ADD HL,HL, exactly as expA.
//
//   npx tsx src/lib/devcomp/z80/z80mac.ts

import { assemble } from '../../morph/z80asm';
import { runOnRealZ80 } from '../../morph/dev/z80run';
import { fmt, toQ, fromQ, dotQ, type Fmt } from './fixed';

const F: Fmt = fmt(16, 8); // Q8.8

// ---- tape layout: scalars, then the two vectors, then the stack -----------
const MEMBYTES = 8192;
let p = 0x1000;
const alloc = (n: number): number => { const a = p; p += n; return a; };
const A = {
	N: alloc(1), WPTR: alloc(2), XPTR: alloc(2), BIAS: alloc(2), RESULT: alloc(2),
	WCUR: alloc(2), XCUR: alloc(2),
	SX: alloc(2), SY: alloc(2), MX: alloc(2), MY: alloc(2), MP: alloc(4),
	P0: alloc(2), P1: alloc(2), P2: alloc(2), P3: alloc(2),
	SP: alloc(4), ACC: alloc(4), SIGN: alloc(1), CROSSC: alloc(1), CNT: alloc(1),
	WARR: alloc(128), XARR: alloc(128)
};
const SYM: Record<string, number> = {
	...A,
	MX0: A.MX, MX1: A.MX + 1, MY0: A.MY, MY1: A.MY + 1,
	SX1: A.SX + 1, SY1: A.SY + 1,
	MP0: A.MP, MP1: A.MP + 1, MP2: A.MP + 2, MP3: A.MP + 3,
	SP0: A.SP, SP1: A.SP + 1, SP2: A.SP + 2, SP3: A.SP + 3,
	ACC0: A.ACC, ACC1: A.ACC + 1, ACC2: A.ACC + 2, ACC3: A.ACC + 3,
	BIAS0: A.BIAS, BIAS1: A.BIAS + 1,
	STACKTOP: MEMBYTES
};

// ---- shared subroutine library (one instruction per line) -----------------
const LIB = `
umul16:                          ; MX(2) * MY(2) -> MP(4), unsigned
	LD A,(MY0)
	LD E,A
	LD A,(MX0)
	CALL mul8
	LD (P0),HL                   ; Xl*Yl
	LD A,(MY1)
	LD E,A
	LD A,(MX0)
	CALL mul8
	LD (P1),HL                   ; Xl*Yh
	LD A,(MY0)
	LD E,A
	LD A,(MX1)
	CALL mul8
	LD (P2),HL                   ; Xh*Yl
	LD A,(MY1)
	LD E,A
	LD A,(MX1)
	CALL mul8
	LD (P3),HL                   ; Xh*Yh
	LD HL,(P0)                   ; MP = P0 (low 16), high 16 = 0
	LD (MP0),HL
	LD HL,0
	LD (MP2),HL
	LD HL,(P1)                   ; cross = P1 + P2 ; bit16 -> CROSSC
	LD DE,(P2)
	ADD HL,DE
	LD A,0
	ADC A,0
	LD (CROSSC),A
	LD A,(MP1)                   ; MP += cross << 8
	ADD A,L
	LD (MP1),A
	LD A,(MP2)
	ADC A,H
	LD (MP2),A
	LD A,(MP3)
	ADC A,0
	LD (MP3),A
	LD A,(MP3)                   ; MP3 += CROSSC (cross bit16 in byte3)
	LD B,A
	LD A,(CROSSC)
	ADD A,B
	LD (MP3),A
	LD HL,(P3)                   ; MP += P3 << 16 (bytes 2,3)
	LD DE,(MP2)
	ADD HL,DE
	LD (MP2),HL
	RET

smul16:                          ; SX(2) * SY(2) -> SP(4), signed (sign-magnitude)
	LD A,(SX1)
	AND 0x80
	LD (SIGN),A                  ; SIGN = SX sign bit
	LD DE,(SX)
	LD A,(SX1)
	AND 0x80
	JR Z, sx_pos
	LD A,E                       ; abs(SX): DE = ~DE + 1
	XOR 0xFF
	LD E,A
	LD A,D
	XOR 0xFF
	LD D,A
	INC DE
sx_pos:
	LD (MX),DE
	LD DE,(SY)
	LD A,(SY1)
	AND 0x80
	JR Z, sy_pos
	LD A,E                       ; abs(SY)
	XOR 0xFF
	LD E,A
	LD A,D
	XOR 0xFF
	LD D,A
	INC DE
	LD A,(SIGN)                  ; sign = SXsign XOR SYsign
	XOR 0x80
	LD (SIGN),A
sy_pos:
	LD (MY),DE
	CALL umul16
	LD A,(SIGN)
	AND 0x80
	JR Z, sp_pos
	LD A,(MP0)                   ; SP = -MP (complement 4 bytes, +1 with carry)
	XOR 0xFF
	LD (SP0),A
	LD A,(MP1)
	XOR 0xFF
	LD (SP1),A
	LD A,(MP2)
	XOR 0xFF
	LD (SP2),A
	LD A,(MP3)
	XOR 0xFF
	LD (SP3),A
	LD A,(SP0)
	ADD A,1
	LD (SP0),A
	LD A,(SP1)
	ADC A,0
	LD (SP1),A
	LD A,(SP2)
	ADC A,0
	LD (SP2),A
	LD A,(SP3)
	ADC A,0
	LD (SP3),A
	RET
sp_pos:
	LD HL,(MP0)
	LD (SP0),HL
	LD HL,(MP2)
	LD (SP2),HL
	RET

acc_add:                         ; ACC(4) += SP(4)
	LD A,(SP0)
	LD B,A
	LD A,(ACC0)
	ADD A,B
	LD (ACC0),A
	LD A,(SP1)
	LD B,A
	LD A,(ACC1)
	ADC A,B
	LD (ACC1),A
	LD A,(SP2)
	LD B,A
	LD A,(ACC2)
	ADC A,B
	LD (ACC2),A
	LD A,(SP3)
	LD B,A
	LD A,(ACC3)
	ADC A,B
	LD (ACC3),A
	RET

mul8:                            ; A * E -> HL  (8×8 -> 16 unsigned, MSB-first)
	LD HL,0
	LD D,0
	LD B,8
m8:
	ADD A,A
	JR NC, m8n
	ADD HL,HL
	ADD HL,DE
	JR m8c
m8n:
	ADD HL,HL
m8c:
	DJNZ m8
	RET
`;

// ---- Phase 2 main: dot product Σ w·x + bias -> RESULT ----------------------
const DOT_MAIN = `
	LD SP, STACKTOP
	LD A,0                       ; ACC = BIAS << 8 (sign-extended to 32 bits)
	LD (ACC0),A
	LD A,(BIAS0)
	LD (ACC1),A
	LD A,(BIAS1)
	LD (ACC2),A
	LD A,(BIAS1)                 ; sext -> ACC3
	ADD A,A
	LD A,0
	SBC A,0
	LD (ACC3),A
	LD HL,(WPTR)
	LD (WCUR),HL
	LD HL,(XPTR)
	LD (XCUR),HL
	LD A,(N)                     ; counter in MEMORY (mul8 clobbers B)
	LD (CNT),A
loop:
	LD HL,(WCUR)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (WCUR),HL
	LD (SX),DE
	LD HL,(XCUR)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (XCUR),HL
	LD (SY),DE
	CALL smul16
	CALL acc_add
	LD A,(CNT)
	DEC A
	LD (CNT),A
	JR NZ, loop
	LD HL,(ACC1)                 ; RESULT = low 16 of (ACC>>8) = bytes ACC1,ACC2
	LD (RESULT),HL
	HALT
`;

const CODE_DOT = assemble((DOT_MAIN + LIB).split('\n').map((l) => l.trim()), SYM);

// ---- JS harness -----------------------------------------------------------
function setI16(tape: Uint8Array, addr: number, v: number): void { tape[addr] = v & 0xff; tape[addr + 1] = (v >> 8) & 0xff; }
function getI16(tape: Uint8Array, addr: number): number { const u = tape[addr] | (tape[addr + 1] << 8); return u >= 0x8000 ? u - 0x10000 : u; }

/** Run the Z80 dot product on (w · x + bias), all Q8.8; returns the Q8.8 result int. */
function z80Dot(wQ: number[], xQ: number[], biasQ: number): number {
	const tape = new Uint8Array(MEMBYTES);
	tape.set(CODE_DOT, 0);
	tape[A.N] = wQ.length;
	setI16(tape, A.WPTR, A.WARR);
	setI16(tape, A.XPTR, A.XARR);
	setI16(tape, A.BIAS, biasQ);
	wQ.forEach((v, i) => setI16(tape, A.WARR + 2 * i, v));
	xQ.forEach((v, i) => setI16(tape, A.XARR + 2 * i, v));
	const { mem, halted } = runOnRealZ80(tape, MEMBYTES, 5_000_000);
	if (!halted) throw new Error('dot program did not HALT');
	return getI16(mem, A.RESULT);
}

console.log('=== Phase 2: signed fixed-point MAC on a REAL Z80 (Q8.8) ===');
console.log(`code ${CODE_DOT.length} B\n`);

// trace a trivial case: [1.0]·[1.0] + 0 should give 256 (=1.0)
{
	const tape = new Uint8Array(MEMBYTES);
	tape.set(CODE_DOT, 0);
	tape[A.N] = 1;
	setI16(tape, A.WPTR, A.WARR); setI16(tape, A.XPTR, A.XARR); setI16(tape, A.BIAS, 0);
	setI16(tape, A.WARR, 256); setI16(tape, A.XARR, 256);
	const { mem } = runOnRealZ80(tape, MEMBYTES, 5_000_000);
	const accBytes = [mem[A.ACC], mem[A.ACC + 1], mem[A.ACC + 2], mem[A.ACC + 3]];
	const spBytes = [mem[A.SP], mem[A.SP + 1], mem[A.SP + 2], mem[A.SP + 3]];
	console.log(`  trace 1.0·1.0+0: ACC bytes=[${accBytes.map((b) => b.toString(16).padStart(2, '0')).join(' ')}] SP=[${spBytes.map((b) => b.toString(16).padStart(2, '0')).join(' ')}] RESULT=${getI16(mem, A.RESULT)} (want 256)`);
	console.log(`  WARR=[${mem[A.WARR]} ${mem[A.WARR + 1]}] XARR=[${mem[A.XARR]} ${mem[A.XARR + 1]}] WCUR=${getI16(mem, A.WCUR)} (want ${A.WARR + 2})\n`);
}

// deterministic pseudo-random (no Math.random — keep it reproducible)
let seed = 12345;
const rnd = (): number => { seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff; return seed / 0x7fffffff; };

// (1) random signed dot products vs the fixed-point reference
let worst = 0, mism = 0, tested = 0;
for (let t = 0; t < 300; t++) {
	const n = 1 + (t % 48);
	const w: number[] = [], x: number[] = [];
	for (let k = 0; k < n; k++) { w.push(toQ((rnd() * 2 - 1) * 0.7, F)); x.push(toQ((rnd() * 2 - 1) * 10, F)); }
	const bias = toQ((rnd() * 2 - 1) * 2, F);
	const z = z80Dot(w, x, bias);
	const ref = dotQ(w, x, bias, F);
	tested++;
	if (z !== ref) { mism++; if (mism <= 4) console.log(`  MISMATCH n=${n}: z80=${z} ref=${ref} (Δ=${fromQ(z - ref, F).toFixed(4)})`); }
	worst = Math.max(worst, Math.abs(fromQ(z - ref, F)));
}
console.log(`[${mism === 0 ? 'PASS' : 'FAIL'}] Z80 dot == fixed-point reference on ${tested} random signed vectors (${mism} mismatches, worst Δ ${worst.toFixed(4)})`);

// (2) the real thing: an actual W1 row of the E1 gate · a real perceive vector
import { readFileSync } from 'node:fs';
import { EDIM, experimentById, seedGrid, perceive } from '../rule';
const cfg = EDIM;
const exp = experimentById('e1_gate')!;
const par = new Float64Array(JSON.parse(readFileSync(new URL('../params/e1_gate.json', import.meta.url), 'utf8')));
const s0 = seedGrid(cfg, exp, [0, 1]);
const percBuf = new Float64Array(cfg.PERC);
perceive(cfg, s0, exp.outputCells[0], percBuf);
const percQ = Array.from(percBuf, (v) => toQ(v, F));
let rowMism = 0, rowWorst = 0;
for (let hh = 0; hh < cfg.HD; hh++) {
	const wRow: number[] = [];
	for (let k = 0; k < cfg.PERC; k++) wRow.push(toQ(par[cfg.W1O + hh * cfg.PERC + k], F));
	const biasQ = toQ(par[cfg.B1O + hh], F);
	const z = z80Dot(wRow, percQ, biasQ);
	const ref = dotQ(wRow, percQ, biasQ, F);
	if (z !== ref) rowMism++;
	rowWorst = Math.max(rowWorst, Math.abs(fromQ(z - ref, F)));
}
console.log(`[${rowMism === 0 ? 'PASS' : 'FAIL'}] Z80 dot == reference on all ${cfg.HD} real W1 rows · real perceive (${rowMism} mismatches, worst Δ ${rowWorst.toFixed(4)})`);

if (mism || rowMism) { console.error('\nFAIL: the MAC does not run identically on the Z80.'); process.exit(1); }
console.log('\nPASS: the rule\'s core arithmetic — a signed fixed-point MAC — runs bit-exactly on a real Z80.');
console.log('The full cell update is this loop, wrapped over perceive + the hidden/output layers (Phase 2b).');

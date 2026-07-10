// PHASE 2b — ONE full cell update of the trained rule on a REAL Z80.
//
// perceive([id,gx,gy,lap] per channel) → h = relu(W1·perc + b1) → dl = W2·h + b2
// → state' = tanh(state + dl), all signed Q8.8, weights in memory, tanh a LUT.
// Wraps the Phase-2 signed MAC (`dotsub`) over 48 hidden + 12 output units and
// adds perceive (needs a signed >>1 — the SRA/RR we added to the assembler) and
// a direct tanh lookup. Validated on real neighbourhoods from the E1 rollout
// against the bit-faithful fixed reference (EXACT) and f64 (quantization error).
//
//   npx tsx src/lib/devcomp/z80/z80cell.ts

import { readFileSync } from 'node:fs';
import { assemble } from '../../morph/z80asm';
import { runOnRealZ80 } from '../../morph/dev/z80run';
import { EDIM, experimentById, seedGrid, step, type RuleConfig } from '../rule';
import { fmt, toQ, fromQ, dotQ, reluQ, type Fmt } from './fixed';

const cfg = EDIM;
const exp = experimentById('e1_gate')!;
const par = new Float64Array(JSON.parse(readFileSync(new URL('../params/e1_gate.json', import.meta.url), 'utf8')));
const F: Fmt = fmt(16, 8);
const { C, HD, PERC, FEAT, W1O, B1O, W2O, B2O } = cfg;

// ---- tanh LUT: index = pre + LUTHALF, covers pre ∈ [−LUTHALF, LUTHALF−1] ----
const LUTHALF = 4096; // ±16.0 in Q8.8 — |state+dl| ≤ ~10.3, so no clamp needed
const LUTN = 2 * LUTHALF;
const TANH = new Int16Array(LUTN);
for (let i = 0; i < LUTN; i++) TANH[i] = toQ(Math.tanh(fromQ(i - LUTHALF, F)), F);
const tanhLUT = (pre: number): number => TANH[Math.max(0, Math.min(LUTN - 1, pre + LUTHALF))];

// ---- bit-faithful TS reference for ONE cell (matches the Z80 exactly) ------
function cellFixedLUT(nb: { self: number[]; r: number[]; l: number[]; u: number[]; d: number[] }): number[] {
	const perc: number[] = new Array(PERC);
	for (let ch = 0; ch < C; ch++) {
		const b = ch * FEAT;
		perc[b] = nb.self[ch];
		perc[b + 1] = (nb.r[ch] - nb.l[ch]) >> 1;
		perc[b + 2] = (nb.d[ch] - nb.u[ch]) >> 1;
		perc[b + 3] = nb.r[ch] + nb.l[ch] + nb.u[ch] + nb.d[ch] - 4 * nb.self[ch];
	}
	const h: number[] = new Array(HD);
	for (let hh = 0; hh < HD; hh++) {
		const w: number[] = [];
		for (let k = 0; k < PERC; k++) w.push(toQ(par[W1O + hh * PERC + k], F));
		h[hh] = reluQ(dotQ(w, perc, toQ(par[B1O + hh], F), F));
	}
	const out: number[] = new Array(C);
	for (let c = 0; c < C; c++) {
		const w: number[] = [];
		for (let hh = 0; hh < HD; hh++) w.push(toQ(par[W2O + c * HD + hh], F));
		const dl = dotQ(w, h, toQ(par[B2O + c], F), F);
		out[c] = tanhLUT(nb.self[c] + dl);
	}
	return out;
}

// ---- memory map -----------------------------------------------------------
const MEMBYTES = 32768;
let p = 0x1000;
const alloc = (n: number): number => { const a = p; p += n; return a; };
// MAC/dot scratch (Phase 2)
const WPTR = alloc(2), XPTR = alloc(2), BIAS = alloc(2), RESULT = alloc(2), N = alloc(1);
const WCUR = alloc(2), XCUR = alloc(2), SX = alloc(2), SY = alloc(2), MX = alloc(2), MY = alloc(2);
const MP = alloc(4), P0 = alloc(2), P1 = alloc(2), P2 = alloc(2), P3 = alloc(2), SP = alloc(4), ACC = alloc(4);
const SIGN = alloc(1), CROSSC = alloc(1), CNT = alloc(1);
// cell state
const SELF = alloc(C * 2), NR = alloc(C * 2), NL = alloc(C * 2), NU = alloc(C * 2), ND = alloc(C * 2);
const PERCB = alloc(PERC * 2), HBUF = alloc(HD * 2), OUT = alloc(C * 2);
const pS = alloc(2), pR = alloc(2), pL = alloc(2), pU = alloc(2), pD = alloc(2), pP = alloc(2);
const T_SELF = alloc(2), T_SR = alloc(2), T_SL = alloc(2), T_SU = alloc(2), T_SD = alloc(2), T4 = alloc(2);
const WROW = alloc(2), BROW = alloc(2), HOUT = alloc(2), COUT = alloc(2), PCNT = alloc(1), HCNT = alloc(1), CCNT = alloc(1), STASH = alloc(2), PRE = alloc(2);
// weights + tanh table
const W1DAT = alloc(HD * PERC * 2), B1DAT = alloc(HD * 2), W2DAT = alloc(C * HD * 2), B2DAT = alloc(C * 2);
const TANHLUT = alloc(LUTN * 2);

const SYM: Record<string, number> = {
	WPTR, XPTR, BIAS, RESULT, N, WCUR, XCUR, SX, SY, MX, MY, MP, P0, P1, P2, P3, SP, ACC, SIGN, CROSSC, CNT,
	SELF, NR, NL, NU, ND, PERCB, HBUF, OUT, pS, pR, pL, pU, pD, pP,
	T_SELF, T_SR, T_SL, T_SU, T_SD, T4, WROW, BROW, HOUT, COUT, PCNT, HCNT, CCNT, STASH, PRE,
	W1DAT, B1DAT, W2DAT, B2DAT,
	MX0: MX, MX1: MX + 1, MY0: MY, MY1: MY + 1, SX1: SX + 1, SY1: SY + 1,
	MP0: MP, MP1: MP + 1, MP2: MP + 2, MP3: MP + 3, SP0: SP, SP1: SP + 1, SP2: SP + 2, SP3: SP + 3,
	ACC0: ACC, ACC1: ACC + 1, ACC2: ACC + 2, ACC3: ACC + 3, BIAS0: BIAS, BIAS1: BIAS + 1,
	TANHBASE2: TANHLUT + LUTHALF * 2, ROWBYTES: PERC * 2,
	STACKTOP: MEMBYTES
};

// ---- the program ----------------------------------------------------------
const MAIN = `
	LD SP, STACKTOP
	; ---- perceive: for each of C channels, fill 4 features ----
	LD HL,SELF
	LD (pS),HL
	LD HL,NR
	LD (pR),HL
	LD HL,NL
	LD (pL),HL
	LD HL,NU
	LD (pU),HL
	LD HL,ND
	LD (pD),HL
	LD HL,PERCB
	LD (pP),HL
	LD A,${C}
	LD (PCNT),A
p_loop:
	LD HL,(pS)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (pS),HL
	LD (T_SELF),DE
	LD HL,(pR)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (pR),HL
	LD (T_SR),DE
	LD HL,(pL)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (pL),HL
	LD (T_SL),DE
	LD HL,(pU)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (pU),HL
	LD (T_SU),DE
	LD HL,(pD)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (pD),HL
	LD (T_SD),DE
	; feature 0 = self
	LD HL,(pP)
	LD DE,(T_SELF)
	LD (HL),E
	INC HL
	LD (HL),D
	INC HL
	; feature 1 = (sr - sl) >> 1
	LD HL,(T_SR)
	LD DE,(T_SL)
	LD A,L
	SUB E
	LD L,A
	LD A,H
	SBC A,D
	LD H,A
	SRA H
	RR L
	LD (STASH),HL
	LD HL,(pP)
	INC HL
	INC HL
	LD DE,(STASH)
	LD (HL),E
	INC HL
	LD (HL),D
	; feature 2 = (sd - su) >> 1
	LD HL,(T_SD)
	LD DE,(T_SU)
	LD A,L
	SUB E
	LD L,A
	LD A,H
	SBC A,D
	LD H,A
	SRA H
	RR L
	LD (STASH),HL
	LD HL,(pP)
	INC HL
	INC HL
	INC HL
	INC HL
	LD DE,(STASH)
	LD (HL),E
	INC HL
	LD (HL),D
	; feature 3 = sr + sl + su + sd - 4*self
	LD HL,(T_SELF)
	ADD HL,HL
	ADD HL,HL
	LD (T4),HL
	LD HL,(T_SR)
	LD DE,(T_SL)
	ADD HL,DE
	LD DE,(T_SU)
	ADD HL,DE
	LD DE,(T_SD)
	ADD HL,DE
	LD DE,(T4)
	LD A,L
	SUB E
	LD L,A
	LD A,H
	SBC A,D
	LD H,A
	LD (STASH),HL
	LD HL,(pP)
	INC HL
	INC HL
	INC HL
	INC HL
	INC HL
	INC HL
	LD DE,(STASH)
	LD (HL),E
	INC HL
	LD (HL),D
	; advance pP by 8 (4 features × 2 bytes)
	LD HL,(pP)
	LD DE,8
	ADD HL,DE
	LD (pP),HL
	LD A,(PCNT)
	DEC A
	LD (PCNT),A
	JP NZ, p_loop

	; ---- hidden: h[hh] = relu(dot(W1 row, PERC, b1[hh])) ----
	LD HL,W1DAT
	LD (WROW),HL
	LD HL,B1DAT
	LD (BROW),HL
	LD HL,HBUF
	LD (HOUT),HL
	LD A,${HD}
	LD (HCNT),A
h_loop:
	LD HL,(WROW)
	LD (WPTR),HL
	LD HL,PERCB
	LD (XPTR),HL
	LD A,${PERC}
	LD (N),A
	LD HL,(BROW)
	LD E,(HL)
	INC HL
	LD D,(HL)
	LD (BIAS),DE
	CALL dotsub
	LD HL,(RESULT)
	LD A,H
	AND 0x80
	JR Z, h_pos
	LD HL,0
h_pos:
	LD (STASH),HL
	LD HL,(HOUT)
	LD DE,(STASH)
	LD (HL),E
	INC HL
	LD (HL),D
	INC HL
	LD (HOUT),HL
	LD HL,(WROW)
	LD DE,ROWBYTES
	ADD HL,DE
	LD (WROW),HL
	LD HL,(BROW)
	INC HL
	INC HL
	LD (BROW),HL
	LD A,(HCNT)
	DEC A
	LD (HCNT),A
	JP NZ, h_loop

	; ---- output: out[c] = tanh(SELF[c] + dot(W2 row, H, b2[c])) ----
	LD HL,W2DAT
	LD (WROW),HL
	LD HL,B2DAT
	LD (BROW),HL
	LD HL,OUT
	LD (COUT),HL
	LD HL,SELF
	LD (pS),HL
	LD A,${C}
	LD (CCNT),A
c_loop:
	LD HL,(WROW)
	LD (WPTR),HL
	LD HL,HBUF
	LD (XPTR),HL
	LD A,${HD}
	LD (N),A
	LD HL,(BROW)
	LD E,(HL)
	INC HL
	LD D,(HL)
	LD (BIAS),DE
	CALL dotsub
	; pre = SELF[c] + dl
	LD HL,(pS)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (pS),HL
	LD HL,(RESULT)
	ADD HL,DE
	; tanh: ptr = TANHBASE2 + 2*pre
	ADD HL,HL
	LD DE,TANHBASE2
	ADD HL,DE
	LD E,(HL)
	INC HL
	LD D,(HL)
	LD (PRE),DE
	LD HL,(COUT)
	LD DE,(PRE)
	LD (HL),E
	INC HL
	LD (HL),D
	INC HL
	LD (COUT),HL
	LD HL,(WROW)
	LD DE,ROWBYTES
	ADD HL,DE
	LD (WROW),HL
	LD HL,(BROW)
	INC HL
	INC HL
	LD (BROW),HL
	LD A,(CCNT)
	DEC A
	LD (CCNT),A
	JP NZ, c_loop
	HALT

dotsub:                          ; WPTR·XPTR (N terms, Q8.8) + BIAS -> RESULT
	LD A,0
	LD (ACC0),A
	LD A,(BIAS0)
	LD (ACC1),A
	LD A,(BIAS1)
	LD (ACC2),A
	LD A,(BIAS1)
	ADD A,A
	LD A,0
	SBC A,0
	LD (ACC3),A
	LD HL,(WPTR)
	LD (WCUR),HL
	LD HL,(XPTR)
	LD (XCUR),HL
	LD A,(N)
	LD (CNT),A
dloop:
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
	JR NZ, dloop
	LD HL,(ACC1)
	LD (RESULT),HL
	RET

umul16:
	LD A,(MY0)
	LD E,A
	LD A,(MX0)
	CALL mul8
	LD (P0),HL
	LD A,(MY1)
	LD E,A
	LD A,(MX0)
	CALL mul8
	LD (P1),HL
	LD A,(MY0)
	LD E,A
	LD A,(MX1)
	CALL mul8
	LD (P2),HL
	LD A,(MY1)
	LD E,A
	LD A,(MX1)
	CALL mul8
	LD (P3),HL
	LD HL,(P0)
	LD (MP0),HL
	LD HL,0
	LD (MP2),HL
	LD HL,(P1)
	LD DE,(P2)
	ADD HL,DE
	LD A,0
	ADC A,0
	LD (CROSSC),A
	LD A,(MP1)
	ADD A,L
	LD (MP1),A
	LD A,(MP2)
	ADC A,H
	LD (MP2),A
	LD A,(MP3)
	ADC A,0
	LD (MP3),A
	LD A,(MP3)
	LD B,A
	LD A,(CROSSC)
	ADD A,B
	LD (MP3),A
	LD HL,(P3)
	LD DE,(MP2)
	ADD HL,DE
	LD (MP2),HL
	RET

smul16:
	LD A,(SX1)
	AND 0x80
	LD (SIGN),A
	LD DE,(SX)
	LD A,(SX1)
	AND 0x80
	JR Z, sx_pos
	LD A,E
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
	LD A,E
	XOR 0xFF
	LD E,A
	LD A,D
	XOR 0xFF
	LD D,A
	INC DE
	LD A,(SIGN)
	XOR 0x80
	LD (SIGN),A
sy_pos:
	LD (MY),DE
	CALL umul16
	LD A,(SIGN)
	AND 0x80
	JR Z, sp_pos
	LD A,(MP0)
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

acc_add:
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

mul8:
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

const CODE = assemble(MAIN.split('\n').map((l) => l.trim()), SYM);
console.log('=== Phase 2b: ONE full cell update of the trained rule on a REAL Z80 ===');
console.log(`code ${CODE.length} B, tape ${MEMBYTES} B\n`);
if (CODE.length >= 0x1000) throw new Error(`code (${CODE.length}) overruns data at 0x1000`);

function setI16(t: Uint8Array, a: number, v: number): void { t[a] = v & 0xff; t[a + 1] = (v >> 8) & 0xff; }
function getI16(t: Uint8Array, a: number): number { const u = t[a] | (t[a + 1] << 8); return u >= 0x8000 ? u - 0x10000 : u; }

function buildTape(): Uint8Array {
	const t = new Uint8Array(MEMBYTES);
	t.set(CODE, 0);
	for (let hh = 0; hh < HD; hh++) for (let k = 0; k < PERC; k++) setI16(t, W1DAT + (hh * PERC + k) * 2, toQ(par[W1O + hh * PERC + k], F));
	for (let hh = 0; hh < HD; hh++) setI16(t, B1DAT + hh * 2, toQ(par[B1O + hh], F));
	for (let c = 0; c < C; c++) for (let hh = 0; hh < HD; hh++) setI16(t, W2DAT + (c * HD + hh) * 2, toQ(par[W2O + c * HD + hh], F));
	for (let c = 0; c < C; c++) setI16(t, B2DAT + c * 2, toQ(par[B2O + c], F));
	for (let i = 0; i < LUTN; i++) setI16(t, TANHLUT + i * 2, TANH[i]);
	return t;
}
const TAPE0 = buildTape();

function z80Cell(nb: { self: number[]; r: number[]; l: number[]; u: number[]; d: number[] }): number[] {
	const t = TAPE0.slice();
	for (let ch = 0; ch < C; ch++) {
		setI16(t, SELF + ch * 2, nb.self[ch]); setI16(t, NR + ch * 2, nb.r[ch]);
		setI16(t, NL + ch * 2, nb.l[ch]); setI16(t, NU + ch * 2, nb.u[ch]); setI16(t, ND + ch * 2, nb.d[ch]);
	}
	const { mem, halted } = runOnRealZ80(t, MEMBYTES, 20_000_000);
	if (!halted) throw new Error('cell program did not HALT');
	const out: number[] = [];
	for (let ch = 0; ch < C; ch++) out.push(getI16(mem, OUT + ch * 2));
	return out;
}

// neighbourhoods from a REAL E1 rollout (seed → step), quantized to Q8.8
function neighborhood(s: Float64Array, i: number): { self: number[]; r: number[]; l: number[]; u: number[]; d: number[] } {
	const g = (cell: number): number[] => Array.from({ length: C }, (_, ch) => toQ(s[cell * C + ch], F));
	return { self: g(i), r: g(i + 1), l: g(i - 1), u: g(i - cfg.SW), d: g(i + cfg.SW) };
}

// (A) per-cell exactness: Z80 cell vs fixed reference across real neighbourhoods
const testCells = [exp.outputCells[0], exp.inputCells[0] + 1, 4 * cfg.SW + 4, 3 * cfg.SW + 5, 5 * cfg.SW + 6, 2 * cfg.SW + 3];
let exactMism = 0, tested = 0, worstVsRef = 0, worstVsF64 = 0;
for (const inp of exp.cases.map((c) => c.in)) {
	let s = seedGrid(cfg, exp, inp);
	for (let stp = 0; stp < 6; stp++) {
		const nxtF64 = step(cfg, par, s, exp, inp);
		for (const i of testCells) {
			const nb = neighborhood(s, i);
			const z = z80Cell(nb);
			const ref = cellFixedLUT(nb);
			tested++;
			for (let ch = 0; ch < C; ch++) {
				if (z[ch] !== ref[ch]) exactMism++;
				worstVsRef = Math.max(worstVsRef, Math.abs(fromQ(z[ch] - ref[ch], F)));
				worstVsF64 = Math.max(worstVsF64, Math.abs(fromQ(z[ch], F) - nxtF64[i * C + ch]));
			}
		}
		s = nxtF64;
	}
}
console.log(`(A) Z80 cell vs bit-faithful fixed reference: ${exactMism === 0 ? 'PASS (exact)' : `FAIL (${exactMism} channel mismatches)`}  over ${tested} cell updates, worst Δ ${worstVsRef.toFixed(5)}`);
console.log(`    Z80 cell vs f64 rule (quantization error): worst |Δ| = ${worstVsF64.toExponential(2)} per channel  (Q8.8 res ${(1 / 256).toExponential(2)})`);
if (exactMism) { console.error('\nFAIL: the full cell does not run identically on the Z80.'); process.exit(1); }

// (B) WHOLE COMPUTER on the Z80: sweep the cell over every interior cell for T
// steps and read the XOR truth table. Every cell update is executed by the real
// Z80. Memoized by neighbourhood — early uniform fields repeat, so it stays fast.
const cellCache = new Map<string, number[]>();
function z80CellMemo(nb: { self: number[]; r: number[]; l: number[]; u: number[]; d: number[] }): number[] {
	const key = [nb.self, nb.r, nb.l, nb.u, nb.d].map((v) => v.join(',')).join('|');
	let out = cellCache.get(key);
	if (!out) { out = z80Cell(nb); cellCache.set(key, out); }
	return out;
}
function z80Rollout(inputs: number[], steps: number): Int32Array {
	const { SW, SH, N } = cfg;
	let s = new Int32Array(N * C);
	const s0 = seedGrid(cfg, exp, inputs);
	for (let j = 0; j < s.length; j++) s[j] = toQ(s0[j], F);
	for (let t = 0; t < steps; t++) {
		const ns = new Int32Array(N * C); // border stays 0
		for (let y = 1; y < SH - 1; y++)
			for (let x = 1; x < SW - 1; x++) {
				const i = y * SW + x;
				const g = (cell: number): number[] => Array.from({ length: C }, (_, ch) => s[cell * C + ch]);
				const out = z80CellMemo({ self: g(i), r: g(i + 1), l: g(i - 1), u: g(i - SW), d: g(i + SW) });
				for (let ch = 0; ch < C; ch++) ns[i * C + ch] = out[ch];
			}
		for (let k = 0; k < exp.inputCells.length; k++) ns[exp.inputCells[k] * C + 0] = toQ(inputs[k], F);
		s = ns;
	}
	return s;
}

console.log('\n(B) the WHOLE XOR gate, every cell update executed on the real Z80:');
console.log('    case      out(Z80)   class   expected');
let truthOk = true;
const t0 = tested; // reuse var namespace note
for (const cs of exp.cases) {
	const s = z80Rollout(cs.in, exp.tGrow);
	const outQ = s[exp.outputCells[0] * C + 0];
	const out = fromQ(outQ, F);
	const cls = out > 0.5 ? 1 : 0;
	if (cls !== cs.out[0]) truthOk = false;
	console.log(`    [${cs.in.join(',')}] → ${out.toFixed(4).padStart(8)}     ${cls}       ${cs.out[0]}`);
}
console.log(`    truth table (in-substrate): ${truthOk ? 'PASS' : 'FAIL'}   (${cellCache.size} distinct cell updates cached)`);

if (!truthOk) { console.error('\nFAIL: the in-substrate gate did not compute XOR.'); process.exit(1); }
console.log('\nPASS: the trained developmental rule — grown by gradient descent — computes XOR');
console.log('as a real Z80 program. The learned computer is literally a program in a real ISA.');

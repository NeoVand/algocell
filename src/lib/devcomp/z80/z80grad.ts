// PHASE 3 — the training GRADIENT through the running Z80, for the real rule.
//
// Forward-mode AD: every value carries a tangent v̇ = d(value)/dθ for one weight
// θ. The cell update becomes a DUAL cell: dualdot accumulates the value Σ wv·xv
// and the product-rule tangent Σ(wv·xd + wd·xv); ReLU gates both; tanh maps the
// value through the LUT and the tangent through a (1−tanh²) derivative LUT. We
// seed θ's tangent = 1 (all field tangents 0, since the seed is constant in θ),
// run ONE cell on the real Z80, and check d(out)/dθ vs finite differences — the
// exact test Exp A ran for a single multiply, now for the whole trained MLP rule.
//
//   npx tsx src/lib/devcomp/z80/z80grad.ts

import { readFileSync } from 'node:fs';
import { assemble } from '../../morph/z80asm';
import { runOnRealZ80 } from '../../morph/dev/z80run';
import { EDIM, experimentById, seedGrid, step, readOutputs } from '../rule';
import { fmt, toQ, fromQ, type Fmt } from './fixed';

const cfg = EDIM;
const exp = experimentById('e1_gate')!;
const par = new Float64Array(JSON.parse(readFileSync(new URL('../params/e1_gate.json', import.meta.url), 'utf8')));
const F: Fmt = fmt(16, 8);
const { C, HD, PERC, FEAT, W1O, B1O, W2O, B2O } = cfg;

// tanh + derivative LUTs (index = pre + LUTHALF)
const LUTHALF = 4096, LUTN = 2 * LUTHALF;
const TANH = new Int16Array(LUTN), DERIV = new Int16Array(LUTN);
for (let i = 0; i < LUTN; i++) { const th = Math.tanh(fromQ(i - LUTHALF, F)); TANH[i] = toQ(th, F); DERIV[i] = toQ(1 - th * th, F); }

// ---- memory map -----------------------------------------------------------
const MEMBYTES = 65536;
let p = 0x1000;
const alloc = (n: number): number => { const a = p; p += n; return a; };
// dual-dot scratch
const WVPTR = alloc(2), WDPTR = alloc(2), XVPTR = alloc(2), XDPTR = alloc(2), N = alloc(1);
const BIASV = alloc(2), BIASD = alloc(2), RESULT = alloc(2), RESULTD = alloc(2);
const WVCUR = alloc(2), WDCUR = alloc(2), XVCUR = alloc(2), XDCUR = alloc(2);
const TWV = alloc(2), TWD = alloc(2), TXV = alloc(2), TXD = alloc(2);
const SX = alloc(2), SY = alloc(2), MX = alloc(2), MY = alloc(2), MP = alloc(4);
const P0 = alloc(2), P1 = alloc(2), P2 = alloc(2), P3 = alloc(2), SP = alloc(4);
const ACC = alloc(4), ACCD = alloc(4), SIGN = alloc(1), CROSSC = alloc(1), CNT = alloc(1);
// cell state (values + tangents)
const SELF = alloc(C * 2), NR = alloc(C * 2), NL = alloc(C * 2), NU = alloc(C * 2), ND = alloc(C * 2);
const PERCB = alloc(PERC * 2), PERCD = alloc(PERC * 2), HVAL = alloc(HD * 2), HTAN = alloc(HD * 2), OUTV = alloc(C * 2), OUTD = alloc(C * 2);
const pS = alloc(2), pR = alloc(2), pL = alloc(2), pU = alloc(2), pD = alloc(2), pP = alloc(2);
const T_SELF = alloc(2), T_SR = alloc(2), T_SL = alloc(2), T_SU = alloc(2), T_SD = alloc(2), T4 = alloc(2);
const WVROW = alloc(2), WDROW = alloc(2), BVROW = alloc(2), BDROW = alloc(2), HOUTV = alloc(2), HOUTT = alloc(2), COUTV = alloc(2), COUTD = alloc(2);
const PCNT = alloc(1), HCNT = alloc(1), CCNT = alloc(1), STASH = alloc(2), PREV = alloc(2);
// weights (value + tangent) + LUTs
const W1V = alloc(HD * PERC * 2), W1D = alloc(HD * PERC * 2), B1V = alloc(HD * 2), B1D = alloc(HD * 2);
const W2V = alloc(C * HD * 2), W2D = alloc(C * HD * 2), B2V = alloc(C * 2), B2D = alloc(C * 2);
const TANHLUT = alloc(LUTN * 2), DERIVLUT = alloc(LUTN * 2);

const SYM: Record<string, number> = {
	WVPTR, WDPTR, XVPTR, XDPTR, N, BIASV, BIASD, RESULT, RESULTD, WVCUR, WDCUR, XVCUR, XDCUR, TWV, TWD, TXV, TXD,
	SX, SY, MX, MY, MP, P0, P1, P2, P3, SP, ACC, ACCD, SIGN, CROSSC, CNT,
	SELF, NR, NL, NU, ND, PERCB, PERCD, HVAL, HTAN, OUTV, OUTD, pS, pR, pL, pU, pD, pP,
	T_SELF, T_SR, T_SL, T_SU, T_SD, T4, WVROW, WDROW, BVROW, BDROW, HOUTV, HOUTT, COUTV, COUTD, PCNT, HCNT, CCNT, STASH, PREV,
	W1V, W1D, B1V, B1D, W2V, W2D, B2V, B2D,
	MX0: MX, MX1: MX + 1, MY0: MY, MY1: MY + 1, SX1: SX + 1, SY1: SY + 1,
	MP0: MP, MP1: MP + 1, MP2: MP + 2, MP3: MP + 3, SP0: SP, SP1: SP + 1, SP2: SP + 2, SP3: SP + 3,
	ACC0: ACC, ACC1: ACC + 1, ACC2: ACC + 2, ACC3: ACC + 3, ACCD0: ACCD, ACCD1: ACCD + 1, ACCD2: ACCD + 2, ACCD3: ACCD + 3,
	BV0: BIASV, BV1: BIASV + 1, BD0: BIASD, BD1: BIASD + 1,
	TANHBASE2: TANHLUT + LUTHALF * 2, DERIVBASE2: DERIVLUT + LUTHALF * 2, ROWBYTES: PERC * 2,
	STACKTOP: 0xff00
};

const MAIN = `
	LD SP, STACKTOP
	; ---- perceive VALUES -> PERCB (tangents PERCD stay 0: seed is constant in θ) ----
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
	LD HL,(pP)
	LD DE,(T_SELF)
	LD (HL),E
	INC HL
	LD (HL),D
	INC HL
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
	LD HL,(pP)
	LD DE,8
	ADD HL,DE
	LD (pP),HL
	LD A,(PCNT)
	DEC A
	LD (PCNT),A
	JP NZ, p_loop

	; ---- hidden: (hv,hd) = relu(dualdot(W1 row, PERC, b1)) ----
	LD HL,W1V
	LD (WVROW),HL
	LD HL,W1D
	LD (WDROW),HL
	LD HL,B1V
	LD (BVROW),HL
	LD HL,B1D
	LD (BDROW),HL
	LD HL,HVAL
	LD (HOUTV),HL
	LD HL,HTAN
	LD (HOUTT),HL
	LD A,${HD}
	LD (HCNT),A
h_loop:
	LD HL,(WVROW)
	LD (WVPTR),HL
	LD HL,(WDROW)
	LD (WDPTR),HL
	LD HL,PERCB
	LD (XVPTR),HL
	LD HL,PERCD
	LD (XDPTR),HL
	LD A,${PERC}
	LD (N),A
	LD HL,(BVROW)
	LD E,(HL)
	INC HL
	LD D,(HL)
	LD (BIASV),DE
	LD HL,(BDROW)
	LD E,(HL)
	INC HL
	LD D,(HL)
	LD (BIASD),DE
	CALL dualdot
	LD HL,(RESULT)
	LD A,H
	AND 0x80
	JR Z, hrelu_pos
	LD HL,0
	LD (RESULT),HL
	LD (RESULTD),HL
hrelu_pos:
	LD HL,(RESULT)
	LD (STASH),HL
	LD HL,(HOUTV)
	LD DE,(STASH)
	LD (HL),E
	INC HL
	LD (HL),D
	INC HL
	LD (HOUTV),HL
	LD HL,(RESULTD)
	LD (STASH),HL
	LD HL,(HOUTT)
	LD DE,(STASH)
	LD (HL),E
	INC HL
	LD (HL),D
	INC HL
	LD (HOUTT),HL
	LD HL,(WVROW)
	LD DE,ROWBYTES
	ADD HL,DE
	LD (WVROW),HL
	LD HL,(WDROW)
	LD DE,ROWBYTES
	ADD HL,DE
	LD (WDROW),HL
	LD HL,(BVROW)
	INC HL
	INC HL
	LD (BVROW),HL
	LD HL,(BDROW)
	INC HL
	INC HL
	LD (BDROW),HL
	LD A,(HCNT)
	DEC A
	LD (HCNT),A
	JP NZ, h_loop

	; ---- output: outv = tanh(self+dl.v) ; outd = (1-tanh^2)(self+dl.v) * dl.d ----
	LD HL,W2V
	LD (WVROW),HL
	LD HL,W2D
	LD (WDROW),HL
	LD HL,B2V
	LD (BVROW),HL
	LD HL,B2D
	LD (BDROW),HL
	LD HL,OUTV
	LD (COUTV),HL
	LD HL,OUTD
	LD (COUTD),HL
	LD HL,SELF
	LD (pS),HL
	LD A,${C}
	LD (CCNT),A
c_loop:
	LD HL,(WVROW)
	LD (WVPTR),HL
	LD HL,(WDROW)
	LD (WDPTR),HL
	LD HL,HVAL
	LD (XVPTR),HL
	LD HL,HTAN
	LD (XDPTR),HL
	LD A,${HD}
	LD (N),A
	LD HL,(BVROW)
	LD E,(HL)
	INC HL
	LD D,(HL)
	LD (BIASV),DE
	LD HL,(BDROW)
	LD E,(HL)
	INC HL
	LD D,(HL)
	LD (BIASD),DE
	CALL dualdot
	LD HL,(pS)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (pS),HL
	LD HL,(RESULT)
	ADD HL,DE
	LD (PREV),HL
	ADD HL,HL
	LD DE,TANHBASE2
	ADD HL,DE
	LD E,(HL)
	INC HL
	LD D,(HL)
	LD (STASH),DE
	LD HL,(COUTV)
	LD DE,(STASH)
	LD (HL),E
	INC HL
	LD (HL),D
	INC HL
	LD (COUTV),HL
	LD HL,(PREV)
	ADD HL,HL
	LD DE,DERIVBASE2
	ADD HL,DE
	LD E,(HL)
	INC HL
	LD D,(HL)
	LD (SX),DE
	LD HL,(RESULTD)
	LD (SY),HL
	CALL mulq8
	LD (STASH),HL
	LD HL,(COUTD)
	LD DE,(STASH)
	LD (HL),E
	INC HL
	LD (HL),D
	INC HL
	LD (COUTD),HL
	LD HL,(WVROW)
	LD DE,ROWBYTES
	ADD HL,DE
	LD (WVROW),HL
	LD HL,(WDROW)
	LD DE,ROWBYTES
	ADD HL,DE
	LD (WDROW),HL
	LD HL,(BVROW)
	INC HL
	INC HL
	LD (BVROW),HL
	LD HL,(BDROW)
	INC HL
	INC HL
	LD (BDROW),HL
	LD A,(CCNT)
	DEC A
	LD (CCNT),A
	JP NZ, c_loop
	HALT

dualdot:                         ; Σ wv·xv+bv -> RESULT ; Σ(wv·xd+wd·xv)+bd -> RESULTD
	LD A,0
	LD (ACC0),A
	LD A,(BV0)
	LD (ACC1),A
	LD A,(BV1)
	LD (ACC2),A
	LD A,(BV1)
	ADD A,A
	LD A,0
	SBC A,0
	LD (ACC3),A
	LD A,0
	LD (ACCD0),A
	LD A,(BD0)
	LD (ACCD1),A
	LD A,(BD1)
	LD (ACCD2),A
	LD A,(BD1)
	ADD A,A
	LD A,0
	SBC A,0
	LD (ACCD3),A
	LD HL,(WVPTR)
	LD (WVCUR),HL
	LD HL,(WDPTR)
	LD (WDCUR),HL
	LD HL,(XVPTR)
	LD (XVCUR),HL
	LD HL,(XDPTR)
	LD (XDCUR),HL
	LD A,(N)
	LD (CNT),A
ddloop:
	LD HL,(WVCUR)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (WVCUR),HL
	LD (TWV),DE
	LD HL,(WDCUR)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (WDCUR),HL
	LD (TWD),DE
	LD HL,(XVCUR)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (XVCUR),HL
	LD (TXV),DE
	LD HL,(XDCUR)
	LD E,(HL)
	INC HL
	LD D,(HL)
	INC HL
	LD (XDCUR),HL
	LD (TXD),DE
	LD HL,(TWV)
	LD (SX),HL
	LD HL,(TXV)
	LD (SY),HL
	CALL smul16
	CALL acc_add
	LD HL,(TWV)
	LD (SX),HL
	LD HL,(TXD)
	LD (SY),HL
	CALL smul16
	CALL acc_addd
	LD HL,(TWD)
	LD (SX),HL
	LD HL,(TXV)
	LD (SY),HL
	CALL smul16
	CALL acc_addd
	LD A,(CNT)
	DEC A
	LD (CNT),A
	JP NZ, ddloop
	LD HL,(ACC1)
	LD (RESULT),HL
	LD HL,(ACCD1)
	LD (RESULTD),HL
	RET

mulq8:                           ; (SX·SY)>>8 -> HL  (Q8.8 multiply)
	CALL smul16
	LD HL,(SP1)
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

acc_addd:
	LD A,(SP0)
	LD B,A
	LD A,(ACCD0)
	ADD A,B
	LD (ACCD0),A
	LD A,(SP1)
	LD B,A
	LD A,(ACCD1)
	ADC A,B
	LD (ACCD1),A
	LD A,(SP2)
	LD B,A
	LD A,(ACCD2)
	ADC A,B
	LD (ACCD2),A
	LD A,(SP3)
	LD B,A
	LD A,(ACCD3)
	ADC A,B
	LD (ACCD3),A
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
console.log('=== Phase 3: the training gradient through a running Z80 (dual numbers, real rule) ===');
console.log(`code ${CODE.length} B\n`);
if (CODE.length >= 0x1000) throw new Error(`code (${CODE.length}) overruns data`);

function setI16(t: Uint8Array, a: number, v: number): void { t[a] = v & 0xff; t[a + 1] = (v >> 8) & 0xff; }
function getI16(t: Uint8Array, a: number): number { const u = t[a] | (t[a + 1] << 8); return u >= 0x8000 ? u - 0x10000 : u; }

// base tape: value weights + LUTs (θ-independent)
const BASE = new Uint8Array(MEMBYTES);
BASE.set(CODE, 0);
for (let hh = 0; hh < HD; hh++) for (let k = 0; k < PERC; k++) setI16(BASE, W1V + (hh * PERC + k) * 2, toQ(par[W1O + hh * PERC + k], F));
for (let hh = 0; hh < HD; hh++) setI16(BASE, B1V + hh * 2, toQ(par[B1O + hh], F));
for (let c = 0; c < C; c++) for (let hh = 0; hh < HD; hh++) setI16(BASE, W2V + (c * HD + hh) * 2, toQ(par[W2O + c * HD + hh], F));
for (let c = 0; c < C; c++) setI16(BASE, B2V + c * 2, toQ(par[B2O + c], F));
for (let i = 0; i < LUTN; i++) { setI16(BASE, TANHLUT + i * 2, TANH[i]); setI16(BASE, DERIVLUT + i * 2, DERIV[i]); }

// neighbourhood of the output cell from the seed (field tangents all 0)
const s0 = seedGrid(cfg, exp, [0, 1]);
const oc = exp.outputCells[0];
const nbCells = { SELF: oc, NR: oc + 1, NL: oc - 1, NU: oc - cfg.SW, ND: oc + cfg.SW };
function loadNeighborhood(t: Uint8Array): void {
	const put = (addr: number, cell: number): void => { for (let ch = 0; ch < C; ch++) setI16(t, addr + ch * 2, toQ(s0[cell * C + ch], F)); };
	put(SELF, nbCells.SELF); put(NR, nbCells.NR); put(NL, nbCells.NL); put(NU, nbCells.NU); put(ND, nbCells.ND);
}

/** run the dual cell with θ's tangent seeded to 1; return d(out ch0)/dθ (real). */
function z80Grad(theta: number): { val: number; grad: number } {
	const t = BASE.slice();
	loadNeighborhood(t);
	// tangent weight arrays: all 0 (fresh tape), set θ's tangent = 1.0
	if (theta >= W1O && theta < B1O) setI16(t, W1D + (theta - W1O) * 2, toQ(1, F));
	else if (theta >= B1O && theta < W2O) setI16(t, B1D + (theta - B1O) * 2, toQ(1, F));
	else if (theta >= W2O && theta < B2O) setI16(t, W2D + (theta - W2O) * 2, toQ(1, F));
	else setI16(t, B2D + (theta - B2O) * 2, toQ(1, F));
	const { mem, halted } = runOnRealZ80(t, MEMBYTES, 50_000_000);
	if (!halted) throw new Error('dual cell did not HALT');
	return { val: fromQ(getI16(mem, OUTV), F), grad: fromQ(getI16(mem, OUTD), F) };
}

/** f64 ground-truth gradient d(out ch0 after 1 step)/dθ via central differences. */
function f64Grad(theta: number, h = 1e-4): number {
	const outAt = (delta: number): number => { const q = Float64Array.from(par); q[theta] += delta; return readOutputs(cfg, step(cfg, q, s0, exp, [0, 1]), exp)[0]; };
	return (outAt(h) - outAt(-h)) / (2 * h);
}

const THETAS: { name: string; idx: number }[] = [
	{ name: 'b2[0]        ', idx: B2O + 0 },
	{ name: 'W2[ch0,h5]   ', idx: W2O + 0 * HD + 5 },
	{ name: 'W2[ch0,h20]  ', idx: W2O + 0 * HD + 20 },
	{ name: 'b1[10]       ', idx: B1O + 10 },
	{ name: 'W1[h10,perc7]', idx: W1O + 10 * PERC + 7 },
	{ name: 'W1[h3,perc44]', idx: W1O + 3 * PERC + 44 }
];

const valZ80 = z80Grad(THETAS[0].idx).val;
const valF64 = readOutputs(cfg, step(cfg, par, s0, exp, [0, 1]), exp)[0];
console.log(`output value (sanity): Z80 ${valZ80.toFixed(4)}  f64 ${valF64.toFixed(4)}  |Δ| ${Math.abs(valZ80 - valF64).toExponential(2)}\n`);
console.log('d(output ch0, 1 step)/dθ — Z80 dual tangent vs f64 finite-diff:');
console.log('  weight          Z80 grad      f64 grad      |err|');
let worst = 0;
for (const th of THETAS) {
	const g = z80Grad(th.idx).grad;
	const gRef = f64Grad(th.idx);
	const err = Math.abs(g - gRef);
	worst = Math.max(worst, err);
	console.log(`  ${th.name}   ${g.toFixed(6).padStart(10)}   ${gRef.toFixed(6).padStart(10)}   ${err.toExponential(2)}`);
}
console.log(`\nworst |err| = ${worst.toExponential(2)}  (Q8.8 resolution ${(1 / 256).toExponential(2)})`);
if (worst > 2 / 256) { console.error('\nFAIL: the in-substrate gradient did not match.'); process.exit(1); }
console.log('\nPASS: a real Z80 (z80-emulator v2.3.0) carried a forward-mode tangent d(out)/dθ of the');
console.log('single-step cell map, matching finite differences to Q8.8 resolution. This is the PRIMITIVE');
console.log('for an in-substrate gradient (a single-cell JVP for a few weights) — NOT yet the full BPTT');
console.log('training gradient ∂L/∂θ over the T-step rollout. Rebuild at Q16.16 + propagate through the');
console.log('rollout to reach gradient-grade (Phase 1.5 shows the fixed-point tangent is clean at Q16.16).');

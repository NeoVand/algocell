// EXPERIMENT A — can we carry an exact gradient THROUGH a running Z80 program?
//
// Forward-mode automatic differentiation via DUAL NUMBERS, implemented in
// fixed-point Q8.8 as an actual Z80 program (runs identically on Zilion). A
// value is a pair (v, v̇); arithmetic transports the tangent v̇ by the chain
// rule. We compute f(θ)=θ² as a dual multiply and check the returned tangent
// equals f'(θ)=2θ — against a fixed-point reference AND finite differences.
//
// The crux: multiply is the only nonlinear op, and it needs a fixed-point
// multiply. The in-repo assembler has NO shift opcodes — but `ADD A,A` (shift
// a byte left, MSB→carry) and `ADD HL,HL` (shift HL left) are enough for a
// shift-add multiply. So no assembler change, no trap.
//
//   npx tsx src/lib/morph/dev/expA.ts

import { assemble } from '../z80asm';
import { runOnRealZ80 } from './z80run';

// ---- fixed-point Q8.8 reference (mirrors exactly what the Z80 computes) ----
const ONE = 256; // 1.0 in Q8.8
const MASK = 0xffff;

const toQ = (x: number): number => Math.round(x * ONE) & MASK;
const fromQ = (q: number): number => q / ONE;

/** Q8.8 * Q8.8 -> Q8.8 (unsigned): exact 16x16->32 product, then >>8, low 16 bits. */
const fixmul = (x: number, y: number): number => Math.floor((x * y) / 256) & MASK;

interface Dual {
	v: number; // value  (Q8.8)
	d: number; // tangent (Q8.8)
}
const dualMul = (a: Dual, b: Dual): Dual => ({
	v: fixmul(a.v, b.v),
	d: (fixmul(a.v, b.d) + fixmul(a.d, b.v)) & MASK
});

// ---- tape memory map (data + stack sit above the code) --------------------
const ADDR = {
	AV: 768, AD: 770, BV: 772, BD: 774, RV: 776, RD: 778,
	MULX: 780, MULY: 782, P0: 784, P1a: 786, P1b: 788, P2: 790, TMP1: 792
};
const MEMBYTES = 1024;
const SYM: Record<string, number> = {
	...ADDR,
	MULX0: ADDR.MULX, MULX1: ADDR.MULX + 1,
	MULY0: ADDR.MULY, MULY1: ADDR.MULY + 1,
	P0HI: ADDR.P0 + 1, P2LO: ADDR.P2,
	STACKTOP: MEMBYTES
};

// ---- the Z80 program: dual multiply of (AV,AD) x (BV,BD) -> (RV,RD) --------
const SOURCE = `
		LD SP, STACKTOP
		; RV = fixmul(AV, BV)
		LD HL,(AV)
		LD (MULX),HL
		LD HL,(BV)
		LD (MULY),HL
		CALL fixmul
		LD (RV),HL
		; TMP1 = fixmul(AV, BD)
		LD HL,(AV)
		LD (MULX),HL
		LD HL,(BD)
		LD (MULY),HL
		CALL fixmul
		LD (TMP1),HL
		; RD = fixmul(AD, BV) + TMP1
		LD HL,(AD)
		LD (MULX),HL
		LD HL,(BV)
		LD (MULY),HL
		CALL fixmul
		LD DE,(TMP1)
		ADD HL,DE
		LD (RD),HL
		HALT

	fixmul:                     ; Q8.8 (MULX) * (MULY) -> HL, via four 8x8 partials
		LD A,(MULY0)
		LD E,A
		LD A,(MULX0)
		CALL mul8               ; P0 = Xl*Yl
		LD (P0),HL
		LD A,(MULY0)
		LD E,A
		LD A,(MULX1)
		CALL mul8               ; P1a = Xh*Yl
		LD (P1a),HL
		LD A,(MULY1)
		LD E,A
		LD A,(MULX0)
		CALL mul8               ; P1b = Xl*Yh
		LD (P1b),HL
		LD A,(MULY1)
		LD E,A
		LD A,(MULX1)
		CALL mul8               ; P2 = Xh*Yh
		LD (P2),HL
		; result = highbyte(P0) + P1a + P1b + (lowbyte(P2) << 8)   [low 16 bits]
		LD A,(P0HI)
		LD L,A
		LD H,0
		LD DE,(P1a)
		ADD HL,DE
		LD DE,(P1b)
		ADD HL,DE
		LD A,(P2LO)
		LD D,A
		LD E,0
		ADD HL,DE
		RET

	mul8:                       ; A * E -> HL (8x8 -> 16, unsigned, MSB-first shift-add)
		LD HL,0
		LD D,0                  ; DE = E (multiplicand)
		LD B,8
	m8:
		ADD A,A                 ; A <<= 1, bit7 -> carry
		JR NC, m8n
		ADD HL,HL               ; HL = HL*2 + DE
		ADD HL,DE
		JR m8c
	m8n:
		ADD HL,HL               ; HL = HL*2
	m8c:
		DJNZ m8
		RET
`
	.split('\n')
	.map((l) => l.trim());

const CODE = assemble(SOURCE, SYM);

function u16(tape: Uint8Array, addr: number): number {
	return tape[addr] | (tape[addr + 1] << 8);
}
function setU16(tape: Uint8Array, addr: number, val: number): void {
	tape[addr] = val & 0xff;
	tape[addr + 1] = (val >> 8) & 0xff;
}

/** Run the dual-multiply program on (a × b); returns the dual result read from the tape. */
function runDualMul(a: Dual, b: Dual): Dual {
	const tape = new Uint8Array(MEMBYTES);
	tape.set(CODE, 0);
	setU16(tape, ADDR.AV, a.v);
	setU16(tape, ADDR.AD, a.d);
	setU16(tape, ADDR.BV, b.v);
	setU16(tape, ADDR.BD, b.d);
	const { mem, halted } = runOnRealZ80(tape, MEMBYTES, 200000);
	if (!halted) throw new Error('program did not HALT');
	return { v: u16(mem, ADDR.RV), d: u16(mem, ADDR.RD) };
}

/** f(θ) = θ² as a dual multiply: (θ,1)·(θ,1) = (θ², 2θ). */
function squareOnZ80(theta: number): Dual {
	const t = toQ(theta);
	return runDualMul({ v: t, d: ONE }, { v: t, d: ONE });
}

function main() {
	console.log('=== EXP A: forward-mode AD through a running Z80 (dual numbers, Q8.8) ===');
	console.log(`code ${CODE.length} B, memBytes ${MEMBYTES}\n`);

	// (1) Substrate correctness: Z80 dual-multiply must EXACTLY equal the fixed ref.
	let worst = 0;
	let mism = 0;
	for (let i = 0; i < 400; i++) {
		const a: Dual = { v: (i * 97 + 13) & MASK, d: (i * 31 + 7) & MASK };
		const b: Dual = { v: (i * 53 + 3) & MASK, d: (i * 71 + 11) & MASK };
		const z = runDualMul(a, b);
		const r = dualMul(a, b);
		if (z.v !== r.v || z.d !== r.d) {
			mism++;
			if (mism <= 3) console.log(`  MISMATCH: z=(${z.v},${z.d}) ref=(${r.v},${r.d})`);
		}
	}
	console.log(`[${mism === 0 ? 'PASS' : 'FAIL'}] Z80 dual-multiply == fixed-point reference on 400 random pairs (${mism} mismatches)`);

	// (2) The gradient test: tangent of f(θ)=θ² must equal 2θ (and finite differences).
	console.log('\n  θ      f=θ²(Z80)   f\'(Z80)   2θ(exact)   finite-diff   |err|');
	console.log('  ----   ---------   -------   ---------   -----------   -----');
	const h = 0.0625; // finite-difference step
	for (const theta of [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]) {
		const r = squareOnZ80(theta);
		const val = fromQ(r.v);
		const tan = fromQ(r.d);
		const fdPlus = fromQ(squareOnZ80(theta + h).v);
		const fdMinus = fromQ(squareOnZ80(theta - h).v);
		const fd = (fdPlus - fdMinus) / (2 * h);
		const err = Math.abs(tan - 2 * theta);
		worst = Math.max(worst, err);
		console.log(
			`  ${theta.toFixed(2)}   ${val.toFixed(4).padStart(9)}   ${tan.toFixed(3).padStart(7)}   ${(2 * theta).toFixed(3).padStart(9)}   ${fd.toFixed(3).padStart(11)}   ${err.toFixed(4)}`
		);
	}
	console.log(`\nworst |tangent − 2θ| over the sweep: ${worst.toFixed(4)} (Q8.8 resolution = ${(1 / ONE).toFixed(4)})`);

	if (mism !== 0 || worst > 1 / ONE + 1e-9) {
		console.error('\nFAIL: gradient did not transport exactly through the Z80.');
		process.exit(1);
	}
	console.log('\nPASS: an exact analytic gradient was carried THROUGH a running Z80 program.');
	console.log('Each θ is an independent program → Zilion computes the whole (value, gradient) curve in one parallel dispatch.');

	// (3) The payoff: gradient DESCENT driven purely by the Z80's in-substrate AD.
	// Minimize L(θ) = (f(θ) − t)² with f(θ)=θ², using ONLY the (value, tangent)
	// the Z80 reports about its own execution — no evolution, no finite diffs.
	// Converges to θ = √t. This is learning by gradient, not by black-box search.
	console.log('\n=== gradient descent to solve θ² = t, using the Z80-computed gradient ===');
	console.log('  t     θ*(√t)    θ found    L final     iters');
	console.log('  ---   -------   --------   ---------   -----');
	const lossAt = (theta: number, t: number): number => {
		const f = fromQ(squareOnZ80(Math.max(0.02, theta)).v);
		return (f - t) * (f - t);
	};
	let allConverged = true;
	for (const t of [2, 3, 5, 7]) {
		let theta = 1.0;
		let iters = 0;
		let L = Infinity;
		for (; iters < 100; iters++) {
			const r = squareOnZ80(theta); // Z80 returns f=θ² (value) and f'=2θ (tangent)
			const f = fromQ(r.v);
			const fp = fromQ(r.d);
			L = (f - t) * (f - t);
			if (L < 1e-4) break;
			const gradL = 2 * (f - t) * fp; // dL/dθ = 2(f−t)·f' — f' is the in-substrate tangent
			// Backtracking line search: shrink the step until the loss actually drops.
			// (Proves the in-substrate gradient really points downhill.)
			let step = 1.0;
			while (step > 1e-5 && lossAt(theta - step * gradL, t) >= L) step *= 0.5;
			theta = Math.max(0.02, theta - step * gradL);
		}
		const ok = Math.abs(theta - Math.sqrt(t)) < 0.03;
		allConverged = allConverged && ok;
		console.log(
			`  ${t}     ${Math.sqrt(t).toFixed(4)}   ${theta.toFixed(4).padStart(8)}   ${L.toExponential(2)}   ${String(iters).padStart(5)}   ${ok ? '' : '  <-- off'}`
		);
	}
	if (!allConverged) {
		console.error('\nFAIL: gradient descent did not converge using the in-substrate gradient.');
		process.exit(1);
	}
	console.log('\nPASS: optimization converged using ONLY gradients the Z80 computed about its own run.');
	console.log('This is the marriage: a discrete program that hands back an exact gradient — descent, not evolution.');
}

main();

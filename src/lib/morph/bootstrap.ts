// The fixed Z80 "bootstrap": a von-Neumann-5 outer-totalistic CA update loop.
//
// This program is the SAME for every candidate in evolution — only the genome
// table (data in the tape) changes. It reads the FRONT grid, computes each
// interior cell's next state via GENOME[ SELFBASE[self] + neighborSum ], writes
// it to BACK, copies BACK->FRONT, and repeats T times, then HALTs.
//
// Register roles inside the inner cell:
//   HL = front cell pointer p      DE = back cell pointer
//   B  = self                      C  = running neighbor sum
//   A  = scratch
// The stack is never used (there is no free RAM for one — every tape byte is
// code, tables, or grid), so pointer saves go through fixed memory words
// (PSAVE/SUMSAVE) via LD (nn),HL / LD (nn),A instead of PUSH/POP.
//
// M0 note: BACK->FRONT is copied with LDIR each step for obviously-correct
// double buffering. The throughput-oriented pointer-swap variant is a later
// optimization (see MORPHOGENESIS roadmap); correctness comes first.

import { assemble } from './z80asm';
import type { MorphParams, MemoryMap } from './ca';

export function bootstrapSource(p: MorphParams, map: MemoryMap): string[] {
	return `
		; ---- init developmental-step counter ----
		LD A, T
		LD (TCOUNT), A
	t_loop:
		LD HL, FRONT_FIRST      ; first interior cell (row 1, col 1)
		LD DE, BACK_FIRST
		LD A, HMINUS2           ; interior rows
		LD (YCOUNT), A
	y_loop:
		LD A, WMINUS2           ; interior cells per row
		LD (XCOUNT), A
	x_loop:
		; --- gather self + von-Neumann-5 neighbor sum (border reads are 0) ---
		LD A,(HL)               ; self
		LD B,A
		DEC HL                  ; p-1 (left)
		LD C,(HL)               ; sum = left
		INC HL
		INC HL                  ; p+1 (right)
		LD A,(HL)
		ADD A,C
		LD C,A                  ; sum += right
		DEC HL                  ; p
		LD A,L                  ; HL = p - W (up)
		SUB W
		LD L,A
		LD A,H
		SBC A,0
		LD H,A
		LD A,(HL)
		ADD A,C
		LD C,A                  ; sum += up
		LD A,L                  ; HL = p (restore +W)
		ADD A,W
		LD L,A
		LD A,H
		ADC A,0
		LD H,A
		LD A,L                  ; HL = p + W (down)
		ADD A,W
		LD L,A
		LD A,H
		ADC A,0
		LD H,A
		LD A,(HL)
		ADD A,C
		LD C,A                  ; sum += down  (C = full neighbor sum)
		LD A,L                  ; HL = p (restore -W)
		SUB W
		LD L,A
		LD A,H
		SBC A,0
		LD H,A
		; --- next = GENOME[ SELFBASE[self] + sum ] ---
		LD (PSAVE), HL          ; save front pointer (no stack available)
		LD A,C
		LD (SUMSAVE), A         ; stash sum
		LD H,0
		LD L,B                  ; HL = self
		LD BC, SELFBASE
		ADD HL, BC              ; HL = SELFBASE + self
		LD A,(HL)               ; A = self*K
		LD HL, SUMSAVE
		ADD A,(HL)              ; A = self*K + sum = genome index
		LD H,0
		LD L,A                  ; HL = index
		LD BC, GENOME
		ADD HL, BC              ; HL = GENOME + index
		LD A,(HL)               ; A = next state
		LD HL, (PSAVE)          ; restore front pointer
		LD (DE), A              ; write next -> back buffer
		INC HL
		INC DE
		LD A,(XCOUNT)
		DEC A
		LD (XCOUNT),A
		JP NZ, x_loop
		INC HL                  ; skip this row's right border + next row's left border
		INC HL
		INC DE
		INC DE
		LD A,(YCOUNT)
		DEC A
		LD (YCOUNT),A
		JP NZ, y_loop
		LD HL, BACK             ; copy BACK -> FRONT for the next step
		LD DE, FRONT
		LD BC, WH
		LDIR
		LD A,(TCOUNT)
		DEC A
		LD (TCOUNT),A
		JP NZ, t_loop
		HALT
	`
		.split('\n')
		.map((l) => l.trim());
}

/** Symbols the bootstrap references (fixed addresses + grid constants). */
export function bootstrapSymbols(p: MorphParams, map: MemoryMap): Record<string, number> {
	return {
		T: p.T,
		W: p.W,
		WMINUS2: p.W - 2,
		HMINUS2: p.H - 2,
		WH: p.W * p.H,
		TCOUNT: map.TCOUNT,
		YCOUNT: map.YCOUNT,
		XCOUNT: map.XCOUNT,
		SUMSAVE: map.SUMSAVE,
		PSAVE: map.PSAVE,
		SELFBASE: map.SELFBASE,
		GENOME: map.GENOME,
		FRONT: map.FRONT,
		BACK: map.BACK,
		FRONT_FIRST: map.FRONT + p.W + 1,
		BACK_FIRST: map.BACK + p.W + 1
	};
}

/** Assemble the bootstrap to bytes for the given params + memory map. */
export function assembleBootstrap(p: MorphParams, map: MemoryMap): Uint8Array {
	return assemble(bootstrapSource(p, map), bootstrapSymbols(p, map));
}

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

// ---------------------------------------------------------------------------
// SUBSTRATE V2 — SUM×DIR16 bootstrap (directional perception).
//
// Same T/Y/X scaffolding + LDIR + HALT as v1; only the inner cell body changes.
// Per neighbor, the same byte already loaded for the sum is tested for aliveness
// and OR'd into a 4-bit `dir` code (up=8, right=4, down=2, left=1). The genome
// index is tbase*16 + dir, where tbase = self*K + sum is v1's index. tbase*16 is
// built with ADD HL,HL x4 (no shift/rotate opcode exists in z80asm), and dir is
// merged into the zeroed low nibble. Register roles: HL=front ptr, DE=back ptr,
// B=dir, C=sum, A=scratch; self is re-read from (HL) at index time. No stack.
// ---------------------------------------------------------------------------

export function bootstrapSourceV2(p: MorphParams, map: MemoryMap): string[] {
	return `
		LD A, T
		LD (TCOUNT), A
	t_loop:
		LD HL, FRONT_FIRST
		LD DE, BACK_FIRST
		LD A, HMINUS2
		LD (YCOUNT), A
	y_loop:
		LD A, WMINUS2
		LD (XCOUNT), A
	x_loop:
		LD B,0                  ; dir = 0
		DEC HL                  ; p-1 (left)
		LD A,(HL)
		LD C,A                  ; sum = left
		OR A
		JR Z, sk_l
		INC B                   ; dir |= 1 (left)
	sk_l:
		INC HL
		INC HL                  ; p+1 (right)
		LD A,(HL)
		ADD A,C
		LD C,A                  ; sum += right
		LD A,(HL)               ; reload right (ADD clobbered A)
		OR A
		JR Z, sk_r
		LD A,B
		ADD A,4                 ; dir |= 4 (right)
		LD B,A
	sk_r:
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
		LD A,(HL)
		OR A
		JR Z, sk_u
		LD A,B
		ADD A,8                 ; dir |= 8 (up)
		LD B,A
	sk_u:
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
		LD C,A                  ; sum += down (C = full neighbor sum)
		LD A,(HL)
		OR A
		JR Z, sk_d
		LD A,B
		ADD A,2                 ; dir |= 2 (down)
		LD B,A
	sk_d:
		LD A,L                  ; HL = p (restore -W)
		SUB W
		LD L,A
		LD A,H
		SBC A,0
		LD H,A
		; --- index = (self*K + sum)*16 + dir ---
		LD (PSAVE), HL          ; save front pointer
		LD A,C
		LD (SUMSAVE), A         ; stash sum
		LD A,B
		LD (DIRSAVE), A         ; stash dir
		LD A,(HL)               ; re-read self (HL = p)
		LD H,0
		LD L,A                  ; HL = self
		LD BC, SELFBASE
		ADD HL, BC              ; HL = SELFBASE + self
		LD A,(HL)               ; A = self*K
		LD HL, SUMSAVE
		ADD A,(HL)              ; A = self*K + sum = tbase (0..S*K-1)
		LD H,0
		LD L,A                  ; HL = tbase
		ADD HL, HL              ; *2
		ADD HL, HL              ; *4
		ADD HL, HL              ; *8
		ADD HL, HL              ; *16 (low nibble of L now 0)
		LD A,(DIRSAVE)
		OR L                    ; merge dir into low nibble
		LD L,A                  ; HL = tbase*16 + dir = index
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
		INC HL
		INC HL
		INC DE
		INC DE
		LD A,(YCOUNT)
		DEC A
		LD (YCOUNT),A
		JP NZ, y_loop
		LD HL, BACK
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

export function bootstrapSymbolsV2(p: MorphParams, map: MemoryMap): Record<string, number> {
	if (map.DIRSAVE === undefined) throw new Error('v2 memory map is missing DIRSAVE');
	return {
		...bootstrapSymbols(p, map),
		DIRSAVE: map.DIRSAVE
	};
}

/** Assemble the v2 SUM×DIR16 bootstrap. */
export function assembleBootstrapV2(p: MorphParams, map: MemoryMap): Uint8Array {
	return assemble(bootstrapSourceV2(p, map), bootstrapSymbolsV2(p, map));
}

// Morphogenesis cellular automaton — the substrate-independent definition.
//
// A W x H grid of cells, each holding a state in 0..S-1. The border ring is a
// fixed background (state 0) and never updates; only the interior evolves. The
// update is *outer-totalistic* on the von-Neumann-5 neighborhood:
//
//     index = self * K + (up + down + left + right)      K = 4*(S-1) + 1
//     next  = genome[index]
//
// so a "genome" is just the lookup table `genome[0 .. S*K - 1]` of next-states.
// This same rule is implemented three ways that MUST agree byte-for-byte:
//   1. `stepCA` here (pure TS reference),
//   2. the hand-written Z80 program in `bootstrap.ts` (run on a real Z80),
//   3. the same Z80 bytes on Zilion's GPU core.
// The dev harness diff-tests (1) against (2); Zilion's own conformance covers (3).

export interface MorphParams {
	/** grid width (includes the 1-cell background border) */
	W: number;
	/** grid height (includes the 1-cell background border) */
	H: number;
	/** number of cell states, 0..S-1 (state 0 = background) */
	S: number;
	/** developmental steps per run */
	T: number;
}

/** K = neighbor-sum span + 1 = the stride of `self` in the genome index. */
export function kOf(S: number): number {
	return 4 * (S - 1) + 1;
}

/** Genome table length for S states. */
export function genomeLength(S: number): number {
	return S * kOf(S);
}

/** SELFBASE[self] = self * K — a tiny lookup so the Z80 avoids multiplying. */
export function selfbaseTable(S: number): Uint8Array {
	const K = kOf(S);
	const t = new Uint8Array(S);
	for (let s = 0; s < S; s++) t[s] = (s * K) & 0xff;
	return t;
}

/**
 * One reference CA step. `grid` is length W*H (row-major); returns a fresh grid.
 * Border cells are held at 0. Neighbors outside the interior read as 0 (the
 * border), matching the Z80 program (whose border bytes are always 0).
 */
export function stepCA(grid: Uint8Array, genome: Uint8Array, p: MorphParams): Uint8Array {
	const { W, H, S } = p;
	const K = kOf(S);
	const next = new Uint8Array(W * H); // borders stay 0
	for (let y = 1; y < H - 1; y++) {
		for (let x = 1; x < W - 1; x++) {
			const i = y * W + x;
			const self = grid[i];
			const sum = grid[i - 1] + grid[i + 1] + grid[i - W] + grid[i + W];
			next[i] = genome[self * K + sum];
		}
	}
	return next;
}

/** Run the reference CA for T steps from a seed grid. */
export function runCA(seed: Uint8Array, genome: Uint8Array, p: MorphParams): Uint8Array {
	let g: Uint8Array = seed.slice();
	for (let t = 0; t < p.T; t++) g = stepCA(g, genome, p);
	return g;
}

/** A single live seed cell of `state` at the grid center. */
export function centerSeed(p: MorphParams, state = 1): Uint8Array {
	const g = new Uint8Array(p.W * p.H);
	const cx = p.W >> 1;
	const cy = p.H >> 1;
	g[cy * p.W + cx] = state;
	return g;
}

// --- fixed tape memory map -------------------------------------------------
//
// Data lives at fixed absolute addresses (independent of code length) so the
// bootstrap can reference them as constants. Code occupies [0, DATA_ORG); the
// tape builder asserts the assembled code actually fits.

export interface MemoryMap {
	DATA_ORG: number;
	TCOUNT: number;
	YCOUNT: number;
	XCOUNT: number;
	SUMSAVE: number;
	PSAVE: number; // 2 bytes
	DIRSAVE?: number; // 1 byte (v2 only — directional alive-code scratch)
	SELFBASE: number; // S bytes
	GENOME: number; // genomeBytes
	FRONT: number; // W*H bytes
	BACK: number; // W*H bytes
	memBytes: number;
	genomeBytes: number; // length of the genome region (v1: S*K, v2: S*K*16)
}

function nextPow2(n: number): number {
	let p = 4;
	while (p < n) p <<= 1;
	return p;
}

/** Compute the tape layout. `codeReserve` is the room set aside for code. */
export function computeMemoryMap(p: MorphParams, codeReserve = 384): MemoryMap {
	const S = p.S;
	const WH = p.W * p.H;
	const DATA_ORG = codeReserve;
	const TCOUNT = DATA_ORG;
	const YCOUNT = DATA_ORG + 1;
	const XCOUNT = DATA_ORG + 2;
	const SUMSAVE = DATA_ORG + 3;
	const PSAVE = DATA_ORG + 4; // +5
	const SELFBASE = DATA_ORG + 6;
	const GENOME = SELFBASE + S;
	const FRONT = GENOME + genomeLength(S);
	const BACK = FRONT + WH;
	const end = BACK + WH;
	const memBytes = nextPow2(end);
	return { DATA_ORG, TCOUNT, YCOUNT, XCOUNT, SUMSAVE, PSAVE, SELFBASE, GENOME, FRONT, BACK, memBytes, genomeBytes: genomeLength(S) };
}

/**
 * Build the initial tape: code at 0, the SELFBASE + genome tables and the
 * seeded FRONT grid at their fixed offsets, BACK zeroed.
 */
export function buildTape(
	code: Uint8Array,
	genome: Uint8Array,
	seed: Uint8Array,
	map: MemoryMap,
	p: MorphParams
): Uint8Array {
	if (code.length > map.DATA_ORG) {
		throw new Error(`code (${code.length} B) overruns data region at ${map.DATA_ORG}`);
	}
	if (genome.length !== map.genomeBytes) {
		throw new Error(`genome length ${genome.length} != ${map.genomeBytes}`);
	}
	if (seed.length !== p.W * p.H) throw new Error(`seed length ${seed.length} != ${p.W * p.H}`);

	const tape = new Uint8Array(map.memBytes);
	tape.set(code, 0);
	tape.set(selfbaseTable(p.S), map.SELFBASE);
	tape.set(genome, map.GENOME);
	tape.set(seed, map.FRONT);
	// BACK stays zero.
	return tape;
}

/** Extract the FRONT grid (the current/output buffer) from a tape image. */
export function readFront(tape: Uint8Array, map: MemoryMap, p: MorphParams): Uint8Array {
	return tape.slice(map.FRONT, map.FRONT + p.W * p.H);
}

/** ASCII render for eyeballing growth in a terminal. */
export function asciiGrid(grid: Uint8Array, p: MorphParams): string {
	const glyphs = [' ', '.', 'o', '#', '@', '%', '*', '+'];
	const rows: string[] = [];
	for (let y = 0; y < p.H; y++) {
		let row = '';
		for (let x = 0; x < p.W; x++) {
			row += glyphs[grid[y * p.W + x] % glyphs.length];
		}
		rows.push(row);
	}
	return rows.join('\n');
}

// ===========================================================================
// SUBSTRATE V2 — SUM×DIR16 (directional perception; breaks the isotropy wall)
// ===========================================================================
//
// v1 is isotropic: index = self*K + sum, so from a symmetric seed the CA can
// only make symmetric patterns. v2 keeps that isotropic base and APPENDS a
// 4-bit directional "alive" code — one bit per von-Neumann neighbor — so the
// rule can tell up/down/left/right apart:
//
//     dir   = (up>0)*8 | (right>0)*4 | (down>0)*2 | (left>0)*1     (0..15)
//     tbase = self*K + sum                                          (0..S*K-1)
//     index = tbase*16 + dir
//     next  = genome[index]
//
// The genome is thus 16 directional variants of the entire v1 table. It is
// warm-startable byte-exactly from an evolved v1 genome (see liftV1toV2), so
// v2 can never regress below v1's proven capability. Bit assignment is
// load-bearing: the Z80 bootstrap and this reference MUST agree exactly.

/** Genome length for the v2 SUM×DIR16 rule. */
export function genomeLengthV2(S: number): number {
	return S * kOf(S) * 16;
}

/** v2 tape layout (adds a DIRSAVE scratch byte; larger genome; more code room). */
export function computeMemoryMapV2(p: MorphParams, codeReserve = 512): MemoryMap {
	const S = p.S;
	const WH = p.W * p.H;
	const DATA_ORG = codeReserve;
	const TCOUNT = DATA_ORG;
	const YCOUNT = DATA_ORG + 1;
	const XCOUNT = DATA_ORG + 2;
	const SUMSAVE = DATA_ORG + 3;
	const PSAVE = DATA_ORG + 4; // +5
	const DIRSAVE = DATA_ORG + 6;
	const SELFBASE = DATA_ORG + 7;
	const GENOME = SELFBASE + S;
	const FRONT = GENOME + genomeLengthV2(S);
	const BACK = FRONT + WH;
	const end = BACK + WH;
	const memBytes = nextPow2(end);
	return {
		DATA_ORG, TCOUNT, YCOUNT, XCOUNT, SUMSAVE, PSAVE, DIRSAVE, SELFBASE, GENOME, FRONT, BACK,
		memBytes, genomeBytes: genomeLengthV2(S)
	};
}

/** One reference step of the v2 SUM×DIR16 rule (mirrors the Z80 bootstrap exactly). */
export function stepCAv2(grid: Uint8Array, genome: Uint8Array, p: MorphParams): Uint8Array {
	const { W, H, S } = p;
	const K = kOf(S);
	const next = new Uint8Array(W * H); // borders stay 0
	for (let y = 1; y < H - 1; y++) {
		for (let x = 1; x < W - 1; x++) {
			const i = y * W + x;
			const self = grid[i];
			const up = grid[i - W];
			const down = grid[i + W];
			const left = grid[i - 1];
			const right = grid[i + 1];
			const sum = up + down + left + right;
			const dir = (up > 0 ? 8 : 0) | (right > 0 ? 4 : 0) | (down > 0 ? 2 : 0) | (left > 0 ? 1 : 0);
			const tbase = self * K + sum;
			next[i] = genome[tbase * 16 + dir];
		}
	}
	return next;
}

/** Run the v2 reference for T steps from a seed grid. */
export function runCAv2(seed: Uint8Array, genome: Uint8Array, p: MorphParams): Uint8Array {
	let g: Uint8Array = seed.slice();
	for (let t = 0; t < p.T; t++) g = stepCAv2(g, genome, p);
	return g;
}

/** Broadcast a v1 genome into all 16 directional slots — the warm-start lift. */
export function liftV1toV2(v1: Uint8Array, S: number): Uint8Array {
	const len1 = genomeLength(S);
	if (v1.length !== len1) throw new Error(`v1 genome length ${v1.length} != ${len1}`);
	const v2 = new Uint8Array(genomeLengthV2(S));
	for (let t = 0; t < len1; t++) {
		const base = t * 16;
		for (let d = 0; d < 16; d++) v2[base + d] = v1[t];
	}
	return v2;
}

/**
 * Asymmetric seed: center cell (state 1) plus a fixed off-center landmark cell
 * (state `landmark`, default 3, one cell to the LEFT of center). The landmark
 * breaks the symmetry the isotropy wall depends on — even under the warm-start
 * (inert-dir) genome it makes the neighbor SUM asymmetric, giving the GA a
 * non-zero fitness gradient from generation 0.
 */
export function asymmetricSeed(p: MorphParams, center = 1, landmark = 3): Uint8Array {
	const g = new Uint8Array(p.W * p.H);
	const cx = p.W >> 1;
	const cy = p.H >> 1;
	g[cy * p.W + cx] = center;
	g[cy * p.W + (cx - 1)] = landmark;
	return g;
}

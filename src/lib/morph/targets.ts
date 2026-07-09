// Procedural target shapes for morphogenesis evolution.
//
// A target is a W*H grid of states (0 = background). Foreground uses `state`.
// Shapes are centered on the seed cell (W>>1, H>>1) so growth and target share
// an origin. The border ring is always background (the CA never writes it).
//
// Note on achievability: the CA grows from a single seed under a von-Neumann-5
// neighborhood, so *diamonds* (L1 balls) are the shape it produces most
// naturally — they are the friendliest early targets. Discs/crosses are harder
// (they demand the rule round corners or break isotropy) and make good stress
// tests once diamonds work.

import type { MorphParams } from './ca';

function blank(p: MorphParams): Uint8Array {
	return new Uint8Array(p.W * p.H);
}

function isInterior(p: MorphParams, x: number, y: number): boolean {
	return x > 0 && y > 0 && x < p.W - 1 && y < p.H - 1;
}

/** Filled L1 ball (diamond): |dx| + |dy| <= radius. */
export function makeDiamond(p: MorphParams, radius: number, state = 1): Uint8Array {
	const g = blank(p);
	const cx = p.W >> 1;
	const cy = p.H >> 1;
	for (let y = 0; y < p.H; y++)
		for (let x = 0; x < p.W; x++)
			if (isInterior(p, x, y) && Math.abs(x - cx) + Math.abs(y - cy) <= radius) g[y * p.W + x] = state;
	return g;
}

/** Filled L2 ball (disc): dx^2 + dy^2 <= radius^2. */
export function makeDisc(p: MorphParams, radius: number, state = 1): Uint8Array {
	const g = blank(p);
	const cx = p.W >> 1;
	const cy = p.H >> 1;
	const r2 = radius * radius;
	for (let y = 0; y < p.H; y++)
		for (let x = 0; x < p.W; x++) {
			const dx = x - cx;
			const dy = y - cy;
			if (isInterior(p, x, y) && dx * dx + dy * dy <= r2) g[y * p.W + x] = state;
		}
	return g;
}

/** Hollow diamond ring: inner < |dx|+|dy| <= outer. */
export function makeRing(p: MorphParams, outer: number, inner: number, state = 1): Uint8Array {
	const g = blank(p);
	const cx = p.W >> 1;
	const cy = p.H >> 1;
	for (let y = 0; y < p.H; y++)
		for (let x = 0; x < p.W; x++) {
			const d = Math.abs(x - cx) + Math.abs(y - cy);
			if (isInterior(p, x, y) && d <= outer && d > inner) g[y * p.W + x] = state;
		}
	return g;
}

/** Plus / cross of the given half-thickness. */
export function makeCross(p: MorphParams, half: number, state = 1): Uint8Array {
	const g = blank(p);
	const cx = p.W >> 1;
	const cy = p.H >> 1;
	for (let y = 0; y < p.H; y++)
		for (let x = 0; x < p.W; x++)
			if (isInterior(p, x, y) && (Math.abs(x - cx) <= half || Math.abs(y - cy) <= half)) g[y * p.W + x] = state;
	return g;
}

// --- multi-color (M2) targets ---------------------------------------------
//
// Concentric shapes are *radially symmetric*, so an isotropic outer-totalistic
// rule can reach them (the neighbor-sum encodes distance-from-seed). Left/right
// asymmetric shapes (the flag) are NOT reachable by such a rule from a centered
// seed — they are included to demonstrate that wall empirically.

/** Concentric diamond bands: ringStates[0] innermost, each `ringWidth` cells thick. */
export function makeBullseye(p: MorphParams, ringStates: number[], ringWidth = 2): Uint8Array {
	const g = blank(p);
	const cx = p.W >> 1;
	const cy = p.H >> 1;
	for (let y = 0; y < p.H; y++)
		for (let x = 0; x < p.W; x++) {
			if (!isInterior(p, x, y)) continue;
			const d = Math.abs(x - cx) + Math.abs(y - cy);
			const band = Math.floor(d / ringWidth);
			if (band < ringStates.length) g[y * p.W + x] = ringStates[band];
		}
	return g;
}

/** Filled diamond with a differently-colored core. */
export function makeCoreShell(
	p: MorphParams,
	coreRadius: number,
	outerRadius: number,
	coreState = 2,
	shellState = 1
): Uint8Array {
	const g = blank(p);
	const cx = p.W >> 1;
	const cy = p.H >> 1;
	for (let y = 0; y < p.H; y++)
		for (let x = 0; x < p.W; x++) {
			if (!isInterior(p, x, y)) continue;
			const d = Math.abs(x - cx) + Math.abs(y - cy);
			if (d <= coreRadius) g[y * p.W + x] = coreState;
			else if (d <= outerRadius) g[y * p.W + x] = shellState;
		}
	return g;
}

/** Three vertical colour bands — the *anisotropic* case (unreachable by an isotropic rule). */
export function makeFrenchFlag(p: MorphParams, states: [number, number, number] = [1, 2, 3]): Uint8Array {
	const g = blank(p);
	const lo = 1;
	const hi = p.W - 2;
	const span = hi - lo + 1;
	for (let y = 1; y < p.H - 1; y++)
		for (let x = lo; x <= hi; x++) {
			const third = Math.min(2, Math.floor(((x - lo) * 3) / span));
			g[y * p.W + x] = states[third];
		}
	return g;
}

// --- arbitrary bitmap targets (asymmetric — the substrate-v2 proving ground) --
//
// Define any small shape as rows of characters, centered in the grid. A digit
// '1'..'9' sets that state (for multi-color); any other non-'.'/' ' char sets
// `state`; '.' or ' ' is background. These are the shapes that an isotropic rule
// CANNOT grow (a letter 'F' has no symmetry at all), so they are the acid test
// that directional perception actually works.

export function fromBitmap(p: MorphParams, rows: string[], state = 1): Uint8Array {
	const g = blank(p);
	const h = rows.length;
	const w = Math.max(...rows.map((r) => r.length));
	const ox = (p.W - w) >> 1;
	const oy = (p.H - h) >> 1;
	for (let r = 0; r < h; r++)
		for (let c = 0; c < rows[r].length; c++) {
			const ch = rows[r][c];
			let s = 0;
			if (ch >= '1' && ch <= '9') s = ch.charCodeAt(0) - 48;
			else if (ch !== '.' && ch !== ' ') s = state;
			if (s > 0) {
				const x = ox + c;
				const y = oy + r;
				if (isInterior(p, x, y)) g[y * p.W + x] = s;
			}
		}
	return g;
}

/** Letter 'F' — no symmetry at all: the canonical asymmetry test. */
export function letterF(p: MorphParams, state = 1): Uint8Array {
	return fromBitmap(p, ['#####', '#....', '####.', '#....', '#....', '#....'], state);
}

/** A rightward arrow — clear left/right asymmetry. */
export function arrow(p: MorphParams, state = 1): Uint8Array {
	return fromBitmap(p, ['..#...', '..##..', '######', '..##..', '..#...'], state);
}

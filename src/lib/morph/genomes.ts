// Genome helpers shared by the headless test and the browser route.
//
// A genome is the outer-totalistic rule table `genome[self*K + sum]` of length
// `genomeLength(S)`. `growthGenome` is a hand-designed rule used for M0 demos;
// `randomGenome` (deterministic) is used to stress-test every table entry.

import { genomeLength, kOf } from './ca';

/** Deterministic PRNG (Mulberry32) — reproducible random genomes. */
export function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		a |= 0;
		a = (a + 0x6d2b79f5) | 0;
		let t = Math.imul(a ^ (a >>> 15), 1 | a);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

/** A hand-designed growth rule: floods outward from the seed and differentiates. */
export function growthGenome(S: number): Uint8Array {
	const K = kOf(S);
	const g = new Uint8Array(genomeLength(S));
	const rule = (self: number, sum: number): number => {
		if (self === 0) return sum >= 1 ? 1 : 0; // birth on any live neighbor
		if (self === 1) return sum >= 3 ? 2 : 1; // mature when well-surrounded
		if (self === 2) return sum >= 5 ? 3 : 2;
		return S - 1; // top state is stable
	};
	for (let self = 0; self < S; self++) {
		for (let sum = 0; sum <= 4 * (S - 1); sum++) {
			g[self * K + sum] = rule(self, sum);
		}
	}
	return g;
}

/** A pseudo-random genome — exercises every table index and the sum extremes. */
export function randomGenome(S: number, seed: number): Uint8Array {
	const rng = mulberry32(seed);
	const g = new Uint8Array(genomeLength(S));
	for (let i = 0; i < g.length; i++) g[i] = Math.floor(rng() * S);
	return g;
}

/** Count differing cells between two equal-length grids. */
export function gridDiff(a: Uint8Array, b: Uint8Array): number {
	let d = 0;
	for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) d++;
	return d;
}

// Shape-aware fitness + behavior descriptors for morphogenesis evolution.
//
// The plain foreground-weighted accuracy in evolve.ts is fooled two ways: a
// large background inflates it (a blob scores well on a small target), and it
// is a cliff (a shape shifted one cell scores like a random shape). This module
// fixes both:
//   - Dice/IoU on the silhouette (or per-class) structurally removes the
//     all-background optimum: Dice(empty, nonempty) = 0.
//   - A distance-transform "soft" credit gives a foreground cell partial credit
//     for being NEAR the target, smoothing the landscape.
// And it exposes the behavior descriptor MAP-Elites needs, whose *horizontal
// asymmetry* axis is exactly the dimension the isotropy wall could not cross.

import type { MorphParams } from './ca';
import type { FitnessMode } from './evolve';

/** Binary Dice over foreground (state>0). Dice(∅, target) = 0. */
export function diceBinary(grid: Uint8Array, target: Uint8Array): number {
	let inter = 0;
	let ga = 0;
	let ta = 0;
	for (let i = 0; i < grid.length; i++) {
		const g = grid[i] > 0 ? 1 : 0;
		const t = target[i] > 0 ? 1 : 0;
		inter += g & t;
		ga += g;
		ta += t;
	}
	if (ga + ta === 0) return 1;
	return (2 * inter) / (ga + ta);
}

/** Mean per-class Dice over the states that appear in the target (color-aware). */
export function diceExact(grid: Uint8Array, target: Uint8Array, S: number): number {
	let sum = 0;
	let classes = 0;
	for (let c = 1; c < S; c++) {
		let inter = 0;
		let ga = 0;
		let ta = 0;
		for (let i = 0; i < grid.length; i++) {
			const g = grid[i] === c ? 1 : 0;
			const t = target[i] === c ? 1 : 0;
			inter += g & t;
			ga += g;
			ta += t;
		}
		if (ta === 0) continue; // class not in target
		classes++;
		sum += (2 * inter) / (ga + ta || 1);
	}
	return classes === 0 ? 0 : sum / classes;
}

/**
 * Chamfer distance transform of the target foreground (cells' L1 distance to the
 * nearest target-foreground cell; 0 on foreground). Two-pass, precomputed once.
 */
export function distanceTransform(target: Uint8Array, p: MorphParams): Int32Array {
	const { W, H } = p;
	const BIG = W + H;
	const dt = new Int32Array(W * H);
	for (let i = 0; i < dt.length; i++) dt[i] = target[i] > 0 ? 0 : BIG;
	// forward
	for (let y = 0; y < H; y++)
		for (let x = 0; x < W; x++) {
			const i = y * W + x;
			if (x > 0) dt[i] = Math.min(dt[i], dt[i - 1] + 1);
			if (y > 0) dt[i] = Math.min(dt[i], dt[i - W] + 1);
		}
	// backward
	for (let y = H - 1; y >= 0; y--)
		for (let x = W - 1; x >= 0; x--) {
			const i = y * W + x;
			if (x < W - 1) dt[i] = Math.min(dt[i], dt[i + 1] + 1);
			if (y < H - 1) dt[i] = Math.min(dt[i], dt[i + W] + 1);
		}
	return dt;
}

/**
 * Soft-Dice with distance-transform credit: a predicted foreground cell earns
 * `1/(1+dt)` (full credit on-target, decaying with distance) instead of 0/1.
 * Smooths the shape landscape so a near-miss is a gentle slope, not a cliff.
 */
export function softShape(grid: Uint8Array, target: Uint8Array, dt: Int32Array): number {
	let credit = 0;
	let ga = 0;
	let ta = 0;
	for (let i = 0; i < grid.length; i++) {
		const g = grid[i] > 0 ? 1 : 0;
		const t = target[i] > 0 ? 1 : 0;
		if (g) credit += 1 / (1 + dt[i]);
		ga += g;
		ta += t;
	}
	if (ga + ta === 0) return 1;
	return (2 * credit) / (ga + ta);
}

/** Behavior descriptor for MAP-Elites: (coverage, horizontal asymmetry). */
export function behaviorDescriptor(grid: Uint8Array, p: MorphParams): { coverage: number; hAsym: number } {
	const { W, H } = p;
	let fg = 0;
	let left = 0;
	let right = 0;
	const cx = W / 2;
	for (let y = 0; y < H; y++)
		for (let x = 0; x < W; x++) {
			if (grid[y * W + x] > 0) {
				fg++;
				if (x < cx) left++;
				else right++;
			}
		}
	const coverage = fg / (W * H);
	const hAsym = fg === 0 ? 0 : (right - left) / fg; // [-1, 1]
	return { coverage, hAsym };
}

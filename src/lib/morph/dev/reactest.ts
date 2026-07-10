// Quick test: is the (persistence-trained, NOT reactive-trained) held XOR rule already
// reactive? Hold input b0, flip to b1 mid-rollout, read the output — does it migrate to b1's
// answer? Sweeps placements × answer-changing transitions.  npx tsx src/lib/morph/dev/reactest.ts
import { readFileSync } from 'node:fs';
import { IDIM, forwardMarkers, loadParams } from '../../devcomp/rule';
const cfg = IDIM, C = cfg.C, SW = cfg.SW, inCh = [3, 4];
const par = loadParams(cfg, JSON.parse(readFileSync('src/lib/devcomp/params/xor_invariant.json', 'utf8')) as number[]);
function mulberry(s: number) { let a = s >>> 0; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const rng = mulberry(42), CASES = [[0, 0], [0, 1], [1, 0], [1, 1]];
let ok = 0, tot = 0;
for (let p = 0; p < 25; p++) {
	const rx = () => 1 + Math.floor(rng() * (SW - 2)), ry = () => 1 + Math.floor(rng() * (cfg.SH - 2));
	const x0 = rx(), y0 = ry(), used = new Set([y0 * SW + x0]);
	const near = () => { for (let t = 0; t < 60; t++) { const x = rx(), y = ry(), c = y * SW + x; if (!used.has(c)) { used.add(c); return c; } } let c = ry() * SW + rx(); while (used.has(c)) c = ry() * SW + rx(); used.add(c); return c; };
	const ins = [y0 * SW + x0, near()], out = near();
	for (const b0 of CASES) for (const b1 of CASES) {
		if (b0[0] === b1[0] && b0[1] === b1[1]) continue; // only answer-changing flips
		const st = forwardMarkers(cfg, par, ins, [out], b0, inCh, { steps: 200, switchAt: 100, bits2: b1 });
		const o = st[200][out * C + 0], tgt = b1[0] ^ b1[1];
		tot++; if (Math.abs(o - tgt) < 0.3) ok++;
	}
}
console.log(`held rule reactivity (NO reactive training): ${ok}/${tot} = ${(100 * ok / tot).toFixed(0)}% migrate correctly`);
console.log(`  (25 placements × up-to-12 answer-changing flips; flip at step 100, read output at 200)`);

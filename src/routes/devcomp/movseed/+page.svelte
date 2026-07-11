<script lang="ts">
	// Multi-seed rigor for the POSITION-INVARIANT (movable) rules, on the GPU, in-browser.
	// Trains the movable WIRE and the (held) movable XOR from N random seeds — each a full
	// distance-ramp run from scratch — and reports the accuracy distribution + success rate
	// over random port placements. Gives the position-invariance headline error bars.
	import { onMount } from 'svelte';
	import { makeConfig, type RuleConfig, type MovSample } from '$lib/devcomp/rule';
	import { GPUTrainer, type Sample } from '$lib/devcomp/gpuTrainer';

	let out = $state('starting…');

	function mulberry32(seed: number) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
	function wilson(k: number, n: number): [number, number] { if (!n) return [0, 0]; const z = 1.96, p = k / n, d = 1 + z * z / n; const c = p + z * z / (2 * n), h = z * Math.sqrt((p * (1 - p) + z * z / (4 * n)) / n); return [Math.max(0, (c - h) / d), Math.min(1, (c + h) / d)]; }

	function placement(rng: () => number, cfg: RuleConfig, nIn: number, maxDist: number) {
		const rx = () => 1 + Math.floor(rng() * (cfg.SW - 2)), ry = () => 1 + Math.floor(rng() * (cfg.SH - 2));
		const x0 = rx(), y0 = ry(), used = new Set([y0 * cfg.SW + x0]);
		const near = () => { for (let t = 0; t < 80; t++) { const x = rx(), y = ry(), c = y * cfg.SW + x; if (Math.abs(x - x0) + Math.abs(y - y0) <= maxDist && !used.has(c)) { used.add(c); return c; } } let c = ry() * cfg.SW + rx(); while (used.has(c)) c = ry() * cfg.SW + rx(); used.add(c); return c; };
		const ins = [y0 * cfg.SW + x0]; for (let k = 1; k < nIn; k++) ins.push(near());
		return { ins, out: near() };
	}

	type Mode = { name: string; nIn: number; inCh: number[]; aliveFrom: number; cases: number[][]; target: (b: number[]) => number; iters: number; B: number; T: number };
	const WIRE: Mode = { name: 'movable WIRE (route a bit)', nIn: 1, inCh: [0], aliveFrom: 3, cases: [[0], [1]], target: (b) => b[0], iters: 400, B: 32, T: 40 };
	const XOR: Mode = { name: 'movable XOR (held, position-invariant compute)', nIn: 2, inCh: [3, 4], aliveFrom: 5, cases: [[0, 0], [0, 1], [1, 0], [1, 1]], target: (b) => b[0] ^ b[1], iters: 700, B: 48, T: 60 };

	function makeSamples(rng: () => number, cfg: RuleConfig, m: Mode, nPlace: number, maxDist: number): (MovSample & Sample)[] {
		const s: (MovSample & Sample)[] = [];
		for (let p = 0; p < nPlace; p++) { const pl = placement(rng, cfg, m.nIn, maxDist); for (const bits of m.cases) s.push({ inPorts: pl.ins, outPort: pl.out, bits, inCh: m.inCh, target: m.target(bits) }); }
		return s;
	}
	function initFrom(cfg: RuleConfig, m: Mode, seed: number): Float32Array {
		const rng = mulberry32(seed), par = new Float32Array(cfg.P);
		for (let j = 0; j < cfg.P; j++) par[j] = (rng() - 0.5) * 0.1;
		for (let j = cfg.W2O; j < cfg.P; j++) par[j] *= 0.4;
		for (let hh = 0; hh < cfg.HD; hh++) par[cfg.W2O + 0 * cfg.HD + hh] = 0; // readout row 0 → separated channels propagate
		par[cfg.B2O + 0] = 0;
		return par;
	}

	async function runMode(cfg: RuleConfig, m: Mode, nSeeds: number, header: () => string): Promise<{ acc: number; solved: boolean }[]> {
		const FULL = cfg.SW + cfg.SH, nPl = m.B / m.cases.length;
		const tr = await GPUTrainer.create(cfg, { B: m.B, T: m.T, aliveFrom: m.aliveFrom, whold: 1 });
		const evalSet = makeSamples(mulberry32(90001), cfg, m, m.B / m.cases.length, 999); // fixed eval, exactly B samples
		const evalAcc = async () => { const o = await tr.evalOutputs(evalSet, 0, 7); let ok = 0; evalSet.forEach((s, i) => { if (Math.abs(o[i] - s.target) < 0.3) ok++; }); return ok / evalSet.length; };
		const res: { acc: number; solved: boolean }[] = [];
		for (let seed = 0; seed < nSeeds; seed++) {
			tr.setParams(initFrom(cfg, m, 100 + seed));
			const rng = mulberry32(7000 + seed * 131);
			let best = 0;
			for (let it = 1; it <= m.iters; it++) {
				const p = Math.max(0, Math.min(1, (it - 0.12 * m.iters) / (0.5 * m.iters)));
				const maxDist = Math.min(FULL, Math.round(2 + p * FULL));
				tr.setBatch(makeSamples(rng, cfg, m, nPl, maxDist), 0);
				const cos = 0.5 * (1 + Math.cos(Math.PI * (it / m.iters)));
				tr.trainStep(Math.min(1, it / 40) * 0.004 * (0.15 + 0.85 * cos), it);
				if (it % 60 === 0) { const a = await evalAcc(); if (a > best) best = a; }
				if (it % 40 === 0) { out = header() + `\n  seed ${seed}: training… best ${(best * 100).toFixed(0)}% (it ${it}/${m.iters})`; await new Promise((r) => setTimeout(r)); }
			}
			const a = await evalAcc(); best = Math.max(best, a);
			res.push({ acc: best, solved: best >= 0.9 });
			out = header() + '\n' + res.map((r, i) => `  seed ${i}: ${(r.acc * 100).toFixed(0)}%${r.solved ? ' ✓' : ''}`).join('\n');
		}
		tr.destroy();
		return res;
	}

	function summ(name: string, res: { acc: number; solved: boolean }[]): string {
		const n = res.length, k = res.filter((r) => r.solved).length;
		const mean = res.reduce((a, r) => a + r.acc, 0) / n, std = Math.sqrt(res.reduce((a, r) => a + (r.acc - mean) ** 2, 0) / n);
		const [lo, hi] = wilson(k, n);
		return `${name}: ${k}/${n} ≥90% (Wilson95 [${(100 * lo).toFixed(0)}%,${(100 * hi).toFixed(0)}%]); mean acc ${(100 * mean).toFixed(0)}±${(100 * std).toFixed(0)}%`;
	}

	onMount(async () => {
		const W = window as unknown as Record<string, unknown>;
		try {
			const cfg = makeConfig(17, 17, 16, 96, true); // IDIM: markers, 17×17
			let log = '';
			const wr = await runMode(cfg, WIRE, 10, () => log + `\n[WIRE] 10 seeds:`);
			const wireSum = summ('WIRE', wr);
			log += '\n' + wireSum + '\n'; out = log;
			W.__movseed = { wire: wireSum, wireRaw: wr, done: false };
			const xr = await runMode(cfg, XOR, 6, () => log + `\n[XOR] 6 seeds:`);
			const xorSum = summ('XOR (held)', xr);
			log += '\n' + xorSum + '\nDONE'; out = log;
			W.__movseed = { wire: wireSum, wireRaw: wr, xor: xorSum, xorRaw: xr, done: true };
		} catch (e) {
			out = 'ERROR: ' + (e as Error).message;
			W.__movseed = { ...(W.__movseed as object ?? {}), error: (e as Error).message };
		}
	});
</script>

<pre id="mv-result" style="padding:1rem;font-size:13px;white-space:pre-wrap">{out}</pre>

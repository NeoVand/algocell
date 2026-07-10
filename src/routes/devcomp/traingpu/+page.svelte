<script lang="ts">
	import { onMount } from 'svelte';
	import { RDIM, lossAndGradMarkers, forwardMarkers, type MovSample, type RuleConfig } from '$lib/devcomp/rule';
	import { GPUTrainer, type Sample } from '$lib/devcomp/gpuTrainer';
	import xorInit from '$lib/devcomp/params/xor_invariant.json'; // warm-start: computes XOR but doesn't hold

	interface Row { label: string; a: string; b: string; metric: string; ok: boolean; }
	let rows = $state<Row[]>([]);
	let status = $state('running…');
	let trainStatus = $state('');

	function mulberry32(seed: number): () => number {
		let a = seed >>> 0;
		return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
	}

	// XOR channel scheme: signal channels [3,4], readout ch0, alive from ch5.
	const IN_CH = [3, 4];
	const ALIVE_FROM = 5;
	const XOR_CASES = [[0, 0], [0, 1], [1, 0], [1, 1]];

	/** A random port placement on a grid: 2 inputs + 1 output, distinct interior cells. */
	function placement(rng: () => number, cfg: RuleConfig, maxDist = 999): { ins: number[]; out: number } {
		const rx = () => 1 + Math.floor(rng() * (cfg.SW - 2)), ry = () => 1 + Math.floor(rng() * (cfg.SH - 2));
		const x0 = rx(), y0 = ry(), used = new Set([y0 * cfg.SW + x0]);
		const near = () => { for (let t = 0; t < 60; t++) { const x = rx(), y = ry(); const c = y * cfg.SW + x; if (Math.abs(x - x0) + Math.abs(y - y0) <= maxDist && !used.has(c)) { used.add(c); return c; } } let c = ry() * cfg.SW + rx(); while (used.has(c)) c = ry() * cfg.SW + rx(); used.add(c); return c; };
		return { ins: [y0 * cfg.SW + x0, near()], out: near() };
	}

	function makeSamples(rng: () => number, cfg: RuleConfig, nPlace: number, maxDist = 999): (MovSample & Sample)[] {
		const out: (MovSample & Sample)[] = [];
		for (let p = 0; p < nPlace; p++) {
			const pl = placement(rng, cfg, maxDist);
			for (const bits of XOR_CASES) out.push({ inPorts: pl.ins, outPort: pl.out, bits, inCh: IN_CH, target: bits[0] ^ bits[1] });
		}
		return out;
	}

	// REACTIVE samples: each flips ONE input mid-rollout, so the XOR answer always changes
	// (0→1 or 1→0). Trains the rule to migrate to the new answer live, without re-seeding.
	function makeReactiveSamples(rng: () => number, cfg: RuleConfig, nPlace: number, tSwitch: number, maxDist = 999): (MovSample & Sample)[] {
		const out: (MovSample & Sample)[] = [];
		for (let p = 0; p < nPlace; p++) {
			const pl = placement(rng, cfg, maxDist);
			for (const bits0 of XOR_CASES) {
				const bits1 = bits0.slice(); bits1[Math.floor(rng() * 2)] ^= 1; // flip one input → answer flips
				out.push({ inPorts: pl.ins, outPort: pl.out, inCh: IN_CH, bits: bits0, target: bits0[0] ^ bits0[1], bits2: bits1, target2: bits1[0] ^ bits1[1], tSwitch });
			}
		}
		return out;
	}

	function readoutZeroInit(cfg: RuleConfig, seed: number): Float64Array {
		const rng = mulberry32(seed), par = new Float64Array(cfg.P);
		for (let j = 0; j < cfg.P; j++) par[j] = (rng() - 0.5) * 0.1;
		for (let j = cfg.W2O; j < cfg.P; j++) par[j] *= 0.4;
		for (let hh = 0; hh < cfg.HD; hh++) par[cfg.W2O + 0 * cfg.HD + hh] = 0; // readout row (ch0) = 0
		par[cfg.B2O + 0] = 0;
		return par;
	}

	async function run() {
		const out: Row[] = [];
		try {
			const cfg = RDIM; // C=16, HD=96, markers, fireRate 0.5 (async/stochastic updates) — reactive XOR

			// ---------- 1. gradient validation: REACTIVE rollout (transition + 2 hold windows) ----------
			const Tg = 8, WH = 2, TSW = 4, nPlace = 3; // B=12; whold 2; input flips at state 4 → validates windows A+B
			const samples = makeReactiveSamples(mulberry32(11), cfg, nPlace, TSW);
			const B = samples.length;
			const par = new Float64Array(cfg.P); // small random init → generically nonzero grads (a strong check)
			const rngP = mulberry32(7); for (let j = 0; j < cfg.P; j++) par[j] = (rngP() - 0.5) * 0.1;

			// CPU analytic grad (oracle) — reactive two-window objective + STOCHASTIC updates (seed SEED)
			const SEED = 12345;
			const { L: Lcpu, grad: gcpu } = lossAndGradMarkers(cfg, par, samples, Tg, WH, SEED);

			// 1a. finite-diff self-check of the CPU oracle on a few params
			const eps = 1e-4; let fdMaxRel = 0;
			for (const j of [11, 900, cfg.B1O + 5, cfg.W2O + 40, cfg.B2O + 3]) {
				const pp = par.slice(); pp[j] += eps; const pm = par.slice(); pm[j] -= eps;
				const fd = (lossAndGradMarkers(cfg, pp, samples, Tg, WH, SEED).L - lossAndGradMarkers(cfg, pm, samples, Tg, WH, SEED).L) / (2 * eps);
				fdMaxRel = Math.max(fdMaxRel, Math.abs(fd - gcpu[j]) / (Math.abs(fd) + 1e-8));
			}
			out.push({ label: `CPU analytic vs finite-diff (5 params, reactive whold=${WH})`, a: '', b: '', metric: fdMaxRel.toExponential(2), ok: fdMaxRel < 1e-3 });

			// GPU grad — same reactive objective (input flips at TSW)
			const trainer = await GPUTrainer.create(cfg, { B, T: Tg, aliveFrom: ALIVE_FROM, whold: WH });
			trainer.setParams(new Float32Array(par));
			trainer.setBatch(samples, TSW);
			const ggpu = await trainer.computeGrad(SEED);

			// 1b. GPU grad vs CPU analytic grad over all P
			let maxAbs = 0, maxRel = 0, denom = 0;
			for (let j = 0; j < cfg.P; j++) { const d = Math.abs(ggpu[j] - gcpu[j]); maxAbs = Math.max(maxAbs, d); maxRel = Math.max(maxRel, d / (Math.abs(gcpu[j]) + 1e-3)); denom = Math.max(denom, Math.abs(gcpu[j])); }
			out.push({ label: `GPU grad vs CPU grad (all ${cfg.P} params, T=${Tg}, B=${B})`, a: `‖g‖∞=${denom.toExponential(2)}`, b: `maxAbs=${maxAbs.toExponential(2)}`, metric: `maxRel=${maxRel.toExponential(2)}`, ok: maxRel < 5e-3 });

			// 1c. final-output agreement (CPU forwardMarkers vs GPU)
			const outsGpu = await trainer.readFinalOutputs(samples);
			let maxOutDiff = 0;
			for (let i = 0; i < B; i++) {
				const s = samples[i];
				const cpuO = forwardMarkers(cfg, par, s.inPorts, [s.outPort], s.bits, s.inCh, { steps: Tg, switchAt: s.tSwitch, bits2: s.bits2, seed: SEED })[Tg][s.outPort * cfg.C + 0];
				maxOutDiff = Math.max(maxOutDiff, Math.abs(cpuO - outsGpu[i]));
			}
			out.push({ label: 'final output: CPU vs GPU', a: `Lcpu=${Lcpu.toExponential(2)}`, b: '', metric: `maxDiff=${maxOutDiff.toExponential(2)}`, ok: maxOutDiff < 1e-3 });
			trainer.destroy();

			rows = out;
			const allOk = out.every((r) => r.ok);
			status = (allOk ? 'GRAD PASS' : 'GRAD FAIL') + ` — ${out.filter((r) => r.ok).length}/${out.length}`;
			console.log('[traingpu]', status);

			// ---------- 2. TRAIN movable XOR: varied-from-start, LARGE batch (only if grads pass) ----------
			// The user's methodology: no fixed-placement phase — every sample is a different port
			// location from iter 1. That needs a big batch so the position-invariant signal averages
			// out of the placement noise (small batch collapses to 0.5). The GPU makes big B cheap.
			if (allOk) {
				// REACTIVE + ASYNC, FROM SCRATCH, no phases: async (fireRate 0.5) updates damp the ring, so
				// we build reactivity in from iter 1 — every sample flips an input at TSW (settle answer0 in
				// window A → flip → re-settle answer1 in window B), full-range placements, random init.
				const Tt = 100, WHOLD = 24, TSW = 44, Bt = 64, iters = Number(new URLSearchParams(location.search).get('iters') ?? 2200);
				const nPl = Bt / 4;                       // placements per batch (×4 flips)
				const FULL = cfg.SW + cfg.SH;
				const ES = 7;                             // fixed eval seed → stable metric
				const tr = await GPUTrainer.create(cfg, { B: Bt, T: Tt, aliveFrom: ALIVE_FROM, whold: WHOLD });
				tr.setParams(new Float32Array(readoutZeroInit(cfg, 7))); // FROM SCRATCH
				const rng = mulberry32(2024);
				// FIXED eval sets — post-switch: did the output MIGRATE to the new answer (target2)?
				const evalMid = makeReactiveSamples(mulberry32(888), cfg, nPl, TSW, 12);
				const evalFull = makeReactiveSamples(mulberry32(999), cfg, nPl, TSW, 999);
				const accOn = async (es: typeof evalFull) => {
					const o = await tr.evalOutputs(es, TSW, ES);
					let ok = 0; es.forEach((s, i) => { if (Math.abs(o[i] - (s.target2 ?? s.target)) < 0.3) ok++; });
					return ok / es.length;
				};
				const t0 = performance.now();
				let best = 0, bestParams: Float32Array | null = null;
				for (let it = 1; it <= iters; it++) {
					// varied + reactive from iter 1; distance RAMPS from close→full so the XOR combine can
					// catch at first (near-identity init can't bootstrap it at full range), then generalizes.
					const p = Math.max(0, Math.min(1, (it - 0.12 * iters) / (0.45 * iters)));
					const maxDist = Math.min(FULL, Math.round(2 + p * FULL));
					tr.setBatch(makeReactiveSamples(rng, cfg, nPl, TSW, maxDist), TSW);
					const cos = 0.5 * (1 + Math.cos(Math.PI * (it / iters)));
					const lr = Math.min(1, it / 40) * 0.004 * (0.1 + 0.9 * cos); // from-scratch schedule
					tr.trainStep(lr, it);                    // seed = it → masks vary per iter
					if (it % 50 === 0 || it === iters) {
						const accM = await accOn(evalMid);
						const accF = await accOn(evalFull);              // post-switch migration, full range
						if (accF > best) { best = accF; bestParams = await tr.readParams(); } // keep-best snapshot
						const itps = (it / (performance.now() - t0)) * 1000;
						trainStatus = `iter ${it}/${iters} · react mid ${(accM * 100).toFixed(0)}% · full ${(accF * 100).toFixed(0)}% · best ${(best * 100).toFixed(0)}% · ${itps.toFixed(1)} it/s (B=${Bt},T=${Tt},async)`;
						console.log('[traingpu-train]', trainStatus);
						(window as unknown as Record<string, unknown>).__xorTrain = { it, iters, accM, accF, best, itps };
					}
				}
				const params = Array.from(bestParams ?? await tr.readParams()); // export the BEST rule, not the last
				(window as unknown as Record<string, unknown>).__xorTrained = { params, acc: best };
				trainStatus = `DONE · best full-range react-acc ${(best * 100).toFixed(0)}% — params on window.__xorTrained`;
				console.log('[traingpu-train]', trainStatus);
				tr.destroy();
			}
		} catch (e) { status = 'error: ' + (e as Error).message; console.error(e); }
	}

	onMount(run);
</script>

<svelte:head><title>devcomp — GPU trainer validation</title></svelte:head>

<main>
	<h1>GPU/browser trainer — gradient validation vs CPU reference</h1>
	<p class="status" class:pass={status.includes('PASS')} class:fail={status.includes('FAIL') || status.startsWith('error')}>{status}</p>
	{#if trainStatus}<p class="train" class:pass={trainStatus.startsWith('DONE')}>{trainStatus}</p>{/if}
	<table>
		<thead><tr><th>check</th><th></th><th></th><th>metric</th><th></th></tr></thead>
		<tbody>
			{#each rows as r (r.label)}
				<tr class:bad={!r.ok}><td>{r.label}</td><td>{r.a}</td><td>{r.b}</td><td>{r.metric}</td><td>{r.ok ? '✓' : '✗'}</td></tr>
			{/each}
		</tbody>
	</table>
</main>

<style>
	main { font-family: ui-monospace, monospace; padding: 28px; color: #e6edf3; background: #0a0d12; min-height: 100vh; }
	h1 { font-size: 19px; }
	.status { font-size: 15px; font-weight: 700; }
	.status.pass { color: #34d399; } .status.fail { color: #f87171; }
	.train { font-size: 14px; color: #7dd3fc; } .train.pass { color: #34d399; font-weight: 700; }
	table { border-collapse: collapse; margin-top: 14px; font-size: 13px; }
	th, td { border: 1px solid #23303d; padding: 4px 10px; text-align: left; }
	tr.bad { background: rgba(248, 113, 113, 0.12); }
</style>

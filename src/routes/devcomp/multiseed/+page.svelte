<script lang="ts">
	// Multi-seed rigor, run in-browser on the GPU: train the 1-bit adder from N random
	// seeds (one full distance-curriculum run each, no restarts = honest reproducibility)
	// and report the success rate + a Wilson 95% CI. This is the S8-bar rigor the review
	// asked for, for a headline beyond the gate.
	import { onMount } from 'svelte';
	import { ADIM } from '$lib/devcomp/rule';
	import { GPUTrainer, type Sample } from '$lib/devcomp/gpuTrainer';

	let out = $state('starting…');
	const NSEEDS = 8;

	function wilson(k: number, n: number): [number, number] {
		if (n === 0) return [0, 0];
		const z = 1.96, p = k / n, d = 1 + z * z / n;
		const c = p + z * z / (2 * n), h = z * Math.sqrt((p * (1 - p) + z * z / (4 * n)) / n);
		return [Math.max(0, (c - h) / d), Math.min(1, (c + h) / d)];
	}

	onMount(async () => {
		try {
			const cfg = ADIM, Tt = 30, aliveFrom = 1, B = 8;
			const iy = cfg.SH >> 1, inCol = 2, OUT = cfg.SW - 3;
			const inPorts = [(iy - 1) * cfg.SW + inCol, iy * cfg.SW + inCol, (iy + 1) * cfg.SW + inCol];
			const inCh = [0, 0, 0];
			const truth = (a: number, b: number, c: number) => [a ^ b ^ c, a + b + c >= 2 ? 1 : 0];
			const mkCases = (col: number): Sample[] => {
				const outPorts = [(iy - 1) * cfg.SW + col, (iy + 1) * cfg.SW + col];
				const cs: Sample[] = [];
				for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let c = 0; c < 2; c++)
					cs.push({ inPorts, inCh, bits: [a, b, c], outPorts, targets: truth(a, b, c) });
				return cs;
			};
			const t = await GPUTrainer.create(cfg, { B, T: Tt, aliveFrom, whold: 1 });

			const results: { seed: number; acc: number; solved: boolean; secs: number }[] = [];
			for (let seed = 0; seed < NSEEDS; seed++) {
				const t0 = performance.now();
				// seeded init
				let s = (seed + 1) * 2654435761 >>> 0;
				const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff; return s / 0x7fffffff; };
				const par = new Float32Array(cfg.P);
				for (let j = 0; j < cfg.P; j++) par[j] = (rnd() - 0.5) * 0.12;
				for (let j = cfg.W2O; j < cfg.P; j++) par[j] *= 0.5;
				t.setParams(par);
				let acc = 0;
				for (let col = inCol + 2; col <= OUT; col++) {
					const warm = col > inCol + 2;
					const lrHi = warm ? 0.003 : 0.008, lrLo = warm ? 0.0006 : 0.003;
					const cases = mkCases(col);
					t.setBatch(cases, 0);
					const iters = warm ? 300 : 500;
					for (let it = 1; it <= iters; it++) {
						const cos = 0.5 * (1 + Math.cos(Math.PI * (it / iters)));
						t.trainStep(Math.min(1, it / 20) * (lrLo + (lrHi - lrLo) * cos), it);
						if (it % 50 === 0) await new Promise((r) => setTimeout(r)); // yield: keep the page responsive + let the GPU queue drain
					}
					out = `seed ${seed}/${NSEEDS - 1} … training col ${col}/${OUT}` + (results.length ? '\n\n' + results.map((r) => `  seed ${r.seed}: ${r.solved ? 'SOLVED' : 'failed'} ${r.acc}/8`).join('\n') : '');
					if (col === OUT) {
						const o = await t.readFinalOutputsMulti(cases);
						acc = cases.filter((c, i) => c.targets!.every((tg, k) => Math.abs(o[i][k] - tg) < 0.3)).length;
					}
				}
				const secs = (performance.now() - t0) / 1000;
				results.push({ seed, acc, solved: acc === 8, secs });
				const nSolved = results.filter((r) => r.solved).length;
				const [lo, hi] = wilson(nSolved, results.length);
				out =
					`1-bit adder — multi-seed on GPU (${results.length}/${NSEEDS} seeds done)\n\n` +
					results.map((r) => `  seed ${r.seed}: ${r.solved ? 'SOLVED' : 'failed'}  ${r.acc}/8  (${r.secs.toFixed(0)}s)`).join('\n') +
					`\n\n  success ${nSolved}/${results.length}  Wilson95 [${(100 * lo).toFixed(0)}%, ${(100 * hi).toFixed(0)}%]`;
			}
			t.destroy();
			const nSolved = results.filter((r) => r.solved).length;
			const [lo, hi] = wilson(nSolved, NSEEDS);
			out += `\n\nDONE. success ${nSolved}/${NSEEDS} = ${(100 * nSolved / NSEEDS).toFixed(0)}%  Wilson95 [${(100 * lo).toFixed(0)}%, ${(100 * hi).toFixed(0)}%]`;
		} catch (e) {
			out = 'ERROR: ' + (e as Error).message + '\n' + (e as Error).stack;
		}
	});
</script>

<pre id="ms-result" style="padding:1rem;font-size:13px;white-space:pre-wrap">{out}</pre>

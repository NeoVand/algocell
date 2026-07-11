<script lang="ts">
	// Validation: the GPU trainer's NON-MARKER multi-input/multi-output path vs the
	// finite-diff-checked CPU reference (lossAndGradFixed). Uses the ADIM adder config
	// (3 inputs → 2 outputs, no markers). A pass here means the N-in/M-out extension
	// computes the same gradient as the CPU — the discipline the marker path already meets.
	import { onMount } from 'svelte';
	import { ADIM, lossAndGradFixed, type FixedSample } from '$lib/devcomp/rule';
	import { GPUTrainer, type Sample } from '$lib/devcomp/gpuTrainer';

	let result = $state('running…');
	let ok = $state(false);

	onMount(async () => {
		try {
			const cfg = ADIM; // 11×11, C=16, HD=64, non-markers
			const T = 20, aliveFrom = 1, whold = 1;
			const iy = cfg.SH >> 1, inCol = 2, outCol = cfg.SW - 3;
			const inPorts = [(iy - 1) * cfg.SW + inCol, iy * cfg.SW + inCol, (iy + 1) * cfg.SW + inCol];
			const outPorts = [(iy - 1) * cfg.SW + outCol, (iy + 1) * cfg.SW + outCol];
			const inCh = [0, 0, 0];
			const cases: FixedSample[] = [];
			for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let cin = 0; cin < 2; cin++) {
				const sum = a ^ b ^ cin, carry = a + b + cin >= 2 ? 1 : 0;
				cases.push({ inPorts, inCh, bits: [a, b, cin], outPorts, targets: [sum, carry] });
			}
			const B = cases.length; // 8

			// unsaturated near-identity params → clean gradient (last layer 0)
			const par = new Float64Array(cfg.P);
			let s = 12345; const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff; return s / 0x7fffffff; };
			for (let j = 0; j < cfg.P; j++) par[j] = (rnd() - 0.5) * 0.1;
			for (let j = cfg.W2O; j < cfg.P; j++) par[j] = 0;

			const cpu = lossAndGradFixed(cfg, par, cases, T, aliveFrom, whold);

			const t = await GPUTrainer.create(cfg, { B, T, aliveFrom, whold });
			t.setParams(Float32Array.from(par));
			t.setBatch(cases as Sample[], 0);
			const gGPU = await t.computeGrad(0);
			const outs = await t.readFinalOutputsMulti(cases as Sample[]);
			t.destroy();

			let maxAbs = 0, maxRel = 0, gnorm = 0;
			for (let j = 0; j < cfg.P; j++) {
				const a = cpu.grad[j], b = gGPU[j];
				maxAbs = Math.max(maxAbs, Math.abs(a - b));
				maxRel = Math.max(maxRel, Math.abs(a - b) / (Math.abs(a) + 1e-6));
				gnorm += a * a;
			}
			ok = maxRel < 3e-3 && maxAbs < 1e-4;
			result =
				`[1] GRADIENT — NON-MARKER GPU vs CPU (ADIM adder, ${B} cases, T=${T}, 3-in/2-out)\n` +
				`    maxAbs ${maxAbs.toExponential(2)}  maxRel ${maxRel.toExponential(2)}  ‖gradCPU‖ ${Math.sqrt(gnorm).toExponential(2)}  => ${ok ? 'PASS' : 'FAIL'}\n\n[2] end-to-end training…`;

			await trainAdderGPU(cfg);
		} catch (e) {
			result = 'ERROR: ' + (e as Error).message + '\n' + (e as Error).stack;
		}
	});

	// End-to-end: train the 1-bit adder from scratch on the GPU via a distance curriculum,
	// then report accuracy + throughput (the payoff — this is what makes multi-seed affordable).
	async function trainAdderGPU(cfg: typeof ADIM): Promise<void> {
		const Tt = 30, aliveFrom = 1, B = 8;
		const iy = cfg.SH >> 1, inCol = 2, OUT = cfg.SW - 3;
		const inPorts = [(iy - 1) * cfg.SW + inCol, iy * cfg.SW + inCol, (iy + 1) * cfg.SW + inCol];
		const inCh = [0, 0, 0];
		const truth = (a: number, b: number, cin: number) => [a ^ b ^ cin, a + b + cin >= 2 ? 1 : 0];
		const mkCases = (col: number): Sample[] => {
			const outPorts = [(iy - 1) * cfg.SW + col, (iy + 1) * cfg.SW + col];
			const cs: Sample[] = [];
			for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) for (let cin = 0; cin < 2; cin++)
				cs.push({ inPorts, inCh, bits: [a, b, cin], outPorts, targets: truth(a, b, cin) });
			return cs;
		};
		const t = await GPUTrainer.create(cfg, { B, T: Tt, aliveFrom, whold: 1 });
		let s = 999; const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff; return s / 0x7fffffff; };
		const par = new Float32Array(cfg.P);
		for (let j = 0; j < cfg.P; j++) par[j] = (rnd() - 0.5) * 0.12;
		for (let j = cfg.W2O; j < cfg.P; j++) par[j] *= 0.5;
		t.setParams(par);
		const t0 = performance.now();
		let steps = 0, log = '';
		for (let col = inCol + 2; col <= OUT; col++) {
			const warm = col > inCol + 2;                 // warm stages: REFINE gently (don't kick out of basin)
			const lrHi = warm ? 0.003 : 0.008, lrLo = warm ? 0.0006 : 0.003;
			const cases = mkCases(col);
			t.setBatch(cases, 0);
			const iters = warm ? 300 : 500;
			for (let it = 1; it <= iters; it++) {
				const cos = 0.5 * (1 + Math.cos(Math.PI * (it / iters)));
				const lr = Math.min(1, it / 20) * (lrLo + (lrHi - lrLo) * cos);
				t.trainStep(lr, it); steps++;
			}
			const outs = await t.readFinalOutputsMulti(cases);
			const acc = cases.filter((c, i) => c.targets!.every((tg, k) => Math.abs(outs[i][k] - tg) < 0.3)).length;
			log += `    col ${col}: acc ${acc}/8\n`;
			result = result.replace(/\[2\][\s\S]*/, `[2] end-to-end training (distance curriculum, GPU):\n${log}`);
		}
		const secs = (performance.now() - t0) / 1000;
		const cases = mkCases(OUT); t.setBatch(cases, 0);
		const outs = await t.readFinalOutputsMulti(cases);
		const acc = cases.filter((c, i) => c.targets!.every((tg, k) => Math.abs(outs[i][k] - tg) < 0.3)).length;
		t.destroy();
		result = result.replace(/\[2\][\s\S]*/, `[2] end-to-end training (distance curriculum, GPU):\n${log}` +
			`    FINAL 1-bit adder on GPU: ${acc}/8 cases  in ${steps} steps, ${secs.toFixed(1)}s (${(steps / secs).toFixed(0)} it/s)\n` +
			`    => ${acc === 8 ? 'PASS — GPU trains the multi-IO adder end-to-end' : 'partial'}`);
		ok = ok && acc === 8;
	}
</script>

<pre id="valfixed-result" style="padding:1rem;font-size:13px;white-space:pre-wrap">{result}</pre>
<div id="valfixed-ok" data-ok={ok}></div>

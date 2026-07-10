// S8 orchestrator — runs the ablation × multi-seed grid with CPU concurrency
// control, aggregates per condition, writes s8_results.json.
//   npx tsx src/lib/morph/dev/s8_run.ts
import { execFile } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { cpus } from 'node:os';

const CONDS = ['baseline', 'iso', 'id', 'norelu', 'hd4', 'hd8', 'hd16', 'hd32', 'hd96'];
const SEEDS = [0, 1, 2, 3, 4, 5, 6, 7];
const ITERS = '300', RESTARTS = '2';
const CONC = Math.max(2, cpus().length - 2);

interface SeedResult { cond: string; seed: number; solved: boolean; maxSolvedDist: number; finalLoss: number; secs: number; ok: boolean; }

function runOne(cond: string, seed: number): Promise<SeedResult> {
	return new Promise((resolve) => {
		const t0 = Date.now();
		execFile('npx', ['tsx', 'src/lib/morph/dev/s8.ts'], { env: { ...process.env, COND: cond, SEEDS: String(seed), ITERS, RESTARTS }, timeout: 400_000, maxBuffer: 1 << 20 },
			(err, stdout) => {
				const secs = (Date.now() - t0) / 1000;
				const line = (stdout || '').split('\n').find((l) => l.startsWith('RESULT '));
				if (err || !line) { console.log(`  [FAIL] ${cond} seed ${seed} (${secs.toFixed(0)}s) ${err ? String(err.message).slice(0, 60) : 'no RESULT'}`); return resolve({ cond, seed, solved: false, maxSolvedDist: -1, finalLoss: NaN, secs, ok: false }); }
				const r = JSON.parse(line.slice(7));
				const p = r.per[0];
				console.log(`  ${cond.padEnd(9)} seed ${seed}: ${p.solved ? 'SOLVED' : 'failed'} d${p.maxSolvedDist}/5 loss ${p.finalLoss.toFixed(4)} (${secs.toFixed(0)}s)`);
				resolve({ cond, seed, solved: p.solved, maxSolvedDist: p.maxSolvedDist, finalLoss: p.finalLoss, secs, ok: true });
			});
	});
}

async function main() {
	const jobs: { cond: string; seed: number }[] = [];
	for (const c of CONDS) for (const s of SEEDS) jobs.push({ cond: c, seed: s });
	console.log(`S8 grid: ${CONDS.length} conditions × ${SEEDS.length} seeds = ${jobs.length} runs, concurrency ${CONC}\n`);
	const results: SeedResult[] = [];
	let idx = 0;
	async function worker(): Promise<void> { while (idx < jobs.length) { const j = jobs[idx++]; results.push(await runOne(j.cond, j.seed)); } }
	const t0 = Date.now();
	await Promise.all(Array.from({ length: CONC }, () => worker()));
	// aggregate per condition
	const agg = CONDS.map((cond) => {
		const rs = results.filter((r) => r.cond === cond && r.ok);
		const losses = rs.map((r) => r.finalLoss).sort((a, b) => a - b);
		const mean = losses.reduce((a, b) => a + b, 0) / (losses.length || 1);
		const std = Math.sqrt(losses.reduce((a, b) => a + (b - mean) ** 2, 0) / (losses.length || 1));
		const nSolved = rs.filter((r) => r.solved).length;
		return { cond, nSeeds: rs.length, nSolved, successRate: nSolved / (rs.length || 1), meanLoss: mean, stdLoss: std, medianLoss: losses[losses.length >> 1] ?? NaN, meanSolvedDist: rs.reduce((a, r) => a + r.maxSolvedDist, 0) / (rs.length || 1) };
	});
	const out = { config: { CONDS, SEEDS, ITERS, RESTARTS, task: 'XOR gate, 9x9, distance curriculum to d=5' }, agg, results, totalSecs: (Date.now() - t0) / 1000 };
	writeFileSync('docs/s8_results.json', JSON.stringify(out, null, 2));
	console.log('\n=== S8 SUMMARY (XOR gate, 8 seeds, distance curriculum → d=5) ===');
	console.log('condition   P      success   meanLoss±std        meanSolvedDist/5');
	for (const a of agg) console.log(`  ${a.cond.padEnd(9)} ${String(results.find((r) => r.cond === a.cond)?.ok ? '' : '').padEnd(0)} ${a.nSolved}/${a.nSeeds} (${(100 * a.successRate).toFixed(0)}%)   ${a.meanLoss.toFixed(4)}±${a.stdLoss.toFixed(4)}   ${a.meanSolvedDist.toFixed(2)}`);
	console.log(`\ntotal ${out.totalSecs.toFixed(0)}s → docs/s8_results.json`);
}
main();

// One line per saved tick of each run file (the look.json written by scripts/free-look.mjs).
//   node docs/free-running/summarise.mjs docs/free-running/runs/b_seed1.json ...
import { readFileSync } from 'node:fs';
for (const dir of process.argv.slice(2)) {
	const j = JSON.parse(readFileSync(dir.endsWith('.json') ? dir : dir + '/look.json', 'utf8'));
	console.log(dir.split('/').pop(), JSON.stringify(j.q), 'heads', j.heads);
	for (const r of j.rows) {
		const best = r.sameRight.map((v, i) => [v, i + 1]).sort((a, b) => b[0] - a[0]).slice(0, 3).map(([v, k]) => `${k}:${v.toFixed(2)}`).join(' ');
		console.log(`  t=${String(r.tick).padStart(7)} H=${r.entropy.toFixed(2)} zero=${(r.zero * 100).toFixed(1)}% run(mean/med last)=${r.meanRunLength}/${r.lastRunMedian} never=${r.headsNeverRestarted} wait=${r.waitingHeads} lag[${best}] down1=${r.sameDown[0]} bytes ${r.topBytes.slice(0, 4).map((b) => b[0] + ' ' + (b[1] * 100).toFixed(0) + '%').join(', ')} | pairs ${r.topPairs.slice(0, 4).map((b) => b[0] + ' ' + (b[1] * 100).toFixed(0) + '%').join(', ')} | ${r.seconds}s`);
	}
}

// What are processors doing in the waiting state? Reads a sheet and processors saved by
// scripts/free-look.mjs (sheet_<tick>.u8, heads_<tick>.u32).
//   node docs/free-running/waiting.mjs <run folder> <tick>
import { readFileSync } from 'node:fs';
const [dir, tick] = process.argv.slice(2);
// Index of each register inside a processor's 40 words (freeLayout(settings).state).
const L = {"a":0,"f":1,"b":2,"c":3,"d":4,"e":5,"h":6,"l":7,"a_":8,"f_":9,"b_":10,"c_":11,"d_":12,"e_":13,"h_":14,"l_":15,"ix":16,"iy":17,"sp":18,"pc":19,"i":20,"r":21,"iff1":22,"iff2":23,"halted":24};
const sheet = new Uint8Array(readFileSync(`${dir}/sheet_${tick}.u8`));
const hb = readFileSync(`${dir}/heads_${tick}.u32`);
const heads = new Uint32Array(hb.buffer, hb.byteOffset, hb.byteLength / 4);
const W = Math.sqrt(sheet.length), HW = 40, n = heads.length / HW;
const signed = (b) => (b >= 128 ? b - 256 : b), wrap = (v) => ((v % W) + W) % W;
const place = (o, a) => {
	const f = heads[o + 39]; const al = signed(a & 255), ac = signed((a >> 8) & 255);
	let dx = f & 1 ? ac : al, dy = f & 1 ? al : ac; if (f & 2) dx = -dx; if (f & 4) dy = -dy;
	return wrap(heads[o + 33] + dy) * W + wrap(heads[o + 32] + dx);
};
const op = new Map(); let rstLoop = 0, at38 = 0, pcRow0 = 0; const ages = [];
for (let k = 0; k < n; k++) {
	const o = k * HW, pc = heads[o + L.pc] & 0xffff, b = sheet[place(o, pc)];
	op.set(b, (op.get(b) ?? 0) + 1);
	if (pc >> 8 === 0) pcRow0++;
	if (pc === 0x38) at38++;
	if (pc === 0x38 && b === 0xff) { rstLoop++; ages.push(heads[o + 34]); }
}
const top = [...op.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10).map(([b, c]) => `${b.toString(16).padStart(2, '0')} ${(100 * c / n).toFixed(1)}%`);
ages.sort((a, b) => a - b);
console.log(JSON.stringify({ dir: dir.split('/').pop(), tick, processors: n, nextOpcodeTop: top, pcInLandingLine: +(pcRow0 / n).toFixed(3), pcAt0x38: at38, pcAt0x38OnFF: rstLoop, shareOnRst38Loop: +(rstLoop / n).toFixed(4), loopAgeMedian: ages[ages.length >> 1] ?? null }));

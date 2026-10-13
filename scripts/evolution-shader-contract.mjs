import { build } from 'vite';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const directory = await mkdtemp(path.join(os.tmpdir(), 'algocell-shader-contract-'));
const hashes = {};
const sha = (code) => createHash('sha256').update(code).digest('hex');
const normalize = (code) =>
	code
		.replace(/\/\/[^\n]*/g, '')
		.replace(/\/\*[\s\S]*?\*\//g, '')
		.replace(/\s+/g, '');
for (const [name, repo] of [
	['baseline', process.env.BASELINE_REPO ?? '/tmp/algocell-evolution-ee67698'],
	['candidate', process.cwd()]
]) {
	const output = path.join(directory, name);
	await build({
		configFile: false,
		root: repo,
		logLevel: 'warn',
		resolve: { alias: { $lib: path.join(repo, 'src/lib') } },
		build: {
			outDir: output,
			emptyOutDir: true,
			lib: {
				entry: path.join(repo, 'src/lib/gpu/shaders.ts'),
				formats: ['es'],
				fileName: () => 'shaders.mjs'
			},
			minify: true
		}
	});
	const { createSimShader } = await import(pathToFileURL(path.join(output, 'shaders.mjs')));
	hashes[name] = [];
	for (const [isa, compat, variant] of [
		['z80', '0.2', 'classic'],
		['z80', '0.3', 'classic'],
		['z80', '0.3', 'zilog-nmos'],
		['m6502', '0.3', 'nmos'],
		['i8086', '0.3', 'nmos']
	]) {
		const settings = {
			isa,
			compat,
			variant,
			length: 16,
			ring: 32,
			steps: 128,
			width: 200,
			height: 200,
			topology: 'square',
			seed: 6,
			pairAttempts: 8192,
			mutationExponent: 4,
			lethalProbability: 0,
			randomRegisters: false,
			ablation: [],
			stack: isa === 'z80' ? 'legacy-sp' : 'first-push',
			dialMode: isa === 'z80' ? 'historical-pcg' : 'counter'
		};
		const wgsl = createSimShader('square', settings, false);
		hashes[name].push({
			isa,
			compat,
			variant,
			bytes: wgsl.length,
			raw: sha(wgsl),
			normalized: sha(normalize(wgsl))
		});
	}
}
const results = hashes.baseline.map((baseline, i) => ({
	isa: baseline.isa,
	compat: baseline.compat,
	variant: baseline.variant,
	baseline,
	candidate: hashes.candidate[i],
	identical: baseline.raw === hashes.candidate[i].raw,
	identicalTokens: baseline.normalized === hashes.candidate[i].normalized
}));
const output = process.env.CONTRACT_OUTPUT ?? 'docs/review/evolution/shader-contract.json';
await mkdir(path.dirname(output), { recursive: true });
await writeFile(
	output,
	JSON.stringify(
		{
			method:
				'Generate the actual default uninstrumented createSimShader for each core from independent Vite bundles. SHA-256 raw WGSL and WGSL after removing comments/whitespace only; no operators, identifiers or values normalized. No GPU is acquired.',
			results
		},
		null,
		2
	) + '\n'
);
for (const result of results)
	console.log(
		`${result.isa} ${result.compat} ${result.variant}: raw=${result.identical}, tokens=${result.identicalTokens}`
	);
if (results.some((result) => !result.identicalTokens)) process.exitCode = 1;

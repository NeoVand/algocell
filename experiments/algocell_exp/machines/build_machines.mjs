#!/usr/bin/env node
// Regenerate, or check, the Zilion machines vendored in this directory.
//
//   (cd <zilion> && npm run build)
//   node experiments/algocell_exp/machines/build_machines.mjs [--zilion <dir>]           # write
//   node experiments/algocell_exp/machines/build_machines.mjs [--zilion <dir>] --check   # verify, write nothing
//   node experiments/algocell_exp/machines/build_machines.mjs [--zilion <dir>] --spec worlds_assays.json [--check]
//
// For each entry of the worlds*.json files beside this script it calls Zilion's buildMachine (codegen stays in TypeScript, as in
// zilion's scripts/export-machines.mjs) and writes <name>.wgsl and <name>.manifest.json, then one provenance file per
// spec file with the Zilion commit and each machine's SHA-256s: PROVENANCE.md for worlds.json, PROVENANCE_<x>.md for
// worlds_<x>.json. `--spec <file>` builds (or checks) only the machines of that spec file and its provenance file, so
// a study can vendor its machines from a newer Zilion commit without regenerating anyone else's. Unlike zilion's
// exporter it also accepts the skeleton cores (m6502, bff), which ZSoup uses to show its plumbing on a second ISA; a
// skeleton machine is marked as such and is not for results.
//
// --check exits 1 unless every .wgsl and .manifest.json it builds equals what is on disk, and each provenance file
// equals what would be written for the commit it records; it also reports whether that commit is the checkout's HEAD. The default Zilion checkout is ../zilion beside the Algocell repository root.

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../../..");
const args = process.argv.slice(2);
const opt = (n, d) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const zdir = resolve(opt("--zilion", process.env.ZILION_DIR ?? join(repoRoot, "../zilion")));
const check = args.includes("--check");
const onlySpec = opt("--spec", null);

const dist = (p) => pathToFileURL(join(zdir, "dist", p)).href;
if (!existsSync(join(zdir, "dist/index.js"))) {
  console.error(`no Zilion build at ${zdir}/dist (run npm run build there, or pass --zilion <dir>)`);
  process.exit(2);
}
const Z = await import(dist("index.js"));
const { m6502 } = await import(dist("cores/m6502/index.js"));
const { bff } = await import(dist("cores/bff/index.js"));
const CORES = { z80: Z.z80, m6502, bff };

const git = (...a) => execFileSync("git", ["-C", zdir, ...a], { encoding: "utf8" }).trim();
const head = git("rev-parse", "HEAD");
const branch = git("rev-parse", "--abbrev-ref", "HEAD");
const dirty = git("status", "--porcelain", "--untracked-files=no") !== "";
const sha256 = (s) => createHash("sha256").update(s).digest("hex");

// Every worlds*.json beside this script (worlds.json first, then the others by name), so studies vendor machines in
// their own spec files; a machine name may appear only once across them.
const specFiles = readdirSync(here).filter((f) => /^worlds.*\.json$/.test(f)).sort((a, b) => (a === "worlds.json" ? -1 : b === "worlds.json" ? 1 : a.localeCompare(b)));
const specs = {};
for (const f of specFiles) {
  for (const [name, s] of Object.entries(JSON.parse(readFileSync(join(here, f), "utf8")))) {
    if (name in specs) throw new Error(`machine '${name}' is defined twice (${specs[name].__file} and ${f})`);
    specs[name] = { ...s, __file: f };
  }
}
if (onlySpec !== null && !specFiles.includes(onlySpec)) throw new Error(`--spec ${onlySpec}: no such spec file beside this script (have: ${specFiles.join(", ")})`);
const builtSpecs = onlySpec === null ? specFiles : [onlySpec];
const provFile = (f) => (f === "worlds.json" ? "PROVENANCE.md" : `PROVENANCE_${f.replace(/^worlds_?/, "").replace(/\.json$/, "")}.md`);
const files = {};
const perSpec = Object.fromEntries(builtSpecs.map((f) => [f, { rows: [], z80Plain: [], names: [] }]));
for (const [name, s] of Object.entries(specs)) {
  if (!builtSpecs.includes(s.__file)) continue;
  const { rows, z80Plain, names } = perSpec[s.__file];
  names.push(name);
  if (!/^[\w.-]+$/.test(name)) throw new Error(`invalid machine name '${name}'`);
  const { note, __file, ...spec } = s; // a note is documentation, never part of the machine
  const coreOf = (id) => {
    const isa = CORES[id];
    if (!isa) throw new Error(`${name}: unknown isa '${id}' (known: ${Object.keys(CORES).join(", ")})`);
    return isa;
  };
  if (spec.core === undefined) throw new Error(`${name}: give the core as "core": { "isa": …, "compat": … }`);
  // Zilion's Z80 default is its newer semantics; every Algocell machine names the semantics it runs (the published
  // results ran on compat '0.2'), so a Z80 world without an explicit compat is refused here.
  if (spec.core.isa === "z80" && spec.core.compat === undefined) throw new Error(`${name}: a Z80 machine must state its compat ('0.2' for Algocell's published semantics)`);
  const m = Z.buildMachine({ ...spec, core: { ...spec.core, isa: coreOf(spec.core.isa) } });
  const manifest = JSON.stringify(m.manifest, null, 1) + "\n";
  files[`${name}.wgsl`] = m.wgsl;
  files[`${name}.manifest.json`] = manifest;
  const c = m.manifest.core;
  const mem = m.manifest.memory;
  rows.push(
    `| \`${name}\` | ${c.id} ${c.coreVersion} (${c.status}; hooks: ${c.hooks.join(", ") || "none"}) | **${c.compat}**${spec.core.compat !== undefined ? " (stated)" : " (default)"} | ` +
      `${mem.segments.map((g) => `${g.name}:${g.bytes}`).join(" + ")} on ${mem.ringBytes} | \`${m.manifest.sha256.core}\` | ` +
      `\`${m.manifest.sha256.world}\` | \`${m.manifest.sha256.wgsl}\` | ${Buffer.byteLength(m.wgsl)} |`,
  );
  if (sha256(m.wgsl) !== m.manifest.sha256.wgsl) throw new Error(`${name}: manifest sha256.wgsl does not match the WGSL`);
  if (c.id === "z80" && c.hooks.length === 0) {
    if (m.manifest.sha256.core !== sha256(Z.Z80_CORE_WGSL)) throw new Error(`${name}: a hook-free Z80 machine whose core is not Z80_CORE_WGSL`);
    z80Plain.push(`\`${name}\``);
  }
}

const provenance = (f, commit, br, wasDirty) => {
  const { rows, z80Plain, names } = perSpec[f];
  return `# Vendored Zilion machines: provenance

Generated by \`build_machines.mjs\` from \`${f}\`; do not edit by hand. Regenerate with
\`(cd ../zilion && npm run build) && node experiments/algocell_exp/machines/build_machines.mjs${f === "worlds.json" ? "" : ` --spec ${f}`}\`; verify with \`--check\`.

- Zilion: \`@neovand/zilion\` ${Z.ZILION_VERSION}, commit \`${commit}\` (branch \`${br}\`${wasDirty ? ", WORKING TREE DIRTY" : ""})
- Manifest schema: \`${Z.MANIFEST_SCHEMA}\`
- Core SHA-256 = \`manifest.sha256.core\` (the core text as emitted, with its hooks); world SHA-256 = canonical JSON
  of the world spec; WGSL SHA-256 = the whole generated shader (checked against the file by \`zmachine.load_machine\`).
- \`Z80_CORE_WGSL\` (the 0.2.0 Z80 core text, byte-identical to the v0.2.0 tag by zilion's \`npm run check:bytes\`):
  SHA-256 \`${sha256(Z.Z80_CORE_WGSL)}\`. A Z80 machine with no hooks emits exactly this text (${z80Plain.join(", ") || "none in this file"});
  the others insert the hook calls at zilion's fixed insertion points.
- Compat: every Z80 machine states \`compat: '0.2'\` in ${f} (the exporter refuses a Z80 world without one), the
  semantics of the 0.2.0 core that all published Algocell results ran on; "Core" is the Z80 descriptor's version, whose
  frozen \`'0.2'\` compat emits the 0.2.0 core text (the core SHA-256 below).
${names.some((n) => n.startsWith("m6502skel_")) ? "- \`m6502skel_*\` uses the SKELETON 6502 core (a handful of opcodes): plumbing only, never results.\n" : ""}
| Machine | Core | Compat | Segments on ring P | Core SHA-256 | World SHA-256 | WGSL SHA-256 | WGSL bytes |
|---|---|---|---|---|---|---|---|
${rows.join("\n")}
`;
};

if (check) {
  let stale = 0;
  for (const [f, text] of Object.entries(files)) {
    const p = join(here, f);
    const same = existsSync(p) && readFileSync(p, "utf8") === text;
    if (!same) stale++;
    console.log(`${same ? "ok   " : "STALE"} ${f}`);
  }
  const notes = [];
  for (const sf of builtSpecs) {
    const pf = provFile(sf);
    const provPath = join(here, pf);
    const old = existsSync(provPath) ? readFileSync(provPath, "utf8") : "";
    const rec = /commit `([0-9a-f]{40})` \(branch `([^`]+)`(, WORKING TREE DIRTY)?\)/.exec(old);
    const provOk = !!rec && old === provenance(sf, rec[1], rec[2], !!rec[3]);
    if (!provOk) stale++;
    console.log(`${provOk ? "ok   " : "STALE"} ${pf}`);
    if (rec && rec[1] !== head) notes.push(`note: ${pf} records zilion ${rec[1].slice(0, 12)}, the checkout is at ${head.slice(0, 12)}`);
  }
  const builtAt = statSync(join(zdir, "dist/index.js")).mtimeMs / 1000;
  const committedAt = Number(git("log", "-1", "--format=%ct"));
  if (builtAt < committedAt) console.log(`note: ${zdir}/dist is older than HEAD's commit; run npm run build there before trusting --check`);
  for (const n of notes) console.log(`${n} (outputs ${stale ? "differ" : "unchanged"})`);
  process.exit(stale ? 1 : 0);
} else {
  for (const [f, text] of Object.entries(files)) {
    writeFileSync(join(here, f), text);
    console.log(`wrote ${f}`);
  }
  for (const sf of builtSpecs) {
    writeFileSync(join(here, provFile(sf)), provenance(sf, head, branch, dirty));
    console.log(`wrote ${provFile(sf)} (zilion ${head.slice(0, 12)}${dirty ? ", dirty" : ""})`);
  }
}

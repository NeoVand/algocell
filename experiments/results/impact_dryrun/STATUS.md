# Emulator impact study: status (paused 2026-10-11, usage credit limited)

Branch `feat/impact` (worktree `/Users/neo/repos/algocell-wt-impact`). Draft protocol: `experiments/planning/IMPACT_STUDY_DRAFT.md`
(unchanged). Nothing registered; the registered run has not been run.

## Built and tested (commit 9a658f8)

- **Machines.** `algocell_exp/machines/worlds_impact.json` (generator `algocell_exp/impact_worlds.py`): 53 worlds (P = 2L at
  every registered L plus L = 19; Stage F's padded and odd rings; lethal tar at L = 16, 32; the six DZ dials; randreg and
  randsp at L = 16–64), each as `imp_<world>_c0` (compat 0.2), `_c1` (0.3 classic), `_c2` (0.3 zilog-nmos), one shared world
  spec. Built at zilion ed93b7c; `build_machines.mjs --check` passes. The z80c02 machines of worlds.json were regenerated
  too (header comment and manifest coreVersion only); `m6502skel_pair_L16` now builds zilion's full 6502 core (the census
  branch renames it: expect a merge conflict there and in PROVENANCE.md; resolve by regenerating).
- **C0 proofs** (`results/impact_equiv/EQUIV.md`, git 9a658f8): every C0 machine against Algocell's own executor for its
  world on archived pairs of its registered conditions (own budgets, masks, PCG seeds), plus random pairs with and without
  random masks: 53 of 53 worlds, 3.73 M pairs, 0 mismatches in memory, write counts and registers; 67 of 67 negative
  controls fail as they must. L = 19 uses the executor export-sim.ts would write (the derivation regenerates all 70+
  exported executors byte for byte); randreg/randsp at L = 20–64 use plain-condition archived pairs.
- **Tooling.** `algocell_exp/impact.py` (runner, affected-pair rate with Clopper–Pearson, attribution: repaired pass +
  probes + decisive event, 'other' stops the run; GPU path `FastAttributor`, CPU reference `Attributor`, which also handles
  the dial), `algocell_exp/impact_corpus.py` (682 conditions: 12 random rings, 598 draft-tier, 73 added-tier; seed streams;
  dry-run exclusion), `impact_study.py` (equiv, dryrun, bench, exclusion, report, guarded `run`; every GPU command runs in
  resumable chunks of about 60 s).
- **Tests** (`tests/test_impact.py`, all passed before the pause): machine invariants (same world SHA, hooks, world code
  identical outside the core text), executor derivation, classifier, attribution on constructed programs for every class
  (plain, padded, lethal, dial, randreg, randsp worlds), decisive event after a transient one, stop on 'other' and on a
  failed check, stepping reproduces single dispatches, fast path equals reference, Clopper–Pearson, corpora and exclusion.

## Remaining

1. **The dry run** (1,000 pairs per condition, all three cores, census of every pair). A first, complete run at the
   uncommitted state before 9a658f8 is not used (its outputs are not kept here); it took about 38 min of GPU work, so the
   chunked run needs about 40 chunks with equal pauses.
2. **The bench** (timing at 100,000 pairs, for the projection), the exclusion check, and the report (`REPORT.md`, with
   the protocol issues, written by `impact_study.py report`).
3. **The independent review** (C0 proofs; C1/C2 differ only in the core; attribution cannot misclass; dry run excluded),
   then fixes and a commit. A reviewer was started and stopped at the pause; no findings were received.
4. **Corpus 3 (assays)** is pending on assays running on Zilion machines (branch feat/assays); the hook is
   `impact_study.corpus3_assays` (raises NotImplementedError).

## Resume (from `/Users/neo/repos/algocell-wt-impact/experiments`; machine in use: foreground only, ~60 s chunks, equal pauses)

Each GPU command does one chunk of about 60 s and exits 3 while work remains; run it again after a pause as long as the
chunk took. A driver that does this in the foreground:

```sh
PY=/Users/neo/repos/algocell/experiments/.venv/bin/python
chunks() { while :; do t0=$(date +%s); "$@"; rc=$?; [ $rc -eq 3 ] || return $rc; sleep $(( $(date +%s) - t0 )); done; }
chunks $PY impact_study.py dryrun          # → results/impact_dryrun/dryrun_conditions.csv, dryrun_affected_pairs.csv, dryrun_meta.json
chunks $PY impact_study.py bench           # → results/impact_dryrun/bench.csv
$PY impact_study.py exclusion              # CPU only → results/impact_dryrun/exclusion.csv
$PY impact_study.py report                 # → results/impact_dryrun/REPORT.md
```

Progress is kept in `results/impact_dryrun/_dryrun_parts.jsonl` (and `_bench_parts.jsonl`); an interrupted chunk loses at
most the condition it was on. Tests (GPU; run in small groups, e.g. `pytest tests/test_impact.py -k classify`) need
`PYTHONPATH=.`. The registered run (`impact_study.py run --registered`) refuses until the study is registered in
`REVISION_PREREG.md` with every C1/C2 core SHA-256 and the dry run exists.

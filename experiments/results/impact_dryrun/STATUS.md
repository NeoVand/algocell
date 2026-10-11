# Emulator impact tooling and dry run: completed

Branch `feat/impact`; draft `experiments/planning/IMPACT_STUDY_DRAFT.md` unchanged. Nothing registered and no confirmatory comparisons executed. Assays remain the `corpus3_assays` hook.

Read [README.md](README.md) for the generated results and [PROTOCOL.md](PROTOCOL.md) for issues to settle before registration. All numerical results in README are derived from CSV tables.

- Registry: 787 conditions, 92 worlds, three cores per world. The exploratory ring sweep adds final/emergence archives; scheduled intermediate snapshots remain outside that sampling scope. R2 includes every final invasion-closer soup with its original suppression masks. `input_manifest.json` freezes sources, draws, budgets and masks.
- Attribution: both CPU and GPU repair preserve Q's flag-write relation when copying F. `test_q_repair_preserves_flag_write_relation` and `test_q_repair_does_not_invent_scf_ccf_events` catch the earlier false SCF/CCF attribution and retain genuine Q events. Regression evidence is in `test_summary.json`.
- C0 proof: 6,060,000 pairs, 0 mismatches, 120 negative controls caught. Original proof tables stay in `../impact_equiv/`; supplemental archived/random/masked checks are `equiv_groups.csv` and `equiv_controls.csv`. These are single-pair executor comparisons; the report states the soup-shader and stochastic-stream limitations.
- Cross-platform gate: 856,976 identical Metal/Vulkan bytes, including canonical attribution. `cross_local.json` and `cross_remote.json` carry component hashes.
- Detached L40S job: 34.13 minutes; app `ap-KLBng9CqyaldZWMusPVUOS`, call `fc-01M4MWM7CVDQQ05Y3A01KMX43Q`. Modal reports it stopped. Its per-proof, per-condition and per-benchmark checkpoints remain committed on `algocell-atlas-runs` at `impact_dryrun/c964d145c1b95cde`. Only sampled inputs were uploaded.
- Spend: $2.50 estimated, $1.212445 billed for the cross-check and worker apps; below the hard cap. `billing.json` contains only those apps.
- Exclusion: 0 overlaps across 787 conditions; regenerated registered draws were never executed. The transported and executed draw/pair digests match the archive.
- Verification: 473 passed, 2 provenance skips because Zilion HEAD is newer than the recorded core commit. Manual regeneration against rebuilt Zilion passed with unchanged core output. Impact and existing ZSoup tests ran in foreground bursts with equal pauses; no local GPU process remains. The benchmark timer correction and projection tests also pass.
- Telemetry correction: the executed worker's benchmark loop overwrote its total-wall start variable. `bench_raw.csv` preserves the original output. `bench.csv` drops the invalid wall column and retains valid execution/attribution phase timings; condition/job wall times were unaffected. The timer is fixed and has a CPU regression. `execution_source.json` verifies the exact executed source against Git.

## Reproduce tables (CPU only)

From this worktree's `experiments` directory:

```sh
/Users/neo/repos/algocell/experiments/.venv/bin/python impact_study.py exclusion
/Users/neo/repos/algocell/experiments/.venv/bin/python impact_study.py report
```

Exclusion needs ignored `inputs.npz`, retained locally and on the volume. If absent, fetch only the bundle:

```sh
XDG_RUNTIME_DIR=/tmp /Users/neo/repos/algocell/experiments/.venv/bin/modal volume get --force --profile neovand algocell-atlas-runs impact_dryrun/c964d145c1b95cde/inputs.npz results/impact_dryrun/inputs.npz
```

A fresh dry run uses `impact_modal.py prepare`, a brief foreground `cross-local`, `modal_impact.py::cross`, then `modal run --detach --profile neovand modal_impact.py::main`. Recompute its estimate before launch. The worker rejects stale code/input hashes, checkpoints each chunk and stops on any “other”. Do not rerun a completed bundle with changed worker code; prepare and cross-check fresh inputs first.

# Registered S1 scorer

`score6502.py` implements the S1 section of `experiments/REVISION_PREREG.md` without changing that registration.
Validation uses only archived Z80 DZ lp0 worlds and handmade inputs. Registered 6502 inputs remain uninspected.

The scorer preserves `dial_soups.analyse`'s chronological top-three walk, ≥ 0.005 share cutoff, tape cache, assay
draws and trace draws. The first qualifying tape is the largest-share tape, retaining recorded order for ties.
It stops after the sample with both endpoints measured. P2 checks every qualifying heritable tape before closure;
a tape at the closure sample does not qualify. Without a closure event, all samples through the horizon count.
Missing t_rep and t_closed are right-censored, stored as JSON null / empty CSV; KM medians use `dial_soups._km_median`.

Partner source and counts are exactly the frozen DZ implementation: uniform random byte partners, with judge RNG
`[seed, 0, 11]`, 64 culture partners and 16 traced encounters per cached tape. Final classifications call the frozen
`population_classes.classify` with RNG `[seed, 23]`: 64 random cells, 32 shared culture partners, 16 traces per cell,
up to 16 heritable cells scanned, 16 alternative bytes per site and 16 partners per mutant generation. This retains
the DZ final-class counts, including its 32-partner population culture test. It does not substitute contemporary soup
partners for these frozen uniform draws. Classes of unscanned confined cells retain the frozen proportional estimator.

For scanned heritable confined cells, d is the culture assay's modal cyclic copy offset; no additional assay or random
draw is introduced. P3's equality denominator includes scanned confined cells with measured d ≥ 6. If none are eligible,
the offset prediction is not tested. Cells lacking a copy offset have null d. The cell table records sites and equality.

PC sequences come from deterministic prefix-budget replays (0 through the world's instruction budget) in the executor
interface. Every replay starts with exactly the same organism, partner and initial state. Prefixes are batched using
per-lane budgets when available. The state record supplies `pc`; no opcode disassembly, instruction-start bitmap or
ring-position heuristic is used. A decrease in any of the 16 traces is a loop, except exactly 65535 → 0. A jump to its
own instruction boundary has no decrease and therefore does not meet this registered definition. Open code can acquire
a measured loop by entering random partner instructions; the handmade straight-line tests use NOP partners to isolate
the requested code's behavior. Actual scoring uses the frozen random partners.

All encounters use the world's vendored machine, including final cultures, traces and mutants. Arm C's 512-instruction
budget reaches every assay; D's BRK-halt request is checked against its machine. Machine core/world/WGSL hashes must
match the run's condition. The executor permits exactly the NMOS world's `illegal: as-chip` trap policy (ordinary JAM
behavior), retaining other halting-rule refusals. Input tape and final snapshot hashes are checked; incomplete worlds
are rejected. Full machine and scoring-source hashes accompany each per-world result.

Only a complete arm A (all twenty registered seeds, L=16, 128 instructions, faithful machine, final 10,000,000 snapshot)
receives P1–P4 and null/kill verdicts. An incomplete arm is descriptive. With fewer than five t_rep events, P1–P3 are
not tested and P4 is evaluated. The kill threshold is evaluated independently of the null rule, as registered. B–D
receive descriptive endpoints and no kill verdict.

Local invocation after release:

```sh
python score6502.py --runs /chosen/arm/A --machine m6502_pair_maps_L16 --out results/s6502/A
python score6502.py --gather /chosen/scored-jsons --out results/s6502
```

Modal invocation for the archived control only:

```sh
XDG_RUNTIME_DIR=/tmp /Users/neo/repos/algocell/experiments/.venv/bin/modal run modal_score6502.py \
  --archive /Users/neo/repos/algocell/experiments/runs/dial/dial --out results/s6502
```

Each call uses a single-use GPU container (L40S, A10G or L4) and writes JSON to the scratch prefix, then returns it to the local entrypoint
for generated tables and the control comparison. Calls are awaited; there are no detached jobs or automatic retries.
The saved JSON is read back from the volume and compared exactly with the returned result. `--gather-only` gathers
existing score JSONs without rescoring (`--prefix` must identify a scratch test when not using `--registered`).
When GPU allocation is unavailable, `--cpu-test` validates archived DZ scratch inputs using Mesa software Vulkan in
one CPU container per world, executing the same vendored WGSL through the same executor. Its only additional image
requirement is Debian `mesa-vulkan-drivers` (already installed in the cached image during validation); it cannot access registered worlds. Production
scoring remains on GPUs. No pip/npm packages are added to the cached DZ image.
The dependency image is the immutable cached `modal_dial.py` image `im-T2VaU4HVHQXvzDSkbRye1y`; source and vendored
machines are mounted from this worktree. Mesa is layered before source mounts, as Modal requires.
The software test supplies its device explicitly through the executor interface; the production GPU-only selection
guard remains intact. Three GPU-only validation attempts allocated zero containers and were stopped (only our apps);
the full Modal test therefore used software Vulkan. The physical Modal GPU execution path was not exercised here.
Test mode rejects the registered prefix, traversal, non-DZ arms and registered seeds. Production mode uses predetermined
filenames under `s6502/<arm>/`, writes `s6502_scored/<arm>/`, and requires all scoring code and machines to be committed.
Existing scored results are refused rather than overwritten. No volume listing is needed. Production invocation is
reserved for the coordinator after this scorer is committed; it was not invoked in this task.

Validation commands (temporary files remain inside this worktree):

```sh
PYTHONPATH=experiments /Users/neo/repos/algocell/experiments/.venv/bin/python -m pytest -q \
  experiments/tests/test_score6502.py --basetemp=experiments/.scorer_tmp
PYTHONPATH=experiments /Users/neo/repos/algocell/experiments/.venv/bin/python -m pytest -q \
  experiments/tests/test_assays_any_isa.py \
  -k 'frozen_functions_on_a_machine or refusals or pack_fetch_masks_roundtrip or m6502_world_state' \
  --basetemp=experiments/.scorer_regression_tmp
```

Generated evidence: `SCORER_CONTROL.md`, `MODAL_SCORER_TEST.md`, `s1_worlds.csv`, `s1_cells.csv`, `s1_p2.csv`,
`s1_verdicts.json`, `S1.md` and `worlds/lp0_s*.json`. Those S1 tables describe Z80 controls only, not registered S1 data.
The final validation passed 31 scorer tests and four focused executor regression tests. Both the local GPU control
and the Modal software-Vulkan control matched all ten lp0 worlds in every frozen column; their complete scientific
results (including cells, judged tapes and P2) and machine provenance were identical. Source hashes match the final
files. Local control bursts were below one second with equal pauses; timings and a second control table are preserved
in `local_control/`. Empty cell/P2 tables retain CSV headers. No registered 6502 output was accessed.

# S1 6502 Modal smoke and cost gate

Smoke app: `ap-6dyE1zpZSEbaRZr7NbHjha`; launched 2026-10-11T07:32:06.505434+00:00.
Runner commit: `948ac2b5cfef2b96840681036739181bbb8db37e`.
Four L40S containers, arms A–D, seed 999, 30,000 steps each. Volume: `algocell-atlas-runs/s6502_smoke/<arm>/`.
The pilot and S1 registration are unchanged. No disassembly, assays, or tape classification were performed.

All structural checks passed with `check_s6502_smoke.py`: 116 samples per arm at exactly the registered steps; three distinct exact tapes per sample, with finite ordered shares summing to at most one; soup snapshots at 20,000 and 30,000 steps, decompressed size 20,000 × L and matching stored checksum; arm parameters, L40S Vulkan adapter, worktree commit, runner SHA-256, and machine core/world/WGSL SHA-256s match the vendored manifests. Histograms, concentration fields, active pairs and completion summaries also passed.

| Arm | Instructions | Execution steps/s | Whole-smoke wall steps/s | Projected 10M world |
|---|---:|---:|---:|---:|
| A | 128 | 4,842.71 | 2,173.41 | 34.89 min |
| B | 128 | 4,623.17 | 1,785.31 | 36.83 min |
| C | 512 | 2,678.26 | 1,586.43 | 62.67 min |
| D | 128 | 4,987.60 | 2,156.59 | 33.86 min |

Projection: 10,000,000 / aggregate measured execution steps/s, plus 864 measured sample overheads, five checkpoint allowances (the slower measured snapshot), and startup once. Whole-smoke wall rates are reported separately because their dense initial schedule and fixed startup costs do not scale with the registered horizon.

Baseline: 56.09 L40S-hours, $109.37 for 80 worlds. Current `modal billing rates --profile neovand --json` confirms $1.95/L40S-hour. Allow one CPU core and one GiB per container ($3.10), then add a 25% wall-time margin: **$140.58 total**, below the $160 gate.

This is a sizing estimate from the unclassified smoke, not a guaranteed bill. Arm C projects to 62.67 minutes; the registered function timeout is four hours (over 3.8× that estimate). No retries; single-use containers; volume reload before each world; commit after each snapshot and at the end. Completed worlds are skipped, incomplete worlds retained and reported.

Smoke artifact download: 12 files, 2,943,899 bytes, retained locally under `smoke/` and on the volume. The source volume is authoritative; raw smoke artifacts and CLI logs are excluded from git. `smoke_check.json`, `modal_rates.json`, and `smoke_launch.json` retain check, price and launch metadata.

Verification: 11 mocked CPU-only tests, including hash-collision separation, complete registered sample/checkpoint schedules, overwrite refusal, Modal skip/incomplete handling, seed separation, and fixed-overhead cost projection. No local GPU runs.

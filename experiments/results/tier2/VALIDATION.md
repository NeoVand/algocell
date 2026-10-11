# I1-T2 preparation validation

No registered I1-T2 world or endpoint scoring was run. GPU work was on Modal; the CPU-only regression suite passed 6 tests. An accidentally broad local GPU test invocation was stopped after about 10 seconds; it is not evidence used here.

| Check | Coverage | Result |
|---|---|---|
| Original Soup executor vs ZSoup bridge, C0, identical drawn/active lanes | 27 worlds × 128 steps; plus three mutation-off arms × 128; 31,457,280 pair slots | zero memory/write-count mismatches; CPU comparator rechecks agree; active writes/changes observed |
| Mapped C0 machines vs original pair executor | 27 worlds × 65,536; plus 65,536 i8080-masked L32 pairs; 1,835,008 pairs | zero final-memory/write-count/register mismatches |
| Original traces vs mapped C0 machines | 27 worlds, budgets 0/1/128; 331,776 pairs; hook routing explicitly exercised | zero memory/fetch-map mismatches |
| Negative controls | 27 worlds × 8,192 pairs at budget 127 instead of 128 | all 27 detect a mismatch |
| Frozen default runner path | DZ, E, R4 runner output against 0970b82 using the same deterministic CPU test double | returned fields, journal bytes and snapshot bytes identical |
| Machine identity/world twins | 54 assay machines, 27 C0/C1 twins; complete manifest/spec and WGSL SHA checks | identical execution world, only core differs; maps observational |
| Frozen scoring numerics | scoring_sources.json AST checks | numerical report/KM/permutation/classification/serial/branching functions unchanged |
| C1 Modal smoke | one 2,000-step world per DZ, E, L20, R4 randreg, F L8/P32 | all five output-shape checks pass, 20,000 × L uint8; journal/snapshot schemas and 80 mapped-assay pairs checked |

The lockstep proof copies pairs after prepare_batch and compares every active/inactive slot, preserving lane indices and batch seeds through raw-hook PCG streams. It proves executor/bridge equivalence without asserting identical independently scheduled soup trajectories. Inactive lanes have budget zero. Negative controls and non-vacuity counters are in validate.json; mutation-off and 8080 mask coverage is in variants.json.

Smoke data live only at Modal volume `algocell-atlas-runs:/tier2_smoke/{DZ,E,L,R4,F}`. The per-world shape metadata are in smoke.json; no smoke endpoint is interpreted or used as a registered result. The original E/R4 endpoint overshoot is preserved.

Modal apps (completed):
- [five C1 smoke worlds](https://modal.com/apps/neovand/main/ap-mjFVnR10ieF6PjWFCnbiEM)
- [C0 lockstep and pair proofs](https://modal.com/apps/neovand/main/ap-p1eOi9geSVgyKNeQ8GLFjo)
- [C0 trace proofs](https://modal.com/apps/neovand/main/ap-mhSyJBSmvZeZlaCbxiYrG0)
- [mutation-off and 8080 mask proofs](https://modal.com/apps/neovand/main/ap-xKz40FpVY1O15eQMjVMvh8)
- [L40S fresh timing](https://modal.com/apps/neovand/main/ap-VKyf1NuEr2UYPJ7H2vqMjN)
- [L40S mature timing](https://modal.com/apps/neovand/main/ap-zGpfV0QtPQMfJxhvloGleA)
- [L4 fresh/mature timing](https://modal.com/apps/neovand/main/ap-BqJN7Or4nX58o0wyCYZQUZ)

Timing probes use seed 424242, excluded from every original world design; archived mature soups are read only and used only to time C1 execution. The preparation launcher bounds every GPU action and contains no registered-run dispatch path.

# I1-T2 running spend ledger

Updated UTC: 2026-10-11 13:53:46.

COST.md rates: L4 $0.000222/s + one CPU core $0.0000131/s + 8 GiB × $0.00000222/s = $0.00025286/container-second. No credits or automatic retries assumed.

| Experiment | Required worlds/jobs | Forecast USD |
|---|---:|---:|
| DZ | 60 | 1.8762 |
| E | 90 | 2.0588 |
| E-L32L | 20 | 0.5531 |
| L | 20 | 22.7468 |
| R4 | 32 | 0.2750 |
| F | 10 | 0.2835 |
| StageE-L10 | 10 | 0.5000 |
| Serial | 14 | 1.3333 |
| Branching | 7 | 0.6667 |
| Later soup scoring reserve | — | 2.0000 |
| Startup / compilation reserve | — | 1.0000 |
| Storage / egress reserve | — | 0.2000 |
| Preparation reserve (prior work) | — | 2.0000 |

Soup costs retain COST.md's slowest fresh-state rate, 20% throughput reduction and 10% runner overhead. Stage E L10 adds a conservative $0.50. Serial/branching execution uses $2 of the already registered $4 assay/scoring reserve; $2 remains reserved for subsequent soup scoring. Panels emit measurement tables only; claim reports are deferred.

Initial all-in forecast: **$35.4933**. Cap: **$40**.
Runtime estimate so far: **$0.0000**; required remaining: **$30.2933**; other reserves: **$5.20**; current all-in forecast: **$35.4933**.
Modal app-attributed metered spend: unavailable before launch; status script queries billing if supported.

Runtime estimates use per-job start/end markers and allocated resource rates, excluding queue time. Modal billing is authoritative when available and may lag; runtime estimates are not claimed as actual charges. Each snapshot refreshes a marker; the status command can refresh this ledger with --update-ledger. Dispatch checks the remaining-work projection before launch and between batches, and containers check it before starting. If it exceeds $40, pending dispatch stops without shortening any design; incomplete work is never scored.

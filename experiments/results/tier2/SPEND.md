# I1-T2 running spend ledger

Scoring closeout UTC: 2026-10-12T04:35:22.033817+00:00.

All registered world/panel work has finished. Stage L keeps its 20 failed end-check markers, but all 20 worlds ran 10M steps and have every frozen scoring input; all were scored. No new tier-2 world was launched. The dispatcher projection counting replacement Stage L runs is not a remaining-work budget for this scoring task.

| kind          | app_id                    |   metered_usd |
|:--------------|:--------------------------|--------------:|
| worlds/panels | ap-GpoWKGnLbLpyPF0kyUWKsT |   19.16065586 |
| scoring       | ap-bAUeWQx7pVbtC7D51FI205 |    0.39176397 |
| scoring       | ap-oRCTXTA8C8njp8zlxGpi9I |    0.00405097 |

New scoring actual metered spend: **$0.39581494**, including the failed Stage E plotting attempt and its table-only completion; scoring ceiling **$5**.
Tier-2 world/panel and scoring apps combined: **$19.55647080**. Remaining under $40: **$20.44352920**. Retaining the original $2 preparation reserve leaves **$18.44352920**. Preparation outside these apps is not claimed as metered here.

Function-body runtime estimates: worlds/panels $17.72714790; scoring $0.27012659. These estimates exclude startup and are not the bill.

Billing is authoritative app-attributed usage and can lag. The report requests the full current-hour interval because Modal excludes a partial end interval; `score_spend.json` records the query bounds. All eight successful scoring jobs completed, plus the preserved failed plotting attempt. The allocated-runtime ceilings were $3.337752 initially and $0.151716 for the single Stage E retry, before startup overhead.

## Historical launch forecast and early snapshot


Updated UTC: 2026-10-11 13:57:19.

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
Runtime estimate so far: **$0.6519**; required remaining: **$29.5453**; other reserves: **$5.20**; current all-in forecast: **$35.3972**.
Modal app-attributed metered spend: $0.0000 (metered; may lag).

Runtime estimates use per-job start/end markers and allocated resource rates, excluding queue time. Modal billing is authoritative when available and may lag; runtime estimates are not claimed as actual charges. Each snapshot refreshes a marker; the status command can refresh this ledger with --update-ledger. Dispatch checks the remaining-work projection before launch and between batches, and containers check it before starting. If it exceeds $40, pending dispatch stops without shortening any design; incomplete work is never scored.

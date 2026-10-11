# I1-T2 cost estimate (generated)

Rates: [Modal pricing](https://modal.com/pricing), fetched 2026-10-11: L4 $0.000222/s; L40S $0.000542/s; CPU $0.0000131/core/s; memory $0.00000222/GiB/s. No free credits assumed.

`bench.json`, `mature.json`, and `l4.json` record C1 timing on Modal. Project using the **slowest fresh-state repetition per world**, reduce throughput a further 20%, then add 10% for runner sampling/serialization. Mature Stage L states are faster and receive no discount in this budget. Allocate one physical CPU core plus 8 GiB for the full container lifespan; specify these bounds in the registered launcher.

| GPU | experiment | worlds | registered steps | budget USD |
|---|---|---|---|---|
| L40S | DZ | 60 | 18,000,000 | 3.19 |
| L40S | E | 110 | 27,000,000 | 4.75 |
| L40S | L | 20 | 200,000,000 | 42.17 |
| L40S | R4 | 32 | 2,400,000 | 0.54 |
| L40S | F | 10 | 3,000,000 | 0.52 |
| L4 | DZ | 60 | 18,000,000 | 1.88 |
| L4 | E | 110 | 27,000,000 | 2.61 |
| L4 | L | 20 | 200,000,000 | 22.75 |
| L4 | R4 | 32 | 2,400,000 | 0.27 |
| L4 | F | 10 | 3,000,000 | 0.28 |

**L4: worlds $27.79; complete preparation + tier-2 envelope $34.99.** Add $4 for all frozen scoring/serial/branching assays, $1 startup/compilation, $0.20 storage/egress, and $2 preparation smoke/proofs/benchmarks. These are reserves, not measured charges. The $4 scoring reserve covers about 4.4 L4 GPU-container hours at the allocated CPU/RAM price.

**L40S: worlds $51.17; complete preparation + tier-2 envelope $58.37.** Add $4 for all frozen scoring/serial/branching assays, $1 startup/compilation, $0.20 storage/egress, and $2 preparation smoke/proofs/benchmarks. These are reserves, not measured charges. The $4 scoring reserve covers about 4.4 L4 GPU-container hours at the allocated CPU/RAM price.

Choose **L4** for the complete 232-world design plus the full 14-genotype serial and five-genotype/two-arm branching controls. Nothing is dropped; all six DZ p levels, all original seeds/horizons, E's mutation-off tests and L32 environment controls, and both Stage L lengths remain intact. Long Stage L containers need at least a 2-hour timeout (4 hours recommended).

The L40S-only envelope exceeds the $40 cap; changing GPU to L4 preserves the design. Do not silently truncate seeds, p levels, or horizons to enforce the cap. Use a global spend ledger: stop dispatching if the forecast including the remaining required worlds exceeds $40; checkpoint unfinished work and request the coordinator's revised registration. A budget stop is an incomplete test, never a verdict.

Residual uncertainty: C1 may evolve states slower than the C0 archive performance inputs; pure dispatch timing omits some CPU scoring and runner I/O. The fresh-state floor, 20% rate reduction, runner uplift and explicit reserves address this without claiming a guaranteed bill. No auto-retries are budgeted; a retry must consume the same global reserve. Raw timing was on NVIDIA GPUs; no registered world was run.

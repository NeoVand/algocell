# Reset-source intervention correction — R2-label STOP, 2026-10-12

**STOP under the revised brief's R2-label gate.** Reset CX intervention repairs
`51 B9`: it remains literal at both lengths and budgets. However, the complete
L16/128 audit changes **19 open memory-fed encounter labels to other**. In
particular, `AD 50` changes from **12/16 to 2/16** encounters jointly open and
memory-fed, crossing the registered ≥9/16 tape criterion. This is a qualifying
stop independent of descriptive literal/other counts. No R2 definition,
threshold, prediction, biological result or population outcome is rewritten.
R0 and R1 cannot change from this classifier correction; raw execution and fetch
openness are identical.

The previous reset-source STOP rationale is superseded. The remaining three
census settings, random-program differential, N2/growing-ledger/restart-time
repairs, randomized system simulation, $85/$95 forecast comparison, $95 cap
freeze and refreeze are **not completed because this stop gate fired**. The
pending $95 approval is not recorded as granted. There is no release, launch,
paid compute, provider call, population run or push.

## Verified reset and read/fetch interventions

`prefix_causality8086.py` derives its observer/intervention machine from the
unchanged vendored WGSL. An extra three-word per-lane input independently
selects either one exact fetch/data-read event or one reset register. Reset
alternatives XOR the register's full word with `5555` and `AAAA`, with every
other register and initial memory at frozen reset. The predicted byte follows
its low/high word layout and any modeled literal increment/decrement. Exact
read/fetch interventions XOR only that returned byte with `55` and `AA`;
initial executable memory is unchanged. An unsupported source remains other.

Each effective partner write, including previously omitted no-change RMW,
requires both exact-value checks and the same ordered instruction starts and
fetch/read/write addresses through that write. A verified neutral RMW creates
no new channel. Changes after the claimed write are allowed. Numeric equality
alone never verifies provenance. Interrupt/vector writes reject the affected
episode before MOVS source claims can be emitted.

## Independent decisive counterexample

At L16/128 with zero partner, `AD 50`'s exact source is event **74**:
`[step=14, data read, A15, value=50]`. The later claimed partner write is event
**92**: `[step=17, write, B16, value=50]`. A prior PUSH writes this same high byte
to position18, which is fetched as a displacement before event92.

| Substitution | Value at claimed write | Earlier event86 data-read position | Prefix through event92 |
|---|---|---|---|
| `50 XOR 55 = 05` | `05`, exactly predicted | 16 → 5 | Changed; fails |
| `50 XOR AA = FA` | `FA`, exactly predicted | 16 → 26 | Changed; fails |

The intervening RMW write at event87 moves with that read. Both alternatives
fail **before** event92; this is not rejection of permissible later divergence.
The second hook has a constant exact event target and no extra intervention
input binding. It independently reproduces both results, with complete actual
baseline/intervened traces in `reset_stop_focus.json.gz` and its generated
WGSL/manifest in `reset_gate_independent.*`.

## Complete correction for the audited setting

All **4,998 L16/128 archived encounters** were replayed against the uninstrumented
machine and checked against every original lane-record word, memory byte and
archived trace/hash. There are **0 execution/trace mismatches**. There are
**76,226 alternative-value checks**, with **5,685 failed checks**; unsuccessful
claims fail closed. Every baseline label equals the existing production
classifier before the diagnostic correction.

The **231 changed encounter labels** are all listed individually, including
length, budget, word, partner, old/new label, openness, causal source and first
failing write identity, in
`experiments/results/census8086/reset_correction_labels.csv`. This is a dated
**diagnostic correction**, not a replacement production census. Aggregate
before/after counts are in `reset_correction_counts.csv`; identities, transitions
and scope are in `reset_correction_provenance.json`. The separate proposed full
L16/128 table is `reset_mechanisms_L16_b128.csv` in prelaunch. All non-classifier
columns match the production table.

| L16/128 scope | Literal, before → after | Memory one-pass, before → after | Confined loop, before → after | Other, before → after |
|---|---:|---:|---:|---:|
| All 4,998 encounters | 488 → 276 | 339 → 320 | 6 → 6 | 4,165 → 4,396 |
| Nonzero period-2 words exact on zero partner | 12 → 3 | 2 → 2 | 0 → 0 | 6 → 15 |

The descriptive transitions are **212 literal → other**. These do not cause the
STOP. The qualifying transitions are **19 open memory-fed → other**:

| Word | Archived partner IDs that change |
|---|---|
| `36 FF` | 10 |
| `8B 52` | 0, 2, 5 |
| `AB 58` | 6 |
| `AD 50` | 0, 3, 5, 6, 7, 9, 11, 12, 13, 15, 16 |
| `AD AB` | 6 |
| `C5 52` | 0, 5 |

Partner0 is the zero reference; partners1–16 are the frozen random panel. The
`AD 50` tape's joint-open/memory numerator moves from12 to2 among the16 random
encounters. The remaining **16,184 census encounters** are not audited under
this correction. No result from the old 8,192-program differential is claimed
as validation of the corrected contract.

## Required named witnesses

- **`51 B9`: literal at L16/L32, budgets128/512, zero partner.** Every claimed
  write passes both alternatives, including the first reset-CX push. That first
  partner byte is `55`/`AA` with identical prefix; the high byte follows word
  layout. A disabled intervention exactly matches the ordinary observer's full
  record and memory in all four settings.
- **`B9 51`: other at all four zero-partner settings.** It fails the all-writes
  causal gate despite its baseline literal tags; resetting CX does not resolve
  failures of later immediate/read-source claims.
- **`07 A5`: unchanged.** All17 L16/128 encounters pass, including all1,088
  partner writes and2,176 alternative checks. The prior isolated-event
  prefix audit verifies all68 archived encounters/10,880 writes; the unchanged
  source-event implementation and prior evidence are retained. The other51
  encounters were not newly re-audited in the stopped full-census run.
- **Far-JMP witness `8e0e0e00ea0a000200900ef490900200`: other.** Its immediate-CS
  alternatives change a full instruction start before the claimed PUSH-CS
  write. Its old literal label is a descriptive correction.
- **N1: other.** Unsupported vector/stack traffic is rejected before accepting
  zero-count REP's apparent MOVS channel. The baseline agrees with raw execution.

## Validation and frozen disposition

**22 CPU regressions pass**, including reset low/high layout, exact immediate
fetch identity, literal arithmetic carry, unchanged-RMW treatment, N1,
unsupported claims, permissible later divergence, earlier bus/start changes,
and both archived independent `AD 50` failures. Python source parsing, correction
counts, unchanged non-classifier columns, generated shader/manifest hashes and
archive integrity pass. The focused diagnosis records **73 GPU bursts**, maximum
**1.035s**, each with an equal pause. Every census GPU call passed the ≤60s
burst guard and equal pause; its early gate return did not archive individual
census timing rows. This timing-evidence limitation is explicit.

Reproduce the diagnostic audit with:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments \
  /Users/neo/repos/algocell/experiments/.venv/bin/python \
  experiments/audit_reset8086.py
```

`diagnose_reset_stop8086.py` reproduces the focused stop diagnosis. This is
fixed-input diagnostic work only. The raw word/trace tables, contracts,
registration, production classifiers, spend code and approval state are
unchanged. All **270 existing freeze file identities** match. Existing freeze
SHA-256:
`353d57feec44548be1552704872aacf924b84249c9fe0aa99872e6ddee39c1c0`.
The existing unreleased **$85 freeze is preserved**, with no replacement freeze
or launch receipt. Exported shaders preserve inherited trailing-space bytes;
these exact source-archive exceptions are recorded in validation.

**STOP: R2-label gate changed. Coordinator report, no launch.**

# Mandatory provenance STOP — 2026-10-12

**STOP / NO-GO.** The requested causal contract changes `07 A5` and the
registered exact-zero-partner provenance counts. The instruction “If they
changed, stop and report; do not alter R2” is therefore active. No class repair,
complete census re-audit/random differential, spend-guard repair, randomized
system simulation, cap amendment, replacement freeze or launch was performed.
The existing production classifier still contains Astra's reported defects.
The existing $85 freeze remains unreleased and is not superseded by this report.

`ASTRA_RECHECK2.md` was read completely and committed first, unchanged, in
`3ff9e40`. Baseline reviewed code is `a078045`.

## Independent intervention evidence

The local harness `experiments/intervention_stop8086.py` executes the actual
vendored L16/L32 WGSL, without a CPU emulator or a second provenance-tag bank.
It restores the same frozen parent/partner for every encounter and substitutes
the copied source byte using both fixed alternatives, `s XOR 0x55` and
`s XOR 0xAA`. It checks the exact corresponding write and **every instruction
start through the encounter's full 128/512-step budget**.

Two intervention locations are tested independently:

1. Replace the source byte in the initial memory, then rerun the unmodified
   machine with the existing observation hooks.
2. Replace only the value returned by the specific source data read. The
   original executable bytes and all earlier reads/fetches remain intact.
   This rules out an earlier opcode-fetch change as the explanation for the
   rejected provenance claim. This second test is an additional necessary
   causal check, not a substitute for the initial-memory test.

For each of four length/budget settings, all 17 baseline trace hashes match the
archived census. Baseline observer memory and every original record word match
the uninstrumented frozen machine. With substitution disabled, the intervention
observer's entire memory and complete diagnostic record match the ordinary
observer. No vendored machine file or production observer was modified.

## `07 A5` changes in 68/68 encounters

The current classifier labels every encounter memory-fed one-pass copier.
Each encounter has at least one claimed copy that fails the required two-value
test. Thus **all 68 labels must become other under the requested contract**.
Both intervention locations independently reject all 68 encounters.

For zero partners, the first claimed partner write copies source byte `07`:

| Length | Source data read | Partner write | `07 XOR 55 = 52` | First differing full start |
|---:|---|---|---|---|
| 16 | step 1, A0 | step 1, B16 | writes `52` exactly; control changes | start index 33: original 33, intervened 36 |
| 32 | step 17, A16 | step 17, B32 | writes `52` exactly; control changes | start index 49: original 49, intervened 52 |

The same divergence occurs at budgets 128 and 512. For these zero partners,
`07 XOR AA = AD` writes `AD` exactly and preserves the full starts. The contract
requires **both** alternatives to pass, so the `55` failure rejects the claim.
The written byte really is copied; the requested full-control-flow condition
also constrains its later execution after the copied byte reaches B.

The registered nonzero-word, k=2, exact-zero-partner counts consequently cannot
remain unchanged. Applying **only** the proven `07 A5` rejection gives:

| Length | Budget | Memory-fed before | Memory-fed after this rejection | Other before | Other after this rejection | Literal before/after |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 128 | 2 | 1 | 6 | 7 | 12 |
| 16 | 512 | 1 | 0 | 16 | 17 | 12 |
| 32 | 128 | 5 | 4 | 10 | 11 | 11 |
| 32 | 512 | 4 | 3 | 11 | 12 | 11 |

These are not complete new census totals: other old claims may also fail.
Intervention verification only rejects old claims; it cannot restore this
rejected word's label by finding a different successful encounter. The L16/512
memory-fed exact-zero count is necessarily zero under this conservative
verification, because `07 A5` was its only member. Raw word tables, archived
traces, production label CSVs, registration and R0–R2 are left unchanged.

## Astra N1 independently fails exact-byte causality

The actual frozen machine reproduces parent
`eb06f49002f10000bc10009df3a5f400`, partner
`02010000000000000000000000000000`, and full starts
`[0,8,11,12,61698]`. The current classifier returns memory-fed one-pass copier.

At step 3, replacing only vector data read A4 (`02`) with `57` or `A8` leaves
the pushed B16 flags byte at `02`, rather than the substituted value. Both
interventions also change the vector destination/start sequence. This is
independent execution evidence for **other**, without reproducing the faulty
MOVS/vector ordering inference.

## Artifacts and validation

Reproduce from repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments \
  /Users/neo/repos/algocell/experiments/.venv/bin/python \
  experiments/intervention_stop8086.py
```

`causal_stop_summary.json` records counts, original and derived shader identities,
archive SHA-256 and the stopped disposition. `causal_stop_validation.json.gz`
contains baseline full starts, both alternatives' actual write values, full
intervened traces/start sequences, trace hashes, and GPU timings. The three
`causal_stop_*.wgsl` / `.manifest.json` pairs preserve the exact isolated-read
intervention implementations, derived from the frozen observer.

The successful harness run used 41 local GPU bursts; maximum burst was 1.033s,
with an equal pause after each. All replay/equality/rejection assertions passed.
No Modal/provider calls, paid compute, hardware benchmark, population run,
launch receipt or push occurred. The full existing test suite and requested
randomized dispatcher simulation were not run after the mandatory stop; no
claim of completed class repairs or replacement validation is made.

Evidence compilation, archive/hash checks, derived shader/manifest hashes and
all 270 existing freeze file identities passed. Staged whitespace checks pass
for the harness, report and JSON. The three preserved derived WGSL files each
carry three whitespace-only lines inherited from the frozen observer; these
nine exact-source archival exceptions are retained and disclosed.

The coordinator must resolve the changed scientific provenance basis before
authorizing further work. The requested $95 cap has not been frozen and no
user approval is asserted. **STOP; R2 unchanged.**

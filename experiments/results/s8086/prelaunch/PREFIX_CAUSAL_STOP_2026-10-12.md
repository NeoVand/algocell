# Corrected prefix causality — mandatory count-change STOP

The full-encounter start-sequence requirement in the previous brief was wrong.
Its rejection of `07 A5` is superseded. The corrected isolated-read, prefix-through-
write test **passes `07 A5` in all 68 archived encounters**, including every one
of its **10,880 claimed partner writes**, with **21,760 successful alternative-
value checks**. Changing an executed copy later is permitted.

Nevertheless, the corrected brief's fail-safe changes the registered zero-
partner provenance basis. `51 B9` starts with **PUSH CX from reset CX**, before
any source data read or immediate-operand fetch exists. That partner write cannot
have either of the two admitted causal sources. The required episode label is
therefore **other**. This exact-zero word was registered as a literal writer at
all four length/budget settings. The instruction **“If they change under this
corrected contract, stop and report. Do not alter R2”** applies.

**STOP.** No production class repair, spend repair, randomized system simulation,
$95 cap amendment, replacement freeze, release, launch or push was performed.
R0–R2, registration, production classifiers, raw census tables, label CSVs and
the existing unreleased $85 freeze remain unchanged. No paid compute or provider
calls occurred. This report does not claim completion of the remaining brief.

## Intervention and verification

`prefix_causality8086.py` derives a diagnostic execution machine from the actual
vendored L16/L32 WGSL. A separate per-lane input identifies the exact event whose
read/fetch return value is XORed with `55` or `AA`; initial executable memory is
unchanged. Each source is identified by its ordered event, not an equal-valued
byte elsewhere. Each alternative checks the write's exact substituted value,
instruction starts and ordered fetch/read/write addresses through and including
that write. Later structural divergence and intervened trace hashes are recorded.
The observer records ring memory addresses and the actual instruction-start
hook argument, as in the frozen census.

Each of the four `07 A5` baseline panels has the archived trace hashes. Baseline
observer memory and every original lane-record word equal the unmodified machine.
With intervention disabled, complete observer memory and diagnostic records
equal the ordinary observer. All 68 episodes retain their memory-fed one-pass
label; neither the `55` nor `AA` check fails for any claimed partner write.
Initial-memory substitution remains secondary: the new conclusion comes from
isolated source returns, and does not require the old initial-memory tests to
preserve earlier opcode execution.

## The remaining, independent contract change

For `51 B9`, SP=1 and CX=0 at reset. Step 0 fetches only opcode `51`, starts and
decodes PUSH CX, then writes the low reset byte to partner P−1 and the high byte
to parent 0. The first source data read has not occurred; the first immediate
operand fetch belongs to the **following** B9 instruction. A later operand cannot
causally supply the earlier write. The reset register is not either admitted
intervention point. Full frozen-machine replay and archived trace hashes agree
at both lengths and budgets.

| L | Budget | First partner write | Baseline literal | After only this rejection | Baseline other | After only this rejection |
|---:|---:|---|---:|---:|---:|---:|
| 16 | 128 | step 0, B31 = 00 | 12 | 11 | 6 | 7 |
| 16 | 512 | step 0, B31 = 00 | 12 | 11 | 16 | 17 |
| 32 | 128 | step 0, B63 = 00 | 11 | 10 | 10 | 11 |
| 32 | 512 | step 0, B63 = 00 | 11 | 10 | 11 | 12 |

These are necessary corrections from one proved rejection, **not complete new
census totals**. Memory-fed counts are unchanged by this rejection. The exact
word totals and raw execution results are unchanged. No tally or prediction was
rewritten to preserve the rationale.

The first setting's conservative re-audit covered **4,998 archived encounters**,
with all baseline hashes and labels checked. Its all-writes fail-safe produced
478 label rejections; the 20 nonzero exact-zero words changed from
literal/memory/other = **12/2/6** to **0/1/19**. Reset-derived stores and previously
omitted, unverified RMW stores are unsupported. These are the diagnostic's
conservative classifications, not a claim to have completed all instruction-
class proof procedures. The decisive reset-source counterexample above does not
depend on that broader extractor. The audit stopped at this first count-change
gate; the remaining 16,184 encounters and random-program differential were **not
completed**. The old 8,192-program differential is not evidence for this changed
contract.

## N1 and earlier witnesses

- **Astra N1:** required label **other**. Six vector/stack stores reject the step
  before accepting any MOVS channels, including zero-count REP. Rechecking the
  archived isolated vector-read interventions under prefix causality preserves
  the prefix through B16 but writes `02`, whereas the alternatives are `57` and
  `A8`. Both exact-byte tests fail. Fresh unmodified-machine/observer replay
  reproduces the baseline. The unrepaired production classifier still emits its
  old label; no claim of a released production fix is made.
- **Earlier immediate far-JMP witness**
  `8e0e0e00ea0a000200900ef490900200`: raw execution is unchanged, but its old
  literal classification also fails this contract. Changing the immediate CS
  operand changes an instruction start **before** the claimed PUSH CS write.
  Both alternatives therefore fail prefix equality. The new diagnostic label is
  **other**, independently of the N1 vector defect.
- **B9 51 and AB 58 zero references:** the first setting's audit rejects these
  under the all-writes fail-safe too. For example, B9 51's first unverified store
  is step 12, B25; AB 58's is step 1, B26. An unchanged numerical RMW store was
  previously omitted, rather than causally verified. This does not invalidate
  `07 A5`'s directly verified copy channel.

## Artifacts and checks

Reproduce the focused evidence from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments \
  /Users/neo/repos/algocell/experiments/.venv/bin/python \
  experiments/audit_prefix8086.py
```

The optional `--census-first-setting` audits L16/128 and reports its safety gate;
it does not claim to complete all four settings. `prefix_focus_validation.json.gz`
contains every source/write event identity and both alternatives' results for
all 68 encounters, reset-source traces, N1 and far-JMP evidence. Its summary
records scope, hashes and STOP disposition. `prefix_causal_validation.json.gz`
preserves the completed first-setting audit separately. The two exact derived
WGSL/manifest pairs preserve the isolated-event implementation.

The focused run used **378 GPU bursts**, maximum **0.591s**; the first-setting
audit used **729 bursts**, maximum **1.041s**. Every burst had an equal pause.
No burst exceeded 60 seconds. **11 focused CPU regressions pass**, covering both
alternatives, allowed later divergence, earlier start/read/write/fetch changes,
wrong byte values, reset-source rejection, immediate fetch identity, N1 and
unverified RMW stores. Python parsing, archive hashes and both shader/manifest
hashes pass. The six trailing-space lines in the two exported WGSL files are
inherited verbatim from the frozen observer; they are source-archive exceptions.

All **270 existing freeze file identities** remain committed and unchanged.
Existing freeze SHA-256:
`353d57feec44548be1552704872aacf924b84249c9fe0aa99872e6ddee39c1c0`.
This report supersedes the previous full-sequence rejection rationale, **not**
the freeze. The $95 approval is still pending, and no new freeze is asserted.

**STOP; R2 unchanged.**

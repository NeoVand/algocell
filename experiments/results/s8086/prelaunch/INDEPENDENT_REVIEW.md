# S2 independent implementation review — 2026-10-12

Reviewer: a separate agent pass, assigned only this report. No implementation
files were edited by the reviewer. No GPU, Modal, soup stepping or population
observations were performed by the reviewer. The approved draft was read in
full, together with the additional L2 specification.

Status: **independent code review completed; no launch approval**. Findings
below were sent to the implementing agent before the freeze. Implementation
findings have been fixed and inspected. The approved L2 design defect in
finding 1 remains and requires the coordinator's independent design review.

## Blocking findings

1. **The approved L2 prediction is false for a permitted nonzero source start.**
   At L=32, d=16, use the 16-byte periodic seed beginning
   `BE 0C 00 BF 1C 00 49 F3 A4`, followed by seven arbitrary bytes. This sets
   SI=12, DI=28, CX=FFFF and runs forward REP MOVSB. Its kernel reads s+k and
   writes s+d+k with s=12, d=16. The baseline is periodic, the kernel covers B,
   there are no original-partner reads, and the baseline offspring resets into
   the same family. Its instruction footprint is C={0,...,8}. Its causal seed
   is original coordinates 12...27, however: original primary coordinates
   9...11 are erased, and originals 25...27 are retained. The approved claim
   that free coordinates are primary 0...d-1 minus C, with all originals d...L-1
   erased, does not follow. A CPU-generated ordered trace reproduces the
   extractor's acceptance and source map [16,...,27,12,...,15] repeated twice;
   the baseline output equals the parent, while the assay's canonical source
   check rejects it. The implementation must not silently change the approved
   theorem, rotate the seed, exclude this candidate based on retention, or call
   that baseline rejection a scientific discovery. A source-origin hypothesis
   or a shifted-seed law requires an explicit design resolution. This is an
   approved-design defect, independently of an implementation defect.

2. **Partial-register writes discarded preserved-byte dependencies.** The
   initial dependency transfer replaced all word-register taint for a byte MOV.
   Example at L=d=32: `BF 20 00 8B 0E 1E 00 88 C1 F3 A4`, zero fill, byte31=1.
   `MOV CX,[30]; MOV CL,AL` preserves payload-derived CH. REP's baseline count
   is 256, while changing byte31 to zero makes it zero. The initial extractor
   omitted that dependence. The implementing agent has changed to separate
   low/high-byte dependencies; a reviewer CPU synthetic trace now includes
   coordinate31 in C. Final regression verification remains required.

3. **Modulo-reducing control origins concealed erased-coordinate influence.**
   Initial C construction reduced fetched/data-control roots modulo d. A fetch
   or condition/address/count read rooted in original position >=d can then be
   counted as fixed primary core even though an erased-position intervention
   changes it before overwrite. The implementing agent now rejects these
   original erased-coordinate code/control dependencies. This guard is
   necessary for the primary-seed theorem; it does not repair finding 1.

4. **World settlements exceeded reservations and hard timeouts exceeded
   liabilities.** Initial projection reserved deadline*RATE while the remote
   world returned (deadline+120)*RATE. The original remote hard timeout was
   12 hours despite smaller planned allocations; cooperative checks cannot
   interrupt a hung compilation, GPU call or storage call. The dispatcher now
   uses per-job Modal `with_options(timeout=...)`. Verify reservations include
   the same termination/startup allowance and scoring/control allocations use
   that binding too.

5. **The original dispatcher did not dispatch scoring, extraction, laws or
   required controls through the spend guard.** The implementing agent is
   adding these stages. The later eligibility commit must be attested by a
   host-side comparison with the exact committed local eligibility files,
   rather than accepting a nonempty commit string at the remote function.
   Jobs must remain exclusively reserved and failed attempts retained, with
   automatic retries disabled.

6. **Constructed controls did not preserve all original intervention streams.**
   Initially `assay_case(control=...)` replaced serial partner streams only;
   joint assignments still came from new S2 streams. Required controls need
   original census single/joint assignments and original serial streams, and
   all ten L/d/operation combinations at budget128 must complete. Certificate
   checks alone are not that full positive-control assay.

7. **Fixed timing omitted actual runner submission overhead.** The original
   benchmark submitted 256 steps with one synchronization and reused one pair
   of parameter buffers; the runner uses at most50 steps, updates parameters
   per step in a ring, and synchronously reads the active-pair tally each
   submission. The benchmark must account for those actual costs while
   restoring soup, pairs and masks before every dispatch.

8. **Frozen projection and remote archive paths require binding.** The CLI's
   projection argument was independent of the immutable freeze manifest.
   Alternative deadlines/liabilities must be rejected. Remote images initially
   included only runtime Python files, algocell_exp and prelaunch artifacts,
   while wrappers read every path in freeze.files. Any registration, draft,
   tests or census contracts included in that manifest must be packaged too,
   or a separately frozen runtime identity list must be used consistently.

## Protocol observations

The chronological scorer scans every scheduled sample and keeps R0/R1,
occurrence, persistence and the route kill separate. Its culture threshold,
share gate, genotype tie rule and 11-sample persistence interval match the
approved draft. Unassessable qualifying top tapes mark the world unscoreable.
Random-cell multiplicities remain in the 64-cell denominator, while exact
genotypes are deduplicated for execution. Mechanisms come from the frozen
ordered full-address census classifier; numeric-PC-decrease heuristics are not
used. Full uninstrumented and diagnostic lane records are compared.

L1 uses the actual evolved baseline tape, checks the six-byte core and declared
displacement, and requires the constructed ordered-copy geometry. Its source
start is zero. The L2 dependency verifier deliberately leaves unsupported
opcodes unresolved; that admitted subset and any unresolved evolved
certificates must be disclosed rather than asserted to cover arbitrary 8086
loops.

The assay stages preserve mismatches and stop the affected case for saved-input
replays. A static source map is used for successful rows; its validity depends
on the independent noninterference certificate. The nonzero-source flaw shows
why an arbitrary constant displacement cannot substitute for proving the
registered source map.

## Final closure

Findings 2 through 8 are closed by final source inspection:

- Byte-granular taint preserves CH through a CL write, and INC/DEC conservatively
  propagate carries across word dependencies. Retained REP decode is preserved
  through continuations rather than lost on the first empty fetch group.
- Original erased-coordinate code/control dependencies are rejected before any
  retention assay. The primary core footprint is no longer formed by reducing
  those original roots modulo d.
- World and scoring/control reservations include120 seconds beyond the
  enforced function allocation. The dispatcher binds Modal hard timeouts to
  the individual planned allocation; cooperative ticks also cover extraction.
  Reservations reject nonfinite and negative amounts.
- All world, score, extraction, law and constructed-control stages dispatch
  through the append-only spend guard. Exact planned liabilities remain
  reserved for unstarted work. Eligibility data must match the released
  current Git commit and exact local committed files before assay dispatch.
- The ten constructed controls use census single/joint assignments and census
  serial partner streams. The complete control runner uses budget128 only.
- Benchmark submissions now reproduce the runner's50-step cap, parameter
  ring writes and synchronous active-pair tally reads, while restoring fixed
  soup, pairs and active masks before every individual dispatch. The final
  benchmark measures homogeneous named census witnesses, every constructed
  REP geometry/operation, a constructed non-REP loop, and the frozen mixture;
  the forecast uses the slowest repetition across these panels. Complete
  largest-d REP and non-REP law workloads are timed separately, with assay
  logging/flush costs, and the forecast uses the slower measured assay rate.
  Every panel's pair memory and complete lane records must match the direct
  machine executor before timing. The inherited soup readback's `(n,2,L)`
  shape is normalized to `(n,2L)` for that comparison. Remote NumPy/RNG fixtures
  must match the frozen fixtures before the benchmark starts, and actual
  Python/platform/NumPy/wgpu/Brotli versions are recorded in its result.
- The dispatcher rejects projections different from the frozen allocation
  plan. The remote image conditionally packages every additional path from
  the full freeze manifest; world, scorer and control wrappers verify those
  content identities.

A follow-up finding in the new law report was also fixed: an arm verdict now
requires20 complete scoreable records with the registered seed set. A partial
arm cannot report support. The L2-without-L1 cap-exclusion denominator removes
the excluded L1 cases consistently. Unresolved extraction certificates suspend
support rather than disappearing from the arm verdict.

The last closure pass also checked two assay fixes. Machine identity is compared
after JSON normalization, so a committed certificate's segment lists match
the runtime description's tuples without weakening its hashes/reset checks.
Mismatch diagnosis uses `baseline=False` only when replaying the deliberately
primary-only intervention: the baseline extraction still requires the actual
sampled parent to be d-periodic. A nonperiodic primary-only mutant is therefore
not rejected merely because the intervention followed the registered protocol.
Its copy geometry, execution identity, ancestry and expected output still have
to be diagnosed. Local GPU calls now record burst/pause durations and use a
finally block to preserve the equal pause after a raised exception.

The reviewer independently ran the initial CPU-only S2 protocol subset:
**12 passed, 8 deselected, 0.22 seconds**, then the expanded final CPU subset:
**19 passed, 8 deselected, 0.34 seconds**. These checks covered RNG fixtures, sample
cadence and ties, full-horizon/persistence boundaries, route cutoffs, candidate
selection/multiplicity, spend-liability/hash-chain checks, frozen fixed inputs,
known payload/core/source mismatches, overflow handling, snapshot identity,
and the archived ten-world DZ endpoint control. The expanded subset additionally
checked incomplete-arm law verdicts, all three injected pipeline mismatch
positions, nonfinite spend rejection without ledger mutation, and complete/
incomplete synthetic record validation. The implementation owner reported an
earlier full-suite result of **25 passed**, followed by the final combined
`tests/test_s8086.py` plus `tests/test_run6502.py` result: **38 passed in
34.46 seconds**, with one existing Modal-local mock warning. That full suite
completed all ten constructed law controls, totaling **1,185,792 encounters**;
exact census replay for both phases and budgets; the byte-count dependency and
nonzero-source counterexamples; and fixed pair-bridge/active-tally memory and
complete-record identity checks. The WGSL observer's reserved identifier
`active` was changed to `active_mask` before the successful validation. These
owner GPU results and the approved Modal benchmark are separate freeze
artifacts, not reviewer compute. No further implementation blocker was found
in the final separate pass.

Finding 1 is **not closed**. The implementation intentionally preserves the
approved L2 eligibility and prediction literally and adds a fixed-input
nonzero-source counterexample regression. It must be disclosed as a known
prelaunch design defect, never as a newly observed evolved-law failure. A
source-start restriction or shifted causal-seed prediction would be a design
amendment. This code review does not authorize population dispatch, and the
coordinator should resolve that scientific defect before approving any launch.
L1 is unaffected by that counterexample because its source start is zero.

Reviewed source identities (SHA-256, before final registration/benchmark
artifact assembly):

| Source | SHA-256 |
|---|---|
| s8086.py | `145b18c2b7a3634f8cc5cb2f1b8b5b7273e3d0041213cf81390196d036a3ae65` |
| run8086.py | `ff08fe041a2592c621c1f5fd49a4d185477677a16f9c8ee5ee429c78f991e585` |
| score8086.py | `be5fd460203563f8fe1e78ecc5c834b6f9e7536d4561765abaae1dc9991ea8d8` |
| law8086.py | `a9596088abf91b2879b8928755b84231e54cc9e99790c5020cfd9e6da0b6ea2e` |
| dispatch8086.py | `9895472337f6f2c28869ccb0265d8d22158224b465178a3487dbdbb577bb817a` |
| benchmark8086.py | `47f07b76a684496011334bf49f8d0ac9ec1c7676b477a3a0444439316839ed7f` |
| modal_s8086.py | `fa9244f0d653e8b25a486abe1947c51dca4268899c1e465afa9a636ff8f739f9` |
| modal_score8086.py | `b5b632e1dbcd7dd3eab747d583689d7e76032d8b010a87f9885ca27437b2af99` |

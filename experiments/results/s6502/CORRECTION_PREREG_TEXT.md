### Correction to the outcome of S1 (2026-10-11)

The independent audit (results/s6502/AUDIT.md) found a multi-byte sequential-PC-wrap defect, two conflicts between the registration’s prose and its cited frozen definitions, and the shared packed-word mutation race. The original outcome therefore should not be described as implemented exactly as registered. Re-scoring all 80 fixed worlds on Modal with the corrected loop detector is recorded in results/s6502/v2/S1_v2.md and results/s6502/CORRECTION.md; original outputs are preserved.

**Arm A.** P1 remains not supported: 14 of 20 first replicators are confined and have a non-sequential 16-bit PC decrease (originally 14/20); all 20 are confined with random partners. P2 is supported (0 violating worlds). P3 remains not tested: both classes occur at the frozen n=32 culture count (94 transmitters, 209 regenerators), and at the literal n=64 count (94, 209); the offset component has 0 / 0 eligible cells. P4 is supported, KM median t_rep 1,125,000. The null rule does not apply, and the kill rule is not triggered. The registered verdicts are reported alongside the original in the correction tables.

**Protocol conflicts.** The registration says heritable over 64 partners, while the frozen DZ final-cell estimator uses 32; both complete final-cell classifications are now reported. It says ‘soup partners’, while exec_trace and the frozen DZ scorer draw seeded uniform-byte partners. Random tracing remains the reported registered analysis because it matches the cited frozen definition. The separate soup-snapshot sensitivity uses 16 partners from each world’s nearest saved snapshot at or after t_rep, and does not re-estimate P2 or t_closed.

| arm | first_replicators | random_confined | soup_confined | changed_confined | random_open | soup_open | changed_open | random_loop | soup_loop | changed_loop | changed_any |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 20 | 20 | 20 | 0 | 0 | 0 | 0 | 14 | 14 | 0 | 0 |
| B | 4 | 4 | 4 | 0 | 0 | 0 | 0 | 4 | 4 | 0 | 0 |
| C | 20 | 19 | 19 | 0 | 1 | 1 | 0 | 15 | 15 | 0 | 0 |
| D | 18 | 18 | 18 | 0 | 0 | 0 | 0 | 17 | 17 | 0 | 0 |

**Descriptive B–D.**

| arm | protocol | with_rep | first_confined | first_open | first_loop | first_confined_loop | first_open_straight | p2_worlds | km_t_rep | transmitter | regenerator | offset_eligible | offset_equal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | original32 | 4 | 4 | 0 | 4 | 4 | 0 | 0 | not tested / censored | 39 | 57 | 0 | 0 |
| B | v2_frozen32 | 4 | 4 | 0 | 4 | 4 | 0 | 0 | not tested / censored | 39 | 57 | 0 | 0 |
| B | v2_literal64 | 4 | 4 | 0 | 4 | 4 | 0 | 0 | not tested / censored | 39 | 57 | 0 | 0 |
| C | original32 | 20 | 19 | 1 | 15 | 14 | 0 | 0 | 837500.0 | 44 | 229 | 0 | 0 |
| C | v2_frozen32 | 20 | 19 | 1 | 15 | 14 | 0 | 0 | 837500.0 | 44 | 229 | 0 | 0 |
| C | v2_literal64 | 20 | 19 | 1 | 15 | 14 | 0 | 0 | 837500.0 | 44 | 229 | 0 | 0 |
| D | original32 | 18 | 18 | 0 | 17 | 17 | 0 | 0 | 1650000.0 | 86 | 232 | 0 | 0 |
| D | v2_frozen32 | 18 | 18 | 0 | 17 | 17 | 0 | 0 | 1650000.0 | 86 | 232 | 0 | 0 |
| D | v2_literal64 | 18 | 18 | 0 | 17 | 17 | 0 | 0 | 1650000.0 | 86 | 232 | 0 | 0 |

**Remaining limits.** Sequential instruction increments through $FFFF are now exempted using the executed Zilion instruction length. Forward alias returns remain invisible to the registered 16-bit-PC-decrease measure; the prior position-space result remains exploratory and does not change P1. The nominal 512 mutations per step are attempts: the audit demonstrates a lost update when distinct byte mutations share a packed word. This is the same shared, disclosed property as in the Z80 kernel, not a 6502-specific correction. No mutation-policy rerun was performed, so its effect on the world outcomes is unmeasured. The updated scorer retains exact Z80 DZ p=0 positive-control parity, 10/10 on every column.

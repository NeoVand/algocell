# Search for an open full replicator in wrap-BFF (generated)

- enumerated: 16,447,860 straight-line programs (prefix ≤ 3 instructions over `<>{}+-.,`, tiling unit of period 2–5 with a copy instruction, repeated to 64 bytes); 2 random partners each, 2^13 steps, wrapping pointer; alphabet filter passed (≥ 75% of each partner's bytes in the program's alphabet): 284,192; both partners ≥ 50% copied: 15947
- stage 2 (32 random partners): programs with copies ≥ 0.5: 378; with copies ≥ 0.5 and gen2 ≥ 0.3: 8; best copies: 0.84 (`,,,` + `,,,,<`, self-damage 0.03, gen2 0.03)
- wall time 1094s

| prefix   | unit   |   copies |   score |   gen2 |   self_damage |   entered |   executed |
|:---------|:-------|---------:|--------:|-------:|--------------:|----------:|-----------:|
| ,,,      | ,,,,<  |     0.84 |    0.73 |   0.03 |          0.03 |      1.00 |    7433.12 |
| },{      | ,<,,,  |     0.84 |    0.72 |   0.03 |          0.03 |      1.00 |    7433.12 |
| -><      | ,,<,,  |     0.84 |    0.69 |   0.09 |          0.06 |      1.00 |    7433.12 |
| },,      | ,,,<   |     0.84 |    0.68 |   0.03 |          0.91 |      1.00 |    7433.12 |
| },>      | ,<,,,  |     0.84 |    0.68 |   0.09 |          0.06 |      1.00 |    7433.12 |
| ,,       | ,,,<   |     0.84 |    0.69 |   0.03 |          0.03 |      1.00 |    7433.12 |
| -,       | ,,,<   |     0.84 |    0.68 |   0.03 |          0.91 |      1.00 |    7433.12 |
| +><      | ,,<,,  |     0.84 |    0.69 |   0.09 |          0.06 |      1.00 |    7433.12 |
| ,{.      | ,,<,,  |     0.84 |    0.69 |   0.03 |          0.06 |      1.00 |    7433.12 |
| ,>,      | ,,,,<  |     0.84 |    0.71 |   0.03 |          0.03 |      1.00 |    7433.12 |
| +,,      | ,,<,   |     0.84 |    0.68 |   0.03 |          0.91 |      1.00 |    7433.12 |
| +{.      | ,,,,<  |     0.84 |    0.68 |   0.03 |          0.06 |      1.00 |    7433.12 |
| -{.      | ,,,<,  |     0.84 |    0.68 |   0.03 |          0.06 |      1.00 |    7433.12 |
| +,       | ,,,<   |     0.84 |    0.68 |   0.03 |          0.91 |      1.00 |    7433.12 |
| ,        | <,,,,  |     0.84 |    0.70 |   0.09 |          0.06 |      1.00 |    7433.12 |
| ,>>      | ,,,,<  |     0.84 |    0.70 |   0.03 |          0.03 |      1.00 |    7433.12 |
| ,><      | ,,<,,  |     0.84 |    0.70 |   0.09 |          0.03 |      1.00 |    7433.12 |
| -<       | ,,,,<  |     0.84 |    0.69 |   0.09 |          0.06 |      1.00 |    7433.12 |
| },>      | ,,,,<  |     0.84 |    0.68 |   0.03 |          0.06 |      1.00 |    7433.12 |
| +>,      | ,,,,<  |     0.84 |    0.70 |   0.03 |          0.03 |      1.00 |    7433.12 |

## Nature of the 'copies' (post hoc characterisation, 32 fresh partners each, all 130 programs with copies ≥ 0.75)

- offspring tapes contain on average 9.7 distinct byte values (min 4.2, max 20.7); the modal byte fills 0.83 of the offspring (min 0.67). Every passing program is a constant fill: it writes the byte under a fixed head over the whole ring, and the parent itself is 0.78 that same byte on average, so the 75% match is met by the parent's own low information content, not by a copy.
- the 8 programs with copies ≥ 0.75 and gen2 ≥ 0.3 have units [',,,,<', ',,,<,', ',,<,,', ',<,,,']: comma-fills whose offspring (a tape of `,`) count as heritable only because the partner's own stray head moves and `,` instructions, executed when the pointer runs into the partner, write the fill onward.
- conclusion: no straight-line program in this family copies a genome of period ≥ 2 into its partner under a wrapping pointer; what exists is one-symbol tar, the BFF analogue of the Z80 zero flood. The open full replicator of the Z80 soup needs a literal write channel (code = data = literal), which BFF lacks.

# Executed-address traces (2026-10-09; pre-registered as P in `REVISION_PREREG.md`)

Every first replicator and final dominant of Stages G (80 worlds), K (20) and I (10, lethal rule) was run against 256
seeded random partners on a traced copy of the culture-test executor (`algocell_exp/gen_trace_shader.py`, byte-identical
execution, plus a bitmap of every address fetched as instruction stream). Table `per_replicator.csv`; summary
`AGREEMENT.md`.

## Pointer confinement, now measured in the Z80

- First replicators: 110 of 110 fetch partner bytes in every one of 256 encounters (entered 1.00). Final dominants:
  89 of 110 never fetch a partner byte (entered 0.00) and 21 do so in every encounter; nothing in between.
- The 89 pointer-closed finals are exactly the loop-bearing ones (67 in G, 12 in K, 10 in I). Pointer closure and zero
  inflow agree in 106 of 110 finals; the four disagreements are the Stage G finals that keep one to three partner bytes
  (pointer-closed, H > 0). No final is pointer-open with zero inflow: in the Z80 the two criteria never pull apart the
  way they do in BFF. P met; its kill criterion (a zero-inflow final that enters the partner) did not fire.

## The executed set

- The pusher fetches essentially its whole tape as instruction stream (median 35 of L positions per encounter, union
  equal to L): in a load–push tiling every byte is either an opcode or an operand.
- The closers fetch a small part: the RET NZ design 8 of 16 (its second half is regenerated, not executed), the 4-byte
  LDIR tilings 4 of L. Their unexecuted bytes are not inherited either: the block copy rewrites the ring as the periodic
  extension of the executed seed, so a mutation outside the seed is erased (0 transmissible sites in the scan).

## Two kinds of heritable variation

- The pusher's transmissible sites (1,128 across Stage G firsts) are all executed positions: operand bytes, read as
  instruction stream and pushed as data. Variation is inherited through interpreted bytes.
- The closers' transmissible sites, where they exist, are copied-but-unexecuted bytes: 82 of the 518 transmissible
  sites among Stage G finals (all in the five worlds whose genome has a jump before a block copy), 20 of 24 in the Stage I
  finals (e.g. seed 4009: 9 sites, 8 never executed). Variation inherited through uninterpreted bytes: a genotype segment
  in Langton's sense, copied as data and never run as code.

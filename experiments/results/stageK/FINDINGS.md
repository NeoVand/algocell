# Stage K — closure at the aligned intermediate length L = 32 (2026-10-09, early; pre-registered in `REVISION_PREREG.md`)

20 worlds, Stage G `none@closure1M` conditions at L = 32 (2L = 64 divides 65,536, so no relative jump can land through the
16-bit wrap), seeds 5001–5020, one million steps, run on Modal (≈ 200 s each on an L40S; batch `stageK`). Scored by the Stage G
pipeline (`stage_k_pipeline.sh`); tables in `stageK/`, scoring in `SCORING.md`.

| prediction | outcome |
|---|---|
| K1 first replicator a load–push word in ≥ 18/20 | **met**: 20/20 (`01 c5` 16, `11 d5` 3, `21 e5` 1) |
| K2 closed successor by 1M steps in ≥ 15/20 (kill < 10) | **not met, not killed**: 12/20, all twelve by a block copy (`LDIR`), copies 1.00, self-damage 0.00; the other 8 worlds still hold the open pusher (copies 0.73–0.82, damage 0.16–0.26) |
| K3 median first-replicator step 300–1,000 | **met**: 425 |

Reading. At an aligned length the open → closed order holds wherever closure has arrived, and the closer is the block copy,
not a relative jump. With L = 16 (20/20 by 300k, RET NZ and LDIR), L = 32 (12/20 by 1M, LDIR) and L = 64 (8/20 by 1M, seven by LDIR
or LDDR and one by JP (HL); corrected 2026-10-09), closure at aligned lengths is monotone in L; the fast jump closures at L = 20 and 50 are enabled by the address
wrap of the mod-2L memory and should be set apart in the text. The pair-invasion result at L = 50 is a result about that
machine as defined, not about a true ring.

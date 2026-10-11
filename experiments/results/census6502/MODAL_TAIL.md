# Period-4 BRK-halt census: chunks 201–255 on Modal

Chunks 0–200 of `m6502_brkhalt_maps_L16_k4_r16_b128-512` ran locally (Apple M4, Metal). Chunks 201–255 ran on Modal (`modal_census.py`, app `algocell-census6502`, 2026-10-11). The container first recomputed chunks 0 and 200 and required identical counters and identical hit sets (hits are appended through an atomic counter, so their order is the GPU's). Its record, verbatim:

```json
{"adapter": "NVIDIA L40S (Vulkan, DiscreteGPU)", "checks": [{"chunk": 0, "budget": 128, "counters_identical": true, "hits_identical_as_set": true, "n_hits": [827, 827], "counter_diffs": {}}, {"chunk": 0, "budget": 512, "counters_identical": true, "hits_identical_as_set": true, "n_hits": [827, 827], "counter_diffs": {}}, {"chunk": 200, "budget": 128, "counters_identical": true, "hits_identical_as_set": true, "n_hits": [77, 77], "counter_diffs": {}}, {"chunk": 200, "budget": 512, "counters_identical": true, "hits_identical_as_set": true, "n_hits": [74, 74], "counter_diffs": {}}], "chunks": 256}
```

The header and the first 201 chunk lines of the returned checkpoint are byte-identical to the local checkpoint.

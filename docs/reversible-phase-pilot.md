# Reversible rule: checks and a pilot map of the two dials

11 November 2026. Pilot only. Nothing here is replicated enough to be a result.

## What was checked exactly

`node scripts/reversible-check.mjs` (real GPU pipeline, dev server on port 5176). All six checks pass.

| check                                                                        | outcome                                                                    |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Overwrite rule repeats itself for a seed, one pair per update, 1,500 updates | 0 differing bytes (12,316 bytes changed by the run)                        |
| Reversible at dissipation 1 equals overwrite, mutation on                    | 0 differing bytes                                                          |
| Each closed event obeys `now' = out ^ now ^ before`, `before' = now`         | 300 events, 404 cells whose output changed (401 not at rest), 0 violations |
| 2,000 closed events, layers swapped, same draws replayed in reverse          | 33,678 bytes changed on the way out, 0 not restored                        |
| Same procedure at dissipation 0.5                                            | 23,998 bytes not restored (as it should be)                                |
| Six boundary rules compile and run                                           | all differ from each other                                                 |

The byte counts are from one run of the script and move a little between runs, because the warm-up before
checks 3–5 uses many pairs per update. The exact checks use one pair per update because the engine resolves colliding pairs with a GPU race: with many
pairs per update two runs of the same seed differ after a single update (21,156 of 122,880 bytes in one trial).
Every statistical statement therefore needs replicate runs.

The boundary rules (`reflect`, `cylinder`, `torus`, `mobius`, `klein`, `projective`) are checked only for running
and for differing. Their neighbour relation was derived by hand from the same flag scheme as `games-of-life`
(`wraps_x`, `wraps_y`, `flips_x`, `flips_y`); it is not yet tested cell by cell.

## The pilot sweep

`scripts/reversible-sweep.mjs`, plans `docs/reversible-sweep/plan-pilot-{a,b}.json`, data
`docs/reversible-sweep/pilot.jsonl` (153 runs), figures `docs/reversible-sweep/figures/`.

Z80 (compat 0.2), 148 × 135 cells, tape 16, ring 32, 128 steps, 8,192 pair attempts per update.
Dissipation 0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99, 1 plus the overwrite rule; mutation exponent 2, 4, 6, 8, 10
(2,048 to 8 random bytes per update, 6.4e-3 to 2.5e-5 per byte per update).

- Random start, reflecting edges, 4,000 updates: two runs per cell (seeds 101, 202).
- Replicator start (lattice tiled with `01 c5`), reflecting edges, 2,000 updates: one run per cell.
- Random start, torus, 4,000 updates: one run per cell at mutation exponents 4 and 6 only.

Measures at the last update: byte entropy, share of zero bytes, share of bytes that differ between the two layers
("heat"), share of neighbouring cells holding the same tape, gzip ratio of the lattice, and others in the file.

## What the pilot shows

1. **Dissipation 0 is a different world.** In all 17 runs, from either start and with either boundary, the lattice is indistinguishable from
   random: entropy 8.0 bits, 0.4% zero bytes, 99.6% of bytes differ between the layers, gzip ratio 1.00. A world of
   one replicator is erased within 2,000 updates.
2. **From a random start, 4,000 updates is a map of waiting times.** A run has either been taken over by a
   replicator or is still in the waiting state (about 39–40% zero bytes at low mutation). Counting a run as taken
   over when under 30% of its bytes are blank, the ten runs per column give 0, 3, 6, 3, 7, 8, 5 for dissipation
   0.3 to 1 and 8 for the overwrite rule. Apart from the zero at 0.3, that is noise, not a trend. The portraits
   catch several takeovers mid-spread, as discs.
3. **From a replicator start the picture is smooth.** After 2,000 updates the share of blank bytes is back at the
   waiting level (30–40%) to the left of a line that moves with mutation, and below it to the right. Taking under
   30% blank as the mark, the line is at dissipation 0.99 for exponent 2, 0.8 for exponent 4, and 0.6 for exponents
   6, 8 and 10 (0.295 at exponent 6, so that step is not resolved). One run per cell, a threshold chosen after
   seeing the data, and the lattices are still changing at 2,000 updates: the direction of the line is supported,
   its position is not. The gzip measure is not used for the line because blank bytes compress too: the waiting
   state scores 0.42–0.46 on it.
4. **The same line is visible from the random start at high mutation**, where takeover is fast: at exponent 2
   both runs are structured only from dissipation 0.99 (one of the two at 0.95); at exponent 4, from 0.8.
5. **Dissipation 1 and the overwrite rule look alike**, as the exact check says they must.
6. **Torus against reflecting edges:** the two rows run on a torus show the same pattern and similar numbers. One
   run per cell; this is "no difference seen at this resolution", not a test.

## What the pilot does not show

- Whether anything between dissipation 0.8 and 0.99 differs from the ordinary soup at the same mutation. By these
  summary numbers it does not, and the summary numbers are blind to dynamics (blinking, waves) that are easy to
  see by eye in the app.
- Whether the line in point 3 is a sharp transition or a slow crossover, and where it meets dissipation 0. The
  grid has no point between 0 and 0.3.
- Steady states. No run is long enough, and time to takeover is itself a random variable.

## Cold start in the closed world (one run)

`scripts/reversible-coldstart.mjs`: blank lattice at rest, a disc of random tapes of radius 12, dissipation 0,
no mutation, torus. Blank cells: 97.7% at the start, 96.6% after 250 updates, 93.2% after 1,000, 0% after 4,000;
by then 99.6% of bytes differ between the layers. The disc grows as a random blob and consumes the blank lattice.
Data and picture in `docs/reversible-sweep/cold/`.

## Longer control at the setting that looked alive (two runs per rule)

`docs/reversible-sweep/long-control.jsonl`: mutation exponent 6, reflecting edges, 16,000 updates, reversible at
dissipation 0.9 against the overwrite rule, seeds 301 and 302.

Both rules pass through the same succession: the waiting state, then short-period replicators (`01 c5`, `11 d5`,
`e5 2a`), then, in three of the four runs, a replicator of period eight of the form `xx e3 21 e3 21 yy xx yy` (yy = `c0` or `e0`), holding 22–46%
of cells at 16,000 updates (reversible: 22% and 25%; overwrite: 46% in one run, while the other was still on
`e5 2a` at 14%). The gzip ratio at the end is 0.07 and 0.16 (reversible) against 0.22 and 0.05 (overwrite). With two
runs per rule there is no detectable difference between dissipation 0.9 and the ordinary soup in these measures.
Under the reversible rule 7–8% of bytes differ between the two layers at the end.

## What a proper map needs

- Three starts per cell (random, a seeded patch, a replicator world), so that a boundary can be called independent
  of the start where they agree and history-dependent where they do not.
- Time to takeover as the measured quantity for random starts, with a takeover rule fixed in advance, and enough
  replicates per cell to estimate it.
- A finer dissipation axis below 0.3 and near the line; torus as the default so no cell is special.
- Per-event rates on both axes (mutation per byte per event, dissipation per event), so lattices of different size
  can be compared.
- An observable for dynamics, not only for the still lattice.

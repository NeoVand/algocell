# Nascent LDIR copiers in the top-10 before emergence (generated) — stageD, 512 steps, 1/16

| label            |   n |   emerged |   runs_with_ldir_in_top10_pre |   first_ldir_median |   episodes_median |   episodes_max |   max_share_median |   frac_samples_median |
|:-----------------|----:|----------:|------------------------------:|--------------------:|------------------:|---------------:|-------------------:|----------------------:|
| none             |  20 |         5 |                             2 |              103500 |               0   |              0 |              0     |                 0     |
| stack-write-only |  20 |        19 |                            20 |                 325 |               5.5 |             16 |              0     |                 0.027 |
| stack-read-only  |  20 |        10 |                             7 |              135500 |               0   |              1 |              0     |                 0     |
| push             |  20 |         8 |                             4 |              150500 |               0   |              0 |              0     |                 0     |
| call-rst-write   |  20 |        14 |                            17 |               50500 |               1   |              5 |              0.001 |                 0.005 |

`runs_with_ldir_in_top10_pre`: runs in which an LDIR-bearing tape entered the top-10 before the heritable event (or ever, if none); `episodes_pre`: times such a tape was present at one sample and absent at the next, before the event; `max_share`: largest top-10 share an LDIR tape held before the event.

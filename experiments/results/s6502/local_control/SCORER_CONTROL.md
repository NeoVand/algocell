# S1 scorer positive control (generated)

Archived DZ lp0, seeds 9001–9010, final snapshot 300,000; vendored Z80 compat '0.2'. Exact comparisons, including round-trip parsed floats (NaN equals missing).

| seed | t_rep | t_closed | first_tape | first_open | frac_heritable | frac_confined_of_heritable | frac_regenerator_of_heritable | frac_intermediate_of_heritable | frac_transmitter_of_heritable | median_unexec_sites_transmitters |
|---|---|---|---|---|---|---|---|---|---|---|
| 9001 | match | match | match | match | match | match | match | match | match | match |
| 9002 | match | match | match | match | match | match | match | match | match | match |
| 9003 | match | match | match | match | match | match | match | match | match | match |
| 9004 | match | match | match | match | match | match | match | match | match | match |
| 9005 | match | match | match | match | match | match | match | match | match | match |
| 9006 | match | match | match | match | match | match | match | match | match | match |
| 9007 | match | match | match | match | match | match | match | match | match | match |
| 9008 | match | match | match | match | match | match | match | match | match | match |
| 9009 | match | match | match | match | match | match | match | match | match | match |
| 9010 | match | match | match | match | match | match | match | match | match | match |

| column | exact matches |
|---|---:|
| t_rep | 10/10 |
| t_closed | 10/10 |
| first_tape | 10/10 |
| first_open | 10/10 |
| frac_heritable | 10/10 |
| frac_confined_of_heritable | 10/10 |
| frac_regenerator_of_heritable | 10/10 |
| frac_intermediate_of_heritable | 10/10 |
| frac_transmitter_of_heritable | 10/10 |
| median_unexec_sites_transmitters | 10/10 |

PASS: every compared endpoint and class column matches exactly.

The random streams are reproducible: judge RNG = [seed, 0, 11], final RNG = [seed, 23]; PC replays draw no randomness. At p = 0 the dial is the ordinary executor. All executions used the vendored machine; no shader fallback.

No registered 6502 world was read, listed, fetched or scored during validation.

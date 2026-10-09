# Stage L (ten million steps, L = 16 and 20, ten worlds each): pre-registered scoring

## L = 16
- worlds that closed by 10M steps: 10/10 (first closed snapshot: 6001:10000, 6002:50000, 6003:50000, 6004:10000, 6005:50000, 6006:10000, 6007:20000, 6008:75000, 6009:30000, 6010:50000)
- L1 recovery (more transmissible sites at 10M than the closed reference dominant): 1/10 (needs >= 5) -> NOT MET
- L2 every recovery in a block-copy lineage with unexecuted sites: NOT MET
- L3 (as registered) no return- or push-based dominant with > 1 transmissible site at any snapshot: NOT MET (8 violating snapshots: seed 6003 step 2000 open sites 2; seed 6003 step 2050 open sites 2; seed 6003 step 3000 open sites 2; seed 6003 step 5000 open sites 2; seed 6003 step 10000 open sites 2; seed 6003 step 15000 open sites 2; seed 6003 step 20000 open sites 2; seed 6003 step 30000 open sites 2)
- L3 restricted to pointer-closed non-block dominants: MET (0 violating snapshots)
- worlds whose dominant is open again at 10M after having closed: none
- final dominants with <= 1 transmissible site: 9/10 (kill condition component: >= 8 at both lengths)

## L = 20
- worlds that closed by 10M steps: 10/10 (first closed snapshot: 6001:175000, 6002:200000, 6003:100000, 6004:500000, 6005:20000, 6006:500000, 6007:295000, 6008:500000, 6009:115000, 6010:200000)
- L1 recovery (more transmissible sites at 10M than the closed reference dominant): 0/10 (needs >= 5) -> NOT MET
- L2 every recovery in a block-copy lineage with unexecuted sites: n/a (no recovery)
- L3 (as registered) no return- or push-based dominant with > 1 transmissible site at any snapshot: NOT MET (88 violating snapshots: seed 6001 step 500 open sites 2; seed 6001 step 1000 open sites 2; seed 6001 step 2000 open sites 2; seed 6001 step 30000 open sites 2; seed 6001 step 75000 open sites 2; seed 6002 step 2000 open sites 2; seed 6002 step 2600 open sites 2; seed 6002 step 3000 open sites 2; seed 6002 step 5000 open sites 2; seed 6002 step 10000 open sites 2; seed 6002 step 15000 open sites 2; seed 6002 step 20000 open sites 2)
- L3 restricted to pointer-closed non-block dominants: NOT MET (13 violating snapshots)
- worlds whose dominant is open again at 10M after having closed: none
- final dominants with <= 1 transmissible site: 8/10 (kill condition component: >= 8 at both lengths)

**Kill criterion** (L1 fails at both lengths with <= 1 site in >= 8/10 worlds at both): FIRED

## Per world

```
 L  seed  first_closed_step  ref_step  ref_sites                ref_tape              final_tape  final_share  final_entered  final_sites  final_unexec  final_capacity  final_block  recovered
16  6001              10000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.61060            0.0            0             0        6.000000        False      False
16  6002              50000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.50055            0.0            0             0        6.000000        False      False
16  6003              50000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.46965            0.0            0             0        6.000000        False      False
16  6004              10000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.40495            0.0            0             0        6.000000        False      False
16  6005              50000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.63865            0.0            0             0        6.000000        False      False
16  6006              10000    300000          0 1e 24 ed b0 1e 24 ed b0 6b f3 cb db 9c ed b0 b3      0.00640            0.0            4             1       31.351912         True       True
16  6007              20000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.52900            0.0            0             0        6.000000        False      False
16  6008              75000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.55570            0.0            0             0        6.000000        False      False
16  6009              30000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.61490            0.0            0             0        6.000000        False      False
16  6010              50000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.53740            0.0            0             0        6.000000        False      False
20  6001             175000   1000000          0 1e 2c ed b0 1e 2c ed b0 1e a4 ed b0 1e a4 ed b0      0.19965            0.0            0             0        8.906891         True      False
20  6002             200000   1000000          0 1e a4 ed b0 1e a4 ed b0 1e 04 ed b0 1e 04 ed b0      0.25205            0.0            0             0       10.044394         True      False
20  6003             100000   1000000          3 11 44 6b ed b8 ed c2 9e 11 44 6b ed b8 94 be 5a      0.00265            0.0            3             0       37.121210         True      False
20  6004             500000   1000000         12 00 04 5e 94 41 27 11 94 1e 2c ed b0 1e 2c ed b0      0.18175            0.0            0             0        7.870365         True      False
20  6005              20000   1000000          0 04 5e ed b0 04 5e ed b0 1e a4 ed b0 1e a4 ed b0      0.26430            0.0            0             0        8.906891         True      False
20  6006             500000   1000000          0 11 fd d9 ed b0 11 fd d9 1e 7c ed b0 1e 7c ed b0      0.25135            0.0            0             0        9.199672         True      False
20  6007             295000   1000000          1 11 1d e2 ed b0 11 1d e2 1e 7c ed b0 1e 7c ed b0      0.20165            0.0            0             0        9.076816         True      False
20  6008             500000   1000000          0 a4 5e ed b0 a4 5e ed b0 1e 7c ed b0 1e 7c ed b0      0.23880            0.0            0             0        9.044394         True      False
20  6009             115000   1000000          0 b0 11 ad d9 ed b0 11 ad 1e a4 ed b0 1e a4 ed b0      0.21740            0.0            0             0        9.139551         True      False
20  6010             200000   1000000          3 00 11 e6 26 30 ed b8 62 cb b2 4d 11 e6 26 30 ed      0.00145            0.0            3             0       28.164787         True      False

transmissible sites of the most common tape at key steps (k = thousand steps; '-' no snapshot or not heritable):
L = 16 seed 6001: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6002: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6003: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6004: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6005: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6006: 300k:0 1000k:10 2000k:10 3000k:10 5000k:4 7500k:4 10000k:4
L = 16 seed 6007: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6008: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6009: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6010: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6001: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6002: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6003: 300k:11 1000k:3 2000k:3 3000k:4 5000k:3 7500k:3 10000k:3
L = 20 seed 6004: 300k:3 1000k:12 2000k:5 3000k:5 5000k:0 7500k:0 10000k:0
L = 20 seed 6005: 300k:2 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6006: 300k:2 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6007: 300k:0 1000k:1 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6008: 300k:1 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6009: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6010: 300k:3 1000k:3 2000k:3 3000k:3 5000k:3 7500k:3 10000k:3
```

## Culture-test table of the final dominants (stage_g pipeline)
```
 L  seed  t_rep final_cf final_block  final_copied  final_damaged  final_share
16  6001   1100   RET NZ           -           1.0            0.0       0.6106
16  6002   3700   RET NZ           -           1.0            0.0       0.5006
16  6003   1400   RET NZ           -           1.0            0.0       0.4697
16  6004    200   RET NZ           -           1.0            0.0       0.4049
16  6005    500   RET NZ           -           1.0            0.0       0.6387
16  6006    550        -        LDIR           1.0            0.0       0.0064
16  6007    150   RET NZ           -           1.0            0.0       0.5290
16  6008    200   RET NZ           -           1.0            0.0       0.5557
16  6009    150   RET NZ           -           1.0            0.0       0.6149
16  6010   3450   RET NZ           -           1.0            0.0       0.5374
20  6001    550        -        LDIR           1.0            0.0       0.1996
20  6002    400        -        LDIR           1.0            0.0       0.2520
20  6003   1200        -        LDDR           1.0            1.0       0.0027
20  6004    450        -        LDIR           1.0            0.0       0.1817
20  6005    450        -        LDIR           1.0            0.0       0.2643
20  6006    450        -        LDIR           1.0            0.0       0.2514
20  6007    400        -        LDIR           1.0            0.0       0.2016
20  6008    400        -        LDIR           1.0            0.0       0.2388
20  6009    250        -        LDIR           1.0            0.0       0.2174
20  6010    300  JR NC,d           -           1.0            0.0       0.0013
```

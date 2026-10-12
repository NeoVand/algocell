# Stage L (ten million steps, L = 16 and 20, ten worlds each): pre-registered scoring

## L = 16
- worlds that closed by 10M steps: 10/10 (first closed snapshot: 6001:50000, 6002:10000, 6003:30000, 6004:75000, 6005:15000, 6006:30000, 6007:75000, 6008:20000, 6009:2000, 6010:50000)
- L1 recovery (more transmissible sites at 10M than the closed reference dominant): 0/10 (needs >= 5) -> NOT MET
- L2 every recovery in a block-copy lineage with unexecuted sites: n/a (no recovery)
- L3 (as registered) no return- or push-based dominant with > 1 transmissible site at any snapshot: NOT MET (8 violating snapshots: seed 6003 step 1000 open sites 2; seed 6003 step 1400 open sites 2; seed 6003 step 2000 open sites 2; seed 6003 step 3000 open sites 2; seed 6003 step 5000 open sites 2; seed 6003 step 10000 open sites 2; seed 6003 step 15000 open sites 2; seed 6003 step 20000 open sites 2)
- L3 restricted to pointer-closed non-block dominants: MET (0 violating snapshots)
- worlds whose dominant is open again at 10M after having closed: none
- final dominants with <= 1 transmissible site: 8/10 (kill condition component: >= 8 at both lengths)

## L = 20
- worlds that closed by 10M steps: 10/10 (first closed snapshot: 6001:150000, 6002:115000, 6003:180000, 6004:750000, 6005:300000, 6006:200000, 6007:270000, 6008:300000, 6009:150000, 6010:150000)
- L1 recovery (more transmissible sites at 10M than the closed reference dominant): 1/10 (needs >= 5) -> NOT MET
- L2 every recovery in a block-copy lineage with unexecuted sites: NOT MET
- L3 (as registered) no return- or push-based dominant with > 1 transmissible site at any snapshot: NOT MET (85 violating snapshots: seed 6001 step 75000 open sites 2; seed 6002 step 5000 open sites 2; seed 6002 step 10000 open sites 2; seed 6002 step 15000 open sites 2; seed 6002 step 20000 open sites 2; seed 6002 step 30000 open sites 2; seed 6002 step 50000 open sites 2; seed 6002 step 75000 open sites 2; seed 6002 step 100000 open sites 2; seed 6003 step 10000 open sites 2; seed 6003 step 15000 open sites 2; seed 6003 step 75000 open sites 2)
- L3 restricted to pointer-closed non-block dominants: NOT MET (13 violating snapshots)
- worlds whose dominant is open again at 10M after having closed: none
- final dominants with <= 1 transmissible site: 8/10 (kill condition component: >= 8 at both lengths)

**Kill criterion** (L1 fails at both lengths with <= 1 site in >= 8/10 worlds at both): FIRED

## Per world

```
 L  seed  first_closed_step  ref_step  ref_sites                ref_tape              final_tape  final_share  final_entered  final_sites  final_unexec  final_capacity  final_block  recovered
16  6001              50000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.65555            0.0            0             0        6.000000        False      False
16  6002              10000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.51750            0.0            0             0        6.000000        False      False
16  6003              30000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.58865            0.0            0             0        6.000000        False      False
16  6004              75000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.61355            0.0            0             0        6.000000        False      False
16  6005              15000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.61750            0.0            0             0        6.000000        False      False
16  6006              30000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.57535            0.0            0             0        6.000000        False      False
16  6007              75000    300000          4 28 65 5e ed b0 07 36 7f 6b cb db 7c ed b0 7a 9e      0.00510            0.0            4             2       31.661778         True      False
16  6008              20000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.53970            0.0            0             0        6.000000        False      False
16  6009               2000    300000         10 50 5e c3 6a 06 6a 98 00 6b cb db bc ed b0 97 3d      0.00580            0.0            4             2       31.661778         True      False
16  6010              50000    300000          0 ad e3 21 e3 21 c0 ad c0 ad e3 21 e3 21 c0 ad c0      0.58750            0.0            0             0        6.000000        False      False
20  6001             150000   1000000          0 1e a4 ed b0 1e a4 ed b0 1e 2c ed b0 1e 2c ed b0      0.21380            0.0            0             0        7.942515         True      False
20  6002             115000   1000000          0 1e a4 ed b0 1e a4 ed b0 1e 04 ed b0 1e 04 ed b0      0.19440            0.0            0             0       10.044394         True      False
20  6003             180000   1000000          0 1e 54 ed b0 1e 54 ed b0 1e 2c ed b0 1e 2c ed b0      0.29215            0.0            0             0        8.139551         True      False
20  6004             750000   1000000          0 b0 11 0d d9 ed b0 11 0d 1e 2c ed b0 1e 2c ed b0      0.22085            0.0            0             0        7.870365         True      False
20  6005             300000   1000000          2 21 e5 21 e5 21 4e 10 e5 1e a4 ed b0 1e a4 ed b0      0.20160            0.0            0             0        8.906891         True      False
20  6006             200000   1000000          3 11 44 6b ed b8 00 a1 4c 11 44 6b ed b8 36 b3 7c      0.00340            0.0            3             0       36.240791         True      False
20  6007             270000   1000000          0 1e 7c ed b0 1e 7c ed b0 1e 2c ed b0 1e 2c ed b0      0.17045            0.0            0             0        8.139551         True      False
20  6008             300000   1000000          0 1e a4 ed b0 1e a4 ed b0 b0 1e 54 ed b0 1e 54 ed      0.14530            0.0            0             0       11.436712         True      False
20  6009             150000   1000000          0 7c 5e ed b0 7c 5e ed b0 1e 7c ed b0 1e 7c ed b0      0.20750            0.0            0             0        9.139551         True      False
20  6010             150000   1000000          2 21 e5 21 e5 21 4e 10 e5 11 44 6b ed b8 9f bd 7e      0.00360            0.0            3             0       36.799282         True       True

transmissible sites of the most common tape at key steps (k = thousand steps; '-' no snapshot or not heritable):
L = 16 seed 6001: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6002: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6003: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6004: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6005: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6006: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6007: 300k:4 1000k:0 2000k:4 3000k:0 5000k:10 7500k:10 10000k:4
L = 16 seed 6008: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 16 seed 6009: 300k:10 1000k:11 2000k:11 3000k:10 5000k:4 7500k:4 10000k:4
L = 16 seed 6010: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6001: 300k:5 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6002: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6003: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6004: 300k:3 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6005: 300k:2 1000k:2 2000k:2 3000k:2 5000k:2 7500k:0 10000k:0
L = 20 seed 6006: 300k:3 1000k:3 2000k:3 3000k:3 5000k:2 7500k:3 10000k:3
L = 20 seed 6007: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6008: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6009: 300k:0 1000k:0 2000k:0 3000k:0 5000k:0 7500k:0 10000k:0
L = 20 seed 6010: 300k:2 1000k:2 2000k:2 3000k:3 5000k:3 7500k:3 10000k:3
```

## Culture-test table of the final dominants (stage_g pipeline)
```
 L  seed  t_rep final_cf final_block  final_copied  final_damaged  final_share
16  6001    900   RET NZ           -           1.0            0.0       0.6555
16  6002    850   RET NZ           -           1.0            0.0       0.5175
16  6003    950   RET NZ           -           1.0            0.0       0.5887
16  6004    150   RET NZ           -           1.0            0.0       0.6136
16  6005    800   RET NZ           -           1.0            0.0       0.6175
16  6006   1300   RET NZ           -           1.0            0.0       0.5754
16  6007   1400        -        LDIR           1.0            0.0       0.0051
16  6008    700   RET NZ           -           1.0            0.0       0.5397
16  6009    150        -        LDIR           1.0            0.0       0.0058
16  6010    150   RET NZ           -           1.0            0.0       0.5875
20  6001    400        -        LDIR           1.0            0.0       0.2138
20  6002    250        -        LDIR           1.0            0.0       0.1944
20  6003    700        -        LDIR           1.0            0.0       0.2922
20  6004    800        -        LDIR           1.0            0.0       0.2208
20  6005    950        -        LDIR           1.0            0.0       0.2016
20  6006    650        -        LDDR           1.0            1.0       0.0034
20  6007    450        -        LDIR           1.0            0.0       0.1704
20  6008   1450        -        LDIR           1.0            0.0       0.1453
20  6009    350        -        LDIR           1.0            0.0       0.2075
20  6010    350        -        LDDR           1.0            1.0       0.0036
```

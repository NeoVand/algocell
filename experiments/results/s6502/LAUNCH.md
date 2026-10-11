# S1 registered 6502 launch

**80 worlds spawned detached**: arms A–D × seeds 65001–65020, each for 10,000,000 steps without early stopping.

- App: [`ap-JPFklMO051sIKBw6tRIxCP`](https://modal.com/apps/neovand/main/ap-JPFklMO051sIKBw6tRIxCP).
- Launch time (UTC): `2026-10-11T07:35:47.669732+00:00`.
- Runner/wrapper commit: `5e98673dc363dae043ebe7315ea19bb0650c4e09` on `feat/s6502`.
- Command: `XDG_RUNTIME_DIR=/tmp /Users/neo/repos/algocell/experiments/.venv/bin/modal run --detach --profile neovand modal_s6502.py` from `experiments/`.
- Inputs: one `Function.spawn` per world; defaults select all four arms and all twenty registered seeds.
- GPU: L40S, one input per single-use container, no retries, timeout 14,400 seconds.
- Volume: `algocell-atlas-runs/s6502/<arm>/`. Reload before starting, completed worlds skipped, incomplete worlds reported and retained; commit after each snapshot and at the end.
- Confirmed at `2026-10-11T07:37:15.335717+00:00`: app state `ephemeral (detached)`, **49 active Modal tasks**; 80 submitted calls, all shown as pending/executing with no reported failures. Modal may queue inputs while containers start.
- Volume metadata confirms world files by arm: A=20, B=20, C=9, D=0. No registered JSONL contents or soups were fetched or inspected.
- Smoke: four arms, seed 999, 30,000 steps; every structural/provenance check passed. See [SMOKE.md](SMOKE.md) and [smoke_check.json](smoke_check.json).
- Cost gate: 56.09 baseline L40S-hours, $109.37 GPU cost; **$140.58** including a CPU/memory allowance and 25% extra wall time, below $160. Current L40S rate confirmed at $1.95/hour.
- Launch metadata and per-call status: [launch_calls.json](launch_calls.json); app-list evidence: [launch_app_list.json](launch_app_list.json).
- The impact-study app was untouched. No app was stopped. Completion was not awaited; no push was performed.

## Function call IDs

| Arm | Seed | Call ID |
|---|---:|---|
| A | 65001 | `fc-01M4MY79RBJ5TA5RBJS5CCFX4S` |
| A | 65002 | `fc-01M4MY79W9VA9YZ3Z4XQZF6YYK` |
| A | 65003 | `fc-01M4MY7A6MZ1F3N9Q1TP6EDRJE` |
| A | 65004 | `fc-01M4MY7AC83P6BT22PWGSYHYWY` |
| A | 65005 | `fc-01M4MY7AF58GT390WV44Q697JT` |
| A | 65006 | `fc-01M4MY7ANVMJBM8K1Y7HFD5MEM` |
| A | 65007 | `fc-01M4MY7AV0K1WZW2RH9NWK0PF8` |
| A | 65008 | `fc-01M4MY7B092NWMFPW03J0FBDZX` |
| A | 65009 | `fc-01M4MY7B5RDF51FPXNK6FHJR4W` |
| A | 65010 | `fc-01M4MY7BA8QRX77ZYV0TM3ZP3W` |
| A | 65011 | `fc-01M4MY7BDAT7FVFX4NEHPWZPGH` |
| A | 65012 | `fc-01M4MY7BJ7A6HE8TMYDWTC9K37` |
| A | 65013 | `fc-01M4MY7BP5238NWK8ZAKKMRYVZ` |
| A | 65014 | `fc-01M4MY7BTTKZ7ZDE4CDJKKX39E` |
| A | 65015 | `fc-01M4MY7BY7PBM4KE80H55FV8TM` |
| A | 65016 | `fc-01M4MY7C17SRSS2K7S1PHDNB46` |
| A | 65017 | `fc-01M4MY7C46VNGKHGMDF039YMEJ` |
| A | 65018 | `fc-01M4MY7C76C7MXHYHSA19HYHT0` |
| A | 65019 | `fc-01M4MY7CA6VJVJATPNMW5P4Q4X` |
| A | 65020 | `fc-01M4MY7CCZ20DNYW2NZ4Q9SG02` |
| B | 65001 | `fc-01M4MY7CFYY0PZYEPBRRX244G7` |
| B | 65002 | `fc-01M4MY7CK4EWM76A04S1EQT5MQ` |
| B | 65003 | `fc-01M4MY7CP33AXRC5MKS6119F8W` |
| B | 65004 | `fc-01M4MY7CRXQRGAYA5DZYY9ZW80` |
| B | 65005 | `fc-01M4MY7CVTM9EDBXC0BGGXPE2R` |
| B | 65006 | `fc-01M4MY7CYQAA15QA8D6FJZ7TCT` |
| B | 65007 | `fc-01M4MY7D1XBYKSRDFAMYXBWBVE` |
| B | 65008 | `fc-01M4MY7D5F5SHCEZRYGPPN1774` |
| B | 65009 | `fc-01M4MY7D8AR89HHDR7BVTXWRWH` |
| B | 65010 | `fc-01M4MY7DBCFVGEE6Z636Q2TWMK` |
| B | 65011 | `fc-01M4MY7DE62NWB5C5KHKQ8FBS5` |
| B | 65012 | `fc-01M4MY7DH2CN96C15WD7DG6W04` |
| B | 65013 | `fc-01M4MY7DM0644QT094NWKSCJ81` |
| B | 65014 | `fc-01M4MY7DPZS2CVCDJWJB7Y4J3N` |
| B | 65015 | `fc-01M4MY7DSY2APME5PD1Q8TP65K` |
| B | 65016 | `fc-01M4MY7DWN4S427Z08NKZJAZ3B` |
| B | 65017 | `fc-01M4MY7DZPEE8Y6RC4VFCX9B5Y` |
| B | 65018 | `fc-01M4MY7E310F0AA0V5PNFW4EV7` |
| B | 65019 | `fc-01M4MY7E5TQTETKX01RGK4401C` |
| B | 65020 | `fc-01M4MY7E8TSDRJEQEDH8SE90HZ` |
| C | 65001 | `fc-01M4MY7EBJFRPR8NMW0DY3RX8M` |
| C | 65002 | `fc-01M4MY7EEKKSTEX91SVWAEQKKN` |
| C | 65003 | `fc-01M4MY7EHK7HBGV352F6T8Q529` |
| C | 65004 | `fc-01M4MY7EMYWYD2DFB38Z3R2MM2` |
| C | 65005 | `fc-01M4MY7ERHGWD5ZVH04J2CP1J5` |
| C | 65006 | `fc-01M4MY7EVSNZWQXSA1EFQHMT1K` |
| C | 65007 | `fc-01M4MY7EYS638Z7GZ7E8AKHZFJ` |
| C | 65008 | `fc-01M4MY7F1QGHB5AZ1C1DBDHBT0` |
| C | 65009 | `fc-01M4MY7F4KC5R7PCXQ1Q4DZ2J1` |
| C | 65010 | `fc-01M4MY7F7NQFWPT2AKTCRZXZCG` |
| C | 65011 | `fc-01M4MY7FB6SPKY5FD9CAVJ9ES7` |
| C | 65012 | `fc-01M4MY7FE1BM1MX9NDEBSYRGNR` |
| C | 65013 | `fc-01M4MY7FH0YWJDQVZB5Q6D2G88` |
| C | 65014 | `fc-01M4MY7FKRG7GGCT38DGB86R8Y` |
| C | 65015 | `fc-01M4MY7FPQH5N1AEMQASM0S0AN` |
| C | 65016 | `fc-01M4MY7FSSG7EHSV407XBEZQH4` |
| C | 65017 | `fc-01M4MY7FXMAY6GMTKM99EHGYRT` |
| C | 65018 | `fc-01M4MY7G0ACH14NPKH9M5YR19C` |
| C | 65019 | `fc-01M4MY7G38H5DDHMASZ8CSRX9D` |
| C | 65020 | `fc-01M4MY7G67DNE1J5482P588SWF` |
| D | 65001 | `fc-01M4MY7G9A5JDRBGP7MCSXWC7J` |
| D | 65002 | `fc-01M4MY7GCGDKF1W6GFMQ90VJPJ` |
| D | 65003 | `fc-01M4MY7GFNRYYXCXC0AJ0RQTFD` |
| D | 65004 | `fc-01M4MY7GJG30E2D8SNH7ZRW6PV` |
| D | 65005 | `fc-01M4MY7GPAW8BNYXE6QVPDK1NG` |
| D | 65006 | `fc-01M4MY7GS3A3ZPN1XSPM1VXZZT` |
| D | 65007 | `fc-01M4MY7GWH9TYGBNB5R0RAY4KV` |
| D | 65008 | `fc-01M4MY7GZZMGFVV9ME4NZHQW5R` |
| D | 65009 | `fc-01M4MY7H34YGW8GKMVRDH6FDSF` |
| D | 65010 | `fc-01M4MY7H632SSQR1TBSJ28K6SG` |
| D | 65011 | `fc-01M4MY7H9D7AX6VXFH0YF1G0KM` |
| D | 65012 | `fc-01M4MY7HCPKHR6F97YN0F5R5NC` |
| D | 65013 | `fc-01M4MY7HGK0EMW88E4RCVH1KC2` |
| D | 65014 | `fc-01M4MY7HKMDTCECXFS4X56VNSE` |
| D | 65015 | `fc-01M4MY7HPDHHDPT3WRAVR62PJV` |
| D | 65016 | `fc-01M4MY7HSBW1XP028VDKR89YSY` |
| D | 65017 | `fc-01M4MY7HWR17QF9TTD1EK4K3GW` |
| D | 65018 | `fc-01M4MY7HZC9E1B43FYEKPXHKEB` |
| D | 65019 | `fc-01M4MY7J2642ZXQMF3DYJ9PCXM` |
| D | 65020 | `fc-01M4MY7J5BE4XCCZZM95EF7EBT` |

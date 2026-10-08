# Two-byte word census at L = 16 (generated)

- 65,536 words tiled to 16 bytes, culture test with 32 random partners, 128 steps: heritable (gen2 ≥ 0.3) 254, faithful 8, score ≥ 0.5 60
- heritable words with a control-flow instruction: 246/254; partner test (256 partners): copied median 0.00 (max 0.81), self-damage median 1.00 (min 0.14); words copying ≥ 0.95 of partners: 0

| word   | mnemonics              | control_flow   |   score |   gen2 | faithful   |   copied |   damaged |
|:-------|:-----------------------|:---------------|--------:|-------:|:-----------|---------:|----------:|
| 2a e5  | LD HL,(nn) ; PUSH HL   | -              |    0.83 |   0.59 | True       |     0.70 |      0.39 |
| e5 2a  | PUSH HL ; LD HL,(nn)   | -              |    0.82 |   0.57 | True       |     0.81 |      0.14 |
| 21 e5  | LD HL,nn ; PUSH HL     | -              |    0.87 |   0.55 | True       |     0.66 |      0.37 |
| 01 c5  | LD BC,nn ; PUSH BC     | -              |    0.84 |   0.55 | True       |     0.69 |      0.34 |
| c5 01  | PUSH BC ; LD BC,nn     | -              |    0.70 |   0.52 | True       |     0.48 |      0.92 |
| 8d d4  | ADC A,L ; CALL NC,nn   | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4b c4  | LD C,E ; CALL NZ,nn    | CALL NZ,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4b d4  | LD C,E ; CALL NC,nn    | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4b f4  | LD C,E ; CALL P,nn     | CALL P,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8b d4  | ADC A,E ; CALL NC,nn   | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8c cc  | ADC A,H ; CALL Z,nn    | CALL Z,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8d cc  | ADC A,L ; CALL Z,nn    | CALL Z,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8d cd  | ADC A,L ; CALL nn      | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4b cd  | LD C,E ; CALL nn       | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8d f4  | ADC A,L ; CALL P,nn    | CALL P,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4d c4  | LD C,L ; CALL NZ,nn    | CALL NZ,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4d d4  | LD C,L ; CALL NC,nn    | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8b f4  | ADC A,E ; CALL P,nn    | CALL P,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 6b d4  | LD L,E ; CALL NC,nn    | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 8b cd  | ADC A,E ; CALL nn      | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 6d d4  | LD L,L ; CALL NC,nn    | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| d4 02  | CALL NC,nn ; LD (BC),A | CALL NC,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4d cd  | LD C,L ; CALL nn       | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4d f4  | LD C,L ; CALL P,nn     | CALL P,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 6b cd  | LD L,E ; CALL nn       | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 6b f4  | LD L,E ; CALL P,nn     | CALL P,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4b e4  | LD C,E ; CALL PO,nn    | CALL PO,nn     |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 6d cd  | LD L,L ; CALL nn       | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 6d f4  | LD L,L ; CALL P,nn     | CALL P,nn      |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 0d cd  | DEC C ; CALL nn        | CALL nn        |    0.50 |   0.50 | False      |     0.00 |      1.00 |
| 4d e4  | LD C,L ; CALL PO,nn    | CALL PO,nn     |    0.50 |   0.49 | False      |     0.00 |      1.00 |
| 6d e4  | LD L,L ; CALL PO,nn    | CALL PO,nn     |    0.50 |   0.49 | False      |     0.00 |      1.00 |
| 6b e4  | LD L,E ; CALL PO,nn    | CALL PO,nn     |    0.50 |   0.49 | False      |     0.00 |      1.00 |
| 6a e4  | LD L,D ; CALL PO,nn    | CALL PO,nn     |    0.50 |   0.45 | False      |     0.00 |      1.00 |
| 8a d4  | ADC A,D ; CALL NC,nn   | CALL NC,nn     |    0.50 |   0.45 | False      |     0.00 |      1.00 |
| d4 05  | CALL NC,nn ; DEC B     | CALL NC,nn     |    0.44 |   0.45 | False      |     0.00 |      1.00 |
| 8c d4  | ADC A,H ; CALL NC,nn   | CALL NC,nn     |    0.50 |   0.45 | False      |     0.00 |      1.00 |
| 32 c4  | LD (nn),A ; CALL NZ,nn | CALL NZ,nn     |    0.44 |   0.45 | False      |     0.00 |      1.00 |
| 8c f4  | ADC A,H ; CALL P,nn    | CALL P,nn      |    0.50 |   0.45 | False      |     0.00 |      1.00 |
| 4a e4  | LD C,D ; CALL PO,nn    | CALL PO,nn     |    0.50 |   0.45 | False      |     0.00 |      1.00 |

## Deterministic fixed points against the all-zero partner (THEOREMS.md Proposition 3)

- words whose tiling reappears in the zero partner after one 128-step encounter: 6 / 65,536 — `00 00`, `01 c5`, `11 d5`, `21 e5`, `2a e5`, `e5 2a`
- of these, organism intact after the encounter: 6

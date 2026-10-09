# BFF dials: lethality (D1) and write ratio (D2) (2026-10-09; pre-registered in `REVISION_PREREG.md`, D)

Six soups per setting (seeds 1–6), 2¹⁷ programs, 16,384 epochs, Modal batches `bff_dial_hp0.01/0.03/0.1/0.3` (wrap + literal,
an unmatched bracket halts with probability p) and `bff_dial_r2/r3` (literal push, no wrap, the literal writes r copies of its
word); p = 0 (`wraplitnh`, 12 soups), p = 1 (`wraplit`, 12) and r = 1 (`lit`, 12) are the existing cells. Scored by
`micro/bff_analysis.py` (`results/<batch>/runs.csv`) and `dials_score.py` (`d1_lethality.csv`, `d2_bandwidth.csv`, `SCORING.md`).
r = 2 outcomes are appended below when its last three soups land.

| prediction | outcome |
|---|---|
| D1-1 the open all-`P` population persists (share ≥ 10% at 16,384 epochs) in ≥ 4/6 at p = 0.01 and ≤ 1/6 at p = 0.3 | **half met**: 6/6 at p = 0.01 (and 6/6 at 0.03 and 0.1, share 0.98 throughout); at p = 0.3 the share falls below 1% at epochs 832–896 in 6/6 and recovers to 0.100–0.111 by 16,384 in 5/6, so the persistence rule is met marginally and the ≤ 1/6 clause is not |
| D1-2 the collapse epoch falls monotonically in p within a factor of three of 1/p | **not met**: no collapse at p ≤ 0.1; 896 at p = 0.3; 192 at p = 1 (1/p scaling from p = 1 predicts 1,920 at p = 0.1) |
| D1-3 no closure by control flow at any p | **met**: no loop-bearing dominant and no closed final in 24/24 dial soups |
| Kill (extinction at p = 0.01 in ≥ 4/6) | not fired |
| D2-1 inflow ≤ 1 bit at r ≥ 2 against 8 bits at r = 1 | **met at r = 3**: 0.00 bits in 6/6 (one offspring string over 256 partners) against 8.00 bits in 12/12 at r = 1 |
| D2-2 the first replicator still the loop-free, pointer-open tiling | **met**: 6/6 loop-free, pointer enters the partner in 1.00 of encounters, copies 1.00 |
| D2-3 collapse into bracket tar slower or absent at r ≥ 2 (all-`P` share at epoch 1,024) | **met, absent at r = 3**: all-`P` share 3.6–7.3% at epoch 1,024 in 5/6 (0 in one) against 0–0.24% at r = 1; every random tape heritable at 16,384 epochs in 6/6 (`final_heritable` 1.0), the soup a cloud of `P`-tilings with harmless operands (`50 50`, `50 c9`, `50 73`, `21 50`, `50 af`) |

## Reading

- Lethality is a threshold, not a dial. Up to p = 0.1 the open replicator holds 98% of the soup for 16,384 epochs as if the
  brackets were harmless; at p = 0.3 it collapses within ~900 epochs and settles at a tenth of the soup with 0.31–0.44 of
  random tapes heritable; at p = 1 it collapses within ~200 epochs to a 0.2% minority. Model 5's window closes when the
  tar's kill rate exceeds the copier's reproduction rate, and the crossover lies between p = 0.1 and 0.3 for this copier.
- The write ratio is the other boundary of Theorem 2, now measured: at r = 3 (write ratio 2.0) the copy completes within
  one pass, the first replicator is information-closed from birth (0 bits) without a loop and without a wrapping pointer,
  and the lethal tar never forms because no copy is partial; the open phase then lasts for as long as we watched.
- Both dials change the end of the open phase, not its beginning: the first replicator is the loop-free literal tiling in
  all 30 dial soups scored so far.

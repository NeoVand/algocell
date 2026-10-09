"""Round-2 review check (2026-10-09): run the reviewer-constructed 8080 closer (L = 32) in our executor, under the i8080 suppression and unsuppressed. Output: results/review_r2/check_8080_closer.txt"""
import sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from algocell_exp import exectrace as X
from algocell_exp import assay as A
import make_conds

core = bytes.fromhex("21 20 00 31 40 00 16 10 2b 46 2b 4e c5 15 c2 08 00 c3 11 00".replace(" ", ""))
rng = np.random.default_rng(7)
L = 32
for label, sup in (("i8080", make_conds.ABLATIONS["i8080"]), ("full Z80", [])):
    exact_all, conf_all = [], []
    for trial in range(4):
        payload = rng.integers(0, 256, size=L - len(core), dtype=np.uint8)
        tape = np.concatenate([np.frombuffer(core, np.uint8), payload])
        P = rng.integers(0, 256, size=(256, L), dtype=np.uint8)
        res, masks = X.execute_pairs_traced(np.concatenate([np.repeat(tape[None], 256, 0), P], 1), L, 128, suppress=sup)
        off, me = res[:, L:], res[:, :L]
        ex = X.exec_positions(masks, 2 * L)
        exact_all.append(float((off == tape).all(1).mean()))
        conf_all.append(float((~ex[:, L:].any(1)).mean()))
        intact = float((me == tape).all(1).mean())
        # serial transfer: the copy of a copy, eight times, against fresh partners
        t = tape.copy(); ok = True
        for g in range(8):
            Q = rng.integers(0, 256, size=(1, L), dtype=np.uint8)
            r2, _ = X.execute_pairs_traced(np.concatenate([t[None], Q], 1), L, 128, suppress=sup)
            t2 = r2[0, L:]
            ok &= bool((t2 == tape).all()); t = t2
        print(f"{label}: trial {trial}: exact copy {exact_all[-1]:.3f} of 256 partners, pointer confined {conf_all[-1]:.3f}, parent intact {intact:.3f}, 8 serial transfers exact: {ok}")
    # culture test (the paper's heredity assay) on the last tape
    r = A.assay(tape.tobytes(), z80_steps=128, suppress=list(sup), n=64, seed=1)
    print(f"{label}: culture test score {r.get('score', float('nan')):.3f}, gen2 {r.get('gen2_score', float('nan')):.3f}")

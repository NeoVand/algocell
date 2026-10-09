"""A BFF primordial soup (Agüera y Arcas et al. 2024, §2.1) with the two switches of THEORY.md P1.

    python -m micro.bff_soup --out runs/bff/std_s1 --seed 1 [--ip-wrap] [--density k] [--n 131072] [--epochs 16384]

Published setup, reproduced: 2^17 programs of 64 uniformly random bytes; each epoch every program is put in exactly
one random ordered pair, the pair is concatenated, executed for 2^13 steps (or until the program ends) and split;
background mutation replaces each byte with a random byte with probability 2^-12 ≈ 0.024% per epoch.
Switches: `--ip-wrap` makes the instruction pointer wrap modulo 128 instead of ending the encounter when it leaves
the pair; `--density k` maps k byte values (instead of 1) to each of the ten instructions.

Recorded, every epoch (epochs.csv): mean steps executed, fraction of encounters whose pointer entered the partner
(population openness), mean writes into the partner, copy events in a sample of 1,024 pairs (partner became a
≥ 75% copy of the program at the best cyclic shift, forwards or reversed), the fraction of those copies made with
the pointer confined to the program ("closed copies"), mean chunk transfer (newly written partner bytes that match
the program at the best alignment), zero-byte fraction; every `--sample-every` epochs (samples.jsonl): byte entropy,
high-order entropy (H0 − brotli-q2 bits/byte), unique-tape fraction, the ten most common tape classes (a tape and its
reverse are one class) with shares, culture tests of the top three (copies, gen2, pointer-entered-partner,
self-damage, has_loop) and the heritable fraction of 32 random tapes. Snapshots of the whole soup (brotli) at
`--snapshot-epochs`, at the first sample that meets the Z80 replicator criterion (top-3 class share ≥ 0.5% and
gen2 ≥ 0.3) and at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import brotli
import numpy as np

from micro.bff import BFF, PAIR, TAPE, assay, density_map, has_loop


def batch_best_similarity(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row: best fraction of B equal to a cyclic shift of A or of reversed A → (best, shift, reversed)."""
    n = A.shape[0]
    best = np.zeros(n)
    arg_s = np.zeros(n, dtype=np.int16)
    arg_r = np.zeros(n, dtype=bool)
    for rev in (False, True):
        src = A[:, ::-1] if rev else A
        for s in range(TAPE):
            v = (B == np.roll(src, s, axis=1)).mean(1)
            m = v > best
            best[m] = v[m]
            arg_s[m] = s
            arg_r[m] = rev
    return best, arg_s, arg_r


def chunk_transfer(A: np.ndarray, B0: np.ndarray, B1: np.ndarray) -> np.ndarray:
    """Newly written partner bytes (B1 != B0) that equal the program at the best alignment (shift, reversal)."""
    n = A.shape[0]
    best = np.zeros(n, dtype=np.int32)
    new = B1 != B0
    for rev in (False, True):
        src = A[:, ::-1] if rev else A
        for s in range(TAPE):
            v = ((B1 == np.roll(src, s, axis=1)) & new).sum(1)
            best = np.maximum(best, v)
    return best


VOID = np.dtype((np.void, TAPE))


def lex_less(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise lexicographic a < b for uint8 matrices."""
    diff = a != b
    anyd = diff.any(1)
    first = diff.argmax(1)
    idx = np.arange(a.shape[0])
    return anyd & (a[idx, first] < b[idx, first])


def canonical_classes(soup: np.ndarray, top: int = 10) -> list[tuple[bytes, int]]:
    """Most common tape classes; a tape and its reverse are one class (BFF copiers often write the copy reversed)."""
    rev = soup[:, ::-1]
    canon = np.where(lex_less(rev, soup)[:, None], rev, soup)
    vals, counts = np.unique(np.ascontiguousarray(canon).view(VOID).ravel(), return_counts=True)
    order = np.argsort(-counts)[:top]
    return [(vals[i].tobytes(), int(counts[i])) for i in order]


def unique_fraction(soup: np.ndarray) -> float:
    return len(np.unique(np.ascontiguousarray(soup).view(VOID).ravel())) / soup.shape[0]


def entropy_bits(soup: np.ndarray) -> float:
    c = np.bincount(soup.ravel(), minlength=256).astype(float)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())


def high_order_entropy(soup: np.ndarray) -> tuple[float, float]:
    h0 = entropy_bits(soup)
    comp = len(brotli.compress(soup.tobytes(), quality=2))
    return h0, h0 - 8.0 * comp / soup.size


def run(out: str, n: int, epochs: int, seed: int, ip_wrap: bool, density: int, sample_every: int, snapshot_epochs: list[int],
        steps: int = 1 << 13, mutation: float = 2.0 ** -12, sample_pairs: int = 512, literal: bool = False, nohalt: bool = False,
        halt_p: float = 1.0, lit_rep: int = 1) -> dict:
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    amap = None if density <= 1 else density_map(density, seed=0)
    if literal and amap is not None:
        amap = amap.copy(); amap[ord("P")] = 11
    bff = BFF(max_pairs=n // 2, steps=steps, ip_wrap=ip_wrap, alphabet=amap, literal=literal, nohalt=nohalt, halt_p=halt_p, lit_rep=lit_rep)
    amap_eff = bff.alphabet
    soup = rng.integers(0, 256, size=(n, TAPE), dtype=np.uint8)
    cond = {"n_programs": n, "tape": TAPE, "steps": steps, "mutation": mutation, "ip_wrap": ip_wrap, "density": density, "literal": literal, "nohalt": nohalt, "halt_p": halt_p, "lit_rep": lit_rep, "seed": seed,
            "epochs": epochs, "sample_every": sample_every, "snapshot_epochs": snapshot_epochs, "sample_pairs": sample_pairs}
    json.dump(cond, open(os.path.join(out, "cond.json"), "w"), indent=1)
    ep_f = open(os.path.join(out, "epochs.csv"), "w")
    ep_f.write("epoch,executed_mean,frac_entered,writesB_mean,copy_frac,closed_copy_frac,chunk_mean,chunk_p90,zero_frac,elapsed_s\n")
    sm_f = open(os.path.join(out, "samples.jsonl"), "w")
    t0 = time.time()
    t_rep = None
    snaps = set(snapshot_epochs)

    def snapshot(tag):
        with open(os.path.join(out, f"soup_{tag}.u8.br"), "wb") as fh:
            fh.write(brotli.compress(soup.tobytes(), quality=5))

    def sample(epoch, stats):
        h0, hoe = high_order_entropy(soup)
        uniq = unique_fraction(soup)
        classes = canonical_classes(soup)
        tops = []
        for rank, (tape_b, cnt) in enumerate(classes[:3]):
            t = np.frombuffer(tape_b, dtype=np.uint8)
            r = assay(bff, t, n=64, seed=seed * 7919 + epoch + rank)
            tops.append({"rank": rank, "share": cnt / n, "tape": t.tobytes().hex(), "has_loop": has_loop(t, amap_eff), **{k: round(v, 4) for k, v in r.items()}})
        idx = rng.choice(n, size=32, replace=False)
        her = [assay(bff, soup[i], n=16, seed=seed * 104729 + epoch + j)["gen2"] >= 0.3 for j, i in enumerate(idx)]
        rec = {"epoch": epoch, "H0": round(h0, 4), "HOE": round(hoe, 4), "unique_frac": round(uniq, 5),
               "classes": [{"tape": c[0].hex(), "share": c[1] / n} for c in classes],
               "top": tops, "frac_heritable": float(np.mean(her)), **stats}
        sm_f.write(json.dumps(rec) + "\n")
        sm_f.flush()
        return rec

    for epoch in range(epochs + 1):
        if epoch in snaps:
            snapshot(f"e{epoch}")
        perm = rng.permutation(n)
        pairs = soup[perm].reshape(n // 2, PAIR)
        pre = pairs[:sample_pairs].copy()
        mem, outp = bff.execute(pairs)
        # per-epoch statistics
        A, B1 = pre[:, :TAPE], mem[:sample_pairs, TAPE:]
        best, _, _ = batch_best_similarity(A, B1)
        copies = best >= 0.75
        ch = chunk_transfer(A, pre[:, TAPE:], B1)
        entered = outp[:, 1].astype(bool)
        stats = {"executed_mean": float(outp[:, 0].mean()), "frac_entered": float(entered.mean()), "writesB_mean": float(outp[:, 3].mean()),
                 "copy_frac": float(copies.mean()), "closed_copy_frac": float((~entered[:sample_pairs][copies]).mean()) if copies.any() else float("nan"),
                 "chunk_mean": float(ch.mean()), "chunk_p90": float(np.percentile(ch, 90)), "zero_frac": float((soup == 0).mean())}
        ep_f.write(f"{epoch},{stats['executed_mean']:.1f},{stats['frac_entered']:.4f},{stats['writesB_mean']:.3f},{stats['copy_frac']:.4f},"
                   f"{stats['closed_copy_frac']:.4f},{stats['chunk_mean']:.3f},{stats['chunk_p90']:.1f},{stats['zero_frac']:.4f},{time.time() - t0:.1f}\n")
        soup[perm] = mem.reshape(n, TAPE)
        # background mutation
        n_mut = rng.binomial(n * TAPE, mutation)
        if n_mut:
            pos = rng.integers(0, n * TAPE, size=n_mut)
            soup.reshape(-1)[pos] = rng.integers(0, 256, size=n_mut, dtype=np.uint8)
        if epoch % sample_every == 0:
            rec = sample(epoch, stats)
            if t_rep is None and any(t["share"] >= 0.005 and t["gen2"] >= 0.3 for t in rec["top"]):
                t_rep = epoch
                snapshot("emergence")
            if epoch % (sample_every * 16) == 0:
                top = rec["top"][0]
                print(f"epoch {epoch} HOE {rec['HOE']:.3f} uniq {rec['unique_frac']:.3f} entered {stats['frac_entered']:.2f} copies {stats['copy_frac']:.3f} "
                      f"top share {top['share']:.4f} gen2 {top['gen2']:.2f} loop {top['has_loop']} entered {top['entered']:.2f} heritable {rec['frac_heritable']:.2f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            ep_f.flush()
    snapshot("final")
    summary = {**cond, "t_rep": t_rep, "elapsed_s": time.time() - t0}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    ep_f.close()
    sm_f.close()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1 << 17)
    ap.add_argument("--epochs", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ip-wrap", action="store_true")
    ap.add_argument("--literal", action="store_true", help="add the literal-push instruction P (byte 0x50)")
    ap.add_argument("--nohalt", action="store_true", help="unmatched brackets are no-ops instead of halting (benign tar)")
    ap.add_argument("--density", type=int, default=1)
    ap.add_argument("--sample-every", type=int, default=64)
    ap.add_argument("--snapshot-epochs", default="0,1024,2048,4096,8192")
    ap.add_argument("--steps", type=int, default=1 << 13)
    a = ap.parse_args()
    snaps = [int(x) for x in a.snapshot_epochs.split(",") if x]
    s = run(a.out, a.n, a.epochs, a.seed, a.ip_wrap, a.density, a.sample_every, snaps, steps=a.steps, literal=a.literal, nohalt=a.nohalt)
    print("done", json.dumps(s))


if __name__ == "__main__":
    main()

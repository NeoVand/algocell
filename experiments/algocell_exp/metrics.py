"""Soup-level measurements: species (per-cell hash) statistics, byte entropy,
and the Computational Life paper's high-order entropy (byte Shannon entropy
minus the normalised compressed size)."""

from __future__ import annotations

import brotli
import numpy as np

from .isa import mechanisms


def species_stats(hashes: np.ndarray) -> dict:
    n = hashes.size
    uniq, counts = np.unique(hashes, return_counts=True)
    p = counts / n
    order = np.argsort(-counts)
    top = counts[order[0]] / n
    H = float(-(p * np.log2(p)).sum())
    simpson = float(1.0 - (p * p).sum())
    richness = int((counts >= max(1, n // 1000)).sum())
    return {
        "unique": int(uniq.size),
        "top_share": float(top),
        "top_hash": int(uniq[order[0]]),
        "top3_hashes": [int(h) for h in uniq[order[:3]]],
        "top3_shares": [float(c / n) for c in counts[order[:3]]],
        "H_species": H,
        "simpson": simpson,
        "richness": richness,
    }


def q_radius(tape_length: int) -> int:
    """Quasispecies Hamming radius: a quarter of the tape (4 for L=16, 1 for L=4, 25 for L=100)."""
    return max(1, -(-tape_length // 4))


def quasispecies_share(soup: np.ndarray, top_tape: np.ndarray) -> dict:
    """Fraction of cells within Hamming distance <= r of the dominant tape, for
    r = L/8 (q_half) and r = L/4 (q, the primary occupancy measure).

    Exact-hash species undercount a replicator's occupancy: mutation keeps
    fragmenting it into near-identical variants (pilot: the classic replicator
    plateaus near 10% by exact hash while the grid is visibly full of it). The
    Hamming ball around the dominant tape is the quasispecies occupancy. The
    radius scales with tape length so the measure means the same thing for a
    4-byte and a 100-byte organism (a fixed radius 4 covers all of a 4-byte tape)."""
    L = soup.shape[1]
    r = q_radius(L)
    r_half = max(1, r // 2)
    d = (soup != top_tape[None, :]).sum(axis=1)
    return {"q_radius": r, "q_half_share": float((d <= r_half).mean()), "q_share": float((d <= r).mean())}


def motif_share(soup: np.ndarray, top_tape: np.ndarray, min_repeats: int | None = None) -> dict:
    """Fraction of cells that contain the dominant tape's most frequent 2-byte
    word at least `min_repeats` times (default L/4, min 2). Catches cells that
    carry the replicator's core instruction pair but have drifted beyond the
    quasispecies radius, which is what the eye sees as 'the grid is full of it'."""
    if min_repeats is None:
        min_repeats = max(2, q_radius(top_tape.size))
    pairs = top_tape[:-1].astype(np.uint16) | (top_tape[1:].astype(np.uint16) << 8)
    vals, counts = np.unique(pairs, return_counts=True)
    motif = vals[np.argmax(counts)]
    lo, hi = np.uint8(motif & 0xFF), np.uint8(motif >> 8)
    hits = ((soup[:, :-1] == lo) & (soup[:, 1:] == hi)).sum(axis=1)
    return {"motif": f"{int(lo):02x} {int(hi):02x}", "motif_share": float((hits >= min_repeats).mean())}


# Raw byte-pattern census (population level, mechanism-agnostic). These count
# byte values wherever they sit, so operands inflate them slightly; the
# replication assay on exemplars gives the executed truth. Cheap enough to run
# at every sample.
_BLOCK_COPY_2ND = np.array([0xA0, 0xA8, 0xB0, 0xB8], dtype=np.uint8)  # LDI LDD LDIR LDDR after ED
_PUSH = np.array([0xC5, 0xD5, 0xE5, 0xF5], dtype=np.uint8)
_LD_HL_W = np.array([0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x77, 0x36], dtype=np.uint8)  # LD (HL),r / LD (HL),n


def census(soup: np.ndarray) -> dict:
    n = soup.shape[0]
    a, b = soup[:, :-1], soup[:, 1:]
    ldir_pair = (a == 0xED) & np.isin(b, _BLOCK_COPY_2ND)
    push_cnt = np.isin(soup, _PUSH).sum(axis=1)
    out = {
        "c_blockcopy": float(ldir_pair.any(axis=1).mean()),       # cells containing ED A0/A8/B0/B8
        "c_push2": float((push_cnt >= 2).mean()),                 # cells with >= 2 PUSH bytes
        "c_ex_sp": float((soup == 0xE3).any(axis=1).mean()),      # EX (SP),HL
        "c_rst": float((soup == 0xFF).any(axis=1).mean()),        # RST 38 (the smear maker)
        "c_ld_hl_w": float(np.isin(soup, _LD_HL_W).any(axis=1).mean()),
        "c_zero8": float(((soup == 0).sum(axis=1) >= min(8, soup.shape[1])).mean()),  # NOP-flooded cells
        "c_cb_hl": float(((a == 0xCB) & ((b & 7) == 6)).any(axis=1).mean()),  # CB-page ops on (HL)
    }
    # most common 4-grams across the soup (sliding windows), as hex
    if soup.shape[1] >= 4:
        w = (soup[:, :-3].astype(np.uint32) | (soup[:, 1:-2].astype(np.uint32) << 8)
             | (soup[:, 2:-1].astype(np.uint32) << 16) | (soup[:, 3:].astype(np.uint32) << 24)).reshape(-1)
        vals, cnts = np.unique(w, return_counts=True)
        top = np.argsort(-cnts)[:5]
        total = w.size
        out["top_4grams"] = [
            {"gram": " ".join(f"{(int(v) >> (8 * i)) & 0xFF:02x}" for i in range(4)), "share": float(c / total)}
            for v, c in zip(vals[top], cnts[top])
        ]
    return out


def byte_entropy(counts: np.ndarray) -> float:
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def high_order_entropy(soup: np.ndarray, quality: int = 5) -> dict:
    """H0 (bits/byte) minus brotli bits/byte. Positive = structure beyond byte frequencies."""
    data = soup.reshape(-1)
    counts = np.bincount(data, minlength=256)
    h0 = byte_entropy(counts)
    comp = brotli.compress(data.tobytes(), quality=quality)
    bpb = len(comp) * 8.0 / data.size
    return {"H0": h0, "brotli_bpb": bpb, "hoe": h0 - bpb}


def minimal_period(tape: bytes | np.ndarray) -> int:
    """Smallest p such that tape[i] == tape[i mod p] for all i (len(tape) if aperiodic).
    A soup whose dominant tapes have small periods is tiled by a short motif."""
    b = np.frombuffer(bytes(tape), dtype=np.uint8) if not isinstance(tape, np.ndarray) else tape
    n = b.size
    for p in range(1, n):
        if np.array_equal(b[p:], b[:-p]):
            return p
    return n


def exemplars(soup: np.ndarray, hashes: np.ndarray, top_hashes: list[int]) -> list[dict]:
    out = []
    for h in top_hashes:
        idx = int(np.argmax(hashes == np.uint32(h)))
        tape = soup[idx].tobytes()
        out.append({"hash": int(h), "tape": tape.hex(" "), "mechanisms": mechanisms(tape)})
    return out

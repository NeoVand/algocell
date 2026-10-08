"""Well-mixed pairing (Stage H control). Needs the local GPU; ≈ 20 s.

(a) documents the lattice pairing every stage so far ran under: on the square grid every active pair is at
    lattice Manhattan distance 1 (x = idx % 160, y = idx // 160);
(b) the mixed variant draws the partner uniformly from the whole soup: the distance-1 fraction is < 1%, the
    x and y marginals of j pass a chi-square uniformity test (16 bins, p > 0.001; statistic and p computed here,
    scipy is not installed) and the active-pair count is within 25% of the square case;
(c) a 300-step mixed run records grid and shader, and tar forms as in the square soup: the zero-byte fraction
    rises above 0.2 by step 300 (it does so at step 8, as on the lattice). What follows differs and is printed, not
    asserted: under mixing the flood is displaced within ≈ 100 steps by a species sweeping the whole population;
(d) the derived shader differs from the square shader only in the neighbour-selection block;
(e) the Stage H condition file: 10 worlds, seeds 3001–3010, every other field identical to Stage G L = 16.
Run with `-s` to see the measured numbers.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np

from algocell_exp.gen_mixed_shader import REPLACEMENT, SQUARE_BLOCK, mixed_source
from algocell_exp.run import run
from algocell_exp.soup import SHADER_DIR, Soup

ROOT = Path(__file__).resolve().parents[1]
W, H, L = 160, 125, 16
STAGE_G = dict(width=W, height=H, tape_length=L, pair_count=8192, z80_steps=128, noise_exp=4)   # Stage G L = 16 settings
SEED = 2001


# ── chi-square without scipy: survival function via the regularized upper incomplete gamma Q(df/2, x/2) ──

def chi2_sf(x: float, df: int) -> float:
    """P(X > x) for X ~ chi-square(df) = Q(df/2, x/2); series for P below a+1, continued fraction (modified Lentz) above."""
    a, z = df / 2.0, x / 2.0
    if z <= 0:
        return 1.0
    lg = math.lgamma(a)
    if z < a + 1:
        ap, term, total = a, 1.0 / a, 1.0 / a
        for _ in range(10_000):
            ap += 1
            term *= z / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return 1.0 - total * math.exp(-z + a * math.log(z) - lg)
    tiny = 1e-300
    b = z + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return math.exp(-z + a * math.log(z) - lg) * h


def chi2_uniform(values: np.ndarray, n_values: int, bins: int = 16) -> tuple[float, float, np.ndarray]:
    """Chi-square statistic of integers in [0, n_values) against uniformity, binned into `bins` integer-edged bins
    (expected count ∝ integers per bin, so unequal bins such as 125 rows into 16 are handled exactly). Returns (statistic, p, counts)."""
    values = np.asarray(values, dtype=np.int64)
    assert values.min() >= 0 and values.max() < n_values
    edges = np.array([math.ceil(b * n_values / bins) for b in range(bins + 1)])
    counts = np.histogram(values, bins=edges)[0]
    expected = len(values) * np.diff(edges) / n_values
    stat = float(((counts - expected) ** 2 / expected).sum())
    return stat, chi2_sf(stat, bins - 1), counts


def test_chi2_sf_matches_tabulated_critical_values():
    assert abs(chi2_sf(37.697, 15) - 0.001) < 2e-5       # chi-square(15) 99.9th percentile
    assert abs(chi2_sf(24.996, 15) - 0.05) < 2e-4        # 95th percentile
    assert abs(chi2_sf(3.841, 1) - 0.05) < 2e-4
    assert abs(chi2_sf(15.0, 15) - 0.4514) < 1e-3        # x = df, both branches
    assert abs(chi2_sf(1e-6, 15) - 1.0) < 1e-6 and chi2_sf(200.0, 15) < 1e-30


# ── pair statistics after one step ──

def _one_step(grid: str, seed: int = SEED):
    soup = Soup(grid=grid, seed=seed, **STAGE_G)
    soup.step(1)
    pairs, active = soup.read_pairs()
    assert pairs.shape == (8192, 2) and active.shape == (8192,)
    return pairs.astype(np.int64), active


def _manhattan(pairs: np.ndarray) -> np.ndarray:
    i, j = pairs[:, 0], pairs[:, 1]
    return np.abs(i % W - j % W) + np.abs(i // W - j // W)


def test_square_pairs_are_lattice_neighbours():
    pairs, active = _one_step("square")
    n_active = int(active.sum())
    d = _manhattan(pairs[active])
    # Every stage so far: i random, j one of its four lattice neighbours (edges reflected), so never a self-pair.
    assert 3000 < n_active < 6000, n_active                      # the parallel collision claim leaves ≈ 48–56% of 8192 draws
    assert (d == 1).all(), np.unique(d, return_counts=True)
    assert (pairs[:, 0] != pairs[:, 1]).all()
    d_all = _manhattan(pairs)
    assert (d_all == 1).all()                                     # inactive draws are lattice neighbours too
    print(f"\n[square] active pairs {n_active}/8192 ({n_active / 8192:.3f}); Manhattan distance == 1 in {int((d == 1).sum())}/{n_active} active pairs")


def test_mixed_pairs_are_uniform_over_the_soup():
    pairs_sq, active_sq = _one_step("square")
    pairs, active = _one_step("mixed")
    n_sq, n_mx = int(active_sq.sum()), int(active.sum())
    act = pairs[active]
    d = _manhattan(act)
    frac1 = float((d == 1).mean())
    frac1_expected = 4 / (W * H)                                  # a uniform partner is one of the ≤ 4 neighbours with probability ≈ 4/20000
    assert frac1 < 0.01, frac1
    assert (act[:, 0] != act[:, 1]).all()                         # j == i stays inactive
    j = act[:, 1]
    stat_x, p_x, cx = chi2_uniform(j % W, W)
    stat_y, p_y, cy = chi2_uniform(j // W, H)
    assert p_x > 0.001, (stat_x, p_x, cx.tolist())
    assert p_y > 0.001, (stat_y, p_y, cy.tolist())
    # the drawn (not only the active) partners are uniform as well
    stat_xa, p_xa, _ = chi2_uniform(pairs[:, 1] % W, W)
    stat_ya, p_ya, _ = chi2_uniform(pairs[:, 1] // W, H)
    assert p_xa > 0.001 and p_ya > 0.001, (stat_xa, p_xa, stat_ya, p_ya)
    assert abs(n_mx - n_sq) / n_sq < 0.25, (n_mx, n_sq)
    print(f"\n[mixed] active pairs {n_mx}/8192 ({n_mx / 8192:.3f}) vs square {n_sq} ({n_mx / n_sq - 1:+.1%}); "
          f"distance-1 fraction {frac1:.5f} ({int((d == 1).sum())}/{n_mx}; uniform expectation {frac1_expected:.5f}); "
          f"median distance {np.median(d):.0f}; chi2 x: {stat_x:.2f} p = {p_x:.3f}, y: {stat_y:.2f} p = {p_y:.3f} (df 15; active); "
          f"all draws x: {stat_xa:.2f} p = {p_xa:.3f}, y: {stat_ya:.2f} p = {p_ya:.3f}")


# ── smoke run: one 300-step mixed run at the Stage G settings ──

def test_mixed_smoke_run_records_grid_and_forms_tar():
    buf = io.StringIO()
    s = run(grid="mixed", tape=L, width=W, height=H, seed=SEED, pairs=8192, z80_steps=128, noise_exp=4,
            horizon=300, sample_every=50, stop_share=-1, sample_steps=[1, 2, 3, 5, 8, 13, 21, 34], quiet=True, out=buf)
    recs = [json.loads(l) for l in buf.getvalue().splitlines()]
    cond, samples = recs[0], [r for r in recs if r["kind"] == "sample"]
    zf = {r["step"]: r["zero_frac"] for r in samples}
    assert cond["kind"] == "condition" and cond["grid"] == "mixed" and cond["provenance"]["shader_file"] == "sim_mixed_L16.wgsl"
    assert s["grid"] == "mixed" and s["steps_run"] == 300 and s["provenance"]["shader_file"] == "sim_mixed_L16.wgsl"
    assert [r["step"] for r in samples] == [1, 2, 3, 5, 8, 13, 21, 34, 50, 100, 150, 200, 250, 300]
    # Tar forms as in the square soup: the zero-byte fraction rises from the random soup's 1/256 above 0.2 within the first
    # steps (lattice: 0.22 at step 8, 0.34 at 50; mixed, 2026-10-08: 0.23 at 8, 0.36 at 50). What happens next differs and
    # is NOT asserted: in the well-mixed soup the flood is displaced within ≈ 100 steps by a species that sweeps the whole
    # population (zero_frac at step 300 was 0.09, 0.11 and 0.19 in three same-seed runs; GPU non-determinism moves the
    # sweep), whereas the lattice soup keeps ≈ 0.37 to step 1000.
    assert zf[1] > 1 / 256 and max(zf.values()) > 0.2, zf
    assert all(3000 < r["active_pairs"] < 6000 for r in samples)
    top = {r["step"]: (r["top_share"], r["q_share"]) for r in samples}
    print("\n[mixed smoke] zero_frac by step: " + ", ".join(f"{t}: {zf[t]:.3f}" for t in sorted(zf))
          + "; top/q share: " + ", ".join(f"{t}: {top[t][0]:.3f}/{top[t][1]:.3f}" for t in (50, 100, 200, 300))
          + f"; active pairs at 300: {samples[-1]['active_pairs']}; wall {s['wall_s']} s for 300 steps + 14 samples")


# ── derived shader ──

def test_mixed_shader_differs_only_in_the_neighbour_block():
    sq = (SHADER_DIR / "sim_square_L16.wgsl").read_text()
    mx = (SHADER_DIR / "sim_mixed_L16.wgsl").read_text()
    assert sq.count(SQUARE_BLOCK) == 1 and mx == sq.replace(SQUARE_BLOCK, REPLACEMENT) == mixed_source(sq)
    assert REPLACEMENT.strip() == "let j = rand_bounded(w * h);"
    assert "Topology-specific" not in mx and mx.count("let j = rand_bounded(w * h);") == 1
    soup = Soup(grid="mixed", seed=1, **STAGE_G)
    assert soup.shader_file.name == "sim_mixed_L16.wgsl" and soup.grid == "mixed"
    assert Soup(grid="square", seed=1, **STAGE_G).shader_file.name == "sim_square_L16.wgsl"


# ── Stage H condition file ──

def test_stage_h_conditions_match_stage_g_L16():
    from algocell_exp.batch import run_stem
    from make_conds import ABLATIONS, STAGES, ablation_of, stage_h

    text = open(ROOT / "conds" / "stageH.json").read()
    conds = json.loads(text)
    assert json.dumps(stage_h(), indent=0) == text and STAGES["stageH"] is stage_h, "condition file differs from make_conds output"
    assert len(conds) == 10 and [c["seed"] for c in conds] == list(range(3001, 3011))
    for c in conds:
        assert c["label"] == "mixed@closure" and c["grid"] == "mixed" and c["tape"] == 16 and c["z80_steps"] == 128 and c["noise_exp"] == 4
        assert c["horizon"] == 300_000 and c["stop_share"] == -1 and c["suppress"] == ""
    g16 = [c for c in json.load(open(ROOT / "conds" / "stageG.json")) if c["tape"] == 16]
    assert len(g16) == 20
    ref = {k: v for k, v in g16[0].items() if k not in ("label", "seed")}
    for c in conds:
        assert {k: v for k, v in c.items() if k not in ("label", "seed", "grid")} == ref   # sample schedule, snapshots, random tapes, census, stop_share …
    stems = [run_stem(c) for c in conds]
    assert len(set(stems)) == 10 and stems[0] == "mixed@closure_L16_st128_k4_s3001"
    assert ablation_of("mixed@closure") in ABLATIONS and ABLATIONS["mixed"] == []   # preflight / zoo resolve the label; nothing suppressed
    assert (SHADER_DIR / "sim_mixed_L16.wgsl").exists() and (SHADER_DIR / "z80_test_L16.wgsl").exists()

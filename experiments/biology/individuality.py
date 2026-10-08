"""Pre-registered analysis B — individuality as zero information inflow (BIOLOGY_PREREG.md, section B, 2026-10-08).

    python biology/individuality.py [--out results/biology/individuality] [--n-partners 256] [--skip-bff]

Claim under test (Krakauer et al. 2020; Lemma 7 of THEOREMS.md). With the organism X fixed and the offspring a deterministic
function O = F(X, E) of the random context E, the information flowing from the environment into the offspring is
I(O; E | X) = H(O | X) − H(O | X, E) = H(O | X), because H(O | X, E) = 0 under determinism. It is estimated by intervention:
every first and final replicator tape of the 80 Stage G worlds (Z80) and of the 64 life-producing BFF soups is run as
program A against 256 uniformly random partners (the partner distribution of both culture tests), one encounter each
(Z80: 128 instructions through `algocell_exp.assay.execute_pairs`, the culture-test kernel; BFF: the soup's own executor
flags and steps through `micro.bff.BFF`), and the plug-in entropy of the resulting partner halves is taken.

Measures per replicator (fixed in the pre-registration):
  H_bits        plug-in Shannon entropy (bits) of the empirical distribution of the 256 offspring byte strings (max 8 bits)
  H_class_bits  the same over three classes by best-cyclic-shift similarity of the offspring to X:
                faithful (≥ 0.95), partial (≥ 0.75), other
  p_modal       share of the most common offspring string
  H_self_bits   the plug-in entropy of the organism's own half afterwards (self-damage)
  copied_frac   fraction of offspring with similarity ≥ 0.75 (cross-check against `copied`/`copies` of the input tables)
  damaged_frac  fraction of encounters in which the organism lost ≥ 25% of its bytes (as closure.partner_test / bff.assay)
Diagnostics of where the inflow sits (not pre-registered, descriptive): n_var_pos = offspring positions that take more than
one value across the partners; n_partner_pos = positions still holding the partner's original byte in ≥ 50% of encounters;
partner_byte_share = share of offspring bytes equal to the partner's original byte; BFF only: entered_frac (pointer entered
the partner), exec_steps_mean and halted_frac (encounter ended before the step budget) from the kernel's per-pair record.
Similarity: Z80 = best cyclic shift, forwards (`algocell_exp.assay._best_shift_rows`, as in the Z80 culture test);
BFF = best cyclic shift forwards or reversed (`micro.bff_soup.batch_best_similarity`, as in `micro.bff.assay`, because BFF
copiers often write the copy reversed). "Has a loop instruction": Z80 = `has_cf | has_block` of stage_g_runs.csv (jump,
relative jump, DJNZ, CALL/RET, RST, or LDIR/LDDR-type block repeat); BFF = both `[` and `]` present under the soup's
alphabet (`micro.bff.has_loop`). Partners: numpy default_rng seeded per replicator with [20261008, machine, group, seed, which].

Outputs (--out): per_replicator.csv, summary.csv, predictions.csv, offspring_hist.json, fig_z80_slope.{pdf,svg,png},
fig_bff_slope.{pdf,svg,png}, NUMBERS_INDIVIDUALITY.md (generated). FINDINGS.md is written by hand from the numbers file.
No soup is run; the only GPU use is one pair-executor launch per tape length (Z80) and per flag set (BFF).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXP not in sys.path:
    sys.path.insert(0, EXP)

import figstyle as fs  # noqa: E402
from algocell_exp.assay import _best_shift_rows, execute_pairs  # noqa: E402
from micro.bff import BFF, LIT, TAPE as BFF_TAPE, ascii_map, density_map, has_loop  # noqa: E402

try:
    from micro.bff_soup import batch_best_similarity as _bff_batch_similarity  # fwd | rev, as the BFF culture test
except ImportError:  # pragma: no cover — brotli missing; same computation as micro.bff_soup.batch_best_similarity
    _bff_batch_similarity = None

SEED0 = 20261008
Z80_STEPS = 128
N_PARTNERS = 256
Z80_CSV = os.path.join(EXP, "results", "stageG", "stageG", "stage_g_runs.csv")
BFF_CSV = os.path.join(EXP, "results", "bff", "runs.csv")
BFF_RUNS = os.path.join(EXP, "runs", "bff_modal", "bff")
DEFAULT_OUT = os.path.join(EXP, "results", "biology", "individuality")
VARIANTS = ["std", "wrap", "lit", "wraplit", "wraplitnh"]  # fixed order; runs.csv calls "lit" "stdlit"
VARIANT_FLAGS = {  # variant → (ip_wrap, literal, nohalt), as micro.bff_analysis.load_run builds the name
    "std": (False, False, False), "wrap": (True, False, False), "lit": (False, True, False),
    "wraplit": (True, True, False), "wraplitnh": (True, True, True)}
LIT_VARIANTS = ["lit", "wraplit", "wraplitnh"]
# Pre-registered thresholds: name → (fraction needed for "met", fraction below which the prediction is killed)
PRED = {"Q1": (0.90, 0.60), "Q2": (0.90, 0.60), "Q3": (0.75, 0.50), "Q4a": (0.90, 0.60), "Q4b": (0.90, 0.60)}
GREY, GREY_DARK, ACCENT = "#BDBDBD", "#737373", fs.CONCEPT["closed"]  # grey plus one accent colour (vermilion)


# ----------------------------------------------------------------------------------------------------------------------
# measures
def hb(s: str) -> np.ndarray:
    """Hex string (with or without spaces) → uint8 array."""
    return np.frombuffer(bytes.fromhex(s.replace(" ", "")), dtype=np.uint8).copy()


def plugin_entropy(rows: np.ndarray) -> tuple[float, int, float, np.ndarray]:
    """Plug-in Shannon entropy (bits) of the empirical distribution of the byte strings in `rows` (N, L):
    (H, number of distinct strings, modal share, counts sorted descending). Each row is hashed exactly (void view)."""
    rows = np.ascontiguousarray(rows, dtype=np.uint8)
    v = rows.view(np.dtype((np.void, rows.shape[1]))).ravel()
    _, counts = np.unique(v, return_counts=True)
    p = counts / counts.sum()
    H = float(max(0.0, -(p * np.log2(p)).sum()))  # max(0, ·) clears the negative zero of a single class
    return H, int(len(counts)), float(p.max()), np.sort(counts)[::-1]


def class_entropy(sim: np.ndarray) -> float:
    """Plug-in entropy (bits) over three outcome classes: faithful (≥ 0.95), partial (≥ 0.75), other."""
    cls = np.where(sim >= 0.95, 2, np.where(sim >= 0.75, 1, 0))
    counts = np.bincount(cls, minlength=3)
    p = counts[counts > 0] / counts.sum()
    return float(max(0.0, -(p * np.log2(p)).sum()))


def z80_similarity(O: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Best cyclic-shift similarity (forwards) of each offspring row to X — the Z80 culture test's measure."""
    return _best_shift_rows(O, np.repeat(X[None], O.shape[0], axis=0))[0]


def bff_similarity(O: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Best cyclic-shift similarity, forwards or reversed — the BFF culture test's measure (micro.bff.similarity)."""
    A = np.repeat(X[None], O.shape[0], axis=0)
    if _bff_batch_similarity is not None:
        return _bff_batch_similarity(A, O)[0]
    best = np.zeros(O.shape[0])
    for src in (A, A[:, ::-1]):
        for s in range(X.size):
            best = np.maximum(best, (O == np.roll(src, s, axis=1)).mean(1))
    return best


def measure(X: np.ndarray, Xp: np.ndarray, O: np.ndarray, R: np.ndarray, sim: np.ndarray) -> tuple[dict, np.ndarray]:
    """All per-replicator measures from the organism X, its own half afterwards Xp (N, L), the offspring O (N, L), the
    partners before the encounter R (N, L) and the offspring similarities; also returns the sorted offspring counts for
    offspring_hist.json. Where the inflow sits: n_var_pos = offspring positions that take more than one value across the
    partners; n_partner_pos = positions that still hold the partner's original byte in ≥ 50% of encounters (unwritten
    positions); partner_byte_share = fraction of all offspring bytes equal to the partner's original byte."""
    H, n_unique, p_modal, counts = plugin_entropy(O)
    H_self, n_unique_self, _, _ = plugin_entropy(Xp)
    keep = (O == R).mean(0)
    return {
        "H_bits": H, "H_class_bits": class_entropy(sim), "p_modal": p_modal, "H_self_bits": H_self,
        "copied_frac": float((sim >= 0.75).mean()), "damaged_frac": float(((Xp != X[None]).mean(1) >= 0.25).mean()),
        "faithful_frac": float((sim >= 0.95).mean()), "sim_mean": float(sim.mean()),
        "self_intact_frac": float((Xp == X[None]).all(1).mean()),
        "n_unique_offspring": n_unique, "n_unique_self": n_unique_self,
        "n_var_pos": int((O != O[0][None, :]).any(0).sum()), "n_partner_pos": int((keep >= 0.5).sum()), "partner_byte_share": float(keep.mean()),
    }, counts


def verdict(frac: float, name: str) -> str:
    met, kill = PRED[name]
    if np.isnan(frac):
        return "no data"
    return "met" if frac >= met else ("KILL" if frac < kill else "between (neither met nor killed)")


# ----------------------------------------------------------------------------------------------------------------------
# Z80: Stage G first and final tapes
def run_z80(n_partners: int, hist: dict) -> list[dict]:
    G = pd.read_csv(Z80_CSV)
    rows = []
    for L, g in G.groupby("L"):
        L = int(L)
        items = []
        for _, r in g.iterrows():
            for wi, which in enumerate(("first", "final")):
                X = hb(r[f"{which}_tape"])
                assert X.size == L, (r["seed"], which, X.size, L)
                seed = [SEED0, 0, L, int(r["seed"]), wi]
                R = np.random.default_rng(seed).integers(0, 256, size=(n_partners, L), dtype=np.uint8)
                items.append((r, which, X, R, seed))
        pairs = np.concatenate([np.concatenate([np.repeat(X[None], n_partners, 0), R], axis=1) for _, _, X, R, _ in items], axis=0)
        res = execute_pairs(pairs, L, Z80_STEPS)  # one launch per tape length
        for i, (r, which, X, R, seed) in enumerate(items):
            blk = res[i * n_partners:(i + 1) * n_partners]
            Xp, O = blk[:, :L], blk[:, L:]
            sim = z80_similarity(O, X)
            m, counts = measure(X, Xp, O, R, sim)
            has_cf, has_block = bool(r[f"{which}_has_cf"]), bool(r[f"{which}_has_block"])
            world = f"L{L}_s{int(r['seed'])}"
            rows.append({
                "machine": "z80", "world": world, "group": L, "which": which, "has_loop": has_cf or has_block, **m,
                "entered_frac": np.nan, "exec_steps_mean": np.nan, "halted_frac": np.nan,
                "csv_copied": float(r[f"{which}_copied"]), "csv_damaged": float(r[f"{which}_damaged"]), "csv_has_loop": has_cf or has_block,
                "loop_kind": "+".join(k for k, v in (("cf:" + str(r[f"{which}_cf"]), has_cf), ("block:" + str(r[f"{which}_block"]), has_block)) if v) or "-",
                "sim_kind": "fwd", "n_partners": n_partners, "steps": Z80_STEPS, "partner_seed": json.dumps(seed),
                "tape_hex": r[f"{which}_tape"].replace(" ", ""),
            })
            hist[f"z80|{world}|{which}"] = counts.tolist()
        print(f"z80 L = {L}: {len(items)} replicators × {n_partners} partners done", flush=True)
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# BFF: first and final tapes of every life-producing soup, under each soup's own executor flags
def bff_flags(run: str) -> dict:
    d = os.path.join(BFF_RUNS, run)
    for name in ("cond.json", "summary.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            c = json.load(open(p))
            return {"ip_wrap": bool(c.get("ip_wrap", False)), "literal": bool(c.get("literal", False)), "nohalt": bool(c.get("nohalt", False)),
                    "density": int(c.get("density", 1)), "steps": int(c.get("steps", 1 << 13)), "tape": int(c.get("tape", BFF_TAPE)), "source": p}
    raise FileNotFoundError(f"no cond.json/summary.json for BFF run {run} under {BFF_RUNS}")


def bff_alphabet(flags: dict) -> np.ndarray | None:
    """None → the executor's default ASCII map (with P when literal); density > 1 → density_map(seed=0) as the soup used."""
    if flags["density"] <= 1:
        return None
    amap = density_map(flags["density"], seed=0)
    if flags["literal"]:
        amap = amap.copy()
        amap[ord(LIT)] = 11
    return amap


def run_bff(n_partners: int, hist: dict, notes: list[str]) -> list[dict]:
    B = pd.read_csv(BFF_CSV)
    B["variant"] = B["variant"].replace({"stdlit": "lit"})
    live = B[B["t_top"].notna()].copy()
    notes.append(f"BFF soups in runs.csv: {len(B)}; life-producing (t_top not null): {len(live)} "
                 f"({', '.join(f'{v} {int((live.variant == v).sum())}' for v in VARIANTS)}); the {int(B['t_top'].isna().sum())} soups without a "
                 f"replicator have no first tape and are not part of the analysis.")
    groups: dict[tuple, list] = {}
    for _, r in live.iterrows():
        flags = bff_flags(r["run"])
        expect = VARIANT_FLAGS[r["variant"]]
        got = (flags["ip_wrap"], flags["literal"], flags["nohalt"])
        if got != expect:
            notes.append(f"SKIPPED {r['run']}: executor flags {got} (ip_wrap, literal, nohalt) from {flags['source']} do not match variant '{r['variant']}' {expect}.")
            continue
        if flags["tape"] != BFF_TAPE:
            notes.append(f"SKIPPED {r['run']}: tape length {flags['tape']} ≠ {BFF_TAPE}.")
            continue
        key = (flags["ip_wrap"], flags["literal"], flags["nohalt"], flags["density"], flags["steps"])
        for wi, which in enumerate(("first", "final")):
            hx = r[f"{which}_tape"]
            if not isinstance(hx, str) or len(hx) != 2 * BFF_TAPE:
                notes.append(f"SKIPPED {r['run']} {which}: tape missing or not {BFF_TAPE} bytes in runs.csv.")
                continue
            X = hb(hx)
            seed = [SEED0, 1, VARIANTS.index(r["variant"]), int(r["seed"]), wi]
            R = np.random.default_rng(seed).integers(0, 256, size=(n_partners, BFF_TAPE), dtype=np.uint8)
            groups.setdefault(key, []).append((r, which, X, R, seed, flags))
    rows = []
    for key, items in groups.items():
        ip_wrap, literal, nohalt, density, steps = key
        flags0 = items[0][5]
        amap = bff_alphabet(flags0)
        amap_eff = ascii_map(literal) if amap is None else amap
        pairs = np.concatenate([np.concatenate([np.repeat(X[None], n_partners, 0), R], axis=1) for _, _, X, R, _, _ in items], axis=0)
        bff = BFF(max_pairs=pairs.shape[0], steps=steps, ip_wrap=ip_wrap, alphabet=amap, literal=literal, nohalt=nohalt)
        mem, out = bff.execute(pairs)  # one launch per flag set
        for i, (r, which, X, R, seed, flags) in enumerate(items):
            sl = slice(i * n_partners, (i + 1) * n_partners)
            Xp, O = mem[sl, :BFF_TAPE], mem[sl, BFF_TAPE:]
            sim = bff_similarity(O, X)
            m, counts = measure(X, Xp, O, R, sim)
            loop = has_loop(X, amap_eff)
            rows.append({
                "machine": "bff", "world": r["run"], "group": r["variant"], "which": which, "has_loop": loop, **m,
                "entered_frac": float(out[sl, 1].mean()), "exec_steps_mean": float(out[sl, 0].mean()), "halted_frac": float((out[sl, 0] < steps).mean()),
                "csv_copied": float(r[f"{which}_copies"]), "csv_damaged": float(r[f"{which}_self_damage"]), "csv_has_loop": bool(r[f"{which}_loop"]),
                "loop_kind": "[ ]" if loop else "-",
                "sim_kind": "fwd|rev", "n_partners": n_partners, "steps": steps, "partner_seed": json.dumps(seed),
                "tape_hex": r[f"{which}_tape"],
            })
            hist[f"bff|{r['run']}|{which}"] = counts.tolist()
        print(f"bff flags ip_wrap={ip_wrap} literal={literal} nohalt={nohalt} density={density} steps={steps}: {len(items)} replicators done", flush=True)
    return rows


# ----------------------------------------------------------------------------------------------------------------------
# summaries, predictions
def q(x: pd.Series, p: float) -> float:
    return float(x.quantile(p)) if len(x) else float("nan")


def summarise(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for machine in ("z80", "bff"):
        M = P[P["machine"] == machine]
        if M.empty:
            continue
        groups = ["all"] + ([int(v) for v in sorted(M["group"].astype(int).unique())] if machine == "z80" else [v for v in VARIANTS if v in set(M["group"])])
        for grp in groups:
            d = M if grp == "all" else M[M["group"].astype(str) == str(grp)]
            for which in ("first", "final"):
                x = d[d["which"] == which]
                if x.empty:
                    continue
                loop = x["has_loop"].astype(bool)
                rows.append({
                    "machine": machine, "group": grp, "which": which, "n": len(x),
                    "H_median": float(x["H_bits"].median()), "H_q1": q(x["H_bits"], 0.25), "H_q3": q(x["H_bits"], 0.75),
                    "H_class_median": float(x["H_class_bits"].median()), "p_modal_median": float(x["p_modal"].median()),
                    "H_self_median": float(x["H_self_bits"].median()), "copied_median": float(x["copied_frac"].median()),
                    "damaged_median": float(x["damaged_frac"].median()),
                    "frac_H_gt_1bit": float((x["H_bits"] > 1).mean()), "frac_H_lt_0p5bit": float((x["H_bits"] < 0.5).mean()),
                    "n_loop": int(loop.sum()), "frac_loop_H_lt_0p5bit": float((x["H_bits"][loop] < 0.5).mean()) if loop.any() else float("nan"),
                    "frac_loop": float(loop.mean()),
                    "drop_median": float("nan"), "drop_q1": float("nan"), "drop_q3": float("nan"), "frac_drop_ge_2bits": float("nan"),
                })
            f = d[d["which"] == "first"].set_index("world")["H_bits"]
            l = d[d["which"] == "final"].set_index("world")["H_bits"]
            drop = (f - l.reindex(f.index)).dropna()
            if len(drop):
                rows.append({
                    "machine": machine, "group": grp, "which": "first-final", "n": len(drop),
                    "H_median": float("nan"), "H_q1": float("nan"), "H_q3": float("nan"), "H_class_median": float("nan"), "p_modal_median": float("nan"),
                    "H_self_median": float("nan"), "copied_median": float("nan"), "damaged_median": float("nan"),
                    "frac_H_gt_1bit": float("nan"), "frac_H_lt_0p5bit": float("nan"), "n_loop": np.nan, "frac_loop_H_lt_0p5bit": float("nan"), "frac_loop": float("nan"),
                    "drop_median": float(drop.median()), "drop_q1": q(drop, 0.25), "drop_q3": q(drop, 0.75), "frac_drop_ge_2bits": float((drop >= 2).mean()),
                })
    return pd.DataFrame(rows)


def predictions(P: pd.DataFrame) -> pd.DataFrame:
    def row(name, machine, population, statement, n, count):
        frac = count / n if n else float("nan")
        return {"prediction": name, "machine": machine, "population": population, "statement": statement, "n": int(n), "count": int(count),
                "fraction": frac, "threshold_met": PRED[name][0], "threshold_kill": PRED[name][1], "outcome": verdict(frac, name)}
    out = []
    Z = P[P["machine"] == "z80"]
    if not Z.empty:
        f = Z[Z["which"] == "first"].set_index("world")
        l = Z[Z["which"] == "final"].set_index("world")
        out.append(row("Q1", "z80", "all Stage G worlds", "H(O | X_first) > 1 bit", len(f), (f["H_bits"] > 1).sum()))
        lw = l[l["has_loop"].astype(bool)]
        out.append(row("Q2", "z80", "worlds whose final dominant carries a loop instruction", "H(O | X_final) < 0.5 bit", len(lw), (lw["H_bits"] < 0.5).sum()))
        drop = (f["H_bits"] - l["H_bits"].reindex(f.index)).dropna()
        out.append(row("Q3", "z80", "all Stage G worlds", "H(O | X_first) − H(O | X_final) ≥ 2 bits", len(drop), (drop >= 2).sum()))
    Bf = P[(P["machine"] == "bff") & (P["which"] == "first")]
    if not Bf.empty:
        std = Bf[Bf["group"] == "std"]
        out.append(row("Q4a", "bff", "life-producing soups of the published variant (std)", "H(O | X_first) < 0.5 bit", len(std), (std["H_bits"] < 0.5).sum()))
        lit = Bf[Bf["group"].isin(LIT_VARIANTS)]
        out.append(row("Q4b", "bff", "life-producing soups of the literal-push variants (lit, wraplit, wraplitnh)", "H(O | X_first) > 1 bit", len(lit), (lit["H_bits"] > 1).sum()))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------------------------------
# figures
def slope_figure(P: pd.DataFrame, machine: str, groups: list, titles: dict, path: str, loop_label: str, width: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fs.setup()
    plt.rcParams.update({"xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6, "axes.titlesize": 6.5, "axes.labelsize": 6})  # type ≥ 6 pt
    M = P[P["machine"] == machine]
    fig, axes = plt.subplots(1, len(groups), figsize=(width, 2.25), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, grp in zip(axes, groups):
        d = M[M["group"].astype(str) == str(grp)]
        worlds = sorted(d["world"].unique())
        for w in worlds:
            f = d[(d["world"] == w) & (d["which"] == "first")]
            l = d[(d["world"] == w) & (d["which"] == "final")]
            if f.empty or l.empty:
                continue
            f, l = f.iloc[0], l.iloc[0]
            j = rng.uniform(-0.07, 0.07)  # small, equal jitter at both ends keeps coincident worlds visible as a bundle
            ax.plot([j, 1 + j], [f["H_bits"], l["H_bits"]], color=GREY, lw=0.55, alpha=0.7, zorder=1, solid_capstyle="round")
            for x, rec in ((j, f), (1 + j, l)):
                if bool(rec["has_loop"]):
                    ax.plot(x, rec["H_bits"], "o", ms=2.8, mfc=ACCENT, mec=ACCENT, mew=0.5, zorder=3)
                else:
                    ax.plot(x, rec["H_bits"], "o", ms=2.8, mfc="white", mec=GREY_DARK, mew=0.6, zorder=2)
        for y in (8.0, 1.0, 0.5):
            ax.axhline(y, ls=":", lw=0.5, color=GREY, zorder=0)
        ax.set_xticks([0, 1], ["first", "final"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(-0.45, 8.75)
        ax.set_yticks([0, 2, 4, 6, 8])
        ax.set_title(f"{titles[grp]}\n(n = {len(worlds)})")
        fs.tidy(ax)
    axes[0].set_ylabel("H(O | X) (bits)")
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
    last = axes[-1]
    last.text(1.47, 8.0, "ceiling, 256 partners", fontsize=6, color=GREY_DARK, va="center", ha="left", clip_on=False)
    last.text(1.47, 1.0, "1 bit (Q1)", fontsize=6, color=GREY_DARK, va="bottom", ha="left", clip_on=False)
    last.text(1.47, 0.5, "0.5 bit (Q2)", fontsize=6, color=GREY_DARK, va="top", ha="left", clip_on=False)
    handles = [Line2D([], [], marker="o", ls="none", ms=2.8, mfc=ACCENT, mec=ACCENT, label=loop_label),
               Line2D([], [], marker="o", ls="none", ms=2.8, mfc="white", mec=GREY_DARK, label="no loop instruction")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
    fig.subplots_adjust(wspace=0.15)
    fs.save(fig, path)


# ----------------------------------------------------------------------------------------------------------------------
# numbers file
def md_table(df: pd.DataFrame, **kw) -> str:
    try:
        return df.to_markdown(index=False, **kw)
    except ImportError:  # pragma: no cover
        return "```\n" + df.to_string(index=False) + "\n```"


def write_numbers(P: pd.DataFrame, S: pd.DataFrame, V: pd.DataFrame, notes: list[str], out: str, n_partners: int) -> None:
    rel = os.path.relpath(out, EXP)
    md = ["# Individuality as zero information inflow — numbers (generated by biology/individuality.py; do not edit)\n",
          f"Every number below is computed from `{rel}/per_replicator.csv` (this analysis; columns named in each table) unless a "
          f"source is given in the heading. Inputs: `results/stageG/stageG/stage_g_runs.csv` (Z80 first/final tapes, `*_has_cf`, "
          f"`*_has_block`, `*_copied`, `*_damaged`) and `results/bff/runs.csv` with `runs/bff_modal/bff/<run>/cond.json` (BFF tapes, "
          f"`*_copies`, `*_self_damage`, `*_loop`, executor flags). Partner test: {n_partners} uniformly random partners per replicator "
          f"(numpy default_rng, seed per replicator in column `partner_seed`), organism in the first half, one encounter "
          f"(Z80: {Z80_STEPS} instructions, `algocell_exp.assay.execute_pairs`; BFF: the soup's own flags and steps, `micro.bff.BFF`). "
          f"H = plug-in entropy in bits of the {n_partners} offspring strings (maximum log2({n_partners}) = {np.log2(n_partners):.0f} bits; "
          f"a replicator whose offspring are all distinct sits at the ceiling, so H is a lower bound there). Under determinism "
          f"I(O; E | X) = H(O | X) − H(O | X, E) = H(O | X).\n"]
    for n in notes:
        md.append(f"- {n}")
    md.append("")
    # verdicts
    md.append("## Pre-registered predictions (BIOLOGY_PREREG.md section B; `predictions.csv`)\n")
    Vt = V.copy()
    Vt["fraction"] = Vt["fraction"].map(lambda v: f"{v:.3f}")
    md.append(md_table(Vt[["prediction", "machine", "population", "statement", "count", "n", "fraction", "threshold_met", "threshold_kill", "outcome"]]) + "\n")
    # summaries
    md.append("## Summary by machine and group (`summary.csv`)\n")
    St = S.copy()
    for c in ("H_median", "H_q1", "H_q3", "H_class_median", "p_modal_median", "H_self_median", "copied_median", "damaged_median",
              "frac_H_gt_1bit", "frac_H_lt_0p5bit", "frac_loop_H_lt_0p5bit", "frac_loop", "drop_median", "drop_q1", "drop_q3", "frac_drop_ge_2bits"):
        St[c] = St[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    St["n_loop"] = St["n_loop"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    md.append(md_table(St) + "\n")
    # per machine detail
    for machine, label in (("z80", "Z80 (Stage G, 80 worlds)"), ("bff", "BFF (life-producing soups)")):
        M = P[P["machine"] == machine]
        if M.empty:
            md.append(f"## {label}: no data\n")
            continue
        f = M[M["which"] == "first"].set_index("world")
        l = M[M["which"] == "final"].set_index("world").reindex(f.index)
        md.append(f"## {label}: per replicator\n")
        md.append(f"- H(O | X_first): median {f['H_bits'].median():.3f} bits (IQR {q(f['H_bits'], 0.25):.3f}–{q(f['H_bits'], 0.75):.3f}; range {f['H_bits'].min():.3f}–{f['H_bits'].max():.3f}); "
                  f"> 1 bit in {int((f['H_bits'] > 1).sum())}/{len(f)}; < 0.5 bit in {int((f['H_bits'] < 0.5).sum())}/{len(f)}; at the {np.log2(n_partners):.0f}-bit ceiling "
                  f"(all offspring distinct) in {int((f['n_unique_offspring'] == n_partners).sum())}/{len(f)}.")
        md.append(f"- H(O | X_final): median {l['H_bits'].median():.3f} bits (IQR {q(l['H_bits'], 0.25):.3f}–{q(l['H_bits'], 0.75):.3f}; range {l['H_bits'].min():.3f}–{l['H_bits'].max():.3f}); "
                  f"> 1 bit in {int((l['H_bits'] > 1).sum())}/{len(l)}; < 0.5 bit in {int((l['H_bits'] < 0.5).sum())}/{len(l)}; exactly 0 (one offspring string) in {int((l['n_unique_offspring'] == 1).sum())}/{len(l)}.")
        loop_l = l["has_loop"].astype(bool)
        md.append(f"- loop instruction present: first {int(f['has_loop'].astype(bool).sum())}/{len(f)}, final {int(loop_l.sum())}/{len(l)}; "
                  f"H(O | X_final) < 0.5 bit among loop-bearing finals {int((l['H_bits'][loop_l] < 0.5).sum())}/{int(loop_l.sum())}, "
                  f"among loop-free finals {int((l['H_bits'][~loop_l] < 0.5).sum())}/{int((~loop_l).sum())}.")
        drop = f["H_bits"] - l["H_bits"]
        md.append(f"- paired drop H_first − H_final: median {drop.median():.3f} bits (IQR {q(drop, 0.25):.3f}–{q(drop, 0.75):.3f}); ≥ 2 bits in {int((drop >= 2).sum())}/{len(drop)}; "
                  f"negative (final more partner-dependent than first) in {int((drop < 0).sum())}/{len(drop)}.")
        md.append(f"- class entropy H_class(O | X): first median {f['H_class_bits'].median():.3f}, final median {l['H_class_bits'].median():.3f}; "
                  f"p_modal: first median {f['p_modal'].median():.3f}, final median {l['p_modal'].median():.3f}; "
                  f"H(X' | X) (self): first median {f['H_self_bits'].median():.3f}, final median {l['H_self_bits'].median():.3f}; "
                  f"organism byte-intact after the encounter: first median {f['self_intact_frac'].median():.3f}, final median {l['self_intact_frac'].median():.3f}.")
        if machine == "bff":
            md.append(f"- pointer entered the partner (BFF kernel record; `entered_frac`): first median {f['entered_frac'].median():.3f}, final median {l['entered_frac'].median():.3f}; "
                      f"replicators with entered_frac = 0 and H = 0: first {int(((f['entered_frac'] == 0) & (f['H_bits'] == 0)).sum())}/{len(f)}, final {int(((l['entered_frac'] == 0) & (l['H_bits'] == 0)).sum())}/{len(l)}; "
                      f"entered_frac = 0 but H > 0: first {int(((f['entered_frac'] == 0) & (f['H_bits'] > 0)).sum())}, final {int(((l['entered_frac'] == 0) & (l['H_bits'] > 0)).sum())}.")
        # cross-check against the input tables
        tol = 0.10 if machine == "z80" else 0.15
        dc = (M["copied_frac"] - M["csv_copied"]).abs()
        dd = (M["damaged_frac"] - M["csv_damaged"]).abs()
        src = "`*_copied`/`*_damaged` of stage_g_runs.csv (256 partners, other seed)" if machine == "z80" else "`*_copies`/`*_self_damage` of runs.csv (culture test, 64 partners)"
        md.append(f"- cross-check of copied_frac against {src}: median |Δ| {dc.median():.3f}, max |Δ| {dc.max():.3f}, within {tol:.2f} in {int((dc <= tol).sum())}/{len(M)}; "
                  f"mean copied_frac {M['copied_frac'].mean():.3f} vs mean of the table {M['csv_copied'].mean():.3f}; "
                  f"damaged_frac: median |Δ| {dd.median():.3f}, max |Δ| {dd.max():.3f}.")
        bad = M[dc > tol]
        if len(bad):
            md.append("  - beyond tolerance: " + "; ".join(f"{r.world} {r.which} {r.copied_frac:.3f} vs {r.csv_copied:.3f}" for r in bad.itertuples()))
        agree = int((M["has_loop"].astype(bool) == M["csv_has_loop"].astype(bool)).sum())
        md.append(f"- has_loop agreement with the input table: {agree}/{len(M)}.")
        # where the inflow sits (columns n_var_pos, n_partner_pos, partner_byte_share, exec_steps_mean, halted_frac)
        md.append(f"- where the inflow sits — offspring positions taking more than one value across the {n_partners} partners (`n_var_pos`, of L) and positions still holding "
                  f"the partner's original byte in ≥ 50% of encounters (`n_partner_pos`): first median {f['n_var_pos'].median():.0f} / {f['n_partner_pos'].median():.0f}, "
                  f"final median {l['n_var_pos'].median():.0f} / {l['n_partner_pos'].median():.0f}; share of offspring bytes equal to the partner's original byte "
                  f"(`partner_byte_share`): first median {f['partner_byte_share'].median():.3f}, final median {l['partner_byte_share'].median():.3f}.")
        exc = l[l["has_loop"].astype(bool) & (l["H_bits"] >= 0.5)]
        if len(exc):
            md.append("- loop-bearing finals with H(O | X) ≥ 0.5 bit: " + "; ".join(
                f"{w} ({r.loop_kind}; H {r.H_bits:.2f}, mean offspring similarity {r.sim_mean:.3f}, copied {r.copied_frac:.2f}, faithful {r.faithful_frac:.2f}, "
                f"variable positions {int(r.n_var_pos)}/{len(hb(r.tape_hex))}, partner-byte positions {int(r.n_partner_pos)}, H_class {r.H_class_bits:.2f})" for w, r in exc.iterrows()) + ".")
        nl = l[~l["has_loop"].astype(bool)]
        if len(nl):
            md.append(f"- loop-free finals: {len(nl)} ({', '.join(sorted(nl.index))}); H median {nl['H_bits'].median():.3f} bits, copied median {nl['copied_frac'].median():.3f}, "
                      f"faithful median {nl['faithful_frac'].median():.3f}, H_class median {nl['H_class_bits'].median():.3f}.")
        if machine == "bff":
            pc = f[(f["entered_frac"] == 0) & (f["H_bits"] > 0)]
            if len(pc):
                md.append("- pointer-closed first replicators (entered_frac = 0) with H > 0: " + "; ".join(
                    f"{w} ({r.group}; H {r.H_bits:.2f}, p_modal {r.p_modal:.3f}, copied {r.copied_frac:.2f}, faithful {r.faithful_frac:.2f}, variable positions {int(r.n_var_pos)}, "
                    f"partner-byte positions {int(r.n_partner_pos)})" for w, r in pc.iterrows()) + ".")
            md.append(f"- encounter length (BFF kernel; `exec_steps_mean`, `halted_frac` = share of encounters that ended before the step budget): " + "; ".join(
                f"{v} first {f.loc[f['group'] == v, 'exec_steps_mean'].median():.0f} steps, halted early {f.loc[f['group'] == v, 'halted_frac'].median():.3f}" for v in VARIANTS if (f["group"] == v).any()) + ".")
        md.append("")
        if machine == "z80":
            grp_rows = []
            for L, g in M.groupby("group"):
                gf, gl = g[g["which"] == "first"].set_index("world"), g[g["which"] == "final"].set_index("world")
                gl = gl.reindex(gf.index)
                grp_rows.append({"L": int(L), "n": len(gf), "H_first_median": gf["H_bits"].median(), "H_first_>1bit": int((gf["H_bits"] > 1).sum()),
                                 "H_final_median": gl["H_bits"].median(), "final_loop": int(gl["has_loop"].astype(bool).sum()),
                                 "H_final_<0.5_among_loop": int((gl["H_bits"][gl["has_loop"].astype(bool)] < 0.5).sum()),
                                 "H_final_<0.5_among_no_loop": int((gl["H_bits"][~gl["has_loop"].astype(bool)] < 0.5).sum()),
                                 "drop_≥2bits": int(((gf["H_bits"] - gl["H_bits"]) >= 2).sum()),
                                 "copied_first_median": gf["copied_frac"].median(), "copied_final_median": gl["copied_frac"].median(),
                                 "p_modal_first_median": gf["p_modal"].median(), "H_class_first_median": gf["H_class_bits"].median(),
                                 "n_var_pos_first_median": gf["n_var_pos"].median(), "partner_byte_share_first_median": gf["partner_byte_share"].median()})
            md.append("per tape length:\n")
            md.append(md_table(pd.DataFrame(grp_rows), floatfmt=".3f") + "\n")
        else:
            grp_rows = []
            for v in VARIANTS:
                g = M[M["group"] == v]
                if g.empty:
                    continue
                gf, gl = g[g["which"] == "first"].set_index("world"), g[g["which"] == "final"].set_index("world")
                gl = gl.reindex(gf.index)
                grp_rows.append({"variant": v, "n": len(gf), "H_first_median": gf["H_bits"].median(), "H_first_<0.5bit": int((gf["H_bits"] < 0.5).sum()),
                                 "H_first_>1bit": int((gf["H_bits"] > 1).sum()), "H_final_median": gl["H_bits"].median(), "H_final_<0.5bit": int((gl["H_bits"] < 0.5).sum()),
                                 "first_loop": int(gf["has_loop"].astype(bool).sum()), "final_loop": int(gl["has_loop"].astype(bool).sum()),
                                 "entered_first_median": gf["entered_frac"].median(), "entered_final_median": gl["entered_frac"].median(),
                                 "copied_first_median": gf["copied_frac"].median(), "copied_final_median": gl["copied_frac"].median(),
                                 "p_modal_first_median": gf["p_modal"].median(), "faithful_first_median": gf["faithful_frac"].median(),
                                 "n_var_pos_first_median": gf["n_var_pos"].median(), "n_partner_pos_first_median": gf["n_partner_pos"].median(),
                                 "exec_steps_first_median": gf["exec_steps_mean"].median(), "halted_first_median": gf["halted_frac"].median()})
            md.append("per variant:\n")
            md.append(md_table(pd.DataFrame(grp_rows), floatfmt=".3f") + "\n")
        T = pd.DataFrame({
            "world": f.index, "group": f["group"].values,
            "H_first": f["H_bits"].values, "H_final": l["H_bits"].values, "loop_first": f["has_loop"].astype(bool).values, "loop_final": l["has_loop"].astype(bool).values,
            "H_class_first": f["H_class_bits"].values, "H_class_final": l["H_class_bits"].values, "p_modal_final": l["p_modal"].values,
            "H_self_first": f["H_self_bits"].values, "H_self_final": l["H_self_bits"].values,
            "copied_first": f["copied_frac"].values, "table_copied_first": f["csv_copied"].values,
            "copied_final": l["copied_frac"].values, "table_copied_final": l["csv_copied"].values,
        })
        md.append("per world (H in bits; `table_*` = the input table's partner/culture test):\n")
        md.append(md_table(T, floatfmt=".3f") + "\n")
    with open(os.path.join(out, "NUMBERS_INDIVIDUALITY.md"), "w") as fh:
        fh.write("\n".join(md))


# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-partners", type=int, default=N_PARTNERS)
    ap.add_argument("--skip-bff", action="store_true")
    ap.add_argument("--skip-z80", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hist: dict = {}
    notes: list[str] = []
    rows: list[dict] = []
    if not a.skip_z80:
        rows += run_z80(a.n_partners, hist)
    if not a.skip_bff:
        rows += run_bff(a.n_partners, hist, notes)
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(a.out, "per_replicator.csv"), index=False)
    json.dump(hist, open(os.path.join(a.out, "offspring_hist.json"), "w"))
    S = summarise(P)
    S.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    V = predictions(P)
    V.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    if (P["machine"] == "z80").any():
        Ls = sorted(P.loc[P["machine"] == "z80", "group"].astype(int).unique())
        slope_figure(P, "z80", Ls, {L: f"L = {L}" for L in Ls}, os.path.join(a.out, "fig_z80_slope"),
                     "loop instruction (jump, CALL/RET, RST or LDIR/LDDR)", fs.COL15)
    if (P["machine"] == "bff").any():
        vs = [v for v in VARIANTS if v in set(P.loc[P["machine"] == "bff", "group"])]
        titles = {"std": "BFF as published", "wrap": "wrapping pointer", "lit": "literal push", "wraplit": "wrap + literal push", "wraplitnh": "wrap + literal,\nno halt"}
        slope_figure(P, "bff", vs, titles, os.path.join(a.out, "fig_bff_slope"), "loop instruction ([ and ] present)", fs.DOUBLE)
    write_numbers(P, S, V, notes, a.out, a.n_partners)
    print(V[["prediction", "machine", "count", "n", "fraction", "outcome"]].to_string(index=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

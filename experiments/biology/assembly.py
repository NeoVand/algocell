"""Pre-registered analysis A (BIOLOGY_PREREG.md, section A): does the assembly measure of selection fire on the
sterile order (tar) that precedes the first heritable replicator?  CPU only; uses data already on disk.

    .venv/bin/python biology/assembly.py [--out results/biology/assembly] [--workers 10] [--skip-e] [--skip-g]

Measures (fixed in the prereg; see docstrings):
  * a(s): Re-Pair upper bound on the assembly index of a byte string (`repair_index`), exact on the three worked
    examples of the prereg (`self_test`).
  * A of a snapshot: sum_i e^{a_i} (n_i - 1) / N_T over unique tapes with n_i >= 2 (Sharma et al. 2023, eq. 1).
  * A_top10(t): the same sum over the ten exemplar classes of the sample record at step t, n_i = round(share_i N_T).
  * baseline: A_top10 at the first sample (step 1).
Predictions P1-P3 and their kill criteria are stated in the prereg; this script only produces the numbers.

Ground truth and strata for P3 are those of detectors.py (`event` = step >= t_rep > 0 from assays.csv; samples at the
snapshot-step schedule `STEPS`, stratified per L and per step; `auc` = Mann-Whitney with ties at 0.5).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import get_context

import brotli
import numpy as np
import pandas as pd

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (EXP, os.path.join(EXP, "manuscript", "figures")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import figstyle as fs  # noqa: E402
from algocell_exp.batch import select_summaries  # noqa: E402
from detectors import STEPS, auc  # noqa: E402

RUNS_G = os.path.join(EXP, "runs", "stageG")
RUNS_E = os.path.join(EXP, "runs", "stageE")
STAGE_G_CSV = os.path.join(EXP, "results", "stageG", "stageG", "stage_g_runs.csv")
STAGE_E_ASSAYS = os.path.join(EXP, "results", "stageE", "assays.csv")
OUT_DEFAULT = os.path.join(EXP, "results", "biology", "assembly")

N_T_G = 20000           # Stage G soups (prereg)
CROSS_FACTOR = 10.0     # P2: A_top10 > CROSS_FACTOR x baseline
EXEMPLAR_WORLD = "none@closure_L16_st128_k4_s2001"
L_MIN_P3 = 25
ACCENT = "#0072B2"      # one accent colour (Okabe-Ito blue) plus greys
GREY = "#8c8c8c"
LIGHT = "#c8c8c8"


# ----------------------------------------------------------------------------------------------------------------
# Assembly index (Re-Pair upper bound)
# ----------------------------------------------------------------------------------------------------------------

def repair_index(seq) -> int:
    """Re-Pair upper bound on the assembly index of `seq` (any sequence of hashable symbols; bytes work).

    Repeatedly replace the most frequent adjacent pair (non-overlapping occurrences, counted greedily left to right;
    ties broken by the pair whose first occurrence in the current sequence is earliest) by a new symbol until no pair
    occurs twice.  a = number of rules + (length of the final sequence - 1): every rule is one join, and the final
    sequence of m symbols needs m - 1 joins.  Exact for periodic strings; an upper bound in general.
    """
    s = list(seq)
    if len(s) <= 1:
        return 0
    rules = 0
    while True:
        counts: dict = {}
        first: dict = {}
        last: dict = {}
        for i in range(len(s) - 1):
            p = (s[i], s[i + 1])
            if last.get(p) == i - 1:          # would overlap the previously counted occurrence of this pair
                continue
            last[p] = i
            c = counts.get(p)
            if c is None:
                counts[p] = 1
                first[p] = i
            else:
                counts[p] = c + 1
        best, bc = None, 1
        for p, c in counts.items():           # insertion order = order of first occurrence
            if c > bc or (c == bc and best is not None and first[p] < first[best]):
                best, bc = p, c
        if best is None:
            break
        new = ("R", rules)
        out = []
        i, n = 0, len(s)
        b0, b1 = best
        while i < n:
            if i + 1 < n and s[i] == b0 and s[i + 1] == b1:
                out.append(new)
                i += 2
            else:
                out.append(s[i])
                i += 1
        s = out
        rules += 1
    return rules + (len(s) - 1)


_A_CACHE: dict[str, int] = {}


def a_hex(tape: str) -> int:
    """a(s) of a space-separated hex tape, cached (one process-local dict)."""
    a = _A_CACHE.get(tape)
    if a is None:
        a = repair_index(bytes.fromhex(tape))
        _A_CACHE[tape] = a
    return a


def assembly(a_values, counts, n_total: int) -> float:
    """A = sum_i e^{a_i} (n_i - 1) / N_T over classes with n_i >= 2 (singletons contribute zero)."""
    a = np.asarray(a_values, dtype=float)
    n = np.asarray(counts, dtype=float)
    m = n >= 2
    if not m.any():
        return 0.0
    return float(np.sum(np.exp(a[m]) * (n[m] - 1.0)) / n_total)


def sample_assembly(r: dict, cells: int) -> tuple[float, int, int, float]:
    """A_top10 of a sample record (its ten exemplar classes), the a of the modal class, the number of the ten classes
    with n >= 2, and the modal share."""
    shares = r["top10_shares"]
    tapes = [e["tape"] for e in r["exemplars"]]
    n = [int(round(s * cells)) for s in shares]
    a = [a_hex(t) for t in tapes]
    return assembly(a, n, cells), a[0], int(sum(1 for x in n if x >= 2)), float(shares[0])


def self_test() -> None:
    """The three worked examples of the prereg plus edge cases."""
    assert repair_index(bytes.fromhex("01 c5" * 8)) == 4, "01 c5 x 8 must give 4"
    assert repair_index(bytes(16)) == 4, "sixteen zero bytes must give 4"
    assert repair_index(bytes(range(16))) == 15, "16 bytes with no repeated pair must give 15"
    assert repair_index(b"") == 0 and repair_index(b"a") == 0 and repair_index(b"ab") == 1
    assert repair_index(b"aaa") == 2            # (a a) once only -> no rule; 3 symbols
    assert repair_index(b"abab") == 2           # ab -> X; X X -> Y: 2 rules, length 1
    assert a_hex("01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5 01 c5") == 4
    assert abs(assembly([4, 4, 9], [3, 1, 2], 20000) - (np.exp(4) * 2 + np.exp(9) * 1) / 20000) < 1e-12
    assert assembly([4], [1], 20000) == 0.0


# ----------------------------------------------------------------------------------------------------------------
# Stage G: one world -> per-sample rows, per-world row, per-snapshot rows
# ----------------------------------------------------------------------------------------------------------------

def _read_soup(path: str, L: int) -> np.ndarray:
    raw = brotli.decompress(open(path, "rb").read())
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, L)


def _snapshot_step(name: str, summary: dict) -> float:
    if name.startswith("t"):
        return int(name[1:])
    if name == "final":
        return int(summary.get("steps_run", -1))
    if name == "emergence":
        fe = summary.get("first_emergent") or {}
        return int(fe.get("step", -1))
    return float("nan")


def stage_g_world(task: dict) -> dict:
    """Everything for one Stage G world (run in a worker)."""
    world, L, seed, t_rep, first_tape = task["world"], task["L"], task["seed"], task["t_rep"], task["first_tape"]
    stem = os.path.join(RUNS_G, world)
    summary = json.load(open(stem + ".summary.json"))
    assert summary["cells"] == N_T_G, (world, summary["cells"])
    samples = []
    rows_s = []
    for line in open(stem + ".jsonl"):
        if '"kind": "sample"' not in line:
            continue
        r = json.loads(line)
        shares, hashes, ex = r["top10_shares"], r["top10_hashes"], r["exemplars"]
        assert len(shares) == 10 and len(ex) == 10, (world, r["step"])
        assert all(shares[i] >= shares[i + 1] for i in range(9)), ("shares not sorted", world, r["step"])
        assert [e["hash"] for e in ex] == hashes, ("exemplar order", world, r["step"])
        A, a_modal, n_ge2, top_share = sample_assembly(r, N_T_G)
        rows_s.append(dict(world=world, L=L, seed=seed, step=r["step"], A_top10=A, hoe=r.get("hoe"), unique=r["unique"],
                           top_share=top_share, a_modal=a_modal, n_top10_ge2=n_ge2))
        tapes = [e["tape"] for e in ex]
        ns = [int(round(s * N_T_G)) for s in shares]
        samples.append((r["step"], ex[0]["tape"], top_share, A, first_tape in tapes, max((a_hex(t) for t, n in zip(tapes, ns) if n >= 2), default=0)))
    samples.sort(key=lambda s: s[0])
    steps = [s[0] for s in samples]
    assert steps[0] == 1, (world, steps[:3])
    assert len(set(steps)) == len(steps), ("duplicate sample steps", world)
    baseline = samples[0][3]
    a_first = a_hex(first_tape)
    before = [s for s in samples if s[0] < t_rep]
    assert before, (world, t_rep)
    tar_step, tar_tape, tar_share, tar_A = before[-1][:4]
    a_tar = a_hex(tar_tape)
    cross = next((s for s in samples if s[3] > CROSS_FACTOR * baseline), None)
    at_trep = next((s for s in samples if s[0] == t_rep), None)
    A_by_step = {s[0]: s[3] for s in samples}
    # post hoc companion (not pre-registered): the first tape may already be modal below the 0.5% share that t_rep
    # requires (assay_batch.SHARE_MIN), so also take the tar as the modal tape at the last sample before the first
    # tape enters the top ten at all ("pre-arrival").
    arrival = next((s for s in samples if s[4]), None)
    arrival_step = arrival[0] if arrival else np.nan
    pre = [s for s in samples if arrival and s[0] < arrival[0]]
    if pre:
        pre_step, pre_tape, pre_share, pre_A = pre[-1][:4]
        a_pre = a_hex(pre_tape)
    else:
        pre_step, pre_tape, pre_share, pre_A, a_pre = np.nan, None, np.nan, np.nan, np.nan
    row_w = dict(world=world, L=L, seed=seed, horizon=task["horizon"], t_rep=t_rep,
                 first_tape=first_tape, a_first=a_first,
                 tar_step=tar_step, tar_modal_tape=tar_tape, tar_modal_share=tar_share, a_tar_modal=a_tar,
                 tar_is_first_tape=(tar_tape == first_tape),
                 p1_le=(a_first <= a_tar), p1_eq=(a_first == a_tar),
                 arrival_step=arrival_step, prearrival_step=pre_step, prearrival_modal_tape=pre_tape, prearrival_modal_share=pre_share,
                 a_prearrival_modal=a_pre, p1_prearrival_le=(a_first <= a_pre) if pre else np.nan,
                 baseline=baseline, threshold=CROSS_FACTOR * baseline, baseline_max_a=samples[0][5],
                 step_first_cross=(cross[0] if cross else np.nan), A_first_cross=(cross[3] if cross else np.nan),
                 cross_before_trep=(cross is not None and cross[0] < t_rep),
                 cross_before_arrival=(cross is not None and arrival is not None and cross[0] < arrival[0]),
                 A_at_tar_sample=tar_A, A_at_prearrival_sample=pre_A, A_at_trep=(at_trep[3] if at_trep else np.nan),
                 n_samples=len(samples), A_final=samples[-1][3])
    # snapshots
    rows_snap = []
    for f in sorted(glob.glob(stem + ".soup_*.u8.br")):
        name = os.path.basename(f).split(".soup_")[1].split(".u8")[0]
        soup = _read_soup(f, L)
        n_cells = len(soup)
        u, c = np.unique(soup, axis=0, return_counts=True)
        m = c >= 2
        idx = np.nonzero(m)[0]
        a_vals = np.array([a_hex(u[i].tobytes().hex(" ")) for i in idx], dtype=float)
        n_vals = c[idx].astype(float)
        A_exact = assembly(a_vals, n_vals, n_cells)
        contrib = np.exp(a_vals) * (n_vals - 1.0) / n_cells if len(idx) else np.zeros(0)
        tot = contrib.sum()
        j = int(np.argmax(contrib)) if len(idx) else -1
        order = np.argsort(-c, kind="stable")[:10]
        A_top10_snap = assembly([a_hex(u[i].tobytes().hex(" ")) for i in order], c[order], n_cells)
        i_mod = order[0]
        modal_tape = u[i_mod].tobytes().hex(" ")
        step = _snapshot_step(name, summary)
        rows_snap.append(dict(world=world, L=L, seed=seed, snapshot=name, step=step, cells=n_cells, A_exact=A_exact,
                              n_classes=len(u), n_classes_ge2=int(m.sum()), a_modal=a_hex(modal_tape), modal_n=int(c[i_mod]),
                              modal_tape=modal_tape, A_top10_snapshot=A_top10_snap,
                              A_top10_sample=A_by_step.get(step, np.nan), t_rep=t_rep,
                              A_exact_share_n_le3=(float(contrib[n_vals <= 3].sum() / tot) if tot > 0 else np.nan),
                              top_contrib_a=(int(a_vals[j]) if j >= 0 else np.nan), top_contrib_n=(int(n_vals[j]) if j >= 0 else np.nan),
                              top_contrib_share=(float(contrib[j] / tot) if tot > 0 else np.nan)))
    return dict(samples=rows_s, world=row_w, snapshots=rows_snap, cache=len(_A_CACHE))


def stage_g_tasks() -> list[dict]:
    g = pd.read_csv(STAGE_G_CSV)
    tasks = []
    for _, r in g.iterrows():
        world = f"{r['label']}_L{int(r['L'])}_st128_k4_s{int(r['seed'])}"
        path = os.path.join(RUNS_G, world + ".jsonl")
        assert os.path.exists(path), path
        assert int(r["t_rep"]) > 0, (world, r["t_rep"])
        tasks.append(dict(world=world, L=int(r["L"]), seed=int(r["seed"]), t_rep=int(r["t_rep"]), horizon=int(r["horizon"]),
                          first_tape=r["first_tape"]))
    assert len(tasks) == 80, len(tasks)
    allowed = {os.path.basename(p)[: -len(".summary.json")] for p in select_summaries(RUNS_G)}
    assert {t["world"] for t in tasks} <= allowed, "a Stage G world of the table is not in conds/stageG.json"
    return tasks


# ----------------------------------------------------------------------------------------------------------------
# Stage E: one run -> per-sample rows (all recorded samples)
# ----------------------------------------------------------------------------------------------------------------

def stage_e_run(summary_path: str) -> dict:
    s = json.load(open(summary_path))
    cells = int(s["cells"])
    stem = summary_path[: -len(".summary.json")]
    file = os.path.basename(summary_path)
    rows = []
    for line in open(stem + ".jsonl"):
        if '"kind": "sample"' not in line:
            continue
        r = json.loads(line)
        A, a_modal, n_ge2, top_share = sample_assembly(r, cells)
        rows.append((file, r["step"], A, r.get("hoe"), r["unique"], top_share, a_modal, n_ge2))
    return dict(file=file, cells=cells, label=s["label"], tape_len=int(s.get("tape_length", 16)), seed=s["seed"],
                steps=s["z80_steps"], k=s["noise_exp"], rows=rows)


# ----------------------------------------------------------------------------------------------------------------
# AUC strata (detectors.py conventions)
# ----------------------------------------------------------------------------------------------------------------

DETECTORS = ("A_top10", "hoe")


def auc_rows(df: pd.DataFrame, **keys) -> list[dict]:
    t = df["event"].to_numpy(bool)
    out = []
    for det in DETECTORS:
        score = df[det].astype(float).fillna(-np.inf).to_numpy()
        out.append(dict(**keys, detector=det, AUC=auc(score, t), n=len(df), n_pos=int(t.sum()), n_neg=int((~t).sum())))
    return out


def build_auc_table(de: pd.DataFrame) -> pd.DataFrame:
    """de: all Stage E samples with columns label, tape_len, step, event, A_top10, hoe, on_schedule."""
    rows = []
    big = de["tape_len"] >= L_MIN_P3
    nominal = de["label"].isin(["none@nominal", "stack-write-only@nominal"])
    sched = de["on_schedule"]
    for samples_name, smask in (("snapshot steps", sched), ("all samples", pd.Series(True, index=de.index))):
        rows += auc_rows(de[big & smask], stratum=f"E | L>={L_MIN_P3} | all arms | {samples_name}", samples=samples_name, arms="all", L=f">={L_MIN_P3}", step="all")
        rows += auc_rows(de[big & nominal & smask], stratum=f"E | L>={L_MIN_P3} | @nominal | {samples_name}", samples=samples_name, arms="@nominal", L=f">={L_MIN_P3}", step="all")
        rows += auc_rows(de[~big & smask], stratum=f"E | L<{L_MIN_P3} | all arms | {samples_name}", samples=samples_name, arms="all", L=f"<{L_MIN_P3}", step="all")
    for L, g in de[sched].groupby("tape_len"):
        rows += auc_rows(g, stratum=f"E | L={L} | all arms | snapshot steps", samples="snapshot steps", arms="all", L=str(L), step="all")
    for L, g in de[sched & nominal].groupby("tape_len"):
        rows += auc_rows(g, stratum=f"E | L={L} | @nominal | snapshot steps", samples="snapshot steps", arms="@nominal", L=str(L), step="all")
    for st, g in de[sched & big].groupby("step"):
        rows += auc_rows(g, stratum=f"E | L>={L_MIN_P3} | all arms | step {st}", samples="snapshot steps", arms="all", L=f">={L_MIN_P3}", step=str(st))
    for lab, g in de[sched & big].groupby("label"):
        rows += auc_rows(g, stratum=f"E | L>={L_MIN_P3} | {lab} | snapshot steps", samples="snapshot steps", arms=lab, L=f">={L_MIN_P3}", step="all")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------------------------

def _figcheck(fig, name: str) -> bool:
    import figcheck
    return figcheck.print_report(figcheck.check(fig), name)


def figure_exemplar(ps: pd.DataFrame, w: pd.Series, out: str) -> bool:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = ps[ps["world"] == w["world"]].sort_values("step")
    fig, ax = plt.subplots(figsize=(fs.SINGLE, 2.35))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(d["step"], d["A_top10"], color=ACCENT, lw=0.9, label="assembly $A_{top10}$ (left)", zorder=3)
    ax.axhline(w["threshold"], color=GREY, ls=":", lw=0.7, zorder=1)
    ax.axvline(w["t_rep"], color="black", ls="--", lw=0.7, zorder=2)
    if np.isfinite(w["step_first_cross"]):
        ax.plot([w["step_first_cross"]], [w["A_first_cross"]], marker="o", ms=3.5, mfc="white", mec=ACCENT, mew=0.8, ls="none", zorder=4)
    ax.set_xlabel("step")
    ax.set_ylabel("$A_{top10}$ = Σ e$^{a_i}$ (n$_i$ − 1) / N")
    ax.set_xlim(0.8, d["step"].max() * 1.3)
    lo, hi = d["A_top10"].min(), d["A_top10"].max()
    ax.set_ylim(lo / 3, hi * 40)
    ax2 = ax.twinx()
    ax2.plot(d["step"], d["hoe"], color=GREY, lw=0.7, label="high-order entropy (right)", zorder=2)
    ax2.set_ylabel("HOE (bits per byte)", color="#555555")
    ax2.tick_params(axis="y", colors="#555555")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#555555")
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, max(2.5, float(np.nanmax(d["hoe"])) * 1.9))
    # annotations placed in the free upper band
    ax.text(w["t_rep"] * 1.15, hi * 12, f"$t_{{rep}}$ = {int(w['t_rep']):,}", fontsize=5.5, ha="left", va="center")
    ax.text(1.0, w["threshold"] * 1.9, "10 × baseline", fontsize=5.5, ha="left", va="bottom", color="#555555")
    handles = [Line2D([], [], color=ACCENT, lw=0.9), Line2D([], [], color=GREY, lw=0.7),
               Line2D([], [], color="black", ls="--", lw=0.7), Line2D([], [], color=GREY, ls=":", lw=0.7),
               Line2D([], [], marker="o", ms=3.5, mfc="white", mec=ACCENT, mew=0.8, ls="none")]
    labels = ["$A_{top10}$ (left axis)", "HOE (right axis)", "$t_{rep}$ (first heritable replicator)", "10 × baseline", "first crossing"]
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=5.5, columnspacing=1.2, handlelength=1.8)
    ok = _figcheck(fig, "assembly_exemplar_L16_s2001")
    fs.save(fig, os.path.join(out, "assembly_exemplar_L16_s2001"))
    return ok


def figure_cross_vs_trep(pw: pd.DataFrame, out: str) -> bool:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(fs.SINGLE * 0.72, 2.4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    markers = {16: "o", 20: "s", 50: "^", 64: "D"}
    lo = min(pw["t_rep"].min(), pw["step_first_cross"].min()) / 1.6
    hi = pw["t_rep"].max() * 1.6
    ax.plot([lo, hi], [lo, hi], color=LIGHT, lw=0.7, ls="--", zorder=1, label="_diag")
    never = pw[~np.isfinite(pw["step_first_cross"])]
    rng = np.random.default_rng(0)
    for L in sorted(markers):
        g = pw[(pw["L"] == L) & np.isfinite(pw["step_first_cross"])]
        jit = np.exp(rng.uniform(-0.04, 0.04, len(g)))
        ax.plot(g["t_rep"] * jit, g["step_first_cross"] * np.exp(rng.uniform(-0.04, 0.04, len(g))), ls="none", marker=markers[L], ms=3.2,
                mfc="none", mec=ACCENT, mew=0.7, alpha=0.85, label=f"L = {L} (n = {len(g)})", zorder=3)
    if len(never):
        ax.plot(never["t_rep"], np.full(len(never), hi / 1.15), ls="none", marker="x", ms=3.5, color="black",
                label=f"no crossing (n = {len(never)}),\ndrawn at the top edge", zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("$t_{rep}$: first heritable replicator (step)")
    ax.set_ylabel("first $A_{top10}$ > 10 × baseline (step)")
    ax.set_xticks([100, 1000])
    ax.set_xticklabels(["100", "1,000"])
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1,000"])
    # region labels in the two free corners (upper left above the diagonal, mid right below it)
    ax.text(lo * 1.25, hi / 1.25, "fires after $t_{rep}$", fontsize=5.5, ha="left", va="top", color="#555555")
    ax.text(lo * 1.15, 1.9, "fires before $t_{rep}$", fontsize=5.5, ha="left", va="center", color="#555555")
    fs.outside_legend(ax, fontsize=5.5, handletextpad=0.3)
    ok = _figcheck(fig, "assembly_first_cross_vs_trep")
    fs.save(fig, os.path.join(out, "assembly_first_cross_vs_trep"))
    return ok


# ----------------------------------------------------------------------------------------------------------------
# NUMBERS
# ----------------------------------------------------------------------------------------------------------------

def pct(k: int, n: int) -> str:
    return f"{k}/{n} = {100.0 * k / n:.1f}%"


def write_numbers(out: str, pw: pd.DataFrame, ps: pd.DataFrame, snap: pd.DataFrame, au: pd.DataFrame | None, de: pd.DataFrame | None,
                  fig_ok: dict, elapsed: float) -> None:
    md = [f"# NUMBERS — analysis A, assembly fires on sterile order (generated by biology/assembly.py, {time.strftime('%Y-%m-%d %H:%M')})\n",
          "Every number below names the table it was read from (all in this directory). Definitions: BIOLOGY_PREREG.md section A.\n"]
    n = len(pw)
    md.append("## Re-Pair self-test (biology/assembly.py::self_test)\n")
    md.append("`01 c5` × 8 → 4; sixteen `00` bytes → 4; 16 bytes with no repeated adjacent pair → 15: all pass (asserted on every run).\n")

    md.append("## P1 — a(first_tape) vs a(tar modal tape at the last sample before t_rep)  [per_world.csv]\n")
    le, eq, lt, gt = int(pw["p1_le"].sum()), int(pw["p1_eq"].sum()), int((pw["a_first"] < pw["a_tar_modal"]).sum()), int((pw["a_first"] > pw["a_tar_modal"]).sum())
    md.append(f"- a_first ≤ a_tar: {pct(le, n)} (prediction ≥ 80%; kill < 50%)")
    md.append(f"- of which equal: {pct(eq, n)}; strictly less: {pct(lt, n)}; a_first > a_tar (against P1): {pct(gt, n)}")
    md.append(f"- tar modal tape identical to first_tape (comparison degenerate): {pct(int(pw['tar_is_first_tape'].sum()), n)}")
    by = pw.groupby("L").agg(n=("world", "size"), le=("p1_le", "sum"), eq=("p1_eq", "sum"), a_first=("a_first", lambda s: sorted(set(s))),
                             a_tar_min=("a_tar_modal", "min"), a_tar_median=("a_tar_modal", "median"), a_tar_max=("a_tar_modal", "max"))
    md.append("\n| L | worlds | a_first ≤ a_tar | equal | a_first values | a_tar min / median / max |\n|---|---|---|---|---|---|")
    for L, r in by.iterrows():
        md.append(f"| {L} | {int(r['n'])} | {int(r['le'])} | {int(r['eq'])} | {r['a_first']} | {int(r['a_tar_min'])} / {r['a_tar_median']:g} / {int(r['a_tar_max'])} |")
    md.append("\nTar modal tapes (last sample before t_rep), by pattern [per_world.csv, column tar_modal_tape]:\n")
    pat = pw["tar_modal_tape"].map(_tape_pattern).value_counts()
    for k, v in pat.items():
        md.append(f"- {k}: {v}")
    md.append("\nFirst tapes, by pattern [per_world.csv, column first_tape]:\n")
    for k, v in pw["first_tape"].map(_tape_pattern).value_counts().items():
        md.append(f"- {k}: {v}")
    md.append("\n### P1 companion, post hoc (not pre-registered): tar = modal tape at the last sample before the first tape enters the top ten  [per_world.csv, columns arrival_step, prearrival_*]\n")
    md.append("Why: t_rep requires a heritable top-3 exemplar with share ≥ 0.5% (assay_batch.py, SHARE_MIN), so the first tape is often already modal below that share at the last sample before t_rep; this companion removes that degeneracy.\n")
    pre = pw[pw["prearrival_modal_tape"].notna()]
    ple = int(pre["p1_prearrival_le"].astype(bool).sum())
    peq = int((pre["a_first"] == pre["a_prearrival_modal"]).sum())
    plt_ = int((pre["a_first"] < pre["a_prearrival_modal"]).sum())
    md.append(f"- worlds with a sample before the first tape's arrival in the top ten: {len(pre)}/{n}; arrival step: min {int(pw['arrival_step'].min())}, median {pw['arrival_step'].median():g}, max {int(pw['arrival_step'].max())}; t_rep − arrival: median {(pw['t_rep'] - pw['arrival_step']).median():g} steps")
    md.append(f"- a_first ≤ a_prearrival: {pct(ple, len(pre))}; equal: {pct(peq, len(pre))}; strictly less: {pct(plt_, len(pre))}; a_first > a_prearrival: {pct(len(pre) - ple, len(pre))}")
    md.append(f"- pre-arrival modal tape identical to the first tape: {int((pre['prearrival_modal_tape'] == pre['first_tape']).sum())} (by construction 0)")
    byp = pre.groupby("L").agg(n=("world", "size"), le=("p1_prearrival_le", lambda s: int(s.astype(bool).sum())), a_pre_min=("a_prearrival_modal", "min"), a_pre_med=("a_prearrival_modal", "median"), a_pre_max=("a_prearrival_modal", "max"), share=("prearrival_modal_share", "median"))
    md.append("\n| L | worlds | a_first ≤ a_prearrival | a_prearrival min / median / max | median modal share |\n|---|---|---|---|---|")
    for L, r in byp.iterrows():
        md.append(f"| {L} | {int(r['n'])} | {int(r['le'])} | {int(r['a_pre_min'])} / {r['a_pre_med']:g} / {int(r['a_pre_max'])} | {r['share']:.4f} |")
    md.append("\nPre-arrival modal tapes, by pattern:\n")
    for k, v in pre["prearrival_modal_tape"].map(_tape_pattern).value_counts().items():
        md.append(f"- {k}: {v}")

    md.append("\n## P2 — first sample with A_top10 > 10 × baseline, vs t_rep  [per_world.csv]\n")
    cb = int(pw["cross_before_trep"].sum())
    never = int((~np.isfinite(pw["step_first_cross"])).sum())
    md.append(f"- crossing before t_rep (step_first_cross < t_rep): {pct(cb, n)} (prediction ≥ 75%; kill < 50%)")
    md.append(f"- never crossing within the horizon: {never}; crossing at or after t_rep: {n - cb - never}")
    late = pw[np.isfinite(pw["step_first_cross"]) & ~pw["cross_before_trep"]]
    if len(late):
        md.append("- crossing at or after t_rep: " + "; ".join(f"{r['world']} (t_rep {int(r['t_rep'])}, cross {int(r['step_first_cross'])})" for _, r in late.iterrows()))
    inflated = pw[pw["baseline_max_a"] >= pw["L"] / 2]
    md.append(f"- worlds whose step-1 top ten holds a duplicated essentially random tape (a ≥ L/2 with n ≥ 2): {len(inflated)}: "
              + ", ".join(f"{r['world']} (max a = {int(r['baseline_max_a'])}, baseline {r['baseline']:.3g})" for _, r in inflated.iterrows()))
    rest = pw[pw["baseline_max_a"] < pw["L"] / 2]
    md.append(f"- among the other {len(rest)} worlds: crossing before t_rep in {pct(int(rest['cross_before_trep'].sum()), len(rest))}; never crossing: {int((~np.isfinite(rest['step_first_cross'])).sum())}")
    cba = int(pw["cross_before_arrival"].sum())
    md.append(f"- post hoc: crossing before the first tape enters the top ten at all (step_first_cross < arrival_step): {pct(cba, n)} — these crossings are carried by tapes other than the replicator")
    byA = pw.groupby("L").agg(n=("world", "size"), cba=("cross_before_arrival", "sum"))
    md.append("  by L: " + ", ".join(f"L = {L}: {int(r['cba'])}/{int(r['n'])}" for L, r in byA.iterrows()))
    fin = pw[np.isfinite(pw["step_first_cross"])]
    md.append(f"- step_first_cross: min {int(fin['step_first_cross'].min())}, median {fin['step_first_cross'].median():g}, max {int(fin['step_first_cross'].max())} (sample steps are 1, 2, 3, 5, 8, 13, 21, 34, 50, 100, …)")
    md.append(f"- t_rep: min {int(pw['t_rep'].min())}, median {pw['t_rep'].median():g}, max {int(pw['t_rep'].max())}")
    lead = (fin["t_rep"] / fin["step_first_cross"])
    md.append(f"- lead t_rep / step_first_cross: min {lead.min():.2g}, median {lead.median():.3g}, max {lead.max():.3g}")
    md.append(f"- baseline A_top10 at step 1: min {pw['baseline'].min():.3g}, median {pw['baseline'].median():.3g}, max {pw['baseline'].max():.3g}")
    md.append(f"- A_top10 at the tar sample / baseline: median {(pw['A_at_tar_sample'] / pw['baseline']).median():.3g}; at the t_rep sample / baseline: median {(pw['A_at_trep'] / pw['baseline']).median():.3g}")
    by2 = pw.groupby("L").agg(n=("world", "size"), cb=("cross_before_trep", "sum"), med_cross=("step_first_cross", "median"), med_trep=("t_rep", "median"))
    md.append("\n| L | worlds | cross before t_rep | median step_first_cross | median t_rep |\n|---|---|---|---|---|")
    for L, r in by2.iterrows():
        md.append(f"| {L} | {int(r['n'])} | {int(r['cb'])} | {r['med_cross']:g} | {r['med_trep']:g} |")

    ex = pw[pw["world"] == EXEMPLAR_WORLD]
    if len(ex):
        e = ex.iloc[0]
        md.append(f"\n## Exemplar world {EXEMPLAR_WORLD}  [per_world.csv, per_sample.csv]\n")
        md.append(f"- t_rep = {int(e['t_rep'])}; first_tape `{e['first_tape']}`, a = {int(e['a_first'])}; tar modal tape at step {int(e['tar_step'])}: `{e['tar_modal_tape']}` (share {e['tar_modal_share']:.4f}), a = {int(e['a_tar_modal'])}")
        md.append(f"- baseline A_top10(step 1) = {e['baseline']:.4g}; threshold = {e['threshold']:.4g}; first crossing at step {e['step_first_cross']:g} with A_top10 = {e['A_first_cross']:.4g}")
        md.append(f"- A_top10 at the tar sample = {e['A_at_tar_sample']:.4g}; at t_rep = {e['A_at_trep']:.4g}; at the end = {e['A_final']:.4g}")

    md.append("\n## Exact snapshot assembly vs the top-10 proxy  [per_snapshot.csv]\n")
    ts = snap[snap["snapshot"].str.startswith("t") & np.isfinite(snap["A_top10_sample"])]
    ratio_snap = (ts["A_top10_snapshot"] / ts["A_exact"]).replace([np.inf], np.nan).dropna()
    ratio_samp = (ts["A_top10_sample"] / ts["A_exact"]).replace([np.inf], np.nan).dropna()
    md.append(f"- snapshots: {len(snap)} ({snap['snapshot'].nunique()} names over {snap['world'].nunique()} worlds); timed snapshots with a sample at the same step: {len(ts)}")
    md.append(f"- A_top10 (from the snapshot's own ten most common classes) / A_exact: median {ratio_snap.median():.3f}, 10th pct {ratio_snap.quantile(0.1):.3f}, min {ratio_snap.min():.3f}")
    md.append(f"- A_top10 (from the sample record at the same step) / A_exact: median {ratio_samp.median():.3f}, 10th pct {ratio_samp.quantile(0.1):.3f}, min {ratio_samp.min():.3f}")
    md.append(f"- classes with n ≥ 2 per snapshot: median {snap['n_classes_ge2'].median():g}, max {int(snap['n_classes_ge2'].max())}; modal class a: min {int(snap['a_modal'].min())}, max {int(snap['a_modal'].max())}")
    md.append(f"- share of A_exact carried by classes with n ≤ 3 copies: median {snap['A_exact_share_n_le3'].median():.3f} (by L: "
              + ", ".join(f"L = {L}: {g['A_exact_share_n_le3'].median():.3f}" for L, g in snap.groupby("L")) + ")")
    md.append(f"- the single class contributing most to A_exact: median share {snap['top_contrib_share'].median():.3f}; its copy number median {snap['top_contrib_n'].median():g}, its a median {snap['top_contrib_a'].median():g} (by L: "
              + ", ".join(f"L = {L}: n {g['top_contrib_n'].median():g}, a {g['top_contrib_a'].median():g}" for L, g in snap.groupby("L")) + ")")
    pre = ts[ts["step"] < ts["t_rep"]]
    post = ts[ts["step"] >= ts["t_rep"]]
    if len(pre):
        md.append(f"- timed snapshots before t_rep: {len(pre)} (only steps 500–1,000 precede t_rep in some worlds); A_exact median {pre['A_exact'].median():.3g}; after/at t_rep: {len(post)}, A_exact median {post['A_exact'].median():.3g}")
    else:
        md.append(f"- no timed snapshot precedes t_rep in any world (earliest snapshot is step 500); A_exact over all timed snapshots: median {ts['A_exact'].median():.3g}")

    if au is not None and de is not None:
        md.append("\n## P3 — AUC of A_top10 against `event` on Stage E, with HOE on the same samples  [auc.csv, per_sample_E.csv]\n")
        md.append(f"Runs: {de['file'].nunique()} (conds/stageE.json); samples on the detectors.py schedule: {int(de['on_schedule'].sum()):,}; all recorded samples: {len(de):,}. "
                  f"N_T per run = the run's own cell count (20,000 except the `bytes` arm). `event` = step ≥ t_rep > 0 (results/stageE/assays.csv).\n")
        md.append("| stratum | detector | AUC | n | n_pos | n_neg |\n|---|---|---|---|---|---|")
        head = au[au["step"].eq("all") & au["L"].isin([f">={L_MIN_P3}", f"<{L_MIN_P3}"]) & au["arms"].isin(["all", "@nominal"])]
        for _, r in head.iterrows():
            md.append(f"| {r['stratum']} | {r['detector']} | {r['AUC']:.3f} | {r['n']} | {r['n_pos']} | {r['n_neg']} |")
        prim = au[(au["stratum"] == f"E | L>={L_MIN_P3} | all arms | snapshot steps")].set_index("detector")
        v = prim.loc["A_top10", "AUC"]
        verdict = "prediction met" if v <= 0.65 else ("KILL criterion met" if v >= 0.75 else "neither the prediction (≤ 0.65) nor the kill criterion (≥ 0.75): pre-registered grey zone")
        md.append(f"\nPrimary P3 number (prereg: pooled over L ≥ {L_MIN_P3}, strata of detectors.py): AUC(A_top10) = {v:.3f} — {verdict}; "
                  f"HOE on the same samples = {prim.loc['hoe', 'AUC']:.3f}.\n")
        md.append("Cross-check: the HOE AUCs of the @nominal per-L strata below reproduce results/detectors/NUMBERS_DETECTORS.md (same samples, same `auc`).\n")
        md.append("### by L (snapshot steps; all arms, then @nominal only — the latter matches the strata of results/detectors/NUMBERS_DETECTORS.md)\n")
        md.append("| L | arms | n | n_pos | AUC A_top10 | AUC hoe |\n|---|---|---|---|---|---|")
        byL = au[au["step"].eq("all") & ~au["L"].str.startswith((">", "<"))]
        for (L, arms), g in byL.groupby(["L", "arms"], sort=False):
            g = g.set_index("detector")
            md.append(f"| {L} | {arms} | {int(g.loc['A_top10', 'n'])} | {int(g.loc['A_top10', 'n_pos'])} | {g.loc['A_top10', 'AUC']:.3f} | {g.loc['hoe', 'AUC']:.3f} |")
        md.append(f"\n### by step (snapshot steps; L ≥ {L_MIN_P3}, all arms)\n")
        md.append("| step | n | n_pos | AUC A_top10 | AUC hoe |\n|---|---|---|---|---|")
        for st, g in au[~au["step"].eq("all")].groupby("step", sort=False):
            g = g.set_index("detector")
            md.append(f"| {st} | {int(g.loc['A_top10', 'n'])} | {int(g.loc['A_top10', 'n_pos'])} | {g.loc['A_top10', 'AUC']:.3f} | {g.loc['hoe', 'AUC']:.3f} |")
        md.append(f"\n### by arm (snapshot steps; L ≥ {L_MIN_P3})\n")
        md.append("| arm | n | n_pos | AUC A_top10 | AUC hoe |\n|---|---|---|---|---|")
        for lab, g in au[au["step"].eq("all") & au["L"].eq(f">={L_MIN_P3}") & ~au["arms"].isin(["all", "@nominal"])].groupby("arms", sort=False):
            g = g.set_index("detector")
            md.append(f"| {lab} | {int(g.loc['A_top10', 'n'])} | {int(g.loc['A_top10', 'n_pos'])} | {g.loc['A_top10', 'AUC']:.3f} | {g.loc['hoe', 'AUC']:.3f} |")
    else:
        md.append("\n## P3 — not computed in this run (--skip-e)\n")

    md.append("\n## Figures\n")
    for k, v in fig_ok.items():
        md.append(f"- {k}: figcheck {'clean' if v else 'NOT clean (see console)'}")
    md.append(f"\nRuntime {elapsed:.0f} s; a-cache sizes are printed on the console.\n")
    with open(os.path.join(out, "NUMBERS_ASSEMBLY.md"), "w") as fh:
        fh.write("\n".join(md))
    print("\n".join(md))


def _tape_pattern(tape: str) -> str:
    b = bytes.fromhex(tape)
    L = len(b)
    if all(x == 0 for x in b):
        return f"all-zero (L = {L})"
    for p in (1, 2, 4, 8):
        if p < L and all(b[i] == b[i % p] for i in range(L)):
            unit = b[:p].hex(" ")
            return f"period {p} `{unit}` × {L // p}" + (f" + {L % p}" if L % p else "")
    for p in (2, 4, 8):
        mism = sum(1 for i in range(L) if b[i] != b[i % p])
        if mism <= 2:
            return f"period {p} with {mism} mismatched byte(s)"
    return "other"


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--skip-e", action="store_true", help="skip Stage E (P3)")
    ap.add_argument("--skip-g", action="store_true", help="skip Stage G (P1, P2, snapshots, figures); reuse tables in --out")
    ap.add_argument("--reuse-e", action="store_true", help="reuse per_sample_E_all.csv.gz in --out instead of re-reading Stage E")
    ap.add_argument("--write-all-samples", action="store_true", help="also write per_sample_E_all.csv.gz (all 858k Stage E samples, ~23 MB)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    self_test()
    print("self-test: Re-Pair worked examples pass")
    ctx = get_context("spawn")

    # ---------------- Stage G ----------------
    if not a.skip_g:
        tasks = stage_g_tasks()
        with ctx.Pool(a.workers) as pool:
            res = pool.map(stage_g_world, tasks, chunksize=1)
        ps = pd.DataFrame([r for x in res for r in x["samples"]]).sort_values(["L", "seed", "step"])
        pw = pd.DataFrame([x["world"] for x in res]).sort_values(["L", "seed"])
        snap = pd.DataFrame([r for x in res for r in x["snapshots"]]).sort_values(["L", "seed", "step", "snapshot"])
        ps.to_csv(os.path.join(a.out, "per_sample.csv"), index=False)
        pw.to_csv(os.path.join(a.out, "per_world.csv"), index=False)
        snap.to_csv(os.path.join(a.out, "per_snapshot.csv"), index=False)
        print(f"Stage G: {len(pw)} worlds, {len(ps)} samples, {len(snap)} snapshots in {time.time() - t0:.0f} s")
    else:
        ps = pd.read_csv(os.path.join(a.out, "per_sample.csv"))
        pw = pd.read_csv(os.path.join(a.out, "per_world.csv"))
        snap = pd.read_csv(os.path.join(a.out, "per_snapshot.csv"))

    # ---------------- Stage E ----------------
    au = de = None
    if not a.skip_e:
        cache_path = os.path.join(a.out, "per_sample_E_all.csv.gz")
        if a.reuse_e and os.path.exists(cache_path):
            de = pd.read_csv(cache_path)
        else:
            t1 = time.time()
            paths = select_summaries(RUNS_E)
            with ctx.Pool(a.workers) as pool:
                res = pool.map(stage_e_run, paths, chunksize=2)
            rows = []
            meta = []
            for x in res:
                rows += x["rows"]
                meta.append(dict(file=x["file"], cells=x["cells"], label=x["label"], tape_len=x["tape_len"], seed=x["seed"], steps=x["steps"], k=x["k"]))
            de = pd.DataFrame(rows, columns=["file", "step", "A_top10", "hoe", "unique", "top_share", "a_modal", "n_top10_ge2"])
            de = de.merge(pd.DataFrame(meta), on="file", how="left")
            asy = pd.read_csv(STAGE_E_ASSAYS)[["file", "t_rep"]]
            missing = set(de["file"]) - set(asy["file"])
            assert not missing, f"{len(missing)} runs without an assays.csv row, e.g. {sorted(missing)[:3]}"
            de = de.merge(asy, on="file", how="left")
            de["event"] = (de["t_rep"] > 0) & (de["step"] >= de["t_rep"])
            de["on_schedule"] = de["step"].isin(STEPS)
            de = de.sort_values(["label", "tape_len", "seed", "step"])
            if a.write_all_samples:
                de.to_csv(cache_path, index=False)
            print(f"Stage E: {de['file'].nunique()} runs, {len(de)} samples ({int(de['on_schedule'].sum())} on the detectors.py schedule) in {time.time() - t1:.0f} s")
        de[de["on_schedule"]].to_csv(os.path.join(a.out, "per_sample_E.csv"), index=False)
        au = build_auc_table(de)
        au.to_csv(os.path.join(a.out, "auc.csv"), index=False)

    # ---------------- figures ----------------
    fs.setup()
    fig_ok = {}
    exw = pw[pw["world"] == EXEMPLAR_WORLD]
    if len(exw):
        fig_ok["assembly_exemplar_L16_s2001"] = figure_exemplar(ps, exw.iloc[0], a.out)
    fig_ok["assembly_first_cross_vs_trep"] = figure_cross_vs_trep(pw, a.out)

    write_numbers(a.out, pw, ps, snap, au, de, fig_ok, time.time() - t0)


if __name__ == "__main__":
    main()

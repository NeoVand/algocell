"""Generate T2.md and comparison tables from frozen C1 scorer outputs (no GPU).

python tier2_report.py --data results/tier2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dial_soups import DIAL, _km_median, _perm_spearman
from score_L import is_block
from stats_tests import cochran_armitage


def met(ok):
    return "met" if ok else "not met"


def table(frame):
    return frame.to_markdown(index=False, floatfmt=".4g")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("results/tier2"))
    args = ap.parse_args()
    data = args.data
    scored = data / "scored"
    baseline = pd.read_csv(data / "PUBLISHED_VERDICTS.csv")
    values = {}
    scripts = []

    def put(claim, component, verdict, evidence):
        key = claim, component
        assert key not in values, key
        values[key] = verdict, evidence

    def script(claim, component, verdict, evidence):
        scripts.append(dict(claim=claim,script_component=component,verdict=verdict,evidence=evidence))

    # DZ uses the frozen report's numerical functions. The reversal count is
    # explicitly registered but absent from that report, so is separate here.
    d = pd.read_csv(scored / "DZ/dial_worlds.csv")
    assert len(d) == 60 and d.groupby("variant").size().eq(10).all()
    ranks = {v: i for i, v in enumerate(DIAL)}
    d["rank"] = d.variant.map(ranks)
    g = d.groupby("rank")
    opens = g.first_open.apply(lambda s: int(s.fillna(False).astype(bool).sum()))
    z, p2 = cochran_armitage(opens.values, g.size().values, opens.index.values)
    p1 = p2 / 2 if z < 0 else 1 - p2 / 2
    r2, p2s = _perm_spearman(d["rank"], d.t_closed.fillna(d.horizon).astype(float))
    r3, p3 = _perm_spearman(d["rank"], d.frac_transmitter_of_heritable)
    medians = [_km_median(d[d.variant == v].t_closed.tolist(), 300000) for v in DIAL]
    reversals = sum(b < a for a, b in zip(medians, medians[1:]))
    n0, n1 = int(opens.loc[0]), int(opens.loc[5])
    tr0 = int((d[d.variant == "lp0"].frac_transmitter_of_heritable > .5).sum())
    tr1 = int((d[d.variant == "lp1"].frac_transmitter_of_heritable > .5).sum())
    put("DZ1", "open_p0_ge8", met(n0 >= 8), f"{n0}/10 open first (threshold >=8)")
    put("DZ1", "open_p1_le2", met(n1 <= 2), f"{n1}/10 open first (threshold <=2)")
    put("DZ1", "falling_rank_trend", met(z < 0 and p1 < .05), f"z={z:.6g}; one-sided P={p1:.6g}")
    put("DZ2", "KM_medians_le1_reversal", met(reversals <= 1), f"{reversals} reversal; medians=" + ", ".join(f"{x:g}" for x in medians))
    put("DZ2", "positive_spearman_perm_lt05", met(r2 > 0 and p2s < .05), f"rho={r2:.6g}; frozen 20000-permutation P={p2s:g}")
    put("DZ3", "positive_transmitter_trend_perm_lt05", met(r3 > 0 and p3 < .05), f"rho={r3:.6g}; frozen 20000-permutation P={p3:g}")
    put("DZ3", "p0_transmitter_majority_le3", met(tr0 <= 3), f"{tr0}/10 transmitter-majority worlds")
    put("DZ3", "p1_transmitter_majority_ge7", met(tr1 >= 7), f"{tr1}/10 transmitter-majority worlds")
    script("DZ2", "correlation_only_report", met(r2 > 0 and p2s < .05), "DIAL.md omits KM reversal verdict; registered count is evaluated separately")
    dz = pd.DataFrame([dict(p=DIAL[v], worlds=10, open_first=int(opens.loc[ranks[v]]),
                               KM_t_closed=medians[ranks[v]], transmitter_majority=int((d[d.variant == v].frac_transmitter_of_heritable > .5).sum())) for v in DIAL])
    dz.to_csv(data / "T2_DZ.csv", index=False)

    e = pd.read_csv(scored / "E/offset_switch.csv")
    assert len(e) == 110
    mix = e[e.start == "mix50"]
    for L, tar in ((16,"benign"),(32,"benign")):
        arm = mix[(mix.L == L) & (mix.tar == tar) & (mix.mut == "off")]
        assert len(arm) == 10
        n = int(arm.T_share_end.between(.3,.7).sum())
        put("E1", f"L{L}_benign_off_mix50_7of10_in_03_07", met(n >= 7), f"{n}/10 in [.3,.7]")
    for L, tar in ((16,"benign"),(16,"lethal"),(32,"benign")):
        arm = mix[(mix.L == L) & (mix.tar == tar) & (mix.mut == "on")]
        assert len(arm) == 10
        interval = arm.T_share_end.between(.05,.4)
        nscript = int(interval.sum())
        nliteral = int((interval & (arm.T_share_end < arm.T_share_start)).sum())
        put("E2", f"L{L}_{tar}_on_mix50_7of10_below_start_in_005_04", met(nliteral >= 7), f"literal below-start AND interval {nliteral}/10; frozen printed interval count {nscript}/10")
        script("E2", f"L{L}_{tar}_interval_only", met(nscript >= 7), f"{nscript}/10; interval-only count omits below-start conjunction")
    put("E3", "lethal_direction", "no direction registered", "All lethal-arm tables descriptive; no winner test")
    arm = mix[(mix.L == 32) & (mix.tar == "lethal") & (mix.mut == "on")]
    assert len(arm) == 10
    nwin, nkill = int((arm.T_share_end > .5).sum()), int((arm.T_share_end < .4).sum())
    put("E-L32L", "7of10_Tshare_gt_half", met(nwin >= 7), f"{nwin}/10 T share >.5; median {arm.T_share_end.median():.6g}, range {arm.T_share_end.min():.6g}–{arm.T_share_end.max():.6g}")
    put("E-L32L", "kill_7of10_Tshare_lt04", "fired" if nkill >= 7 else "not fired", f"{nkill}/10 T share <.4")
    script("E-L32L", "frozen_report", "descriptive", "offset_switch.py labels lethal arms E3 descriptive; registered >.5/<.4 thresholds evaluated separately")

    cap = pd.read_csv(scored / "L/capacity_time_L/capacity_over_time.csv").drop_duplicates(["stage","L","seed","step"]).sort_values(["L","seed","step"])
    cap["herit"] = cap.ctrl_herit.fillna(False).astype(bool)
    l = pd.read_csv(scored / "L/scoring_L.csv")
    assert len(l) == 20 and l.groupby("L").size().eq(10).all()
    ls = []
    for L in (16,20):
        arm = l[l.L == L]
        rec = arm[arm.recovered]
        n = len(rec)
        ns = int((arm.final_sites.fillna(0) <= 1).sum())
        frozen_ok = all(r.final_block and (r.final_unexec or 0) >= max(1,(r.final_sites or 0)) * .5 for r in rec.itertuples())
        literal_ok = all(r.final_block and r.final_unexec == r.final_sites for r in rec.itertuples())
        fverdict = "n/a (no recovery)" if n == 0 else met(frozen_ok)
        literal = "n/a (no recovery)" if n == 0 else met(literal_ok)
        snaps = cap[(cap.L == L) & cap.herit & cap.tape.notna()]
        viol = snaps[(~snaps.tape.astype(str).apply(is_block)) & (snaps.n_sites > 1)]
        vclosed = viol[viol.entered_frac < .5]
        put("L1", f"L{L}_recovery_ge5of10", met(n >= 5), f"{n}/10 recover; final <=1 site {ns}/10")
        detail = "; ".join(f"seed {int(r.seed)}: reference {int(r.ref_sites)} -> final {int(r.final_sites)} sites, {int(r.final_unexec)} unexecuted" for r in rec.itertuples())
        put("L2", f"L{L}_block_and_unexecuted_recovery", literal, f"{n} recoveries; literal all-site verdict {literal}; frozen >=half verdict {fverdict}" + ("; " + detail if detail else ""))
        put("L3", f"L{L}_nonblock_dominant_le1_all_snapshots", met(viol.empty), f"{len(viol)} violating snapshots; {len(vclosed)} closed-only violations (restriction unregistered)")
        script("L2", f"L{L}_half_unexecuted", fverdict, f"{n} recoveries; >=half sites unexecuted, versus literal all sites")
        ls.append(dict(L=L,recoveries=n,final_le1=ns,L2_frozen=fverdict,L2_literal=literal,L3_violations=len(viol),L3_closed_only_violations=len(vclosed)))
    kill = all(x["recoveries"] < 5 and x["final_le1"] >= 8 for x in ls)
    put("L", "joint_kill_both_lengths", "fired" if kill else "not fired", "; ".join(f"L{x['L']}: recovery {x['recoveries']}/10, final <=1 {x['final_le1']}/10" for x in ls))
    lsummary = pd.DataFrame(ls)
    lsummary.to_csv(data / "T2_L.csv", index=False)

    # Journal motif counts are at the horizon. Frozen final-soup assays retain
    # the original overshoot and legacy absolute-path-length RNG salt.
    endpoints = pd.read_csv(data / "c1_r4_endpoints.csv")
    r4 = pd.read_csv(scored / "R4/r4_inv.csv")
    assert len(endpoints) == len(r4) == 32
    joined = endpoints.merge(r4,on=["variant","resident","invader","seed"],validate="one_to_one")
    forward = joined[(joined.variant == "randreg") & (joined.resident == "pusher") & (joined.invader == "SI13")]
    reverse = joined[(joined.variant == "randreg") & (joined.resident == "SI13") & (joined.invader == "pusher")]
    assert len(forward) == len(reverse) == 5
    nmotif = int((forward.SI13 > .5).sum())
    nconf = int((forward.confined_given_heritable > .5).sum())
    ncombined = int(((forward.SI13 > .5) & (forward.confined_given_heritable > .5)).sum())
    nreverse = int((reverse.pusher < .05).sum())
    put("C4b", "SI13_motif_gt50_and_confined_4of5", met(nmotif >= 4), f"literal SI13 motif >.5 {nmotif}/5 (decision 3 motif component); descriptive motif AND confinement {ncombined}/5")
    put("C4b", "most_heritable_confined_4of5", met(nconf >= 4), f"{nconf}/5 most heritable confined; range {forward.confined_given_heritable.min():.6g}–{forward.confined_given_heritable.max():.6g}")
    put("C4b", "reverse_pusher_lt05_4of5", met(nreverse >= 4), f"{nreverse}/5 pusher <.05; horizon share range {reverse.pusher.min():.6g}–{reverse.pusher.max():.6g}")
    put("C4c", "invasion_direction", "no direction registered", "C4c tables descriptive; closed-descendant reading is not a winner test")
    script("C4b", "frozen_report", "descriptive", "r4_analyse.py emits final-cell classes, not a binary verdict; named motif and descendant confinement remain separate")
    oldend = pd.read_csv(data / "published_r4_endpoints.csv")
    oldr4 = pd.read_csv(Path(__file__).resolve().parent / "results/r4/r4_inv.csv")
    old = oldend.merge(oldr4,on=["variant","resident","invader","seed"],validate="one_to_one")
    c4rows=[]
    for protocol,frame in (("published",old),("C1",joined)):
        for (res,inv),a in frame[frame.variant == "std"].groupby(["resident","invader"]):
            c4rows.append(dict(protocol=protocol,resident=res,invader=inv,n=len(a),SI13_mean=a.SI13.mean(),R4_mean=a.R4.mean(),R4_min=a.R4.min(),R4_max=a.R4.max(),heritable_min=a.heritable.min(),heritable_max=a.heritable.max(),confined_min=a.confined_given_heritable.min(),confined_max=a.confined_given_heritable.max()))
    c4 = pd.DataFrame(c4rows)
    c4.to_csv(data / "T2_C4c.csv",index=False)

    f = pd.read_csv(scored / "F/rings.csv").iloc[0]
    put("F4c", "heritable_lt10of10_OR_KM_gt2000", met(f.t_rep_n < 10 or f.t_rep_km > 2000), f"{int(f.t_rep_n)}/10 heritable; KM median {f.t_rep_km} (not reached if inf)")
    sm = pd.read_csv(scored / "Serial/sr_summary.csv").set_index("key")
    st = pd.read_csv(scored / "Serial/sr_sites.csv")
    assert len(sm) == 14
    keys = ["pusher50","pusher64"]
    put("S1", "complete_registered_genotype_panel", met(all(sm.loc[k,"sites_g4"] >= max(10,sm.loc[k,"sites_g1"]/2) for k in keys)), "; ".join(f"{k}: g1={sm.loc[k,'sites_g1']}, g4={sm.loc[k,'sites_g4']}" for k in keys))
    keys = ["ret16","ldir20","ldir32","jr50"]
    put("S2", "complete_registered_genotype_panel", met(all(sm.loc[k,"sites_g4"] <= 2 and sm.loc[k,"erased_g1"] > sm.loc[k,"retained_g1"] for k in keys)), "; ".join(f"{k}: g4={sm.loc[k,'sites_g4']}; erased/retained g1={sm.loc[k,'erased_g1']:.4g}/{sm.loc[k,'retained_g1']:.4g}" for k in keys))
    keys = ["closer32_8080_p1","closer32_8080_p2"]
    payload = {k:int(st[(st.key == k) & (st.gen == 8) & (st.pos >= 20)].transmissible.sum()) for k in keys}
    put("S3", "complete_registered_genotype_panel", met(all(payload[k] >= 11 and sm.loc[k,"ctrl_alive_g8"] >= .99 for k in keys)), "; ".join(f"{k}: payload={payload[k]}/12; control alive={sm.loc[k,'ctrl_alive_g8']:g}" for k in keys))
    keys = ["genome16_s6006","genome20_s6003","genome20_s6004"]
    put("S4", "complete_registered_genotype_panel", met(all(sm.loc[k,"sites_g4"] >= 5 for k in keys)), "; ".join(f"{k}: g4={sm.loc[k,'sites_g4']}" for k in keys))
    put("S5", "complete_registered_genotype_panel", "met (descriptive)" if sm.loc["pusher16","sites_g4"] <= 2 else "not met (descriptive)", f"pusher16: g4={sm.loc['pusher16','sites_g4']}")

    b = pd.read_csv(scored / "Branching/branching_summary.csv")
    assert b.registered_params.all() and len(b[["key","arm"]].drop_duplicates()) == 7
    b8 = b[(b.arm == "U") & (b.gen == 8)].set_index("key")
    p16 = b8.loc["pusher16"]
    put("X1-1", "registered_U_endpoint", met(p16.ctrl_alive >= .5), f"pusher16 U: {int(p16.ctrl_alive_n)}/{int(p16.ctrl)} alive at g8")
    put("X1-2", "registered_U_endpoint", met(p16.sites >= 1), f"pusher16 U: {int(p16.sites)} sites at g8; K: {int(b[(b.key == 'pusher16') & (b.arm == 'K') & (b.gen == 8)].iloc[0].sites)} sites")
    keys = ["ret16","ldir32","genome20_s6003"]
    put("X1-3", "registered_U_endpoint", met(b8.loc[keys[0],"sites"] <= 2 and b8.loc[keys[1],"sites"] <= 2 and b8.loc[keys[2],"sites"] >= 5), "; ".join(f"{k}: {int(b8.loc[k,'sites'])} sites at g8" for k in keys))

    expected = set(zip(baseline.claim,baseline.registered_component))
    assert set(values) == expected and len(values) == 36, (expected - set(values), set(values) - expected)
    components = baseline.copy()
    components["C1_verdict"] = [values[(r.claim,r.registered_component)][0] for r in baseline.itertuples()]
    components["evidence"] = [values[(r.claim,r.registered_component)][1] for r in baseline.itertuples()]
    components["comparison"] = np.where(components.published_verdict == components.C1_verdict,"same","changed")
    components.to_csv(data / "T2_COMPONENTS.csv",index=False)
    claims = pd.DataFrame([dict(claim=claim,status="stands" if a.comparison.eq("same").all() else "changed",components="; ".join(f"{r.registered_component}: {r.published_verdict} -> {r.C1_verdict}" for r in a.itertuples())) for claim,a in components.groupby("claim",sort=False)])
    claims.to_csv(data / "T2_CLAIMS.csv",index=False)
    pd.DataFrame(scripts).to_csv(data / "T2_SCRIPT_COMPONENTS.csv",index=False)

    # Stage E is a registered descriptive addendum, outside the binary ledger.
    se = pd.read_csv(scored / "StageE-L10/size_arms.csv")
    assert len(se) == 1 and se.iloc[0].n == 10
    seo = pd.read_csv(Path(__file__).resolve().parent / "results/stageE/stage_e/size_arms.csv")
    seo = seo[(seo.ablation == "none") & (seo.arm == "nominal") & (seo.tape_len == 10)]
    secols = ["n","t_rep_n","t_rep_km","t_faith_n","t_faith_km","whole_tape_n","final_rep_n","func_rnd_median"]
    sedesc = pd.concat([seo[secols].assign(protocol="published"),se[secols].assign(protocol="C1")],ignore_index=True)[["protocol",*secols]]
    sedesc.to_csv(data / "T2_StageE_L10.csv",index=False)
    sea = pd.read_csv(scored / "StageE-L10/analysis/assays.csv")
    assert len(sea) == 10
    seworld = sea[["seed","steps_run","t_rep","t_faith","final_replicator","final_func_rnd"]]
    seworld.to_csv(data / "T2_StageE_L10_worlds.csv",index=False)

    diag = json.loads((data / "stageL_diagnosis.json").read_text())
    assert len(diag) == 20 and all(x["scoreable"] for x in diag)
    diagtable = pd.DataFrame([dict(L=x["L"],seed=x["seed"],steps_run=x["steps_run"],last_journal_step=x["journal_final_step"],final_bytes=x["final_bytes"],saved_snapshots=len(x["actual_snapshot_steps"]),missing_steps=",".join(map(str,x["missing_config_snapshots"])),scoreable=x["scoreable"]) for x in diag])
    diagtable.to_csv(data / "T2_StageL_status.csv",index=False)
    spend = json.loads((data / "score_spend.json").read_text())
    billing = pd.DataFrame(spend["apps"])
    assert billing.metered_usd.notna().all()
    scoring_usd = billing[billing.kind == "scoring"].metered_usd.sum()
    all_usd = billing.metered_usd.sum()
    assert scoring_usd < 5 and all_usd < 40
    commands=[]
    for p in sorted(scored.glob("*/execution.json")):
        execution = json.loads(p.read_text())
        assert execution["status"] == "complete", p
        for cmd in execution["commands"]:
            assert cmd["returncode"] == 0
            commands.append(dict(experiment=execution["experiment"],command=" ".join(cmd["argv"]),seconds=cmd["finished_epoch"]-cmd["started_epoch"],returncode=cmd["returncode"]))
    pd.DataFrame(commands).to_csv(data / "T2_COMMANDS.csv",index=False)
    changed = components[components.comparison == "changed"]
    text = ["# I1-T2 — frozen scoring under C1", "",
            "Generated by `tier2_report.py` from the committed scorer tables. No new tier-2 world was run. All assays use C1 on Modal NVIDIA L4 (one CPU core, 8 GiB, zero automatic retries).",
            "",
            f"**{int(claims.status.eq('stands').sum())}/{len(claims)} ledger claims stand; {len(changed)}/36 registered components change.** A claim stands only when every registered verdict equals its published verdict, including previously not-met verdicts. E3 and C4c retain their no-direction status; this is not a positive test result. Stage E L10 is descriptive and outside the 36-component ledger.",
            "", "## Per claim", "", table(claims), "", "## Changed components", "",
            table(changed[["claim","registered_component","published_verdict","C1_verdict","evidence"]]) if len(changed) else "None.",
            "", "## Registered components", "", table(components[["claim","registered_component","published_verdict","C1_verdict","comparison","evidence"]]),
            "", "## Frozen scripts versus literal registration (decision 4)", "", table(pd.DataFrame(scripts)),
            "", "Frozen reports are retained under `scored/`: DZ `DIAL.md`, E `REPORT.md` and `OFFSET_ENCOUNTER.md`, L `SCORING.md`, R4 `R4_INVASIONS.md`, F `NUMBERS_F.md`, serial `REPORT.md`, and branching `BRANCHING.md`. Numerical formulas and thresholds were not changed.",
            "", "The C4b ledger retains its original component identifier containing `and_confined`; decision 3 makes named SI13 motif majority, confinement, and reverse pusher suppression separate verdicts. The published descendant-based interpretation remains descriptive, rather than replacing the failed literal motif test.",
            "", "### DZ measurements", "", table(dz), "", "A permutation P of 0 means zero of the frozen 20,000 permutations reached the observed statistic; it is not a claim of exactly zero population probability.",
            "", "### Stage L measurements", "", table(lsummary), "", table(l[["L","seed","ref_step","ref_sites","final_sites","final_unexec","final_block","recovered"]]),
            "", "## C4c descriptive comparison", "", "Motif shares use the horizon journal (100,000 for C4c); culture and confinement use the frozen saved final soup at 100,250. Each world uses its own convention. No invasion direction was registered. Unseeded controls are included.", "", table(c4),
            "", "Complete horizon motif tables are `published_r4_endpoints.csv` and `c1_r4_endpoints.csv`; final-cell classes are `scored/R4/r4_inv.csv`. C4b horizon/final steps are 50,000/50,250. The original absolute-path-length RNG salt is retained.",
            "", "## Stage E nominal L10 descriptive addendum", "", table(sedesc), "", table(seworld),
            "", "`inf` means the KM median was not reached; -1 means no recorded heritable/faithful emergence. No direction or binary claim was added. The first full Stage E plotting attempt failed on an empty finite-KM array. The table-only completion calls the frozen `stage_e.interaction_clock` and `stage_e.cell_rows` unchanged, after repeating only its C1 post hoc assays; it introduces no world run, new assay design, or threshold. The failed attempt and its spend are preserved.",
            "", "## Stage L diagnosis and scoring status", "",
            "**Scoreable: 20/20. Dispatcher completion remains 0/20 (20 end-check failures).** All journals parse and end at 10,000,000; summaries record the full horizon; final soups decompress to 20,000 × L bytes. All 23 scheduled snapshots that existed in the original archives exist here, with their interaction censuses. No missing snapshot was synthesized, and no archived C0 soup was substituted.",
            "", "Step 7500 comes from `make_conds.py`: `SNAPSHOTS_10M` extends the shared `SNAPSHOTS` tuple, including 7500, while `stage_l()` selects sample_every=5000. The original archived summaries request 7500, as do the registered C1 configs. Both use samples every 50 until 5000, then every 5000, with explicit sample steps [1,2,3,5,8,13,21,34]. `algocell_exp.run.next_sample_step` merges the sample grid and explicit sample steps, not the snapshot list. It therefore jumps from 5000 to 10000. Snapshot/census creation occurs only at a sampled target in the snapshot set, while `batch.check_outputs` independently requires every configured snapshot. The original worlds also have no step-7500 snapshot. The end check consequently reports exactly the missing step-7500 snapshot and census after each full C1 run.",
            "", "Frozen scorer requirements: `assay_batch.py` reads summary, journal, and final soup; `c4_functional.py` assays recorded journal random tapes at available steps and does not read censuses; `stage_g.py` consumes those assay/functional tables; `capacity_over_time.py` enumerates saved files through `soup_stills.snapshot_files`, without requiring step 7500; `score_L.py` uses the resulting capacity table and reference/final rows. Every required reference (300,000 at L16; 1,000,000 at L20), final row and original saved snapshot is present. Missing 7500 does not block that frozen pipeline. Its outputs are scored with this fact disclosed.",
            "", table(diagtable), "", "Full filenames, saved/census steps, archive comparison, marker errors and final-soup hashes are in `stageL_diagnosis.json`. The source volume was read only; scorer analysis was written in container scratch and published under the separate `tier2_scores/` prefix.",
            "", "## Actual spend", "", table(billing), "",
            f"Metered tier-2 world/panel app plus scoring apps: **${all_usd:.6f}**; new scoring, including the failed plotting attempt: **${scoring_usd:.6f} < $5**. Remaining under $40: **${40-all_usd:.6f}**. Billing is app-attributed and can lag; sampled at {spend['queried_utc']}. This reports actual metered charges, without substituting resource-time estimates. Preparation outside these apps is not represented as measured spend; retaining the original $2 preparation reserve leaves ${40-all_usd-2:.6f}.",
            "", f"Measured function-body runtime estimate at the registered resource rate: existing worlds/panels ${spend['run_runtime_estimate_usd']:.6f}; scoring ${spend['scoring_runtime_estimate_usd']:.6f}. These exclude container startup and are not the bill.",
            "", "`tier2_status.py`'s remaining-work forecast treats the Stage L end-check failures as 20 uncompleted worlds and projects rerunning them. That forecast is not adopted here: all 20 reached the horizon and their frozen scoring inputs are intact. Failed dispatcher markers remain unchanged; no replacement run is authorized or launched.",
            "", "## Reproducibility", "",
            "Frozen sources and helpers were exported from registration/launch commit `7821ff0`, rather than taking post-registration helper edits from the worktree tip. All 14 prepared-script hashes in `scoring_sources.json` match the Modal image; the mapped C1 machines come from the same commit. Execution records retain commands, GPU, hashes, timestamps and return codes. Serial and branching reuse all completed C1 measurement tables (14 and 7 jobs) and run only their frozen reports, retaining the registered fixed genotypes and K pools.",
            "", "`score_calls.json` and `score_retry_calls.json` identify Modal calls and output prefix. `T2_COMMANDS.csv` records successful scoring invocations. Scorer files under `scored/`, comparison CSVs and `score_spend.json` are the report inputs. Rebuild with `/Users/neo/repos/algocell/experiments/.venv/bin/python tier2_report.py --data results/tier2` from `experiments/`. This report generator performs only CPU table/statistical work.", ""]
    (data / "T2.md").write_text("\n".join(text))
    print(table(claims[["claim","status"]]))
    print(table(changed[["claim","registered_component","published_verdict","C1_verdict","evidence"]]))


if __name__ == "__main__":
    main()

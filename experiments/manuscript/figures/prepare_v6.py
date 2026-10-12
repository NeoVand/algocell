"""Generate auditable plotting tables from archived outputs (no simulation).

Optional --runs points to the historical archive; --c1-runs to downloaded C1
E logs. When absent, C1 trajectories are explicitly missing, not interpolated.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
sys.path.insert(0, str(EXP))
R = EXP / 'results'
D = HERE / 'out/v6/data'
SOURCES = []


def record(p):
    p = Path(p)
    SOURCES.append({'path': str(p.relative_to(EXP)) if p.is_relative_to(EXP) else str(p),
                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})


def read(p, **kwargs):
    p = R / p
    record(p)
    return pd.read_csv(p, low_memory=False, **kwargs)


def save(name, d):
    d.to_csv(D / (name + '.csv'), index=False)


def capacity():
    e = read('capacity/edit_eligibility.csv')
    c = read('capacity/capacity_cases.csv.gz')
    c['passed'] = (c.exact_passed == c.partners) & (c.provenance_passed == c.partners) & (c.core_passed == c.partners)
    assert c.passed.all()
    a = c[c.direction == 'forward'].query("kind=='single'")
    # A coordinate is measured free only if ALL 255 alternatives pass ALL partners.
    m = a.groupby(['edit_id', 'position']).agg(alternatives=('value', 'nunique'), passed=('passed', 'all')).reset_index()
    assert (m.alternatives == 255).all()
    counts = m[m.passed].groupby('edit_id').size()
    e['measured_free'] = e.edit_id.map(counts).fillna(0).astype(int)
    assert (e.measured_free == e.predicted_free).all()
    save('capacity_edits', e)
    all_m = c[c.kind=='single'].groupby(['edit_id','direction','position']).agg(alternatives=('value','nunique'),passed=('passed','all')).reset_index()
    assert (all_m.alternatives==255).all()
    caps = all_m.groupby(['edit_id','direction']).size().reset_index(name='measured_free')
    design = c[['edit_id','direction','d']].drop_duplicates().merge(caps,on=['edit_id','direction'],how='left').fillna({'measured_free':0})
    design['predicted_free']=design.d-4
    assert (design.predicted_free==design.measured_free).all()
    save('capacity_direct',design)
    s = read('capacity/serial.csv.gz')
    save('capacity_serial', s.groupby(['edit_id', 'direction', 'generation']).agg(
        lineages=('lineage', 'nunique'), exact=('exact', 'sum'), core_exact=('core_exact', 'sum'),
        d=('actual_d', 'first')).reset_index())
    save('capacity_challenges', c.groupby(['direction', 'd', 'kind']).agg(
        cases=('passed', 'size'), passed=('passed', 'sum'), partners=('partners', 'first')).reset_index())
    save('capacity_haplotypes', s[(s.lineage == 0) & (s.generation == 8)])
    eligible = read('evolved6502/eligibility.csv')
    exact = read('evolved6502/exact_cases.csv.gz')
    exact['passed'] = (exact.exact_passed == exact.partners) & (exact.provenance_passed == exact.partners)
    assert exact.passed.all()
    em = exact[exact.kind == 'single'].groupby(['key', 'position']).agg(alternatives=('value', 'nunique'), passed=('passed', 'all')).reset_index()
    assert (em.alternatives == 255).all()
    eligible['measured_free'] = eligible.key.map(em.groupby('key').size()).fillna(0).astype(int)
    save('evolved6502_capacity', eligible[eligible.eligible == 1])
    save('evolved6502_exclusions', eligible.groupby(['eligible', 'reason'], dropna=False).size().reset_index(name='roles'))
    save('evolved6502_challenges', exact.groupby('kind').agg(cases=('passed', 'size'), passed=('passed', 'sum'), partners=('partners', 'first')).reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, default=Path('/Users/neo/repos/algocell/experiments/runs'))
    ap.add_argument('--c1-runs', type=Path)
    args = ap.parse_args()
    D.mkdir(parents=True, exist_ok=True)
    for core, path in [('Historical', 'branching/branching_summary.csv'), ('C1', 'tier2/scored/Branching/branching_summary.csv')]:
        b = read(path)
        b['cohort'] = core
        if core == 'Historical': branch = b
        else: branch = pd.concat([branch, b])
    save('branching', branch)
    trace=read('exectrace/per_replicator.csv')
    refs=[]
    for key in ['pusher16','ret16']:
        tape=branch[branch.key==key].tape.iloc[0]
        q=trace[trace.tape==tape].sort_values(['stage','seed','which'])
        if len(q):
            row=q.iloc[0].to_dict();row['key']=key
            # exec_trace.py's archived 256-partner protocol (not assay controls).
            row['encounters']=256
            row['entered_n']=int(round(row['entered_frac']*row['encounters']))
            refs.append(row)
    save('reference_traces',pd.DataFrame(refs))
    sr = read('serial_retention/sr_mutants.csv.gz')
    sr = sr.query("gen==8 and key in ['pusher16','ret16','ldir32','genome20_s6003']").copy()
    sr['outcome'] = np.select([sr.retained > 0, sr.erased > 0], [2, 1], default=0)
    save('retention_maps', sr)
    save('retention_sites', read('serial_retention/sr_sites.csv').query('gen==8'))
    pe = read('pe/pe_matrix.csv')
    from serial_retention import R_CTRL, R_MUT
    pe['control_n'] = R_CTRL
    pe['control_alive_n'] = np.rint(pe.alive_g8 * R_CTRL).astype(int)
    pe['lineages_per_alternative'] = R_MUT
    pe['alternatives_per_position'] = 255
    save('parent_environment', pe)
    o = read('offset/offset_cells.csv')
    cc = read('crystal/cells.csv')
    ex = read('crystal/cell_exceptions.csv')
    cc = cc[cc.source == 'offset/offset_cells.csv']
    assert len(cc) == len(o)
    o['eligible_dividing'] = (o.L % o.offset == 0)
    o['predicted'] = o.offset - 4
    o['fetched_extra'] = o.n_sites - o.n_unexec
    o['residual'] = o.n_unexec - o.predicted
    o['world'] = o.stage + '_L' + o.L.astype(str) + '_s' + o.seed.astype(str)
    o['exception'] = cc.sort_values('source_row').exception_class.to_numpy()
    save('z80_payload', o)
    save('z80_worlds', o[o.eligible_dividing].groupby(['world','predicted']).agg(
        cells=('n_unexec','size'), measured=('n_unexec','mean'), residual=('residual','mean'),
        fetched_extra=('fetched_extra','mean')).reset_index())
    save('z80_exceptions', ex[ex.source == 'offset/offset_cells.csv'])
    controls = read('census6502/table_closers_law.csv')
    save('constructed6502', controls)
    capacity()
    save('allele_dial', read('crystal/alleles_ldir32.csv'))
    save('allele_encounters', read('crystal/allele_encounters_ldir32.csv'))
    cohorts = []
    for core, path in [('Historical','offset/offset_switch.csv'),('C1','tier2/scored/E/offset_switch.csv')]:
        t = read(path).query("L==32 and mut=='on'").copy()
        t['cohort'] = core
        cohorts.append(t)
    endpoints = pd.concat(cohorts)
    save('competition_endpoints', endpoints)
    trajectories, missing = [], []
    for row in endpoints.to_dict('records'):
        root = args.runs / 'offset' if row['cohort']=='Historical' else args.c1_runs
        name = f"L32_{row['tar']}_muton_{row['start']}_s{row['seed']}.jsonl"
        candidates = list(root.rglob(name)) if root and root.exists() else []
        # Reconcile every archived endpoint; never choose an unmatched duplicate.
        matched = None
        for p in candidates:
            logs = [json.loads(x) for x in p.read_text().splitlines()]
            t = pd.DataFrame([x for x in logs if 'core' in x])
            final = t.iloc[-1]
            share = final['T'] / (final['R']+final['T']) if final['R']+final['T'] else np.nan
            if np.isclose(share,row['T_share_end'],equal_nan=True) and np.isclose(final.core/20000,row['core_share_end']):
                matched=(p,t); break
        if matched is None:
            missing.append({k:row[k] for k in ['cohort','tar','start','seed']})
            continue
        p,t = matched
        record(p)
        t['T_share'] = t['T']/(t['R']+t['T']).replace(0,np.nan)
        t['core_share'] = t.core / 20000
        for k in ['cohort','tar','start','seed']:t[k]=row[k]
        t['cells']=20000
        trajectories.append(t)
    save('competition_trajectories',pd.concat(trajectories))
    save('missing_trajectories',pd.DataFrame(missing,columns=['cohort','tar','start','seed']))
    dial=[]
    for core,path in [('Historical','dial/dial_worlds.csv'),('C1','tier2/scored/DZ/dial_worlds.csv')]:
        t=read(path);t['cohort']=core;dial.append(t)
    save('dial_worlds',pd.concat(dial))
    save('c1_dial_summary',read('tier2/T2_DZ.csv'))
    save('components',read('tier2/T2_COMPONENTS.csv'))
    save('claims',read('tier2/T2_CLAIMS.csv'))
    save('impact',read('impact/outcome.csv'))
    save('validation',read('impact/validation.csv'))
    baseline=Path('/Users/neo/repos/zilion-wt-8086/test/baselines/8086-architectural.json')
    record(baseline)
    ledger=json.loads(baseline.read_text())
    save('8086_validation',pd.DataFrame([ledger['summary']['totals']]))
    video=args.runs/'video/video_L16_st128_k4_s2002.jsonl'
    record(video)
    logs=[json.loads(x) for x in video.read_text().splitlines()]
    condition=logs[0]
    t=pd.DataFrame([{'step':r['step'],'zero_fraction':r['zero_frac'],'dominant_neighbourhood':r['q_share'],
                     'distinct':r['unique'],'cells':condition['cells'],'L':condition['tape_length'],'seed':condition['seed']}
                    for r in logs if r.get('kind')=='sample'])
    save('world_trajectory',t)
    save('sources',pd.DataFrame(SOURCES).drop_duplicates())
    print('Generated plotting tables; missing trajectory worlds:',len(missing))


if __name__=='__main__':main()

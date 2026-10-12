"""Freeze v7 coordinates from v6.2 preparation and the previous figure tables.

No simulation or guessed values. V6.2 preparation remains the source of its
coordinates; this adds the historical KM and serial tables used by make_figures.
"""
from pathlib import Path
import ast
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

import make_figures as old

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
D = HERE / 'out/v7/data'
USED = ['branching', 'world_trajectory', 'validation_fractions', 'law_population',
        'law_z80_c1', 'law_6502_c2', 'law_6502_controls', 'capacity_edits',
        'evolved6502_capacity', 'lineage_strips', 'dial_coordinates',
        'competition_medians', 'competition_trajectories', 'competition_endpoints',
        'endpoint_positions', 'lethality_summary', 'erasure_ordered',
        'missing_trajectories', 'parent_environment', 'ed_joint_challenges',
        'dial_worlds', 'erasure_states', 'z80_payload']


def main():
    D.mkdir(parents=True, exist_ok=True)
    sources = []
    def record(p):
        sources.append(dict(path=str(p.relative_to(EXP)),
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    def save(name, data):
        data.to_csv(D / (name + '.csv'), index=False)
    for name in USED:
        p = HERE / 'out/v6/data' / (name + '.csv')
        record(p)
        shutil.copyfile(p, D / p.name)
    for name in ['c1_download_audit.json', 'erasure_manifest.json']:
        p = HERE / 'out/v6/data' / name
        record(p)
        shutil.copyfile(p, D / name)
    for name, src in [('serial_summary', 'serial_retention/sr_summary.csv'),
                      ('serial_sites', 'serial_retention/sr_sites.csv')]:
        p = EXP / 'results' / src
        record(p)
        save(name, pd.read_csv(p, low_memory=False))
    # Protocol numbers in the final labels are extracted from the committed
    # scorer rather than entered independently in the drawing code.
    method_rows=[]
    p=EXP/'impact_i1_report.py';record(p);tree=ast.parse(p.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decision')
    bound=ast.literal_eval(next(n for n in ast.walk(fn) if isinstance(n,ast.Compare)).comparators[0])
    p=EXP/'results/impact/decisions.csv';record(p);confidence=pd.read_csv(p).confidence.unique()
    assert len(confidence)==1
    method_rows += [dict(name='ring_rate_bound',value=bound,source='impact_i1_report.py::decision'),
                    dict(name='ring_confidence',value=float(confidence[0]),source='results/impact/decisions.csv')]
    p=EXP/'algocell_exp/assay.py';record(p);tree=ast.parse(p.read_text())
    minimum=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='GEN2_MIN' for t in n.targets))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_norm_gain_rows')
    gate=ast.literal_eval(fn.args.defaults[-1])
    method_rows += [dict(name='gen2_minimum',value=minimum,source='algocell_exp/assay.py::GEN2_MIN'),
                    dict(name='match_gate',value=gate,source='algocell_exp/assay.py::_norm_gain_rows')]
    save('methods',pd.DataFrame(method_rows))
    p = EXP / 'results/stageE/assays.csv'
    record(p)
    a = pd.read_csv(p, low_memory=False)
    a = a[(a.label == 'none@nominal') & (a.steps == 128) & (a.k == 4) & a.replicate.isna()]
    a = a[a.tape_len.isin([8, 9, 16, 36, 64, 100])]
    save('emergence_worlds', a[['tape_len', 'seed', 'horizon', 't_rep']])
    coords = []
    for length, g in a.groupby('tape_len'):
        assert len(g) == 10 and set(g.horizon) == {300000}
        x, y = old.km_curve(g.t_rep.tolist(), int(g.horizon.iloc[0]))
        for xx, yy in zip(x, y):
            coords.append(dict(L=length, step=max(40, xx), emergence=1-yy,
                               worlds=len(g), detected=int((g.t_rep >= 0).sum())))
    save('emergence_curves', pd.DataFrame(coords))
    b = pd.read_csv(D / 'branching.csv')
    b = b[(b.arm == 'U') & (b.gen == 8) & b.key.isin(old_keys())]
    hist, c1 = (b[b.cohort == c].set_index('key').sort_index() for c in ['Historical', 'C1'])
    cols = ['lost', 'erased', 'retained', 'ctrl_alive_n', 'sites', 'L', 'n_identifiable', 'reps']
    np.testing.assert_allclose(hist[cols], c1[cols])
    save('branching_endpoints', b)
    # Beeswarm at the actual v7 endpoint axes (45 x 20.5 mm), rather than the
    # larger v6.2 axes. Ordinate values are never displaced.
    ends = pd.read_csv(D / 'competition_endpoints.csv')
    placed_rows=[]
    for measure in ['T_share_end','core_share_end','T_distinct_tails_end']:
        span=1.10 if measure!='T_distinct_tails_end' else ends[measure].max()*1.12
        for (cohort,tar),group in ends.groupby(['cohort','tar']):
            placed=[]
            candidates=[0]+[s*x for x in [.09,.18,.27,.36,.45] for s in [-1,1]]
            for row in group[group[measure].notna()].sort_values([measure,'start','seed']).to_dict('records'):
                y=row[measure]
                ds=[np.array([((off-x)/.085)**2+((y-v)/(span*.060))**2 for x,v in placed]) for off in candidates]
                costs=[int((dist<1).sum()) for dist in ds]
                best=min(costs);valid=[j for j,cost in enumerate(costs) if cost==best]
                index=max(valid,key=lambda j:ds[j].min()) if best else valid[0]
                off=candidates[index];placed.append((off,y))
                placed_rows.append(dict(**row,offset=off,measure=measure,near_collisions=best))
    save('endpoint_positions',pd.DataFrame(placed_rows))
    # Abstract provenance examples: two equal-length tapes, six displayed units
    # per tape. Units aggregate byte positions; c=4, L=12, d=4 or d=L.
    rows = []
    for displacement in [4, 12]:
        for unit in range(12):
            position = unit * 2
            source = position % displacement
            rows.append(dict(L=12, c=4, d=displacement, unit=unit,
                             position=position, source=source,
                             state='core' if source < 4 else 'payload'))
    save('geometry_units', pd.DataFrame(rows))
    save('sources_v7', pd.DataFrame(sources))
    ledger = [dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
              for p in sorted(D.iterdir()) if p.is_file() and p.name != 'hashes.csv']
    save('hashes', pd.DataFrame(ledger))
    assert pd.read_csv(D / 'missing_trajectories.csv').empty
    print('v7 tables generated; historical/C1 branching endpoints agree; 80 competition worlds present.')


def old_keys():
    return ['pusher16', 'ret16', 'ldir32', 'genome20_s6003']


if __name__ == '__main__':
    main()

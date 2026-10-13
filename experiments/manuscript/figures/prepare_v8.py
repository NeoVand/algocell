"""Freeze v7 plotting inputs and generate Fig. 5 from the completed S2 tables. CPU only."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[1]
D=HERE/'out/v8/data'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    D.mkdir(parents=True,exist_ok=True)
    for p in (HERE/'out/v7/data').iterdir():
        if p.is_file() and p.name not in ['hashes.csv','sources_v7.csv']:
            shutil.copy2(p,D/p.name)
    sources=pd.read_csv(HERE/'out/v7/data/sources_v7.csv').to_dict('records')
    def source(name):
        p=EXP/name;sources.append(dict(path=name,sha256=sha(p)));return p
    def read(name):return pd.read_csv(source(name),low_memory=False)
    def save(name,rows):pd.DataFrame(rows).to_csv(D/(name+'.csv'),index=False)
    z=read('results/stageG/stageG/stage_g_runs.csv')
    c=read('results/s6502/s1_worlds.csv').query("arm=='A'")
    e=read('results/s8086/tables/worlds.csv')
    km=read('results/s8086/tables/kaplan_meier.csv').query("endpoint=='rep'")
    census=read('results/census8086/table_hits.csv')
    for p in ['results/stageG/FINDINGS.md','results/s6502/S1.md','results/s8086/S2.md','results/literal_channel/CHECK.md','results/census6502/CENSUS.md','results/census6502/table_main.csv','results/census6502/control_z80.csv']:
        source(p)
    assert len(z)==80 and z.t_rep.ge(0).all() and not z.first_has_cf.any()
    assert len(c)==20 and c.first_confined.all() and not c.first_open.any()
    assert len(e)==20 and e.scoreable.all() and e.rep_right_censored.sum()==6
    assert e.first_fetch.value_counts().to_dict()=={'confined':12,'open':2}
    witness=census.query("L==16 and budget==128 and word=='b951'")
    assert len(witness)==1 and witness.iloc[0].zero_match==16
    rows=[]
    for name,d in [('Z80 G',z.query('L==16')),('6502 A',c),('8086 S2',e)]:
        for r in d.itertuples():
            t=float(r.t_rep) if pd.notna(r.t_rep) and r.t_rep>=0 else np.nan
            horizon=int(r.censor_step) if name=='8086 S2' else int(r.horizon)
            rows.append(dict(machine=name,seed=int(r.seed),L=16,budget=128,event=bool(np.isfinite(t)),
                             detection=t,observed=t if np.isfinite(t) else horizon,horizon=horizon))
    worlds=pd.DataFrame(rows);save('arrival_worlds',worlds)
    curves=[];censors=[];summary=[]
    for name,d in worlds.groupby('machine',sort=False):
        risk=len(d);survival=1.;coords=[(40.,0.)]
        for time,g in d.groupby('observed',sort=True):
            events=int(g.event.sum());censored=len(g)-events
            if events:survival*=1-events/risk;coords.append((time,1-survival))
            if censored:censors.append(dict(machine=name,step=time,detected=1-survival,censored=censored))
            risk-=len(g)
        horizon=int(d.horizon.iloc[0]);coords.append((horizon,1-survival))
        for x,y in coords:curves.append(dict(machine=name,step=x,detected=y))
        median=next((x for x,y in coords if y>=.5-1e-12),np.nan)
        summary.append(dict(machine=name,worlds=len(d),events=int(d.event.sum()),censored=int((~d.event).sum()),horizon=horizon,median=median))
        if name=='8086 S2':
            actual=np.array(coords[1:-1]);np.testing.assert_allclose(actual[:,0],km.step)
            np.testing.assert_allclose(actual[:,1],1-km.survival,atol=1e-12)
    save('arrival_curves',curves);save('arrival_censors',censors);save('arrival_summary',summary)
    save('route_by_machine',[dict(machine='Z80',cohort='G; L16/20/50/64 bytes',worlds=80,open=80,confined=0,censored=0),
                             dict(machine='6502',cohort='S1 arm A; L16 bytes',worlds=20,open=0,confined=20,censored=0),
                             dict(machine='8086',cohort='S2; L16 bytes',worlds=20,open=2,confined=12,censored=6)])
    save('literal_matrix',[dict(machine='Z80',width=2,pusher='Yes',witness='01 C5',route='Open',confined=0,detected=80),
                           dict(machine='6502',width=1,pusher='No',witness='',route='Confined',confined=20,detected=20),
                           dict(machine='8086',width=2,pusher='Yes',witness='B9 51',route='Confined majority',confined=12,detected=14)])
    tiles=[]
    for r in e[e.first_tape.fillna('').str.startswith('5fa4')].itertuples():
        tape=bytes.fromhex(r.first_tape);assert len(tape)==16 and tape==tape[:4]*4 and r.first_fetch=='confined'
        for i,v in enumerate(tape):tiles.append(dict(seed=r.seed,detection=int(r.t_rep),position=i,byte=f'{v:02X}',
                 role='shared' if i%4<2 else 'variable',mechanism=r.first_majority_mechanism))
    assert len(tiles)==160 and len({r['seed'] for r in tiles})==10
    save('first_copier_tiles',tiles)
    save('sources_v8',sources)
    save('hashes',[dict(path=p.name,sha256=sha(p)) for p in sorted(D.iterdir()) if p.is_file() and p.name!='hashes.csv'])
    print(pd.DataFrame(summary).to_string(index=False))
    print('Frozen v8 inputs; S2 KM verified; all ten 16-byte tile tapes aligned without rotation.')


if __name__=='__main__':main()

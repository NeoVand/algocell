"""EXPLORATORY post hoc crystal checks. CPU tables, bounded local encounter bursts.

Use the archive-owning checkout's Python. From experiments:
  python crystal_check.py tables
  python crystal_check.py probe --key pusher16   # repeat for X1's five keys
  python crystal_check.py alleles --key ldir32  # also genome20_s6003
  python crystal_check.py offspring --source classes --L 16 --stage G
  python crystal_check.py report
No archives are modified; no soup, large mutation scan, Modal job or GPU process
is launched in the background. Each encounter command is killed at 58 seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import time
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from algocell_exp import assay as A
import branching_transfer as BT
from serial_retention import panel, hx

HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results'
OUT = RESULTS/'crystal'
STATUS = 'EXPLORATORY'
KEYS = BT.KEYS
INPUTS = {}


def read(path, **kw):
    path = Path(path)
    INPUTS[str(path.relative_to(HERE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, low_memory=False, **kw)


def write(name, rows):
    d = rows.copy() if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    if 'status' not in d:
        d.insert(0,'status',STATUS)
    OUT.mkdir(exist_ok=True)
    d.to_csv(OUT/name,index=False)
    return d


def manifest():
    INPUTS.pop('results/crystal/inputs.csv',None)
    write('inputs.csv',[{'path':k,'sha256':v} for k,v in sorted(INPUTS.items())])


def parse(tape):
    return np.frombuffer(bytes.fromhex(str(tape)),np.uint8).copy()


@lru_cache(None)
def periods(tape):
    b = bytes.fromhex(tape)
    L = len(b)
    cyclic = next(p for p in range(1,L+1) if L%p == 0 and b == (b[:p]*(L//p)))
    finite = next(p for p in range(1,L+1) if b[p:] == b[:-p] or p == L)
    return cyclic, finite


def family(tape):
    b = parse(tape)
    if len(b)>=4 and list(b[1:4])==[0x5e,0xed,0xb0]: return 'XX 5e ed b0'
    if len(b)>=4 and b[0]==0x1e and list(b[2:4])==[0xed,0xb0]: return '1e NN ed b0'
    s = bytes(b).hex()+bytes(b).hex()
    if 'edb0' in s or 'edb8' in s: return 'other block copy (motif)'
    if any(x in b for x in (0xc0,0xc8,0xc9,0xd0,0xd8)): return 'return-bearing (motif)'
    return 'other'


def bounds(row):
    p,L,e,u = row['period'],row['L'],row.get('n_executed',np.nan),row.get('n_unexec',np.nan)
    if not np.isfinite(e) or not np.isfinite(u):
        return pd.Series({'payload_lower':np.nan,'payload_upper':np.nan,'test':'missing fetched/unexecuted counts'})
    # Number of distinct fetched residues from an aggregate address union.
    lo = p-min(p,int(e))
    hi = p-math.ceil(int(e)/(L//p))
    if np.isfinite(row.get('fetched_residues',np.nan)):
        lo=hi=p-int(row['fetched_residues'])
    label = 'equal (exact from aggregate)' if lo==hi==u else (
        'deficit' if u<lo else 'excess' if u>hi else 'compatible; per-site mask missing')
    return pd.Series({'payload_lower':lo,'payload_upper':hi,'test':label})


def exception_class(row):
    labels=[]
    if row['test']=='deficit':labels.append('sampled unexecuted count below fetched-residue bound; mechanism unresolved')
    elif row['test']=='excess':labels.append('sampled unexecuted count above fetched-residue bound; minimal-period model fails')
    elif 'missing' in row['test']:labels.append('not addressable: per-site fetched mask or counts missing')
    if np.isfinite(row.get('n_unexec',np.nan)) and row.n_sites>row.n_unexec:
        labels.append('additional fetched-site transmission under sampled scorer')
    if row.source=='offset/offset_cells.csv':
        if row.test=='offset payload deficit':labels.append('offset payload deficit: targeted copy-gate rejection')
        if row.test=='offset outside dividing-d conditions':labels.append('excluded: Proposition 8 offset geometry')
        if row.offset_divides==True:
            if row.period<row.offset:labels.append('minimal period below copying prefix; mutations can expand unit cell')
            if row.period>row.offset:labels.append('unmutated parent not d-periodic; offspring projection changes period')
    return '; '.join(labels) or 'none for fetched-only sampled comparison'


def tables():
    frames=[]
    sources = [('offset/offset_cells.csv','n_unexec',None),
               ('population/classes_cells.csv','n_sites_unexecuted','n_executed'),
               ('population/population_cells.csv',None,None),
               ('s6502/v2/s1_cells.csv','n_sites_unexecuted','n_executed'),
               ('s6502/v2/s1_cells_64.csv','n_sites_unexecuted','n_executed')]
    for src,uc,ec in sources:
        d=read(RESULTS/src);d['source']=src;d['source_row']=np.arange(len(d))
        if 'L' not in d:
            d['L']=d.tape.map(lambda t:len(parse(t)))
        d['n_unexec']=d[uc] if uc else np.nan
        d['n_executed']=d[ec] if ec else np.nan
        frames.append(d)
    trace=read(RESULTS/'exectrace/per_replicator.csv')
    for folder in ('mutscan','mutscan_I','mutscan_K'):
        src=f'{folder}/mutscan_tapes.csv';d=read(RESULTS/src)
        read(RESULTS/folder/'mutscan_sites.csv')
        d=d.merge(trace[trace.stage=={'mutscan':'G','mutscan_I':'I','mutscan_K':'K'}[folder]][
            ['label','n_sites_unexecuted','exec_union']],on='label',how='left',validate='one_to_one')
        d['source']=src;d['source_row']=np.arange(len(d));d['n_unexec']=d.n_sites_unexecuted;d['n_executed']=d.exec_union
        frames.append(d)
    for src in ('serial_retention/sr_summary.csv','serial_retention/posthoc_cells.csv'):
        d=read(RESULTS/src,dtype={'executed':str})
        for g in BT.GENS_REPORT:
            e=d.copy();e['gen']=g;e['source']=src;e['source_row']=np.arange(len(e))
            e['n_sites']=e[f'sites_g{g}'];e['n_unexec']=e[f'sites_unexec_g{g}'];e['n_executed']=e.executed.map(lambda s:s.count('1'))
            e['fetched_residues']=e.apply(lambda r:len({i%periods(r.tape)[0] for i,b in enumerate(r.executed) if b=='1'}),axis=1)
            frames.append(e)
    d=pd.concat(frames,ignore_index=True)
    d['family']=d.tape.map(family)
    ps=d.tape.map(periods);d['period']=[x[0] for x in ps];d['finite_period']=[x[1] for x in ps]
    d['small_unit']=d.period<d.L
    d=pd.concat([d,d.apply(bounds,axis=1)],axis=1)
    # Offset rows lack masks but identify a core footprint, not general eligibility.
    off=d.source=='offset/offset_cells.csv'
    d.loc[off,'offset_divides']=d.loc[off].apply(lambda r:0<r.offset<=r.L and r.L%r.offset==0,axis=1)
    d.loc[off,'offset_prediction']=d.loc[off,'offset']-4
    for idx,r in d[off].iterrows():
        if r.offset_divides:
            d.loc[idx,'test']=('offset payload equal' if r.n_unexec==r.offset-4 else 'offset payload deficit' if r.n_unexec<r.offset-4 else 'offset payload excess')
        else:d.loc[idx,'test']='offset outside dividing-d conditions'
    d['exception_class']=d.apply(exception_class,axis=1)
    keep=['source','source_row','stage','arm','world','seed','L','snapshot','step','cell','key','which','gen','confined','heritable','family','tape','period','finite_period','small_unit','n_sites','n_unexec','n_executed','fetched_residues','payload_lower','payload_upper','test','offset','offset_divides','offset_prediction','exception_class']
    d=write('cells.csv',d[[c for c in keep if c in d]])
    write('cell_exceptions.csv',d[d.exception_class!='none for fetched-only sampled comparison'])
    write('cell_groups.csv',d.groupby(['source','family','test'],dropna=False).agg(
        cells=('tape','size'),small_units=('small_unit','sum'),median_period=('period','median'),
        positive_unexec=('n_unexec',lambda x:int((x>0).sum()))).reset_index())
    first=[]
    for stage in ('G','I'):
        for _,r in read(RESULTS/f'stage{stage}/stage{stage}/stage_g_runs.csv').iterrows():
            p,f=periods(r.first_tape)
            tr=trace[(trace.stage==stage)&(trace.L==r.L)&(trace.seed==r.seed)&(trace.which=='first')]
            rec={'dataset':f'Z80 {stage}','endpoint':'first replicator','stage':stage,'L':r.L,'seed':r.seed,'step':r.t_rep,'tape':r.first_tape,'period':p,'finite_period':f,'small_unit':p<r.L}
            if len(tr)==1:
                t=tr.iloc[0];rec.update({'confined':t.entered_frac==0,'n_sites':t.n_sites,'n_unexec':t.n_sites_unexecuted,'entered_frac':t.entered_frac,'H_bits':t.H_bits})
            first.append(rec)
            if rec.get('confined'):
                first.append({**rec,'endpoint':'first scored closed-heritable (same first tape)'})
    # Every cell at the earliest sampled closed-heritable snapshot, not one selected modal tape.
    c=d[(d.source=='population/classes_cells.csv') & (d.stage.isin(['G','I'])) & (d.confined==True)].copy()
    c['sample_step']=c.step.fillna(c.groupby(['stage','L','seed']).step.transform('max')+1)
    c=c[c.sample_step==c.groupby(['stage','L','seed']).sample_step.transform('min')]
    for _,r in c.iterrows():
        first.append({'dataset':f'Z80 {r.stage}','endpoint':'earliest sampled closed-heritable','stage':r.stage,'L':r.L,'seed':r.seed,'step':r.step,'snapshot':r.snapshot,'cell':r.cell,'tape':r.tape,'period':r.period,'finite_period':r.finite_period,'small_unit':r.period<r.L,'n_sites':r.n_sites,'n_unexec':r.n_unexec,'class': 'transmitter' if r.n_sites>=5 else 'intermediate' if r.n_sites>=3 else 'regenerator'})
    available={(r['stage'],r['L'],r['seed']) for r in first if r['endpoint']=='earliest sampled closed-heritable'}
    for r in list(first):
        if r['dataset'].startswith('Z80') and r['endpoint']=='first replicator' and (r['stage'],r['L'],r['seed']) not in available:
            first.append({'dataset':r['dataset'],'endpoint':'no sampled closed-heritable cell','stage':r['stage'],'L':r['L'],'seed':r['seed']})
    for _,r in read(RESULTS/'s6502/v2/s1_worlds.csv').iterrows():
        if pd.isna(r.first_tape):
            first.append({'dataset':f'6502 {r.arm}','endpoint':'censored first replicator','L':r.L,'seed':r.seed,'world':r.world})
            continue
        p,f=periods(r.first_tape)
        rec={'dataset':f'6502 {r.arm}','endpoint':'first replicator','L':r.L,'seed':r.seed,'world':r.world,'step':r.t_rep,'tape':r.first_tape,'period':p,'finite_period':f,'small_unit':p<r.L,'confined':r.first_confined}
        first.append(rec)
        if r.first_confined and r.t_rep==r.t_closed:
            first.append({**rec,'endpoint':'first closed-heritable (same scored tape)'})
        elif np.isfinite(r.t_closed):
            path=RESULTS/'s6502/v2/worlds'/f'{r.world}.json'
            INPUTS[str(path.relative_to(HERE))]=hashlib.sha256(path.read_bytes()).hexdigest()
            scored=json.loads(path.read_text())
            closed=next(t for t in scored['judged_tapes'] if t['heritable'] and t['confined'])
            p,f=periods(closed['tape'])
            first.append({**rec,'endpoint':'first closed-heritable (scored cache order)','step':r.t_closed,'tape':closed['tape'],'period':p,'finite_period':f,'small_unit':p<r.L,'confined':True})
    f=pd.DataFrame(first)
    f['world_id']=f.L.astype(str)+'/'+f.seed.astype(str)
    f=write('first_cells.csv',f)
    write('first_exceptions.csv',f[(f.period==f.L)|f.endpoint.str.contains('censored|no sampled')|((f.dataset=='Z80 I')&f.endpoint.str.startswith('first')&(f.n_sites<5))])
    write('first_groups.csv',f.groupby(['dataset','endpoint'],dropna=False).agg(cells=('seed','size'),worlds=('world_id','nunique'),small_units=('small_unit',lambda x:(x==True).sum()),median_period=('period','median'),min_period=('period','min'),max_period=('period','max')).reset_index())
    manifest()
    print(f'EXPLORATORY: {len(d)} scan records; {len(f)} first/earliest closed records (CPU).',flush=True)


def gt_for(key):
    return next(g for g in panel() if g['key']==key)


def timeout():
    def stop(*_):raise TimeoutError('EXPLORATORY: stopped encounter command at 58 seconds')
    signal.signal(signal.SIGALRM,stop);signal.alarm(58)


def roles_rows(key,roles,phase):
    out=[]
    for j in range(roles.shape[-1]):
        r=roles[:,j]
        out.append({'key':key,'phase':phase,'pos':j,'n_encounters':len(r),'opcode_frac':float(((r&1)!=0).mean()),'operand_frac':float(((r&2)!=0).mean()),'data_read_frac':float(((r&4)!=0).mean()),'unread_frac':float(((r&7)==0).mean()),'write_frac':float(((r&8)!=0).mean())})
    return out


def probe(key):
    timeout();t0=time.monotonic()
    from crystal_roles import run,source
    gt=gt_for(key);x=gt['tape'];L=len(x);N=16
    pos,val,mut,wcode,ident=BT.mutants(gt,32,BT.SEED)
    rng=np.random.default_rng([BT.SEED,zlib.crc32(key.encode()),0,730])
    Q=BT.partners(rng,N,L,None)
    basepairs=np.concatenate([np.tile(x,(N,1)),Q],1)
    wt,pr=run(basepairs,L,gt['zh']);off=wt[:,L:]
    assert np.array_equal(wt,BT.execute(basepairs,L,list(gt['sup']),gt['zh']))
    pairs=np.concatenate([np.repeat(mut,N,0),np.tile(Q,(len(mut),1))],1)
    child,rr=run(pairs,L,gt['zh'])
    assert np.array_equal(child,BT.execute(pairs,L,list(gt['sup']),gt['zh']))
    from algocell_exp import exectrace as X
    _,masks=X.execute_pairs_traced(pairs,L,128,zero_halts=gt['zh'])
    assert np.array_equal((rr&3)!=0,X.exec_positions(masks,2*L))
    child=child[:,L:]
    changed=(child!=np.tile(off,(len(mut),1)))
    verb=changed & (child==np.repeat(val,N)[:,None])
    from serial_retention import window_codes
    marker=(window_codes(child)==np.repeat(wcode,N)[:,None]).any(axis=1)
    copies=A._best_shift_match_rows(child,np.repeat(mut,N,0))>=.75
    raw=[]
    for m in range(len(mut)):
        for q in range(N):
            a=m*N+q
            raw.append({'key':key,'parent_pos':int(pos[m]),'val':int(val[m]),'identifiable':bool(ident[m]),'partner':q,'changed_positions':' '.join(map(str,np.flatnonzero(changed[a]))),'verbatim_positions':' '.join(map(str,np.flatnonzero(verb[a]))),'child_copy':bool(copies[a]),'contains_mutant_window':bool(marker[a])})
    write(f'probes_{key}.csv',raw)
    alive_counts=copies.reshape(len(mut),N).sum(1)
    retained_counts=(copies&marker).reshape(len(mut),N).sum(1)
    ok=(alive_counts>0)&(2*retained_counts>=alive_counts)
    first_sites=[]
    for i in range(L):
        sel=(pos==i)&ident
        first_sites.append({'key':key,'pos':i,'identifiable_values':int(sel.sum()),'values_ok_without_background':int(ok[sel].sum()),'site_without_background':bool(2*ok[sel].sum()>=sel.sum())})
    write(f'first_encounter_sites_{key}.csv',first_sites)
    # The child's own next execution, including roles on mutated offspring.
    nextQ=BT.partners(rng,N,L,None)
    _,nr=run(np.concatenate([child,np.tile(nextQ,(len(mut),1))],1),L,gt['zh'])
    _,wr=run(np.concatenate([off,nextQ],1),L,gt['zh'])
    write(f'roles_{key}.csv',roles_rows(key,pr[:,:L],'wild-type parent')+roles_rows(key,wr[:,:L],'wild-type offspring next execution')+roles_rows(key,nr[:,:L],'mutant offspring next execution'))
    # Also record roles on each changed descendant position, so address union
    # cannot conceal how a particular probe allele is executed next.
    edge=[];byq=[]
    for i in range(L):
        sel=np.flatnonzero(pos==i);C=changed.reshape(len(mut),N,L)[sel];V=verb.reshape(len(mut),N,L)[sel]
        role=nr[:,:L].reshape(len(mut),N,L)[sel]
        byq.append(V.mean(axis=0)>=.5)
        for j in range(L):
            if C[:,:,j].any():
                v=V[:,:,j]
                edge.append({'key':key,'parent_pos':i,'child_pos':j,'n_probes':v.size,'n_changed':int(C[:,:,j].sum()),'n_verbatim':int(v.sum()),'verbatim_fraction':float(v.mean()),'verbatim_opcode_next':int((v & ((role[:,:,j]&1)!=0)).sum()),'verbatim_operand_next':int((v & ((role[:,:,j]&2)!=0)).sum()),'verbatim_data_next':int((v & ((role[:,:,j]&4)!=0)).sum())})
    write(f'provenance_{key}.csv',edge)
    maps=np.array(byq).transpose(1,0,2) # partner, parent, child
    op=((wr[:,:L]&1)!=0)
    predictions=[];site_predictions=[]
    for strict in (False,True):
        reach=np.tile(np.eye(L,dtype=bool),(N,1,1))
        for g in range(1,9):
            reach=np.matmul(reach.astype(np.int32),maps.astype(np.int32))>0
            if strict:reach &= ~op[:,None,:]
            if g in BT.GENS_REPORT:
                n=int((reach.any(axis=2).mean(axis=0)>=.5).sum())
                predictions.append({'key':key,'arm':'U','gen':g,'prediction':'no opcode in child executions' if strict else 'verbatim paths only','sites':n,'scope':'frozen wild-type local map; 32 values/site; 16 partners; >=half probes edge; >=half contexts site'})
                for i,fraction in enumerate(reach.any(axis=2).mean(axis=0)):
                    site_predictions.append({'key':key,'arm':'U','gen':g,'pos':i,'prediction':'no opcode in child executions' if strict else 'verbatim paths only','context_fraction':fraction,'predicted':fraction>=.5})
    write(f'prediction_{key}.csv',predictions)
    write(f'predicted_sites_{key}.csv',site_predictions)
    # These are local-map forecasts, not a new branching score or invariant-family proof.
    _,oh,th=source(L,gt['zh'])
    write(f'execution_{key}.csv',[{'key':key,'mode':'probe','executor':A.executor_file(L),'sha256_executor':oh,'sha256_roles_shader':th,'budget':128,'partners':N,'values':32,'memory_validation_cases':len(pairs)+N,'fetch_mask_validation_cases':len(pairs),'wall_s':time.monotonic()-t0}])
    print(f'EXPLORATORY: {key} differential/role checks {time.monotonic()-t0:.2f}s',flush=True)


def alleles(key):
    timeout();t0=time.monotonic()
    gt=gt_for(key);x=gt['tape'];L=len(x);N=16
    # genome20 is the actual six-byte LDDR architecture, not a substituted LDIR recipe.
    site=0 if key=='ldir32' else 2
    assert key=='ldir32' or list(x[:6])==[0x5e,0x11,0x04,0xc5,0xed,0xb8]
    T=np.tile(x,(256,1));T[:,site]=np.arange(256)
    rng=np.random.default_rng([BT.SEED,L,741]);Q=rng.integers(0,256,(N,L),dtype=np.uint8)
    Q2=rng.integers(0,256,(N,L),dtype=np.uint8)
    TT=np.repeat(T,N,0);RR=np.tile(Q,(256,1));RR2=np.tile(Q2,(256,1))
    pairs=np.concatenate([TT,RR],1)
    res=BT.execute(pairs,L,[],False);off=res[:,L:]
    res2=BT.execute(np.concatenate([off,RR2],1),L,[],False);off2=res2[:,L:]
    before=A._best_shift_match_rows(RR,TT);after,shifts=A._best_shift_rows(off,TT)
    before2=A._best_shift_match_rows(RR2,TT);after2=A._best_shift_match_rows(off2,TT)
    score=A._norm_gain_rows(before,after,256,N);g2=A._norm_gain_rows(before2,after2,256,N)
    # First-instruction register/memory evidence for Proposition 8 prerequisites.
    first,regs=A._shader_execute_pairs(pairs,L,1,return_regs=True)
    rows=[];enc=[]
    for v in range(256):
        a=slice(v*N,(v+1)*N);ps=[periods(hx(t))[0] for t in off[a]]
        d_initial=v%(2*L) if site==0 else (0xc500+v)%(2*L)
        d=d_initial if site==0 else ((0xc500+v-1)%(2*L)-(0xffff%(2*L)))%(2*L)
        eligible=False;reason='LDDR; six-byte core; 16-bit source wrap; Proposition 8 forward-prefix conditions absent'
        if site==0:
            e=regs[a];protected=(first[a,:max(d,1)]==pairs[a,:max(d,1)]).all(1)
            bc=256*e[:,2]+e[:,3];hl=256*e[:,6]+e[:,7]
            eligible=bool(4<=d<=L and L%d==0 and (e[:,9]==1).all() and protected.all() and (hl%(2*L)==0).all() and ((256*e[:,4])%(2*L)==0).all() and ((bc==0)|(bc>=126)).all())
            reason='sampled first-instruction prerequisites met; forward core, dividing d and coverage' if eligible else 'first-instruction/protected-prefix or dividing-d conditions fail'
        inherited=(off[np.arange(v*N,(v+1)*N),np.mod(site+shifts[a],L)]==v)
        changed=(off[a]!=off[(int(x[site]))*N:(int(x[site])+1)*N]).any(1)
        rows.append({'key':key,'site':site,'value':v,'hex':f'{v:02x}','initial_ring_displacement':d_initial,'effective_d':d,'core_c':4 if site==0 else 6,'prop8_eligible':eligible,'eligibility_reason':reason,'predicted_payload':d-4 if eligible else np.nan,'nominal_d_minus_c':d-(4 if site==0 else 6),'score':float(score[v]),'gen2_score':float(g2[v]),'culture_replicator':bool(g2[v]>=A.GEN2_MIN),'copy_fraction':float((after[a]>=.75).mean()),'faithful':bool(g2[v]>=.3 and (after[a]>=.75).mean()>=.5),'offspring_periods':' '.join(map(str,sorted(set(ps)))),'allele_aligned_frac':float(inherited.mean()),'allele_at_original_pos_frac':float((off[a,site]==v).mean()),'causally_changed_fraction':float(changed.mean()),'causal_inherited_frac':float((inherited & changed).mean()) if v!=int(x[site]) else np.nan})
        for q in range(N):
            j=v*N+q
            enc.append({'key':key,'value':v,'partner':q,'partner_tape':hx(Q[q]),'offspring':hx(off[j]),'offspring_period':ps[q],'best_shift':int(shifts[j]),'similarity':float(after[j]),'inherited_aligned':bool(inherited[q]),'inherited_original_pos':bool(off[j,site]==v)})
    write(f'alleles_{key}.csv',rows);write(f'allele_encounters_{key}.csv',enc)
    write(f'allele_execution_{key}.csv',[{'key':key,'executor':A.executor_file(L),'budget':128,'partners':N,'values':256,'wall_s':time.monotonic()-t0}])
    print(f'EXPLORATORY: {key} 256-allele panel {time.monotonic()-t0:.2f}s',flush=True)


def offspring(src,L,stage):
    timeout();t0=time.monotonic()
    from algocell_exp import executor as E
    c=pd.read_csv(OUT/'cells.csv',low_memory=False);name={'classes':'population/classes_cells.csv','6502':'s6502/v2/s1_cells.csv'}[src]
    d=c[(c.source==name)&(c.L==L)]
    if stage:d=d[d.stage==stage] if src=='classes' else d[d.arm==stage]
    if not len(d):return
    unique=d.drop_duplicates('tape');tapes=np.array([parse(t) for t in unique.tape])
    Q=np.random.default_rng([BT.SEED,L,751]).integers(0,256,(2,L),dtype=np.uint8)
    par=np.repeat(tapes,2,0);pairs=np.concatenate([par,np.tile(Q,(len(tapes),1))],1)
    if src=='classes':
        from make_conds import ABLATIONS
        from population_sample import STAGES
        _,_,zh,abl=STAGES[stage]
        sup=list(ABLATIONS[abl]) if abl else []
        mem=A.execute_pairs(pairs,L,128,sup,zero_halts=zh);executor=A.executor_file(L,zero_halts=zh)
    else:
        m={'A':'m6502_pair_maps_L16','B':'m6502_pair_maps_L32','C':'m6502_pair_maps_L16','D':'m6502_brkhalt_maps_L16'}[stage]
        ex=E.ZilionExecutor(m)
        with E.using(ex):mem=A.execute_pairs(pairs,L,128,zero_halts=stage=='D')
        executor=m
    rows=[]
    for i,t in enumerate(unique.tape):
        ps=[periods(hx(o))[0] for o in mem[2*i:2*i+2,L:]]
        rows.append({'source':name,'stage_or_arm':stage,'L':L,'tape':t,'offspring_period_min':min(ps),'offspring_period_max':max(ps),'offspring_periods':' '.join(map(str,ps)),'offspring_tape_0':hx(mem[2*i,L:]),'offspring_tape_1':hx(mem[2*i+1,L:]),'executor':executor,'budget':128,'partners':2,'wall_s':time.monotonic()-t0})
    write(f'offspring_{src}_{stage}_L{L}.csv',rows)
    print(f'EXPLORATORY: {src} {stage} L{L}, {len(tapes)} unique tapes, {time.monotonic()-t0:.2f}s',flush=True)


def offset_examples():
    """Two pre-identified exceptions only, not a new population scan."""
    timeout();t0=time.monotonic()
    c=read(RESULTS/'offset/offset_cells.csv');rows=[]
    examples=[('unit-cell expands under mutation',c[(c.L==32)&(c.offset==8)&c.tape.map(lambda t:periods(t)[0]==4)].iloc[0]),
              ('copy gate deficit',c[(c.L==16)&(c.seed==4004)&(c.offset==8)&(c.n_sites==0)].iloc[0])]
    for name,r in examples:
        x=parse(r.tape);L=len(x);d=int(r.offset)
        vals=[8,96,127,255]
        mut=np.array([x.copy() for i in range(4,d) for v in vals])
        pos=np.repeat(np.arange(4,d),len(vals));val=np.tile(vals,d-4)
        mut[np.arange(len(mut)),pos]=val
        T=np.repeat(mut,16,0);wt=np.tile(x,(16,1))
        P=np.random.default_rng([BT.SEED,L,761])
        present=np.ones(len(T),bool);alive=np.ones(len(T),bool)
        for g in range(1,9):
            q=P.integers(0,256,(16,L),dtype=np.uint8)
            mem=A.execute_pairs(np.concatenate([T,np.tile(q,(len(mut),1))],1),L,128,zero_halts=r.stage=='I')
            off=mem[:,L:];cp=A._best_shift_match_rows(off,T)>=.75
            present &= off[np.arange(len(T)),np.repeat(pos,16)]==np.repeat(val,16)
            alive &= cp
            woff=A.execute_pairs(np.concatenate([wt,q],1),L,128,zero_halts=r.stage=='I')[:,L:]
            if g in BT.GENS_REPORT:
                for m,(i,v) in enumerate(zip(pos,val)):
                    a=slice(m*16,(m+1)*16)
                    rows.append({'example':name,'stage':r.stage,'L':L,'seed':r.seed,'snapshot':r.snapshot,'parent_tape':hx(x),'parent_period':periods(hx(x))[0],'offset':d,'site':int(i),'value':int(v),'gen':g,'continuous_verbatim_fraction':float(present[a].mean()),'continuous_copy_gate_fraction':float(alive[a].mean()),'offspring_periods':' '.join(map(str,sorted({periods(hx(t))[0] for t in off[a]}))),'wildtype_g1_similarity':float(A._best_shift_match_rows(woff,np.tile(x,(16,1))).mean()) if g==1 else np.nan,'offspring_example':hx(off[m*16])})
            T=off;wt=woff
    write('offset_examples.csv',rows)
    write('offset_example_execution.csv',[{'examples':2,'budget':128,'partners':16,'values_per_site':4,'generations':8,'wall_s':time.monotonic()-t0}])
    print(f'EXPLORATORY: two offset exception interventions {time.monotonic()-t0:.2f}s',flush=True)


def report():
    old=pd.read_csv(OUT/'inputs.csv')
    INPUTS.update(dict(zip(old.path,old.sha256)))
    c=read(OUT/'cells.csv');f=read(OUT/'first_groups.csv');pr=pd.concat([read(OUT/f'prediction_{key}.csv') for key in KEYS])
    reg=read(RESULTS/'branching/branching_summary.csv')
    preds=pr.merge(reg[['key','arm','gen','sites']].rename(columns={'sites':'n_sites'}),on=['key','arm','gen'],how='left')
    write('prediction_comparison.csv',preds)
    site_preds=pd.concat([read(p) for p in sorted(OUT.glob('predicted_sites_*.csv'))])
    sites=read(RESULTS/'branching/branching_sites.csv')
    site_preds=site_preds.merge(sites[['key','arm','gen','pos','transmissible','executed']],on=['key','arm','gen','pos'],how='left',validate='many_to_one')
    site_preds['disagreement']=site_preds.predicted!=site_preds.transmissible
    write('site_prediction_comparison.csv',site_preds)
    first_scores=pd.concat([read(p) for p in sorted(OUT.glob('first_encounter_sites_*.csv'))])
    write('first_encounter_sites.csv',first_scores)
    offspring_files=sorted(OUT.glob('offspring_classes_*.csv'))+sorted(OUT.glob('offspring_6502_*.csv'))
    if offspring_files:
        o=pd.concat([read(p) for p in offspring_files]);write('offspring_periods.csv',o)
        # Both stages and arms have their own rule; preserve those distinctions on joining.
        c['stage_or_arm']=c.stage.fillna(c.arm)
        offset_o=o[o.source=='population/classes_cells.csv'].copy();offset_o['source']='offset/offset_cells.csv'
        joined=pd.concat([o,offset_o])
        c=c.merge(joined.drop(columns=['status','wall_s']),on=['source','stage_or_arm','L','tape'],how='left',validate='many_to_one')
        c['offspring_both_small']=c.offspring_period_max<c.L
        c['offspring_period_varies']=c.offspring_period_min!=c.offspring_period_max
        write('cells_with_offspring.csv',c)
        measured=c[c.offspring_period_max.notna()].copy()
        write('offspring_groups.csv',measured.groupby(['source','confined','family'],dropna=False).agg(
            scan_rows=('tape','size'),parent_small_units=('small_unit','sum'),
            both_offspring_small=('offspring_both_small','sum'),partner_period_varies=('offspring_period_varies','sum'),
            median_offspring_period=('offspring_period_max','median')).reset_index())
    lines=['# EXPLORATORY — crystal / unit-cell / copy-provenance checks','',
           '**EXPLORATORY throughout: post hoc, nothing registered.** Generated by `experiments/crystal_check.py` from the CSVs in this directory. Existing registered numbers are comparators, not registrations for this analysis.','',
           '## EXPLORATORY H1 — not addressable as a complete quantitative X1 explanation; local forecast contradicted','',
           'Differential encounters use the existing X1 default Algocell executor shaders from the checked-out base commit 4cdb503, 128 instructions, the serial panel definitions, X1\'s 32 seeded alternative values/site, and 16 uniform partners (arm U distribution). Mutant and control encounters share partners. The observational shader records every opcode/prefix fetch, operand/displacement fetch, non-fetch data read and write. All differential and wild-type final memories are checked against the uninstrumented executor; all mutant fetch masks also match the existing `exectrace` tooling. Roles are address roles during the offspring\'s own next encounter; self modification can change the value occupying an address. `probes_*` records every intervention, copy gate and mutant window; `provenance_*` records changed and verbatim edges and the corresponding address roles in their next execution; `roles_*` records every offspring position.','',
           'The forecast freezes the wild-type local map: an edge exists when >=half sampled values copy verbatim at a particular destination/context; compose that map, and require survival in >=half contexts. The strict forecast removes opcode destinations at each child generation; the transmission-only forecast does not. Neither includes X1\'s branching survival, 75% copy gate, five-byte window, background subtraction, cap or fresh context resampling. Neither is an invariant-family certificate. Those omissions are explicit mechanisms by which a local map can fail to predict X1.','',
           '| EXPLORATORY key | g | paths only | no opcode | registered X1 U |','|---|---:|---:|---:|---:|']
    for (key,g),e in preds.groupby(['key','gen']):
        a=e[e.prediction=='verbatim paths only'].iloc[0];b=e[e.prediction!='verbatim paths only'].iloc[0]
        lines.append(f'| {key} | {g} | {int(a.sites)} | {int(b.sites)} | {int(a.n_sites)} |')
    lines+=['','A separate first-encounter diagnostic adds the 75% copy gate and five-byte window, conditioning on copy survivors and requiring >=half sampled values/site, with no background/branching correction. Its g1 site counts are '+', '.join(f'{key}: {int(d.site_without_background.sum())}' for key,d in first_scores.groupby('key'))+'. Per-site disagreements with X1 are preserved in `site_prediction_comparison.csv`. This diagnostic explains why isolated byte equality is too permissive for genome20: positions 6–14 preserve their complete windows in every sampled encounter, whereas the displacement operands can match a byte without reproducing their mutant window or a viable lineage. It is still a different assay from X1.']
    lines+=['','Arm K is a registered comparison only: pusher64 32/14/0/0; no K-context map was measured. The U map cannot quantitatively explain K. Opcode-fetched and operand-fetched are distinct; the older `exectrace` annotation calls both “executed”. A verbatim differential response is local causal evidence, not proof that one source supplies the byte for every value and descendant. Changes to opcodes/addresses can produce nonlinear changed positions. Rare working core alleles are not ruled out by a zero thresholded site score.','',
            '## EXPLORATORY H2 — contradicted as a universal equality; restricted offset count supported','',
            'A cyclic period p is the smallest divisor of L whose prefix tiles the tape. Every finite tape has trivial cyclic period L; “periodic” without p<L is unfalsifiable. `finite_period` separately uses the finite-factor definition T[i]=T[i+p] without a divisibility requirement. Sources remain separate: repeated tapes, snapshots, generation endpoints and the alternative 6502 64-partner scoring are not independent replicates.','']
    off=c[c.source=='offset/offset_cells.csv'];eligible=off[off.offset_divides==True]
    lines.append(f'Offset table: {len(off)} rows; {len(eligible)} have 0<d<=L and d|L; {int((eligible.n_unexec==eligible.offset-4).sum())}/{len(eligible)} have n_unexec=d−4; {int((eligible.n_sites==eligible.offset-4).sum())}/{len(eligible)} have total sites=d−4. These are scan scores, not all-value serial capacities. A core motif plus dividing d alone does not establish all Proposition 8 prerequisites.')
    lines.append('')
    bad=eligible[eligible.n_unexec!=eligible.offset-4]
    for _,r in bad.iterrows():lines.append(f'- EXPLORATORY deficit: {r.stage}, L={int(r.L)}, seed {int(r.seed)}, {r.snapshot}: p={int(r.period)}, d={int(r.offset)}, n_unexec={int(r.n_unexec)}, predicted={int(r.offset)-4}; `{r.tape}`. Prefix tiling agrees at {sum(parse(r.tape)==np.tile(parse(r.tape)[:int(r.offset)],int(r.L/r.offset)))}/{int(r.L)} original positions; the 75% viability gate can reject the projected offspring. This is a gate diagnosis, not proof of loss of eligible payload.')
    excess=eligible[eligible.n_sites>eligible.offset-4]
    lines.append(f'- EXPLORATORY extra total sites: {len(excess)} rows; all have predicted unexecuted counts: {bool((excess.n_unexec==excess.offset-4).all())}. Their excess is executed-site transmission under the sampled scorer; per-site masks/alleles are absent from this table.')
    outside=off[off.offset_divides!=True]
    lines.append(f'- EXPLORATORY excluded offset geometry: {len(outside)} rows; fully listed in `cell_exceptions.csv` (nondividing offsets or d>L).')
    nested=eligible[eligible.period<eligible.offset]
    for _,r in nested.iterrows():
        lines.append(f'- EXPLORATORY minimal-period counterexample: {r.stage}, L={int(r.L)}, seed {int(r.seed)}, {r.snapshot}, `{r.tape}` has p={int(r.period)}, d={int(r.offset)}, and {int(r.n_unexec)} unexecuted scored sites. The minimal four-byte unit is entirely core, predicting zero under p−c, whereas the copying prefix has eight bytes and four free positions. Mutable capacity is set by the invariant copying family, not the minimal period of its unmutated member.')
    ex=read(OUT/'offset_examples.csv')
    for name,e in ex.groupby('example'):
        g8=e[e.gen==8]
        lines.append(f'- EXPLORATORY targeted validation ({name}): four payload values/site, 16 partners, eight transfers. Minimum continuous verbatim fraction at g8={g8.continuous_verbatim_fraction.min():.3f}; minimum continuous 75% copy-gate fraction={g8.continuous_copy_gate_fraction.min():.3f}. Offspring periods at g8: {", ".join(map(str,sorted(g8.offspring_periods.unique())))}. Wild-type first-transfer similarity={e[e.gen==1].wildtype_g1_similarity.iloc[0]:.3f}. Full interventions in `offset_examples.csv`.')
    lines+=['','For generic population cells the tables provide fetched union e, not the fetched residue set. The admissible payload interval is [p−min(p,e), p−ceil(e/(L/p))]. With p=L this is exact; with smaller p it is a bound. Serial tables have explicit fetched bitmaps, which are collapsed into unit-cell residues exactly. “Equal” is compatibility of sampled unexecuted counts with a fetched-only model, not evidence that unfetched positions persist for arbitrary values. Copy-source reads are explicitly excluded from this fetched-only formula: subtracting every “used” position would incorrectly remove the payload being copied. Address/count/control use cannot be reconstructed from aggregate counts.','',
            '| EXPLORATORY family/panel | cells | small units | exact equality | deficit | excess | mask missing / compatible |','|---|---:|---:|---:|---:|---:|---:|']
    core=c.family.isin(['XX 5e ed b0','1e NN ed b0'])
    for label,d in [('non-offset confined population',c[(c.source=='population/classes_cells.csv')&(c.confined==True)&~core]),('6502 final primary',c[c.source=='s6502/v2/s1_cells.csv']),('6502 final 64-partner sensitivity',c[c.source=='s6502/v2/s1_cells_64.csv'])]:
        lines.append(f'| {label} | {len(d)} | {int(d.small_unit.sum())} | {int((d.test=="equal (exact from aggregate)").sum())} | {int((d.test=="deficit").sum())} | {int((d.test=="excess").sum())} | {int(d.test.str.contains("missing|compatible").sum())} |')
    for key in ('ldir32','genome20_s6003'):
        role=read(OUT/f'roles_{key}.csv');role=role[role.phase=='wild-type parent']
        lines.append(f'\nEXPLORATORY observed read roles, {key}: {int((role.data_read_frac==1).sum())}/{len(role)} parent positions are data-read in every sampled encounter; {int(((role.opcode_frac+role.operand_frac)>0).sum())} are instruction-fetched.')
    lines.append('\nEXPLORATORY subtracting all read positions would predict no free payload for genome20 despite its nine persistent X1 sites. Its p=20 minus six fetched positions also overpredicts the nine sites: five unfetched tail coordinates are not protected by copying. Unfetched status alone is insufficient.')
    lines+=['','Every non-equal or unaddressable record, additional fetched-site transmission, and offset/minimal-period discrepancy is classified in `cell_exceptions.csv`; family/stage/source aggregates are in `cell_groups.csv`. Generic deficits/excesses remain mechanistically unresolved: aggregate site scores do not identify copying geometry, control use, copy-gate failure or long-term persistence. Missing masks and missing counts are separate from actual deficits/excesses. `cells.csv` covers all listed Z80 mutation, population and serial scan endpoints plus both 6502 final tables. Mutation-table unexecuted counts are joined from the archived trace table; the two partner panels differ, so they are annotations, not identical interventions.','']
    if offspring_files:
        o=read(OUT/'offspring_periods.csv')
        lines.append(f'Cheap offspring check: {len(o)} unique tape/rule combinations executed against two fixed uniform partners; `offspring_periods.csv` gives both actual tapes and periods. This is a sampled output check, not a partner-universal periodicity certificate. It does not change which rows are eligible for the theorem.')
        lines+=['','| EXPLORATORY offspring panel | scanned rows measured | both offspring p<L | period differs between partners | median offspring p |','|---|---:|---:|---:|---:|']
        for label,d in [('non-offset confined population',c[(c.source=='population/classes_cells.csv')&(c.confined==True)&~core]),('offset population',c[c.source=='offset/offset_cells.csv']),('6502 primary',c[c.source=='s6502/v2/s1_cells.csv'])]:
            d=d[d.offspring_period_max.notna()]
            lines.append(f'| {label} | {len(d)} | {int(d.offspring_both_small.sum())} | {int(d.offspring_period_varies.sum())} | {d.offspring_period_max.median():g} |')
        lines+=['','Detailed motif-family counts are in `offspring_groups.csv`. Applying the parent\'s fetched union to a changed offspring is not justified; the offspring\'s per-site machinery/data-read map is not archived for the full population. Thus the universal *persistent* law cannot be established by these single-generation site scans, even where the count happens to agree.']
    lines+=['','## EXPLORATORY H3 — contradicted as a universal statement and as lethal-first large transmitters; small-first tendency supported','',
            'Z80 first tapes come from the Stage G/I scored world tables. Earliest closed-heritable Z80 cells are all scanned confined cells at the earliest sampled snapshot in each world, not the exact first arrival or a resolved ancestral transition. One Stage G world has no sampled confined cell. 6502 first tapes come from corrected S1 v2 scored outputs, all four arms. Where the first tape is confined and t_rep=t_closed, it is also the earliest scored closed tape; no payload site scan exists for the 6502 first tape. For the single open-first arm C world, the earliest closed tape is recovered from the scorer\'s chronological insertion-order `judged_tapes` cache (same rule as `score6502.endpoints`), at t_closed. Censored and full-period exceptions are listed in `first_exceptions.csv`.','',
            '| EXPLORATORY panel | endpoint | cells | worlds | p<L | median p | range |','|---|---|---:|---:|---:|---:|---|']
    for _,r in f.iterrows():lines.append(f'| {r.dataset} | {r.endpoint} | {int(r.cells)} | {int(r.worlds)} | {int(r.small_units)} | {r.median_period:g} | {r.min_period:g}–{r.max_period:g} |')
    fc=read(OUT/'first_cells.csv');i=fc[(fc.dataset=='Z80 I')&(fc.endpoint=='earliest sampled closed-heritable')]
    lines.append(f'\nEXPLORATORY lethal Z80 earliest sampled closed cells: {int((i["class"]=="transmitter").sum())}/{len(i)} have >=5 scored sites; {int((i.period==i.L).sum())}/{len(i)} have full-length unit cells. A world can contribute multiple sampled cells; `first_cells.csv` preserves all exceptions and stages. “Small” here means nontrivial p<L; the full period distributions are retained, rather than choosing an accommodating size cutoff.')
    lethal=fc[(fc.dataset=='Z80 I')&(fc.endpoint=='first scored closed-heritable (same first tape)')]
    lines.append(f'\nEXPLORATORY the earlier scored Stage I first tapes are already confined in {len(lethal)}/10 worlds, all with p=4 or 8 and total scored sites <=4; {int((lethal.n_sites>=5).sum())}/10 are transmitters by the >=5-site class. Thus “lethal tar first produces large-unit transmitters” is contradicted by these first tapes, even though later earliest population snapshots frequently contain transmitters. The 6502 exception to universally small first tapes is arm B: two detected first tapes have p=L=32; the other two have p=8. Censoring is retained for the remaining 16 B worlds and two D worlds.')
    lines+=['','## EXPLORATORY H4 — supported for a restricted working-allele set; genome20 outside Proposition 8','',
            'All 256 settings were tested at position 0 of ldir32 and position 2 (LD DE low immediate) of the actual genome20_s6003 tape `5e 11 04 c5 ed b8 …`. The latter is an LDDR transmitter with six initial core bytes and 16-bit source wrap, not `XX 5e ed b0`; applying d−4 to it would test the wrong parent. `initial_ring_displacement` is DE mod 2L before copying; for LDDR `effective_d` is the ring displacement after HL first wraps to 0xffff. For its original value 04 these differ: 36 initially, then 20. That discontinuity prevents assuming a constant modulo-ring copy offset.','',
            'Every allele has one first encounter per member of a fixed 16-uniform-partner panel. Culture self-copying uses the existing two-generation normalised gain and threshold gen2>=0.3, so a second encounter against a second fixed uniform panel is required; all raw first offspring and periods are preserved. Faithfulness additionally requires >=half offspring be >=75% copies. Inheritance is both original-position and best-shift-aligned byte equality, with changed-output comparison to wild type. Equality alone can be incidental or partner-sourced; the differential flag supports a local causal effect, not a full multigeneration parameter certificate.','']
    for key in ('ldir32','genome20_s6003'):
        a=read(OUT/f'alleles_{key}.csv');w=a[a.culture_replicator]
        e=w[w.prop8_eligible];inherit=w[w.allele_aligned_frac>=.5]
        lines.append(f'- EXPLORATORY {key}: {len(w)}/256 culture-working alleles; {int(w.faithful.sum())} faithful; {len(inherit)} working with >=half aligned allele equality; {len(e)} working satisfying sampled Proposition 8 prerequisites. Working values (hex): '+', '.join(w.hex.astype(str)))
        if len(e):lines[-1]+=f' Eligible d range {int(e.effective_d.min())}–{int(e.effective_d.max())}; predicted fixed-core payload range {int(e.predicted_payload.min())}–{int(e.predicted_payload.max())}; values and exclusions in `alleles_{key}.csv`.'
        if key.startswith('genome'):lines[-1]+=' Every Prop 8 payload prediction is missing by design because its forward-prefix conditions fail. `nominal_d_minus_c` is recorded as arithmetic only, not promoted to an inherited payload prediction. The working range must therefore be described by measured offspring periods/allele inheritance, not that proposition.'
    lines+=['','## EXPLORATORY reproducibility and limits','',
            'CPU computed table periods, joins, bounds and summaries. Local GPU commands reuse the assay executor and append observational role bits only; each command has a 58 s alarm and each role dispatch checks elapsed time. No new worlds or large site scans. `execution_*` records source/derived shader hashes and equality-check counts; `allele_execution_*` and offspring rows record executor and runtime. `inputs.csv` hashes the result tables consumed to generate the report. Original archive root `/Users/neo/repos/algocell/experiments/runs` remained read-only; these checks use scored tables and no raw archive writes.','',
            'Reproduce from the worktree `experiments` directory with `/Users/neo/repos/algocell/experiments/.venv/bin/python crystal_check.py tables`; then `probe --key KEY` for the five X1 keys, `alleles --key KEY` for ldir32/genome20_s6003, `offset-examples`, and `offspring --source classes --L L --stage STAGE` per observed stage/length, or `offspring --source 6502 --L L --stage ARM`. Finish with `report`. Report-only generation requires no GPU.']
    (OUT/'CRYSTAL.md').write_text('\n'.join(lines)+'\n')
    manifest()
    print('EXPLORATORY: report generated.',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['tables','probe','alleles','offspring','offset-examples','report'])
    ap.add_argument('--key',choices=KEYS)
    ap.add_argument('--source',choices=['classes','6502'],default='classes')
    ap.add_argument('--L',type=int,default=16)
    ap.add_argument('--stage',default='')
    args=ap.parse_args();OUT.mkdir(exist_ok=True)
    if args.mode=='tables':tables()
    elif args.mode=='probe':probe(args.key)
    elif args.mode=='alleles':alleles(args.key)
    elif args.mode=='offspring':offspring(args.source,args.L,args.stage)
    elif args.mode=='offset-examples':offset_examples()
    else:report()


if __name__=='__main__':main()

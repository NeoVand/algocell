"""C2 evolved panel: qualify -> commit ledger -> assay -> commit CSVs -> report."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from algocell_exp import assay as A
from algocell_exp.census8086_trace import same_execution
from algocell_exp.zmachine import load_machine
from capacity_common import HERE, SEED, UNKNOWN, Bursts, committed, fresh, hx, parse, partners, sha, write
from evolved6502_trace import Trace6502
from serial_retention import window_codes

OUT=HERE/'results/evolved6502'
V2=HERE/'results/s6502/v2'


def sample(archive):
    rows=[];coverage=[]
    worlds=pd.read_csv(V2/'s1_worlds.csv',keep_default_na=False)
    finals=pd.read_csv(V2/'final_cultures.csv')
    # The table repeats each of the 64 draws under 32- and 64-partner
    # evaluations. Use corrected S1's frozen32 primary calls, not 128 draws.
    assert (finals.groupby(['world','cell']).tape.nunique()==1).all()
    finals=finals[finals.partners==32]
    for w in worlds.to_dict('records'):
        common={k:w[k] for k in ('world','arm','seed','L','machine')};common['archived_heritable']=''
        if w['first_tape']:
            rows.append({**common,'archived_heritable':1,'role':'first','cell':-1,'tape':w['first_tape'],'key':w['world']+':first'})
        selected=finals[finals.world==w['world']].to_dict('records')
        selected.sort(key=lambda r:sha(f'C2:{SEED}:{r["world"]}:{r["cell"]}'.encode()))
        assert len(selected)==64
        for f in selected[:8]:
            rows.append({**common,'archived_heritable':int(f['heritable']),'role':'final_random','cell':f['cell'],'tape':hx(parse(f['tape'])),'key':f'{w["world"]}:cell{f["cell"]}'})
        log=archive/(w['world']+'.jsonl')
        expected=json.loads((V2/'worlds'/f'{w["world"]}.json').read_text())['input_sha256']
        available=log.exists()
        if available:
            assert sha(log.read_bytes())==expected,'raw log does not match corrected S1 input'
            records=[json.loads(x) for x in log.read_text().splitlines()]
            last=[x for x in records if x.get('kind')=='sample'][-1]
            assert last['step']==10_000_000
            for rank,t in enumerate(last['top3']):
                tape=t['tape'];assert sha(bytes.fromhex(tape))==t['sha256'] if 'sha256' in t else True
                rows.append({**common,'role':'final_top3','cell':rank,'tape':hx(parse(tape)),'key':f'{w["world"]}:top{rank}'})
        coverage.append({'world':w['world'],'arm':w['arm'],'first_available':int(bool(w['first_tape'])),'random_draws':8,'final_top3_available':int(available),'raw_log_sha256':expected,'raw_path':str(log)})
    for r in rows:
        # Reuse the registered WORLD:CELL hash for final-draw ordering and
        # descriptive selection; first tapes use their declared cell=-1.
        sampling_key=f'{r["world"]}:{r["cell"]}' if r['role']!='final_top3' else r['key']
        r['sampling_hash']=sha(f'C2:{SEED}:{sampling_key}'.encode())
    write(OUT/'panel.csv',rows);write(OUT/'coverage.csv',coverage)
    write(OUT/'inputs.csv',[{'path':str(p.relative_to(HERE)),'sha256':sha(p.read_bytes())} for p in (V2/'s1_worlds.csv',V2/'final_cultures.csv')])
    return rows


def geometry(log,labels,L,opcodes):
    if log is None:return {'eligible':0,'reason':'trace_overflow','d':'','core_c':'','core_positions':'','free_positions':'','predicted_free':'','directions':'','copy_writes':0,'trace_sha256':''}
    writes=log[log[:,1]==2];reason=[];d=None;directions='';free=[];footprint=set()
    if not len(writes):reason.append('no_writes')
    # A prelude may write outside the protected seed. The qualifying operation
    # starts with the first proven verbatim copy; subsequent writes must all be
    # proven copy channels, not incidental equal bytes.
    pure=(writes[:,4]!=UNKNOWN)&(writes[:,5]!=UNKNOWN)
    if not pure.any():reason.append('no_proven_copy_channel')
    else:
        first=int(np.flatnonzero(pure)[0]);episode=writes[first:]
        if not pure[first:].all():reason.append('interfering_or_unresolved_write')
        src=episode[:,5].astype(np.int64);dst=episode[:,2].astype(np.int64)
        ds=(dst-src)%(2*L);unique=np.unique(ds)
        if len(unique)==1:d=int(unique[0])
        else:reason.append('nonconstant_displacement')
        directions='source:'+','.join(map(str,np.unique(np.diff(src)%(2*L))))+';destination:'+','.join(map(str,np.unique(np.diff(dst)%(2*L))))
        if not (np.all(np.diff(src)%(2*L)==1) and np.all(np.diff(dst)%(2*L)==1)):reason.append('not_sequential_forward')
        if d is not None:
            if not (0<d<=L and L%d==0):reason.append('nondividing_or_out_of_range')
            if src[0]!=0 or dst[0]!=d:reason.append('entry_phase_not_primary_seed')
            if len(episode)<2*L-d:reason.append('insufficient_coverage')
            if (writes[:first,2]<d).any():reason.append('prelude_changes_seed')
            if not np.array_equal(labels[L:],np.arange(L)%max(d,1)):reason.append('not_primary_seed_provenance')
            fetch=log[log[:,1]==0]
            footprint=set(map(int,fetch[:,4]))
            # The only accepted control loop uses flags independent of payload:
            # BVC/BVS, absolute JMP, or a straight-line fixed episode. Unknown
            # opcodes remain outside the proof certificate, even with equal bytes.
            safe={'LDA','STA','STX','STY','PHA','PLA','TSX','INX','DEX','INY','DEY','JMP','BVC','BVS','NOP','CMP','CPX','CPY','CLC','SEC','CLV','CLI','SEI','CLD','SED'}
            for step,kind,pos,op,origin,source in log[log[:,1]==3]:
                info=opcodes[int(op)];name=info['mnemonic']
                if name not in safe:reason.append('uncertified_control_or_address_dependency')
                if name=='JMP' and info.get('mode')=='ind':reason.append('indirect_control_dependency')
                if info.get('mode') in ('izx','izy'):
                    # The zero-page pointer is machinery, even when it was
                    # never fetched. Freeze its causal seed coordinates too.
                    pointer_reads=log[(log[:,0]==step)&(log[:,1]==1)][:2]
                    if len(pointer_reads)!=2:reason.append('unresolved_pointer_dependency')
                    footprint.update(map(int,pointer_reads[:,4]))
            if UNKNOWN in footprint or any(p>=d for p in footprint):reason.append('footprint_outside_protected_prefix')
            if np.any((log[:,1]<=1)&(log[:,4]>=L)&(log[:,4]!=UNKNOWN)):reason.append('original_partner_inflow')
            free=sorted(set(range(d))-footprint)
    return {'eligible':int(not reason),'reason':';'.join(sorted(set(reason))),'d':d if d is not None else '',
            'core_c':len(footprint) if d is not None else '', 'core_positions':';'.join(map(str,sorted(footprint))),
            'free_positions':';'.join(map(str,free)), 'predicted_free':len(free) if d is not None else '', 'directions':directions,
            'copy_writes':int(pure.sum()) if len(writes) else 0,'trace_sha256':sha(log.tobytes())}


def qualify(archive):
    OUT.mkdir(parents=True,exist_ok=True);rows=sample(archive);burst=Bursts(OUT)
    opcodes=json.loads((HERE/'results/census6502/m6502_opcodes.json').read_text())
    if isinstance(opcodes,dict):opcodes=opcodes.get('instructions',opcodes.get('opcodes'))
    evidence=[];ledger=[]
    event_file=gzip.open(OUT/'geometry_traces.csv.gz','wt',newline='')
    event_writer=csv.DictWriter(event_file,fieldnames=['key','partner','step','kind','position','byte','origin','copy_source_address'])
    event_writer.writeheader()
    for machine in sorted(set(r['machine'] for r in rows)):
        base=burst.run('compile-'+machine,lambda:load_machine(machine))
        trace=burst.run('trace-compile-'+machine,lambda:Trace6502(base))
        group=[r for r in rows if r['machine']==machine];L=int(group[0]['L'])
        for lo in range(0,len(group),16):
            chunk=group[lo:lo+16];T=np.stack([parse(r['tape']) for r in chunk]);Q=partners(L)
            pairs=np.concatenate((np.repeat(T,17,0),np.tile(Q,(len(T),1))),1)
            raw=burst.run('geometry-raw-'+machine,lambda:base.run(pairs,budget=128))
            result=burst.run('geometry-trace-'+machine,lambda:trace.run(pairs,budget=128))
            assert same_execution(raw,result,base.record_words),'instrumentation changed 6502 execution'
            logs=trace.logs(result);labels=trace.labels(result)
            for j,row in enumerate(chunk):
                gs=[]
                for p in range(17):
                    ix=j*17+p;g=geometry(logs[ix],labels[ix],L,opcodes);gs.append(g)
                    evidence.append({'key':row['key'],'partner':p,**g})
                    if logs[ix] is not None:
                        event_writer.writerows({'key':row['key'],'partner':p,'step':int(step),'kind':int(kind),'position':int(pos),'byte':int(value),'origin':int(origin),'copy_source_address':int(source)} for step,kind,pos,value,origin,source in logs[ix])
                g=gs[0].copy();g['eligible']=int(all(x['eligible'] for x in gs) and len({(x['d'],x['core_c'],x['free_positions']) for x in gs})==1)
                g['reason']=';'.join(sorted(set(y for x in gs for y in x['reason'].split(';') if y)))
                ledger.append({**row,**g,'base_sha256':base.m['sha256']['wgsl'],'trace_shader_sha256':trace.m['sha256']['wgsl']})
            print(f'C2 geometry {machine} {min(lo+16,len(group))}/{len(group)}',flush=True)
    event_file.close()
    write(OUT/'eligibility.csv',ledger);write(OUT/'geometry.csv',evidence)
    descriptive=[]
    for arm in 'ABCD':
        for role in ('first','final_random'):
            choices=sorted([r for r in ledger if r['arm']==arm and r['role']==role and not r['eligible'] and r['archived_heritable']==1],key=lambda r:r['sampling_hash'])[:2]
            descriptive.extend(choices)
    write(OUT/'descriptive_panel.csv',descriptive)
    print(f'C2 {len(ledger)} sampled roles, eligible {sum(r["eligible"] for r in ledger)}, descriptive {len(descriptive)}',flush=True)


def descriptive_scan(row,base,burst):
    t=parse(row['tape']);L=len(t);key=row['key'];rng=np.random.default_rng([SEED,int(sha(key.encode())[:8],16)])
    positions=[];values=[];mut=[]
    for pos in range(L):
        for val in rng.choice([v for v in range(256) if v!=t[pos]],32,replace=False):
            a=t.copy();a[pos]=val;positions.append(pos);values.append(val);mut.append(a)
    T=np.stack(mut);M=len(T);pos=np.array(positions);val=np.array(values)
    codes=window_codes(T)[np.arange(M),(pos-2)%L];ident=~np.isin(codes,window_codes(t[None])[0])
    current=np.repeat(T,4,0);control=np.tile(t,(64,1));alive=np.ones(len(current),bool);ac=np.ones(64,bool);rows=[]
    for g in range(1,9):
        Q=fresh(L,key,g,len(current)+64)
        rr=burst.run(f'descriptive-{key}-g{g}',lambda:base.run(np.concatenate((np.vstack((current,control)),Q),1),budget=128))
        child=rr.memory[:,L:];alive &= A._best_shift_match_rows(child[:len(current)],current)>=.75;ac &= A._best_shift_match_rows(child[len(current):],control)>=.75
        current,control=child[:len(current)],child[len(current):]
        present=(window_codes(current)==np.repeat(codes,4)[:,None]).any(1)
        cc=window_codes(control[ac]);al=alive.reshape(M,4);pr=(present&alive).reshape(M,4)
        for i in range(M):
            background=float((cc==codes[i]).any(1).mean()) if len(cc) else 0.
            share=float(pr[i].sum()/al[i].sum()) if al[i].any() else 0.
            rows.append({'key':key,'arm':row['arm'],'role':row['role'],'generation':g,'position':int(pos[i]),'value':int(val[i]),'identifiable':int(ident[i]),'alive':float(al[i].mean()),'retained':float(pr[i].mean()),'carrying_alive':share,'background':background,'ok':int(share-background>=.5),'control_alive':float(ac.mean())})
    return rows


def assay():
    for p in (OUT/'eligibility.csv',OUT/'descriptive_panel.csv',HERE/'evolved6502_retention.py',HERE/'evolved6502_trace.py',HERE/'REVISION_PREREG.md'):committed(p)
    ledger=pd.read_csv(OUT/'eligibility.csv');eligible=ledger[ledger.eligible==1]
    if len(eligible):
        # Keep any qualifying result visible; never silently skip promised work.
        exact_assay(eligible)
    else:
        write(OUT/'law_verdict.csv',[{'eligible':0,'verdict':'no eligible architecture','scope':'observed panel; missing top-3 remains incomplete'}])
    burst=Bursts(OUT);panel=pd.read_csv(OUT/'descriptive_panel.csv');machines={};rows=[]
    for row in panel.to_dict('records'):
        name=row['machine']
        if name not in machines:machines[name]=burst.run('compile-'+name,lambda:load_machine(name))
        rows.extend(descriptive_scan(row,machines[name],burst));write(OUT/'descriptive_mutants.csv.gz',rows)
        print('C2 descriptive complete '+row['key'],flush=True)


def exact_assay(eligible):
    # Generic seed family, arbitrary noncontiguous code footprint, frozen d/c.
    burst=Bursts(OUT);outputs=[];serial=[]
    for row in eligible.to_dict('records'):
        base=burst.run('exact-compile',lambda:load_machine(row['machine']));trace=burst.run('exact-trace-compile',lambda:Trace6502(base))
        t=parse(row['tape']);L=len(t);d=int(row['d']);free=[int(p) for p in str(row['free_positions']).split(';') if p!='nan' and p];cases=[]
        for kind,ps in [('single',free),('erased',range(d,L))]:
            for p in ps:
                for v in range(256):
                    if v==int(t[p]):continue
                    a=t.copy();a[p]=v;cases.append((kind,p,v,a))
        rng=np.random.default_rng([SEED,int(sha(row['key'].encode())[:8],16)])
        for j in range(64):
            a=t.copy();a[free]=rng.integers(0,256,len(free),dtype=np.uint8);cases.append(('joint',-1,j,a))
        for lo in range(0,len(cases),64):
            batch=cases[lo:lo+64];T=np.stack([c[3] for c in batch]);current=T.copy();expected=np.tile(T[:,:d],(1,L//d));Q=partners(L)
            pairs=np.concatenate((np.repeat(T,17,0),np.tile(Q,(len(T),1))),1)
            raw=burst.run('exact-raw',lambda:base.run(pairs,budget=128));r=burst.run('exact-trace',lambda:trace.run(pairs,budget=128));assert same_execution(raw,r,base.record_words)
            off=r.memory[:,L:].reshape(len(T),17,L);labels=trace.labels(r)[:,L:].reshape(len(T),17,L)
            for j,(kind,p,v,a) in enumerate(batch):outputs.append({'key':row['key'],'kind':kind,'position':p,'value':v,'exact_passed':int((off[j]==expected[j]).all(1).sum()),'provenance_passed':int((labels[j]==np.arange(L)%d).all(1).sum()),'partners':17})
            for g in range(1,9):
                Q=fresh(L,f'{row["key"]}:{lo}',g,len(T));pairs=np.concatenate((current,Q),1)
                raw=burst.run('serial-raw',lambda:base.run(pairs,budget=128));r=burst.run('serial-trace',lambda:trace.run(pairs,budget=128));assert same_execution(raw,r,base.record_words)
                current=r.memory[:,L:];labels=trace.labels(r)[:,L:]
                for j,(kind,p,v,a) in enumerate(batch):serial.append({'key':row['key'],'kind':kind,'position':p,'value':v,'generation':g,'exact':int(np.array_equal(current[j],expected[j])),'provenance_exact':int(np.array_equal(labels[j],np.arange(L)%d)),'offspring':hx(current[j]),'sources':';'.join(map(str,labels[j]))})
        write(OUT/'exact_cases.csv.gz',outputs);write(OUT/'exact_serial.csv.gz',serial)
    bad=sum(r['exact_passed']!=17 or r['provenance_passed']!=17 for r in outputs)+sum(not(r['exact'] and r['provenance_exact']) for r in serial)
    write(OUT/'law_verdict.csv',[{'eligible':len(eligible),'verdict':'FAIL' if bad else 'PASS','scope':'observed frozen eligible panel'}])


def report():
    for name in ('eligibility.csv','coverage.csv','law_verdict.csv','descriptive_mutants.csv.gz'):committed(OUT/name)
    ledger=pd.read_csv(OUT/'eligibility.csv');cover=pd.read_csv(OUT/'coverage.csv');m=pd.read_csv(OUT/'descriptive_mutants.csv.gz');v=pd.read_csv(OUT/'law_verdict.csv');b=pd.read_csv(OUT/'bursts.csv')
    sites=m[m.identifiable==1].groupby(['key','arm','role','generation','position']).ok.mean().reset_index();sites['site']=sites.ok>=.5
    counts=sites.groupby(['key','arm','role','generation']).site.sum().reset_index()
    text=['# C2 — evolved 6502 retention','', 'Generated by `evolved6502_retention.py report` from committed CSVs. Registration: `REVISION_PREREG.md`, commit `213a653`.',
          f'Conditional law verdict: **{v.verdict.iloc[0]}**. Eligibility was committed before interventions; footprint and d were extracted from code and ordered provenance, never from observed sites.',
          f'Panel completeness: **{"complete" if cover.final_top3_available.all() else "INCOMPLETE"}**. {len(cover)} worlds, all four arms; {int(cover.first_available.sum())} first replicators; {int(cover.random_draws.sum())} final random draws; final top-3 available in {int(cover.final_top3_available.sum())}/{len(cover)} worlds. Raw logs are matched to corrected S1 input hashes. Missing logs are listed in `coverage.csv`; no substitute tapes were used.',
          f'Observed panel: {len(ledger)} sampling roles, {ledger.tape.nunique()} distinct tapes, {int(ledger.eligible.sum())} eligible. No eligible architecture is a scope result for these tapes; missing final top-3 prevents a complete-panel conclusion.',
          '', '| arm | role | sampled | eligible |','|---|---|---:|---:|']
    for (arm,role),g in ledger.groupby(['arm','role']):text.append(f'| {arm} | {role} | {len(g)} | {g.eligible.sum()} |')
    text += ['', 'Descriptive serial scan (X1-style thresholds; four mutant lineages, 64 controls, 32 alternatives/coordinate, independent fresh uniform partners). These outcomes are descriptive and do not test the exact overlap law.',
             '', '| key | role | sites g1 / g2 / g4 / g8 | control alive g8 | mutant alive g8 | retained g8 |','|---|---|---|---:|---:|---:|']
    for key,g in m.groupby('key'):
        a=g[(g.generation==8)&(g.identifiable==1)];ss=counts[counts.key==key].set_index('generation').site
        text.append(f'| {key} | {g.role.iloc[0]} | '+ ' / '.join(str(int(ss.get(x,0))) for x in (1,2,4,8))+f' | {a.control_alive.iloc[0]:.3f} | {a.alive.mean():.3f} | {a.retained.mean():.3f} |')
    text += ['', 'Exclusion counts (overlapping reasons; failures remain in the frozen ledger):']
    from collections import Counter
    reasons=Counter(reason for s in ledger.reason.fillna('') for reason in s.split(';') if reason)
    text += [f'- {reason}: {n}' for reason,n in sorted(reasons.items())]
    text += ['',f'Local active compute including compilation/readback {b.active_s.sum():.2f} s, with equal pauses; maximum burst {b.active_s.max():.2f} s. No paid compute.',
             'All traced encounters match the raw frozen 6502 machine memory and architectural record. Pure-copy source labels are propagated through loads, PLA/PHA, transfers and stores; arithmetic/unknown channels are marked unresolved. Reverse movement, nonconstant displacement and payload-dependent control do not receive a fitted d−c prediction.',
             'Files: `panel.csv`, `coverage.csv`, `inputs.csv`, `eligibility.csv`, `geometry.csv`, `geometry_traces.csv.gz`, `descriptive_panel.csv`, `descriptive_mutants.csv.gz`, `law_verdict.csv`, `bursts.csv`.']
    (OUT/'RETENTION.md').write_text('\n'.join(text)+'\n');print('\n'.join(text[:6]))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['qualify','assay','report']);ap.add_argument('--archive',type=Path,default=Path('/private/tmp/algocell-s1audit'))
    args=ap.parse_args()
    qualify(args.archive) if args.command=='qualify' else globals()[args.command]()

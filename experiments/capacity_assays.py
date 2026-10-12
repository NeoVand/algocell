"""C1: qualify, commit ledger, assay, commit CSVs, report. No paid compute."""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import pandas as pd

from capacity_common import HERE, SEED, Bursts, Z80Provenance, committed, fresh, hx, parse, partners, sha, write

OUT=HERE/'results/capacity'


def first_ok(tape, family, burst):
    L=len(tape);d=int(tape[0 if family=='XX 5e ed b0' else 1])%(2*L)
    if not (4<=d<=L and L%d==0): return False,'nondividing_or_out_of_range'
    Q=partners(L);mem=np.concatenate((np.tile(tape,(len(Q),1)),Q),1)
    r,e=burst.run(f'first-L{L}',lambda:__import__('algocell_exp.assay',fromlist=['x'])._shader_execute_pairs(mem,L,1,return_regs=True))
    bc=256*e[:,2]+e[:,3];hl=256*e[:,6]+e[:,7]
    good=(e[:,9]==(1 if family.startswith('XX') else 2)) & (hl%(2*L)==0) & ((256*e[:,4])%(2*L)==0) & ((bc==0)|(bc>=126)) & (r[:,:d]==mem[:,:d]).all(1)
    return bool(good.all()),'first_instruction_conditions' if not good.all() else ''


def qualify():
    OUT.mkdir(parents=True,exist_ok=True);burst=Bursts(OUT)
    df=pd.read_csv(HERE/'results/offset/offset_cells.csv');df['row']=range(len(df))
    exclusions=[];eligible=[];sources=[]
    # Only the first instruction differs among cells of a given L/family/value.
    cache={}
    for row in df.to_dict('records'):
        t=parse(row['tape']);family=row['family'];v=int(t[0 if family.startswith('XX') else 1]);key=(len(t),family,v)
        if key not in cache: cache[key]=first_ok(t,family,burst)
        ok,reason=cache[key]
        row.update(eligible=int(ok),reason=reason,sampling_hash=sha(f'C1:{SEED}:{row["row"]}'.encode()))
        exclusions.append({k:row[k] for k in ('row','stage','L','seed','snapshot','family','tape','eligible','reason','sampling_hash')})
        if ok: eligible.append(row)
    selected=sorted(eligible,key=lambda r:r['sampling_hash'])[:20]
    selected=[{'row':-2,'family':'XX 5e ed b0','tape':'04 5e ed b0 '*8,'key':'ldir32'},
              {'row':-1,'family':'1e NN ed b0','tape':'1e 04 ed b0 '*5,'key':'ldir20'}]+selected
    ledgers=[];trace_rows=[];parent_rows=[]
    pipes={};alleles={}
    for parent in selected:
        t=parse(parent['tape']);L=len(t);fam=parent['family'];site=0 if fam.startswith('XX') else 1;orig=int(t[site]);key=parent.get('key',f'offset_row{parent["row"]}')
        ok,reason=first_ok(t,fam,burst);assert ok,(key,reason)
        parent_rows.append({'key':key,'source_row':parent['row'],'L':L,'family':fam,'site':site,'original_value':orig,'original_d':orig%(2*L),'tape':hx(t)})
        if L not in pipes: pipes[L]=Z80Provenance(L,burst,events=1024)
        pipe=pipes[L]
        af=(L,fam)
        if af not in alleles:
            choices={}
            for d in range(4,L+1):
                if L%d: continue
                for v in range(256):
                    if v%(2*L)!=d: continue
                    a=t.copy();a[site]=v
                    valid,_=first_ok(a,fam,burst)
                    if valid: choices[d]=v;break
            alleles[af]=choices
        vs=sorted(set(alleles[af].values())|{orig})
        for v in vs:
            a=t.copy();a[site]=v;d=v%(2*L);Q=partners(L)
            mem,regs=pipe.run(np.concatenate((np.tile(a,(17,1)),Q),1))
            logs=pipe.logs(regs);expected=np.arange(L)%d
            passed=True;reasons=[]
            for p,log in enumerate(logs):
                # Reads/writes in the LDIR episode follow the same sequential
                # operation. First-instruction side effects precede that episode.
                w=log[(log[:,1]==2)&(log[:,0]>=(2 if site==0 else 1))]
                r=log[(log[:,1]==1)&(log[:,0]>=(2 if site==0 else 1))]
                forward=len(w)==len(r) and len(w)>=2*L-d and np.all(r[:,2]==np.arange(len(r))%(2*L)) and np.all(w[:,2]==(np.arange(len(w))+d)%(2*L))
                causal=np.array_equal(pipe.labels(regs)[p,L:],expected)
                core=np.array_equal(mem[p,L:L+4],a[:4])
                passed &= bool(forward and causal and core)
                for step,kind,pos,value,origin in log:
                    trace_rows.append({'key':key,'value':v,'partner':p,'step':int(step),'kind':int(kind),'position':int(pos),'byte':int(value),'source':int(origin)})
                if not forward: reasons.append('nonsequential_or_insufficient_copy')
                if not causal: reasons.append('provenance_geometry')
                if not core: reasons.append('core_not_preserved')
            ledgers.append({'edit_id':f'{key}_v{v:02x}','key':key,'L':L,'family':fam,'site':site,'original_value':orig,'value':v,'d':d,'core_c':4,'predicted_free':d-4,'free_positions':';'.join(map(str,range(4,d))),'erased_positions':';'.join(map(str,range(d,L))),'eligible':int(passed),'reason':';'.join(sorted(set(reasons))),'parent':hx(a),'base_sha256':pipe.base_hash,'trace_shader_sha256':pipe.source_hash,'trace_sha256':sha(b''.join(x.tobytes() for x in logs))})
        sources.append({'path':str(HERE/'results/offset/offset_cells.csv'),'sha256':sha((HERE/'results/offset/offset_cells.csv').read_bytes())})
    write(OUT/'candidate_eligibility.csv',exclusions)
    write(OUT/'parents.csv',parent_rows);write(OUT/'edit_eligibility.csv',ledgers)
    write(OUT/'geometry_traces.csv.gz',trace_rows)
    write(OUT/'inputs.csv',sources[:1])
    # Freeze partners explicitly, so the assay cannot resample eligibility.
    write(OUT/'partners.csv',[{'L':L,'partner':p,'tape':hx(q)} for L in pipes for p,q in enumerate(partners(L))])
    print(f'C1 frozen candidates {len(exclusions)}, parents {len(selected)}, edits {len(ledgers)}, eligible edits {sum(r["eligible"] for r in ledgers)}',flush=True)


def variants(base,d,key):
    cases=[]
    for kind,positions in [('single',range(4,d)),('erased',range(d,len(base)))]:
        for pos in positions:
            for v in range(256):
                if v==int(base[pos]): continue
                a=base.copy();a[pos]=v;cases.append((kind,pos,v,a))
    rng=np.random.default_rng([SEED,int(sha(key.encode())[:8],16)])
    for j in range(64):
        a=base.copy();a[4:d]=rng.integers(0,256,d-4,dtype=np.uint8);cases.append(('joint',-1,j,a))
    return cases


def assay():
    for name in ('edit_eligibility.csv','parents.csv','partners.csv','geometry_traces.csv.gz'):
        committed(OUT/name)
    committed(HERE/'REVISION_PREREG.md');committed(HERE/'capacity_assays.py');committed(HERE/'capacity_common.py')
    burst=Bursts(OUT);ledger=pd.read_csv(OUT/'edit_eligibility.csv');pipes={};serial=[];capacity=[]
    for row in ledger.to_dict('records'):
        if not row['eligible']: continue
        L=int(row['L']);site=int(row['site']);t=parse(row['parent']);key=row['edit_id']
        if L not in pipes: pipes[L]=Z80Provenance(L,burst)
        pipe=pipes[L];current=np.tile(t,(16,1))
        for direction,v in [('forward',int(row['value'])),('reciprocal',int(row['original_value']))]:
            current[:,site]=v;d=v%(2*L)
            expected=np.tile(current[:,:d],(1,L//d))
            for g in range(1,9):
                Q=fresh(L,f'{key}:{direction}',g,16)
                off,reg=pipe.run(np.concatenate((current,Q),1));off=off[:,L:];labels=pipe.labels(reg)[:,L:]
                _,first=pipe.run(np.concatenate((current,Q),1),steps=2 if site==0 else 1)
                actual=((256*first[:,4]+first[:,5])-(256*first[:,6]+first[:,7])).astype(np.int64)%(2*L)
                for r in range(16):
                    serial.append({'edit_id':key,'direction':direction,'lineage':r,'generation':g,'value':v,'predicted_d':d,'actual_d':int(actual[r]),'core':hx(off[r,:4]),'offspring':hx(off[r]),'sources':';'.join(map(str,labels[r])),'partner':hx(Q[r]),'exact':int(np.array_equal(off[r],expected[r])),'provenance_exact':int(np.array_equal(labels[r],np.arange(L)%d)),'core_exact':int(np.array_equal(off[r,:4],expected[r,:4]))})
                current=off
            # Each actual g8 descendant is tested directly, without replacing
            # its payload with a construction. Cases are pooled only if tapes
            # are exactly identical; every lineage remains in the CSV.
            for child_bytes in sorted(set(map(bytes,current))):
                indices=[r for r in range(16) if bytes(current[r])==child_bytes]
                base=np.frombuffer(child_bytes,np.uint8);cases=variants(base,d,f'{key}:{direction}:{child_bytes.hex()}')
                for lo in range(0,len(cases),128):
                    batch=cases[lo:lo+128];A=np.stack([c[3] for c in batch]);Q=partners(L)
                    mem=np.concatenate((np.repeat(A,17,0),np.tile(Q,(len(A),1))),1)
                    off,reg=pipe.run(mem);off=off[:,L:].reshape(len(A),17,L);labels=pipe.labels(reg)[:,L:].reshape(len(A),17,L)
                    for j,(kind,pos,val,a) in enumerate(batch):
                        expected=np.tile(a[:d],L//d);equal=(off[j]==expected).all(1);causal=(labels[j]==(np.arange(L)%d)).all(1)
                        capacity.append({'edit_id':key,'direction':direction,'lineages':';'.join(map(str,indices)),'d':d,'kind':kind,'position':pos,'value':val,'parent':hx(a),'partners':17,'exact_passed':int(equal.sum()),'provenance_passed':int(causal.sum()),'core_passed':int((off[j,:,:4]==a[:4]).all(1).sum()),'output_sha256':sha(off[j].tobytes())})
                        if not equal.all() or not causal.all():
                            write(OUT/'diagnosis.csv',[{'edit_id':key,'direction':direction,'kind':kind,'position':pos,'value':val,'partner':p,'offspring':hx(off[j,p]),'sources':';'.join(map(str,labels[j,p]))} for p in range(17) if not(equal[p] and causal[p])])
                write(OUT/'capacity_cases.csv.gz',capacity)
            write(OUT/'serial.csv.gz',serial)
            print(f'C1 {key} {direction} d={d} g8 and capacity complete',flush=True)


def report():
    for name in ('edit_eligibility.csv','serial.csv.gz','capacity_cases.csv.gz'):
        committed(OUT/name)
    led=pd.read_csv(OUT/'edit_eligibility.csv');s=pd.read_csv(OUT/'serial.csv.gz');c=pd.read_csv(OUT/'capacity_cases.csv.gz')
    expected_ids=set(led.loc[led.eligible==1,'edit_id'])
    assert set(s.edit_id)==set(c.edit_id)==expected_ids,'incomplete edit coverage'
    counts=s.groupby(['edit_id','direction']).size()
    assert len(counts)==2*len(expected_ids) and (counts==128).all(),'incomplete serial panel'
    for (edit_id,direction,lineages),cases in c.groupby(['edit_id','direction','lineages']):
        row=led[led.edit_id==edit_id].iloc[0]
        d=int(row.d) if direction=='forward' else int(row.original_value)%(2*int(row.L))
        assert len(cases)==255*(int(row.L)-4)+64,'incomplete direct capacity panel'
        assert (cases.kind=='single').sum()==255*(d-4)
        assert (cases.kind=='erased').sum()==255*(int(row.L)-d)
        assert (cases.kind=='joint').sum()==64
    for _,cases in c.groupby(['edit_id','direction']):
        lineage_groups=set(cases.lineages.astype(str))
        assert sorted(int(x) for group in lineage_groups for x in group.split(';'))==list(range(16)),'missing g8 descendant'
    bads=s[(s.exact!=1)|(s.provenance_exact!=1)|(s.core_exact!=1)|(s.actual_d!=s.predicted_d)]
    badc=c[(c.exact_passed!=17)|(c.provenance_passed!=17)|(c.core_passed!=17)]
    text=['# C1 — Z80 displacement inheritance and descendant capacity','',
          'Generated by `capacity_assays.py report` from committed CSVs. Registration: `REVISION_PREREG.md`, commit `213a653`.',
          f'','Conditional prediction: **'+('PASS' if len(bads)==len(badc)==0 else 'FAIL — diagnosis required')+f'**. {len(bads)} serial mismatches; {len(badc)} direct-capacity mismatches. Zero tolerance; no post-outcome exclusions.',
          f'{led.key.nunique()} archived parents; {int(led.eligible.sum())}/{len(led)} frozen edits eligible. Each edit has 16 serial lineages, forward and reciprocal passages, all generations 1–8.',
          'Eligibility/geometry was frozen at commit `4855ac3`. Additional parents are CSV cells, not independent worlds; repeated tape values and world origins remain in the ledger.',
          f'Full core, actual executed displacement, offspring tapes and source labels are in `serial.csv.gz` ({len(s):,} rows). Direct tests use each actual g8 descendant; identical descendants share execution and keep all lineage IDs.',
          '', '| direction | d | edits | predicted free d−4 | single cases | joint cases | erased cases | mismatches |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for (direction,d),a in c.groupby(['direction','d']):
        bad=((a.exact_passed!=17)|(a.provenance_passed!=17)|(a.core_passed!=17)).sum()
        text.append(f'| {direction} | {d} | {a.edit_id.nunique()} | {d-4} | {(a.kind=="single").sum()} | {(a.kind=="joint").sum()} | {(a.kind=="erased").sum()} | {bad} |')
    b=pd.read_csv(OUT/'bursts.csv')
    text += ['',f'Local active compute including compilation and readback: {b.active_s.sum():.2f} s; equal pauses recorded in `bursts.csv`; maximum burst {b.active_s.max():.2f} s. No paid compute.',
             'All-value means all 255 alternatives at every declared coordinate. Joint tests are 64 samples, not exhaustive combinations. Each case checks every byte and source label against zero plus 16 uniform partners, without a similarity gate or cyclic realignment. Erased-coordinate equality alone is not retention: final source labels must come from the protected primary seed.',
             'Eligibility was frozen from code and geometry before retention. The initial archived tail need not be d-periodic: first-generation regeneration does not imply lineage loss. The invariant offspring family is the primary seed tiled to L.',
             'Sources: `candidate_eligibility.csv`, `parents.csv`, `edit_eligibility.csv`, `geometry_traces.csv.gz`, `partners.csv`, `inputs.csv`, `serial.csv.gz`, `capacity_cases.csv.gz`.']
    (OUT/'CAPACITY.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text[:7]))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=['qualify','assay','report'])
    globals()[ap.parse_args().command]()

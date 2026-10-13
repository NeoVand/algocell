"""Corrected causal-contract safety gate, archived census and random inputs."""
from __future__ import annotations
import csv
import gzip
import json
from collections import Counter, defaultdict

import numpy as np

import prefix_causality8086 as C
from algocell_exp.census8086_trace import DiagnosticMachine, mechanism, same_execution
from algocell_exp.zmachine import load_machine
from algocell_exp.zcensus import tile_tapes
from census8086 import pairs
from s8086 import HERE, PRE, sha, read_json

OUT=HERE/'results/census8086'


def archived(L, budget, selected=None):
    rows=list(csv.DictReader((OUT/f'mechanisms_L{L}_b{budget}.csv').open()))
    logs=defaultdict(list)
    for part in sorted(OUT.glob(f'traces_L{L}_b{budget}_part*.csv.gz')):
        with gzip.open(part, 'rt') as f:
            for r in csv.DictReader(f):
                key=int(r['k']),int(r['index']),int(r['partner'])
                if selected is None or key in selected:
                    logs[key].append([int(r[x]) for x in ('step','kind','pos','value','original_partner')])
    for r in rows:
        key=int(r['k']),int(r['index']),int(r['partner'])
        if selected is not None and key not in selected: continue
        log=None if r['trace_sha256']=='overflow' else np.asarray(logs.pop(key),np.uint32)
        if log is not None: assert sha(log.tobytes())==r['trace_sha256']
        yield r, log


def save(name, value):
    raw=(json.dumps(value, separators=(',', ':'))+'\n').encode()
    (PRE/name).write_bytes(gzip.compress(raw,mtime=0) if name.endswith('.gz') else raw)


def focused():
    """All 68 copy witnesses, N1, and exact-zero reset-source counterexamples."""
    result={'contract':'isolated source event XOR 55/AA; exact write and identical starts/read/write addresses through that write; later divergence allowed',
            'baseline_commit':'364007c','no_paid_compute':True,'no_provider_calls':True,
            'no_population_runs':True,'R2_altered':False,'release_authorized':False,
            'witnesses':[], 'zero_counts':[], 'source_less_writes':[], 'earlier_witnesses':[]}
    for L in (16,32):
        base=C.burst(f'focused-base-L{L}',lambda:load_machine(f'i8086_pair_maps_L{L}'))
        obs=C.burst(f'focused-observer-L{L}',lambda:DiagnosticMachine(base,events=8192,storage_trace=True))
        intervention=C.burst(f'focused-intervention-L{L}',lambda:C.Interventions(obs))
        (PRE/f'prefix_intervention_L{L}.wgsl').write_text(intervention.wgsl)
        save(f'prefix_intervention_L{L}.manifest.json',intervention.m)
        for budget in (128,512):
            chosen={(2,0x07a5,p) for p in range(17)} | {(2,0x51b9,0)}
            stored=dict(((int(r['index']),int(r['partner'])),(r,log)) for r,log in archived(L,budget,chosen))
            mem=pairs(tile_tapes(np.asarray([0x07a5],np.uint64),2,L),L)
            raw=C.burst(f'focused-raw-{L}-{budget}',lambda:base.run(mem,budget=budget))
            observed=C.burst(f'focused-trace-{L}-{budget}',lambda:obs.run(mem,budget=budget))
            assert same_execution(raw,observed,base.record_words)
            logs=obs.logs(observed)
            for p,log in enumerate(logs): assert sha(log.tobytes())==stored[0x07a5,p][0]['trace_sha256']
            inactive=C.burst(f'focused-inactive-{L}-{budget}',lambda:intervention.run(mem,budget=budget,state=np.full((17,1),0xffffffff,np.uint32)))
            assert np.array_equal(inactive.memory,observed.memory) and np.array_equal(inactive.record,observed.record)
            labels,hypotheses,evidence=C.verify_batch(obs,intervention,mem,logs,L,budget)
            for p,(log,new,(cc,reason)) in enumerate(zip(logs,labels,hypotheses)):
                assert new=='memory-fed one-pass copier'
                result['witnesses'].append({'L':L,'budget':budget,'partner':p,'word':'07a5','label':new,
                    'writes':len(cc),'baseline_trace_sha256':sha(log.tobytes()),
                    'claims':cc,'checks':[e for e in evidence if e['lane']==p]})
            # 51 B9 executes PUSH CX before any data read or operand fetch.
            mem=pairs(tile_tapes(np.asarray([0x51b9],np.uint64),2,L),L)[:1]
            raw=C.burst(f'reset-source-raw-{L}-{budget}',lambda:base.run(mem,budget=budget))
            observed=C.burst(f'reset-source-trace-{L}-{budget}',lambda:obs.run(mem,budget=budget))
            assert same_execution(raw,observed,base.record_words)
            log=obs.logs(observed)[0];before=stored[0x51b9,0][0]
            assert sha(log.tobytes())==before['trace_sha256']
            cc,reason=C.claims(log,L);i,tag=cc[0]
            assert tag==C.UNKNOWN and log[i].tolist()==[0,2,2*L-1,0,1]
            assert not np.any(log[:i,1]==1) and list(log[:i][log[:i,1]==0,3])==[0x51]
            assert mechanism(log,L)[0]=='literal writer'
            result['source_less_writes'].append({'L':L,'budget':budget,'word':'51b9','partner':0,
                'old_label':before['mechanism'],'required_label':'other','reason':'first PUSH CX stores reset CX; no source data read or immediate operand fetch exists',
                'write_event':i,'trace':log.tolist(),'trace_sha256':sha(log.tobytes())})
            rows=list(csv.DictReader((OUT/f'mechanisms_L{L}_b{budget}.csv').open()))
            exact={int(r['index']) for r in csv.DictReader((OUT/f'words_L{L}_k2_b{budget}.csv').open()) if int(r['index']) and int(r['zero_match'])==L}
            assert 0x51b9 in exact
            counts=Counter(r['mechanism'] for r in rows if r['k']=='2' and r['partner']=='0' and int(r['index']) in exact)
            changed=counts.copy();changed['literal writer']-=1;changed['other']+=1
            result['zero_counts'].append({'L':L,'budget':budget,'baseline':dict(counts),
                'after_only_proven_51b9_rejection':dict(changed),'unchanged':False,'complete_reaudit':False})
            print(f'focused L{L}/{budget}: 07 A5 verified 17/17; 51 B9 has no causal source for first write',flush=True)
    for name,parent,partner in [('N1','eb06f49002f10000bc10009df3a5f400','02010000000000000000000000000000'),
                                ('far_jump','8e0e0e00ea0a000200900ef490900200','00'*16)]:
        mem=np.frombuffer(bytes.fromhex(parent+partner),np.uint8)[None]
        base=C.burst(name+'-base',lambda:load_machine('i8086_pair_maps_L16'))
        obs=C.burst(name+'-observer',lambda:DiagnosticMachine(base,events=8192,storage_trace=True))
        intervention=C.burst(name+'-intervention',lambda:C.Interventions(obs))
        raw=C.burst(name+'-raw',lambda:base.run(mem,budget=128))
        observed=C.burst(name+'-trace',lambda:obs.run(mem,budget=128))
        assert same_execution(raw,observed,base.record_words)
        logs=obs.logs(observed)
        labels,cc,ev=C.verify_batch(obs,intervention,mem,logs,16,128)
        assert labels==['other']
        archived_n1=None
        if name=='N1':
            old=read_json(PRE/'causal_stop_validation.json.gz')['N1']
            baseline=np.asarray(old['trace'],np.uint32)
            assert np.array_equal(baseline,logs[0])
            source=int(np.flatnonzero((baseline[:,0]==3)&(baseline[:,1]==1)&(baseline[:,2]==4))[0])
            write=int(np.flatnonzero((baseline[:,0]==3)&(baseline[:,1]==2)&(baseline[:,2]==16))[0])
            archived_n1=[C.compare(baseline,np.asarray(t['trace'],np.uint32),write,source,mask)
                         for mask,t in zip((0x55,0xaa),old['tests'])]
            assert all(r['prefix_identical'] and not r['passes'] and r['written']==2 for r in archived_n1)
        result['earlier_witnesses'].append({'name':name,'parent':parent,'partner':partner,
            'old_label':mechanism(logs[0],16)[0],'required_label':labels[0],
            'claims':cc,'evidence':ev,'trace':logs[0].tolist(),
            'archived_N1_isolated_read_checks':archived_n1,
            'archived_N1_sha256':sha((PRE/'causal_stop_validation.json.gz').read_bytes()) if name=='N1' else None})
    result.update(disposition='MANDATORY STOP: 51 B9 changes registered zero-partner provenance counts under the fail-safe rule',
        existing_classifier_unrepaired=True,existing_freeze_superseded=False,
        complete_census_reaudit=False,random_program_differential='NOT RUN: mandatory count-change stop',
        spend_repairs=False,fault_simulation=False,cap_95_frozen=False,refreeze=False)
    save('prefix_focus_validation.json.gz',{**result,'gpu_bursts':C.BURSTS})
    summary={k:v for k,v in result.items() if k not in ('witnesses','source_less_writes','earlier_witnesses')}
    summary['07a5_encounters']=len(result['witnesses'])
    summary['07a5_verified']=sum(r['label']=='memory-fed one-pass copier' for r in result['witnesses'])
    summary['07a5_claimed_writes']=sum(r['writes'] for r in result['witnesses'])
    summary['earlier_witnesses']=[{k:v for k,v in r.items() if k not in ('trace','claims','evidence')} for r in result['earlier_witnesses']]
    summary['gpu_bursts']=len(C.BURSTS);summary['max_gpu_burst_s']=max(r['burst_s'] for r in C.BURSTS)
    summary['archive_sha256']=sha((PRE/'prefix_focus_validation.json.gz').read_bytes())
    save('prefix_focus_summary.json',summary)
    print(result['disposition'],flush=True)


def census_first_setting():
    result={'contract':'isolated source event XOR 55/AA; exact write value and identical starts/read/write addresses through that write; later divergence allowed',
            'baseline_commit':'364007c', 'no_paid_compute':True, 'no_provider_calls':True,
            'no_population_runs':True, 'R2_altered':False, 'release_authorized':False,
            'census':[], 'changes':[], 'evidence':[], 'witnesses':[]}
    changed_gate=False
    for L in (16,):
        base=C.burst(f'compile-L{L}',lambda:load_machine(f'i8086_pair_maps_L{L}'))
        observer=C.burst(f'observer-L{L}',lambda:DiagnosticMachine(base,events=8192,storage_trace=True))
        intervention=C.burst(f'intervention-L{L}',lambda:C.Interventions(observer))
        (PRE/f'prefix_intervention_L{L}.wgsl').write_text(intervention.wgsl)
        save(f'prefix_intervention_L{L}.manifest.json',intervention.m)
        for budget in (128,):
            exact={}
            for k in (1,2):
                exact[k]={int(r['index']) for r in csv.DictReader((OUT/f'words_L{L}_k{k}_b{budget}.csv').open())
                          if int(r['index']) and int(r['zero_match'])==L}
            encounters=list(archived(L,budget))
            before=Counter(); after=Counter(); witnesses=[]
            for lo in range(0,len(encounters),64):
                work=encounters[lo:lo+64]
                memories=np.stack([pairs(tile_tapes(np.asarray([int(r['index'])],np.uint64),int(r['k']),L),L)[int(r['partner'])]
                                   for r, log in work])
                logs=[log for _, log in work]
                labels,hypotheses,evidence=C.verify_batch(observer,intervention,memories,logs,L,budget)
                for e in evidence:
                    r=work[e['lane']][0]
                    e.update(L=L,budget=budget,k=int(r['k']),index=int(r['index']),partner=int(r['partner']))
                result['evidence'].extend(evidence)
                for (r,log),new,(cc,reason) in zip(work,labels,hypotheses):
                    assert mechanism(log,L)[0]==r['mechanism']
                    if new!=r['mechanism']:
                        result['changes'].append({'L':L,'budget':budget,'k':r['k'],'word':r['word'],
                                                 'partner':r['partner'],'old':r['mechanism'],'new':new,'reason':reason})
                    if int(r['partner'])==0 and int(r['index']) in exact[int(r['k'])]:
                        before[int(r['k']),r['mechanism']]+=1; after[int(r['k']),new]+=1
                    if r['word']=='07a5':
                        witnesses.append(new)
                        result['witnesses'].append({'L':L,'budget':budget,'partner':int(r['partner']),
                                                    'word':'07a5','label':new,'writes':len(cc)})
                if lo%512==0:print(f'prefix census L{L}/{budget}: {min(lo+64,len(encounters))}/{len(encounters)}',flush=True)
            gate=before==after and witnesses==['memory-fed one-pass copier']*17
            changed_gate |= not gate
            result['census'].append({'L':L,'budget':budget,'encounters':len(encounters),
                'before':{f'{k}:{cls}':n for (k,cls),n in before.items()},
                'after':{f'{k}:{cls}':n for (k,cls),n in after.items()},
                'zero_counts_unchanged':before==after,'07a5_unchanged':witnesses==['memory-fed one-pass copier']*17})
            save('prefix_causal_validation.json.gz',{**result,'gpu_bursts':C.BURSTS})
            print(f'prefix census L{L}/{budget}: gate={gate}, total changes={len(result["changes"])}',flush=True)
    result['disposition']='MANDATORY STOP: registered zero-partner counts or 07 A5 changed' if changed_gate else 'CAUSAL GATE PASSES; spend repairs may proceed'
    result.update(existing_freeze_superseded=False,existing_classifier_unrepaired=True,
                  complete_census_reaudit=False,
                  random_program_differential='NOT RUN: mandatory count-change stop' if changed_gate else 'NOT RUN: first-setting diagnostic only')
    save('prefix_causal_validation.json.gz',{**result,'gpu_bursts':C.BURSTS})
    summary={k:v for k,v in result.items() if k not in ('evidence','witnesses','changes')}
    summary['label_changes']=len(result['changes'])
    summary['07a5_encounters']=sum(r.get('word')=='07a5' for r in result['witnesses'])
    summary['07a5_verified']=sum(r.get('word')=='07a5' and r['label']=='memory-fed one-pass copier' for r in result['witnesses'])
    summary['gpu_bursts']=len(C.BURSTS)
    summary['max_gpu_burst_s']=max(r['burst_s'] for r in C.BURSTS)
    summary['archive_sha256']=sha((PRE/'prefix_causal_validation.json.gz').read_bytes())
    save('prefix_causal_summary.json',summary)
    print(result['disposition'],flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census-first-setting',action='store_true')
    args=parser.parse_args()
    census_first_setting() if args.census_first_setting else focused()

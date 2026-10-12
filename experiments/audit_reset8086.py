"""Complete reset/event intervention census gate. Fixed inputs; no provider calls."""
from collections import Counter
import csv, gzip, json
import numpy as np
import prefix_causality8086 as C
from audit_prefix8086 import archived, save, OUT
from algocell_exp.census8086_trace import DiagnosticMachine, mechanism, same_execution
from algocell_exp.zmachine import load_machine
from algocell_exp.zcensus import tile_tapes
from census8086 import pairs
from s8086 import PRE, sha


def audit():
    result={'date':'2026-10-12','baseline_commit':'58a2f4b','no_paid_compute':True,
            'no_population':True,'R0_R1_unchanged':True,'R2_unchanged':True,
            'runs':[],'changes':[],'gate_changes':[],'focused':[]}
    with gzip.open(PRE/'reset_intervention_checks.jsonl.gz','wt') as evfile:
      for L in (16,32):
        base=C.burst(f'base-{L}',lambda:load_machine(f'i8086_pair_maps_L{L}'))
        obs=C.burst(f'observer-{L}',lambda:DiagnosticMachine(base,events=8192,storage_trace=True))
        inter=C.burst(f'intervention-{L}',lambda:C.Interventions(obs))
        (PRE/f'reset_intervention_L{L}.wgsl').write_text(inter.wgsl)
        save(f'reset_intervention_L{L}.manifest.json',inter.m)
        for budget in (128,512):
            entries=list(archived(L,budget)); before=Counter();after=Counter();zb=Counter();za=Counter();checks=failures=0
            exact={k:{int(r['index']) for r in csv.DictReader((OUT/f'words_L{L}_k{k}_b{budget}.csv').open())
                      if int(r['index']) and int(r['zero_match'])==L} for k in (1,2)}
            updated=[]
            for lo in range(0,len(entries),64):
                work=entries[lo:lo+64]
                mem=np.stack([pairs(tile_tapes(np.asarray([int(r['index'])],np.uint64),int(r['k']),L),L)[int(r['partner'])] for r,_ in work])
                logs=[log for _,log in work]
                raw=C.burst(f'raw-{L}-{budget}-{lo}',lambda:base.run(mem,budget=budget))
                observed=C.burst(f'observed-{L}-{budget}-{lo}',lambda:obs.run(mem,budget=budget))
                assert same_execution(raw,observed,base.record_words)
                fresh=obs.logs(observed)
                for a,b in zip(logs,fresh): assert (a is None and b is None) or np.array_equal(a,b)
                labels,hyp,ev=C.verify_batch(obs,inter,mem,logs,L,budget)
                checks+=2*len(ev);failures+=sum(not c['passes'] for e in ev for c in e['checks'])
                for e in ev:
                    row=work[e['lane']][0]
                    evfile.write(json.dumps(dict(e,L=L,budget=budget,k=int(row['k']),index=int(row['index']),partner=int(row['partner'])))+'\n')
                for (r,log),new,(cc,reason) in zip(work,labels,hyp):
                    before[r['mechanism']]+=1;after[new]+=1
                    row={**r,'mechanism':new,'evidence':'two_alternative_reset_or_event_prefix_interventions' if new!='other' else 'unsupported_or_failed_causal_intervention'}
                    updated.append(row)
                    opened=log is not None and bool(np.any((log[:,1]==0)&(log[:,2]>=L)))
                    if new!=r['mechanism']:
                        change={'L':L,'budget':budget,'k':int(r['k']),'word':r['word'],'partner':int(r['partner']),'old':r['mechanism'],'new':new,'open':opened,'reason':reason}
                        result['changes'].append(change)
                        if opened and ('memory-fed one-pass copier' in (new,r['mechanism'])):result['gate_changes'].append(change)
                    if int(r['partner'])==0 and int(r['index']) in exact[int(r['k'])]:zb[r['mechanism']]+=1;za[new]+=1
                    if r['word']=='07a5':
                        assert new=='memory-fed one-pass copier','07 A5 changed; STOP'
                    if int(r['partner'])==0 and r['word'] in ('b951','51b9','075a','07a5'):
                        result['focused'].append({'L':L,'budget':budget,'word':r['word'],'label':new,'claims':len(cc)})
                if lo%512==0:print(f'reset census L{L}/{budget}: {min(lo+64,len(entries))}/{len(entries)}; gate changes {len(result["gate_changes"])}',flush=True)
            result['runs'].append({'L':L,'budget':budget,'encounters':len(entries),'before':dict(before),'after':dict(after),'zero_exact_before':dict(zb),'zero_exact_after':dict(za),'alternative_checks':checks,'failed_checks':failures,'full_record_memory_trace_mismatches':0})
            # Keep proposed tables distinct until the safety gate is resolved.
            with (PRE/f'reset_mechanisms_L{L}_b{budget}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(updated[0]));w.writeheader();w.writerows(updated)
            save('reset_census_validation.json',result)
            if result['gate_changes']:
                result['disposition']='STOP: open memory-fed labels changed'
                result.update(gpu_bursts=len(C.BURSTS),max_gpu_burst_s=max(x['burst_s'] for x in C.BURSTS))
                save('reset_census_validation.json',result);save('reset_local_bursts.json',C.BURSTS)
                return result
        if L==16:
            for name,parent,partner in [('N1','eb06f49002f10000bc10009df3a5f400','02010000000000000000000000000000'),('far_jump','8e0e0e00ea0a000200900ef490900200','00'*16)]:
                mem=np.frombuffer(bytes.fromhex(parent+partner),np.uint8)[None]
                raw=C.burst(name+'-raw',lambda:base.run(mem,budget=128))
                observed=C.burst(name+'-observer',lambda:obs.run(mem,budget=128));assert same_execution(raw,observed,base.record_words)
                log=obs.logs(observed)[0]
                labels,hyp,ev=C.verify_batch(obs,inter,mem,[log],16,128)
                result['focused'].append({'name':name,'old':mechanism(log,16)[0],'label':labels[0],'claims':hyp,'checks':ev,'trace':log.tolist()})
    result['disposition']='PASS: 07 A5 and all open memory-fed labels unchanged'
    result['gpu_bursts']=len(C.BURSTS);result['max_gpu_burst_s']=max(x['burst_s'] for x in C.BURSTS)
    save('reset_census_validation.json',result);save('reset_local_bursts.json',C.BURSTS)
    print(result['disposition'],flush=True)
    return result

if __name__=='__main__':audit()

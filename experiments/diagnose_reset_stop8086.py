"""Focused diagnosis of the R2-label STOP, never continuing launch preparation."""
import copy, csv, gzip, json
from collections import Counter
import numpy as np
import prefix_causality8086 as C
from audit_prefix8086 import archived,save,OUT
from algocell_exp.census8086_trace import DiagnosticMachine,mechanism,same_execution
from algocell_exp.zmachine import load_machine,ZMachine
from algocell_exp.zcensus import tile_tapes
from census8086 import pairs
from s8086 import PRE,sha,read_json


def main():
    result={'date':'2026-10-12','scope':'STOP diagnosis only; not remaining census/random/spend/refreeze work',
            'witnesses':[],'direct_gate_witness':{}}
    audit=read_json(PRE/'reset_census_validation.json')
    with gzip.open(PRE/'reset_intervention_checks.jsonl.gz','rt') as f:
        evidence=[json.loads(line) for line in f]
    first=next(e for e in evidence if e['index']==0xad50 and e['partner']==0 and any(not c['passes'] for c in e['checks']))
    for L in (16,32):
        base=C.burst(f'diagnose-base-{L}',lambda:load_machine(f'i8086_pair_maps_L{L}'))
        obs=C.burst(f'diagnose-observer-{L}',lambda:DiagnosticMachine(base,events=8192,storage_trace=True))
        inter=C.burst(f'diagnose-intervention-{L}',lambda:C.Interventions(obs))
        (PRE/f'reset_intervention_L{L}.wgsl').write_text(inter.wgsl);save(f'reset_intervention_L{L}.manifest.json',inter.m)
        for budget in (128,512):
            mem=np.stack([pairs(tile_tapes(np.asarray([idx],np.uint64),2,L),L)[0] for idx in (0x51b9,0xb951)])
            raw=C.burst(f'diagnose-raw-{L}-{budget}',lambda:base.run(mem,budget=budget))
            observed=C.burst(f'diagnose-trace-{L}-{budget}',lambda:obs.run(mem,budget=budget));assert same_execution(raw,observed,base.record_words)
            inactive=C.burst(f'diagnose-inactive-{L}-{budget}',lambda:inter.run(mem,budget=budget,state=np.full((2,3),0xffffffff,np.uint32)))
            assert np.array_equal(inactive.memory,observed.memory) and np.array_equal(inactive.record,observed.record)
            logs=obs.logs(observed)
            stored={int(r['index']):(r,log) for r,log in archived(L,budget,{(2,idx,0) for idx in (0x51b9,0xb951)})}
            labels,hyp,ev=C.verify_batch(obs,inter,mem,logs,L,budget)
            for j,idx in enumerate((0x51b9,0xb951)):
                assert np.array_equal(logs[j],stored[idx][1])
                result['witnesses'].append({'L':L,'budget':budget,'word':f'{idx:04x}','partner':0,'old':stored[idx][0]['mechanism'],
                    'label':labels[j],'claims':len(hyp[j][0]),'first_claim':hyp[j][0][0],
                    'checks':[e for e in ev if e['lane']==j], 'trace_sha256':sha(logs[j].tobytes())})
            print(f'diagnose L{L}/{budget}: 51 B9 {labels[0]}; B9 51 {labels[1]}',flush=True)
        if L==16:
            for name,parent,partner in [('N1','eb06f49002f10000bc10009df3a5f400','02010000000000000000000000000000'),('far_jump','8e0e0e00ea0a000200900ef490900200','00'*16)]:
                mem=np.frombuffer(bytes.fromhex(parent+partner),np.uint8)[None]
                raw=C.burst(name+'-raw',lambda:base.run(mem,budget=128))
                observed=C.burst(name+'-obs',lambda:obs.run(mem,budget=128));assert same_execution(raw,observed,base.record_words)
                log=obs.logs(observed)[0];labels,hyp,ev=C.verify_batch(obs,inter,mem,[log],16,128)
                assert labels==['other']
                result['witnesses'].append({'name':name,'old':mechanism(log,16)[0],'label':labels[0],'reason':hyp[0][1],'checks':ev,'trace':log.tolist()})
            # Independent constant-target hook has no intervention input binding.
            mem=pairs(tile_tapes(np.asarray([0xad50],np.uint64),2,16),16)[:1]
            raw=C.burst('gate-raw',lambda:base.run(mem,budget=128))
            observed=C.burst('gate-observer',lambda:obs.run(mem,budget=128));assert same_execution(raw,observed,base.record_words)
            log=obs.logs(observed)[0];source=first['source'][1];write=first['write_event']
            src=obs.wgsl.replace('let p=zil_pos(addr); let v=census_real_read(addr);',
                f'let p=zil_pos(addr); var v=census_real_read(addr); if(census_n=={source}u && kind==1u) {{ v=v^(zil_params.seed & 255u); }}')
            manifest=copy.deepcopy(obs.m);manifest['sha256']['wgsl']=sha(src.encode())
            direct=C.burst('gate-independent-hook',lambda:ZMachine(src,manifest,name='constant-event-stop-witness',device=obs.dev))
            (PRE/'reset_gate_independent.wgsl').write_text(src);save('reset_gate_independent.manifest.json',manifest)
            tests=[]
            for mask in (0x55,0xaa):
                changed=obs.logs(C.burst(f'gate-independent-{mask}',lambda:direct.run(mem,budget=128,seed=mask)))[0]
                check=C.compare(log,changed,write,source,mask)
                # Independent comparison of the explicit prefix rows.
                a=[r[:3].tolist() for r in log[:write+1] if r[1] in (0,1,2,4)]
                b=[r[:3].tolist() for r in changed[:write+1] if r[1] in (0,1,2,4)]
                assert a!=b and not check['passes'] and check['written']==check['substitute']
                tests.append(dict(check,trace=changed.tolist()))
            result['direct_gate_witness']={'L':16,'budget':128,'word':'ad50','partner':0,
                'old':'memory-fed one-pass copier','required_label':'other','full_encounter_open':True,
                'source_event':source,'source_row':log[source].tolist(),'write_event':write,'write_row':log[write].tolist(),
                'baseline_trace':log.tolist(),'checks':tests}
    result['gpu_bursts']=C.BURSTS
    save('reset_stop_focus.json.gz',result)
    summary={k:v for k,v in result.items() if k not in ('witnesses','direct_gate_witness','gpu_bursts')}
    summary['witnesses']=[{k:v for k,v in x.items() if k not in ('checks','trace')} for x in result['witnesses']]
    summary['direct_gate_witness']={k:v for k,v in result['direct_gate_witness'].items() if k not in ('baseline_trace','checks')}
    summary.update(gpu_bursts=len(C.BURSTS),max_gpu_burst_s=max(x['burst_s'] for x in C.BURSTS),archive_sha256=sha((PRE/'reset_stop_focus.json.gz').read_bytes()))
    save('reset_stop_focus_summary.json',summary)
    # The original run ended at the requested gate. Describe its real scope.
    audit.update(R2_predictions_unchanged=True,R2_label_changes=len(audit['gate_changes']),
        complete_census_reaudit=False,random_program_differential='NOT RUN: open-memory label STOP',
        spend_repairs=False,fault_simulation=False,cap_95_frozen=False,refreeze=False,
        existing_freeze_superseded=False,existing_classifier_unrepaired=True,
        checks_archive_sha256=sha((PRE/'reset_intervention_checks.jsonl.gz').read_bytes()))
    save('reset_census_validation.json',audit)
    with (OUT/'reset_correction_labels.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit['changes'][0]));w.writeheader();w.writerows(audit['changes'])
    print('STOP confirmed independently; no refreeze',flush=True)

if __name__=='__main__':main()

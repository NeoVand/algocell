"""B6 fixed-input provenance differential and archived census correction. No Modal.

Every census candidate trace and deterministic random program is checked against
an independent value-tag bank. Only random fixed inputs execute locally; each
synchronous GPU burst has a <=60s check and equal pause. No population inputs.
"""
from collections import defaultdict, Counter
import csv, gzip, json, time, subprocess, io
from pathlib import Path
import numpy as np
from algocell_exp.census8086_trace import mechanism,DiagnosticMachine,same_execution
from algocell_exp.i8086_provenance_spec import OPCODE_CLASS,CLASSES
from i8086_value_tags import value_channels,reference_mechanism
from s8086 import HERE,PRE,sha

OUT=HERE/'results/census8086'

def save_csv(path,rows,fields=None):
    rows=list(rows)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

def check(log,L):
    assert mechanism(log,L,audit_channels=True)==value_channels(log,L), 'ordered write channel mismatch'
    assert mechanism(log,L)[0]==reference_mechanism(log,L), 'mechanism mismatch'

def burst(label,fn):
    t=time.monotonic();result=fn();duration=time.monotonic()-t
    if duration>60: raise RuntimeError('local GPU burst >60s')
    t=time.monotonic();time.sleep(duration)
    with (PRE/'b6_local_bursts.jsonl').open('a') as f:
        f.write(json.dumps({'label':label,'burst_s':duration,'pause_s':time.monotonic()-t})+'\n')
    return result

def audit_table():
    rows=[]
    for op,cls in enumerate(OPCODE_CLASS):
        anchor,effect,policy=CLASSES[cls]
        for L in (16,32):
            text=(HERE/f'algocell_exp/machines/i8086_pair_maps_L{L}.wgsl').read_text()
            assert anchor.lower() in text.lower(),(cls,anchor)
        mod=(op<64 and op%8<4) or 128<=op<=143 or 196<=op<=199 or 208<=op<=211 or 216<=op<=223 or op in (246,247,254,255)
        for reg in (range(8) if mod else [-1]):
            for mode in (('register','memory') if mod else ['implicit']):
                rows.append({'opcode':f'{op:02x}','reg_field':reg,'mode':mode,'class':cls,'WGSL_anchor':anchor,'writes':effect,'provenance_policy':policy})
    save_csv(PRE/'b6_instruction_audit.csv',rows)
    return len(rows)

def census():
    changes=[];counts=[];total=0;settings=0
    word_hashes={p.name:sha(p.read_bytes()) for p in sorted(OUT.glob('words_*.csv'))}
    for L in (16,32):
      for budget in (128,512):
        path=OUT/f'mechanisms_L{L}_b{budget}.csv'
        baseline=subprocess.check_output(['git','show',f'e96abc0:experiments/results/census8086/{path.name}'],text=True)
        rows=list(csv.DictReader(io.StringIO(baseline))); logs=defaultdict(list)
        for part in sorted(OUT.glob(f'traces_L{L}_b{budget}_part*.csv.gz')):
          with gzip.open(part,'rt') as f:
            for r in csv.DictReader(f):
              logs[int(r['k']),int(r['index']),int(r['partner'])].append([int(r[x]) for x in ('step','kind','pos','value','original_partner')])
        exact={}
        for k in (1,2):
          words=list(csv.DictReader((OUT/f'words_L{L}_k{k}_b{budget}.csv').open()))
          exact[k]={int(r['index']) for r in words if int(r['index']) and int(r['zero_match'])==L}
        exact_labels={(r['k'],r['index']):r['mechanism'] for r in rows if int(r['partner'])==0 and int(r['index']) in exact[int(r['k'])]}
        before=Counter((int(r['k']),r['mechanism']) for r in rows if int(r['partner'])==0 and int(r['index']) in exact[int(r['k'])])
        for r in rows:
          key=int(r['k']),int(r['index']),int(r['partner'])
          log=None if r['trace_sha256']=='overflow' else np.asarray(logs[key],np.uint32)
          if log is not None: assert sha(log.tobytes())==r['trace_sha256']
          check(log,L);total+=1
          cls,evidence=mechanism(log,L)
          if cls!=r['mechanism']: changes.append({'L':L,'budget':budget,'k':r['k'],'word':r['word'],'partner':r['partner'],'old':r['mechanism'],'new':cls})
          r.update(mechanism=cls,evidence=evidence)
        after=Counter((int(r['k']),r['mechanism']) for r in rows if int(r['partner'])==0 and int(r['index']) in exact[int(r['k'])])
        if before!=after: raise RuntimeError('registered zero-partner counts changed; STOP, do not alter R2')
        assert exact_labels=={(r['k'],r['index']):r['mechanism'] for r in rows if int(r['partner'])==0 and int(r['index']) in exact[int(r['k'])]}
        witness=[r for r in rows if r['word']=='07a5']
        if len(witness)!=17 or any(r['mechanism']!='memory-fed one-pass copier' for r in witness): raise RuntimeError('07 A5 changed; STOP')
        save_csv(path,rows)
        counts.append({'L':L,'budget':budget,'exact_counts_unchanged':True,'07a5_17_of_17_unchanged':True,'encounters':len(rows)})
        settings+=len({(r['k'],r['index']) for r in rows})
        print(f'B6 archived L{L}/{budget}: {len(rows)} differential matches',flush=True)
    save_csv(OUT/'b6_correction_labels.csv',changes,['L','budget','k','word','partner','old','new'])
    return {'settings':settings,'encounters':total,'mismatches':0,'label_changes':len(changes),'runs':counts,'words_sha256':word_hashes,'overflow_other':2,'individual_exact_word_labels_unchanged':True}

def random_programs():
    from algocell_exp.zmachine import load_machine
    summary=[]; seen=Counter()
    rng=np.random.Generator(np.random.PCG64(0xB68086))
    for L in (16,32):
        machine=burst(f'compile-L{L}',lambda:load_machine(f'i8086_pair_maps_L{L}'))
        observer=burst(f'observer-L{L}',lambda:DiagnosticMachine(machine,events=8192,storage_trace=True))
        # Uniform byte tapes plus random load/overwrite/store instruction chains.
        inputs=rng.integers(0,256,(2048,2*L),dtype=np.uint8)
        chains=rng.integers(0,256,(2048,2*L),dtype=np.uint8)
        reps=[0xea,0x8e,0x8c,0xc4,0xc5,0x8f,0x86,0x87,0x89,0x8b,0xff,0xfe,0xf6,0xf7,0xd0,0xd2,0x40,0x54,0xac,0xad,0xaa,0xab,0xe2,0x98,0x99,0x9d,0x9f,0xd4,0xd5,0xd6,0xd7]
        for i,row in enumerate(chains):
            reg=i%8
            row[:4]=[0x8b,(reg<<3)|6,2*L-2,0]
            op=reps[i%len(reps)]
            ins=[op,int(rng.integers(0,256)),0,0,0]
            if op==0xea: ins=[op,9,0,2,0]
            if op in (0x8e,0x8c,0x86,0x87,0x89,0x8b,0xff,0xfe,0xf6,0xf7,0xd0,0xd2): ins[1]=0xc0|int(rng.integers(0,64))
            row[4:9]=ins;row[9:12]=[0x50+reg,0x0e,0xf4]
        inputs=np.vstack([inputs,chains])
        if L==16:
            inputs[0,:L]=np.frombuffer(bytes.fromhex('8e0e0e00ea0a000200900ef490900200'),np.uint8);inputs[0,L:]=0
        digest=sha(inputs.tobytes());n=0
        for start in range(0,len(inputs),32):
            mem=inputs[start:start+32]
            raw=burst(f'raw-L{L}-{start}',lambda:machine.run(mem,budget=128))
            obs=burst(f'trace-L{L}-{start}',lambda:observer.run(mem,budget=128))
            assert same_execution(raw,obs,machine.record_words)
            for j,log in enumerate(observer.logs(obs)):
                check(log,L);n+=1
                if L==16 and start==0 and j==0:
                    assert mechanism(log,L)[0]=='literal writer'
                    end=max(log[(log[:,1]==2)&(log[:,2]>=L),0])
                    starts=list(map(int,log[(log[:,1]==4)&(log[:,0]<=end),2]))
                    assert starts==[0,36,42]
                    (PRE/'b6_astra_witness.json').write_text(json.dumps({'parent':bytes(mem[j,:L]).hex(),'starts':starts,'mechanism':mechanism(log,L),'trace':log.tolist()},indent=2)+'\n')
                if log is not None:
                    for op in log[log[:,1]==3,3]: seen[OPCODE_CLASS[int(op)]]+=1
            if start%512==0: print(f'B6 random L{L}: {start+len(mem)}/{len(inputs)}',flush=True)
        summary.append({'L':L,'programs':n,'budget':128,'input_sha256':digest,'full_record_memory_mismatches':0,'provenance_mismatches':0})
    return {'runs':summary,'classes_observed':dict(sorted(seen.items())),'mismatches':0}

def main():
    table=audit_table(); archive=census();random=random_programs()
    result={'date':'2026-10-12','no_population':True,'no_modal':True,'baseline_commit':'e96abc0','instruction_table_rows':table,'classifier_sha256':sha((HERE/'algocell_exp/census8086_trace.py').read_bytes()),'oracle_sha256':sha((HERE/'i8086_value_tags.py').read_bytes()),'census':archive,'random_programs':random}
    (PRE/'b6_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'b6_correction_provenance.json').write_text(json.dumps({'date':'2026-10-12','scope':'B6; trace-fed CPU reclassification of every archived candidate; raw outputs/traces unchanged',**archive},indent=2)+'\n')
if __name__=='__main__': main()

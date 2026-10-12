"""Independent random-program causal differential; fixed inputs, local bursts only.

The scalar oracle uses a separately written prefix comparison, source-expression
arithmetic, and class decision. Production extractors supply hypotheses, not
truth: every admitted claim is checked by two actual isolated interventions.
This validates causal verification, not arbitrary unmodeled 8086 semantics.
"""
import json
from collections import Counter
from unittest.mock import patch
import numpy as np
import prefix_causality8086 as C
from algocell_exp.census8086_trace import DiagnosticMachine, same_execution
from algocell_exp.zmachine import load_machine
from s8086 import PRE, HERE, sha


def scalar_prediction(tag, root, mask):
    cache={}
    def evaluate(t):
        identity=id(t)
        if identity in cache:return cache[identity]
        source=t[1]
        if isinstance(source,int):result=t[2] ^ (mask if root==('event',source) else 0)
        elif source[0]=='reset':
            word=int(source[1]==4)
            if root==('reset',source[1]):word ^= mask | mask<<8
            result=(word//(256**source[2]))%256
        else:
            lo,hi=map(evaluate,source[1]);word=(lo+256*hi+source[2])%65536
            result=(word//(256**source[3]))%256
        cache[identity]=result
        return result
    return evaluate(tag)


def scalar_prefix(original, changed, write, source, mask, predicted=None):
    if changed is None or len(changed)<=write:return False,False,None
    identical=True
    for i in range(write+1):
        if int(original[i,1]) in (0,1,2,4) and tuple(original[i,:3])!=tuple(changed[i,:3]):
            identical=False
    wanted=int(original[source,3])^mask if predicted is None else predicted
    value=int(changed[write,3])
    return identical, identical and value==wanted and int(changed[write,1])==2, value


def scalar_label(log,L,hypothesis,evidence,lane):
    cc,_=hypothesis
    if not cc:return 'other'
    failed={e['write_event'] for e in evidence if e['lane']==lane and any(not c['passes'] for c in e['checks'])}
    channels=set()
    for write,tag in cc:
        if write in failed or tag[0]=='other' or not C.roots(tag):return 'other'
        if tag[0]!='neutral':channels.add(tag[0])
    if channels=={'literal'}:return 'literal writer'
    if channels!={'memory'}:return 'other'
    end=max(int(log[write,0]) for write,_ in cc)
    starts=[];fetches_b=False
    for step,kind,address,*_ in log:
        if step>end:continue
        if kind==4:starts.append(int(address))
        if kind==0 and address>=L:fetches_b=True
    if len(starts)==len(set(starts)):return 'memory-fed one-pass copier'
    return 'other' if fetches_b else 'confined copy loop'


def verify_differential(observer,intervention,mem,logs,L,budget):
    comparisons=[0];predictions=[0]
    production_compare=C.compare;production_predict=C.predict
    def compare(*args,**kwargs):
        result=production_compare(*args,**kwargs)
        identical,passes,value=scalar_prefix(*args,**kwargs)
        assert (result['prefix_identical'],result['passes'],result['written'])==(identical,passes,value)
        comparisons[0]+=1
        return result
    def predict(*args):
        result=production_predict(*args);assert result==scalar_prediction(*args)
        predictions[0]+=1;return result
    with patch.object(C,'compare',compare),patch.object(C,'predict',predict):
        labels,hyp,ev=C.verify_batch(observer,intervention,mem,logs,L,budget)
    assert labels==[scalar_label(log,L,h,ev,i) for i,(log,h) in enumerate(zip(logs,hyp))]
    return labels,comparisons[0],predictions[0]


def random_programs():
    rng=np.random.Generator(np.random.PCG64(0xB68086));runs=[]
    for L in (16,32):
        base=C.burst(f'random-base-{L}',lambda:load_machine(f'i8086_pair_maps_L{L}'))
        observer=C.burst(f'random-observer-{L}',lambda:DiagnosticMachine(base,events=8192,storage_trace=True))
        intervention=C.burst(f'random-intervention-{L}',lambda:C.Interventions(observer))
        mem=rng.integers(0,256,(4096,2*L),dtype=np.uint8)
        # Half uniform programs; half load/overwrite/store chains. Include
        # prefixes, resets, far transfers, REP and exceptional/vector traffic.
        ops=[0xea,0x8e,0x8c,0xc4,0xc5,0x8f,0x86,0x87,0x89,0x8b,0xff,0xfe,0xf6,0xf7,0xd0,0xd2,0x40,0x54,0xac,0xad,0xaa,0xab,0xe2,0x98,0x99,0x9d,0x9f,0xd4,0xd5,0xd6,0xd7,0xf3,0xf2,0x51,0xcc]
        for i,row in enumerate(mem[2048:]):
            reg=i%8;row[:4]=[0x8b,(reg<<3)|6,2*L-2,0]
            op=ops[i%len(ops)];ins=[op,int(rng.integers(0,256)),0,0,0]
            if op==0xea:ins=[op,9,0,2,0]
            if op in (0xf3,0xf2):ins=[op,0xa5,0xf4,0,0]
            if op in (0x8e,0x8c,0x86,0x87,0x89,0x8b,0xff,0xfe,0xf6,0xf7,0xd0,0xd2):ins[1]=0xc0|int(rng.integers(0,64))
            row[4:9]=ins;row[9:12]=[0x50+reg,0x0e,0xf4]
        checks=predictions=0;counts=Counter();overflows=0
        for lo in range(0,len(mem),32):
            batch=mem[lo:lo+32]
            raw=C.burst(f'random-raw-{L}-{lo}',lambda:base.run(batch,budget=128))
            traced=C.burst(f'random-traced-{L}-{lo}',lambda:observer.run(batch,budget=128))
            assert same_execution(raw,traced,base.record_words)
            logs=observer.logs(traced);overflows+=sum(log is None for log in logs)
            labels,c,p=verify_differential(observer,intervention,batch,logs,L,128)
            checks+=c;predictions+=p;counts.update(labels)
            if lo%512==0:print(f'random causal L{L}: {lo+len(batch)}/4096',flush=True)
        runs.append({'L':L,'programs':len(mem),'input_sha256':sha(mem.tobytes()),'counts':dict(counts),'prefix_comparisons':checks,'scalar_predictions':predictions,'overflow_other':overflows,'full_record_memory_mismatches':0,'causal_differential_mismatches':0})
    result={'date':'2026-10-12','programs':8192,'runs':runs,'mismatches':0,'no_population':True,'no_paid_compute':True,'oracle_scope':__doc__,'classifier_sha256':sha((HERE/'prefix_causality8086.py').read_bytes()),'oracle_sha256':sha(__import__('pathlib').Path(__file__).read_bytes()),'gpu_bursts':len(C.BURSTS),'max_gpu_burst_s':max(r['burst_s'] for r in C.BURSTS)}
    (PRE/'descriptive_random_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (PRE/'descriptive_random_bursts.json').write_text(json.dumps(C.BURSTS,indent=2)+'\n')
    print('8192 random fixed programs; 0 differential mismatches',flush=True)
    return result

if __name__=='__main__':random_programs()

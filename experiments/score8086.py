"""S2 chronological endpoints, full-horizon occurrence/persistence and candidates."""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import brotli
import numpy as np
from algocell_exp import assay as A, executor as E
from algocell_exp.census8086_trace import DiagnosticMachine, mechanism, same_execution
from s8086 import ARMS, HORIZON, SEEDS, SNAPSHOTS, PERSISTENCE, samples, key, rng, partners, sha, safe, write_new, verify_freeze,diagnostic_for

def endpoints(records, judge):
    first = closed = None; occurrence = []; persistence = {}; unresolved = []
    blocked_rep = blocked_closed = False
    previous = None
    for sample in records:
        events = []
        for tape,share in sorted(sample['top3'],key=lambda x:(-x[1],bytes.fromhex(x[0]))):
            if share < .005: continue
            r = judge(tape)
            if r.get('gen2') is None or not math.isfinite(r['gen2']):
                unresolved.append({'step':sample['step'],'tape':tape,'reason':'unassessable_culture'})
                blocked_rep = blocked_rep or first is None
                blocked_closed = blocked_closed or closed is None
                continue
            if not r['heritable']: continue
            event = {'step':sample['step'],'preceding_step':previous,'baseline':sample['step']==0,
                     'tape':tape,'share':share,**r}
            if first is None and not blocked_rep: first = event
            if closed is None and not blocked_closed and r['confined']: closed = event
            if r['open_memory_fed']: events.append(event); occurrence.append(event)
        if sample['step'] in PERSISTENCE: persistence[sample['step']] = {r['tape'] for r in events}
        previous = sample['step']
    persistent = set(persistence) == set(PERSISTENCE) and all(persistence.values())
    same = bool(persistent and set.intersection(*persistence.values()))
    return {'scoreable':not unresolved,'unresolved':unresolved,'first':first,'closed':closed,
            't_rep':first['step'] if first else None,'t_closed':closed['step'] if closed else None,
            'open_memory_occurrence':bool(occurrence),'occurrence':occurrence,
            'persistent_open_memory':persistent,'same_genotype_persistence':same,
            'persistence_samples':{str(s):sorted(t) for s,t in persistence.items()}}

def judge_tape(tape, arm, seed, ex, diagnostic):
    parent = np.frombuffer(bytes.fromhex(tape),np.uint8); L = len(parent)
    if L != ARMS[arm]['L']: raise ValueError('tape length mismatch')
    seed64 = int.from_bytes(key(arm,seed,L,'culture',tape)[:8],'little')
    culture = A.assay_many(parent[None],z80_steps=128,n=64,seed=seed64,zero_halts=False)[0]
    gen2 = culture['gen2_score']
    crng = rng(arm,seed,L,'culture',tape)
    cultures = [crng.integers(0,256,(64,L),dtype=np.uint8) for _ in range(2)]
    p = partners(arm,seed,L,tape)
    panel = np.vstack((np.zeros((1,L),np.uint8),p))
    mem = np.concatenate((np.repeat(parent[None],17,axis=0),panel),axis=1)
    m = ex.machine_for(L,zero_halts=False,maps=True)
    raw = m.run(mem,budget=128); obs = diagnostic.run(mem,budget=128)
    if not same_execution(raw,obs,m.record_words): raise ValueError('diagnostic execution differs from full lane record')
    entered = raw.fetch_map[1:,L:].any(axis=1)
    logs = diagnostic.logs(obs); diags = diagnostic.diagnostics(obs)
    rows = []
    for v in range(1,17):
        cls,reason = mechanism(logs[v],L)
        rows.append({'partner':v-1,'mechanism':cls,'reason':reason,'open':bool(entered[v-1]),
                     'partner_write_coverage':int(diags[v,4]),'write_transactions':int(raw.writes('B')[v]),
                     'original_partner_fetch_reads':int(diags[v,0]),'original_partner_data_reads':int(diags[v,1]),
                     'trace_sha256':sha(logs[v].tobytes()) if logs[v] is not None else None,
                     'record_sha256':sha(raw.record[v].tobytes()),'output_sha256':sha(raw.memory[v].tobytes())})
    counts = Counter(r['mechanism'] for r in rows)
    majority = next((k for k,v in counts.items() if v >= 9),None)
    numerator = sum(r['open'] and r['mechanism']=='memory-fed one-pass copier' for r in rows)
    return safe({'gen2':gen2,'heritable':bool(np.isfinite(gen2) and gen2>=.3), 'culture':culture,
                 'culture_partners':[c.tobytes().hex() for c in cultures], 'culture_seed':seed64,
                 'route_partners':[t.tobytes().hex() for t in p], 'entered':int(entered.sum()),
                 'open':bool(entered.sum()>=8),'confined':bool(not entered.any()),
                 'fetch_class':'open' if entered.sum()>=8 else 'confined' if not entered.any() else 'intermediate',
                 'majority_mechanism':majority,'mechanism_counts':dict(counts),'encounters':rows,
                 'open_memory_numerator':numerator,'open_memory_fed':bool(numerator>=9),
                 'zero_reference':{'coverage':int(diags[0,4]),'write_transactions':int(raw.writes('B')[0]),
                                   'original_partner_fetch_reads':int(diags[0,0]),
                                   'original_partner_data_reads':int(diags[0,1]),
                                   'mechanism':mechanism(logs[0],L)},
                 'max_B_hamming_from_zero':int((raw.memory[1:,L:]!=raw.memory[0,L:]).sum(axis=1).max()),
                 'input_sha256':sha(mem.tobytes()),'record_sha256':sha(raw.record.tobytes()),
                 'diagnostic_identity':True})

def load_snapshot(path, record, L):
    if Path(record['file']).name != record['file']: raise ValueError('unsafe snapshot filename')
    compressed = (Path(path).parent/record['file']).read_bytes()
    if sha(compressed) != record['sha256_compressed']: raise ValueError('compressed snapshot hash mismatch')
    raw = brotli.decompress(compressed)
    if len(raw) != 20000*L or len(raw) != record['raw_bytes'] or record['shape'] != [20000,L]:
        raise ValueError('snapshot size/shape mismatch')
    if sha(raw) != record['sha256_raw']: raise ValueError('snapshot hash mismatch')
    return np.frombuffer(raw,np.uint8).reshape(20000,L)

def read_input(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not records or records[0].get('schema') != 's2-world/1': raise ValueError('missing S2 condition')
    c = records[0]; arm,seed = c['arm'],c['seed']
    if arm not in ARMS or seed not in SEEDS or c['steps'] != HORIZON:
        raise ValueError('unregistered condition')
    if any(c[k] != v for k,v in ARMS[arm].items() if k != 'machine') or c['machine_name'] != ARMS[arm]['machine']:
        raise ValueError('condition differs from registered machine/arm')
    for k,v in {'width':160,'height':125,'pair_count':8192,'noise_exp':4,
                'mutation_attempts_per_step':512,'sample_steps':samples(),'snapshot_steps':list(SNAPSHOTS)}.items():
        if c[k] != v: raise ValueError(f'condition differs: {k}')
    sr = [r for r in records if r.get('kind') == 'sample']
    if [r['step'] for r in sr] != samples(): raise ValueError('missing/corrupt/duplicate sample')
    summaries = [r for r in records if r.get('kind') == 'summary']
    if (len(summaries)!=1 or summaries[0]['steps_run']!=HORIZON or summaries[0]['samples']!=585
            or any(r.get('kind')=='incomplete' for r in records)):
        raise ValueError('incomplete record; cannot be a biological negative')
    parsed=[]; prev_active=0
    for r in sr:
        tapes=[]
        for t in r['top3']:
            b=bytes.fromhex(t['tape'])
            if (len(b)!=c['L'] or sha(b)!=t['sha256'] or not isinstance(t['count'],int)
                    or not 0<t['count']<=20000 or t['share']!=t['count']/20000):
                raise ValueError('top tape/count/hash/share mismatch')
            tapes.append((t['tape'],t['share']))
        if (not 1<=len(tapes)<=3 or len({t for t,s in tapes})!=len(tapes)
                or tapes!=sorted(tapes,key=lambda x:(-x[1],bytes.fromhex(x[0])))):
            raise ValueError('top-three ordering/duplicates mismatch')
        if (r['pair_draws_total']!=r['step']*8192 or not prev_active<=r['active_pairs_total']<=r['pair_draws_total']):
            raise ValueError('active pair accounting mismatch')
        prev_active=r['active_pairs_total']; parsed.append({'step':r['step'],'top3':tapes})
    snapshots=[r for r in records if r.get('kind')=='snapshot']
    if [r['step'] for r in snapshots]!=list(SNAPSHOTS): raise ValueError('missing/duplicate checkpoint')
    for r in snapshots:
        loaded=load_snapshot(path,r,c['L'])
        sample=next(s for s in sr if s['step']==r['step'])
        if sha(loaded.tobytes())!=sample['soup_sha256']: raise ValueError('sample/checkpoint identity mismatch')
        if r['step']==0 and sha(loaded.tobytes())!=c['initial_sha256']: raise ValueError('initial identity mismatch')
    return c,parsed,snapshots[-1]

def score_world(path, *, device=None, tick=lambda:None):
    verify_freeze()
    c,records,snapshot=read_input(path); arm,seed,L=c['arm'],c['seed'],c['L']
    ex=E.ZilionExecutor(ARMS[arm]['machine'],device=device)
    m=ex.machine_for(L,zero_halts=False,maps=True)
    for k in ('sha256_core','sha256_world','sha256_wgsl'):
        if m.describe()[k]!=c['machine'][k]: raise ValueError('world machine hash mismatch')
    if c['freeze_sha256']!=sha((Path(__file__).parent/'results/s8086/prelaunch/freeze.json').read_bytes()):
        raise ValueError('world freeze identity mismatch')
    trace=diagnostic_for(m); cache={}
    def judge(tape):
        tick()
        if tape not in cache: cache[tape]=judge_tape(tape,arm,seed,ex,trace)
        return cache[tape]
    with E.using(ex):
        ep=endpoints(records,judge)
        soup=load_snapshot(path,snapshot,L)
        # Draw indices before reading any cell genotype or candidate score.
        indices=rng(arm,seed,L,'cells-final').choice(20000,64,replace=False)
        candidates=[]
        for source,event in (('first',ep['first']),('closed',ep['closed'])):
            if event: candidates.append({'source':source,'tape':event['tape'],'step':event['step']})
        for t,s in records[-1]['top3']: candidates.append({'source':'final-top','tape':t,'share':s,'step':HORIZON})
        for i in indices: candidates.append({'source':'random-cell','cell':int(i),'tape':soup[i].tobytes().hex(),'step':HORIZON})
        for r in candidates: r['culture']=judge(r['tape'])['culture']
    world={'world':Path(path).stem,'arm':arm,'seed':seed,'L':L,'horizon':HORIZON,
           'scoreable':ep['scoreable'],'t_rep':ep['t_rep'],'t_closed':ep['t_closed'],
           'first':ep['first'],'closed':ep['closed'],'open_memory_occurrence':ep['open_memory_occurrence'],
           'persistent_open_memory':ep['persistent_open_memory'],
           'same_genotype_persistence':ep['same_genotype_persistence']}
    return safe({'schema':'s2-score/1','world':world,'endpoints':ep,'candidates':candidates,
                 'random_cell_indices':indices.tolist(),'random_cell_denominator':64,
                 'random_cell_unknowns':sum(cache[r['tape']]['gen2'] is None for r in candidates if r['source']=='random-cell'),
                 'judged_tapes':[{'tape':t,**r} for t,r in sorted(cache.items())],
                 'input_sha256':sha(Path(path).read_bytes()),'executor':ex.describe(),'usage':E.usage()})

def km(rows, field):
    events=Counter(r[field] for r in rows if r[field] is not None)
    at_risk=len(rows); survival=1.; curve=[]; median=None
    for t,n in sorted(events.items()):
        survival*=1-n/at_risk; curve.append({'step':t,'at_risk':at_risk,'events':n,'survival':survival})
        at_risk-=n
        if median is None and survival<=.5: median=t
    return {'curve':curve,'median':median,'right_censored':at_risk,'censor_step':HORIZON}

def binomial_interval(k,n):
    if not n: return None
    z=1.959963984540054; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'count':k,'n':n,'wilson95':[max(0.,center-half),min(1.,center+half)],
            'descriptive_only':True}

def arm_verdict(results, arm):
    rows=[r['world'] for r in results if r['world']['arm']==arm]
    if len({r['seed'] for r in rows})!=len(rows): raise ValueError('duplicate world')
    complete=(len(rows)==20 and {r['seed'] for r in rows}==set(SEEDS)
              and all(r['scoreable'] and r['horizon']==HORIZON and r['L']==ARMS[arm]['L'] for r in rows))
    scoreable=[r for r in rows if r['scoreable']]
    rep=[r for r in scoreable if r['t_rep'] is not None]; D=len(rep)
    O=sum(r['first']['open'] for r in rep); closed=sum(r['first']['confined'] for r in rep)
    occurrence=sum(r['open_memory_occurrence'] for r in rows); persistent=sum(r['persistent_open_memory'] for r in rows)
    counts=Counter((r['first']['fetch_class'],r['first']['majority_mechanism'] or 'heterogeneous') for r in rep)
    encounter_counts=Counter((r['first']['fetch_class'],e['open'],e['mechanism']) for r in rep for e in r['first']['encounters'])
    return {'arm':arm,'role':'primary' if arm=='A' else 'secondary separate scope','status':'complete' if complete else 'incomplete/provisional',
            'worlds':len(rows),'scoreable_worlds':len(scoreable),'unscoreable_worlds':len(rows)-len(scoreable),
            'D':D,'O':O,'first_confined':closed,'occurrence_worlds':occurrence,'persistence_worlds':persistent,
            'R0':D>=10 if complete else None,'R1':O>=math.ceil(.8*D) if complete and D>=10 else None,
            'R2a':occurrence>=2 if complete else None,'R2b':persistent>=1 if complete else None,
            'R2_joint':occurrence>=2 and persistent>=1 if complete else None,
            'kill':closed>=math.ceil(.25*D) if complete and D>=10 else None,
            'km_rep':km(scoreable,'t_rep'),'km_closed':km(scoreable,'t_closed'),
            'descriptive_binomial_intervals':{'R0':binomial_interval(D,20),'R1':binomial_interval(O,D),
                    'R2a':binomial_interval(occurrence,20),'R2b':binomial_interval(persistent,20)} if complete else None,
            'first_tape_distribution':[{'fetch':k[0],'mechanism':k[1],'count':v} for k,v in sorted(counts.items())],
            'first_encounter_distribution':[{'tape_fetch':k[0],'open':k[1],'mechanism':k[2],'count':v} for k,v in sorted(encounter_counts.items())]}

def report(results,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    write_new(out/'route_verdicts.json',{'arms':[arm_verdict(results,a) for a in ARMS]})
    for r in results: write_new(out/'worlds'/f"{r['world']['world']}.json.gz",r)

if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    out=a.out if a.out.suffix=='.gz' else Path(str(a.out)+'.gz')
    write_new(out,score_world(a.input))

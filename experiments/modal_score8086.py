"""Future S2 scoring/extraction/assays. Frozen launch release required; no retries."""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path
import modal
if Path('/root/experiments').is_dir(): sys.path.insert(0,'/root/experiments')
from modal_s8086 import image,volume
from s8086 import HERE,PRE,ARMS,SEEDS,RATE,sha,launch_receipt
app=modal.App('algocell-s8086-scoring')

def locations(arm,seed,stage):
    if arm not in ARMS or seed not in SEEDS or stage not in ('score','extract','assay'):
        raise ValueError('unregistered scoring input')
    from run8086 import world_name
    name=world_name(arm,seed)
    return f'/vol/s8086/{arm}/{name}.jsonl',f'/vol/s8086_{stage}/{arm}/{name}'

@app.function(image=image,gpu='L4',cpu=(1,1),memory=(8192,8192),timeout=12*3600,
              retries=0,single_use_containers=True,max_containers=1,volumes={'/vol':volume})
def score(arm,seed,stage,receipt,deadline_s,eligibility_commit=''):
    import sys
    sys.path.insert(0,'/root/experiments')
    import s8086,score8086,law8086
    from algocell_exp.zmachine import load_machine
    source,destination=locations(arm,seed,stage)
    freeze=json.loads((PRE/'freeze.json').read_text())
    for filename,digest in freeze['files'].items():
        if sha((HERE/filename).read_bytes())!=digest: raise ValueError('remote frozen source differs')
    if (receipt.get('population_launch_approved') is not True or
            receipt.get('freeze_sha256')!=sha((PRE/'freeze.json').read_bytes())):
        raise ValueError('coordinator release required')
    s8086.committed=lambda p:sha(Path(p).read_bytes())
    volume.reload(); out=Path(destination); start=time.monotonic()
    # Reserve output exclusively before compute; errors keep marker and liability.
    out.parent.mkdir(parents=True,exist_ok=True)
    marker=Path(str(out)+'.reserved.json')
    s8086.write_new(marker,{'arm':arm,'seed':seed,'stage':stage,'deadline_s':deadline_s})
    volume.commit()
    def tick():
        if time.monotonic()-start>deadline_s-30: raise TimeoutError('scoring spend stop; incomplete')
    try:
        if stage=='score':
            result=score8086.score_world(Path(source),tick=tick); s8086.write_new(Path(str(out)+'.json'),result)
        else:
            m=load_machine(ARMS[arm]['machine'])
            scored=Path(f'/vol/s8086_score/{arm}/{out.name}.json')
            if stage=='extract':
                result=law8086.freeze_candidates(json.loads(scored.read_text()),Path(str(out)+'.json'),m,tick=tick)
            else:
                eligibility=Path(f'/vol/s8086_extract/{arm}/{out.name}.json')
                # The later local data commit must be attested in the release;
                # remote extraction alone never authorizes retention assays.
                digest=sha(eligibility.read_bytes())
                if (len(eligibility_commit)!=40 or receipt.get('eligibility_commit')!=eligibility_commit
                        or receipt.get('eligibility_sha256',{}).get(out.name)!=digest):
                    raise ValueError('committed eligibility data release required before interventions')
                law8086.committed=lambda p:sha(Path(p).read_bytes())
                result=law8086.assay_frozen(eligibility,out,m,tick=tick)
    finally: volume.commit()
    wall=time.monotonic()-start
    if wall>deadline_s: raise RuntimeError('scoring allocation exceeded')
    return {'result_path':str(out),'resource_time_bound_usd':(deadline_s+120)*RATE}

@app.function(image=image,gpu='L4',cpu=(1,1),memory=(8192,8192),timeout=3600,
              retries=0,single_use_containers=True,max_containers=1,volumes={'/vol':volume})
def positive_controls(receipt,deadline_s):
    import sys
    sys.path.insert(0,'/root/experiments')
    import law8086
    from algocell_exp.zmachine import load_machine
    from algocell_exp.census8086_trace import DiagnosticMachine
    freeze=json.loads((PRE/'freeze.json').read_text())
    for filename,digest in freeze['files'].items():
        if sha((HERE/filename).read_bytes())!=digest: raise ValueError('control frozen source differs')
    if (receipt.get('population_launch_approved') is not True or
            receipt.get('freeze_sha256')!=sha((PRE/'freeze.json').read_bytes())):
        raise ValueError('coordinator release required')
    volume.reload(); start=time.monotonic()
    def tick():
        if time.monotonic()-start>deadline_s-30: raise TimeoutError('control allocation stop')
    machines={}
    for L in (16,32):
        tick(); m=load_machine(f'i8086_pair_maps_L{L}'); machines[L]=(m,DiagnosticMachine(m,events=8192))
    try: result=law8086.controls(Path('/vol/s8086_controls'),machines,tick=tick)
    finally: volume.commit()
    return {'results':result,'resource_time_bound_usd':(deadline_s+120)*RATE}

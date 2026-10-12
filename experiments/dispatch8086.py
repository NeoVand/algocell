"""Serialized dispatch accounting: append-only hash chain, liabilities, no retries.

One coordinator owns the local ledger. Reservations are recorded before spawn;
crash/unknown calls retain their entire liability and cannot be automatically
retried. Container timeouts bound running obligations despite billing lag.
"""
from __future__ import annotations
import fcntl
import json
import math
import os
import time
from pathlib import Path
from s8086 import CAP, RATE, sha, safe, launch_receipt

class SpendGuard:
    def __init__(self, path, plan, *, cap=CAP, overhead=1.):
        self.path=Path(path); self.plan=dict(plan); self.cap=cap; self.overhead=overhead
        if (not math.isfinite(cap) or not 0<cap<=CAP or not math.isfinite(overhead) or overhead<0
                or any(not math.isfinite(v) or v<0 for v in plan.values())):
            raise ValueError('invalid spend plan')
        self.path.parent.mkdir(parents=True,exist_ok=True)

    def transact(self, event):
        if 'usd' in event and (not isinstance(event['usd'],(int,float)) or
                               not math.isfinite(event['usd']) or event['usd']<0):
            raise ValueError('nonfinite/negative spend')
        with self.path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX); f.seek(0)
            lines=[json.loads(s) for s in f if s.strip()]; previous='0'*64
            jobs={}; billed={}; known_cost=0.; reserved=0.
            for row in lines:
                digest=row.pop('hash')
                if row['previous']!=previous or sha(json.dumps(row,sort_keys=True,separators=(',',':')).encode())!=digest:
                    raise ValueError('ledger hash chain corrupt')
                previous=digest
                if row['kind']=='plan':
                    if row['plan']!=self.plan or row['cap']!=self.cap or row['overhead']!=self.overhead:
                        raise ValueError('spend plan changed')
                elif row['kind']=='reserve': jobs[row['job']]={'reservation':row['usd'],'status':'reserved'}
                elif row['kind']=='started': jobs[row['job']]['status']='running'
                elif row['kind']=='settled':
                    jobs[row['job']]['status']='settled'; jobs[row['job']]['cost']=row['usd']
                elif row['kind']=='metered': billed[row['app_id']]=row['usd']
            if not lines:
                init={'kind':'plan','plan':self.plan,'cap':self.cap,'overhead':self.overhead,'previous':previous,'time':time.time()}
                previous=sha(json.dumps(init,sort_keys=True,separators=(',',':')).encode())
                f.write(json.dumps({**init,'hash':previous})+'\n'); f.flush(); os.fsync(f.fileno())
            for name,j in jobs.items():
                if j['status']=='settled': known_cost+=j['cost']
                else: reserved+=j['reservation']
            unstarted=sum(v for k,v in self.plan.items() if k not in jobs)
            total=max(known_cost,sum(billed.values()))+reserved+unstarted+self.overhead
            kind=event['kind']
            if kind=='reserve':
                name=event['job']
                if name in jobs: raise FileExistsError('job already reserved; no automatic retry')
                if name not in self.plan or event['usd']<self.plan[name]: raise ValueError('unplanned/underreserved job')
                total+=event['usd']-self.plan[name]
                if total>self.cap: raise RuntimeError(f'spend stop: complete liabilities ${total:.4f} > ${self.cap}')
            elif kind=='settled':
                if event['job'] not in jobs or jobs[event['job']]['status']=='settled': raise ValueError('invalid settlement')
                if not math.isfinite(event['usd']) or event['usd']<0 or event['usd']>jobs[event['job']]['reservation']:
                    raise ValueError('resource-time bound exceeded reservation')
            elif kind=='started':
                if event['job'] not in jobs or jobs[event['job']]['status']!='reserved': raise ValueError('invalid start')
                if not event.get('call_id'): raise ValueError('call provenance required')
            elif kind=='metered':
                if not math.isfinite(event['usd']) or event['usd']<billed.get(event['app_id'],0):
                    raise ValueError('billing must be cumulative/nondecreasing')
            else: raise ValueError('unknown spend event')
            row={'previous':previous,'time':time.time(),**safe(event)}
            row['hash']=sha(json.dumps(row,sort_keys=True,separators=(',',':')).encode())
            f.seek(0,2); f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
            return {'projected_liabilities_usd':total,'cap':self.cap}

    def reserve(self,job,usd,**provenance): return self.transact({'kind':'reserve','job':job,'usd':usd,**provenance})
    def start(self,job,call_id,app_id): return self.transact({'kind':'started','job':job,'call_id':call_id,'app_id':app_id})
    def settle(self,job,usd,**provenance): return self.transact({'kind':'settled','job':job,'usd':usd,**provenance})

def dispatch_worlds(release, ledger, projection):
    """Future launch entry; every world/scorer/extractor/assay remains reserved."""
    launch_receipt(release)
    frozen=json.loads((Path(__file__).parent/'results/s8086/prelaunch/projection.json').read_text())
    if projection!=frozen: raise ValueError('projection differs from frozen allocation plan')
    from modal_s8086 import app,world
    from s8086 import ARMS,SEEDS,HERE
    import subprocess
    plan=projection['job_liabilities_usd']
    guard=SpendGuard(ledger,plan,overhead=projection['storage_termination_reserve_usd'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    receipt=json.loads(Path(release).read_text())
    # No detach: app lifetime and stop are owned by the same dispatcher.
    bounded={a:world.with_options(timeout=math.ceil(projection['world_deadline_s'][a])) for a in ARMS}
    with app.run():
        for arm in ARMS:
            for seed in SEEDS:
                job=f'world:{arm}:{seed}'; allocation=projection['world_deadline_s'][arm]
                guard.reserve(job,plan[job],git_commit=commit)
                call=bounded[arm].spawn(arm,seed,commit,receipt,allocation)
                guard.start(job,call.object_id,app.app_id)
                result=call.get()  # errors retain liability; zero retries
                guard.settle(job,result['resource_time_bound_usd'],app_id=app.app_id,call_id=call.object_id)

def dispatch_scoring(stage,release,ledger,projection):
    launch_receipt(release)
    frozen=json.loads((Path(__file__).parent/'results/s8086/prelaunch/projection.json').read_text())
    if projection!=frozen: raise ValueError('projection differs from frozen allocation plan')
    from modal_score8086 import app,score,positive_controls
    from s8086 import ARMS,SEEDS,HERE,committed
    import subprocess
    receipt=json.loads(Path(release).read_text())
    plan=projection['job_liabilities_usd']
    guard=SpendGuard(ledger,plan,overhead=projection['storage_termination_reserve_usd'])
    jobs=[]
    if stage=='controls': jobs=[('controls',None,None,positive_controls)]
    else:
        kind='law' if stage=='assay' else stage
        jobs=[(f'{kind}:{a}:{s}',a,s,score) for a in ARMS for s in SEEDS]
    commit=receipt.get('eligibility_commit','')
    if stage=='assay':
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
        if head!=commit: raise ValueError('assay data release must identify current committed eligibility ledger')
        for _,arm,seed,_ in jobs:
            from run8086 import world_name
            name=world_name(arm,seed)
            local=HERE/f'results/s8086/eligibility/{arm}/{name}.json'
            if committed(local)!=receipt.get('eligibility_sha256',{}).get(name):
                raise ValueError('eligibility ledger differs from released data commit')
    bounded=[]
    for job,arm,seed,function in jobs:
        deadline=math.floor(plan[job]/RATE-120)
        if deadline<=30: raise ValueError('allocation too small')
        bounded.append((job,arm,seed,function.with_options(timeout=deadline),deadline))
    with app.run():
        for job,arm,seed,function,deadline in bounded:
            guard.reserve(job,plan[job],stage=stage,eligibility_commit=commit)
            call=(function.spawn(receipt,deadline) if stage=='controls' else
                  function.spawn(arm,seed,stage,receipt,deadline,commit))
            guard.start(job,call.object_id,app.app_id)
            result=call.get()
            guard.settle(job,result['resource_time_bound_usd'],app_id=app.app_id,call_id=call.object_id)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release',type=Path,required=True); ap.add_argument('--ledger',type=Path,required=True)
    ap.add_argument('--projection',type=Path,required=True)
    ap.add_argument('--stage',choices=('worlds','score','extract','assay','controls'),default='worlds')
    a=ap.parse_args()
    if a.stage=='worlds': dispatch_worlds(a.release,a.ledger,json.loads(a.projection.read_text()))
    else: dispatch_scoring(a.stage,a.release,a.ledger,json.loads(a.projection.read_text()))

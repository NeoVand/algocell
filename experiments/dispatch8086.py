"""Serialized dispatch accounting: append-only hash chain, liabilities, no retries.

One coordinator owns the local ledger. Reservations are recorded before spawn;
crash/unknown calls retain their entire liability and cannot be automatically
retried. An independent lifecycle process bounds all attempts and stopping; billing
lag never renews the clock. Unknown termination retains the full reservation.
"""
from __future__ import annotations
import fcntl
import json
import math
import os
import time
from pathlib import Path
from s8086 import CAP, RATE, sha, safe, launch_receipt, released_registration

def clone_accounting(jobs,billed,plan):
    # The replay state contains only scalar fields, not the event payloads.
    return {name:dict(job) for name,job in jobs.items()},dict(billed),dict(plan)

class SpendGuard:
    def __init__(self, path, plan, *, cap=CAP, overhead=1.):
        self.path=Path(path); self.plan=dict(plan); self.cap=cap; self.overhead=overhead
        self._replay=None
        if (not math.isfinite(cap) or not 0<cap<=CAP or not math.isfinite(overhead) or overhead<0
                or any(not math.isfinite(v) or v<0 for v in plan.values())):
            raise ValueError('invalid spend plan')
        self.path.parent.mkdir(parents=True,exist_ok=True)

    def transact(self, event, *, nonblocking=False):
        if 'usd' in event and (not isinstance(event['usd'],(int,float)) or
                               not math.isfinite(event['usd']) or event['usd']<0):
            raise ValueError('nonfinite/negative spend')
        with self.path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)); f.seek(0)
            raw=f.read(); previous='0'*64
            jobs={}; billed={}; known_cost=0.; reserved=0.; plan=dict(self.plan)
            offset=0
            if self._replay:
                cached=self._replay; offset=cached['offset']
                if cached['config']!=(self.plan,self.cap,self.overhead):
                    raise ValueError('spend plan changed')
                if len(raw)<offset or sha(raw[:offset].encode())!=cached['digest']:
                    raise ValueError('ledger hash chain corrupt: verified prefix changed')
                jobs,billed,plan=clone_accounting(*cached['state']);previous=cached['previous']
            lines=[json.loads(s) for s in raw[offset:].splitlines() if s.strip()]
            for row in lines:
                digest=row.pop('hash')
                if row['previous']!=previous or sha(json.dumps(row,sort_keys=True,separators=(',',':')).encode())!=digest:
                    raise ValueError('ledger hash chain corrupt')
                previous=digest
                if row['kind']=='plan':
                    if row['plan']!=self.plan or row['cap']!=self.cap or row['overhead']!=self.overhead:
                        raise ValueError('spend plan changed')
                elif row['kind']=='reserve': jobs[row['job']]={'reservation':row['usd'],'status':'reserved'}
                elif row['kind']=='started': jobs[row['job']].update(status='running',app_id=row['app_id'],call_id=row['call_id'])
                elif row['kind']=='lifecycle': jobs[row['job']]['app_id']=row['app_id']
                elif row['kind']=='stopped': jobs[row['job']]['stopped']=True
                elif row['kind']=='uncertain': jobs[row['job']]['unknown_bound']=True
                elif row['kind']=='terminal': jobs[row['job']]['experiment_stopped']=True
                elif row['kind']=='liability':
                    jobs[row['job']].update(liability=row['usd'],elapsed_usd=row.get('elapsed_usd',0),app_id=row.get('app_id',jobs[row['job']].get('app_id')))
                elif row['kind']=='settled':
                    jobs[row['job']]['status']='settled'; jobs[row['job']]['cost']=row['usd']
                elif row['kind']=='metered': billed[row['app_id']]=row['usd']
                elif row['kind']=='plan_extension': plan.update(row['jobs'])
            if raw:
                self._replay={'offset':len(raw),'digest':sha(raw.encode()),'previous':previous,
                              'config':(dict(self.plan),self.cap,self.overhead),
                              'state':clone_accounting(jobs,billed,plan)}
            if not raw:
                init={'kind':'plan','plan':self.plan,'cap':self.cap,'overhead':self.overhead,'previous':previous,'time':time.time()}
                previous=sha(json.dumps(init,sort_keys=True,separators=(',',':')).encode())
                f.write(json.dumps({**init,'hash':previous})+'\n'); f.flush(); os.fsync(f.fileno())
            def complete_projection():
                # Observed elapsed cost is a spend floor. Billing for that same
                # app overlaps elapsed cost; it is never added twice. Remaining
                # server/termination liabilities and all future work are added.
                total=self.overhead+sum(v for k,v in plan.items() if k not in jobs)
                accounted_apps=set()
                for j in jobs.values():
                    app=j.get('app_id');meter=billed.get(app,0);accounted_apps.add(app)
                    if j['status']=='settled':total+=max(j['cost'],meter)
                    else:
                        liability=j.get('liability',j['reservation']);elapsed=j.get('elapsed_usd',0)
                        total+=max(meter,elapsed)+(liability-elapsed)
                total+=sum(v for app,v in billed.items() if app not in accounted_apps)
                return total
            total=complete_projection()
            unknown=[name for name,j in jobs.items() if j.get('unknown_bound')]
            experiment_stopped=any(j.get('experiment_stopped') for j in jobs.values())
            kind=event['kind']
            if unknown and kind in ('reserve','plan_extension'):
                raise RuntimeError('spend stop: unknown physical liabilities require reconciliation')
            if experiment_stopped and kind=='reserve':raise RuntimeError('INCOMPLETE experiment; dispatch stopped')
            if kind=='reserve':
                name=event['job']
                if name in jobs: raise FileExistsError('job already reserved; no automatic retry')
                if name not in plan or event['usd']<plan[name]: raise ValueError('unplanned/underreserved job')
                total+=event['usd']-plan[name]
                if total>self.cap: raise RuntimeError(f'spend stop: complete liabilities ${total:.4f} > ${self.cap}')
            elif kind=='lifecycle':
                if event['job'] not in jobs or jobs[event['job']]['status']!='reserved' or not event.get('app_id'):
                    raise ValueError('invalid lifecycle start')
            elif kind=='stopped':
                j=jobs.get(event['job'])
                if (not j or j.get('app_id')!=event.get('app_id') or event.get('confirmed') is not True
                        or event.get('tasks_empty') is not True): raise ValueError('unconfirmed stopping')
            elif kind=='uncertain':
                if event['job'] not in jobs or not event.get('reason'): raise ValueError('unknown uncertainty event')
                unknown=sorted(set(unknown)|{event['job']})
            elif kind=='terminal':
                if event['job'] not in jobs: raise ValueError('unknown terminal job')
            elif kind=='check': pass
            elif kind=='liability':
                j=jobs.get(event['job'])
                if not j or j['status']=='settled': raise ValueError('unknown/live liability job')
                elapsed=event.get('elapsed_usd',0)
                if not math.isfinite(elapsed) or not 0<=elapsed<=event['usd']:raise ValueError('invalid observed elapsed cost')
                j.update(liability=event['usd'],elapsed_usd=elapsed,app_id=event.get('app_id',j.get('app_id')))
                total=complete_projection()
            elif kind=='settled':
                if event['job'] not in jobs or jobs[event['job']]['status']=='settled': raise ValueError('invalid settlement')
                if not math.isfinite(event['usd']) or event['usd']<0 or event['usd']>max(jobs[event['job']]['reservation'],jobs[event['job']].get('liability',0)):
                    raise ValueError('resource-time bound exceeded reservation')
                if not jobs[event['job']].get('stopped'): raise ValueError('confirmed stopping required before settlement')
                if jobs[event['job']].get('unknown_bound'): raise ValueError('physical resource bound unresolved; liability retained')
                jobs[event['job']].update(status='settled',cost=event['usd'])
                total=complete_projection()
            elif kind=='started':
                if event['job'] not in jobs or jobs[event['job']]['status']!='reserved': raise ValueError('invalid start')
                if not event.get('call_id') or not event.get('app_id'): raise ValueError('call/app provenance required')
            elif kind=='metered':
                if not math.isfinite(event['usd']) or event['usd']<billed.get(event['app_id'],0):
                    raise ValueError('billing must be cumulative/nondecreasing')
                billed[event['app_id']]=event['usd']
                total=complete_projection()
            elif kind=='plan_extension':
                extra=event['jobs']
                if (not event.get('reason') or not extra or set(extra)&set(plan)
                        or any(not math.isfinite(v) or v<0 for v in extra.values())):
                    raise ValueError('invalid prospective manual plan extension')
                total+=sum(extra.values())
                if total>self.cap: raise RuntimeError('spend stop: extended complete liabilities exceed cap')
            else: raise ValueError('unknown spend event')
            row={'previous':previous,'time':time.time(),**safe(event)}
            row['hash']=sha(json.dumps(row,sort_keys=True,separators=(',',':')).encode())
            f.seek(0,2); f.write(json.dumps(row)+'\n'); f.flush(); os.fsync(f.fileno())
            return {'projected_liabilities_usd':total,'cap':self.cap,'stop_required':total>self.cap or bool(unknown) or experiment_stopped,'unknown_physical_liabilities':unknown}

    def reserve(self,job,usd,**provenance): return self.transact({'kind':'reserve','job':job,'usd':usd,**provenance})
    def start(self,job,call_id,app_id): return self.transact({'kind':'started','job':job,'call_id':call_id,'app_id':app_id})
    def settle(self,job,usd,**provenance): return self.transact({'kind':'settled','job':job,'usd':usd,**provenance})

def bounded_call(app,function,args,guard,job,execution_s,*,watchdog_factory=None):
    """One app per job; no next job until provider-confirmed termination.

    The independent process is armed before spawn/get, and continues when
    either blocks or the dispatcher dies. Body results never settle a live app.
    """
    from lifecycle8086 import WatchdogProcess,dispatch_seconds
    factory=watchdog_factory or WatchdogProcess
    if guard.plan[job]+1e-12<dispatch_seconds(execution_s)*RATE:
        raise ValueError('lifecycle envelope underreserved')
    guard.reserve(job,guard.plan[job],execution_s=execution_s,lifecycle_s=dispatch_seconds(execution_s))
    monitor=None; call=None; result=None; failure=None; acknowledged={'phase':'reserved','step':None}
    try:
        # Clock/independent app discovery are armed before app.run setup.
        monitor=factory(guard,job,None,execution_s)
        with app.run(name=monitor.app_name):
            guard.transact({'kind':'lifecycle','job':job,'app_id':app.app_id})
            monitor.bind_app(app.app_id)
            try:
                monitor.permit()  # current complete-cap decision before spawn
                call=function.spawn(*args)
                monitor.bind(call.object_id)
                guard.start(job,call.object_id,app.app_id)
                acknowledged={'phase':'spawn_acknowledged','step':None}
                result=call.get()
                acknowledged={'phase':'result_acknowledged','summary':result.get('summary'),'step':None}
            except BaseException as exc:
                failure=exc
                guard.transact({'kind':'terminal','job':job,'app_id':app.app_id,
                                'call_id':call.object_id if call else None,'status':'incomplete/invalid',
                                'error':type(exc).__name__,'reason':str(exc),
                                'last_acknowledged_progress':acknowledged})
            finally:
                monitor.finish()
                try: monitor.wait_confirmed()
                except BaseException as exc:
                    guard.transact({'kind':'terminal','job':job,'app_id':app.app_id,
                                    'call_id':call.object_id if call else None,'status':'stopping_unconfirmed',
                                    'error':type(exc).__name__,'reason':str(exc),
                                    'original_error':str(failure) if failure else None,
                                    'last_acknowledged_progress':acknowledged,'liability_retained':True})
                    if failure is None: failure=exc
    except BaseException as exc:
        if monitor: monitor.finish()
        guard.transact({'kind':'terminal','job':job,'status':'incomplete/invalid','error':type(exc).__name__,
                        'reason':str(exc),'last_acknowledged_progress':acknowledged,'liability_retained':True})
        raise
    if failure is not None: raise failure
    charged=max(guard.plan[job],monitor.confirmation().get('charged_usd',0))
    guard.settle(job,charged,app_id=app.app_id,call_id=call.object_id,
                 stopping_confirmed=True,accounting='dynamic task-attempt liabilities and confirmed stopping')
    return result


def dispatch_worlds(release, ledger, projection):
    receipt=launch_receipt(release); registration=released_registration(receipt)
    frozen=json.loads((Path(__file__).parent/'results/s8086/prelaunch/projection.json').read_text())
    if projection!=frozen or projection.get('registration')!=registration.as_dict():
        raise ValueError('projection differs from prospective registered allocation plan')
    from prebuild8086 import require_prebuild
    image_id=require_prebuild()
    from modal_s8086 import app,world,PREBUILT_IMAGE_ID
    if PREBUILT_IMAGE_ID!=image_id:raise ValueError('fresh dispatcher must load the prebuilt image; builds forbidden')
    from s8086 import SEEDS,HERE
    import subprocess
    guard=SpendGuard(ledger,projection['job_liabilities_usd'],overhead=projection['storage_termination_reserve_usd'])
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    for arm in registration.arms:
        for seed in SEEDS:
            job=f'world:{arm}:{seed}'; deadline=projection['execution_deadline_s'][job]
            function=world.with_options(timeout=deadline,max_containers=1,buffer_containers=0,retries=0)
            bounded_call(app,function,(arm,seed,commit,receipt,deadline),guard,job,deadline)


def dispatch_scoring(stage,release,ledger,projection):
    receipt=launch_receipt(release); registration=released_registration(receipt)
    frozen=json.loads((Path(__file__).parent/'results/s8086/prelaunch/projection.json').read_text())
    if projection!=frozen or projection.get('registration')!=registration.as_dict():
        raise ValueError('projection differs from prospective registered allocation plan')
    from prebuild8086 import require_prebuild
    image_id=require_prebuild()
    from modal_s8086 import PREBUILT_IMAGE_ID
    if PREBUILT_IMAGE_ID!=image_id:raise ValueError('fresh dispatcher must load the prebuilt image; builds forbidden')
    from modal_score8086 import app,score,positive_controls
    from s8086 import SEEDS,HERE,committed
    import subprocess
    guard=SpendGuard(ledger,projection['job_liabilities_usd'],overhead=projection['storage_termination_reserve_usd'])
    if stage=='controls': jobs=[('controls',None,None,positive_controls)]
    else:
        kind='law' if stage=='assay' else stage
        jobs=[(f'{kind}:{a}:{s}',a,s,score) for a in registration.arms for s in SEEDS]
    commit=receipt.get('eligibility_commit','')
    if stage=='assay':
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
        if head!=commit: raise ValueError('assay data release must identify current committed eligibility ledger')
        for _,arm,seed,_ in jobs:
            from run8086 import world_name
            name=world_name(arm,seed)
            local=HERE/f'results/s8086/eligibility/{arm}/{name}.json.gz'
            if committed(local)!=receipt.get('eligibility_sha256',{}).get(name):
                raise ValueError('eligibility ledger differs from released data commit')
    for job,arm,seed,function in jobs:
        deadline=projection['execution_deadline_s'][job]
        bounded=function.with_options(timeout=deadline,max_containers=1,buffer_containers=0,retries=0)
        args=(receipt,deadline) if stage=='controls' else (arm,seed,stage,receipt,deadline,commit)
        bounded_call(app,bounded,args,guard,job,deadline)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release',type=Path,required=True); ap.add_argument('--ledger',type=Path,required=True)
    ap.add_argument('--projection',type=Path,required=True)
    ap.add_argument('--stage',choices=('worlds','score','extract','assay','controls'),default='worlds')
    a=ap.parse_args()
    if a.stage=='worlds': dispatch_worlds(a.release,a.ledger,json.loads(a.projection.read_text()))
    else: dispatch_scoring(a.stage,a.release,a.ledger,json.loads(a.projection.read_text()))

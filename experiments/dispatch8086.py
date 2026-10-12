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

def parallel_projection(original):
    """Operational timeout amendment; retain every non-world allocation."""
    from copy import deepcopy
    from lifecycle8086 import WORLD_TIMEOUT_S,dispatch_seconds,FULL_WORLD_RECOVERIES
    p=deepcopy(original)
    for name in p['execution_deadline_s']:
        if name.startswith('world:'):
            p['execution_deadline_s'][name]=WORLD_TIMEOUT_S
            p['job_liabilities_usd'][name]=dispatch_seconds(WORLD_TIMEOUT_S)*RATE
    world_cost=dispatch_seconds(WORLD_TIMEOUT_S)*RATE
    p['world_deadline_s']={arm:WORLD_TIMEOUT_S for arm in p['registration']['arms']}
    p['longest_world_container_s']=WORLD_TIMEOUT_S
    for row in p['components']:row['world_seconds']=WORLD_TIMEOUT_S
    p['component_totals_usd']={stage:sum(v for n,v in p['job_liabilities_usd'].items() if n.startswith(stage+':'))
                             for stage in ('world','score','extract','law')}
    p['component_totals_usd']['fixed']=sum(v for n,v in p['job_liabilities_usd'].items() if ':' not in n)+p['storage_termination_reserve_usd']
    p['projection_usd']=sum(p['job_liabilities_usd'].values())+p['storage_termination_reserve_usd']
    p['recovery_budget_usd']=FULL_WORLD_RECOVERIES*world_cost
    p['cap_headroom_usd']=CAP-p['projection_usd']
    p['full_world_restarts_absorbed']=int(p['cap_headroom_usd']//world_cost)
    p['launch_fits']=p['projection_usd']<=CAP
    p['lifecycle_policy'].update(one_app_per_job=False,max_containers_per_job=20,
        concurrent_worlds=20,world_timeout_s=WORLD_TIMEOUT_S,
        measured_world_minutes=63,timeout_disposition='infrastructure failure; restart step 0, same seed',
        remaining_registered_full_world_recoveries=FULL_WORLD_RECOVERIES-1,
        world_fixed_recovery_s={a:FULL_WORLD_RECOVERIES*dispatch_seconds(WORLD_TIMEOUT_S) for a in p['registration']['arms']})
    return p

def clone_accounting(jobs,billed,plan):
    # The replay state contains only scalar fields, not the event payloads.
    return {name:dict(job) for name,job in jobs.items()},dict(billed),dict(plan)

def world_timeout(exc):
    from modal.exception import FunctionTimeoutError
    return (isinstance(exc,FunctionTimeoutError) or
            isinstance(exc,TimeoutError) and str(exc)=='spend/allocation stop; incomplete world' or
            isinstance(exc,RuntimeError) and str(exc)=='world allocation exceeded; retain liability')

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
                elif row['kind']=='batch_liability':
                    base=sum(jobs[n]['reservation'] for n in row['jobs'])
                    for n in row['jobs']:
                        fraction=jobs[n]['reservation']/base
                        jobs[n].update(liability=row['usd']*fraction,elapsed_usd=row.get('elapsed_usd',0)*fraction)
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
                apps={}
                for j in jobs.values():
                    app=j.get('app_id');floor,remaining=apps.get(app,(0.,0.))
                    if j['status']=='settled':floor+=j['cost']
                    else:
                        liability=j.get('liability',j['reservation']);elapsed=j.get('elapsed_usd',0)
                        floor+=elapsed;remaining+=liability-elapsed
                    apps[app]=(floor,remaining)
                total+=sum(max(billed.get(app,0),floor)+remaining for app,(floor,remaining) in apps.items())
                total+=sum(v for app,v in billed.items() if app not in apps)
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
                experiment_stopped=True
            elif kind=='check': pass
            elif kind=='liability':
                j=jobs.get(event['job'])
                if not j or j['status']=='settled': raise ValueError('unknown/live liability job')
                elapsed=event.get('elapsed_usd',0)
                if not math.isfinite(elapsed) or not 0<=elapsed<=event['usd']:raise ValueError('invalid observed elapsed cost')
                j.update(liability=event['usd'],elapsed_usd=elapsed,app_id=event.get('app_id',j.get('app_id')))
                total=complete_projection()
            elif kind=='batch_liability':
                names=event['jobs'];elapsed=event.get('elapsed_usd',0)
                if (not names or len(set(names))!=len(names) or
                        any(n not in jobs or jobs[n]['status']=='settled' or jobs[n].get('app_id')!=event['app_id'] for n in names)
                        or not math.isfinite(elapsed) or not 0<=elapsed<=event['usd']):
                    raise ValueError('invalid concurrent liabilities')
                base=sum(jobs[n]['reservation'] for n in names)
                if event['usd']+1e-12<base:raise ValueError('concurrent reservations cannot be released before stopping')
                for n in names:
                    fraction=jobs[n]['reservation']/base
                    jobs[n].update(liability=event['usd']*fraction,elapsed_usd=elapsed*fraction)
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
            app_liabilities={}
            for j in jobs.values():
                app=j.get('app_id');floor,remaining=app_liabilities.get(app,(0.,0.))
                if j['status']=='settled':floor+=j['cost']
                else:
                    elapsed=j.get('elapsed_usd',0);floor+=elapsed
                    remaining+=j.get('liability',j['reservation'])-elapsed
                app_liabilities[app]=(floor,remaining)
            return {'projected_liabilities_usd':total,'cap':self.cap,'cap_exceeded':total>self.cap,
                    'app_liabilities_usd':{app:max(billed.get(app,0),floor)+remaining
                                          for app,(floor,remaining) in app_liabilities.items()},
                    'experiment_stopped':experiment_stopped,
                    'stop_required':total>self.cap or bool(unknown) or experiment_stopped,
                    'unknown_physical_liabilities':unknown}

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


def bounded_parallel(app,jobs,guard,*,watchdog_factory=None):
    """One shared app/watchdog; one input per single-use container, at most 20.

    Reserve the complete cohort before setup. Keep all nominal reservations
    until provider STOPPED with no tasks; never settle individual live results.
    """
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from lifecycle8086 import WatchdogProcess,dispatch_seconds,FULL_WORLD_RECOVERIES
    if not jobs or len(jobs)>20 or len({j[0] for j in jobs})!=len(jobs):
        raise ValueError('invalid parallel cohort')
    deadlines={name:deadline for name,_,_,deadline in jobs}
    for name,deadline in deadlines.items():
        if guard.plan[name]+1e-12<dispatch_seconds(deadline)*RATE:
            raise ValueError('lifecycle envelope underreserved')
    for name,deadline in deadlines.items():
        guard.reserve(name,guard.plan[name],execution_s=deadline,concurrent_jobs=list(deadlines))
    monitor=(watchdog_factory or WatchdogProcess)(guard,jobs[0][0],None,max(deadlines.values()),
                                                 jobs=deadlines,max_containers=len(jobs))
    calls={};current_calls={};results={};failure=None;recoveries=0
    reservations={n:guard.plan[n] for n in deadlines}
    try:
        with app.run(name=monitor.app_name):
            for name in deadlines:guard.transact({'kind':'lifecycle','job':name,'app_id':app.app_id})
            monitor.bind_app(app.app_id)
            try:
                for name,function,args,_ in jobs:
                    monitor.permit()
                    call=function.spawn(*args);calls[name]=call;current_calls[name]=call
                    monitor.bind_call(name,call.object_id)
                    guard.start(name,call.object_id,app.app_id)
                # RPC retrieval is concurrent too; all submissions precede any
                # result wait. The independent stop path owns no host locks.
                executor=ThreadPoolExecutor(max_workers=len(jobs))
                futures={executor.submit(call.get):name for name,call in calls.items()}
                try:
                    while futures:
                        future=next(as_completed(futures));name=futures.pop(future)
                        try:results[name]=future.result()
                        except Exception as exc:
                            # This classification precedes any attempt-file
                            # inspection. A fresh single-use input uses exactly
                            # the original arguments/seed and starts at step 0.
                            if not world_timeout(exc) or not name.startswith('world:') or recoveries>=FULL_WORLD_RECOVERIES-1:raise
                            recoveries+=1
                            retry=f'{name}:timeout-recovery:{recoveries}'
                            reservation=reservations[name];deadline=deadlines[name]
                            guard.transact({'kind':'plan_extension','jobs':{retry:reservation},
                                'reason':'registered infrastructure timeout recovery; same seed, step 0',
                                'failed_call_id':current_calls[name].object_id,'attempt_files_inspected':False})
                            guard.reserve(retry,reservation,execution_s=deadline,recovery_of=name)
                            guard.transact({'kind':'lifecycle','job':retry,'app_id':app.app_id})
                            deadlines[retry]=deadline;reservations[retry]=reservation
                            monitor.add_recovery(retry,deadline,reservation);monitor.permit()
                            _,function,args,_=next(j for j in jobs if j[0]==name)
                            call=function.spawn(*args);calls[retry]=call;current_calls[name]=call
                            monitor.bind_call(retry,call.object_id);guard.start(retry,call.object_id,app.app_id)
                            futures[executor.submit(call.get)]=name
                finally:executor.shutdown(wait=False,cancel_futures=True)
            except BaseException as exc:
                failure=exc
                guard.transact({'kind':'terminal','job':jobs[0][0],'app_id':app.app_id,
                                'status':'INCOMPLETE','error':type(exc).__name__,'reason':str(exc),
                                'route_verdict':None,'law_verdict':None,'concurrent_jobs':list(deadlines)})
            finally:
                monitor.finish()
                try:monitor.wait_confirmed()
                except BaseException as exc:
                    if failure is None:failure=exc
            if failure is not None:raise failure
    except BaseException as exc:
        monitor.finish()
        guard.transact({'kind':'terminal','job':jobs[0][0],'status':'INCOMPLETE',
                        'error':type(exc).__name__,'reason':str(exc),'liability_retained':True,
                        'route_verdict':None,'law_verdict':None})
        raise
    base=sum(reservations.values())
    charged=max(base,monitor.confirmation().get('charged_usd',base))
    for name in deadlines:
        guard.settle(name,charged*(reservations[name]/base),app_id=app.app_id,call_id=calls[name].object_id,
                     stopping_confirmed=True,accounting='cohort liability share after app stopped and empty')
    return results


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
    jobs=[];pools={}
    for arm in registration.arms:
        for seed in SEEDS:
            job=f'world:{arm}:{seed}';deadline=projection['execution_deadline_s'][job]
            if deadline not in pools:
                pools[deadline]=world.with_options(timeout=deadline,max_containers=20,buffer_containers=0,retries=0)
            function=pools[deadline]
            jobs.append((job,function,(arm,seed,commit,receipt,deadline),deadline))
    bounded_parallel(app,jobs,guard)


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
    calls=[];pools={}
    for job,arm,seed,function in jobs:
        deadline=projection['execution_deadline_s'][job]
        key=(function,deadline)
        if key not in pools:
            pools[key]=function.with_options(timeout=deadline,max_containers=min(20,len(jobs)),buffer_containers=0,retries=0)
        bounded=pools[key]
        args=(receipt,deadline) if stage=='controls' else (arm,seed,stage,receipt,deadline,commit)
        calls.append((job,bounded,args,deadline))
    if len(calls)==1:bounded_call(app,*calls[0][1:3],guard,calls[0][0],calls[0][3])
    else:bounded_parallel(app,calls,guard)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release',type=Path,required=True); ap.add_argument('--ledger',type=Path,required=True)
    ap.add_argument('--projection',type=Path,required=True)
    ap.add_argument('--stage',choices=('worlds','score','extract','assay','controls'),default='worlds')
    a=ap.parse_args()
    if a.stage=='worlds': dispatch_worlds(a.release,a.ledger,json.loads(a.projection.read_text()))
    else: dispatch_scoring(a.stage,a.release,a.ledger,json.loads(a.projection.read_text()))

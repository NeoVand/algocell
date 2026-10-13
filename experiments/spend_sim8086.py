"""Randomized system simulation using production SpendGuard and Lifecycle.

No RPCs or GPU. A conforming single-container provider supplies cursor events,
setup/termination delays, preemptions and asynchronous result acknowledgement.
Actual resource seconds are accumulated separately from the guard's charges.
The provider honouring stop/server limits remains the registered residual risk.
"""
from collections import Counter
import json
import random
import io
import tempfile
from pathlib import Path
from unittest.mock import patch
from dispatch8086 import SpendGuard
from lifecycle8086 import Lifecycle, ATTEMPT_TERMINATION_S, STARTUP_TIMEOUT_S
from s8086 import PRE, RATE, CAP, read_json, sha
from meter8086 import AppMeterPoller

class Journal:
    def __init__(self,path):self.rows=[]
    def append(self,row):self.rows.append(row)
    def flush(self,*args):return True

class MemoryFile(io.StringIO):
    def fileno(self):return 0

class Ledger:
    """Same append-only serialization/replay; disk latency is tested separately."""
    def __init__(self):self.file=MemoryFile()
    def open(self,mode):return self
    def __enter__(self):return self.file
    def __exit__(self,*args):return False
    def read_text(self):return self.file.getvalue()
    def unlink(self):pass

class Provider:
    def __init__(self):self.events=[];self.tasks=[];self.stopped=False;self.meter=0.;self.stops=0
    def sample(self,app):return {'tasks':self.tasks,'attempt_events':self.events,'stopped':self.stopped,'metered_usd':self.meter}
    def stop(self,app):self.stops+=1
    def cancel(self,call):pass


def simulate(sequences=1000,seed=0x95958086,*,output=True):
    projection=read_json(PRE/'projection.json');plan=projection['job_liabilities_usd'];overhead=projection['storage_termination_reserve_usd']
    prng=random.Random(seed);violations=[];outcomes=Counter();max_charge=max_actual=0.;recovered=0;restart_sequences=0
    completion_before_finish=finish_before_completion=0
    with tempfile.TemporaryDirectory() as temp, patch('lifecycle8086.AsyncJournal',Journal), patch('dispatch8086.fcntl.flock'), patch('dispatch8086.os.fsync'), patch('lifecycle8086.Lifecycle.bookkeep',lambda self,event:self.guard.transact(event,nonblocking=True)):
      for sequence in range(sequences):
        cap=CAP if sequence%4 else projection['projection_usd']+1e-7
        guard=SpendGuard(Path(temp)/f'ledger-{sequence}',plan,cap=cap,overhead=overhead)
        guard.path=Ledger()
        # Real complete future plan; reserve preparation/fixed jobs too. Body
        # deadlines remain fixed. Failures select seeds/attempt timing only.
        restart_count=0 if sequence%5==0 else prng.randint(0,6)
        restart_sequences+=bool(restart_count)
        failures=Counter(prng.choices([j for j in plan if j.startswith('world:')],k=restart_count))
        settled=0.;actual=overhead+plan['preparation'];failed_seconds=0.;completed=0;stopped=False;expected_cost=projection['projection_usd']
        for job,deadline in projection['execution_deadline_s'].items():
            if stopped:break
            app=f'app-{sequence}-{job}';guard.reserve(job,plan[job]);guard.start(job,'call',app)
            clock=[0.];state=Lifecycle(Path(temp)/'events',app_id=app,call_id='call',execution_s=deadline,guard=guard,job=job,started=0,now=lambda:clock[0],setup=True)
            setup=prng.uniform(0,120);clock[0]=setup;state.body_started=setup
            provider=Provider();current_actual=setup;elapsed_failures=0.;attempt=0
            def check_charge():
                nonlocal max_charge,max_actual
                charged=overhead+plan['preparation']+settled+state.charged_usd
                observed=actual+current_actual*RATE
                max_charge=max(max_charge,charged);max_actual=max(max_actual,observed)
                if charged>cap+1e-8:violations.append((sequence,'b charged',charged,cap))
                if observed>cap+1e-8 or observed>charged+1e-8:violations.append((sequence,'b actual',observed,charged,cap))
            def poll(finish=False):
                state.poll(provider,finish=finish);check_charge()
            # Observe first startup before the first input. Its server liability
            # includes startup and body; there is no fictitious extra attempt.
            provider.tasks=['task-0'];provider.events=[{'attempt':'attempt-0','task':'task-0','state':'live','elapsed_s':0}]
            poll()
            for failure in range(failures[job]):
                if state.incomplete:break
                # Including near-full failures puts the original app deadline
                # under pressure and exercises the finite recovery extension.
                duration=(deadline+STARTUP_TIMEOUT_S)*prng.uniform(.02,1.)
                clock[0]+=duration;current_actual+=duration;failed_seconds+=duration+ATTEMPT_TERMINATION_S
                elapsed_failures+=duration+ATTEMPT_TERMINATION_S
                provider.tasks=[];provider.events=[{'attempt':f'attempt-{attempt}','task':f'task-{attempt}','state':'preempted','elapsed_s':duration}]
                provider.meter=current_actual*RATE if prng.random()<.3 else provider.meter
                poll()
                if state.incomplete:break
                attempt+=1;recovered+=1
                provider.tasks=[f'task-{attempt}'];provider.events=[{'attempt':f'attempt-{attempt}','task':f'task-{attempt}','state':'live','elapsed_s':0}]
                poll()
            if not state.incomplete:
                # Successful resource lifetime <= startup+body forecast.
                duration=deadline*prng.uniform(.75,1.)
                clock[0]+=duration;current_actual+=duration
                provider.tasks=[];provider.events=[{'attempt':f'attempt-{attempt}','task':f'task-{attempt}','state':'completed','elapsed_s':duration}]
                provider.meter=current_actual*RATE if prng.random()<.5 else provider.meter
                before=provider.stops
                finish_first=bool(prng.getrandbits(1))
                finish_before_completion+=finish_first;completion_before_finish+=not finish_first
                poll(finish=finish_first)
                # a: under the complete conservative budget, asynchronous
                # successful completion cannot manufacture a cap stop.
                reserved_jobs=guard_jobs(guard.path)
                allowed=settled+overhead+max(plan[job],(setup+elapsed_failures+duration+ATTEMPT_TERMINATION_S)*RATE)+sum(v for k,v in plan.items() if k not in reserved_jobs)<=cap+1e-8
                if allowed and state.incomplete:violations.append((sequence,'a success',job,state.stop_reason))
                expected_cost+=max(0.,(setup+elapsed_failures+duration+ATTEMPT_TERMINATION_S)*RATE-plan[job])
            if state.incomplete:
                stopped=True;outcomes['INCOMPLETE']+=1
                provider.tasks=[];provider.events=[];provider.stopped=True
                poll()
                guard.transact({'kind':'terminal','job':job,'status':'INCOMPLETE','route_verdict':None,'law_verdict':None})
                result=guard.transact({'kind':'check'})
                if not result['stop_required']:violations.append((sequence,'d cap stop not terminal'))
                if restart_count==0:violations.append((sequence,'a nominal stopped',state.stop_reason))
                # c is conservative admission: an entire remaining full run
                # must fit, including all ended/unknown attempt liabilities.
                if projection['projection_usd']+failed_seconds*RATE<=cap+1e-8:violations.append((sequence,'c sufficient recovery budget',job))
            else:
                delay=prng.uniform(0,30);clock[0]+=delay
                # Count the stop delay separately; terminated attempt allowances
                # exceed it, so actual never outruns charged resource seconds.
                current_actual+=delay;provider.stopped=True;provider.events=[];provider.tasks=[];poll(finish=True)
                if not state.confirmed or state.incomplete:violations.append((sequence,'c completion not confirmed',job))
                charged=max(plan[job],state.charged_usd);guard.settle(job,charged)
                settled+=charged;actual+=current_actual*RATE
                if job.startswith('world:'):completed+=1
        if not stopped:
            outcomes['COMPLETE']+=1
            if completed!=20:violations.append((sequence,'c not all worlds',completed))
            if guard.transact({'kind':'check'})['stop_required']:violations.append((sequence,'a final false stop'))
        guard.path.unlink()
    result={'date':'2026-10-12','sequences':sequences,'seed':seed,'violations':len(violations),'violation_details':violations[:30],
        'invariants':{'a':'successful completions within complete forecast do not trigger a stop','b':'charged current/settled and independently accumulated actual spend <= sequence cap <= $95','c':'all 20 worlds complete when conservative full-run plus recovery budget fits','d':'cap stop is terminal INCOMPLETE with null route/law verdicts, never a negative'},
        'outcomes':dict(outcomes),'restart_sequences':restart_sequences,'recovered_attempts':recovered,'completion_before_finish':completion_before_finish,'finish_before_completion':finish_before_completion,'max_charged_usd':max_charge,'max_actual_usd':max_actual,
        'provider_scope':__doc__,'projection_sha256':sha((PRE/'projection.json').read_bytes()),'simulation_sha256':sha(Path(__file__).read_bytes()),'no_paid_compute':True,'no_population':True}
    if output:(PRE/'spend_simulation.json').write_text(json.dumps(result,indent=2)+'\n')
    if violations:raise AssertionError(json.dumps(result))
    return result


def guard_jobs(path):
    # Separate read-only test oracle, independent from SpendGuard's replay cache.
    return {r['job'] for r in map(json.loads,path.read_text().splitlines()) if r['kind']=='reserve'}

def simulate_parallel(sequences=1000,seed=0x20268086,*,output=False,carried_usd=0.):
    """20 interleaved worlds; independent birth/end/cleanup resource oracle.

    Provider timeouts terminate an input and admit a separately reserved,
    identical-seed step-0 input. Transient stop/cancel RPC failures are retried
    within the unchanged termination allowance. No paid compute or result data.
    """
    projection=read_json(PRE/'projection.json');plan=projection['job_liabilities_usd']
    jobs={n:d for n,d in projection['execution_deadline_s'].items() if n.startswith('world:')}
    deadline=max(jobs.values());overhead=projection['storage_termination_reserve_usd']+carried_usd
    nominal=projection['projection_usd']+carried_usd
    rng=random.Random(seed);violations=[];outcomes=Counter();restarts=0;max_charge=max_actual=0.;interleavings=0;queue_limits=Counter()
    faults=Counter();timeout_recoveries=0
    class FaultGuard(SpendGuard):
        locked=False
        def transact(self,event,*,nonblocking=False):
            if nonblocking and self.locked:
                faults['ledger_lock_contentions']+=1
                raise BlockingIOError('injected concurrent ledger lock owner')
            return super().transact(event,nonblocking=nonblocking)
    class FaultProvider(Provider):
        def __init__(self):super().__init__();self.stop_failures=0;self.cancel_failures=0;self.poller=None
        def sample(self,app):
            sample=super().sample(app)
            if self.poller:sample.update(self.poller.refresh())
            return sample
        def stop(self,app):
            super().stop(app)
            if self.stop_failures:
                self.stop_failures-=1;faults['failed_stop_rpcs']+=1
                raise TimeoutError('injected stop RPC timeout')
        def cancel(self,call):
            if self.cancel_failures:
                self.cancel_failures-=1;faults['failed_cancel_rpcs']+=1
                raise RuntimeError('injected cancel RPC failure')
    with tempfile.TemporaryDirectory() as temp,patch('lifecycle8086.AsyncJournal',Journal),patch('dispatch8086.fcntl.flock'),patch('dispatch8086.os.fsync'),patch('lifecycle8086.LOCK_RETRY_S',0):
      for sequence in range(sequences):
        cap=nominal+1e-7 if sequence%4==0 else CAP
        guard=FaultGuard(Path(temp)/'parallel',plan,cap=cap,overhead=overhead);guard.path=Ledger()
        for name in jobs:guard.reserve(name,plan[name]);guard.start(name,'call-'+name,'shared-app')
        clock=[0.];state=Lifecycle(Path(temp)/'events',app_id='shared-app',job=next(iter(jobs)),execution_s=deadline,
                                  guard=guard,jobs=jobs,max_containers=20,started=0,now=lambda:clock[0],setup=True,
                                  initial_account=guard.transact({'kind':'check'}))
        setup=rng.uniform(0,120);state.body_started=setup;clock[0]=setup
        state.call_ids=['call-'+n for n in jobs]
        provider=FaultProvider();schedule=[];intervals={};termination_delays={};active={};finished=set()
        provider.stop_failures=rng.randint(0,2);provider.cancel_failures=rng.randint(0,2)
        failure_count=0 if sequence%5==0 else rng.randint(1,6)
        failures=Counter(rng.choices(list(jobs),k=failure_count))
        timeouts=Counter(rng.choices(list(jobs),k=0 if sequence%5==0 else rng.randint(0,2)))
        slots=rng.choice((1,2,4,8,20));queue_limits[slots]+=1;available=[setup]*slots
        failure_seconds=0.
        for name in jobs:
            slot=min(range(slots),key=lambda i:available[i]);born=available[slot]
            kinds=['preempted']*failures[name]+['timeout']*timeouts[name]
            rng.shuffle(kinds);kinds.append('completed')
            for attempt,terminal in enumerate(kinds):
                key=name+'-'+str(attempt);startup=rng.uniform(0,120)
                if startup>=90:faults['slow_startups']+=1
                duration=(deadline+startup if terminal=='timeout' else
                          (deadline+startup)*rng.uniform(.02,1.) if terminal=='preempted' else
                          startup+deadline*rng.uniform(.75,1.))
                end=born+duration;intervals[key]=(born,end)
                termination_delays[key]=rng.uniform(0,30)
                schedule.append((born,'live',key,name,0.))
                schedule.append((end,terminal,key,name,duration))
                if terminal=='preempted':failure_seconds+=duration+ATTEMPT_TERMINATION_S
                if terminal!='completed':born=end+termination_delays[key]+rng.uniform(0,30)
            available[slot]=end+termination_delays[key]
        schedule.sort(key=lambda e:(e[0],e[1]!='live',rng.random()))
        expected=nominal+failure_seconds*RATE+sum(plan[n]*count for n,count in timeouts.items())
        stopped=False;last_job=None;sequence_timeouts=0
        def actual_seconds(timestamp):
            return setup+sum(max(0,min(timestamp,end)-born)+max(0,min(timestamp-end,termination_delays[k]))
                             for k,(born,end) in intervals.items())
        billing_rng=random.Random(seed+sequence)
        def read_billing():
            # Whole sequences without any valid meter, intermittent throttling,
            # other failed/unknown reads, billing lag, then recovery. The same
            # production app poller serves every one of the concurrent jobs.
            fault=billing_rng.random()
            if sequence%5==0 or fault<.4:
                faults['rate_limited_billing_reads']+=1
                raise RuntimeError('Rate limit exceeded for workspace billing report requests')
            if fault<.5:
                faults['failed_billing_reads']+=1;raise TimeoutError('billing RPC timeout')
            if fault<.55:
                faults['unknown_billing_reads']+=1;return {'metered_usd':None}
            faults['successful_shared_app_billing_reads']+=1
            if provider.poller.failures:faults['billing_recoveries']+=1
            if 'Rate limit exceeded' in (provider.poller.error or ''):faults['rate_limit_recoveries']+=1
            return {'metered_usd':provider.meter}
        provider.poller=AppMeterPoller('shared-app',0,Path(temp)/f'meter-{sequence}',
                                     read=read_billing,now=lambda:clock[0],wall=lambda:clock[0])
        def check():
            nonlocal max_charge,max_actual
            charged=overhead+plan['preparation']+state.charged_usd
            actual=overhead+plan['preparation']+actual_seconds(clock[0])*RATE
            max_charge=max(max_charge,charged);max_actual=max(max_actual,actual)
            if charged>cap+1e-8 or actual>charged+1e-8 or actual>cap+1e-8:
                violations.append((sequence,'b',charged,actual,cap))
        for timestamp,kind,key,name,duration in schedule:
            clock[0]=timestamp
            if kind=='live':active[key]=name
            else:
                active.pop(key,None)
                if kind=='completed':finished.add(name)
            if last_job and last_job!=name:interleavings+=1
            last_job=name
            provider.tasks=list(active)
            provider.events=[{'attempt':key,'task':key,'state':'completed' if kind=='timeout' else kind,'elapsed_s':duration}]
            provider.events.extend({'attempt':k,'task':k,'state':'live','elapsed_s':timestamp-intervals[k][0]} for k in active if k!=key)
            if rng.random()<.3:provider.meter=max(provider.meter,actual_seconds(timestamp)*RATE)
            guard.locked=billing_rng.random()<.25
            state.poll(provider);check()
            if guard.locked and state.stop_reason=='observation/accounting unavailable':
                violations.append((sequence,'a lock contention became terminal'))
            if kind=='timeout' and not state.incomplete:
                faults['world_timeouts']+=1;sequence_timeouts+=1
                retry=f'{name}:timeout-recovery:{sequence_timeouts}'
                try:
                    guard.transact({'kind':'plan_extension','jobs':{retry:plan[name]},'reason':'timeout infrastructure recovery, step 0, same seed'})
                    guard.reserve(retry,plan[name]);guard.start(retry,'call-'+retry,'shared-app')
                    state.jobs[retry]=deadline;state.reservations[retry]=plan[name]
                    state.call_ids.append('call-'+retry);timeout_recoveries+=1
                except RuntimeError:state.stop_now(provider,'timeout recovery admission exceeds complete cap')
            if state.incomplete:
                if expected<=cap+1e-8:violations.append((sequence,'a/c sufficient complete recovery budget stopped',expected,cap,state.stop_reason))
                stopped=True
                # Failed control RPCs neither erase reservations nor prevent a
                # later bounded successful stop. Count cleanup independently.
                for _ in range(3):state.poll(provider)
                provider.stopped=True;provider.tasks=[];provider.events=[]
                guard.locked=False
                clock[0]+=30;state.poll(provider);check()
                guard.transact({'kind':'terminal','job':next(iter(jobs)),'status':'INCOMPLETE','route_verdict':None,'law_verdict':None})
                if not guard.transact({'kind':'check'})['stop_required']:violations.append((sequence,'d nonterminal'))
                try:guard.reserve('controls',plan['controls'])
                except RuntimeError:pass
                else:violations.append((sequence,'d further dispatch allowed'))
                restarts+=sum(a['state']=='preempted' for a in state.attempts.values())
                break
        if stopped:outcomes['INCOMPLETE']+=1
        else:
            guard.locked=False
            provider.tasks=[];provider.events=[]
            # Exercise failed normal-finish RPCs before eventual confirmation.
            for _ in range(3):state.poll(provider,finish=True)
            provider.stopped=True
            clock[0]=max(clock[0]+1,max(end+termination_delays[k] for k,(_,end) in intervals.items()))
            state.poll(provider,finish=True);check()
            if not state.confirmed or state.incomplete or len(finished)!=20:violations.append((sequence,'c incomplete cohort'))
            base=sum(state.reservations.values())
            for name in state.jobs:guard.settle(name,state.charged_usd*(state.reservations[name]/base))
            if guard.transact({'kind':'check'})['stop_required']:violations.append((sequence,'a completion stop'))
            outcomes['COMPLETE']+=1;restarts+=sum(failures.values())
        provider.poller.close()
    result={'date':'2026-10-12','sequences':sequences,'seed':seed,'concurrent_worlds':20,
            'violations':len(violations),'violation_details':violations[:30],
            'invariants':{'a':'successful completions within conservative complete budget never cause a stop',
                          'b':'simultaneous charged current/settled and independently accumulated actual spend stay below cap <= $95',
                          'c':'all 20 worlds complete whenever the conservative full-run plus registered recovery budget fits',
                          'd':'a cap stop is terminal INCOMPLETE with null route/law verdicts and forbids further dispatch'},
            'outcomes':dict(outcomes),'restarts':restarts,'timeout_step0_same_seed_recoveries':timeout_recoveries,
            'faults':dict(faults),'shared_app_jobs':20,'interleaved_job_transitions':interleavings,'available_gpu_limits':dict(queue_limits),
            'carried_usd':carried_usd,'max_charged_usd':max_charge,'max_actual_usd':max_actual,
            'world_timeout_s':deadline,'projection_sha256':sha((PRE/'projection.json').read_bytes()),
            'simulation_sha256':sha(Path(__file__).read_bytes()),'no_paid_compute':True,'no_population':True}
    if output:Path(output).write_text(json.dumps(result,indent=2)+'\n')
    if violations:raise AssertionError(json.dumps(result))
    return result

if __name__=='__main__':
    result=simulate();print(f"{result['sequences']} randomized sequences; {result['violations']} violations; {result['outcomes']}")

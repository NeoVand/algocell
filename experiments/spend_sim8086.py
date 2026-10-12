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

if __name__=='__main__':
    result=simulate();print(f"{result['sequences']} randomized sequences; {result['violations']} violations; {result['outcomes']}")

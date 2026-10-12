"""Independent host watchdog. No provider contact occurs on import.

One app and one container pool per job. All attempts share a single monotonic
clock; a restart never renews the resource envelope. Reservation is released
only after provider STOPPED plus empty TaskList, never merely call.get().
"""
from __future__ import annotations
import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from s8086 import RATE,sha,safe

STARTUP_TIMEOUT_S=120
STARTUP_ATTEMPTS=2
TERMINATION_GRACE_S=60
WATCHDOG_POLL_S=1
RPC_TIMEOUT_S=10
# A whole startup/restart envelope, termination, one polling interval and one
# bounded stop RPC, in addition to the frozen conservative execution allocation.
LIFECYCLE_EXTRA_S=STARTUP_TIMEOUT_S*STARTUP_ATTEMPTS+TERMINATION_GRACE_S+WATCHDOG_POLL_S+6*RPC_TIMEOUT_S

def envelope_seconds(execution_s):
    if not math.isfinite(execution_s) or execution_s<=0: raise ValueError('invalid execution envelope')
    return math.ceil(execution_s)+LIFECYCLE_EXTRA_S


def atomic_json(path,value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(safe(value),f); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


class Lifecycle:
    def __init__(self,path,*,app_id,execution_s,call_id=None,started=None,now=time.monotonic,guard=None,job=None):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.app_id,self.call_id,self.now,self.guard,self.job=app_id,call_id,now,guard,job
        self.started=now() if started is None else started
        self.reservation_seconds=envelope_seconds(execution_s)
        self.expires_at=self.started+self.reservation_seconds
        self.stop_at=self.expires_at-TERMINATION_GRACE_S-WATCHDOG_POLL_S-6*RPC_TIMEOUT_S
        self.confirmed=False; self.confirmed_at=None; self.requested=False
        self.tasks=set(); self.last_meter=0.; self.last_progress=None; self.unknown_bound=False
        self.emit('armed',expires_at=self.expires_at,stop_at=self.stop_at,reservation_seconds=self.reservation_seconds)

    @property
    def liability_retained(self): return not self.confirmed

    def emit(self,kind,**fields):
        with self.path.open('a') as f:
            f.write(json.dumps(safe({'kind':kind,'monotonic':self.now(),'app_id':self.app_id,
                                    'call_id':self.call_id,**fields}))+'\n'); f.flush(); os.fsync(f.fileno())

    def uncertainty(self,reason):
        if not self.unknown_bound:
            self.unknown_bound=True
            if self.guard: self.guard.transact({'kind':'uncertain','job':self.job,'reason':reason})

    def poll(self,provider,*,finish=False,parent_alive=True):
        stopping=finish or not parent_alive or self.now()>=self.stop_at
        try:
            sample=provider.sample(self.app_id)
            active=sample['tasks']
            for task in set(active)-self.tasks:
                self.tasks.add(task); self.emit('attempt_observed',task=task,attempts_observed=len(self.tasks))
            # Cap observed restarts too. Even unobserved short attempts consume
            # the same physical wall envelope; there is no per-attempt renewal.
            stopping |= len(self.tasks)>STARTUP_ATTEMPTS or len(active)>1
            if len(active)>1: self.uncertainty('concurrent containers exceed physical rate bound')
            self.last_progress=sample.get('progress',self.last_progress)
            meter=sample.get('metered_usd')
            if meter is not None and meter>=self.last_meter:
                self.last_meter=meter
                if self.guard:
                    account=self.guard.transact({'kind':'metered','app_id':self.app_id,'usd':meter})
                    stopping |= account['stop_required']
            if sample['stopped'] and not active:
                self.confirmed=True; self.confirmed_at=self.now()
                if self.confirmed_at>self.expires_at: self.uncertainty('termination confirmed after reserved resource envelope')
                self.emit('stopped_confirmed',attempts_observed=len(self.tasks),last_progress=self.last_progress,
                          within_envelope=self.confirmed_at<=self.expires_at)
                if self.guard:
                    self.guard.transact({'kind':'stopped','job':self.job,'app_id':self.app_id,
                                         'confirmed':True,'tasks_empty':True,'within_envelope':self.confirmed_at<=self.expires_at})
                return True
        except Exception as exc:
            self.emit('observation_error',error=f'{type(exc).__name__}: {exc}')
            # Failure to monitor cannot grant extra resource time.
            stopping=True
        if stopping:
            self.requested=True
            self.emit('stop_requested',finish=finish,parent_alive=parent_alive,last_progress=self.last_progress)
            if self.call_id:
                try: provider.cancel(self.call_id)
                except Exception as exc: self.emit('cancel_error',error=f'{type(exc).__name__}: {exc}')
            try: provider.stop(self.app_id)
            except Exception as exc: self.emit('stop_error',error=f'{type(exc).__name__}: {exc}')
        if self.now()>self.expires_at and not self.confirmed:
            self.uncertainty('termination not confirmed by reserved envelope expiry')
            self.emit('unconfirmed_overrun',liability_retained=True,last_progress=self.last_progress)
        return False


class ModalProvider:
    """Pinned 1.6.1 protocol; bounded RPCs, independent of dispatcher client."""
    def __init__(self):
        from modal.client import _Client
        self.loop=asyncio.new_event_loop()
        self.client=self.loop.run_until_complete(asyncio.wait_for(_Client.from_env(),RPC_TIMEOUT_S))
        self.last_bill=0.; self.last_bill_at=-math.inf

    def request(self,name,message):
        return self.loop.run_until_complete(asyncio.wait_for(getattr(self.client._stub,name)(message),RPC_TIMEOUT_S))

    def sample(self,app):
        from modal_proto import api_pb2 as P
        lifecycle=self.request('AppGetLifecycle',P.AppGetLifecycleRequest(app_id=app))
        tasks=self.request('TaskList',P.TaskListRequest(app_id=app))
        return {'stopped':lifecycle.lifecycle.app_state==P.APP_STATE_STOPPED,
                'tasks':[t.task_id for t in tasks.tasks],
                'progress':{'provider_tasks':[{'task_id':t.task_id,'started_at':t.started_at,'enqueued_at':t.enqueued_at} for t in tasks.tasks]}}

    def cancel(self,call):
        from modal_proto import api_pb2 as P
        self.request('FunctionCallCancel',P.FunctionCallCancelRequest(function_call_id=call,terminate_containers=True))

    def stop(self,app):
        from modal_proto import api_pb2 as P
        self.request('AppStop',P.AppStopRequest(app_id=app,source=P.APP_STOP_SOURCE_CLI))

    def metering(self,app,started_epoch):
        from meter8086 import billing
        result=self.loop.run_until_complete(billing(app,started_epoch))
        return result['metered_usd']


def watch(config_path):
    from dispatch8086 import SpendGuard
    config=json.loads(Path(config_path).read_text())
    guard=SpendGuard(config['ledger'],config['plan'],cap=config['cap'],overhead=config['overhead'])
    provider=ModalProvider()  # Authentication/monitor setup must succeed before spawn.
    state=Lifecycle(config['events'],app_id=config['app_id'],execution_s=config['execution_s'],
                    started=config['started_monotonic'],guard=guard,job=config['job'])
    atomic_json(config['status'],{'ready':True,'pid':os.getpid(),'heartbeat':time.monotonic()})
    last_meter=0.
    while not state.confirmed:
        control=json.loads(Path(config_path).read_text()); state.call_id=control.get('call_id')
        try: os.kill(config['parent_pid'],0); alive=True
        except ProcessLookupError: alive=False
        except PermissionError: alive=True
        # Meter query is independent of blocked call.get and cumulative billing
        # lag cannot extend the physical deadline. Skip near stop deadline.
        if time.monotonic()-last_meter>=60 and time.monotonic()+60<state.stop_at:
            try:
                meter=provider.metering(config['app_id'],config['started_epoch'])
                if guard.transact({'kind':'metered','app_id':config['app_id'],'usd':meter})['stop_required']:
                    control['finish']=True
            except Exception as exc: state.emit('meter_error',error=f'{type(exc).__name__}: {exc}')
            last_meter=time.monotonic()
        state.poll(provider,finish=control.get('finish',False),parent_alive=alive)
        atomic_json(config['status'],{'ready':True,'pid':os.getpid(),'heartbeat':time.monotonic(),
                                    'confirmed':state.confirmed,'confirmed_at':state.confirmed_at,
                                    'within_envelope':state.confirmed and state.confirmed_at<=state.expires_at})
        if not state.confirmed: time.sleep(WATCHDOG_POLL_S)


class WatchdogProcess:
    """A new session survives a blocked/crashed dispatcher. Mockable controller."""
    def __init__(self,guard,job,app_id,execution_s):
        prefix=guard.path.parent/f'lifecycle-{sha(job.encode())[:16]}'
        self.path=Path(str(prefix)+'.control.json'); self.status=Path(str(prefix)+'.status.json')
        self.events=Path(str(prefix)+'.events.jsonl')
        if self.path.exists(): raise FileExistsError('lifecycle already reserved; no restart')
        self.config={'ledger':str(guard.path.resolve()),'plan':guard.plan,'cap':guard.cap,'overhead':guard.overhead,
                     'job':job,'app_id':app_id,'execution_s':execution_s,'events':str(self.events.resolve()),
                     'status':str(self.status.resolve()),'parent_pid':os.getpid(),
                     'started_monotonic':time.monotonic(),'started_epoch':time.time()}
        atomic_json(self.path,self.config)
        with Path(str(prefix)+'.watchdog.log').open('x') as log:
            self.process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--watch',str(self.path.resolve())],
                                          stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        limit=time.monotonic()+RPC_TIMEOUT_S+5
        while time.monotonic()<limit:
            if self.process.poll() is not None: raise RuntimeError('watchdog failed before spawn')
            if self.status.exists() and json.loads(self.status.read_text()).get('ready'): return
            time.sleep(.05)
        self.finish(); raise TimeoutError('watchdog not ready; spawn forbidden')

    def bind(self,call_id):
        self.config['call_id']=call_id; atomic_json(self.path,self.config)

    def finish(self):
        self.config['finish']=True; atomic_json(self.path,self.config)

    def confirmation(self):
        return json.loads(self.status.read_text()) if self.status.exists() else {}

    def wait_confirmed(self):
        limit=time.monotonic()+TERMINATION_GRACE_S
        while time.monotonic()<limit:
            status=self.confirmation()
            if status.get('confirmed'):
                if not status.get('within_envelope'): raise RuntimeError('late stop; retained liability needs reconciliation')
                return status
            time.sleep(.1)
        raise TimeoutError('stopping unconfirmed; liability retained, watchdog continues')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--watch',required=True,type=Path)
    watch(ap.parse_args().watch)

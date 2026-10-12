"""Independent, fail-closed dynamic attempt watchdog. No provider calls on import.

App/task cursor events preserve short starts/restarts/preemptions between polls.
Stop and cancellation start in separate sessions before logging or bookkeeping.
Ledger lock acquisition is nonblocking and the entire transaction is timed.
"""
from __future__ import annotations
import argparse,asyncio,json,math,os,queue,subprocess,sys,threading,time,uuid
from pathlib import Path
from s8086 import RATE,sha,safe

STARTUP_TIMEOUT_S=120
STARTUP_ATTEMPTS=2  # historical nominal forecast allowance, NOT a restart limit
SETUP_TIMEOUT_S=120
PREBUILD_TIMEOUT_S=600
TERMINATION_GRACE_S=60
WATCHDOG_POLL_S=1
RPC_TIMEOUT_S=10
BOOKKEEP_TIMEOUT_S=RPC_TIMEOUT_S  # incremental replay; bounded independently of stop requests
LIFECYCLE_EXTRA_S=STARTUP_TIMEOUT_S*2+TERMINATION_GRACE_S+WATCHDOG_POLL_S+6*RPC_TIMEOUT_S
ATTEMPT_TERMINATION_S=TERMINATION_GRACE_S+WATCHDOG_POLL_S+6*RPC_TIMEOUT_S
FULL_WORLD_RECOVERIES=3
WORLD_TIMEOUT_S=9000  # measured 63-minute world; operational 2.5-hour limit

def recovery_seconds(execution_s, job=None):
    # Shared cap, charged as incurred. Each world's fixed lifetime can absorb
    # the entire cohort's three-full-world recovery allowance at that seed.
    return FULL_WORLD_RECOVERIES*dispatch_seconds(execution_s) if job and job.startswith('world:') else 0

def envelope_seconds(execution_s):
    if not math.isfinite(execution_s) or execution_s<=0: raise ValueError('invalid execution envelope')
    return math.ceil(execution_s)+LIFECYCLE_EXTRA_S

def dispatch_seconds(execution_s): return SETUP_TIMEOUT_S+envelope_seconds(execution_s)

def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f: json.dump(safe(value),f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

class AsyncJournal:
    """Only a daemon writer touches the append-only watchdog log."""
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.q=queue.Queue();self.error=None
        def writer():
            while True:
                row=self.q.get()
                try:
                    with self.path.open('a') as f: f.write(json.dumps(safe(row))+'\n');f.flush();os.fsync(f.fileno())
                except BaseException as exc: self.error=str(exc)
                finally: self.q.task_done()
        threading.Thread(target=writer,daemon=True).start()
    def append(self,row): self.q.put_nowait(row)
    def flush(self,timeout=1):
        deadline=time.monotonic()+timeout
        while self.q.unfinished_tasks and time.monotonic()<deadline: time.sleep(.001)
        return not self.q.unfinished_tasks and self.error is None

class AsyncStatus:
    def __init__(self,path):
        self.q=queue.Queue();self.path=path
        def writer():
            while True:
                row=self.q.get()
                try:atomic_json(self.path,row)
                except Exception:pass  # stale heartbeat forbids dispatch
                finally:self.q.task_done()
        threading.Thread(target=writer,daemon=True).start()
    def publish(self,row):self.q.put_nowait(row)
    def flush(self,timeout=1):
        deadline=time.monotonic()+timeout
        while self.q.unfinished_tasks and time.monotonic()<deadline:time.sleep(.001)

class Lifecycle:
    def __init__(self,path,*,app_id,execution_s,call_id=None,started=None,now=time.monotonic,guard=None,job=None,setup=False,jobs=None,max_containers=1):
        self.path=Path(path);self.journal=AsyncJournal(path)
        self.app_id,self.call_id,self.now,self.guard,self.job=app_id,call_id,now,guard,job
        self.started=now() if started is None else started
        self.execution_s=execution_s;self.setup=setup;self.body_started=None if setup else self.started
        self.jobs=dict(jobs) if jobs else None;self.max_containers=max_containers;self.call_ids=[]
        self.reservations={n:guard.plan[n] for n in jobs} if jobs else {}
        self.reservation_seconds=dispatch_seconds(execution_s) if setup else envelope_seconds(execution_s)
        self.recovery_seconds=recovery_seconds(execution_s,job)
        self.expires_at=self.started+self.reservation_seconds+self.recovery_seconds
        if jobs:
            # Queueing may serialize the cohort; server-side body limits stay
            # per container. The app clock is fixed once, never renewed.
            self.expires_at=self.started+sum(dispatch_seconds(v) for v in jobs.values())+self.recovery_seconds
        self.stop_at=self.expires_at-ATTEMPT_TERMINATION_S
        self.confirmed=False;self.confirmed_at=None;self.requested=False;self.stop_reason=None;self.incomplete=False
        self.tasks=set();self.attempts={};self.last_meter=0.;self.last_progress=None;self.unknown_bound=False
        self.charged_usd=0.;self.elapsed_usd=0.;self.account=None;self.work_complete=False
        self.emit('armed',expires_at=self.expires_at,stop_at=self.stop_at,reservation_seconds=self.reservation_seconds,setup_reserved=setup)
    @property
    def liability_retained(self): return not self.confirmed or self.unknown_bound
    def emit(self,kind,**fields):
        self.journal.append({'kind':kind,'monotonic':self.now(),'app_id':self.app_id,'call_id':self.call_id,**fields})
    def flush_log(self): return self.journal.flush()
    def bookkeep(self,event):
        if not self.guard:return {'stop_required':False}
        result=queue.Queue(maxsize=1)
        def transact():
            try: result.put((True,self.guard.transact(event,nonblocking=True)))
            except BaseException as exc: result.put((False,exc))
        threading.Thread(target=transact,daemon=True).start()
        try:ok,value=result.get(timeout=BOOKKEEP_TIMEOUT_S)
        except queue.Empty: raise TimeoutError('watchdog ledger IO exceeded bounded RPC allowance')
        if not ok: raise value
        return value
    def uncertainty(self,reason):
        if not self.unknown_bound:
            self.unknown_bound=True;self.emit('uncertain',reason=reason,liability_retained=True)
            # This must never be required to request provider stop.
            try:self.bookkeep({'kind':'uncertain','job':self.job,'reason':reason})
            except Exception as exc:self.emit('ledger_unavailable',error=str(exc))
    def stop_now(self,provider,reason):
        if reason!='normal_finish':self.incomplete=True
        self.requested=True;self.stop_reason=reason
        # Production methods only Popen independent processes. Neither awaits
        # bookkeeping, provider RPC completion nor the other control action.
        try:provider.stop(self.app_id)
        except Exception as exc:self.emit('stop_error',error=str(exc))
        for call in self.call_ids or ([self.call_id] if self.call_id else []):
            try:provider.cancel(call)
            except Exception as exc:self.emit('cancel_error',error=str(exc))
        self.emit('stop_requested',reason=reason,last_progress=self.last_progress,status='INCOMPLETE')
    def observe(self,sample):
        active=set(sample['tasks']);events=sample.get('attempt_events',[])
        # Test mocks may use TaskList alone; production always supplies cursor
        # events as well, including tasks already gone from the live list.
        events=events+[{'attempt':t,'task':t,'state':'live','elapsed_s':0} for t in active if not any(e.get('task')==t for e in events)]
        for event in events:
            key=event['attempt'];task=event.get('task',key)
            if key not in self.attempts:
                self.attempts[key]={'first':self.now(),'observed_at':self.now(),'elapsed_s':0.,'live':True,'task':task,'state':'live'}
                self.tasks.add(key);self.emit('attempt_observed',task=task,attempt=key,attempts_observed=len(self.tasks),recovery='restart_step_0')
            attempt=self.attempts[key]
            elapsed=event.get('elapsed_s',self.now()-attempt['first'])
            if not math.isfinite(elapsed) or elapsed<0:raise ValueError('invalid attempt elapsed time')
            attempt['elapsed_s']=max(attempt['elapsed_s'],elapsed)
            attempt['observed_at']=self.now()
            attempt['live']=event.get('state') not in ('completed','preempted','failed')
            attempt['state']=event.get('state','live')
            self.emit('attempt_state',**event)
            if event.get('state')=='completed': self.work_complete=True
        if sample['stopped'] and not active:
            # STOPPED plus an empty TaskList resolves even vanished attempts.
            # Their elapsed and cleanup charges remain; no server can still
            # run and no queued replacement can start in this stopped app.
            for attempt in self.attempts.values():
                if attempt['live']:
                    attempt['elapsed_s']+=max(0.,self.now()-attempt['observed_at'])
                    attempt.update(live=False,state='stopped',observed_at=self.now())
        # A vanished task is retained at its FULL server liability until an
        # explicit terminal cursor event confirms the end of that attempt.
        limit=self.execution_s+STARTUP_TIMEOUT_S
        seconds=0.;elapsed_seconds=0.
        for attempt in self.attempts.values():
            elapsed=attempt['elapsed_s']+max(0.,self.now()-attempt['observed_at']) if attempt['live'] else attempt['elapsed_s']
            elapsed_seconds+=elapsed
            seconds+=elapsed+(max(0.,limit-elapsed) if attempt['live'] else 0.)+ATTEMPT_TERMINATION_S
        if not self.jobs and not any(a['live'] for a in self.attempts.values()) and self.body_started is not None and not self.work_complete:
            seconds+=limit+ATTEMPT_TERMINATION_S  # reserve pending input before its first task
        setup_elapsed=min(SETUP_TIMEOUT_S,max(0.,(self.now() if self.body_started is None else self.body_started)-self.started)) if self.setup else 0.
        self.charged_usd=(seconds+setup_elapsed)*RATE
        self.elapsed_usd=(elapsed_seconds+setup_elapsed)*RATE
        if self.jobs:
            # Retain every nominal job reservation until the WHOLE app stops.
            # Each ended preemption additionally reserves its consumed time
            # and termination; the nominal floor reserves its step-0 retry.
            # Vanished tasks without terminal cursor evidence retain a full
            # server liability, including possible overlapping replacements.
            base=sum(self.reservations.values())
            failed=sum(a['elapsed_s']+ATTEMPT_TERMINATION_S for a in self.attempts.values() if a['state'] in ('preempted','failed'))
            completed=sum(a['state']=='completed' for a in self.attempts.values())
            pending=0 if sample['stopped'] and not active else max(0,len(self.jobs)-completed-len(active))
            # TaskList absence never confirms an end. Retain that attempt's
            # full server bound AND reserve queued/replacement inputs. This
            # also covers overlap before all 20 original jobs have started.
            dynamic=(seconds+pending*(limit+ATTEMPT_TERMINATION_S)+setup_elapsed)*RATE
            # Ended attempts still consume cleanup resource time. That time
            # already sits inside seconds and the retained nominal cohort.
            # Include its conservative allowance in the accounted spend floor
            # before overlapping cumulative app billing, rather than adding
            # the bill's cleanup excess on top of the same reservation again.
            ended=sum(not a['live'] for a in self.attempts.values())
            self.elapsed_usd+=(ended*ATTEMPT_TERMINATION_S)*RATE
            self.charged_usd=max(base+failed*RATE,dynamic,self.elapsed_usd)
        self.emit('liability_charged',usd=self.charged_usd,attempts=len(self.attempts),seconds=seconds,setup_s=setup_elapsed)
        return active
    def poll(self,provider,*,finish=False,parent_alive=True):
        self.work_complete |= finish
        due=finish or not parent_alive or self.now()>=self.stop_at or self.requested
        if self.setup and self.body_started is None and self.now()>=self.started+SETUP_TIMEOUT_S:due=True
        if due:
            reason=self.stop_reason if self.requested else ('normal_finish' if finish and parent_alive and self.now()<self.stop_at else 'parent/deadline')
            self.stop_now(provider,reason)
        try:
            sample=provider.sample(self.app_id);active=self.observe(sample)
            self.last_progress=sample.get('progress',self.last_progress)
            if len(active)>self.max_containers:
                self.stop_now(provider,'concurrent container limit exceeded');self.uncertainty('concurrent containers exceed rate envelope')
            if self.guard:
                event={'kind':'batch_liability' if self.jobs else 'liability','job':self.job,'app_id':self.app_id,'usd':self.charged_usd,'elapsed_usd':self.elapsed_usd,'attempts':safe(self.attempts)}
                if self.jobs:event['jobs']=list(self.jobs)
                self.account=self.bookkeep(event)
                if self.account['stop_required']:self.stop_now(provider,'meter plus live attempt liabilities plus remaining work exceeds cap')
            meter=sample.get('metered_usd')
            if meter is not None:
                if not math.isfinite(meter) or meter<self.last_meter:raise ValueError('invalid cumulative meter')
                self.last_meter=meter
                self.account=self.bookkeep({'kind':'metered','app_id':self.app_id,'usd':meter})
                if self.account['stop_required']:self.stop_now(provider,'metered spend exceeds complete cap')
            if sample['stopped'] and not active:
                self.confirmed=True;self.confirmed_at=self.now()
                if self.confirmed_at>self.expires_at:self.uncertainty('termination confirmed after reserved envelope')
                self.emit('stopped_confirmed',attempts_observed=len(self.tasks),last_progress=self.last_progress,within_envelope=self.confirmed_at<=self.expires_at)
                try:
                    for job in self.jobs or [self.job]:self.bookkeep({'kind':'stopped','job':job,'app_id':self.app_id,'confirmed':True,'tasks_empty':True})
                except Exception as exc:self.emit('ledger_unavailable',error=str(exc))
                return True
        except Exception as exc:
            self.stop_now(provider,'observation/accounting unavailable');self.emit('observation_error',error=f'{type(exc).__name__}: {exc}')
        if self.now()>self.expires_at and not self.confirmed:
            self.uncertainty('termination not confirmed by reserved envelope expiry');self.emit('unconfirmed_overrun',liability_retained=True)
        return False

class ModalProvider:
    """Pinned SDK protocol; durable task-state cursor plus active task/app state."""
    def __init__(self):
        from modal.client import _Client
        self.loop=asyncio.new_event_loop();self.client=self.loop.run_until_complete(asyncio.wait_for(_Client.from_env(),RPC_TIMEOUT_S))
        self.cursor='';self.task_birth={};self.task_generation={};self.pending={};self.controls=[];self.containers={};self.last_meter=None
    def request(self,name,message):
        return self.loop.run_until_complete(asyncio.wait_for(getattr(self.client._stub,name)(message),RPC_TIMEOUT_S))
    def discover(self,description,started):
        from modal_proto import api_pb2 as P
        from modal._object import _get_environment_name
        matches=[a for a in self.request('AppList',P.AppListRequest(environment_name=_get_environment_name())).apps if a.description==description and a.created_at>=started-1]
        if len(matches)>1:raise RuntimeError('nonunique ephemeral app identity')
        return matches[0].app_id if matches else None
    def task_events(self,app):
        from modal_proto import api_pb2 as P
        async def collect():
            events=[]
            request=P.AppGetLogsRequest(app_id=app,last_entry_id=self.cursor,timeout=.1)
            async for batch in self.client._stub.AppGetLogs.unary_stream(request):
                for item in batch.items:
                    task=batch.task_id
                    if not task or not item.task_state:continue
                    timestamp=item.timestamp or time.time()
                    state=P.TaskState.Name(item.task_state)
                    generation=self.task_generation.setdefault(task,0)
                    fresh_container=item.container_id and self.containers.get(task) and item.container_id!=self.containers[task]
                    if fresh_container or (state in ('TASK_STATE_CREATED','TASK_STATE_CREATING_CONTAINER') and self.pending.get(task) in ('preempted','completed')):
                        generation+=1;self.task_generation[task]=generation
                        self.containers.pop(task,None)
                    if item.container_id:self.containers[task]=item.container_id
                    key=f'{task}:{generation}'
                    birth=self.task_birth.setdefault(key,timestamp)
                    terminal={'TASK_STATE_COMPLETED':'completed','TASK_STATE_PREEMPTED':'preempted'}
                    status=terminal.get(state,'live');self.pending[task]=status
                    events.append({'attempt':key,'task':task,'container_id':item.container_id,'state':status,'provider_state':state,'elapsed_s':max(0.,timestamp-birth),'provider_timestamp':timestamp})
                if batch.entry_id:self.cursor=batch.entry_id
            return events
        return self.loop.run_until_complete(asyncio.wait_for(collect(),RPC_TIMEOUT_S))
    def sample(self,app):
        from modal_proto import api_pb2 as P
        events=self.task_events(app)
        life=self.request('AppGetLifecycle',P.AppGetLifecycleRequest(app_id=app))
        tasks=self.request('TaskList',P.TaskListRequest(app_id=app))
        # Give active tasks the same generation key as cursor events; this is
        # also the fallback for starts whose first state message is delayed.
        for task in tasks.tasks:
            key=f'{task.task_id}:{self.task_generation.get(task.task_id,0)}'
            if not any(e['attempt']==key for e in events):
                birth=task.enqueued_at or task.started_at or time.time();self.task_birth.setdefault(key,birth)
                events.append({'attempt':key,'task':task.task_id,'state':'live','elapsed_s':max(0.,time.time()-self.task_birth[key])})
        return {'stopped':life.lifecycle.app_state==P.APP_STATE_STOPPED,'tasks':[t.task_id for t in tasks.tasks],'attempt_events':events,'event_cursor':self.cursor,'metered_usd':self.last_meter}
    def cancel(self,call):
        self.controls.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--cancel',call],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True))
    def stop(self,app):
        if not app:return
        self.controls=[p for p in self.controls if p.poll() is None]
        # Repeated independent CLI requests remain possible if an earlier one
        # hangs; each has a local RPC/command timeout in its own process.
        self.controls.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--stop',app],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True))
    def metering(self,app,started):
        from meter8086 import billing
        return self.loop.run_until_complete(asyncio.wait_for(billing(app,started),RPC_TIMEOUT_S))['metered_usd']


def watch(config_path):
    from dispatch8086 import SpendGuard
    config=json.loads(Path(config_path).read_text());guard=SpendGuard(config['ledger'],config['plan'],cap=config['cap'],overhead=config['overhead'])
    provider=ModalProvider()
    state=Lifecycle(config['events'],app_id=config.get('app_id'),execution_s=config['execution_s'],started=config['started_monotonic'],guard=guard,job=config['job'],setup=True,jobs=config.get('jobs'),max_containers=config.get('max_containers',1))
    status_writer=AsyncStatus(config['status'])
    status_writer.publish({'ready':True,'heartbeat':time.monotonic()})
    last_meter=0.
    while not state.confirmed:
        control=json.loads(Path(config_path).read_text());state.call_id=control.get('call_id')
        state.call_ids=list(control.get('call_ids',{}).values())
        if state.jobs:
            state.jobs.update(control['jobs'])
            state.reservations.update(control.get('reservations',{}))
        if control.get('body_started') is not None:state.body_started=control['body_started']
        try:os.kill(config['parent_pid'],0);alive=True
        except ProcessLookupError:alive=False
        if not state.app_id:
            try:state.app_id=control.get('app_id') or provider.discover(config['app_name'],config['started_epoch'])
            except Exception as exc:state.requested=True;state.emit('discovery_error',error=str(exc))
        if state.app_id:
            if time.monotonic()-last_meter>=60 and time.monotonic()+60<state.stop_at:
                try:
                    provider.last_meter=provider.metering(state.app_id,config['started_epoch'])
                except Exception as exc:state.stop_now(provider,'meter unavailable');state.emit('meter_error',error=str(exc))
                last_meter=time.monotonic()
            state.poll(provider,finish=control.get('finish',False),parent_alive=alive)
        elif time.monotonic()>=state.started+SETUP_TIMEOUT_S or not alive or control.get('finish'):
            state.requested=True;state.emit('setup_incomplete',status='INCOMPLETE')
            # Keep discovering: a late app-creation response must still be stopped.
        status_writer.publish({'ready':True,'heartbeat':time.monotonic(),'app_id':state.app_id,'confirmed':state.confirmed,'confirmed_at':state.confirmed_at,'within_envelope':state.confirmed and not state.unknown_bound and state.confirmed_at<=state.expires_at,'stop_required':state.requested,'incomplete':state.incomplete,'charged_usd':state.charged_usd,'dispatch_permitted':state.body_started is not None and not state.requested and state.account is not None})
        if not state.confirmed:time.sleep(WATCHDOG_POLL_S)
    state.flush_log();status_writer.flush()

class WatchdogProcess:
    def __init__(self,guard,job,app_id,execution_s,*,jobs=None,max_containers=1):
        prefix=guard.path.parent/f'lifecycle-{sha(job.encode())[:16]}'
        self.path=Path(str(prefix)+'.control.json');self.status=Path(str(prefix)+'.status.json');self.events=Path(str(prefix)+'.events.jsonl')
        if self.path.exists():raise FileExistsError('lifecycle already reserved; no restart')
        self.config={'ledger':str(guard.path.resolve()),'plan':guard.plan,'cap':guard.cap,'overhead':guard.overhead,'job':job,'app_id':app_id,'app_name':f's2-{uuid.uuid4().hex}','execution_s':execution_s,'events':str(self.events.resolve()),'status':str(self.status.resolve()),'parent_pid':os.getpid(),'started_monotonic':time.monotonic(),'started_epoch':time.time()}
        if jobs:self.config.update(jobs=dict(jobs),reservations={n:guard.plan[n] for n in jobs},max_containers=max_containers,call_ids={})
        atomic_json(self.path,self.config)
        with Path(str(prefix)+'.watchdog.log').open('x') as log:
            self.process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--watch',str(self.path.resolve())],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        limit=time.monotonic()+RPC_TIMEOUT_S+5
        while time.monotonic()<limit:
            if self.process.poll() is not None:raise RuntimeError('watchdog failed before setup')
            if self.confirmation().get('ready'):return
            time.sleep(.05)
        self.finish();raise TimeoutError('watchdog not ready; setup forbidden')
    @property
    def app_name(self):return self.config['app_name']
    def bind_app(self,app_id):
        if time.monotonic()>self.config['started_monotonic']+SETUP_TIMEOUT_S:
            self.config['app_id']=app_id;self.finish();raise TimeoutError('app setup timeout')
        self.config.update(app_id=app_id,body_started=time.monotonic());atomic_json(self.path,self.config)
    def permit(self):
        limit=time.monotonic()+3*RPC_TIMEOUT_S+5
        while time.monotonic()<limit:
            status=self.confirmation()
            if status.get('stop_required') or self.process.poll() is not None:raise RuntimeError('watchdog spend/observation stop; INCOMPLETE')
            if status.get('dispatch_permitted') and time.monotonic()-status['heartbeat']<3*RPC_TIMEOUT_S+WATCHDOG_POLL_S:return
            time.sleep(.05)
        self.finish();raise TimeoutError('no current watchdog dispatch permit')
    def bind(self,call_id):self.config['call_id']=call_id;atomic_json(self.path,self.config)
    def bind_call(self,job,call_id):
        self.config['call_ids'][job]=call_id;atomic_json(self.path,self.config)
    def add_recovery(self,job,execution_s,reservation):
        self.config['jobs'][job]=execution_s;self.config['reservations'][job]=reservation
        atomic_json(self.path,self.config)  # fixed original clock is not renewed
    def finish(self):self.config['finish']=True;atomic_json(self.path,self.config)
    def confirmation(self):return json.loads(self.status.read_text()) if self.status.exists() else {}
    def wait_confirmed(self):
        limit=time.monotonic()+TERMINATION_GRACE_S
        while time.monotonic()<limit:
            status=self.confirmation()
            if status.get('confirmed'):
                if not status.get('within_envelope'):raise RuntimeError('late/unknown stop; liability retained')
                if status.get('incomplete'):raise RuntimeError('watchdog cap/infrastructure stop; experiment INCOMPLETE')
                return status
            time.sleep(.1)
        raise TimeoutError('stopping unconfirmed; liability retained, watchdog continues')

if __name__=='__main__':
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--watch',type=Path);group.add_argument('--stop');group.add_argument('--cancel');a=ap.parse_args()
    if a.watch:watch(a.watch)
    elif a.stop:
        subprocess.run([sys.executable,'-m','modal','app','stop','--yes',a.stop],timeout=RPC_TIMEOUT_S,check=True)
    else:
        from modal_proto import api_pb2 as P
        provider=ModalProvider();provider.request('FunctionCallCancel',P.FunctionCallCancelRequest(function_call_id=a.cancel,terminate_containers=True))

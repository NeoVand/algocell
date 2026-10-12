"""Coordinator B5 acceptance paths. Real locks/processes, mock provider only."""
import asyncio,ast,json,os,signal,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import pytest
import s8086 as S
from dispatch8086 import SpendGuard,bounded_call
from lifecycle8086 import Lifecycle,dispatch_seconds,ModalProvider,SETUP_TIMEOUT_S,ATTEMPT_TERMINATION_S

class Provider:
    def __init__(self):self.events=[];self.tasks=[];self.stopped=False;self.stops=0;self.cancels=[];self.meter=0
    def sample(self,app):return {'stopped':self.stopped,'tasks':self.tasks,'attempt_events':self.events,'metered_usd':self.meter}
    def stop(self,app):self.stops+=1
    def cancel(self,call):self.cancels.append(call)


def test_sigstop_dispatcher_holding_real_ledger_lock(tmp_path):
    guard=SpendGuard(tmp_path/'ledger',{'a':1},overhead=0);guard.reserve('a',1);guard.start('a','call','app')
    script=tmp_path/'locked_dispatcher.py'
    script.write_text('''import fcntl,os,signal,sys,time
from pathlib import Path
with Path(sys.argv[1]).open('a+') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 Path(sys.argv[2]).write_text(str(os.getpid()))
 os.kill(os.getpid(),signal.SIGSTOP)
 time.sleep(60)
''')
    parent=subprocess.Popen([sys.executable,str(script),str(guard.path),str(tmp_path/'locked')],start_new_session=True)
    try:
        limit=time.monotonic()+3
        while not (tmp_path/'locked').exists() and time.monotonic()<limit:time.sleep(.01)
        assert (tmp_path/'locked').exists()
        # Another independent process uses the actual SpendGuard and actual
        # dispatcher-held flock; the test runner only oversees/cleans up.
        watch=tmp_path/'mock_watchdog.py'
        watch.write_text('''import sys,time
from pathlib import Path
from dispatch8086 import SpendGuard
from lifecycle8086 import Lifecycle
root=Path(sys.argv[1]);guard=SpendGuard(root/'ledger',{'a':1},overhead=0)
class Mock:
 def sample(self,app):return {'stopped':False,'tasks':['attempt'],'metered_usd':0}
 def stop(self,app):(root/'stop').write_text(app)
 def cancel(self,call):(root/'cancel').write_text(call)
state=Lifecycle(root/'events',app_id='app',call_id='call',execution_s=1,guard=guard,job='a')
t=time.monotonic();state.poll(Mock());state.flush_log()
(root/'elapsed').write_text(str(time.monotonic()-t))
''')
        child=subprocess.Popen([sys.executable,str(watch),str(tmp_path)],start_new_session=True)
        child.wait(timeout=3)
        assert child.returncode==0
        assert (tmp_path/'stop').read_text()=='app' and (tmp_path/'cancel').read_text()=='call'
        assert float((tmp_path/'elapsed').read_text())<.5
        os.kill(parent.pid,0) # suspended dispatcher remains live, as in Astra
        assert any('observation_error' in row for row in (tmp_path/'events').read_text().splitlines())
    finally:
        os.kill(parent.pid,signal.SIGCONT);parent.kill();parent.wait()


def test_logging_and_bookkeeping_cannot_delay_stop(tmp_path,monkeypatch):
    state=Lifecycle(tmp_path/'log',app_id='app',call_id='call',execution_s=1,started=0,now=lambda:1000)
    p=Provider();order=[]
    p.stop=lambda app:order.append('stop');p.cancel=lambda call:order.append('cancel')
    def blocked_emit(*a,**kw):
        assert order[:2]==['stop','cancel']
    monkeypatch.setattr(state,'emit',blocked_emit)
    state.poll(p)
    assert order[:2]==['stop','cancel']


def test_timed_ledger_io_stops_even_after_nonblocking_lock_acquired(tmp_path,monkeypatch):
    g=SpendGuard(tmp_path/'ledger',{'a':1},overhead=0);g.reserve('a',1)
    monkeypatch.setattr('lifecycle8086.BOOKKEEP_TIMEOUT_S',.05)
    monkeypatch.setattr(g,'transact',lambda *a,**kw:time.sleep(10))
    s=Lifecycle(tmp_path/'events',app_id='app',call_id='call',execution_s=1,guard=g,job='a')
    p=Provider();start=time.monotonic();s.poll(p)
    assert time.monotonic()-start<.2 and p.stops and p.cancels==['call']


@pytest.mark.parametrize('terminal',['preempted','completed','failed'])
def test_every_short_attempt_between_polls_is_charged(tmp_path,terminal):
    g=SpendGuard(tmp_path/'ledger',{'a':.5,'remaining':94.4},overhead=0);g.reserve('a',.5)
    s=Lifecycle(tmp_path/'events',app_id='app',call_id='call',execution_s=100,guard=g,job='a',started=0,now=lambda:1)
    p=Provider();p.events=[{'attempt':f'attempt-{i}','task':f'task-{i}','state':terminal,'elapsed_s':.01} for i in range(25)]
    s.poll(p)
    assert len(s.attempts)==25 and s.charged_usd>=25*ATTEMPT_TERMINATION_S*S.RATE
    assert p.stops and s.incomplete and s.account['projected_liabilities_usd']>S.CAP
    assert s.flush_log()
    assert sum(json.loads(x)['kind']=='attempt_observed' for x in s.path.read_text().splitlines())==25


def test_restarts_share_deadline_and_accumulate_live_liabilities(tmp_path):
    g=SpendGuard(tmp_path/'ledger',{'a':1,'remaining':93},overhead=0);g.reserve('a',1)
    clock=[0];s=Lifecycle(tmp_path/'events',app_id='app',call_id='call',execution_s=5000,guard=g,job='a',started=0,now=lambda:clock[0])
    p=Provider();p.tasks=['first'];s.poll(p);deadline=s.expires_at
    assert not p.stops
    clock[0]=5;p.tasks=['restart'];s.poll(p)
    assert p.stops and s.expires_at==deadline and len(s.attempts)==2
    assert s.charged_usd>=2*(5000+120+ATTEMPT_TERMINATION_S)*S.RATE


def test_meter_and_liabilities_are_both_enforced_and_no_new_work(tmp_path):
    g=SpendGuard(tmp_path/'ledger',{'a':1,'remaining':93},overhead=0);g.reserve('a',1)
    s=Lifecycle(tmp_path/'events',app_id='app',execution_s=100,guard=g,job='a')
    p=Provider();p.meter=2;p.tasks=['live'];s.poll(p)
    assert p.stops and s.incomplete
    g.transact({'kind':'terminal','job':'a','status':'INCOMPLETE'})
    with pytest.raises(RuntimeError,match='INCOMPLETE'):g.reserve('remaining',83)


@pytest.mark.parametrize('fault',['sample','stop','cancel'])
def test_control_failures_retain_liability(tmp_path,fault):
    s=Lifecycle(tmp_path/'events',app_id='app',call_id='call',execution_s=1,started=0,now=lambda:1000)
    p=Provider()
    def fail(*a):raise RuntimeError('mock control failure')
    setattr(p,fault,fail);s.poll(p)
    assert s.liability_retained and s.incomplete and s.unknown_bound
    assert s.flush_log()
    assert any('unconfirmed_overrun' in x for x in s.path.read_text().splitlines())


def test_setup_is_charged_and_can_stop_before_body(tmp_path):
    s=Lifecycle(tmp_path/'events',app_id='app',execution_s=100,setup=True,started=0,now=lambda:SETUP_TIMEOUT_S+1)
    p=Provider();s.poll(p)
    assert p.stops and s.incomplete and s.charged_usd>=SETUP_TIMEOUT_S*S.RATE
    assert s.reservation_seconds==dispatch_seconds(100)


def test_dispatch_arms_before_setup_and_checks_cap_before_spawn(tmp_path):
    from contextlib import contextmanager
    order=[]
    class App:
        app_id='app'
        @contextmanager
        def run(self,**kw):order.append('setup');yield
    class Monitor:
        app_name='unique'
        def __init__(self,*args):order.append('armed')
        def bind_app(self,app):order.append('bind_app')
        def permit(self):order.append('permit');raise RuntimeError('mock cap stop')
        def finish(self):order.append('finish')
        def wait_confirmed(self):order.append('confirmed')
    class Function:
        def spawn(self,*args):raise AssertionError('new work dispatched after cap stop')
    g=SpendGuard(tmp_path/'ledger',{'a':dispatch_seconds(1)*S.RATE},overhead=0)
    with pytest.raises(RuntimeError,match='mock cap stop'):bounded_call(App(),Function(),(),g,'a',1,watchdog_factory=Monitor)
    assert order[:4]==['armed','setup','bind_app','permit']


def test_production_stop_and_cancel_launch_independent_processes(monkeypatch):
    import lifecycle8086 as L
    calls=[]
    class Child:
        def poll(self):return None
    monkeypatch.setattr(L.subprocess,'Popen',lambda cmd,**kw:(calls.append((cmd,kw)) or Child()))
    provider=object.__new__(ModalProvider);provider.controls=[]
    provider.stop('app');provider.cancel('call')
    assert [c[0][-2:] for c in calls]==[['--stop','app'],['--cancel','call']]
    assert all(c[1]['start_new_session'] for c in calls)
    assert "'modal','app','stop'" in Path(L.__file__).read_text()


def test_cursor_catches_preempted_and_restarted_container_same_task():
    from modal_proto import api_pb2 as P
    provider=object.__new__(ModalProvider)
    provider.loop=asyncio.new_event_loop();provider.cursor='';provider.task_birth={};provider.task_generation={};provider.pending={};provider.containers={}
    batches=[P.TaskLogsBatch(task_id='ta-1',entry_id='1',items=[P.TaskLogs(task_state=P.TASK_STATE_CREATED,timestamp=1,container_id='old'),P.TaskLogs(task_state=P.TASK_STATE_PREEMPTED,timestamp=2,container_id='old')]),
             P.TaskLogsBatch(task_id='ta-1',entry_id='2',items=[P.TaskLogs(task_state=P.TASK_STATE_CREATED,timestamp=3,container_id='new'),P.TaskLogs(task_state=P.TASK_STATE_COMPLETED,timestamp=4,container_id='new')])]
    class Stream:
        async def unary_stream(self,request):
            assert request.last_entry_id==''
            for b in batches:yield b
    provider.client=SimpleNamespace(_stub=SimpleNamespace(AppGetLogs=Stream()))
    events=provider.task_events('app');provider.loop.close()
    assert {e['attempt'] for e in events}=={'ta-1:0','ta-1:1'} and provider.cursor=='2'
    assert [e['state'] for e in events]==['live','preempted','live','completed']


def test_image_prebuild_receipt_gate_and_nominal_forecast(tmp_path,monkeypatch):
    from prebuild8086 import require_prebuild,validated_image_id
    from benchmark8086 import project
    monkeypatch.delenv('S2_PREBUILT_IMAGE_RECEIPT',raising=False)
    with pytest.raises(ValueError,match='prebuild'):require_prebuild()
    p=tmp_path/'receipt';p.write_text(json.dumps({'image_id':'im-mock','stopped_confirmed':True,'freeze_sha256':S.sha((S.PRE/'freeze.json').read_bytes()),'recipe_sha256':S.sha((S.HERE/'modal_s8086.py').read_bytes())}))
    assert validated_image_id(p)=='im-mock'
    p.write_text('{}')
    with pytest.raises(ValueError):validated_image_id(p)
    forecast=project(S.read_json(S.PRE/'benchmark.json'),.7999038189802217)
    assert forecast['projection_usd']==pytest.approx(83.79381223898022)
    assert forecast['projection_usd']<=S.CAP and 'prebuild' in forecast['job_liabilities_usd']
    assert forecast['lifecycle_policy']['setup_reserved_each_app']


def test_recovery_starts_step0_and_never_publishes_partial_history(tmp_path):
    from recovery8086 import attempt_directory,publish_complete
    reg=S.Registration(1000000,('A',));a,meta=attempt_directory(tmp_path,'world',attempt_id='failed')
    assert meta['policy']=='restart_step_0' and meta['checkpoint_resume'] is False
    (a/'world.jsonl').write_text(json.dumps({'kind':'incomplete','steps_run':20000})+'\n')
    with pytest.raises(ValueError,match='INCOMPLETE'):publish_complete(a,tmp_path,'world',reg)
    b,meta=attempt_directory(tmp_path,'world',attempt_id='restarted')
    assert meta['prior_attempt_ids']==['failed']
    rows=[{'kind':'condition','seed':S.SEEDS[0],'infrastructure_recovery':meta}]
    for step in reg.sample_steps:
        rows.append({'kind':'sample','step':step})
        if step in reg.snapshots:
            name=f'world.t{step}.u8.br';(b/name).write_bytes(b'mock')
            rows.append({'kind':'snapshot','step':step,'file':name})
    rows.append({'kind':'summary','steps_run':reg.horizon})
    (b/'world.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    publish_complete(b,tmp_path,'world',reg)
    assert json.loads((tmp_path/'world.jsonl').read_text().splitlines()[1])['step']==0
    assert 'incomplete' not in (tmp_path/'world.jsonl').read_text()


def test_blocked_watchdog_journal_writer_cannot_block_stop(tmp_path):
    path=tmp_path/'journal.fifo';os.mkfifo(path)
    state=Lifecycle(path,app_id='app',call_id='call',execution_s=1,started=0,now=lambda:1000)
    p=Provider();start=time.monotonic();state.poll(p)
    assert p.stops and p.cancels==['call'] and time.monotonic()-start<.2
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK)
    os.close(fd)


def test_repeated_observation_is_not_an_additional_attempt(tmp_path):
    s=Lifecycle(tmp_path/'events',app_id='app',execution_s=100,started=0,now=lambda:1)
    p=Provider();p.tasks=['task'];p.events=[{'attempt':'task:container','task':'task','state':'live','elapsed_s':1}]
    s.poll(p);charge=s.charged_usd;s.poll(p)
    assert len(s.attempts)==1 and s.charged_usd==charge


def test_nominal_single_attempt_complete_plan_does_not_double_bill_elapsed(tmp_path):
    projection=S.read_json(S.PRE/'projection.json');g=SpendGuard(tmp_path/'ledger',projection['job_liabilities_usd'],overhead=projection['storage_termination_reserve_usd'])
    for job,deadline in projection['execution_deadline_s'].items():
        app=f'app-{job}';g.reserve(job,g.plan[job]);g.start(job,'call',app)
        clock=[120.];s=Lifecycle(tmp_path/(S.sha(job.encode())+'.events'),app_id=app,execution_s=deadline,guard=g,job=job,started=0,now=lambda:clock[0],setup=True);s.body_started=120.
        p=Provider();p.tasks=['task'];p.events=[{'attempt':'task','task':'task','state':'live','elapsed_s':0}];s.poll(p)
        clock[0]=120+deadline;p.tasks=[];p.stopped=True;p.meter=deadline*S.RATE
        p.events=[{'attempt':'task','task':'task','state':'completed','elapsed_s':deadline}]
        s.poll(p,finish=True)
        assert not s.incomplete and s.confirmed and s.account['projected_liabilities_usd']<=S.CAP
        g.settle(job,g.plan[job]);assert s.flush_log()
    result=g.transact({'kind':'check'})
    assert result['projected_liabilities_usd']==pytest.approx(projection['projection_usd']) and not result['stop_required']


def test_prebuild_has_its_own_timed_accounted_step(tmp_path,monkeypatch):
    from contextlib import contextmanager
    import modal,modal_s8086,prebuild8086,lifecycle8086
    order=[]
    monkeypatch.setattr(prebuild8086,'launch_receipt',lambda _: {'freeze_sha256':S.sha((S.PRE/'freeze.json').read_bytes())})
    class App:
        app_id='build-app'
        @contextmanager
        def run(self,**kwargs):order.append('app_setup');yield
    class Monitor:
        app_name='unique-build'
        def __init__(self,guard,job,*args):self.guard,self.job=guard,job;order.append('armed')
        def bind_app(self,app):self.app=app
        def permit(self):order.append('permit')
        def finish(self):order.append('finish')
        def wait_confirmed(self):self.guard.transact({'kind':'stopped','job':self.job,'app_id':self.app,'confirmed':True,'tasks_empty':True})
        def confirmation(self):return {}
    class Image:
        object_id='im-mock'
        class Build:
            async def aio(self,app):order.append('image_build')
        build=Build()
    real_wait=asyncio.wait_for
    async def timed(awaitable,timeout):
        assert timeout==600;order.append('timed_prebuild');return await real_wait(awaitable,timeout)
    monkeypatch.setattr(modal,'App',App);monkeypatch.setattr(modal_s8086,'build_image_spec',lambda:Image())
    monkeypatch.setattr(lifecycle8086,'WatchdogProcess',Monitor);monkeypatch.setattr(prebuild8086.asyncio,'wait_for',timed)
    prebuild8086.prebuild('mock',tmp_path/'ledger',tmp_path/'image_receipt')
    assert order==['armed','app_setup','permit','timed_prebuild','image_build','finish']
    rows=[json.loads(x) for x in (tmp_path/'ledger').read_text().splitlines()]
    assert next(r for r in rows if r['kind']=='reserve')['job']=='prebuild'
    assert json.loads((tmp_path/'image_receipt').read_text())['stopped_confirmed']


def test_sdk_supported_overrides_and_all_declared_server_controls():
    import inspect,modal
    allowed=set(inspect.signature(modal.Function.with_options).parameters)
    tree=ast.parse((S.HERE/'dispatch8086.py').read_text())
    overrides=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='with_options']
    assert overrides and all(set(k.arg for k in n.keywords)<=allowed for n in overrides)
    for name in ('modal_s8086.py','modal_score8086.py'):
        tree=ast.parse((S.HERE/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='function':
                fields={k.arg:k.value for k in node.keywords}
                assert 'timeout' in fields and 'startup_timeout' in fields
                for key,value in [('retries',0),('single_use_containers',True),('max_containers',1),('min_containers',0),('buffer_containers',0)]:assert ast.literal_eval(fields[key])==value


def test_independent_setup_discovery_uses_exact_unique_name_and_environment(monkeypatch):
    from modal_proto import api_pb2 as P
    import modal._object
    monkeypatch.setattr(modal._object,'_get_environment_name',lambda:'mock-env')
    p=object.__new__(ModalProvider)
    def request(method,message):
        assert method=='AppList' and message.environment_name=='mock-env'
        return P.AppListResponse(apps=[P.AppListResponse.AppListItem(app_id='right',description='unique',created_at=10),P.AppListResponse.AppListItem(app_id='wrong',description='other',created_at=10)])
    p.request=request
    assert p.discover('unique',10)=='right'

"""Modal v1 publication and simultaneous shared-app liability regressions."""
import json
import os
from contextlib import contextmanager
import pytest
import s8086 as S
from dispatch8086 import SpendGuard,bounded_parallel
from lifecycle8086 import Lifecycle,dispatch_seconds
from recovery8086 import attempt_directory,publish_complete
from test_b5_enforcement import Provider

def complete(root,attempt_id='first'):
    reg=S.Registration(250000,('A',));attempt,_=attempt_directory(root,'world',attempt_id=attempt_id)
    rows=[{'kind':'sample','step':s} for s in reg.sample_steps]
    for step in reg.snapshots:
        filename=f'world.t{step}';(attempt/filename).write_bytes(attempt_id.encode())
        rows.append({'kind':'snapshot','step':step,'file':filename})
    rows.append({'kind':'summary','steps_run':reg.horizon})
    (attempt/'world.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return attempt,reg

def test_publish_without_hardlinks_and_duplicate_does_not_clobber(tmp_path,monkeypatch):
    def unsupported(*args):raise PermissionError(1,'Modal volume rejects hard links')
    monkeypatch.setattr(os,'link',unsupported)
    attempt,reg=complete(tmp_path);publish_complete(attempt,tmp_path,'world',reg)
    before={p.name:p.read_bytes() for p in tmp_path.glob('world.*')}
    duplicate,_=complete(tmp_path,'duplicate');publish_complete(duplicate,tmp_path,'world',reg)
    assert before=={p.name:p.read_bytes() for p in tmp_path.glob('world.*')}
    assert json.loads((duplicate/'publication.json').read_text())['selected'] is False

def test_interrupted_publish_retains_exclusive_marker_and_never_clobbers(tmp_path,monkeypatch):
    attempt,reg=complete(tmp_path)
    replace=os.replace
    def interrupt(source,target):
        if str(target).endswith('world.jsonl'):raise OSError('injected publication failure')
        return replace(source,target)
    monkeypatch.setattr(os,'replace',interrupt)
    with pytest.raises(OSError):publish_complete(attempt,tmp_path,'world',reg)
    assert not (tmp_path/'world.jsonl').exists()
    assert (tmp_path/'world.jsonl.publish-reserved').exists()
    another,_=complete(tmp_path,'another')
    with pytest.raises(FileExistsError):publish_complete(another,tmp_path,'world',reg)

def test_shared_app_meter_is_counted_once_with_all_live_reservations(tmp_path):
    guard=SpendGuard(tmp_path/'ledger',{'a':2.,'b':3.,'future':4.},overhead=1.)
    for name in ('a','b'):guard.reserve(name,guard.plan[name]);guard.start(name,'call-'+name,'shared')
    guard.transact(dict(kind='batch_liability',jobs=['a','b'],app_id='shared',usd=5.,elapsed_usd=2.))
    result=guard.transact(dict(kind='metered',app_id='shared',usd=2.))
    assert result['projected_liabilities_usd']==pytest.approx(10.)
    result=guard.transact(dict(kind='metered',app_id='shared',usd=3.))
    assert result['projected_liabilities_usd']==pytest.approx(11.)
    assert SpendGuard(guard.path,guard.plan,overhead=1.).transact(dict(kind='check'))==result

def test_20_parallel_tasks_keep_reservations_until_whole_app_stops(tmp_path):
    jobs={f'world:A:{seed}':100 for seed in S.SEEDS};plan={n:dispatch_seconds(d)*S.RATE for n,d in jobs.items()}
    guard=SpendGuard(tmp_path/'ledger',plan,overhead=0)
    for name in jobs:guard.reserve(name,plan[name]);guard.start(name,'call','shared')
    state=Lifecycle(tmp_path/'events',app_id='shared',execution_s=100,guard=guard,job=next(iter(jobs)),
                    jobs=jobs,max_containers=20,started=0,now=lambda:50,setup=True)
    state.body_started=0.;provider=Provider();provider.tasks=[str(i) for i in range(20)]
    state.poll(provider)
    assert not state.incomplete and state.charged_usd==pytest.approx(sum(plan.values()))
    provider.tasks.pop();provider.events=[dict(attempt='19',task='19',state='completed',elapsed_s=50)]
    state.poll(provider)
    assert not state.incomplete and state.charged_usd==pytest.approx(sum(plan.values()))
    assert not state.confirmed
    provider.stopped=True;provider.tasks=[];provider.events=[];state.poll(provider,finish=True)
    assert state.confirmed and not state.incomplete

def test_parallel_stop_starts_before_unavailable_bookkeeping(tmp_path,monkeypatch):
    jobs={'world:A:1':1,'world:A:2':1};guard=SpendGuard(tmp_path/'ledger',{n:1. for n in jobs},overhead=0)
    state=Lifecycle(tmp_path/'events',app_id='shared',execution_s=1,guard=guard,job=next(iter(jobs)),
                    jobs=jobs,max_containers=2,started=0,now=lambda:10000)
    state.call_ids=['call-1','call-2'];provider=Provider();order=[]
    provider.stop=lambda app:order.append('stop');provider.cancel=lambda call:order.append(call)
    def blocked(event):
        assert order[:3]==['stop','call-1','call-2']
        raise BlockingIOError('ledger held by suspended dispatcher')
    monkeypatch.setattr(state,'bookkeep',blocked);state.poll(provider)
    assert state.incomplete and order[:3]==['stop','call-1','call-2']

def test_vanished_attempt_and_replacement_keep_queued_world_reserved(tmp_path):
    jobs={'world:A:1':5000,'world:A:2':5000};plan={n:dispatch_seconds(d)*S.RATE for n,d in jobs.items()}
    guard=SpendGuard(tmp_path/'ledger',plan,overhead=0)
    for name in jobs:guard.reserve(name,plan[name]);guard.start(name,'call','shared')
    state=Lifecycle(tmp_path/'events',app_id='shared',execution_s=5000,guard=guard,job=next(iter(jobs)),
                    jobs=jobs,max_containers=2,started=0,now=lambda:10,setup=True)
    state.body_started=0.;provider=Provider();provider.tasks=['old'];state.poll(provider)
    provider.tasks=['replacement'];state.poll(provider)
    assert len(state.attempts)==2 and state.attempts['old']['live']
    # Unknown old task, live replacement and one still-queued original world.
    assert state.charged_usd>=3*(5000+120+121)*S.RATE

def test_parallel_dispatch_submits_every_job_before_wait_and_settles_after_stop(tmp_path):
    order=[];plan={f'world:A:{i}':dispatch_seconds(1)*S.RATE for i in range(20)}
    guard=SpendGuard(tmp_path/'ledger',plan,overhead=0)
    class App:
        app_id='shared'
        @contextmanager
        def run(self,**kw):order.append('app');yield
    class Call:
        def __init__(self,name):self.object_id=name
        def get(self):
            assert sum(x=='spawn' for x in order)==20
            return {'body_complete':True}
    class Function:
        def spawn(self,name):order.append('spawn');return Call(name)
    class Monitor:
        app_name='unique'
        def __init__(self,g,job,app,deadline,**kw):
            assert len(kw['jobs'])==20 and kw['max_containers']==20
            assert sum(json.loads(s)['kind']=='reserve' for s in g.path.read_text().splitlines())==20
            order.append('armed')
        def bind_app(self,app):pass
        def permit(self):pass
        def bind_call(self,name,call):pass
        def finish(self):order.append('finish')
        def wait_confirmed(self):
            order.append('stopped')
            for name in plan:guard.transact(dict(kind='stopped',job=name,app_id='shared',confirmed=True,tasks_empty=True))
        def confirmation(self):return {'charged_usd':sum(plan.values())}
    result=bounded_parallel(App(),[(n,Function(),(n,),1) for n in plan],guard,watchdog_factory=Monitor)
    assert len(result)==20 and order[:2]==['armed','app']
    rows=[json.loads(s) for s in guard.path.read_text().splitlines()]
    assert max(i for i,r in enumerate(rows) if r['kind']=='stopped')<min(i for i,r in enumerate(rows) if r['kind']=='settled')

def test_parallel_fault_simulation():
    from spend_sim8086 import simulate_parallel
    result=simulate_parallel(64)
    assert result['violations']==0 and result['concurrent_worlds']==20
    assert result['outcomes']['COMPLETE'] and result['outcomes']['INCOMPLETE']
    assert result['interleaved_job_transitions'] and result['restarts']

def test_completed_cleanup_meter_overlaps_retained_cohort(tmp_path):
    jobs={'world:A:1':100,'world:A:2':100};plan={n:dispatch_seconds(d)*S.RATE for n,d in jobs.items()}
    base=sum(plan.values());guard=SpendGuard(tmp_path/'ledger',plan,cap=base+1e-7,overhead=0)
    for name in jobs:guard.reserve(name,plan[name]);guard.start(name,'call','shared')
    state=Lifecycle(tmp_path/'events',app_id='shared',execution_s=100,guard=guard,job=next(iter(jobs)),
                    jobs=jobs,max_containers=2,started=0,now=lambda:60,setup=True)
    state.body_started=0.;provider=Provider();provider.tasks=['live']
    provider.events=[dict(attempt='ended',task='ended',state='completed',elapsed_s=50),
                     dict(attempt='live',task='live',state='live',elapsed_s=60)]
    provider.meter=(50+60+10)*S.RATE  # ten cleanup seconds beyond observed body
    state.poll(provider)
    assert not state.incomplete and state.charged_usd==pytest.approx(base)
    assert state.account['projected_liabilities_usd']==pytest.approx(base)
    assert SpendGuard(guard.path,plan,cap=base+1e-7,overhead=0).transact(dict(kind='check'))==state.account

@pytest.mark.parametrize('server_timeout',[True,False])
def test_world_timeout_recovery_reserves_before_same_seed_step0_spawn(tmp_path,server_timeout):
    from modal.exception import FunctionTimeoutError
    order=[];job='world:A:8086001';reservation=dispatch_seconds(9000)*S.RATE
    guard=SpendGuard(tmp_path/'ledger',{job:reservation},overhead=0)
    class App:
        app_id='shared'
        @contextmanager
        def run(self,**kw):yield
    class Call:
        def __init__(self,n):self.n=n;self.object_id=f'call-{n}'
        def get(self):
            if self.n==1:
                raise FunctionTimeoutError('server limit') if server_timeout else TimeoutError('spend/allocation stop; incomplete world')
            return {'body_complete':True}
    class Function:
        def spawn(self,arm,seed):
            rows=[json.loads(s) for s in guard.path.read_text().splitlines()]
            assert sum(r['kind']=='reserve' for r in rows)==len(order)+1
            order.append((arm,seed));return Call(len(order))
    class Monitor:
        app_name='unique'
        def __init__(self,*args,**kw):self.jobs=dict(kw['jobs'])
        def bind_app(self,app):pass
        def permit(self):pass
        def bind_call(self,name,call):pass
        def add_recovery(self,name,deadline,reservation):self.jobs[name]=deadline
        def finish(self):pass
        def wait_confirmed(self):
            for name in self.jobs:guard.transact(dict(kind='stopped',job=name,app_id='shared',confirmed=True,tasks_empty=True))
        def confirmation(self):return {'charged_usd':2*reservation}
    assert bounded_parallel(App(),[(job,Function(),('A',8086001),9000)],guard,watchdog_factory=Monitor)[job]['body_complete']
    assert order==[('A',8086001),('A',8086001)]
    rows=[json.loads(s) for s in guard.path.read_text().splitlines()]
    extension=next(r for r in rows if r['kind']=='plan_extension')
    assert extension['failed_call_id']=='call-1' and extension['attempt_files_inspected'] is False
    assert sum(r['usd'] for r in rows if r['kind']=='settled')==pytest.approx(2*reservation)

def test_confirmed_app_stop_resolves_vanished_tasks_without_releasing_cohort(tmp_path):
    jobs={'world:A:1':100,'world:A:2':100};plan={n:dispatch_seconds(d)*S.RATE for n,d in jobs.items()}
    guard=SpendGuard(tmp_path/'ledger',plan,overhead=0)
    for name in jobs:guard.reserve(name,plan[name]);guard.start(name,'call','shared')
    state=Lifecycle(tmp_path/'events',app_id='shared',execution_s=100,guard=guard,job=next(iter(jobs)),
                    jobs=jobs,max_containers=2,started=0,now=lambda:60,setup=True)
    state.body_started=0.;provider=Provider();provider.tasks=['one','two'];state.poll(provider)
    provider.tasks=[];provider.stopped=True;state.poll(provider,finish=True)
    assert state.confirmed and not any(a['live'] for a in state.attempts.values())
    assert state.charged_usd==pytest.approx(sum(plan.values()))

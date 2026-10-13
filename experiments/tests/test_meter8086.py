"""Rate limits are missing observations; resource time still enforces the cap."""
import fcntl
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from dispatch8086 import SpendGuard
from lifecycle8086 import Lifecycle,dispatch_seconds
from meter8086 import AppMeterPoller
from s8086 import RATE
from test_b5_enforcement import Provider

def test_rate_limit_backoff_preserves_cumulative_app_meter(tmp_path):
    clock=[0.];reads=[]
    def read():
        reads.append(clock[0])
        if len(reads)==1:return {'metered_usd':2.}
        if len(reads)<4:raise RuntimeError('Rate limit exceeded for workspace billing report requests')
        return {'metered_usd':3.}
    meter=AppMeterPoller('app',0,tmp_path,read=read,now=lambda:clock[0],wall=lambda:clock[0])
    assert meter.refresh()['metered_usd']==2.
    for at in (120,240):clock[0]=at;assert meter.refresh()['metered_usd']==2.
    assert meter.next_poll==480
    clock[0]=479;meter.refresh();assert reads==[0,120,240]
    clock[0]=480;assert meter.refresh()['metered_usd']==3.
    assert not meter.error and meter.failures==0
    meter.close()

def test_twenty_jobs_share_one_poller_and_one_app_charge(tmp_path):
    reads=[];entered=threading.Event();release=threading.Event()
    def read():
        reads.append('app');entered.set();assert release.wait(2)
        return {'metered_usd':2.}
    meter=AppMeterPoller('app',0,tmp_path/'meter',read=read)
    with pytest.raises(BlockingIOError):AppMeterPoller('app',0,tmp_path/'meter',read=read)
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures=[pool.submit(meter.refresh) for _ in range(20)]
        assert entered.wait(1);release.set()
        for f in futures:f.result()
    assert reads==['app']
    plan={f'score:{i}':.2 for i in range(20)}
    guard=SpendGuard(tmp_path/'ledger',plan,overhead=0)
    for job in plan:guard.reserve(job,plan[job]);guard.start(job,'call','app')
    guard.transact(dict(kind='batch_liability',jobs=list(plan),app_id='app',usd=4.,elapsed_usd=2.))
    for _ in plan:result=guard.transact(dict(kind='metered',app_id='app',usd=meter.last_meter))
    assert result['projected_liabilities_usd']==pytest.approx(4.)
    meter.close()

@pytest.mark.parametrize('failure',['rate_limit','timeout','unknown'])
def test_failed_billing_uses_actual_live_container_accrual(tmp_path,failure):
    clock=[120.]
    def read():
        if failure=='unknown':return {'metered_usd':None}
        if failure=='timeout':raise TimeoutError('billing timeout')
        raise RuntimeError('Rate limit exceeded for workspace billing report requests')
    meter=AppMeterPoller('app',0,tmp_path/'meter',read=read,now=lambda:clock[0],wall=lambda:clock[0])
    jobs={f'score:{i}':100 for i in range(20)};plan={n:dispatch_seconds(d)*RATE for n,d in jobs.items()}
    base=sum(plan.values());guard=SpendGuard(tmp_path/'ledger',plan,cap=base+1e-7,overhead=0)
    for n in jobs:guard.reserve(n,plan[n]);guard.start(n,'call','app')
    state=Lifecycle(tmp_path/'events',app_id='app',execution_s=100,guard=guard,job=next(iter(jobs)),
                    jobs=jobs,max_containers=20,started=0,now=lambda:clock[0],setup=True)
    state.body_started=0;provider=Provider();provider.tasks=[str(i) for i in range(8)]
    sample=provider.sample
    provider.sample=lambda app:{**sample(app),**meter.refresh()}
    state.poll(provider)
    assert state.last_meter is None and state.elapsed_usd>=0
    assert not state.incomplete and not state.account['cap_exceeded']
    clock[0]=150;state.poll(provider)
    assert state.elapsed_usd>=8*30*RATE
    assert not state.incomplete
    # Real elapsed resource time beyond the cap still stops with no billing.
    clock[0]=2000;state.poll(provider)
    assert state.incomplete and state.account['cap_exceeded']
    assert state.stop_reason=='complete spend projection exceeds cap'
    meter.close()

def test_workspace_backoff_applies_to_other_apps(tmp_path):
    clock=[0.];reads=[]
    def read():reads.append(clock[0]);raise TimeoutError('billing timed out')
    a=AppMeterPoller('a',0,tmp_path,read=read,now=lambda:clock[0],wall=lambda:clock[0])
    b=AppMeterPoller('b',0,tmp_path,read=read,now=lambda:clock[0],wall=lambda:clock[0])
    a.refresh();b.refresh();assert len(reads)==1
    clock[0]=120;a.refresh();b.refresh();assert len(reads)==2
    assert a.next_poll==360 and b.next_poll==360
    a.close();b.close()

def test_short_real_ledger_lock_collision_is_retried(tmp_path):
    guard=SpendGuard(tmp_path/'ledger',{'a':1.},overhead=0)
    guard.reserve('a',1.);guard.start('a','call','app')
    held=threading.Event()
    def holder():
        with guard.path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);held.set();time.sleep(.08)
    thread=threading.Thread(target=holder);thread.start();assert held.wait(1)
    state=Lifecycle(tmp_path/'events',app_id='app',execution_s=1,guard=guard,job='a')
    state.poll(Provider());thread.join()
    assert not state.incomplete and not state.account['stop_required']

def test_terminal_state_does_not_claim_the_zero_meter_exceeds_cap(tmp_path):
    guard=SpendGuard(tmp_path/'ledger',{'a':1.},overhead=0)
    guard.reserve('a',1.);guard.start('a','call','app')
    guard.transact(dict(kind='terminal',job='a',status='INCOMPLETE'))
    state=Lifecycle(tmp_path/'events',app_id='app',execution_s=1,guard=guard,job='a')
    state.poll(Provider())
    assert state.account['experiment_stopped'] and not state.account['cap_exceeded']
    assert state.stop_reason=='terminal incomplete dispatch'
    assert state.last_meter==0

def test_sustained_contention_accrues_and_recovers_without_terminal(tmp_path,monkeypatch):
    clock=[0.];jobs={'score:1':100,'score:2':100}
    plan={n:dispatch_seconds(d)*RATE for n,d in jobs.items()}
    guard=SpendGuard(tmp_path/'ledger',plan,overhead=0)
    for n in jobs:guard.reserve(n,plan[n]);guard.start(n,'call','app')
    state=Lifecycle(tmp_path/'events',app_id='app',execution_s=100,guard=guard,job='score:1',
                    jobs=jobs,max_containers=2,started=0,now=lambda:clock[0],setup=True)
    state.body_started=0.;provider=Provider();provider.tasks=['one','two']
    with guard.path.open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX)
        monkeypatch.setattr('lifecycle8086.LOCK_RETRY_S',.01)
        for at in (0,10,30):clock[0]=at;state.poll(provider)
        assert state.accounting_pending and state.elapsed_usd>=2*30*RATE
        assert not state.incomplete and provider.stops==0
        assert not any(json.loads(r)['kind']=='terminal' for r in guard.path.read_text().splitlines())
    clock[0]=31;state.poll(provider)
    assert not state.accounting_pending and not state.incomplete
    assert state.accounted_meter==0 and state.elapsed_usd>=2*31*RATE

def test_real_cap_still_stops_during_ledger_contention(tmp_path,monkeypatch):
    clock=[0.];jobs={'score:1':100,'score:2':100}
    plan={n:dispatch_seconds(d)*RATE for n,d in jobs.items()};cap=sum(plan.values())+1e-7
    guard=SpendGuard(tmp_path/'ledger',plan,cap=cap,overhead=0)
    for n in jobs:guard.reserve(n,plan[n]);guard.start(n,'call','app')
    state=Lifecycle(tmp_path/'events',app_id='app',execution_s=100,guard=guard,job='score:1',
                    jobs=jobs,max_containers=2,started=0,now=lambda:clock[0],setup=True)
    state.body_started=0.;provider=Provider();provider.tasks=['one','two'];state.poll(provider)
    with guard.path.open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX);monkeypatch.setattr('lifecycle8086.LOCK_RETRY_S',.01)
        clock[0]=600;state.poll(provider)
    assert state.incomplete and state.account['cap_exceeded'] and state.accounting_pending
    assert state.stop_reason=='complete spend projection exceeds cap'

def test_physically_stopped_app_waiting_for_lock_never_becomes_a_deadline_stop(tmp_path,monkeypatch):
    clock=[0.];guard=SpendGuard(tmp_path/'ledger',{'a':1.},overhead=0)
    guard.reserve('a',1.);guard.start('a','call','app')
    state=Lifecycle(tmp_path/'events',app_id='app',execution_s=1,guard=guard,job='a',started=0,now=lambda:clock[0])
    provider=Provider();provider.stopped=True
    with guard.path.open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX);monkeypatch.setattr('lifecycle8086.LOCK_RETRY_S',.01)
        state.poll(provider,finish=True)
        assert state.confirmed_at==0 and not state.confirmed and state.accounting_pending
        clock[0]=1000;state.poll(provider,finish=True)
        assert not state.incomplete and not state.unknown_bound and state.stop_reason=='normal_finish'
    clock[0]=1001;state.poll(provider,finish=True)
    assert state.confirmed and not state.incomplete

def test_operational_refreeze_accepts_only_exact_completed_world_bytes(tmp_path,monkeypatch):
    import lifecycle8086 as L
    from s8086 import sha
    world=tmp_path/'world.jsonl';world.write_bytes(b'completed old world')
    prior='a'*64
    manifest={'completed_world_inputs':{'world':{'freeze_sha256':prior,'input_sha256':sha(world.read_bytes())}}}
    (tmp_path/'freeze.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(L,'PRE',tmp_path)
    assert L.valid_world_freeze(world,prior)
    assert not L.valid_world_freeze(world,'b'*64)
    world.write_bytes(b'changed world')
    assert not L.valid_world_freeze(world,prior)
    other=tmp_path/'other.jsonl';other.write_bytes(b'completed old world')
    assert not L.valid_world_freeze(other,prior)

def test_unknown_physical_bound_reports_its_cause_without_a_cap_label(tmp_path):
    guard=SpendGuard(tmp_path/'ledger',{'a':1.},overhead=0)
    guard.reserve('a',1.);guard.start('a','call','app')
    guard.transact(dict(kind='uncertain',job='a',reason='termination unconfirmed'))
    state=Lifecycle(tmp_path/'events',app_id='app',execution_s=1,guard=guard,job='a')
    state.poll(Provider())
    assert state.account['unknown_physical_liabilities']==['a'] and not state.account['cap_exceeded']
    assert state.stop_reason=='unknown physical liabilities'

def test_dispatch_waits_for_lock_recovery_past_rpc_allowance(monkeypatch):
    import lifecycle8086 as L
    from types import SimpleNamespace
    clock=[0.];monitor=object.__new__(L.WatchdogProcess)
    monitor.config={'started_monotonic':0.,'execution_s':100,'job':'score:1','jobs':{'score:1':100}}
    monitor.process=SimpleNamespace(poll=lambda:None)
    reads=[]
    def status():
        reads.append(clock[0]);clock[0]=40.
        return dict(dispatch_permitted=len(reads)>1,heartbeat=clock[0],stop_required=False)
    monitor.confirmation=status
    monkeypatch.setattr(L.time,'monotonic',lambda:clock[0]);monkeypatch.setattr(L.time,'sleep',lambda _:None)
    monitor.permit()
    assert len(reads)==2

def test_host_waits_for_stopped_app_final_lock_recovery(monkeypatch):
    import lifecycle8086 as L
    clock=[0.];monitor=object.__new__(L.WatchdogProcess);reads=[]
    def status():
        reads.append(clock[0]);clock[0]=65.
        return dict(confirmed=len(reads)>1,confirmed_at=1.,accounting_pending=len(reads)==1,
                    within_envelope=True,incomplete=False)
    monitor.confirmation=status
    monkeypatch.setattr(L.time,'monotonic',lambda:clock[0]);monkeypatch.setattr(L.time,'sleep',lambda _:None)
    assert monitor.wait_confirmed()['confirmed'] and len(reads)==2

def test_first_contended_poll_includes_carried_plan_extensions(tmp_path,monkeypatch):
    clock=[0.];jobs={'score:1':100};plan={n:dispatch_seconds(d)*RATE for n,d in jobs.items()}
    guard=SpendGuard(tmp_path/'ledger',plan,overhead=0)
    guard.transact(dict(kind='plan_extension',jobs={'carry:old':94.8},reason='historical charged allocation'))
    guard.reserve('carry:old',94.8);guard.start('carry:old','old-call','old-app')
    guard.transact(dict(kind='liability',job='carry:old',app_id='old-app',usd=94.8,elapsed_usd=94.8))
    guard.reserve('score:1',plan['score:1']);guard.start('score:1','call','app')
    initial=guard.transact(dict(kind='check'))
    state=Lifecycle(tmp_path/'events',app_id='app',execution_s=100,guard=guard,job='score:1',
                    jobs=jobs,started=0,now=lambda:clock[0],setup=True,initial_account=initial)
    state.body_started=0.;provider=Provider();provider.tasks=['one']
    with guard.path.open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX);monkeypatch.setattr('lifecycle8086.LOCK_RETRY_S',.01)
        state.poll(provider)
        assert state.account['projected_liabilities_usd']>=94.8+plan['score:1']
        assert state.accounting_pending and not state.incomplete
        clock[0]=1000;state.poll(provider)
    assert state.incomplete and state.account['cap_exceeded']
    assert state.stop_reason=='complete spend projection exceeds cap'

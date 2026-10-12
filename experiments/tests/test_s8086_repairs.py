"""Prelaunch defect regressions; synthetic inputs, no population or paid calls."""
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
from algocell_exp.census8086_trace import mechanism
import law8086 as Law
import s8086 as S
from dispatch8086 import SpendGuard


def trace_instruction(step, pc, ins, reads=(), writes=()):
    prefix = 0
    while ins[prefix] in (0xf2, 0xf3): prefix += 1
    return [(step,0,pc+i,v,0) for i,v in enumerate(ins)] + [(step,4,pc,ins[0],0),(step,3,pc+prefix+1,ins[prefix],0)] + [(step,1,p,v,0) for p,v in reads] + [(step,2,p,v,0) for p,v in writes]


def test_observer_uses_hook_start_argument(monkeypatch):
    from algocell_exp.census8086_trace import DiagnosticMachine
    from algocell_exp.zmachine import ZMachine
    from types import SimpleNamespace
    src = (S.HERE/'algocell_exp/machines/i8086_pair_maps_L16.wgsl').read_text()
    manifest = json.loads((S.HERE/'algocell_exp/machines/i8086_pair_maps_L16.manifest.json').read_text())
    def capture(self, src, manifest, **kw): self.wgsl=src
    monkeypatch.setattr(ZMachine,'__init__',capture)
    base=SimpleNamespace(m=manifest,wgsl=src,record_words=manifest['record']['wordsPerLane'],name='test',dev=None,
                         segment=lambda _: {'bytes':16},ring_bytes=32)
    obs=DiagnosticMachine(base,events=32)
    hook=obs.wgsl.split('fn on_instruction_start(')[1].split('}')[0]
    assert 'census_event(4u, pc, unit)' in hook


def test_true_start_rep_boundary():
    rows=trace_instruction(0,0,[0xbf,16,0])+trace_instruction(1,3,[0xb9,2,0])
    rows+=trace_instruction(2,6,[0xf3,0xa5],[(0,0xbf),(1,16)],[(16,0xbf),(17,16)])
    rows += [(3,4,6,0xf3,0),(3,3,6,0xa5,0),(3,1,2,0,0),(3,1,3,0xb9,0),(3,2,18,0,0),(3,2,19,0xb9,0)]
    assert mechanism(np.asarray(rows,np.uint32),16)[0]=='confined copy loop'


def test_push_sp_arithmetic_is_not_copy():
    rows=trace_instruction(0,0,[0x8b,0x26,14,0],[(14,0x14),(15,0x12)])
    rows+=trace_instruction(1,4,[0x54],writes=[(18,0x12),(19,0x12)])
    assert mechanism(np.asarray(rows,np.uint32),16)[0]=='other'


def test_byte_pairing_rejects_swapped_equal_members():
    rows=trace_instruction(0,0,[0x8b,6,14,0],[(14,0x14),(15,0x12)])
    rows+=trace_instruction(1,4,[0x50],writes=[(30,0x12),(31,0x14)])
    assert mechanism(np.asarray(rows,np.uint32),16)[0]=='other'


@pytest.mark.parametrize('reg,ins,reads',[(4,[7],[(0,1),(1,2)]),(6,[0xac],[(0,1)]),(7,[0xaa],[]),(1,[0xe2,0],[])])
def test_implicit_arithmetic_invalidates_memory_pair(reg,ins,reads):
    rows=trace_instruction(0,0,[0x8b,(reg<<3)|6,14,0],[(14,0x14),(15,0x12)])
    rows+=trace_instruction(1,4,ins,reads)
    rows+=trace_instruction(2,6,[0x50+reg],writes=[(18,0x14),(19,0x12)])
    assert mechanism(np.asarray(rows,np.uint32),16)[0]=='other'


def shifted_certificate():
    cert=S.read_json(S.PRE/'certificate_L2_source_counterexample.json.gz')
    return cert


def test_shifted_source_amendment():
    cert=shifted_certificate(); p=np.frombuffer(bytes.fromhex(cert['parent']),np.uint8)
    g=Law.copy_certificate(p,np.asarray(cert['ordered_traces'][0],np.uint32),32)
    assert g['R']==list(range(12,28)) and g['K']==list(range(9))
    assert g['C_R']==list(range(16,25)) and g['free']==[12,13,14,15,25,26,27]
    assert g['erasure_controls']==[9,10,11,28,29,30,31] and g['held_erased']==list(range(9))
    assert g['phase']==4
    variants=list(Law.variants(p,g,'B',S.SEEDS[0]))
    for case,kind,x,expected in variants:
        assert np.array_equal(expected,x[g['source_map']])
        assert np.array_equal(x[g['K']],p[g['K']]) and np.array_equal(x[g['C_R']],p[g['C_R']])
        assert Law.check_descendant(expected,expected,g['source_map'],g) is None


def test_new_meter_enforces_current_transaction(tmp_path):
    guard=SpendGuard(tmp_path/'ledger.jsonl',{'a':59},overhead=0)
    r=guard.transact({'kind':'metered','app_id':'app','usd':60})
    assert r['projected_liabilities_usd']==119 and r['stop_required']


def test_lifecycle_watchdog_stops_blocked_dispatcher_and_keeps_unknown(tmp_path):
    from lifecycle8086 import Lifecycle, envelope_seconds
    clock=[0.0]
    class Mock:
        def __init__(self): self.stops=0; self.calls=[]; self.tasks=['attempt-1']; self.confirm=False
        def sample(self,app): return {'stopped':self.confirm,'tasks':self.tasks,'metered_usd':0.0}
        def cancel(self,call): self.calls.append(call)
        def stop(self,app): self.stops+=1
    mock=Mock(); state=Lifecycle(tmp_path/'life.jsonl',app_id='ap-mock',call_id='fc-mock',
                               execution_s=100,now=lambda:clock[0],started=0)
    state.poll(mock); clock[0]=20; mock.tasks=['attempt-2']; state.poll(mock)
    clock[0]=state.stop_at; state.poll(mock)
    assert mock.stops==1 and mock.calls==['fc-mock'] and not state.confirmed
    assert state.reservation_seconds==envelope_seconds(100)
    assert state.liability_retained
    clock[0]+=2; mock.confirm=True; mock.tasks=[]; state.poll(mock)
    assert state.confirmed and not state.liability_retained
    assert state.confirmed_at<state.expires_at
    events=[json.loads(x) for x in state.path.read_text().splitlines()]
    assert any(e['kind']=='attempt_observed' and e['task']=='attempt-2' for e in events)


def test_lifecycle_no_renewal_on_restart_and_meter_stop(tmp_path):
    from lifecycle8086 import Lifecycle
    class Mock:
        def sample(self,app): return {'stopped':False,'tasks':['restart'],'metered_usd':60}
        def cancel(self,call): pass
        def stop(self,app): self.stopped=True
    mock=Mock(); guard=SpendGuard(tmp_path/'ledger.jsonl',{'a':59},overhead=0)
    state=Lifecycle(tmp_path/'life.jsonl',app_id='ap-mock',execution_s=100,started=0,now=lambda:10,
                    guard=guard,job='a')
    expiry=state.expires_at; state.poll(mock)
    assert mock.stopped and state.expires_at==expiry and state.liability_retained


def test_settlement_requires_confirmed_stopping(tmp_path):
    guard=SpendGuard(tmp_path/'ledger.jsonl',{'a':1},overhead=0)
    guard.reserve('a',1); guard.start('a','call','app')
    with pytest.raises(ValueError,match='stopping'): guard.settle('a',.5)
    guard.transact({'kind':'stopped','job':'a','app_id':'app','confirmed':True,'tasks_empty':True})
    guard.settle('a',.5)


@pytest.mark.parametrize('horizon,arms,n,k',[(3000000,('A','B'),585,5),(3000000,('A',),585,5),
                                            (1000000,('A','B'),505,4),(1500000,('A','B'),525,5),(1500000,('A',),525,5)])
def test_parameterized_projection_and_schedule(horizon,arms,n,k):
    from benchmark8086 import project
    reg=S.Registration(horizon,arms)
    result=project(S.read_json(S.PRE/'benchmark.json'),.7999038189802217,registration=reg)
    assert result['registration']==reg.as_dict()
    assert len(reg.sample_steps)==n and len(reg.snapshots)==k
    assert reg.persistence==tuple(range(horizon-250000,horizon+1,25000))
    assert len(reg.persistence)==11 and set(reg.persistence)<=set(reg.sample_steps)
    assert result['population_worlds']==20*len(arms)
    assert result['L1_total_evolved_encounters_cap']==40960000*len(arms)


def test_independent_process_runs_while_dispatcher_is_blocked(tmp_path):
    import os,subprocess,sys,time
    # Separate session, same policy class, mock provider only. The parent does
    # no dispatch/poll work while blocked waiting for the mock call's signal.
    script=tmp_path/'mock_watchdog.py'
    script.write_text('''import os,sys,time
from pathlib import Path
from lifecycle8086 import Lifecycle
root=Path(sys.argv[1])
class Mock:
    stopped=False
    def sample(self,app): return {'stopped':self.stopped,'tasks':[] if self.stopped else ['attempt-1']}
    def cancel(self,call): (root/'cancelled').write_text(call)
    def stop(self,app):
        (root/'stopped').write_text(str(os.getpgrp())); self.stopped=True
provider=Mock()
state=Lifecycle(root/'events.jsonl',app_id='mock',call_id='call',execution_s=1,
                started=time.monotonic()-240.8)
while not state.confirmed:
    state.poll(provider); time.sleep(.01)
''')
    process=subprocess.Popen([sys.executable,str(script),str(tmp_path)],start_new_session=True)
    try:
        process.wait(timeout=3)  # Stand-in for blocking call.get; child must act.
        assert process.returncode==0
        assert (tmp_path/'cancelled').read_text()=='call'
        assert int((tmp_path/'stopped').read_text())!=os.getpgrp()
        records=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
        assert records[-1]['kind']=='stopped_confirmed' and records[-1]['within_envelope']
    finally:
        if process.poll() is None: process.kill(); process.wait()


def test_dispatch_failure_persists_original_and_does_not_settle(tmp_path):
    from contextlib import contextmanager
    from dispatch8086 import bounded_call
    from lifecycle8086 import envelope_seconds
    events=[]
    class App:
        app_id='ap-mock'
        @contextmanager
        def run(self): yield
    class Call:
        object_id='fc-mock'
        def get(self): raise RuntimeError('original device failure')
    class Function:
        def spawn(self,*args): events.append('spawn'); return Call()
    class Monitor:
        def __init__(self,guard,job,app,execution):
            self.guard,self.job,self.app=guard,job,app; events.append('armed')
        def bind(self,call): events.append('bind')
        def finish(self): events.append('finish')
        def wait_confirmed(self):
            events.append('confirmed')
            self.guard.transact({'kind':'stopped','job':self.job,'app_id':self.app,'confirmed':True,'tasks_empty':True})
    guard=SpendGuard(tmp_path/'ledger',{'a':envelope_seconds(1)*S.RATE},overhead=0)
    with pytest.raises(RuntimeError,match='original device failure'):
        bounded_call(App(),Function(),(),guard,'a',1,watchdog_factory=Monitor)
    assert events==['armed','spawn','bind','finish','confirmed']
    ledger=[json.loads(row) for row in guard.path.read_text().splitlines()]
    terminal=next(row for row in ledger if row['kind']=='terminal')
    assert terminal['call_id']=='fc-mock' and terminal['last_acknowledged_progress']['phase']=='spawn_acknowledged'
    assert not any(row['kind']=='settled' for row in ledger)


def test_stop_state_with_live_tasks_cannot_release_liability(tmp_path):
    from lifecycle8086 import Lifecycle
    class Mock:
        def sample(self,app): return {'stopped':True,'tasks':['still-billed']}
        def stop(self,app): pass
    state=Lifecycle(tmp_path/'life.jsonl',app_id='mock',execution_s=1,started=0,now=lambda:1000)
    assert not state.poll(Mock()) and state.liability_retained
    assert any(json.loads(x)['kind']=='unconfirmed_overrun' for x in state.path.read_text().splitlines())


def test_shortened_record_requires_new_horizon_and_exact_r2b_subset():
    import score8086 as Score
    reg=S.Registration(1000000,('A',))
    def judge(tape): return {'gen2':1.,'heritable':True,'confined':False,'open_memory_fed':True}
    rows=[{'step':s,'top3':[('90',1.)]} for s in reg.sample_steps]
    result=Score.endpoints(rows,judge,reg)
    assert result['persistent_open_memory'] and len(result['persistence_samples'])==11
    # 750k..1M has 101 scheduled samples; only 11 exact 25k points matter.
    assert sum(750000<=s<=1000000 for s in reg.sample_steps)==101
    for row in rows:
        if row['step']==752500: row['top3']=[]
    assert Score.endpoints(rows,judge,reg)['persistent_open_memory']
    for row in rows:
        if row['step']==775000: row['top3']=[]
    assert not Score.endpoints(rows,judge,reg)['persistent_open_memory']


def test_rep_decrement_invalidates_loaded_cx_before_push():
    rows=trace_instruction(0,0,[0x8b,0x0e,14,0],[(14,1),(15,0)])
    rows+=trace_instruction(1,4,[0xf3,0xa4],[(0,0x8b)],[(16,0x8b)])
    rows+=trace_instruction(2,6,[0x51],writes=[(30,0),(31,0)])
    assert mechanism(np.asarray(rows,np.uint32),16)[0]=='other'


def test_startup_and_container_limits_are_explicit():
    import ast
    for filename in ('modal_s8086.py','modal_score8086.py'):
        tree=ast.parse((S.HERE/filename).read_text())
        decorators=[d for node in tree.body if isinstance(node,ast.FunctionDef) for d in node.decorator_list
                    if isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute) and d.func.attr=='function']
        assert decorators
        for d in decorators:
            keys={kw.arg:kw.value for kw in d.keywords}
            assert isinstance(keys['startup_timeout'],ast.Name) and keys['startup_timeout'].id=='STARTUP_TIMEOUT'
            assert all(ast.literal_eval(keys[k])==v for k,v in [('max_containers',1),('min_containers',0),('buffer_containers',0),('retries',0)])


def test_malformed_amendment_never_reduces_raw_dependencies():
    cert=shifted_certificate(); p=np.frombuffer(bytes.fromhex(cert['parent']),np.uint8)
    # Raw core coordinate >= d is rejected, even when its residue is small.
    log=np.asarray(cert['ordered_traces'][0],np.uint32).copy()
    first_fetch=np.flatnonzero(log[:,1]==0)[0]; log[first_fetch,2]=16
    with pytest.raises(ValueError,match='raw_core_exceeds'):
        Law.copy_certificate(p,log,32)


def test_runner_initialization_failure_is_terminal_without_gpu_read(tmp_path,monkeypatch):
    import run8086 as Run
    monkeypatch.setattr(Run,'launch_receipt',lambda _: {})
    monkeypatch.setattr(Run,'released_registration',lambda _: S.Registration(1500000,('A',)))
    class Fault:
        def __init__(self,*args,**kwargs): raise RuntimeError('initialization failed')
    monkeypatch.setattr(Run,'S2Soup',Fault)
    with pytest.raises(RuntimeError,match='initialization failed'):
        Run.run_world('A',S.SEEDS[0],tmp_path,release='mock',deadline_s=100,git_commit='mock')
    rows=[json.loads(x) for x in next(tmp_path.glob('*.jsonl')).read_text().splitlines()]
    assert rows[-1]['kind']=='incomplete' and rows[-1]['steps_run']==0


def test_unknown_physical_bound_blocks_new_work_and_settlement(tmp_path):
    guard=SpendGuard(tmp_path/'ledger',{'a':1,'b':1},overhead=0)
    guard.reserve('a',1); guard.start('a','call','app')
    result=guard.transact({'kind':'uncertain','job':'a','reason':'termination unconfirmed past envelope'})
    assert result['stop_required'] and result['unknown_physical_liabilities']==['a']
    with pytest.raises(RuntimeError,match='unknown physical liabilities'): guard.reserve('b',1)
    guard.transact({'kind':'stopped','job':'a','app_id':'app','confirmed':True,'tasks_empty':True})
    with pytest.raises(ValueError,match='bound unresolved'): guard.settle('a',1)

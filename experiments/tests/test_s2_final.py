"""Final S2 prospective/descriptive and operational regression checks."""
import json
import time
from pathlib import Path
import pytest
from dispatch8086 import SpendGuard
from lifecycle8086 import Lifecycle, BOOKKEEP_TIMEOUT_S, FULL_WORLD_RECOVERIES, dispatch_seconds
from s8086 import PRE, CAP, RATE, read_json, sha
from test_b5_enforcement import Provider

@pytest.mark.parametrize('finish_first',[False,True])
def test_successful_completion_without_host_ack_never_reserves_restart(tmp_path,finish_first):
    p=read_json(PRE/'projection.json');job='world:A:8086001';deadline=p['execution_deadline_s'][job]
    g=SpendGuard(tmp_path/'ledger',p['job_liabilities_usd'],overhead=p['storage_termination_reserve_usd'])
    g.reserve(job,g.plan[job]);g.start(job,'call','app');clock=[120.]
    s=Lifecycle(tmp_path/'events',app_id='app',call_id='call',execution_s=deadline,guard=g,job=job,started=0,now=lambda:clock[0],setup=True);s.body_started=120.
    provider=Provider();provider.tasks=['task'];provider.events=[{'attempt':'attempt','task':'task','state':'live','elapsed_s':0}]
    s.poll(provider);clock[0]+=deadline
    provider.tasks=[];provider.events=[{'attempt':'attempt','task':'task','state':'completed','elapsed_s':deadline}]
    s.poll(provider,finish=finish_first)
    assert s.work_complete and not s.incomplete and not s.account['stop_required']
    assert s.charged_usd==(120+deadline+121)*RATE
    provider.stopped=True;s.poll(provider,finish=True)
    assert s.confirmed and not s.incomplete


def test_late_world_preemption_has_time_for_full_recovery(tmp_path):
    p=read_json(PRE/'projection.json');job='world:A:8086001';deadline=p['execution_deadline_s'][job]
    g=SpendGuard(tmp_path/'ledger',p['job_liabilities_usd'],overhead=p['storage_termination_reserve_usd']);g.reserve(job,g.plan[job]);g.start(job,'call','app')
    clock=[120.];s=Lifecycle(tmp_path/'events',app_id='app',call_id='call',execution_s=deadline,guard=g,job=job,started=0,now=lambda:clock[0],setup=True);s.body_started=120.
    provider=Provider();provider.tasks=['old'];provider.events=[{'attempt':'old','task':'old','state':'live','elapsed_s':0}];s.poll(provider)
    clock[0]+=deadline;provider.tasks=[];provider.events=[{'attempt':'old','task':'old','state':'preempted','elapsed_s':deadline}];s.poll(provider)
    assert not s.work_complete and not s.incomplete
    expiry=s.expires_at;provider.tasks=['new'];provider.events=[{'attempt':'new','task':'new','state':'live','elapsed_s':0}];s.poll(provider)
    clock[0]+=deadline;provider.tasks=[];provider.events=[{'attempt':'new','task':'new','state':'completed','elapsed_s':deadline}];s.poll(provider)
    assert not s.incomplete and s.expires_at==expiry and clock[0]<s.stop_at
    assert s.recovery_seconds==FULL_WORLD_RECOVERIES*dispatch_seconds(deadline)


def test_authenticated_incremental_replay_at_full_experiment_scale(tmp_path):
    p=read_json(PRE/'projection.json');job='world:A:8086001';g=SpendGuard(tmp_path/'ledger',p['job_liabilities_usd'],overhead=p['storage_termination_reserve_usd']);g.reserve(job,g.plan[job]);g.start(job,'call','app')
    previous=json.loads(g.path.read_text().splitlines()[-1])['hash']
    # Cap/rate bounds about 375,701 charged resource seconds. Two ledger
    # rows per poll: 750,000 rows cover worlds, scoring and recoveries.
    with g.path.open('a') as f:
        for i in range(750_000):
            row={'previous':previous,'time':i,'kind':'liability','job':job,'app_id':'app','usd':g.plan[job],'elapsed_usd':0,'attempts':1}
            previous=sha(json.dumps(row,sort_keys=True,separators=(',',':')).encode());f.write(json.dumps({**row,'hash':previous})+'\n')
    clock=[120.];s=Lifecycle(tmp_path/'events',app_id='app',execution_s=p['execution_deadline_s'][job],guard=g,job=job,started=0,now=lambda:clock[0],setup=True);s.body_started=120.
    provider=Provider();provider.tasks=['task'];provider.events=[{'attempt':'task','task':'task','state':'live','elapsed_s':8000}]
    start=time.monotonic();s.poll(provider);cold=time.monotonic()-start
    assert cold<BOOKKEEP_TIMEOUT_S and not s.incomplete and not provider.stops
    start=time.monotonic();s.poll(provider);incremental=time.monotonic()-start
    assert incremental<BOOKKEEP_TIMEOUT_S and not s.incomplete and not provider.stops
    assert incremental<cold
    (PRE/'ledger_scale_validation.json').write_text(json.dumps({'date':'2026-10-12',
        'history_rows':750_000,'ledger_bytes':g.path.stat().st_size,
        'cold_reconciliation_s':cold,'incremental_poll_s':incremental,
        'watchdog_timeout_s':BOOKKEEP_TIMEOUT_S,'false_stops':0},indent=2)+'\n')
    # Authenticated prefix checking survives cache reuse and rejects mutation.
    with g.path.open('r+') as f:
        f.seek(100);c=f.read(1);f.seek(100);f.write('X' if c!='X' else 'Y')
    with pytest.raises(ValueError,match='hash chain'):g.transact({'kind':'check'})


def test_randomized_spend_system_invariants():
    from spend_sim8086 import simulate
    result=simulate(1000)
    assert result['sequences']==1000 and result['violations']==0
    assert result['outcomes']['COMPLETE'] and result['outcomes']['INCOMPLETE']
    assert result['completion_before_finish'] and result['finish_before_completion']
    assert result['max_charged_usd']<=CAP and result['max_actual_usd']<=CAP


def test_descriptive_r2_has_no_boolean_verdict_even_above_old_thresholds():
    from test_s8086 import judgement
    from score8086 import arm_verdict
    from s8086 import SEEDS
    from s8086 import HORIZON
    rows=[{'world':{'arm':'A','seed':seed,'L':16,'horizon':HORIZON,'scoreable':True,'t_rep':0,'t_closed':None,'first':judgement(open=True),'open_memory_occurrence':True,'persistent_open_memory':True}} for seed in SEEDS]
    for r in rows:
        r['world']['open_memory_occurrence']=True;r['world']['persistent_open_memory']=True
    result=arm_verdict(rows,'A')
    assert result['R0'] and result['R1'] and not result['kill']
    assert result['occurrence_worlds']==20 and result['persistence_worlds']==20
    assert result['R2a'] is result['R2b'] is result['R2_joint'] is None
    assert result['R2_status'].startswith('descriptive')

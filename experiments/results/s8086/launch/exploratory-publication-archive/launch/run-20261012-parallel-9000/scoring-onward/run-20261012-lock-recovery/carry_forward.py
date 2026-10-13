"""Reconcile stopped physical allocations into a new ledger; never clear the old ledger."""
from collections import defaultdict
import json
from pathlib import Path
from dispatch8086 import SpendGuard
from s8086 import PRE, read_json, sha

RUN=Path(__file__).resolve().parent
projection=read_json(PRE/'projection.json')
prior=read_json(RUN/'prior_ledger_replay.json')
states=read_json(RUN/'historical_app_states.json')
if any(s['state']!='APP_STATE_STOPPED' or s['tasks'] for s in states):
    raise RuntimeError('historical physical allocations unresolved; STOP')
if sum(n.startswith('world:') and j['status']=='settled' for n,j in prior['jobs'].items())!=20:
    raise RuntimeError('20 completed settled worlds required; STOP')
ledger=RUN/'spend.jsonl'
if ledger.exists():raise FileExistsError('carry-forward ledger already exists; no replay retry')
guard=SpendGuard(ledger,projection['job_liabilities_usd'],overhead=projection['storage_termination_reserve_usd'])
carried={};allocations=[]
for name,job in prior['jobs'].items():
    if name=='preparation':target=name
    elif name=='prebuild':target='carry:completed-prebuild:prebuild'
    elif name.startswith('score:'):target='carry:stopped-scoring:'+name
    else:target=name
    bound=max(job['reservation'],job.get('liability',0),job.get('cost',0))
    if target not in guard.plan:carried[target]=bound
    allocations.append((target,name,job,bound))
guard.transact(dict(kind='plan_extension',jobs=carried,reason='explicit scoring-onward release; retain all original charged allocations',
                    source_ledger_sha256=sha((RUN/'prior_spend.jsonl').read_bytes())))
for target,name,job,bound in allocations:
    app=job.get('app','historical-aggregate-repartition')
    guard.reserve(target,bound,historical_source_job=name)
    guard.start(target,job.get('call','historical:'+name),app)
    guard.transact(dict(kind='liability',job=target,app_id=app,usd=bound,elapsed_usd=bound,
                        accounting='stopped historical full allocation retained; app billing overlaps once'))
    guard.transact(dict(kind='stopped',job=target,app_id=app,confirmed=True,tasks_empty=True,
                        physical_confirmation_sha256=sha((RUN/'historical_app_states.json').read_bytes())))
    if job['status']=='settled':guard.settle(target,job['cost'],historical=True)
meters=dict(prior['meters'])
snapshot=read_json(RUN/'billing_snapshot.json')
for app,value in snapshot.get('by_app',{}).items():meters[app]=max(meters.get(app,0),value)
for app,value in meters.items():guard.transact(dict(kind='metered',app_id=app,usd=value))
result=guard.transact(dict(kind='check',reason='all historical allocations plus fresh prebuild/scoring and remaining downstream work'))
if result['stop_required'] or result['projected_liabilities_usd']>95:
    raise RuntimeError(f'carry-forward forecast rejects relaunch; STOP: {result}')
report=dict(prior_ledger=str(RUN.parents[1]/'spend.jsonl'),prior_ledger_sha256=sha((RUN/'prior_spend.jsonl').read_bytes()),
            old_terminal_rows_preserved=True,carried_allocations=[dict(job=n,source_job=old,usd=b,status=j['status'],app_id=j.get('app')) for n,old,j,b in allocations],
            carried_full_bounds_usd=sum(b for _,_,_,b in allocations),metered_usd=sum(meters.values()),
            billing_may_lag=True,conservative_recorded_lifetime_bound_usd=17.48499627932854,
            complete_forecast_usd=result['projected_liabilities_usd'],remaining_stages=['prebuild','score','extract','eligibility_commit_release','assay','controls'],
            worlds_completed=20,worlds_rerun=False,cap=95,shared_reserve_usd=projection['storage_termination_reserve_usd'],check=result)
(RUN/'carry_forward.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'metered_usd':report['metered_usd'],'complete_forecast_usd':report['complete_forecast_usd'],'cap':95,'worlds_completed':20,'worlds_rerun':False}))

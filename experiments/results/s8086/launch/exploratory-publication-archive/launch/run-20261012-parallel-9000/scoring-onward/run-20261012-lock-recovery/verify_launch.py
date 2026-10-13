"""Read-only detached-process, provider, budget, and queue forecast verification."""
import asyncio
from datetime import datetime,timezone,timedelta
import json
import math
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo
from modal.client import _Client
from modal_proto import api_pb2 as P
from dispatch8086 import SpendGuard
from s8086 import PRE, RATE, read_json, sha

RUN=Path(__file__).resolve().parent

async def main():
    pipeline=read_json(RUN/'pipeline.status.json')
    if pipeline['kind']=='pipeline_incomplete':raise RuntimeError('pipeline stopped; inspect status and logs')
    stage=pipeline.get('stage')
    if stage not in ('score','extract','assay','controls'):
        raise RuntimeError('scoring-onward dispatch has not started yet')
    pid=int((RUN/'pipeline.pid').read_text());child=int((RUN/f'{stage}.pid').read_text())
    kind='law' if stage=='assay' else stage
    job='controls' if stage=='controls' else f'{kind}:A:8086001'
    prefix=RUN/f'lifecycle-{sha(job.encode())[:16]}'
    control=read_json(str(prefix)+'.control.json');status=read_json(str(prefix)+'.status.json')
    if (status['incomplete'] or status['stop_required'] or not status['ready']
            or time.monotonic()-status['heartbeat']>31 or not status.get('app_id')):
        raise RuntimeError('watchdog not current and healthy')
    process_lines=subprocess.check_output(['ps','-axo','pid,ppid,pgid,etime,state,command'],text=True).splitlines()
    watches=[s for s in process_lines if '--watch '+str(prefix)+'.control.json' in s]
    if len(watches)!=1:raise RuntimeError('independent watchdog process missing or duplicated')
    watch=int(watches[0].split()[0])
    processes=[s for s in process_lines if s.split() and s.split()[0].isdigit() and int(s.split()[0]) in (pid,child,watch)]
    if len(processes)!=3:raise RuntimeError('pipeline or dispatcher process not live')
    pipe_fields=next(s.split() for s in processes if int(s.split()[0])==pid)
    watch_fields=watches[0].split()
    if pipe_fields[2]!=str(pid) or watch_fields[2]!=str(watch):raise RuntimeError('pipeline/watchdog must own independent sessions')
    client=await _Client.from_env();app=status['app_id']
    life=await asyncio.wait_for(client._stub.AppGetLifecycle(P.AppGetLifecycleRequest(app_id=app)),10)
    tasks=await asyncio.wait_for(client._stub.TaskList(P.TaskListRequest(app_id=app)),10)
    if life.lifecycle.app_state==P.APP_STATE_STOPPED or not tasks.tasks:
        raise RuntimeError('provider app has no live running tasks yet')
    layout=await asyncio.wait_for(client._stub.AppGetLayout(P.AppGetLayoutRequest(app_id=app)),10)
    identities=dict(layout.app_layout.function_ids)
    async def active_ids():
        found=set()
        async for batch in client._stub.AppGetLogs.unary_stream(P.AppGetLogsRequest(app_id=app,timeout=.1)):
            if batch.function_id:found.add(batch.function_id)
        return found
    for identity in await asyncio.wait_for(active_ids(),10):
        if identity not in identities.values():identities[identity]=identity
    functions=[]
    for tag,identity in identities.items():
        stats=await asyncio.wait_for(client._stub.FunctionGetCurrentStats(P.FunctionGetCurrentStatsRequest(function_id=identity)),10)
        definition=await asyncio.wait_for(client._stub.FunctionGetById(P.FunctionGetByIdRequest(function_id=identity)),10)
        functions.append(dict(tag=tag,function_id=identity,num_total_tasks=stats.num_total_tasks,
            num_running_inputs=stats.num_running_inputs,backlog=stats.backlog,
            server_timeout_s=definition.function.timeout_secs,max_containers=definition.function.autoscaler_settings.max_containers))
    plan=read_json(PRE/'projection.json')
    (RUN/'provider_function_inventory.json').write_text(json.dumps(functions,indent=2)+'\n')
    active_functions=[f for f in functions if f['num_running_inputs']>0]
    limit=1 if stage=='controls' else 20
    if not any(f['server_timeout_s']==plan['execution_deadline_s'][job] and f['max_containers']==limit
               for f in active_functions):
        raise RuntimeError('no processing inputs with the exact frozen server timeout/container limit yet')
    guard=SpendGuard(RUN/'spend.jsonl',plan['job_liabilities_usd'],overhead=plan['storage_termination_reserve_usd'])
    budget=guard.transact(dict(kind='check',reason='verified detached scoring-onward provider execution'))
    if budget['stop_required'] or budget['projected_liabilities_usd']>95:
        raise RuntimeError('complete forecast rejected; STOP')
    rows=[json.loads(s) for s in guard.path.read_text().splitlines()]
    submitted=[r for r in rows if r['kind']=='started' and r.get('app_id')==app and not r['job'].startswith('carry:')]
    if stage!='controls' and len(submitted)!=20:raise RuntimeError('cohort has not submitted all 20 inputs yet')
    slots=len(tasks.tasks);waves=math.ceil(20/slots)
    # Same frozen body/setup/startup/cleanup envelopes as the prior forecasts;
    # remove completed worlds, retain conservative diagnosis time.
    per_job={k:plan['job_liabilities_usd'][f'{k}:A:8086001']/RATE for k in ('score','extract','law')}
    remaining=[k for k in ('score','extract','law') if ('score','extract','law').index(k)>=('score','extract','law').index(kind)] if kind!='controls' else []
    seconds=waves*sum(per_job[k] for k in remaining)+(plan['job_liabilities_usd']['controls']+plan['job_liabilities_usd']['diagnosis'])/RATE
    now=datetime.now(timezone.utc)
    carry=read_json(RUN/'carry_forward.json')
    known_meters={}
    for row in rows:
        if row['kind']=='metered':known_meters[row['app_id']]=row['usd']
    report=dict(verified_utc=now.isoformat(),pipeline_pid=pid,dispatcher_pid=child,watchdog_pid=watch,
        stage=stage,app_id=app,app_state=P.AppState.Name(life.lifecycle.app_state),submitted_inputs=len(submitted),
        live_tasks=slots,task_ids=[t.task_id for t in tasks.tasks],functions=functions,processes=processes,watchdog_status=status,
        known_metered_spend_usd=sum(known_meters.values()),billing_may_lag=True,
        conservative_recorded_lifetime_bound_before_relaunch_usd=carry['conservative_recorded_lifetime_bound_usd'],
        complete_forecast=budget,cap=95,remaining_eta_hours=seconds/3600,
        eta_chicago=(now+timedelta(seconds=seconds)).astimezone(ZoneInfo('America/Chicago')).isoformat(),
        eta_assumption=f'{slots} observed containers persist; full frozen non-world envelopes and diagnosis allowance; extra queues/recoveries can extend',
        world_data_mutated=False,worlds_rerun=False,freeze_sha256=sha((PRE/'freeze.json').read_bytes()),
        paths=dict(run_directory=str(RUN),pipeline_log=str(RUN/'pipeline.log'),stage_log=str(RUN/f'{stage}.log'),
                   ledger=str(RUN/'spend.jsonl'),watchdog_status=str(prefix)+'.status.json',watchdog_events=str(prefix)+'.events.jsonl'))
    (RUN/'launch_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

asyncio.run(main())

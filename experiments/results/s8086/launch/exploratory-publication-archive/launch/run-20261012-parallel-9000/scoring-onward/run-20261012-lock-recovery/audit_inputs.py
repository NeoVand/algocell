"""Read-only completion/byte integrity and historical app/billing audit."""
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import modal
from modal.client import _Client
from modal_proto import api_pb2 as P
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.json_format import MessageToDict
from s8086 import DEFAULT_REGISTRATION as reg, SEEDS, sha
from run8086 import world_name

RUN=Path(__file__).resolve().parent
OLD=RUN.parents[1]

def save(name, value):
    (RUN/name).write_text(json.dumps(value,indent=2)+'\n')

async def main():
    client=await _Client.from_env()
    prior=json.loads((RUN/'prior_ledger_replay.json').read_text())
    apps=sorted(a for a in set(prior['meters'])|{j['app'] for j in prior['jobs'].values() if j.get('app')} if a.startswith('ap-'))
    states=[]
    for app in apps:
        life=await asyncio.wait_for(client._stub.AppGetLifecycle(P.AppGetLifecycleRequest(app_id=app)),10)
        tasks=await asyncio.wait_for(client._stub.TaskList(P.TaskListRequest(app_id=app)),10)
        states.append(dict(app_id=app,state=P.AppState.Name(life.lifecycle.app_state),tasks=[t.task_id for t in tasks.tasks]))
    save('historical_app_states.json',states)
    if any(s['state']!='APP_STATE_STOPPED' or s['tasks'] for s in states):
        raise RuntimeError('historical physical liabilities unresolved; STOP')
    start,end=Timestamp(),Timestamp()
    lo=1791792000
    hi=math.ceil(time.time()/3600)*3600
    start.FromSeconds(lo);end.FromSeconds(hi)
    request=P.WorkspaceBillingReportRequest(start_timestamp=start,end_timestamp=end,resolution='h',app_ids=apps)
    totals=defaultdict(float)
    try:
        async with asyncio.timeout(15):
            rows=[MessageToDict(r,preserving_proto_field_name=True) async for r in client._stub.WorkspaceBillingReport.unary_stream(request)]
        for row in rows:totals[row['object_id']]+=float(row['cost'])
        save('billing_snapshot.json',dict(apps=apps,rows=rows,by_app=dict(totals),metered_usd=sum(totals.values()),
             query_start_epoch=lo,query_end_epoch=hi,queried_utc=datetime.now(timezone.utc).isoformat(),billing_may_lag=True))
    except Exception as exc:
        save('billing_snapshot.json',dict(apps=apps,metered_usd=None,error=f'{type(exc).__name__}: {exc}',billing_may_lag=True))

def worlds_audit():
    volume=modal.Volume.from_name('algocell-atlas-runs')
    worlds=[]
    def raw(path):
        return b''.join(volume.read_file(path))
    for seed in SEEDS:
        name=world_name('A',seed);path=f'/s8086/A/{name}.jsonl'
        data=raw(path);records=[json.loads(s) for s in data.splitlines()]
        condition=records[0];summary=records[-1]
        if (condition['kind']!='condition' or condition['seed']!=seed or condition['arm']!='A'
                or condition['registration']!=reg.as_dict() or summary['kind']!='summary'
                or summary['steps_run']!=reg.horizon or any(r['kind']=='incomplete' for r in records)
                or [r['step'] for r in records if r['kind']=='sample']!=list(reg.sample_steps)):
            raise RuntimeError(f'world completion/registration integrity differs: {name}; STOP')
        snapshots=[r for r in records if r['kind']=='snapshot']
        if [r['step'] for r in snapshots]!=list(reg.snapshots):
            raise RuntimeError(f'world snapshot schedule differs: {name}; STOP')
        files=[]
        for row in snapshots:
            target=f"/s8086/A/{row['file']}";compressed=raw(target)
            digest=sha(compressed)
            if digest!=row['sha256_compressed'] or len(compressed)!=row['compressed_bytes']:
                raise RuntimeError(f'world snapshot bytes differ: {target}; STOP')
            files.append(dict(path=target,bytes=len(compressed),sha256=digest))
        worlds.append(dict(world=name,path=path,bytes=len(data),sha256=sha(data),snapshots=files,
                           completed=True,horizon=reg.horizon,freeze_sha256=condition['freeze_sha256']))
        print(name,'complete; five compressed snapshots verified',flush=True)
    outputs={}
    for stage in ('score','extract','assay'):
        try:entries=volume.listdir(f'/s8086_{stage}/A')
        except modal.exception.NotFoundError:entries=[]
        outputs[stage]=[dict(path=e.path,bytes=e.size) for e in entries]
    if any(outputs.values()):
        raise RuntimeError('existing downstream output requires integrity archive before exclusive relaunch; STOP')
    save('world_input_integrity.json',dict(worlds=worlds,worlds_completed=len(worlds),snapshots_verified=100,
         source_volume='algocell-atlas-runs',world_data_mutated=False,worlds_rerun=False,
         downstream_output_inventory=outputs,verified_utc=datetime.now(timezone.utc).isoformat()))
    print('20 completed worlds, 100 matching snapshots, downstream paths clear; historical apps stopped',flush=True)

asyncio.run(main())
worlds_audit()

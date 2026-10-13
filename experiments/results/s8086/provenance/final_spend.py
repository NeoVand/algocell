"""Read-only final provider reconciliation; run with the registered Python -I."""
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))  # trusted, frozen repository modules only
from dispatch8086 import SpendGuard
from s8086 import verify_freeze

OUT = Path(__file__).resolve().parent
RUN = HERE / 'results/s8086/launch/run-20261012-parallel-9000/scoring-onward/run-20261012-lock-recovery'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

async def main():
    verify_freeze()
    from modal.client import _Client
    from modal_proto import api_pb2 as P
    from google.protobuf.timestamp_pb2 import Timestamp
    from google.protobuf.json_format import MessageToDict
    ledger = RUN / 'spend.jsonl'
    original_digest = digest(ledger)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    apps = sorted({r['app_id'] for r in rows if r.get('app_id', '').startswith('ap-')})
    lo = math.floor(min(r['time'] for r in rows) / 86400) * 86400
    hi = math.ceil(time.time() / 3600) * 3600
    client = await _Client.from_env()
    start, end = Timestamp(), Timestamp()
    start.FromDatetime(datetime.fromtimestamp(lo, timezone.utc))
    end.FromDatetime(datetime.fromtimestamp(hi, timezone.utc))
    attempts = []
    for attempt in range(8):
        try:
            request = P.WorkspaceBillingReportRequest(start_timestamp=start, end_timestamp=end,
                                                     resolution='h', app_ids=apps)
            async with asyncio.timeout(30):
                billed = [MessageToDict(r, preserving_proto_field_name=True)
                          async for r in client._stub.WorkspaceBillingReport.unary_stream(request)]
            attempts.append({'attempt': attempt + 1, 'status': 'success', 'time': time.time()})
            break
        except Exception as exc:
            delay = min(1800, 120 * 2 ** attempt)
            attempts.append({'attempt': attempt + 1, 'status': 'error', 'error': f'{type(exc).__name__}: {exc}',
                             'time': time.time(), 'backoff_s': delay})
            (OUT / 'billing_attempts.json').write_text(json.dumps(attempts, indent=2) + '\n')
            print(f'Billing read failed; backoff {delay}s', flush=True)
            if attempt == 7:
                raise
            await asyncio.sleep(delay)
    by_app = {app: 0.0 for app in apps}
    for row in billed:
        assert row['object_id'] in by_app
        by_app[row['object_id']] += float(row['cost'])
    stopped = []
    for app in apps:
        life = await asyncio.wait_for(client._stub.AppGetLifecycle(P.AppGetLifecycleRequest(app_id=app)), 15)
        tasks = await asyncio.wait_for(client._stub.TaskList(P.TaskListRequest(app_id=app)), 15)
        record = {'app_id': app, 'app_state': P.AppState.Name(life.lifecycle.app_state),
                  'active_tasks': [t.task_id for t in tasks.tasks], 'metered_usd': by_app[app]}
        assert life.lifecycle.app_state == P.APP_STATE_STOPPED and not tasks.tasks, record
        stopped.append(record)
    # Replay a copy through the unchanged guard. Preserve the original ledger.
    copy = OUT / 'final_spend_ledger.jsonl'
    shutil.copyfile(ledger, copy)
    guard = SpendGuard(copy, rows[0]['plan'], cap=rows[0]['cap'], overhead=rows[0]['overhead'])
    before = guard.transact({'kind': 'check', 'accounting': 'final report replay of completed pipeline ledger'})
    for app in apps:
        guard.transact({'kind': 'metered', 'app_id': app, 'usd': by_app[app],
                        'accounting': 'fresh app-attributed hourly billing; confirmed stopped and no tasks; billing may lag'})
    after = guard.transact({'kind': 'check', 'accounting': 'final report reconciliation; no compute or dispatch'})
    result = {'queried_utc': datetime.now(timezone.utc).isoformat(), 'query_start_epoch': lo,
              'query_end_epoch': hi, 'apps': stopped, 'rows': billed, 'by_app': by_app,
              'metered_usd': sum(by_app.values()), 'billing_may_lag': True,
              'backoff_policy_s': {'initial': 120, 'maximum': 1800}, 'attempts': attempts,
              'original_ledger_sha256': original_digest, 'original_ledger_unchanged': original_digest == digest(ledger),
              'original_rows': len(rows), 'guard_before': before, 'guard_after': after,
              'shared_storage_termination_reserve_usd': rows[0]['overhead'],
              'conservative_bound_usd': after['projected_liabilities_usd'], 'cap_usd': rows[0]['cap'],
              'cap_headroom_usd': rows[0]['cap'] - after['projected_liabilities_usd'],
              'unallocated_plan_items': sorted(set(guard._replay['state'][2]) - set(guard._replay['state'][0])),
              'status_counts': dict(Counter(v['status'] for v in guard._replay['state'][0].values()))}
    result['retained_historical_running_statuses'] = [
        {'job': name, **value} for name, value in guard._replay['state'][0].items()
        if value['status'] != 'settled']
    (OUT / 'final_spend.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'billing_attempts.json').write_text(json.dumps(attempts, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('metered_usd', 'conservative_bound_usd', 'cap_usd', 'cap_headroom_usd', 'status_counts', 'unallocated_plan_items')}, indent=2), flush=True)

if __name__ == '__main__':
    asyncio.run(main())

"""Read app-attributed hourly billing; preserve cumulative append-only provenance."""
from __future__ import annotations
import asyncio
import json
import math
import fcntl
import threading
import time
from pathlib import Path
from datetime import datetime,timezone
from s8086 import PRE,write_new

POLL_INTERVAL_S=120
MAX_BACKOFF_S=1800

class AppMeterPoller:
    """One app lease, one nonblocking worker, and a workspace-wide RPC gate.

    Billing is cumulative per app. Unknown reads preserve the last successful
    reading; Lifecycle independently accrues actual task/container lifetimes.
    No billing failure supplies a numerical cost or a stop decision.
    """
    def __init__(self,app_id,started_epoch,directory,*,read=None,now=time.monotonic,wall=time.time):
        self.app_id=app_id;self.started_epoch=started_epoch;self.now=now;self.wall=wall
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.lease=(self.directory/(app_id+'.lock')).open('a+')
        try:fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.lease.close();raise
        self.read=read or (lambda:asyncio.run(billing(app_id,started_epoch)))
        self.next_poll=0.;self.failures=0;self.last_meter=None;self.error=None;self.reads=0
        self.worker=None;self.mutex=threading.Lock()
    def _gate(self,delay=None):
        path=self.directory/'workspace.json'
        with path.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
            raw=f.read();state=json.loads(raw) if raw else {}
            due=state.get('next_epoch',0.);now=self.wall()
            if delay is None and now<due:
                self.next_poll=self.now()+due-now;return False
            due=max(due,now+(POLL_INTERVAL_S if delay is None else delay))
            f.seek(0);f.truncate();json.dump({'next_epoch':due},f);f.flush()
            return True
    def snapshot(self):
        return {'metered_usd':self.last_meter,'billing_error':self.error,
                'billing_failures':self.failures,'billing_reads':self.reads,
                'billing_next_poll':self.next_poll}
    def refresh(self):
        # Also used synchronously by fault simulations with a fake clock.
        if not self.mutex.acquire(blocking=False):return self.snapshot()
        try:
            if self.now()<self.next_poll or not self._gate():return self.snapshot()
            self.reads+=1
            try:
                value=self.read()['metered_usd']
                if (not isinstance(value,(int,float)) or not math.isfinite(value) or value<0
                        or self.last_meter is not None and value<self.last_meter):
                    raise ValueError('unknown/invalid cumulative app billing')
                self.last_meter=value;self.failures=0;self.error=None
                self.next_poll=self.now()+POLL_INTERVAL_S
            except Exception as exc:
                self.failures+=1;self.error=f'{type(exc).__name__}: {exc}'
                delay=min(MAX_BACKOFF_S,POLL_INTERVAL_S*2**min(self.failures-1,8))
                self.next_poll=self.now()+delay;self._gate(delay)
            return self.snapshot()
        finally:self.mutex.release()
    def poll(self):
        if self.now()>=self.next_poll and (self.worker is None or not self.worker.is_alive()):
            self.worker=threading.Thread(target=self.refresh,daemon=True);self.worker.start()
        return self.snapshot()
    def close(self):
        if self.worker is not None:self.worker.join(timeout=16)
        self.lease.close()

async def billing(app_id,started_epoch):
    from modal_proto import api_pb2
    from google.protobuf.timestamp_pb2 import Timestamp
    from modal.client import _Client
    client=await _Client.from_env()
    start,end=Timestamp(),Timestamp()
    # Modal hourly billing excludes a partial end interval. Request the full
    # current hour, filtering this unique preparation app ID, and disclose lag.
    lo=math.floor(started_epoch/3600)*3600; hi=math.ceil(datetime.now(timezone.utc).timestamp()/3600)*3600
    start.FromDatetime(datetime.fromtimestamp(lo,timezone.utc)); end.FromDatetime(datetime.fromtimestamp(hi,timezone.utc))
    request=api_pb2.WorkspaceBillingReportRequest(start_timestamp=start,end_timestamp=end,resolution='h',app_ids=[app_id])
    async with asyncio.timeout(15):
        rows=[r async for r in client._stub.WorkspaceBillingReport.unary_stream(request)]
    from google.protobuf.json_format import MessageToDict
    return {'app_id':app_id,'query_start_epoch':lo,'query_end_epoch':hi,'metered_usd':sum(float(r.cost) for r in rows),
            'rows':[MessageToDict(r,preserving_proto_field_name=True) for r in rows],
            'queried_utc':datetime.now(timezone.utc).isoformat(),'billing_may_lag':True}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--output',required=True)
    ap.add_argument('--call',default='benchmark_call.json'); a=ap.parse_args()
    call=json.loads((PRE/a.call).read_text())
    try: result=asyncio.run(billing(call['app_id'],call['started_epoch']))
    except Exception as e: result={'app_id':call['app_id'],'metered_usd':None,'error':f'{type(e).__name__}: {e}'}
    write_new(a.output,result); print(json.dumps(result))

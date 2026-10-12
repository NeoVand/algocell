"""Read app-attributed hourly billing; preserve cumulative append-only provenance."""
from __future__ import annotations
import asyncio
import json
import math
from datetime import datetime,timezone
from s8086 import PRE,write_new

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

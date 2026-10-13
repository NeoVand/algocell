import json,sys,time
from pathlib import Path
sys.path.insert(0,'/Users/neo/repos/algocell-wt-s2/experiments')
from lifecycle8086 import ModalProvider
from dispatch8086 import SpendGuard
from s8086 import PRE,write_new
r=Path('/Users/neo/repos/algocell-wt-s2/experiments/results/s8086/launch/run-20261012-volume-parallel')
p=json.loads((PRE/'projection.json').read_text());g=SpendGuard(r/'spend.jsonl',p['job_liabilities_usd'],overhead=5)
rows=[json.loads(s) for s in g.path.read_text().splitlines()];apps=sorted({x['app_id'] for x in rows if x.get('app_id','').startswith('ap-')})
provider=ModalProvider();records=[]
for app in apps:
 provider.cursor="";sample=provider.sample(app);assert sample['stopped'] and not sample['tasks'],app
 starts=[x['time'] for x in rows if x.get('app_id')==app]
 meter=provider.metering(app,min(starts)-6*86400)
 g.transact(dict(kind='metered',app_id=app,usd=meter,accounting='provider-confirmed stopped historical app; fresh cumulative billing, lag possible'))
 records.append(dict(app_id=app,metered_usd=meter,stopped=True,tasks_empty=True))
result=dict(time=time.time(),apps=records,metered_usd=sum(x['metered_usd'] for x in records),billing_may_lag=True,check=g.transact(dict(kind='check')))
write_new(r/'historical_billing_verified.json',result);print(json.dumps(result,indent=2))

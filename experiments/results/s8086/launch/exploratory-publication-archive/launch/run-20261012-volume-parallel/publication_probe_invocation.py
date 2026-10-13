import sys,time,json,threading,subprocess,os
from pathlib import Path
HERE=Path('/Users/neo/repos/algocell-wt-s2/experiments');sys.path.insert(0,str(HERE));os.chdir(HERE)
from volume_probe8086 import app,probe
from dispatch8086 import SpendGuard
from s8086 import PRE,write_new
from lifecycle8086 import ModalProvider
RUN=HERE/'results/s8086/launch/run-20261012-volume-parallel'
p=json.loads((PRE/'projection.json').read_text());guard=SpendGuard(RUN/'spend.jsonl',p['job_liabilities_usd'],overhead=p['storage_termination_reserve_usd'])
job='preparation:publication-verification';guard.reserve(job,.01,preparation=True,budget_usd=.10,cpu=.125,memory_mib=256,server_timeout_s=60)
started=time.time();done=threading.Event()
def timeout_stop():
 if not done.wait(180):
  if app.app_id:subprocess.Popen([sys.executable,'-m','modal','app','stop','--yes',app.app_id],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
threading.Thread(target=timeout_stop,daemon=True).start()
try:
 with app.run(name='s2-volume-probe-20261012'):
  guard.transact(dict(kind='lifecycle',job=job,app_id=app.app_id))
  call=probe.spawn(False,(HERE/'recovery8086.py').read_text());guard.start(job,call.object_id,app.app_id)
  write_new(RUN/'publication_probe_call.json',dict(app_id=app.app_id,call_id=call.object_id,started_epoch=started))
  result=call.get(timeout=100)
  write_new(RUN/'publication_probe.json',result)
finally:
 if app.app_id:subprocess.run([sys.executable,'-m','modal','app','stop','--yes',app.app_id],stdin=subprocess.DEVNULL,timeout=15,check=True)
done.set();print(json.dumps(result,indent=2))

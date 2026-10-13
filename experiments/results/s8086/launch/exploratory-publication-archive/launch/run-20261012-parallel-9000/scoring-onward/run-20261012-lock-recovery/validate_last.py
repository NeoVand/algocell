"""Validate the complete final version, including first-poll carried liabilities."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import hashlib

RUN=Path(__file__).resolve().parent
HERE=Path('/Users/neo/repos/algocell-wt-s2/experiments')
while True:
    text=(RUN/'validate_final.log').read_text()
    if 'RuntimeError' in text:raise RuntimeError('prior full suite failed; explicit STOP')
    if 'final complete suite passed' in text:break
    time.sleep(.5)
runtime=json.loads((RUN/'test_runtime.json').read_text())
if runtime['exit_code']:raise RuntimeError('prior full suite failed; explicit STOP')
for name in ('full_tests.log','full_tests.xml','test_runtime.json','test_gpu_bursts.jsonl'):
    shutil.copyfile(RUN/name,RUN/('second_'+name))
(RUN/'test_runtime.json').unlink()
sources=['dispatch8086.py','lifecycle8086.py','meter8086.py','spend_sim8086.py','score8086.py',
         'tests/test_b5_enforcement.py','tests/test_meter8086.py']
(RUN/'final_source_sha256.json').write_text(json.dumps({n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in sources},indent=2)+'\n')
with (RUN/'full_tests.log').open('w') as log:
    code=subprocess.call([sys.executable,'-u',str(RUN/'test_runner.py')],stdout=log,stderr=log)
if code:raise RuntimeError('final full suite failed; explicit STOP, no refreeze or launch')
print('final complete suite passed',flush=True)

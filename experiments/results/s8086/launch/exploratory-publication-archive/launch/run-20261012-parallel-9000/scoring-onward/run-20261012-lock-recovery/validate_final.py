"""After the initial passing suite, validate the final fixes and added regressions."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

RUN=Path(__file__).resolve().parent
while True:
    try:initial=json.loads((RUN/'test_runtime.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(.5);continue
    break
if initial['exit_code']:
    raise RuntimeError('initial suite failed; explicit STOP condition, no refreeze or launch')
for name in ('full_tests.log','full_tests.xml','test_runtime.json','test_gpu_bursts.jsonl'):
    shutil.copyfile(RUN/name,RUN/('initial_'+name))
(RUN/'test_runtime.json').unlink()
with (RUN/'full_tests.log').open('w') as log:
    code=subprocess.call([sys.executable,'-u',str(RUN/'test_runner.py')],stdout=log,stderr=log)
if code:raise RuntimeError('final suite failed; explicit STOP condition, no refreeze or launch')
print('final complete suite passed',flush=True)

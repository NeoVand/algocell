"""Release the committed operational refreeze and detach only downstream stages."""
import json
import os
from pathlib import Path
import subprocess
import sys
from s8086 import HERE, PRE, read_json, sha, verify_freeze, write_new
from dispatch8086 import SpendGuard

RUN=Path(__file__).resolve().parent
freeze=verify_freeze()
validation=read_json(PRE/'operational_scoring_onward_validation_2026-10-12.json')
if validation['full_suite']['failed'] or validation['concurrent_fault_simulation']['violations']:
    raise RuntimeError('validation failure; STOP')
projection=read_json(PRE/'projection.json')
guard=SpendGuard(RUN/'spend.jsonl',projection['job_liabilities_usd'],overhead=projection['storage_termination_reserve_usd'])
check=guard.transact(dict(kind='check',reason='committed exact refreeze release; scoring onward only'))
if check['stop_required'] or check['projected_liabilities_usd']>95:
    raise RuntimeError('complete forecast exceeds release envelope; STOP')
old_ledger=RUN.parents[1]/'spend.jsonl'
if sha(old_ledger.read_bytes())!=sha((RUN/'prior_spend.jsonl').read_bytes()):
    raise RuntimeError('old ledger changed since accounting reconciliation; STOP')
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
receipt=read_json(RUN.parents[1]/'launch_receipt.json')
amendment=freeze['operational_scoring_onward_amendment']
carry=read_json(RUN/'carry_forward.json')
receipt.update(approval_source='Explicit user continuation: finish scoring-onward lock fix, validate, operational refreeze and detached downstream-only relaunch',
    freeze_sha256=sha((PRE/'freeze.json').read_bytes()),frozen_code_commit=head,
    required_stages=['prebuild','score','extract','eligibility data commit and release','assay','controls'],
    release_record=amendment['path'],operational_amendment=amendment,
    operational_validation_sha256=sha((PRE/'operational_scoring_onward_validation_2026-10-12.json').read_bytes()),
    relaunch_run_directory=str(RUN.relative_to(HERE)),relaunch_nominal_projection_usd=check['projected_liabilities_usd'],
    carried_metered_usd=carry['metered_usd'],carried_historical_liabilities_usd=carry['carried_full_bounds_usd'],
    carry_forward_sha256=sha((RUN/'carry_forward.json').read_bytes()),
    pipeline_invocation_sha256=sha((RUN/'pipeline_invocation.py').read_bytes()),
    completed_world_input_integrity_sha256=sha((RUN/'world_input_integrity.json').read_bytes()),
    scoring_onward_only=True,worlds_rerun=False,completed_input_worlds=20,
    old_terminal_rows_preserved=True,prior_launch_receipt_sha256=sha((RUN.parents[1]/'launch_receipt.json').read_bytes()),
    prior_freeze_sha256=sha((RUN/'prior_freeze.json').read_bytes()))
# Remove prior world-relaunch timing/recovery claims; their allocations are
# already historical, and this receipt authorizes no fresh world input.
for key in ('parallel_with_remaining_registered_world_recoveries_usd','world1_recovery'):
    receipt.pop(key,None)
write_new(RUN/'launch_receipt.json',receipt)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(HERE))
env.pop('S2_PREBUILT_IMAGE_RECEIPT',None)
with (RUN/'pipeline.log').open('x') as log:
    process=subprocess.Popen([sys.executable,'-u',str(RUN/'pipeline_invocation.py')],cwd=HERE,env=env,
                             stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(RUN/'pipeline.pid').write_text(str(process.pid)+'\n')
record=dict(pipeline_pid=process.pid,freeze_sha256=receipt['freeze_sha256'],refreeze_commit=head,
            detached_session=True,forecast_usd=check['projected_liabilities_usd'],cap=95,
            stages=receipt['required_stages'],completed_input_worlds=20,worlds_rerun=False)
(RUN/'detached_launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))

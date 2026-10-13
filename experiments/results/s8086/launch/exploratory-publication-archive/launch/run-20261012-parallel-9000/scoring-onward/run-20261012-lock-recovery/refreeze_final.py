"""Exact operational refreeze, gated on the final complete suite and concurrent faults."""
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from s8086 import HERE, PRE, read_json, sha

RUN=Path(__file__).resolve().parent
runtime=read_json(RUN/'test_runtime.json')
if runtime['exit_code'] or not (RUN/'initial_test_runtime.json').exists():
    raise RuntimeError('final complete suite required; STOP')
if 'final complete suite passed' not in (RUN/'validate_last.log').read_text():
    raise RuntimeError('final validation controller has not completed; STOP')
cases=list(ET.parse(RUN/'full_tests.xml').iter('testcase'))
failed=[c.attrib for c in cases if c.find('failure') is not None or c.find('error') is not None]
skipped=[c.attrib for c in cases if c.find('skipped') is not None]
baseline=list(ET.parse(RUN.parents[1]/'full_tests.xml').iter('testcase'))
old_skips={(c.get('classname'),c.get('name')) for c in baseline if c.find('skipped') is not None}
if failed or {(c['classname'],c['name']) for c in skipped}!=old_skips:
    raise RuntimeError('full suite failed or a new test was skipped; STOP')
required=['test_failed_billing_uses_actual_live_container_accrual[rate_limit]',
          'test_failed_billing_uses_actual_live_container_accrual[timeout]',
          'test_failed_billing_uses_actual_live_container_accrual[unknown]',
          'test_sigstop_dispatcher_holding_real_ledger_lock',
          'test_first_contended_poll_includes_carried_plan_extensions']
for name in required:
    matching=[c for c in cases if c.get('name')==name]
    if len(matching)!=1 or list(matching[0]):raise RuntimeError('required regression did not pass: '+name)
sim=read_json(RUN/'concurrent_fault_simulation.json')
if (sim['sequences']<1000 or sim['violations'] or sim['simulation_sha256']!=sha((HERE/'spend_sim8086.py').read_bytes())
        or sim.get('lifecycle_sha256')!=sha((HERE/'lifecycle8086.py').read_bytes())
        or sim.get('dispatch_sha256')!=sha((HERE/'dispatch8086.py').read_bytes())
        or sim.get('meter_sha256')!=sha((HERE/'meter8086.py').read_bytes())
        or any(not sim['faults'].get(k) for k in ('ledger_lock_contentions','rate_limited_billing_reads','rate_limit_recoveries'))
        or sim['shared_app_jobs']!=20):
    raise RuntimeError('concurrent fault validation failed or stale; STOP')
carry=read_json(RUN/'carry_forward.json')
if carry['complete_forecast_usd']>95:raise RuntimeError('forecast exceeds $95; STOP')
worlds=read_json(RUN/'world_input_integrity.json')
if worlds['worlds_completed']!=20 or worlds['snapshots_verified']!=100 or worlds['world_data_mutated']:
    raise RuntimeError('completed input audit failed; STOP')

# Preserve the newly generated evidence, then restore historical validation
# artifacts to their original frozen bytes. They remain historical evidence.
for name in ('spend_simulation.json','ledger_scale_validation.json','local_bursts.jsonl'):
    path=PRE/name
    (RUN/('generated_'+name)).write_bytes(path.read_bytes())
    path.write_bytes(subprocess.check_output(['git','show',f'HEAD:{path.relative_to(HERE.parent)}'],cwd=HERE.parent))
sources=['dispatch8086.py','lifecycle8086.py','meter8086.py','spend_sim8086.py','score8086.py',
         'tests/test_b5_enforcement.py','tests/test_meter8086.py']
if read_json(RUN/'final_source_sha256.json')!={n:sha((HERE/n).read_bytes()) for n in sources}:
    raise RuntimeError('source changed after final full-suite start; STOP')
old=read_json(RUN/'prior_freeze.json')
for name,digest in old['files'].items():
    if name not in sources and sha((HERE/name).read_bytes())!=digest:
        raise RuntimeError('unapproved frozen file changed; STOP: '+name)
original=subprocess.check_output(['git','show','HEAD:experiments/score8086.py'],cwd=HERE.parent,text=True)
old_gate="    if c['freeze_sha256']!=sha((Path(__file__).parent/'results/s8086/prelaunch/freeze.json').read_bytes()):"
new_gate="    from lifecycle8086 import valid_world_freeze\n    if not valid_world_freeze(path,c['freeze_sha256']):"
if original.replace(old_gate,new_gate)!=(HERE/'score8086.py').read_text():
    raise RuntimeError('scoring change exceeds exact provenance gate; STOP')
report=dict(schema='s2-operational-scoring-onward-validation/1',date='2026-10-12',
    full_suite=dict(passed=len(cases)-len(skipped),failed=len(failed),skipped=len(skipped),
        skipped_cases=skipped,wall_s=runtime['wall_s'],exit_code=runtime['exit_code'],required_regressions=required,
        junit_sha256=sha((RUN/'full_tests.xml').read_bytes()),log_sha256=sha((RUN/'full_tests.log').read_bytes())),
    concurrent_fault_simulation=sim,
    serial_simulation=read_json(RUN/'generated_spend_simulation.json'),
    ledger_scale=read_json(RUN/'generated_ledger_scale_validation.json'),
    source_sha256={name:sha((HERE/name).read_bytes()) for name in sources},
    world_input_integrity_sha256=sha((RUN/'world_input_integrity.json').read_bytes()),
    completed_worlds=20,snapshots_verified=100,worlds_rerun=False,world_data_mutated=False,
    scoring_rules_eligibility_rules_endpoints_changed=False,scoring_source_change='exact operational provenance gate only',
    known_metered_usd=carry['metered_usd'],conservative_recorded_lifetime_bound_usd=carry['conservative_recorded_lifetime_bound_usd'],
    complete_forecast_usd=carry['complete_forecast_usd'],cap=95,
    run_directory=str(RUN.relative_to(HERE)),artifacts_are_historical_except_this_operational_validation=True)
if (report['serial_simulation']['sequences']<1000 or report['serial_simulation']['violations']
        or report['ledger_scale']['false_stops']):raise RuntimeError('serial/ledger validation failed; STOP')
validation=PRE/'operational_scoring_onward_validation_2026-10-12.json'
validation.write_text(json.dumps(report,indent=2)+'\n')
amendment='results/s8086/prelaunch/S2_SCORING_ONWARD_LOCK_AMENDMENT_2026-10-12.md'
freeze=read_json(PRE/'freeze.json')
for name in sources+[amendment,str(validation.relative_to(HERE))]:
    freeze['files'][name]=sha((HERE/name).read_bytes())
freeze.update(unique_test_cases_covered=report['full_suite']['passed'],
    launch_status='SCORING-ONWARD RELAUNCH AUTHORIZED: explicit user instruction; final validation passed',
    operational_scoring_onward_amendment=dict(path=amendment,sha256=freeze['files'][amendment],
        authorized_by='explicit user continuation and scoring-onward relaunch instruction',
        prior_freeze_sha256=sha((RUN/'prior_freeze.json').read_bytes()),
        validation=str(validation.relative_to(HERE)),validation_sha256=sha(validation.read_bytes()),
        complete_all_in_forecast_usd=carry['complete_forecast_usd'],science_rules_endpoints_unchanged=True),
    completed_world_inputs={w['world']:dict(input_sha256=w['sha256'],freeze_sha256=w['freeze_sha256']) for w in worlds['worlds']})
(PRE/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
paths=['experiments/'+name for name in sources+[amendment,str(validation.relative_to(HERE)),str((PRE/'freeze.json').relative_to(HERE))]]
if any(p.startswith(('results/','paper/','.codex/','AGENTS.md','.claude/launch.json')) for p in paths):
    raise RuntimeError('forbidden commit path')
(RUN/'refreeze_commit_paths.json').write_text(json.dumps(paths,indent=2)+'\n')
print(json.dumps(dict(paths=paths,freeze_sha256=sha((PRE/'freeze.json').read_bytes()),files=len(freeze['files']),
                     passed=report['full_suite']['passed'],simulation_sequences=sim['sequences'],violations=0,
                     forecast_usd=carry['complete_forecast_usd']),indent=2))

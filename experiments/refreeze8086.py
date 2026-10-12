"""CPU-only option (b) artifact generation; never release or contact Modal.

Run inputs() before validation, then manifest() after validation/reporting.
The amendment and complete manifest must be committed together.
"""
from __future__ import annotations
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
from benchmark8086 import project
from s8086 import HERE, PRE, ARMS, SEEDS, CAP, DEFAULT_REGISTRATION, fixtures, read_json, sha, safe


def save(name, value):
    (PRE / name).write_text(json.dumps(safe(value), indent=2, allow_nan=False) + '\n')


def inputs():
    registration = DEFAULT_REGISTRATION
    old = read_json(PRE / 'projection.json')
    preparation = old['job_liabilities_usd']['preparation']
    projection = project(read_json(PRE / 'benchmark.json'), preparation, registration)
    prior_option = read_json(PRE / 'forecast_options_repaired.json')['b']
    if projection['job_liabilities_usd'] != prior_option['job_liabilities_usd']:
        raise ValueError('repaired option (b) allocation changed')
    if projection['projection_usd'] > CAP:
        raise ValueError(f'complete option (b) forecast exceeds ${CAP:g}; STOP')
    projection.update(option='b', approval_date='2026-10-12', launch_status='STOPPED: coordinator review required',
                      benchmark_sha256=sha((PRE / 'benchmark.json').read_bytes()),
                      component_totals_usd=prior_option['component_totals_usd'])
    save('projection.json', projection)
    save('cohort.json', {'option':'b', 'arms':{a:ARMS[a] for a in registration.arms},
         'seeds':SEEDS, 'horizon':registration.horizon, 'registration':registration.as_dict(),
         'samples':registration.sample_steps, 'snapshots':registration.snapshots,
         'persistence':registration.persistence, 'law_cap_per_family_per_world':16,
         'random_cells_per_world':64, 'withdrawn_arms':{'B':'prospectively withdrawn; no population data collected'},
         'population_worlds':20, 'gpu':'L4', 'cpu':1, 'memory_gib':8})
    # Both lengths retain historical/control fixtures, with identical RNG bytes.
    rng = safe(fixtures())
    if rng != read_json(PRE / 'rng_fixtures.json'):
        raise ValueError('frozen RNG fixtures changed')
    save('rng_fixtures.json', rng)
    schemas = read_json(PRE / 'input_schemas.json')
    schemas['registered_option'] = 'b'
    schemas['registration'] = registration.as_dict()
    condition = schemas['world_jsonl_record']['oneOf'][0]
    condition['required'] = list(dict.fromkeys(condition['required'] + ['registration','budget']))
    condition['properties'].update(arm={'enum':['A']}, L={'enum':[16]}, budget={'const':128},
                                  registration={'const':registration.as_dict()})
    schemas['world_jsonl_record']['oneOf'][2]['properties']['shape']['enum'] = [[20000,16]]
    for name in ('score','eligibility_gzip_json'):
        schemas[name]['required'] = list(dict.fromkeys(schemas[name]['required'] + ['registration']))
        schemas[name]['properties']['registration'] = {'const':registration.as_dict()}
    schemas['score']['properties']['schema']['const'] = 's2-score/2-registration'
    schemas['eligibility_gzip_json']['properties']['schema']['const'] = 's2-eligibility/2-source-root-phase'
    schemas['certificate_geometry'] = {
        'required':['d','c','K','R','C_R','F','E','erasure_controls','held_erased','phase',
                    'physical_entry','source_start','source_map','reset_source_map','noninterference'],
        'raw_dependencies':'K; c=1+max(K)<=d; no modulo reduction',
        'source_map':'r(q)=s+((L+q-s) mod d); composed generation-zero ancestry',
        'mutable_erasure':'E\\K', 'held_erased':'E intersect K', 'physical_entry':0}
    schemas['launch_receipt'] = ('population_launch_approved=true, independent_design_review, exact freeze_sha256, '
        'exact registered horizon/arm schedules; option (b) only; absent at freeze. '
        'Retention additionally requires eligibility_commit and per-world committed gzip eligibility_sha256.')
    save('input_schemas.json', schemas)
    save('guard_limits.json', {'approved_date':'2026-10-12','option':'b','cap_usd':CAP,
         'includes_preparation':True, 'forecast_usd':projection['projection_usd'],
         'guard':'dispatch8086.SpendGuard; independent lifecycle8086.WatchdogProcess',
         'rate_usd_per_s':projection['rate_usd_per_s'],
         'lifecycle_extra_s':projection['lifecycle_extra_s'], 'policy':projection['lifecycle_policy'],
         'execution_deadline_s':projection['execution_deadline_s'],
         'job_liabilities_usd':projection['job_liabilities_usd'],
         'storage_termination_reserve_usd':projection['storage_termination_reserve_usd'],
         'launch_status':'STOPPED: coordinator review required; no launch receipt'})
    # Export the current generator without allocating a device or compiling WGSL.
    from algocell_exp.zmachine import ZMachine
    from algocell_exp.census8086_trace import DiagnosticMachine
    observers = {}
    for L in (16,32):
        name = f'i8086_pair_maps_L{L}'
        m = read_json(HERE / f'algocell_exp/machines/{name}.manifest.json')
        base = SimpleNamespace(m=m, record_words=m['record']['wordsPerLane'],
             wgsl=(HERE / f'algocell_exp/machines/{name}.wgsl').read_text(),
             ring_bytes=m['world']['memory']['ring'], name=name, dev=None,
             segment=lambda segment, L=L:{'bytes':L})
        captured = {}
        def capture(self, source, manifest, **kwargs):
            captured.update(source=source, manifest=manifest)
        with patch.object(ZMachine, '__init__', capture):
            DiagnosticMachine(base, events=8192, storage_trace=True)
        export = f'observer_{name}_storage'
        (PRE / f'{export}.wgsl').write_text(captured['source'])
        save(f'{export}.manifest.json', captured['manifest'])
        observers[str(L)] = {'wgsl_sha256':sha(captured['source'].encode()),
             'manifest_sha256':sha((PRE / f'{export}.manifest.json').read_bytes()),
             'record_words':captured['manifest']['record']['wordsPerLane'],
             'events':8192, 'mode':'storage-vulkan',
             'scope':'registered A population and fixed controls' if L==16 else 'fixed controls only; B withdrawn'}
    save('observer_hashes.json', observers)
    registration_text = (HERE / 'REVISION_PREREG.md').read_text()
    (PRE / 'S2_REGISTRATION_OPTION_B.md').write_text(registration_text[registration_text.index('## S2 —'):])
    basis = read_json(PRE / 'registration_basis.json')
    basis.update(approved_cap_usd=CAP, option='b', amended_date='2026-10-12',
                 historical_body_sha256=basis['body_sha256'],
                 amended_body_sha256=sha((PRE / 'S2_REGISTRATION_OPTION_B.md').read_bytes()),
                 amendment='Prospective amendment — user\'s option (b), 2026-10-12; STOP before launch')
    save('registration_basis.json', basis)
    return projection


def manifest():
    freeze = read_json(PRE / 'freeze.json')
    projection = read_json(PRE / 'projection.json')
    validation = read_json(PRE / 'validation.json')
    if projection['cap_usd'] != CAP or not projection['launch_fits'] or projection['projection_usd'] > CAP:
        raise ValueError('forecast/cap validation failed; STOP')
    if validation['failed'] or not validation['passed']:
        raise ValueError('validation failed; STOP')
    paths = set(freeze['files'])
    tracked = subprocess.check_output(['git','ls-files','experiments/results/s8086/prelaunch',
               'experiments/results/census8086','experiments/tests'], cwd=HERE.parent, text=True).splitlines()
    paths.update(str((HERE.parent / p).relative_to(HERE)) for p in tracked)
    paths.update(('refreeze8086.py','reforecast8086.py','reaudit8086.py','lifecycle8086.py','score6502.py',
                  'tests/test_s8086_repairs.py','tests/test_score6502.py','results/s8086/prelaunch/guard_limits.json',
                  'results/s8086/prelaunch/S2_REGISTRATION_OPTION_B.md',
                  'results/s8086/prelaunch/refreeze_local_bursts.jsonl',
                  'results/s8086/prelaunch/refreeze_test_runner.py'))
    paths.discard('results/s8086/prelaunch/freeze.json')
    files = {p:sha((HERE / p).read_bytes()) for p in sorted(paths)}
    freeze.update(schema='s2-freeze/2-option-b', amended_date='2026-10-12', option='b',
        approved_cap_usd=CAP, projection_usd=projection['projection_usd'], population_worlds=20,
        registration=DEFAULT_REGISTRATION.as_dict(), registered_arms=['A'], withdrawn_arms=['B'],
        original_normative_body_sha256=freeze.get('original_normative_body_sha256',freeze['normative_body_sha256']),
        normative_body_sha256=files['results/s8086/prelaunch/S2_REGISTRATION_OPTION_B.md'],
        no_population_runs=True, launch_status='STOPPED: coordinator refreeze review required',
        launch_receipt_created=False, registration_sha256=files['REVISION_PREREG.md'],
        projection_sha256=files['results/s8086/prelaunch/projection.json'],
        validation_sha256=files['results/s8086/prelaunch/validation.json'],
        guard_limits_sha256=files['results/s8086/prelaunch/guard_limits.json'],
        unique_test_cases_covered=validation['passed'],
        independent_review='results/s8086/prelaunch/ASTRA_PRELAUNCH_REVIEW.md',
        independent_refreeze_review='pending coordinator review; previous reviews do not release this freeze',
        repairs_commit='5309961b56d4eef177bcec43744cab05c7f6cb9f',
        limitations=['arbitrary 8086 proof extraction remains unresolved outside explicitly modeled instructions',
                     'physical stop requires functioning host/provider control; unknown overruns retain liabilities'],
        files=files)
    save('freeze.json', freeze)
    return sha((PRE / 'freeze.json').read_bytes()), len(files)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('inputs','manifest'))
    stage = ap.parse_args().stage
    if stage == 'inputs':
        p = inputs()
        print(f"STOPPED option (b): ${p['projection_usd']:.6f} / ${CAP:g}")
    else:
        digest, count = manifest()
        print(f'STOPPED freeze SHA-256 {digest}; {count} exact file identities')

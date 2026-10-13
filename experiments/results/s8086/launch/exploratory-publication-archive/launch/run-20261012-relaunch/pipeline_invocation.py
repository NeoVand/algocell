"""Recorded detached invocation of unchanged frozen entrypoints; no host retries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)
from s8086 import PRE, SEEDS, sha, write_new, read_json, released_registration
from lifecycle8086 import atomic_json
from dispatch8086 import SpendGuard
from run8086 import world_name

RELEASE = RUN / 'launch_receipt.json'
LEDGER = RUN / 'spend.jsonl'
IMAGE = RUN / 'prebuilt_image.json'
PROJECTION = PRE / 'projection.json'
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(HERE))
env.pop('S2_PREBUILT_IMAGE_RECEIPT', None)


def event(kind, **values):
    row = dict(time=time.time(), kind=kind, pipeline_pid=os.getpid(), **values)
    with (RUN / 'pipeline.events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
        f.flush()
        os.fsync(f.fileno())
    atomic_json(RUN / 'pipeline.status.json', row)
    print(json.dumps(row), flush=True)


def stage(name, script, arguments):
    with (RUN / f'{name}.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-u', str(HERE / script), *map(str, arguments)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 cwd=HERE, env=env)
        event('stage_started', stage=name, child_pid=child.pid)
        (RUN / f'{name}.pid').write_text(str(child.pid) + '\n')
        code = child.wait()
    if code:
        raise RuntimeError(f'{name} exited {code}; no retry, review logs and retained liabilities')
    event('stage_complete', stage=name, child_pid=child.pid)


def dispatch(name, release=RELEASE):
    stage(name, 'dispatch8086.py', ['--release', release, '--ledger', LEDGER,
          '--projection', PROJECTION, '--stage', name])


try:
    projection = read_json(PROJECTION)
    guard = SpendGuard(LEDGER, projection['job_liabilities_usd'],
                       overhead=projection['storage_termination_reserve_usd'])
    check = guard.transact({'kind': 'check', 'accounting': 'detached complete-pipeline startup'})
    if check['stop_required']:
        raise RuntimeError(f'carry-forward spend gate rejected launch: {check}')
    receipt = read_json(RELEASE)
    registration = released_registration(receipt)
    event('pipeline_started', registered_worlds=20,
          seeds=list(SEEDS), stages=['prebuild', 'worlds', 'score', 'extract',
                                    'eligibility_commit_release', 'assay', 'controls'],
          projected_liabilities_usd=check['projected_liabilities_usd'])
    stage('prebuild', 'prebuild8086.py', ['--release', RELEASE, '--ledger', LEDGER, '--output', IMAGE])
    from prebuild8086 import validated_image_id
    image_id = validated_image_id(IMAGE)
    env['S2_PREBUILT_IMAGE_RECEIPT'] = str(IMAGE)
    event('image_receipt_validated', image_id=image_id)
    for name in ('worlds', 'score', 'extract'):
        dispatch(name)

    event('eligibility_download_started', worlds=20)
    import modal
    volume = modal.Volume.from_name('algocell-atlas-runs')
    paths, digests = [], {}
    for arm in registration.arms:
        for seed in SEEDS:
            name = world_name(arm, seed)
            target = HERE / f'results/s8086/eligibility/{arm}/{name}.json.gz'
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                for chunk in volume.read_file(f'/s8086_extract/{arm}/{name}.json.gz'):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            data = read_json(target)
            if data.get('registration') != registration.as_dict():
                raise ValueError(f'eligibility registration differs: {name}')
            digests[name] = sha(target.read_bytes())
            paths.append(str(target.relative_to(HERE.parent)))
    if len(paths) != 20:
        raise ValueError('complete eligibility ledger required before evolved interventions')
    subprocess.run(['git', 'add', '--', *paths], cwd=HERE.parent, check=True)
    message = ('Freeze S2 relaunch eligibility before evolved L1/L2 assays\n\n'
               'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>')
    subprocess.run(['git', 'commit', '--only', '-m', message, '--', *paths], cwd=HERE.parent, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE.parent, text=True).strip()
    assay_receipt = dict(receipt, eligibility_commit=commit, eligibility_sha256=digests)
    assay_release = RUN / 'eligibility_release.json'
    write_new(assay_release, assay_receipt)
    event('eligibility_released', eligibility_commit=commit, worlds=20,
          eligibility_sha256=digests)
    dispatch('assay', assay_release)
    dispatch('controls', assay_release)
    event('pipeline_complete', scientific_verdict='read frozen world/scoring/assay outputs')
except BaseException as exc:
    event('pipeline_incomplete', error=type(exc).__name__, reason=str(exc),
          scientific_verdict=None, no_retry=True)
    raise

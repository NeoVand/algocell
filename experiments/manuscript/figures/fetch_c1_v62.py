"""Download-only C1 E trajectory audit. Never imports or calls a Modal function.

Raw downloads stay outside Git. The committed audit binds each trajectory to
the frozen dispatch spec, recorded completion marker, generated machine hashes,
and all endpoint measurements (including the undefined-ratio core-loss world).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
OUT = HERE / 'out/v6/data'
MODAL = '/Users/neo/repos/algocell/experiments/.venv/bin/modal'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--destination', type=Path, default=Path('/tmp/figs-v62-c1'))
    args = ap.parse_args()
    dispatch_path = EXP / 'results/tier2/DISPATCH.json'
    markers_path = EXP / 'results/tier2/run_markers.json'
    endpoint_path = EXP / 'results/tier2/scored/E/offset_switch.csv'
    jobs = json.loads(dispatch_path.read_text())
    markers = json.loads(markers_path.read_text())
    endpoints = pd.read_csv(endpoint_path).query("L==32 and mut=='on'")
    selected = [j for j in jobs if j['kind']=='offset' and
                j['condition']['L']==32 and j['condition']['mut']=='on']
    assert len(selected)==40 and len(endpoints)==40

    def download(job):
        base = args.destination / job['experiment']
        base.mkdir(parents=True, exist_ok=True)
        paths = []
        for suffix in ('.jsonl', '.complete.json'):
            name = job['stem'] + suffix
            path = base / name
            remote = 'tier2/' + job['experiment'] + '/' + name
            if not path.exists():
                for attempt in range(6):
                    result = subprocess.run([MODAL, 'volume', 'get', 'algocell-atlas-runs',
                                             remote, str(path)], capture_output=True, text=True)
                    if result.returncode==0 or 'rate limit' not in result.stderr.lower():
                        break
                    time.sleep(min(30,5*(attempt+1)))
                time.sleep(1)
                if result.returncode:
                    return {'id':job['id'], 'available':False, 'remote':remote,
                            'error':result.stderr.strip()}
            paths.append(path)
        raw, marker_path = paths
        marker = json.loads(marker_path.read_text())
        recorded = markers[job['id']]
        for field in ('id','status','spec_sha256','source_commit','machine'):
            assert marker[field]==recorded[field], (job['id'],field)
        assert marker['status']=='complete'
        assert marker['spec_sha256']==job['spec_sha256']
        # The spec digest was recorded before source_commit was added at launch.
        spec = {k:v for k,v in job.items() if k not in ('spec_sha256','source_commit')}
        assert hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()==job['spec_sha256']
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        condition = rows[0]
        for field in ('L','tar','mut','start','seed','steps'):
            assert condition[field]==job['condition'][field]
        machine_path = EXP / 'algocell_exp/machines' / (job['machine']+'.manifest.json')
        machine = json.loads(machine_path.read_text())
        for field in ('sha256_core','sha256_world','sha256_wgsl'):
            assert condition['machine'][field]==machine['sha256'][field.removeprefix('sha256_')], (job['id'],field)
        assert condition['machine']['machine']==job['machine']
        samples = [r for r in rows if 'core' in r]
        final = samples[-1]
        c = job['condition']
        endpoint = endpoints[(endpoints.tar==c['tar']) & (endpoints.start==c['start']) &
                             (endpoints.seed==c['seed'])].iloc[0]
        share = final['T']/(final['R']+final['T']) if final['R']+final['T'] else np.nan
        for measured,expected in [(final['step'],endpoint.steps),
                                  (share,endpoint.T_share_end),
                                  (final['core']/20000,endpoint.core_share_end),
                                  (final['T_distinct_tails'],endpoint.T_distinct_tails_end)]:
            assert np.isclose(measured,expected,equal_nan=True), (job['id'],measured,expected)
        return {'id':job['id'], 'available':True, 'raw_path':str(raw),
                'remote':'tier2/'+job['experiment']+'/'+raw.name,
                'raw_sha256':sha(raw), 'marker_sha256':sha(marker_path),
                'spec_sha256':job['spec_sha256'], 'source_commit':marker['source_commit'],
                'machine_manifest_sha256':sha(machine_path), 'samples':len(samples),
                'endpoint_verified':True, 'machine_hashes_verified':True,
                'recorded_marker_verified':True}

    with ThreadPoolExecutor(max_workers=1) as pool:
        records = list(pool.map(download,selected))
    audit = {'volume':'algocell-atlas-runs','mode':'modal volume get only',
             'raw_hash_note':'Raw file SHA-256 recorded on download; prior ledger binds spec and machine hashes, not raw log content.',
             'dispatch_sha256':sha(dispatch_path),'recorded_markers_sha256':sha(markers_path),
             'endpoint_table_sha256':sha(endpoint_path),'worlds':records}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'c1_download_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Verified C1 trajectories:',sum(r['available'] for r in records),'of',len(records))
    if any(not r['available'] for r in records):
        print('Unavailable:',[r['id'] for r in records if not r['available']])


if __name__=='__main__':
    main()

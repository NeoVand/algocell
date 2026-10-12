"""S2 fixed-input preparation and gated future world container. No default launch."""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path
import modal
if Path('/root/experiments').is_dir(): sys.path.insert(0,'/root/experiments')
from s8086 import HERE,PRE,RATE,sha,write_new

FILES=('s8086.py','run8086.py','score8086.py','law8086.py','benchmark8086.py','run6502.py',
       'census8086.py','census8086_law.py','dispatch8086.py','modal_s8086.py','modal_score8086.py','lifecycle8086.py','meter8086.py')
image=(modal.Image.from_id('im-T2VaU4HVHQXvzDSkbRye1y')
       .apt_install('mesa-vulkan-drivers')
       .env({'WGPU_BACKEND_TYPE':'Vulkan','XDG_RUNTIME_DIR':'/tmp','PYTHONPATH':'/root/experiments',
             'S2_TRACE_BACKEND':'storage-vulkan'})
       .add_local_dir(str(HERE/'algocell_exp'),remote_path='/root/experiments/algocell_exp')
       .add_local_dir(str(PRE),remote_path='/root/experiments/results/s8086/prelaunch'))
for name in FILES: image=image.add_local_file(str(HERE/name),remote_path=f'/root/experiments/{name}')
if (PRE/'freeze.json').exists():
    # Mount all non-runtime archive identities required by remote validation.
    for name in json.loads((PRE/'freeze.json').read_text())['files']:
        if name not in FILES and not name.startswith(('algocell_exp/','results/s8086/prelaunch/')):
            image=image.add_local_file(str(HERE/name),remote_path=f'/root/experiments/{name}')
app=modal.App('algocell-s8086-prelaunch')
volume=modal.Volume.from_name('algocell-atlas-runs')
BENCHMARK_TIMEOUT=450
STARTUP_TIMEOUT=120
WORLD_TIMEOUT=12*3600  # dispatcher reserves the smaller enforced per-world deadline

@app.function(image=image,gpu='L4',cpu=(1,1),memory=(8192,8192),timeout=BENCHMARK_TIMEOUT,
              startup_timeout=STARTUP_TIMEOUT,retries=0,single_use_containers=True,
              min_containers=0,buffer_containers=0,max_containers=1)
def fixed_benchmark():
    import sys
    sys.path.insert(0,'/root/experiments')
    from benchmark8086 import benchmark
    return benchmark()

@app.function(image=image,gpu='L4',cpu=(1,1),memory=(8192,8192),timeout=WORLD_TIMEOUT,
              volumes={'/vol':volume},startup_timeout=STARTUP_TIMEOUT,retries=0,single_use_containers=True,
              min_containers=0,buffer_containers=0,max_containers=1)
def world(arm,seed,commit,receipt,deadline):
    import sys,tempfile
    sys.path.insert(0,'/root/experiments')
    import run8086 as R
    # Freeze identities are checked on the host before spawn. Remote mounts
    # lack git metadata, so verify content directly and use the host release.
    from s8086 import verify_freeze
    import s8086
    manifest=json.loads((PRE/'freeze.json').read_text())
    for name,digest in manifest['files'].items():
        if sha((HERE/name).read_bytes())!=digest: raise ValueError('remote freeze mismatch')
    if receipt.get('freeze_sha256')!=sha((PRE/'freeze.json').read_bytes()) or receipt.get('population_launch_approved') is not True:
        raise ValueError('unreleased population launch')
    # Git attestation was checked by dispatcher; rebind only that remote check.
    s8086.committed=lambda p:sha(Path(p).read_bytes())
    volume.reload(); start=time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        release=Path(tmp)/'release.json'; release.write_text(json.dumps(receipt))
        try: result=R.run_world(arm,seed,Path('/vol/s8086')/arm,release=release,
                               deadline_s=deadline,git_commit=commit,commit_output=volume.commit)
        except BaseException:
            try: volume.commit()
            except Exception: pass
            raise
        else: volume.commit()
    if time.monotonic()-start>deadline: raise RuntimeError('world allocation exceeded; retain liability')
    return {'summary':result,'body_complete':True}  # host settles only after confirmed stopping

@app.local_entrypoint()
def main():
    raise RuntimeError('STOPPED: historical preparation complete; no new paid calls authorized')

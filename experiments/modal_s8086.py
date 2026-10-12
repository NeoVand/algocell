"""S2 fixed-input preparation and gated future world container. No default launch."""
from __future__ import annotations
import json
import time
from pathlib import Path
import modal
from s8086 import HERE,PRE,RATE,sha,write_new

FILES=('s8086.py','run8086.py','score8086.py','law8086.py','benchmark8086.py','run6502.py',
       'census8086.py','census8086_law.py','dispatch8086.py','modal_s8086.py','modal_score8086.py')
image=(modal.Image.from_id('im-T2VaU4HVHQXvzDSkbRye1y')
       .env({'WGPU_BACKEND_TYPE':'Vulkan','XDG_RUNTIME_DIR':'/tmp'})
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
WORLD_TIMEOUT=12*3600  # dispatcher reserves the smaller enforced per-world deadline

@app.function(image=image,gpu='L4',cpu=(1,1),memory=(8192,8192),timeout=BENCHMARK_TIMEOUT,
              retries=0,single_use_containers=True,max_containers=1)
def fixed_benchmark():
    import sys
    sys.path.insert(0,'/root/experiments')
    from benchmark8086 import benchmark
    return benchmark()

@app.function(image=image,gpu='L4',cpu=(1,1),memory=(8192,8192),timeout=WORLD_TIMEOUT,
              volumes={'/vol':volume},retries=0,single_use_containers=True,max_containers=1)
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
        finally: volume.commit()
    if time.monotonic()-start>deadline: raise RuntimeError('world allocation exceeded; retain liability')
    return {'summary':result,'resource_time_bound_usd':(deadline+120)*RATE}

@app.local_entrypoint()
def main():
    """The only entrypoint is the authorized <=$2 fixed-input benchmark."""
    from dispatch8086 import SpendGuard
    from benchmark8086 import project
    # No retries; an existing reservation makes a second invocation fail.
    guard=SpendGuard(PRE/'spend.jsonl',{'preparation-benchmark':.50},cap=2.,overhead=.50)
    guard.reserve('preparation-benchmark',.50,inputs_sha256=sha((PRE/'benchmark_inputs.json').read_bytes()))
    started=time.time(); call=fixed_benchmark.spawn()
    guard.start('preparation-benchmark',call.object_id,app.app_id)
    write_new(PRE/'benchmark_call.json',{'app_id':app.app_id,'call_id':call.object_id,
                                       'started_epoch':started,'timeout_s':BENCHMARK_TIMEOUT,
                                       'rate_usd_per_s':RATE,'resource_reservation_usd':.50})
    result=call.get(); result['app_id']=app.app_id; result['call_id']=call.object_id
    elapsed=time.time()-started
    # Whole application allocation + 120s termination/startup allowance; not a bill.
    bound=(elapsed+120)*RATE+.05
    if bound>.50: raise RuntimeError('benchmark resource bound exceeded reservation')
    result['whole_app_resource_bound_usd']=bound
    write_new(PRE/'benchmark.json',result)
    guard.settle('preparation-benchmark',bound,app_id=app.app_id,call_id=call.object_id,
                 function_wall_s=result['function_wall_s'])
    write_new(PRE/'projection.json',project(result,bound))
    print(json.dumps({'app_id':app.app_id,'resource_bound_usd':bound,
                      'projection_usd':project(result,bound)['projection_usd']}))

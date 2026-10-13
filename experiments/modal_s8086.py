"""S2 fixed-input preparation and gated future world container. No default launch."""
from __future__ import annotations
import json
import os
import time
import sys
import tempfile
import shutil
from pathlib import Path
import modal
if Path('/root/experiments').is_dir(): sys.path.insert(0,'/root/experiments')
from s8086 import HERE,PRE,RATE,sha,write_new

FILES=('s8086.py','run8086.py','score8086.py','law8086.py','benchmark8086.py','run6502.py',
       'census8086.py','census8086_law.py','dispatch8086.py','modal_s8086.py','modal_score8086.py','lifecycle8086.py','meter8086.py','prebuild8086.py','recovery8086.py')
IMAGE_BUNDLES=[]  # retain local build context until build() finishes
def build_image_spec():
    # One content layer, rather than hundreds of per-file image-build inputs.
    # All recipe content is frozen; no runtime receipt/log is copied in.
    bundle=tempfile.TemporaryDirectory(prefix='s2-image-');IMAGE_BUNDLES.append(bundle)
    directory=Path(bundle.name)
    paths=set(FILES)
    if (PRE/'freeze.json').exists():
        paths.update(json.loads((PRE/'freeze.json').read_text())['files'])
        paths.add('results/s8086/prelaunch/freeze.json')
    paths.update(str(p.relative_to(HERE)) for p in (HERE/'algocell_exp').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in sorted(paths):
        destination=directory/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(HERE/name,destination)
    image=(modal.Image.from_id('im-T2VaU4HVHQXvzDSkbRye1y')
           .apt_install('mesa-vulkan-drivers')
           .env({'WGPU_BACKEND_TYPE':'Vulkan','XDG_RUNTIME_DIR':'/tmp','PYTHONPATH':'/root/experiments',
                 'S2_TRACE_BACKEND':'storage-vulkan'})
           .add_local_dir(str(directory),remote_path='/root/experiments',copy=True))
    return image

# Dispatch requires a matching, separately built receipt and uses only from_id.
# Historical fixed-benchmark definitions retain the recipe but cannot dispatch
# through the coordinator's guarded entry points without that receipt.
if os.environ.get('S2_PREBUILT_IMAGE_RECEIPT'):
    from prebuild8086 import validated_image_id
    PREBUILT_IMAGE_ID=validated_image_id(Path(os.environ['S2_PREBUILT_IMAGE_RECEIPT']))
    image=modal.Image.from_id(PREBUILT_IMAGE_ID)
else:
    PREBUILT_IMAGE_ID=None
    image=build_image_spec()
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
    from recovery8086 import attempt_directory,publish_complete
    root=Path('/vol/s8086')/arm
    attempt,recovery=attempt_directory(root,R.world_name(arm,seed));volume.commit()
    with tempfile.TemporaryDirectory() as tmp:
        release=Path(tmp)/'release.json'; release.write_text(json.dumps(receipt))
        try:
            R.run_world(arm,seed,attempt,release=release,deadline_s=deadline,git_commit=commit,
                        commit_output=volume.commit,recovery=recovery)
            result=publish_complete(attempt,root,R.world_name(arm,seed),s8086.Registration.from_dict(receipt['registration']))
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

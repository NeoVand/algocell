import os,sys,time,json,functools
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
root=Path('/Users/neo/repos/algocell-wt-s2/experiments')
sys.path.insert(0,str(root))
os.chdir(root)
os.environ['PYTHONPATH']=str(root)
import wgpu
import wgpu.backends.wgpu_native._api as native
log=root/'results/s8086/launch/run-20261012-volume-parallel/test_gpu_bursts.jsonl'
log.write_text('')
depth=0
def bounded(fn,wait=False):
    @functools.wraps(fn)
    def call(self,*args,**kwargs):
        global depth
        if depth: return fn(self,*args,**kwargs)
        start=time.monotonic()
        try:
            result=fn(self,*args,**kwargs)
            if wait:
                depth+=1
                try:
                    scratch=self._device.create_buffer(size=4,usage=wgpu.BufferUsage.COPY_SRC|wgpu.BufferUsage.COPY_DST)
                    try: self.read_buffer(scratch)
                    finally: scratch.destroy()
                finally: depth-=1
            return result
        finally:
            elapsed=time.monotonic()-start
            pause=time.monotonic(); time.sleep(elapsed)
            with log.open('a') as f:
                f.write(json.dumps({'api':fn.__qualname__,'burst_s':elapsed,
                    'pause_s':time.monotonic()-pause,'foreground':True})+'\n')
            if elapsed>60: raise RuntimeError('local GPU burst exceeded 60s')
    return call
native.GPUQueue.submit=bounded(native.GPUQueue.submit,wait=True)
native.GPUDevice.create_compute_pipeline=bounded(native.GPUDevice.create_compute_pipeline)
import pytest
start=time.monotonic()
code=pytest.main(['tests','-q','-p','no:cacheprovider','--junitxml=/Users/neo/repos/algocell-wt-s2/experiments/results/s8086/launch/run-20261012-volume-parallel/full_tests.xml'])
Path('/Users/neo/repos/algocell-wt-s2/experiments/results/s8086/launch/run-20261012-volume-parallel/test_runtime.json').write_text(json.dumps({'exit_code':int(code),'wall_s':time.monotonic()-start}))
sys.exit(int(code))

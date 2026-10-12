"""S2 immutable inputs, random streams, identities and conservative file helpers."""
from __future__ import annotations
import hashlib
import gzip
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PRE = HERE / 'results/s8086/prelaunch'
HORIZON = 3_000_000
SEEDS = tuple(range(8086001, 8086021))
ARMS = {a: {'L': L, 'budget': 128, 'machine': f'i8086_pair_maps_L{L}'}
        for a, L in (('A', 16), ('B', 32))}
SNAPSHOTS = (0, 20_000, 100_000, 1_000_000, HORIZON)
PERSISTENCE = tuple(range(2_750_000, HORIZON + 1, 25_000))
RATE = .000222 + .0000131 + 8 * .00000222
CAP = 60.0

@dataclass(frozen=True)
class Registration:
    """Prospective numerical parameters; construction never registers/releases."""
    horizon: int = HORIZON
    arms: tuple[str,...] = ('A','B')

    def __post_init__(self):
        object.__setattr__(self,'arms',tuple(self.arms))
        if (type(self.horizon) is not int or not 250_000 <= self.horizon <= HORIZON
                or self.horizon % 25_000 or not self.arms or len(set(self.arms))!=len(self.arms)
                or any(a not in ARMS for a in self.arms) or self.arms!=tuple(sorted(self.arms))):
            raise ValueError('invalid prospective horizon/arm set')

    @property
    def sample_steps(self):
        schedule = [0,*range(50,2001,50),*range(2250,20001,250),
                    *range(22500,1_000_001,2500),*range(1_025_000,HORIZON+1,25000)]
        return tuple(s for s in schedule if s<=self.horizon)

    @property
    def snapshots(self):
        return tuple(sorted({s for s in SNAPSHOTS if s<=self.horizon}|{self.horizon}))

    @property
    def persistence(self):
        return tuple(range(self.horizon-250_000,self.horizon+1,25_000))

    def as_dict(self):
        return {'horizon':self.horizon,'arms':list(self.arms),
                'sample_steps':list(self.sample_steps),'snapshot_steps':list(self.snapshots),
                'R2b_steps':list(self.persistence),'worlds_per_arm':20,'final_cells':64,
                'cases_per_family_cap':16}

    @classmethod
    def from_dict(cls,data):
        result=cls(data['horizon'],tuple(data['arms']))
        if data!=result.as_dict(): raise ValueError('derived registration schedule differs')
        return result

DEFAULT_REGISTRATION = Registration()

def samples(horizon=HORIZON):
    return list(Registration(horizon).sample_steps)

def released_registration(receipt):
    # Required only by future release paths, after user choice and refreeze.
    freeze=verify_freeze()
    if receipt.get('registration')!=freeze.get('registration') or not receipt.get('registration'):
        raise ValueError('prospective horizon and arms have not been registered/released')
    return Registration.from_dict(receipt['registration'])

def sha(data):
    return hashlib.sha256(data).hexdigest()

def key(arm, seed, L, purpose, tape='', case='', generation=0):
    if arm not in ARMS or L != ARMS[arm]['L'] or not isinstance(seed, int):
        raise ValueError('invalid RNG identity')
    if not isinstance(case, str) or not isinstance(generation, int):
        raise ValueError('invalid RNG case/generation')
    if tape != bytes.fromhex(tape).hex():
        raise ValueError('tape hex must be lowercase canonical hex')
    return hashlib.sha256(json.dumps(['S2-8086-v1', arm, seed, L, purpose, tape, case, generation],
                         ensure_ascii=True, separators=(',', ':')).encode()).digest()

def rng(*args):
    return np.random.Generator(np.random.PCG64(int.from_bytes(key(*args)[:8], 'little')))

def partners(arm, seed, L, tape, purpose='route-trace'):
    return rng(arm, seed, L, purpose, tape).integers(0, 256, (16, L), dtype=np.uint8)

def safe(value):
    if isinstance(value, dict): return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw=(json.dumps(safe(value),indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:
        f.write(gzip.compress(raw,mtime=0) if path.suffix=='.gz' else raw)

def read_json(path):
    path=Path(path); raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)

def committed(path):
    path = Path(path).resolve()
    rel = path.relative_to(HERE.parent)
    raw = subprocess.check_output(['git', 'show', f'HEAD:{rel}'], cwd=HERE)
    if raw != path.read_bytes(): raise ValueError(f'uncommitted freeze: {rel}')
    return sha(raw)

def verify_freeze():
    manifest = json.loads((PRE / 'freeze.json').read_text())
    committed(PRE / 'freeze.json')
    for name, digest in manifest['files'].items():
        p = HERE / name
        if sha(p.read_bytes()) != digest or committed(p) != digest:
            raise ValueError(f'frozen source changed: {name}')
    return manifest

def launch_receipt(path):
    """An explicit later coordinator release is required; preparation has none."""
    freeze = verify_freeze()
    release = json.loads(Path(path).read_text())
    if (release.get('population_launch_approved') is not True
            or release.get('freeze_sha256') != sha((PRE / 'freeze.json').read_bytes())
            or not release.get('independent_design_review')):
        raise ValueError('population launch has not been released by coordinator')
    if freeze['projection_usd'] > CAP: raise ValueError('complete design exceeds $60')
    return release

def fixed_burst(fn):
    """Foreground local GPU call; equal pause, no use by cloud execution."""
    start = time.monotonic()
    try: result = fn()
    finally:
        elapsed = time.monotonic() - start
        pause_start = time.monotonic(); time.sleep(elapsed)
        PRE.mkdir(parents=True,exist_ok=True)
        with (PRE/'local_bursts.jsonl').open('a') as f:
            f.write(json.dumps({'burst_s':elapsed,'pause_s':time.monotonic()-pause_start,
                                'foreground':True})+'\n')
    if elapsed > 60: raise RuntimeError('local GPU burst exceeded 60s')
    return result

def diagnostic_for(machine):
    """Explicit Modal trace backend; no silent execution fallback.

    NVIDIA lost the device compiling the 8192-event private array; Mesa also
    timed out. The bit-identical storage observer avoids that array on L4.
    The original private mode and explicit Mesa mode remain available for
    fixed-input validation. Every full original lane/memory must match raw execution.
    """
    from algocell_exp.census8086_trace import DiagnosticMachine
    backend=os.environ.get('S2_TRACE_BACKEND','')
    if backend not in ('','software-vulkan','storage-vulkan'): raise ValueError('unknown S2 trace backend')
    if backend == 'storage-vulkan': return DiagnosticMachine(machine,events=8192,storage_trace=True)
    if backend == 'software-vulkan':
        import wgpu
        from algocell_exp.zmachine import load_machine
        adapter=wgpu.gpu.request_adapter_sync(force_fallback_adapter=True)
        if str(adapter.info.get('adapter_type'))!='CPU' or str(adapter.info.get('backend_type'))!='Vulkan':
            raise RuntimeError('declared trace backend requires CPU software Vulkan')
        device=adapter.request_device_sync()
        base=load_machine(machine.name,device=device)
        return DiagnosticMachine(base,events=8192)
    return DiagnosticMachine(machine,events=8192)

def fixtures():
    rows = []
    for arm, cfg in ARMS.items():
        L, seed = cfg['L'], SEEDS[0]
        tape = bytes([0x90] * L).hex()
        for purpose in ('culture', 'route-trace', 'cells-final', 'law-rank', 'law-joint', 'law-serial'):
            t = '' if purpose == 'cells-final' else tape
            case = 'joint:256' if purpose in ('law-joint', 'law-serial') else ''
            g = 1 if purpose == 'law-serial' else 0
            args = (arm, seed, L, purpose, t, case, g)
            generator = rng(*args)
            values = (generator.choice(20000, 64, replace=False).tolist() if purpose == 'cells-final'
                      else generator.integers(0, 256, 64, dtype=np.uint8).tobytes().hex())
            rows.append({'args': args, 'digest': key(*args).hex(), 'fixture': values})
    return {'numpy': np.__version__, 'bit_generator': 'PCG64', 'rows': rows}

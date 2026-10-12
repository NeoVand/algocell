"""S2 immutable inputs, random streams, identities and conservative file helpers."""
from __future__ import annotations
import hashlib
import json
import math
import subprocess
import time
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

def samples():
    return [0, *range(50, 2001, 50), *range(2250, 20001, 250),
            *range(22500, 1_000_001, 2500), *range(1_025_000, HORIZON + 1, 25000)]

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
    with path.open('x') as f:
        f.write(json.dumps(safe(value), indent=2, allow_nan=False) + '\n')

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

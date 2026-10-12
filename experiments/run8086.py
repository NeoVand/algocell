"""S2 records only. No smoke/pilot mode; population work requires a later release."""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path
import brotli
import numpy as np
import wgpu
from algocell_exp.metrics import quasispecies_share
from algocell_exp.soup import adapter_summary
from algocell_exp.zsoup import ZSoup
from s8086 import ARMS, HERE, HORIZON, SEEDS, SNAPSHOTS, samples, sha, launch_receipt, released_registration
from run6502 import exact_tapes as _exact_tapes, utc_now

class S2Soup(ZSoup):
    """Observe realized active pairs without changing the shared soup kernels.

    Accumulate at most 50 steps in u32, read before reset, then use a Python
    integer. Mutation races and collision policy are untouched.
    """
    def _build(self):
        super()._build()
        B = wgpu.BufferUsage
        self.tally = self.device.create_buffer(size=4, usage=B.STORAGE | B.COPY_SRC | B.COPY_DST)
        module = self.device.create_shader_module(code='''
@group(0) @binding(0) var<storage, read> active_mask: array<u32>;
@group(0) @binding(1) var<storage, read_write> total: atomic<u32>;
@compute @workgroup_size(64)
fn tally(@builtin(global_invocation_id) id: vec3u) {
  if(id.x < arrayLength(&active_mask) && active_mask[id.x] != 0u) { atomicAdd(&total, 1u); }
}''')
        layout = self.device.create_bind_group_layout(entries=[
            {'binding': i, 'visibility': wgpu.ShaderStage.COMPUTE,
             'buffer': {'type': 'read-only-storage' if i == 0 else 'storage'}} for i in range(2)])
        self.tally_pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[layout]),
            compute={'module': module, 'entry_point': 'tally'})
        self.tally_bg = self.device.create_bind_group(layout=layout, entries=[
            {'binding': 0, 'resource': {'buffer': self.pair_active_buf, 'size': self.pair_count * 4}},
            {'binding': 1, 'resource': {'buffer': self.tally, 'size': 4}}])
        original = self.bridge.encode_execute
        def observe(enc, bg, mbg, count):
            original(enc, bg, mbg, count)
            p = enc.begin_compute_pass()
            p.set_pipeline(self.tally_pipeline); p.set_bind_group(0, self.tally_bg)
            p.dispatch_workgroups(-(-count // 64)); p.end()
        self.bridge.encode_execute = observe
        self.active_pairs_total = 0

    def step(self, n=1):
        if not 1 <= n <= 50: raise ValueError('S2 submission must contain 1..50 steps')
        self.device.queue.write_buffer(self.tally, 0, np.zeros(1, np.uint32))
        super().step(n)
        total = int(np.frombuffer(self.device.queue.read_buffer(self.tally), np.uint32)[0])
        if total > n * self.pair_count: raise RuntimeError('invalid active-pair total')
        self.active_pairs_total += total

def exact_tapes(cells, hashes):
    record, top = _exact_tapes(cells, hashes)
    for t in record['top3']:
        t['count'] = round(t['share'] * len(cells))
    return record, top

def world_name(arm, seed): return f's8086_{arm}_s{seed}'

def run_world(arm, seed, out, *, release, deadline_s, git_commit=None, commit_output=lambda: None):
    receipt=launch_receipt(release); registration=released_registration(receipt)
    if arm not in registration.arms or seed not in SEEDS or not 0 < deadline_s:
        raise ValueError('invalid registered world/allocation')
    cfg = ARMS[arm]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    world = world_name(arm, seed)
    if any(out.glob(f'{world}.*')): raise FileExistsError('world already exists; no overwrite/retry')
    git_commit = git_commit or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    start = time.monotonic(); step = 0
    with (out / f'{world}.jsonl').open('x') as f:
        def emit(r):
            f.write(json.dumps(r, allow_nan=False) + '\n'); f.flush()
        try:
            soup = S2Soup(cfg['machine'], seed=seed, width=160, height=125, pair_count=8192,
                          noise_exp=4, z80_steps=128)
            initial = soup.read_soup().tobytes()
            emit({'kind': 'condition', 'schema': 's2-world/1', 'world': world, 'arm': arm,
                  'seed': seed, **cfg, 'machine_name': cfg['machine'], 'machine': soup.machine_provenance,
                  'steps': registration.horizon, 'width': 160, 'height': 125, 'pair_count': 8192,
                  'noise_exp': 4, 'mutation_attempts_per_step': 512, 'applied_updates': None,
                  'sample_steps': list(registration.sample_steps), 'snapshot_steps': list(registration.snapshots),
                  'registration':registration.as_dict(), 'git_commit': git_commit,
                  'initial_sha256': sha(initial), 'adapter': adapter_summary(), 'started': utc_now(),
                  'allocation': {'gpu': 'L4', 'cpu': 1, 'memory_mib': 8192, 'deadline_s': deadline_s},
                  'freeze_sha256': sha((HERE / 'results/s8086/prelaunch/freeze.json').read_bytes())})
            commit_output()
            for target in registration.sample_steps:
                while step < target:
                    # Leave 30 seconds for final checkpoint, storage and termination.
                    if time.monotonic() - start >= deadline_s - 30:
                        raise TimeoutError('spend/allocation stop; incomplete world')
                    n = min(50, target-step); soup.step(n); step += n
                soup.sync(); cells = soup.read_soup()
                counts, top = exact_tapes(cells, soup.read_hashes())
                hist = soup.read_byte_counts()
                emit({'kind': 'sample', 'step': step, **counts, **quasispecies_share(cells, top),
                      'byte_hist': hist.astype(int).tolist(), 'soup_sha256': sha(cells.tobytes()),
                      'pair_draws_total': step * 8192, 'active_pairs_total': soup.active_pairs_total,
                      'elapsed_s': time.monotonic()-start})
                if step in registration.snapshots:
                    raw = cells.tobytes(); name = f'{world}.t{step}.u8.br'
                    compressed = brotli.compress(raw, quality=9)
                    with (out / name).open('xb') as ck: ck.write(compressed)
                    emit({'kind': 'snapshot', 'step': step, 'file': name, 'shape': [20000,cfg['L']],
                          'raw_bytes': len(raw), 'compressed_bytes': len(compressed),
                          'sha256_raw': sha(raw), 'sha256_compressed': sha(compressed)})
                    commit_output()
            result = {'kind': 'summary', 'world': world, 'steps_run': step, 'samples': len(registration.sample_steps),
                      'snapshots': len(registration.snapshots), 'active_pairs_total': soup.active_pairs_total,
                      'wall_s': time.monotonic()-start, 'finished': utc_now()}
            emit(result); commit_output(); return result
        except BaseException as exc:
            # Never depend on another GPU read/checkpoint to record failure.
            # The independent dispatcher journal covers even broken storage.
            try:
                emit({'kind':'incomplete','steps_run':step,'error':type(exc).__name__,
                      'reason':str(exc),'last_acknowledged_progress':step})
                commit_output()
            except Exception:
                pass  # Preserve the original exception for the host terminal record.
            raise

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=ARMS, required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--deadline-s', type=float, required=True)
    a = ap.parse_args()
    print(json.dumps(run_world(a.arm, a.seed, a.out, release=a.release, deadline_s=a.deadline_s)))

"""Reproduce the mandatory provenance STOP on fixed census inputs; no Modal.

This is a necessary-condition falsification, not a completed classifier audit.
One rejected claimed write suffices to reject an episode under the requested
all-writes/two-substitutions/full-start-sequence contract. Both initial-memory
and isolated data-read interventions are recorded to avoid conflating a data
dependency with an earlier instruction-fetch dependency. The latter changes
only the value returned by that exact read, using unchanged vendored execution.
No successful first-byte test certifies the remaining writes.
"""
from __future__ import annotations

import copy
import csv
import gzip
import json
import time
from collections import Counter

import numpy as np

from algocell_exp.census8086_trace import DiagnosticMachine, mechanism, same_execution
from algocell_exp.zmachine import ZMachine, load_machine
from census8086 import pairs
from s8086 import HERE, PRE, sha


def main():
    bursts = []

    def burst(label, fn):
        start = time.monotonic()
        value = fn()
        elapsed = time.monotonic() - start
        if elapsed > 60:
            raise RuntimeError('local GPU burst exceeded 60s')
        pause = time.monotonic()
        time.sleep(elapsed)
        bursts.append({'label': label, 'burst_s': elapsed,
                       'pause_s': time.monotonic() - pause})
        return value

    def variant(observer, step, pos, kind):
        anchor = 'let p=zil_pos(addr); let v=census_real_read(addr);'
        assert observer.wgsl.count(anchor) == 1
        replacement = (f'let p=zil_pos(addr); var v=census_real_read(addr); '
                       f'if (kind=={kind}u && zil_step=={step}u && p=={pos}u) '
                       '{ v=v^(zil_params.seed & 255u); }')
        source = observer.wgsl.replace(anchor, replacement)
        manifest = copy.deepcopy(observer.m)
        manifest['sha256']['wgsl'] = sha(source.encode())
        machine = burst(f'compile-intervention-{observer.ring_bytes}-{step}-{pos}',
                        lambda: ZMachine(source, manifest, name='read-intervention',
                                         device=observer.dev))
        return machine, source, manifest

    def starts(log):
        return list(map(int, log[log[:, 1] == 4, 2]))

    def check(original, modified, write_step, write_pos, substitute):
        rows = modified[(modified[:, 0] == write_step) &
                        (modified[:, 1] == 2) & (modified[:, 2] == write_pos)]
        follows = len(rows) == 1 and int(rows[0, 3]) == substitute
        unchanged = starts(original) == starts(modified)
        return {'substituted_value': substitute,
                'written_values': list(map(int, rows[:, 3])),
                'written_byte_follows_exactly': bool(follows),
                'full_starts_unchanged': unchanged,
                'passes_necessary_condition': bool(follows and unchanged),
                'full_starts': starts(modified), 'trace': modified.tolist(),
                'trace_sha256': sha(modified.tobytes())}

    runs = []
    generated = {}
    identities = {}
    for L in (16, 32):
        base = burst(f'compile-frozen-{L}', lambda: load_machine(f'i8086_pair_maps_L{L}'))
        observer = burst(f'compile-observer-{L}',
                         lambda: DiagnosticMachine(base, events=8192, storage_trace=True))
        step, source_pos = (1, 0) if L == 16 else (17, 16)
        intervention, src, manifest = variant(observer, step, source_pos, 1)
        generated[f'L{L}'] = {'wgsl': src, 'manifest': manifest}
        identities[str(L)] = {'frozen_machine_sha256': base.wgsl_sha256,
                              'observer_sha256': observer.wgsl_sha256,
                              'read_intervention_sha256': intervention.wgsl_sha256}
        memory = pairs(np.tile(np.frombuffer(bytes.fromhex('07a5'), np.uint8), L // 2)[None], L)
        for budget in (128, 512):
            raw = burst(f'raw-{L}-{budget}', lambda: base.run(memory, budget=budget))
            observed = burst(f'observer-{L}-{budget}', lambda: observer.run(memory, budget=budget))
            assert same_execution(raw, observed, base.record_words)
            inactive = burst(f'inactive-intervention-{L}-{budget}',
                             lambda: intervention.run(memory, budget=budget, seed=0))
            assert np.array_equal(inactive.memory, observed.memory)
            assert np.array_equal(inactive.record, observed.record)
            logs = observer.logs(observed)
            rows = list(csv.DictReader((HERE / f'results/census8086/mechanisms_L{L}_b{budget}.csv').open()))
            archived = {int(r['partner']): r for r in rows if r['word'] == '07a5'}
            tests = {p: {'initial_memory': [], 'isolated_data_read': []} for p in range(17)}
            for mask in (0x55, 0xAA):
                changed = memory.copy()
                changed[:, source_pos] ^= mask
                input_logs = observer.logs(burst(f'initial-byte-{L}-{budget}-{mask}',
                                                 lambda: observer.run(changed, budget=budget)))
                read_logs = observer.logs(burst(f'isolated-read-{L}-{budget}-{mask}',
                                                lambda: intervention.run(memory, budget=budget, seed=mask)))
                for p, log in enumerate(logs):
                    assert sha(log.tobytes()) == archived[p]['trace_sha256']
                    assert mechanism(log, L)[0] == 'memory-fed one-pass copier'
                    writes = log[(log[:, 1] == 2) & (log[:, 2] >= L)]
                    first = writes[0]
                    assert int(first[0]) == step and int(first[2]) == L and int(first[3]) == 7
                    tests[p]['initial_memory'].append(check(log, input_logs[p], step, L, 7 ^ mask))
                    tests[p]['isolated_data_read'].append(check(log, read_logs[p], step, L, 7 ^ mask))
            for p, log in enumerate(logs):
                rejected = any(not r['passes_necessary_condition']
                               for r in tests[p]['isolated_data_read'])
                input_rejected = any(not r['passes_necessary_condition']
                                     for r in tests[p]['initial_memory'])
                runs.append({'L': L, 'budget': budget, 'partner': p, 'word': '07a5',
                             'old_label': archived[p]['mechanism'],
                             'required_label': 'other' if rejected or input_rejected else 'NOT CERTIFIED: remaining writes unaudited',
                             'isolated_read_rejects': rejected, 'initial_memory_rejects': input_rejected,
                             'baseline_full_starts': starts(log), 'baseline_trace_sha256': sha(log.tobytes()),
                             'tests': tests[p]})
            print(f'07 A5 L{L}/{budget}: isolated-read rejects '
                  f'{sum(any(not r["passes_necessary_condition"] for r in tests[p]["isolated_data_read"]) for p in range(17))}/17', flush=True)

    # Astra N1: vector bytes numerically equal flags, but never supply them.
    memory = np.frombuffer(bytes.fromhex('eb06f49002f10000bc10009df3a5f400'
                                         '02010000000000000000000000000000'), np.uint8)[None]
    base = burst('N1-machine', lambda: load_machine('i8086_pair_maps_L16'))
    observer = burst('N1-observer', lambda: DiagnosticMachine(base, events=8192, storage_trace=True))
    raw = burst('N1-raw', lambda: base.run(memory, budget=128))
    observed = burst('N1-trace', lambda: observer.run(memory, budget=128))
    assert same_execution(raw, observed, base.record_words)
    log = observer.logs(observed)[0]
    assert starts(log) == [0, 8, 11, 12, 61698]
    assert mechanism(log, 16)[0] == 'memory-fed one-pass copier'
    intervention, src, manifest = variant(observer, 3, 4, 1)
    generated['N1'] = {'wgsl': src, 'manifest': manifest}
    n1_tests = []
    for mask in (0x55, 0xAA):
        changed = observer.logs(burst(f'N1-intervention-{mask}',
                                      lambda: intervention.run(memory, budget=128, seed=mask)))[0]
        n1_tests.append(check(log, changed, 3, 16, 2 ^ mask))
    assert all(not r['written_byte_follows_exactly'] for r in n1_tests)

    zero_counts = []
    for L in (16, 32):
        for budget in (128, 512):
            rows = list(csv.DictReader((HERE / f'results/census8086/mechanisms_L{L}_b{budget}.csv').open()))
            exact = {int(r['index']) for r in csv.DictReader((HERE / f'results/census8086/words_L{L}_k2_b{budget}.csv').open())
                     if int(r['index']) and int(r['zero_match']) == L}
            old = Counter(r['mechanism'] for r in rows if r['k'] == '2' and r['partner'] == '0' and int(r['index']) in exact)
            witness = next(r for r in runs if r['L'] == L and r['budget'] == budget and r['partner'] == 0)
            assert int('07a5', 16) in exact
            after = old.copy()
            if witness['required_label'] == 'other':
                after['memory-fed one-pass copier'] -= 1
                after['other'] += 1
            zero_counts.append({'L': L, 'budget': budget, 'k': 2, 'baseline_exact_zero_counts': dict(old),
                                'counts_after_only_proven_07a5_rejection': dict(after),
                                'complete_reaudit': False,
                                'unchanged': False if witness['required_label'] == 'other' else 'not established'})

    assert any(r['required_label'] == 'other' and r['partner'] == 0 for r in runs)
    result = {'date': '2026-10-12', 'reviewed_head': 'a078045',
              'rule': 'every claimed write must follow s XOR 0x55 and s XOR 0xAA exactly, with full encounter starts unchanged',
              'disposition': 'MANDATORY STOP: 07 A5 and registered zero-partner provenance counts change',
              'scope': 'necessary-condition counterexample; STOP before class fixes, complete re-audit/differential, spend fixes, simulation or refreeze',
              'R2_altered': False, 'existing_classifier_unrepaired': True,
              'existing_freeze_superseded': False, 'release_authorized': False,
              'no_modal_calls': True, 'no_paid_compute': True, 'no_population_runs': True,
              'identities': identities, '07a5_encounters': len(runs),
              '07a5_definite_rejections': sum(r['required_label'] == 'other' for r in runs),
              '07a5_isolated_read_rejections': sum(r['isolated_read_rejects'] for r in runs),
              'zero_counts': zero_counts, 'runs': runs,
              'N1': {'parent': bytes(memory[0, :16]).hex(), 'partner': bytes(memory[0, 16:]).hex(),
                     'old_label': mechanism(log, 16)[0], 'required_label': 'other',
                     'baseline_full_starts': starts(log), 'trace': log.tolist(), 'tests': n1_tests},
              'local_gpu_bursts': bursts}
    with gzip.GzipFile(filename=str(PRE / 'causal_stop_validation.json.gz'), mode='wb', mtime=0) as f:
        f.write((json.dumps(result, separators=(',', ':')) + '\n').encode())
    summary = {k: v for k, v in result.items() if k not in ('runs', 'N1', 'local_gpu_bursts')}
    summary['N1'] = {k: v for k, v in result['N1'].items() if k not in ('trace', 'tests')}
    summary['gpu_bursts'] = len(bursts)
    summary['maximum_gpu_burst_s'] = max(r['burst_s'] for r in bursts)
    summary['archive_sha256'] = sha((PRE / 'causal_stop_validation.json.gz').read_bytes())
    summary['rejections_by_setting'] = [
        {'L': L, 'budget': budget,
         'initial_memory_rejections': sum(r['initial_memory_rejects'] for r in runs if r['L'] == L and r['budget'] == budget),
         'isolated_data_read_rejections': sum(r['isolated_read_rejects'] for r in runs if r['L'] == L and r['budget'] == budget),
         'zero_partner_rejected': next(r['required_label'] == 'other' for r in runs if r['L'] == L and r['budget'] == budget and r['partner'] == 0)}
        for L in (16, 32) for budget in (128, 512)]
    (PRE / 'causal_stop_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for name, data in generated.items():
        (PRE / f'causal_stop_{name}.wgsl').write_text(data['wgsl'])
        (PRE / f'causal_stop_{name}.manifest.json').write_text(json.dumps(data['manifest'], indent=2) + '\n')
    print(result['disposition'], flush=True)


if __name__ == '__main__':
    main()

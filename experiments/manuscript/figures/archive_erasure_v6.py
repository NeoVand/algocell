"""Archive a COMPLETE ordered erasure trace; local frozen Z80 executor only.

Run with the project's Python. Uses the existing observational provenance
executor, checks its output against the uninstrumented shader, and enforces
58-second active bursts through capacity_common.Bursts. No ISA implementation.
"""
from pathlib import Path
import json
import sys

import numpy as np

EXP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXP))
from capacity_common import Bursts, Z80Provenance, hx, parse, sha, write

OUT = Path(__file__).resolve().parent / 'out/v6/data'


def main():
    import pandas as pd
    source = EXP / 'results/branching/branching_summary.csv'
    tape = parse(pd.read_csv(source).query("key == 'ldir32'").tape.iloc[0])
    # A payload mutation outside the protected core, selected before execution.
    position, marker = 8, 0xA7
    tape[position] = marker
    burst = Bursts(OUT)
    tracer = Z80Provenance(32, burst, events=4096)
    rows, states = [], []
    ancestor_map = np.arange(64)
    for encounter in (1, 2):
        partner = np.zeros(32, np.uint8)
        pair = np.concatenate((tape, partner))[None]
        output, regs = tracer.run(pair, steps=128)
        state = pair[0].copy()
        origins = np.arange(64)
        ancestors = ancestor_map.copy()
        # Include initial and terminal states; every event in between is kept.
        states += [{'encounter': encounter, 'phase': 'initial', 'tape': hx(state),
                    'sources': ';'.join(map(str, origins))}]
        for order, (step, kind, address, byte, origin) in enumerate(tracer.logs(regs)[0]):
            p, k = int(address), int(kind)
            before, prior_source = int(state[p]), int(origins[p])
            if k == 2:
                state[p], origins[p] = byte, origin
                ancestors[p] = ancestor_map[int(origin)]
            rows.append({'encounter': encounter, 'order': order, 'step': int(step),
                         'kind': ['fetch', 'read', 'write'][k], 'address': p,
                         'byte': int(byte), 'source': int(origin),
                         'prior_byte': before, 'prior_source': prior_source,
                         'ancestral_source': int(ancestor_map[int(origin)]),
                         'marker_sources_remaining': int((ancestors == position).sum()),
                         'marker_values_remaining': int((state == marker).sum())})
        assert np.array_equal(state, output[0])
        assert np.array_equal(origins, tracer.labels(regs)[0])
        states += [{'encounter': encounter, 'phase': 'final', 'tape': hx(state),
                    'sources': ';'.join(map(str, origins))}]
        assert not (origins == position).any() and not (state == marker).any()
        tape = output[0, 32:].copy()  # actual descendant reproduces next
        ancestor_map = np.r_[ancestors[32:], np.arange(64,96)]
    write(OUT / 'erasure_ordered.csv', rows)
    write(OUT / 'erasure_states.csv', states)
    for name in ('erasure_ordered.csv', 'erasure_states.csv', 'bursts.csv'):
        path=OUT/name
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    (OUT / 'erasure_manifest.json').write_text(json.dumps({
        'status': 'EXPLORATORY', 'source': str(source.relative_to(EXP)),
        'source_sha256': sha(source.read_bytes()), 'executor_sha256': tracer.base_hash,
        'observational_shader_sha256': tracer.source_hash, 'position': position,
        'marker': marker, 'L': 32, 'steps_per_encounter': 128,
        'events': len(rows), 'complete': True,
        'partner': 'all zero; benign chemistry',
        'ordered_csv_sha256': sha((OUT / 'erasure_ordered.csv').read_bytes())
    }, indent=2) + '\n')
    print(f'Archived {len(rows)} ordered events across two encounters; raw executor equality passed.')


if __name__ == '__main__':
    main()

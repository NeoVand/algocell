"""Meaningful causal-label checks, including equality without copy provenance."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from algocell_exp.census8086_trace import same_execution
from algocell_exp.zmachine import load_machine
from capacity_common import Bursts, HERE, UNKNOWN, partners
from evolved6502_trace import Trace6502
from evolved6502_retention import geometry


def test_frozen_6502_forward_copy_and_arithmetic_labels():
    """A true forward loop qualifies; unchanged ADC output is not a copy."""
    import json
    burst=Bursts(HERE/'results/evolved6502')
    base=burst.run('test-compile',lambda:load_machine('m6502_pair_maps_L16'))
    trace=burst.run('test-trace-compile',lambda:Trace6502(base))
    seed=np.array([0x68,0xba,0x95,0x08,0x50,0xfa,0x33,0x77],np.uint8)
    t=np.tile(seed,2);Q=partners(16);pairs=np.concatenate((np.tile(t,(17,1)),Q),1)
    raw=burst.run('test-forward-raw',lambda:base.run(pairs,budget=128))
    r=burst.run('test-forward-trace',lambda:trace.run(pairs,budget=128))
    assert same_execution(raw,r,base.record_words)
    ops=json.loads((HERE/'results/census6502/m6502_opcodes.json').read_text())['opcodes']
    for log,labels in zip(trace.logs(r),trace.labels(r)):
        g=geometry(log,labels,16,ops)
        assert g['eligible']==1 and g['d']==8 and g['core_c']==6
        assert g['free_positions']=='6;7'
    a=np.zeros(32,np.uint8);a[:6]=[0xa9,5,0x69,0,0x85,0x10]
    raw=burst.run('test-arithmetic-raw',lambda:base.run(a[None],budget=3))
    r=burst.run('test-arithmetic-trace',lambda:trace.run(a[None],budget=3))
    assert same_execution(raw,r,base.record_words)
    assert r.memory[0,16]==5
    assert trace.labels(r)[0,16]==UNKNOWN


def test_reverse_geometry_is_not_relabelled_forward():
    """Direction and displacement exclusions are determined without outcomes."""
    import json
    ops=json.loads((HERE/'results/census6502/m6502_opcodes.json').read_text())['opcodes']
    # Copy stream 2->10, 1->9, 0->8 is constant-d but reverse.
    log=np.array([[0,2,10,7,2,2],[1,2,9,6,1,1],[2,2,8,5,0,0]],np.uint32)
    g=geometry(log,np.arange(32,dtype=np.uint32),16,ops)
    assert not g['eligible']
    assert 'not_sequential_forward' in g['reason']

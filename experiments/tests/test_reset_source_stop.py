"""CPU checks of archived independent reset-source/R2-stop evidence."""
import numpy as np
import pytest
from s8086 import PRE,read_json
from prefix_causality8086 import compare
from algocell_exp.census8086_trace import mechanism


@pytest.mark.parametrize('L,budget',[(16,128),(16,512),(32,128),(32,512)])
def test_51b9_first_reset_CX_write_passes_both_alternatives(L,budget):
    data=read_json(PRE/'reset_stop_focus.json.gz')
    case=next(r for r in data['witnesses'] if r.get('word')=='51b9' and r['L']==L and r['budget']==budget)
    assert case['label']=='literal writer' and case['first_claim']==[3,['literal',['reset',1,0],0]]
    first=next(e for e in case['checks'] if e['write_event']==3)
    assert first['source']==['reset',1]
    assert [r['written'] for r in first['checks']]==[0x55,0xaa]
    assert all(r['prefix_identical'] and r['passes'] for r in first['checks'])


@pytest.mark.parametrize('mask',[0x55,0xaa])
def test_independent_AD50_intervention_rejects_change_before_later_write(mask):
    case=read_json(PRE/'reset_stop_focus.json.gz')['direct_gate_witness']
    log=np.asarray(case['baseline_trace'],np.uint32)
    check=next(r for r in case['checks'] if r['mask']==mask)
    changed=np.asarray(check['trace'],np.uint32)
    assert mechanism(log,16)[0]=='memory-fed one-pass copier'
    fresh=compare(log,changed,case['write_event'],case['source_event'],mask)
    assert fresh['written']==fresh['substitute'] and not fresh['prefix_identical'] and not fresh['passes']
    # This failure occurs before the claimed write, not only in a later lap.
    assert log[86].tolist()==[16,1,16,0,1]
    assert changed[86,2] in (5,26) and 86<case['write_event']


def test_N1_and_far_jump_are_other_under_corrected_fail_safe():
    witnesses=read_json(PRE/'reset_stop_focus.json.gz')['witnesses']
    assert {r['name']:r['label'] for r in witnesses if 'name' in r}=={'N1':'other','far_jump':'other'}
    assert all(r['label']=='other' for r in witnesses if r.get('word')=='b951')

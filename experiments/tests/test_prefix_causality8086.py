"""The corrected prefix gate and fail-safe; CPU synthetic traces only."""
import numpy as np
import pytest

from prefix_causality8086 import UNKNOWN, claims, compare, label
from test_s8086_repairs import trace_instruction as inst


def copy_trace():
    return np.asarray(inst(0,0,[0xa4],reads=[(0,0xa4)],writes=[(16,0xa4)])
                      + inst(1,16,[0x90]),np.uint32)


@pytest.mark.parametrize('mask',[0x55,0xaa])
def test_later_control_divergence_is_permitted(mask):
    original=copy_trace(); modified=original.copy()
    source=int(np.flatnonzero(original[:,1]==1)[0])
    write=int(np.flatnonzero(original[:,1]==2)[0])
    modified[source,3]^=mask;modified[write,3]^=mask
    modified[-2,2]=33
    result=compare(original,modified,write,source,mask)
    assert result['passes'] and result['first_structural_divergence'] is not None


@pytest.mark.parametrize('kind',[0,1,2,4])
def test_every_prefix_bus_address_and_instruction_start_must_match(kind):
    original=copy_trace(); modified=original.copy()
    source=int(np.flatnonzero(original[:,1]==1)[0]); write=int(np.flatnonzero(original[:,1]==2)[0])
    modified[source,3]^=0x55;modified[write,3]^=0x55
    modified[np.flatnonzero(original[:,1]==kind)[0],2]+=1
    assert not compare(original,modified,write,source,0x55)['passes']


def test_exact_substituted_byte_is_required():
    original=copy_trace(); source=int(np.flatnonzero(original[:,1]==1)[0]);write=int(np.flatnonzero(original[:,1]==2)[0])
    modified=original.copy();modified[source,3]^=0x55
    assert not compare(original,modified,write,source,0x55)['passes']


def test_reset_push_has_no_immediate_or_data_source():
    log=np.asarray(inst(0,0,[0x51],writes=[(31,0),(0,0)])+inst(1,1,[0xb9,0x51,0xb9]),np.uint32)
    cc,reason=claims(log,16)
    assert cc==[(3,UNKNOWN)] and label(log,16,cc)=='other'


def test_immediate_literal_claim_names_operand_fetch():
    log=np.asarray(inst(0,0,[0xb9,0x51,0xb9])+inst(1,3,[0x51],writes=[(31,0x51),(0,0xb9)]),np.uint32)
    cc,reason=claims(log,16)
    assert cc==[(8,('literal',1,0x51))]
    assert label(log,16,cc)=='literal writer'


def test_n1_rejects_vector_traffic_before_emitting_channels():
    log=np.asarray(inst(0,0,[0xf3,0xa5],reads=[(4,2),(5,0xf1),(6,0),(7,0)],
                        writes=[(16,2),(17,0xf1),(14,0),(15,0),(12,14),(13,0)]),np.uint32)
    cc,reason=claims(log,16)
    assert reason=='interrupt_or_vector_traffic'
    assert all(tag==UNKNOWN for _,tag in cc) and label(log,16,cc)=='other'


def test_unverified_rmw_write_is_never_silently_omitted():
    log=np.asarray(inst(0,0,[0,0],reads=[(16,0)],writes=[(16,0)]),np.uint32)
    cc,reason=claims(log,16)
    assert len(cc)==1 and cc[0][1]==UNKNOWN and label(log,16,cc)=='other'

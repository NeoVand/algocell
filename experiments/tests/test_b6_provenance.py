"""B6 total opcode inventory and regression per WGSL instruction class."""
import numpy as np
import pytest
from algocell_exp.census8086_trace import mechanism
from algocell_exp.i8086_provenance_spec import OPCODE_CLASS,CLASSES
from i8086_value_tags import value_channels,reference_mechanism
from test_s8086_repairs import trace_instruction as inst
import s8086 as S

@pytest.mark.parametrize('cls', sorted(CLASSES))
def test_every_instruction_class_against_independent_value_bank(cls):
    # Seed memory provenance in every GPR and segment; exercise each admitted
    # opcode/subgroup/mode then probe every byte source. Numeric equality to a
    # previously loaded value must never restore a killed origin.
    for op in [i for i,c in enumerate(OPCODE_CLASS) if c==cls]:
      for modrm in (0x00,0x06,0x36,0xc0,0xc4,0xf0,0xfc,0xff):
        rows=[];step=0
        for r in range(8):
          rows+=inst(step,step*4,[0x8b,(r<<3)|6,14,0],[(14,0),(15,0)]);step+=1
        for s in range(4):
          rows+=inst(step,step*4,[0x8e,(s<<3)|6,14,0],[(14,0),(15,0)]);step+=1
        rows+=inst(step,48,[op,modrm,0,0,0],[(14,0),(15,0),(12,0),(13,0)]);step+=1
        for r in range(8): rows+=inst(step,100+step,[0x50+r],writes=[(30,0),(31,0)]);step+=1
        for s in range(4): rows+=inst(step,100+step,[6+8*s],writes=[(30,0),(31,0)]);step+=1
        log=np.asarray(rows,np.uint32)
        assert mechanism(log,16,audit_channels=True)==value_channels(log,16)
        assert mechanism(log,16)[0]==reference_mechanism(log,16)
    anchor=CLASSES[cls][0]
    for L in (16,32): assert anchor.lower() in (S.HERE/f'algocell_exp/machines/i8086_pair_maps_L{L}.wgsl').read_text().lower()


def test_astra_immediate_far_jump_overwrites_cs_without_touching_other_segments():
    rows=inst(0,0,[0x8e,0x0e,14,0],[(14,2),(15,0)])
    rows+=inst(1,36,[0xea,10,0,2,0]);rows+=inst(2,42,[0x0e],writes=[(31,2),(0,0)])
    assert mechanism(np.asarray(rows,np.uint32),16)[0]=='literal writer'
    # A MOV-to-DS tag survives the far jump; CS and DS have different sources.
    rows=inst(0,0,[0x8e,0x1e,14,0],[(14,2),(15,0)])
    rows+=inst(1,4,[0xea,10,0,2,0]);rows+=inst(2,42,[0x1e],writes=[(31,2),(0,0)])
    assert mechanism(np.asarray(rows,np.uint32),16)[0]=='memory-fed one-pass copier'

@pytest.mark.parametrize('op',[0xc4,0xc5])
def test_register_far_load_trap_cannot_create_empty_then_literal_tag(op):
    rows=inst(0,0,[op,0xc0]);rows+=inst(1,4,[0x40]);rows+=inst(2,5,[0x50],writes=[(30,0),(31,0)])
    assert mechanism(np.asarray(rows,np.uint32),16)[0]=='other'

@pytest.mark.parametrize('op',[0xa4,0xa5,0xa6,0xa7,0xaa,0xab,0xac,0xad,0xae,0xaf])
def test_zero_rep_preserves_loaded_pointer_and_accumulator(op):
    rows=inst(0,0,[0x8b,0x36,14,0],[(14,2),(15,0)])
    rows+=inst(1,4,[0xf3,op]);rows+=inst(2,6,[0x56],writes=[(30,2),(31,0)])
    assert mechanism(np.asarray(rows,np.uint32),16)[0]=='memory-fed one-pass copier'


def test_tf_interrupt_after_admitted_opcode_invalidates_cs():
    rows=inst(0,0,[0x8e,0x0e,14,0],[(14,2),(15,0)])
    rows+=inst(1,4,[0x90],reads=[(4,0),(5,0),(6,2),(7,0)],writes=[(0,0)]*6)
    rows+=inst(2,6,[0x0e],writes=[(31,2),(0,0)])
    assert mechanism(np.asarray(rows,np.uint32),16)[0]=='other'

@pytest.mark.parametrize('cls',['decimal_adjust','lea','sign_extend','immediate_far_call','flags_stack','flags_transfer','return','interrupt','shift','aam_aad_salc','xlat','escape','io','near_call','muldiv_group','unsupported'])
def test_unadmitted_writer_class_cannot_retain_cs_memory_provenance(cls):
    op=next(i for i,c in enumerate(OPCODE_CLASS) if c==cls)
    rows=inst(0,0,[0x8e,0x0e,14,0],[(14,2),(15,0)])
    rows+=inst(1,4,[op,0xc0,0,0,0]);rows+=inst(2,10,[0x0e],writes=[(31,2),(0,0)])
    assert mechanism(np.asarray(rows,np.uint32),16)[0]=='other'

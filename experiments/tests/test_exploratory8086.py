"""EXPLORATORY / post hoc verifier checks against frozen machine semantics."""
import numpy as np
import pytest
from exploratory8086 import burst
from exploratory8086_law import copy_certificate
from algocell_exp.zmachine import load_machine
from algocell_exp.census8086_trace import DiagnosticMachine,same_execution

@pytest.fixture(scope='module')
def machine():
    m=burst('EXPLORATORY test machine compile',lambda:load_machine('i8086_pair_maps_L16'))
    t=burst('EXPLORATORY test trace compile',lambda:DiagnosticMachine(m,events=2048,storage_trace=True))
    return m,t

def execute(machine,parent,budget):
    m,t=machine; mem=np.concatenate((parent,np.zeros(16,np.uint8)))[None,:]
    raw=burst('EXPLORATORY test raw',lambda:m.run(mem,budget=budget))
    obs=burst('EXPLORATORY test observer',lambda:t.run(mem,budget=budget))
    assert same_execution(raw,obs,m.record_words)
    return raw,t.logs(obs)[0]

@pytest.mark.parametrize('op,register',[(7,'es'),(15,'cs'),(23,'ss'),(31,'ds'),(95,'di')])
def test_pop_semantics(machine,op,register):
    p=np.zeros(16,np.uint8);p[:3]=[op,0xa5,0x61]
    r,log=execute(machine,p,1)
    assert r.reg(register)[0]==0x61a5 and r.reg('sp')[0]==3
    assert log[log[:,1]==1,2].tolist()==[1,2]

@pytest.mark.parametrize('prefix,expected_register,expected',[
    ([0x0c,0x1f],'ax',0x1f),([0x31,0xf1],'cx',0),
    ([0xe5,0xa7],'ax',0xffff),([0x21,0xc0],'ax',0)])
def test_other_originally_unmodelled_semantics(machine,prefix,expected_register,expected):
    p=np.zeros(16,np.uint8);p[:len(prefix)]=prefix
    r,_=execute(machine,p,1)
    assert r.reg(expected_register)[0]==expected

@pytest.mark.parametrize('prefix,core',[
    ([0x07,0xa5,0x61,0xfd],4),
    ([0x0c,0x1f,0x07,0xa5,0x61,0xfd],6),
    ([0x31,0xf1,0x07,0xa5,0x61,0xfd],6),
    ([0xe5,0xa7,0x07,0xa5,0x61,0xfd],6)])
def test_full_tape_certificate_and_segment_dependencies(machine,prefix,core):
    p=np.arange(16,dtype=np.uint8);p[:len(prefix)]=prefix
    r,log=execute(machine,p,128);g=copy_certificate(p,log,16)
    assert np.array_equal(r.memory[0,16:],p)
    assert g['d']==16 and g['source_map']==list(range(16))
    assert g['K']==list(range(core)) and g['F']==list(range(core,16))
    assert 1 in g['K'] and 2 in g['K']

def test_pop_di_four_byte_overlap_and_alias(machine):
    p=np.tile(np.array([0x5f,0xa4,0x61,0xfd],np.uint8),4)
    r,log=execute(machine,p,128);g=copy_certificate(p,log,16)
    assert np.array_equal(r.memory[0,16:],p)
    assert g['d']==4 and g['K']==[0,1,2,3] and g['F']==[]
    assert g['source_map']==list(range(4))*4 and g['E']==list(range(4,16))

def test_arithmetic_memory_write_is_not_semantic_copy(machine):
    p=np.tile(np.array([0x21,0x00,0xff,0x58],np.uint8),4)
    _,log=execute(machine,p,128)
    with pytest.raises(ValueError,match='interfering_or_unpaired_write'):copy_certificate(p,log,16)

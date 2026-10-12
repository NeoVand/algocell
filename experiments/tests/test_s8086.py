"""Fixed-input S2 protocol checks. No soup stepping and no population observations."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest
import s8086 as S
import law8086 as Law
import score8086 as Score
from run8086 import exact_tapes
from dispatch8086 import SpendGuard
from benchmark8086 import input_manifest,fixed_inputs

def archive_certificate(name,certificate):
    import gzip
    path=S.PRE/f'{name}.json.gz'; raw=json.dumps(certificate,sort_keys=True).encode()
    if path.exists(): assert gzip.decompress(path.read_bytes())==raw
    else: path.write_bytes(gzip.compress(raw,mtime=0))

def judgement(open=True, confined=False, memory=True, gen2=.3):
    return {'gen2':gen2,'heritable':gen2 is not None and gen2>=.3,'open':open,'confined':confined,
            'open_memory_fed':memory,'fetch_class':'open' if open else 'confined' if confined else 'intermediate',
            'majority_mechanism':'memory-fed one-pass copier','encounters':[]}

def test_rng_fixtures():
    assert json.loads((S.PRE/'rng_fixtures.json').read_text())==S.safe(S.fixtures())
    a=('A',8086001,16,'route-trace','90'*16)
    assert np.array_equal(S.partners('A',8086001,16,'90'*16),S.rng(*a).integers(0,256,(16,16),dtype=np.uint8))
    assert S.key(*a)!=S.key('A',8086001,16,'culture','90'*16)
    with pytest.raises(ValueError): S.key('B',1,16,'culture')

def test_schedule_ties_collision():
    assert len(S.samples())==585 and len(set(S.samples()))==585
    assert S.samples()[0]==0 and S.samples()[-1]==3000000 and len(S.PERSISTENCE)==11
    cells=np.array([[2,2],[1,1],[2,2],[1,1]],np.uint8)
    r,_=exact_tapes(cells,np.ones(4,np.uint32))
    assert [t['tape'] for t in r['top3']]==['0101','0202']
    assert [t['count'] for t in r['top3']]==[2,2]

def test_full_horizon_persistence_share_and_zero():
    records=[{'step':s,'top3':[('02',.005),('01',.005)]} for s in S.samples()]
    result=Score.endpoints(records,lambda _:judgement())
    assert result['t_rep']==0 and result['first']['baseline'] and result['first']['tape']=='01'
    assert result['persistent_open_memory'] and result['same_genotype_persistence']
    records[-2]['top3']=[('01',.004999)]
    result=Score.endpoints(records,lambda _:judgement())
    assert result['open_memory_occurrence'] and not result['persistent_open_memory']
    result=Score.endpoints([{'step':50,'top3':[('01',.005)]}],lambda _:judgement(gen2=None))
    assert not result['scoreable'] and result['t_rep'] is None
    result=Score.endpoints([{'step':0,'top3':[('01',.0049)]}],lambda _:judgement())
    assert result['t_rep'] is None

def test_persistence_after_both_first_events():
    records=[{'step':s,'top3':[('01',.1)]} for s in S.samples()]
    records[-1]['top3']=[('02',.1)]
    result=Score.endpoints(records,lambda t:judgement(open=t=='02',confined=t=='01',memory=t=='02'))
    assert result['t_rep']==result['t_closed']==0
    assert result['open_memory_occurrence'] and not result['persistent_open_memory']

def test_route_cutoffs_separate_arms():
    def make(arm,seed,open=True,confined=False):
        return {'world':{'arm':arm,'seed':seed,'L':S.ARMS[arm]['L'],'scoreable':True,'horizon':S.HORIZON,
                         't_rep':0,'t_closed':None,'first':judgement(open,confined),
                         'open_memory_occurrence':seed==S.SEEDS[0],'persistent_open_memory':False}}
    rows=[make('A',s,i<16,i>=16) for i,s in enumerate(S.SEEDS)]
    v=Score.arm_verdict(rows,'A')
    assert v['R0'] and v['R1'] and not v['kill'] and not v['R2a'] and not v['R2b']
    rows=[make('A',s,i<15,i>=15) for i,s in enumerate(S.SEEDS)]
    v=Score.arm_verdict(rows,'A'); assert not v['R1'] and v['kill']
    assert Score.arm_verdict(rows[:-1],'A')['R1'] is None

def test_candidate_cap_and_multiplicity():
    tapes=[bytes([i]*16).hex() for i in range(30)]
    records=[{'source':'final-top','tape':t} for t in tapes[:3]]
    records += [{'source':'random-cell','tape':t} for t in tapes]*2
    certs={t:{'families':{'L1':{'status':'eligible'}}} for t in tapes}
    result=Law.selection(records,certs,'A',S.SEEDS[0],16,'L1')
    assert len(result['selected'])==16 and set(tapes[:3])<=set(result['selected'])
    assert len(result['cap_excluded'])==14 and len(records)==63
    assert result==Law.selection(list(reversed(records)),certs,'A',S.SEEDS[0],16,'L1')

def test_spend_guard_reserves_every_remaining_liability(tmp_path):
    guard=SpendGuard(tmp_path/'spend.jsonl',{'a':2.,'b':58.},overhead=0)
    guard.reserve('a',2.)
    with pytest.raises(FileExistsError): guard.reserve('a',2.)
    guard.start('a','fc-fixed','ap-fixed'); guard.settle('a',1.)
    guard.reserve('b',58.)
    with pytest.raises(ValueError): guard.settle('b',59.)
    refuse=SpendGuard(tmp_path/'refuse.jsonl',{'a':2.,'b':58.},overhead=.01)
    with pytest.raises(RuntimeError): refuse.reserve('a',2.)
    raw=(tmp_path/'spend.jsonl').read_text().replace('fc-fixed','fc-tampered')
    (tmp_path/'spend.jsonl').write_text(raw)
    with pytest.raises(ValueError,match='hash chain'): guard.transact({'kind':'metered','app_id':'x','usd':1})

def test_fixed_inputs_archive():
    assert input_manifest()==json.loads((S.PRE/'benchmark_inputs.json').read_text())
    for L in (16,32):
        archive=np.load(S.PRE/f'fixed_L{L}.npz')
        for name,value in zip(('soup','pairs','active'),fixed_inputs(L)):
            assert np.array_equal(archive[name],value)

def test_known_payload_core_erasure_mismatches_reject():
    expected=np.arange(16,dtype=np.uint8); g={'d':16}; roots=list(range(16))
    assert Law.check_descendant(expected,expected,roots,g) is None
    for p in (0,6,15):
        wrong=expected.copy(); wrong[p]^=1
        assert Law.check_descendant(wrong,expected,roots,g)=='exact_offspring_mismatch'
    assert Law.check_descendant(expected,expected,[16,*range(1,16)],g)=='parent_source_identity_mismatch'
    assert Law.verdict([],{},True)=='evolved law untested'
    assert Law.verdict(['x'],{'x':{'status':'failed application'}},False)=='failed application'
    assert Law.verdict(['x'],{},True)=='incomplete/invalid'

def test_overflow_to_other():
    from algocell_exp.census8086_trace import mechanism
    assert mechanism(None,16)==('other','trace_overflow')
    with pytest.raises(Law.Unresolved,match='overflow'): Law.copy_certificate(np.zeros(16,np.uint8),None,16)

@pytest.fixture(scope='module')
def machines():
    from algocell_exp.zmachine import load_machine,ZMachine
    from algocell_exp.census8086_trace import DiagnosticMachine
    # Every local GPU API call is synchronous and followed by equal pause.
    old=ZMachine.run
    def burst(self,*args,**kwargs): return S.fixed_burst(lambda:old(self,*args,**kwargs))
    ZMachine.run=burst
    result={}
    for L in (16,32):
        m=S.fixed_burst(lambda:load_machine(f'i8086_pair_maps_L{L}'))
        t=S.fixed_burst(lambda:DiagnosticMachine(m,events=8192)); result[L]=(m,t)
    yield result
    ZMachine.run=old

def test_exact_census_replay_both_phases(machines):
    from census8086 import pairs,matches
    from algocell_exp.census8086_trace import same_execution,mechanism
    wanted={'b951','51b9','07a5','a507','ff30','30ff','ad50','50ad'}
    for L,(m,t) in machines.items():
        for budget in (128,512):
            census={r['word']:r for r in csv.DictReader((S.HERE/f'results/census8086/words_L{L}_k2_b{budget}.csv').open()) if r['word'] in wanted}
            mechanisms={(r['word'],int(r['partner'])):r for r in csv.DictReader((S.HERE/f'results/census8086/mechanisms_L{L}_b{budget}.csv').open()) if r['word'] in wanted}
            for word in sorted(wanted):
                p=np.tile(np.frombuffer(bytes.fromhex(word),np.uint8),L//2)[None]
                mem=pairs(p,L); raw=m.run(mem,budget=budget); obs=t.run(mem,budget=budget)
                assert same_execution(raw,obs,m.record_words)
                b=raw.memory[:,L:]; r=census[word]
                assert S.sha(b.tobytes())==r['output_sha256']
                assert matches(b.reshape(1,17,L),p,2)[0].tolist()==list(map(int,r['matches'].split(';')))
                for v,log in enumerate(t.logs(obs)):
                    if (word,v) in mechanisms:
                        assert mechanism(log,L)[0]==mechanisms[word,v]['mechanism']

def test_constructed_rep_and_non_rep_certificates(machines):
    from census8086_law import GEOMETRIES,parents
    for L,d in GEOMETRIES:
        m,t=machines[L]
        for op in (0xa4,0xa5):
            c=Law.extract(parents(d,L,op),m,t)
            assert c['families']['L1']['status']=='eligible',c['families']
            assert c['families']['L2']['C']==list(range(6))
            assert c['families']['L2']['free']==list(range(6,d))
            archive_certificate(f'certificate_REP_L{L}_d{d}_{op:02x}',c)
    L,d=32,16
    p=np.arange(d,dtype=np.uint8)
    p[:10]=[0xbf,d,0,0xb9,255,255,0xad,0xab,0xe2,0xfc]
    p=np.tile(p,2); m,t=machines[L]
    c=Law.extract(p,m,t)
    assert c['families']['L1']['status']=='excluded'
    assert c['families']['L2']['status']=='eligible',c['families']
    assert c['families']['L2']['C']==list(range(10))
    assert c['families']['L2']['source_map']==[i%d for i in range(L)]
    archive_certificate('certificate_nonREP',c)

def test_all_ten_full_constructed_law_assays(machines,tmp_path):
    result=Law.controls(tmp_path/'controls',machines)
    assert len(result)==10
    assert all(r['status']=='complete-zero-mismatch' for r in result.values())
    assert sum(r['encounters'] for r in result.values())==1_185_792
    path=S.PRE/'constructed_control_validation.json'
    if path.exists(): assert json.loads(path.read_text())==result
    else: S.write_new(path,result)

def test_l2_nonzero_source_counterexample(machines):
    # Fixed-input review counterexample, not an evolved candidate. Preserve
    # the approved L2 definition and show its proposed prediction is false.
    p=np.arange(16,dtype=np.uint8)
    p[:9]=[0xbe,12,0,0xbf,28,0,0x49,0xf3,0xa4]
    p=np.tile(p,2); m,t=machines[32]
    c=Law.extract(p,m,t); g=c['families']['L2']
    assert g['status']=='eligible',g
    assert g['C']==list(range(9)) and g['source_start']==12
    assert set(g['source_map'])==set(range(12,28))
    assert Law.check_descendant(p,p,g['source_map'],g)=='parent_source_identity_mismatch'
    archive_certificate('certificate_L2_source_counterexample',c)

def test_byte_granular_count_dependency(machines):
    p=np.zeros(32,np.uint8)
    p[:11]=[0xbf,32,0,0x8b,0x0e,30,0,0x88,0xc1,0xf3,0xa4]; p[31]=1
    m,t=machines[32]; c=Law.extract(p,m,t); g=c['families']['L2']
    assert g['status']=='eligible',g
    assert 31 in g['C'] and 31 not in g['free']
    assert 30 not in g['C']  # CL overwritten; CH still controls REP.
    archive_certificate('certificate_byte_count_dependency',c)

def test_snapshot_and_record_identity_boundaries(tmp_path):
    import brotli
    cells=np.tile(np.arange(16,dtype=np.uint8),(20000,1)); raw=cells.tobytes()
    compressed=brotli.compress(raw); (tmp_path/'fixed.br').write_bytes(compressed)
    record={'file':'fixed.br','raw_bytes':len(raw),'shape':[20000,16],
            'sha256_raw':S.sha(raw),'sha256_compressed':S.sha(compressed)}
    assert np.array_equal(Score.load_snapshot(tmp_path/'input.jsonl',record,16),cells)
    wrong={**record,'sha256_raw':'0'*64}
    with pytest.raises(ValueError,match='snapshot hash'): Score.load_snapshot(tmp_path/'input.jsonl',wrong,16)
    with pytest.raises(ValueError,match='unsafe'): Score.load_snapshot(tmp_path/'input.jsonl',{**record,'file':'../fixed.br'},16)

def test_no_launch_without_review_release(tmp_path):
    # Direct invocation also cannot produce a population before the freeze.
    import run8086
    with pytest.raises((FileNotFoundError,ValueError)):
        run8086.run_world('A',8086001,tmp_path,release=tmp_path/'absent.json',deadline_s=3600)
    assert not list(tmp_path.glob('s8086*'))

def test_family_arm_incomplete_cannot_pass():
    row={'arm':'A','seed':S.SEEDS[0],'world':'fixed','scoreable':True,'horizon':S.HORIZON,'L':16}
    eligibility=[{'world':row,'candidates':[],'certificates':{},'selection':{'L2':{'eligible':[],'selected':[],'cap_excluded':[]}}}]
    assert Law.family_report(eligibility,{},True,'L2',arm='A')['verdict']=='incomplete/invalid'

@pytest.mark.parametrize('position',[0,6,15])
def test_injected_pipeline_mismatch_stops(position,monkeypatch):
    from types import SimpleNamespace
    from census8086_law import parents
    parent=parents(16,16,0xa4)
    class Fault:
        def run(self,mem,**kw):
            result=mem.copy(); result[:,16:]=parent
            result[0,16+position]^=1
            return SimpleNamespace(memory=result)
    monkeypatch.setattr(Law,'diagnosis',lambda *a:{'status':'failed application','saved_input':True})
    emitted=[]
    geometry={'d':16,'C':list(range(6)),'free':list(range(6,16)),'source_map':list(range(16))}
    r=Law.assay_case(parent,geometry,'A',S.SEEDS[0],Fault(),None,emitted.append)
    assert r['status']=='failed application' and r['encounters']==1
    assert emitted[-1]['kind']=='candidate-mismatch' and emitted[-1]['saved_input']

def test_guard_nan_rejected_without_mutation(tmp_path):
    with pytest.raises(ValueError): SpendGuard(tmp_path/'ledger',{'x':1},overhead=float('nan'))
    guard=SpendGuard(tmp_path/'ledger',{'x':1},overhead=0)
    with pytest.raises(ValueError): guard.reserve('x',float('nan'))
    assert not (tmp_path/'ledger').exists()

def test_frozen_pair_bridge_and_active_total(machines):
    from run8086 import S2Soup
    m,_=machines[16]
    s=S.fixed_burst(lambda:S2Soup(m,device=m.dev,pair_count=128))
    _,pair,_=fixed_inputs(16); pair=pair[:128]; active=np.zeros(128,np.uint32); active[:37]=1
    def dispatch():
        q=s.device.queue
        q.write_buffer(s.pair_data_buf,0,pair); q.write_buffer(s.pair_active_buf,0,active)
        q.write_buffer(s.tally,0,np.zeros(1,np.uint32)); q.write_buffer(s.params_bufs[0],0,s._params(123))
        q.write_buffer(s.zparams_bufs[0],0,s.bridge.machine_params(128,123,s.masks))
        enc=s.device.create_command_encoder()
        s.bridge.encode_execute(enc,s.bind_groups[0],s.machine_bind_groups[0],128)
        q.submit([enc.finish()]); s.sync()
        return s.read_pair_memory(),s.read_records(),int(np.frombuffer(q.read_buffer(s.tally),np.uint32)[0])
    memory,record,total=S.fixed_burst(dispatch)
    expected=m.run(pair,budget=np.where(active!=0,128,0).astype(np.uint32),seed=123)
    assert total==37 and np.array_equal(memory.reshape(128,32),expected.memory) and np.array_equal(record,expected.record)

def test_complete_record_and_incomplete_boundaries(tmp_path):
    import brotli
    cells=np.tile(np.arange(16,dtype=np.uint8),(20000,1)); raw=cells.tobytes(); compressed=brotli.compress(raw)
    top={'tape':cells[0].tobytes().hex(),'sha256':S.sha(cells[0].tobytes()),'count':20000,'share':1.}
    condition={'kind':'condition','schema':'s2-world/1','arm':'A','seed':S.SEEDS[0],**S.ARMS['A'],
               'machine_name':S.ARMS['A']['machine'],'steps':S.HORIZON,'width':160,'height':125,
               'pair_count':8192,'noise_exp':4,'mutation_attempts_per_step':512,
               'sample_steps':S.samples(),'snapshot_steps':list(S.SNAPSHOTS),'initial_sha256':S.sha(raw)}
    records=[condition]
    for step in S.samples():
        records.append({'kind':'sample','step':step,'top3':[top],'pair_draws_total':step*8192,
                        'active_pairs_total':step*1000,'soup_sha256':S.sha(raw)})
        if step in S.SNAPSHOTS:
            name=f'fixed{step}.br'; (tmp_path/name).write_bytes(compressed)
            records.append({'kind':'snapshot','step':step,'file':name,'raw_bytes':len(raw),
                            'shape':[20000,16],'sha256_raw':S.sha(raw),'sha256_compressed':S.sha(compressed)})
    records.append({'kind':'summary','steps_run':S.HORIZON,'samples':585})
    path=tmp_path/'synthetic-fixed-record.jsonl'
    def write(rows): path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    write(records); c,parsed,snap=Score.read_input(path)
    assert len(parsed)==585 and snap['step']==S.HORIZON
    write(records[:-1])
    with pytest.raises(ValueError,match='incomplete'): Score.read_input(path)
    write([r for r in records if r.get('step')!=50])
    with pytest.raises(ValueError,match='sample'): Score.read_input(path)
    altered=json.loads(json.dumps(records)); altered[1]['top3'][0]['count']=19999
    write(altered)
    with pytest.raises(ValueError,match='tape/count'): Score.read_input(path)

def test_instruction_and_retained_rep_boundaries(machines):
    from census8086 import pairs
    from algocell_exp.census8086_trace import same_execution,mechanism
    m,t=machines[16]
    # Sequential byte and multi-byte ring laps, forward aliases, near/backward,
    # far transfers, halt and coherent self modification. Check full records.
    cases=[('90'*16,128),('b900'*8,128),('eb1e'+'90'*14,128),
           ('ebfe'+'90'*14,128),('e90000'+'90'*13,128),
           ('ea00000100'+'90'*11,128),('f4'+'90'*15,128),
           ('c6060900f4'+'90'*11,128),('bf080049f3a4'+'90'*10,3)]
    for tape,budget in cases:
        p=np.frombuffer(bytes.fromhex(tape),np.uint8)[None]; mem=pairs(p,16)
        r=m.run(mem,budget=budget); q=t.run(mem,budget=budget)
        assert same_execution(r,q,m.record_words)
        if tape.startswith('90') or tape.startswith('b900'): assert mechanism(t.logs(q)[0],16)[0]=='other'
        if tape.startswith('f4'): assert int(r.steps[0])==1 and int(r.halt_reason[0])==1
        if tape.startswith('bf'):
            log=t.logs(q)[0]; assert int((log[:,1]==2).sum())==1

def test_culture_n64_trace_n16(machines):
    from algocell_exp import executor as E
    from census8086_law import parents
    m,t=machines[16]; ex=S.fixed_burst(lambda:E.ZilionExecutor(m.name,device=m.dev))
    with E.using(ex): r=Score.judge_tape(parents(8,16,0xa4).tobytes().hex(),'A',S.SEEDS[0],ex,t)
    assert len(r['encounters'])==16 and len(r['route_partners'])==16
    assert all(len(bytes.fromhex(p))==64*16 for p in r['culture_partners'])
    assert r['heritable'] and r['gen2']==1 and r['diagnostic_identity']

def test_archived_dz_endpoint_control():
    # Independent cached historical cultures, own old streams/estimator.
    import pandas as pd
    targets=pd.read_csv(S.HERE/'results/dial/dial_worlds.csv')
    targets=targets[targets.variant=='lp0'].set_index('seed')
    for seed in range(9001,9011):
        archive=json.loads((S.HERE/f'results/s6502/v2/control/worlds/lp0_s{seed}.json').read_text())
        row=archive['world']; cache={r['tape']:r for r in archive['judged_tapes']}
        first=row['first_tape']; closed=next(t for t,r in cache.items() if r['heritable'] and r['confined'])
        records=[{'step':row['t_rep'],'top3':[(first,.5)]},
                 {'step':row['t_closed'],'top3':[(closed,.5)]}]
        if records[0]['step']==records[1]['step']: records=records[:1]
        def judge(t):
            r=cache[t]
            return {**r,'open_memory_fed':False}
        result=Score.endpoints(records,judge)
        assert result['t_rep']==targets.loc[seed,'t_rep']
        assert result['t_closed']==targets.loc[seed,'t_closed']
        assert result['first']['tape']==targets.loc[seed,'first_tape']
        assert result['first']['open']==targets.loc[seed,'first_open']

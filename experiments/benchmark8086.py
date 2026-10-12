"""Frozen inputs only: restore pair, mask and soup BEFORE EVERY dispatch.

No output feeds another world step. The shared world kernels and observation
counter are timed, including preparation/writeback/mutation, on fixed inputs.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import brotli
import numpy as np
import wgpu
from s8086 import PRE,sha,safe,ARMS,SEEDS,HORIZON,RATE,CAP,DEFAULT_REGISTRATION

def fixed_inputs(L):
    from census8086_law import parents
    tapes=[np.tile(np.frombuffer(bytes.fromhex(w),np.uint8),L//2) for w in ('b951','07a5','ff30','ad50')]
    tapes += [parents(L,L,op) for op in (0xa4,0xa5)]
    # Frozen uniform-byte inputs; no population seed or trajectory.
    random=np.random.Generator(np.random.PCG64(0x8086BEEF+L))
    soup=random.integers(0,256,(20000,L),dtype=np.uint8)
    pair=random.integers(0,256,(8192,2*L),dtype=np.uint8)
    for i in range(8192): pair[i,:L]=tapes[i%len(tapes)] if i%8<6 else pair[i,:L]
    active=np.ones(8192,np.uint32)  # worst-case full active slot count
    return soup,pair,active

def input_manifest():
    return {'schema':'s2-fixed-benchmark/1','seed':'PCG64(0x8086BEEF+L)','active':'all 8192 slots',
            'restored_before_every_dispatch':True,'inputs':{
                str(L):{'soup_sha256':sha(s.tobytes()),'pairs_sha256':sha(p.tobytes()),'active_sha256':sha(a.tobytes())}
                for L in (16,32) for s,p,a in [fixed_inputs(L)]},
            'world_panels':{str(L):{name:sha(p.tobytes()) for name,p in world_panels(L).items()} for L in (16,32)},
            'law_panels':'complete single/joint/erasure/serial workload: six-byte MOVSB and constructed LODS/STOS/LOOP; d=L; original S2 fixed streams'}

def nonrep_parent(L):
    p=np.arange(L,dtype=np.uint8)
    p[:10]=[0xbf,L,0,0xb9,255,255,0xad,0xab,0xe2,0xfc]
    return p

def world_panels(L):
    from census8086_law import parents,GEOMETRIES
    _,base,_=fixed_inputs(L)
    panels={'mixture':base}
    for word in ('b951','07a5','ff30','ad50'):
        p=base.copy(); p[:,:L]=np.tile(np.frombuffer(bytes.fromhex(word),np.uint8),L//2)
        panels[word]=p
    for LL,d in GEOMETRIES:
        if LL!=L: continue
        for op in (0xa4,0xa5):
            p=base.copy(); p[:,:L]=parents(d,L,op); panels[f'REP-d{d}-{op:02x}']=p
    p=base.copy(); p[:,:L]=nonrep_parent(L); panels['nonREP']=p
    return panels

def benchmark():
    from run8086 import S2Soup,exact_tapes
    from algocell_exp import executor as E
    from algocell_exp.census8086_trace import DiagnosticMachine
    from score8086 import judge_tape
    from census8086_law import parents
    from law8086 import extract,assay_case
    from census8086 import pairs as census_pairs
    from algocell_exp.census8086_trace import same_execution
    from s8086 import fixtures,diagnostic_for
    start=time.perf_counter(); records=[]
    manifest=json.loads((PRE/'benchmark_inputs.json').read_text())
    if manifest!=input_manifest(): raise ValueError('fixed benchmark input identity changed')
    if json.loads((PRE/'rng_fixtures.json').read_text())!=safe(fixtures()):
        raise ValueError('Modal NumPy/RNG byte fixtures differ from freeze')
    for arm,cfg in ARMS.items():
        L=cfg['L']; t0=time.perf_counter()
        s=S2Soup(cfg['machine'],seed=1,noise_exp=4,z80_steps=128)
        ex=E.ZilionExecutor(cfg['machine'],device=s.device)
        m=ex.machine_for(L,zero_halts=False,maps=True); diagnostic=diagnostic_for(m)
        compilation=time.perf_counter()-t0
        print(json.dumps({'kind':'fixed-compile','L':L,'seconds':compilation}),flush=True)
        backend_validation=[]
        for word in ('b951','51b9','07a5','a507','ff30','30ff','ad50','50ad'):
            parent=np.tile(np.frombuffer(bytes.fromhex(word),np.uint8),L//2)
            memory=census_pairs(parent[None],L)
            for budget in (128,512):
                uninstrumented=m.run(memory,budget=budget)
                traced=diagnostic.run(memory,budget=budget)
                if not same_execution(uninstrumented,traced,m.record_words):
                    raise ValueError('L4/software diagnostic census replay differs')
                backend_validation.append({'word':word,'budget':budget,'partners':17,
                                           'output_sha256':sha(uninstrumented.memory[:,L:].tobytes()),
                                           'record_sha256':sha(uninstrumented.record.tobytes()),'same_execution':True})
        fixed,pair,active=fixed_inputs(L); q=s.device.queue; B=wgpu.BufferUsage
        def buf(data): return s.device.create_buffer_with_data(data=data,usage=B.COPY_SRC)
        sb,pb,ab=buf(fixed),buf(pair),buf(active)
        q.write_buffer(s.zparams_bufs[0],0,s.bridge.machine_params(8192,123,s.masks))
        q.write_buffer(s.params_bufs[0],0,s._params(123))
        def dispatch(n):
            remaining=n
            while remaining:
                submission=min(50,remaining); q.write_buffer(s.tally,0,np.zeros(1,np.uint32))
                done=0
                while done<submission:
                    chunk=min(s.ring,submission-done)
                    for k in range(chunk):
                        seed=s.cpu_rng.next_u32()
                        q.write_buffer(s.params_bufs[k],0,s._params(seed))
                        q.write_buffer(s.zparams_bufs[k],0,s.bridge.machine_params(8192,seed,s.masks))
                    enc=s.device.create_command_encoder()
                    for k in range(chunk):
                        bg=s.bind_groups[k]
                        enc.copy_buffer_to_buffer(sb,0,s.soup_buf,0,sb.size)
                        s.bridge.dispatch(enc,'clear_collision',bg,-(-20000//256))
                        s.bridge.dispatch(enc,'prepare_batch',bg,128)
                        enc.copy_buffer_to_buffer(pb,0,s.pair_data_buf,0,pb.size)
                        enc.copy_buffer_to_buffer(ab,0,s.pair_active_buf,0,ab.size)
                        s.bridge.encode_execute(enc,bg,s.machine_bind_groups[k],8192)
                        s.bridge.dispatch(enc,'absorb_results',bg,128)
                        s.bridge.dispatch(enc,'mutate_soup',bg,8)
                    q.submit([enc.finish()]); done+=chunk
                total=int(np.frombuffer(q.read_buffer(s.tally),np.uint32)[0])
                if total!=8192*submission: raise ValueError('active tally mismatch')
                remaining-=submission
            s.sync()
        rates=[]; world_times=[]
        for name,p in world_panels(L).items():
            pb=buf(p); expected=m.run(p,budget=128); dispatch(1)
            actual=s.read_pair_memory().reshape(8192,2*L); raw=s.read_records()
            if not np.array_equal(actual,expected.memory) or not np.array_equal(raw,expected.record):
                raise ValueError('fixed world dispatch memory/full-record mismatch')
            timing=[]
            for _ in range(3):
                t0=time.perf_counter(); dispatch(128); timing.append(128/(time.perf_counter()-t0))
            rates.extend(timing); world_times.append({'panel':name,'world_steps_per_s':timing})
            print(json.dumps({'kind':'fixed-world-timing','L':L,'panel':name,'steps_per_s':timing}),flush=True)
        # Count/readback/serialization uses fixed data, no endpoint classifier.
        q.write_buffer(s.soup_buf,0,fixed)
        from algocell_exp.metrics import quasispecies_share
        t0=time.perf_counter(); cells=s.read_soup(); record,top=exact_tapes(cells,s.read_hashes()); hist=s.read_byte_counts()
        json.dumps({**record,**quasispecies_share(cells,top),'byte_hist':hist.astype(int).tolist()})
        sample_s=time.perf_counter()-t0
        t0=time.perf_counter(); brotli.compress(cells.tobytes(),quality=9); snapshot_s=time.perf_counter()-t0
        measured=[]
        # Complete capped law workload for a fixed largest-d control. Same
        # batching, RNG, source checks and gzipped append format as future assays.
        import gzip,tempfile
        for name,tape,family in (('REP',parents(L,L,0xa4),'L1'),('nonREP',nonrep_parent(L),'L2')):
            with E.using(ex):
                t0=time.perf_counter(); judge_tape(tape.tobytes().hex(),arm,SEEDS[0],ex,diagnostic)
                judge_s=time.perf_counter()-t0
            t0=time.perf_counter(); certificate=extract(tape,m,diagnostic); extraction_s=time.perf_counter()-t0
            geometry=certificate['families'][family]
            if geometry['status']!='eligible': raise ValueError('constructed control failed extraction')
            with tempfile.TemporaryDirectory() as tmp:
                t0=time.perf_counter()
                with gzip.open(Path(tmp)/'law.jsonl.gz','wt') as f:
                    def emit(row): f.write(json.dumps(safe(row))+'\n'); f.flush()
                    law=assay_case(tape,geometry,arm,SEEDS[0],m,diagnostic,emit)
                law_s=time.perf_counter()-t0
            if law['status']!='complete-zero-mismatch': raise ValueError('fixed law assay mismatch')
            measured.append({'panel':name,'culture_route_s_per_tape':judge_s,'extraction_s_per_tape':extraction_s,
                             'law_encounters':law['encounters'],'law_s':law_s,
                             'law_encounters_per_s':law['encounters']/law_s})
        records.append({'arm':arm,'L':L,'compile_s':compilation,'world_steps_per_s':rates,
                        'world_panels':world_times,'assay_panels':measured,
                        'backend_validation':backend_validation,
                        'sample_s':sample_s,'snapshot_s':snapshot_s,
                        'culture_route_s_per_tape':max(r['culture_route_s_per_tape'] for r in measured),
                        'extraction_s_per_tape':max(r['extraction_s_per_tape'] for r in measured),
                        'law_encounters_per_s':min(r['law_encounters_per_s'] for r in measured),
                        'diagnostic_identity':True,'adapter':dict(s.device.adapter.info),
                        'diagnostic_adapter':dict(diagnostic.dev.adapter.info),
                        'machine':m.describe(),'judged_fixed_input_sha256':sha(tape.tobytes())})
    import sys,platform
    return safe({'schema':'s2-benchmark/1','input_manifest':manifest,'records':records,
                 'dependencies':{'python':sys.version,'platform':platform.platform(),
                                 'numpy':np.__version__,'wgpu':wgpu.__version__,'brotli':brotli.__version__},
                 'function_wall_s':time.perf_counter()-start})

def project(bench,preparation_bound,registration=DEFAULT_REGISTRATION):
    """Worst input/candidate ceilings, both separate law families, 20% slowdown
    and 10% runner margin; startup and CPU scoring/serialization are included.
    """
    from lifecycle8086 import envelope_seconds,LIFECYCLE_EXTRA_S
    jobs={}; execution={}; components=[]; deadlines={}; slowest=0
    n=len(registration.sample_steps); k=len(registration.snapshots)
    for r in bench['records']:
        arm,L=r['arm'],r['L']
        if arm not in registration.arms: continue
        world_s=registration.horizon/min(r['world_steps_per_s'])/.8*1.1 + n*r['sample_s']*1.5 + k*r['snapshot_s']*1.5
        startup=max(120,r['compile_s']*2)
        deadline=math.ceil(world_s+startup+60); deadlines[arm]=deadline; slowest=max(slowest,deadline)
        world_usd=envelope_seconds(deadline)*RATE
        # At most 3 tapes at each of 585 samples plus 69 source records.
        tapes=n*3+69
        scoring_s=tapes*r['culture_route_s_per_tape']*1.5+startup
        extracting_s=69*r['extraction_s_per_tape']*1.5+startup
        # L1 and L2 each select <=16 per world. L2 admits up to L-1
        # free coordinates. Never deduplicate across different certificates.
        l1=(((L-6)*256)+512)*25
        l2=(((L-1)*256)+512)*25
        # Max at d=L; smaller d adds erased*17 but removes free*25.
        law_s=16*(l1+l2)/r['law_encounters_per_s']*1.5+startup
        components.append({'arm':arm,'worlds':20,'world_seconds':deadline,'samples':n,'snapshots':k,'score_tapes_cap':tapes,'score_seconds':scoring_s,
                           'extraction_seconds':extracting_s,'law_seconds':law_s,
                           'L1_encounters_per_case_cap':l1,'L2_encounters_per_case_cap':l2})
        for seed in SEEDS:
            for stage,seconds in (('world',deadline),('score',scoring_s+60),('extract',extracting_s+60),('law',law_s+60)):
                job=f'{stage}:{arm}:{seed}'
                execution[job]=math.ceil(seconds)
                jobs[job]=envelope_seconds(seconds)*RATE
    rate=min(r['law_encounters_per_s'] for r in bench['records'])
    execution['controls']=math.ceil(1_185_792/rate*1.5+120)
    jobs['controls']=envelope_seconds(execution['controls'])*RATE
    diagnosis_s=1290*max(r['extraction_s_per_tape'] for r in bench['records'])*2+240
    execution['diagnosis']=math.ceil(diagnosis_s)
    jobs['diagnosis']=envelope_seconds(diagnosis_s)*RATE
    jobs['preparation']=preparation_bound
    # Packed full-control outputs measured locally; twofold volume margin,
    # plus2 GB for checkpoints/certificates, one paid month and one export.
    sizes=json.loads((PRE/'output_size_validation.json').read_text())
    output_gib=(185_505_792*sizes['worst_gzip_bytes_per_encounter']*2)/(1024**3)+2
    storage_reserve=max(5.,math.ceil(output_gib*(.09+.04)+.5))
    total=sum(jobs.values())+storage_reserve
    return {'schema':'s2-cost/2-lifecycle','registration':registration.as_dict(),'population_worlds':20*len(registration.arms),
            'execution_deadline_s':execution,'lifecycle_extra_s':LIFECYCLE_EXTRA_S,
            'lifecycle_policy':{'startup_timeout_s':120,'startup_attempt_allowance':2,'max_containers_per_job':1,
                'termination_grace_s':60,'watchdog_poll_s':1,'bounded_rpc_allowance_s':60,
                'one_app_per_job':True,'restart_renews_clock':False,'settlement_requires_stopped_and_no_tasks':True},'gpu':'L4','cpu':1,'memory_gib':8,'rate_usd_per_s':RATE,
            'rates_source':'https://modal.com/pricing','rates_checked':'2026-10-12',
            'components':components,'job_liabilities_usd':jobs,'projection_usd':total,
            'world_deadline_s':deadlines,'longest_world_container_s':slowest,
            'storage_termination_reserve_usd':storage_reserve,'projected_output_gib':output_gib,
            'storage_assumption':'twofold measured gzip volume +2 GiB overhead; one paid month and one export; no credits',
            'diagnostic_replays_cap':1290,'cap_usd':CAP,'launch_fits':total<=CAP,
            'L1_total_evolved_encounters_cap':40_960_000*len(registration.arms),'L2_total_evolved_encounters_cap':51_200_000*len(registration.arms),
            'constructed_encounters':1_185_792,
            'warning':'Fixed-input timing cannot prove a worst-case population runtime; allocation stops are incomplete experiments.'}

import math

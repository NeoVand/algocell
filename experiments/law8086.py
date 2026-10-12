"""S2 L1/L2: extraction -> committed data freeze -> exact interventions.

The dependency verifier is conservative: an unmodelled instruction leaves an
unresolved certificate, never a claim of payload independence. L2 permits REP
setups and non-REP loops; numeric PC comparisons are never used.
"""
from __future__ import annotations
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
import numpy as np
from algocell_exp.census8086_trace import DiagnosticMachine, same_execution
from census8086 import panel, pairs
from s8086 import HERE, key, rng, sha, safe, write_new, committed, verify_freeze,diagnostic_for

class Unresolved(ValueError): pass

def copy_certificate(parent, log, L, *, baseline=True):
    """Static dependency admission plus ordered byte-copy/source tracking.

    All admitted instructions have explicit semantics below. Pointer/count/
    condition taint is promoted to the held-fixed core, including core data.
    Pairing is semantic (MOVS, LODS->STOS, MOV load->store), not byte equality.
    Unsupported instructions cannot produce an eligibility verdict.
    """
    if log is None: raise Unresolved('trace_overflow')
    if np.any((log[:,1] < 2) & (log[:,4] != 0)):
        raise ValueError('original_partner_read')
    origins = list(range(2*L))
    regs = [[set(),set()] for _ in range(8)]  # low/high byte dependencies
    held = [[None,None] for _ in range(8)]
    def sources(r): return regs[r][0] | regs[r][1]
    flags = set(); dependencies = set(); fetched_roots = set(); moves = []
    pending_rep = None
    for step in np.unique(log[:,0]):
        group = log[log[:,0] == step]
        fetched = [int(v) for k,p,v,o in group[:,1:] if k == 0]
        for k,p,v,o in group[:,1:]:
            if k == 0: fetched_roots.add(origins[int(p)])
        decodes = [int(v) for k,p,v,o in group[:,1:] if k == 3]
        rr = [(int(p), int(v), origins[int(p)]) for k,p,v,o in group[:,1:] if k == 1]
        ww = [(int(p), int(v)) for k,p,v,o in group[:,1:] if k == 2]
        if not decodes:
            if not rr and not ww: continue
            if pending_rep is None: raise Unresolved('access_without_decode')
            op, ins = pending_rep
        else:
            op = decodes[-1]
            ins = fetched.copy()
            prefixes = []
            while ins and ins[0] in (0x26,0x2e,0x36,0x3e,0xf0,0xf1,0xf2,0xf3):
                prefixes.append(ins.pop(0))
            if any(p in prefixes for p in (0xf0,0xf1)):
                raise Unresolved('lock_or_undefined_prefix')
            if not ins:
                if pending_rep is None or op != pending_rep[0]: raise Unresolved('missing_decode_bytes')
                ins = pending_rep[1]
            if ins[0] != op: raise Unresolved('decode_byte_discrepancy')
            if fetched:
                if op in (0xa4,0xa5) and any(p in prefixes for p in (0xf2,0xf3)):
                    pending_rep = (op, ins)
                else: pending_rep = None
        copy_reads = None
        if 0xb8 <= op <= 0xbf:
            regs[op-0xb8] = [set(),set()]; held[op-0xb8] = [None,None]
        elif 0x40 <= op <= 0x4f:
            flags = sources(op & 7).copy()
            regs[op & 7] = [flags.copy(),flags.copy()]; held[op & 7] = [None,None]
        elif op in (0x90,0xfc): pass
        elif op in (0xa4,0xa5):
            dependencies |= sources(6) | sources(7) | sources(1)
            copy_reads = rr
        elif op in (0xac,0xad):
            dependencies |= sources(6)
            for i,r in enumerate(rr): held[0][i]=r; regs[0][i]={r[2]}
        elif op in (0xaa,0xab):
            dependencies |= sources(7)
            copy_reads = held[0][:1 if op==0xaa else 2]
        elif op in (0x88,0x89,0x8a,0x8b):
            if len(ins) < 2: raise Unresolved('missing_modrm')
            modrm = ins[1]; mod, reg, rm = modrm>>6, (modrm>>3)&7, modrm&7
            byte = op in (0x88,0x8a)
            if byte and (reg >= 4 or (mod == 3 and rm >= 4)):
                raise Unresolved('high_byte_register_dependency')
            if mod == 3:
                dest, source = (reg,rm) if op in (0x8a,0x8b) else (rm,reg)
                for i in range(1 if byte else 2):
                    regs[dest][i] = regs[source][i].copy(); held[dest][i] = held[source][i]
            else:
                address_regs = ((3,6),(3,7),(5,6),(5,7),(6,),(7,),(5,),(3,))[rm]
                if mod == 0 and rm == 6: address_regs = ()
                for r in address_regs: dependencies |= sources(r)
                if op in (0x8a,0x8b):
                    for i,r in enumerate(rr): held[reg][i]=r; regs[reg][i]={r[2]}
                else: copy_reads = held[reg][:1 if byte else 2]
        elif op in (0xe0,0xe1,0xe2,0xe3):
            dependencies |= sources(1)
            if op in (0xe0,0xe1): dependencies |= flags
            held[1] = [None,None]
        elif 0x70 <= op <= 0x7f:
            dependencies |= flags
        elif op in (0xeb,0xe9): pass
        elif op == 0xf4:
            if rr or ww: raise Unresolved('halt_access')
        else:
            raise Unresolved(f'unmodelled_dependency_opcode:{op:02x}')
        if ww:
            if copy_reads is None or len(copy_reads) != len(ww) or any(r is None for r in copy_reads):
                raise ValueError('interfering_or_unpaired_write')
            for (src,value,root),(dst,wvalue) in zip(copy_reads,ww):
                if value != wvalue: raise ValueError('copy_value_mismatch')
                moves.append((src,dst,root)); origins[dst] = root
        elif copy_reads:
            raise ValueError('copy_read_without_write')
    if not moves: raise ValueError('no_copy_kernel')
    displacement = (moves[0][1]-moves[0][0]) % (2*L)
    if not 1 <= displacement <= L or L % displacement:
        raise ValueError('invalid_displacement')
    d = displacement
    if any((dst-src) % (2*L) != d for src,dst,_ in moves):
        raise ValueError('nonconstant_displacement')
    if any(b[0] != (a[0]+1) % (2*L) for a,b in zip(moves,moves[1:])):
        raise ValueError('nonforward_kernel')
    if {dst for _,dst,_ in moves if dst >= L} != set(range(L,2*L)):
        raise ValueError('incomplete_partner_coverage')
    if any(root >= L for root in origins[L:]): raise ValueError('partner_ancestry')
    if baseline and not np.array_equal(parent, np.tile(parent[:d],L//d)):
        raise ValueError('nonperiodic_baseline')
    if any(p >= d for p in fetched_roots | dependencies):
        raise ValueError('original_erased_coordinate_code_or_control_dependency')
    C = sorted(fetched_roots | dependencies)
    # Values at dependencies are held fixed at every repeated core position.
    # Executed stream therefore remains unchanged for arbitrary free payload.
    if not C or max(C)+1 > d: raise ValueError('core_exceeds_seed')
    return {'d': d, 'c': max(C)+1, 'C': C, 'free': sorted(set(range(d))-set(C)),
            'source_start': moves[0][0], 'byte_moves': len(moves),
            'source_map': origins[L:], 'moves_sha256': sha(np.asarray(moves,dtype='<u4').tobytes()),
            'trace_sha256': sha(log.astype('<u4').tobytes())}

def extract(parent, machine, diagnostic=None):
    parent = np.asarray(parent,dtype=np.uint8); L = len(parent)
    if L not in (16,32) or machine.name != f'i8086_pair_maps_L{L}':
        raise ValueError('wrong_parent_or_machine')
    t = diagnostic or diagnostic_for(machine)
    mem = pairs(parent[None], L)
    raw = machine.run(mem,budget=128); obs = t.run(mem,budget=128)
    if not same_execution(raw,obs,machine.record_words): raise Unresolved('observation_changed_execution')
    result = {'parent': parent.tobytes().hex(), 'L': L, 'machine': machine.describe(),
              'partners': [p.tobytes().hex() for p in panel(L)],
              'input_sha256': sha(mem.tobytes()), 'record_sha256': sha(raw.record.tobytes()),
              'output_sha256': sha(raw.memory.tobytes()), 'families': {}}
    logs = t.logs(obs); geometry = []
    try:
        for log in logs: geometry.append(copy_certificate(parent,log,L))
        g = geometry[0]
        for v in geometry[1:]:
            if any(v[k] != g[k] for k in ('d','C','free','source_start','source_map')):
                raise ValueError('context_dependent_certificate')
        if not np.all(raw.memory[:,L:] == parent): raise ValueError('baseline_descendant_not_closed')
        result['families']['L2'] = {'status': 'eligible', **g, 'panel_geometry': geometry,
                                   'proof': 'explicit opcode dependency transfer + ordered semantic byte copies'}
    except (ValueError,Unresolved) as e:
        result['families']['L2'] = {'status': 'unresolved' if isinstance(e,Unresolved) else 'excluded',
                                   'reason': str(e)}
    core = parent[:6]
    d = int(core[1]) + 256*int(core[2])
    canonical = (core[0] == 0xbf and list(core[3:5]) == [0x49,0xf3]
                 and core[5] in (0xa4,0xa5) and 6 <= d <= L and L%d == 0
                 and (core[5] != 0xa5 or d%2 == 0))
    l2 = result['families']['L2']
    if (canonical and l2['status'] == 'eligible' and l2['d'] == d
            and l2['C'] == list(range(6)) and l2['source_start'] == 0
            and all(g['byte_moves'] == 126*(1 if core[5] == 0xa4 else 2) for g in geometry)):
        result['families']['L1'] = {'status': 'eligible', **l2}
    else:
        result['families']['L1'] = {'status': 'excluded', 'reason': 'outside_verified_six_byte_REP_family'}
    result['ordered_traces'] = [None if log is None else log.tolist() for log in logs]
    return safe(result)

def selection(candidates, certificates, arm, seed, L, family):
    eligible = {t for t,c in certificates.items() if c['families'][family]['status'] == 'eligible'}
    priority = {r['tape'] for r in candidates if r['source'] != 'random-cell'} & eligible
    if len(priority) > 5: raise ValueError('more than five distinct event/final tapes')
    randoms = sorted(({r['tape'] for r in candidates if r['source'] == 'random-cell'} & eligible)-priority,
                     key=lambda t:(key(arm,seed,L,'law-rank',t),bytes.fromhex(t)))
    selected = sorted(priority) + randoms[:16-len(priority)]
    return {'selected': selected, 'cap_excluded': sorted(eligible-set(selected)), 'eligible': sorted(eligible)}

def freeze_candidates(scored, out, machine, diagnostic=None, tick=lambda:None):
    """No interventions. All culture-positive candidates, open or confined."""
    verify_freeze()
    world = scored['world']; certs = {}
    judged = {r['tape']:r for r in scored['judged_tapes']}
    for tape in sorted({r['tape'] for r in scored['candidates']}):
        tick()
        if judged[tape]['heritable']:
            certs[tape] = extract(np.frombuffer(bytes.fromhex(tape),np.uint8),machine,diagnostic)
    selections = {f:selection(scored['candidates'],certs,world['arm'],world['seed'],world['L'],f)
                  for f in ('L1','L2')}
    result = {'schema':'s2-eligibility/1','world':world,'certificates':certs,'selection':selections,
              'candidates':scored['candidates'], 'scored_sha256':sha(json.dumps(scored,sort_keys=True).encode()),
              'extractor_sha256':sha(Path(__file__).read_bytes())}
    write_new(out,result); return result

def controls(out, machines, tick=lambda:None):
    """All ten constructed controls, budget128, original census assignments."""
    from census8086_law import GEOMETRIES,parents
    Path(out).mkdir(parents=True,exist_ok=False); results={}
    for L,d in GEOMETRIES:
        m,t=machines[L]
        for op in (0xa4,0xa5):
            tick(); p=parents(d,L,op); cert=extract(p,m,t)
            geometry=cert['families']['L1']
            if geometry['status']!='eligible' or cert['families']['L2']['C']!=list(range(6)):
                raise RuntimeError('constructed control extraction mismatch')
            name=f'L{L}_d{d}_{op:02x}'
            with gzip.open(Path(out)/f'{name}.jsonl.gz','xt') as f:
                def emit(row): f.write(json.dumps(safe(row))+'\n')
                emit({'kind':'input','certificate':cert})
                results[name]=assay_case(p,geometry,'A' if L==16 else 'B',8086001,m,t,emit,
                                        control={'op':op},tick=tick)
                emit({'kind':'summary',**results[name]})
    write_new(Path(out)/'summary.json',results); return results

def family_report(eligibility, outcomes, controls_complete, family, *, arm, exclude_L1=False):
    from s8086 import SEEDS,HORIZON,ARMS
    eligibility=[d for d in eligibility if d['world']['arm']==arm]
    selected=[]; cases={}; worlds=set(); unresolved=0; coverage=Counter()
    for data in eligibility:
        world=data['world']['world']
        coverage['source_records']+=len(data['candidates'])
        coverage['random_cells']+=sum(r['source']=='random-cell' for r in data['candidates'])
        coverage['unique_candidates']+=len({r['tape'] for r in data['candidates']})
        coverage['culture_positive']+=len(data['certificates'])
        for cert in data['certificates'].values():
            unresolved+=cert['families'][family]['status']=='unresolved'
        for source in data['candidates']:
            if source['source']!='random-cell': continue
            score=source['culture']['gen2_score']
            coverage['random_cell_unknowns']+=score is None
            coverage['random_cell_culture_positive']+=score is not None and score>=.3
            cert=data['certificates'].get(source['tape'])
            coverage['random_cell_eligible']+=bool(cert and cert['families'][family]['status']=='eligible'
                and (not exclude_L1 or cert['families']['L1']['status']!='eligible'))
        for tape in data['selection'][family]['eligible']:
            if not exclude_L1 or data['certificates'][tape]['families']['L1']['status']!='eligible':
                coverage['eligible']+=1
        for tape in data['selection'][family]['selected']:
            if exclude_L1 and data['certificates'][tape]['families']['L1']['status']=='eligible': continue
            name=f'{world}:{tape}'; selected.append(name); worlds.add(world)
            if tape in outcomes.get(world,{}).get(family,{}): cases[name]=outcomes[world][family][tape]
        coverage['cap_excluded']+=sum(not exclude_L1 or data['certificates'][t]['families']['L1']['status']!='eligible'
                                      for t in data['selection'][family]['cap_excluded'])
    status=verdict(selected,cases,controls_complete)
    if unresolved and status!='failed application': status='incomplete/invalid'
    rows=[d['world'] for d in eligibility]
    complete=(len(rows)==20 and {r['seed'] for r in rows}==set(SEEDS)
              and all(r['scoreable'] and r['horizon']==HORIZON and r['L']==ARMS[arm]['L'] for r in rows))
    if not complete and status!='failed application': status='incomplete/invalid'
    return {'family':family,'arm':arm,'arm_complete':complete,'exclude_L1':exclude_L1,'verdict':status,'independent_worlds':len(worlds),
            'selected':len(selected),'completed':sum(r['status']=='complete-zero-mismatch' for r in cases.values()),
            'unresolved_extractions':unresolved,'denominators':dict(coverage)}

def variants(parent, geometry, arm, seed):
    L,d = len(parent),geometry['d']; free = geometry['free']; tape = parent.tobytes().hex()
    for pos in free:
        for value in range(256):
            p = parent.copy(); p[pos] = value
            yield f'single:{pos}:{value}', 'single', p, np.tile(p[:d],L//d)
    for index in range(512):
        p = parent.copy()
        p[free] = ((37*np.arange(len(free))+index)%256 if index < 256 else
                   rng(arm,seed,L,'law-joint',tape,f'joint:{index}').integers(0,256,len(free),dtype=np.uint8))
        yield f'joint:{index}', 'joint', p, np.tile(p[:d],L//d)
    for pos in range(d,L):
        for value in range(256):
            p = parent.copy(); p[pos] = value
            yield f'erased:{pos}:{value}', 'erased', p, parent.copy()

def check_descendant(actual, expected, roots, geometry):
    d = geometry['d']
    if not np.array_equal(actual,expected): return 'exact_offspring_mismatch'
    if list(roots) != [i%d for i in range(len(actual))]: return 'parent_source_identity_mismatch'
    return None

def diagnosis(machine, diagnostic, memory, expected, geometry):
    r1 = machine.run(memory[None],budget=128); r2 = machine.run(memory[None],budget=128)
    q = diagnostic.run(memory[None],budget=128)
    identical = np.array_equal(r1.record,r2.record) and np.array_equal(r1.memory,r2.memory)
    same = same_execution(r1,q,machine.record_words)
    log = diagnostic.logs(q)[0]
    reason = None; resolved = False
    try:
        g = copy_certificate(memory[:len(expected)],log,len(expected),baseline=False)
        resolved = True
        reason = check_descendant(r1.memory[0,len(expected):],expected,g['source_map'],geometry)
    except ValueError as exc:
        reason = str(exc)
    return safe({'input':memory.tobytes().hex(),'expected':expected.tobytes().hex(),
                 'output':r1.memory[0].tobytes().hex(),'record':r1.record[0].tolist(),
                 'ordered_trace':None if log is None else log.tolist(), 'identical_replays':identical,
                 'diagnostic_identity':same,'provenance_resolved':resolved,'reason':reason,
                 # Static baseline promise breaking in a mutant is a failed application,
                 # not an outcome-driven removal from the selected panel.
                 'status':'failed application' if identical and same and log is not None
                          and reason and not reason.startswith('unmodelled_') else 'incomplete/invalid'})

def assay_case(parent, geometry, arm, seed, machine, diagnostic, emit, *, control=None, tick=lambda:None):
    """Batch deterministic cases; save actual serial partners and every output hash.

    Algebraic source map is constant because extraction proved free payload
    cannot change code, addresses, count or stop. Diagnostic mismatch replays
    validate this against execution; arbitrary coincident values never imply ancestry.
    """
    L = len(parent); tape = parent.tobytes().hex(); tested = 0
    from itertools import islice
    if control:
        from census8086_law import variants as census_variants
        base = census_variants(L,geometry['d'],control['op'])
        control['indices'] = {c[0]:i for i,c in enumerate(base)}
        fixed = [(c[0],c[1],c[4],np.tile(c[4][:geometry['d']],L//geometry['d'])) for c in base]
        erased = [c for c in variants(parent,geometry,arm,seed) if c[1]=='erased']
        iterator = iter(fixed+erased)
    else: iterator = iter(variants(parent,geometry,arm,seed))
    while batch := list(islice(iterator,256)):
        tick(); parents = np.stack([c[2] for c in batch]); mem = pairs(parents,L)
        raw = machine.run(mem,budget=128)
        children = raw.memory.reshape(len(batch),17,2*L)[:,:,L:]
        for i,(case,kind,p,expect) in enumerate(batch):
            for v, child in enumerate(children[i]):
                mismatch = check_descendant(child,expect,geometry['source_map'],geometry)
                emit({'case':case,'kind':kind,'generation':1,'challenge':'census','partner':v,
                      'input_sha256':sha(mem[i*17+v].tobytes()),'output_sha256':sha(child.tobytes()),
                      'matched':mismatch is None})
                tested += 1
                if mismatch:
                    diag = diagnosis(machine,diagnostic,mem[i*17+v],expect,geometry)
                    emit({'kind':'candidate-mismatch','case':case,'challenge':'census',**diag})
                    return {'status':diag['status'],'encounters':tested}
        serial_batch = [(i,c) for i,c in enumerate(batch) if c[1] != 'erased']
        if not serial_batch: continue
        current = np.stack([c[2] for i,c in serial_batch]); roots = [list(range(L)) for _ in serial_batch]
        for g in range(1,9):
            tick()
            if control:
                from algocell_exp.zcensus import partner_bytes
                # Original census control case ordering/serial stream (budget 128).
                ids = np.array([control['indices'][c[0]] for _,c in serial_batch],np.uint64)
                fresh = partner_bytes(ids,g,L,seed=0x8086C0DE ^ (L<<16) ^ (geometry['d']<<8) ^ control['op'] ^ 128)
            else:
                fresh = np.stack([rng(arm,seed,L,'law-serial',tape,c[0],g).integers(0,256,L,dtype=np.uint8)
                                  for _,c in serial_batch])
            memory = np.concatenate((current,fresh),axis=1); rr = machine.run(memory,budget=128)
            current = rr.memory[:,L:]
            for i,(_,c) in enumerate(serial_batch):
                roots[i] = [roots[i][p] for p in geometry['source_map']]
                mismatch = check_descendant(current[i],c[3],roots[i],geometry)
                emit({'case':c[0],'kind':c[1],'challenge':'serial','generation':g,
                      'partner_bytes':fresh[i].tobytes().hex(),'partner_sha256':sha(fresh[i].tobytes()),
                      'input_sha256':sha(memory[i].tobytes()),'output_sha256':sha(current[i].tobytes()),
                      'matched':mismatch is None})
                tested += 1
                if mismatch:
                    diag = diagnosis(machine,diagnostic,memory[i],c[3],geometry)
                    emit({'kind':'candidate-mismatch','case':c[0],'challenge':'serial','generation':g,**diag})
                    return {'status':diag['status'],'encounters':tested}
    return {'status':'complete-zero-mismatch','encounters':tested,'erased_coordinates':L-geometry['d'],
            'free_coordinates':len(geometry['free']),'single_assignments':len(geometry['free'])*256,
            'single_alternatives':len(geometry['free'])*255,'joint_assignments':512,
            'erased_assignments':(L-geometry['d'])*256,'census_partners':17,'serial_generations':8}

def assay_frozen(path, out, machine, diagnostic=None, tick=lambda:None):
    verify_freeze(); committed(path)
    data = json.loads(Path(path).read_text())
    if data['extractor_sha256'] != sha(Path(__file__).read_bytes()): raise ValueError('extractor changed')
    t = diagnostic or diagnostic_for(machine)
    world = data['world']; outcomes = {}
    Path(out).mkdir(parents=True,exist_ok=False)
    for family in ('L1','L2'):
        outcomes[family] = {}
        for tape in data['selection'][family]['selected']:
            p = np.frombuffer(bytes.fromhex(tape),np.uint8)
            cert = data['certificates'][tape]
            if cert['machine'] != safe(machine.describe()): raise ValueError('wrong machine identity')
            with gzip.open(Path(out)/f'{family}_{sha(p.tobytes())}.jsonl.gz','xt') as f:
                def emit(r): f.write(json.dumps(safe(r),allow_nan=False)+'\n'); f.flush()
                emit({'kind':'assay-input','certificate_sha256':sha(json.dumps(cert,sort_keys=True).encode()),
                      'parent':tape,'geometry':cert['families'][family], 'frozen_partners':cert['partners']})
                result = assay_case(p,cert['families'][family],world['arm'],world['seed'],machine,t,emit,tick=tick)
                outcomes[family][tape] = result; emit({'kind':'summary',**result})
    write_new(Path(out)/'summary.json',outcomes); return outcomes

def verdict(selected, outcomes, controls_complete):
    if any(r['status'] == 'failed application' for r in outcomes.values()): return 'failed application'
    if not controls_complete or set(outcomes) != set(selected) or any(
            r['status'] != 'complete-zero-mismatch' for r in outcomes.values()): return 'incomplete/invalid'
    return 'supported on tested evolved panel' if selected else 'evolved law untested'

if __name__ == '__main__':
    from algocell_exp.zmachine import load_machine
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=('extract','assay')); ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); a = ap.parse_args()
    data = json.loads(a.input.read_text()); L = data['world']['L']
    m = load_machine(f'i8086_pair_maps_L{L}')
    if a.stage == 'extract': freeze_candidates(data,a.out,m)
    else: assay_frozen(a.input,a.out,m)

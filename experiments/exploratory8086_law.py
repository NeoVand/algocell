"""EXPLORATORY / post hoc extension. Registered law8086.py is unchanged.

Copied dependency/geometry verifier from frozen S2 source, with conservative
POP/segment, NMOS conditional aliases, ALU, IN, LES/LDS and far-jump transfers.
Full-word segment taint is deliberately conservative on the 32-byte ring.
"""
import numpy as np
from law8086 import Unresolved
from s8086 import sha

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
    segments = [set() for _ in range(4)]
    source_segment = 3
    source_override = False
    direction = set()
    def sources(r): return regs[r][0] | regs[r][1]
    flags = {k:set() for k in ('CF','PF','AF','ZF','SF','OF')}
    dependencies = set(); fetched_roots = set(); moves = []
    pending_rep = None
    for step in np.unique(log[:,0]):
        group = log[log[:,0] == step]
        fetched = [int(v) for k,p,v,o in group[:,1:] if k == 0]
        for k,p,v,o in group[:,1:]:
            if k == 0:
                fetched_roots.add(origins[int(p)])
                dependencies |= segments[1]
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
            source_segment = next(({0x26:0,0x2e:1,0x36:2,0x3e:3}[p] for p in reversed(prefixes) if p in (0x26,0x2e,0x36,0x3e)), 3)
            source_override = any(p in prefixes for p in (0x26,0x2e,0x36,0x3e))
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
        if 0xb0 <= op <= 0xb7:
            reg, byte = op & 3, int(op >= 0xb4)
            regs[reg][byte] = set(); held[reg][byte] = None
        elif 0xb8 <= op <= 0xbf:
            regs[op-0xb8] = [set(),set()]; held[op-0xb8] = [None,None]
        elif 0x40 <= op <= 0x4f:
            low=regs[op & 7][0].copy(); word=sources(op & 7).copy()
            # Word INC/DEC preserve CF; low result does not depend on the old
            # high byte. Carries/borrows affect the new high byte.
            flags.update({'PF':low.copy(),'AF':low.copy(),'ZF':word.copy(),
                          'SF':word.copy(),'OF':word.copy()})
            regs[op & 7] = [low,word]; held[op & 7] = [None,None]
        elif op in (7,15,23,31) or 0x58 <= op <= 0x5f:
            # Frozen NMOS semantics: read SS:SP low then high; increment SP;
            # POP SP overwrites that increment with the popped word.
            dependencies |= sources(4) | segments[2]
            if len(rr) != 2: raise Unresolved('pop_read_width')
            roots = [{rr[0][2]}, {rr[1][2]}]
            regs[4] = [regs[4][0].copy(), sources(4).copy()]; held[4] = [None,None]
            if op in (7,15,23,31): segments[(op >> 3) & 3] = roots[0] | roots[1]
            else: regs[op & 7] = roots; held[op & 7] = rr.copy()
        elif 0x90 <= op <= 0x97:
            reg = op & 7
            regs[0], regs[reg] = regs[reg], regs[0]
            held[0], held[reg] = held[reg], held[0]
        elif op in (0xfc,0xfd): direction = set()
        elif op in (0xe4,0xe5,0xec,0xed):
            # Frozen host has no port device: IN returns all ones.
            n = 2 if op & 1 else 1
            for i in range(n): regs[0][i] = set(); held[0][i] = None
        elif op in (0xe6,0xe7,0xee,0xef,0x9b,0xfa,0xfb,0xf8,0xf9,0xf5):
            if op in (0xf8,0xf9): flags['CF'] = set()
        elif op == 0xea:
            segments[1] = set()  # far jump's CS and IP are fetched literals
        elif op in (0xc4,0xc5):
            if len(ins) < 2 or ins[1] >> 6 == 3: raise Unresolved('les_lds_register_form')
            modrm=ins[1]; mod,reg,rm=modrm>>6,(modrm>>3)&7,modrm&7
            address_regs=((3,6),(3,7),(5,6),(5,7),(6,),(7,),(5,),(3,))[rm]
            if mod == 0 and rm == 6: address_regs=()
            for r in address_regs: dependencies |= sources(r)
            seg=source_segment
            if not source_override and seg == 3 and (rm in (2,3) or (rm == 6 and mod != 0)): seg=2
            dependencies |= segments[seg]
            if len(rr) != 4: raise Unresolved('les_lds_read_width')
            regs[reg]=[{rr[0][2]},{rr[1][2]}]; held[reg]=rr[:2]
            segments[0 if op == 0xc4 else 3]={rr[2][2],rr[3][2]}
        elif op < 0x40 and op % 8 <= 5:
            # ALU: union taint conservatively; transformed values lose copy identity.
            kind=op//8; width=2 if op&1 else 1
            if op%8 in (4,5):
                deps=set().union(*regs[0][:width]); dest=0
            else:
                if len(ins)<2: raise Unresolved('missing_modrm')
                modrm=ins[1]; mod,reg,rm=modrm>>6,(modrm>>3)&7,modrm&7
                if width==1 and (reg>=4 or (mod==3 and rm>=4)): raise Unresolved('high_byte_register_dependency')
                deps=set().union(*regs[reg][:width])
                if mod==3: deps |= set().union(*regs[rm][:width])
                else:
                    address_regs=((3,6),(3,7),(5,6),(5,7),(6,),(7,),(5,),(3,))[rm]
                    if mod==0 and rm==6: address_regs=()
                    for r in address_regs: dependencies |= sources(r)
                    seg=source_segment
                    if not source_override and seg==3 and (rm in (2,3) or (rm==6 and mod!=0)): seg=2
                    dependencies |= segments[seg]
                    deps |= {r[2] for r in rr}
                dest=reg if op&2 else rm
                if mod!=3 and not op&2 and kind!=7:
                    raise ValueError('interfering_or_unpaired_write')
            if kind in (2,3): deps |= flags['CF']
            flags.update({k:deps.copy() for k in flags})
            if kind in (1,4,6):
                flags['CF']=set(); flags['OF']=set(); flags['AF']=set()
            if kind!=7:
                for i in range(width): regs[dest][i]=deps.copy(); held[dest][i]=None
        elif op in (0xa4,0xa5):
            dependencies |= sources(6) | sources(7) | sources(1) | segments[source_segment] | segments[0] | direction
            copy_reads = rr
        elif op in (0xac,0xad):
            dependencies |= sources(6) | segments[source_segment] | direction
            for i,r in enumerate(rr): held[0][i]=r; regs[0][i]={r[2]}
        elif op in (0xaa,0xab):
            dependencies |= sources(7) | segments[0] | direction
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
                seg = source_segment
                if not source_override and seg == 3 and (rm in (2,3) or (rm == 6 and mod != 0)): seg = 2
                dependencies |= segments[seg]
                if op in (0x8a,0x8b):
                    for i,r in enumerate(rr): held[reg][i]=r; regs[reg][i]={r[2]}
                else: copy_reads = held[reg][:1 if byte else 2]
        elif op in (0xe0,0xe1,0xe2,0xe3):
            dependencies |= sources(1)
            if op in (0xe0,0xe1): dependencies |= flags['ZF']
            held[1] = [None,None]
        elif 0x60 <= op <= 0x7f:
            used=(('OF',),('OF',),('CF',),('CF',),('ZF',),('ZF',),('CF','ZF'),('CF','ZF'),
                  ('SF',),('SF',),('PF',),('PF',),('SF','OF'),('SF','OF'),('ZF','SF','OF'),('ZF','SF','OF'))[op & 15]
            for name in used: dependencies |= flags[name]
        elif op in (0xeb,0xe9): pass
        elif op == 0xf4:
            if rr or ww: raise Unresolved('halt_access')
        else:
            raise Unresolved(f'unmodelled_dependency_opcode:{op:02x}')
        for reg in ((6,) if op in (0xac,0xad) else (6,7) if op in (0xa4,0xa5) else (7,) if op in (0xaa,0xab) else ()):
            regs[reg] = [regs[reg][0].copy(),sources(reg).copy()]
            held[reg] = [None,None]
        if op in (0xe0,0xe1,0xe2) or (pending_rep is not None and op in (0xa4,0xa5)):
            regs[1] = [regs[1][0].copy(),sources(1).copy()]; held[1] = [None,None]
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
    # K is the raw dependency footprint, never reduced modulo d. Held setup
    # roots may be ancestrally erased; arbitrary erasure excludes those roots.
    if any(p >= d for p in fetched_roots | dependencies):
        raise ValueError('raw_core_exceeds_footprint_bound')
    C = sorted(fetched_roots | dependencies)
    # Values at dependencies are held fixed at every repeated core position.
    # Executed stream therefore remains unchanged for arbitrary free payload.
    if not C or max(C)+1 > d: raise ValueError('core_exceeds_seed')
    s = moves[0][0]
    if not 0 <= s <= L-d: raise ValueError('source_root_interval_outside_A')
    R = list(range(s,s+d))
    source_map = [s+((L+q-s)%d) for q in range(L)]
    if origins[L:] != source_map: raise ValueError('ordered_source_map_disagrees_with_roots_phase')
    if set(origins[L:]) != set(R): raise ValueError('incomplete_original_A_root_interval')
    C_R = sorted(source_map[k] for k in C)
    if any(parent[k] != parent[source_map[k]] for k in C):
        raise ValueError('entry_phase_core_not_inherited_exactly')
    E = sorted(set(range(L))-set(R))
    free = sorted(set(R)-set(C_R))
    if len(free) != d-len(C): raise ValueError('noninjective_source_core')
    if any(source_map[source_map[q]] != source_map[q] for q in range(L)):
        raise ValueError('nonidempotent_serial_source_map')
    return {'d':d,'c':max(C)+1,'C':C,'K':C,'R':R,'C_R':C_R,'free':free,'F':free,'E':E,
            'erasure_controls':sorted(set(E)-set(C)), 'held_erased':sorted(set(E)&set(C)),
            'phase':(L-s)%d,'physical_entry':0,'source_start':s,'byte_moves':len(moves),
            'source_map':source_map, 'reset_source_map':source_map,
            'noninterference':'explicit raw dependency transfer; hold K and C_R; F and E\\K disjoint from raw dependencies; same fixed physical reset core',
            'moves_sha256':sha(np.asarray(moves,dtype='<u4').tobytes()),
            'trace_sha256':sha(log.astype('<u4').tobytes())}

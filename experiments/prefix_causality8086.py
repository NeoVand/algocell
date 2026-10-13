"""Fixed-input prefix-causality audit. No provider, population or paid calls.

Claims are hypotheses until BOTH isolated source-fetch interventions succeed.
Reset registers and exact read/fetch returns are independently intervened.
Every claim must pass both alternatives; unsupported/vector traffic stays other.
"""
from __future__ import annotations

import copy
import time
from collections import defaultdict

import numpy as np

from algocell_exp.zmachine import ZMachine
from s8086 import sha

PREFIXES = {0x26, 0x2e, 0x36, 0x3e, 0xf0, 0xf1, 0xf2, 0xf3}
STRINGS = {0xa4, 0xa5, 0xa6, 0xa7, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf}
UNKNOWN = ('other', None, None)


def claims(trace, L):
    """Exact source event for every partner store; no coincident-value matching."""
    if trace is None:
        return [], 'trace_overflow'
    targets = np.flatnonzero((trace[:, 1] == 2) & (trace[:, 2] >= L))
    if not len(targets):
        return [], 'no_partner_write'
    end = int(trace[targets[-1], 0])
    bank = [('literal', ('reset', r, b), (1 if r==4 else 0) >> (8*b))
            for r in range(12) for b in range(2)]
    groups = defaultdict(list)
    for i, row in enumerate(trace):
        if row[0] <= end:
            groups[int(row[0])].append((i, row))
    result = []
    repeating = False

    def slots(reg, n):
        return [2*reg, 2*reg+1] if n == 2 else [2*(reg % 4)+(reg // 4)]
    def get(reg, n):
        return [bank[i] for i in slots(reg, n)]
    def put(reg, n, tags):
        for i, tag in zip(slots(reg, n), (tags+[UNKNOWN]*n)[:n]):
            bank[i] = tag
    def kill(reg):
        put(reg, 2, [UNKNOWN]*2)
    def add(reg, delta):
        tags=get(reg, 2)
        if all(t[0]=='literal' for t in tags):
            value=(tags[0][2]+256*tags[1][2]+delta)&65535
            put(reg, 2, [('literal', ('add', tuple(tags), delta, b), (value>>(8*b))&255) for b in range(2)])
        else: kill(reg)

    for tick, group in sorted(groups.items()):
        decoded = [int(r[3]) for _, r in group if r[1] == 3]
        data = [(i, r) for i, r in group if r[1] == 1]
        stores = [(i, r) for i, r in group if r[1] == 2]
        # Reject before accepting this instruction's own writes. A hardware
        # interrupt always emits six stack bytes, even after zero-count REP.
        if len(stores) >= 6 or any(op in range(0xcc, 0xd0) for op in decoded):
            return [(int(i), UNKNOWN) for i in targets], 'interrupt_or_vector_traffic'
        if not decoded:
            if stores:
                return [(int(i), UNKNOWN) for i in targets], 'undecoded_writer'
            continue
        op = decoded[-1]
        fetch = [(i, r) for i, r in group if r[1] == 0]
        raw = [int(r[3]) for _, r in fetch]
        pref = []
        while raw and raw[0] in PREFIXES:
            pref.append(raw.pop(0)); fetch.pop(0)
        if raw:
            repeating = op in STRINGS and bool(set(pref) & {0xf2, 0xf3})
        reads = [('memory', i, int(r[3])) for i, r in data]
        immediate = [('literal', i, int(r[3])) for i, r in fetch[1:]]
        n = 2 if op % 2 else 1
        m = raw[1] if len(raw) > 1 else 0
        mode, reg, rm = m // 64, (m // 8) % 8, m % 8
        source = []
        if 0x50 <= op <= 0x57:
            if op == 0x54: add(4, -2)
            source = get(op-0x50, 2)
        elif op in (0xfe, 0xff) and reg >= 6:
            if mode == 3:
                if op == 0xff and rm == 4: add(4, -2)
                source = get(rm, n)
            else: source = reads[:n]
        elif op in (0xa4, 0xa5): source = reads[:n]
        elif op in (0xaa, 0xab, 0xa2, 0xa3): source = get(0, n)
        elif op in (0x88, 0x89) and mode != 3: source = get(reg, n)
        elif op in (0xc6, 0xc7): source = immediate[-n:]
        elif op in (6, 14, 22, 30): source = bank[16+2*((op//8)%4):18+2*((op//8)%4)]
        elif op == 0x8c and mode != 3: source = bank[16+2*(reg%4):18+2*(reg%4)]
        elif op == 0x8f and mode != 3: source = reads[:2]
        for j, (i, row) in enumerate(stores):
            if row[2] < L: continue
            tag = source[j] if j < len(source) else UNKNOWN
            rmw=(op<64 and op%8<4) or op in (0x80,0x81,0x82,0x83,0xd0,0xd1,0xd2,0xd3,0xf6,0xf7,0xfe,0xff)
            if not source and rmw:
                # Neutral RMW claims still require exact causal intervention.
                paired=[(ii,rr) for ii,rr in data if rr[2]==row[2] and rr[3]==row[3]]
                if paired: tag=('neutral', paired[-1][0], int(row[3]))
            if tag[2] != int(row[3]): tag = UNKNOWN
            result.append((i, tag))
        if 0xb0 <= op <= 0xbf: put(op%8, 2 if op>=0xb8 else 1, immediate)
        elif op in (0xac, 0xad) and repeating and not reads: pass
        elif op in (0xac, 0xad, 0xa0, 0xa1): put(0, n, reads[:n])
        elif 0x58 <= op <= 0x5f: add(4, 2); put(op-0x58, 2, reads[:2])
        elif op in (7, 15, 23, 31):
            add(4, 2); bank[16+2*((op//8)%4):18+2*((op//8)%4)] = (reads+[UNKNOWN]*2)[:2]
        elif op in (0x88, 0x89, 0x8a, 0x8b):
            if op >= 0x8a: put(reg, n, get(rm, n) if mode==3 else reads[:n])
            elif mode==3: put(rm, n, get(reg, n))
        elif op in (0xc4, 0xc5) and mode != 3 and len(reads) >= 4:
            put(reg, 2, reads[:2]); s=0 if op==0xc4 else 3; bank[16+2*s:18+2*s]=reads[2:4]
        elif op == 0x8e: bank[16+2*(reg%4):18+2*(reg%4)] = ((get(rm, 2) if mode==3 else reads[:2])+[UNKNOWN]*2)[:2]
        elif op == 0x8c and mode==3: put(rm, 2, bank[16+2*(reg%4):18+2*(reg%4)])
        elif 0x40 <= op <= 0x4f: add(op%8, 1 if op<0x48 else -1)
        elif op in (0xc6, 0xc7) and mode==3: put(rm, n, immediate[-n:])
        elif 0x90 <= op <= 0x97:
            a=get(0, 2); put(0, 2, get(op%8, 2)); put(op%8, 2, a)
        elif op in (0x86, 0x87):
            a=get(reg, n); put(reg, n, get(rm, n) if mode==3 else reads[:n])
            if mode==3: put(rm, n, a)
        elif op == 0x8f:
            add(4, 2)
            if mode==3: put(rm, 2, reads[:2])
        elif op in (0xe0, 0xe1, 0xe2): add(1, -1)
        elif op == 0xea: bank[18:20] = immediate[-2:]
        elif op in (0xe3, 0xe9, 0xeb, 0x9b, 0xa8, 0xa9, 0xf4, 0xf5, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd) or 0x60 <= op <= 0x7f: pass
        elif op in STRINGS or 0x50 <= op <= 0x57 or op in (6, 14, 22, 30): pass
        elif op in (0xfe, 0xff) and reg >= 6: pass
        elif op < 64 and op%8<2 and mode!=3: pass
        elif op in (0xc6, 0xc7, 0x88, 0x89, 0x8c) and mode!=3: pass
        else: bank = [UNKNOWN]*24
        if (not repeating or data or stores) and op in (0xa4, 0xa5, 0xa6, 0xa7, 0xac, 0xad): kill(6)
        if (not repeating or data or stores) and op in (0xa4, 0xa5, 0xa6, 0xa7, 0xaa, 0xab, 0xae, 0xaf): kill(7)
        if (not repeating or data or stores) and repeating and op in STRINGS: add(1, -1)
        if 0x50 <= op <= 0x57 or op in (6, 14, 22, 30) or (op in (0xfe, 0xff) and reg>=6):
            if not (op==0x54 or (op==0xff and mode==3 and rm==4)): add(4, -2)
    return result, 'exact_event_claims_only'


def label(log, L, verified):
    if not verified or any(tag[0] == 'other' for _, tag in verified): return 'other'
    channels = {tag[0] for _, tag in verified if tag[0]!='neutral'}
    if channels == {'literal'}: return 'literal writer'
    if channels == {'memory'}:
        end = max(int(log[i, 0]) for i, _ in verified)
        episode = log[log[:, 0] <= end]
        starts = episode[episode[:, 1] == 4, 2]
        if len(set(starts)) == len(starts): return 'memory-fed one-pass copier'
        if not np.any((episode[:, 1] == 0) & (episode[:, 2] >= L)): return 'confined copy loop'
    return 'other'


def compare(original, modified, write_index, source_index, mask, *, predicted=None):
    """Compare ordered starts/fetch-read/data-read/write addresses through w."""
    prefix = original[:write_index+1]
    structural = np.isin(prefix[:, 1], [0, 1, 2, 4])
    other = None if modified is None else modified[:write_index+1]
    identical = (other is not None and len(other)==len(prefix)
                 and np.array_equal(prefix[structural, :3], other[structural, :3]))
    wanted = int(original[source_index, 3]) ^ mask if predicted is None else predicted
    value = None if other is None or len(other)<=write_index else int(other[write_index, 3])
    follows = bool(identical and value == wanted and other[-1, 1] == 2)
    divergence = None
    if modified is not None:
        a = original[np.isin(original[:, 1], [0, 1, 2, 4]), :3]
        b = modified[np.isin(modified[:, 1], [0, 1, 2, 4]), :3]
        n=min(len(a), len(b))
        differences=np.flatnonzero(np.any(a[:n]!=b[:n],axis=1))
        if len(differences): divergence=int(differences[0])
        if divergence is None and len(a)!=len(b): divergence=min(len(a), len(b))
    return {'mask':mask, 'substitute':wanted, 'written':value,
            'prefix_identical':bool(identical), 'passes':follows,
            'first_structural_divergence':divergence,
            'trace_sha256':None if modified is None else sha(modified.tobytes())}


class Interventions(ZMachine):
    """Per-lane exact event intervention, carried in an extra input binding."""
    def __init__(self, observer):
        m = copy.deepcopy(observer.m)
        binding = max(b['binding'] for b in m['bindings'])+1
        m['bindings'].append({'binding':binding, 'group':0, 'type':'read-only-storage',
                              'role':'stateIn', 'wordsPerLane':3})
        source = observer.wgsl
        anchor = 'let p=zil_pos(addr); let v=census_real_read(addr);'
        assert source.count(anchor)==1
        source=source.replace(anchor, 'let p=zil_pos(addr); var v=census_real_read(addr); '
                              'if (census_n == causal_target && (kind==0u || kind==1u)) '
                              '{ v=v^(zil_params.seed & 255u); }')
        anchor='census_clear();'
        assert source.count(anchor)==1
        source=source.replace(anchor, 'census_clear(); causal_target=causal_inputs[3u*(lane-zil_params.lane_base)];')
        anchor='    zil_init_state();'
        assert source.count(anchor)==1
        source=source.replace(anchor, anchor + '''
    let causal_r=causal_inputs[3u*(lane-zil_params.lane_base)+1u];
    let causal_v=causal_inputs[3u*(lane-zil_params.lane_base)+2u];
    if(causal_r<8u) { i8086_r[causal_r]=causal_v; }
    else if(causal_r<12u) { i8086_s[causal_r-8u]=causal_v; }
''')
        source+=f'\n@group(0) @binding({binding}) var<storage, read> causal_inputs: array<u32>;\nvar<private> causal_target: u32;\n'
        m['sha256']['wgsl']=sha(source.encode())
        super().__init__(source, m, name='isolated-source-event', device=observer.dev)


BURSTS=[]
def burst(name, fn):
    start=time.monotonic(); value=fn(); elapsed=time.monotonic()-start
    if elapsed>60: raise RuntimeError('local GPU burst exceeded 60s')
    pause=time.monotonic(); time.sleep(elapsed)
    BURSTS.append({'name':name, 'burst_s':elapsed, 'pause_s':time.monotonic()-pause})
    return value


def roots(tag):
    found=set();seen=set();pending=[tag]
    while pending:
        t=pending.pop()
        if id(t) in seen: continue
        seen.add(id(t));src=t[1]
        if isinstance(src, int): found.add(('event',src))
        elif src is not None and src[0]=='reset': found.add(('reset',src[1]))
        elif src is not None and src[0]=='add': pending.extend(src[1])
    return found


def predict(tag, root, mask):
    # Register arithmetic forms a shared expression graph. Evaluate each node
    # once, rather than expanding both bytes into an exponential tree.
    memo={}
    def byte(t):
        if id(t) in memo: return memo[id(t)]
        src=t[1]
        if isinstance(src, int): v=t[2]^mask if root==('event',src) else t[2]
        elif src[0]=='reset':
            word=1 if src[1]==4 else 0
            if root==('reset',src[1]): word^=mask*257
            v=(word>>(8*src[2]))&255
        elif src[0]=='add':
            word=(byte(src[1][0])+256*byte(src[1][1])+src[2])&65535
            v=(word>>(8*src[3]))&255
        else: raise ValueError('source cannot be intervened on')
        memo[id(t)]=v
        return v
    return byte(tag)


def verify_batch(observer, intervention, memories, logs, L, budget, *, execute=burst, extractor=claims):
    hypotheses=[extractor(log, L) for log in logs]
    verified=[list(c) for c, _ in hypotheses]
    jobs=[]
    for lane, (cc, reason) in enumerate(hypotheses):
        if not cc or any(tag[0]=='other' for _, tag in cc): continue
        by_source=defaultdict(list)
        for write, tag in cc:
            rr=roots(tag)
            if not rr:
                verified[lane]=[(i, t if i!=write else UNKNOWN) for i,t in verified[lane]]
            for root in rr: by_source[root].append((write,tag))
        jobs.extend((lane, root, writes) for root, writes in by_source.items())
    evidence=[]
    for lo in range(0, len(jobs), 64):
        work=jobs[lo:lo+64]
        mem=np.stack([memories[lane] for lane, _, _ in work])
        outcomes=[]
        for mask in (0x55, 0xaa):
            inputs=np.asarray([[root[1] if root[0]=='event' else 0xffffffff,
                                root[1] if root[0]=='reset' else 0xffffffff,
                                ((1 if root[1]==4 else 0) ^ (mask*257)) if root[0]=='reset' else 0]
                               for _,root,_ in work], np.uint32)
            changed=execute(f'intervene-L{L}-b{budget}-{lo}-{mask}',
                            lambda:intervention.run(mem, budget=budget, seed=mask, state=inputs))
            outcomes.append(observer.logs(changed))
        for j, (lane, root, writes) in enumerate(work):
            for write,tag in writes:
                checks=[compare(logs[lane], outcomes[k][j], write,
                                root[1] if root[0]=='event' else 0, mask,
                                predicted=predict(tag,root,mask))
                        for k, mask in enumerate((0x55, 0xaa))]
                if not all(c['passes'] for c in checks):
                    verified[lane]=[(i, t if i!=write else UNKNOWN) for i, t in verified[lane]]
                evidence.append({'lane':lane, 'source':root, 'write_event':write, 'checks':checks})
    return [label(log, L, cc) for log, cc in zip(logs, verified)], hypotheses, evidence


if __name__ == '__main__':
    raise SystemExit('Use audit_prefix8086.py; this module never launches work.')

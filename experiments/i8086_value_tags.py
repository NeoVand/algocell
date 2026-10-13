"""Independent, trace-fed value-tag oracle for the conservative census contract.

No production classifier/spec imports. Storage is a flat 24-byte bank, not the
classifier's paired register lists. Tags name the read event and carry its value;
transformed values are unresolved, irrespective of numerical coincidences.
Unknown operations invalidate the whole bank. This tests provenance (not CPU
arithmetic correctness, which remains covered by the vendored vector ledger).
"""
import numpy as np

PREFIXES = {0x26,0x2e,0x36,0x3e,0xf0,0xf1,0xf2,0xf3}
STRINGS = {0xa4,0xa5,0xa6,0xa7,0xaa,0xab,0xac,0xad,0xae,0xaf}

def value_channels(trace, length):
    if trace is None: return []
    targets = [r for r in trace if r[1] == 2 and r[2] >= length]
    if not targets: return []
    stop = targets[-1][0]
    bank = [('literal',None,None)] * 24
    unknown = ('other',None,None)
    literal = ('literal',None,None)
    channels = []
    repeating = False
    def slots(n, size): return [2*n,2*n+1] if size == 2 else [2*(n%4)+(n//4)]
    def take(n,size): return [bank[i] for i in slots(n,size)]
    def store(n,size,tags):
        for i,t in zip(slots(n,size), (tags+[unknown]*size)[:size]): bank[i]=t
    def change(n):
        ix=slots(n,2)
        fill=literal if all(bank[i][0]=='literal' for i in ix) else unknown
        for i in ix: bank[i]=fill
    groups={}
    for row in trace:
        if row[0] <= stop: groups.setdefault(int(row[0]),[]).append(row)
    for tick,group in sorted(groups.items()):
        decoded=[int(r[3]) for r in group if r[1]==3]
        if not decoded: continue
        op=decoded[-1]
        raw=[int(r[3]) for r in group if r[1]==0]
        pref=[]
        while raw and raw[0] in PREFIXES: pref.append(raw.pop(0))
        if raw: repeating=op in STRINGS and bool(set(pref)&{0xf2,0xf3})
        data=[r for r in group if r[1]==1]; stores=[r for r in group if r[1]==2]
        tags=[('memory',(tick,i,int(r[2])),int(r[3])) for i,r in enumerate(data)]
        n=2 if op%2 else 1
        m=raw[1] if len(raw)>1 else 0
        mode=m//64; dest=(m//8)%8; operand=m%8
        outgoing=[]
        if op in range(0x50,0x58):
            if op==0x54: change(4)
            outgoing=take(op-0x50,2)
        if op in (0xfe,0xff) and dest>=6:
            if mode==3:
                if op==0xff and operand==4: change(4)
                outgoing=take(operand,n)
            else: outgoing=tags[:n]
        if op in (0xa4,0xa5): outgoing=tags[:n]
        if op in (0xaa,0xab,0xa2,0xa3): outgoing=take(0,n)
        if op in (0x88,0x89) and mode!=3: outgoing=take(dest,n)
        if op in (0xc6,0xc7): outgoing=[literal]*n
        if op in (6,14,22,30): outgoing=bank[16+2*((op//8)%4):18+2*((op//8)%4)]
        if op==0x8c and mode!=3: outgoing=bank[16+2*(dest%4):18+2*(dest%4)]
        if op==0x8f and mode!=3: outgoing=tags[:2]
        for index,row in enumerate(stores):
            if row[2]<length: continue
            rmw=(op<64 and op%8<4) or op in (0x80,0x81,0x82,0x83,0xd0,0xd1,0xd2,0xd3,0xf6,0xf7,0xfe,0xff)
            if not outgoing and rmw and any(r[2]==row[2] and r[3]==row[3] for r in data): continue
            tag=outgoing[index] if index<len(outgoing) else unknown
            channels.append(tag[0] if tag[0]!='memory' or tag[2]==row[3] else 'other')
        handled=True
        if 0xb0<=op<=0xbf: store(op%8,2 if op>=0xb8 else 1,[literal]*2)
        elif op in (0xac,0xad) and repeating and not data: pass
        elif op in (0xac,0xad,0xa0,0xa1): store(0,n,tags[:n] or [unknown]*n)
        elif op in range(0x58,0x60): change(4); store(op-0x58,2,tags[:2] if len(tags)>=2 else [unknown]*2)
        elif op in (7,15,23,31):
            change(4); bank[16+2*((op//8)%4):18+2*((op//8)%4)]=tags[:2] if len(tags)>=2 else [unknown]*2
        elif op in (0x88,0x89,0x8a,0x8b):
            if op>=0x8a: store(dest,n,take(operand,n) if mode==3 else tags[:n])
            elif mode==3: store(operand,n,take(dest,n))
        elif op in (0xc4,0xc5) and mode!=3 and len(tags)>=4:
            store(dest,2,tags[:2]); s=0 if op==0xc4 else 3; bank[16+2*s:18+2*s]=tags[2:4]
        elif op==0x8e: bank[16+2*(dest%4):18+2*(dest%4)]=take(operand,2) if mode==3 else tags[:2]
        elif op==0x8c and mode==3: store(operand,2,bank[16+2*(dest%4):18+2*(dest%4)])
        elif 0x40<=op<=0x4f: change(op%8)
        elif op in (0xc6,0xc7) and mode==3: store(operand,n,[literal]*n)
        elif 0x90<=op<=0x97:
            a=take(0,2); b=take(op%8,2); store(0,2,b); store(op%8,2,a)
        elif op in (0x86,0x87):
            a=take(dest,n); b=take(operand,n) if mode==3 else tags[:n]; store(dest,n,b)
            if mode==3: store(operand,n,a)
        elif op==0x8f:
            change(4)
            if mode==3: store(operand,2,tags[:2])
        elif op in (0xe0,0xe1,0xe2): change(1)
        elif op==0xea: bank[18:20]=[literal]*2
        elif op in (0xe3,0xe9,0xeb,0x9b,0xa8,0xa9,0xf4,0xf5,0xf8,0xf9,0xfa,0xfb,0xfc,0xfd) or 0x60<=op<=0x7f: pass
        elif op in (0xa4,0xa5,0xa6,0xa7,0xaa,0xab,0xae,0xaf): pass
        elif 0x50<=op<=0x57 or op in (6,14,22,30): pass
        elif op in (0xfe,0xff) and dest>=6: pass
        elif op<64 and op%8<2 and mode!=3: pass
        elif op in (0xc6,0xc7,0x88,0x89,0x8c) and mode!=3: pass
        else: handled=False
        if not handled: bank=[unknown]*24
        if (not repeating or data or stores) and op in (0xa4,0xa5,0xa6,0xa7,0xac,0xad): change(6)
        if (not repeating or data or stores) and op in (0xa4,0xa5,0xa6,0xa7,0xaa,0xab,0xae,0xaf): change(7)
        if (not repeating or data or stores) and repeating and op in STRINGS: change(1)
        if 0x50<=op<=0x57 or op in (6,14,22,30) or (op in (0xfe,0xff) and dest>=6): change(4)
        if len(stores)>=6: bank=[unknown]*24
    return channels


def reference_mechanism(trace,L):
    if trace is None: return 'other'
    ch=value_channels(trace,L)
    if ch and set(ch)=={'literal'}: return 'literal writer'
    if ch and set(ch)=={'memory'}:
        stop=max(r[0] for r in trace if r[1]==2 and r[2]>=L)
        starts=[int(r[2]) for r in trace if r[1]==4 and r[0]<=stop]
        entered=any(r[1]==0 and r[2]>=L and r[0]<=stop for r in trace)
        if len(set(starts))==len(starts): return 'memory-fed one-pass copier'
        if not entered: return 'confined copy loop'
    return 'other'

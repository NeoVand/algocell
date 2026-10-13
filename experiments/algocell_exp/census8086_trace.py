"""Read/write diagnostics derived from a vendored machine, without changing execution.

The census checks every diagnostic lane against the unmodified machine. Full
traces are bounded; overflow is explicit, never silently classified.
"""
from __future__ import annotations

import copy
import hashlib
import re

import numpy as np

from .zmachine import ZMachine
from .i8086_provenance_spec import OPCODE_CLASS


class DiagnosticMachine(ZMachine):
    def __init__(self, base: ZMachine, events: int = 0, *, storage_trace: bool = False):
        m = copy.deepcopy(base.m)
        old = base.record_words
        self.base, self.events, self.diag_offset = base, events, old
        extra = 6 + 5 * events
        m["record"]["wordsPerLane"] += extra
        m["record"]["fields"].append({"name": "censusDiagnostics", "offset": old, "words": extra})
        src = base.wgsl

        def once(a, b):
            nonlocal src
            if src.count(a) != 1:
                raise ValueError(f"diagnostic anchor not unique: {a!r}")
            src = src.replace(a, b)

        once(f"const ZIL_RECORD_WORDS: u32 = {old}u;", f"const ZIL_RECORD_WORDS: u32 = {old+extra}u;")
        once("fn mem_read(addr: u32)", "fn census_real_read(addr: u32)")
        once("fn mem_write(addr: u32, val: u32)", "fn census_real_write(addr: u32, val: u32)")
        once("let v = mem_read(a); i8086_ip", "let v = census_fetch_read(a); i8086_ip")
        # Decode and start hooks record actual execution, including retained REP.
        for name, event, value in (("on_decode", 3, "key"), ("on_instruction_start", 4, "unit")):
            pat = rf"fn {name}\([^{{]+\{{"
            matches = list(re.finditer(pat, src))
            if len(matches) != 1:
                raise ValueError(f"missing {name} diagnostic hook")
            at = matches[0].end()
            src = src[:at] + f"\n    census_event({event}u, {'pc' if event == 4 else 'i8086_pc()'}, {value});" + src[at:]
        once("fn zil_store_record(o: u32, steps: u32) {", "fn zil_store_record(o: u32, steps: u32) {\n    census_store(o);")
        storage_trace = storage_trace and events > 0
        record_base = "\n    census_record_base = (lane - zil_params.lane_base) * ZIL_RECORD_WORDS;" if storage_trace else ""
        once("fn zil_reset_lane(lane: u32, seed: u32) {", "fn zil_reset_lane(lane: u32, seed: u32) {" + record_base + "\n    census_clear();")
        L, P = base.segment("A")["bytes"], base.ring_bytes
        trace_vars = ("var<private> census_record_base: u32;" if storage_trace else
                      f"var<private> census_log: array<u32, {5*events}>;" if events else "")
        log = f"""
  if (census_n < {events}u) {{
    let o = census_n * 5u;
    census_log[o] = zil_step; census_log[o+1u] = kind;
    census_log[o+2u] = select(p, addr, kind >= 3u); census_log[o+3u] = value;
    census_log[o+4u] = select(0u, 1u, p >= {L}u && census_written[p] == 0u);
  }}
""" if events else ""
        if storage_trace:
            log = f"""
  if (census_n < {events}u) {{
    let o = census_record_base + {old+6}u + census_n * 5u;
    zil_out[o] = zil_step; zil_out[o+1u] = kind;
    zil_out[o+2u] = select(p, addr, kind >= 3u); zil_out[o+3u] = value;
    zil_out[o+4u] = select(0u, 1u, p >= {L}u && census_written[p] == 0u);
  }}
"""
        store = f"for (var i=0u; i < min(census_n, {events}u)*5u; i++) {{ zil_out[o+{old+6}u+i] = census_log[i]; }}" if events else ""
        if storage_trace: store = ""  # every lane owns only its appended record region
        src += f"""
// Census observation only. 0 fetch, 1 data read, 2 write, 3 decode, 4 start.
var<private> census_written: array<u32, {P}>;
var<private> census_counts: array<u32, 3>;
var<private> census_n: u32;
{trace_vars}
fn census_clear() {{
  for(var i=0u;i<{P}u;i++) {{ census_written[i]=0u; }}
  for(var i=0u;i<3u;i++) {{ census_counts[i]=0u; }} census_n=0u;
}}
fn census_event(kind: u32, addr: u32, value: u32) {{
  let p = zil_pos(addr);
  {log}
  census_n += 1u;
}}
fn census_read(addr: u32, kind: u32) -> u32 {{
  let p=zil_pos(addr); let v=census_real_read(addr);
  if(kind==1u) {{ census_counts[2] += 1u; }}
  if(p>={L}u && census_written[p]==0u) {{ census_counts[kind] += 1u; }}
  census_event(kind, addr, v); return v;
}}
fn mem_read(addr: u32) -> u32 {{ return census_read(addr, 1u); }}
fn census_fetch_read(addr: u32) -> u32 {{ return census_read(addr, 0u); }}
fn mem_write(addr: u32, val: u32) {{
  census_event(2u, addr, val & 255u); census_written[zil_pos(addr)] = 1u;
  census_real_write(addr, val);
}}
fn census_store(o: u32) {{
  for(var i=0u;i<3u;i++) {{ zil_out[o+{old}u+i]=census_counts[i]; }}
  var a=0u; var b=0u;
  for(var i=0u;i<{P}u;i++) {{ if(i<{L}u) {{ a += census_written[i]; }} else {{ b += census_written[i]; }} }}
  zil_out[o+{old+3}u]=a; zil_out[o+{old+4}u]=b; zil_out[o+{old+5}u]=census_n;
  {store}
}}
"""
        m["sha256"]["wgsl"] = hashlib.sha256(src.encode()).hexdigest()
        super().__init__(src, m, name=base.name + ("_trace" if events else "_diagnostics"), device=base.dev)

    def diagnostics(self, result):
        return result.record[:, self.diag_offset:self.diag_offset+6]

    def logs(self, result):
        if not self.events:
            raise ValueError("summary-only diagnostics")
        out = []
        for row in result.record:
            n = int(row[self.diag_offset+5])
            if n > self.events:
                out.append(None)
            else:
                out.append(row[self.diag_offset+6:self.diag_offset+6+5*n].reshape(n, 5))
        return out


def same_execution(base_result, diagnostic_result, base_words):
    return (np.array_equal(base_result.memory, diagnostic_result.memory)
            and np.array_equal(base_result.record, diagnostic_result.record[:, :base_words]))


def mechanism(log, L, *, audit_channels=False):
    """Conservative trace classification; memory copies require read/write evidence.

    One-pass means source/destination movement without repeated instruction-start
    positions in the copying episode; 'confined' additionally excludes B fetches.
    The episode ends at the last partner write. Unrecognized writers stay other.
    """
    if log is None:
        return [] if audit_channels else ("other", "trace_overflow")
    writes = log[(log[:, 1] == 2) & (log[:, 2] >= L)]
    if not len(writes):
        return [] if audit_channels else ("other", "no_partner_write")
    end = int(writes[-1, 0])
    episode = log[log[:, 0] <= end]
    reads = episode[episode[:, 1] == 1]
    starts = episode[episode[:, 1] == 4, 2]
    confined = not np.any((episode[:, 1] == 0) & (episode[:, 2] >= L))
    repeated = len(np.unique(starts)) < len(starts)
    ops = {int(s): int(v) for s, kind, p, v, origin in episode if kind == 3}
    # Each byte is (channel, exact read event identity, value). A word store
    # pairs low then high with its semantic source; equal values elsewhere in
    # the read set never establish pairing. Arithmetic kills copy identity.
    literal = ("literal", None, None)
    other = ("other", None, None)
    regs = [[literal, literal] for _ in range(8)]
    segments = [[literal, literal] for _ in range(4)]
    channels = []
    pending_rep = False

    def arithmetic(reg):
        regs[reg] = [literal,literal] if all(b[0] == "literal" for b in regs[reg]) else [other,other]

    def get(reg, word):
        return regs[reg][:] if word else [regs[reg & 3][int(reg >= 4)]]

    def put(reg, word, value):
        value = (value + [other,other])[:2 if word else 1]
        if word: regs[reg] = value[:]
        else: regs[reg & 3][int(reg >= 4)] = value[0]

    # Diagnostic events are ordered by execution step. Partition once rather
    # than rescanning a full 8192-event episode for each retained REP step.
    boundaries = np.flatnonzero(np.diff(episode[:,0])) + 1
    groups = {int(g[0,0]):g for g in np.split(episode,boundaries) if len(g)}
    for step in sorted(ops):
        op = ops[step]
        group = groups[step]
        fetched = list(map(int, group[group[:, 1] == 0, 3]))
        prefixes = []
        while fetched and fetched[0] in (0x26, 0x2e, 0x36, 0x3e, 0xf0, 0xf1, 0xf2, 0xf3):
            prefixes.append(fetched.pop(0))
        if fetched: pending_rep = op in (0xa4,0xa5,0xa6,0xa7,0xaa,0xab,0xac,0xad,0xae,0xaf) and any(p in (0xf2,0xf3) for p in prefixes)
        rr = group[group[:, 1] == 1]
        ww = group[group[:, 1] == 2]
        read = [("memory", (int(step),i,int(r[2])), int(r[3])) for i,r in enumerate(rr)]
        source = []
        modrm = fetched[1] if len(fetched) > 1 else None
        word = op & 1
        mode = modrm >> 6 if modrm is not None else None
        reg = (modrm >> 3) & 7 if modrm is not None else None
        rm = modrm & 7 if modrm is not None else None
        policy = OPCODE_CLASS[op]
        if 0x50 <= op <= 0x57:
            # The kernel implements 8086 PUSH SP's decremented stored value.
            if op == 0x54: arithmetic(4)
            source = get(op-0x50, True)
        elif op in (0xfe,0xff) and reg in (6,7):
            if mode == 3:
                if op == 0xff and rm == 4: arithmetic(4)
                source = get(rm,word)
            else: source = read[:1+word]
        elif op in (0xa4,0xa5): source = read[:1+word]
        elif op in (0xaa,0xab,0xa2,0xa3): source = get(0,word)
        elif op in (0x88,0x89) and mode != 3: source = get(reg,word)
        elif op in (0xc6,0xc7): source = [literal]*(1+word)
        elif op in (6,14,22,30): source = segments[(op>>3)&3][:]
        elif op == 0x8c and mode != 3: source = segments[reg&3][:]
        elif op == 0x8f and mode != 3: source = read[:2]
        for i,write in enumerate(ww):
            if write[2] < L: continue
            rmw = (op < 0x40 and (op & 7) < 4) or op in (0x80,0x81,0x82,0x83,0xd0,0xd1,0xd2,0xd3,0xf6,0xf7,0xfe,0xff)
            if not source and rmw and np.any((rr[:,2] == write[2]) & (rr[:,3] == write[3])):
                # Address-paired unchanged RMW contributes no new copy origin.
                continue
            b = source[i] if i < len(source) else other
            channels.append(b[0] if b[0] != "memory" or int(write[3]) == b[2] else "other")

        # Explicit writes and every implicit general-register update in the
        # vendored execute/string/repeat/push/pop helpers. Unknown semantics
        # poison all registers and segments, including trap/interrupt paths.
        if policy == 'immediate': put(op&7,op>=0xb8,[literal]*(2 if op>=0xb8 else 1))
        elif op in (0xac,0xad) and pending_rep and not len(rr): pass
        elif op in (0xac,0xad,0xa0,0xa1): put(0,word,read[:1+word] or [other]*(1+word))
        elif policy == 'register_pop':
            arithmetic(4); put(op-0x58,True,read[:2] if len(read)>=2 else [other]*2)
        elif policy == 'segment_pop':
            arithmetic(4); segments[(op>>3)&3] = read[:2] if len(read)>=2 else [other]*2
        elif policy == 'mov' and modrm is not None:
            if op in (0x8a,0x8b): put(reg,word,get(rm,word) if mode==3 else read[:1+word])
            elif mode==3: put(rm,word,get(reg,word))
        elif policy == 'far_load' and mode != 3 and len(read) >= 4:
            put(reg,True,read[:2]); segments[0 if op==0xc4 else 3] = read[2:4]
        elif op == 0x8e and modrm is not None: segments[reg&3] = get(rm,True) if mode==3 else read[:2]
        elif op == 0x8c and mode==3: put(rm,True,segments[reg&3])
        elif policy == 'inc_dec': arithmetic(op&7)
        elif op in (0xc6,0xc7) and mode==3: put(rm,word,[literal]*(1+word))
        elif policy == 'ax_exchange':
            regs[0],regs[op&7] = regs[op&7],regs[0]
        elif policy == 'xchg' and modrm is not None:
            old = get(reg,word)
            put(reg,word,get(rm,word) if mode==3 else read[:1+word])
            if mode==3: put(rm,word,old)
        elif op == 0x8f and modrm is not None:
            arithmetic(4)
            if mode==3: put(rm,True,read[:2])
        elif op in (0xa4,0xa5,0xa6,0xa7,0xaa,0xab,0xae,0xaf): pass
        elif op in (0xe0,0xe1,0xe2): arithmetic(1)
        elif policy == 'immediate_far_jump': segments[1] = [literal,literal]
        elif op in (0xe3,0xe9,0xeb,0x9b,0xa8,0xa9,0xf4,0xf5,0xf8,0xf9,0xfa,0xfb,0xfc,0xfd) or 0x60 <= op <= 0x7f: pass
        elif 0x50 <= op <= 0x57 or op in (6,14,22,30): pass
        elif op in (0xfe,0xff) and reg in (6,7): pass
        elif op < 0x40 and (op&7)<4 and modrm is not None and (op&7)<2 and mode!=3: pass
        elif op in (0xc6,0xc7,0x88,0x89,0x8c) and mode!=3: pass
        else:
            regs = [[other,other] for _ in range(8)]
            segments = [[other,other] for _ in range(4)]
        executed_string = not pending_rep or bool(len(rr) or len(ww))
        if executed_string and op in (0xa4,0xa5,0xa6,0xa7,0xac,0xad): arithmetic(6)
        if executed_string and op in (0xa4,0xa5,0xa6,0xa7,0xaa,0xab,0xae,0xaf): arithmetic(7)
        if executed_string and pending_rep and policy == 'string': arithmetic(1)
        if 0x50 <= op <= 0x57 or op in (6,14,22,30) or (op in (0xfe,0xff) and reg in (6,7)):
            arithmetic(4)
        # TF can vector after an otherwise admitted instruction. Interrupt bus
        # traffic includes four IVT reads and six stack writes. No admitted
        # transfer above generates six stores; poison the post-interrupt state
        # even when the instruction's own stores happened to be recognizable.
        if len(ww) >= 6:
            regs = [[other,other] for _ in range(8)]
            segments = [[other,other] for _ in range(4)]
    if audit_channels:
        return channels
    if channels and all(c == "literal" for c in channels):
        return "literal writer", "partner_writes_from_reset_or_immediate_registers"
    if channels and all(c == "memory" for c in channels):
        if confined and repeated:
            return "confined copy loop", "paired_memory_moves_repeated_start_no_partner_fetch"
        if not repeated:
            return "memory-fed one-pass copier", "paired_memory_moves_distinct_starts"
    return "other", "mixed_or_unresolved_write_provenance"

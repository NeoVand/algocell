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


class DiagnosticMachine(ZMachine):
    def __init__(self, base: ZMachine, events: int = 0):
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
            src = src[:at] + f"\n    census_event({event}u, i8086_pc(), {value});" + src[at:]
        once("fn zil_store_record(o: u32, steps: u32) {", "fn zil_store_record(o: u32, steps: u32) {\n    census_store(o);")
        once("fn zil_reset_lane(lane: u32, seed: u32) {", "fn zil_reset_lane(lane: u32, seed: u32) {\n    census_clear();")
        L, P = base.segment("A")["bytes"], base.ring_bytes
        trace_vars = f"var<private> census_log: array<u32, {5*events}>;" if events else ""
        log = f"""
  if (census_n < {events}u) {{
    let o = census_n * 5u;
    census_log[o] = zil_step; census_log[o+1u] = kind;
    census_log[o+2u] = select(p, addr, kind >= 3u); census_log[o+3u] = value;
    census_log[o+4u] = select(0u, 1u, p >= {L}u && census_written[p] == 0u);
  }}
""" if events else ""
        store = f"for (var i=0u; i < min(census_n, {events}u)*5u; i++) {{ zil_out[o+{old+6}u+i] = census_log[i]; }}" if events else ""
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


def mechanism(log, L):
    """Conservative trace classification; memory copies require read/write evidence.

    One-pass means source/destination movement without repeated instruction-start
    positions in the copying episode; 'confined' additionally excludes B fetches.
    The episode ends at the last partner write. Unrecognized writers stay other.
    """
    if log is None:
        return "other", "trace_overflow"
    writes = log[(log[:, 1] == 2) & (log[:, 2] >= L)]
    if not len(writes):
        return "other", "no_partner_write"
    end = int(writes[-1, 0])
    episode = log[log[:, 0] <= end]
    reads = episode[episode[:, 1] == 1]
    starts = episode[episode[:, 1] == 4, 2]
    confined = not np.any((episode[:, 1] == 0) & (episode[:, 2] >= L))
    repeated = len(np.unique(starts)) < len(starts)
    ops = {int(s): int(v) for s, kind, p, v, origin in episode if kind == 3}
    # Trace-local register provenance for the admitted direct writing forms.
    # Unknown register-producing instructions poison the register inventory;
    # a later immediate load restores just that register. Irrelevant data reads
    # do not turn an immediate-loaded PUSH into a memory-fed writer.
    registers = ["literal"] * 8  # the declared reset, including stack phase
    read_values = [[] for _ in range(8)]
    channels = []
    neutral = {0x90, 0xf4, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd}
    for step in sorted(ops):
        op = ops[step]
        group = episode[episode[:, 0] == step]
        fetched = list(map(int, group[group[:, 1] == 0, 3]))
        while fetched and fetched[0] in (0x26, 0x2e, 0x36, 0x3e, 0xf0, 0xf1, 0xf2, 0xf3):
            fetched.pop(0)
        rr = group[group[:, 1] == 1]
        ww = group[(group[:, 1] == 2) & (group[:, 2] >= L)]
        channel, values = "other", []
        if 0x50 <= op <= 0x57:
            channel, values = registers[op-0x50], read_values[op-0x50]
        elif op == 0xff and len(fetched) >= 2 and ((fetched[1] >> 3) & 7) in (6, 7):
            if fetched[1] >> 6 != 3:
                channel, values = "memory", list(rr[:, 3])
            else:
                reg = fetched[1] & 7
                channel, values = registers[reg], read_values[reg]
        elif op in (0xa4, 0xa5):
            channel, values = "memory", list(rr[:, 3])
        elif op in (0xaa, 0xab):
            channel, values = registers[0], read_values[0]
        elif op in (0xc6, 0xc7):
            channel = "literal"
        for write in ww:
            rmw = (op < 0x40 and (op & 7) < 4) or op in (0x80, 0x81, 0x82, 0x83, 0xd0, 0xd1, 0xd2, 0xd3, 0xfe, 0xff)
            if channel == "other" and rmw and np.any((rr[:, 2] == write[2]) & (rr[:, 3] == write[3])):
                # A no-change RMW adds coverage/transactions but no new byte
                # provenance. B9 51's bootstrap executes such ADDs between pushes.
                continue
            channels.append(channel if channel != "memory" or write[3] in values else "other")
        if 0xb8 <= op <= 0xbf:
            registers[op-0xb8] = "literal"; read_values[op-0xb8] = []
        elif op in (0xac, 0xad):
            registers[0] = "memory" if op == 0xad else "other"; read_values[0] = list(rr[:, 3])
        elif 0x58 <= op <= 0x5f:
            registers[op-0x58] = "memory"; read_values[op-0x58] = list(rr[:, 3])
        elif op in (0x8a, 0x8b) and len(fetched) >= 2:
            dest, source = (fetched[1] >> 3) & 7, fetched[1] & 7
            if op == 0x8a: dest &= 3; source &= 3
            if op == 0x8a:
                registers[dest], read_values[dest] = "other", []
            elif fetched[1] >> 6 == 3:
                registers[dest], read_values[dest] = registers[source], read_values[source]
            else:
                registers[dest], read_values[dest] = "memory", list(rr[:, 3])
        elif op in (0xc4, 0xc5) and len(fetched) >= 2:
            dest = (fetched[1] >> 3) & 7
            registers[dest], read_values[dest] = "memory", list(rr[:2, 3])
        elif 0x40 <= op <= 0x4f:
            reg = op & 7
            if registers[reg] != "literal":
                registers[reg] = "other"; read_values[reg] = []
        elif op in (0xc6, 0xc7) and len(fetched) >= 2:
            if fetched[1] >> 6 == 3:
                reg = fetched[1] & 7
                if op == 0xc7 or registers[reg & 3] == "literal":
                    registers[reg if op == 0xc7 else reg & 3] = "literal"
                    read_values[reg if op == 0xc7 else reg & 3] = []
                else:
                    registers[reg & 3] = "other"; read_values[reg & 3] = []
        elif 0x91 <= op <= 0x97:
            reg = op & 7
            registers[0], registers[reg] = registers[reg], registers[0]
            read_values[0], read_values[reg] = read_values[reg], read_values[0]
        elif op in (0xa4, 0xa5, 0xaa, 0xab) or 0x50 <= op <= 0x57 or op in neutral or 0x60 <= op <= 0x7f:
            pass
        elif op < 0x40 and (op & 7) < 6:
            # Memory-destination ALU does not modify general registers.
            if not (len(fetched) >= 2 and (op & 7) < 2 and fetched[1] >> 6 != 3):
                registers = ["other"] * 8; read_values = [[] for _ in range(8)]
        else:
            registers = ["other"] * 8; read_values = [[] for _ in range(8)]
    if channels and all(c == "literal" for c in channels):
        return "literal writer", "partner_writes_from_reset_or_immediate_registers"
    if channels and all(c == "memory" for c in channels):
        if confined and repeated:
            return "confined copy loop", "paired_memory_moves_repeated_start_no_partner_fetch"
        if not repeated:
            return "memory-fed one-pass copier", "paired_memory_moves_distinct_starts"
    return "other", "mixed_or_unresolved_write_provenance"

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
    if not len(reads):
        return "literal writer", "no_data_reads_in_writing_episode"
    starts = episode[episode[:, 1] == 4, 2]
    confined = not np.any((episode[:, 1] == 0) & (episode[:, 2] >= L))
    repeated = len(np.unique(starts)) < len(starts)
    ops = {int(s): int(v) for s, kind, p, v, origin in episode if kind == 3}
    # Every partner write must pair with a same-valued data read in the same
    # MOVS/PUSH-memory step or the immediately preceding LODS/MOV-load step.
    matched = True
    for step, kind, pos, value, origin in writes:
        op = ops.get(int(step), -1)
        if op in (0xa4, 0xa5, 0xff):
            rr = reads[reads[:, 0] == step]
        elif 0x50 <= op <= 0x57 and ops.get(int(step)-1) in (0xac, 0xad, 0x8a, 0x8b):
            rr = reads[reads[:, 0] == step-1]
        else:
            matched = False
            break
        if not np.any(rr[:, 3] == value):
            matched = False
            break
    if matched:
        if confined and repeated:
            return "confined copy loop", "paired_memory_moves_repeated_start_no_partner_fetch"
        if not repeated:
            return "memory-fed one-pass copier", "paired_memory_moves_distinct_starts"
    return "other", "mixed_or_unresolved_write_provenance"

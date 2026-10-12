"""Observation-only 6502 causal labels and ordered accesses on frozen machines.

Pure loads/transfers/stores and PLA/PHA propagate source labels. Arithmetic,
bit masks, status/return pushes and unknown channels conservatively lose their
label. This does not implement an emulator or modify architectural execution.
"""
from __future__ import annotations

import copy
import re

import numpy as np

from algocell_exp.zmachine import ZMachine
from capacity_common import sha


class Trace6502(ZMachine):
    def __init__(self, base, events=1536):
        m=copy.deepcopy(base.m);old=base.record_words;P=base.ring_bytes
        extra=P+1+6*events;m['record']['wordsPerLane']+=extra
        m['record']['fields'].append({'name':'retentionProvenance','offset':old,'words':extra})
        self.offset,self.events,self.P=old,events,P
        src=base.wgsl
        def once(a,b):
            nonlocal src
            assert src.count(a)==1,a
            src=src.replace(a,b)
        once(f'const ZIL_RECORD_WORDS: u32 = {old}u;',f'const ZIL_RECORD_WORDS: u32 = {old+extra}u;')
        once('fn mem_read(addr: u32)','fn obs_real_read(addr: u32)')
        once('fn mem_write(addr: u32, val: u32)','fn obs_real_write(addr: u32, val: u32)')
        once('let v = mem_read(m6502_pc);','let v = obs_read(m6502_pc,0u);')
        once('fn zil_reset_lane(lane: u32, seed: u32) {','fn zil_reset_lane(lane: u32, seed: u32) {\n obs_clear();')
        once('fn zil_store_record(o: u32, steps: u32) {','fn zil_store_record(o: u32, steps: u32) {\n obs_store(o);')
        once('fn m6502_alu(f: u32, v: u32) {','fn m6502_alu(f: u32, v: u32) {\n obs_alu(f);')
        once('    mem_write(addr, w);','    obs_write_origin=0xffffffffu; obs_write_source=0xffffffffu;\n    if(src<3u){obs_write_origin=obs_reg_origin[src];obs_write_source=obs_reg_source[src];}\n    mem_write(addr, w);obs_write_origin=0xffffffffu;obs_write_source=0xffffffffu;')
        once('case 8u: { m6502_push(select(m6502_a, m6502_p | 0x30u, op == 0x08u)); }',
             'case 8u: { if(op==0x48u){obs_write_origin=obs_reg_origin[0];obs_write_source=obs_reg_source[0];} m6502_push(select(m6502_a, m6502_p | 0x30u, op == 0x08u)); obs_write_origin=0xffffffffu;obs_write_source=0xffffffffu; }')
        once('else { m6502_a = v; m6502_nz(v); }','else { obs_reg_origin[0]=obs_read_origin;obs_reg_source[0]=obs_read_source; m6502_a = v; m6502_nz(v); }')
        once('            let dst = (d >> 26u) & 3u;','            let dst = (d >> 26u) & 3u;\n            let source=(d >> 24u)&3u; obs_reg_origin[dst]=select(0xffffffffu,obs_reg_origin[source],step==0u);obs_reg_source[dst]=select(0xffffffffu,obs_reg_source[source],step==0u);')
        once('if ((d & 0x40u) != 0u) { m6502_a = m6502_shift(sop, m6502_a); }','if ((d & 0x40u) != 0u) { obs_reg_origin[0]=0xffffffffu;obs_reg_source[0]=0xffffffffu;m6502_a = m6502_shift(sop, m6502_a); }')
        for name,kind,value in [('on_decode',3,'key'),('on_instruction_start',4,'unit')]:
            match=list(re.finditer(rf'fn {name}\([^{{]+\{{',src));assert len(match)==1
            at=match[0].end();src=src[:at]+f'\n obs_event({kind}u,m6502_pc,{value},0xffffffffu,0xffffffffu);'+src[at:]
        src+=f'''
var<private> obs_origin:array<u32,{P}>;
var<private> obs_reg_origin:array<u32,4>;
var<private> obs_reg_source:array<u32,4>;
var<private> obs_read_origin:u32;
var<private> obs_read_source:u32;
var<private> obs_write_origin:u32;
var<private> obs_write_source:u32;
var<private> obs_n:u32;
var<private> obs_log:array<u32,{events*6}>;
fn obs_clear(){{
 for(var i=0u;i<{P}u;i++){{obs_origin[i]=i;}}
 for(var i=0u;i<4u;i++){{obs_reg_origin[i]=0xffffffffu;obs_reg_source[i]=0xffffffffu;}}
 obs_n=0u;obs_read_origin=0xffffffffu;obs_read_source=0xffffffffu;obs_write_origin=0xffffffffu;obs_write_source=0xffffffffu;
}}
fn obs_event(kind:u32,addr:u32,v:u32,origin:u32,source:u32){{
 if(obs_n<{events}u){{let i=obs_n*6u;obs_log[i]=zil_step;obs_log[i+1u]=kind;obs_log[i+2u]=zil_pos(addr);obs_log[i+3u]=v;obs_log[i+4u]=origin;obs_log[i+5u]=source;}}obs_n+=1u;
}}
fn obs_read(addr:u32,kind:u32)->u32{{
 let p=zil_pos(addr);let v=obs_real_read(addr);obs_read_origin=obs_origin[p];obs_read_source=p;obs_event(kind,addr,v,obs_origin[p],p);return v;
}}
fn mem_read(addr:u32)->u32{{return obs_read(addr,1u);}}
fn mem_write(addr:u32,val:u32){{
 obs_origin[zil_pos(addr)]=obs_write_origin;obs_event(2u,addr,val & 255u,obs_write_origin,obs_write_source);obs_real_write(addr,val);
}}
fn obs_alu(f:u32){{
 if(f==6u || f==11u || f==12u || f==13u){{return;}}
 if(f==5u || f==8u || f==9u || f==10u){{
  if(f==5u || f==10u){{obs_reg_origin[0]=obs_read_origin;obs_reg_source[0]=obs_read_source;}}
  if(f==8u || f==10u){{obs_reg_origin[1]=obs_read_origin;obs_reg_source[1]=obs_read_source;}}
  if(f==9u){{obs_reg_origin[2]=obs_read_origin;obs_reg_source[2]=obs_read_source;}}
 }}else{{
  if(f==19u){{obs_reg_origin[1]=0xffffffffu;obs_reg_source[1]=0xffffffffu;}}
  else{{obs_reg_origin[0]=0xffffffffu;obs_reg_source[0]=0xffffffffu;}}
  if(f==18u || f==20u){{obs_reg_origin[1]=0xffffffffu;obs_reg_source[1]=0xffffffffu;}}
  if(f==20u){{obs_reg_origin[3]=0xffffffffu;obs_reg_source[3]=0xffffffffu;}}
 }}
}}
fn obs_store(o:u32){{
 for(var i=0u;i<{P}u;i++){{zil_out[o+{old}u+i]=obs_origin[i];}}
 zil_out[o+{old+P}u]=obs_n;
 for(var i=0u;i<min(obs_n,{events}u)*6u;i++){{zil_out[o+{old+P+1}u+i]=obs_log[i];}}
}}
'''
        m['sha256']['wgsl']=sha(src.encode())
        super().__init__(src,m,name=base.name+'_retention_trace',device=base.dev)

    def logs(self,r):
        logs=[]
        for row in r.record:
            n=int(row[self.offset+self.P])
            logs.append(None if n>self.events else row[self.offset+self.P+1:self.offset+self.P+1+6*n].reshape(n,6))
        return logs

    def labels(self,r):
        return r.record[:,self.offset:self.offset+self.P]

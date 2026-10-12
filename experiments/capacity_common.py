"""Shared frozen-assay I/O, local burst accounting and observational provenance.

No instruction semantics are implemented here. Z80 labels follow only the
existing LDI memory move; all other writes get an unknown source label.
"""
from __future__ import annotations

import csv
import hashlib
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from algocell_exp import assay as A
from z80_ring_gpu import build

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED = 20261012
UNKNOWN = 0xffffffff


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hx(t):
    return bytes(t).hex(" ")


def parse(t):
    return np.frombuffer(bytes.fromhex(t), np.uint8).copy()


def write(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def committed(path):
    rel = path.relative_to(ROOT)
    frozen = subprocess.check_output(["git", "show", f"HEAD:{rel}"], cwd=ROOT)
    if frozen != path.read_bytes():
        raise RuntimeError(f"commit before assay: {rel}")


def partners(L):
    return np.vstack((np.zeros((1, L), np.uint8),
                      np.random.default_rng([SEED, L]).integers(0, 256, (16, L), dtype=np.uint8)))


def fresh(L, key, gen, n):
    k = int(sha(key.encode())[:8], 16)
    return np.random.default_rng([SEED, L, k, gen]).integers(0, 256, (n, L), dtype=np.uint8)


class Bursts:
    def __init__(self, out):
        self.path = out / "bursts.csv"
        self.rows = []
        if self.path.exists():
            self.rows = list(csv.DictReader(self.path.open()))

    def run(self, label, fn):
        if sum(float(r["active_s"]) for r in self.rows) >= 1800:
            raise RuntimeError("30-minute local compute cap reached; report remaining plan")
        def timeout(*_):
            raise TimeoutError("local burst reached 58 seconds")
        prev = signal.signal(signal.SIGALRM, timeout)
        signal.alarm(58)
        start = time.monotonic()
        try:
            result = fn()
        finally:
            elapsed = time.monotonic() - start
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)
            self.rows.append({"label": label, "active_s": f"{elapsed:.6f}", "pause_s": f"{elapsed:.6f}"})
            write(self.path, self.rows)
            time.sleep(elapsed)
        return result


class Z80Provenance:
    def __init__(self, L, burst, events=0):
        self.L, self.P, self.events, self.burst = L, 2*L, events, burst
        path = A._executor_path(L, None, False)
        src = path.read_text()
        self.base_hash = sha(src.encode())
        def once(old, new):
            nonlocal src
            assert src.count(old) == 1, old
            src = src.replace(old, new)
        extra = self.P + 1 + events*5
        once('fn mem_read(addr: u32) -> u32 { return mem[addr % MEM_LENGTH]; }',
             'fn mem_read(addr: u32) -> u32 { let p=addr%MEM_LENGTH; obs_event(1u,p,mem[p],obs_origin[p]); return mem[p]; }')
        once('\tmem[a] = val & 0xffu;',
             '\tobs_origin[a]=obs_copy_origin; obs_event(2u,a,val & 255u,obs_copy_origin);\n\tmem[a] = val & 0xffu;')
        once('    let val = mem_read(cpu_pc);',
             '    let p=cpu_pc%MEM_LENGTH; let val=mem[p]; obs_event(0u,p,val,obs_origin[p]);')
        once('fn z80_ldi() {\n    let val = mem_read(get_hl());\n    mem_write(get_de(), val);',
             'fn z80_ldi() {\n    let val = mem_read(get_hl());\n    obs_copy_origin=obs_origin[get_hl()%MEM_LENGTH];\n    mem_write(get_de(), val);\n    obs_copy_origin=0xffffffffu;')
        once('\tfor (var s = 0u; s < params.z80_steps; s++) {',
             '\tobs_n=0u; obs_copy_origin=0xffffffffu; for(var i=0u;i<MEM_LENGTH;i++){obs_origin[i]=i;}\n\tfor (var s = 0u; s < params.z80_steps; s++) {\n        obs_step=s;')
        once('\tlet rbase = case_id * 12u;', f'\tlet rbase = case_id * {12+extra}u;')
        anchor='regs[rbase+8u]=cpu_sp; regs[rbase+9u]=cpu_pc; regs[rbase+10u]=cpu_writes_a; regs[rbase+11u]=cpu_writes_b;'
        store=f'\n for(var i=0u;i<MEM_LENGTH;i++){{regs[rbase+12u+i]=obs_origin[i];}} regs[rbase+{12+self.P}u]=obs_n;'
        if events:
            store += f'for(var i=0u;i<min(obs_n,{events}u)*5u;i++){{regs[rbase+{13+self.P}u+i]=obs_log[i];}}'
        once(anchor, anchor+store)
        logvars=f'var<private> obs_log: array<u32,{5*events}>;' if events else ''
        logging=(f'if(obs_n<{events}u){{let i=obs_n*5u;obs_log[i]=obs_step;obs_log[i+1u]=kind;obs_log[i+2u]=p;obs_log[i+3u]=v;obs_log[i+4u]=origin;}}' if events else '')
        src+=f'\nvar<private> obs_origin: array<u32,{self.P}>; var<private> obs_copy_origin:u32; var<private> obs_n:u32; var<private> obs_step:u32; {logvars}\nfn obs_event(kind:u32,p:u32,v:u32,origin:u32){{{logging}obs_n+=1u;}}'
        self.source_hash = sha(src.encode())
        self.pipe, self.layout = burst.run(f"compile-Z80-L{L}-events{events}", lambda: build(A.get_device(), src))
        self.words = 12+extra

    def run(self, pairs, steps=128):
        def dispatch():
            dev = A.get_device()
            n, L = len(pairs), self.L
            wpc=(L+3)//4
            packed=np.zeros((n,8*wpc),np.uint8)
            packed[:,:L], packed[:,4*wpc:4*wpc+L] = pairs[:,:L], pairs[:,L:]
            params=np.zeros(32,np.uint32)
            params[2],params[3],params[4],params[6]=L,2*L,n,steps
            B=A.wgpu.BufferUsage
            io=dev.create_buffer(size=packed.nbytes,usage=B.STORAGE|B.COPY_SRC|B.COPY_DST)
            reg=dev.create_buffer(size=n*self.words*4,usage=B.STORAGE|B.COPY_SRC)
            par=dev.create_buffer(size=params.nbytes,usage=B.UNIFORM|B.COPY_DST)
            try:
                dev.queue.write_buffer(io,0,packed);dev.queue.write_buffer(par,0,params)
                bg=dev.create_bind_group(layout=self.layout,entries=[{'binding':i,'resource':{'buffer':b,'offset':0,'size':b.size}} for i,b in enumerate((par,io,reg))])
                enc=dev.create_command_encoder();p=enc.begin_compute_pass()
                p.set_pipeline(self.pipe);p.set_bind_group(0,bg);p.dispatch_workgroups((n+63)//64);p.end()
                dev.queue.submit([enc.finish()])
                out=np.frombuffer(dev.queue.read_buffer(io),np.uint8).reshape(n,8*wpc)
                regs=np.frombuffer(dev.queue.read_buffer(reg),np.uint32).reshape(n,self.words).copy()
                mem=np.concatenate((out[:,:L],out[:,4*wpc:4*wpc+L]),1).copy()
            finally:
                for b in (io,reg,par): b.destroy()
            return mem,regs
        mem,regs=self.burst.run(f"Z80-L{self.L}-n{len(pairs)}-steps{steps}",dispatch)
        raw,rr=self.burst.run(f"raw-Z80-L{self.L}-n{len(pairs)}-steps{steps}",lambda:A._shader_execute_pairs(pairs,self.L,steps,return_regs=True))
        assert np.array_equal(mem,raw) and np.array_equal(regs[:,:12],rr), "instrumentation changed execution"
        return mem,regs

    def labels(self, regs):
        return regs[:,12:12+self.P]

    def logs(self, regs):
        assert self.events
        logs=[]
        for row in regs:
            n=int(row[12+self.P]);assert n<=self.events,"trace overflow"
            logs.append(row[13+self.P:13+self.P+5*n].reshape(n,5))
        return logs

"""Fresh fixed-input Gate 2 re-audit; foreground local GPU bursts only."""
import csv
import gzip
import json
import io
import subprocess
from pathlib import Path
import numpy as np
from census8086 import OUT,pairs,matches,burst,write_csv,sha
from algocell_exp.zcensus import tile_tapes
from algocell_exp.zmachine import load_machine
from algocell_exp.census8086_trace import DiagnosticMachine,mechanism,same_execution


class TraceParts:
    fields=['k','index','partner','step','kind','pos','value','original_partner']
    def __init__(self,prefix): self.prefix=prefix; self.count=0; self.part=0; self.file=None
    def __enter__(self): return self
    def writerows(self,rows):
        for row in rows:
            if self.count%1_000_000==0:
                if self.file: self.file.close()
                self.part+=1
                path=OUT/f'{self.prefix}_part{self.part:03d}.csv.gz'
                self.file=gzip.open(path,'wt',newline='')
                self.writer=csv.DictWriter(self.file,fieldnames=self.fields); self.writer.writeheader()
            self.writer.writerow(row); self.count+=1
    def __exit__(self,*args):
        if self.file: self.file.close()

def main():
    changes=[]; runs=[]
    for L in (16,32):
        m=burst(f'correction-compile-L{L}',lambda:load_machine(f'i8086_pair_maps_L{L}'))
        d=burst(f'correction-observer-L{L}',lambda:DiagnosticMachine(m,events=8192,storage_trace=True))
        for budget in (128,512):
            path=OUT/f'mechanisms_L{L}_b{budget}.csv'
            old=list(csv.DictReader(io.StringIO(subprocess.check_output(['git','show',f'b21e0f7:experiments/results/census8086/{path.name}'],text=True))))
            lookup={(int(r['k']),int(r['index']),int(r['partner'])):r for r in old}
            wanted=sorted({(k,i) for k,i,v in lookup})
            result=[]; digest=[]
            with TraceParts(f'traces_L{L}_b{budget}') as w:
                for lo in range(0,len(wanted),16):
                    batch=wanted[lo:lo+16]
                    A=np.stack([tile_tapes(np.array([idx],np.uint64),k,L)[0] for k,idx in batch]); mem=pairs(A,L)
                    raw=burst(f'correction-L{L}-b{budget}-{lo}-raw',lambda:m.run(mem,budget=budget))
                    obs=burst(f'correction-L{L}-b{budget}-{lo}-trace',lambda:d.run(mem,budget=budget))
                    assert same_execution(raw,obs,m.record_words)
                    digest.append({'input':sha(mem.tobytes()),'record':sha(raw.record.tobytes()),'memory':sha(raw.memory.tobytes())})
                    logs=d.logs(obs); diags=d.diagnostics(obs)
                    for i,(k,idx) in enumerate(batch):
                        score=matches(raw.memory[i*17:(i+1)*17,L:].reshape(1,17,L),A[i:i+1],k)[0]
                        for v in range(17):
                            log=logs[i*17+v]; cls,evidence=mechanism(log,L); diag=diags[i*17+v]
                            before=lookup[k,idx,v]
                            assert int(before['match'])==int(score[v])
                            assert all(int(before[a])==int(diag[j]) for a,j in [('write_coverage',4),('original_partner_fetch_reads',0),('original_partner_data_reads',1),('events',5)])
                            row={**before,'mechanism':cls,'evidence':evidence,'trace_sha256':sha(log.tobytes()) if log is not None else 'overflow'}
                            result.append(row)
                            if before['mechanism']!=cls:
                                changes.append({'L':L,'budget':budget,'k':k,'word':row['word'],'partner':v,'old':before['mechanism'],'new':cls,'evidence':evidence})
                            if log is not None:
                                w.writerows(dict(zip(w.fields,(k,idx,v,*map(int,event)))) for event in log)
                    print(f'correction L{L} b{budget}: {min(lo+16,len(wanted))}/{len(wanted)} candidates',flush=True)
            write_csv(path,result)
            runs.append({'L':L,'budget':budget,'candidates':len(wanted),'encounters':len(result),'observer_sha256':d.wgsl_sha256,'execution_batches':digest})
    write_csv(OUT/'correction_labels.csv',changes,['L','budget','k','word','partner','old','new','evidence'])
    (OUT/'correction_provenance.json').write_text(json.dumps({'date':'2026-10-12','baseline_commit':'b21e0f7','classifier_sha256':sha(Path(__file__).with_name('algocell_exp').joinpath('census8086_trace.py').read_bytes()),'no_population':True,'all_original_records_and_memory_equal':True,'runs':runs},indent=2)+'\n')
    print(f'{len(changes)} changed labels',flush=True)

if __name__=='__main__': main()

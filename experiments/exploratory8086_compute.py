"""EXPLORATORY / post hoc foreground GPU assays, using frozen executors."""
import collections, csv, gzip, json, types
import numpy as np
import exploratory8086 as E
from algocell_exp.zmachine import load_machine
from algocell_exp.census8086_trace import DiagnosticMachine, same_execution
import law8086 as Law

class Foreground:
    def __init__(self,m):self.m=m
    def run(self,*a,**kw):return E.burst('law: '+self.m.name+': '+str(len(a[0]))+' lanes',lambda:self.m.run(*a,**kw))
    def __getattr__(self,key):return getattr(self.m,key)

def encounters():
    m=E.burst('compile frozen S2 machine',lambda:load_machine('i8086_pair_maps_L16'))
    from algocell_exp import exectrace as X
    rows=[]; events=[]; inputs=[]
    for isa in ('8086','Z80'):
        tiles=['b951','51b9','07a5'] if isa=='8086' else ['01c5']
        for seed in (range(8086001,8086021) if isa=='8086' else range(2001,2021)):
            steps=(0,20000,100000,1000000,3000000) if isa=='8086' else (500,1000,2000,20000,100000)
            for step in steps:
                if isa=='8086':cells,p=E.snapshot(seed,step)
                else:
                    p=E.STAGEG/f'none@closure_L16_st128_k4_s{seed}.soup_t{step}.u8.br'
                    if not p.exists():raise FileNotFoundError(p)
                    import brotli
                    cells=np.frombuffer(brotli.decompress(p.read_bytes()),np.uint8).reshape(-1,16)
                inputs.append({'isa':isa,'world':seed,'step':step,'path':str(p),'sha256':E.digest(p.read_bytes()),'cells':len(cells)})
                rng=np.random.default_rng([E.SEED,seed,step]); ids=rng.choice(len(cells),512,replace=False)
                soup=cells[ids];uniform=rng.integers(0,256,(512,16),dtype=np.uint8)
                for tile in tiles:
                    A=np.tile(np.frombuffer(bytes.fromhex(tile),np.uint8),8)
                    for context,partners in (('soup',soup),('uniform',uniform)):
                        for role in ('A','B'):
                            donor=np.tile(A,(512,1)); mem=np.concatenate((donor,partners),1) if role=='A' else np.concatenate((partners,donor),1)
                            if isa=='8086':
                                result=E.burst(f'encounter {isa} {seed} {step} {tile} {context} {role}',lambda:m.run(mem,budget=128));out=result.memory
                                entered=result.fetch_map[:,16:].any(1)
                                writes=result.writes('B' if role=='A' else 'A')
                            else:
                                out,masks=E.burst(f'encounter {isa} {seed} {step} {tile} {context} {role}',lambda:X.execute_pairs_traced(mem,16,128))
                                entered=X.exec_positions(masks,32)[:,16:].any(1);writes=np.full(512,-1)
                            target=out[:,16:] if role=='A' else out[:,:16];survivor=out[:,:16] if role=='A' else out[:,16:]
                            exact=np.all(target==A,1);phase=np.all(target==np.roll(A,1),1);retained=np.all(survivor==A,1)
                            already=np.all(partners==A,1)|np.all(partners==np.roll(A,1),1)
                            matched=np.maximum((target==A).sum(1),(target==np.roll(A,1)).sum(1))
                            rows.append({'isa':isa,'world':seed,'step':step,'tile':tile,'context':context,'donor_role':role,'partners':512,'physical_phase_exact':int(exact.sum()),'either_tile_phase_exact':int((exact|phase).sum()),'already_exact':int(already.sum()),'new_exact':int(((exact|phase)&~already).sum()),'donor_retained_physical_phase':int(retained.sum()),'partner75':int((matched>=12).sum()),'B_fetch':int(entered.sum()),'mean_target_writes':float(writes.mean()) if isa=='8086' else '', 'input_sha256':E.digest(mem.tobytes()),'output_sha256':E.digest(out.tobytes()),'partner_indices_sha256':E.digest(ids.astype('<u4').tobytes()),'sample_seed':str([E.SEED,seed,step])})
                            for j in range(512):events.append({'isa':isa,'world':seed,'step':step,'tile':tile,'context':context,'donor_role':role,'partner_index':int(ids[j]) if context=='soup' else j,'partner':partners[j].tobytes().hex(),'exact':bool(exact[j]),'either_phase_exact':bool(exact[j] or phase[j]),'already_exact':bool(already[j]),'donor_retained':bool(retained[j]),'A_output':out[j,:16].tobytes().hex(),'B_output':out[j,16:].tobytes().hex(),'B_fetch':bool(entered[j])})
            print(E.LABEL,'encounters',isa,seed,flush=True)
    E.write_csv('encounter_fates.csv',rows);E.write_csv('encounter_fates_raw.csv.gz',events);E.write_csv('stageG_input_hashes.csv',inputs);E.sources()

def transmission():
    m=E.burst('compile transmission machine',lambda:load_machine('i8086_pair_maps_L16'))
    family=E.read_csv(E.OUT/'first_family.csv')
    # A descriptive X1 scan: each byte of the four-byte founder is varied,
    # repeated throughout the donor, then passed eight times to fresh partners.
    cases=[]
    for w in family:
        base=np.frombuffer(bytes.fromhex(w['tape']),np.uint8)
        for pos in range(4):
            for value in range(256):
                founder=base[:4].copy();founder[pos]=value;p=np.tile(founder,4)
                cases.append((w['world'],w['tape'],pos,value,p,value!=base[pos]))
    rows=[];detail=[]
    for lo in range(0,len(cases),256):
        batch=cases[lo:lo+256]; A=np.stack([c[4] for c in batch]);n=len(batch)
        rng=np.random.default_rng([E.SEED,lo]);P=rng.integers(0,256,(n,16,16),dtype=np.uint8)
        mem=np.concatenate((np.repeat(A,16,axis=0),P.reshape(-1,16)),1)
        r=E.burst(f'X1 variants {lo}',lambda:m.run(mem,budget=128)); B=r.memory[:,16:].reshape(n,16,16);ok=np.all(B==A[:,None,:],2)
        current=A.copy();serial=[]
        for g in range(1,9):
            fresh=rng.integers(0,256,(n,16),dtype=np.uint8);mem=np.concatenate((current,fresh),1)
            r=E.burst(f'X1 serial {lo} g{g}',lambda:m.run(mem,budget=128));current=r.memory[:,16:].copy();passed=np.all(current==A,1);serial.append(passed)
            for j,c in enumerate(batch):detail.append({'world':c[0],'baseline':c[1],'position':c[2],'value':c[3],'generation':g,'passed':bool(passed[j]),'partner_sha256':E.digest(fresh[j].tobytes()),'offspring_sha256':E.digest(current[j].tobytes())})
        allserial=np.all(np.array(serial),0)
        for j,c in enumerate(batch):rows.append({'world':c[0],'baseline':c[1],'position':c[2],'value':c[3],'alternative':bool(c[5]),'mutant_tape':c[4].tobytes().hex(),'partners':16,'physical_phase_exact':int(ok[j].sum()),'all16_exact':bool(ok[j].all()),'all8_serial_exact':bool(allserial[j]),'g8_exact':bool(serial[-1][j])})
    E.write_csv('family_transmission.csv',rows);E.write_csv('family_serial.csv.gz',detail)

def law():
    # The exact registered assay function is rebound into independent globals;
    # only the 64-joint subset, extended diagnostic verifier, and foreground
    # execution wrappers differ. The registered module is never monkey-patched.
    pred=E.OUT/'law_panel_predictions.csv'
    subprocess=__import__('subprocess')
    subprocess.run(['git','diff','--exit-code','HEAD','--',str(pred.relative_to(E.HERE.parent))],cwd=E.HERE.parent,check=True)
    subprocess.run(['git','ls-files','--error-unmatch',str(pred.relative_to(E.HERE.parent))],cwd=E.HERE.parent,check=True,stdout=subprocess.DEVNULL)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=E.HERE.parent,text=True).strip()
    m=E.burst('compile law machine',lambda:load_machine('i8086_pair_maps_L16'))
    t=E.burst('compile law storage observer',lambda:DiagnosticMachine(m,events=2048,storage_trace=True))
    fm,ft=Foreground(m),Foreground(t)
    def limited_variants(*args):
        for row in Law.variants(*args):
            if row[1]!='joint' or 256<=int(row[0].split(':')[1])<320:yield row
    env={**Law.__dict__,'variants':limited_variants,'copy_certificate':E.copy_certificate}
    env['diagnosis']=types.FunctionType(Law.diagnosis.__code__,env)
    assay=types.FunctionType(Law.assay_case.__code__,env,argdefs=Law.assay_case.__defaults__)
    assay.__kwdefaults__=Law.assay_case.__kwdefaults__
    summary=[]; counts=[]
    folder=E.OUT/'law_encounters';folder.mkdir(exist_ok=True)
    for i,r in enumerate(E.read_csv(pred)):
        p=np.frombuffer(bytes.fromhex(r['tape']),np.uint8).copy();g=json.loads(r['geometry']);c=collections.Counter()
        with gzip.open(folder/f'{i:02d}.csv.gz','wt',newline='') as f:
            fields=['label','case','kind','generation','challenge','partner','partner_bytes','partner_sha256','input_sha256','output_sha256','matched','generation_zero_source_map','physical_entry','phase']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            def emit(row):
                if row['kind']=='candidate-mismatch':
                    E.dump(f'law_mismatch_{i:02d}.json',row);return
                c[(row['kind'],row['challenge'],row['generation'])]+=1
                w.writerow({'label':E.LABEL,**row})
            result=assay(p,g,'A',int(r['seed']),fm,ft,emit)
        result['joint_assignments']=64
        summary.append({'world':r['world'],'seed':r['seed'],'tape':r['tape'],'prediction_commit':commit,'prediction_csv_sha256':E.digest(pred.read_bytes()),'d':g['d'],'K':json.dumps(g['K']),'F':json.dumps(g['F']),'E':json.dumps(g['E']),'status':result['status'],'encounters':result['encounters'],'single_assignments':256*len(g['F']),'joint_assignments':64,'erased_assignments':256*len(g['erasure_controls']),'held_erased_coordinates':len(g['held_erased']),'serial_generations':8,'details':f'law_encounters/{i:02d}.csv.gz','details_sha256':E.digest((folder/f'{i:02d}.csv.gz').read_bytes())})
        for (kind,challenge,generation),n in sorted(c.items()):counts.append({'world':r['world'],'tape':r['tape'],'kind':kind,'challenge':challenge,'generation':generation,'encounters':n})
        print(E.LABEL,'law',i,r['world'],result['status'],result['encounters'],flush=True)
    E.write_csv('law_assay_summary.csv',summary);E.write_csv('law_assay_denominators.csv',counts)

"""EXPLORATORY, post hoc S2 follow-ups; never calls registered reporters.

Run with the prescribed interpreter and -I. All GPU work is synchronous, capped
by conservative 4096-lane batches, logged, and followed by an equal pause.
"""
from __future__ import annotations
import argparse, collections, csv, gzip, hashlib, json, subprocess, sys, time
from pathlib import Path
import brotli
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from exploratory8086_law import copy_certificate
from law8086 import Unresolved
from s8086 import samples
LABEL='EXPLORATORY / post hoc'
ROOT=HERE/'results/s8086'; OUT=ROOT/'exploratory'
RAW=ROOT/'downloads/worlds_A/A'
STAGEG=Path('/Users/neo/repos/algocell/experiments/runs/stageG')
S2=Path('/Users/neo/repos/algocell-wt-s2/experiments/results/s8086')
SEED=0xE8086001

def digest(b): return hashlib.sha256(b).hexdigest()
def read_csv(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open()) as f:return list(csv.DictReader(f))
def write_csv(name,rows,fields=None):
    rows=list(rows); p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True)
    fields=['label']+(fields or list(rows[0]) if rows else fields or [])
    with (gzip.open(p,'wt',newline='') if str(p).endswith('.gz') else p.open('w',newline='')) as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({'label':LABEL,**r})
def dump(name,data): (OUT/name).write_text(json.dumps({'label':LABEL,**data},indent=2)+'\n')
def burst(label,fn):
    start=time.monotonic()
    try:return fn()
    finally:
        elapsed=time.monotonic()-start
        pause=time.monotonic();time.sleep(elapsed); paused=time.monotonic()-pause
        p=OUT/'gpu_bursts.csv'; exists=p.exists()
        with p.open('a',newline='') as f:
            w=csv.writer(f)
            if not exists:w.writerow(['label','operation','burst_seconds','pause_seconds','foreground'])
            w.writerow([LABEL,label,elapsed,paused,True])
        if elapsed>60:raise RuntimeError('GPU burst exceeded 60 seconds')
def frozen():
    for f in sorted((ROOT/'eligibility/A').glob('*.json.gz')):
        with gzip.open(f,'rt') as h:yield f,json.load(h)
def world_tapes():return read_csv(ROOT/'tables/worlds.csv')
def snapshot(seed,t):
    p=RAW/f's8086_A_s{seed}.t{t}.u8.br'
    b=brotli.decompress(p.read_bytes()); assert len(b)==320000
    return np.frombuffer(b,np.uint8).reshape(-1,16),p

def scan():
    words={'b951','51b9','07a5','a507'}
    for L in (16,32):
        for b in (128,512):
            for r in read_csv(HERE/f'results/census8086/mechanisms_L{L}_b{b}.csv'):
                if r['partner']=='0' and int(r['match'])==L and r['k']=='2' and r['mechanism'] in ('literal writer','memory-fed one-pass copier'):
                    words.add(r['word']);words.add(r['word'][2:]+r['word'][:2])
    rows=[]; hits=[]; inputs=[]; coverage=[]
    for seed in range(8086001,8086021):
        p=RAW/f's8086_A_s{seed}.jsonl'; ss=[]
        with p.open() as f:
            for line in f:
                d=json.loads(line)
                if d['kind']=='sample':ss.append(d)
        assert [r['step'] for r in ss]==samples()
        inputs.append({'source':str(p.relative_to(HERE)), 'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())})
        for r in ss:
            tapes=np.array([np.frombuffer(bytes.fromhex(x['tape']),np.uint8) for x in r['top3']]);weights=np.array([x['count'] for x in r['top3']])
            scan_block(tapes,weights,words,seed,r['step'],'scheduled_top3',rows,hits)
        for t in (0,20000,100000,1000000,3000000):
            cells,p=snapshot(seed,t)
            assert digest(cells.tobytes())==next(r['soup_sha256'] for r in ss if r['step']==t)
            inputs.append({'source':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())})
            scan_block(cells,np.ones(len(cells),int),words,seed,t,'snapshot',rows,hits)
            hist=np.bincount(cells.ravel(),minlength=256)
            coverage.append({'world':seed,'step':t,'cells':len(cells),'zero_fraction':hist[0]/cells.size,'distinct_tapes':len(np.unique(cells.view('V16'))),'modal_byte':int(hist.argmax()),'modal_byte_fraction':hist.max()/cells.size})
        print(f'{LABEL}: scanned {seed}',flush=True)
    write_csv('motif_scan.csv.gz',rows);write_csv('motif_hits.csv.gz',hits)
    write_csv('snapshot_composition.csv',coverage);write_csv('input_hashes.csv',inputs)
    dump('scan_contract.json',{'words':sorted(words),'partial_definition':'maximum linear consecutive alternating-template match >=4 bytes; >=8 also tabulated; a pair >=2 is separate; both absolute phases tested; no cross-cell or artificial circular joins','full_definition':'physical phase zero and phase one exact 16-byte alternating tile separately','scheduled_samples':11700,'top3_records':sum(r['tapes'] for r in rows if r['word']=='b951' and r['source']=='scheduled_top3'),'checkpoint_cells':2000000})

def scan_block(cells,weights,words,seed,t,source,rows,hits):
    for word in sorted(words):
        w=np.frombuffer(bytes.fromhex(word),np.uint8); maxrun=np.zeros(len(cells),int)
        exact=[]
        for phase in (0,1):
            match=cells==np.tile(np.roll(w,phase),8)
            run=np.zeros(len(cells),int)
            for col in range(16):
                run=np.where(match[:,col],run+1,0);maxrun=np.maximum(maxrun,run)
            exact.append(np.all(match,axis=1))
        rows.append({'world':seed,'step':t,'source':source,'word':word,'tapes':len(cells),'cell_weight':int(weights.sum()),'pair_tapes':int((maxrun>=2).sum()),'partial4_tapes':int((maxrun>=4).sum()),'partial8_tapes':int((maxrun>=8).sum()),'full_phase0_tapes':int(exact[0].sum()),'full_phase1_tapes':int(exact[1].sum()),'partial4_weight':int(weights[maxrun>=4].sum()),'full_weight':int(weights[exact[0]|exact[1]].sum()),'max_run':int(maxrun.max())})
        for ix in np.flatnonzero(maxrun>=4):
            hits.append({'world':seed,'step':t,'source':source,'index':int(ix),'word':word,'max_run':int(maxrun[ix]),'full_phase':0 if exact[0][ix] else 1 if exact[1][ix] else -1,'count':int(weights[ix]),'tape':cells[ix].tobytes().hex()})

def reextract():
    rows=[]; certificates={}; reasons=collections.Counter(); family_rows=[]; traces=[]
    first={r['world']:r for r in world_tapes()}
    for path,d in frozen():
        world=d['world']['world']
        for tape,c in sorted(d['certificates'].items()):
            p=np.frombuffer(bytes.fromhex(tape),np.uint8); geoms=[]
            try:
                for trace in c['ordered_traces']:geoms.append(copy_certificate(p,np.array(trace,dtype=np.uint32) if trace is not None else None,16))
                g=geoms[0]
                for v in geoms[1:]:
                    if any(v[k]!=g[k] for k in ('d','K','R','C_R','F','E','held_erased','phase','source_start','source_map')):raise ValueError('context_dependent_certificate')
                status='eligible';reason=''; certificates[world+':'+tape]=g
            except ValueError as e:
                status='unresolved' if isinstance(e,Unresolved) else 'excluded';reason=str(e);g={}
            reasons[(status,reason)]+=1
            rows.append({'world':world,'seed':d['world']['seed'],'tape':tape,'registered_status':c['families']['L2']['status'],'registered_reason':c['families']['L2'].get('reason',''),'status':status,'reason':reason,'d':g.get('d'),'K':json.dumps(g.get('K')),'F':json.dumps(g.get('F')),'E':json.dumps(g.get('E')),'geometry':json.dumps(g,sort_keys=True) if g else '', 'cached_contexts':len(geoms) if status=='eligible' else '', 'baseline_closure_basis':'frozen culture-positive result + ordered A-root source map' if status=='eligible' else ''})
            if tape==first[world]['first_tape'] and tape.startswith('5fa4'):
                log=np.array(c['ordered_traces'][0],dtype=np.uint32);starts=log[log[:,1]==4];fetch=log[log[:,1]==0]; writes=log[log[:,1]==2];reads=log[log[:,1]==1]
                moves=[]; ip=[]
                for step in np.unique(log[:,0]):
                    group=log[log[:,0]==step];rr=group[group[:,1]==1];ww=group[group[:,1]==2];st=group[group[:,1]==4]
                    if len(st):ip.append(int(st[0,2]))
                    if len(ww):
                        assert len(rr)==len(ww)==1
                        moves.append((int(step),int(rr[0,2]),int(ww[0,2]),int(rr[0,3])))
                    for s,k,pos,val,origin in group:traces.append({'world':world,'tape':tape,'step':int(s),'kind':int(k),'position':int(pos),'value':int(val),'original_partner':int(origin)})
                xy=bytes.fromhex(tape)[2:4]; firstDI=0xa4+256*xy[0]
                desc= ('JMP rel16, displacement formed by y and overwritten/fetched byte 4' if xy[0]==0xe9 else 'LOOP rel8, initial CX=0 wraps to 65535' if xy[0]==0xe2 else ['JO','JNO','JB','JAE','JE','JNE','JBE','JA','JS','JNS','JP','JNP','JL','JGE','JLE','JG'][xy[0]&15]+' rel8'+(' (NMOS 60–6F alias)' if xy[0]<0x70 else ''))
                signed=xy[1] if xy[1]<128 else xy[1]-256
                family_rows.append({'world':world,'seed':d['world']['seed'],'t_rep':first[world]['t_rep'],'tape':tape,'x':f'{xy[0]:02x}','y':f'{xy[1]:02x}','decode':desc,'rel8_signed':signed if xy[0]!=0xe9 else '', 'SI_initial':0,'DI_after_POP':firstDI,'DI_ring':firstDI%32,'SP_after_POP':3,'ES':0,'DS':0,'per_copy':'[DS:SI] -> [ES:DI]; SI++, DI++','copies':len(moves),'A_write_coverage':len(set(writes[writes[:,2]<16,2])),'B_write_coverage':len(set(writes[writes[:,2]>=16,2])),'physical_fetch_positions':json.dumps(sorted(set(map(int,fetch[:,2])))), 'instruction_start_positions_mod32':json.dumps(sorted(set(int(x)%32 for x in starts[:,2]))),'constant_displacement':sorted(set((b-a)%32 for _,a,b,_ in moves)),'first_moves':json.dumps(moves[:8]),'raw_start_ips_first8':json.dumps(ip[:8]),'geometry':json.dumps(g,sort_keys=True),'status':status,'reason':reason,'trace_sha256':digest(log.astype('<u4').tobytes())})
        print(f'{LABEL}: re-extracted {world}',flush=True)
    assert len(rows)==779
    write_csv('eligibility.csv',rows);write_csv('eligibility_counts.csv',[{'status':s,'reason':r,'cases':n} for (s,r),n in sorted(reasons.items())]);write_csv('first_family.csv',family_rows);write_csv('first_family_traces.csv.gz',traces)
    family_copies()
    dump('certificates.json',{'certificates':certificates,'verifier_sha256':digest((HERE/'exploratory8086_law.py').read_bytes()),'frozen_source_sha256':digest((HERE/'law8086.py').read_bytes()),'interpretation':'post hoc candidate admission; no change to registered L1/L2 verdicts'})
    # Freeze predictions before any payload assay: deterministic world-balanced
    # 20 full-tape genotypes, ten first-family genotypes, then up to two additions.
    panel=[];eligible=[r for r in rows if r['status']=='eligible']
    for world in sorted({r['world'] for r in eligible}):
        choices=[r for r in eligible if r['world']==world and r['d']==16]
        if choices:panel.append(min(choices,key=lambda r:(len(json.loads(r['K'])),r['tape'])))
    for r in eligible:
        if r['tape']==first[r['world']]['first_tape'] and r['tape'].startswith('5fa4') and r not in panel:panel.append(r)
    for r in sorted(eligible,key=lambda r:(r['d'],r['world'],r['tape'])):
        if len(panel)>=32:break
        if r not in panel:panel.append(r)
    assert len(panel)<=32
    write_csv('law_panel_predictions.csv',[{'world':r['world'],'seed':r['seed'],'tape':r['tape'],'geometry':r['geometry'],'d':r['d'],'K':r['K'],'F':r['F'],'E':r['E'],'predicted_free_count':len(json.loads(r['F']))} for r in panel])

def archive():
    dest=ROOT/'launch/exploratory-publication-archive'; rows=[]
    paths=list((S2/'launch').rglob('*'))+list((S2/'billing-poller').rglob('*'))
    for p in sorted(paths):
        if not p.is_file():continue
        rel=p.relative_to(S2); b=p.read_bytes(); out=dest/rel
        if len(b)>1000000 and not p.suffix in ('.gz','.br','.zip'):
            out=Path(str(out)+'.gz'); stored=gzip.compress(b,mtime=0)
        else:stored=b
        assert len(stored)<50000000
        out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(stored)
        rows.append({'original_path':str(rel),'archive_path':str(out.relative_to(ROOT)),'original_bytes':len(b),'stored_bytes':len(stored),'original_sha256':digest(b),'stored_sha256':digest(stored),'compression':'gzip' if out.suffix=='.gz' and p.suffix!='.gz' else 'unchanged'})
    write_csv('publication_archive.csv',rows)
    (dest/'README.md').write_text('# EXPLORATORY / post hoc publication-provenance archive\n\nByte-preserving archive of the S2 worktree launch directories and billing-poller. Original ledgers, watchdog logs, handoffs and run records retain their historical meaning. Files over 1 MB are deterministically gzipped; manifest: `../../exploratory/publication_archive.csv`. Nothing was uploaded, pushed, or executed from this archive.\n')
    print(LABEL,'archived',len(rows),'files',sum(r['stored_bytes'] for r in rows),'bytes',flush=True)

def family_copies():
    """EXPLORATORY per-copy pointer states reconstructed from verified traces.

    Only one POP DI occurs; DF stays zero and MOVSB is the only pointer update.
    Check reconstructed full 16-bit registers against every physical read/write.
    """
    family={r['world']:r for r in read_csv(OUT/'first_family.csv')}
    groups=collections.defaultdict(list)
    for r in read_csv(OUT/'first_family_traces.csv.gz'):
        groups[(r['world'],int(r['step']))].append(r)
    counts=collections.Counter();rows=[]
    for (world,step),group in sorted(groups.items()):
        writes=[r for r in group if r['kind']=='2']
        if not writes:continue
        reads=[r for r in group if r['kind']=='1'];assert len(reads)==len(writes)==1
        i=counts[world];counts[world]+=1;si=i;di=(int(family[world]['DI_after_POP'])+i)&65535
        assert si%32==int(reads[0]['position']) and di%32==int(writes[0]['position'])
        starts=[int(r['position']) for r in group if r['kind']=='4'];assert len(starts)==1
        rows.append({'world':world,'tape':family[world]['tape'],'step':step,'copy_index':i,'raw_IP':starts[0],'DS':0,'ES':0,'SI_before':si,'DI_before':di,'SI_after':si+1,'DI_after':(di+1)&65535,'read_position':si%32,'write_position':di%32,'source_ancestor':i%4,'value':int(reads[0]['value']),'original_partner_read':reads[0]['original_partner'],'pointer_basis':'reconstructed from reset, single POP, DF=0, MOVSB increments; all physical positions checked'})
    assert len(rows)==640 and all(counts[w]==64 for w in family)
    write_csv('first_family_copies.csv',rows)

def sources():
    """EXPLORATORY hashes of the unchanged executors and extended verifier."""
    from algocell_exp import assay as A
    from algocell_exp.gen_trace_shader import ensure
    raw=A._executor_path(16,None,False);traced=ensure(raw)
    paths=[HERE/'algocell_exp/machines/i8086_pair_maps_L16.wgsl',HERE/'algocell_exp/machines/i8086_pair_maps_L16.manifest.json',raw,traced,HERE/'law8086.py',HERE/'exploratory8086_law.py',HERE/'algocell_exp/exectrace.py']
    write_csv('compute_source_hashes.csv',[{'role':'StageG executor' if p==raw else 'StageG traced executor' if p==traced else 'frozen or exploratory source','path':str(p),'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())} for p in paths])

def world_family_metadata():
    """EXPLORATORY distinguish unjudged tapes from culture-negative tapes."""
    coverage=read_csv(OUT/'family_world_coverage.csv');exemplars=read_csv(OUT/'family_world_exemplars.csv')
    lookup={r['world']:r for r in exemplars}
    for r in coverage:
        seed=int(r['seed']);world=r['world'];tape=r['exemplar'];counts=[]
        with (RAW/(world+'.jsonl')).open() as f:
            for line in f:
                d=json.loads(line)
                if d['kind']=='sample':counts.extend(int(t['count']) for t in d['top3'] if t['tape']==tape)
        snap_genotypes=set();snap_cells=0
        for t in (0,20000,100000,1000000,3000000):
            cells,_=snapshot(seed,t);mask=(cells[:,0]==0x5f)&(cells[:,1]==0xa4)&np.all(cells==np.tile(cells[:,:4],(1,4)),1)
            snap_cells+=int(mask.sum());snap_genotypes.update(x.tobytes().hex() for x in cells[mask])
        p=ROOT/'downloads/score_A/A'/(world+'.json.gz')
        with gzip.open(p,'rt') as f:d=json.load(f)
        judged=next((x for x in d['judged_tapes'] if x['tape']==tape),None)
        status='culture-positive' if judged and judged['heritable'] else 'culture-negative' if judged else 'unjudged' if tape else 'absent'
        r.update({'registered_culture_status':status,'exemplar_max_scheduled_cells':max(counts,default=0),'checkpoint_family_genotypes':len(snap_genotypes),'checkpoint_family_cells':snap_cells,'frozen_score_sha256':digest(p.read_bytes())})
        if status=='unjudged':r['selection']='earliest scheduled family tape; unjudged in register'
        if world in lookup:lookup[world].update({'registered_culture_status':status,'exemplar_max_scheduled_cells':max(counts,default=0),'selection':r['selection']})
    for r in coverage:r.pop('label',None)
    for r in exemplars:r.pop('label',None)
    write_csv('family_world_coverage.csv',coverage);write_csv('family_world_exemplars.csv',exemplars)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('scan','reextract','archive','sources','family_copies','world_family_metadata','world_family','encounters','transmission','law','report'));a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.stage in ('encounters','transmission','law','world_family'):
        import exploratory8086_compute as compute
        getattr(compute,a.stage)()
    elif a.stage=='report':
        from exploratory8086_report import report
        report()
    else: globals()[a.stage]()

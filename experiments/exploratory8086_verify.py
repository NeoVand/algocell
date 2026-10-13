"""EXPLORATORY / post hoc CPU-only audit of committed assay/provenance data."""
import collections,csv,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np
import exploratory8086 as E
import law8086 as Law
from census8086 import panel
from s8086 import rng

def verify():
    out=[];n=0;census_panel=panel(16)
    for i,r in enumerate(E.read_csv(E.OUT/'law_assay_summary.csv')):
        p=np.frombuffer(bytes.fromhex(r['tape']),np.uint8).copy()
        g=next(json.loads(x['geometry']) for x in E.read_csv(E.OUT/'law_panel_predictions.csv') if x['world']==r['world'] and x['tape']==r['tape'])
        cases={}
        for name,kind,parent,expect in Law.variants(p,g,'A',int(r['seed'])):
            if kind=='joint' and not 256<=int(name.split(':')[1])<320:continue
            cases[name]=(kind,parent,expect,E.digest(expect.tobytes()))
        assert len(cases)==int(r['single_assignments'])+int(r['joint_assignments'])+int(r['erased_assignments'])
        counts=collections.Counter();seen=set();details=E.OUT/r['details']
        assert E.digest(details.read_bytes())==r['details_sha256']
        with gzip.open(details,'rt') as f:
            for x in csv.DictReader(f):
                assert x['label']==E.LABEL and x['matched']=='True'
                kind,parent,expect,expected_hash=cases[x['case']]
                assert x['output_sha256']==expected_hash and x['kind']==kind
                assert json.loads(x['generation_zero_source_map'])==g['source_map']
                assert int(x['physical_entry'])==0 and int(x['phase'])==g['phase']
                gen=int(x['generation']); challenge=x['challenge']
                key=(x['case'],challenge,gen,x['partner']);assert key not in seen;seen.add(key)
                if challenge=='census':
                    partner=census_panel[int(x['partner'])];current=parent
                    assert gen==1
                else:
                    assert kind!='erased' and 1<=gen<=8
                    partner=rng('A',int(r['seed']),16,'law-serial',r['tape'],x['case'],gen).integers(0,256,16,dtype=np.uint8)
                    assert partner.tobytes().hex()==x['partner_bytes'] and E.digest(partner.tobytes())==x['partner_sha256']
                    current=parent if gen==1 else expect
                assert E.digest(np.concatenate((current,partner)).tobytes())==x['input_sha256']
                counts[(kind,challenge,gen)]+=1;n+=1
        total=sum(counts.values());assert total==int(r['encounters'])
        for kind,assignments in [('single',int(r['single_assignments'])),('joint',int(r['joint_assignments'])),('erased',int(r['erased_assignments']))]:
            assert counts[(kind,'census',1)]==17*assignments
            for generation in range(1,9):assert counts[(kind,'serial',generation)]==(0 if kind=='erased' else assignments)
        out.append({'check':'law exact byte-hash/input/partner/source-map/denominator audit','world':r['world'],'tape':r['tape'],'rows':total,'passed':True})
        print(E.LABEL,'verified law',i,total,flush=True)
    for r in E.read_csv(E.OUT/'publication_archive.csv'):
        p=E.ROOT/r['archive_path'];stored=p.read_bytes();original=gzip.decompress(stored) if r['compression']=='gzip' else stored
        assert E.digest(stored)==r['stored_sha256'] and E.digest(original)==r['original_sha256']
        assert len(original)==int(r['original_bytes']) and len(stored)==int(r['stored_bytes'])
    out.append({'check':'publication archive byte-preservation','world':'all','tape':'','rows':len(E.read_csv(E.OUT/'publication_archive.csv')),'passed':True})
    for r in E.read_csv(E.OUT/'gpu_bursts.csv'):
        assert float(r['burst_seconds'])<=60 and float(r['pause_seconds'])>=float(r['burst_seconds']) and r['foreground']=='True'
    files=list(E.OUT.rglob('*'))+list((E.ROOT/'launch/exploratory-publication-archive').rglob('*'))
    assert all(p.stat().st_size<50000000 for p in files if p.is_file())
    out.append({'check':'foreground GPU bounds and all output files below 50 MB','world':'all','tape':'','rows':len(E.read_csv(E.OUT/'gpu_bursts.csv')),'passed':True})
    E.write_csv('verification.csv',out)
    print(E.LABEL,'audit complete',n,flush=True)

if __name__=='__main__':verify()

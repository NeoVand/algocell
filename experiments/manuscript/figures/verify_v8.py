"""Verify scientific coordinates, exports and honest image-only Astra clearance."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
import pymupdf as fitz

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[1]
OUT=HERE/'out/v8'
D=OUT/'data'
HEIGHTS={**{f'fig{n}':h for n,h in {1:146,2:136,3:166,4:163,5:162}.items()}, **{f'ed8_{n}':(290 if n==5 else 180 if n in [1,3,7,8,9] else 220) for n in range(1,10)}}
PASS='MEETS OR EXCEEDS REFERENCES - NO SUBSTANTIVE DEFECTS'
FAIL='BELOW STANDARD - SUBSTANTIVE DEFECTS'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def table(name):return pd.read_csv(D/(name+'.csv'),low_memory=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--technical-only',action='store_true');args=ap.parse_args()
    for row in table('sources_v8').itertuples():assert sha(EXP/row.path)==row.sha256,row.path
    for row in table('hashes').itertuples():assert sha(D/row.path)==row.sha256,row.path
    assert (OUT/'LEGENDS_V8.md').read_bytes()==(HERE.parent/'LEGENDS_V8.md').read_bytes()
    assert table('missing_trajectories').empty
    for name in ['law_z80_c1','law_6502_c2']:
        q=table(name);np.testing.assert_allclose(q.predicted_free,q.measured_free)
    assert table('law_population').cells.sum()==2343
    q=table('branching_endpoints');cols=['lost','erased','retained','ctrl_alive_n','sites','L']
    np.testing.assert_allclose(q.query("cohort=='C1'").set_index('key').sort_index()[cols],q.query("cohort=='Historical'").set_index('key').sort_index()[cols])
    assert len(table('competition_endpoints'))==80
    d=table('dial_coordinates');assert sorted(d.value)==list(range(256))
    assert d.culture_replicator.sum()==99 and d.prop8_eligible.sum()==11
    q=table('route_by_machine').set_index('machine')
    np.testing.assert_array_equal(q[['worlds','open','confined','censored']],[[80,80,0,0],[20,0,20,0],[20,2,12,6]])
    assert (q[['open','confined','censored']].sum(axis=1)==q.worlds).all()
    worlds=table('arrival_worlds');assert len(worlds)==60 and worlds.groupby('machine').size().eq(20).all()
    q=table('arrival_summary').set_index('machine')
    np.testing.assert_allclose(q.loc[['Z80 G','6502 A','8086 S2'],'median'],[450,1125000,652500])
    cens=table('arrival_censors');assert len(cens)==1 and cens.iloc[0].censored==6 and cens.iloc[0].step==3000000
    assert abs(cens.iloc[0].detected-.7)<1e-12
    q=table('first_copier_tiles');assert len(q)==160 and q.seed.nunique()==10
    source=pd.read_csv(EXP/'results/s8086/tables/worlds.csv').set_index('seed')
    for seed,g in q.groupby('seed'):
        tape=''.join(g.sort_values('position').byte).lower();assert tape==source.loc[seed,'first_tape']
        assert tape==tape[:8]*4 and tape.startswith('5fa4')
        assert g.detection.eq(source.loc[seed,'t_rep']).all()
        assert json.loads(source.loc[seed,'first_mechanism_counts'])=={source.loc[seed,'first_majority_mechanism']:16}
    check=json.loads((OUT/'figcheck.json').read_text());assert set(check)==set(HEIGHTS) and not any(check.values())
    # Guard every delivered SVG, including any future added exports.
    for svg in OUT.glob('*.svg'):
        raw=svg.read_text().lower()
        assert 'data:image/' not in raw and 'base64,' not in raw, f'Embedded raster forbidden: {svg}'
    from legends_v8 import LEGENDS
    assert all(len(s.split())<=250 for s in LEGENDS)
    ed=json.loads((OUT/'ed_sources.json').read_text())
    assert set(ed)=={f'ed8_{n}' for n in range(1,10)}
    for item in ed.values():
        for path,digest in item['sources'].items():assert sha(EXP/path)==digest,path
    eligible=pd.read_csv(EXP/'results/evolved6502/eligibility.csv').query('eligible==1')
    assert (len(eligible),eligible.tape.nunique(),eligible.world.nunique())==(42,41,11)
    for path in ['results/offset/offset_switch.csv','results/tier2/scored/E/offset_switch.csv']:
        assert len(pd.read_csv(EXP/path))==110
    panel=pd.read_csv(EXP/'results/s8086/exploratory/law_assay_summary.csv')
    assert (len(panel),panel.tape.nunique(),panel.world.nunique(),panel.encounters.sum())==(32,32,20,2224640)
    assert panel.status.eq('complete-zero-mismatch').all()
    assert pd.read_csv(EXP/'results/s8086/tables/law_eligible_cases.csv').empty
    records={}
    for name,height in HEIGHTS.items():
        pdf=fitz.open(OUT/(name+'.pdf'));assert len(pdf)==1
        page=pdf[0];np.testing.assert_allclose([page.rect.width,page.rect.height],np.array([180,height])*72/25.4,atol=.02)
        spans=[s for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans']]
        assert len(spans)>10
        for s in spans:
            x0,y0,x1,y1=s['bbox'];assert x0>=-.1 and y0>=-.1 and x1<=page.rect.width+.1 and y1<=page.rect.height+.1,(name,s)
        if name=='fig3':
            txt=page.get_text();assert 'Registered 8086 law test: untested' in txt and 'Frozen extractor certified no eligible copier' in txt and 'pending' not in txt.lower()
        pdf.close()
        root=ET.parse(OUT/(name+'.svg')).getroot();assert len(list(root.iter('{http://www.w3.org/2000/svg}text')))>10
        linked={}
        for item in root.iter('{http://www.w3.org/2000/svg}image'):
            href=item.get('{http://www.w3.org/1999/xlink}href',item.get('href',''))
            assert not href.lower().startswith('data:'), (name,'embedded raster forbidden')
            assert href and not Path(href).is_absolute() and (OUT/href).is_file(),(name,href)
            linked[href]=sha(OUT/href)
        png=Image.open(OUT/(name+'.png'));assert list(png.size)==[math.ceil(mm/25.4*600) for mm in [180,height]]
        assert png.mode=='RGB' and max(abs(v-600) for v in png.info['dpi'])<.01
        rec=dict(page_mm=[180,height],png_pixels=list(png.size),dpi=list(png.info['dpi']),pdf_text_spans=len(spans),figcheck='clean',files={ext:sha(OUT/(name+'.'+ext)) for ext in ['pdf','svg','png']})
        rec['linked_rasters']=linked
        if not args.technical_only:
            rounds=sorted((OUT/'critique_review'/name).glob('round*'));assert 1<=len(rounds)<=2
            for folder in rounds:
                m=json.loads((folder/'manifest.json').read_text());assert m['returncode']==0 and m['image_only']
                assert m['model']=='gpt-6-astra' and m['reasoning_effort']=='high'
                for filename,digest in m['images'].items():assert sha(folder/filename)==digest
                assert json.loads((folder/'figcheck.json').read_text())[name]==[]
                assert set(m['images'])=={'reference1_fig1.png','reference2_fig4v5.png',name+'.png'}
            last=rounds[-1];review=(last/'critique.md').read_text()
            verdict=review.strip().splitlines()[-1].strip('* ');assert verdict in [PASS,FAIL],verdict
            review_hash=json.loads((last/'manifest.json').read_text())['images'][name+'.png']
            final_reviewed=review_hash==sha(OUT/(name+'.png'))
            rec.update(critique_rounds=len(rounds),final_verdict=verdict,cleared=verdict==PASS and final_reviewed,
                       delivered_image_reviewed=final_reviewed,post_review_revision=not final_reviewed,last_review_png_sha256=review_hash,final_review=str(last.relative_to(OUT)/'critique.md'))
        records[name]=rec
    if args.technical_only:
        print('All input/output hashes, scientific coordinates, vector exports, 600-dpi metadata and figcheck passed.');return
    manifest=dict(technical_checks='passed',figures=records,data_sha256=sha(D/'hashes.csv'),legends_sha256=sha(OUT/'LEGENDS_V8.md'),
                  scope='Full-review figure corrections and nine archived-data Extended Data displays; no simulation or GPU compute',ed_sources_sha256=sha(OUT/'ed_sources.json'),
                  source_scripts={p.name:sha(p) for p in [HERE/(s+'.py') for s in ['prepare_v8','build_v8','build_ed_v8','legends_v8','critique_v8','verify_v8','figcheck']]})
    (OUT/'qa_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report(records)
    print('Technical verification passed. Astra cleared: '+', '.join(k for k,v in records.items() if v['cleared']))


def report(records):
    lines=['# Figures and Extended Data v8','',
           f'Technical verification passed for all {len(records)} figures. '+
           f'Astra cleared {sum(r["cleared"] for r in records.values())}/{len(records)} delivered images. '+
           'Clearance requires a no-substantive-defects verdict and an exact reviewed/delivered PNG hash match.','',
           '| Figure | Page (mm) | figcheck | Review rounds | Clearance |',
           '|---|---|---|---|---|']
    for name,r in records.items():
        state='CLEARED' if r['cleared'] else 'REVISED AFTER REVIEW; NOT CLEARED' if r['post_review_revision'] else 'NOT CLEARED'
        lines.append(f"| [{name}]({name}.png) | {' × '.join(map(str,r['page_mm']))} | clean | {r['critique_rounds']} | [{state}]({r['final_review']}) |")
    lines+=['','PDF and SVG retain editable chart text; PNGs are rendered from PDF at 600 dpi. '+
            'Figure 1 links its raster render to relative companion PNG files. No delivered SVG contains embedded raster data.','',
            'All nine Extended Data displays follow the MAIN_v7 map; legends, source paths and unavailable measurements are in [ED_V8.md](ED_V8.md). '+
            'Pointer randomization is unmeasured and was not an S2 arm.','',
            'The canonical publication legends match the proposed full-review captions (235, 246, 247, 236 and 245 words). '+
            'Registered 8086 law tests remain untested; post hoc results appear only in the explicitly exploratory ED9 panels.','',
            'The prior critique archive is retained. This revision uses `critique_review/`, with at most two image-only gpt-6-astra/high rounds per figure. '+
            'Both original references are attached to every review. Reviews are retained unedited. '+
            'The ED final rounds identified additional terminology issues. Those labels and the complete constructed-control domain were corrected after round 2; '+
            'the resulting ED images have not had a third review and are not represented as Astra-cleared.','',
            '**Reproduce:** Run `prepare_v8.py`, `build_v8.py`, `build_ed_v8.py`, `legends_v8.py`, then `verify_v8.py` with '+
            '`/Users/neo/repos/algocell/experiments/.venv/bin/python`. '+
            'Critique: `critique_v8.py --figure N --round R` (N = 1–5 or ed8_1–ed8_9, R = 1–2).','',
            '[Publication legends](LEGENDS_V8.md) · [QA manifest](qa_manifest.json) · [ED sources](ed_sources.json) · [Frozen main data](data/)','']
    data='\n'.join(lines)
    (OUT/'FIGURES_V8.md').write_text(data)
    import re
    (HERE.parent/'FIGURES_V8.md').write_text(re.sub(r'\]\((?!https?://)([^)]+)\)',r'](figures/out/v8/\1)',data))


if __name__=='__main__':main()

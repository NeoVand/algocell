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
HEIGHTS={1:146,2:136,3:166,4:163,5:162}
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
    check=json.loads((OUT/'figcheck.json').read_text());assert set(check)=={f'fig{i}' for i in HEIGHTS} and not any(check.values())
    records={}
    for n,height in HEIGHTS.items():
        name=f'fig{n}';pdf=fitz.open(OUT/(name+'.pdf'));assert len(pdf)==1
        page=pdf[0];np.testing.assert_allclose([page.rect.width,page.rect.height],np.array([180,height])*72/25.4,atol=.02)
        spans=[s for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans']]
        assert len(spans)>10
        for s in spans:
            x0,y0,x1,y1=s['bbox'];assert x0>=-.1 and y0>=-.1 and x1<=page.rect.width+.1 and y1<=page.rect.height+.1,(name,s)
        if n==3:
            txt=page.get_text();assert '8086 evolved law untested' in txt and 'no eligible certified copier' in txt and 'pending' not in txt.lower()
        pdf.close()
        root=ET.parse(OUT/(name+'.svg')).getroot();assert len(list(root.iter('{http://www.w3.org/2000/svg}text')))>10
        png=Image.open(OUT/(name+'.png'));assert list(png.size)==[math.ceil(mm/25.4*600) for mm in [180,height]]
        assert png.mode=='RGB' and max(abs(v-600) for v in png.info['dpi'])<.01
        rec=dict(page_mm=[180,height],png_pixels=list(png.size),dpi=list(png.info['dpi']),pdf_text_spans=len(spans),figcheck='clean',files={ext:sha(OUT/(name+'.'+ext)) for ext in ['pdf','svg','png']})
        if not args.technical_only:
            rounds=sorted((OUT/'critique'/name).glob('round*'));assert 1<=len(rounds)<=3
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
                       delivered_image_reviewed=final_reviewed,last_review_png_sha256=review_hash,final_review=str(last.relative_to(OUT)/'critique.md'))
        records[name]=rec
    if args.technical_only:
        print('All input/output hashes, scientific coordinates, vector exports, 600-dpi metadata and figcheck passed.');return
    manifest=dict(technical_checks='passed',figures=records,data_sha256=sha(D/'hashes.csv'),legends_sha256=sha(OUT/'LEGENDS_V8.md'),
                  scope='Figures 1-4 cleaned; Figure 5 from completed S2; no new simulation or GPU compute',
                  source_scripts={p.name:sha(p) for p in [HERE/(s+'.py') for s in ['prepare_v8','build_v8','legends_v8','critique_v8','verify_v8']]})
    (OUT/'qa_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report(records)
    print('Technical verification passed. Astra cleared: '+', '.join(k for k,v in records.items() if v['cleared']))


def report(records):
    prefix='figures/out/v8/'
    cleared=[k for k,v in records.items() if v['cleared']]
    lines=['# Figures v8: S2 results and v7 cleanup','',
           f'Technical verification passed for all five figures. Astra cleared {len(cleared)}/5 delivered images. '
           'Clearance requires both a no-substantive-defects verdict and an exact hash match between the reviewed and delivered PNG.','',
           '| Figure | Page (mm) | figcheck | Astra rounds | Clearance |',
           '|---|---|---|---|---|']
    for name,r in records.items():
        state='CLEARED' if r['cleared'] else 'NOT CLEARED'
        lines.append(f"| [{name}]({prefix}{name}.png) | {' × '.join(map(str,r['page_mm']))} | clean | {r['critique_rounds']} | [{state}]({prefix}{r['final_review']}) |")
    lines+=['','All figures are supplied as vector PDF, SVG with editable chart text, and PDF-rendered RGB PNG at 600 dpi. '
            'Every panel is populated. No pending 8086 law entry remains.','',
            'PDFs: '+' · '.join(f'[{name}]({prefix}{name}.pdf)' for name in records)+'. SVGs: '+' · '.join(f'[{name}]({prefix}{name}.svg)' for name in records)+'.','',
            '**Changes.** Figures 1–4 start from the v7 scripts and frozen data. Paragraph-like method notes were moved to '
            '`LEGENDS_V8.md`, preserving direct labels, category headings, inline keys and number columns. '
            'Figures 1 and 2 were compacted after removing prose; Fig. 4 retains its complete dial and paired C1/archive endpoints. '
            'Fig. 3b states that the evolved 8086 law is untested, with no eligible certified copier. '
            'No numerical v7 measurement was changed.','',
            'Fig. 5, “Different machines, different beginnings,” shows all-world route partitions, L16 Kaplan–Meier detection '
            'curves with the six coincident S2 censoring marks directly labelled, a literal-width/pusher/route matrix, and all ten '
            'physical-phase `5F A4 x y` first-copier tapes (160 bytes). The S2 surprise is shown as a teal outlined row: '
            'width 2 and the exact `B9 51` census pusher coexist with 12/14 confined first detections. The two open detections '
            'and six censored worlds remain visible. Registered verdicts are in the legends.','',
            '**Scientific limits.** Route panel a uses all 80 historical Z80 G worlds, while b uses the 20 L16 worlds. '
            'Historical Z80 route evidence, S1 and S2 classifiers, sampling schedules and horizons differ. Timing is descriptive, '
            'with no registered directional timing claim. The S2 law is untested: zero eligible evolved cases do not imply '
            'a successful or failed evolved-law assay. Tile colour shows shared byte values, not certified roots or free payload. '
            'The optional S2 exploratory report was unavailable when panel d was authored.','',
            '**Provenance and review.** Input SHA-256 and plotting-table hashes are frozen under `data/`. S2 survival '
            'coordinates were checked against the frozen Kaplan–Meier table. Every shown first-copier byte and detection '
            'update was checked against the S2 world table. Vector text, PDF page bounds, exact PNG dimensions/DPI, and '
            'clean figcheck were verified. Each Astra round attaches the old `fig1.png` and `fig4v5.png` plus the new image '
            'to the explicitly requested gpt-6-astra/high CLI invocation in an empty directory. Prompt, images, hashes, '
            'unmodified review text and command logs are archived. The first-round prompt incorrectly applied the Fig. 3b law-status requirement to every figure; rounds 2–3 scoped it to Fig. 3b. Every review is retained unchanged. At most three rounds per figure. No simulation, '
            'GPU burst or remote experiment was run. The experiment branch was merged because the local v7 branch '
            'and S2 branch diverged and could not fast-forward.','',
            '**Reproduce.**','', '```sh',
            '/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/figures/prepare_v8.py',
            '/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/figures/build_v8.py',
            '/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/figures/legends_v8.py',
            '/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/figures/verify_v8.py','```','',
            'Astra review: `critique_v8.py --figure N --round R`; archived rounds are never overwritten.','',
            '[Full legends](LEGENDS_V8.md) · [Technical manifest]('+prefix+'qa_manifest.json) · [Frozen data]('+prefix+'data/)','',
            '**Final unedited image-only assessments.**','']
    for name,r in records.items():
        lines += [f'### {name}','',(OUT/r['final_review']).read_text().strip(),'']
    text='\n'.join(lines)
    (HERE.parent/'FIGURES_V8.md').write_text(text)
    (OUT/'FIGURES_V8.md').write_text(text.replace(prefix,''))


if __name__=='__main__':main()

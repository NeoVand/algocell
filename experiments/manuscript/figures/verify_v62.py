"""Export, data-integrity and current-image art-clearance audit for v6.2."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

OUT=Path(__file__).resolve().parent/'out/v6'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks=json.loads((OUT/'figcheck_v62.json').read_text())
    assert len(checks)==32 and not any(checks.values()),checks
    names=[k for k in checks if not k.endswith('_legend')]
    for name in names:
        im=Image.open(OUT/(name+'.png'))
        assert im.width==4252 and abs(im.info['dpi'][0]-600)<.01,(name,im.size,im.info)
        info=subprocess.check_output(['pdfinfo',str(OUT/(name+'.pdf'))],text=True)
        size=re.search(r'Page size:\s+([\d.]+) x ([\d.]+)',info)
        assert size and abs(float(size[1])-180/25.4*72)<.01 and float(size[2])<=170/25.4*72+.01
        root=ET.parse(OUT/(name+'.svg')).getroot()
        assert abs(float(root.attrib['width'].removesuffix('pt'))-180/25.4*72)<.001
        assert len(root.findall('.//{http://www.w3.org/2000/svg}text'))>0,'SVG must retain editable text'
    for name in checks:
        if name.endswith('_legend'):
            im=Image.open(OUT/(name+'.png'))
            assert im.width in (4251,4252) and abs(im.info['dpi'][0]-600)<.01
    data=OUT/'data'
    audit=json.loads((data/'c1_download_audit.json').read_text())
    assert len(audit['worlds'])==40 and all(w['available'] and w['endpoint_verified'] and
        w['machine_hashes_verified'] and w['recorded_marker_verified'] for w in audit['worlds'])
    # Match the downloaded raw hash to the source ledger that generated trajectories.
    sources=pd.read_csv(data/'sources.csv')
    source_hashes=dict(zip(sources.path,sources.sha256))
    for world in audit['worlds']:assert source_hashes[world['raw_path']]==world['raw_sha256']
    assert pd.read_csv(data/'missing_trajectories.csv').empty
    trajectories=pd.read_csv(data/'competition_trajectories.csv')
    assert trajectories.groupby('cohort').apply(lambda g:g[['tar','start','seed']].drop_duplicates().shape[0],include_groups=False).to_dict()=={'C1':40,'Historical':40}
    strips=pd.read_csv(data/'lineage_strips.csv')
    assert len(strips)==64 and set(strips.generation)=={8}
    original=strips[strips.displacement==4];edited=strips[strips.displacement==32]
    assert len(original.query("state=='erased'"))==28 and len(edited.query("state=='transmitted'"))==28
    assert set(original.original_byte)=={4} and set(edited.edited_byte)=={96}
    assert (edited.query("state=='transmitted'").position==edited.query("state=='transmitted'").source_position).all()
    dial=pd.read_csv(data/'dial_coordinates.csv');working=dial[dial.culture_replicator]
    assert len(dial)==256 and len(working)==99 and working.predicted_payload.notna().sum()==11
    assert len(pd.read_csv(data/'ed_parent_environment.csv'))==36
    summary=pd.read_csv(data/'lethality_summary.csv').query("cohort=='C1'").sort_values('p')
    registered=pd.read_csv(data/'c1_dial_summary.csv').sort_values('p')
    assert np.allclose(summary.confinement_km*1000,registered.KM_t_closed)
    for row in pd.read_csv(data/'v62_generated_hashes.csv').itertuples():
        assert sha(data/row.path)==row.sha256, row.path
    clearance=[]
    for name in ['fig1','fig2','fig3','fig4',
                 'ed_v6_reference','ed_v6_erasure','ed_v6_law','ed_v6_joint_table','ed_v6_competition','ed_v6_confinement']:
        rounds=sorted((OUT/'critique_v62'/name).glob('round[1-3]'))
        assert len(rounds)<=3
        final=rounds[-1] if rounds else None
        review=(final/'critique.md').read_text() if final and (final/'critique.md').exists() else ''
        verdict=review.strip().splitlines()[-1].strip('* #') if review else 'NOT REVIEWED'
        manifest=json.loads((final/'manifest.json').read_text()) if final and (final/'manifest.json').exists() else {'images':{},'returncode':None}
        current={file:sha(OUT/file) for file in manifest['images']}
        hashes_match=bool(current) and current==manifest['images']
        clean=verdict=='NO SUBSTANTIVE DEFECTS' and manifest.get('returncode')==0 and hashes_match
        clearance.append(dict(figure=name,figcheck='clean',critic_rounds=len(rounds),last_critic_verdict=verdict,
                              independent_clearance=clean,current_images_match_last_review=hashes_match,
                              reviewed_png_sha256=manifest['images'],current_png_sha256=current,width_mm=180,dpi=600))
    (OUT/'qa_manifest_v62.json').write_text(json.dumps(clearance,indent=2)+'\n')
    pending=[r['figure'] for r in clearance if not r['independent_clearance']]
    print(f'{len(names)} figure/panel exports and 10 legends: exact physical width, 600 dpi and clean figcheck.')
    print('40 C1 trajectories verified; lineage states, formal prediction missingness and generated hashes verified.')
    if pending:
        print('Independent art clearance pending:',pending)
        raise SystemExit(1)
    print('All current images match the latest clean independent critiques.')


if __name__=='__main__':main()

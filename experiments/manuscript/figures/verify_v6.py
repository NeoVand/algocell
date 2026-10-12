"""Verify physical exports and ensure a clean image critique covers current PNGs."""
from pathlib import Path
import hashlib
import json
import subprocess

from PIL import Image

OUT=Path(__file__).resolve().parent/'out/v6'


def main():
    checks=json.loads((OUT/'figcheck.json').read_text())
    assert len(checks)==20 and not any(checks.values()),checks
    status=[]
    for n in range(1,5):
        caption=Image.open(OUT/f'fig{n}_legend.png')
        assert caption.width in (4251,4252)
        assert abs(caption.info['dpi'][0]-600)<.01
        names=[f'fig{n}']+[p.stem for p in sorted(OUT.glob(f'fig{n}[a-d].png'))]
        for name in names:
            im=Image.open(OUT/(name+'.png'))
            assert im.width==4252,(name,im.size)
            assert abs(im.info['dpi'][0]-600)<.01
            for ext in ['pdf','svg']:assert (OUT/(name+'.'+ext)).stat().st_size>1000
            info=subprocess.check_output(['pdfinfo',str(OUT/(name+'.pdf'))],text=True)
            assert '510.236' in info or '510.24' in info,info
        rounds=sorted((OUT/'critique'/f'fig{n}').glob('round[1-3]'))
        final=rounds[-1]
        review=(final/'critique.md').read_text()
        verdict=review.strip().splitlines()[-1].strip('* #')
        cleared=verdict=='NO SUBSTANTIVE DEFECTS'
        manifest=json.loads((final/'manifest.json').read_text())
        assert manifest['returncode']==0 and manifest['image_only']
        current={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in manifest['images']}
        hashes_match=current==manifest['images']
        status.append({'figure':n,'figcheck':'clean','critic_rounds':len(rounds),'last_critic_verdict':verdict,
                       'independent_clearance':cleared and hashes_match,
                       'reviewed_png_sha256':manifest['images'],'current_png_sha256':current,
                       'current_images_match_last_review':hashes_match,'width_mm':180,'dpi':600})
    (OUT/'qa_manifest.json').write_text(json.dumps(status,indent=2)+'\n')
    pending=[s['figure'] for s in status if not s['independent_clearance']]
    print('All 16 figure exports and four captions: exact width, 600 dpi, figcheck clean.')
    if pending:
        print('Independent clearance pending after the three-round cap:',pending)
        raise SystemExit(1)
    print('Current PNG hashes match clean independent critiques.')


if __name__=='__main__':main()

"""Independent Astra art-editor critique; attached PNGs are the ONLY evidence.

Usage: Python critique_v6.py --figure 2 --round 1. No repository access is
requested; the critic runs in an empty temporary directory with no tools.
Preserves exact reviewed images, hashes, prompt, stdout, stderr and verdict.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
OUT=HERE/'out/v6'
CLI='/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex'
PROMPT='''Act as an independent Nature art editor. Your only evidence is the attached rendered PNG images. Do not use any tools, inspect files, consult code, or access a repository. The first image is a 180-mm-wide composed figure; the next images are its panels rendered individually at 180 mm width, 600 dpi; the final image is its accompanying figure caption, also rendered at 180 mm and 6.5 pt type. Judge the composed figure at its stated final print size, not just enlarged panels, and read the rendered caption for outcome definitions and denominators. List substantive defects in overlaps, clipping, glyph rendering, 5-7 pt legibility, visual hierarchy, clutter, colour misuse, misleading encodings, denominators, legend/symbol correspondence and plot limits. Reserved designer panels and explicitly EMPTY/PENDING data slots are intentional; assess their presentation but do not request invented data. A checker's clean report is not evidence available to you. Be demanding but do not invent defects. Give precise panel references and concrete fixes. End with exactly one verdict: SUBSTANTIVE DEFECTS, or NO SUBSTANTIVE DEFECTS. If the latter, say no substantive defect remains. Distinguish optional polish from required fixes.'''
PROMPT_V62='''Act as an independent Nature art editor for a coordinator-directed v6.2 redesign. Your ONLY evidence is the attached PNG images. Do not use any tools, inspect files, consult code or access a repository. The first image is the composed figure at 180 mm width; the next images are enlarged individual panels at 180 mm width; the last image is the separate manuscript legend. All were rendered at 600 dpi. Judge legibility, spacing and hierarchy at the composed figure's final print size. Direction: panels show and legends explain; no panel titles, no explanatory sentences, source notes, counts or status words inside panels; keep axis labels, direct group/line labels and at most one short annotation per panel. Clean empty designer frames are intentional. Teal means transmitter/carrier/benign tar; purple lethal tar only; vermilion erasure or loss; grey structure/historical context. Historical versus C1 uses open/filled markers or faint/solid opacity, never different hues. At most four panels; intentional small multiples in 2a/2b and paired dial axes in 4c are permitted. Figure 3b permits a compact symbol key, including a reserved 8086 symbol with NO 8086 data. Its light grey disc area must represent population multiplicity, with a permitted three-size area key for 1, 100 and 2,000 cells; small display offsets for the other series are permitted. Scores and exact capacity must remain clearly distinct in axis/legend definitions. Figure 3c is two registered offspring position strips. Figure 3d is a 16 by 16 hexadecimal byte map: very light grey for non-working values, white with a thin teal outline for working values outside theorem conditions, and continuous teal fill for working eligible values by predicted free payload from 0 to 28 positions. Black outlines mark 04 and 60. A small continuous colourbar is permitted; no numbers inside cells. Do not assign payload predictions to outside-condition values. Figure 4c uses equally spaced categorical probability settings as defined in the legend. List substantive defects only when supported by the visible images: overlaps, clipping, glyphs, print-size readability, clutter, colour misuse, missing or misleading encodings, symbol correspondence, limits, or numerical/definition conflicts with the legend. Give precise panel references and concrete fixes. A figcheck result is not evidence available to you. Be demanding but do not invent defects. Distinguish optional polish from required fixes. End with exactly one verdict: SUBSTANTIVE DEFECTS, or NO SUBSTANTIVE DEFECTS. If the latter, say no substantive defect remains.'''


def main():
    ap=argparse.ArgumentParser();selection=ap.add_mutually_exclusive_group(required=True)
    selection.add_argument('--figure',type=int)
    selection.add_argument('--extended',choices=['reference','erasure','law','joint_table','competition','confinement'])
    ap.add_argument('--round',type=int,required=True)
    ap.add_argument('--version',choices=['v6','v62'],default='v6');a=ap.parse_args()
    assert 1<=a.round<=3
    prompt=PROMPT_V62 if a.version=='v62' else PROMPT
    name='ed_v6_'+a.extended if a.extended else f'fig{a.figure}'
    if a.extended:
        assert a.version=='v62'
        prompt='''Act as an independent Nature art editor. Your ONLY evidence is the two attached PNGs: an Extended Data figure/table at 180 mm width and 600 dpi, and its separate legend. Do not use tools, inspect files, consult code or access a repository. This is supporting data moved from the main figures during a redesign. The standalone joint numeric table is intentional and explicitly requested in Extended Data; do not require it to become a chart. Other panels use letters only, axis labels and direct labels, without explanatory prose or status words. Teal denotes transmission/carriage/benign tar; purple lethal tar only; vermilion erasure/loss; grey structure/historical context. Historical and C1 use open/filled markers or faint/solid opacity. Check overlaps, clipping, print-size legibility, glyphs, limits, misleading encodings, symbol correspondence, and conflicts with the visible legend. Do not invent defects. Give precise required fixes separately from optional polish. End with exactly one verdict: SUBSTANTIVE DEFECTS, or NO SUBSTANTIVE DEFECTS.'''
    target=OUT/('critique_v62' if a.version=='v62' else 'critique')/name/f'round{a.round}'
    if target.exists():raise RuntimeError('Do not overwrite an archived critique round')
    if a.version=='v62':
        checks=json.loads((OUT/'figcheck_v62.json').read_text())
        relevant={k:v for k,v in checks.items() if k.startswith(name)}
        assert relevant and not any(relevant.values()),'figcheck must be clean before critique'
    target.mkdir(parents=True)
    if a.version=='v62':(target/'figcheck.json').write_text(json.dumps(relevant,indent=2)+'\n')
    files=[OUT/(name+'.png')]+([] if a.extended else sorted(OUT.glob(f'fig{a.figure}[a-d].png')))
    if (OUT/(name+'_legend.png')).exists():files.append(OUT/(name+'_legend.png'))
    hashes={}
    for p in files:
        shutil.copy2(p,target/p.name)
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (target/'prompt.txt').write_text(prompt+'\n')
    with tempfile.TemporaryDirectory(prefix='v6-art-editor-') as cwd:
        command=[CLI,'exec','-m','gpt-6-astra','-c','model_reasoning_effort="high"','--skip-git-repo-check','--ephemeral','-C',cwd,'-o',str(target/'critique.md')]
        for p in files:command+=['-i',str(target/p.name)]
        command+=['--',prompt]
        r=subprocess.run(command,text=True,capture_output=True)
        (target/'stdout.txt').write_text(r.stdout)
        (target/'stderr.txt').write_text(r.stderr)
        (target/'manifest.json').write_text(json.dumps({'model':'gpt-6-astra','reasoning_effort':'high','returncode':r.returncode,'images':hashes,'image_only':True},indent=2)+'\n')
    print(f'{name}, round {a.round}, exit {r.returncode}')
    if (target/'critique.md').exists():print((target/'critique.md').read_text())
    if r.returncode:raise RuntimeError('Art-editor execution failed; inspect archived stderr')


if __name__=='__main__':main()

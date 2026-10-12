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


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--figure',type=int,required=True);ap.add_argument('--round',type=int,required=True);a=ap.parse_args()
    assert 1<=a.round<=3
    target=OUT/'critique'/f'fig{a.figure}'/f'round{a.round}'
    if target.exists():raise RuntimeError('Do not overwrite an archived critique round')
    target.mkdir(parents=True)
    files=[OUT/f'fig{a.figure}.png']+sorted(OUT.glob(f'fig{a.figure}[a-d].png'))
    if (OUT/f'fig{a.figure}_legend.png').exists():files.append(OUT/f'fig{a.figure}_legend.png')
    hashes={}
    for p in files:
        shutil.copy2(p,target/p.name)
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (target/'prompt.txt').write_text(PROMPT+'\n')
    with tempfile.TemporaryDirectory(prefix='v6-art-editor-') as cwd:
        command=[CLI,'exec','-m','gpt-6-astra','-c','model_reasoning_effort="high"','--skip-git-repo-check','--ephemeral','-C',cwd,'-o',str(target/'critique.md')]
        for p in files:command+=['-i',str(target/p.name)]
        command+=['--',PROMPT]
        r=subprocess.run(command,text=True,capture_output=True)
        (target/'stdout.txt').write_text(r.stdout)
        (target/'stderr.txt').write_text(r.stderr)
        (target/'manifest.json').write_text(json.dumps({'model':'gpt-6-astra','reasoning_effort':'high','returncode':r.returncode,'images':hashes,'image_only':True},indent=2)+'\n')
    print(f'Fig {a.figure}, round {a.round}, exit {r.returncode}')
    if (target/'critique.md').exists():print((target/'critique.md').read_text())
    if r.returncode:raise RuntimeError('Art-editor execution failed; inspect archived stderr')


if __name__=='__main__':main()

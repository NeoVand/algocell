"""Image-only Astra comparison with the previous, accepted figure standard."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
OUT=HERE/'out/v8'
CLI='/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex'
REFS={n:['fig1','fig4v5'] for n in range(1,6)}
PROMPT='''Act as an independent scientific art editor. Your ONLY evidence is the attached PNG images. Do not use tools, inspect files, read code, consult a repository, or infer hidden data. All images except the LAST are the previous reference figures that establish the required standard. The LAST is the new v8 figure, printed at 180 mm width, exported at 600 dpi. Does it meet or exceed the references in polish, information, and self-explanation? Rich direct labels, category headers, inline keys and number columns are required. Every panel must contain substantive content. Paragraph-like method notes were deliberately moved to separate manuscript legends; judge whether direct labels, units, encodings, cohorts and the main results remain understandable without requiring the full methods inside the image. Scientific content and panel counts differ intentionally. The evolved-8086-law statement is required only for Figure 3b, where it must say untested, with no eligible certified copier. Do not demand that statement in Figures 1, 2, 4 or 5, which have different subjects. No empty panel is permitted. The references use 5-7 pt body/ticks and 8 pt panel letters. Judge actual print readability relative to the references; do not impose a different universal minimum. List EVERY supported defect: apparent missing content, gaps, layout, typography, overlap, clipping, glyphs, misleading counts or areas, ambiguous symbols, missing denominators, unexplained units, or essential labels requiring a caption to understand the main result. Separate optional polish from substantive defects, and propose concrete panel-specific fixes. Be demanding, candid and specific. Do not infer unshown results or demand complete methods in the figure. If no substantive defect remains, explicitly state that the new figure meets or exceeds the references. Finish with exactly one line: MEETS OR EXCEEDS REFERENCES - NO SUBSTANTIVE DEFECTS, or BELOW STANDARD - SUBSTANTIVE DEFECTS.'''



def main():
    ap=argparse.ArgumentParser();ap.add_argument('--figure',type=int,required=True,choices=range(1,6));ap.add_argument('--round',type=int,required=True,choices=range(1,4));a=ap.parse_args()
    name=f'fig{a.figure}';target=OUT/'critique'/name/f'round{a.round}'
    prompt=PROMPT+f" The new image is Figure {a.figure}. Apply requirements only to its scientific scope."
    if a.round==3:
        prompt+=''' Calibration for this final comparison: the required standard is the supplied previous figures, whose publication typography uses 5-7 pt body/ticks and 8 pt panel letters. Do not substitute a different preferred minimum such as 8 pt for all body copy. Flag actual unreadability relative to those references. This is a scientific figure, with full experimental protocols and preregistration details supplied in separate manuscript legends; judge whether the mechanism, measurements, outcomes, units, encodings and relevant comparisons are understandable from the image, without requiring it to reproduce the complete Methods. Do not excuse real defects, and do not infer missing results. The final verdict must still explicitly assess polish, information, and self-explanation against the attached references.'''
    assert not target.exists(), 'Never overwrite an archived round'
    checks=json.loads((OUT/'figcheck.json').read_text())
    assert checks[name]==[], 'figcheck must be clean before art review'
    target.mkdir(parents=True)
    files=[HERE/'out'/(ref+'.png') for ref in REFS[a.figure]]+[OUT/(name+'.png')]
    images={}
    for i,p in enumerate(files):
        dest=target/(f'reference{i+1}_'+p.name if i<len(files)-1 else p.name)
        shutil.copy2(p,dest);images[dest.name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    (target/'prompt.txt').write_text(prompt+'\n')
    (target/'figcheck.json').write_text(json.dumps({name:checks[name]},indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='v8-image-critic-') as cwd:
        cmd=[CLI,'exec','-m','gpt-6-astra','-c','model_reasoning_effort="high"',
             '--skip-git-repo-check','--ephemeral','-C',cwd,'-o',str(target/'critique.md')]
        for file in images:cmd+=['-i',str(target/file)]
        cmd+=['--',prompt]
        r=subprocess.run(cmd,text=True,capture_output=True)
    (target/'stdout.txt').write_text(r.stdout)
    (target/'stderr.txt').write_text(r.stderr)
    (target/'manifest.json').write_text(json.dumps(dict(model='gpt-6-astra',reasoning_effort='high',returncode=r.returncode,image_only=True,images=images),indent=2)+'\n')
    print(f'{name}, round {a.round}, return {r.returncode}')
    if (target/'critique.md').exists():print((target/'critique.md').read_text())
    if r.returncode:raise SystemExit(r.returncode)


if __name__=='__main__':main()

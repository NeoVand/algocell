"""Export the canonical, reviewed v8 captions from MAIN_v8.md.

Scientific caption text lives in the manuscript. Rebuilding the figure package
must not restore stale labels or the superseded long captions.
"""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out' / 'v8'
MANUSCRIPT = HERE.parent / 'MAIN_v8.md'


def main():
    text = MANUSCRIPT.read_text(encoding='utf-8')
    section = text.split('## Figure legends\n', 1)[1].split('## Methods\n', 1)[0].strip()
    captions = re.findall(
        r'(?ms)^\*\*Figure (\d)\. (.*?)(?=^\*\*Figure \d\.|\Z)', section
    )
    if [number for number, _ in captions] != ['1', '2', '3', '4', '5']:
        raise ValueError('Expected exactly Figures 1–5 in manuscript order')
    for number, caption in captions:
        words = len(f'Figure {number}. {caption}'.split())
        if words > 250:
            raise ValueError(f'Figure {number}: {words} words exceeds 250')
    output = '# Legends for Figures v8\n\n' + section + '\n'
    OUT.mkdir(parents=True, exist_ok=True)
    for destination in (HERE.parent / 'LEGENDS_V8.md', OUT / 'LEGENDS_V8.md'):
        destination.write_text(output, encoding='utf-8')
    print('Exported five canonical MAIN_v8 captions; each is at most 250 words.')


if __name__ == '__main__':
    main()

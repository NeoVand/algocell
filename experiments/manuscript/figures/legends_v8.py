"""Export final label-matched legends without editing MAIN_v8.md.

Exact sentence replacements are exported to LEGEND_UPDATES.md for integration.
"""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out' / 'v8'
MANUSCRIPT = HERE.parent / 'MAIN_v8.md'
CHANGES = {
    "1": [
        [
            "Ochre marks expected hardware divergences; vermilion marks flagged or changed comparisons; grey marks agreement under the row-specific criterion.",
            "Ochre marks expected hardware divergences; vermilion marks flagged or different comparisons; grey marks comparisons that meet the row-specific criterion."
        ],
        [
            "Core C1 denotes corrected Z80 semantics; rows must not be pooled as one accuracy estimate.",
            "Corrected core denotes core C1; rows must not be pooled as one accuracy estimate."
        ],
        [
            "The pusher uses the same bytes as instructions and literal output: `01 c5 01` loads BC, and `c5` pushes it into partner memory.",
            "The pusher reads and writes the same bytes: `01 c5 01` loads BC; `c5` pushes BC into partner memory."
        ],
        [
            "Cumulative detection is one minus Kaplan–Meier survival for historical Stage E worlds at six tape lengths L (bytes), with 10 worlds per length and nondetections censored at 300,000 updates.",
            "Historical Stage E detection is one minus Kaplan–Meier survival: six lengths L (bytes), 10 worlds/length, censoring at 300,000 updates."
        ],
        [
            "The 10,083 hardware divergences are expected; none is unexpected.",
            "All 10,083 hardware divergences are expected."
        ],
        [
            "Execution can enter the partner; a separate return-containing design redirects execution into its own tape.",
            "Execution can enter the partner; separate RET NZ copying returns to A’s start if the zero flag is clear."
        ],
        [
            "Ellipses abbreviate tapes; `e3` exchanges HL with stack memory.",
            "Selected bytes are shown; ellipses omit positions; `e3` exchanges HL with stack memory."
        ],
        [
            "The ring criterion is a 95% upper bound on changed-memory rate below 0.01%.",
            "Ring stability requires a 95% upper bound <0.01% changed-memory pairs; small nonzero observed changes are permitted."
        ]
    ],
    "2": [
        [
            "Open circles score at transfer one but not eight; filled circles mark transfer-eight sites.",
            "Open circles score at transfer one but fall below the site criterion at transfer eight; filled circles score at transfer eight."
        ],
        [
            "The L50 pusher uses 32 alternatives/site; others use 255.",
            "The lethal-chemistry L16 tape is a confined transmitter with nine scored sites at transfer eight. The L50 pusher uses 32 alternatives/site; others use 255."
        ],
        [
            "Each member meets two fresh partners; only ≥ 75% best-cyclic-shift matches to the immediate parent continue, uniformly capped at 32 members.",
            "Each member meets two fresh partners; ≥75% best-cyclic-shift immediate-parent matches continue, uniformly capped at 32 members."
        ],
        [
            "Sites score when carriage among surviving mutant populations exceeds surviving-control carriage by ≥ 0.5 for at least half their identifiable alternatives.",
            "Sites score when surviving-mutant carrier fraction minus surviving-unmutated carrier fraction is ≥0.5 for ≥half identifiable alternatives."
        ],
        [
            "Grey marks instruction-stream fetch, including operands, across 64 reference encounters; white marks no fetch.",
            "Grey/white mark instruction-stream fetch/no fetch across 64 reference encounters, including operands."
        ],
        [
            "The L20 transmitter uses seed 6004, versus 6003 in a,b.",
            "L20 uses seed 6004 (6003 in a,b)."
        ],
        [
            "At zero-based execution step 6, address 4 supplies ancestry from original byte 0, overwriting marker 0xA7 at address 8 and erasing its last ancestry label.",
            "At zero-based execution step 6, address 4 supplies original-byte-0 ancestry, overwriting 0xA7 at address 8 and erasing its last ancestry label."
        ],
        [
            "Generation-eight bars partition mutant starts: lost, alive with < 50% marker carriage or alive with ≥ 50% carriage.",
            "Transfer-eight bars partition starts: lost, alive with <50% or ≥50% of members carrying the marker window."
        ]
    ],
    "3": [
        [
            "For fixed core c and protected prefix d, free payload r = d−c.",
            "For fixed core length c and displacement d, predicted free payload r = d − c."
        ],
        [
            "An exploratory 256-value scan finds 157 nonworking, 88 working outside sampled eligibility and 11 eligible, shaded by predicted payload.",
            "An exploratory 256-value scan finds 157 nonworking, 88 working outside the sampled screen and 11 that pass the sampled screen, shaded by predicted payload."
        ],
        [
            "Teal rings show exact capacities for 101 registered settings, including 22 original-value controls, across 22 parents.",
            "Teal rings: exact capacities for 101 registered settings (22 original-value controls included; 22 parents)."
        ],
        [
            "Ochre triangles show exact capacities for 42 eligible 6502 roles: 41 distinct tapes from 11 worlds.",
            "Ochre triangles: exact capacities for 42 eligible 6502 roles (41 tapes, 11 worlds)."
        ],
        [
            "Squares show 296 constructed 6502 sampled-carriage cases, not exact capacities.",
            "Squares: 296 constructed 6502 sampled-carriage cases, not exact capacities."
        ],
        [
            "The red point retains four positions in targeted transfers but scores zero under the population ≥ 75% whole-tape gate.",
            "The red point carries four positions in targeted transfers but scores zero under the ≥75% whole-tape gate."
        ]
    ],
    "4": [
        [
            "Styles/markers identify starting mixtures; light/open denotes historical implementations and dark/filled corrected core C1.",
            "Styles/markers identify starts; light/open = Historical, dark/filled = Corrected (Z80 core C1)."
        ],
        [
            "Zero opcodes execute normally under benign chemistry and halt under lethal chemistry.",
            "Under benign chemistry, a zero opcode executes as no operation; under lethal chemistry, fetching a zero opcode halts."
        ],
        [
            "Endpoint columns give core C1 then historical medians.",
            "Endpoint columns give Corrected then Historical medians."
        ],
        [
            "Transmitter majority requires estimated share > 50% of heritable cells in the confined ≥ 5-site class.",
            "Transmitter majority means >50% of heritable cells are confined with ≥5 byte sites retaining introduced values under the carriage-excess criterion."
        ]
    ],
    "5": [
        [
            "Frozen mechanism labels hold in 16/16 encounters: eight one-pass and two copy-loop classifications.",
            "Frozen full-address classes hold in 16/16 encounters: eight One-pass class and two Loop class classifications. Classes use full instruction-start addresses; all ten post hoc tapes repeatedly copy mapped ring positions at displacement d = 4 bytes."
        ],
        [
            "Historical Z80 Stage G has 80 worlds, 20 each at L = 16,20,50,64, all open first.",
            "Historical Z80 Stage G: 80 worlds (20 each at L = 16,20,50,64), all open first."
        ],
        [
            "Primary 6502 S1 A has 20 L16 worlds, all confined first.",
            "Primary 6502 S1 A: 20 L16 worlds, all confined first."
        ],
        [
            "Coherent-fetch 8086 S2 has 20 L16 worlds: two open, 12 confined and six censored at 3,000,000 updates.",
            "Coherent-fetch 8086 S2: 20 L16 worlds—two open, 12 confined, six censored at 3,000,000 updates."
        ],
        [
            "For S1/S2, open means partner instruction-stream fetch in ≥ 8/16 traces; confined means none.",
            "S1/S2: open = partner instruction-stream fetch in ≥8/16 traces; confined = none."
        ],
        [
            "Kaplan–Meier curves show cumulative detection for 20 L16 worlds/machine at budget 128.",
            "Kaplan–Meier detection: 20 L16 worlds/machine; 128-instruction budget."
        ],
        [
            "One terminal mark represents six coincident 8086 censors.",
            "One terminal mark: six coincident 8086 censors."
        ],
        [
            "All ten `5F A4 x y` first-copier tapes from 14 detected 8086 worlds appear in physical phase with detection times.",
            "All ten `5F A4 x y` first-copier tapes (14 detected worlds) appear in physical phase with detection times."
        ],
        [
            "Later overlap analysis is exploratory and does not change registered verdicts.",
            "Exploratory overlap analysis leaves registered verdicts unchanged."
        ]
    ]
}


def final_legends():
    source=MANUSCRIPT.read_text(encoding='utf-8')
    section=source.split('## Figure legends\n',1)[1].split('## Methods\n',1)[0].strip()
    for replacements in CHANGES.values():
        for old,new in replacements:
            assert section.count(old)==1,old
            section=section.replace(old,new)
    return '# Legends for Figures v8\n\n'+section+'\n'


def main():
    output=final_legends()
    captions=re.findall(r'(?ms)^\*\*Figure (\d)\. (.*?)(?=^\*\*Figure \d\.|\Z)',output)
    assert [number for number,_ in captions]==['1','2','3','4','5']
    counts={n:len(f'Figure {n}. {caption}'.split()) for n,caption in captions}
    assert all(n<=250 for n in counts.values()),counts
    for destination in (HERE.parent/'LEGENDS_V8.md',OUT/'LEGENDS_V8.md'):
        destination.write_text(output,encoding='utf-8')
    lines=['# Legend updates for the final main figures','',
           'Every required final sentence replacement, exact old → exact new. MAIN_v8.md remains unedited. The delivered LEGENDS_V8.md applies these changes; compact replacements keep every final legend within 250 words.','']
    for n,replacements in CHANGES.items():
        lines += [f'## Figure {n}','']
        for old,new in replacements:
            lines += ['**Old sentence:**','',old,'','**New sentence:**','',new,'']
    lines += ['## Integration check','',
              '']
    lines[-1]='All old sentences match MAIN_v8.md exactly once. Applying all replacements in memory reproduces the delivered legends. Final Figure 1–5 word counts: '+', '.join(str(counts[str(n)]) for n in range(1,6))+'. MAIN_v8.md was not edited.'
    (OUT/'LEGEND_UPDATES.md').write_text('\n'.join(lines)+'\n')
    print('Final main legend word counts:',counts)


if __name__=='__main__':main()

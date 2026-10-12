"""Write the coordinator-facing figure inventory from current exports and QA."""
from pathlib import Path
import json
import pandas as pd

from captions_v6 import captions

HERE=Path(__file__).resolve().parent
OUT=HERE/'out/v6'
D=OUT/'data'


def main():
    checks=json.loads((OUT/'figcheck.json').read_text())
    rows=[]
    for n in range(1,5):
        rounds=sorted((OUT/'critique'/f'fig{n}').glob('round[1-3]'))
        verdicts=[]
        for r in rounds:
            path=r/'critique.md'
            verdict=path.read_text().strip().splitlines()[-1] if path.exists() else 'PENDING'
            verdicts.append(f"[round {r.name[-1]}](figures/out/v6/critique/fig{n}/{r.name}/critique.md): {verdict}")
        status='clean' if not any(v for k,v in checks.items() if k.startswith(f'fig{n}')) else 'ISSUES'
        rows.append(f"| {n} | {status} | {'; '.join(verdicts)} |")
    s=pd.read_csv(D/'sources.csv')
    bursts=pd.read_csv(D/'bursts.csv')
    text='''# Figures v6: data build and coordinator handoff

Branch: `feat/figs-v6`, based on `feat/experiments-runner` at `b4e834e`.
Worktree: `/Users/neo/repos/algocell-wt-figs`. Nothing is pushed.

All requested data panels are exported individually and in composed figures at
`figures/out/v6/`: PDF, editable-text SVG, and PNG. Pages are exactly 180 mm wide;
PNGs are rendered from the PDFs by Poppler at 600 dpi (4,252 pixels wide).
Type is Arial, 5.5–8 pt (body predominantly 6–6.5 pt); figures contain at most
four labelled panels. Figure 4 is 170 mm deep. Teal/vermilion and structural
greys serve Figures 1–3; teal/house purple serve Figure 4. No conceptual drawing
was made. Layout decisions precede code in `figures/LAYOUT_V6.md`.

**Completion limits.** Figure 4a has the complete historical trajectories but
only C1 endpoint medians: the 40 corrected-core C1 E trajectory JSONLs are not
in the local archives. The slots state that trajectories are unavailable;
no line is interpolated between endpoints. Figure 3d has exact joint challenges
and a separate constructed eligibility screen, but the existing tables do not
contain joint controls for altered phase or inadequate forward coverage.
Those two controls remain outstanding and are disclosed in the caption.
Designer panels 1a, 1b, 3a await `DESIGNER_ROUND6.md`; evolved 8086 points and
Figure 5 wait for S2 (the prospective second-stage 8086 study). Figure 4d was
not requested. These limits keep the current files reviewable without inventing
results and must be resolved before submission.

**Independent art clearance.** Figures 1 and 2 passed the third image-only
critique. Figure 3's third critique asked for separate score/capacity labels
and a defined constructed ordinate; Figure 4's asked for operational event
definitions and registration thresholds. These were corrected after round 3.
No fourth critique was run, in accordance with the user's three-round cap.
The latest Figures 3 and 4 are technically clean but are **not claimed to have
independent final art clearance**. `out/v6/post_critique_fixes.json` records the
corrections and image hashes; the coordinator must resolve this sign-off.

The label **C1 in Figure 3** denotes the registered displacement-capacity
intervention. **C1 in Figures 1, 2 and 4** denotes the corrected-core rerun
cohort. Their source tables and units are kept separate.

## Panel inventory

| Panels | Content | Generated sources (relative to experiments/) | Status |
|---|---|---|---|
| 1a, 1b | Abstract encounter and mechanism briefs | `manuscript/DESIGNER_ROUND6.md` | Designer reserved |
| 1c | Illustrative world; separate branching, entry and inflow reference assays | archived `runs/video/video_L16_st128_k4_s2002.jsonl`; `results/branching/branching_summary.csv`; `results/exectrace/per_replicator.csv` | Built; historical, not a median history |
| 1d | Distinct instruction/encounter/assay/ecological validation units | `results/impact/outcome.csv`, `validation.csv`; `results/tier2/T2_COMPONENTS.csv`, `T2_CLAIMS.csv`; absolute generated Zilion 8086 architectural JSON baseline | Built; 8086 ecology blank |
| 2a | Controls alive, mutant populations alive and carriage; C1 coincides with historical; X1-2 prediction failed | historical and `results/tier2/scored/Branching/branching_summary.csv` | Built; registered |
| 2b | Every alternative at every position, loss/eraser/carrier, fetched ticks and rare-position counts | `results/serial_retention/sr_mutants.csv.gz`, `sr_sites.csv` | Built; historical serial, any-carrier map distinct from scored sites |
| 2c | Complete ordered erasure archive and descendant reproduction | frozen existing `capacity_common.Z80Provenance`; `figures/out/v6/data/erasure_ordered.csv`, `erasure_states.csv`, `erasure_manifest.json` | Built; exploratory |
| 2d | Full six-parent × six-environment result | `results/pe/pe_matrix.csv`; registered serial protocol | Built; lineage counts and site counts distinct |
| 3a | Abstract copy geometry brief | `manuscript/DESIGNER_ROUND6.md` | Designer reserved |
| 3b | Sampled Z80 non-core payload; Z80 world means; exact evolved 6502 and constructed controls | `results/offset/offset_cells.csv`; `results/crystal/cells.csv`, `cell_exceptions.csv`; `results/evolved6502/eligibility.csv`, `exact_cases.csv.gz`; `results/census6502/table_closers_law.csv` | Built; Z80 exploratory; evolved 8086 empty |
| 3c | C1 inherited edits; measured g8 descendant capacity 0–60; reciprocal edit; exact full descendant sequence example in caption | `results/capacity/edit_eligibility.csv`, `serial.csv.gz`, `capacity_cases.csv.gz`; exploratory `results/crystal/alleles_ldir32.csv`, `allele_encounters_ldir32.csv` | Built; 101 edits, 22 archived parents; exploratory dial separated |
| 3d | Exact all-value, joint and erased challenges; constructed eligibility count | `results/capacity/capacity_cases.csv.gz`; `results/evolved6502/exact_cases.csv.gz`; `results/census6502/table_closers_law.csv` | Built data; phase/coverage joint controls outstanding; 8086 empty |
| 4a | Historical L32 trajectories; side-by-side C1 endpoint strips at 300k | historical `runs/offset/*`; historical and `results/tier2/scored/E/offset_switch.csv` | Partial: needs C1 raw trajectories |
| 4b | Every world's ratio, absolute core abundance and nested tail diversity; undefined ratio disclosed | historical and C1 `offset_switch.csv` | Built; 80 worlds, one core-loss world retained |
| 4c | Historical/C1 open-first, transmitter-majority, KM confinement; original failures and remaining reversal | historical and `results/tier2/scored/DZ/dial_worlds.csv`; `results/tier2/T2_DZ.csv`, `T2_COMPONENTS.csv` | Built; world unit, censoring explicit |

## Build and audit

Use the specified Python from this worktree:

```sh
FIGS_PY=/Users/neo/repos/algocell/experiments/.venv/bin/python
"$FIGS_PY" experiments/manuscript/figures/archive_erasure_v6.py
"$FIGS_PY" experiments/manuscript/figures/prepare_v6.py
"$FIGS_PY" experiments/manuscript/figures/build_v6.py
"$FIGS_PY" experiments/manuscript/figures/verify_v6.py
"$FIGS_PY" experiments/manuscript/figures/write_v6_manifest.py
```

`prepare_v6.py --c1-runs
/path/to/local/C1/logs` will reconcile every raw endpoint before adding its
trajectory. The default historical archive is read-only outside the worktree.
Plotting CSVs and source hashes are committed under `figures/out/v6/data/`;
every plotted point and displayed count is derived by the committed scripts.
The full generated result tables remain their existing committed sources.
`data/sources.csv` hashes every source table/raw log and the absolute 8086
baseline. Coincident scatter coordinates are merged; counts remain in tables.
World means are grouped by world and prediction; cells are not treated as
independent worlds. No error bar or interval is implied where none is drawn.

`archive_erasure_v6.py` records all fetch/read/write events, checks final memory
and registers against the uninstrumented executor, and tracks the original
marker ancestry through the actual descendant. Local GPU only; no Modal was
invoked. Burst accounting includes compilation and readback, and enforces a
58-second alarm. The archived run's longest burst and total activity are below.

Every build invokes the existing `figcheck.py` for all panels, compositions and
rendered captions. Additional checks catch text leaving its reserved slot and
scatter points outside plot limits. `figcheck.json` contains the current reports.
The independent critic is `codex exec -m gpt-6-astra` with high reasoning in an
empty temporary directory. It receives only PNGs and is instructed not to use
tools. Each round archives its actual inputs, image hashes, prompt, review and
CLI logs in `out/v6/critique/figN/roundN/`. Rounds 1 and 2 were followed by fixes;
round 3 includes the captions as rendered PNGs. Required label/caption fixes
for Figures 3 and 4 were then applied within the three-round limit, without
claiming a further independent review. `invocation_failed/` archives
an initial CLI argument-parsing failure, not an art-editor round. There were
exactly three actual critiques per figure, no fourth. `qa_manifest.json` records
the physical-export checks, current/reviewed PNG hashes and independent-clearance
status separately. `verify_v6.py` exits 1 while art clearance remains pending,
even when all technical checks pass.

| Figure | figcheck (panels, composition, caption) | Image-only critic rounds |
|---|---|---|
'''
    text+='\n'.join(rows)+'\n'
    text+=f"\nHashed source inputs: {len(s)}. Longest local GPU burst: {bursts.active_s.max():.6f} s; total active: {bursts.active_s.sum():.6f} s; pauses recorded beside each burst.\n"
    text+='\n## Caption and legend notes\n\nThese notes update the older pending wording in `MAIN_v6.md` without editing the coordinator\'s manuscript. Each is also supplied as a 180 mm, 600 dpi, 6.5 pt caption PNG for the art-editor review.\n'
    for n,caption in captions().items():text+='\n### Figure '+str(n)+'\n\n'+caption+'\n'
    (HERE.parent/'FIGURES_V6.md').write_text(text)
    print('Wrote experiments/manuscript/FIGURES_V6.md')


if __name__=='__main__':main()

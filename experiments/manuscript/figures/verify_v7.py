"""Verify exported figures and frozen data; publish an honest clearance ledger."""
from pathlib import Path
import hashlib
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
import pymupdf as fitz

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
OUT = HERE / 'out/v7'
D = OUT / 'data'
HEIGHTS = {1: 167, 2: 150, 3: 166, 4: 170}
PASS = 'MEETS OR EXCEEDS REFERENCES - NO SUBSTANTIVE DEFECTS'
FAIL = 'BELOW STANDARD - SUBSTANTIVE DEFECTS'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def table(name):
    return pd.read_csv(D / (name + '.csv'), low_memory=False)


def main():
    # Check both the copied input ledger and every frozen output table.
    for row in table('sources_v7').itertuples():
        assert sha(EXP / row.path) == row.sha256, row.path
    for row in table('hashes').itertuples():
        assert sha(D / row.path) == row.sha256, row.path
    np.testing.assert_allclose(table('methods').set_index('name').loc[
        ['ring_rate_bound','ring_confidence','gen2_minimum','match_gate'],'value'],
        [.0001,.95,.3,.75])
    assert (OUT/'LEGENDS_V7.md').read_bytes()==(HERE.parent/'LEGENDS_V7.md').read_bytes()
    assert table('missing_trajectories').empty
    audit = json.loads((D / 'c1_download_audit.json').read_text())['worlds']
    assert len(audit) == 40
    assert all(all(r[k] for k in ['available', 'endpoint_verified',
                                 'machine_hashes_verified', 'recorded_marker_verified'])
               for r in audit)
    worlds = table('emergence_worlds')
    assert worlds.groupby('tape_len').size().eq(10).all()
    assert set(worlds.horizon) == {300000}
    branching = table('branching_endpoints')
    columns = ['lost', 'erased', 'retained', 'ctrl_alive_n', 'sites', 'L']
    hist, c1 = (branching.query('cohort==@cohort').set_index('key').sort_index()
                for cohort in ['Historical', 'C1'])
    np.testing.assert_allclose(hist[columns], c1[columns])
    np.testing.assert_allclose(c1[['lost', 'erased', 'retained']].sum(axis=1), 1)
    for name in ['law_z80_c1', 'law_6502_c2']:
        law = table(name)
        np.testing.assert_allclose(law.predicted_free, law.measured_free)
    population = table('law_population')
    assert population.cells.sum() == 2343
    assert len(population[population.predicted != population.n_unexec]) == 1
    # Neither scientific ordinates nor exact capacities receive display jitter.
    strips = table('lineage_strips')
    assert len(strips) == 64 and set(strips.generation) == {8}
    for edit, state in [('ldir32_v04', 'erased'), ('ldir32_v60', 'transmitted')]:
        q = strips[strips.edit_id == edit]
        assert q.position.nunique() == 32
        assert (q.state == 'core').sum() == 4 and (q.state == state).sum() == 28
    dial = table('dial_coordinates')
    assert sorted(dial.value) == list(range(256))
    assert dial.culture_replicator.sum() == 99 and dial.prop8_eligible.sum() == 11
    assert dial.loc[~dial.prop8_eligible, 'predicted_payload'].isna().all()
    eligible = dial[dial.prop8_eligible]
    np.testing.assert_allclose(eligible.predicted_payload, eligible.effective_d - 4)
    ends = table('competition_endpoints')
    assert ends.groupby(['cohort', 'tar']).size().eq(20).all() and len(ends) == 80
    assert ends.T_share_end.isna().sum() == 1
    points = table('endpoint_positions')
    for measure, group in points.groupby('measure'):
        source = ends[ends[measure].notna()]
        merged = group.merge(source, on=['cohort', 'tar', 'start', 'seed'],
                             suffixes=('_plot', '_source'), validate='one_to_one')
        assert len(merged) == len(source)
        np.testing.assert_allclose(merged[measure + '_plot'], merged[measure + '_source'])
    lethal = table('lethality_summary')
    assert len(lethal) == 12 and lethal.worlds.eq(10).all()

    checks = json.loads((OUT / 'figcheck.json').read_text())
    assert set(checks) == {'fig1', 'fig2', 'fig3', 'fig4'}
    assert not any(checks.values()), checks
    records = {}
    for n, height in HEIGHTS.items():
        name = f'fig{n}'
        pdf = fitz.open(OUT / (name + '.pdf'))
        assert len(pdf) == 1
        page = pdf[0]
        np.testing.assert_allclose([page.rect.width, page.rect.height],
                                  np.array([180, height]) * 72 / 25.4, atol=.02)
        spans = [s for b in page.get_text('dict')['blocks'] if b['type'] == 0
                 for line in b['lines'] for s in line['spans']]
        assert spans, 'PDF must preserve vector text'
        for span in spans:
            x0, y0, x1, y1 = span['bbox']
            assert x0 >= -.1 and y0 >= -.1 and x1 <= page.rect.width + .1 and y1 <= page.rect.height + .1, span
        pdf.close()
        svg = ET.parse(OUT / (name + '.svg')).getroot()
        assert len(list(svg.iter('{http://www.w3.org/2000/svg}text'))) > 10
        png = Image.open(OUT / (name + '.png'))
        expected = [math.ceil(mm / 25.4 * 600) for mm in [180, height]]
        assert list(png.size) == expected, (png.size, expected)
        assert max(abs(v - 600) for v in png.info['dpi']) < .01
        assert png.mode == 'RGB'
        rounds = sorted((OUT / 'critique' / name).glob('round*'))
        assert len(rounds) == 3, 'Respect the three-round cap and preserve all reviews'
        last = rounds[-1]
        review = (last / 'critique.md').read_text()
        verdict = review.strip().splitlines()[-1].strip('* ')
        assert verdict in [PASS, FAIL], verdict
        manifest = json.loads((last / 'manifest.json').read_text())
        assert manifest['returncode'] == 0 and manifest['image_only']
        assert manifest['model'] == 'gpt-6-astra' and manifest['reasoning_effort'] == 'high'
        review_hash=manifest['images'][name + '.png']
        delivered_hash=sha(OUT / (name + '.png'))
        reviewed_final=review_hash==delivered_hash
        for folder in rounds:
            m = json.loads((folder / 'manifest.json').read_text())
            assert m['returncode'] == 0
            for filename, digest in m['images'].items():
                assert sha(folder / filename) == digest
        records[name] = dict(
            page_mm=[180, height], png_pixels=list(png.size),
            dpi=list(png.info['dpi']), pdf_text_spans=len(spans),
            figcheck='clean', critique_rounds=len(rounds),
            final_verdict=verdict, cleared=verdict == PASS and reviewed_final,
            delivered_image_reviewed=reviewed_final,
            last_review_png_sha256=review_hash,
            post_review_revision='Measurement definitions corrected after round 3; no fourth review permitted',
            files={ext: sha(OUT / (name + '.' + ext)) for ext in ['pdf', 'svg', 'png']},
            final_review=str(last.relative_to(OUT) / 'critique.md'))
    manifest = dict(technical_checks='passed', figures=records,
                    data_sha256=sha(D / 'hashes.csv'),
                    legends_sha256=sha(OUT / 'LEGENDS_V7.md'),
                    scope='Figures 1-4; Figure 5 deferred for validated 8086 ecology',
                    source_scripts={p.name: sha(p) for p in
                                    [HERE / (s + '.py') for s in
                                     ['prepare_v7', 'build_v7', 'legends_v7', 'critique_v7', 'verify_v7']]})
    (OUT / 'qa_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    publish_report(records)
    print('All export, source-hash, data, figcheck and archived-review hash checks passed.')
    print('Astra cleared: ' + ', '.join(k for k, v in records.items() if v['cleared']))


def publish_report(records):
    prefix = 'figures/out/v7/'
    lines = ['# Figures v7 — rebuilt from the previous figure standard', '',
             'Generated by ¦figures/verify_v7.py¦. Technical verification passed; '
             'Astra clearance is a separate requirement, recorded below. All four remain '
             '**NOT CLEARED**. The final delivered images correct the third-round definitions, '
             'but have no further Astra review because the three-round cap was reached.', '',
             '| Figure | Page (mm) | figcheck | Astra rounds | Final clearance |',
             '|---|---|---|---|---|']
    for name, row in records.items():
        state = 'CLEARED' if row['cleared'] else 'NOT CLEARED'
        lines.append(f"| [{name}]({prefix}{name}.png) | {' × '.join(map(str,row['page_mm']))} "
                     f"| clean | 3 | [{state}]({prefix}{row['final_review']}) |")
    lines += ['', 'Each figure is supplied as a vector PDF, an SVG with editable chart text, '
              'and a PDF-rendered 600 dpi PNG. Figures 1–4 contain real content in every panel. '
              'No Figure 5 or empty 8086 panel is emitted.',
              '', 'PDFs: ' + ' · '.join(f'[{name}]({prefix}{name}.pdf)' for name in records) +
              '. SVGs: ' + ' · '.join(f'[{name}]({prefix}{name}.svg)' for name in records) + '.',
              '', 'The attached previous PNGs, reviewed PNG, prompt, image hashes, model output '
              'and command logs are preserved for all three rounds under ¦figures/out/v7/critique/¦. '
              'The final reviewed PNGs are preserved unchanged. The delivered PNGs differ because '
              'the remaining definitions were corrected after round three; both hashes are recorded '
              'in the technical manifest. No final revision is claimed to be cleared. Round three explicitly '
              'calibrated typography to the supplied references’ 5–7 pt body/ticks and 8 pt panel '
              'letters, while continuing to require legibility, self-explanation and no substantive defects.',
              '', '**What was rebuilt.** Figure 1 reuses the approved designer 3D panel from '
              'the previous Figure 1 vector PDF, beside the original Kaplan–Meier calculation, '
              'a recorded representative world, and labelled validation counts. Figure 2 adapts '
              '¦fig4v5¦’s outcome bars, serial curves, and per-byte strips, with WT survival and '
              'site-count columns and a complete ordered erasure/ancestry trace. Figure 3 adds '
              'the overlap mechanism, exact-coordinate law scatter, inherited 04→60 intervention, '
              'and all 256 byte values. Figure 4 shows complete C1/archive trajectories, endpoint '
              'distributions, numeric endpoint medians and the full lethality dial.',
              '', 'The layouts start from ¦make_figures.py¦, ¦concept.py¦ and ¦figstyle.py¦: '
              'the previous palette, category headers, descriptive rows, inline keys, numeric '
              'columns and vector schematic primitives are reused. The locally archived three.js '
              'directory contains completed renders but no runnable scene source. Figure 3a '
              'therefore uses the permitted careful 2D vector fallback; each unit represents '
              'two bytes and its provenance is generated in ¦geometry_units.csv¦.',
              '', '**Data and limits.** V6.2 preparation is reused, with the historical KM and '
              'serial-retention tables added by ¦prepare_v7.py¦. All plotted numerical labels '
              'read generated tables. Input and output SHA-256 ledgers are committed. All 40 C1 '
              'competition logs were already downloaded and audited by the v6.2 scripts; the '
              'copied audit verifies endpoints, machine hashes and completion markers. All 40 '
              'historical competition worlds are also present. No simulation, remote compute '
              'or GPU run was performed for v7.',
              '', 'Historical/C1 branching endpoints agree in every displayed value and control '
              'count. Serial strips remain historical and are identified as such. One C1 benign '
              'core-loss world has an undefined T/(R+T): its ratio point is omitted, its zero '
              'absolute abundance and payload count remain plotted, and the displayed denominator '
              'is 19 rather than 20. Horizontal endpoint collision offsets never change measured '
              'ordinates. Z80 population disc areas scale exactly with counts; exact capacities '
              'remain on their measured coordinates.',
              '', 'Validated 8086 ecology and controlled joint tests of insufficient coverage or '
              'altered phase have no completed outcomes in the supplied tables. Figure 5 waits '
              'for those data. The 8086 appears only as a labelled pending key entry in Figure 3. '
              'The copied joint/parent-environment supporting tables are available without '
              'inventing outcomes or leaving gaps.',
              '', '**Verification.** ¦qa_manifest.json¦ records format sizes, 600 dpi metadata, '
              'editable/vector text, PDF text inside page bounds, complete frozen-table hashes, '
              'C1/archive source reconciliation, capacities, all 256 dial values, exact lineages, '
              '80 competition endpoints, unchanged plotted ordinates, clean figcheck, and all '
              'archived review-image hashes and separately identified final-image hashes. '
              'Visual inspection caught and fixed an axis/footer collision '
              'that automated figcheck did not detect.',
              '', '**Reproduce from the worktree root.**',
              '', 'Uses the existing figure Python environment, plus PyMuPDF and Poppler’s '
              '¦pdftoppm¦ for vector composition and 600 dpi rendering.',
              '', '¦¦¦sh',
              '/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/figures/prepare_v7.py',
              '/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/figures/build_v7.py',
              '/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/figures/legends_v7.py',
              '/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/figures/verify_v7.py',
              '¦¦¦', '',
              'Art review uses ¦critique_v7.py --figure N --round R¦, invoking the required '
              '¦codex exec -m gpt-6-astra -c model_reasoning_effort=\'"high"\' -i reference.png '
              '-i new.png¦ command in an empty temporary directory with tools forbidden in the '
              'prompt. Archived rounds cannot be overwritten; the three-round cap is exhausted.',
              '', '[Full legends](LEGENDS_V7.md) · [Frozen tables](' + prefix + 'data/) · '
              '[Technical manifest](' + prefix + 'qa_manifest.json)', '',
              '**Corrections after round three (not Astra cleared).** Figure 1 names hardware '
              'states, ring-memory stability, identical assay metrics, whole-claim verdicts and '
              'constituent pass/fail tests; the ring rate-bound criterion is explicit. Figure 2 '
              'identifies parent and partner address regions, modulo-ring wrapping, the initial '
              'single-byte ancestry label and the circled source-0 → destination-8 erasing write. '
              'Figure 3 names exact, qualifying-site and sampled-carriage measurements, defines '
              'the whole-tape match gate and its zero-count deficit, and gives the normalized '
              'gain formula and matching baseline. Figure 4 names the first-confined-copier '
              'Kaplan–Meier event and consistently calls C1/archive execution implementations. '
              'Data coordinates and measured values are unchanged. These edits received clean '
              'figcheck and local visual/technical verification only.', '',
              '**Last image-only assessments.** The following are the unedited third-round '
              'reviews of the archived snapshots, before the label corrections above. Clearance '
              'requires a reviewed final image and an explicit no-substantive-defects verdict; '
              'neither is claimed here.', '']
    for name, row in records.items():
        lines += [f'### {name}', '', (OUT / row['final_review']).read_text().strip(), '']
    report = '\n'.join(lines).replace('¦', chr(96))
    (HERE.parent / 'FIGURES_V7.md').write_text(report)
    (OUT / 'FIGURES_V7.md').write_text(report.replace(prefix, ''))


if __name__ == '__main__':
    main()

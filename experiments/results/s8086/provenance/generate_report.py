"""CPU-only verification and presentation of unchanged frozen S2 reporters.

Run from experiments with the registered interpreter and -I. No execution of
downloaded code, no GPU or Modal functions, and no changes to scientific code.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))  # trusted repository source, never downloads
import numpy as np
import s8086 as S
import score8086 as Score
import law8086 as Law
from census8086 import panel
from census8086_law import GEOMETRIES, parents, variants as control_variants
from algocell_exp.zcensus import partner_bytes

OUT = HERE / 'results/s8086'
PROV = OUT / 'provenance'
TABLES = OUT / 'tables'
RAW = OUT / 'downloads'
RUN = OUT / 'launch/run-20261012-parallel-9000/scoring-onward/run-20261012-lock-recovery'
CLASSES = ('literal writer', 'memory-fed one-pass copier', 'confined copy loop', 'other')

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(S.safe(value), indent=2, allow_nan=False) + '\n')

def csv_file(name, rows, fields):
    path = TABLES / name
    with (gzip.open(path, 'wt', newline='') if path.suffix == '.gz' else path.open('w', newline='')) as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(row[k], separators=(',', ':')) if isinstance(row.get(k), (dict, list))
                        else row.get(k) for k in fields})

def md_table(fields, rows):
    def cell(value):
        return str(value if value is not None else '—').replace('|', '\\|')
    return '\n'.join(['| ' + ' | '.join(fields) + ' |',
                      '| ' + ' | '.join('---' for _ in fields) + ' |'] +
                     ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in rows])

def check_controls(summary):
    """Validate archived encounter hashes against frozen assignments on CPU.

    This reconstructs inputs and registered expected outputs; it does not run
    an emulator or replace the saved mismatch flags or frozen law verdicts.
    """
    expected_names = {f'L{L}_d{d}_{op:02x}' for L, d in GEOMETRIES for op in (0xa4, 0xa5)}
    assert set(summary) == expected_names
    checks = []
    for L, d in GEOMETRIES:
        frozen_panel = [p.tobytes() for p in panel(L)]
        for op in (0xa4, 0xa5):
            name = f'L{L}_d{d}_{op:02x}'
            path = RAW / 'controls/s8086_controls' / (name + '.jsonl.gz')
            baseline = parents(d, L, op)
            cases = control_variants(L, d, op)
            ids = {c[0]: i for i, c in enumerate(cases)}
            inputs = {c[0]: c[4].tobytes() for c in cases}
            expected = {c[0]: np.tile(c[4][:d], L // d).tobytes() for c in cases}
            geometry = summary[name]['geometry']
            for case, kind, p, result in Law.variants(baseline, geometry, 'A' if L == 16 else 'B', 8086001):
                if kind == 'erased':
                    inputs[case] = p.tobytes()
                    expected[case] = result.tobytes()
            serial_partners = {g: partner_bytes(np.array(list(ids.values()), np.uint64), g, L,
                seed=0x8086C0DE ^ (L << 16) ^ (d << 8) ^ op ^ 128) for g in range(1, 9)}
            seen = set()
            counts = Counter()
            with gzip.open(path, 'rt') as f:
                first = json.loads(next(f))
                assert first['kind'] == 'input'
                cert = first['certificate']
                assert cert['parent'] == baseline.tobytes().hex()
                assert cert['families']['L1'] == geometry
                assert cert['partners'] == [p.hex() for p in frozen_panel]
                assert cert['input_sha256'] == S.sha(b''.join(baseline.tobytes() + p for p in frozen_panel))
                for line in f:
                    row = json.loads(line)
                    if row.get('kind') == 'summary':
                        assert {k: v for k, v in row.items() if k != 'kind'} == summary[name]
                        continue
                    assert row.get('kind') != 'candidate-mismatch'
                    case, challenge, generation = row['case'], row['challenge'], row['generation']
                    if challenge == 'census':
                        partner = int(row['partner'])
                        pbytes = frozen_panel[partner]
                        input_bytes = inputs[case] + pbytes
                    else:
                        assert challenge == 'serial' and case in ids and 1 <= generation <= 8
                        partner = None
                        pbytes = serial_partners[generation][ids[case]].tobytes()
                        assert row['partner_bytes'] == pbytes.hex() and row['partner_sha256'] == S.sha(pbytes)
                        input_bytes = (inputs[case] if generation == 1 else expected[case]) + pbytes
                    identity = (case, challenge, generation, partner)
                    assert identity not in seen
                    seen.add(identity)
                    assert row['input_sha256'] == S.sha(input_bytes)
                    assert row['output_sha256'] == S.sha(expected[case])
                    assert row['matched'] is True
                    assert row['generation_zero_source_map'] == geometry['source_map']
                    assert row['physical_entry'] == row['phase'] == 0
                    counts[challenge] += 1
            assert sum(counts.values()) == summary[name]['encounters']
            assert counts['census'] == len(inputs) * 17
            assert counts['serial'] == len(ids) * 8
            checks.append({'control': name, 'input_output_hashes_verified': sum(counts.values()),
                           'census': counts['census'], 'serial': counts['serial'], 'mismatches': 0})
            print(f'Verified {name}: {sum(counts.values())} encounter hashes', flush=True)
    return checks

def main():
    assert sys.flags.isolated
    frozen = S.verify_freeze()
    TABLES.mkdir(exist_ok=True)
    release = S.read_json(RUN / 'eligibility_release.json')
    registration = S.Registration.from_dict(release['registration'])
    assert registration == S.DEFAULT_REGISTRATION
    assert S.sha((S.PRE / 'freeze.json').read_bytes()) == release['freeze_sha256']
    integrity = S.read_json(RUN / 'world_input_integrity.json')
    assert S.sha((RUN / 'world_input_integrity.json').read_bytes()) == release['completed_world_input_integrity_sha256']
    assert S.read_json(RUN / 'pipeline.status.json')['kind'] == 'pipeline_complete'
    event_rows = [json.loads(l) for l in (RUN / 'pipeline.events.jsonl').read_text().splitlines()]
    assert {r['stage'] for r in event_rows if r['kind'] == 'stage_complete'} == {'prebuild', 'score', 'extract', 'assay', 'controls'}
    assert next(r['time'] for r in event_rows if r['kind'] == 'eligibility_released') < next(
        r['time'] for r in event_rows if r['kind'] == 'stage_started' and r['stage'] == 'assay')
    scores, eligibility, outcomes, inputs = [], [], {}, {}
    verified = []
    for entry in integrity['worlds']:
        name = entry['world']
        world_path = RAW / 'worlds_A/A' / (name + '.jsonl')
        for record in [entry, *entry['snapshots']]:
            path = world_path.parent / Path(record['path']).name
            assert path.stat().st_size == record['bytes'] and S.sha(path.read_bytes()) == record['sha256']
            verified.append({'remote_path': record['path'], 'local_path': str(path.relative_to(HERE)),
                             'sha256': record['sha256'], 'verification': 'recorded pre-score byte digest'})
        condition, samples, snapshot = Score.read_input(world_path, registration)
        assert condition['freeze_sha256'] == entry['freeze_sha256']
        assert frozen['completed_world_inputs'][name]['input_sha256'] == entry['sha256']
        scored = S.read_json(RAW / 'score_A/A' / (name + '.json.gz'))
        assert scored['input_sha256'] == entry['sha256'] and scored['registration'] == registration.as_dict()
        judged = {r['tape']: r for r in scored['judged_tapes']}
        assert Score.endpoints(samples, lambda tape: judged[tape], registration) == scored['endpoints']
        assert scored['world']['scoreable'] and not scored['endpoints']['unresolved']
        assert scored['random_cell_indices'] == S.rng('A', condition['seed'], 16, 'cells-final').choice(20000, 64, replace=False).tolist()
        data_path = RAW / 'extract_A/A' / (name + '.json.gz')
        assert S.sha(data_path.read_bytes()) == release['eligibility_sha256'][name]
        committed = subprocess.check_output(['git', 'show', release['eligibility_commit'] + ':experiments/results/s8086/eligibility/A/' + name + '.json.gz'], cwd=HERE)
        assert committed == data_path.read_bytes()
        elig = S.read_json(data_path)
        assert elig['scored_sha256'] == S.sha(json.dumps(scored, sort_keys=True).encode())
        assert elig['extractor_sha256'] == frozen['files']['law8086.py']
        assert elig['world'] == scored['world'] and elig['candidates'] == scored['candidates']
        assert elig['registration'] == registration.as_dict()
        for tape, cert in elig['certificates'].items():
            assert judged[tape]['heritable']
            assert cert['parent'] == tape and cert['L'] == 16
            assert cert['partners'] == [p.tobytes().hex() for p in panel(16)]
            assert cert['input_sha256'] == S.sha(b''.join(bytes.fromhex(tape) + p.tobytes() for p in panel(16)))
        for family in ('L1', 'L2'):
            assert Law.selection(elig['candidates'], elig['certificates'], 'A', condition['seed'], 16, family) == elig['selection'][family]
        outcome = S.read_json(RAW / 'assay_A/A' / name / 'summary.json')
        assert set(outcome) == {'L1', 'L2'}
        for family in outcome:
            assert set(outcome[family]) == set(elig['selection'][family]['selected'])
        scores.append(scored)
        eligibility.append(elig)
        outcomes[name] = outcome
        inputs[name] = samples
    assert len(scores) == len(eligibility) == 20
    control_summary = S.read_json(RAW / 'controls/s8086_controls/summary.json')
    control_checks = check_controls(control_summary)
    controls_complete = len(control_summary) == 10 and all(r['status'] == 'complete-zero-mismatch' for r in control_summary.values())
    # Execute the unchanged frozen report functions, retain their exact outputs.
    report_dir = OUT / 'frozen_report'
    if report_dir.exists():
        shutil.rmtree(report_dir)  # only this script's generated artifact
    steps = []
    for step, action in [('score8086.report', lambda: Score.report(scores, report_dir, registration)),
                         ('law8086.report', lambda: Law.report(eligibility, outcomes, controls_complete,
                                                              report_dir / 'law_verdicts.json', registration))]:
        try:
            action()
            steps.append({'step': step, 'status': 'complete', 'source_sha256': frozen['files'][step.split('.')[0] + '.py']})
        except Exception:
            error = traceback.format_exc()
            steps.append({'step': step, 'status': 'failed', 'error': error})
            dump(PROV / 'report_steps.json', steps)
            (OUT / 'S2.md').write_text('# S2 frozen report failed\n\n' + step + '\n\n```text\n' + error + '\n```\nNo analysis code was patched.\n')
            raise
    dump(PROV / 'report_steps.json', steps)
    route = S.read_json(report_dir / 'route_verdicts.json')['arms'][0]
    laws = S.read_json(report_dir / 'law_verdicts.json')['families']
    dump(TABLES / 'law_outcomes.json', outcomes)
    csv_file('law_eligible_cases.csv', laws[0]['source_root_core_phase_certificates'] + laws[1]['source_root_core_phase_certificates'],
             ['case', 'family', 'd', 'c', 'K', 'R', 'C_R', 'F', 'E', 'held_erased', 'erasure_controls', 'phase', 'physical_entry', 'source_map'])
    csv_file('law_denominators.csv', [{'family': r['family'], 'exclude_L1': r['exclude_L1'], 'frozen_verdict': r['verdict'],
             'eligible': r['denominators'].get('eligible', 0),
             'independent_worlds': r['independent_worlds'], 'selected': r['selected'], 'completed': r['completed'],
             'unresolved_extractions': r['unresolved_extractions'], **r['denominators']} for r in laws],
             ['family', 'exclude_L1', 'frozen_verdict', 'source_records', 'random_cells', 'unique_candidates', 'culture_positive',
              'random_cell_unknowns', 'random_cell_culture_positive', 'random_cell_eligible', 'eligible', 'selected', 'completed',
              'cap_excluded', 'independent_worlds', 'unresolved_extractions'])
    worlds, encounters, others, copiers, persistence, law_candidates, law_worlds, occurrence = [], [], [], [], [], [], [], []
    for scored, elig in zip(scores, eligibility):
        w, ep = scored['world'], scored['endpoints']
        name = w['world']
        first, closed = w['first'], w['closed']
        worlds.append({'world': name, 'seed': w['seed'], 'scoreable': w['scoreable'], 't_rep': w['t_rep'],
            'rep_preceding_step': first['preceding_step'] if first else None,
            'rep_right_censored': first is None, 'rep_baseline': first['baseline'] if first else False,
            't_closed': w['t_closed'], 'closed_preceding_step': closed['preceding_step'] if closed else None,
            'closed_right_censored': closed is None, 'censor_step': registration.horizon,
            'first_tape': first['tape'] if first else None, 'first_share': first['share'] if first else None,
            'first_gen2': first['gen2'] if first else None, 'first_fetch': first['fetch_class'] if first else None,
            'first_open_encounters': first['entered'] if first else None,
            'first_majority_mechanism': first['majority_mechanism'] if first else None,
            'first_mechanism_counts': first['mechanism_counts'] if first else None,
            'first_max_B_hamming_from_zero': first['max_B_hamming_from_zero'] if first else None,
            'closed_tape': closed['tape'] if closed else None,
            'occurrence': w['open_memory_occurrence'], 'persistent': w['persistent_open_memory'],
            'same_genotype_persistence': w['same_genotype_persistence']})
        judged = {r['tape']: r for r in scored['judged_tapes']}
        detected = {}
        for sample in inputs[name]:
            for tape, share in sample['top3']:
                if share >= .005 and judged[tape]['heritable']:
                    detected.setdefault(tape, []).append(sample['step'])
        for tape, r in judged.items():
            scope = {'world': name, 'seed': w['seed'], 'tape': tape, 'heritable': r['heritable'],
                     'scheduled_detection': tape in detected, 'first_copier': bool(first and tape == first['tape']),
                     'fetch_class': r['fetch_class']}
            if r['heritable']:
                copiers.append({**scope, 'first_detection': min(detected[tape]) if tape in detected else None,
                    'last_detection': max(detected[tape]) if tape in detected else None,
                    'detected_samples': len(detected.get(tape, [])), 'gen2': r['gen2'],
                    'majority_mechanism': r['majority_mechanism'], 'mechanism_counts': r['mechanism_counts'],
                    'open_memory_fed': r['open_memory_fed'], 'entered': r['entered'],
                    'max_B_hamming_from_zero': r['max_B_hamming_from_zero'], 'zero_reference': r['zero_reference']})
            for e in r['encounters']:
                encounter = {**scope, **e}
                encounters.append(encounter)
                if e['mechanism'] == 'other':
                    others.append({**encounter, 'score_path': 'downloads/score_A/A/' + name + '.json.gz',
                                   'partner_bytes': r['route_partners'][e['partner']], 'input_sha256': r['input_sha256']})
            if r['zero_reference']['mechanism'][0] == 'other':
                others.append({**scope, 'partner': -1, 'mechanism': 'other',
                    'reason': r['zero_reference']['mechanism'][1], 'zero_reference': r['zero_reference'],
                    'partner_bytes': '00' * 16, 'input_sha256': r['input_sha256'],
                    'score_path': 'downloads/score_A/A/' + name + '.json.gz'})
        for step, tapes in ep['persistence_samples'].items():
            persistence.append({'world': name, 'seed': w['seed'], 'step': int(step), 'qualifying_tapes': tapes,
                                'event_present': bool(tapes)})
        occurrence.extend({'world': name, 'seed': w['seed'], **r} for r in ep['occurrence'])
        for tape in sorted({r['tape'] for r in elig['candidates']}):
            cert = elig['certificates'].get(tape)
            sources = [r['source'] for r in elig['candidates'] if r['tape'] == tape]
            for family in ('L1', 'L2'):
                g = cert['families'][family] if cert else {'status': 'culture-negative', 'reason': 'culture_not_positive'}
                law_candidates.append({'world': name, 'tape': tape, 'family': family, 'sources': sources,
                    'random_cell_multiplicity': sources.count('random-cell'), 'gen2': judged[tape]['gen2'],
                    'status': g['status'], 'reason': g.get('reason'),
                    'selected': tape in elig['selection'][family]['selected'], 'geometry': g if g['status'] == 'eligible' else None})
        cells = [r for r in elig['candidates'] if r['source'] == 'random-cell']
        law_worlds.append({'world': name, 'source_records': len(elig['candidates']), 'random_cells': len(cells),
            'unique_candidates': len({r['tape'] for r in elig['candidates']}), 'culture_positive': len(elig['certificates']),
            'random_cell_culture_positive': sum(r['culture']['gen2_score'] is not None and r['culture']['gen2_score'] >= .3 for r in cells),
            'random_cell_unknowns': sum(r['culture']['gen2_score'] is None for r in cells),
            'L1_eligible': len(elig['selection']['L1']['eligible']), 'L2_eligible': len(elig['selection']['L2']['eligible']),
            'L2_unresolved': sum(c['families']['L2']['status'] == 'unresolved' for c in elig['certificates'].values())})
    csv_file('worlds.csv', worlds, list(worlds[0]))
    csv_file('copier_tapes.csv', copiers, list(copiers[0]))
    encounter_fields = list(encounters[0])
    csv_file('judged_encounters.csv.gz', encounters, encounter_fields)
    csv_file('first_copier_encounters.csv', [r for r in encounters if r['first_copier']], encounter_fields)
    csv_file('persistence.csv', persistence, ['world', 'seed', 'step', 'qualifying_tapes', 'event_present'])
    csv_file('law_candidates.csv', law_candidates, list(law_candidates[0]))
    csv_file('law_worlds.csv', law_worlds, list(law_worlds[0]))
    with gzip.open(TABLES / 'other_cases.jsonl.gz', 'wt') as f:
        for row in others:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    csv_file('other_cases.csv', others, ['world', 'seed', 'tape', 'heritable', 'scheduled_detection', 'first_copier',
             'fetch_class', 'partner', 'mechanism', 'reason', 'open', 'partner_write_coverage', 'write_transactions',
             'original_partner_fetch_reads', 'original_partner_data_reads', 'trace_sha256', 'record_sha256', 'output_sha256',
             'score_path', 'input_sha256'])
    dump(TABLES / 'open_memory_occurrence.json', occurrence)
    csv_file('first_tape_mechanisms.csv', route['first_tape_distribution'], ['fetch', 'mechanism', 'count'])
    csv_file('first_encounter_mechanisms.csv', route['first_encounter_distribution'], ['tape_fetch', 'open', 'mechanism', 'count'])
    km_rows = [{'endpoint': endpoint, **r} for endpoint in ('rep', 'closed') for r in route['km_' + endpoint]['curve']]
    csv_file('kaplan_meier.csv', km_rows, ['endpoint', 'step', 'at_risk', 'events', 'survival'])
    control_rows = [{'control': name, 'L': int(name.split('_')[0][1:]), 'd': v['geometry']['d'],
        'operation': name.split('_')[-1], **{k: value for k, value in v.items() if k != 'geometry'}}
        for name, v in sorted(control_summary.items())]
    csv_file('constructed_controls.csv', control_rows, list(control_rows[0]))
    csv_file('control_hash_verification.csv', control_checks, list(control_checks[0]))
    census = list(csv.DictReader((HERE / 'results/census8086/table_mechanisms.csv').open()))
    csv_file('census_mechanisms.csv', census, list(census[0]))
    witnesses = []
    for L in (16, 32):
        for budget in (128, 512):
            for r in csv.DictReader((HERE / f'results/census8086/mechanisms_L{L}_b{budget}.csv').open()):
                if r['word'] == '07a5' and r['k'] == '2':
                    witnesses.append({'L': L, 'budget': budget, **r})
    csv_file('census_07a5.csv', witnesses, list(witnesses[0]))
    inventory = []
    bases = {'worlds_A': '/s8086', 'score_A': '/s8086_score', 'extract_A': '/s8086_extract', 'assay_A': '/s8086_assay', 'controls': '/'}
    for folder, remote_base in bases.items():
        for path in sorted((RAW / folder).rglob('*')):
            if path.is_file():
                rel = str(path.relative_to(RAW / folder))
                inventory.append({'local_path': str(path.relative_to(HERE)), 'remote_path': remote_base.rstrip('/') + '/' + rel,
                                  'bytes': path.stat().st_size, 'sha256': S.sha(path.read_bytes())})
    csv_file('download_hashes.csv', inventory, ['local_path', 'remote_path', 'bytes', 'sha256'])
    verification = {'python': sys.executable, 'isolated': bool(sys.flags.isolated), 'verified_utc': datetime.now(timezone.utc).isoformat(),
        'volume': 'algocell-atlas-runs', 'download_method': 'modal volume get (download only)',
        'world_records_verified': 20, 'compressed_and_raw_checkpoints_verified': 100,
        'score_content_digests_verified_against_preassay_eligibility': 20, 'eligibility_byte_digests_and_commit_verified': 20,
        'eligibility_commit': release['eligibility_commit'], 'empty_assay_summaries_verified': 20,
        'controls': control_checks, 'control_encounter_hashes_verified': sum(r['input_output_hashes_verified'] for r in control_checks),
        'frozen_report_steps': steps, 'frozen_source_unchanged': True, 'download_files': len(inventory),
        'download_bytes': sum(r['bytes'] for r in inventory), 'world_hash_records': verified,
        'limitations': ['No independently pre-recorded byte digest for assay/control export files was found in the launch receipts. Their byte digests are recorded in download_hashes.csv; controls are additionally checked against all saved input/output hashes and summary counts.',
                       'World-attempt archives outside the 20 frozen canonical worlds have locally recorded export digests; they do not enter scientific denominators.',
                       'No emulator execution was repeated. Frozen scoring endpoints were replayed from archived judgments; report functions ran unchanged.']}
    dump(PROV / 'verification.json', verification)
    spend = S.read_json(PROV / 'final_spend.json')
    exclusions = laws[1]['extraction_failure_reasons']
    l2_worlds = sum(r['L2_unresolved'] > 0 for r in law_worlds)
    first_others = sum(r['first_copier'] and r['partner'] >= 0 for r in others)
    text = f'''# S2 — registered coherent-fetch 8086 verdicts

Authority: [REVISION_PREREG.md, S2 and amendments](../../REVISION_PREREG.md#s2--the-third-machine-the-coherent-fetch-8086-registered-and-amended-2026-10-12-before-any-s2-world-is-run-option-b-all-in-cap-85-approved-by-the-user-2026-10-12), including the prospective R2 withdrawal/$95 amendment and the L2 source-root/core/phase amendment. Only A remains: L16, budget 128, seeds 8086001–8086020, 20 complete and scoreable worlds through 3,000,000 steps, 585 scheduled samples each and five complete checkpoints each. B/L32 is withdrawn as a population arm. Coherent-fetch instruction semantics are the registered abstraction; there is no directional architecture-comparison timing prediction.

The completed pipeline is `launch/run-20261012-parallel-9000/scoring-onward/run-20261012-lock-recovery`. The [completion record](provenance/run/pipeline.status.json) and [events](provenance/run/pipeline.events.jsonl) show all five compute stages complete. Eligibility was committed as `{release['eligibility_commit']}` before evolved assays. Frozen source commit: `{release['frozen_code_commit']}`; freeze SHA-256: `{release['freeze_sha256']}`. All source hashes and all 20 world/100 checkpoint hashes verified. [Verification](provenance/verification.json) and [export hashes](tables/download_hashes.csv) disclose the scope of each check.

## Registered route verdicts

| Test | Observed | Registered rule | Verdict |
| --- | --- | --- | --- |
| R0 | D = {route['D']}/20 | D ≥ 10 | **Pass** |
| R1 | O/D = {route['O']}/{route['D']} ({100*route['O']/route['D']:.2f}%) | D ≥ 10 and O ≥ ceil(0.8D) = {math.ceil(.8*route['D'])} | **Fail** |
| Route kill | {route['first_confined']}/{route['D']} first copiers confined | With D ≥ 10, confined ≥ ceil(0.25D) = {math.ceil(.25*route['D'])} | **Triggered** |

The registered census-motivated open-beginning claim for the 8086 is withdrawn. Both open first copiers remain in D, as do every confined and `other` first copier; there are zero intermediate first copiers. R1 uses full-encounter B fetch in ≥8/16 traces; confinement requires zero B fetch in all 16. Mechanism labels do not decide R1. R2 has **descriptive status only**; no R2 pass, failure, joint verdict or mechanism-based scientific withdrawal is asserted.

## L1 and amended L2

| Family | Eligible genotype-world cases | Selected/completed | Independent evolved worlds | Registered interpretation | Exact frozen report status |
| --- | --- | --- | --- | --- | --- |
| L1: narrow six-byte REP | 0 | 0/0 | 0 | **Untested** | `evolved law untested` |
| L2: forward family, with L1 | 0 | 0/0 | 0 | **Untested; support suspended by unresolved certificates** | `incomplete/invalid` |
| L2: forward family, excluding L1 | 0 | 0/0 | 0 | **Untested; support suspended by unresolved certificates** | `incomplete/invalid` |

There are no eligible evolved genotypes to list. Evolved single-coordinate assignments/alternatives, joint assignments, mutable-erasure interventions, held-erasure controls, partner challenges and serial-generation encounters are all **zero** for both families. No evolved mismatch was observed because no evolved case was assayed; this supplies neither support nor a failed application. There are no certified evolved roots R, core K/source-core C_R, free set F, erased set E, mutable/held erasure denominators, source maps or physical phases. Empty [eligible-case table](tables/law_eligible_cases.csv) and [assay outcomes](tables/law_outcomes.json) are retained.

Across 20 worlds the candidate register contains {laws[0]['denominators']['source_records']:,} source records, {laws[0]['denominators']['unique_candidates']:,} distinct genotype-world candidates, and {laws[0]['denominators']['culture_positive']:,} culture-positive genotype-world candidates. All 1,280 random cells remain in the cell denominator, including duplicate genotypes: {laws[0]['denominators']['random_cell_culture_positive']}/1,280 culture-positive, 0 unknown, 0 eligible for either family. No case was excluded by the per-family assay cap. [Candidate/exclusion ledger](tables/law_candidates.csv), [world coverage](tables/law_worlds.csv), and [family denominators](tables/law_denominators.csv) retain every case and multiplicity.

L1 excludes all 779 culture-positive candidates as `outside_verified_six_byte_REP_family`. L2 has **724 unresolved extractions across {l2_worlds} independently evolved worlds**, and 55 exclusions for `original_partner_read`:

{md_table(['L2 extraction reason', 'Genotype-world cases'], [(k, v) for k, v in sorted(exclusions.items())])}

The frozen L2 reporter converts unresolved certificates to `incomplete/invalid`; this is disclosed without modifying the reporter or recoding it as a law failure. The registered three-outcome reading is untested, with support suspended. Pipeline completion and route scoreability remain complete. L2 does not rescue L1. The amended L2 rule requires independently proven roots and physical-phase offspring, with K and C_R held, arbitrary F assignments and erasure substitutions only at E\\K; no favorable rotation or inferred alignment is introduced.

All **10 constructed controls** (five L/d geometries × MOVSB/MOVSW at budget 128, including fixed-input L32) completed with **zero mismatches over 1,185,792 encounters**. All recorded encounter input/output hashes, serial partner hashes, generation-zero source maps, phase-zero comparisons and encounter counts were verified on CPU against the frozen assignments. [Control results](tables/constructed_controls.csv) and [hash verification](tables/control_hash_verification.csv) retain each geometry's assignments and denominators. Constructed success never supplies evolved support. No frozen report step failed, and no analysis code was patched.

## R2 — descriptive mechanisms and persistence

First-copier tape majority classes (≥9/16; heterogeneous profiles are separate):

{md_table(['Full-encounter tape fetch', *CLASSES, 'Heterogeneous', 'Tapes'], [[fetch, *[sum(r['count'] for r in route['first_tape_distribution'] if r['fetch'] == fetch and r['mechanism'] == c) for c in (*CLASSES, 'heterogeneous')], sum(r['count'] for r in route['first_tape_distribution'] if r['fetch'] == fetch)] for fetch in ('open', 'intermediate', 'confined')])}

First-copier encounter classes (all 224 random-partner encounters):

{md_table(['Tape fetch', 'Encounter B fetch', *CLASSES, 'Encounters'], [[fetch, opened, *[sum(r['count'] for r in route['first_encounter_distribution'] if r['tape_fetch'] == fetch and r['open'] == opened and r['mechanism'] == c) for c in CLASSES], sum(r['count'] for r in route['first_encounter_distribution'] if r['tape_fetch'] == fetch and r['open'] == opened)] for fetch, opened in (('open', True), ('open', False), ('intermediate', True), ('intermediate', False), ('confined', False))])}

Open memory-fed occurrence is **{route['occurrence_worlds']}/20 worlds**; persistence at every one of the 11 points 2.75M, 2.775M, …, 3M is **{route['persistence_worlds']}/20**; same-genotype persistence is **{sum(w['same_genotype_persistence'] for w in worlds)}/20**. These are counts, without confirmatory cutoffs. The qualifying occurrence is in {', '.join(str(w['seed']) for w in worlds if w['occurrence'])}; its complete [occurrence events](tables/open_memory_occurrence.json) are retained. [Persistence table](tables/persistence.csv) lists all 220 world-time points and exact qualifying tapes; no missing point is interpolated.

All 16 partner labels, write coverage/transactions, original-partner fetch/data-access proxies, causal intervention evidence and stored trace/record/output identities appear in [first-copier encounters](tables/first_copier_encounters.csv) and [all judged encounters](tables/judged_encounters.csv.gz). Every `other` case is listed in [the case table](tables/other_cases.csv), with full available causal checks and partner bytes in [its evidence archive](tables/other_cases.jsonl.gz), including zero-reference and culture-negative cases for completeness. The first-copier subset contains **{first_others} `other` random-partner encounters**. These fail-safe labels do not erase cases or imply a causal mechanism. [Copier tapes](tables/copier_tapes.csv) includes every culture-positive judged tape and separately flags actual scheduled detections versus final-cell-only cultures. Openness, episode confinement, access proxies and zero-reference output dependence remain separate measurements.

## Detection and confinement context

Kaplan–Meier detection median is **{route['km_rep']['median']:,} steps**, with 14 detections and six worlds right-censored at 3M. Confinement median is **{route['km_closed']['median']:,} steps**, with 12 `t_closed` detections and eight right-censored worlds. There are zero step-zero detections. Detection times are scheduled observations with their preceding-sample intervals, not exact births. `t_closed` is the first culture-positive tape confined in all 16 full-encounter traces, not proof of a loop, causal closure or disappearance of every other open tape. [KM risk/event/survival table](tables/kaplan_meier.csv) and [world records](tables/worlds.csv) retain censoring and both intervals.

{md_table(['Seed', 't_rep', 't_closed', 'First fetch', 'First majority mechanism', 'Exact first-copier tape'], [[w['seed'], w['t_rep'] if w['t_rep'] is not None else 'censored@3M', w['t_closed'] if w['t_closed'] is not None else 'censored@3M', w['first_fetch'], w['first_majority_mechanism'], '`' + w['first_tape'] + '`' if w['first_tape'] else '—'] for w in worlds])}

## Census-based descriptive context

Use the final causally verified [census correction](../census8086/DESCRIPTIVE_CORRECTION_2026-10-12.md), rather than earlier superseded class labels. Nonzero period-two exact copies against the zero partner:

{md_table(['L / budget', *CLASSES, 'Total'], [[f'{L}/{budget}', *[sum(int(r['zero_exact_nonzero']) for r in census if int(r['L']) == L and int(r['budget']) == budget and r['k'] == '2' and r['mechanism'] == c) for c in CLASSES], sum(int(r['zero_exact_nonzero']) for r in census if int(r['L']) == L and int(r['budget']) == budget and r['k'] == '2')] for L, budget in ((16, 128), (16, 512), (32, 128), (32, 512))])}

Nonzero period-one exact-zero counts are all zero. The census's **07 A5** witness remains causally verified memory-fed one-pass in all 68 encounters (four L/budget settings × 17 partners), with all 64 random-partner encounters open. At L16/budget128 it is exact in zero plus 16/16 random partners; the other exact-zero memory-fed word AB 58 is exact in 0/16 random partners. These demonstrate available mechanisms and context robustness, not establishment probabilities or a predicted route frequency. The census audits 1,246 settings/21,182 encounters; its final correction changes 1,247 labels without changing outputs/matches. [Copied census class table](tables/census_mechanisms.csv) and [07 A5 panel](tables/census_07a5.csv) preserve the descriptive evidence. L32 is fixed-input context only.

## Final spend and reproducibility

Fresh app-attributed hourly billing at `{spend['queried_utc']}` is **${spend['metered_usd']:.8f}**, including preparation and earlier attempts. Billing may lag; this is a final reconciliation reading, not an assertion of a settled invoice. All {len(spend['apps'])} attributed apps were provider-confirmed `APP_STATE_STOPPED` with no active tasks. The read used a 120-second initial exponential backoff policy capped at 1,800 seconds; {len(spend['attempts'])} attempt(s) were needed.

The unchanged spend guard, replayed on a copied ledger with fresh billing, retains a **${spend['conservative_bound_usd']:.8f} conservative all-in bound**, below the **$95 cap** by **${spend['cap_headroom_usd']:.8f}**. The shared $5 storage/termination reserve is counted once; the unallocated diagnosis allowance remains reserved. Earlier failed/stopped attempts retain their full carried charges, and overlapping app billing is not added twice. Historical accounting statuses retain 22 `running` entries with stopped flags and full charged liabilities; they are not active provider tasks. No costs were released merely to reduce the bound. [Billing, resource rows and guard reconciliation](provenance/final_spend.json), [attempt log](provenance/billing_attempts.json) and [copied hash-chained ledger](provenance/final_spend_ledger.jsonl) retain the evidence; the original ledger is unchanged.

The only new Modal operations were free `modal volume get` downloads and read-only provider billing/lifecycle queries. No compute or dispatch occurred. Downloaded JSON/JSONL/gzip/Brotli data were read with `/Users/neo/repos/algocell/experiments/.venv/bin/python -I`. Unchanged `score8086.report` and `law8086.report` produced [the exact frozen JSON verdicts](frozen_report/); this presentation wrapper renders their outputs and adds CPU-only integrity checks/tables. Scoring endpoints were also replayed unchanged from archived judgments. All downloaded source directories, including failed-attempt world archives outside the canonical denominator, remain local and are hash-inventoried. Export-byte hashes for assay/control files are newly recorded; no prior byte attestation is invented.

Reproduce the report without compute from experiments/:

```sh
/Users/neo/repos/algocell/experiments/.venv/bin/python -I results/s8086/provenance/generate_report.py
```

The [presentation source](provenance/generate_report.py), [billing reader](provenance/final_spend.py), frozen source/registration identities and run provenance are committed with explicit paths. No push.
'''
    (OUT / 'S2.md').write_text(text)
    S.verify_freeze()
    print(json.dumps({'R0': route['R0'], 'D': route['D'], 'O': route['O'], 'R1': route['R1'], 'kill': route['kill'],
                      'laws': [(r['family'], r['exclude_L1'], r['verdict']) for r in laws],
                      'download_files': len(inventory), 'other_cases': len(others), 'control_hashes': verification['control_encounter_hashes_verified']}, indent=2))

if __name__ == '__main__':
    main()

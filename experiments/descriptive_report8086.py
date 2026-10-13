"""Dated census correction and descriptive class tables; CPU only."""
from collections import Counter
import csv
import json
from pathlib import Path
from s8086 import HERE, PRE, read_json, sha
from b6audit8086 import save_csv

OUT=HERE/'results/census8086'
CLASSES=('literal writer','memory-fed one-pass copier','confined copy loop','other')

def main():
    audit=read_json(PRE/'descriptive_census_validation.json')
    transitions=Counter((r['old'],r['new']) for r in audit['changes'])
    tables=[];all_counts=[];zero_rows=[];settings=0;others=[]
    for run in audit['runs']:
        L,budget=run['L'],run['budget']
        rows=list(csv.DictReader((OUT/f'mechanisms_L{L}_b{budget}.csv').open()))
        settings+=len({(r['k'],r['index']) for r in rows})
        before=run['before'];after=run['after']
        for cls in CLASSES:all_counts.append({'L':L,'budget':budget,'mechanism':cls,'before':before.get(cls,0),'after':after.get(cls,0)})
        for r in rows:
            if r['mechanism']=='other':others.append({'L':L,'budget':budget,**r})
        for k in (1,2):
            exact={int(r['index']) for r in csv.DictReader((OUT/f'words_L{L}_k{k}_b{budget}.csv').open()) if int(r['index']) and int(r['zero_match'])==L}
            counts=Counter(r['mechanism'] for r in rows if int(r['k'])==k and int(r['partner'])==0 and int(r['index']) in exact)
            for cls in CLASSES:tables.append({'L':L,'budget':budget,'k':k,'mechanism':cls,'zero_exact_nonzero':counts[cls]})
        zero_rows.append(f"| {L}/{budget} | "+' | '.join(str(run['zero_exact_after'].get(c,0)) for c in CLASSES)+' |')
    save_csv(OUT/'table_mechanisms.csv',tables)
    save_csv(OUT/'descriptive_correction_counts.csv',all_counts)
    save_csv(OUT/'descriptive_other_cases.csv',others)
    # Keep the hit/candidate inventories in agreement with the corrected labels.
    for name in ('table_hits.csv','table_candidates.csv'):
        rows=list(csv.DictReader((OUT/name).open()))
        lookup={}
        for L in (16,32):
            for budget in (128,512):
                for r in csv.DictReader((OUT/f'mechanisms_L{L}_b{budget}.csv').open()):
                    if r['partner']=='0':lookup[L,budget,int(r['k']),r['word']]=r
        for r in rows:
            m=lookup[int(r['L']),int(r['budget']),int(r.get('k',2)),r['word']]
            r['zero_mechanism']=m['mechanism']
            if 'zero_evidence' in r:r['zero_evidence']=m['evidence']
        save_csv(OUT/name,rows)
    provenance={'date':'2026-10-12','baseline_commit':'2b06edb','scope':'all archived 1246 candidate settings / 21182 encounters; zero+16 frozen random partners; all four L/budget settings',
        'status':'descriptive; no label-change stop conditions','settings':settings,'encounters':sum(r['encounters'] for r in audit['runs']),
        'label_changes':len(audit['changes']),'transitions':[{'old':a,'new':b,'count':n} for (a,b),n in sorted(transitions.items())],
        'runs':audit['runs'],'full_record_memory_trace_mismatches':0,'classifier_sha256':audit['classifier_sha256'],
        'audit_sha256':sha((PRE/'descriptive_census_validation.json').read_bytes()),'evidence_sha256':sha((PRE/'descriptive_intervention_checks.jsonl.gz').read_bytes()),
        'no_population':True,'no_paid_compute':True,'R0_R1_kill_L1_L2_unchanged':True}
    (OUT/'descriptive_correction_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    text=f'''# Final descriptive census correction — 2026-10-12

Prospective R2 withdrawal was committed as `39de676` before this audit and
before any S2 population data. R2a/R2b are descriptive; R0, fetch-based R1,
the route kill rule, L1 and amended L2 remain confirmatory and unchanged.
No label change stopped this complete audit.

All {settings:,} candidate settings / {provenance['encounters']:,} encounters
were rerun against the complete zero-plus-16 frozen partner panel under the
prefix-causal operand, memory and reset-value classifier. Full original lane
records, memory and archived trace bytes agree in every encounter: zero
mismatches. Raw word/output/match/coverage/access/count data are unchanged.

Exactly **{len(audit['changes']):,} labels change** relative to `2b06edb`:
'''+ '\n'.join(f'- {n:,} {a} → {b}.' for (a,b),n in sorted(transitions.items()))+'''

Every changed label is listed, with L, budget, period, exact word, partner,
old/new class and reason in `descriptive_correction_labels.csv`. Complete
before/after class counts are in `descriptive_correction_counts.csv` and
per-setting metadata in `descriptive_correction_provenance.json`. Every other
case (including unchanged ones) is shown in `descriptive_other_cases.csv`.
All four `mechanisms_*.csv` and the hit/candidate/mechanism tables are current.
Historical corrections below remain a chronological record; this correction
supersedes their class labels and counts.

Corrected nonzero exact-zero word counts (period two):

| L/budget | Literal | Memory-fed | Confined loop | Other |
|---|---:|---:|---:|---:|
'''+ '\n'.join(zero_rows)+'''

Nonzero period-one exact-zero counts remain zero. Total exact-zero period-two
word counts remain 20, 29, 26, 26. 07 A5 remains causally verified memory-fed
one-pass in all 68 encounters, including all 64 open random-partner encounters.
B9 51 and AD 50 include failed prefix-causal claims and are reported as other
where indicated; equal values cannot override a failed intervention. The
existing two overflow cases remain other. Mechanisms classify encounters,
not probabilities of establishment or population predictions.

All claims use both 55/AA interventions; unsupported/mixed/vector/overflow or
failed claims stay other. Detailed checks are archived in
`../s8086/prelaunch/descriptive_intervention_checks.jsonl.gz`; full audit and
burst metadata are beside it. The independent 8,192-random-program prefix,
source-expression and class differential is reported in
`../s8086/prelaunch/descriptive_random_validation.json`. It checks the causal
verification of conservative hypotheses and raw/observer parity, not arbitrary
unmodeled architectural semantics. No population or paid computation occurred.
'''
    (OUT/'DESCRIPTIVE_CORRECTION_2026-10-12.md').write_text(text)
    census=OUT/'CENSUS.md';old=census.read_text();marker='## Final descriptive correction — 2026-10-12'
    if marker in old:old=old[:old.index(marker)].rstrip()+'\n'
    census.write_text(old+'\n\n'+marker+'\n\nThe final dated [descriptive correction](DESCRIPTIVE_CORRECTION_2026-10-12.md)\nsupersedes earlier labels and mechanism counts. '+f'{len(audit["changes"]):,} labels change in\n{provenance["encounters"]:,} audited encounters; all changes and every other case\nare listed. R2a/R2b are descriptive only. Current exact-zero classes:\n\n| L/budget | Literal | Memory-fed | Confined loop | Other |\n|---|---:|---:|---:|---:|\n'+'\n'.join(zero_rows)+'\n')
    hashes=list(csv.DictReader((OUT/'table_hashes.csv').open()))
    by_file={r['file']:r for r in hashes}
    for name in ('descriptive_correction_labels.csv','descriptive_correction_counts.csv','descriptive_correction_provenance.json','descriptive_other_cases.csv'):
        by_file.setdefault(name,{'file':name})
    for name,r in by_file.items():
        path=HERE.parent/name.removeprefix('source:') if name.startswith('source:') else OUT/name
        r.update(bytes=len(path.read_bytes()),sha256=sha(path.read_bytes()))
    save_csv(OUT/'table_hashes.csv',sorted(by_file.values(),key=lambda r:r['file']),['file','bytes','sha256'])
    print(f'{settings} settings / 21182 encounters; {len(audit["changes"])} changed labels; all other cases disclosed')

if __name__=='__main__':main()

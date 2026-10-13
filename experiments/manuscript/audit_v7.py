#!/usr/bin/env python3
"""Read-only, reproducible numbers/references audit of MAIN_v7.

Run from any directory with experiments/.venv/bin/python. No simulations, network,
source edits, or git operations. Only AUDIT_v7.md and audit_v7.csv are written.
Verification uses named fields, filtered rows, aggregates, and contextual report
excerpts; mere presence of the same numeral is explicitly insufficient.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
DRAFT = HERE / "MAIN_v7.md"
V8 = HERE / "figures/out/v8"
STATUSES = {"verified", "mismatch", "source missing", "number not found in source", "no source bracket"}
WORDS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split(), range(21)))
WORDS.update({"ninety-nine": 99, "half": .5, "one half": .5, "none": 0})
NUM = re.compile(r"(?<![\w.])(?:\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|M)?)(?!\w|\.\d)|\b(?:" + "|".join(sorted(WORDS, key=len, reverse=True)) + r")\b", re.I)
PATH = re.compile(r"(?:/Users/[^\s\];`,)]+|(?:[\w.-]+/)+[\w./-]+|REVISION_PREREG\.md)")
BRACKET = re.compile(r"\[([^\]]*)\]")
CPU = re.compile(r"\b(?:Z80|6502|8086|8088|8080)\b")
FIG = re.compile(r"\b(?:(Extended Data\s+)(?:Fig(?:ure)?\.?|Table)|Fig(?:ure)?\.?)\s+(\d+)\s*([a-z](?:\s*[,–-]\s*[a-z])*)?(?!\w)")


def squash(s):
    return re.sub(r"\s+", " ", s).strip()


def fmt(x):
    return format(Decimal(str(x)).normalize(), "f") if isinstance(x, (int, float, Decimal)) else str(x)


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def resolve(name):
    p = Path(name)
    if p.is_absolute():
        return p
    if name.startswith("experiments/"):
        return REPO / name
    return EXP / name


OPENED = set()


@lru_cache(None)
def source_text(name):
    p = resolve(name)
    OPENED.add(p)
    return p.read_text(errors="replace") if p.is_file() else ""


def stream(name):
    p = resolve(name)
    OPENED.add(p)
    with (gzip.open(p, "rt", newline="") if name.endswith(".gz") else p.open(newline="")) as f:
        yield from csv.DictReader(f)


@lru_cache(None)
def table(name):
    return list(stream(name))


def truth(x):
    return str(x).lower() in {"true", "1"}


def val(row, key):
    return float(row[key])


def paths(s):
    return list(dict.fromkeys(m.group().rstrip(".") for m in PATH.finditer(s) if m.group().startswith(('/', 'results/', 'manuscript/', 'algocell_exp/', 'experiments/')) or m.group().endswith('/') or re.search(r'\.[a-z]{2,5}(?:\.gz)?$',m.group())))


@dataclass(frozen=True)
class Fact:
    value: float
    source: str
    meaning: str
    evidence: str = ""


FACTS = {}


def fact(key, value, source, meaning, evidence=""):
    FACTS[key] = Fact(float(value), source, meaning, evidence)
    return key


def colfacts(prefix, name, row, columns, selector):
    for c in columns:
        if row.get(c, "") != "":
            fact(prefix + "." + c, val(row, c), name, f"{selector}; field {c}")


def build_facts():
    b = "results/branching/branching_summary.csv"
    for key in ["pusher16", "pusher64", "ret16", "ldir32", "genome20_s6003"]:
        rows = [r for r in table(b) if r["key"] == key and r["arm"] == "U"]
        r = next(r for r in rows if r["gen"] == "8")
        colfacts(key, b, r, ["gen", "L", "ctrl_alive_n", "ctrl", "mut_alive", "retained", "sites", "n_mutants", "values", "reps", "k", "cap", "gens"], f"key={key}, arm=U, gen=8")
        fact(key + ".mutation_types_carried", val(r, "frac_ok_mutants") * val(r, "n_mutants"), b, f"{key}, U, g8: frac_ok_mutants × n_mutants; mutation types passing criterion")
    s = "results/tier2/T2_CLAIMS.csv"
    fact("claims.total", len(table(s)), s, "number of registered claim rows")
    fact("claims.same", sum(r["status"] == "stands" for r in table(s)), s, "claim status=stands count")
    s = "results/tier2/T2_COMPONENTS.csv"
    fact("components.total", len(table(s)), s, "number of registered component rows")
    fact("components.changed", sum(r["comparison"] != "same" for r in table(s)), s, "component comparison != same count")
    s = "manuscript/figures/out/v8/data/validation_fractions.csv"
    for i, r in enumerate(table(s)):
        colfacts(f"validation{i}", s, r, ["unchanged", "total", "changed"], r["level"])
    fact("validation.units", len(table(s)), s, "number of distinct validation rows")
    s = "manuscript/figures/out/v8/data/methods.csv"
    for r in table(s):
        fact(r["name"], r["value"], s, f"name={r['name']}; value")
    s = "results/impact/decisions.csv"
    for core in ["C1", "C2"]:
        rows = [r for r in table(s) if r["core"] == core and r["decision_scope"] == "1"]
        fact(core + ".ringtotal", len(rows), s, f"{core}, decision_scope=1: condition count")
        fact(core + ".flagged", sum(r["decision"] != "unaffected" for r in rows), s, f"{core}, decision_scope=1: non-unaffected condition count")
    s = "results/impact/assay_identity.csv"
    fact("assays.total", len(table(s)), s, "isolated-assay row count")
    for c in ["c1", "c2"]:
        fact(c + ".identical", sum(truth(r[c + "_identical"]) for r in table(s)), s, f"sum {c}_identical")
    s = "results/crystal/alleles_ldir32.csv"
    a = table(s)
    elig = [r for r in a if truth(r["prop8_eligible"])]
    working = [r for r in a if truth(r["culture_replicator"])]
    for key, n, meaning in [("dial.total", len(a), "all tested byte values"), ("dial.working", len(working), "culture_replicator=True"), ("dial.inherited", sum(val(r, "allele_aligned_frac") >= .5 for r in working), "working and allele_aligned_frac >= 0.5"), ("dial.eligible", len(elig), "prop8_eligible=True"), ("dial.nonworking", len(a)-len(working), "culture_replicator=False"), ("dial.excluded", len(working)-len(elig), "working outside sampled theorem eligibility")]:
        fact(key, n, s, meaning)
    for c in ["effective_d", "predicted_payload"]:
        for fn in [min, max]:
            fact(f"dial.{c}.{fn.__name__}", fn(val(r,c) for r in elig), s, f"{fn.__name__} {c} among eligible values")
    s = "results/crystal/alleles_genome20_s6003.csv"
    fact("reverse.working", sum(truth(r["culture_replicator"]) for r in table(s)), s, "LDDR scan: culture_replicator=True")
    s = "results/offset/CORE_FAMILIES.md"
    t = source_text(s)
    for k, pattern in [("core.all", r"cells scanned: (\d+)"), ("core.motif", r"or `1e NN ed b0`\): (\d+)")]:
        m = re.search(pattern,t)
        if m: fact(k, m[1], s, pattern, m[0])
    s = "results/offset/offset_cells.csv"
    a = [r for r in table(s) if val(r,"offset") > 0 and val(r,"L") % val(r,"offset") == 0]
    # Column names are taken from the archived table, not fitted to manuscript outcomes.
    ncol = "n_unexec" if "n_unexec" in a[0] else "sites_unexec"
    for r in a: r["offset_prediction"] = str(val(r,"offset")-4)
    pcol = "offset_prediction"
    fact("population.total", len(a), s, "offset_divides=True sampled motif cells")
    fact("population.equal", sum(val(r,ncol)==val(r,pcol) for r in a), s, f"dividing offsets: {ncol} = {pcol}")
    sitecol = "n_sites" if "n_sites" in a[0] else "sites"
    fact("population.extra", sum(val(r,sitecol)>val(r,pcol) for r in a), s, f"dividing offsets: {sitecol} > {pcol}")
    s = "results/crystal/offset_examples.csv"
    r = next(r for r in table(s) if r["seed"] == "5006")
    colfacts("example", s,r,["L","seed","parent_period","offset"],"stage K, seed 5006")
    s = "results/capacity/parents.csv"
    fact("C1.parents", len(table(s)), s,"archived parent row count")
    s = "results/capacity/edit_eligibility.csv"
    e = table(s)
    fact("C1.edits", len(e), s, "frozen displacement-setting rows (including original controls)")
    for c in ["predicted_free", "core_c"]:
        for fn in [min,max]:fact(f"C1.{c}.{fn.__name__}", fn(val(r,c) for r in e),s,f"{fn.__name__} {c} across edits")
    s = "results/capacity/serial.csv.gz"
    n=bad=0;lineages=set();gens=set()
    for r in stream(s):
        n+=1;lineages.add(r["lineage"]);gens.add(int(r["generation"]))
        bad+=not all(truth(r[c]) for c in ["exact","provenance_exact","core_exact"])
    for key,x,desc in [("C1.serial",n,"serial row count"),("C1.lineages",len(lineages),"distinct lineage indices per edit/direction"),("C1.generations",max(gens),"maximum serial generation"),("C1.serial_bad",bad,"serial rows failing exact/provenance/core")]:fact(key,x,s,desc)
    s = "results/capacity/capacity_cases.csv.gz"
    n=bad=0;partners=set();joint=Counter();single=Counter()
    for r in stream(s):
        n+=1;partners.add(int(r["partners"]));bad+=not all(r[c]==r["partners"] for c in ["exact_passed","provenance_passed","core_passed"])
        if r["kind"]=="joint":joint[(r["edit_id"],r["direction"])]+=1
        if r["kind"]=="single":single[(r["edit_id"],r["direction"],r["position"])]+=1
    for key,x,desc in [("C1.direct",n,"direct intervention row count (not encounters)"),("C1.bad",bad,"direct rows failing any partner on exact/provenance/core"),("C1.partners",min(partners),"partners per direct case"),("C1.joints",min(joint.values()),"joint cases per edit/direction"),("C1.alternatives",min(single.values()),"single alternatives per free coordinate")]:fact(key,x,s,desc)
    s="results/evolved6502/panel.csv";a=table(s)
    for key,x,desc in [("C2.roles",len(a),"sampling-role row count"),("C2.worlds",len({r['world'] for r in a}),"distinct evolved worlds"),("C2.tapes",len({r['tape'] for r in a}),"distinct tape values"),("C2.first",sum(r['role']=='first' for r in a),"first-detection roles"),("C2.random",sum(r['role']=='final_random' for r in a),"final random roles"),("C2.top",sum(r['role']=='final_top3' for r in a),"final top-three roles")]:fact(key,x,s,desc)
    s="results/evolved6502/eligibility.csv";a=[r for r in table(s) if truth(r["eligible"])]
    fact("C2.eligible",len(a),s,"eligible=1 roles")
    for c in ["core_c","predicted_free"]:
        for fn in [min,max]:fact(f"C2.{c}.{fn.__name__}",fn(val(r,c) for r in a),s,f"{fn.__name__} {c} among eligible roles")
    for label,s,fields in [("C2.direct","results/evolved6502/exact_cases.csv.gz",["exact_passed","provenance_passed"]),("C2.serial","results/evolved6502/exact_serial.csv.gz",["exact","provenance_exact"])]:
        n=bad=0
        for r in stream(s):
            n+=1
            bad+=not all((r[c]==r['partners'] if c.endswith('_passed') else truth(r[c])) for c in fields)
        fact(label,n,s,"intervention row count" if label.endswith('direct') else "serial observation row count")
        fact(label+"_bad",bad,s,"rows failing exact/provenance checks (full expected offspring, including held core)")
    s="results/census6502/table_closers_law.csv";a=table(s)
    # Frozen constructed controls: qualifying rows have reported payload equal to d-6.
    qualifying=[r for r in a if truth(r['hypotheses']) and truth(r['law holds'])]
    fact("controls6502",len(qualifying),s,"constructed control rows with payload carried = d−6")
    s="results/s6502/v2/arm_comparison.csv"
    r=next(r for r in table(s) if r['protocol']=='v2_frozen32' and r['arm']=='A')
    colfacts("S1",s,r,["worlds","with_rep","first_confined","first_loop","km_t_rep"],"protocol=v2_frozen32, arm=A")
    s="results/tier2/scored/E/offset_switch.csv"
    for tar in ['benign','lethal']:
        for start in ['mix50','R_into_T','T_into_R']:
            a=[r for r in table(s) if r['L']=='32' and r['mut']=='on' and r['tar']==tar and r['start']==start]
            m=median(val(r,'T_share_end') for r in a if r['T_share_end'] and r['T_share_end'].lower()!='nan')
            fact(f"competition.{tar}.{start}",m,s,f"core C1 L32 mutation-on {tar} {start}: median T_share_end")
            fact(f"competition.{tar}.{start}.worlds",len(a),s,f"core C1 L32 mutation-on {tar} {start}: independent-world count")
            fact(f"competition.{tar}.{start}.majority",sum(val(r,'T_share_end')>.5 for r in a if r['T_share_end'] and r['T_share_end'].lower()!='nan'),s,f"{tar} {start}: T_share_end > .5 worlds")
    s="manuscript/figures/out/v8/data/lethality_summary.csv"
    for cohort in ['C1','Historical']:
        r=next(r for r in table(s) if r['cohort']==cohort and val(r,'p')==1)
        colfacts('lethal.'+cohort,s,r,['T_majority','worlds'],f'{cohort} p=1')
    s="results/ladder/population_worlds.csv"
    for stage,L in [('G','16'),('G','20'),('G','50'),('I','16')]:
        a=[r for r in table(s) if r['stage']==stage and r['L']==L and r['threshold']=='any']
        for cls in ['open','regenerator','transmitter']:
            fact(f"ladder.{stage}.{L}.{cls}",sum(val(r,f'final_frac_{cls}_of_heritable')>.5 for r in a),s,f'stage={stage}, L={L}, threshold=any; final {cls} > 1/2 heritable cells')
        fact(f"ladder.{stage}.{L}.worlds",len(a),s,f'stage={stage}, L={L}, threshold=any world rows')
    s="results/crystal/first_cells.csv";a=[r for r in table(s) if r['dataset']=='Z80 I' and r['endpoint']=='first replicator']
    fact('firstI.worlds',len(a),s,'Z80 I first detected tapes')
    fact('firstI.inflow',max(val(r,'H_bits') for r in a),s,'maximum H_bits, Z80 I first tapes')
    fact('firstI.sites',max(val(r,'n_sites') for r in a),s,'maximum sampled n_sites, Z80 I first tapes')
    s="results/s8086/tables/worlds.csv";a=table(s)
    for key,x,meaning in [('S2.worlds',len(a),'independent-world rows'),('S2.detected',sum(bool(r['t_rep']) for r in a),'nonempty t_rep'),('S2.open',sum(r['first_fetch']=='open' for r in a),'first_fetch=open'),('S2.confined',sum(r['first_fetch']=='confined' for r in a),'first_fetch=confined'),('S2.rep_censored',sum(truth(r['rep_right_censored']) for r in a),'rep_right_censored=True'),('S2.closed_censored',sum(truth(r['closed_right_censored']) for r in a),'closed_right_censored=True'),('S2.occurrence',sum(truth(r['occurrence']) for r in a),'occurrence=True'),('S2.persistent',sum(truth(r['persistent']) for r in a),'persistent=True'),('S2.seed.min',min(val(r,'seed') for r in a),'minimum seed'),('S2.seed.max',max(val(r,'seed') for r in a),'maximum seed'),('S2.horizon',max(val(r,'censor_step') for r in a),'censor_step maximum')]:fact(key,x,s,meaning)
    s="results/s8086/tables/kaplan_meier.csv"
    for endpoint in ['rep','closed']:
        rows=[r for r in table(s) if r['endpoint']==endpoint and val(r,'survival')<=.5]
        fact('S2.km.'+endpoint,min(val(r,'step') for r in rows),s,f'endpoint={endpoint}; first KM survival <= 0.5 (world-level, censoring retained)')
    s="results/s8086/tables/law_denominators.csv"
    for family in ['L1','L2']:
        r=next(r for r in table(s) if r['family']==family and not truth(r['exclude_L1']))
        colfacts(family,s,r,['unique_candidates','culture_positive','eligible','unresolved_extractions','cap_excluded'],f'{family}, exclude_L1=False')
    s="results/s8086/tables/law_candidates.csv";a=[r for r in table(s) if r['family']=='L2']
    fact('L2.partner_reads',sum(r['status']=='excluded' and 'original_partner' in r['reason'] for r in a),s,'L2 exclusions for original-partner input')
    fact('L2.pop_es',sum(r['status']=='unresolved' and '07' in r['reason'] for r in a),s,'L2 unresolved opcode 07 / POP ES')
    s="results/s8086/exploratory/eligibility_counts.csv";a=table(s)
    for status in ['eligible','excluded','unresolved']:
        fact('posthoc.'+status,sum(val(r,'cases') for r in a if r['status']==status),s,'sum cases for '+status)
    s="results/s8086/exploratory/law_panel_predictions.csv";a=table(s)
    fact('posthoc.panel',len(a),s,'purposive selected genotype-world rows')
    fact('posthoc.worlds',len({r['world'] for r in a}),s,'distinct worlds in purposive panel')
    s="results/s8086/exploratory/law_assay_denominators.csv"
    fact('posthoc.encounters',sum(val(r,'encounters') for r in table(s)),s,'sum encounters across kind/challenge/generation rows')
    s="results/s8086/exploratory/law_assay_summary.csv";a=table(s)
    fact('posthoc.bad',sum(r['status']!='complete-zero-mismatch' for r in a),s,'panel cases not complete-zero-mismatch')
    for c in ['joint_assignments','serial_generations']:
        fact('posthoc.'+c,min(val(r,c) for r in a),s,'minimum '+c+' per selected genotype')
    s="results/s8086/exploratory/motif_scan.csv.gz"
    a=table(s)
    # Two scanned words share the same denominator; count B951 only.
    a=[r for r in a if r['word'].lower()=='b951']
    for source,field,key in [('scheduled_top3','tapes','motif.top'),('checkpoint','cell_weight','motif.cells')]:
        fact(key,sum(val(r,field) for r in a if r['source']==source),s,f'word=b951, source={source}: sum {field}')
    fact('motif.hits',sum(val(r,'partial4_tapes') for r in a),s,'b951 alternating runs >=4, summed partial4_tapes')
    s="results/s8086/exploratory/first_family.csv";a=table(s)
    fact('family.worlds',len(a),s,'5FA4xy first-family world rows')
    r=a[0];geom=json.loads(r['geometry'])
    for key,x,meaning in [('family.d',json.loads(r['constant_displacement'])[0],'first-family constant displacement'),('family.core',len(geom['C']),'constrained footprint size'),('family.free',len(geom['F']),'free coordinate count')]:fact(key,x,s,meaning)
    s="results/s8086/tables/constructed_controls.csv";a=table(s)
    fact('REP.controls',len(a),s,'constructed control rows')
    fact('REP.encounters',sum(val(r,'encounters') for r in a),s,'sum constructed-control encounters')
    fact('REP.bad',sum(r['status']!='complete-zero-mismatch' for r in a),s,'controls not complete-zero-mismatch')
    s="results/s8086/provenance/final_spend.json";o=json.loads(source_text(s))
    for key,x in o.items():
        if isinstance(x,(int,float)) and not isinstance(x,bool):fact('spend.'+key,x,s,'JSON field '+key)
    s="manuscript/figures/out/v8/data/route_by_machine.csv"
    for r in table(s):colfacts('route.'+r['machine'],s,r,['worlds','open','confined','censored'],r['machine'])
    s="manuscript/figures/out/v8/data/literal_matrix.csv"
    for r in table(s):colfacts('width.'+r['machine'],s,r,['width','confined','detected'],r['machine'])
    s="manuscript/figures/out/v8/data/first_copier_tiles.csv";a=table(s)
    fact('tiles.worlds',len({r['seed'] for r in a}),s,'distinct world seeds')
    fact('tiles.bytes',len({r['position'] for r in a}),s,'positions per tape')
    for fn in [min,max]:fact('tiles.pos.'+fn.__name__,fn(val(r,'position') for r in a),s,fn.__name__+' plotted position')
    for name in ['memory-fed one-pass copier','confined copy loop']:
        fact('tiles.'+name,len({r['seed'] for r in a if r['mechanism']==name}),s,'distinct seeds with mechanism='+name)


@dataclass
class Claim:
    offset: int
    end: int
    token: str
    value: float
    paragraph: str
    pstart: int
    section: str
    source: str
    source_paths: list[str]
    all_paths: list[str]
    kind: str = 'number'
    fact_key: str = ''


def mask_span(chars,a,b):
    chars[a:b] = ' ' * (b-a)


def extract(text):
    claims=[]; excluded=Counter(); section='Preamble'; figure=''
    for pm in re.finditer(r'\S[\s\S]*?(?=\n\s*\n|\Z)',text):
        p=pm[0]
        if p.startswith('## '):section=p[3:].strip();continue
        if p.startswith('**Figure '):
            figure=re.search(r'Figure (\d+)',p)[1];continue
        if p.startswith(('#','|','*Manuscript','**Genuinely pending')):continue
        chars=list(p)
        brackets=list(BRACKET.finditer(p))
        sources=[m for m in brackets if paths(m[1])]
        for m in brackets:mask_span(chars,m.start(),m.end())
        for m in re.finditer(r'`[^`]*`',p):
            code=m[0][1:-1]
            # Byte literals used in edits are claims; instruction encodings/names/hashes are identifiers.
            if re.fullmatch(r'[0-9A-Fa-f]{2}',code):
                mask_span(chars,m.start(),m.end())
                if re.search(r'changed|hexadecimal|byte|Editing|rims|status|FLAGS',p,re.I):
                    chars[m.start()+1:m.end()-1]=code
            else:mask_span(chars,m.start(),m.end())
        for m in CPU.finditer(p):mask_span(chars,m.start(),m.end());excluded['processor identifiers']+=1
        for m in FIG.finditer(p):mask_span(chars,m.start(),m.end());excluded['figure/display identifiers']+=1
        for m in re.finditer(r'\b(?:[A-Z]+\d+(?:-\d+)?|gen\d+)(?:\([a-z]\))?',p):
            if m[0].startswith('L') and m[0][1:].isdigit() and int(m[0][1:]) >= 8:continue
            mask_span(chars,m.start(),m.end());excluded['protocol/metric identifiers']+=1
        for m in re.finditer(r'one another|(?:any\s+)?one\s+member|an internal one|one route|none of (?:the measurements|these statements)|almost none|none enlarges|One trace|zero-page|zero-EA|zero-SP|one-pass|NaN is unassessable, not zero|not zero heritable variation',p,re.I):mask_span(chars,m.start(),m.end())
        # Include numeric coefficients, L16/L32 shorthand, millions and byte literals.
        raw=''.join(chars)
        tokens=list(NUM.finditer(raw))
        tokens += list(re.finditer(r'(?<=L)\d+(?!\w)|(?<!\w)\d+(?:\.\d+)?(?=[LrDw](?:\b|[+−-]))',raw))
        for m in sorted(tokens,key=lambda m:m.start()):
            token=m[0];w=token.lower()
            if w in WORDS:n=WORDS[w]
            else:
                n=float(token.replace('$','').replace(',','').replace('%','').replace('M',''))
                if token.endswith('%'):n/=100
                if token.endswith('M'):n*=1_000_000
                if p[max(0,m.start()-1):m.start()]=='`':n=int(token,16)
                elif re.search(r'hexadecimal\s+[0-9a-fA-F]{2}(?:\s+to\s+|\s+at.*?and\s+)?$|hexadecimal\s+$',p[max(0,m.start()-100):m.start()]):n=int(token,16)
                elif re.search(r'rims mark [0-9A-Fa-f]{2} and $',p[max(0,m.start()-50):m.start()]):n=int(token,16)
                elif re.search(r'million\b',p[m.end():m.end()+9]):n*=1_000_000
            # Nearest following bracket in current sentence, else nearest source in paragraph.
            after=[b for b in sources if b.start()>=m.end() and not re.search(r'\.(?:\s|$)',p[m.end():b.start()])]
            b=after[0] if after else (min(sources,key=lambda b:min(abs(b.start()-m.end()),abs(m.start()-b.end()))) if sources else None)
            source=b[1] if b else ''
            claims.append(Claim(pm.start()+m.start(),pm.start()+m.end(),token,n,p,pm.start(),section+(f' / Figure {figure}' if section=='Figure legends' else ''),source,paths(source),[q for b in sources for q in paths(b[1])]))
    return claims,excluded


def extra_facts():
    def report(key, name, pattern, meaning, group=1):
        m=re.search(pattern,source_text(name),re.I|re.S)
        if m:
            fact(key,float(m[group].replace(',','')),name,meaning,squash(m[0])[:400])
    s='results/s8086/provenance/final_spend.json';o=json.loads(source_text(s))
    fact('spend.bound',o['conservative_bound_usd'],s,'JSON conservative_bound_usd; retained liability bound, not metered spend')
    fact('spend.cap',o['cap_usd'],s,'JSON cap_usd; approved amended cap')
    s='results/s8086/exploratory/motif_scan.csv.gz';a=[r for r in table(s) if r['word']=='b951']
    fact('motif.top',sum(r['source']=='scheduled_top3' for r in a),s,'b951 scheduled_top3 record rows; each record scans three tapes (35,100 tape observations)')
    fact('motif.cells',sum(val(r,'cell_weight') for r in a if r['source']=='snapshot'),s,'b951 snapshot rows: sum cell_weight, including initialization')
    s='results/s8086/exploratory/eligibility.csv';a=[r for r in table(s) if r['status']=='eligible']
    for d in [4,16]:fact(f'posthoc.d{d}',sum(val(r,'d')==d for r in a),s,f'eligible genotype-world candidates with d={d}')
    s='results/s8086/tables/census_07a5.csv';a=table(s)
    fact('witness.contexts',len(a),s,'07a5 audited L/budget/partner contexts')
    s='results/census8086/table_hits.csv';hits=[r for r in table(s) if r['word']=='07a5']
    assert all(val(r,'entered_encounters')==17 for r in hits)
    fact('witness.open_random',sum(val(r,'entered_encounters')-1 for r in hits),s,'07a5 full-encounter partner entry: all 17 contexts per setting; subtract one zero partner per setting, retaining 64 random contexts. Pre-write original-partner reads are a different metric.')
    s='results/census8086/table_hits.csv'
    for L in [16,32]:
        r=next(r for r in table(s) if r['L']==str(L) and r['budget']=='128' and r['word'].lower()=='b951')
        fact(f'pusher8086.L{L}',r['random_exact'],s,f'word=b951, L={L}, budget=128; random_exact')
    s='results/ladder/population_worlds.csv';a=[r for r in table(s) if r['stage']=='G' and r['threshold']=='any']
    fact('order.before',sum(r['R_T_order']=='R before T' for r in a),s,'benign Stage G, threshold=any; R before T world rows')
    fact('order.both',sum(r['R_T_order'] in ['R before T','T before R','same sample'] for r in a),s,'benign Stage G, both classes detected')
    s='results/s8086/exploratory/first_family.csv';a=table(s);g=json.loads(a[0]['geometry'])
    fact('family.seed_length',len(g['R']),s,'protected original-parent roots, |R|')
    s='results/offset/CORE_FAMILIES.md'
    report('core.motif',s,r'four-byte core.*?\):\s*(\d+)','four-byte motif count in confined heritable scan')
    s='results/census8086/DESCRIPTIVE_CORRECTION_2026-10-12.md'
    report('census.settings',s,r'All ([\d,]+) candidate settings','audited census setting count')
    report('census.encounters',s,r'settings / ([\d,]+) encounters','audited census encounter count')
    report('census.changed',s,r'Exactly \*\*([\d,]+) labels change','changed causal labels, outputs unchanged')
    s='results/census8086/table_main.csv';a=table(s)
    for k in [1,2]:fact(f'census.words{k}',max(val(r,'words') for r in a if r['k']==str(k)),s,f'complete period-k={k} word denominator per L/budget arm')
    s='results/census8086/table_law.csv';a=table(s)
    fact('census.law',sum(val(r,'partner_assays')/(val(r,'single_cases')+val(r,'joint_cases')) for r in a),s,'sum partner_assays/(single_cases+joint_cases) per geometry/budget; 20 geometries × 17 partners')
    s='results/serial_retention/sr_mutants.csv.gz'
    a=[r for r in stream(s) if r['key']=='ldir32' and r['pos']=='0' and r['val'] in ['8','96'] and r['gen']=='8']
    fact('old_edit.gen',min(val(r,'gen') for r in a if val(r,'retained')>.5),s,'ldir32 pos=0 val=8/96: retained markers at g8')
    # Protocol constants are taken from cited generated metadata rather than generic numeral search.
    s='manuscript/figures/out/v8/data/branching_endpoints.csv';a=table(s)
    r=a[0]
    for c in ['values','reps','k','cap','gens','ctrl']:
        fact('branch.'+c,r[c],s,'branching uniform endpoint metadata; '+c)
    s='manuscript/figures/out/v8/data/serial_summary.csv';a=table(s)
    fact('serial.designs',len(a),s,'displayed archived design rows')
    for r in a:
        fact('serial.'+r['key']+'.L',r['L'],s,'key='+r['key']+'; tape length')
    s='manuscript/figures/out/v8/data/erasure_manifest.json';o=json.loads(source_text(s))
    for c in ['position','marker','L','steps_per_encounter']:fact('erasure.'+c,o[c],s,'JSON '+c)
    s='manuscript/figures/out/v8/data/erasure_ordered.csv';a=table(s)
    hit=next(r for r in a if r['encounter']=='1' and r['kind']=='write' and r['address']=='8' and r['step']=='6')
    for c in ['step','address','byte','ancestral_source']:fact('erasure.'+c,hit[c],s,'encounter=1, step=6, write address=8; '+c)
    fact('erasure.encounters',len({r['encounter'] for r in a}),s,'distinct encounter numbers in ordered trace')
    read=next(r for r in a if r['encounter']=='1' and r['step']=='6' and r['kind']=='read')
    fact('erasure.source_read',read['address'],s,'encounter=1, step=6: read address preceding circled write; source coordinate is separately 0',str(read))
    fact('erasure.marker_remaining',hit['marker_sources_remaining'],s,'marker ancestry remaining after step=6 overwrite')
    s='manuscript/figures/out/v8/data/geometry_units.csv';a=table(s)
    for c in ['L','c']:fact('geometry.'+c,a[0][c],s,'schematic '+c)
    fact('geometry.dmin',min(val(r,'d') for r in a),s,'small displacement in schematic')
    fact('geometry.dmax',max(val(r,'d') for r in a),s,'full-parent displacement in schematic')
    units=len({r['source'] for r in a if r['d']=='12' and r['state']=='payload'})
    fact('geometry.units',units,s,'independent payload source units at d=12, excluding their repeated descendants in the other tape')
    fact('geometry.free',units*2,s,'two bytes per display unit × independent payload source units at d=12')
    fact('geometry.bytes_per_unit',val(a[1],'position')-val(a[0],'position'),s,'adjacent display-unit position difference, in bytes')
    s='manuscript/figures/out/v8/data/emergence_worlds.csv';a=table(s)
    counts=Counter(r['tape_len'] for r in a)
    fact('emergence.worlds',min(counts.values()),s,'independent-world count per plotted length')
    fact('emergence.horizon',max(val(r,'horizon') for r in a),s,'Stage E observation horizon')
    s='manuscript/figures/out/v8/data/emergence_curves.csv'
    fact('emergence.axis',min(val(r,'step') for r in table(s)),s,'minimum generated curve x coordinate')
    s='manuscript/figures/out/v8/data/arrival_worlds.csv';a=table(s)
    fact('arrival.worlds',20,s,'per-machine independent worlds; count validated in all three groups')
    assert set(Counter(r['machine'] for r in a).values())=={20}
    for c in ['L','budget']:fact('arrival.'+c,a[0][c],s,'L16 machine comparison '+c)
    for machine in ['Z80 G','6502 A','8086 S2']:
        fact('arrival.'+machine+'.horizon',max(val(r,'horizon') for r in a if r['machine']==machine),s,machine+' horizon')
    s='manuscript/figures/out/v8/data/competition_endpoints.csv';a=table(s)
    for cohort in ['C1','Historical']:
        fact('competition.'+cohort+'.logs',sum(r['cohort']==cohort for r in a),s,'independent logged worlds in '+cohort)
    fact('competition.endpoint',max(val(r,'steps') for r in a),s,'L32 mutation-on endpoint steps')
    fact('competition.chem_worlds',sum(r['cohort']=='C1' and r['tar']=='benign' for r in a),s,'C1 benign world rows; lethal and historical also 20')
    a=[r for r in a if r['cohort']=='C1' and r['tar']=='benign']
    fact('competition.undefined',sum(not r['T_share_end'] or r['T_share_end'].lower()=='nan' for r in a),s,'C1 benign undefined T/(R+T) worlds')
    fact('competition.defined',sum(bool(r['T_share_end']) and r['T_share_end'].lower()!='nan' for r in a),s,'C1 benign defined T/(R+T) worlds')
    s='manuscript/figures/out/v8/data/lineage_strips.csv';a=table(s)
    for edit in ['ldir32_v04','ldir32_v60']:
        r=next(r for r in a if r['edit_id']==edit)
        for c in ['generation','lineage','displacement','original_byte','edited_byte']:fact(edit+'.'+c,r[c],s,'edit_id='+edit+'; '+c)
        for state in ['core','erased','payload','transmitted']:
            fact(edit+'.'+state,sum(r['state']==state for r in a if r['edit_id']==edit),s,'edit='+edit+', state='+state+' coordinate count')
    s='algocell_exp/machines/worlds_8086.json';o=json.loads(source_text(s));r=o['i8086_pair_maps_L16']
    fact('reset.zero',r['world']['init']['set']['ax'],s,'L16 init.set.ax (general registers/segments reset zero)')
    fact('reset.flags',r['world']['init']['set']['flags'],s,'L16 init.set.flags')
    fact('reset.SP',1,s,'world note: SP=1 via firstPushAt:lastByte on ring P=32/64',r['note'])


# Semantic bindings for the frozen draft. Ordinal positions are guarded by paragraph
# anchors. Unbound numbers remain conservative contextual checks, never auto-passes.
BINDINGS = {
 17: ('Darwinian evolution requires', 'pusher16.gens pusher16.mutation_types_carried pusher16.n_mutants S2.confined S2.detected posthoc.encounters'),
 62: ('We validated the processor', 'claims.same claims.total components.changed components.total'),
 79: ('In all 80 historical', 'route.Z80.worlds'),
 86: ('To ask what such', 'pusher16.k pusher16.ctrl_alive_n pusher16.ctrl pusher16.L pusher16.gens pusher16.mut_alive pusher16.gens pusher16.mutation_types_carried pusher16.n_mutants'),
 95: ('Three confined Z80', '- - - - genome20_s6003.sites genome20_s6003.gens - pusher64.L pusher64.mutation_types_carried pusher64.n_mutants'),
 120: ('The Z80 soups evolve', 'C1.core_c.min C1.core_c.min core.motif core.all'),
 127: ('In an exploratory regrouping', 'population.total population.equal C1.core_c.min population.extra C1.core_c.min'),
 136: ('The law also corrects', '- example.parent_period - example.offset'),
 149: ('If the displacement is encoded', '- - - old_edit.gen C1.parents C1.edits C1.lineages C1.generations C1.generations C1.direct C1.predicted_free.min C1.predicted_free.max'),
 153: ('An exploratory scan', 'dial.total dial.working dial.inherited - dial.eligible dial.effective_d.min dial.effective_d.max dial.predicted_payload.min dial.predicted_payload.max reverse.working'),
 167: ('Without mutation', '- competition.benign.mix50 - competition.lethal.mix50 - competition.lethal.mix50.majority -'),
 173: ('Both invasion directions', 'competition.benign.R_into_T competition.benign.T_into_R competition.lethal.R_into_T competition.lethal.T_into_R competition.benign.R_into_T.worlds lethal.C1.T_majority lethal.C1.worlds lethal.Historical.T_majority lethal.Historical.worlds'),
 177: ('The lethality dial also', 'ladder.G.16.regenerator ladder.G.16.worlds - ladder.G.20.regenerator ladder.G.20.worlds - - order.before order.both firstI.worlds firstI.inflow firstI.sites ladder.I.16.transmitter ladder.I.16.worlds'),
 196: ('The comparison across processors', 'width.6502.width - - - - - - S1.first_confined S1.km_t_rep S1.first_loop'),
 198: ('The copying law nevertheless', 'controls6502 C2.roles C2.worlds C2.eligible C2.core_c.min C2.core_c.max C2.predicted_free.min C2.predicted_free.max C2.direct C2.serial C1.generations'),
 200: ('The coherent-fetch', 'S2.detected S2.worlds S2.open S2.detected S2.confined S2.km.rep S2.rep_censored'),
 202: ('The registered law test', 'L2.eligible L2.unresolved_extractions L2.pop_es'),
 204: ('Exploratory, post hoc follow-ups', '- motif.top - motif.cells'),
 206: ('The dominant first-copier', 'family.worlds S2.detected family.d family.seed_length family.core family.free - family.free'),
 208: ('Extending the dependency', 'posthoc.eligible posthoc.d16 - posthoc.d4 - posthoc.panel posthoc.worlds posthoc.encounters'),
 231: ('**b,** Kaplan-Meier cumulative emergence', '- - emergence.worlds emergence.horizon - emergence.axis'),
 247: ('**d,** Five separate validation', 'validation.units validation0.unchanged validation0.total validation0.changed - validation1.unchanged validation1.total ring_confidence ring_rate_bound validation2.unchanged validation2.total validation3.unchanged validation3.total validation4.unchanged validation4.total -'),
 269: ('**a,** Generation-eight', 'branch.gens branch.values branch.reps branch.k branch.ctrl - pusher16.ctrl_alive_n pusher16.ctrl pusher16.mut_alive pusher16.retained pusher16.sites pusher16.L ret16.ctrl_alive_n ret16.ctrl ret16.mut_alive ret16.retained ret16.sites ret16.L ldir32.ctrl_alive_n ldir32.ctrl ldir32.mut_alive ldir32.retained ldir32.sites ldir32.L genome20_s6003.ctrl_alive_n genome20_s6003.ctrl genome20_s6003.mut_alive genome20_s6003.retained genome20_s6003.sites genome20_s6003.L - - -'),
 307: ('**d,** One exploratory', '- erasure.position - erasure.encounters erasure.steps_per_encounter erasure.step erasure.ancestral_source erasure.source_read erasure.byte erasure.address - erasure.marker_remaining erasure.marker_remaining - - - - - - erasure.ancestral_source erasure.address erasure.steps_per_encounter -'),
 324: ('**a,** Abstract forward-overlap', '- geometry.L geometry.c - geometry.dmin - geometry.dmax - -'),
 337: ('**b,** Prediction versus', '- - population.total - C1.edits C1.parents C2.eligible controls6502 - - match_gate L1.eligible L1.eligible'),
 358: ('**c,** Two parallel', '- ldir32_v04.generation - ldir32_v04.original_byte ldir32_v04.displacement ldir32_v04.core ldir32_v04.erased - ldir32_v04.original_byte ldir32_v60.edited_byte ldir32_v60.displacement - ldir32_v04.lineage C1.alternatives C1.partners C1.lineages C1.generations'),
 370: ('**d,** Exploratory sweep', 'dial.total - - C1.core_c.min - dial.nonworking dial.excluded dial.eligible C1.core_c.min dial.predicted_payload.min dial.predicted_payload.max - - - - C1.core_c.min - - gen2_minimum - - match_gate'),
 389: ('**a,** Mutation-on', '- - competition.benign.mix50.worlds - competition.benign.R_into_T.worlds - competition.benign.T_into_R.worlds competition.C1.logs competition.Historical.logs - -'),
 406: ('**b,** World-level', 'competition.endpoint competition.chem_worlds - competition.undefined competition.defined -'),
 432: ('**a,** First detected', 'route.Z80.worlds - - - - - S1.worlds - S2.worlds - S2.open S2.confined S2.rep_censored S2.horizon - - - - - - - - route.Z80.worlds S1.first_loop S1.worlds S1.first_confined S1.worlds S2.detected - S2.open S2.detected - S2.confined -'),
 454: ('**b,** Kaplan-Meier cumulative detection', '- - arrival.L arrival.budget arrival.worlds - - arrival.Z80 G.horizon arrival.6502 A.horizon arrival.8086 S2.horizon S2.rep_censored S2.horizon - S2.rep_censored'),
 467: ('**c,** Compact literal-channel', 'width.Z80.width width.6502.width - route.Z80.open route.Z80.worlds S1.first_confined S1.worlds S2.confined S2.detected S2.open width.8086.width width.6502.width'),
 480: ('**d,** All ten', 'tiles.worlds - tiles.bytes - - - tiles.pos.min tiles.pos.max tiles.worlds tiles.memory-fed one-pass copier tiles.confined copy loop tiles.worlds S2.detected family.core'),
 503: ('S2 used the Zilion', 'reset.zero reset.SP - -'),
 511: ('S2’s sole population', 'S2.worlds arrival.L S2.seed.min S2.seed.max S2.horizon - - - - - - - S2.horizon'),
 517: ('Culture scores', 'match_gate - gen2_minimum -'),
 521: ('Historical serial assays', '- - - branch.gens branch.k branch.cap branch.reps branch.values branch.ctrl - -'),
 547: ('Displacement experiment C1', 'C1.parents C1.edits - -'),
 549: ('Forward passages followed', 'C1.lineages C1.generations C1.serial C1.generations C1.alternatives C1.joints C1.partners C1.direct'),
 553: ('C2 shares registration', '- C2.roles -'),
 555: ('The complete panel', 'C2.first C2.random C2.top - C2.worlds C2.tapes C2.eligible - C1.alternatives C1.joints - - C1.generations C2.direct C2.serial - - - - -'),
 565: ('The validated 8086 census', 'census.words1 - census.words2 - - - - - - - census.settings census.encounters census.changed'),
 569: ('Repeated classifier corrections', 'witness.contexts witness.open_random - - pusher8086.L16 - - pusher8086.L32 - - -'),
 573: ('R0 requires', '- S2.worlds - - - - - - S2.detected S2.open S2.confined S2.occurrence S2.worlds S2.persistent S2.worlds S2.km.closed S2.closed_censored'),
 575: ('Law candidates comprise', '- - - L1.unique_candidates L1.culture_positive L2.partner_reads L2.unresolved_extractions L2.pop_es'),
 579: ('Constructed controls are', 'census.law - - - - - - - - REP.controls REP.bad REP.encounters -'),
 585: ('The final reconciliation', 'spend.metered_usd spend.metered_usd spend.bound spend.cap'),
 591: ('A separate extended', 'L1.culture_positive posthoc.eligible posthoc.excluded posthoc.unresolved posthoc.panel posthoc.worlds posthoc.joint_assignments - - posthoc.serial_generations posthoc.encounters - -'),
 595: ('Validation distinguishes', 'validation0.unchanged validation0.total validation0.changed -'),
 597: ('I1 compared historical', 'C1.flagged C1.ringtotal C2.flagged C2.ringtotal c1.identical assays.total claims.same claims.total components.changed components.total - -'),
}
# Fact keys containing spaces are kept out of whitespace-separated bindings.
BINDINGS[454]=(BINDINGS[454][0],['-','-','arrival.L','arrival.budget','arrival.worlds','-','-','arrival.Z80 G.horizon','arrival.6502 A.horizon','arrival.8086 S2.horizon','S2.rep_censored','S2.horizon','-','S2.rep_censored'])
BINDINGS[480]=(BINDINGS[480][0],['tiles.worlds','-','tiles.bytes','-','-','-','tiles.pos.min','tiles.pos.max','tiles.worlds','tiles.memory-fed one-pass copier','tiles.confined copy loop','tiles.worlds','S2.detected','family.core'])


def bind_claims(claims,text):
    issues=[]
    for start in sorted({c.pstart for c in claims}):
        cs=[c for c in claims if c.pstart==start]
        line=text.count('\n',0,start)+1
        if line not in BINDINGS:continue
        anchor,keys=BINDINGS[line]
        keys=keys.split() if isinstance(keys,str) else keys
        if not cs[0].paragraph.startswith(anchor) or len(keys)!=len(cs):
            issues.append(f'L{line}: semantic binding disabled: anchor or token count changed ({len(keys)} expected, {len(cs)} extracted).')
            continue
        for c,k in zip(cs,keys):
            if k!='-':
                if k not in FACTS:issues.append(f'L{line}: missing fact {k}')
                else:c.fact_key=k
    return issues

STOP = set('the a an and or of in to at with for from by on is are was were be as this that these those its their all every only not no exact sampled registered historical corrected core figure panel source report results table bytes byte cases count rows value values generation generations tested test against per first final whole'.split())


def terms(s):
    s=BRACKET.sub(' ',CPU.sub(' ',s))
    return {w[:7] for w in re.findall(r'[a-zA-Z]{3,}',s.lower()) if w not in STOP and w not in WORDS}


def displayed_equal(c, f):
    """Return exact agreement or explicit nearest rounding (half up)."""
    actual=Decimal(str(f.value)); claimed=Decimal(str(c.value))
    if actual==claimed:return True,'exact; no rounding'
    token=c.token.replace('$','').replace(',','')
    if c.token.lower() in WORDS or c.token.startswith('0') and len(c.token)==2:
        return False,''
    scale=Decimal(100) if token.endswith('%') else Decimal(1)
    numeric=token.rstrip('%M')
    places=len(numeric.split('.')[1]) if '.' in numeric else 0
    if actual*scale==(actual*scale).to_integral_value() and scale==1:return False,''
    rounded=(actual*scale).quantize(Decimal(1).scaleb(-places),rounding=ROUND_HALF_UP)
    if rounded==claimed*scale:
        return True,f"rounded {fmt(f.value)} to {places} decimal place(s)" + (' in percent units' if scale==100 else '') + ' (nearest, half up)'
    return False,''


@lru_cache(None)
def prose_numbers(name):
    """Context candidates only from prose reports; CSV equality is never evidence."""
    t=source_text(name)
    out=[]
    for m in NUM.finditer(t):
        tok=m[0].lower()
        if tok in WORDS:n=float(WORDS[tok])
        else:
            n=float(tok.replace('$','').replace(',','').replace('%','').replace('m',''))
            if tok.endswith('%'):n/=100
            if tok.endswith('m'):n*=1000000
        # A narrow window prevents an unrelated number elsewhere in a long report from passing.
        context=squash(t[max(0,m.start()-90):m.end()+90])
        out.append((n,context,t.count('\n',0,m.start())+1))
    return out


def contextual(c):
    off=c.offset-c.pstart
    near=squash(c.paragraph[max(0,off-90):off+len(c.token)+90])
    want=terms(near)
    if len(want)<3:return None
    best=None
    for name in c.source_paths:
        if not name.endswith('.md') or not resolve(name).is_file():continue
        for n,context,line in prose_numbers(name):
            got=terms(context);common=len(want & got)
            score=common/max(len(want),len(got),1)
            if common<3 or score<.5:continue
            f=Fact(n,name,'same local quantitative context in cited report',f'L{line}: {context}')
            eq,rounding=displayed_equal(c,f)
            if eq and (best is None or score>best[0]):best=(score,f,rounding)
    return best[1:] if best else None


def location(text,offset):
    line=text.count('\n',0,offset)+1
    col=offset-text.rfind('\n',0,offset)
    return f'experiments/manuscript/MAIN_v7.md:{line}:{col}'


def audit_numbers(claims,text):
    out=[]
    for i,c in enumerate(claims,1):
        row={'id':f'N{i:04d}','location':location(text,c.offset),'section':c.section,'value':c.token,'normalized_value':fmt(c.value),'source':squash(c.source),'status':'','correct_value':'','source_value':'','rounding':'','meaning':'','evidence_source':'','evidence':'','context':squash(c.paragraph[max(0,c.offset-c.pstart-90):c.end-c.pstart+110])}
        if not c.source_paths:
            row.update(status='no source bracket',meaning='No generated-table/report source bracket in this paragraph.')
        elif all(not resolve(s).exists() for s in c.source_paths):
            row.update(status='source missing',meaning='All paths in attached bracket are missing.')
        else:
            f=FACTS.get(c.fact_key)
            # Fact may be in a cited underlying table or in the cited report's sibling
            # table. Always disclose that evidence path; never substitute it silently.
            if f:
                ok,rounding=displayed_equal(c,f)
                row.update(status='verified' if ok else 'mismatch',correct_value='' if ok else fmt(f.value),source_value=fmt(f.value),rounding=rounding,meaning=f.meaning,evidence_source=f.source,evidence=f.evidence or f.meaning)
            else:
                match=contextual(c)
                if match:
                    f,rounding=match
                    row.update(status='verified',source_value=fmt(f.value),rounding=rounding,meaning=f.meaning,evidence_source=f.source,evidence=f.evidence)
                else:
                    row.update(status='number not found in source',meaning='No occurrence with established same meaning in cited source. Literal equality alone does not verify; may be a protocol/formula/display constant requiring a more specific citation.')
        assert row['status'] in STATUSES
        out.append(row)
    return out


def audit_paths(text):
    records=[]
    for m in BRACKET.finditer(text):
        if m[1].startswith('fill:'):continue
        for name in paths(m[1]):records.append((m.start(),name,'source bracket'))
    for m in re.finditer(r'\[[^\]]*\]\(([^)]*)\)',text):
        name=m[1].strip('<>').split('#',1)[0]
        if name and not re.match(r'https?://|mailto:',name):records.append((m.start(),name,'link'))
    # Also check file paths and rooted-directory paths in the supporting map/prose.
    for m in re.finditer(r'`([^`]+)`',text):
        if '/' in m[1] and not re.search(r'\s',m[1]):
            for name in paths(m[1]):records.append((m.start(),name,'prose/map path'))
    result=[]
    for off,name,kind in records:
        p=(HERE/name if kind=='link' and not Path(name).is_absolute() else resolve(name))
        exists=p.exists()
        outside=not p.resolve().is_relative_to(REPO.resolve())
        result.append({'location':location(text,off),'path':name,'kind':kind,'resolved':str(p),'status':'exists' if exists else 'missing','portability':'non-portable absolute path outside repository' if Path(name).is_absolute() and outside else 'repository-relative' if not Path(name).is_absolute() else 'absolute path inside repository','suggested_path':{'out/v6/data/erasure_states.csv':'manuscript/figures/out/v6/data/erasure_states.csv','tables/kaplan_meier.csv':'results/s8086/tables/kaplan_meier.csv'}.get(name,'') if not exists else ''})
        # Open every cited file; CSVs are opened without retaining million-row tables.
        if exists and p.is_file():
            OPENED.add(p)
            if name.endswith(('.csv','.csv.gz')):
                next(stream(name),None)
            elif kind!='link':source_text(name)
    return result


def audit_figures(text):
    import pymupdf
    legends=(V8/'LEGENDS_V8.md').read_text()
    panels={}
    assets=[]
    for m in re.finditer(r'\*\*Figure (\d+)\.[\s\S]*?(?=\*\*Figure |\*\*Method notes|\Z)',legends):
        n=int(m[1]);ps=set(re.findall(r'\*\*([a-z]),\*\*',m[0]))
        pdf=V8/f'fig{n}.pdf';printed=set()
        if pdf.exists():
            with pymupdf.open(pdf) as d:
                for page in d:
                    for b in page.get_text('dict')['blocks']:
                        for line in b.get('lines',[]):
                            for s in line['spans']:
                                # Panel labels are standalone bold 8pt spans. Small
                                # payload letters inside schematics do not count.
                                if re.fullmatch('[a-z]',s['text'].strip()) and s['size']>=7.9 and (s['flags'] & 16):printed.add(s['text'].strip())
        panels[n]=(ps,printed)
        assets.append({'figure':f'Figure {n}','legend_panels':','.join(sorted(ps)),'pdf_panels':','.join(sorted(printed)),'pdf':str(pdf.relative_to(REPO)),'png_exists':(V8/f'fig{n}.png').exists(),'status':'verified' if ps==printed and pdf.exists() else 'missing asset or panel'})
    refs=[]
    for m in FIG.finditer(text):
        # Figure heading numbers define figures, rather than citing them.
        if text[max(0,m.start()-2):m.start()]=='**':continue
        extended=bool(m[1]);n=int(m[2]);letters=[]
        if m[3]:
            raw=re.sub(r'\s','',m[3]);letters=re.findall('[a-z]',raw)
            if re.search('[–-]',raw) and len(letters)==2:letters=[chr(x) for x in range(ord(letters[0]),ord(letters[1])+1)]
        if extended:status='missing Extended Data display';detail='Supporting-material map is not a rendered figure/table.'
        elif n not in panels:status='missing figure';detail='No v8 legend/asset.'
        else:
            legend,printed=panels[n];missing=set(letters)-(legend & printed)
            status='verified' if not missing else 'missing panel'
            detail='Both v8 legend and PDF panel labels exist.' if not missing else 'Missing '+','.join(sorted(missing))
        refs.append({'location':location(text,m.start()),'reference':m[0],'status':status,'detail':detail})
    ed=[{'display':f'Extended Data {m[1]}','location':location(text,m.start()),'status':'mapped only; no numbered rendered Extended Data figure/table','detail':'Source families exist but a map does not supply a submission display.'} for m in re.finditer(r'^\| Extended Data (\d+) \|',text,re.M)]
    return assets,refs,ed


def audit_citations(text):
    refhead=re.search(r'^##?\s+(?:References|Bibliography)\s*$',text,re.M|re.I)
    body=text[:refhead.start()] if refhead else text
    entries=[]
    if refhead:
        end=re.search(r'^##? ',text[refhead.end():],re.M)
        tail=text[refhead.end():refhead.end()+end.start()] if end else text[refhead.end():]
        entries=[squash(p) for p in re.split(r'\n\s*\n',tail) if p.strip()]
    cites={}
    for b in BRACKET.finditer(body):
        if paths(b[1]) or b[1].startswith('fill:'):continue
        for part in b[1].split(';'):
            part=squash(part)
            m=re.fullmatch(r'(.+?)\s+((?:18|19|20)\d{2}[a-z]?)',part)
            if m:cites.setdefault(part,[]).append(location(text,b.start()))
    result=[];used=set()
    for cite,locs in sorted(cites.items()):
        author,year=cite.rsplit(' ',1);surname=author.split(',')[0].split(' & ')[0].split(' et al.')[0]
        matches=[i for i,e in enumerate(entries) if surname.casefold() in e.casefold() and year in e]
        used.update(matches)
        result.append({'citation':cite,'occurrences':len(locs),'locations':'; '.join(locs),'status':'matched' if len(matches)==1 else 'ambiguous' if matches else 'missing reference-list entry','missing_details':'Full author/editor list; title; journal/book/preprint venue or publisher; year in reference entry; volume and pages/article number where applicable; DOI/URL/accession where available.' if not matches else ''})
    uncited=[e for i,e in enumerate(entries) if i not in used]
    incomplete=[]
    for e in entries:
        missing=[]
        if not re.search(r'(?:18|19|20)\d{2}',e):missing.append('year')
        if not re.search(r'https?://|doi\b|10\.\d{4,9}/',e,re.I):missing.append('DOI/URL (if available)')
        if not re.search(r'\d+\s*[–-]\s*\d+|\b[A-Za-z]?\d{4,}\b',e):missing.append('page range/article identifier (if applicable)')
        if len(e.split())<8:missing.extend(['complete author list','title','venue/publisher'])
        if missing:incomplete.append({'entry':e,'missing_details':'; '.join(missing)})
    return bool(refhead),entries,result,uncited,incomplete


def word_count(s,remove_brackets=True):
    if remove_brackets:s=BRACKET.sub('',s)
    s=re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',s)
    s=re.sub(r'^#+ .*$', '',s,flags=re.M)
    s=re.sub(r'^\*\*Figure \d+\.[^\n]*', '',s,flags=re.M)
    s=re.sub(r'\*\*[a-z],\*\*','',s)
    return len(re.findall(r"\b[\w]+(?:[’'−–-][\w]+)*(?:[.,][\d]+)*\b",s))


def counts(text):
    def section(name):
        m=re.search(r'^## '+re.escape(name)+r'\s*\n([\s\S]*?)(?=^## |\Z)',text,re.M)
        return m[1] if m else ''
    out=[]
    for name in ['Abstract','Main text','Methods']:
        s=section(name);out.append({'section':name,'words_excluding_brackets':word_count(s),'words_including_brackets':word_count(s,False)})
    s=section('Figure legends')
    for m in re.finditer(r'\*\*Figure (\d+)\.[\s\S]*?(?=\*\*Figure |\Z)',s):
        out.append({'section':'Figure '+m[1]+' legend','words_excluding_brackets':word_count(m[0]),'words_including_brackets':word_count(m[0],False)})
    return out


def mdtable(rows,cols):
    def escape(x):return squash(str(x)).replace('|','\\|').replace('`','')
    return ['| '+' | '.join(cols)+' |','| '+' | '.join('---' for _ in cols)+' |']+['| '+' | '.join(escape(r.get(c,'')) for c in cols)+' |' for r in rows]


def write_report(text,numbers,paths_result,figs,refs,ed,citations,wcounts,bindings,excluded):
    counts_status=Counter(r['status'] for r in numbers)
    has_refs,entries,cites,uncited,incomplete=citations
    lines=['# Mechanical audit of MAIN_v7','',
      f'Draft: `experiments/manuscript/MAIN_v7.md` · SHA-256 `{digest(DRAFT)}`.',
      f'Audit script: `experiments/manuscript/audit_v7.py`. CSV: `experiments/manuscript/audit_v7.csv`.',
      '',f"Extracted **{len(numbers)} numeric occurrences**: "+'; '.join(f'{s}: {counts_status[s]}' for s in sorted(STATUSES))+'.',
      f'Found **{len(cites)} distinct author–year citations** ({sum(r["occurrences"] for r in cites)} occurrences), **{len(entries)} reference entries**, '
      f'**{sum(r["status"]=="missing reference-list entry" for r in cites)} missing entries** and **{len(uncited)} uncited entries**.',
      f'Checked **{len(paths_result)} path occurrences** ({len({r["resolved"] for r in paths_result})} unique resolved paths); '
      f'**{sum(r["status"]=="missing" for r in paths_result)} missing occurrences**. '
      f'**{len({r["resolved"] for r in paths_result if r["portability"].startswith("non-portable")})} unique absolute external paths** are non-portable. '
      f'Required source-bracket/link path checks: **{sum(r["kind"] in ["source bracket","link"] and r["status"]=="missing" for r in paths_result)} missing**; remaining missing paths are optional prose shorthands.',
      f'Checked **{len(refs)} figure references**; **{sum(r["status"]!="verified" for r in refs)} unresolved**. All numbered Extended Data map entries are assessed separately below.',
      '', '## Scope and verification rules','',
      'Numbers include digit and word cardinals, zero/none, half, percentages, currency, range endpoints, ratio operands, generation/step counts, lengths, seeds, numeric equation coefficients and isolated hexadecimal byte edits. Each repeated occurrence is a separate CSV row. Processor names, protocol/claim IDs, hashes, bibliographic years, figure numbers, numeric characters inside instruction encodings/path names and nonquantitative idioms are excluded. Figure numbers and citation years have their own inventories. Compound “one half” is one fraction. Version identifiers are not scientific counts.',
      'Nearest source means the next source bracket within the current sentence; otherwise the closest source bracket within the same paragraph. The exact bracket is retained in the CSV. Sources resolve from experiments/, as the draft specifies. Additional supporting/map code paths are checked as well as all bracket and local-link paths. Missing files are distinguished from files that exist but do not establish the stated number.',
      'Named-field checks filter the stated cohort, arm, genotype, generation and endpoint before aggregating. Gzip tables are read directly. C1/C2 intervention rows are counted separately from encounter counts; serial observations retain their generation denominator. Kaplan–Meier medians use the first generated survival coordinate at or below 0.5. A cited report’s underlying table may supply stronger evidence; its exact path and calculation are disclosed in evidence_source/meaning. Contextual checks of cited prose require at least three shared distinctive terms in a narrow window and report the excerpt. A bare occurrence of the numeral never passes. “Number not found in source” includes a number present literally whose meaning is not established; these rows require citation or manual semantic resolution and are not assertions of numerical falsity.',
      'No simulations, paid compute, network lookup, source edits or source regeneration are performed. This audit verifies archived evidence and does not independently establish theorem correctness, statistical independence, eligibility validity, or bibliographic accuracy outside the repository.',
      '', '## Numeric mismatches','']
    mismatch=[r for r in numbers if r['status']=='mismatch']
    lines+=mdtable(mismatch,['id','location','value','correct_value','evidence_source','meaning']) if mismatch else ['No established numeric mismatch. Unresolved evidence is listed separately; it is not counted as verified.']
    lines+=['','## Rounding used','']
    rounded=[r for r in numbers if r['rounding'] and not r['rounding'].startswith('exact')]
    lines+=mdtable(rounded,['id','location','value','source_value','rounding','meaning'])
    lines+=['','## Numbers requiring source resolution','']
    lines+=mdtable([r for r in numbers if r['status'] not in ['verified','mismatch']],['id','location','value','source','status','context'])
    lines+=['','## Paths','']
    unique=[]
    for p in sorted({r['resolved'] for r in paths_result}):
        group=[r for r in paths_result if r['resolved']==p];r=dict(group[0]);r['occurrences']=len(group);r['locations']='; '.join(x['location'] for x in group);unique.append(r)
    lines+=mdtable(unique,['path','resolved','kind','status','portability','suggested_path','occurrences','locations'])
    lines+=['','## Figure and panel cross-references','',
       'Panel checks use the exported PDF’s standalone bold panel-label spans (8 pt or larger), excluding small payload letters within schematics, and the panel headings in figures/out/v8/LEGENDS_V8.md. PNG existence is checked. This is a label/asset check, not a visual-design review.']
    lines+=['']+mdtable(figs,['figure','legend_panels','pdf_panels','pdf','png_exists','status'])
    lines+=['']+mdtable(refs,['location','reference','status','detail'])
    lines+=['','## Extended Data map','',
       'The draft explicitly calls these a map, not already-rendered displays. No “Extended Data Fig./Table N” in-text callouts were found unless listed above. Map entries are nevertheless submission gaps: neither a source directory nor an unnumbered legacy display supplies the named numbered figure/table.']
    lines+=['']+mdtable(ed,['display','location','status','detail'])
    lines+=['','## Citation and bibliography audit','',
       'No References/Bibliography section exists in MAIN_v7.md.' if not has_refs else 'Reference-list entries were matched by first author and year.',
       'Bibliographic checks are local. Missing reference entries require the full bibliographic record; none is inferred from author/year alone.']
    lines+=['']+mdtable(cites,['citation','occurrences','status','missing_details','locations'])
    lines+=['','Uncited entries: '+ ('none (reference list is empty).' if not uncited else '; '.join(uncited)),
       'Existing entries with missing details: '+ ('none to inspect (reference list is empty).' if not entries else str(len(incomplete))) ]
    if incomplete:lines+=['']+mdtable(incomplete,['entry','missing_details'])
    lines+=['','## Word counts','',
       'Counts exclude section/title/figure headings and panel-label markup. The primary column removes every square-bracket citation/source/editorial annotation. A word is a Unicode alphanumeric token, allowing internal apostrophes, hyphens/dashes and decimal/grouping separators; inline code and formula tokens count. The secondary column includes bracket contents for transparent comparison. Main text runs from Main text through its final subsection, excluding legends, Methods, supporting-material map and availability.']
    lines+=['']+mdtable(wcounts,['section','words_excluding_brackets','words_including_brackets'])
    lines+=['','## Binding checks and reproducibility','']
    lines+=bindings or ['All manuscript-specific semantic binding anchors and number counts matched.']
    lines+=['','Excluded identifier categories: '+json.dumps(dict(excluded),sort_keys=True)+'.',
       '', '```sh','/Users/neo/repos/algocell/experiments/.venv/bin/python experiments/manuscript/audit_v7.py','```',
       '', 'The CSV contains every extracted numeric occurrence, including verified rows, source values, correct values for mismatches, rounding, exact evidence paths, meaning and local context. Exit status is 0 for a completed audit with findings; --strict returns 1 when findings remain. Outputs are deterministic for the same draft and source bytes.',
       '', '## Evidence file fingerprints','',
       'Only files actually opened by the audit are fingerprinted below. Files outside the repository retain their absolute location. Directory paths are existence checks.']
    fingerprints=[]
    for p in sorted(OPENED):
        if p.is_file():fingerprints.append({'path':str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p),'sha256':digest(p)})
    lines+=['']+mdtable(fingerprints,['path','sha256'])
    (HERE/'AUDIT_v7.md').write_text('\n'.join(lines)+'\n')


def protocol_facts():
    def from_text(key,name,pattern,meaning,group=1):
        m=re.search(pattern,source_text(name),re.I|re.S)
        if not m:return
        tok=m[group].lower();number=WORDS.get(tok)
        if number is None:number=float(tok.replace(',',''))
        fact(key,number,name,meaning,squash(m[0])[:450])
    s='results/branching/BRANCHING.md'
    from_text('branch.half',s,r'at least (half) of the identifiable values','site scoring: at least half of identifiable alternative values')
    from_text('branch.excess',s,r'background by at least (0\.5)','site-scoring excess over unmutated control carriage')
    from_text('branch.window',s,r'allele window is the (5)-byte window','centred mutant marker window width')
    from_text('branch.budget',s,r'G = 8;.*?; (128) instructions','branching assay execution budget')
    s='results/evolved6502/RETENTION.md'
    for key,pat,meaning in [('C2.alternatives',r'All (255) alternatives per predicted coordinate','all alternatives per free/erased coordinate'),('C2.joints',r'(64) joint assignments per eligible tape','joint assignments per eligible tape'),('C2.partners',r'zero\+(16) partners','uniform partner count, plus zero'),('C2.generations',r'serial observations through g(8)','maximum serial generation'),('C2.original_roles',r'original (714)-role ledger','original frozen role ledger')]:from_text(key,s,pat,meaning)
    s='results/s8086/prelaunch/freeze.json';o=json.loads(source_text(s));r=o['registration']
    fact('schedule.records',len(r['sample_steps']),s,'registration.sample_steps count per world (initialization included)')
    for i,x in enumerate(r['snapshot_steps']):fact('schedule.checkpoint'+str(i),x,s,'registration.snapshot_steps['+str(i)+']')
    for field in ['final_cells','cases_per_family_cap']:fact('schedule.'+field,r[field],s,'registration.'+field)
    fact('operational.timeout',o['world_server_timeout_s'],s,'world_server_timeout_s after operational amendment')
    s='manuscript/figures/out/v8/data/world_trajectory.csv';r=table(s)[0]
    for c in ['seed','cells','L']:fact('world.'+c,r[c],s,'illustrative archived world metadata '+c)
    s='manuscript/figures/out/v8/data/competition_trajectories.csv';r=table(s)[0]
    fact('competition.cells',r['cells'],s,'population denominator for absolute motif prevalence')
    s='manuscript/figures/out/v8/data/serial_summary.csv';a=table(s)
    for key in ['pusher16','pusher50','ret16','ldir32','genome20_s6003','genome20_s6004']:
        r=next(r for r in a if r['key']==key)
        fact('serial.'+key+'.alternatives',val(r,'n_mutants')/val(r,'L'),s,'key='+key+': n_mutants/L; alternatives per site')
    s='results/branching/branching_summary.csv';a=[r for r in table(s) if r['arm']=='U' and r['gen']=='8']
    keys=['ret16','ldir32','genome20_s6003']
    fact('branch.confined_designs',len([r for r in a if r['key'] in keys]),s,'three named confined design rows: ret16, ldir32, genome20_s6003')
    fact('branch.regenerators',len([r for r in a if r['key'] in ['ret16','ldir32']]),s,'named return/block-copy regenerators')
    fact('branch.return_design',sum(r['key']=='ret16' for r in a),s,'one named return regenerator')
    fact('branch.block_design',sum(r['key']=='ldir32' for r in a),s,'one named block-copy regenerator')
    s='results/crystal/cell_exceptions.csv';a=[r for r in table(s) if r['stage']=='K' and val(r,'L')==32 and val(r,'seed')==5006 and val(r,'period')==4 and r['offset'] and val(r,'offset')==8]
    fact('example.cells',len(a),s,'K L32 seed5006 period4 displacement8 exception rows')
    fact('example.sites',a[0]['n_sites'],s,'K L32 seed5006 period4 displacement8: total n_sites=6; n_unexec=4. Correct total is 6, or qualify manuscript 4 as unexecuted payload positions.',str({k:a[0][k] for k in ['source_row','snapshot','n_sites','n_unexec','period','offset','tape']}))
    s='results/serial_retention/sr_mutants.csv.gz'
    for value in [8,96]:
        rows=[r for r in stream(s) if r['key']=='ldir32' and r['pos']=='0' and r['val']==str(value) and r['gen']=='8']
        assert rows and val(rows[0],'retained')>.5
        fact('old_edit.value'+str(value),rows[0]['val'],s,'ldir32 pos=0, gen=8; retained marker value '+str(value))
    s='results/serial_retention/sr_summary.csv';r=next(r for r in table(s) if r['key']=='ldir32')
    generations=sorted(int(re.search(r'g(\d+)$',k)[1]) for k in r if re.fullmatch(r'sites_g\d+',k))
    for i,g in enumerate(generations):fact('serial.sample'+str(i),g,s,'serial scored generation encoded in sites_g'+str(g)+' column')
    fact('old_edit.original',int(r['tape'].split()[0],16),s,'ldir32 archived displacement byte at position 0')
    s='results/census6502/table_main.csv';a=[r for r in table(s) if r['L']=='16']
    for col in ['L','k','budget']:
        for fn in [min,max]:fact('census6502.'+col+'.'+fn.__name__,fn(val(r,col) for r in a),s,fn.__name__+' '+col+' in L16 census')
    fact('census6502.zero',sum(val(r,'exact, zero partner') for r in a),s,'exact zero-partner writer count summed across L16 census arms')
    s='results/census8086/table_main.csv';a=table(s)
    for col in ['L','budget']:
        for fn in [min,max]:fact('census8086.'+col+'.'+fn.__name__,fn(val(r,col) for r in a),s,fn.__name__+' '+col+' across census arms')
    s='results/census8086/table_hits.csv';r=next(r for r in table(s) if r['word']=='b951' and r['budget']=='128' and r['L']=='16')
    fact('census8086.random_partners',len(r['original_partner_data_reads'].split(';'))-1,s,'full frozen partner panel access fields minus one zero partner')
    s='results/census8086/table_law.csv';a=table(s)
    for L in [16,32]:
        values=sorted({val(r,'free_payload') for r in a if r['L']==str(L)})
        for i,x in enumerate(values):fact(f'REP.L{L}.free{i}',x,s,f'L={L}, eligible constructed d settings; sorted free_payload[{i}]')
    s='results/s8086/tables/worlds.csv';a=table(s)
    fact('family.trace_partners',next(iter(json.loads(next(r for r in a if r['first_tape'].startswith('5fa4'))['first_mechanism_counts']).values())),s,'all first-family encounter mechanism_counts=16; denominator of frozen descriptive trace panel')
    s='results/literal_channel/census_audit.csv';a=[r for r in table(s) if r['machine']=='Z80' and r['source'].endswith('census_p2.csv')]
    fact('literal.Z80.all',len(a),s,'nontrivial period-two Z80 census writer rows, unique source census')
    fact('literal.Z80.literal',sum(truth(r['literal_channel']) for r in a),s,'period-two Z80 literal_channel=True')
    s='results/census8086/LAW_CONTRACT.md'
    for k,p,meaning in [('REP.core',r'Both have (six) core bytes','MOV-DI/DEC-CX/REP family fixed core size'),('REP.17',r'two budgets, (17) partners','frozen census partner panel size'),('REP.values',r'every one of (256) values','all-value single-coordinate design'),('REP.joint1',r'(256)\s+counterbalanced','counterbalanced joint assignments'),('REP.joint2',r'(256) fixed random joint assignments','seeded joint assignments'),('REP.gen',r'serially passed to g=(8)','planned serial generations')]:from_text(k,s,p,meaning)
    s='results/s8086/S2.md'
    for k,p,meaning in [('route.required',r'(?:R0.*?D\s*≥\s*|R0.*?requirement\s*)(10)','R0 minimum detected world count'),('route.share',r'(0\.8)D','R1 required open proportion multiplier'),('route.kill',r'(0\.25)D','route kill required confined proportion multiplier')]:from_text(k,s,p,meaning)
    # These constants are from the dated registration, checked by full semantic phrases.
    s='REVISION_PREREG.md'
    from_text('route.required',s,r'R0 remains D≥(10)/20','R0 minimum detection denominator')
    from_text('route.share',s,r'R1 remains O≥ceil\((0\.8)D\)','R1 open-first cutoff multiplier')
    from_text('route.kill',s,r'kill rule remains at least ceil\((0\.25)D\)','route kill cutoff multiplier')
    s='results/evolved6502/descriptive_panel.csv';a=table(s)
    for role in ['first','final_random']:
        n=Counter(r['arm'] for r in a if r['role']==role)
        assert len(n)==4
        fact('descriptive.'+role,min(n.values()),s,'descriptive scan '+role+' roles per arm')
    s='results/evolved6502/RETENTION.md'
    for k,p,meaning in [('descriptive.reps',r'four mutant lineages','mutant lineages per descriptive alternative'),('descriptive.controls',r'(64) controls, 32 alternatives','unmutated descriptive controls'),('descriptive.values',r'(32) alternatives/coordinate','descriptive alternative values per coordinate')]:
        if k.endswith('reps'):
            from_text(k,s,r'(four) mutant lineages',meaning)
        else:from_text(k,s,p,meaning)
    # Reports explicitly separate unexpected divergences from expected raw mismatches.
    s='algocell_exp/machines/PROVENANCE_8086.md'
    from_text('validation.unexpected',s,r'(?:unexpected[^\n]{0,30}[:=]\s*)(0)','unexpected architectural vector divergences')


def supplemental_claims(text,claims):
    """Add hexadecimal marker/reset literals and explicit negative count claims."""
    supplement=[]
    paragraphs={c.pstart:c.paragraph for c in claims}
    for start,p in paragraphs.items():
        sources=[b for b in BRACKET.finditer(p) if paths(b[1])]
        for pat,group,kind in [(r'FLAGS is `(F002)`',1,'hex'),(r'and (FF) at the lower right',1,'hex'),(r'over marker (A7)',1,'hex'),(r'\b(no) (?:mismatch(?:es)?|evolved candidate was certified|eligible evolved member|free payload|evolved law intervention occurred|paid compute|nonzero exact zero-partner self-writer|cap excluded a case)',1,'implicit_zero')]:
            for m in re.finditer(pat,p,re.I):
                a,b=m.span(group)
                if any(c.offset<=start+a<c.end for c in claims):continue
                bracket=min(sources,key=lambda x:min(abs(x.start()-b),abs(a-x.end()))) if sources else None
                section=next(c.section for c in claims if c.pstart==start)
                token=m[group];number=int(token,16) if kind=='hex' else 0
                c=Claim(start+a,start+b,token,number,p,start,section,bracket[1] if bracket else '',paths(bracket[1]) if bracket else [],[n for x in sources for n in paths(x[1])],kind=kind)
                if kind=='hex':
                    if token=='F002':c.fact_key='reset.flags'
                    if token=='A7':c.fact_key='erasure.marker'
                    if token=='FF':
                        s='results/crystal/alleles_ldir32.csv';fact('dial.maxbyte',max(val(r,'value') for r in table(s)),s,'maximum tested byte value, hex FF');c.fact_key='dial.maxbyte'
                elif 'paid compute' in m[0]:pass
                elif 'nonzero exact' in m[0]:c.fact_key='census6502.zero'
                elif 'cap excluded' in m[0]:c.fact_key='L1.cap_excluded'
                elif 'free payload' in m[0]:c.fact_key='C1.predicted_free.min' if 'C1' in p else 'family.free'
                elif 'evolved candidate' in m[0] or 'eligible evolved member' in m[0] or 'evolved law intervention' in m[0]:c.fact_key='L1.eligible'
                elif 'results/evolved6502/' in p:c.fact_key='C2.serial_bad'
                elif 'results/s8086/exploratory/' in p:c.fact_key='posthoc.bad'
                elif 'results/capacity/' in p:c.fact_key='C1.bad'
                elif 'eligible evolved member' in m[0] or 'evolved candidate' in m[0] or 'evolved law intervention' in m[0]:c.fact_key='L1.eligible'
                if kind=='implicit_zero' and 'paid compute' in m[0]:c.fact_key=''
                supplement.append(c)
    for start,p in paragraphs.items():
        for code in re.finditer(r'`([^`]+)`',p):
            if not re.search(r'[()/=]\s*[-−]?\d|\d\s*[-−+*/]',code[1]) or paths(code[1]):continue
            for m in re.finditer(r'\d+(?:\.\d+)?',code[1]):
                a=start+code.start(1)+m.start();b=start+code.start(1)+m.end()
                if any(c.offset<=a<c.end for c in claims+supplement):continue
                base=next(c for c in claims if c.pstart==start)
                c=Claim(a,b,m[0],float(m[0]),p,start,base.section,base.source,base.source_paths,base.all_paths,kind='inline formula')
                supplement.append(c)
    return sorted(claims+supplement,key=lambda c:c.offset)


MORE_BINDINGS={
 50:{0:'encounter.tapes',1:'encounter.tapes'},
 95:{0:'branch.confined_designs',1:'branch.regenerators',2:'branch.return_design',3:'branch.block_design',6:'branch.half'},
 136:{0:'example.cells',1:'example.parent_period',2:'example.sites',3:'example.offset'},
 149:{0:'old_edit.original',1:'old_edit.value8',2:'old_edit.value96'},
 153:{},
 177:{2:'population.G16.L',5:'population.G20.L',6:'population.G50.L'},
 196:{2:'census6502.k.min',3:'census6502.k.max',4:'census6502.L.min',5:'census6502.budget.min',6:'census6502.budget.max'},
 198:{10:'C2.generations'},
 200:{},
 239:{1:'world.seed',2:'world.cells',3:'world.L',5:'world.L'},
 247:{4:'validation.unexpected'},
 269:{30:'branch.half',31:'branch.window',32:'branch.half'},
 286:{1:'serial.sample0',2:'serial.sample1',3:'serial.sample2',4:'serial.sample3',6:'serial.pusher16.alternatives'},
 295:{7:'serial.genome20_s6004.L',10:'serial.pusher50.L',11:'serial.pusher50.alternatives',12:'serial.pusher16.alternatives',14:'branch.half'},
 307:{13:'branch.window'},
 324:{0:'geometry.bytes_per_unit',7:'geometry.units',8:'geometry.free'},
 358:{2:'world.L32',11:'ldir32_v60.transmitted'},
 370:{13:'old_edit.value96',16:'dial.effective_d.max',19:'C1.uniform_partners'},
 389:{0:'world.L32',1:'competition.cells'},
 432:{28:'route.required',31:'route.required_open',33:'route.kill_count'},
 480:{3:'family.trace_partners',4:'family.trace_partners'},
 511:{7:'schedule.records',8:'schedule.checkpoint0',9:'schedule.checkpoint1',10:'schedule.checkpoint2',11:'schedule.checkpoint3'},
 517:{},
 521:{0:'branch.window',9:'branch.half',10:'branch.excess'},
 527:{0:'world.L32',1:'competition.endpoint',2:'competition.benign.mix50.worlds',3:'competition.benign.R_into_T.worlds'},
 547:{3:'C1.uniform_partners'},
 549:{},
 553:{0:'C2.original_roles'},
 555:{8:'C2.alternatives',9:'C2.joints',11:'C2.partners',12:'C2.generations',15:'descriptive.first',16:'descriptive.final_random',17:'descriptive.values',18:'descriptive.reps',19:'descriptive.controls'},
 561:{5:'literal.Z80.literal',6:'literal.Z80.all'},
 565:{4:'census8086.L.min',5:'census8086.L.max',6:'census8086.budget.min',7:'census8086.budget.max',9:'census8086.random_partners'},
 569:{2:'census8086.budget.min',5:'census8086.random_partners',6:'census8086.L.min',8:'census8086.random_partners',9:'census8086.L.max'},
 573:{0:'route.required',2:'route.share',3:'route.kill'},
 575:{1:'schedule.final_cells',2:'REP.core'},
 577:{2:'schedule.cases_per_family_cap',3:'REP.values',4:'REP.joint1',5:'REP.joint2',6:'REP.17',7:'REP.gen'},
 579:{1:'REP.core',2:'REP.L16.free0',3:'REP.L16.free1',4:'census8086.L.min',5:'REP.L32.free0',6:'REP.L32.free1',7:'REP.L32.free2',8:'census8086.L.max',12:'census8086.L.max'},
 583:{0:'S2.seed.min',1:'operational.timeout'},
 591:{8:'REP.17'},
 595:{3:'validation.unexpected'},
}


def more_bindings(claims,text):
    from math import ceil
    s='algocell_exp/machines/worlds_8086.json';o=json.loads(source_text(s))
    fact('encounter.tapes',len(o['i8086_pair_maps_L16']['world']['memory']['segments']),s,'focal A and partner B tape segments per encounter')
    s='results/capacity/partners.csv';fact('C1.uniform_partners',len({r['partner'] for r in table(s) if r['partner']!='0'}),s,'16 uniform frozen partner identities per L, excluding zero partner identity 0')
    s='results/ladder/population_worlds.csv'
    for L in [16,20,50]:
        r=next(r for r in table(s) if r['stage']=='G' and r['L']==str(L));fact(f'population.G{L}.L',r['L'],s,f'Stage G L={L} world metadata')
    s='manuscript/figures/out/v8/data/lineage_strips.csv';a=[r for r in table(s) if r['edit_id']=='ldir32_v60']
    fact('world.L32',len(a),s,'32 physical offspring coordinates in L32 intervention strip')
    s='REVISION_PREREG.md'
    fact('route.required_open',ceil(FACTS['route.share'].value*FACTS['S2.detected'].value),s,'ceil(0.8 × 14 detected worlds); required open first copiers')
    fact('route.kill_count',ceil(FACTS['route.kill'].value*FACTS['S2.detected'].value),s,'ceil(0.25 × 14 detected worlds); confined-first kill threshold')
    # Original base claims only, preserving supplemental-zero/hex offsets.
    for start in sorted({c.pstart for c in claims}):
        cs=[c for c in claims if c.pstart==start and c.kind=='number']
        if not cs:continue
        line=text.count('\n',0,start)+1
        for index,key in MORE_BINDINGS.get(line,{}).items():
            if index<len(cs) and key in FACTS:cs[index].fact_key=key


def law_facts_and_bindings(claims,text):
    """Check theorem/definition constants against the specific cited statements."""
    def capture(key,name,pattern,meaning,group=1):
        m=re.search(pattern,source_text(name),re.S|re.I)
        if not m:return
        token=m[group].lower()
        x=WORDS[token] if token in WORDS else float(token.replace(',',''))
        fact(key,x,name,meaning,squash(m[0])[:450])
    s='manuscript/MAIN_nature.md'
    for k,p,meaning in [
        ('theorem.Lmax',r'Z80 soup at tape length L ≤ (64)','Proposition 8 maximum L'),
        ('theorem.ring',r'pair memory is a ring of (2)L bytes','Proposition 8 ring length coefficient'),
        ('theorem.core',r'satisfy (4) ≤ d ≤ L','Proposition 8 minimum protected-prefix/core displacement'),
        ('theorem.entry',r'XX \(a\) passes control to address (1)','Proposition 8 required successor instruction address'),
        ('theorem.HL',r'leaves HL ≡ (0)','Proposition 8 reset/source-base congruence'),
        ('theorem.Dscale',r'and (256)·D ≡ 0','Proposition 8 high-byte register scale'),
        ('theorem.Dzero',r'256·D ≡ (0)','Proposition 8 destination-base congruence'),
        ('theorem.BCzero',r'BC either (0) or at least','Proposition 8 BC zero counter convention'),
        ('theorem.moves',r'BC either 0 or at least (126)','Proposition 8 minimum count / available block repetitions'),
        ('theorem.immediate',r'block move has (127) repetitions','Proposition 8 immediate variant repetition count'),
        ('entropy.outputs',r'plug-in entropy of the offspring over (256) uniformly random partners','Lemma 7 finite partner/output sample'),
        ('entropy.bits',r'log₂ 256 = (8) bits','Lemma 7 sample entropy ceiling'),
        ('entropy.zero',r'λ\(x\) = (0): perfect fidelity','Lemma 7 zero inflow condition'),
        ('class.R',r'a regenerator \(at most (two) transmissible sites\)','population classification maximum regenerator sites'),
        ('class.Imin',r'intermediate \((three) or four\)','population classification intermediate minimum sites'),
        ('class.Imax',r'intermediate \(three or (four)\)','population classification intermediate maximum sites'),
        ('class.T',r'transmitter \((five) or more\)','population classification minimum transmitter sites'),
        ('serial.lineages',r'Each mutant founded (16) lineages','historical serial mutant lineages per alternative'),
        ('serial.controls',r'unmutated tape (256)','historical serial control starts'),
        ('serial.half',r'for at least (half) of its values','historical serial site threshold'),
    ]:capture(k,s,p,meaning)
    s='results/literal_channel/CHECK.md'
    capture('literal.opcode',s,r'load has (one) opcode byte','repeated-pair grammar opcode-byte width')
    capture('literal.register',s,r'opcodes sharing (one)\s+register','width-one mixture shared-register grammar')
    capture('literal.width1',s,r'Width (1) has a period-16 mixture','finite width-one witness grammar')
    capture('literal.width3',s,r'Width (3) has mixture writers','width-three counterexample grammar')
    capture('literal.bound2',s,r'=2w\+(2)-\\gcd','corrected Fine–Wilf period bound constant')
    if 'literal.bound2' in FACTS:
        fact('literal.coeff2',2,s,'coefficient 2 multiplying w in corrected N≥2w+2−gcd(2,w)',r'N\ge (w+2)+w-\gcd(w+2,w)=2w+2-\gcd(2,w).')
        fact('literal.gcd2',2,s,'constant 2 in gcd(2,w) in corrected Fine–Wilf bound',r'N\ge (w+2)+w-\gcd(w+2,w)=2w+2-\gcd(2,w).')
    s='results/s8086/S2.md'
    for k,p,meaning in [('route.trace_min',r'B fetch in ≥(8)/16 traces','S2 full-encounter open-first threshold'),('route.traces',r'B fetch in ≥8/(16) traces','S2 route trace denominator'),('route.confined_zero',r'confinement requires (zero) B fetch','S2 confinement requirement')]:capture(k,s,p,meaning)
    s='manuscript/figures/out/v8/data/8086_validation.csv'
    if resolve(s).exists():
        r=table(s)[0]
        for field in ['unexpected','unexpected_arch','fail_unexpected']:
            if field in r:fact('validation.unexpected',r[field],s,'architecture vector '+field)
    # Plot sources preserve the same 8088 pass/expected-divergence accounting.
    candidates=[V8/'data/8086_validation.csv',HERE/'figures/out/v6/data/8086_validation.csv']
    for p in candidates:
        if p.exists():
            name=str(p.relative_to(EXP));r=table(name)[0]
            for field in r:
                if 'unexpected' in field and re.fullmatch(r'\d+(?:\.\d+)?',r[field]):fact('validation.unexpected',r[field],name,'hardware-vector '+field)
    changes={
      196:{0:'literal.width1'},
      247:{4:'validation.unexpected'},
      286:{7:'serial.lineages'},
      295:{13:'serial.lineages'},
      432:{14:'route.trace_min',15:'route.traces',16:'route.confined_zero',17:'route.traces'},
      499:{},
      517:{1:'culture.generation',3:'culture.partners'},
      519:{2:'class.R',3:'class.Imin',4:'class.Imax',5:'class.T'},
      521:{1:'serial.lineages',2:'serial.controls'},
      523:{0:'entropy.zero',1:'entropy.zero',2:'entropy.zero',3:'entropy.outputs',4:'entropy.bits'},
      533:{1:'theorem.ring',2:'dial.total',3:'entropy.bits'},
      535:{0:'theorem.Lmax',1:'theorem.ring',2:'theorem.core',3:'theorem.entry',4:'theorem.HL',5:'theorem.Dscale',6:'theorem.Dzero',7:'theorem.ring',8:'theorem.BCzero',9:'theorem.moves',10:'theorem.moves',11:'theorem.immediate'},
      559:{0:'literal.opcode',1:'literal.register',2:'literal.bound2',3:'literal.coeff2',4:'literal.bound2',5:'literal.gcd2'},
      561:{0:'literal.width1',2:'literal.width3',3:'literal.width1'},
      573:{4:'route.trace_min',5:'route.traces',7:'route.traces'},
      577:{1:'REP.core'},
      595:{3:'validation.unexpected'},
    }
    for start in sorted({c.pstart for c in claims}):
        cs=[c for c in claims if c.pstart==start]
        line=text.count('\n',0,start)+1
        for i,key in changes.get(line,{}).items():
            if i<len(cs) and key in FACTS:cs[i].fact_key=key


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--strict',action='store_true',help='nonzero exit if any audit finding remains');args=ap.parse_args()
    before=digest(DRAFT);text=DRAFT.read_text()
    fact_issues=[]
    for build in [build_facts,extra_facts,protocol_facts]:
        try:build()
        except (FileNotFoundError,KeyError,ValueError,StopIteration) as exc:
            fact_issues.append(f'{build.__name__}: source absent or schema changed: {type(exc).__name__}: {exc}')
    claims,excluded=extract(text);bindings=bind_claims(claims,text)
    for bind in [more_bindings,law_facts_and_bindings]:
        try:bind(claims,text)
        except (FileNotFoundError,KeyError,ValueError,StopIteration) as exc:
            fact_issues.append(f'{bind.__name__}: source absent or schema changed: {type(exc).__name__}: {exc}')
    claims=supplemental_claims(text,claims);bindings+=fact_issues
    numbers=audit_numbers(claims,text);path_result=audit_paths(text)
    figs,refs,ed=audit_figures(text);cites=audit_citations(text);wcounts=counts(text)
    with (HERE/'audit_v7.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(numbers[0]),lineterminator='\n');w.writeheader();w.writerows(numbers)
    write_report(text,numbers,path_result,figs,refs,ed,cites,wcounts,bindings,excluded)
    assert before==digest(DRAFT),'Draft changed during audit; outputs are not valid for a stable draft.'
    print(json.dumps({'numbers':len(numbers),'statuses':dict(Counter(r['status'] for r in numbers)),'missing_path_occurrences':sum(r['status']=='missing' for r in path_result),'citation_keys':len(cites[2]),'reference_entries':len(cites[1]),'binding_issues':bindings,'word_counts':wcounts},indent=2))
    if args.strict and (bindings or any(r['status']!='verified' for r in numbers) or any(r['status']=='missing' or r['portability'].startswith('non-portable') for r in path_result) or any(r['status']!='verified' for r in refs) or ed or any(r['status']!='matched' for r in cites[2]) or cites[3] or cites[4]):return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Local read-only audit of manuscript v8 and SI; no network or execution of experiments.

Run: experiments/.venv/bin/python experiments/manuscript/audit_v8.py [MAIN.md SI.md]
Deliverables: AUDIT_v8.md and audit_v8.csv (or --output-dir equivalents).
A disposable, input-hash-validated fact cache is written in the system temp directory.
The v7 module is imported for its archived fact builders, never its main routine.
Unbound scientific numbers stay unsourced: numeral equality is not verification.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from statistics import median
from urllib.parse import urlsplit

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
OTHER = Path('/Users/neo/repos/algocell-wt-figs')
STATUS = ('verified', 'derived', 'rounding', 'definition', 'mismatch', 'unsourced', 'source missing')
BRACKET = re.compile(r'\[(?!\[)([^\]\n]*)\]')
WORDS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(), range(21)))
WORDS.update({'half':.5, 'one half':.5, 'none':0, 'no':0})
for ten,n in [('twenty',20),('thirty',30),('forty',40),('fifty',50),('sixty',60),('seventy',70),('eighty',80),('ninety',90)]:
    WORDS[ten]=n
    for unit,u in list(WORDS.items()):
        if isinstance(u,int) and 1<=u<=9:WORDS[ten+'-'+unit]=n+u
ORDINAL = dict(zip('first second third fourth fifth sixth seventh eighth ninth tenth'.split(), range(1,11)))
# Numbers in identifiers, versions and source paths are inventoried too, explicitly
# tagged as identifiers rather than silently omitted or treated as measurements.
DIGIT = r'0[xX][0-9a-fA-F]+|\d+(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?(?:[eE][+−-]?\d+)?'
TOKEN = re.compile(r'(?<![\w])(?:0[xX][0-9a-fA-F]+)(?!\w)|\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+−-]?\d+)?(?:[%kKM])?|\b(?:' + '|'.join(sorted((w for w in WORDS if w!='no'),key=len,reverse=True)) + r')\b|[⁰¹²³⁴⁵⁶⁷⁸⁹]+', re.I)
N = r'(?:0[xX][0-9a-fA-F]+|\d+(?:,\d{3})*(?:\.\d+)?%?|(?<![\w-])(?:'+ '|'.join(sorted(WORDS,key=len,reverse=True)) + r')(?![\w-]))'
READ = {}
LOAD_ISSUES = []
FACT_META = {}


def fmt(x):
    try:return format(Decimal(str(x)).normalize(), 'f')
    except Exception:return str(x)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    p=Path(p).resolve()
    data=p.read_bytes()
    READ.setdefault(p,hashlib.sha256(data).hexdigest())
    return data.decode('utf-8',errors='replace')


def label(p):
    p=Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def location(p,text,offset):
    return f'{label(p)}:{text.count(chr(10),0,offset)+1}:{offset-text.rfind(chr(10),0,offset)}'


def squash(t):return re.sub(r'\s+',' ',t).strip()


def norm(token):
    s=token.lower().replace('−','-')
    if s in WORDS:return float(WORDS[s])
    if s.startswith('0x'):return float(int(s,16))
    if re.fullmatch('[⁰¹²³⁴⁵⁶⁷⁸⁹]+',s):s=s.translate(str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹','0123456789'))
    mul=.01 if s.endswith('%') else 1000 if s.endswith('k') else 1000000 if s.endswith('m') else 1
    return float(Decimal(s.rstrip('%km').replace(',',''))*Decimal(str(mul)))


def load_base(use_cache=True):
    spec=importlib.util.spec_from_file_location('audit_v7_factbase',HERE/'audit_v7.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    read(HERE/'audit_v7.py')
    spec.loader.exec_module(module)
    # The large archived streams need scanning only when an input changes. The
    # cache is disposable, outside the repository, and every input hash is
    # checked before reuse. It never substitutes for or alters source data.
    cache=Path(tempfile.gettempdir())/('algocell-audit-v8-'+hashlib.sha256(str(HERE).encode()).hexdigest()[:12]+'.json')
    inputs_before=set(READ)
    # Fingerprint each source before the v7 helpers open it, including gzip inputs.
    old_text,old_stream=module.source_text,module.stream
    def source_text(name):
        p=module.resolve(name)
        if p.is_file():READ.setdefault(p.resolve(),sha(p))
        return old_text(name)
    def stream(name):
        p=module.resolve(name)
        if p.is_file():READ.setdefault(p.resolve(),sha(p))
        yield from old_stream(name)
    module.source_text=source_text
    module.stream=stream
    try:
        saved=json.loads(cache.read_text())
        if use_cache and saved['schema']==1 and all(Path(p).is_file() and sha(Path(p))==h for p,h in saved['inputs'].items()):
            READ.update({Path(p):h for p,h in saved['inputs'].items()})
            module.FACTS.update({k:module.Fact(**v) for k,v in saved['facts'].items()})
            LOAD_ISSUES.extend(saved['issues'])
            return module,[module.Claim(**c) for c in saved['claims']],saved['resolution']
    except (OSError,ValueError,KeyError,TypeError):pass
    for build in (module.build_facts,module.extra_facts,module.protocol_facts):
        try:build()
        except (OSError,KeyError,ValueError,StopIteration,AssertionError) as e:
            LOAD_ISSUES.append(f'{build.__name__}: {type(e).__name__}: {e}')
    old=read(HERE/'MAIN_v7.md')
    cs,_=module.extract(old)
    LOAD_ISSUES.extend(module.bind_claims(cs,old))
    for build in (module.more_bindings,module.law_facts_and_bindings):
        try:build(cs,old)
        except (OSError,KeyError,ValueError,StopIteration,AssertionError) as e:
            LOAD_ISSUES.append(f'{build.__name__}: {type(e).__name__}: {e}')
    cs=module.supplemental_claims(old,cs)
    resolution={}
    for line in read(HERE/'AUDIT_v7_RESOLUTION.md').splitlines():
        m=re.match(r'\| (N\d+) \|.*?\| (Source|Derivation|Correction) \| (.*) \|$',line)
        if m:resolution[m[1]]=(m[2],m[3])
    with tempfile.NamedTemporaryFile(mode='w',dir=cache.parent,prefix=cache.name+'.',delete=False) as f:
        json.dump({'schema':1,'inputs':{str(p):h for p,h in READ.items() if p not in inputs_before or p==HERE/'audit_v7.py'},'facts':{k:asdict(v) for k,v in module.FACTS.items()},'claims':[asdict(c) for c in cs],'resolution':resolution,'issues':LOAD_ISSUES},f,sort_keys=True)
    Path(f.name).replace(cache)
    return module,cs,resolution


@dataclass
class Document:
    path: Path
    text: str
    role: str


def source_bracket(b):
    return bool(re.search(r'(?:results/|manuscript/|algocell_exp/|/Users/|[\w.-]+\.py\b|REVISION_PREREG\.md)',b))


def brackets_removed(t):
    return BRACKET.sub(lambda m:'' if source_bracket(m[1]) else m[0],t)


def blocks(doc):
    section='Preamble'
    figure=''
    for m in re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)',doc.text):
        p=m[0]
        heading=re.match(r'^#{1,6}\s+(.+)',p)
        if heading:section=heading[1]
        fig=re.match(r'^\*\*Figure\s+(\d+)',p)
        if fig:figure=fig[1]
        yield m.start(),p,section+(f' / Figure {figure}' if section=='Figure legends' and figure else '')


def mask_identifiers(p):
    marks=[]
    def add(pattern,kind):
        for m in re.finditer(pattern,p,re.I):marks.append((m.start(),m.end(),kind))
    for b in BRACKET.finditer(p):
        if source_bracket(b[1]):marks.append((b.start(),b.end(),'source annotation'))
        elif re.fullmatch(r'\s*\d+(?:\s*[,–-]\s*\d+)*\s*',b[1]):marks.append((b.start(),b.end(),'citation'))
        elif re.search(r'AUTHOR|SUPPLY|TODO|FUNDING|CONTRIBUTIONS|ACKNOWLEDGEMENTS|COMPETING|DISCLOSURE|ACCESSION|RELEASE',b[1],re.I):marks.append((b.start(),b.end(),'placeholder'))
    add(r'https?://[^\s]+','URL / DOI')
    add(r'(?<!\w)[0-9a-f]{7,64}(?!\w)','hash / revision')
    for m in re.finditer(r'(?:[\w.-]+/)+[\w.*-]+',p):
        if m[0].startswith(('results/','algocell_exp/','manuscript/','experiments/','Users/')) or re.search(r'\.[A-Za-z]{2,5}(?:\.gz)?$',m[0]):
            marks.append((m.start(),m.end(),'path / identifier'))
    add(r'\b(?:Z80|6502|8086|8088|8080)\b','processor identifier')
    add(r'\b(?:C[012]|S[12]|L[12]|R[012]|P[1-4]|I1(?:-T2)?|X[1-3](?:-\d+)?|N[1-6](?:-\d+)?|DZ[123])\b','protocol identifier')
    add(r'\b(?:[A-KM-Z]\d+[a-z]?(?:-\d+)?|N\d+-\d+)\b','protocol identifier')
    add(r'\b(?:pusher\d+|ret\d+|ldir\d+|genome\d+_s\d+|lethal\d+_s\d+|closer\d+_\d+_p\d+|ed8_\d+)\b','specimen / target identifier')
    add(r'\bv[678]\b','revision identifier')
    add(r'\bzero flag\b','processor flag identifier')
    add(r'\b0xXX[0-9a-f]+\b','symbolic word encoding')
    add(r'\bN6[-/]\d+(?:/\d+)*(?:′)?','protocol identifier')
    add(r'\bed8_\*\.pdf\b','path / identifier')
    for m in re.finditer(r'`([^`\n]+)`',p):
        code=m[1]
        if re.fullmatch(r'(?:[0-9a-f]{2}|[xyNXdHLob]+)(?:\s+(?:[0-9a-f]{2}|[xyNXdHLob]+))*',code,re.I) or (not re.fullmatch(r'0x[0-9a-f]+|[0-9a-f]{2,4}',code,re.I) and not re.search(r'[=+−*/^]|\bmod\b',code)):
            marks.append((m.start(1),m.end(1),'instruction encoding / code identifier'))
    add(r'\b(?:one another|one route|one aspect|an internal one|zero-page|zero-EA|zero-SP|one-pass|one-sided|two-sided|almost none|none of the measurements)\b','non-count idiom')
    add(r'\b(?:none enlarges|None of these statements|one accuracy estimate|any one member)\b','non-count idiom')
    return marks


def extract(doc,base):
    rows=[]
    for start,p,section in blocks(doc):
        bs=[b for b in BRACKET.finditer(p) if source_bracket(b[1])]
        # SI table source cells are not enclosed in square brackets.
        source_cells=base.paths(p) if p.startswith('|') else []
        inherited=[]
        if doc.role=='legends' or section.startswith('Figure legends'):
            inherited=['manuscript/SI_v8.md','manuscript/AUDIT_v7_RESOLUTION.md']
        marks=mask_identifiers(p)
        tokens=list(TOKEN.finditer(p))
        tokens+=list(re.finditer(r'\bno(?= (?:mismatch(?:es)?|eligible evolved|evolved law intervention|paid compute|nonzero exact zero-partner|free payload|unevaluated words))',p,re.I))
        # Decimal-less hexadecimal literals such as F002, A7 and FF are quantities
        # only when separately displayed as a literal rather than part of an opcode.
        for m in re.finditer(r'(?<=`)(?:F[0-9a-f]{3}|A7|FF)(?=`)',p,re.I):
            tokens=[t for t in tokens if t.end()<=m.start() or t.start()>=m.end()]
            tokens.append(m)
        hex_sets=[]
        for sm in re.finditer(r'\b[RT]\s*=\s*\{([^}]+)\}',p):
            for hm in re.finditer(r'(?<!\w)[0-9a-f]{2}(?!\w)',p[sm.start(1):sm.end(1)],re.I):
                a=sm.start(1)+hm.start();b=sm.start(1)+hm.end()
                tokens=[t for t in tokens if t.end()<=a or t.start()>=b]
                m=re.compile(re.escape(p[a:b])).search(p,a,b)
                tokens.append(m);hex_sets.append((a,b))
        for m in sorted(tokens,key=lambda m:m.start()):
            kind=next((k for a,b,k in marks if a<=m.start()<b),'quantity')
            # Headings and figure callouts retain a separate structural inventory.
            if p.startswith('#'):kind='heading identifier'
            if re.match(r'^\*\*Figure \d+',p):kind='figure identifier'
            if section=='References':kind='bibliographic metadata'
            if re.search(r'(?:Fig(?:ure)?\.?s?|Notes?|Tables?)\s+[\d,– −a-z]*$',p[max(0,m.start()-40):m.start()],re.I):kind='display / section reference'
            tok=m[0]
            if re.search(r'status\s+`?$',p[:m.start()],re.I) and re.match(r'`?\s+hexadecimal',p[m.end():],re.I):kind='quantity'
            if re.fullmatch(r'F[0-9a-f]{3}|A7|FF',tok,re.I):kind='quantity'
            if p.startswith('|') and re.search(r'(?:^|\n)\|\s*$',p[:m.start()]) and re.match(r'\s*(?:\.\s*[^|]*)?\|',p[m.end():]):kind='table entry identifier'
            if (m.start(),m.end()) in hex_sets:kind='quantity'
            try:v=int(tok,16) if re.fullmatch(r'F[0-9a-f]{3}|A7|FF',tok,re.I) or (m.start(),m.end()) in hex_sets else norm(tok)
            except ValueError:continue
            # Isolated quoted byte values and the stated 6502 status are hex.
            if kind=='quantity' and re.search(r'hexadecimal',p[max(0,m.start()-40):m.end()+30],re.I) and re.fullmatch(r'\d{2}',tok):v=int(tok,16)
            after=[b for b in bs if b.start()>=m.end() and not re.search(r'\.(?:\s|$)',p[m.end():b.start()])]
            b=after[0] if after else min(bs,key=lambda b:min(abs(b.start()-m.end()),abs(m.start()-b.end()))) if bs else None
            attached=base.paths(b[1]) if b else source_cells
            all_paths=list(dict.fromkeys([q for b in bs for q in base.paths(b[1])]+source_cells+inherited))
            before=p[max(0,m.start()-100):m.start()]
            following=p[m.end():m.end()+130]
            # Decimal/grouping punctuation, negative signs and million suffixes.
            if tok[-1:].lower() not in ('m','k','%') and re.match(r'\s*million\b',following):v*=1000000
            if m.start() and p[m.start()-1]=='−' and not re.search(r'[\w)\]}]\s*$',before[:-1]):v=-v
            rows.append({'id':'','location':location(doc.path,doc.text,start+m.start()),'file':label(doc.path),'line':doc.text.count('\n',0,start+m.start())+1,'col':start+m.start()-doc.text.rfind('\n',0,start+m.start()),'section':section,'kind':kind,'value':tok,'normalized_value':fmt(v),'source':squash(b[1]) if b else '; '.join(attached),'status':'unsourced','fact_key':'','denominator':'','declared_denominator':'','source_value':'','correct_value':'','rounding':'','meaning':'','evidence_source':'','evidence':'','resolution_ids':'','context':squash(before+tok+following),'_value':v,'_start':start+m.start(),'_end':start+m.end(),'_pstart':start,'_p':p,'_local':m.start(),'_attached':attached,'_paths':all_paths,'_doc':doc})
    return rows


DENOMS={
 'pusher16.mutation_types_carried':'512 mutation types; pusher16/U/g8',
 'pusher16.mut_alive':'all pusher16/U mutant starts; 512 types × 4 populations',
 'pusher16.ctrl_alive_n':'64 unmutated pusher16/U populations',
 'pusher64.mutation_types_carried':'2,048 mutation types; pusher64/U/g8',
 'claims.same':'22 registered claims; all components must agree',
 'components.changed':'36 registered components',
 'population.equal':'2,343 dividing-offset sampled motif cells',
 'C1.direct':'direct intervention cases; not independent worlds or encounters',
 'C1.serial':'101 settings × 16 lineages × 8 generations × 2 directions',
 'C2.roles':'62 first + 640 final random + 240 top-three sampling roles',
 'C2.eligible':'942 roles from 80 S1 worlds; 42 eligible roles',
 'C2.eligible_tapes':'42 eligible roles; 41 distinct tapes',
 'C2.eligible_worlds':'42 eligible roles; 11 independent worlds',
 'S2.open':'14 detected S2 worlds', 'S2.confined':'14 detected S2 worlds',
 'S2.detected':'20 registered S2 worlds','S2.rep_censored':'20 registered S2 worlds',
 'S1.first_loop':'20 primary S1/A detected worlds',
 'motif.top':'scheduled top-three sample records, NOT individual ranked tapes',
 'motif.entries':'ranked tape observations: three tapes per scheduled record',
 'motif.cells':'checkpoint-cell observations, including initialization',
 'competition.defined':'20 core-C1 benign competition worlds; defined ratios only',
}


def denominator(key,f):
    if key in DENOMS:return DENOMS[key]
    if key.startswith('validation') and '.' in key:
        prefix=key.split('.')[0]
        if prefix+'.total' in B.FACTS:return f'{fmt(B.FACTS[prefix+".total"].value)} {f.meaning.split(";")[0]}'
    if key.startswith('competition.') and not key.endswith(('.worlds','.majority')):return 'T/(R+T); '+key.removeprefix('competition.')+'; defined world-level endpoint ratios'
    if key in ('C2.direct','C2.serial'):return 'complete eligible C2 panel; intervention cases' if key.endswith('direct') else 'complete eligible C2 panel; serial observations'
    return f.meaning


RULES=[]


def rule(pattern,keys,scope=None,derived=False):
    if not keys:return
    RULES.append((re.compile(pattern,re.I),keys.split() if isinstance(keys,str) else keys,scope,derived))


def new_fact(key,value,source,meaning,evidence=''):
    B.fact(key,value,source,meaning,evidence)


def additional_facts():
    s='results/evolved6502/eligibility.csv'
    a=[r for r in B.table(s) if B.truth(r['eligible'])]
    for field,key in [('tape','C2.eligible_tapes'),('world','C2.eligible_worlds')]:
        new_fact(key,len({r[field] for r in a}),s,f'eligible=True; distinct {field}')
    s='results/s8086/tables/law_candidates.csv'
    a=B.table(s)
    new_fact('L2.source_records',sum(float(r.get('source_records','1')) for r in a),s,'source-record multiplicity; see source denominators') if a and 'source_records' in a[0] else None
    s='results/s8086/exploratory/law_panel_predictions.csv'
    a=B.table(s)
    for d in (4,16):new_fact(f'posthoc.panel.d{d}',sum(float(r['d'])==d for r in a),s,f'purposive panel genotype-world rows with d={d}')
    s='results/s8086/provenance/final_spend.json'
    o=json.loads(B.source_text(s))
    for k,v in o.items():
        if isinstance(v,(int,float)) and not isinstance(v,bool):new_fact('spend.'+k,v,s,'JSON '+k)
    s='results/impact/conditions.csv';a=B.table(s)
    new_fact('impact.conditions',len(a),s,'all archived/random condition rows')
    new_fact('impact.sweep',len(a)-B.FACTS['C1.ringtotal'].value,s,'all conditions minus decision_scope=1',f'{len(a)} − {fmt(B.FACTS["C1.ringtotal"].value)}')
    for core in ('C1','C2'):new_fact(core+'.unaffected',B.FACTS[core+'.ringtotal'].value-B.FACTS[core+'.flagged'].value,'results/impact/decisions.csv',f'{core} decision_scope=1 and decision=unaffected')
    s='results/s8086/exploratory/motif_scan.csv.gz'
    new_fact('motif.entries',B.FACTS['motif.top'].value*3,s,'scheduled_top3 record count × 3 tapes per record')
    new_fact('S1.remaining',B.FACTS['S1.worlds'].value-B.FACTS['S1.first_loop'].value,'results/s6502/v2/arm_comparison.csv','primary arm A worlds minus first_loop',f'{fmt(B.FACTS["S1.worlds"].value)} − {fmt(B.FACTS["S1.first_loop"].value)}')
    new_fact('S2.remaining',B.FACTS['S2.worlds'].value-B.FACTS['S2.detected'].value,'results/s8086/tables/worlds.csv','complete S2 worlds minus detected worlds')
    new_fact('C1.changed_settings',B.FACTS['C1.edits'].value-B.FACTS['C1.parents'].value,'results/capacity/edit_eligibility.csv','all settings minus original-value controls')
    # This does not use a prose assertion as the source of an empirical count.
    s='manuscript/figures/out/v8/data/geometry_units.csv';a=B.table(s)
    new_fact('geometry.settings',len({r['d'] for r in a}),s,'number of distinct schematic displacements')
    s='manuscript/figures/out/v8/data/emergence_worlds.csv';a=B.table(s)
    new_fact('emergence.lengths',len({r['tape_len'] for r in a}),s,'distinct plotted tape lengths')
    s='results/crystal/cell_exceptions.csv';a=B.table(s)
    a=[r for r in a if r['stage']=='K' and float(r['L'])==32 and float(r['seed'])==5006 and r['offset'] and float(r['offset'])==8]
    if a:new_fact('example.unexecuted',a[0]['n_unexec'],s,'stage=K,L=32,seed=5006,offset=8; n_unexec',str({k:a[0][k] for k in ('stage','L','seed','period','offset','n_sites','n_unexec')}))
    # Supplemental quantitative facts unique to SI v8.
    s='results/s8086/tables/law_denominators.csv'
    for r in B.table(s):
        if r['family']=='L2' and not B.truth(r['exclude_L1']):
            for col in ('source_records','unique_candidates','culture_positive','eligible','cap_excluded'):
                new_fact('L2.'+col,float(r[col]),s,'family=L2, exclude_L1=False; field '+col)
    s='results/impact/metrics.csv'
    for r in B.table(s):
        if re.fullmatch(r'\d+(?:\.\d+)?',r['value']):new_fact('impact.'+r['metric'],float(r['value']),s,'metric='+r['metric']+'; value')
    s='results/s8086/exploratory/law_assay_summary.csv';a=B.table(s)
    for col in ('single_assignments','joint_assignments','erased_assignments','encounters'):
        new_fact('posthoc.sum.'+col,sum(float(r[col]) for r in a),s,'sum '+col+' across all selected genotypes')
    new_fact('posthoc.coordinates',sum(float(r['single_assignments'])/256 for r in a),s,'sum single_assignments/256: genotype-coordinate pairs')
    s='results/s8086/exploratory/family_transmission.csv';a=B.table(s)
    for key,value,meaning in [('family.assignments',len(a),'all founder-value assignments'),('family.alternatives',sum(B.truth(r['alternative']) for r in a),'alternative=True assignments'),('family.passed',sum(B.truth(r['alternative']) and B.truth(r['all16_exact']) and B.truth(r['all8_serial_exact']) for r in a),'alternative=True and all16_exact and all8_serial_exact')]:new_fact(key,value,s,meaning)
    s='results/literal_channel/direct_run.json';o=json.loads(B.source_text(s))
    for col in ('width1_orders','max_period'):new_fact('literal.direct.'+col,o[col],s,'JSON '+col)
    s='results/literal_channel/pair_layouts.csv';a=B.table(s)
    new_fact('literal.layouts',len(a),s,'all enumerated load/push layouts')
    s='results/literal_channel/minimal_symbolic_search.csv';a=B.table(s)
    new_fact('literal.width1.cases',sum(float(r['width'])==1 for r in a),s,'width=1 minimal-cycle count cases')
    s='results/literal_channel/6502_finite_harness.json';o=json.loads(B.source_text(s))
    new_fact('literal.replays',len(o['trials']),s,'JSON trials count')
    for i,budget in enumerate(sorted({t['budget'] for t in o['trials']})):new_fact('literal.replay.budget'+str(i),budget,s,'distinct trial budgets, sorted index '+str(i))
    s='results/s8086/exploratory/publication_archive.csv'
    new_fact('archive.files',len(B.table(s)),s,'publication archive rows')
    s='results/s8086/exploratory/first_family.csv';a=B.table(s)
    if a:
        for col in ('copies','SI_initial','ES','DS','DI_ring'):new_fact('family.'+col,min(float(r[col]) for r in a),s,'uniform first-family '+col)
    s='results/s8086/prelaunch/freeze.json';o=json.loads(B.source_text(s))
    for col,v in o.get('registration',{}).items():
        if isinstance(v,(int,float)) and not isinstance(v,bool):new_fact('schedule.'+col,v,s,'registration.'+col)
    s='results/evolved6502/panel.csv';a=B.table(s)
    for fn in (min,max):new_fact('S1.seed.'+fn.__name__,fn(float(r['seed']) for r in a),s,fn.__name__+' source S1 seed')
    s='manuscript/figures/out/v8/data/serial_summary.csv';a=B.table(s)
    for key in ('genome20_s6003','genome20_s6004'):
        r=next(r for r in a if r['key']==key)
        new_fact(key+'.seed',int(key.split('_s')[1]),s,'row key='+key+'; encoded source seed')
    s='results/tier2/T2_DZ.csv';a=B.table(s)
    for i,r in enumerate(a):
        for col in ('p','worlds','open_first','KM_t_closed','transmitter_majority'):new_fact(f'dial.C1.{i}.{col}',float(r[col]),s,f'p={r["p"]}; {col}')
    new_fact('dial.C1.reversals',sum(float(a[i]['KM_t_closed'])<float(a[i-1]['KM_t_closed']) for i in range(1,len(a))),s,'adjacent KM_t_closed decreases as p increases')
    s='algocell_exp/soup.py'
    tree=ast.parse(B.source_text(s))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Soup')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    defaults={a.arg:ast.literal_eval(d) for a,d in zip(fn.args.args[-len(fn.args.defaults):],fn.args.defaults) if isinstance(d,ast.Constant)}
    for name in ('width','height','pair_count','z80_steps','noise_exp'):
        new_fact('soup.'+name,defaults[name],s,'Soup.__init__ default '+name)
    new_fact('soup.cells',defaults['width']*defaults['height'],s,'width × height; default lattice cells')
    new_fact('soup.mutation_attempts',defaults['pair_count']/2**defaults['noise_exp'],s,'pair_count / 2**noise_exp; declared attempted replacements')
    s='results/s8086/exploratory/encounter_fates.csv';a=B.table(s)
    for tile in ('b951','51b9'):
        for context in ('soup','uniform'):
            for role in ('A','B'):
                rows=[r for r in a if r['isa']=='8086' and r['step']=='20000' and r['tile']==tile and r['context']==context and r['donor_role']==role]
                field='either_tile_phase_exact' if role=='A' else 'donor_retained_physical_phase'
                if rows:new_fact(f'fates.{tile}.{context}.{role}',sum(float(r[field]) for r in rows)/sum(float(r['partners']) for r in rows),s,f'8086,step=20000,tile={tile},context={context},donor_role={role}; pooled {field}/partners')


# Explicit v8 extensions. All selectors ignore the displayed numeral.
def anchored_fact(key, source, pattern, meaning, group=1, classification='definition'):
    t=read(B.resolve(source));m=re.search(pattern,t,re.I|re.S)
    if not m:raise ValueError('source anchor missing: '+key+' / '+pattern)
    token=m[group]
    value=norm(token)
    new_fact(key,value,source,meaning,squash(m[0]))
    FACT_META[key]=(classification,location(B.resolve(source),t,m.start(group)))
    return key


def selected_fact(key, source, field, filters, aggregate=None, classification='verified'):
    a=B.table(source)
    def equal(actual,want):
        if isinstance(want,(int,float)):
            try:return Decimal(actual)==Decimal(str(want))
            except ArithmeticError:return False
        return actual==str(want)
    selected=[(i,r) for i,r in enumerate(a,2) if all(equal(r[k],v) for k,v in filters.items())]
    vals=[float(r[field]) for _,r in selected if r[field] and r[field].lower()!='nan'] if field and aggregate not in ('count','distinct') else []
    if aggregate=='count':value=len(selected)
    elif aggregate=='defined':value=len(vals)
    elif aggregate=='undefined':value=len(selected)-len(vals)
    elif aggregate=='median':value=median(vals)
    elif aggregate=='min':value=min(vals)
    elif aggregate=='max':value=max(vals)
    elif aggregate=='distinct':value=len({r[field] for _,r in selected})
    else:
        if len(set(vals))!=1:raise ValueError('nonuniform selected field: '+key)
        value=vals[0]
    sel=', '.join(k+'='+str(v) for k,v in filters.items()) or 'all rows'
    evidence=f'filter [{sel}]; '+(aggregate+' ' if aggregate else 'uniform ')+(field or 'rows')+'; CSV lines '+(','.join(str(i) for i,_ in selected) if len(selected)<30 else f'{len(selected)} selected rows')
    new_fact(key,value,source,evidence,evidence)
    FACT_META[key]=(classification,source)
    return key


def comprehensive_facts():
    q='('+N.replace('[\\w-]','[\\w]')+')'
    s='results/tier2/scored/E/offset_switch.csv'
    for L,tar in [(16,'benign'),(16,'lethal'),(32,'benign')]:
        for mut,lower,upper,claim in [('off',.3,.7,'E1'),('on',.05,.4,'E2')]:
            a=[(i,r) for i,r in enumerate(B.table(s),2) if r['L']==str(L) and r['tar']==tar and r['mut']==mut and r['start']=='mix50']
            passed=[i for i,r in a if r['T_share_end'] and lower<=float(r['T_share_end'])<=upper and (mut=='off' or float(r['T_share_end'])<float(r['T_share_start']))]
            key=f'{claim}.{L}.{tar}'
            new_fact(key,len(passed),s,f'L={L},tar={tar},mut={mut},start=mix50; '+f'{lower}<=T_share_end<={upper}'+(' and T_share_end<T_share_start' if mut=='on' else ''),f'passing CSV lines {passed}; denominator {len(a)} worlds')
            new_fact(key+'.worlds',len(a),s,f'L={L},tar={tar},mut={mut},start=mix50; count rows')
    selected_fact('E2.core16',s,'core_share_end',dict(L=16,tar='benign',mut='on',start='mix50'),'median')
    selected_fact('E2.share16',s,'T_share_end',dict(L=16,tar='benign',mut='on',start='mix50'),'median')
    s='manuscript/figures/out/v8/data/competition_endpoints.csv'
    f=dict(cohort='C1',L=32,tar='benign',mut='on',start='T_into_R')
    for agg in ['defined','undefined','count','median']:selected_fact('competition.benign.T_into_R.'+agg,s,'T_share_end',f,agg)
    # The complete loss is present as zero absolute abundance and tail diversity.
    a=[(i,r) for i,r in enumerate(B.table(s),2) if all(r[k]==str(v) for k,v in f.items()) and (not r['T_share_end'] or r['T_share_end'].lower()=='nan')]
    for field in ['core_share_end','T_distinct_tails_end']:
        new_fact('competition.loss.'+field,float(a[0][1][field]),s,'C1,L=32,tar=benign,mut=on,start=T_into_R; undefined T_share_end; field '+field,f'CSV line {a[0][0]}: {a[0][1]}')
    s='results/dial/dial_worlds.csv';a=B.table(s)
    # Use the generated historical summary, not a hardcoded reversal count.
    s='manuscript/figures/out/v8/data/lethality_summary.csv';a=[r for r in B.table(s) if r['cohort']=='Historical'];a.sort(key=lambda r:float(r['p']))
    new_fact('dial.Historical.reversals',sum(float(b['confinement_km'])<float(a['confinement_km']) for a,b in zip(a,a[1:])),s,'cohort=Historical, sort p ascending; adjacent decreases in confinement_km',str([(r['p'],r['confinement_km']) for r in a]))
    s='results/capacity/edit_eligibility.csv'
    for eid in ['ldir32_v04','ldir32_v60']:
        for field in ['site','original_value','value','d','L','core_c','predicted_free']:
            selected_fact('edit.'+eid+'.'+field,s,field,dict(edit_id=eid))
    new_fact('edit.ring',B.FACTS['edit.ldir32_v60.L'].value*B.FACTS['encounter.tapes'].value,s,'2 tape segments × L for edit_id=ldir32_v60','encounter.tapes × edit.ldir32_v60.L = 2 × 32 = 64')
    # All first-family geometry constants are checked for uniformity across worlds.
    s='results/s8086/exploratory/first_family.csv';a=B.table(s)
    for field in ['d','c','phase','source_start','physical_entry','byte_moves']:
        vals={json.loads(r['geometry'])[field] for r in a}
        if len(vals)!=1:raise ValueError('first-family nonuniform '+field)
        new_fact('family.geometry.'+field,next(iter(vals)),s,'all first-family rows; geometry JSON.'+field)
    s='results/s8086/exploratory/family_serial.csv.gz'
    # Avoid a new stream: the registered generation count is a frozen procedure.
    anchored_fact('family.generations','results/s8086/EXPLORATORY.md',r'(?:through|over|across|all|at|generation|generations?=|g=)\s*(8|eight)\b','first-family transmission serial generations')
    s='results/s8086/tables/law_denominators.csv'
    for fam in ['L1','L2']:
        for field in ['eligible','culture_positive','unique_candidates','source_records','cap_excluded']:
            selected_fact(fam+'.ledger.'+field,s,field,dict(family=fam,exclude_L1='False'))
    s='manuscript/figures/out/v8/data/arrival_worlds.csv'
    for machine,prefix in [('Z80 G','arrival.Z80'),('6502 A','arrival.6502'),('8086 S2','arrival.8086')]:
        for field in ['L','budget','horizon']:
            selected_fact(prefix+'.'+field,s,field,dict(machine=machine))
        selected_fact(prefix+'.worlds',s,None,dict(machine=machine),'count')
    s='results/stageG/stageG/stage_g_runs.csv'
    a=B.table(s)
    for L in [16,20,50,64]:
        selected_fact('route.Z80.L'+str(L)+'.worlds',s,None,dict(L=L),'count')
        selected_fact('route.Z80.L'+str(L),s,'L',dict(L=L))
    new_fact('route.Z80.allworlds',sum(B.FACTS['route.Z80.L'+str(L)+'.worlds'].value for L in [16,20,50,64]),s,'L in {16,20,50,64}; sum world rows','20 + 20 + 20 + 20 = 80')
    # Additional registration and procedure constants, captured from their source.
    reg='REVISION_PREREG.md'
    captures=[
      ('E.interval.low',r'lies in (0\.3)–0\.7','E1 interval lower bound'),('E.interval.high',r'lies in 0\.3–(0\.7)','E1 interval upper bound'),
      ('E.required',r'lies in 0\.3–0\.7 in ≥ (7) of 10','E1/E2 required passing-world count'),('E.total',r'lies in 0\.3–0\.7 in ≥ 7 of (10)','E1/E2 worlds per condition'),
      ('E2.low',r'lies in (0\.05)–0\.4','E2 endpoint interval lower bound'),('E2.high',r'lies in 0\.05–(0\.4)','E2 endpoint interval upper bound'),
      ('fetch.open',r'partner byte as instruction stream in ≥ (8) of the 16','open trace numerator'),('fetch.total',r'partner byte as instruction stream in ≥ 8 of the (16)','trace-panel denominator'),
      ('fetch.confined',r'\*\*Confined\.\*\* It fetches (none)','confined zero partner-fetch traces'),
      ('class.regenerator',r'\*\*Regenerator:\*\* confined, ≤ (2)','site-score regenerator threshold'),('class.transmitter',r'\*\*Transmitter:\*\* confined, ≥ (5)','site-score transmitter threshold'),
      ('detection.share',r'top-three exact tape at share ≥(0\.005)','S2 culture-positive detection share threshold'),
      ('culture.normalization',r's_after−s_before\)/\((1)−s_before','normalized culture gain unit constant'),
      ('culture.generations',r'each of (two) generations for every tape','S2 culture generations'),('culture.S2.partners',r'n=(64) fresh uniform-byte partners','S2 culture partners per generation'),
      ('literal.opcodewidth',r'load has (one) opcode byte','literal grammar opcode bytes'),
    ]
    for key,pat,meaning in captures:
        try:anchored_fact(key,'results/literal_channel/CHECK.md' if key.startswith('literal.') else reg,pat,meaning)
        except ValueError as e:LOAD_ISSUES.append(str(e))
    # Definition records from the SI are allowed for MAIN, never for SI itself.
    # Curated below; they cannot turn empirical counts into definitions.


def comprehensive_rules():
    q='('+N.replace('[\\w-]','[\\w]')+')'
    rule(q+r'-byte pusher','pusher64.L',lambda p:'2,048' in p)
    rule(r'at least '+q+r' of first-generation offspring','branch.half',lambda p:'alleles_ldir32' in p)
    rule(r'd\s*−\s*'+q,'theorem.core',lambda p:'offset_cells.csv' in p)
    rule(q+r' full-tape d\s*=\s*'+q+r' copiers and '+q+r' d\s*=\s*'+q+r' regenerators',['posthoc.d16','edit.full_d','posthoc.d4','family.d'])
    rule(r'Amended L2 also admitted '+q,'L2.ledger.eligible')
    rule(r'(no) mismatches','L2.ledger.eligible',lambda p:'L2 is therefore untested' in p)
    rule(r'initializes a displacement d\s*=\s*'+q,'family.d')
    rule(r'first '+q+r' parental bytes','family.seed_length')
    rule(r'(?:this |With this )'+q+r'-byte footprint','family.core')
    rule(r'(?:free payload is |Zero free payload is again not )'+q,'family.free',lambda p:'family_transmission' in p)
    rule(r'('+N+r') free payload','family.free',lambda p:'family_transmission' in p)
    rule(r'through '+q+r' passages','family.generations',lambda p:'family_transmission' in p)
    rule(r'only '+q+r' of '+q+r' corrected-core benign worlds','E1.16.benign E1.16.benign.worlds')
    rule(r'(?:the|required )'+q+r'[–-]'+q+r' interval','E.interval.low E.interval.high',lambda p:'E1' in p)
    rule(r'required '+q+r' of '+q,'E.required E.total',lambda p:'E1' in p or 'E2' in p)
    rule(r'end between '+q+r' and '+q,'E2.low E2.high',lambda p:'E2' in p)
    rule(r'only '+q+r' of '+q+r' benign worlds at L\s*=\s*'+q,['E2.16.benign','E2.16.benign.worlds','arrival.Z80.L'])
    rule(q+r' of '+q+r' lethal worlds at L\s*=\s*'+q,['E2.16.lethal','E2.16.lethal.worlds','arrival.Z80.L'],lambda p:'E2' in p)
    # The two benign-length clauses need explicit preceding syntax.
    rule(r'and '+q+r' of '+q+r' benign worlds at L\s*=\s*'+q,['E2.32.benign','E2.32.benign.worlds','world.L32'],lambda p:'E2' in p)
    rule(r'at L\s*=\s*'+q+r' under benign chemistry','arrival.Z80.L',lambda p:'E2' in p)
    rule(r'median of '+q+r' of all cells','E2.core16',lambda p:'E2' in p)
    rule(r'tape length '+q+r', started with equal shares','world.L32')
    rule(r'all '+q+r' lethal worlds exceeded '+q,'competition.lethal.mix50.worlds branch.half')
    rule(r'benign T-into-R median of '+q+r' uses '+q+r' defined ratios because '+q+r' world has R\s*\+\s*T\s*=\s*'+q,['competition.benign.T_into_R.median','competition.benign.T_into_R.defined','competition.benign.T_into_R.undefined','competition.loss.core_share_end'])
    rule(r'retains '+q+r' absolute abundance','competition.loss.core_share_end')
    rule(r'failed with '+q+r' reversals','dial.Historical.reversals')
    rule(r'historically it was not met \('+q+r' reversals','dial.Historical.reversals')
    rule(r'in all '+q+r' worlds','population.G50.worlds',lambda p:'open cells remained' in p)
    rule(r'L\s*=\s*'+q+r' open cells remained','population.G50.L')
    rule(r'all '+q+r' first detected tapes were confined','lethal.first_worlds')
    rule(r'had '+q+r' sampled partner inflow','lethal.inflow.max')
    rule(r'at most '+q+r' scored sites','lethal.sites.max',lambda p:'first_cells.csv' in p)
    rule(q+r' qualifying cases','controls6502',lambda p:'census6502/' in p)
    rule(r'representing '+q+r' distinct tapes from '+q+r' worlds','C2.eligible_tapes C2.eligible_worlds')
    # Legend empirical and protocol fields.
    rule(q+r' marker carriage','branch.half',lambda p:'branching transfers' in p)
    rule(q+r' carriage','branch.half',lambda p:'branching transfers' in p)
    rule(r'at least '+q+r' their identifiable alternatives','branch.half')
    rule(r'transfer '+q+r' but not '+q,'serial.sample0 branch.gens')
    rule(r'transfer[ -]'+q+r' sites','branch.gens')
    rule(r'across '+q+r' reference encounters','serial.fetch_partners')
    rule(r'The L'+q+r' pusher','serial.pusher50.L')
    rule(r'The L'+q+r' transmitter','serial.genome20_s6004.L')
    rule(r'generation[ -]'+q+r' descendants','C1.generations',lambda p:'lineage zero' in p)
    rule(r'Editing byte '+q+r' \('+q+r'→'+q+r'\) gives displacement '+q+r' mod '+q,['edit.ldir32_v60.site','edit.ldir32_v60.original_value','edit.ldir32_v60.value','edit.ldir32_v60.value','edit.ring'])
    rule(r'Rims mark '+q+r'/'+q,'edit.ldir32_v04.value edit.ldir32_v60.value')
    rule(r'with '+q+r' uniform partners','C1.uniform_partners',lambda p:'scan finds' in p)
    rule(r'L'+q+r' mutation-on competitions','world.L32')
    rule(r'has '+q+r' equal-start worlds and '+q+r' per reciprocal '+q+r' invasion','competition.benign.mix50.worlds competition.benign.R_into_T.worlds competition.invasion',lambda p:'chemistry/implementation' in p)
    rule(q+r' ratios remain overall, '+q+r' in its '+q+r'-world T-into-R','competition.defined competition.benign.T_into_R.defined competition.benign.T_into_R.count')
    rule(r'(One|'+N+r') corrected-core benign world has R\+T\s*=\s*'+q,'competition.undefined competition.loss.core_share_end')
    rule(r'uses L'+q+r' and '+q+r' worlds per probability','arrival.Z80.L dial.C1.0.worlds')
    rule(r'≥\s*'+q+r'/'+q+r' traces','fetch.open fetch.total')
    rule(r'>\s*'+q+r' of heritable cells','branch.half')
    rule(r'≥\s*'+q+r'-site class','class.transmitter')
    rule(r'has '+q+r' worlds, '+q+r' each at L\s*=\s*'+q+r','+q+r','+q+r','+q,['route.Z80.allworlds','route.Z80.L16.worlds','route.Z80.L16','route.Z80.L20','route.Z80.L50','route.Z80.L64'])
    rule(r'Primary 6502 S1 A has '+q+r' L'+q+r' worlds','S1.worlds arrival.6502.L')
    rule(r'Coherent-fetch 8086 S2 has '+q+r' L'+q+r' worlds: '+q+r' open, '+q+r' confined and '+q+r' censored','S2.worlds arrival.8086.L S2.open S2.confined S2.rep_censored')
    rule(r'holds in '+q+r'/'+q,'S1.first_loop S1.worlds',lambda p:'loop requirement' in p)
    rule(r'for '+q+r' L'+q+r' worlds/machine','arrival.Z80.worlds arrival.Z80.L')
    rule(r'Horizons are '+q+r' Z80, '+q+r' 6502 and '+q+r' 8086','arrival.Z80.horizon arrival.6502.horizon arrival.8086.horizon')
    rule(r'All '+q+r' `5F A4 x y` first-copier tapes from '+q+r' detected','family.worlds S2.detected')
    rule(r'hold in '+q+r'/'+q+r' encounters: '+q+r' one-pass and '+q+r' copy-loop','family.trace_partners family.trace_partners tiles.memory-fed one-pass copier tiles.confined copy loop')
    # Methods exact denominators and constants that previously escaped rules.
    rule(r'arm C uses '+q,'census6502.budget.max')
    rule(r'S2.*?arm comprised '+q+r' L'+q+r' worlds','S2.worlds arrival.8086.L')
    rule(r'first scheduled top[ -]'+q,'schedule.top3')
    rule(r'at share at least '+q,'detection.share')
    rule(r'\(after−before\)/\('+q+r'−before\)','culture.normalization')
    rule(r'generation[ -]'+q+r' score','culture.generations')
    rule(r'S2 used '+q+r' uniform partners','culture.S2.partners')
    rule(r'(?:at least )'+q+r' of (?:qualifying aligned offspring|its tested identifiable values|its members)','branch.half')
    rule(r'at most '+q+r' scored sites as regenerators','class.regenerator')
    rule(q+r' or '+q+r' as intermediate','class.intermediate.low class.intermediate.high')
    rule(r'at least '+q+r' as transmitters','class.transmitter')
    rule(r'more than '+q+r' qualify, '+q+r' are sampled','branch.cap branch.cap')
    rule(r'centred '+q+r'-byte mutant window','branch.window')
    rule(r'each survivor '+q+r' partners, caps populations at '+q+r' and uses '+q+r' populations for each of '+q,'branch.k branch.cap branch.reps branch.values')
    rule(r'registered '+q+r' excess over control carriage','branch.excess')
    rule(r'KM.*?p\s*=\s*'+q+r', '+q+r', '+q+r', '+q+r', '+q+r' and '+q,['dial.C1.0.p','dial.C1.1.p','dial.C1.2.p','dial.C1.3.p','dial.C1.4.p','dial.C1.5.p'])
    rule(r'were '+q+r', '+q+r', '+q+r', '+q+r', '+q+r' and '+q+r' updates',['dial.C1.0.KM_t_closed','dial.C1.1.KM_t_closed','dial.C1.2.KM_t_closed','dial.C1.3.KM_t_closed','dial.C1.4.KM_t_closed','dial.C1.5.KM_t_closed'],lambda p:'lethality dial' in p)
    rule(r'series has '+q+r' reversal','dial.C1.reversals')
    rule(r'top[ -]'+q+r' tape at ≥\s*'+q,'schedule.top3 detection.share')
    rule(r'in '+q+r' of '+q+r' encounters','fetch.total fetch.total',lambda p:'first sample containing' in p)
    rule(r'R0 requires at least '+q+r' detections among '+q+r' worlds','route.required S2.worlds')
    rule(r'ceil\('+q+r'D\) open','route.share')
    rule(r'confined means '+q+r' in all '+q,'fetch.confined fetch.total')
    rule(r'confined\s*=\s*'+q,'S2.confined',lambda p:'Observed D' in p)
    rule(r'occurrence in '+q+r'/'+q+r' worlds and persistence in '+q+r'/'+q,'S2.occurrence S2.worlds S2.persistent S2.worlds')
    rule(r'Of '+q+r' distinct genotype-world candidates, '+q+r' were culture-positive','L2.ledger.unique_candidates L2.ledger.culture_positive')
    rule(r'L2 excluded '+q+r' for original-partner reads and left '+q,'L2.partner_reads L2.unresolved_extractions')
    rule(r'(No|'+N+r') cap excluded','L2.ledger.cap_excluded')
    rule(r'could select at most '+q+r' genotype-world cases','schedule.cases_per_family_cap')
    rule(r'all '+q+r' values at free','REP.values')
    rule(q+r' counterbalanced plus '+q+r' seeded joint assignments','REP.joint1 REP.joint2')
    rule(r'all '+q+r' census partners and '+q+r' serial generations','REP.17 REP.gen')
    rule(r'All '+q+r' byte/word REP controls','REP.controls')
    rule(r'For d\s*=\s*'+q+r', free payload is empty','family.d')
    rule(r'for d\s*=\s*'+q+r', erased coordinates are empty','arrival.8086.L')
    rule(r'ratios from '+q+r' worlds','competition.benign.T_into_R.count',lambda p:'benign corrected-core T-into-R' in p)
    rule(r'has '+q+r' defined ratios','competition.benign.T_into_R.defined',lambda p:'benign corrected-core T-into-R' in p)


def phrase_definition(pattern, source='manuscript/SI_v8.md', source_pattern=None):
    """Bind numeric slots in a named defining clause, not a bag of numerals."""
    t=read(B.resolve(source));m=re.search(source_pattern or pattern,t,re.I|re.S)
    if not m:
        LOAD_ISSUES.append('definition anchor missing: '+source+' / '+(source_pattern or pattern));return
    keys=[]
    for i,token in enumerate(m.groups(),1):
        key='definition.'+hashlib.sha256((source+'\n'+pattern+'\n'+str(i)).encode()).hexdigest()[:14]
        new_fact(key,norm(token),source,'definition slot '+str(i)+'; '+squash(m[0]),squash(m[0]))
        FACT_META[key]=('definition',location(B.resolve(source),t,m.start(i)))
        keys.append(key)
    rule(pattern,keys,lambda p:'S1' in p or '6502' in p if pattern.startswith('every ') else True)


def definitions_v8():
    q='('+N.replace('[\\w-]','[\\w]')+')'
    si='manuscript/SI_v8.md';reg='REVISION_PREREG.md';check='results/literal_channel/CHECK.md'
    def d(p,sp=None,s=si):phrase_definition(p,s,sp)
    # Mathematical statements/definitions in the explicitly named SI notes.
    d(r'ring of '+q+r'L bytes',r'P\s*=\s*'+q+r'L')
    d(r'separates '+q+r' properties',r'the '+q+r' properties are partner-information closure')
    d(r'use '+q+r' frozen, census-defined byte classes',r'comparison names '+q+r' designs, R and T')
    d(r'the '+q+r' designs fare',r'comparison names '+q+r' designs, R and T')
    d(q+r'-byte immediate-displacement copier',r'same '+q+r' held core bytes')
    d(r'channel is '+q+r' byte wide',r'load opcodes followed by '+q+r' literal byte',check)
    # The legend's display choices are definitions documented in Note 5.
    d(r'(One|'+N+r') illustrative L',r'illustrative marker trace has encounter indices one and two within '+q+r' lineage')
    d(r'(one|'+N+r') illustrative world',r'illustrative marker trace has encounter indices one and two within '+q+r' lineage')
    d(r'schematic shows '+q+r' Z80 copying mechanisms',r'comparison names '+q+r' designs, R and T')
    d(r'detection is '+q+r' minus Kaplan',r'Cumulative detection is '+q+r' minus Kaplan')
    d(r'generation[ -]'+q+r' normalized culture gain',r'normalized generation[ -]'+q+r' gain')
    d(r'≥\s*'+q+r'/'+q+r' unshifted positions',r'matching at least '+q+r' of '+q+r' bytes of the contemporaneous modal tape')
    d(r'(zero|'+N+r') bytes/all bytes',r'('+N+r') fraction uses all bytes')
    d(r'(One|'+N+r') exploratory lineage',r'encounter indices one and two within '+q+r' lineage')
    d(r'in '+q+r' 128-instruction encounters',r'encounter indices one and two within one lineage, not '+q+r' replicate lineages')
    d(r'(zero|'+N+r')-based execution step',r'lineage '+q+r', generation eight')
    d(r'copies '+q+r' L\s*=\s*'+q+r' tapes',r'aggregates '+q+r' bytes per displayed unit: L\s*=\s*'+q)
    d(r'inset: '+q+r'[–-]'+q,r'the '+q+r'[–-]'+q+r' enlargement preserves coordinates')
    d(r'red point retains '+q+r' positions',r'deficit at predicted '+q+r' versus sampled zero')
    d(r'but scores '+q+r' under the population',r'deficit at predicted four versus sampled '+q)
    d(r'Working requires generation[ -]'+q,r'working threshold is '+q+r'-generation')
    d(r'`5e ed b0` at bytes '+q+r'[–-]'+q,r'byte motif at positions '+q+r'[–-]'+q)
    d(r'other byte[ -]'+q+r' variants',r'byte '+q+r' is in R')
    # Tail excludes the four held core bytes; bind its endpoints to classifier code.
    d(r'(Zero|'+N+r') opcodes execute normally',r'fetched (0x00) halts',reg)
    d(r'fetching a '+q+r' opcode',r'fetched (0x00) halts',reg)
    d(r'confinement requires '+q,r'\*\*Confined\.\*\* It fetches '+q,reg)
    d(r'confined means '+q,r'\*\*Confined\.\*\* It fetches '+q,reg)
    d(r'(Zero|'+N+r') absolute abundance and tail diversity',r'fifth remains at '+q+r' absolute abundance')
    d(r'(One|'+N+r') terminal mark',r'coincide at '+q+r' 3,000,000-update coordinate')
    # Protocol/reset and grammar details. Source values are captured, not inserted.
    d(r'a fetched '+q+r' opcode',r'fetched '+q+r' halts',reg)
    d(r'SP\s*=\s*'+q,r'SP='+q,s='algocell_exp/machines/worlds_8086.json')
    d(r'(zero|'+N+r') count copies nothing',r'CX='+q+r' at reset',s='results/census8086/LAW_CONTRACT.md')
    # All S1 arm settings/schedule slots use the frozen S1 protocol, preserving arm.
    for arm in ['A','B','C','D']:
        sp=r'\('+arm+r'(?:, primary)?\) L\s*=\s*'+q+r', '+q
        tp=arm+r', L\s*=\s*'+q+r'(?:/budget |/)'+q
        d(tp,sp,reg)
    for tp,sp in [
        (r'top[ -]'+q+r' tapes every '+q+r' updates to '+q,r'The '+q+r' most common tapes.*?sampled every '+q+r' steps to '+q),
    ]:d(tp,sp,reg)
    for tp,sp in [
        (r'S1 specified '+q+r' culture partners',r'gen2 ≥ 0\.3 over '+q+r' partners'),
        (r'final-cell estimator used '+q,r'estimator uses '+q),
        (r'top-tape cultures used '+q,r'Top-tape cultures retain '+q+r' partners'),
        (r'implementation used '+q+r' uniform-byte partners',r'trace draws retain '+q+r' uniform-byte partners'),
        (r'Corrected frozen'+q,r'final-cell estimator uses '+q),
        (r'Complete '+q+r'-partner final-cell reclassification',r'n=32 and n='+q),
    ]:d(tp,sp,reg if tp.startswith('S1 specified') else 'results/s6502/CORRECTION.md')
    d(r'NaN is unassessable, not '+q,r'NaN is unassessable, not '+q,reg)
    d(r's\+d−'+q,r's\+d−'+q,reg)
    d(r'c\s*=\s*'+q+r'\+max\(K\)',r'c\s*=\s*'+q+r'\+max\(K\)',reg)
    d(r'physical entry '+q,r'physical entry '+q,si)
    d(r'd-periodic at physical entry '+q,r'entry phase '+q,s='results/census8086/LAW_CONTRACT.md')
    d(r'plus '+q+r' frozen partners',r'same frozen '+q+r'-partner panel',s='results/census8086/LAW_CONTRACT.md')
    d(r'(zero|'+N+r') plus 16',r'each against '+q+r' and the same frozen',s='results/census8086/LAW_CONTRACT.md')
    d(r'top[ -]'+q+r' roles',r'The '+q+r' most common tapes',reg)
    d(q+r'-byte load/push opcodes',r'A load has '+q+r' opcode byte',check)
    d(r'into '+q+r' register',r'push opcodes sharing\s+'+q+r'\s+register',check)
    d(r'periods are w\+'+q,r'ℓ=w\+'+q,check)
    d(r'N\s*≥\s*'+q+r'w\+'+q+r'−gcd\('+q+r',w\)',r'=('+N+r')w\+('+N+r')-\\gcd\(('+N+r'),w\)',check)
    d(r'n_P satisfies '+q+r' < n_P',r'('+N+r')<nP<d',check)
    d(r'Width[ -]'+q+r' mixtures provide counterexamples',r'Width '+q+r' has mixture writers',check)
    d(r'finite width[ -]'+q+r' factor witness',r'width '+q+r' has a period-16 mixture',check)
    d(r'archived '+q+r'-partner bootstrap',r'6502.*?'+q+r' initial A',check)
    d(r'nontrivial Z80 period[ -]'+q+r' writers',r'five nontrivial period-'+q,check)
    d(r'(None|'+N+r') of these statements',r'(no) such width-1 periodic self-writer',check)
    d(r'period[ -]'+q+r' and 65,536',r'All 256 period-'+q+r' and',reg)
    d(r'period[ -]'+q+r' words at L',r'65,536 period-'+q+r' words',reg)
    d(r'words at L'+q+r'/L'+q+r' and budgets '+q+r'/'+q,r'words were evaluated at each L='+q+r'/'+q+r' and budget='+q+r'/'+q,reg)
    d(r'with (no) unevaluated words',r'There are (no) unevaluated words',reg)
    d(r'partners at L'+q+r' and',r'For L='+q+r', d=8',s='results/census8086/LAW_CONTRACT.md')
    d(r'at L'+q+r', so it is',r'for L='+q+r', d=8/16/32',s='results/census8086/LAW_CONTRACT.md')
    d(r'(zero|'+N+r')-partner dependent',r'each against '+q+r' and the same frozen',s='results/census8086/LAW_CONTRACT.md')
    d(r'physical entry zero, '+q+r' ≤ d',r'('+N+r') <= d <= L',s='results/census8086/LAW_CONTRACT.md')
    # Bind the entire construction tuple to its frozen contract, in target order.


def completion_facts_and_rules():
    q='('+N.replace('[\\w-]','[\\w]')+')'
    # Explicit field bindings for the few remaining empirical statements.
    s='results/crystal/first_cells.csv'
    filters=dict(dataset='Z80 I',stage='I',endpoint='first replicator')
    a=[(i,r) for i,r in enumerate(B.table(s),2) if all(r[k]==v for k,v in filters.items())]
    if not a:raise ValueError('missing lethal first cohort')
    for key,field,fn in [('lethal.first_worlds',None,len),('lethal.inflow.max','H_bits',max),('lethal.sites.max','n_sites',max)]:
        value=len(a) if field is None else fn(float(r[field]) for _,r in a)
        new_fact(key,value,s,'dataset=Z80 I,stage=I,endpoint=first replicator; '+('count rows' if field is None else 'max '+field),f'CSV lines {[i for i,_ in a]}')
    new_fact('population.G50.worlds',B.FACTS['route.Z80.L50.worlds'].value,'results/stageG/stageG/stage_g_runs.csv','L=50; independent world rows')
    selected_fact('edit.full_d','results/s8086/exploratory/eligibility.csv','d',dict(status='eligible',d='16'))
    anchored_fact('serial.fetch_partners','serial_retention.py',r'Q = rng\.integers\(0, 256, size=\((64), L\)','reference instruction-fetch annotation partners')
    anchored_fact('class.intermediate.low','results/ladder/LADDER.md',r'M means (3)–4','intermediate site class lower endpoint')
    anchored_fact('class.intermediate.high','results/ladder/LADDER.md',r'M means 3–(4)','intermediate site class upper endpoint')
    anchored_fact('schedule.top3','REVISION_PREREG.md',r'The (three) most common tapes with their shares','top-tape sampling slots')
    anchored_fact('competition.invasion','REVISION_PREREG.md',r'(1%)','rare-invasion starting share')
    # Both endpoints of the exact loss criterion are intrinsic definitions of zero.
    anchored_fact('definition.inflow.zero','manuscript/SI_v8.md',r'hence (zero) inflow','zero partner-information inflow in the deterministic conditional identity')
    anchored_fact('definition.sampled.entropy.zero','manuscript/SI_v8.md',r'(zero) over the sample is sampled independence','sampled zero entropy is not universal closure')
    anchored_fact('definition.overlap.zero','manuscript/SI_v8.md',r'displacement d with (0) < d < P','strict positive displacement lower boundary')
    anchored_fact('definition.alphabet','manuscript/SI_v8.md',r'therefore (256)\^r members','byte alphabet cardinality')
    anchored_fact('definition.bits','manuscript/SI_v8.md',r'= (8)r ideal noiseless bits','bits per unconstrained byte')
    anchored_fact('definition.theorem.ring','manuscript/SI_v8.md',r'take P = (2)L','two-tape encounter-ring coefficient')
    anchored_fact('definition.theorem.maxL','manuscript/SI_v8.md',r'tape length L ≤ (64)','Z80 specialization tape-length bound')
    anchored_fact('definition.theorem.core','manuscript/SI_v8.md',r'require (4) ≤ d ≤ L','Z80 specialization held-core lower bound')
    # Machine source values are taken from executable contracts, without execution.
    s='algocell_exp/shader/z80_test_L16.wgsl'
    anchored_fact('reset.Z80.zero',s,r'cpu_a\s*=\s*(0)u; cpu_f=0u; cpu_b=0u','Z80 encounter reset: cpu_a zero')
    anchored_fact('io.Z80.zero',s,r'port reads (0)','Z80 IN port policy')
    s='algocell_exp/machines/worlds_8086.json'
    anchored_fact('reset.8086.zero',s,r'"ax": (0)','i8086_pair_maps_L16.world.init.set: general registers reset zero')
    s='algocell_exp/machines/i8086_pair_maps_L16.wgsl'
    anchored_fact('push.word',s,r'fn i8086_push\(v: u32\).*? - (2)u','word stack decrement in i8086_push')
    anchored_fact('push.mask',s,r'fn i8086_push\(v: u32\).*?& (0xffff)u','word stack architectural wrap mask')
    new_fact('push.modulus',B.FACTS['push.mask'].value+1,s,'i8086_push wrap modulus = mask + 1','65535 + 1 = 65536')
    new_fact('push.low.from_end',B.FACTS['push.word'].value-B.FACTS['reset.sp'].value if 'reset.sp' in B.FACTS else B.FACTS['push.word'].value-1,s,'SP=1 minus word decrement 2 gives last physical ring byte','(1 - 2) mod P = P - 1')
    new_fact('push.high',0,s,'i8086_ww writes high byte at independently wrapped offset (SP+1); SP=1, word decrement=2','(1 - 2 + 1) mod P = 0')
    # Boolean/bit value 1 in all-one means a filled input byte, not one input byte.
    anchored_fact('io.8086.byte',s,r'case 228u, 229u, 230u, 231u, 236u, 237u, 238u, 239u:.*?(255)u','input all-one low byte')
    new_fact('io.8086.one',int(B.FACTS['io.8086.byte'].value==2**8-1),s,'each input bit is set: low byte 255 = 2**8-1','255 = 0b11111111; each bit = 1')
    # Frozen classifier set entries are bound to the AST assignment, including hex.
    s='offset_switch.py';t=read(B.resolve(s));tree=ast.parse(t)
    for n in tree.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['R_VALS','T_VALS']:
            name=n.targets[0].id;values=ast.literal_eval(n.value)[32]
            for v in values:
                new_fact('classifier.'+name+'.'+format(v,'02x'),v,s,f'{name}[32] set member 0x{v:02X}',t.splitlines()[n.lineno-1])
                FACT_META['classifier.'+name+'.'+format(v,'02x')]=('definition',f'{label(B.resolve(s))}:{n.lineno}')
    anchored_fact('definition.tail.start','offset_switch.py',r'tails = s\[isT\]\[:, (4):','transmitter tail begins after four core bytes')
    # Display max position is derived from the frozen L32 classifier context.
    new_fact('definition.tail.end',B.FACTS['world.L32'].value-1,'offset_switch.py','L=32 zero-based tail endpoint = L-1','32 - 1 = 31')
    # Definitions and empirical rules, with scopes when more than one cohort shares prose.
    rule(r'(?:the |required )'+q+r'[–-]'+q+r' interval','E.interval.low E.interval.high',lambda p:'E1' in p)
    rule(r'and '+q+r' of '+q+r' worlds at L\s*=\s*'+q,['population.G20.majority','population.G20.worlds','population.G20.L'],lambda p:'regenerators were the final majority' in p)
    rule(r'all '+q+r' first detected tapes were confined','lethal.first_worlds')
    rule(r'had '+q+r' sampled partner inflow','lethal.inflow.max')
    rule(r'at most '+q+r' scored sites','lethal.sites.max',lambda p:'Z80 I' in p or 'first_cells.csv' in p)
    rule(r'full-tape d\s*=\s*'+q+r' copiers','edit.full_d')
    rule(r'(Zero|'+N+r') partner inflow is our','definition.inflow.zero')
    rule(r'(?:implies|but) '+q+r' inflow','definition.inflow.zero')
    rule(r'Sampled '+q+r' entropy','definition.sampled.entropy.zero')
    rule(r'criterion is a '+q+r' upper bound','ring_confidence')
    rule(r'across '+q+r' reference encounters','serial.fetch_partners')
    rule(r'labels hold in '+q+r'/'+q+r' encounters: '+q+r' one-pass and '+q+r' copy-loop',['family.trace_partners','family.trace_partners','tiles.memory-fed one-pass copier','tiles.confined copy loop'])
    rule(r'registers and flags to '+q+r' except SP','reset.Z80.zero')
    rule(r'I/O reads return '+q,'io.Z80.zero')
    rule(r'General registers, segments and entry IP reset to '+q,'reset.8086.zero')
    rule(r'word push decrements SP by '+q+r' modulo '+q,'push.word push.modulus')
    rule(r'ring positions P−'+q+r' and '+q,'push.low.from_end push.high',derived=True)
    rule(r'input supplies all-'+q+r' bytes','io.8086.one')
    rule(r'four arms of '+q+r' worlds, each run for '+q,'S1.worlds arrival.6502.horizon')
    rule(r'at budget '+q+r', every admitted role','branch.budget')
    rule(r'budget '+q+r' even for candidates','branch.budget',lambda p:'retention' in p)
    rule(r'followed through '+q+r' fresh-partner generations','C2.generations',lambda p:'evolved6502/' in p)
    rule(r'roles from '+q+r' worlds','C2.worlds',lambda p:'complete panel contains' in p)
    rule(q+r' first and '+q+r' final roles per arm, '+q+r' alternatives/site, '+q+r' mutant lineages and '+q+r' controls','descriptive.first descriptive.final_random descriptive.values descriptive.reps descriptive.controls')
    rule(r'Only three of the '+q+r' nontrivial Z80','literal.Z80.all')
    rule(r'At budget '+q+r', `B9 51`','census8086.budget.min')
    rule(r'the purposive '+q+r'-genotype panel','posthoc.panel')
    rule(r'joint assignments in place of '+q,'posthoc.original_joints')
    rule(q+r' census partners and '+q+r' serial transfers','REP.17 posthoc.serial_generations')
    rule(r'(none) unexpected','validation.unexpected',lambda p:'hidden-latch divergences' in p)
    rule(r'with '+q+r' < d < P','definition.overlap.zero')
    rule(r'family of '+q+r'\^r sequences, or '+q+r'r ideal','definition.alphabet definition.bits')
    rule(r'For the Z80 .*?L ≤ '+q+r', d = XX mod '+q+r'L, '+q+r' ≤ d ≤ L','definition.theorem.maxL definition.theorem.ring definition.theorem.core')
    rule(r'modulo '+q+r'L','definition.theorem.ring',lambda p:'For the Z80' in p)
    rule(r'classify.*?≤ '+q+r'/ ≥ '+q+r' site-score','class.regenerator class.transmitter')
    rule(r'≤ '+q+r'/ ≥ '+q+r' site-score','class.regenerator class.transmitter')
    rule(q+r' equal-start worlds and '+q+r' per invasion direction','competition.benign.mix50.worlds competition.benign.R_into_T.worlds')
    rule(r'p\s*=\s*'+q+r', '+q+r', '+q+r', '+q+r', '+q+r' and '+q+r' were',['dial.C1.0.p','dial.C1.1.p','dial.C1.2.p','dial.C1.3.p','dial.C1.4.p','dial.C1.5.p'])
    rule(r'its '+q+r'-byte construction','REP.core')
    rule(r'construction gives '+q+r'/'+q+r' free coordinates at L'+q+r' and '+q+r'/'+q+r'/'+q+r' at L'+q,'REP.L16.free0 REP.L16.free1 census8086.L.min REP.L32.free0 REP.L32.free1 REP.L32.free2 census8086.L.max')
    rule(r'ratio.*?R\+T = '+q,'competition.loss.core_share_end')
    # Small semantic variations, kept independent of printed figures/line positions.
    for pat,key,srcpat,src in [
        (r'S1 comprised '+q+r' arms','definition.S1.arms',r'Panel: corrected S1 v2 all (four) arms','REVISION_PREREG.md'),
        (r'at least '+q+r' its tested identifiable values','definition.scan.half',r'at least (half) of the identifiable values','results/branching/BRANCHING.md'),
        (r'at least '+q+r' its members carry','definition.carriage.half',r'at least (half) of its members','REVISION_PREREG.md'),
        (r'(one|'+N+r') trace or matching byte','definition.certificate.trace',r'core in (one) trace','manuscript/SI_v8.md'),
        (r'R\+T = '+q,'definition.ratio.zero',r'R\+T = (0)','manuscript/SI_v8.md'),
    ]:
        if src is not None:anchored_fact(key,src,srcpat,pat);rule(pat,key)


def final_main_bindings():
    q='('+N.replace('[\\w-]','[\\w]')+')'
    s='REVISION_PREREG.md'
    anchored_fact('chemistry.zero',s,r'a fetched (0x00) halts the encounter','zero opcode in lethal-chemistry contract')
    anchored_fact('partner.zero','results/census8086/LAW_CONTRACT.md',r'each against (zero) and the same frozen','all-zero census partner definition')
    anchored_fact('motif.minimum','results/s8086/exploratory/scan_contract.json',r'match >=(4) bytes','minimum linear alternating pusher run')
    anchored_fact('posthoc.original_joints',s,r'(512) joint assignments','registered evolved-law joint-assignment count')
    # The source manifest explicitly selects two conceptual specimens for Fig 1a.
    # Actual displayed byte coordinates give a one-byte-per-box representation.
    a=B.table('manuscript/figures/out/v8/data/serial_sites.csv')
    new_fact('display.bytebox',min(b-a for a,b in zip(sorted({int(r['pos']) for r in a if r.get('key')=='pusher50'}),sorted({int(r['pos']) for r in a if r.get('key')=='pusher50'})[1:])),'manuscript/figures/out/v8/data/serial_sites.csv','key=pusher50; consecutive physical pos difference',str(sorted({int(r['pos']) for r in a if r.get('key')=='pusher50'})[:3])) if a and 'pos' in a[0] else None
    rule(q+r' byte/box','display.bytebox')
    rule(r'chemistry of the '+q+r' byte','chemistry.zero')
    rule(r'where a '+q+r' opcode','chemistry.zero')
    rule(r'all-'+q+r' partners','partner.zero')
    rule(r'against '+q+r' and copies','partner.zero')
    rule(r'exact '+q+r'[- ]partner','partner.zero')
    rule(r'run of at least '+q+r' bytes','motif.minimum')
    rule(r'top[ -]'+q+r' (?:sample records|records|tapes and)','schedule.top3')
    rule(r'At L'+q+r', classification','world.L32')
    rule(r'byte '+q+r' in R','edit.ldir32_v60.site')
    rule(r'`5e ed b0` at bytes '+q+r'[–-]'+q,'classifier.motif.start classifier.motif.end')
    rule(r'tail sequences at bytes '+q+r'[–-]'+q,'definition.tail.start definition.tail.end',derived=True)
    rule(r'(?:retaining absolute loss and |retains )'+q+r' tail diversity','competition.loss.T_distinct_tails_end')
    rule(r'frozen'+q+r' primary calls','definition.frozen32')
    rule(r'(no) evolved law intervention','L2.ledger.eligible')
    rule(r'Historical.*?confined transmitter populations in '+q+r' historical worlds','dial.missed_worlds')
    rule(r'It missed confined transmitter populations in '+q+r' historical worlds','dial.missed_worlds')
    rule(r'not applicable at L'+q+r' and not met at L'+q,'stageL.L16 stageL.L20')
    rule(r'original extractor and '+q+r'-role ledger','C2.original_roles')
    rule(r'unchanged extraction extended to '+q+r' roles','C2.roles')
    rule(r'transfer, an offspring qualifies only if at least '+q,'match_gate')
    # Exact S1 schedule clause with its first phase prevents matching a different
    # cohort's otherwise similar later phases in a multirow SI table.
    sp=r'The (three) most common tapes.*?sampled every (50) steps to (2,000),\s*every (250) to (20,000), every (2,500) to (1,000,000) and every (25,000)'
    phrase_definition(r'(?:top-|Top )'+q+r'(?: tapes)? every '+q+r'(?: updates)? to '+q+r', every '+q+r' to '+q+r', every '+q+r' to '+q+r' and every '+q+r' thereafter',s,sp)
    # Concrete class boundaries and examples have definition sources or table fields.
    anchored_fact('definition.frozen32','results/s6502/CORRECTION.md',r'estimator uses (32)','primary final-cell culture partner count')
    anchored_fact('classifier.motif.start','manuscript/SI_v8.md',r'byte motif at positions (1)–3','motif first physical coordinate')
    anchored_fact('classifier.motif.end','manuscript/SI_v8.md',r'byte motif at positions 1–(3)','motif last physical coordinate')
    # Independently count the two explicitly named missed worlds in registration.
    t=read(B.resolve(s));m=re.search(r'endpoint .*?misses closed transmitter populations: worlds (\d+) .*? and (\d+)\s*\(p = 1\)',t,re.S)
    if not m:raise ValueError('dial missed-world registration anchor missing')
    new_fact('dial.missed_worlds',len(set(m.groups())),s,'distinct world identities in frozen endpoint limitation',squash(m[0]))
    o=json.loads(read(B.resolve('results/tier2/stageL_diagnosis.json')))
    for L in [16,20]:
        new_fact('stageL.L'+str(L),next(r['L'] for r in o if r['L']==L),'results/tier2/stageL_diagnosis.json','Stage L metadata L='+str(L))

def classifier_bindings(rows):
    source='offset_switch.py';text=read(B.resolve(source));tree=ast.parse(text);values={};source_lines={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['R_VALS','T_VALS']:
            d=n.value
            node=next(v for k,v in zip(d.keys,d.values) if ast.literal_eval(k)==32)
            values[n.targets[0].id[0]]=[ast.literal_eval(e) for e in node.elts];source_lines[n.targets[0].id[0]]=n.lineno
    groups=defaultdict(list)
    for r in rows:groups[(r['file'],r['_pstart'])].append(r)
    for group in groups.values():
        p=group[0]['_p']
        for m in re.finditer(r'\b([RT])\s*=\s*\{([^}]+)\}',p):
            if not re.fullmatch(r'\s*(?:[0-9a-f]{2}\s*,\s*)*[0-9a-f]{2}\s*',m[2],re.I):continue
            members=[r for r in group if m.start(2)<=r['_local']<m.end(2) and r['kind']=='quantity']
            expected=values[m[1]]
            if len(members)!=len(expected):
                LOAD_ISSUES.append('classifier set cardinality changed: '+group[0]['location']);continue
            for i,(r,v) in enumerate(zip(members,expected)):
                key='classifier.'+m[1]+'_VALS.slot'+str(i)
                new_fact(key,v,source,m[1]+'_VALS[32]; source AST literal slot '+str(i),'source set order; '+str(expected))
                FACT_META[key]=('definition',f'{label(B.resolve(source))}:'+str(source_lines[m[1]]))
                match_fact(r,key)


def si_facts_and_rules():
    q='('+N.replace('[\\w-]','[\\w]')+')'
    # Small table-derived facts; no new simulation or large log replay.
    s='results/s8086/exploratory/family_transmission.csv';a=B.table(s)
    passed=[r for r in a if B.truth(r['alternative']) and B.truth(r['all16_exact']) and B.truth(r['all8_serial_exact'])]
    worlds=sorted({r['world'] for r in a})
    new_fact('family.transmission.genotypes',len(worlds),s,'distinct world/baseline identities in founder transmission panel')
    selected_fact('family.transmission.partners',s,'partners',{})
    for pos,letter in [(0,'byte0'),(2,'x'),(3,'y')]:
        counts=[sum(r['world']==w and int(r['position'])==pos for r in passed) for w in worlds]
        for agg,fn in [('min',min),('positive',lambda x:sum(c>0 for c in x)),('max',max)]:
            new_fact('family.transmission.'+letter+'.'+agg,fn(counts),s,f'alternative=True,all16_exact=True,all8_serial_exact=True,position={pos}; per-world {agg} count',str(dict(zip(worlds,counts))))
    s='results/s8086/exploratory/family_world_coverage.csv';a=B.table(s)
    new_fact('family.covered.worlds',sum(bool(r['exemplar']) for r in a),s,'exemplar nonempty; count world rows')
    new_fact('family.covered.total',len(a),s,'all world rows')
    new_fact('family.covered.unjudged',sum(r['registered_culture_status']=='unjudged' for r in a),s,'registered_culture_status=unjudged; count world rows')
    s='results/s8086/exploratory/family_world_exemplars.csv'
    selected_fact('family.covered.partners',s,'random16_physical_phase_exact',{})
    new_fact('family.covered.lanes',B.FACTS['family.covered.worlds'].value*B.FACTS['REP.17'].value,s,'world exemplars × census contexts','18 × 17 = 306')
    s='results/s8086/exploratory/law_panel_predictions.csv';a=B.table(s)
    for d in [4,16]:selected_fact('posthoc.panel.d'+str(d),s,None,dict(d=d),'count')
    for field in ['phase','source_start']:
        gs=[json.loads(r['geometry']) for r in a]
        if len({g[field] for g in gs})!=1:raise ValueError('panel nonuniform '+field)
        new_fact('posthoc.geometry.'+field,gs[0][field],s,'all selected panel rows; geometry JSON.'+field)
    s='results/s8086/exploratory/eligibility.csv';a=B.table(s)
    pop=[r for r in a if r['registered_reason']=='unmodelled_dependency_opcode:07']
    for status in ['eligible','excluded','unresolved']:
        new_fact('posthoc.pop_es.'+status,sum(r['status']==status for r in pop),s,'registered_reason=unmodelled_dependency_opcode:07 (POP ES), status='+status+'; count genotype-world rows')
    s='results/s8086/exploratory/encounter_fates.csv'
    selected_fact('fates.partners',s,'partners',dict(isa='8086',step=20000,tile='b951',context='soup',donor_role='A'))
    selected_fact('fates.worlds',s,'world',dict(isa='8086',step=20000,tile='b951',context='soup',donor_role='A'),'distinct')
    selected_fact('fates.step',s,'step',dict(isa='8086',step=20000,tile='b951',context='soup',donor_role='A'))
    s='results/literal_channel/minimal_symbolic_search.csv';a=B.table(s)
    width1=[r for r in a if float(r['width'])==1]
    new_fact('literal.width1.orders',sum(float(r['token_orders_starting_load']) for r in width1),s,'width=1; sum token_orders_starting_load') if a and 'token_orders_starting_load' in a[0] else None
    s='results/literal_channel/minimal_run.json';o=json.loads(read(B.resolve(s)))
    for field in ['max_period','control_period']:new_fact('literal.search.'+field,o[field],s,'JSON '+field)
    # Actual contrasts rather than numeral presence in the generated report.
    s='results/toxin/causal.csv'
    for field in ['diff','lo','hi']:selected_fact('toxin.intact.'+field,s,field,dict(L=32,tar='lethal',variant='native-detox',metric='intact'))
    anchored_fact('toxin.bootstrap','payload_toxin.py',r'for _ in range\((4000)\)','host-bootstrap replicate count')
    anchored_fact('toxin.percentile.low','payload_toxin.py',r'"lo": np.percentile\(bs, (2.5)\)','lower host-bootstrap percentile')
    anchored_fact('toxin.percentile.high','payload_toxin.py',r'"hi": np.percentile\(bs, (97.5)\)','upper host-bootstrap percentile')
    # Percentile numeric constants are displayed in percent units.
    for key in ['toxin.percentile.low','toxin.percentile.high']:
        f=B.FACTS[key];new_fact(key,f.value/100,f.source,f.meaning,f.evidence)
    anchored_fact('toxin.threshold','REVISION_PREREG.md',r'(?:≥|at least|>=|by)\s*(0\.02)','registered minimum explanatory effect')
    # Every source-root/erased coordinate endpoint is read from the geometry JSON.
    s='results/s8086/exploratory/first_family.csv';a=B.table(s);g=json.loads(a[0]['geometry'])
    for field in ['R','E','K']:
        if any(json.loads(r['geometry'])[field]!=g[field] for r in a):raise ValueError('family geometry differs '+field)
        for i,v in enumerate(g[field]):new_fact('family.'+field+'.'+str(i),v,s,'all first-family rows; geometry JSON.'+field+'['+str(i)+']')
    new_fact('family.ring',len(bytes.fromhex(a[0]['tape']))*B.FACTS['encounter.tapes'].value,s,'first-family tape length × two encounter segments','16 × 2 = 32')
    new_fact('family.repeat',len(bytes.fromhex(a[0]['tape']))/len(g['R']),s,'first-family physical tape length / root count','16 / 4 = 4')
    # Remaining MAIN definition occurrences.
    anchored_fact('definition.measurements','manuscript/SI_v8.md',r'The (four) measurements are culture','enumerated assay definitions: culture, confinement, inflow, retention')
    rule(q+r' measurements','definition.measurements')
    rule(r'L'+q+r' population arm B','census8086.L.max')
    rule(r'L'+q+r' remains fixed-input census','census8086.L.max')
    # Definitions and additional phrasings in SI prose.
    pairs=[
      (r'\b'+q+r' source records, '+q+r' distinct genotype-world candidates','L2.source_records L2.unique_candidates'),
      (r'its '+q+r'-byte construction','REP.core'),
      (r'allowing '+q+r' or '+q+r' payload positions at L'+q,'REP.L16.free0 REP.L16.free1 census8086.L.min'),
      (r'at L'+q+r' and '+q+r'/'+q+r'/'+q+r' at L'+q,'census8086.L.min REP.L32.free0 REP.L32.free1 REP.L32.free2 census8086.L.max'),
      (r'all '+q+r' constructed byte/word controls','REP.controls'),
      (r'budget '+q+r', all '+q+r' values','arrival.8086.budget REP.values'),
      (r'against '+q+r' census partners and '+q+r' serial','REP.17 REP.gen'),
      (r'and '+q+r' census partners','REP.17'),
      (r'through '+q+r' serial generations','C2.generations'),
      (r'Each survivor meets '+q+r' fresh partners','branch.k'),
      (r'caps populations at '+q,'branch.cap'),
      (r'X1 tests '+q+r' seeded alternatives at every position, '+q+r' starting populations per alternative and '+q+r' unmutated','branch.values branch.reps branch.ctrl'),
      (r'at least '+q+r' of its members','branch.half'),
      (r'at least '+q+r' of identifiable tested alternatives','branch.half'),
      (r'generations '+q+r', '+q+r', '+q+r', '+q,'serial.sample0 serial.sample1 serial.sample2 serial.sample3'),
      (r'Historical serial assays use '+q+r' lineages per alternative and '+q+r' unmutated controls','serial.lineages serial.controls'),
      (r'all '+q+r' alternatives for L ≤ '+q+r' and use '+q+r' seeded alternatives','C1.alternatives world.L32 branch.values'),
      (r'S2 has '+q+r' detections, of which '+q+r' are open and '+q+r' confined; '+q+r' worlds are right-censored at '+q,'S2.detected S2.open S2.confined S2.rep_censored S2.horizon'),
      (r'confinement endpoint has '+q+r' censored worlds and median '+q,'S2.closed_censored S2.km.closed'),
      (r'median '+q+r' uses '+q+r' defined ratios among '+q+r' worlds','competition.benign.T_into_R.median competition.benign.T_into_R.defined competition.benign.T_into_R.count'),
      (r'there are '+q+r' defined ratios among '+q+r' worlds','competition.defined competition.chem_worlds'),
      (r'only '+q+r' of '+q+r' corrected-core worlds at each tested length, short of '+q,'E1.16.benign E.total E.required'),
      (r'lie in '+q+r'[–-]'+q+r' at '+q+r' updates in ≥ '+q+r'/'+q,'E2.low E2.high competition.endpoint E.required E.total'),
      (r'only '+q+r'/'+q+r' L'+q+r' benign, '+q+r'/'+q+r' L'+q+r' lethal and '+q+r'/'+q+r' L'+q+r' benign worlds','E2.16.benign E.total arrival.Z80.L E2.16.lethal E.total arrival.Z80.L E2.32.benign E.total world.L32'),
      (r'corrected-core L'+q+r' benign equal-start worlds, median T share was '+q+r' but census-defined core abundance was only '+q,'arrival.Z80.L E2.share16 E2.core16'),
      (r'there are '+q+r' 6502 sampling roles','C2.eligible'),
      (r'series are '+q+r' Z80 settings and '+q+r' 6502 sampling roles, and the '+q+r' constructed','C1.edits C2.eligible controls6502'),
      (r'Fig\. 3’s exploratory population circles summarize '+q,'population.total'),
      (r'Ten of the '+q+r' registered first copiers are '+q+r' repeats','S2.detected family.repeat'),
      (r'\b'+q+r' of the fourteen registered first copiers','family.worlds'),
      (r'congruent to '+q+r' modulo '+q,'family.d family.ring'),
      (r'SI\s*=\s*'+q+r' and DS\s*=\s*ES\s*=\s*'+q,'family.SI_initial family.ES'),
      (r'displacement '+q+r'\. Each traced exemplar makes '+q+r' byte copies in '+q+r' execution steps','family.d family.copies arrival.8086.budget'),
      (r'final ancestry is \['+q+r','+q+r','+q+r','+q+r'\]','family.R.0 family.R.1 family.R.2 family.R.3'),
      (r'R = K = C_R = \{'+q+r','+q+r','+q+r','+q+r'\}, E = \{'+q+r',…,'+q+r'\}','family.R.0 family.R.1 family.R.2 family.R.3 family.E.0 family.E.11'),
      (r'(no) free payload under that held '+q+r'-byte footprint','family.free family.core'),
      (r'each of the '+q+r' founder coordinates through all '+q+r' values','family.core REP.values'),
      (r'in all '+q+r' homologous tiles','family.repeat'),
      (r'Each assignment faces '+q+r' fresh uniform partners and '+q+r' serial fresh-partner encounters','family.transmission.partners family.generations'),
      (r'Across '+q+r' genotypes there are','family.transmission.genotypes'),
      (r'at least '+q+r' transmitted y alternatives, '+q+r' have '+q+r' x alternatives, and '+q+r' have '+q+r' byte-zero alternative','family.transmission.y.min family.transmission.x.positive family.transmission.x.max family.transmission.byte0.positive family.transmission.byte0.max'),
      (r'free count '+q+r'\. These homologous','family.free'),
      (r'includes '+q+r' of '+q+r' worlds\. All '+q+r' copy exactly against '+q+r' fresh partners; '+q+r' remain unjudged','family.covered.worlds family.covered.total family.covered.worlds family.covered.partners family.covered.unjudged'),
      (r'observer’s '+q+r' lanes \('+q+r'×'+q+r' contexts\)','family.covered.lanes family.covered.worlds REP.17'),
      (r'outside both the '+q+r'-candidate extraction denominator and the '+q+r'-genotype law panel','L2.ledger.culture_positive posthoc.panel'),
      (r'against all '+q+r' original contexts','REP.17'),
      (r'Among the '+q+r' cases originally unresolved at `POP ES`, '+q+r' qualify, '+q+r' become excluded and '+q+r' remain unresolved','L2.pop_es posthoc.pop_es.eligible posthoc.pop_es.excluded posthoc.pop_es.unresolved'),
      (r'ten d\s*=\s*'+q+r' copiers','family.d'),
      (r'source start and physical phase '+q,'posthoc.geometry.phase'),
      (r'in each of '+q+r' worlds, all '+q+r' first-family genotypes and '+q+r' deterministic additional cases: '+q+r' full-tape and '+q+r' '+q+r'-byte genotypes','posthoc.worlds family.worlds posthoc.additions posthoc.panel.d16 posthoc.panel.d4 family.core'),
      (r'and '+q+r' independent seeded joint assignments replacing the registered '+q+r'-joint panel','posthoc.joint_assignments posthoc.original_joints'),
      (r'against '+q+r' census partners and '+q+r' fresh-partner serial','REP.17 posthoc.serial_generations'),
      (r'\('+q+r'\+'+q+r'\)×\('+q+r'\+'+q+r'\)\+'+q+r'×'+q,'posthoc.sum.single_assignments posthoc.sum.joint_assignments REP.17 posthoc.serial_generations posthoc.sum.erased_assignments REP.17'),
      (r'For d\s*=\s*'+q+r', F is empty, so the '+q+r' joint assignments','family.d posthoc.joint_assignments'),
      (r'For d\s*=\s*'+q+r', E is empty','edit.full_d'),
      (r'preserves '+q+r' provenance files','archive.files'),
      (r'panel samples '+q+r' distinct cells','fates.partners'),
      (r'At '+q+r' updates, `b9 51`','fates.step'),
      (r'Each pooled entry has '+q+r'×'+q+r' =','fates.worlds fates.partners'),
      (r'within '+q+r' worlds','fates.worlds',lambda p:'encounter_fates.csv' in p),
      (r'scheduled.*?'+q+r' worlds, '+q+r' samples each','S2.worlds schedule.records'),
      (r'\b'+q+r' width-one count cases through execution period '+q,'literal.width1.cases literal.search.max_period'),
      (r'the '+q+r' unreduced orders','literal.width1.orders'),
      (r'with replacement '+q+r' times and reports percentile '+q+r'/'+q+r' limits','toxin.bootstrap toxin.percentile.low toxin.percentile.high'),
      (r'native–detox intact-survival difference is \+'+q+r', '+q+r' interval \['+q+r','+q+r'\]','toxin.intact.diff ring_confidence toxin.intact.lo toxin.intact.hi'),
      (r'explanatory threshold \+'+q,'toxin.threshold'),
    ]
    for entry in pairs:rule(*entry)
    new_fact('posthoc.additions',B.FACTS['posthoc.panel'].value-B.FACTS['posthoc.worlds'].value-B.FACTS['family.worlds'].value,'results/s8086/exploratory/law_panel_predictions.csv','selected rows minus per-world selection minus first-family genotypes','32 - 20 - 10 = 2')


TABLE_RECIPES={}

def si_table_facts():
    s='make_conds.py';t=read(B.resolve(s));tree=ast.parse(t)
    early=next(n for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='EARLY_STEPS')
    for i,v in enumerate(ast.literal_eval(early.value)):
        key='schedule.early'+str(i);new_fact(key,v,s,'EARLY_STEPS['+str(i)+']',t.splitlines()[early.lineno-1]);FACT_META[key]=('definition',f'{label(B.resolve(s))}:{early.lineno}')
    anchored_fact('schedule.E.early_interval',s,r'"sample_every_early": (50)','historical early sampling interval')
    anchored_fact('schedule.E.early_until',s,r'"early_until": (5_000)','historical end of early interval')
    anchored_fact('schedule.E.late_interval',s,r'SAMPLE_EVERY = (500)','historical later sampling interval')
    anchored_fact('schedule.G.long_interval',s,r'1_000_000, (1000)\) for s in SEEDS_G','Stage G long-run sampling interval')
    s='manuscript/figures/out/v8/data/emergence_worlds.csv'
    for field,agg in [('seed','min'),('seed','max')]:selected_fact('emergence.seed.'+agg,s,field,{},agg)
    s='results/stageG/stageG/stage_g_runs.csv'
    for agg in ['min','max']:selected_fact('stageG.seed.'+agg,s,'seed',{},agg)
    for L in [16,20,50,64]:selected_fact('stageG.L'+str(L)+'.horizon',s,'horizon',dict(L=L))
    s='REVISION_PREREG.md'
    for arm in 'ABCD':
        for i,field in [(1,'L'),(2,'budget')]:
            anchored_fact('S1.arm'+arm+'.'+field,s,r'\('+arm+r'(?:, primary)?\) L\s*=\s*(\d+),\s*(\d+)',arm+' frozen '+field,group=i)
    schedule_pattern=r'The (three) most common tapes.*?sampled every (50) steps to (2,000),\s*every (250) to (20,000), every (2,500) to (1,000,000) and every (25,000)'
    for i in range(1,9):anchored_fact('schedule.S1.slot'+str(i),s,schedule_pattern,'S1 chronological schedule slot '+str(i),group=i)
    s='results/dial/dial_worlds.csv'
    for agg in ['min','max']:selected_fact('dial.seed.'+agg,s,'seed',{},agg)
    # The dial's schedule is separately defined, despite sharing the first phases.
    s='dial_soups.py'
    anchored_fact('schedule.dial.early_interval',s,r'return (50) if step < 2000','dial initial sample spacing')
    anchored_fact('schedule.dial.early_until',s,r'return 50 if step < (2000)','dial initial interval end')
    anchored_fact('schedule.dial.middle_interval',s,r'else \((250) if step','dial middle spacing')
    anchored_fact('schedule.dial.middle_until',s,r'250 if step < (20000)','dial middle interval end')
    anchored_fact('schedule.dial.late_interval',s,r'else (2500)','dial final sample spacing')
    TABLE_RECIPES.update({
      'Historical Stage E, nominal unablated subset':('emergence.worlds emergence.seed.min emergence.seed.max arrival.budget emergence.horizon '+' '.join('schedule.early'+str(i) for i in range(8))+' schedule.E.early_interval schedule.E.early_until schedule.E.late_interval schedule.top3 detection.share').split(),
      'Historical Stage G':('route.Z80.L16.worlds route.Z80.L16 route.Z80.L20 route.Z80.L50 route.Z80.L64 stageG.seed.min stageG.seed.max arrival.budget route.Z80.L16 route.Z80.L50 stageG.L16.horizon route.Z80.L20 route.Z80.L64 stageG.L20.horizon schedule.E.late_interval schedule.E.early_until schedule.G.long_interval schedule.top3').split(),
      '6502 S1 A–D':('S1.worlds S1.seed.min S1.seed.max arrival.6502.horizon '+' '.join('S1.arm'+arm+'.'+field for arm in 'ABCD' for field in ['L','budget'])+' '+' '.join('schedule.S1.slot'+str(i) for i in range(1,9))+' schedule.top3 detection.share').split(),
      'Coherent-fetch 8086 S2 A':('S2.worlds arrival.8086.L arrival.8086.budget S2.horizon census8086.L.max S2.horizon schedule.records schedule.checkpoint0 schedule.checkpoint1 schedule.checkpoint2 schedule.checkpoint3 schedule.checkpoint4 schedule.top3 detection.share').split(),
      'Historical/core-C1 lethality dial':('dial.C1.0.worlds '+' '.join('dial.C1.'+str(i)+'.p' for i in range(6))+' dial.seed.min dial.seed.max arrival.Z80.L arrival.budget emergence.horizon schedule.top3 schedule.dial.early_interval schedule.dial.early_until schedule.dial.middle_interval schedule.dial.middle_until schedule.dial.late_interval').split(),
      'L32 mutation-on displacement competitions':('world.L32 competition.benign.mix50.worlds competition.benign.R_into_T.worlds competition.invasion competition.benign.T_into_R.worlds competition.invasion competition.endpoint competition.chem_worlds').split(),
    })


def apply_table_recipes(rows):
    groups=defaultdict(list)
    for r in rows:
        if r['_doc'].role=='si' and r['kind']=='quantity':groups[r['line']].append(r)
    for group in groups.values():
        line=group[0]['_doc'].text.splitlines()[group[0]['line']-1]
        for prefix,keys in TABLE_RECIPES.items():
            if not line.startswith('| '+prefix+' |'):continue
            if len(group)!=len(keys):
                LOAD_ISSUES.append(f'SI cohort row {prefix}: {len(keys)} slots expected, {len(group)} found');continue
            for r,key in zip(group,keys):
                if key not in B.FACTS:LOAD_ISSUES.append('missing cohort fact '+key)
                else:match_fact(r,key)


def mathematical_contract_rules():
    q='('+N.replace('[\\w-]','[\\w]')+')'
    scope=lambda p:'literal_channel/CHECK.md' in p or 'Fine–Wilf' in p
    for pattern,key in [
        (q+r'w','literal.coeff2'),(r'(?:periods? |gcd\()w\+'+q,'literal.bound2'),(r'gcd\('+q+r',w\)','literal.gcd2'),(r'w/'+q,'literal.gcd2'),
        (q+r'-byte push opcode','literal.opcode'),(q+r' data register','literal.register'),
    ]:rule(pattern,key,scope)
    # These constants define mathematical parity cases in the independent proof.
    for key,sp in [('literal.odd.g',r'For odd w, g=(1)'),('literal.even.g',r'For even w, g=(2)'),('literal.odd.add',r'bound is \*\*2w\+(1) for odd')]:
        anchored_fact(key,'results/literal_channel/CHECK.md',sp,key)
    rule(r'g\s*=\s*'+q+r' forces a = b','literal.odd.g',scope)
    rule(r'g\s*=\s*'+q+r' forces alternation','literal.even.g',scope)
    rule(r'finite bound is '+q+r'w\+'+q+r' for odd widths and '+q+r'w',['literal.coeff2','literal.odd.add','literal.coeff2'],scope)
    # The input finite witness is decoded directly from its frozen word and grammar.
    s='results/literal_channel/finite_witness.json';o=json.loads(read(B.resolve(s)));word=o['word'];a=169;b=72;ip=0;loads=pushes=0
    while ip<len(word):
        if word[ip]==a:loads+=1;ip+=2
        elif word[ip]==b:pushes+=1;ip+=1
        else:raise ValueError('unknown finite witness opcode')
    for key,value,desc in [('literal.witness.loads',loads,'loads in one primitive input period'),('literal.witness.pushes',pushes,'pushes in one primitive input period'),('literal.witness.L',len(word),'word length'),('literal.witness.next',o['copy_length']+1,'first disagreeing byte ordinal')]:new_fact(key,value,s,desc,str(word)+'; decode distinct one-byte load/push, width=1')
    rule(r'The '+q+r'-byte input','literal.witness.L',lambda p:'finite width-one factor witness' in p)
    rule(r'Its '+q+r' loads and '+q+r' pushes produce a matching '+q+r'-byte factor','literal.witness.loads literal.witness.pushes literal.witness.L')
    rule(r'disagree at byte '+q+r'; the input primitive period is '+q+r' and output period '+q,'literal.witness.next literal.witness.L literal.witness.pushes')
    rule(r'those periods is '+q+r'\+'+q+r'−gcd\('+q+r','+q+r'\)','literal.witness.L literal.witness.pushes literal.witness.L literal.witness.pushes')
    rule(q+r' unreduced orders','literal.width1.orders')


def strengthen_bindings():
    q='('+N.replace('[\\w-]','[\\w]')+')'
    # Replace archived manuscript theorem anchors with the defining SI statements
    # for MAIN. SI cannot cite itself and remains unresolved when no external
    # contract supplies its particular proof constant.
    si='manuscript/SI_v8.md'
    for key,pattern in [
        ('theorem.ring',r'take P = (2)L'),('theorem.core',r'require (4) ≤ d'),
        ('theorem.Lmax',r'tape length L ≤ (64)'),('theorem.entry',r'control to address (1)'),
        ('theorem.HL',r'leave HL ≡ (0)'),('theorem.Dscale',r'and (256)D ≡ 0'),
        ('theorem.Dzero',r'256D ≡ (0)'),('theorem.BCzero',r'BC either (zero)'),
        ('theorem.moves',r'BC either zero or at least (126)'),('theorem.immediate',r'and (127) copying executions'),
        ('entropy.outputs',r'entropy over (256) partners'),('entropy.bits',r'log₂256 = (8) bits'),
    ]:anchored_fact(key,si,pattern,key)
    for key,pattern in [
        ('literal.coeff2',r'=([2])w\+2-\\gcd'),('literal.bound2',r'=2w\+([2])-\\gcd'),
        ('literal.gcd2',r'\\gcd\(([2]),w\)'),('literal.opcode',r'A load has (one) opcode byte'),
        ('literal.register',r'opcodes sharing\s+(one)\s+register'),('literal.width1',r'width (1) has a period-16 mixture'),
        ('literal.width3',r'Width (3) has mixture writers'),
    ]:anchored_fact(key,'results/literal_channel/CHECK.md',pattern,key)
    # The earlier 714-role freeze is verifiable in the local immutable git object.
    object_name='943aeee:experiments/results/evolved6502/panel.csv'
    proc=subprocess.run(['git','show',object_name],cwd=REPO,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    a=list(csv.DictReader(proc.stdout.decode().splitlines()))
    new_fact('C2.original_roles',len(a),'results/evolved6502/panel.csv','all sampling-role rows at frozen registration/extractor commit 943aeee','git show '+object_name+'; all rows; blob SHA256='+hashlib.sha256(proc.stdout).hexdigest())
    FACT_META['C2.original_roles']=('verified','git:'+object_name+' (all rows)')
    # Figure-specific counts retain their own observational unit.
    s='results/concept/traces.json';o=json.loads(read(B.resolve(s)));names=['pusher_L16','retnz_L16']
    if not all(n in o for n in names):raise ValueError('concept specimens missing')
    new_fact('concept.mechanisms',len(names),s,'JSON keys in {pusher_L16,retnz_L16}; count selected copying designs',str(names))
    selected_fact('world.illustrative', 'manuscript/figures/out/v8/data/world_trajectory.csv','seed',dict(L=16,seed=2002),'distinct')
    rule(r'schematic shows '+q+r' Z80 copying mechanisms','concept.mechanisms')
    rule(r'(One|'+N+r') illustrative L','world.illustrative')
    rule(r'(one|'+N+r') illustrative world','world.illustrative')
    rule(r'copies '+q+r' L\s*=\s*'+q+r' tapes','encounter.tapes geometry.L')
    rule(r'(Zero|'+N+r') absolute abundance and tail diversity','competition.loss.core_share_end')
    anchored_fact('law.zero_tolerance','REVISION_PREREG.md',r'There is (zero) tolerance for an exact','exact-law mismatch tolerance')
    rule(r'(no) mismatches','law.zero_tolerance',lambda p:'L2 is therefore untested' in p)


SI_REASONS={
  '1':'Primary overlap proof, Z80 algebra and counterexample constants: no independent defining clause or keyed certificate field resolved; self-citation is forbidden.',
  '2':'Primary certificate/information identities and logical zero/one constants: the SI introduces these statements; empirical entropy samples do not establish the universal identities.',
  '3':'Proof/construction operands, scoped BFF copy-cost constants or symbolic counterexample details without an unambiguous independent defining clause/field; no matching by numeral alone.',
  '4':'Constructed 6502 reset/stack algebra and geometry-specific protocol or assay operands whose exact defining source slot is not resolved; empirical and planned denominators remain distinct.',
  '5':'Historical cohort schedule/seed details, display definitions, statistical constants or diagnostic examples needing an exact independent source slot; proof/measurement definitions cannot cite this same SI.',
  '6':'Chronology dates, inherited protocol identifiers and genealogy thresholds/counts without a unique frozen registration line or generated ledger field binding.',
  '7':'Impact/Stage-L protocol and diagnostic quantities requiring a specific frozen scorer/source-grid/ledger field; no source ambiguity is converted to verification.',
  '8':'Mechanism witness instruction counts, phases or construction operands needing keyed causal-prefix/census records, rather than unchanged output matches.',
  '9':'Remaining exploratory trace/geometry/selection or sampling operands needing a keyed source field; archive absence cannot supply a causal population rate.',
  'index':'Extended Data/index display parameters and cohort descriptions without a separate defining/filtered field binding.',
}


def assign_parts(rows):
    for r in rows:
        doc=r['_doc']
        if doc.role=='si':
            r['part']='SI'
            notes=[m for m in re.finditer(r'^## Supplementary Note (\d+)\.',doc.text,re.M) if m.start()<=r['_start']]
            index=re.search(r'^## Supplementary Tables',doc.text,re.M)
            r['si_note']='index' if index and r['_start']>=index.start() else notes[-1][1] if notes else 'preamble'
        elif doc.role=='legends':r['part']='exported legends';r['si_note']=''
        else:
            boundaries={name:doc.text.find('## '+name+'\n') for name in ['Abstract','Main text','Figure legends','Methods']}
            pos=r['_start']
            r['part']='abstract' if boundaries['Abstract']<=pos<boundaries['Main text'] else 'main' if boundaries['Main text']<=pos<boundaries['Figure legends'] else 'legends' if boundaries['Figure legends']<=pos<boundaries['Methods'] else 'Methods'
            r['si_note']=''
        r['unresolved_reason']=SI_REASONS.get(r['si_note'],'No unique independent source field/defining line resolved.') if r['status'] in ('unsourced','source missing') else ''
    return {part:{s:sum(r['part']==part and r['status']==s for r in rows) for s in STATUS} for part in ['abstract','main','legends','Methods','SI','exported legends']}


def last_si_bindings():
    q='('+N.replace('[\\w-]','[\\w]')+')'
    # Generated Markdown tables are selected by their row labels and field slots.
    source='results/lod_n6/N6_SCORE.md';t=read(B.resolve(source))
    m=re.search(r'^\| N6-4 \|[^\n]*?\| (\d+) and (\d+) of (\d+);',t,re.M)
    if not m:raise ValueError('N6-4 generated table row missing')
    for i,field in [(2,'executed_non_copier'),(3,'worlds')]:
        key='N6.'+field;new_fact(key,float(m[i]),source,'generated row prediction=N6-4; observed field '+field,m[0])
        FACT_META[key]=('verified',location(B.resolve(source),t,m.start(i)))
    anchored_fact('N6.required_fraction','REVISION_PREREG.md',r'non-copier.*?≥ (0\.6) n','N6-4 non-copier fraction threshold')
    # Recompute the integer cutoff on that independently bound threshold.
    new_fact('N6.required',math.ceil(B.FACTS['N6.required_fraction'].value*B.FACTS['N6.worlds'].value),'REVISION_PREREG.md','N6-4 executed-byte criterion: ceil(fraction × recorded worlds)','ceil(N6.required_fraction × N6.worlds) = ceil(0.6 × 20) = 12')
    anchored_fact('dial.permutations','dial_soups.py',r'def _perm_spearman\(x, y, n=(20000)','seeded permutation resamples')
    anchored_fact('km.median','manuscript/figures/prepare_v8.py',r'if y>=\s*(\.5)-1e-12','KM cumulative-detection median threshold')
    anchored_fact('display.emergence.min','manuscript/figures/build_v8.py',r'xlim=\((40), 350000\)','emergence x-axis lower display limit')
    anchored_fact('detection.cells','REVISION_PREREG.md',r'at least (100)/20,000 cells','S2 culture-positive detection cell threshold')
    # Bootstrap coverage is arithmetic on its own percentile limits.
    new_fact('toxin.confidence',B.FACTS['toxin.percentile.high'].value-B.FACTS['toxin.percentile.low'].value,'payload_toxin.py','bootstrap coverage = upper percentile - lower percentile','0.975 - 0.025 = 0.95')
    selected_fact('impact.c0.checked_source_rows','results/impact/unit_table_c0_validation.csv','source_row',{},'distinct')
    # These text variants preserve the same documented observational units.
    pairs=[
      (r'in '+q+r' new worlds','N6.worlds',lambda p:'N6 tested' in p),
      (r'only '+q+r' of '+q+r' writers were non-copiers.*?against '+q+r' required','N6.executed_non_copier N6.worlds N6.required',lambda p:'N6-4' in p),
      (r'the median is the first update where survival reaches '+q,'km.median'),
      (r'with average ranks, '+q+r' seeded permutations','dial.permutations'),
      (r'none of '+q+r', not an exactly','dial.permutations'),
      (r'native–detox intact-survival difference is \+'+q+r', '+q+r' interval \['+q+r','+q+r'\]','toxin.intact.diff toxin.confidence toxin.intact.lo toxin.intact.hi'),
      (r'All '+q+r' cited source-table cases were checked','impact.c0.checked_source_rows'),
      (r'catalogue covers '+q+r' tape/world/mask combinations','assays.total'),
      (r'corrected core flags '+q+r'/'+q+r' conditions and the vendor-specific Zilog-NMOS core '+q+r'/'+q,'C1.flagged C1.ringtotal C2.flagged C2.ringtotal'),
      (r'All '+q+r' corrected-core flags','C1.flagged'),
      (r'Stage L is complete and scorable in all '+q+r' worlds','stageL.worlds'),
      (r'dispatcher completion remains '+q+r' of '+q,'stageL.completed stageL.worlds'),
      (r'all journals reach '+q+r' updates','stageL.horizon'),
      (r'all '+q+r' originally saved snapshots','stageL.snapshots'),
      (r'the '+q+r' rows in Fig\. 1d','validation.units'),
      (r'uses '+q+r' byte/word geometries at budget '+q,'REP.controls arrival.8086.budget'),
      (r'All '+q+r' controls completed','REP.controls'),
      (r'L'+q+r' controls remain fixed-input evidence after withdrawal of the L'+q+r' population arm','census8086.L.max census8086.L.max'),
      (r'final top '+q+r' and '+q+r' uniformly sampled final cells','schedule.top3 schedule.final_cells'),
      (r'L1 admits '+q,'L1.ledger.eligible'),
      (r'evolved assay denominators are '+q,'L1.ledger.eligible'),
      (r'at most '+q+r' eligible genotype-world cases','schedule.cases_per_family_cap'),
      (r'and the '+q+r' census partners','REP.17'),
      (r'motif at positions '+q+r'[–-]'+q,'classifier.motif.start classifier.motif.end'),
      (r'byte '+q+r' is in R','edit.ldir32_v60.site'),
      (r'R includes partial-copy d = '+q+r' and d = '+q+r' designs with '+q+r' and '+q+r' predicted payload coordinates','classifier.d8 classifier.d16 classifier.free4 classifier.free12'),
      (r'regenerators have at most '+q+r' scored sites and confined transmitters at least '+q,'class.regenerator class.transmitter'),
      (r'the fifth remains at '+q+r' absolute abundance and '+q+r' tail diversity','competition.loss.core_share_end competition.loss.T_distinct_tails_end'),
      (r'two bytes per displayed unit: L = '+q+r', c = '+q+r' and d = '+q+r' or '+q+r', with '+q+r' or '+q+r' free byte coordinates','geometry.L geometry.c geometry.dmin geometry.dmax geometry.zero geometry.free'),
      (r'deficit at predicted '+q+r' versus sampled '+q,'population.deficit.predicted population.deficit.observed'),
      (r'Stage K/L'+q+r'/seed '+q+r', has '+q+r' total scored sites, '+q+r' unexecuted, with displacement '+q,'example.L example.seed example.sites example.unexecuted example.offset'),
      (r'illustrative world in Fig\. 1c is historical seed '+q+r', L'+q+r', with '+q+r' cells','world.seed world.L world.cells'),
      (r'matching at least '+q+r' of '+q+r' bytes','display.modal.matches world.L'),
      (r'The '+q+r' serial-curve keys','display.serial_keys'),
      (r'The '+q+r' byte-strip keys','display.strip_keys'),
      (r'population-disc.*?'+q+r' scale exemplars '+q+r', '+q+r' and '+q,'display.scales display.scale0 display.scale1 display.scale2'),
      (r'emergence display begins at '+q+r' updates','display.emergence.min'),
      (r'Competition plots select '+q+r' starting categories','display.competition.starts'),
      (r'reciprocal '+q+r' invasions have initial T shares '+q+r' or '+q,'competition.invasion competition.start.low competition.start.high'),
      (r'dial reports '+q+r' outcome fields','display.dial.fields'),
      (r'Arrival curves select '+q+r' machine/cohort groups','display.arrival.groups'),
    ]
    for entry in pairs:rule(*entry)
    # Compute display selections from their actual generated coordinate tables.
    for key,file,field in [('display.serial_keys','serial_summary','key'),('display.strip_keys','serial_sites','key'),('display.competition.starts','competition_endpoints','start'),('display.arrival.groups','arrival_worlds','machine')]:
        selected_fact(key,'manuscript/figures/out/v8/data/'+file+'.csv',field,{},'distinct')
    # serial_sites additionally includes nonplotted old specimens; the six plot
    # keys are defined by the preparation function rather than every source row.
    s='manuscript/figures/prepare_v7.py';t=read(B.resolve(s))
    for fn,key in [('old_keys','display.serial_keys')]:
        tree=ast.parse(t);f=next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn),None)
        if f:
            ret=next(n for n in ast.walk(f) if isinstance(n,ast.Return))
            vals=ast.literal_eval(ret.value);new_fact(key,len(vals),s,'function '+fn+' returned specimen keys',str(vals));FACT_META[key]=('definition',f'{label(B.resolve(s))}:{ret.lineno}')
    s='manuscript/figures/build_v8.py';t=read(B.resolve(s));tree=ast.parse(t)
    keys=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='rows' for x in n.targets) and isinstance(n.value,ast.List) and n.value.elts and isinstance(n.value.elts[0],ast.Tuple) and isinstance(n.value.elts[0].elts[0],ast.Constant) and n.value.elts[0].elts[0].value=='pusher50')
    vals=ast.literal_eval(keys.value)
    new_fact('display.strip_keys',len(vals),s,'Fig 2c selected byte-strip keys',str(vals))
    FACT_META['display.strip_keys']=('definition',f'{label(B.resolve(s))}:{keys.lineno}')
    s='results/tier2/stageL_diagnosis.json';a=json.loads(read(B.resolve(s)))
    for key,value,desc in [('stageL.worlds',len(a),'all world records'),('stageL.completed',sum(not r['marker_error'] and not r['missing_config_snapshots'] for r in a),'marker_error empty AND missing_config_snapshots empty'),('stageL.horizon',min(r['steps_run'] for r in a),'uniform steps_run'),('stageL.snapshots',min(len(r['actual_snapshot_steps']) for r in a),'uniform actual_snapshot_steps count')]:new_fact(key,value,s,desc)
    # Source-rendered one-byte matching gate and symbol-area exemplars.
    source='manuscript/figures/build_v8.py';t=read(B.resolve(source))
    new_fact('display.modal.matches',B.FACTS['match_gate'].value*B.FACTS['world.L'].value,'algocell_exp/metrics.py','L=16; modal-match threshold 0.75 × L','0.75 × 16 = 12')
    s='results/offset/OFFSET_CENSUS.md';t=read(B.resolve(s))
    # The census rows bind effective d and free=d-core without the manuscript.
    for d in [8,16]:
        member='08' if d==8 else '50'
        new_fact('classifier.d'+str(d),B.FACTS['classifier.R_VALS.'+member].value%B.FACTS['edit.ring'].value,'offset_switch.py','R_VALS[32] member 0x'+member+' reduced modulo ring length','0x08 mod 64 = 8' if d==8 else '0x50 mod 64 = 16')
    for n,d in [(4,8),(12,16)]:new_fact('classifier.free'+str(n),B.FACTS['classifier.d'+str(d)].value-B.FACTS['geometry.c'].value,'offset_switch.py','partial-copy d minus constrained core','d - c = '+str(d)+' - 4 = '+str(n))
    selected_fact('population.deficit.predicted','manuscript/figures/out/v8/data/law_population.csv','predicted',dict(predicted=4,n_unexec=0))
    selected_fact('population.deficit.observed','manuscript/figures/out/v8/data/law_population.csv','n_unexec',dict(predicted=4,n_unexec=0))
    new_fact('geometry.zero',B.FACTS['geometry.dmin'].value-B.FACTS['geometry.c'].value,'manuscript/figures/out/v8/data/geometry_units.csv','d=4: free=d-c','4 - 4 = 0')
    selected_fact('example.L','results/crystal/cell_exceptions.csv','L',dict(stage='K',L=32,seed=5006,offset=8))
    selected_fact('example.seed','results/crystal/cell_exceptions.csv','seed',dict(stage='K',L=32,seed=5006,offset=8))
    new_fact('competition.start.low',B.FACTS['competition.invasion'].value,'REVISION_PREREG.md','1% rare invasion: initial T share for T-into-R','competition.invasion = 0.01')
    new_fact('competition.start.high',1-B.FACTS['competition.invasion'].value,'REVISION_PREREG.md','R-into-T: complement of R invasion fraction','1 - 0.01 = 0.99')


def derived_bindings():
    """Record arithmetic with explicit independently bound operands and filters."""
    anchored_fact('reset.8086.sp','algocell_exp/machines/i8086_pair_maps_L16.wgsl',r'i8086_r\[4\] = (0x1)u; // first push','frozen pair reset SP')
    anchored_fact('word.next_byte','algocell_exp/machines/i8086_pair_maps_L16.wgsl',r'fn i8086_ww.*?\(off \+ (1)u\)','independently wrapped word high-byte offset')
    specs=[
      ('edit.ring',['encounter.tapes','edit.ldir32_v60.L'],'{0} * {1}'),
      ('push.modulus',['push.mask','word.next_byte'],'{0} + {1}'),
      ('push.low.from_end',['push.word','reset.8086.sp'],'{0} - {1}'),
      ('push.high',['reset.8086.sp','push.word','word.next_byte'],'{0} - {1} + {2}'),
      ('geometry.free',['geometry.units','geometry.bytes_per_unit'],'{0} * {1}'),
      ('geometry.zero',['geometry.dmin','geometry.c'],'{0} - {1}'),
      ('family.ring',['encounter.tapes','tiles.bytes'],'{0} * {1}'),
      ('family.repeat',['tiles.bytes','family.seed_length'],'{0} / {1}'),
      ('family.covered.lanes',['family.covered.worlds','REP.17'],'{0} * {1}'),
      ('classifier.d8',['classifier.R_VALS.08','edit.ring'],'{0} mod {1}'),
      ('classifier.d16',['classifier.R_VALS.50','edit.ring'],'{0} mod {1}'),
      ('classifier.free4',['classifier.d8','geometry.c'],'{0} - {1}'),
      ('classifier.free12',['classifier.d16','geometry.c'],'{0} - {1}'),
      ('definition.tail.end',['world.L32','word.next_byte'],'{0} - {1}'),
      ('posthoc.coordinates',['posthoc.sum.single_assignments','REP.values'],'{0} / {1}'),
      ('posthoc.additions',['posthoc.panel','posthoc.worlds','family.worlds'],'{0} - {1} - {2}'),
      ('competition.start.high',['word.next_byte','competition.invasion'],'{0} - {1}'),
      ('toxin.confidence',['toxin.percentile.high','toxin.percentile.low'],'{0} - {1}'),
      ('display.modal.matches',['match_gate','world.L'],'{0} * {1}'),
    ]
    for key,operands,expression in specs:
        facts=[B.FACTS[k] for k in operands]
        expanded=expression.format(*(fmt(f.value) for f in facts))
        value=arithmetic_value(expanded)
        if float(value)!=B.FACTS[key].value:raise ValueError('derivation/source conflict: '+key)
        f=B.FACTS[key]
        detail=expanded+' = '+fmt(value)+'; operands: '+'; '.join(k+'='+fmt(v.value)+' ['+v.source+'; '+v.meaning+']' for k,v in zip(operands,facts))
        new_fact(key,value,f.source,f.meaning,detail)
        FACT_META[key]=('derived','; '.join(dict.fromkeys(v.source for v in facts)))
    f=B.FACTS['N6.required']
    FACT_META['N6.required']=('derived',f.source+'; results/lod_n6/N6_SCORE.md: observed N6-4 denominator')


def threshold_definitions():
    q='('+N.replace('[\\w-]','[\\w]')+')'
    anchored_fact('carriage.population_min','REVISION_PREREG.md',r'it carries the allele if at least (half) of its members','X1 population marker-carriage fraction')
    anchored_fact('scan.value_min','mutscan.py',r'tr_ok = .*?>= (0\.5)','single-mutant carriage fraction among qualifying copies')
    anchored_fact('scan.site_min','mutscan.py',r'site\["transmissible"\] = site\["frac_both"\] >= (0\.5)','site fraction of identifiable variants satisfying both tests')
    anchored_fact('dial.inherit_min','crystal_check.py',r'inherit=w\[w.allele_aligned_frac>=(\.5)\]','working byte-dial alternatives matching their allele in first-generation offspring')
    anchored_fact('majority.dial_min','dial_soups.py',r'\(tr > (0\.5)\).sum\(\)','dial majority threshold with heritable-cell denominator')
    # The L32 competition majority cutoff is independently registered.
    anchored_fact('majority.competition_min','tier2_report.py',r'nwin, nkill = int\(\(arm.T_share_end > (\.5)\).sum\(\)\)','competition full-tape-class majority cutoff')
    ruleset=[
      (q+r' marker carriage','carriage.population_min',lambda p:'branching' in p.lower()),
      (q+r' carriage','carriage.population_min',lambda p:'Generation-eight bars' in p),
      (r'at least '+q+r' (?:of )?its members','carriage.population_min'),
      (r'alive with fewer than '+q+r' carrying','carriage.population_min'),
      (r'at least '+q+r' of qualifying aligned offspring','scan.value_min'),
      (r'at least '+q+r' (?:of )?its tested identifiable values','scan.site_min'),
      (r'at least '+q+r' of identifiable tested alternatives','scan.site_min'),
      (r'at least '+q+r' of first-generation offspring','dial.inherit_min'),
      (r'(?:>|more than )\s*'+q+r' of heritable cells','majority.dial_min'),
      (r'exceeded '+q+r', meeting the registered endpoint','majority.competition_min'),
    ]
    for entry in ruleset:rule(*entry)
    # Cohort-wide claims check every stated length, not only the first one.
    facts=[B.FACTS['E1.'+str(L)+'.benign'] for L in [16,32]]
    if len({f.value for f in facts})!=1:raise ValueError('E1 benign lengths have different passing counts')
    new_fact('E1.benign.each_length',facts[0].value,facts[0].source,'L in {16,32}, tar=benign, mut=off, start=mix50; per-L count satisfying 0.3<=T_share_end<=0.7; both counts equal', '; '.join(f'L={L}: '+f.evidence for L,f in zip([16,32],facts)))
    rule(r'only '+q+r' of '+q+r' corrected-core benign worlds at each length','E1.benign.each_length E.total')
    rule(r'only '+q+r' of '+q+r' corrected-core worlds at each tested length, short of '+q,'E1.benign.each_length E.total E.required')
    pattern=r'E2: with mutation on, it falls below its start and\s*lies in 0.05–0.4 at 300,000 steps in ≥ (7) of (10)'
    for group,key in [(1,'E2.required'),(2,'E2.total')]:anchored_fact(key,'REVISION_PREREG.md',pattern,'E2 frozen passing-world criterion',group=group)
    rule(r'lie in '+q+r'[–-]'+q+r' at '+q+r' updates in ≥ '+q+r'/'+q,'E2.low E2.high competition.endpoint E2.required E2.total')
    rule(r'against the required '+q+r' of '+q+r'\. At L = '+q,'E2.required E2.total arrival.Z80.L',lambda p:'E2' in p)
    anchored_fact('reset.6502.status','REVISION_PREREG.md',r'A = X = Y = 0, P = (0x20), PC = 0','S1 NMOS 6502 reset processor-status byte')
    rule(r'status `'+q+r'` hexadecimal','reset.6502.status')


def finish_exploratory_bindings():
    q='('+N.replace('[\\w-]','[\\w]')+')'
    s='results/s8086/prelaunch/freeze.json';o=json.loads(read(B.resolve(s)))
    new_fact('schedule.checkpoints',len(o['registration']['snapshot_steps']),s,'registration.snapshot_steps length; initialization included',str(o['registration']['snapshot_steps']))
    s='results/s8086/exploratory/encounter_fates.csv'
    for agg in ['min','max','distinct']:selected_fact('fates.Z80.world.'+agg,s,'world',dict(isa='Z80'),agg)
    steps=sorted({int(r['step']) for r in B.table(s) if r['isa']=='Z80'})
    for i,v in enumerate(steps):new_fact('fates.Z80.step'+str(i),v,s,'isa=Z80; sorted distinct step field ['+str(i)+']',str(steps))
    source='results/s8086/exploratory/first_family.csv';a=B.table(source)
    aliases=[(i,r) for i,r in enumerate(a,2) if 'NMOS' in r['decode']]
    if not aliases:raise ValueError('alias description missing')
    m=re.search(r'NMOS ([0-9a-f]{2})[–-]([0-9a-f]{2}) alias',aliases[0][1]['decode'],re.I)
    if not m:raise ValueError('alias definition missing')
    for i,key in [(1,'alias.min'),(2,'alias.max')]:new_fact(key,int(m[i],16),source,'decode field; world='+aliases[0][1]['world']+'; NMOS opcode alias endpoint',f'CSV line {aliases[0][0]}: '+aliases[0][1]['decode'])
    ex=[(i,r) for i,r in enumerate(a,2) if r['x']=='e9']
    if len(ex)!=1:raise ValueError('JMP rel16 first-family example ambiguous')
    new_fact('family.extra_fetch',max(json.loads(ex[0][1]['physical_fetch_positions'])),source,'x=e9; max physical_fetch_positions; one selected first-family example',f'CSV line {ex[0][0]}: '+ex[0][1]['physical_fetch_positions'])
    pairs=[
      (r'\b'+q+r' worlds ×'+q+r' scheduled samples, '+q+r' ranked tapes per record and '+q+r' worlds ×'+q+r' checkpoints ×'+q+r' cells','S2.worlds schedule.records schedule.top3 S2.worlds schedule.checkpoints soup.cells'),
      (r'using budget '+q+r' and physical entry A','arrival.8086.budget',lambda p:'encounter panel samples' in p),
      (r'panel uses all '+q+r' Stage G L'+q+r' seeds '+q+r'[–-]'+q+r' and snapshots at '+q+r'; '+q+r'; '+q+r'; '+q+r'; '+q+r' updates','fates.Z80.world.distinct route.Z80.L16 fates.Z80.world.min fates.Z80.world.max fates.Z80.step0 fates.Z80.step1 fates.Z80.step2 fates.Z80.step3 fates.Z80.step4'),
      (r'S2 lacks dense checkpoints before '+q+r' updates','schedule.checkpoint1'),
      (r'NMOS `'+q+r'`[–-]`'+q+r'` conditional-jump aliases','alias.min alias.max'),
      (r'exemplar additionally fetches physical byte '+q,'family.extra_fetch'),
      (r'byte-'+q+r' alternative','family.R.0'),
      (r'abundance never reaches its '+q+r'-cell gate','detection.cells'),
    ]
    for entry in pairs:rule(*entry)
    # Actual rendering configuration defines symbol exemplars and axis limits.
    source='manuscript/figures/build_v8.py';t=read(B.resolve(source));tree=ast.parse(t)
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and all(isinstance(x,ast.Name) for x in n.target.elts) and [x.id for x in n.target.elts]==['x','n'] and isinstance(n.iter,ast.List))
    tuples=ast.literal_eval(loop.iter)
    for i,(_,value) in enumerate(tuples):
        new_fact('display.scale'+str(i),value,source,'disc-area legend tuple ['+str(i)+'].n',str(tuples));FACT_META['display.scale'+str(i)]=('definition',f'{label(B.resolve(source))}:{loop.lineno}')
    new_fact('display.scales',len(tuples),source,'disc-area legend exemplar count',str(tuples));FACT_META['display.scales']=('definition',f'{label(B.resolve(source))}:{loop.lineno}')
    rule(r'with '+q+r' scale exemplars '+q+r', '+q+r' and '+q,'display.scales display.scale0 display.scale1 display.scale2')
    for group,key in [(1,'display.arrival.min'),(2,'display.arrival.max')]:anchored_fact(key,source,r'xlim=\((40),(1\.2e7)\)','arrival x-axis display boundary',group=group)
    rule(r'arrival axis spans '+q+r'[–-]'+q,'display.arrival.min display.arrival.max')
    for group,key in [(1,'display.inset.min'),(2,'display.inset.width')]:anchored_fact(key,source,r'Rectangle\(\((0),0\),(10),10','inset zoom box boundary',group=group)
    rule(r'the '+q+r'[–-]'+q+r' enlargement preserves coordinates','display.inset.min display.inset.width')
    selected_fact('dial.maxbyte','manuscript/figures/out/v8/data/dial_coordinates.csv','value',{},'max')
    rule(r'entries run `00`[–-]`(FF)`','dial.maxbyte')
    # In the negated comparison, zero is the mathematical null, not an observed
    # count of all heritable variants. The fixed-footprint model has no admitted
    # nonbaseline payload assignments; the broader founder assay has 245 passes.
    key=anchored_fact('definition.variation_null','manuscript/SI_v8.md',r'fixed-footprint free count (zero)','null cardinality in the logical comparison; not a measured zero-variation result')
    f=B.FACTS[key]
    new_fact(key,f.value,f.source,f.meaning,f.evidence+'; held-footprint product family: 256^0 - 1 = 0 nonbaseline payload assignments; broader founder substitutions: family.passed='+fmt(B.FACTS['family.passed'].value)+' with alternative=True, all16_exact=True, all8_serial_exact=True in results/s8086/exploratory/family_transmission.csv')
    rule(q+r' free payload is again not '+q+r' heritable variation',['family.free',key])


def revised_legend_bindings():
    """Caption noun/field variants, independent of panel letters and line numbers."""
    q='('+N.replace('[\\w-]','[\\w]')+')'
    selected_fact('erasure.encounters','manuscript/figures/out/v8/data/erasure_ordered.csv','encounter',{},'distinct')
    anchored_fact('serial.lineages','serial_retention.py',r'R_MUT = (16)','historical serial mutant starts per alternative')
    anchored_fact('serial.controls','serial_retention.py',r'R_CTRL = (256)','historical serial unmutated control starts')
    # An alive branching population is nonempty: its minimum member count is one.
    source='REVISION_PREREG.md';t=read(B.resolve(source));m=re.search(r'a population is alive at g if it has a\s*member',t,re.I)
    if not m:raise ValueError('branching nonempty-population definition missing')
    new_fact('branch.minimum_members',1,source,'X1 alive iff a member exists; minimum finite cardinality = 1',squash(m[0])+'; exists(member) iff member_count >= 1')
    FACT_META['branch.minimum_members']=('definition',location(B.resolve(source),t,m.start()))
    anchored_fact('branch.site_fraction','REVISION_PREREG.md',r'if, for at least (half) of its tested values','X1 fraction of identifiable variants meeting the excess-carriage criterion')
    pairs=[
      (q+r' lengths L \(bytes\)','emergence.lengths',lambda p:'Stage E detection' in p),
      (q+r' worlds/length','emergence.worlds',lambda p:'Stage E detection' in p),
      (r'censoring at '+q+r' updates','emergence.horizon',lambda p:'Stage E detection' in p),
      (r'budget of '+q+r' execution steps','arrival.budget'),
      (q+r' upper bound','ring_confidence',lambda p:'Ring stability' in p),
      (r'upper bound < '+q+r' changed-memory pairs','ring_rate_bound'),
      (q+r' of members carrying the marker window','carriage.population_min'),
      (q+r' or ≥ '+q+r' of members carrying the marker window','carriage.population_min carriage.population_min'),
      (r'fractions among surviving populations are ≥ '+q+r' for at least '+q+r' the identifiable alternatives','branch.excess branch.site_fraction'),
      (r'score at transfer '+q+r' but','serial.sample0'),
      (r'criterion at transfer '+q,'branch.gens'),
      (r'filled circles score at transfer '+q,'branch.gens'),
      (r'lethal-chemistry L'+q+r' tape','serial.lethal16_s4009.L'),
      (r'L'+q+r' uses seed '+q+r' \('+q+r' in a,b\)','serial.genome20_s6004.L genome20_s6004.seed genome20_s6003.seed'),
      (r'in '+q+r' encounters of '+q+r' execution steps','erasure.encounters erasure.steps_per_encounter'),
      (r'address '+q+r' supplies original-byte-'+q+r' ancestry','erasure.source_read erasure.ancestral_source'),
      (r'overwriting '+q+r' at address '+q,'erasure.marker erasure.address'),
      (q+r' tapes, '+q+r' worlds\)','C2.eligible_tapes C2.eligible_worlds',lambda p:'eligible 6502 roles' in p),
      (r'red point carries '+q+r' positions in targeted transfers but scores '+q,'population.deficit.predicted population.deficit.observed'),
      (q+r' that pass the sampled screen','dial.eligible',lambda p:'256-value scan' in p),
      (r'fetching a '+q+r' opcode halts','chemistry.zero'),
      (r'a '+q+r' opcode executes as no operation','chemistry.zero'),
      (r'≥ '+q+r' byte sites retaining introduced values','class.transmitter'),
      (r'Stage G: '+q+r' worlds \('+q+r' each at L = '+q+r','+q+r','+q+r','+q+r'\)','route.Z80.allworlds route.Z80.L16.worlds route.Z80.L16 route.Z80.L20 route.Z80.L50 route.Z80.L64'),
      (r'Primary 6502 S1 A: '+q+r' L'+q+r' worlds','S1.worlds arrival.6502.L'),
      (r'Coherent-fetch 8086 S2: '+q+r' L'+q+r' worlds[—–-]'+q+r' open, '+q+r' confined, '+q+r' censored','S2.worlds arrival.8086.L S2.open S2.confined S2.rep_censored'),
      (r'confined = '+q,'fetch.confined',lambda p:'S1/S2' in p),
      (r'detection: '+q+r' L'+q+r' worlds/machine','arrival.Z80.worlds arrival.Z80.L'),
      (r'All '+q+r' `5F A4 x y` first-copier tapes \('+q+r' detected worlds\)','family.worlds S2.detected'),
      (r'classes hold in '+q+r'/'+q+r' encounters: '+q+r' one-pass and '+q+r' loop classifications',['family.trace_partners','family.trace_partners','tiles.memory-fed one-pass copier','tiles.confined copy loop']),
      (r'all '+q+r' tapes repeatedly copying mapped ring positions at displacement d = '+q+r' bytes','family.worlds family.geometry.d'),
      (r'at least '+q+r' member remains','branch.minimum_members'),
    ]
    for entry in pairs:rule(*entry)


def rules():
    # Semantic noun/cohort anchors are independent of the manuscript's line numbers
    # and displayed values. A changed numeral still binds to the same source field.
    Q='('+N+')'
    rule(Q+r' of '+Q+r' (?:tested )?(?:mutation types|tested variants|tested mutant types)','pusher16.mutation_types_carried pusher16.n_mutants',lambda p:'representative early' in p or 'pusher16' in p or '16-byte pusher' in p)
    rule(r'(?:none) of '+Q+r' tested variants','pusher16.n_mutants',lambda p:'branching/' in p)
    rule(r'(?P<count>none) of '+Q+r' tested variants',['pusher16.mutation_types_carried','pusher16.n_mutants'],lambda p:'branching/' in p)
    rule(Q+r' of '+Q+r' unmutated populations','pusher16.ctrl_alive_n pusher16.ctrl',lambda p:'pusher' in p)
    rule(Q+r'%? (?:were )?(?:still alive|mutant populations survived)','pusher16.mut_alive',lambda p:'pusher16' in p or '16-byte pusher' in p)
    rule(Q+r' tested mutation types','pusher16.n_mutants',lambda p:'pusher16' in p or '16-byte pusher' in p)
    rule(r'for (none) of','pusher16.mutation_types_carried',lambda p:'16-byte pusher' in p)
    rule(Q+r' of '+Q+r' tested mutant types','pusher64.mutation_types_carried pusher64.n_mutants',lambda p:'64-byte pusher' in p)
    rule(Q+r' (?:scored )?sites through all '+Q+r' generations','genome20_s6003.sites genome20_s6003.gens')
    rule(r'retained '+Q+r' of '+Q+r' claim verdicts','claims.same claims.total')
    rule(r'changed '+Q+r' of '+Q+r' (?:registered )?components','components.changed components.total')
    rule(Q+r' of '+Q+r' claims stand','claims.same claims.total')
    rule(Q+r' of '+Q+r' components change','components.changed components.total')
    rule(r'occurs in '+Q+r' of '+Q+r' sampled confined heritable cells','core.motif core.all')
    rule(r'Of '+Q+r' such cells, '+Q+r' had','population.total population.equal')
    rule(r'In all '+Q+r' cells whose total score','population.extra')
    rule(Q+r' (?:exploratory Z80|exploratory population|such) cells','population.total')
    rule(Q+r' total scored sites','example.sites',lambda p:'5006' in p)
    rule(Q+r' scored transmissible sites','example.sites',lambda p:'5006' in p)
    rule(Q+r' (?:of them )?unexecuted(?: payload positions)?','example.unexecuted',lambda p:'5006' in p)
    rule(r'displacement (?:is )?'+Q,'example.offset',lambda p:'5006' in p)
    rule(r'spans? displacements of '+Q+r' to '+Q,'dial.effective_d.min dial.effective_d.max')
    rule(r'predicted payloads of '+Q+r' to '+Q,'dial.predicted_payload.min dial.predicted_payload.max',lambda p:'alleles_' in p)
    rule(Q+r' (?:archived )?parents','C1.parents',lambda p:'capacity/' in p or 'registered settings' in p)
    rule(Q+r' (?:frozen )?(?:displacement )?settings','C1.edits',lambda p:'capacity/' in p or 'registered settings' in p)
    rule(Q+r' original-value controls','C1.parents',lambda p:'capacity/' in p or 'registered settings' in p)
    rule(Q+r' changed settings','C1.changed_settings',derived=True)
    rule(Q+r' lineages per setting','C1.lineages')
    rule(Q+r' direct cases','C1.direct')
    rule(Q+r' serial rows','C1.serial')
    rule(r'capacities from '+Q+r' to '+Q+r' positions','C1.predicted_free.min C1.predicted_free.max')
    rule(Q+r' (?:values give a )?working replicator','dial.working')
    rule(Q+r' of these match','dial.inherited')
    rule(Q+r' values that pass the sampled eligibility','dial.eligible')
    rule(Q+r' nonworking','dial.nonworking',lambda p:'256' in p)
    rule(Q+r' working (?:values )?outside','dial.excluded',lambda p:'256' in p)
    rule(Q+r' eligible(?: working)?(?:,| values)','dial.eligible',lambda p:'256' in p)
    rule(Q+r' working values','reverse.working',lambda p:'LDDR' in p)
    rule(Q+r' qualifying (?:cases|rows)','controls6502',lambda p:'census6502' in p)
    rule(Q+r' constructed 6502 sampled-carriage cases','controls6502')
    rule(Q+r' sampling roles','C2.eligible',lambda p:'representing' in p or 'eligible' in p)
    rule(Q+r' eligible 6502 (?:roles|worlds|tapes)','C2.eligible')
    rule(Q+r' (?:distinct )?tapes from '+Q+r' worlds','C2.eligible_tapes C2.eligible_worlds',lambda p:'42' in p or 'eligible' in p)
    rule(r'from '+Q+r' worlds and '+Q+r' distinct tapes','C2.worlds C2.tapes',lambda p:'roles' in p and 'evolved6502' in p)
    rule(Q+r' roles from (?:all )?'+Q+r' (?:S1 )?worlds','C2.roles C2.worlds',lambda p:'retention' in p or 'C2' in p)
    rule(Q+r' first-detection roles, '+Q+r' final random-cell draws and '+Q+r' final top-three roles','C2.first C2.random C2.top')
    rule(r'totaling '+Q+r' roles','C2.roles')
    rule(r'Exactly '+Q+r' roles qualify','C2.eligible')
    rule(Q+r' intervention cases and '+Q+r' serial observations','C2.direct C2.serial')
    rule(r'(?:remaining|leaving) '+Q,'S1.remaining',lambda p:'additional registered loop' in p or 'S1' in p and '14/20' in p,derived=True)
    rule(r'Only '+Q+r' met the additional registered loop','S1.first_loop')
    rule(r'(?:median detection time of|median of) '+Q+r' updates','S1.km_t_rep',lambda p:'S1 primary' in p)
    rule(Q+r' first detected replicators were confined','S1.first_confined',lambda p:'S1' in p)
    rule(Q+r' of '+Q+r' first copiers fetched no partner','S2.confined S2.detected')
    rule(r'copiers in '+Q+r' of '+Q+r' worlds','S2.detected S2.worlds',lambda p:'S2' in p)
    rule(r'only '+Q+r' of those '+Q+r' began open','S2.open S2.detected')
    rule(r'the '+Q+r' confined beginnings','S2.confined')
    rule(r'median of '+Q+r' updates','S2.km.rep',lambda p:'coherent-fetch 8086' in p)
    rule(Q+r' worlds censored','S2.rep_censored',lambda p:'S2' in p or '8086' in p)
    rule(Q+r' certificates remained unresolved','L2.unresolved_extractions')
    rule(Q+r' of them at the unmodelled','L2.pop_es')
    rule(r'leaves '+Q+r' unresolved, including '+Q,'L2.unresolved_extractions L2.pop_es',lambda p:'L2' in p)
    rule(r'L2 excludes '+Q+r' for original-partner reads','L2.partner_reads')
    rule(Q+r' culture-positive candidates','L1.culture_positive',lambda p:'registered ledger' in p or 're-extracted all' in p)
    rule(Q+r' scheduled top-three sample records','motif.top')
    rule(Q+r' ranked tape entries','motif.entries')
    rule(Q+r' checkpoint(?:-cell observations| cells| observations)','motif.cells')
    rule(r'(Ten|'+N+r') of the '+Q+r' first copiers','family.worlds S2.detected')
    rule(r'admitted '+Q+r' genotype[–-]world candidates','posthoc.eligible')
    rule(r'(?:are |geometries are )?'+Q+r' full-tape d\s*=\s*'+Q+r' copiers and '+Q+r' d\s*=\s*'+Q+r' (?:regenerators|copiers)',['posthoc.d16','-','posthoc.d4','-'])
    rule(r'admitting '+Q+r' and leaving '+Q+r' excluded and '+Q+r' unresolved','posthoc.eligible posthoc.excluded posthoc.unresolved')
    rule(r'Of '+Q+r' candidates, '+Q+r' become eligible, '+Q+r' excluded and '+Q+r' unresolved','L1.culture_positive posthoc.eligible posthoc.excluded posthoc.unresolved')
    rule(Q+r' (?:genotype|genotypes|cases) (?:panel|from|complete)','posthoc.panel',lambda p:'post hoc' in p or 'posthoc' in p or '2,224,640' in p)
    rule(r'panel (?:represented|of .*? from) '+Q+r' worlds','posthoc.worlds',lambda p:'purposive' in p)
    rule(Q+r' encounters','posthoc.encounters',lambda p:'exploratory/law_assay' in p or '2,224,640' in p)
    rule(Q+r' raw passes among '+Q+r' cases','validation0.unchanged validation0.total')
    rule(Q+r' (?:expected hidden-latch|hardware) divergences','validation0.changed')
    rule(Q+r'/'+Q+r' claim verdicts','claims.same claims.total')
    rule(Q+r'/'+Q+r' claims','claims.same claims.total')
    rule(Q+r'/'+Q+r' components','components.changed components.total')
    rule(r'flags '+Q+r'/'+Q+r' conditions','C1.flagged C1.ringtotal',lambda p:'Core C1 flags' in p)
    rule(r'core C2 '+Q+r'/'+Q,'C2.flagged C2.ringtotal')
    rule(r'flagged '+Q+r'/'+Q+r' and '+Q+r'/'+Q,'C1.flagged C1.ringtotal C2.flagged C2.ringtotal')
    rule(Q+r'/'+Q+r' (?:isolated )?combinations','c1.identical assays.total')
    rule(r'unaffected counts are '+Q+r' and '+Q,'C1.unaffected C2.unaffected',derived=True)
    rule(r'executes '+Q+r' archived/random conditions','impact.conditions')
    rule(Q+r' sample records','motif.top',lambda p:'motif' in p)
    rule(r'(?:cap|amended) (?:was )?\$'+Q,'spend.cap',lambda p:'spend' in p or 'accounting' in p)
    rule(r'bound was \$'+Q,'spend.bound')
    rule(r'reported as \$'+Q,'spend.metered_usd')
    rule(Q+r' in app-attributed charges','spend.metered_usd')
    rule(r'(?:Contains|at) '+Q+r' tape lengths','emergence.lengths')
    rule(r'(?:at|worlds at) '+Q+r' tape lengths','emergence.lengths')
    rule(r'for historical Stage E worlds at '+Q+r' tape lengths','emergence.lengths')
    rule(Q+r' worlds per length','emergence.worlds',lambda p:'Stage E' in p)
    rule(r'censored at '+Q+r' updates','emergence.horizon',lambda p:'Stage E' in p)
    protocol_rules()


def match_fact(row,key,derived=False,extra=''):
    f=B.FACTS.get(key)
    if not f:return False
    if B.resolve(f.source).resolve()==row['_doc'].path.resolve():return False
    c=B.Claim(0,0,row['value'],row['_value'],'',0,'','',[],[],kind='number')
    ok,rounding=B.displayed_equal(c,f)
    row.update(fact_key=key,denominator=denominator(key,f),source_value=fmt(f.value),correct_value='' if ok else fmt(f.value),status='derived' if ok and derived else 'rounding' if ok and not rounding.startswith('exact') else 'verified' if ok else 'mismatch',rounding=rounding if ok else '',meaning=f.meaning,evidence_source=f.source,evidence=(f.evidence or f.meaning)+(('; '+extra) if extra else ''))
    if key in FACT_META:
        classification,where=FACT_META[key]
        row['evidence_source']=where
        if ok and classification in ('definition','derived') and rounding.startswith('exact'):row['status']=classification
    return True


def protocol_rules():
    """Definitions/protocols shared by main text, SI and the short captions.

    Rules select nouns and source families, never the displayed numeral. A
    substituted threshold or sample size therefore yields a mismatch.
    """
    # Compound numerical words must work in generation-eight and four-byte.
    q='('+N.replace('[\\w-]','[\\w]')+')'
    capacity=lambda p:'results/capacity/' in p
    retention=lambda p:'results/evolved6502/' in p
    branch=lambda p:'branching' in p.lower() or 'branching/' in p or 'Figure 2' in p
    serial=lambda p:'serial_' in p or 'Historical serial' in p or 'historical serial' in p
    literal=lambda p:'literal_channel/' in p
    s2=lambda p:'s8086/' in p or 'Coherent-fetch 8086' in p
    geometry=lambda p:'geometry_units.csv' in p or 'Forward overlap' in p or 'Forward overlapping' in p
    for scope,prefix in [(capacity,'C1'),(retention,'C2')]:
        rule(q+r' (?:serial )?lineages',prefix+'.lineages',scope)
        rule(r'(?:generation[ -]|through generation |for |through |and )'+q+r' (?:generations|transfers|successive fresh-partner passages)',prefix+'.generations',scope)
        rule(r'generation[ -]'+q,prefix+'.generations',scope)
        rule(q+r' (?:joint assignments|joint cases)',prefix+'.joints',scope)
        rule(r'\ball '+q+r' alternatives',prefix+'.alternatives',scope)
        rule(r'\ball '+q+r' frozen partners',prefix+'.partners' if prefix=='C1' else 'REP.17',scope)
        rule(q+r' census partners',prefix+'.partners' if prefix=='C1' else 'REP.17',scope)
        rule(r'(?:zero plus |zero\+|zero and )'+q+r' uniform',prefix+'.uniform_partners' if prefix=='C1' else 'C2.partners',scope)
    rule(r'displacement byte was changed from `'+q+r'` to `'+q+r'` or to `'+q+r'`','old_edit.original old_edit.value8 old_edit.value96')
    rule(q+r' transfers','old_edit.gen',lambda p:'sr_mutants.csv.gz' in p and 'val 8' in p)
    rule(r'registered branching.*?'+q+r' alternatives/site, '+q+r' populations/alternative and '+q+r' controls','branch.values branch.reps branch.ctrl')
    rule(q+r' (?:seeded )?alternatives (?:at every position|per (?:site|position)|/site)','branch.values',lambda p:branch(p) and not serial(p))
    rule(q+r' starting populations per alternative','branch.reps',branch)
    rule(q+r' (?:unmutated populations|unmutated controls|controls per design)','branch.ctrl',lambda p:branch(p) and not serial(p))
    rule(q+r' (?:fresh partners|fresh partners per generation)','branch.k',branch)
    rule(r'cap(?:ped|ping)? (?:at|to) '+q+r' members','branch.cap',branch)
    rule(q+r' offspring qualify, '+q+r' are sampled','branch.cap branch.cap',branch)
    rule(r'(?:Generation[ -]|generation[ -]|through |for |and )'+q+r'(?: bars| generations| transfers| descendants)','branch.gens',lambda p:branch(p) and not capacity(p) and not retention(p))
    rule(q+r' populations/alternative','branch.reps',branch)
    rule(q+r' controls per design','branch.ctrl',branch)
    rule(r'Historical serial[^.]*?'+q+r' alternatives/site','serial.pusher16.alternatives')
    rule(q+r' starts/alternative','serial.lineages',serial)
    rule(q+r' lineages per alternative','serial.lineages',serial)
    rule(q+r' unmutated controls','serial.controls',serial)
    rule(r'The panel uses '+q+r' mutant lineages, '+q+r' controls','serial.lineages serial.controls')
    rule(q+r' starts/alternative, '+q+r' transfers and '+q+r' controls','serial.lineages branch.gens serial.controls')
    rule(q+r' reference encounters','serial.controls',lambda p:False)  # separately sourced below
    rule(r'(?:at least |fewer than |least )'+q+r' (?:of (?:its|their|the) (?:members|identifiable|tested)|its identifiable)','branch.half',branch)
    rule(q+r'-byte window','branch.window',branch)
    rule(r'≥\s*'+q+r'(?=%? (?:best|whole))','match_gate')
    rule(r'(?:below |under |≥\s*)'+q+r'(?:%? whole-tape|%? best-cyclic|%? whole tape|%? of its positions)','match_gate')
    rule(r'≥\s*'+q+r' marker carriage','branch.half',branch)
    rule(r'<\s*'+q+r' marker carriage','branch.half',branch)
    rule(r'by ≥\s*'+q,'branch.excess',branch)
    rule(r'(?:gain|score) (?:≥\s*|at least )'+q,'gen2_minimum')
    rule(r'before-match below '+q,'match_gate')
    rule(q+r' uniform partners per generation','culture.partners',s2)
    rule(r'generation[ -]'+q+r' (?:normalised|normalized|score|gain)','culture.generation')
    rule(q+r'-partner generation-'+q+r' rule','culture.partners culture.generation',s2)
    rule(q+r' uniform-byte partners','C2.partners',retention)
    rule(q+r' (?:raw passes|raw passes among)','validation0.unchanged')
    rule(r'(none) (?:is|were) unexpected','validation.unexpected',lambda p:'hardware' in p)
    rule(q+r' validation comparisons','validation.units')
    rule(q+r'% upper bound','ring_confidence')
    rule(r'rate below '+q,'ring_rate_bound')
    rule(q+r' (?:historical worlds of the registered benign|worlds of the registered benign)','route.Z80.worlds')
    rule(r'(?:Two|'+N+r') regenerators',['branch.regenerators'],branch)
    rule(r'(Three|'+N+r') confined Z80 replicators','branch.confined_designs',branch)
    rule(r'('+N+r') returning control','branch.return_design',branch)
    rule(r'('+N+r') using a block copy','branch.block_design',branch)
    rule(r'(?:the |a |in the )'+q+r'-byte core','theorem.core',lambda p:'Z80' in p or '5e ed b0' in p)
    rule(r'r\s*=\s*d\s*−\s*'+q,'theorem.core',lambda p:'Z80' in p or 'Note 1' in p)
    rule(r'd\s*−\s*'+q+r' scored','theorem.core',lambda p:'offset_cells.csv' in p)
    rule(r'repeats every '+q+r' bytes','example.parent_period',lambda p:'5006' in p)
    rule(r'(One|'+N+r') evolved tape repeats','example.cells',lambda p:'5006' in p)
    rule(r'(?:For |on two )L\s*=\s*'+q,'geometry.L',geometry)
    rule(r'c\s*=\s*'+q,'geometry.c',geometry)
    rule(r'd\s*=\s*'+q+r' regenerates','geometry.dmin',geometry)
    rule(r'd\s*=\s*'+q+r' preserves','geometry.dmax',geometry)
    rule(r'preserves '+q+r' payload positions','geometry.free',geometry)
    rule(r'(?:boxes|box) represent(?:s)? '+q+r' bytes','geometry.bytes_per_unit',geometry)
    rule(r'no free payload',( ),lambda p:False)
    rule(q+r' tested variants','pusher16.n_mutants',lambda p:'representative early' in p)
    rule(r'survived '+q+r' generations','pusher16.gens',lambda p:'representative early' in p or '16-byte pusher' in p)
    rule(q+r'-byte pusher','pusher16.L',lambda p:'48 of 64' in p)
    rule(r'at generation '+q,'pusher16.gens',lambda p:'16-byte pusher' in p)
    # The negative counts use the literal zero/none vocabulary, so unrelated
    # occurrences of e.g. a zero partner do not bind to mismatch counts.
    for scope,key in [(capacity,'C1.bad'),(retention,'C2.direct_bad'),(lambda p:'law_assay' in p,'posthoc.bad')]:
        rule(r'(zero|none) mismatches',key,scope)
    rule(r'(?:medians, with|with) '+q+r' worlds per direction','competition.benign.R_into_T.worlds',lambda p:'invasion directions' in p)
    rule(q+r' and '+q+r' under benign chemistry','competition.benign.R_into_T competition.benign.T_into_R',lambda p:'invasion directions' in p)
    rule(q+r' and '+q+r' under lethal chemistry','competition.lethal.R_into_T competition.lethal.T_into_R',lambda p:'invasion directions' in p)
    rule(r'median .*?shares.*?were '+q+r' under benign chemistry','competition.benign.mix50',lambda p:'started with equal shares' in p)
    rule(r'and '+q+r' under lethal chemistry','competition.lethal.mix50',lambda p:'started with equal shares' in p)
    rule(r'All '+q+r' lethal worlds exceeded','competition.lethal.mix50.majority')
    rule(q+r' of '+q+r' fully lethal worlds','lethal.C1.T_majority lethal.C1.worlds')
    rule(r'against '+q+r' of '+q+r' historically','lethal.Historical.T_majority lethal.Historical.worlds',lambda p:'fully lethal' in p)
    rule(q+r' of '+q+r' worlds at L\s*=\s*'+q,'ladder.G.16.regenerator ladder.G.16.worlds population.G16.L',lambda p:'regenerators were the final majority' in p)
    rule(r'and '+q+r' of '+q+r' at L\s*=\s*'+q,'ladder.G.20.regenerator ladder.G.20.worlds population.G20.L',lambda p:'regenerators were the final majority' in p)
    rule(r'Among the '+q+r' benign worlds containing both classes.*?in '+q,'order.both order.before')
    rule(r'(?:share|shares) (?:at least|≥)\s*'+q,'schedule.share_gate',s2)
    rule(q+r' scheduled records','schedule.records',s2)
    rule(r'(?:budget |budget is |budget fixed at )'+q,'branch.budget',lambda p:'branching' in p)
    rule(q+r'-instruction (?:budget|encounters)','branch.budget',lambda p:'Encounters have' in p or 'exploratory lineage' in p)
    rule(r'at zero-based (?:execution |instruction )?step '+q,'erasure.step',lambda p:'marker' in p)
    rule(r'address '+q+r' supplies ancestry','erasure.source_read')
    rule(r'original byte '+q,'erasure.ancestral_source',lambda p:'marker' in p)
    rule(r'marker '+q+r' at address '+q,'erasure.marker erasure.address')
    rule(r'(?:execution |cumulative )?step '+q+r'[^.]*?processor resets','erasure.steps_per_encounter',lambda p:'marker' in p)
    rule(r'lineage (zero)','ldir32_v04.lineage',lambda p:'generation-eight' in p)
    rule(r'transmitting '+q+r' payload positions','ldir32_v60.transmitted',lambda p:'0x60' in p and '96' in p)
    rule(q+r' distinct tapes from '+q+r' worlds','C2.eligible_tapes C2.eligible_worlds',lambda p:'6502 roles' in p or 'roles qualify' in p)
    rule(q+r' roles qualify','C2.eligible',retention)
    rule(r'footprints spanned '+q+r'[–-]'+q+r' bytes','C2.core_c.min C2.core_c.max',retention)
    rule(r'predicted payloads '+q+r'[–-]'+q+r' positions','C2.predicted_free.min C2.predicted_free.max',retention)
    rule(r'case.*?'+q+r' (?:controls|byte/word REP controls)','REP.controls',lambda p:False)
    rule(r'All '+q+r' (?:byte/word REP )?controls','REP.controls')
    rule(q+r' encounters','REP.encounters',lambda p:'constructed_controls.csv' in p and not 'law_assay' in p)
    rule(q+r' geometry/operation/budget/partner rows','census.law')
    rule(q+r' (?:fixed-input census rows|geometry/partner contexts)','census.law')
    rule(r'all '+q+r' period-one (?:and|word)','census.words1',lambda p:'census8086' in p)
    rule(q+r' period-two (?:words|word)','census.words2',lambda p:'census8086' in p)
    rule(r'covers '+q+r' settings and '+q+r' encounters','census.settings census.encounters')
    rule(q+r' labels changed','census.changed')
    rule(r'Exactly '+q+r' labels change','census.changed')
    rule(r'(?:all|in all) '+q+r' audited contexts','witness.contexts')
    rule(r'all '+q+r' random-partner contexts','witness.open_random')
    rule(q+r'/'+q+r' random partners at L16','pusher8086.L16 census8086.random_partners')
    rule(q+r'/'+q+r' at L32','pusher8086.L32 census8086.random_partners',lambda p:'bootstraps' in p)
    rule(r'(?<!-)'+q+r' sample records','motif.top',lambda p:'motif archive' in p)
    rule(r'20×585='+q,'motif.top',derived=True)
    rule(r'11,700×3='+q,'motif.entries',derived=True)
    rule(r'20×5×20,000='+q,'motif.cells',derived=True)
    # Definition and theorem coefficients retain their specifically cited facts.
    rule(q+r'L bytes','theorem.ring',lambda p:'ring of' in p)
    rule(q+r' tapes of L bytes','encounter.tapes')
    rule(r'over '+q+r' partners','entropy.outputs',lambda p:'entropy' in p)
    rule(r'(?:=|ceiling is.*?=)'+q+r' bits','entropy.bits',lambda p:'entropy' in p)
    rule(r'width[ -]'+q,'literal.width1',lambda p:literal(p) and 'Width-one mixtures' in p)
    rule(q+r' nontrivial period-two Z80','literal.Z80.all',literal)
    rule(r'(three)\s*,? (?:not five|of the five)','literal.Z80.literal',literal)
    rule(r'Both .*?'+q+r' core bytes','REP.core',lambda p:'LAW_CONTRACT.md' in p)
    rule(q+r'-byte REP family','REP.core')
    rule(r'at most '+q+r' eligible genotype-world','schedule.cases_per_family_cap',s2)
    rule(q+r' uniformly sampled final cells','schedule.final_cells',s2)
    rule(r'final top three and '+q,'schedule.final_cells',s2)
    rule(q+r' independent (?:seeded )?joint assignments','posthoc.joint_assignments',lambda p:'EXPLORATORY' in p or 'law_assay' in p)
    rule(q+r' fresh-partner serial transfers','posthoc.serial_generations',lambda p:'EXPLORATORY' in p or 'law_assay' in p)
    rule(q+r' source records, '+q+r' distinct genotype-world candidates and '+q+r' culture-positive candidates','L2.source_records L2.unique_candidates L2.culture_positive')
    rule(q+r' single assignments over '+q+r' genotype-coordinate pairs','posthoc.sum.single_assignments posthoc.coordinates')
    rule(q+r' joint assignments and '+q+r' erasure assignments','posthoc.sum.joint_assignments posthoc.sum.erased_assignments')
    rule(q+r' historical launch and accounting files','archive.files')
    rule(q+r' layouts for widths','literal.layouts')
    rule(q+r' width-one count cases','literal.width1.cases')
    rule(q+r' token orders through period '+q,'literal.direct.width1_orders literal.direct.max_period')
    rule(r'All '+q+r' replays are retained','literal.replays')
    rule(r'budgets '+q+r', '+q+r' and '+q,'literal.replay.budget0 literal.replay.budget1 literal.replay.budget2',lambda p:'finite_harness.json' in p)
    rule(q+r' assignments, including '+q+r' non-baseline alternatives','family.assignments family.alternatives')
    rule(q+r' alternatives pass every direct and serial challenge','family.passed')
    rule(q+r' byte copies in '+q+r' steps','family.copies soup.z80_steps',lambda p:'5f a4' in p.lower())
    rule(q+r' pairs per core','impact.total_pairs_per_core')
    rule(q+r' confirmatory pairs each','impact.pairs_per_condition')
    rule(r'n\s*=\s*'+q+r' draws','impact.pairs_per_condition',lambda p:'Clopper' in p)
    rule(q+r' output fields','impact.assay_score_columns',lambda p:'isolated-assay' in p)
    rule(q+r' additional source-table cases','impact.local_assay_combinations')
    rule(q+r' cells','soup.cells',lambda p:'lattice' in p or 'all core-bearing cells divided' in p or 'illustrative' in p)
    rule(q+r' × '+q+r' lattice','soup.width soup.height')
    rule(q+r'×'+q+r'='+q+r' cells','soup.width soup.height soup.cells',derived=True)
    rule(q+r' ordered pair slots','soup.pair_count')
    rule(q+r' drawn pair slots','soup.pair_count')
    rule(q+r' drawn pair draws','soup.pair_count')
    rule(r'Each update draws '+q,'soup.pair_count')
    rule(q+r' uniformly located byte replacements','soup.mutation_attempts')
    rule(q+r' uniform byte-replacement attempts','soup.mutation_attempts')
    rule(r'noise exponent '+q,'soup.noise_exp')
    rule(r'baseline encounter budget is '+q,'soup.z80_steps')
    rule(r'copies into soup partners in '+q+r' of challenges versus '+q+r' into uniform partners','fates.b951.soup.A fates.b951.uniform.A')
    rule(r'receiving role in only '+q+r' versus '+q,'fates.b951.soup.B fates.b951.uniform.B')
    rule(r'active-copy success '+q+r' versus '+q,'fates.51b9.soup.A fates.51b9.uniform.A',lambda p:'opposite' in p)
    rule(r'seed '+q+r', (?:contains|containing)','world.seed',lambda p:'illustrative' in p)
    rule(r'world, seed '+q,'world.seed',lambda p:'illustrative' in p)
    rule(r'illustrative L\s*=\s*'+q,'world.L')
    rule(r'others use '+q,'serial.pusher16.alternatives',lambda p:'L50 pusher' in p)
    rule(r'L50 pusher uses '+q+r' alternatives/site','serial.pusher50.alternatives')
    rule(r'L20 transmitter (?:uses|is) seed '+q+r', (?:versus|rather than) '+q,'genome20_s6004.seed genome20_s6003.seed')
    rule(q+r' registered settings','C1.edits')
    rule(q+r'-value (?:byte-zero )?scan','dial.total')
    rule(r'mutation-on competitions (?:run )?(?:through|to) '+q,'competition.endpoint')
    rule(r'mutation-on runs through '+q,'competition.endpoint')
    rule(r'divided by all '+q+r' cells','competition.cells')
    rule(r'censored at '+q+r'(?:\.| updates)','emergence.horizon',lambda p:'lethality dial' in p or 'Random-code' in p)
    rule(r'Horizons are '+q+r' Z80, '+q+r' 6502 and '+q+r' 8086','arrival.Z80 G.horizon arrival.6502 A.horizon arrival.8086 S2.horizon')
    rule(r'worlds/machine at budget '+q,'arrival.budget')
    rule(r'six censored at '+q+r' updates','S2.horizon',s2)
    rule(r'S2.*?seeds '+q+r'[–-]'+q+r', through '+q,'S2.seed.min S2.seed.max S2.horizon',s2)
    rule(r'Seeds '+q+r'[–-]'+q+r' were reused across arms','S1.seed.min S1.seed.max')
    rule(r'four arms of '+q+r' worlds, each run for '+q,'S1.worlds arrival.6502 A.horizon')
    rule(r'complete checkpoints at '+q+r', '+q+r', '+q+r', '+q+r' and '+q,'schedule.checkpoint0 schedule.checkpoint1 schedule.checkpoint2 schedule.checkpoint3 S2.horizon',s2)
    rule(r'budget fixed at '+q,'branch.budget',retention)
    rule(r'intervention budget (?:is |was )?(?:fixed at )'+q,'branch.budget',retention)
    rule(r'C2 used budget '+q,'branch.budget')
    rule(q+r' first detections, '+q+r' final random draws and '+q+r' final top-three roles','C2.first C2.random C2.top')
    rule(r'It has '+q+r' distinct tapes and '+q+r' eligible roles','C2.tapes C2.eligible')
    rule(q+r' candidate settings','census.settings',lambda p:'descriptive correction' in p)
    rule(q+r' confined.*?and six censored','S2.confined',lambda p:False)
    rule(r'LL?\s*≤\s*'+q,'theorem.Lmax',lambda p:'MAIN_nature.md' in p or 'Proposition 8' in p)
    rule(r'leave(?:s)? HL\s*≡\s*'+q,'theorem.HL')
    rule(q+r'D\s*≡','theorem.Dscale',lambda p:'protected prefix' in p)
    rule(r'D\s*≡\s*'+q,'theorem.Dzero',lambda p:'protected prefix' in p)
    rule(r'BC (zero) or at least '+q,'theorem.BCzero theorem.moves')
    rule(q+r' block moves remain','theorem.moves')
    rule(r'form leaves '+q,'theorem.immediate',lambda p:'1e NN ed b0' in p)
    rule(r'conditional payload r\s*=\s*d\s*−\s*'+q,'theorem.core',lambda p:'Z80' in p)
    rule(r'at least '+q+r' bits','entropy.bits',lambda p:False)
    rule(r'an '+q+r'-bit sample ceiling','entropy.bits')
    rule(q+r' outputs and an','entropy.outputs',lambda p:'entropy' in p)
    for key in ('C1.bad','C2.direct_bad','posthoc.bad','REP.bad'):
        scope=capacity if key=='C1.bad' else retention if key=='C2.direct_bad' else (lambda p:'law_assay' in p) if key=='posthoc.bad' else (lambda p:'constructed_controls.csv' in p)
        rule(r'(no|zero) mismatch(?:es)?',key,scope)
    rule(r'(no) eligible evolved member','L1.eligible')
    rule(r'(no) nonzero exact zero-partner self-writer','census6502.zero')
    rule(r'period '+q+r' to '+q+r' at L\s*=\s*'+q+r' under the declared '+q+r'- and '+q+r'-step arms','census6502.k.min census6502.k.max census6502.L.min census6502.budget.min census6502.budget.max')
    rule(r'FLAGS is `(F[0-9a-f]{3})`','reset.flags')
    # Local set counts are derived from the explicit enumeration, not assumed
    # to be independent experiment denominators.


def apply_rules(rows):
    byblock=defaultdict(list)
    for r in rows:byblock[(r['file'],r['_pstart'])].append(r)
    for group in byblock.values():
        p=group[0]['_p']
        for pattern,keys,scope,derived in RULES:
            if scope and not (scope(p) or scope(p.lower())):continue
            for m in pattern.finditer(p):
                for i,key in enumerate(keys,1):
                    if key=='-' or i>m.re.groups:continue
                    a,b=m.span(i)
                    row=next((r for r in group if r['_local']==a and r['_local']+len(r['value'])==b and r['kind']=='quantity'),None)
                    if row and not (key in FACT_META and FACT_META[key][0]=='definition' and FACT_META[key][1].startswith(row['file']+':')):match_fact(row,key,derived)


def transfers(rows,old,resolution):
    def frame(s):
        return ['#' if x[0].isdigit() else x for x in re.findall(r'[a-z]+|\d+(?:\.\d+)?|[/=+−]',s.lower())]
    templates=[]
    for i,c in enumerate(old,1):
        rid=f'N{i:04d}'
        if not c.fact_key or c.fact_key not in B.FACTS:continue
        local=c.offset-c.pstart
        # Retain distinctive terms on both sides of this occurrence. Matching does
        # not compare numerals; changing a numeral must not disable the binding.
        t=c.paragraph[max(0,local-65):c.end-c.pstart+85]
        pre=frame(c.paragraph[max(0,local-45):local])[-4:]
        post=frame(c.paragraph[c.end-c.pstart:c.end-c.pstart+55])[:4]
        templates.append((c.fact_key,B.terms(t),rid,c.all_paths,resolution.get(rid),pre,post))
    for r in rows:
        if r['kind']!='quantity' or r['fact_key']:continue
        want=B.terms(r['context'])
        if len(want)<4:continue
        candidates=[]
        pre=frame(r['_p'][max(0,r['_local']-45):r['_local']])[-4:]
        post=frame(r['_p'][r['_local']+len(r['value']):r['_local']+len(r['value'])+55])[:4]
        for key,got,rid,paths,res,oldpre,oldpost in templates:
            if not set(paths)&set(r['_paths']):continue
            # A bag of paragraph words is insufficient to identify which nearby
            # number is meant. Require the immediate phrase on both sides.
            if len(pre)<2 or len(post)<2 or pre[-2:]!=oldpre[-2:] or post[:2]!=oldpost[:2]:continue
            shared=len(want&got)
            score=shared/max(len(want),len(got),1)
            if shared>=5 and score>=.7:candidates.append((score,key,rid,res))
        candidates.sort(reverse=True)
        if candidates:
            best=candidates[0]
            rival=next((c for c in candidates[1:] if c[1]!=best[1]),None)
            if rival and best[0]-rival[0]<.1:continue
            if best[3] and best[3][0]=='Correction':continue
            # A fuzzy transfer is a verification aid, not sufficient evidence
            # to prescribe a changed value. Explicit noun/field rules and
            # arithmetic produce authoritative mismatches independently.
            claim=B.Claim(0,0,r['value'],r['_value'],'',0,'','',[],[])
            if not B.displayed_equal(claim,B.FACTS[best[1]])[0]:continue
            if match_fact(r,best[1],bool(best[3] and best[3][0]=='Derivation'),f'v7 {best[2]} context transfer; score={best[0]:.3f}'):
                r['resolution_ids']=best[2] if best[3] else ''


def arithmetic_value(expression):
    s=expression.replace(',','').replace('×','*').replace('−','-').replace('^','**')
    s=re.sub(r'\bmod\b','%',s)
    node=ast.parse(s,mode='eval')
    def visit(n):
        if isinstance(n,ast.Expression):return visit(n.body)
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)):return Decimal(str(n.value))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -visit(n.operand)
        if isinstance(n,ast.BinOp):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
            if isinstance(n.op,ast.Div):return a/b
            if isinstance(n.op,ast.Mod):return a%b
            if isinstance(n.op,ast.Pow) and abs(b)<100:return a**int(b)
        raise ValueError('not a numeric expression')
    return visit(node)


def arithmetic(rows):
    byblock=defaultdict(list)
    for r in rows:byblock[(r['file'],r['_pstart'])].append(r)
    num=r'\d+(?:,\d{3})*(?:\.\d+)?'
    expr=re.compile(r'(?<![\w.])((?:\(?'+num+r'\)?\s*(?:[+−×*/^]|\bmod\b)\s*)+(?:\(?'+num+r'\)?))\s*=\s*('+num+r')(?!\w|\.\d)')
    for group in byblock.values():
        p=brackets_removed(group[0]['_p'])
        # Removing brackets changes offsets; evaluate on original and skip source
        # annotations rather than constructing locations from modified text.
        p=group[0]['_p']
        for m in expr.finditer(p):
            result=next((r for r in group if r['_local']==m.start(2) and r['kind']=='quantity'),None)
            if not result:continue
            try:value=arithmetic_value(m[1])
            except (ValueError,SyntaxError,ArithmeticError):continue
            operands=[r for r in group if m.start(1)<=r['_local']<m.end(1) and r['kind']=='quantity']
            supported=bool(operands) and all(r['fact_key'] and r['status'] in ('verified','definition','derived','rounding') for r in operands)
            if not supported:continue
            ok=Decimal(result['normalized_value'])==value
            result.update(status='derived' if ok else 'mismatch',fact_key='arithmetic:'+re.sub(r'\s+','',m[1]),source_value=fmt(value),correct_value='' if ok else fmt(value),meaning='arithmetic on independently bound operands',evidence_source='; '.join(dict.fromkeys(r['evidence_source'] for r in operands)),evidence=f'{m[1]} = {fmt(value)}; operands: '+', '.join(r['fact_key']+'='+r['source_value'] for r in operands),denominator='same units as the bound operands')
    # Mandatory examples and commonly repeated complements have independently
    # sourced operands, rather than trusting a printed equation.
    for r in rows:
        if r['kind']!='quantity':continue
        left=r['_p'][max(0,r['_local']-45):r['_local']]
        if re.search(r'remaining\s*$',left,re.I) and 'S1' in r['_p'] and 'loop' in r['_p']:match_fact(r,'S1.remaining',True)
        if re.search(r'(?:six|\d+) coincident 8086',r['context'],re.I) and re.search(r'(?:six|\d+)\s*$',left+r['value'],re.I):
            if r['value'].lower()=='six':match_fact(r,'S2.remaining',True)


def finish_rows(rows):
    structural=Counter()
    for i,r in enumerate(rows,1):
        r['id']=f'N{i:05d}'
        if r['kind']!='quantity':
            structural[r['kind']]+=1
            r.update(status='verified',meaning='structural/lexical occurrence; not an empirical quantity',denominator='not applicable',evidence_source=r['file'],evidence=r['kind']+' inventory; references/targets checked separately')
            continue
        if r['fact_key']:continue
        p=r['_p'];local=r['_local'];left=p[max(0,local-45):local];right=p[local+len(r['value']):local+len(r['value'])+90]
        if r['_attached'] and all(not B.resolve(p).exists() for p in r['_attached']):
            r.update(status='source missing',meaning='all attached source paths are absent')
            continue
        r['meaning']='No unambiguous field, derivation or independent defining-line binding; requires source resolution.'
    return structural


def citations(docs):
    entries=defaultdict(list);seen=[];calls=[];findings=[]
    for doc in docs[:2]:
        in_refs=False
        for line in doc.text.splitlines(keepends=True):
            if re.match(r'^##\s+References\s*$',line):in_refs=True;continue
            if in_refs and line.startswith('## '):in_refs=False
            m=re.match(r'^\s*(\d+)\.\s+(.+)',line)
            if in_refs and m:entries[int(m[1])].append((doc.path,m[2]))
        refstart=re.search(r'^## References\s*$',doc.text,re.M)
        body=doc.text[:refstart.start()] if refstart else doc.text
        for b in BRACKET.finditer(body):
            if not re.fullmatch(r'\s*\d+(?:\s*[,–-]\s*\d+)*\s*',b[1]):continue
            if b.start() and body[b.start()-1].isalnum():continue  # A[1], K[0]
            if re.search(r'(?:ancestry|source map|coordinates)\s*(?:is|=|:)\s*$',body[max(0,b.start()-35):b.start()],re.I):continue
            for n in expand_numbers(b[1]):
                calls.append({'location':location(doc.path,doc.text,b.start()),'citation':str(n),'status':'resolved' if n in entries else 'missing entry'})
                if n not in seen:seen.append(n)
    for n,values in sorted(entries.items()):
        if len(values)>1:findings.append(f'Duplicate reference entry {n}: {len(values)} entries')
        if n not in seen:findings.append(f'Uncited reference entry [{n}]')
    for c in calls:
        c['status']='resolved' if int(c['citation']) in entries else 'missing entry'
        if c['status']!='resolved':findings.append(c['location']+': missing ['+c['citation']+']')
    expected=list(range(1,len(entries)+1))
    if sorted(entries)!=expected:findings.append(f'Reference numbering is not contiguous: {sorted(entries)}')
    if seen!=expected:findings.append(f'First citation order {seen}; expected {expected}')
    urls=[]
    for doc in docs[:2]:
        for m in re.finditer(r'https?://[^\s<>`\]]+',doc.text):
            url=m[0].rstrip('.,;')
            # Markdown destinations can contain legitimate DOI parentheses.
            while url.endswith(')') and url.count(')')>url.count('('):url=url[:-1]
            issue=''
            try:
                parts=urlsplit(url)
                if not parts.hostname or '.' not in parts.hostname or parts.username or parts.password:issue='malformed host'
                if re.search(r'[\s\\]|%(?![0-9a-fA-F]{2})',url):issue='malformed URL characters/escape'
                if parts.hostname in ('doi.org','dx.doi.org') and not re.fullmatch(r'/10\.\d{4,9}/[^\s]+',parts.path):issue='malformed DOI'
                _=parts.port
            except ValueError:issue='malformed URL'
            urls.append({'location':location(doc.path,doc.text,m.start()),'URL':url,'status':issue or 'syntax valid; not fetched'})
        for m in re.finditer(r'(?i)\bdoi\s*:\s*([^\s]+)',doc.text):
            doi=m[1].rstrip('.,;')
            if not re.fullmatch(r'10\.\d{4,9}/[^\s]+',doi):urls.append({'location':location(doc.path,doc.text,m.start()),'URL':doi,'status':'malformed DOI'})
    return entries,seen,calls,findings,urls


def expand_numbers(s):
    out=[]
    for part in re.split(r'\s*,\s*',s):
        m=re.fullmatch(r'\s*(\d+)\s*[–−-]\s*(\d+)\s*',part)
        if m:
            a,b=map(int,m.groups());out.extend(range(a,b+1) if b>=a else (a,b))
        else:
            m=re.match(r'\s*(\d+)',part)
            if m:out.append(int(m[1]))
    return out


def figures(docs,other):
    v8=HERE/'figures/out/v8';alternate=other/'experiments/manuscript/figures/out/v8'
    notes=set(int(m[1]) for m in re.finditer(r'^## Supplementary Note (\d+)',docs[1].text,re.M))
    tables=set(int(m[1]) for m in re.finditer(r'\*\*Supplementary Table (\d+)',docs[1].text))
    # The generated-record index explicitly defines Supplementary Tables 3–15.
    tableblock=re.search(r'\| (?:Supplementary )?Table\b[\s\S]*?(?=\n\n|\Z)',docs[1].text,re.I)
    for start,p,section in blocks(docs[1]):
        if p.startswith('|') and 'Generated' in p and ('| 3 |' in p or '| 15 |' in p):tables.update(int(m[1]) for m in re.finditer(r'^\| (\d+) \|',p,re.M))
    inventory=[]
    panels={}
    legends=next((d for d in docs if d.role=='legends'),docs[0])
    for n in range(1,6):
        match=re.search(r'\*\*Figure '+str(n)+r'\.[\s\S]*?(?=\*\*Figure \d|\n## |\Z)',legends.text)
        panels[n]=set(re.findall(r'\*\*([a-z]),\*\*',match[0])) if match else set()
        inventory.append({'figure':str(n),'panels':','.join(sorted(panels[n])),'primary':'; '.join(label(p) for p in sorted(v8.glob(f'fig{n}.*')) if p.suffix in ('.pdf','.png','.svg')),'worktree':'; '.join(str(p) for p in sorted(alternate.glob(f'fig{n}.*')) if p.suffix in ('.pdf','.png','.svg'))})
    ed=[]
    requested={}
    for m in re.finditer(r'manuscript/figures/out/v8/(ed8_(\d+)[\w-]*\.pdf)',docs[1].text):requested[int(m[2])]=m[1]
    for n in sorted(set(range(1,10))|set(requested)):
        primary=sorted(p for p in v8.glob(f'ed8_{n}*') if re.match(r'ed8_'+str(n)+r'(?:_|\.)',p.name))
        alt=sorted(p for p in alternate.glob(f'ed8_{n}*') if re.match(r'ed8_'+str(n)+r'(?:_|\.)',p.name))
        name=requested.get(n,f'ed8_{n}.pdf')
        pdfs=[p for p in primary+alt if p.suffix=='.pdf']
        ed.append({'item':str(n),'requested':name,'primary':'; '.join(label(p) for p in primary),'worktree':'; '.join(str(p) for p in alt),'status':'exact target exists' if (v8/name).is_file() or (alternate/name).is_file() else 'PDF exists under different basename' if pdfs else 'missing PDF'})
    refs=[]
    pattern=re.compile(r'\b(?:(?P<ed>Extended Data)\s+(?:Fig(?:ure)?s?\.?|Tables?)|(?P<supp>Supplementary\s+)?(?P<kind>Fig(?:ure)?s?\.?|Notes?|Tables?))\s+(?P<ids>\d+(?:\s*[–−-]\s*\d+)?(?:\s*,\s*\d+(?:\s*[–−-]\s*\d+)?)*)(?P<panels>[a-z](?:\s*[,–-]\s*[a-z])*)?',re.I)
    for doc in docs[:2]:
        for m in pattern.finditer(doc.text):
            # Path/link text is an inventory of filenames, not an in-text callout.
            for n in expand_numbers(m['ids']):
                kind=m['kind'] or 'Fig.'
                detail='';ok=True
                if m['ed']:
                    record=next((r for r in ed if int(r['item'])==n),None)
                    ok=bool(record and record['status']!='missing PDF');detail=record['status'] if record else 'no display map entry'
                elif kind.lower().startswith('fig'):
                    ok=bool(list(v8.glob(f'fig{n}.pdf'))+list(alternate.glob(f'fig{n}.pdf')))
                    if m['panels']:
                        wanted=set(re.findall(r'[a-z]',m['panels'].lower()))
                        if '–' in m['panels'] or '-' in m['panels']:
                            ends=list(re.findall(r'[a-z]',m['panels'].lower()));wanted=set(chr(i) for i in range(ord(ends[0]),ord(ends[-1])+1))
                        if not wanted<=panels.get(n,set()):ok=False;detail='missing legend panel '+','.join(sorted(wanted-panels.get(n,set())))
                    detail=detail or f'fig{n}.pdf; panels '+','.join(sorted(panels.get(n,set())))
                elif kind.lower().startswith('note'):ok=n in notes;detail=f'SI Supplementary Note {n}'
                else:ok=n in tables;detail=f'SI Supplementary Table {n} heading/generated-record index'
                refs.append({'location':location(doc.path,doc.text,m.start()),'reference':m[0],'target':str(n),'status':'resolved' if ok else 'unresolved','detail':detail})
    return inventory,ed,refs


def leftovers(docs):
    out=[]
    pattern=re.compile(r'\bTODO\b|AUTHOR TO SUPPLY|initial file check|parallel (?:figure )?job|older manifest|v7|superseded',re.I)
    for doc in docs[:2]:
        code=[(m.start(),m.end()) for m in re.finditer(r'```[\s\S]*?```|`[^`\n]*`',doc.text)]
        for m in pattern.finditer(doc.text):out.append({'location':location(doc.path,doc.text,m.start()),'kind':'placeholder/review history','text':squash(doc.text[max(0,m.start()-45):m.end()+65])})
        for m in BRACKET.finditer(doc.text):
            if not source_bracket(m[1]) and re.search(r'AUTHOR|TODO|SUPPLY|CONTRIBUTIONS|FUNDING|ACKNOWLEDGEMENTS|COMPETING|DISCLOSURE|ACCESSION|RELEASE|PLACEHOLDER|TBD|INSERT|CONFIRM',m[1],re.I):out.append({'location':location(doc.path,doc.text,m.start()),'kind':'bracketed placeholder','text':m[0]})
        for m in re.finditer("'",doc.text):
            if not any(a<=m.start()<b for a,b in code):out.append({'location':location(doc.path,doc.text,m.start()),'kind':'straight apostrophe outside code','text':squash(doc.text[max(0,m.start()-30):m.start()+31])})
        for start,p,section in blocks(doc):
            if '\n' in p and not p.startswith(('#','|','```','    ','- ','* ',r'\[','$$')) and not re.match(r'\d+\. ',p):
                lines=p.splitlines()
                if len(lines)>1:out.append({'location':location(doc.path,doc.text,start),'kind':'hard-wrapped paragraph','text':f'{len(lines)} physical lines: '+squash(p)[:180]})
    return out


WORD=re.compile(r'[^\W_]+(?:[’\'\-–−.,][^\W_]+)*',re.UNICODE)


def wc(text,remove_sources=False):
    if remove_sources:text=brackets_removed(text)
    # Remove headings, numbered reference labels and panel labels; retain ordinary
    # prose, inline formulas, numeric expressions and hyphenated lexical words.
    text=re.sub(r'^#{1,6}\s+.*$','',text,flags=re.M)
    text=re.sub(r'\*\*[a-z],\*\*','',text)
    text=re.sub(r'(?<!\w)\[\s*[1-9]\d*(?:\s*[,–-]\s*\d+)*\s*\]','',text)
    return len(WORD.findall(text))


def word_counts(docs,reference_count,ed):
    main=docs[0].text
    parts={}
    heads=list(re.finditer(r'^## (.+)$',main,re.M))
    for i,m in enumerate(heads):parts[m[1]]=main[m.end():heads[i+1].start() if i+1<len(heads) else len(main)]
    title=re.search(r'^# (.+)',main)[1]
    rows=[{'section':'Title','current':len(WORD.findall(title)),'source_brackets_removed':len(WORD.findall(brackets_removed(title)))}]
    for name,key in [('Abstract','Abstract'),('Main text','Main text')]:
        t=parts.get(key,'')
        # Journal main text always excludes provenance/source brackets. Numbered
        # literature citations remain, like ordinary citation markers in text.
        rows.append({'section':name,'current':wc(t,key=='Main text'),'source_brackets_removed':wc(t,True)})
    intro=parts.get('Figure legends','').split('**Figure ')[0]
    legend_total_current=wc(intro);legend_total_clean=wc(intro,True)
    legends=parts.get('Figure legends','')
    for m in re.finditer(r'\*\*Figure (\d+)\.([^\n]+)\*\*\n([\s\S]*?)(?=\n\*\*Figure \d|\Z)',legends):
        # Caption titles count as legend words; the Figure N label does not.
        t=m[2]+'\n'+m[3]
        a,b=wc(t),wc(t,True)
        rows.append({'section':'Legend '+m[1],'current':a,'source_brackets_removed':b})
        legend_total_current+=a;legend_total_clean+=b
    rows.append({'section':'Legends total (including common preface)','current':legend_total_current,'source_brackets_removed':legend_total_clean})
    rows.append({'section':'Methods','current':wc(parts.get('Methods','')),'source_brackets_removed':wc(parts.get('Methods',''),True)})
    rows.append({'section':'References (entries, not words)','current':reference_count,'source_brackets_removed':reference_count})
    rows.append({'section':'Main display items (figures)','current':len(set(re.findall(r'\*\*Figure (\d+)\.',legends))),'source_brackets_removed':len(set(re.findall(r'\*\*Figure (\d+)\.',legends)))})
    rows.append({'section':'Extended Data items (mapped / PDFs available)','current':str(len(ed))+' / '+str(sum(r['status']!='missing PDF' for r in ed)),'source_brackets_removed':str(len(ed))+' / '+str(sum(r['status']!='missing PDF' for r in ed))})
    rows.append({'section':'SI words (all prose and table cells)','current':wc(docs[1].text),'source_brackets_removed':wc(docs[1].text,True)})
    siheads=list(re.finditer(r'^## (.+)$',docs[1].text,re.M))
    for i,m in enumerate(siheads):
        t=docs[1].text[m.end():siheads[i+1].start() if i+1<len(siheads) else len(docs[1].text)]
        rows.append({'section':'SI: '+m[1],'current':wc(t),'source_brackets_removed':wc(t,True)})
    for name,t in parts.items():
        if name not in ('Abstract','Main text','Figure legends','Methods','References'):rows.append({'section':name,'current':wc(t),'source_brackets_removed':wc(t,True)})
    return rows


def paths_audit(docs):
    out=[]
    for doc in docs[:2]:
        for name in dict.fromkeys(B.paths(doc.text)):
            p=B.resolve(name)
            out.append({'document':label(doc.path),'source_path':name,'resolved':label(p),'status':'exists' if p.exists() else 'missing'})
    return out


def csv_read(name):return B.table('manuscript/figures/out/v8/data/'+name+'.csv')


def data_checks():
    """Compare semantic figure aggregates/fields with the archived experiment facts.

    Also check all numeric columns in each copied endpoint table and arithmetic
    partitions. No equality-by-numeral search across arbitrary unrelated columns.
    """
    out=[]
    def compare(key,value,path,evidence,denom=''):
        f=B.FACTS.get(key)
        if not f:return
        same=math.isclose(float(value),f.value,rel_tol=1e-12,abs_tol=1e-12)
        out.append({'fact_key':key,'location':path,'value':fmt(value),'expected':fmt(f.value),'denominator':denom or denominator(key,f),'status':'consistent' if same else 'inconsistent','evidence':f.source+': '+f.meaning+'; '+evidence})
    for name,key,col in [('law_z80_c1','C1.edits','edits'),('law_6502_c2','C2.eligible','roles'),('law_6502_controls','controls6502','cases'),('law_population','population.total','cells')]:
        a=csv_read(name)
        if not a or col not in a[0]:
            if name=='law_6502_controls':col='cells' if a and 'cells' in a[0] else 'cases'
            if not a or col not in a[0]:LOAD_ISSUES.append(name+': aggregate column missing');continue
        compare(key,sum(float(r[col]) for r in a),'figures/out/v8/data/'+name+'.csv',f'sum {col}')
    for rownum,r in enumerate(csv_read('route_by_machine'),2):
        for col,key in ({'worlds':'route.Z80.worlds','open':'route.Z80.open'} if r['machine']=='Z80' else {'worlds':'S1.worlds','confined':'S1.first_confined'} if r['machine']=='6502' else {'worlds':'S2.worlds','open':'S2.open','confined':'S2.confined','censored':'S2.rep_censored'}).items():compare(key,r[col],f'figures/out/v8/data/route_by_machine.csv:{rownum}:{col}',f'machine={r["machine"]}; {col}')
    for rownum,r in enumerate(csv_read('arrival_summary'),2):
        if r['machine']=='8086 S2':
            for col,key in [('worlds','S2.worlds'),('events','S2.detected'),('censored','S2.rep_censored'),('horizon','S2.horizon'),('median','S2.km.rep')]:compare(key,r[col],f'figures/out/v8/data/arrival_summary.csv:{rownum}:{col}','8086 S2 '+col)
        if r['machine']=='6502 A':compare('S1.km_t_rep',r['median'],f'figures/out/v8/data/arrival_summary.csv:{rownum}:median','6502 A median')
    # Every numeric plotted endpoint field is compared to its cohort/arm/key/gen
    # source row. Missing source row and denominator drift are explicit findings.
    archived={}
    for cohort,name in [('Historical','results/branching/branching_summary.csv'),('C1','results/tier2/scored/Branching/branching_summary.csv')]:
        for r in B.table(name):archived[(cohort,r['key'],r['arm'],r['gen'])]=(r,name)
    for lineno,r in enumerate(csv_read('branching_endpoints'),2):
        cohort='Historical' if r['cohort'].lower().startswith('histor') else 'C1'
        pair=archived.get((cohort,r['key'],r['arm'],r['gen']))
        if not pair:out.append({'fact_key':'branching source row','location':f'branching_endpoints.csv:{lineno}','value':r['key'],'expected':'matching cohort/key/arm/gen','denominator':r['cohort'],'status':'inconsistent','evidence':'no archival row'});continue
        original,name=pair
        for col,value in r.items():
            if col not in original or not re.fullmatch(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?',value,re.I):continue
            if not re.fullmatch(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?',original[col],re.I):continue
            # wall times are preparation metadata, not plotted scientific outcomes.
            if 'wall' in col:continue
            same=math.isclose(float(value),float(original[col]),rel_tol=1e-12,abs_tol=1e-12)
            out.append({'fact_key':f'branching.{cohort}.{r["key"]}.{col}','location':f'figures/out/v8/data/branching_endpoints.csv:{lineno}:{col}','value':value,'expected':original[col],'denominator':f'{cohort}/{r["key"]}/{r["arm"]}/g{r["gen"]}; column={col}','status':'consistent' if same else 'inconsistent','evidence':name+f'; key={r["key"]},arm={r["arm"]},gen={r["gen"]}; field={col}'})
    for i,r in enumerate(csv_read('validation_fractions')):
        for col in ('unchanged','total','changed'):compare(f'validation{i}.{col}',r[col],f'figures/out/v8/data/validation_fractions.csv:{i+2}:{col}',r['level'])
        value=float(r['unchanged'])/float(r['total'])
        if not math.isclose(value,float(r['fraction']),abs_tol=1e-12):out.append({'fact_key':f'validation{i}.fraction','location':f'validation_fractions.csv:{i+2}:fraction','value':r['fraction'],'expected':fmt(value),'denominator':r['total']+' '+r['level'],'status':'inconsistent','evidence':'unchanged/total'})
    for key in ('C1','C2'):
        a=csv_read('law_z80_c1' if key=='C1' else 'law_6502_c2')
        for r in a:
            if float(r['predicted_free'])!=float(r['measured_free']):out.append({'fact_key':key+'.free','location':'law_z80_c1.csv' if key=='C1' else 'law_6502_c2.csv','value':r['measured_free'],'expected':r['predicted_free'],'denominator':'settings' if key=='C1' else 'roles','status':'inconsistent','evidence':'predicted_free vs measured_free'})
    # prepare_v8.py copies these tables unchanged from v7. Byte identity checks
    # every value AND its full row/column denominator; a changed copy receives
    # field-level differences instead of an unrelated numeral-presence pass.
    new={'arrival_worlds.csv','arrival_curves.csv','arrival_censors.csv','arrival_summary.csv','route_by_machine.csv','literal_matrix.csv','first_copier_tiles.csv','sources_v8.csv','hashes.csv'}
    previous=HERE/'figures/out/v7/data'
    for p in sorted((HERE/'figures/out/v8/data').glob('*.csv')):
        q=previous/p.name
        if p.name in new or not q.is_file():continue
        atext,btext=read(p),read(q)
        a,b=list(csv.DictReader(atext.splitlines())),list(csv.DictReader(btext.splitlines()))
        if a==b:
            out.append({'fact_key':'copied_table.'+p.stem,'location':label(p),'value':str(len(a))+' rows; all fields identical','expected':str(len(b))+' rows; all fields identical','denominator':'complete ordered rows and column names; all denominator/selector fields included','status':'consistent','evidence':label(q)+'; unchanged-copy selection in figures/prepare_v8.py'})
            continue
        if len(a)!=len(b):out.append({'fact_key':'copied_table.'+p.stem+'.rows','location':label(p),'value':str(len(a)),'expected':str(len(b)),'denominator':'copied table rows','status':'inconsistent','evidence':label(q)})
        for lineno,(x,y) in enumerate(zip(a,b),2):
            for col in sorted(set(x)|set(y)):
                if x.get(col)!=y.get(col):out.append({'fact_key':'copied_table.'+p.stem+'.'+col,'location':label(p)+f':{lineno}:{col}','value':x.get(col,''),'expected':y.get(col,''),'denominator':'full copied row '+json.dumps({k:v for k,v in y.items() if not re.fullmatch(r'-?\d+(?:\.\d+)?',v or '')},sort_keys=True),'status':'inconsistent','evidence':label(q)+f':{lineno}:{col}'})
    # Hash inventories are consistency evidence for the figure inputs, not
    # evidence for experimental validity. Do not trust a recorded hash blindly.
    for name in ('hashes','sources_v8'):
        for r in csv_read(name):
            p=(HERE/'figures/out/v8/data'/r['path']) if name=='hashes' else B.resolve(r['path'])
            if not p.is_file():
                out.append({'fact_key':'manifest.'+name,'location':r['path'],'value':'missing file','expected':r['sha256'],'denominator':'file identity','status':'inconsistent','evidence':name+'.csv'});continue
            read(p)
            current=sha(p)
            if current!=r['sha256']:out.append({'fact_key':'manifest.'+name,'location':label(p),'value':current,'expected':r['sha256'],'denominator':'file identity','status':'inconsistent','evidence':name+'.csv; recorded source/input hash'})
    # All sixty machine/world arrival rows must retain their raw identity, event,
    # horizon and detection, including right censoring rather than substituting 0.
    archives={}
    for machine,name in [('Z80 G','results/stageG/stageG/stage_g_runs.csv'),('6502 A','results/s6502/s1_worlds.csv'),('8086 S2','results/s8086/tables/worlds.csv')]:
        for r in B.table(name):
            if machine=='Z80 G' and r['L']!='16' or machine=='6502 A' and r['arm']!='A':continue
            archives[(machine,r['seed'])]=(r,name)
    arrival=csv_read('arrival_worlds')
    for lineno,r in enumerate(arrival,2):
        original,name=archives[(r['machine'],r['seed'])]
        event=bool(original.get('t_rep','')) and float(original['t_rep'])>=0
        horizon=float(original['censor_step'] if r['machine']=='8086 S2' else original['horizon'])
        expected={'event':event,'observed':float(original['t_rep']) if event else horizon,'horizon':horizon,'L':16,'budget':128}
        for col,val in expected.items():
            actual=B.truth(r[col]) if col=='event' else float(r[col])
            if actual!=val:out.append({'fact_key':'arrival.'+r['machine']+'.'+col,'location':f'arrival_worlds.csv:{lineno}:{col}','value':r[col],'expected':str(val),'denominator':r['machine']+' world seed='+r['seed'],'status':'inconsistent','evidence':name+'; matching seed/arm/L, source '+('t_rep/censor_step' if col in ('event','observed','horizon') else 'declared comparison selection')})
        if event and float(r['detection'])!=float(original['t_rep']):out.append({'fact_key':'arrival.detection','location':f'arrival_worlds.csv:{lineno}:detection','value':r['detection'],'expected':original['t_rep'],'denominator':r['machine']+' seed='+r['seed'],'status':'inconsistent','evidence':name+'; t_rep'})
    # Recalculate empirical KM coordinates from world events/censors. This reads
    # archived events and performs arithmetic; it does not execute an experiment.
    for machine in sorted({r['machine'] for r in arrival}):
        a=[r for r in arrival if r['machine']==machine];risk=len(a);survival=1.;coords=[(40.,0.)];censors=[]
        for time in sorted({float(r['observed']) for r in a}):
            at=[r for r in a if float(r['observed'])==time];events=sum(B.truth(r['event']) for r in at);censored=len(at)-events
            if events:survival*=1-events/risk;coords.append((time,1-survival))
            if censored:censors.append((time,1-survival,censored))
            risk-=len(at)
        coords.append((float(a[0]['horizon']),1-survival))
        for name,expected,cols in [('arrival_curves',coords,('step','detected')),('arrival_censors',censors,('step','detected','censored'))]:
            actual=[tuple(float(r[c]) for c in cols) for r in csv_read(name) if r['machine']==machine]
            if len(actual)!=len(expected) or any(any(not math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12) for x,y in zip(a,b)) for a,b in zip(actual,expected)):
                out.append({'fact_key':name+'.'+machine,'location':name+'.csv','value':str(actual),'expected':str(expected),'denominator':str(len(a))+' worlds; event counts / at-risk worlds, retaining censoring','status':'inconsistent','evidence':'arrival_worlds.csv; independent KM risk-set calculation'})
    out.append({'fact_key':'arrival.all_numeric_fields','location':'arrival_worlds/curves/censors.csv','value':'all machine/seed rows and KM coordinates checked','expected':'archived events and censoring','denominator':'per-machine worlds and time-specific risk sets','status':'consistent' if not any(r['status']=='inconsistent' and r['fact_key'].startswith(('arrival.','arrival_')) for r in out) else 'inconsistent','evidence':'worlds source tables and independent KM calculation'})
    raw={r['seed']:r for r in B.table('results/s8086/tables/worlds.csv') if r['first_tape'].startswith('5fa4')}
    seen=Counter()
    for lineno,r in enumerate(csv_read('first_copier_tiles'),2):
        original=raw.get(r['seed']);pos=int(r['position']);seen[r['seed']]+=1
        if original and 0<=pos<len(bytes.fromhex(original['first_tape'])):
            expected={'detection':str(int(float(original['t_rep']))),'byte':f'{bytes.fromhex(original["first_tape"])[pos]:02X}','mechanism':original['first_majority_mechanism'],'role':'shared' if pos%4<2 else 'variable'}
            for col,value in expected.items():
                if r[col]!=value:out.append({'fact_key':'tiles.'+col,'location':f'first_copier_tiles.csv:{lineno}:{col}','value':r[col],'expected':value,'denominator':'physical byte position '+str(pos)+'; world seed='+r['seed'],'status':'inconsistent','evidence':'results/s8086/tables/worlds.csv; first_tape/t_rep/first_majority_mechanism'})
        else:out.append({'fact_key':'tiles.identity','location':f'first_copier_tiles.csv:{lineno}','value':r['seed']+'/'+r['position'],'expected':'existing first-family world/physical position','denominator':'physical position','status':'inconsistent','evidence':'results/s8086/tables/worlds.csv'})
    if set(seen)!=set(raw) or set(seen.values())!={16}:out.append({'fact_key':'tiles.coverage','location':'first_copier_tiles.csv','value':str(dict(seen)),'expected':'16 physical positions per each of the ten first-family worlds','denominator':'world/position','status':'inconsistent','evidence':'results/s8086/tables/worlds.csv'})
    out.append({'fact_key':'tiles.all_fields','location':'first_copier_tiles.csv','value':str(sum(seen.values()))+' rows checked','expected':'first-family physical byte and world identities','denominator':'10 world tapes × 16 positions','status':'consistent' if not any(r['status']=='inconsistent' and r['fact_key'].startswith('tiles.') for r in out) else 'inconsistent','evidence':'results/s8086/tables/worlds.csv'})
    # Every figure-data CSV is inventoried and fingerprinted, including sources
    # not used by the manuscript. They are not all asserted to be independent raw
    # validation: copies and unique derived display tables are identified below.
    inventory=[]
    for p in sorted((HERE/'figures/out/v8/data').glob('*.csv')):
        t=read(p)
        reader=csv.DictReader(t.splitlines());a=list(reader)
        inventory.append({'file':label(p),'rows':len(a),'columns':', '.join(reader.fieldnames or []),'numeric_cells':sum(bool(re.fullmatch(r'-?\d+(?:\.\d+)?(?:e[+-]?\d+)?',v or '',re.I)) for r in a for v in r.values() if isinstance(v,str))})
    return out,inventory


def cross_documents(rows,data):
    groups=defaultdict(list);out=[]
    for r in rows:
        if r['kind']=='quantity' and r['fact_key']:groups[r['fact_key']].append(r)
    units={'C2.roles':'roles','C2.eligible':'roles','C2.eligible_tapes':'tapes','C2.eligible_worlds':'worlds','C2.worlds':'worlds','C2.tapes':'tapes','C1.edits':'settings','C1.direct':'cases','C2.direct':'cases','C1.serial':'rows','C2.serial':'observations'}
    for r in rows:
        key=r['fact_key']
        if key in units:
            after=r['_p'][r['_local']+len(r['value']):r['_local']+len(r['value'])+65]
            after=re.split(r'[.;\[\n]',after)[0]
            m=re.search(r'\b(roles|tapes|worlds|settings|cases|rows|observations)\b',after)
            if m:
                r['declared_denominator']=m[1]
                if m[1]!=units[key]:out.append({'fact_key':key,'locations':r['location'],'values':r['value'],'denominators':m[1]+' stated; '+units[key]+' required','issue':'declared observational unit differs','correct_value':r['source_value']+' '+units[key],'evidence':r['evidence_source']+'; '+r['meaning']})
        if key.startswith('competition.') and key in B.FACTS and 'median' in r['_p'].lower():
            m=re.search(r'\bT\s*/\s*\(([^)]+)\)',r['_p'])
            if m:
                actual=re.sub(r'\s+','',m[1])
                r['declared_denominator']='T/('+actual+')'
                if actual not in ('R+T','T+R'):out.append({'fact_key':key,'locations':r['location'],'values':r['value'],'denominators':'T/('+actual+') stated; T/(R+T) required','issue':'declared share denominator differs','correct_value':r['source_value']+' with T/(R+T)','evidence':r['evidence_source']+'; field T_share_end'})
    for key,group in sorted(groups.items()):
        if len(group)<2:continue
        values={r['source_value'] if r['status']=='rounding' else r['normalized_value'] for r in group}
        denoms={r['denominator'] for r in group}
        if len(values)>1 or len(denoms)>1:out.append({'fact_key':key,'locations':'; '.join(r['location'] for r in group),'values':'; '.join(r['value'] for r in group),'denominators':'; '.join(sorted(denoms)),'issue':'value differs' if len(values)>1 else 'denominator differs','correct_value':fmt(B.FACTS[key].value) if key in B.FACTS else 'see arithmetic','evidence':B.FACTS[key].source if key in B.FACTS else 'numeric expression'})
    for r in data:
        if r['status']=='inconsistent':out.append({'fact_key':r['fact_key'],'locations':r['location'],'values':r['value'],'denominators':r['denominator'],'issue':'figure-data value/denominator differs from archive','correct_value':r['expected'],'evidence':r['evidence']})
    return out,groups


def legend_consistency(docs):
    if len(docs)<3:return []
    main=re.search(r'^## Figure legends\s*\n([\s\S]*?)(?=^## |\Z)',docs[0].text,re.M)
    if not main:return [{'fact_key':'legends','locations':label(docs[0].path),'values':'missing legend section','denominators':'','issue':'legend section missing','correct_value':'five legends','evidence':label(docs[2].path)}]
    a=main[1].strip();b=re.sub(r'^# [^\n]+\n','',docs[2].text).strip()
    if a==b:return []
    # Report paragraph pairs, including denominator prose even when the numeric
    # tokens coincide. Text identity is stronger than comparing numeral bags.
    out=[]
    for n in range(1,6):
        pat=r'\*\*Figure '+str(n)+r'\.[\s\S]*?(?=\*\*Figure \d|\Z)'
        x,y=re.search(pat,a),re.search(pat,b)
        if not x or not y or squash(x[0])!=squash(y[0]):
            nums=lambda text:[fmt(norm(m[0])) for m in TOKEN.finditer(text) if not any(a<=m.start()<b for a,b,k in mask_identifiers(text) if k=='non-count idiom')]
            nx,ny=nums(x[0]) if x else [],nums(y[0]) if y else []
            out.append({'fact_key':f'legend{n}','locations':label(docs[0].path)+'; '+label(docs[2].path),'values':'numeric token sequences identical' if nx==ny else json.dumps(nx)+' vs '+json.dumps(ny),'denominators':'caption prose differs; inspect declared units','issue':'editorial caption drift' if nx==ny else 'caption number sequence differs','correct_value':'synchronize caption text','evidence':'exact per-figure text and normalized number comparison'})
    return out


def table(rows,columns):
    if not rows:return ['None.']
    def cell(value):return str(value).replace('|','&#124;').replace('\n',' ')
    return ['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']+['| '+' | '.join(cell(r.get(c,'')) for c in columns)+' |' for r in rows]


def write_report(args,docs,rows,structural,refs,figs,ed,figrefs,left,words,paths,data,inventory,cross,groups,unstable):
    entries,seen,calls,findings,urls=refs
    counts=Counter(r['status'] for r in rows)
    substantive=Counter(r['status'] for r in rows if r['kind']=='quantity')
    lines=['# Mechanical audit of manuscript v8 and SI v8','',
      'Read-only snapshot audit. No simulations, paid computation, network calls, source regeneration or git writes.',
      '', '## Numeric status counts','']
    lines+=table([{'status':s,'all occurrences':counts[s],'scientific/formula quantities':substantive[s]} for s in STATUS],['status','all occurrences','scientific/formula quantities'])
    lines+=['','Counts by requested part (exported legends are reported separately, not added to MAIN legends):','']
    bypart=assign_parts(rows)
    lines+=table([{'part':p,**c} for p,c in bypart.items()],['part',*STATUS])
    lines+=['','Scientific/formula quantities only (structural labels and lexical idioms excluded):','']
    lines+=table([{'part':p,**{s:sum(r['part']==p and r['kind']=='quantity' and r['status']==s for r in rows) for s in STATUS}} for p in bypart],['part',*STATUS])
    lines+=['',f'Total: {len(rows)} numeric occurrences. One row per occurrence in MAIN, SI and the exported legend copy; numbers in paths, citation labels, processor names and code are explicitly inventoried as structural occurrences. Figure-data cells are audited in keyed comparisons and inventoried below, not duplicated into the manuscript-occurrence CSV.',
      '', '## Reproducibility and scope','',
      'Draft paths are positional arguments or --main/--si. Defaults resolve beside the script, independent of the working directory. Outputs are deterministic for identical input bytes and filesystem inventories. --strict exits 1 for unresolved audit findings; input changes during a run exit 2 and are explicitly flagged. The audit reads each draft once and checks fingerprints after source analysis. It never edits the manuscript, SI, fact base or source data.',
      '', 'Archived fact builders are reused from audit_v7.py. MAIN_v7 and AUDIT_v7_RESOLUTION supply guarded context-transfer bindings to independent facts; no v7 line-number binding is applied directly to v8. Empirical verification requires a named field/filtered aggregate or an immutable registration/contract record. Bare numeral equality and lexical report matching cannot verify a quantity. The explicit definition class binds mathematical, protocol and display parameters to an independent defining line, with its location and excerpt. MAIN definitions may cite the SI; no document supplies its own scientific evidence. Structural/lexical occurrences remain a separately disclosed inventory, not measurements.',
      '', 'Derived entries check arithmetic on bound operands, or complements of independently sourced counts. Printed equations with unresolved operands do not receive a derived pass. Rounding uses the inherited decimal nearest/half-up rule, with the unrounded value retained. Denominators distinguish worlds, sampling roles, genotypes, mutation types, starting populations, intervention cases, encounters, serial observations and repeated scan records. A disposable hash-checked cache outside the repository avoids rescanning archived streams; any changed archived input invalidates it. The 714-role original ledger is independently counted from the local git object at 943aeee, never inferred from the draft.',
      '', 'The cross-document registry compares every unambiguous fact binding, full caption text, and cohort/field keyed figure tables. Unbound quantities cannot receive a semantic cross-document pass and remain an explicit coverage gap in the unresolved table. The CSV includes source brackets, evidence paths, denominator, meaning, source value, correction and v7 resolution IDs.',
      '', 'Draft snapshot fingerprints:', '']
    lines+=table([{'file':label(d.path),'sha256':READ[d.path.resolve()]} for d in docs],['file','sha256'])
    lines+=['','Concurrent-edit check: '+('INPUT CHANGED; rerun required: '+', '.join(unstable) if unstable else 'all opened inputs unchanged during this run.'),
      '', '## Numeric mismatches','']
    lines+=table([r for r in rows if r['status']=='mismatch'],['id','location','value','correct_value','denominator','evidence_source','evidence'])
    lines+=['','## Cross-document inconsistencies','']+table(cross,['fact_key','locations','values','denominators','issue','correct_value','evidence'])
    lines+=['',f'Compared {len(groups)} bound fact groups and {len(data)} keyed figure-data checks. Exported legends: '+('identical to manuscript legends.' if not legend_consistency(docs) else 'differ; see findings above.'),
      '', '## Word counts','',
      'Unicode word tokens allow internal apostrophes, hyphens/dashes and decimal/grouping punctuation. Headings, panel labels and numbered literature citation markers are excluded from body counts; caption titles count as legend words. Title is counted separately. Main text excludes source brackets in both columns, following the requested journal convention, and excludes legends, Methods, bibliography, availability and SI. The current column otherwise includes source annotations; the second removes result/provenance brackets, retaining editorial placeholders. Counts are mechanical estimates: journal submission systems may tokenize formulas differently.', '']
    lines+=table(words,['section','current','source_brackets_removed'])
    lines+=['','## Numbered references','',f'{len(entries)} entries; {len(calls)} citation-number occurrences. First appearances: '+', '.join(map(str,seen))+'.', '']
    lines+=(findings or ['Every numbered citation resolves, every entry is cited, and first appearances are in numerical order.'])
    lines+=['']+table(calls,['location','citation','status'])
    lines+=['','DOI/URL checks are syntax-only; no fetch or bibliographic validation was performed.','']+table(urls,['location','URL','status'])
    lines+=['','## Figure and Supplementary references','']+table(figrefs,['location','reference','target','status','detail'])
    lines+=['','Main figure exports in both trees:','']+table(figs,['figure','panels','primary','worktree'])
    lines+=['','## Extended Data targets in both trees','',
      'The requested descriptive filenames and the renderer’s short ed8_N basenames are checked separately. A short-name PDF establishes a rendered numbered target, but does not establish that a descriptive path in the SI resolves or that its content/legend is correct.','']+table(ed,['item','requested','primary','worktree','status'])
    lines+=['','## Placeholders, review history, apostrophes and wrapping','']+table(left,['location','kind','text'])
    lines+=['','## Unresolved numeric provenance','']+table([r for r in rows if r['status'] in ('unsourced','source missing')],['id','location','value','status','source','context'])
    lines+=['','Unresolved SI occurrences by note and reason:','']
    si_groups=defaultdict(list)
    for r in rows:
        if r['part']=='SI' and r['status'] in ('unsourced','source missing'):si_groups[r['si_note']].append(r)
    lines+=table([{'SI note':n,'remaining':len(g),'reason':g[0]['unresolved_reason'],'locations':', '.join(str(x) for x in sorted({r['line'] for r in g}))} for n,g in sorted(si_groups.items(),key=lambda x:(not x[0].isdigit(),int(x[0]) if x[0].isdigit() else x[0]))],['SI note','remaining','reason','locations'])
    lines+=['','## Definition bindings','']+table([r for r in rows if r['status']=='definition'],['id','location','value','fact_key','evidence_source','evidence'])
    lines+=['','## Rounding','']+table([r for r in rows if r['status']=='rounding'],['location','value','source_value','rounding','meaning'])
    lines+=['','## Derived arithmetic','']+table([r for r in rows if r['status']=='derived'],['location','value','source_value','evidence_source','evidence'])
    lines+=['','## Source paths','']+table(paths,['document','source_path','resolved','status'])
    lines+=['','## Figure-data coverage','']+table(inventory,['file','rows','numeric_cells','columns'])
    lines+=['','Keyed figure-data comparisons:','']+table(data,['fact_key','location','value','expected','denominator','status','evidence'])
    lines+=['','## Binding diagnostics','']+(LOAD_ISSUES or ['All v7 fact builders and binding anchors loaded.'])
    lines+=['','Structural occurrence categories: '+json.dumps(dict(structural),sort_keys=True)+'.',
      '', '## Exact rerun command','', '```sh',f'/Users/neo/repos/algocell/experiments/.venv/bin/python {HERE / "audit_v8.py"} {docs[0].path} {docs[1].path}','```',
      '', '## Evidence fingerprints','']
    lines+=table([{'file':label(p),'sha256_at_read':h} for p,h in sorted(READ.items())],['file','sha256_at_read'])
    (args.output_dir/'AUDIT_v8.md').write_text('\n'.join(lines)+'\n')
    fields=[k for k in rows[0] if not k.startswith('_')] if rows else ['id','location','status']
    with (args.output_dir/'audit_v8.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n',extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    return counts,substantive


def main():
    global B
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('drafts',nargs='*',metavar='DRAFT',help='MAIN and SI paths, in that order')
    ap.add_argument('--main',type=Path)
    ap.add_argument('--si',type=Path)
    ap.add_argument('--output-dir',type=Path,default=HERE)
    ap.add_argument('--figure-worktree',type=Path,default=OTHER)
    ap.add_argument('--strict',action='store_true')
    ap.add_argument('--no-cache',action='store_true',help='rescan all archived fact inputs; no simulations')
    args=ap.parse_args()
    read(Path(__file__))
    if len(args.drafts)>2:ap.error('supply at most MAIN and SI paths')
    mainpath=(args.main or (Path(args.drafts[0]) if args.drafts else HERE/'MAIN_v8.md')).resolve()
    sipath=(args.si or (Path(args.drafts[1]) if len(args.drafts)>1 else HERE/'SI_v8.md')).resolve()
    args.output_dir=args.output_dir.resolve()
    for output in (args.output_dir/'AUDIT_v8.md',args.output_dir/'audit_v8.csv'):
        if output.resolve() in (mainpath,sipath):ap.error('an output path would overwrite an input draft')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    docs=[Document(mainpath,read(mainpath),'main'),Document(sipath,read(sipath),'si')]
    legend=HERE/'figures/out/v8/LEGENDS_V8.md'
    if legend.exists():docs.append(Document(legend,read(legend),'legends'))
    else:LOAD_ISSUES.append('exported LEGENDS_V8.md missing')
    B,old,resolution=load_base(not args.no_cache)
    try:additional_facts()
    except (OSError,KeyError,ValueError,StopIteration) as e:LOAD_ISSUES.append(f'additional_facts: {type(e).__name__}: {e}')
    comprehensive_facts()
    rules()
    comprehensive_rules()
    definitions_v8()
    completion_facts_and_rules()
    final_main_bindings()
    si_facts_and_rules()
    si_table_facts()
    mathematical_contract_rules()
    strengthen_bindings()
    last_si_bindings()
    derived_bindings()
    threshold_definitions()
    finish_exploratory_bindings()
    revised_legend_bindings()
    rows=[r for doc in docs for r in extract(doc,B)]
    apply_rules(rows)
    transfers(rows,old,resolution)
    classifier_bindings(rows)
    apply_table_recipes(rows)
    arithmetic(rows)
    structural=finish_rows(rows)
    part_counts=assign_parts(rows)
    refs=citations(docs)
    figs,ed,figrefs=figures(docs,args.figure_worktree)
    left=leftovers(docs)
    words=word_counts(docs,len(refs[0]),ed)
    paths=paths_audit(docs)
    try:data,inventory=data_checks()
    except (OSError,KeyError,ValueError,StopIteration) as e:
        LOAD_ISSUES.append(f'data_checks: {type(e).__name__}: {e}');data,inventory=[],[]
    cross,groups=cross_documents(rows,data)
    cross+=legend_consistency(docs)
    unstable=[label(p) for p,h in READ.items() if not p.is_file() or sha(p)!=h]
    counts,substantive=write_report(args,docs,rows,structural,refs,figs,ed,figrefs,left,words,paths,data,inventory,cross,groups,unstable)
    print(json.dumps({'numbers':len(rows),'statuses':{s:counts[s] for s in STATUS},'counts_by_part':part_counts,'scientific_quantities':{s:substantive[s] for s in STATUS},'priority_unbound':sum(r['part'] in ('abstract','main','legends','Methods') and r['status'] not in ('verified','definition','derived','rounding') for r in rows),'si_remaining_by_note':dict(Counter(r['si_note'] for r in rows if r['part']=='SI' and r['status'] in ('unsourced','source missing'))),'mismatches':[{k:r[k] for k in ('location','value','correct_value','evidence_source','evidence')} for r in rows if r['status']=='mismatch'],'cross_document_inconsistencies':cross,'word_counts':words,'reference_findings':refs[3],'unresolved_figure_references':sum(r['status']=='unresolved' for r in figrefs),'extended_data':[{k:r[k] for k in ('item','requested','status')} for r in ed],'leftovers':len(left),'load_issues':LOAD_ISSUES,'changed_inputs':unstable,'output_dir':str(args.output_dir)},indent=2))
    if unstable:return 2
    has_findings=any(counts[s] for s in ('mismatch','unsourced','source missing')) or cross or refs[3] or any('malformed' in r['status'] for r in refs[4]) or any(r['status']=='unresolved' for r in figrefs) or left or LOAD_ISSUES or any(r['status']!='exact target exists' for r in ed) or any(r['status']=='missing' for r in paths)
    return 1 if args.strict and has_findings else 0


if __name__=='__main__':raise SystemExit(main())

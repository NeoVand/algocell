#!/usr/bin/env python3
"""Local read-only audit of manuscript v8 and SI; no network or execution of experiments.

Run: experiments/.venv/bin/python experiments/manuscript/audit_v8.py [MAIN.md SI.md]
Only AUDIT_v8.md and audit_v8.csv are written (or --output-dir equivalents).
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import median
from urllib.parse import urlsplit

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
OTHER = Path('/Users/neo/repos/algocell-wt-figs')
STATUS = ('verified', 'derived', 'rounding', 'mismatch', 'unsourced', 'source missing')
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
N = r'(?<![\w-])(?:0[xX][0-9a-fA-F]+|\d+(?:,\d{3})*(?:\.\d+)?%?|'+ '|'.join(sorted(WORDS,key=len,reverse=True)) + r')(?![\w-])'
READ = {}
LOAD_ISSUES = []


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


def load_base():
    spec=importlib.util.spec_from_file_location('audit_v7_factbase',HERE/'audit_v7.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    read(HERE/'audit_v7.py')
    spec.loader.exec_module(module)
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
    for m in re.finditer(r'`([^`\n]+)`',p):
        code=m[1]
        if re.fullmatch(r'(?:[0-9a-f]{2}|[xyNXdHLob]+)(?:\s+(?:[0-9a-f]{2}|[xyNXdHLob]+))*',code,re.I) or (not re.fullmatch(r'0x[0-9a-f]+|[0-9a-f]{2,4}',code,re.I) and not re.search(r'[=+−*/^]|\bmod\b',code)):
            marks.append((m.start(1),m.end(1),'instruction encoding / code identifier'))
    add(r'\b(?:one another|one route|one aspect|one member|an internal one|zero-page|zero-EA|zero-SP|one-pass|one-sided|two-sided|almost none|none of the measurements)\b','non-count idiom')
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
        for m in sorted(tokens,key=lambda m:m.start()):
            kind=next((k for a,b,k in marks if a<=m.start()<b),'quantity')
            # Headings and figure callouts retain a separate structural inventory.
            if p.startswith('#'):kind='heading identifier'
            if re.match(r'^\*\*Figure \d+',p):kind='figure identifier'
            if section=='References':kind='bibliographic metadata'
            if re.search(r'(?:Fig(?:ure)?\.?s?|Notes?|Tables?)\s+[\d,– −a-z]*$',p[max(0,m.start()-40):m.start()],re.I):kind='display / section reference'
            tok=m[0]
            if re.fullmatch(r'F[0-9a-f]{3}|A7|FF',tok,re.I):kind='quantity'
            if p.startswith('|') and re.search(r'(?:^|\n)\|\s*$',p[:m.start()]) and re.match(r'\s*(?:\.\s*[^|]*)?\|',p[m.end():]):kind='table entry identifier'
            try:v=int(tok,16) if re.fullmatch(r'F[0-9a-f]{3}|A7|FF',tok,re.I) else norm(tok)
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
            if m.start() and p[m.start()-1]=='−' and not re.search(r'\d\s*$',before[:-1]):v=-v
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
    c=B.Claim(0,0,row['value'],row['_value'],'',0,'','',[],[],kind='number')
    ok,rounding=B.displayed_equal(c,f)
    row.update(fact_key=key,denominator=denominator(key,f),source_value=fmt(f.value),correct_value='' if ok else fmt(f.value),status='derived' if ok and derived else 'rounding' if ok and not rounding.startswith('exact') else 'verified' if ok else 'mismatch',rounding=rounding if ok else '',meaning=f.meaning,evidence_source=f.source,evidence=(f.evidence or f.meaning)+(('; '+extra) if extra else ''))
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
    rule(r'none (?:is|were) unexpected',[],lambda p:False)
    rule(r'(none) (?:is|were) unexpected','validation.unexpected',lambda p:'hardware' in p)
    rule(q+r' validation comparisons','validation.units')
    rule(q+r'% upper bound','ring_confidence')
    rule(r'rate below '+q,'ring_rate_bound')
    rule(q+r' (?:historical worlds of the registered benign|worlds of the registered benign)','route.Z80.worlds')
    rule(r'first detected replicator.*?(?:was open in the sampled encounters)',[],lambda p:False)
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
    rule(r'(?:No|no) mismatch',[],lambda p:False)
    rule(r'(?:no|zero) (?:mismatch|mismatches)',[],lambda p:False)
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
    rule(r'first confinement.*?'+q,[],lambda p:False)
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
    rule(q+r' measurements.*?culture',[],lambda p:False)


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
                    if row:match_fact(row,key,derived)


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
            supported=bool(result['_paths']) and any(B.resolve(q).is_file() for q in result['_paths'])
            if not supported:continue
            ok=Decimal(result['normalized_value'])==value
            result.update(status='derived' if ok else 'mismatch',fact_key='arithmetic:'+re.sub(r'\s+','',m[1]),source_value=fmt(value),correct_value='' if ok else fmt(value),meaning='explicit arithmetic; operand provenance is separately audited',evidence_source='; '.join(result['_paths']),evidence=f'{m[1]} = {fmt(value)}',denominator='same units as the stated operands')
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
        if r['value'].lower() in ('four','two') and re.match(r'\s+(measurements|properties)',right) and ('culture' in p or 'closure' in p or 'inflow' in p):
            expected=4 if right.lstrip().startswith('measurements') else 2
            r.update(status='derived' if r['_value']==expected else 'mismatch',source_value=str(expected),correct_value='' if r['_value']==expected else str(expected),fact_key='definition.assays' if expected==4 else 'definition.closure_and_transmission',denominator='explicitly enumerated concepts',meaning='set cardinality in the paragraph',evidence_source=r['file'],evidence='culture, execution confinement, partner inflow, transfer retention' if expected==4 else 'partner-information closure and parental-variation transmission',resolution_ids='N0014' if expected==4 else '')
            continue
        if r['_attached'] and all(not B.resolve(p).exists() for p in r['_attached']):
            r.update(status='source missing',meaning='all attached source paths are absent')
            continue
        # Preserve the v7 conservative contextual-report rule. It discloses the
        # exact excerpt and never uses CSV numeral presence as evidence.
        c=B.Claim(r['_start'],r['_end'],r['value'],r['_value'],r['_p'],r['_pstart'],r['section'],r['source'],r['_attached'],r['_paths'])
        matched=B.contextual(c)
        if matched:
            f,rounding=matched
            r.update(status='rounding' if not rounding.startswith('exact') else 'verified',source_value=fmt(f.value),rounding=rounding,meaning=f.meaning,evidence_source=f.source,evidence=f.evidence,denominator='contextual report scope; not independently field-bound')
        else:r['meaning']='No unambiguous field, derivation or same-context report binding; requires source resolution.'
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
    lines+=['',f'Total: {len(rows)} numeric occurrences. One row per occurrence in MAIN, SI and the exported legend copy; numbers in paths, citation labels, processor names and code are explicitly inventoried as structural occurrences. Figure-data cells are audited in keyed comparisons and inventoried below, not duplicated into the manuscript-occurrence CSV.',
      '', '## Reproducibility and scope','',
      'Draft paths are positional arguments or --main/--si. Defaults resolve beside the script, independent of the working directory. Outputs are deterministic for identical input bytes and filesystem inventories. --strict exits 1 for unresolved audit findings; input changes during a run exit 2 and are explicitly flagged. The audit reads each draft once and checks fingerprints after source analysis. It never edits the manuscript, SI, fact base or source data.',
      '', 'Fact builders and conservative report-context rules are reused from audit_v7.py. MAIN_v7 and AUDIT_v7_RESOLUTION supply guarded context-transfer bindings; no v7 line-number binding is applied directly to v8. Empirical verification requires a named field/filtered aggregate, or the exact same local quantitative report context. Bare numeral equality cannot verify a quantity. Definitions, proof constants and display choices require a source binding too. Unbound occurrences remain unsourced, and the audit does not imply that those statements are false.',
      '', 'Derived entries check explicit numeric arithmetic or a complement from independently sourced counts. Printed-equation operands are separately audited and their unresolved provenance is retained; derived arithmetic establishes the calculation, not new empirical evidence. Rounding uses the inherited decimal nearest/half-up rule, with the unrounded value retained. Denominators distinguish worlds, sampling roles, genotypes, mutation types, starting populations, intervention cases, encounters, serial observations and repeated scan records.',
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
    B,old,resolution=load_base()
    try:additional_facts()
    except (OSError,KeyError,ValueError,StopIteration) as e:LOAD_ISSUES.append(f'additional_facts: {type(e).__name__}: {e}')
    rules()
    rows=[r for doc in docs for r in extract(doc,B)]
    apply_rules(rows)
    transfers(rows,old,resolution)
    arithmetic(rows)
    structural=finish_rows(rows)
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
    print(json.dumps({'numbers':len(rows),'statuses':{s:counts[s] for s in STATUS},'scientific_quantities':{s:substantive[s] for s in STATUS},'mismatches':[{k:r[k] for k in ('location','value','correct_value','evidence_source','evidence')} for r in rows if r['status']=='mismatch'],'cross_document_inconsistencies':cross,'word_counts':words,'reference_findings':refs[3],'unresolved_figure_references':sum(r['status']=='unresolved' for r in figrefs),'extended_data':[{k:r[k] for k in ('item','requested','status')} for r in ed],'leftovers':len(left),'load_issues':LOAD_ISSUES,'changed_inputs':unstable,'output_dir':str(args.output_dir)},indent=2))
    if unstable:return 2
    has_findings=any(counts[s] for s in ('mismatch','unsourced','source missing')) or cross or refs[3] or any('malformed' in r['status'] for r in refs[4]) or any(r['status']=='unresolved' for r in figrefs) or left or LOAD_ISSUES or any(r['status']!='exact target exists' for r in ed) or any(r['status']=='missing' for r in paths)
    return 1 if args.strict and has_findings else 0


if __name__=='__main__':raise SystemExit(main())

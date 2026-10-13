"""Render the nine MAIN_v7 Extended Data displays from archived evidence, CPU only.

Reuse the frozen v6.2/v8 plotting inputs and the original figure palette and
primitives. No new simulation, synthetic observations, or hidden coordinate jitter.
Full ledgers/proofs remain linked from ED_V8.md; each rendered panel has source hashes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import build_v8 as b

HERE=b.HERE
EXP=HERE.parents[1]
OUT=b.OUT
SOURCES={}
CURRENT=0
HEIGHTS={n:(290 if n==5 else 180 if n in [1,3,7,8,9] else 220) for n in range(1,10)}
TITLES={1:'Validation across instruction, encounter, assay and ecology',2:'Complete survival and marker-carriage maps',3:'Registered displacement interventions and exploratory controls',4:'Reset dependence, sampling and detection order',5:'Environmental competition trajectories and intruder controls',6:'Finite censuses, execution contracts and 6502 retention',7:'BFF literal, bandwidth, pointer and halting controls',8:'Genealogy discovery and N6 confirmation',9:'Registered S2 evidence and separate post hoc assays'}
LEGENDS={
1:'a, Five validation comparisons, each with its own denominator. b, Core C1/C2 impact flags and isolated-assay differences. These implementation labels differ from the displacement and 6502 experiment names. c, All 22 I1-T2 claim verdict comparisons. Preserved verdicts include failures. d, All changed constituent components; four of 36 changed. e, Assay columns that changed under either corrected core. f, Stage-L dispatch records: archived final data were scoreable despite missing-snapshot alerts. Mutation writes can collide within a GPU batch; expected hardware divergences and stochastic rerun differences do not establish a causal core-repair effect. Full attribution, omitted-case validation and execution contracts accompany the source ledgers.',
2:'a, Every historical serial design at generation eight, partitioned over all mutant starts into lost, alive below 50% carriage, and alive at or above 50% carriage. Control survival and scored sites appear alongside. b, Complete registered branching generation-eight map, including uniform-byte (U) and archived soup-partner (K) arms; the historical and corrected-core endpoints coincide. c, Serial sites at transfers one and eight for every design, with separately scaled tapes. d, Generation-eight marker-carriage fractions across all parent/environment combinations. e, Identifiable mutation-type counts and control-start denominators; different alternative panels remain explicit. f, Ordered marker ancestry through two 128-instruction encounters. Read address 4 carries ancestry from original byte 0 and overwrites the last marker descendant at address 8 at step 6. Survival requires a qualifying match to the immediate parent, while marker carriage is a separate thresholded measurement. Minority carriers may remain in the middle category.',
3:'a, All 101 registered settings in 22 Z80 parents: predicted and measured free positions. b, Direct joint and erased-coordinate challenges by displacement and passage direction, with cases distinguished from partners. c, Actual generation-eight descendant ancestry for the 0x04 and 0x60 intervention arms. d, Exploratory first-byte dial: all 256 values, with sampled eligibility kept separate from culture competence. e, Sampled Z80 site-score exceptions, including fetched-site scores and the whole-tape-match gate deficit. f, Frozen eligibility accounting for the 2,347 sampled parents. The registered assay comprises 25,856 serial rows and 1,067,608 direct cases without mismatch; free and erased alternatives are tested with the core held fixed. Dial working scores and sampled-site agreement are not universal inheritance certificates. Displacement, protected prefix, coverage, phase, noninterference and serial closure remain conditions of the capacity theorem.',
4:'a, All 20 random-register confirmation worlds at six observation updates; confinement is conditional on operational heritability. b, Modal-tape abundance versus the fraction of sampled heritable cells identical to it. c, Class denominators by stage and tape length; these are sampled cells nested within worlds. d, Marker-positive and marker-negative confinement fractions: the sequence marker is not a reliable confinement classifier. e, First observed regenerator/transmitter ordering under the any-detection threshold for every archived stage and tape length. f, Complete sensitivity to majority and greater-than-10% thresholds. Regenerator/intermediate/transmitter here denote site-score classes (at most two, three or four, and at least five sites), distinct from competition R/T byte classes. Observation order is not ancestry. Random reset is a declared contract intervention; this panel does not establish a universal route or arrival-order law.',
5:'a-d, Every historical and corrected-core L32 mutation-on competition world, through 300,000 updates, with within-group median curves. Line styles identify equal starts and reciprocal 1% invasions. R and T are frozen partial-copy and full-tape byte classes. Each group contains 10 equal-start worlds and five per invasion direction. e, Complete endpoint ledger for all 110 historical and 110 corrected-core worlds: both tape lengths, all tar/mutation/start conditions, relative share, absolute motif prevalence and tail diversity. Mutation-off worlds end at 100,000 updates; mutation-on at 300,000. Other byte-0 variants remain in core abundance. One corrected-core benign world has undefined T/(R+T), but zero absolute abundance and tail diversity remain in the summaries. f, Exploratory intruder continuation counts and host-damage controls, kept separate from registered competitions. The E1 stability endpoint failed; between-world variation does not establish neutral drift. Rejected mutation-only equilibrium interpretations are not fitted. Tail sequences are nested within worlds.',
6:'a, Complete finite 6502 census accounting by length, BRK contract, budget and period, including four-billion-word period-four arms. b, Complete finite 8086 census counts; causal mechanism labels supersede the earlier raw-address labels. c, Constructed 6502 carriage controls compared with d minus constrained footprint; these are sampled-carriage cases. d, S1 first-route and loop counts for all four arms, each with 20 worlds and paired seeds. e, Complete C2 eligibility accounting by sampling role. f, Exact C2 capacities for all eligible roles, with constrained footprints and independent-world counts explicit. There are 942 sampled roles overall; 42 eligible roles represent 41 tapes from 11 worlds. Single-site, joint and erasure tests total 91,428 direct cases and 731,424 serial observations, with no mismatch. C2 does not rescore S1. Literal-channel results concern declared channels and finite contracts, not all copiers. Pointer randomization was unmeasured and was not an S2 arm. Full proofs and finite counterexamples are linked with the source tables.',
7:'a, All archived BFF variants: modal life is the first sample with modal-tape generation-two gain ≥0.3; population life is the first sample with ≥50% of random tapes operationally heritable. Collapse means terminal heritable share <10% after population life. std has no literal channel or wrap; lit adds literal writing, wrap allows pointer wrap, and nh makes brackets harmless. b, First-copier partner-fetch fraction versus offspring entropy conditional on the parent; each point is a world-derived tape. c, Write-bandwidth controls show that open fetch can coexist with zero partner inflow: literal repetition two and three yield zero entropy while partner entry remains one. d, Halting-probability controls, with all-world counts and terminal shares. e, First-versus-final inflow and partner-entry summaries across BFF variants. f, The complete registered individuality prediction ledger, retaining the failed literal-variant component. Entropy, instruction-stream confinement and dependence on initial partner contents are different measurements. Registered dial failures remain failures; no smooth equilibrium mechanism is inferred. World, tape and partner-challenge denominators are separate.',
8:'a, Discovery and confirmation genealogy denominators, with sampled lines nested in worlds. b, First founders versus later conversions in each archive. c, N6 first-founder mechanisms. d, Whole-tape donor function versus executed-byte donor function, retaining the stricter functional definition. e, Non-copier chains and assembly times for all first founders. f, Complete registered N6 component verdicts; N6-4 remains not met. Discovery analyses and corrected genealogy definitions are distinct from the N6 confirmation. A tape that copies its executed bytes need not be a whole-tape copier. Observation order on sampled genealogies does not establish a universal succession. Point-mutation null and donor-abundance comparisons use their registered definitions; founders and later conversions are not pooled.',
9:'a, All 20 registered S2 worlds with first-copier detection or terminal censoring, and fetch-route labels. b, Descriptive persistence profiles (R2), without a directional registered prediction. c, Registered candidate accounting: no eligible evolved-law case; L1/L2 remain untested and L2 support is suspended. d, Exclusion versus unresolved-extraction reasons, with unresolved POP ES cases visible. e, Separate exploratory extraction admissions and the selected post hoc exact-law panel, grouped by displacement. f, Exploratory first-family copy geometry and encounter fates. The post hoc panel comprises 32 distinct genotypes from 20 worlds and 2,224,640 encounters without mismatch. Empty d4 payload joints and d16 erasure sets are displayed as empty challenge sets. These points are not promoted to registered evidence or added to the main law plot. Operational attempts, input hashes and accounting are reproducibility records, not independent biological replication.'}


def read(path, **kw):
    p=EXP/path
    if not p.exists(): raise FileNotFoundError(f'Missing archived data: {path}')
    SOURCES.setdefault(CURRENT,{})[str(p.relative_to(EXP))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p,low_memory=False,**kw)


def frozen(name):return read(f'manuscript/figures/out/v8/data/{name}.csv')


def page(n):
    global CURRENT
    CURRENT=n
    return b.page(290 if n==5 else 220)


def panel(f,i,label,plot=True):
    x=2+92*(i%2); top=f._height_mm-6-70*(i//2)
    b.title(f,chr(97+i),label,x,top)
    return b.axes(f,x+14,top-50,65,39) if plot else b.axes(f,x+1,top-59,82,51,True)


def foot(f,i,s,color=b.INK):
    x=3+92*(i%2);top=f._height_mm-6-70*(i//2)
    f.text(x/180,(top-65)/f._height_mm,s,fontsize=5.5,color=color,va='center')


def tab(a,headers,rows,widths=None,size=5.5):
    widths=widths or [82/len(headers)]*len(headers)
    xs=np.r_[0,np.cumsum(widths)]
    top=48;step=min(4.5,43/max(len(rows),1))
    for x,h in zip(xs,headers):b.text(a,x,top,str(h),size=size,bold=True)
    for i,row in enumerate(rows):
        for x,value in zip(xs,row):
            a.text(x,top-5-i*step,str(value),fontsize=size,color=b.INK,va='center')
    a.plot([0,sum(widths)-1],[44.7,44.7],color=b.RULE,lw=.4)


def note(a,y,s):
    a.text(0,y,s,fontsize=5.5,color=b.INK,va='center')


def frac(a,x='',y='Fraction'):
    a.set(ylim=(-.03,1.05));b.tidy(a,x,y,yticks=[0,.5,1])


def save(f,n):
    f.set_size_inches(180*b.fs.MM,HEIGHTS[n]*b.fs.MM);f._height_mm=HEIGHTS[n]
    b.save(f,f'ed8_{n}')


def ed1():
    f=page(1);d=frozen('validation_fractions')
    a=panel(f,0,'Validation: distinct denominators',False)
    labs=['8088/hardware: state agreement','C1/C0: ring stability','C1/C0: identical assay metrics','C1/archive: same claim verdict','C1/archive: same component outcome']
    tab(a,['Comparison / criterion','Criterion met/n','Different n'],[(lab,f'{int(r.unchanged):,}/{int(r.total):,}',f'{int(r.changed):,}') for lab,r in zip(labs,d.itertuples())],[42,26,14],size=5.3)
    note(a,14,'10,083 hardware divergences: expected; none unexpected')
    foot(f,0,'Ring stability: 95% upper bound on changed-memory rate <0.01%\nEach row names its agreement criterion and denominator')
    d=read('results/impact/outcome.csv');a=panel(f,1,'I1: flags remain flags',False)
    tab(a,['Core','Flagged / conditions','Different assays'],[(r.core,f'{r.flagged}/{r.decision_conditions}',f'{r.assay_differences}/{r.assay_combinations}') for r in d.itertuples()],[14,38,30])
    note(a,24,'Baseline = historical compatibility core C0')
    note(a,17,'C1 = corrected classic; C2 = vendor-specific semantics')
    note(a,10,'Separate sets: 690 ring conditions; 417 isolated assays')
    foot(f,1,'Flag: 95% upper bound on changed-memory rate ≥0.01%\nZero flags +identical assays criterion: not met')
    d=read('results/tier2/T2_CLAIMS.csv');a=panel(f,2,'I1-T2: all 22 claim verdicts',False)
    half=(len(d)+1)//2;rows=[]
    for i in range(half):
        r=d.iloc[i];s=d.iloc[i+half] if i+half<len(d) else None
        rows.append([r['claim'],r.status,s['claim'] if s is not None else '',s.status if s is not None else ''])
    tab(a,['Claim','vs archive','Claim','vs archive'],rows,[13,29,13,27])
    foot(f,2,'DZ2 timing; DZ3 transmitter trend; L2 recovery sites\nStands = same as archive; E1 stability remains failed; 19/22 stand')
    d=read('results/tier2/T2_COMPONENTS.csv');q=d[d.published_verdict!=d.C1_verdict];a=panel(f,3,'Four changed components of 36',False)
    tab(a,['Component','Historical','Core C1'],[(r.claim+(' L16' if 'L16' in r.registered_component else ' L20' if 'L20' in r.registered_component else ''),r.published_verdict.replace('n/a (no recovery)','n/a'),r.C1_verdict.replace('n/a (no recovery)','n/a')) for r in q.itertuples()],[28,27,27])
    note(a,20,'n/a = not applicable: no recovered lineage')
    note(a,12,'L = tape bytes; these are the constituent tests in a')
    foot(f,3,'Historical = published archive; DZ timing/share; L2 sites\nComplete 36-component ledger retained in Source Data')
    d=read('results/impact/assay_columns.csv');a=panel(f,4,'Assay changes by measurement',False)
    names={'n_informative':'Informative partners, generation 1','n_informative2':'Informative partners, generation 2','offspring_within_q':'Offspring ≥75% parent match','gen2_score':'Generation-2 normalized gain','gen2_cond':'Generation-2 conditional gain','score':'Normalized culture gain'}
    tab(a,['Metric','Differing C1','Differing C2'],[(names.get(r.column,r.column.replace('_',' ')),r.c1_changed,r.c2_changed) for r in d.itertuples()],[48,17,17],size=5.2)
    foot(f,4,'Counts /417 assays vs C0; rows overlap (not additive)\nA = parent; B = partner; match threshold q = 75%')
    d=read('results/tier2/T2_StageL_status.csv');a=panel(f,5,'Stage-L scoring and snapshot availability',False)
    rows=[]
    for L,q in d.groupby('L'):
        rows.append([L,len(q),int(q.scoreable.sum()),int(q.missing_steps.fillna('').astype(str).ne('').sum())])
    tab(a,['L bytes','Worlds','Scoreable\nworlds','Worlds missing\nsnapshot'],rows,[14,17,23,28])
    note(a,24,'Missing snapshot: step 7,500 world updates')
    note(a,17,'Scoreable from retained final bytes and journals')
    note(a,10,'10 independent recovery worlds / tape length')
    foot(f,5,'Stage L: closure-recovery worlds, 10 / tape length\nSnapshot alerts do not prevent scoring retained final data')
    for p in ['results/impact/supplement_manifest.json','results/tier2/stageL_diagnosis.json']:
        path=EXP/p;SOURCES[1][p]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(f,1)


def partition(a,d,labels,control,sites):
    n=len(d);h=34/max(n,1);top=40
    b.text(a,0,48,'Design (L bytes)',size=5.5,bold=True);b.text(a,55,48,'Controls\nalive',size=5.5,bold=True);b.text(a,69,48,'Scored\nsites',size=5.5,bold=True)
    for i,(idx,r) in enumerate(d.iterrows()):
        y=top-i*h
        a.text(0,y,labels[i],fontsize=5.1,va='center',color=b.INK)
        left=28
        for col,color in [('lost',b.old.LOST_C),('erased',b.old.ERASED_C),('retained',b.TEAL)]:
            v=float(r[col]);a.add_patch(Rectangle((left,y-.8),23*v,1.6,fc=color,ec='none'));left+=23*v
        a.text(55,y,control[i],fontsize=5.1,va='center');a.text(70,y,str(sites[i]),fontsize=5.1,va='center')
    for x,lab in [(28,'0'),(39.5,'0.5'),(51,'1')]:a.text(x,4,lab,fontsize=5,ha='center',va='center')
    a.text(39.5,1,'Share of mutant starts',fontsize=5,ha='center',va='center')


def design(key):
    names={'ret16':'Closed: return L16','ldir20':'Closed: block-copy L20','ldir32':'Closed: block-copy L32','jr50':'Open: jump L50','lethal16_s4009':'Closed: lethal L16','genome16_s6006':'Evolved transmitter L16/6','genome20_s6003':'Evolved transmitter L20/3','genome20_s6004':'Evolved transmitter L20/4','closer32_8080_p1':'Constructed 8080 L32/1','closer32_8080_p2':'Constructed 8080 L32/2'}
    return names.get(key,'Open: pusher L'+key.removeprefix('pusher'))


def ed2():
    f=page(2);ss=frozen('serial_summary');sites=frozen('serial_sites')
    q=ss.copy();q['lost']=q.lost_g8;q['erased']=q.erased_g8;q['retained']=q.retained_g8
    a=panel(f,0,'All historical serial designs: transfer 8',False)
    partition(a,q,[design(r.key) for r in q.itertuples()],[f'{round(r.ctrl_alive_g8*256)}/256' for r in q.itertuples()],q.sites_g8.tolist())
    foot(f,0,'Dark grey lost; light grey alive <50%; teal alive ≥50% carrying\nCarriage = members containing mutant window / population members\nSite: mutant-survivor carriage − WT-survivor carriage ≥0.5 at ≥half values')
    d=frozen('branching').query("cohort=='Historical' and gen==8")
    a=panel(f,1,'Branching generation-8 outcomes: U / K',False)
    partition(a,d,[design(r.key)+f' {r.arm}' for r in d.itertuples()],[f'{r.ctrl_alive_n}/{r.ctrl}' for r in d.itertuples()],d.sites.tolist())
    foot(f,1,'Same partition/site key as a; controls = unmutated populations\nU uniform-byte; K archived soup partners; all archived pairs shown\n32 values/site ×4 populations/value; 64 controls/design')
    a=panel(f,2,'Complete serial site maps: 1 / 8',False)
    for i,r in enumerate(ss.itertuples()):
        y=47-i*3.35;L=int(r.L)
        a.text(0,y,design(r.key),fontsize=5.1,va='center')
        for gen,off in [(1,.6),(8,-.6)]:
            s=sites.query('key==@r.key and gen==@gen').set_index('pos').transmissible.reindex(range(L)).fillna(False)
            for j,v in enumerate(s):a.add_patch(Rectangle((31+j*49/L,y+off-.48),49/L,.96,fc=b.TEAL if v else '#EEEEEE',ec='white',lw=.1))
    foot(f,2,'Upper track: transfer 1; lower: transfer 8; teal = scored site\nSite criterion as a; each row spans byte 0 to L−1 (L in label)')
    d=frozen('parent_environment');pivot=d.pivot(index='parent',columns='environment',values='retained_g8')
    a=panel(f,3,'All partner-environment carriage tests')
    pos=a.get_position();a.set_position([126/180,pos.y0,51/180,pos.height])
    from matplotlib.colors import LinearSegmentedColormap
    a.imshow(pivot.values,vmin=0,vmax=1,cmap=LinearSegmentedColormap.from_list('carriage',['#F4F5F5',b.TEAL]),aspect='auto')
    environments={'random partners, benign':'Random\nbenign','random partners, lethal':'Random\nlethal','closed benign world':'Closed\nbenign\nsoup','lethal world':'Lethal\nsoup','open phase (t5000)':'Open\nsoup\n5k','random registers':'Random\nreset'}
    labels=[environments.get(s,s.replace(' partners,','').replace(' ','\n',1)) for s in pivot.columns]
    lengths=d.groupby('parent').tape.first().str.split().map(len)
    a.set_xticks(range(len(pivot.columns)),labels,fontsize=5);a.set_yticks(range(len(pivot)),[s.replace('block-copy R','Partial block-copy').replace('block-copy T','Full block-copy')+f' L{lengths[s]}' for s in pivot.index],fontsize=5.1)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):a.text(j,i,f'{pivot.iloc[i,j]:.2f}',ha='center',va='center',fontsize=5.3,color='white' if pivot.iloc[i,j]>.6 else b.INK)
    b.tidy(a,'Partner source / rule / reset','')
    foot(f,3,'Serial: one descendant/transfer; g8 carrier contains mutant window\nStarts/design = L×255×16; L bytes in labels; 256 WT controls\nBenign zero executes / lethal halts; open soup at 5,000 updates')
    a=panel(f,4,'Identifiable alternatives and controls',False)
    tab(a,['Design','Ident. / byte alternatives','WT starts'],[(design(r.key),f'{r.n_identifiable}/{r.n_mutants}',256) for r in ss.itertuples()],[31,35,16],size=5.1)
    foot(f,4,'Identifiable = marker window absent from wild type (WT)\nAlternative types = L byte positions × tested values\n16 starts/value; pusher50/64 & jump50: 32 values, others 255')
    a=panel(f,5,'Ordered ancestry: two full encounters')
    d=frozen('erasure_ordered')
    for e,off in [(1,0),(2,128)]:
        q=d.query('encounter==@e');a.step(np.r_[off,q.step+off],np.r_[1 if e==1 else 0,q.marker_sources_remaining],where='post',color=b.RED,lw=1)
    a.axvline(128,color=b.RULE,ls='--',lw=.6);a.set(xlim=(-4,260),ylim=(-.04,1.1));b.tidy(a,'Instruction step','Marker ancestry copies',[0,128,256],[0,1])
    a.text(10,.8,'One block-copy L32 lineage;\nmarker original byte 8 = 0xA7',fontsize=5.3)
    a.text(26,.43,'1st encounter',fontsize=5.5);a.text(150,.43,'2nd encounter',fontsize=5.5)
    a.annotate('Step 6: last\nmarker ancestor lost',xy=(6,.02),xytext=(20,.22),fontsize=5.2,color=b.RED,arrowprops=dict(arrowstyle='-',color=b.RED,lw=.5))
    foot(f,5,'Counted unit: bytes with byte-8 ancestry; all-zero byte partners\nStep 6 reads address 4 (ancestry byte 0), writes 0x04 over 8\nAt 128: offspring becomes parent; processor reset',b.RED)
    save(f,2)


def ed3():
    f=page(3);d=frozen('capacity_edits');a=panel(f,0,'Registered: 101 settings, 22 parents')
    a.plot([0,60],[0,60],color=b.RULE,lw=.6);a.scatter(d.predicted_free,d.measured_free,s=12,facecolors='none',edgecolors=b.TEAL,lw=.7);a.set(xlim=(-3,63),ylim=(-3,63));b.tidy(a,'Predicted free byte positions','Measured free byte positions',[0,20,40,60],[0,20,40,60])
    a.text(3,53,'Measured = predicted: 101/101 settings',fontsize=5.5)
    counts=[f'{int(r)}:{len(q)}' for r,q in d.groupby('predicted_free')]
    a.text(3,45,'Coincident counts (positions:settings)',fontsize=5.2)
    a.text(3,39,', '.join(counts[:4]),fontsize=5.2)
    a.text(3,33,', '.join(counts[4:]),fontsize=5.2)
    foot(f,0,'Overlapping settings: coordinate counts above; 22 controls /101\nFree = arbitrary byte alternatives jointly inherited with causal roots\nFixed core; 22 parents; diagonal = equality')
    d=frozen('ed_joint_challenges').query("system=='Z80 C1'");a=panel(f,1,'Direct challenges: all passage directions',False)
    rows=[(f'{di} / {ki}',f'{int(q.cases.sum()):,}',f'{int(q.passed.sum()):,}') for (di,ki),q in d.groupby(['direction','kind'])]
    rows.append(('Total',f'{int(d.cases.sum()):,}',f'{int(d.passed.sum()):,}'))
    tab(a,['Edit / challenge','Cases','Passed'],rows,[41,22,19])
    foot(f,1,'Reciprocal = restore original displacement; single = 1 site\nJoint = payload jointly varied; erased = lost-coordinate control\nPass: exact offspring, ancestry and fixed core; 17 partners/case')
    d=frozen('lineage_strips');serial=read('results/capacity/serial.csv.gz');a=panel(f,2,'Actual descendants: lineage 0, generation 8',False)
    for key,y in [('ldir32_v04',35),('ldir32_v60',15)]:
        q=d.query('edit_id==@key');r=q.iloc[0];b.text(a,0,y+6,f'Parent byte 0 = 0x{int(r.edited_byte):02X}; copy offset {int(r.displacement)} bytes',size=6.5,bold=True)
        for r in q.itertuples():a.add_patch(Rectangle((r.position*2.5,y),2.5,3,fc={'core':b.GREY,'erased':b.RED,'transmitted':b.TEAL}[r.state],ec='white',lw=.2))
        for pos in [0,4,16,31]:b.text(a,(pos+.5)*2.5,y-3,str(pos),size=5.5,ha='center')
        sample=serial.query('edit_id==@key and lineage==0 and generation==8 and direction=="forward"');assert len(sample)==1 and sample.exact.all() and sample.provenance_exact.all()
        haplotype=sample.offspring.iloc[0].split();assert len(haplotype)==32
        for j in range(2):a.text(0,y-7-j*3,' '.join(haplotype[j*16:(j+1)*16]).upper(),fontsize=5.1,fontfamily='DejaVu Sans Mono',va='center')
    foot(f,2,'Byte positions 0–31; hex rows = actual descendant bytes\nGrey core | red original-parent ancestry lost | teal retained')
    d=frozen('dial_coordinates');a=panel(f,3,'Exploratory: all 256 first-byte values')
    color=np.array([0 if not r.culture_replicator else 2 if r.prop8_eligible else 1 for r in d.itertuples()]).reshape(16,16)
    from matplotlib.colors import ListedColormap
    a.imshow(color,cmap=ListedColormap(['#CFD4D7','#FFFFFF',b.TEAL]),vmin=0,vmax=2)
    for x in np.arange(-.5,16,.999999):a.axvline(x,color='#D0D5D8',lw=.15)
    for y in np.arange(-.5,16,.999999):a.axhline(y,color='#D0D5D8',lw=.15)
    a.set_xticks([0,4,8,12,15],list('048CF'));a.set_yticks([0,4,8,12,15],list('048CF'));b.tidy(a,'Low hex digit','High hex digit')
    for yy,col,label in [(82,'#CFD4D7','157 nonworking: generation-2 normalized culture gain <0.3'),(79,'white','88 working (gain ≥0.3), excluded by prefix/offset/phase'),(76,b.TEAL,'11 working (gain ≥0.3), meets prefix/offset/phase conditions')]:
        f.patches.append(Rectangle((95/180,(yy-.7)/220),1.5/180,1.4/220,transform=f.transFigure,fc=col,ec=b.GREY,lw=.4))
        f.text(98/180,yy/220,label,fontsize=5.5,va='center',color=b.INK)
    d=frozen('z80_payload');a=panel(f,4,'Exploratory: sampled-score exceptions',False)
    tab(a,['Measurement','Cells'],[['Motif-bearing cells',len(d)],['Cells with dividing offset',int(d.eligible_dividing.sum())],['Cells with extra fetched-site scores',int(d.query('eligible_dividing').fetched_extra.gt(0).sum())],['Cells failing whole-tape match gate',int(d.query('eligible_dividing').residual.ne(0).sum())]],[62,20])
    note(a,20,'53 and 1 are disjoint cell subsets of the 2,343 cases')
    note(a,13,'Extra = fetched-site retention beyond unexecuted sites')
    note(a,6,'Gate deficit: 4 alleles retained; offspring <75% parent match')
    foot(f,4,'Exploratory Z80 G/I/K/L cells: 5E ED B0 motif (same family as c)\n2,347 sampled cells nested in worlds; 4 nondividing excluded')
    d=read('results/capacity/candidate_eligibility.csv');a=panel(f,5,'Frozen candidate eligibility',False)
    q=d.groupby(['eligible','reason'],dropna=False).size().reset_index(name='n')
    tab(a,['Eligibility / reason','Candidates'],[(('Eligible' if r.eligible else str(r.reason).replace('_',' ')),r.n) for r in q.itertuples()],[65,17],size=5.3)
    note(a,25,'Same motif-cell candidates as e: d divides L, 4≤d≤L')
    note(a,18,'d = copy offset; L = tape length in bytes')
    note(a,11,'Serial mismatch count 0 /25,856 descendant-transfer records')
    foot(f,5,'Each record checked separately for exact bytes, core and ancestry\nShared denominator for each check; conditional overlap family')
    save(f,3)


def ed4():
    f=page(4);d=read('results/r4/r4_rep.csv');a=panel(f,0,'Separate reset cohort: L16, 20 worlds')
    for seed,q in d.groupby('seed'):a.plot(q.step/1e6,q.confined_given_heritable,color=b.TEAL,lw=.5,alpha=.4)
    q=d.groupby('step').confined_given_heritable.median();a.plot(q.index/1e6,q,color=b.INK,lw=1);a.set(xlim=(0,3.05));frac(a,'World updates (millions)','Confined / heritable')
    a.text(1.2,.55,'2/20 worlds majority confined at 3M',fontsize=5.5)
    a.text(.7,.1,'18 zero trajectories overlap median',fontsize=5.5)
    foot(f,0,'Thin = worlds; black median; cells confined / heritable cells\nConfined: no partner instruction-stream fetch in 16/16 traces')
    d=read('results/population/population_worlds.csv');a=panel(f,1,'Random cells versus the modal tape')
    a.scatter(d.modal_share,d.identical_to_modal_given_heritable,s=9,color=b.GREY,alpha=.7);a.set(xlim=(-.03,1.05));frac(a,'Cells identical to modal / all cells','Modal-identical / heritable cells')
    foot(f,1,f'One dot = final world sample; {len(d)} stage/length/seed worlds\nY: random sampled heritable cells identical to modal tape')
    d=read('results/population/classes_cells.csv');a=panel(f,2,'Class denominators by cohort',False)
    rows=[]
    for (stage,L),q in d.groupby(['stage','L']):rows.append([f'{stage}, L{L}',q.seed.nunique(),len(q),int(q.confined.sum()),int(q['class'].eq('transmitter').sum())])
    tab(a,['Cohort','Worlds','Cell rows','Conf. rows','T rows'],rows,[28,13,14,16,11],size=5.2)
    foot(f,2,'Pooled sampled-cell rows; confined and T are subsets\nG benign Z80; I lethal Z80; K aligned L32; L 10M extensions\nM 8080 subset; L in labels = tape bytes (not cohort L)')
    d=read('results/marker_check/marker_check.csv');a=panel(f,3,'Sequence marker is not confinement')
    a.scatter(d.confined_given_no_marker,d.confined_given_marker,s=9,color=b.RED);a.plot([0,1],[0,1],color=b.RULE,lw=.6);a.set(xlim=(-.03,1.05));frac(a,'Confined / no marker','Confined / marker')
    a.text(.05,.84,'Marker 20 F0: e.g. 5/5 confined',fontsize=5.5)
    a.text(.05,.74,'No marker in same sample: 0/507',fontsize=5.5)
    foot(f,3,f'L50 matched-pair invasion: {d.seed.nunique()} worlds, {len(d)} samples\n20 F0 marker; axes: confinement among positive/negative cells\n512 cells/sample; equal-fraction diagonal')
    d=read('results/ladder/population_groups.csv');q=d.query("threshold=='any'");a=panel(f,4,'Exploratory detection order: any detection',False)
    tab(a,['Cohort','n','R<T','Same','T<R','R only','T only','None'],[[f'{r.stage} L{r.L}',r.n,r['R before T'],r['same sample'],r['T before R'],r['R only'],r['T only'],r['neither']] for _,r in q.iterrows()],[18,8,9,9,9,10,10,9])
    foot(f,4,'R = confined ≤2 scored sites; T = confined ≥5 scored sites\nR<T: R observed first; Same: same sampled update; T<R: T first\nAny = positive estimated class share; n = worlds; not ancestry')
    a=panel(f,5,'Detection-order threshold sensitivity',False)
    rows=[]
    for _,r in d.query("threshold!='any'").iterrows():rows.append([f'{r.stage} L{r.L} '+('>50%' if r.threshold=='majority' else '>10%'),r['R before T'],r['same sample'],r['T before R'],r['both_seen_n'],r.n])
    tab(a,['Threshold','R<T','Same','T<R','Both','n'],rows,[32,10,10,10,10,10])
    foot(f,5,'Threshold = class share among heritable cells; Both = sum of orders\nOther worlds have one/neither; site scored when ≥half tested values\nare heritable and mutant allele retained in ≥50% qualifying copies')
    save(f,4)


def ed5():
    f=page(5);d=frozen('competition_trajectories');starts={'mix50':('-',b.GREY),'R_into_T':('--',b.RED),'T_into_R':(':',b.TEAL)}
    for i,(cohort,tar) in enumerate([('Historical','benign'),('C1','benign'),('Historical','lethal'),('C1','lethal')]):
        a=panel(f,i,f'{cohort} / {tar}: L32, mutation on')
        q=d.query('cohort==@cohort and tar==@tar')
        for (start,seed),g in q.groupby(['start','seed']):a.plot(g.step/1000,g.T_share,color=starts[start][1],ls=starts[start][0],lw=.45,alpha=.45)
        for start,g in q.groupby('start'):
            m=g.groupby('step').T_share.median();ls,col=starts[start];a.plot(m.index/1000,m,color=col,ls=ls,lw=1.2)
        a.set(xlim=(0,305));frac(a,'World updates (thousands)','Full-tape T / (R + T)')
        foot(f,i,'Thin worlds; thick medians; solid: 50:50 (10 worlds)\nDashed: R at 1% into T (5); dotted: T at 1% into R (5)\nR partial-copy / T full-tape; C1 corrected Z80 / historical C0')
    archive=read('results/offset/offset_switch.csv');c1=read('results/tier2/scored/E/offset_switch.csv')
    assert len(archive)==len(c1)==110
    b.title(f,'e','Complete environmental endpoint ledger: 110 worlds per implementation',2,144)
    a=b.axes(f,3,85,174,51,True);rows=[]
    for (L,tar,mut,start),q in archive.groupby(['L','tar','mut','start']):
        r=c1.query('L==@L and tar==@tar and mut==@mut and start==@start')
        star='*' if (L,tar,mut,start)==(32,'benign','on','T_into_R') else ''
        rows.append([L,tar,mut,{'mix50':'50:50','R_into_T':'R at 1%','T_into_R':'T at 1%'}[start],f'{len(r)}/{len(q)}',f'{r.T_share_end.median():.2f}'+star,f'{q.T_share_end.median():.2f}',f'{r.core_share_end.median():.2f}',f'{q.core_share_end.median():.2f}',f'{r.T_distinct_tails_end.median():.0f}',f'{q.T_distinct_tails_end.median():.0f}'])
    tab(a,['L bytes','Zero rule','Mutation','Start','n C1/H','T C1','T H','Core C1','Core H','T tails C1','T tails H'],rows,[9,18,13,21,16,15,15,15,15,17,18],size=5.5)
    f.text(3/180,79/290,'World medians; H = historical C0. Benign zero executes; lethal zero halts. Mutation off: 100k; on: 300k updates.\n'+
           'T = T/(R+T); core = 5E ED B0-bearing cells /20,000; T tails = distinct sequences at bytes 4–(L−1) per world.\n'+
           '*C1 L32 benign T-at-1%: T median uses 4/5 ratios; core/tail medians use all 5 worlds, including the zero world.',fontsize=5.5,color=b.INK,va='center')
    d=read('results/ladder/n2_groups.csv');a=b.axes(f,3,15,174,51,True)
    b.title(f,'f','EXPLORATORY continuation and host controls; registered stability verdict',2,74)
    tab(a,['Source soup','Execution zero rule','Worlds','Mean T fraction, start','Mean T fraction, end'],[(r.composition,r.rule,r.worlds,f'{r.mean_T_share_start:.3f}',f'{r.mean_T_share_end:.3f}') for r in d.itertuples()],[34,35,20,42,43],size=6)
    note(a,21,'Continuation: historical Z80 L32; mean T/(R+T), end = 2,000 updates; both classes pre-exist.')
    toxin=read('results/toxin/causal.csv');read('results/n2/demography.csv')
    host=toxin.query('tar=="lethal" and variant in ["native","detox"]')
    native=host.query('variant=="native"').iloc[0];detox=host.query('variant=="detox"').iloc[0]
    note(a,15,f'Historical lethal L32 hosts: fraction unchanged native {native.intact:.4f}, detox {detox.intact:.4f}; n=400×64 encounters.')
    note(a,11,'Detox replaces payload 0x00/0x76 with random non-toxic bytes; host is partner B of soup executors.')
    vals=[]
    for L in [16,32]:
        q=c1.query('L==@L and tar=="benign" and mut=="off" and start=="mix50"')
        vals.append(f'L{L}: {int(q.T_share_end.between(.3,.7).sum())}/10')
    note(a,7,'E1, corrected C1 benign mutation-off: ≥7/10 with T/(R+T) 0.3–0.7; '+', '.join(vals)+'; failed.')
    f.text(3/180,9/290,'Continuation and host tests are distinct from registered competitions; variable endpoints do not establish neutral drift.',fontsize=5.5,color=b.INK)
    save(f,5)


def ed6():
    f=page(6);d=read('results/census6502/table_main.csv');a=panel(f,0,'6502 finite census: all completed arms',False)
    rows=[]
    for (L,brk,budget),q in d.groupby(['L','BRK','budget']):rows.append([f'{L}/{brk[:4]}/{budget}',','.join(map(str,q.k)),f'{int(q.done.sum()):,}',int(q['exact, zero partner'].sum())])
    tab(a,['L/BRK/budget','Periods','Words done','Exact'],rows,[31,13,28,10],size=5.1)
    foot(f,0,'L = tape bytes; budget = CPU instructions; k = tiled-word period\nfait = faithful BRK; halt = halt before BRK; Exact = self-writers\nWords done sums k; each L16 k4 arm alone exhausts 256⁴ words')
    d=read('results/census8086/table_main.csv');a=panel(f,1,'8086 finite census: declared reset',False)
    tab(a,['L/budget/k','Words','Zero exact','≥75% ≥8/16'],[(f'{r.L}/{r.budget}/{r.k}',f'{r.words:,}',r.zero_exact_nonzero,r.random75_ge8_nonzero) for r in d.itertuples()],[24,19,18,21],size=5.2)
    foot(f,1,'L bytes; budget instructions; k = word period; SP at ring end\nCounts of nonzero words: exact output into zero partner; or\n≥75% byte match in ≥8 of 16 random-partner encounters')
    d=frozen('law_6502_controls');a=panel(f,2,'Constructed 6502: sampled carriage')
    a.plot([0,28],[0,28],color=b.RULE,lw=.6);a.scatter(d.iloc[:,0],d.iloc[:,1],s=22,marker='s',facecolors='none',edgecolors=b.GREY);a.set(xlim=(-1,29),ylim=(-1,29));b.tidy(a,'d - 6 (byte positions)','Sampled carried positions',[0,14,28],[0,14,28])
    a.text(1,26,'Equality y = x; five coordinate groups',fontsize=5.5)
    foot(f,2,f'{int(d.cases.sum())} constructed copier/contract configurations; points coincide\nCoordinate counts 2:96, 4:40, 10:96, 14:32, 26:32; d offset bytes\nSite: ≥half tested alternatives heritable and allele retained ≥50%')
    d=read('results/s6502/s1_worlds.csv');a=panel(f,3,'S1: complete four-arm route ledger',False)
    tab(a,['Arm L/budget','n','Conf.','Loop','Open','Undet.'],[(f'{arm} {q.L.iloc[0]}/{q.budget.iloc[0]}',len(q),int(q.first_confined.sum()),int(q.first_loop.sum()),int(q.first_open.sum()),int(q.t_rep.fillna(-1).lt(0).sum())) for arm,q in d.groupby('arm')],[28,10,11,11,11,11])
    foot(f,3,'S1 = 6502 worlds; Conf. = zero partner fetch; Open = ≥8/16 traces\nUndet. = no heritable copier detected; Loop overlaps Conf. (PC return)\n20 worlds/arm; L bytes / CPU budget; A–C faithful BRK, D pre-halt')
    d=read('results/evolved6502/eligibility.csv');a=panel(f,4,'Registered C2: complete sampling roles',False)
    labels={'final_random':'Final random cell','final_top3':'Final top-3 by abundance','first':'First detected tape'}
    tab(a,['Sampling role','Sampled','Eligible','Elig. worlds'],[(labels[role],len(q),int(q.eligible.sum()),q.query('eligible==1').world.nunique()) for role,q in d.groupby('role')],[33,17,16,16])
    q=d.query('eligible==1');b.text(a,0,15,f'Eligible: {len(q)} roles; {q.tape.nunique()} unique tapes; {q.world.nunique()} worlds',size=6.5,bold=True)
    b.text(a,0,7,'Pointer randomization: unmeasured (not an S2 arm)',size=5.5)
    foot(f,4,'Eligible = certified fixed-core, pure forward-copy family\n942 roles; world counts overlap across roles (do not sum)')
    d=frozen('evolved6502_capacity');a=panel(f,5,'C2 exact capacities: all eligible roles')
    a.plot([0,10],[0,10],color=b.RULE,lw=.6);a.scatter(d.predicted_free,d.measured_free,s=13,marker='^',color='#C98E2F');a.set(xlim=(-1,11),ylim=(-1,11));b.tidy(a,'Predicted free positions','Measured free positions',[0,5,10],[0,5,10])
    a.text(.4,9.7,'Equality y = x; 42 roles overlap',fontsize=5.5)
    foot(f,5,'Free byte positions: arbitrary alternatives retained with causal roots\nExact-copy/core/ancestry mismatches: 0/91,428 direct cases\nand 0/731,424 serial observations; 42 roles /41 tapes /11 worlds')
    for p in ['results/literal_channel/direct_search.csv','results/literal_channel/finite_search.csv','results/literal_channel/CHECK.md','results/s6502/CORRECTION.md','results/census8086/reset_correction_counts.csv']:
        path=EXP/p;SOURCES[6][p]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(f,6)


def ed7():
    f=page(7);d=read('results/bff/runs.csv');extra=read('results/bff_stdnh/runs.csv');d=pd.concat([d,extra],ignore_index=True)
    a=panel(f,0,'BFF contracts: detection and persistence',False)
    variants={'std':'Standard','stdlit':'Literal, no wrap','stdnh':'Standard, harmless','wrap':'Pointer wrap','wraplit':'Wrap + literal','wraplitnh':'Wrap + literal, harmless','lit':'Literal, no wrap'}
    tab(a,['Variant','Worlds','Modal\ndetected','Pop.\ndetected','Later\ncollapse'],[(variants[v],len(q),int(q.t_top.notna().sum()),int(q.t_her.notna().sum()),int(q.collapsed.fillna(False).sum())) for v,q in d.groupby('variant')],[27,14,13,13,15])
    foot(f,0,'Literal emits operands; harmless = unmatched brackets do not halt\nModal generation-2 normalized culture gain ≥0.3; pop ≥50% heritable\nLater collapse = terminal share <10% after population detection')
    h=read('results/biology/individuality/per_replicator.csv');q=h.query("machine=='bff' and which=='first'");a=panel(f,1,'First copiers: fetch versus partner inflow')
    palette={'std':b.GREY,'lit':b.INK,'wrap':'#C98E2F','wraplit':b.RED,'wraplitnh':b.TEAL}
    short={'std':'Std','lit':'Literal','wrap':'Wrap','wraplit':'W+lit','wraplitnh':'W+lit+harmless'}
    for group,g in q.groupby('group'):
        counts=g.groupby(['entered_frac','H_bits']).size().reset_index(name='n')
        a.scatter(counts.entered_frac,counts.H_bits,s=4*counts.n,label=f'{short[group]} (n={len(g)})',color=palette[group])
    a.set(xlim=(-.05,1.05),ylim=(-.4,max(9,q.H_bits.max()+1)));b.tidy(a,'Partner-fetch encounter fraction','Conditional offspring H (bits)')
    handles=[b.Line2D([],[],marker='o',ls='none',ms=3,color=palette[g],label=f'{short[g]} (n={len(r)})') for g,r in q.groupby('group')]
    a.legend(handles=handles,fontsize=5,loc='center',bbox_to_anchor=(.56,.55))
    a.text(.34,.23,'Coincident tape count',fontsize=5,transform=a.transAxes)
    for xx,n in [(.38,1),(.52,5),(.68,10)]:
        a.scatter(xx,.14,s=4*n,facecolors='none',edgecolors=b.GREY,lw=.5,transform=a.transAxes)
        a.text(xx,.04,str(n),ha='center',fontsize=5,transform=a.transAxes)
    foot(f,1,f'{len(q)} first-modal tapes, one/world; circle area = coincident count\nH = offspring entropy conditional on parent, random partners\nStandard harmless subset has no archived H panel')
    d2=read('results/bff_dials/d2_bandwidth.csv');a=panel(f,2,'Bandwidth controls: open fetch, zero inflow',False)
    tab(a,['Operand copies r','Worlds','H bits','Fetch fraction'],[(r,len(q),f'{q.H_bits.median():.1f}',f'{q.entered_frac.median():.2f}') for r,q in d2.groupby('lit_rep')],[30,17,17,18])
    note(a,23,'Literal P a b emits a,b repeated r times')
    foot(f,2,'World medians of conditional H and partner-fetch fraction\nr≥2: zero partner inflow with fetch in every encounter',b.TEAL)
    d1=read('results/bff_dials/d1_lethality.csv');a=panel(f,3,'Halting dial: all worlds included in counts',False)
    tab(a,['Halt p','Worlds','P tapes\npersist','All-world median\nfinal P share'],[(f'{p:g}',len(q),int(q.persists.sum()),f'{q.share_last.median():.3f}') for p,q in d1.groupby('halt_p')],[18,18,23,23])
    foot(f,3,'p: unmatched bracket halts; P = literal-copy instruction\nPersistence: P-tape share ≥10% at epoch 16,384; final share median')
    q=h.query("machine=='bff'");a=panel(f,4,'First / final inflow and fetch',False)
    rows=[]
    for g,r in q.groupby('group'):
        first=r.query("which=='first'");final=r.query("which=='final'")
        rows.append([variants[g],len(first),f'{first.H_bits.median():.2f} / {final.H_bits.median():.2f}',f'{first.entered_frac.median():.2f} / {final.entered_frac.median():.2f}'])
    tab(a,['Variant','n worlds','H bits\nfirst / final','Fetch fraction\nfirst / final'],rows,[29,10,22,21],size=5.2)
    note(a,15,'First = first modal-life tape; final = terminal modal tape')
    foot(f,4,'Paired world sampling; values are medians across worlds\nFetch fraction and conditional H use the same units as b,c')
    q=read('results/biology/individuality/predictions.csv');a=panel(f,5,'Registered individuality verdicts',False)
    names={'Q1':'Z80 first H>1','Q2':'Z80 loop final H<0.5','Q3':'Z80 H drop ≥2','Q4a':'BFF standard first H<0.5','Q4b':'BFF literal first H>1'}
    tab(a,['Claim (bits)','Worlds pass/n','Verdict'],[(r.prediction+' '+names[r.prediction],f'{r["count"]}/{r.n}',r.outcome.replace('between (neither met nor killed)','between').replace('KILL','rejected')) for _,r in q.iterrows()],[46,21,15],size=5.1)
    foot(f,5,'Q1–3 Z80 G; Q2 only 67 loop-bearing finals; Q4a standard BFF\nQ4b pools literal/no-wrap, wrap+literal, wrap+literal/harmless (36)\nSupport ≥90% (Q3 ≥75%); reject ≤60% (Q3 ≤50%); else between',b.RED)
    path=EXP/'results/bff_dials/FINDINGS.md';SOURCES[7][str(path.relative_to(EXP))]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(f,7)


def ed8():
    f=page(8);d=read('results/ladder/lod_groups.csv');a=panel(f,0,'Genealogy cohorts: worlds and nested lines',False)
    tab(a,['Cohort','Worlds','Confined\nworlds','Lines','Open-first\nlines'],[(r.cohort,r.worlds,r.closed_worlds,r.sampled_lines,r.open_first_lines) for r in d.itertuples()],[25,13,13,14,17])
    note(a,24,'N3-v6 = discovery; N6 = prospective confirmation')
    note(a,17,'Open-first: earliest copier on each sampled ancestry has partner fetch')
    foot(f,0,'World = simulation; line = nested sampled-cell ancestry\nConfined worlds have a first confined founder')
    a=panel(f,1,'First founders versus later conversions',False)
    ds={name:read(f'results/{family}/founders2.csv') for name,family in [('N3-v6','lod'),('N6','lod_n6')]}
    tab(a,['Cohort','First\nfounders','Later\nconversions','Events'],[(name,int(q['first'].sum()),int((~q['first'].astype(bool)).sum()),len(q)) for name,q in ds.items()],[27,17,20,18])
    note(a,24,'20 worlds/cohort; at most one first founder/world')
    note(a,17,'Later conversion events can repeat within a world')
    foot(f,1,'First founders and subsequent conversion events stay separate')
    d=read('results/lod_n6/founders2.csv');q=d[d['first'].astype(bool)];a=panel(f,2,'N6 first-founder mechanisms',False)
    tab(a,['Recorded mechanism','Founders'],list(q.F_kind.value_counts().items()),[61,21])
    note(a,25,'novel = rewrite, <75% match to either parent (18)')
    note(a,18,'damage = partial overwrite, ≥75% parent match (2)')
    foot(f,2,'Separate N6-1 source rule: copy/rewrite 18/20, mutation 0/20\nRecorded event categories and causal source classes differ')
    d=read('results/lod_n6/founding.csv');a=panel(f,3,'Functional donors: two definitions',False)
    tab(a,['Criterion','Whole-tape','Executed bytes'],[['Non-copier',int(d.executor_class.eq('non-copier').sum()),int(d.executor_exec_copier.lt(.5).sum())],['Copier',int(d.executor_class.ne('non-copier').sum()),int(d.executor_exec_copier.ge(.5).sum())]],[34,24,24])
    b.text(a,0,16,'Executed-byte non-copiers: 11/20',size=6.5,bold=True)
    b.text(a,0,7,'All whole-tape noncopier executors vs soup-abundance null: P=0.062',size=6,color=b.RED)
    foot(f,3,'20 N6 executors; whole tape vs only instruction-stream bytes\nCopier: ≥75% scoped bytes copied in ≥50% partner encounters\nN6-4 needs 12/20 executed-byte noncopiers; 11/20 observed: not met')
    d=read('results/lod_n6/chain_v6b.csv');first=ds['N6'].query('first==True');q=first;assert len(q)==q.seed.nunique()==20;a=panel(f,4,'N6: all first-founder chain lengths')
    a.scatter(q.steps_elapsed,q.chain_records,s=15,color=b.TEAL);b.tidy(a,'Updates from chain start','Contributing chain records')
    a.set_xlabel('Updates: chain start to first founder')
    foot(f,4,f'N6: one point per first founder, {len(q)} worlds\nRecords = contributing chain events; intervening whole-tape\ndonors are non-copiers (executed-byte function kept separate)')
    a=panel(f,5,'Registered N6: complete component ledger',False)
    p=EXP/'results/lod_n6/N6_SCORE.md';SOURCES[8][str(p.relative_to(EXP))]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows=[['N6-1','Copy/rewrite; mutation','18/20; 0/20','met'],['N6-2','Whole-tape noncopier chains','20/20','met'],['N6-3','Executed-word share','Observed 0.00190\n7.9× baseline','met'],['N6-4','Whole/exec-byte noncopiers','20/20; 11/20\nP = 0.062','not met'],['N6-5′','Expected mutation founders','0.0064 in cohort','met']]
    tab(a,['ID','Component','Observed','Verdict'],rows,[12,32,24,14],size=5.2)
    foot(f,5,'N6-1 copy/rewrite ≥80%, mutation ≤10%; N6-2 whole-chain ≥75%\nN6-3 median cells carrying executed word ≥5×random baseline 0.00024\nN6-5′ expected mutation founders <0.05/cohort; ≥25% mutation rejects')
    save(f,8)


def ed9():
    f=page(9);d=read('results/s8086/tables/worlds.csv');a=panel(f,0,'Registered S2: all 20 worlds')
    for i,r in enumerate(d.itertuples()):
        t=r.censor_step if r.rep_right_censored else r.t_rep
        a.plot(t/1e6,i,marker='|' if r.rep_right_censored else 'o',color=b.GREY if r.rep_right_censored else b.TEAL if r.first_fetch=='confined' else b.RED,ms=3.5,ls='none')
    a.set(xlim=(-.1,3.15),ylim=(-1,20));a.set_yticks([0,5,10,15,19],[str(x) for x in d.seed.iloc[[0,5,10,15,19]]]);b.tidy(a,'Detection / censor time (million updates)','World seed')
    foot(f,0,'S2: coherent-fetch 8086; first detected heritable copier\nTeal: no partner stream fetch; red: fetch ≥8/16 encounters\nGrey ticks: 6 nondetections at 3 million world updates')
    p=read('results/s8086/tables/persistence.csv');a=panel(f,1,'R2: open memory-fed persistence')
    q=p.groupby('step').event_present.sum();a.plot(q.index/1e6,q,color=b.TEAL,lw=1,marker='o',ms=2);a.set(xlim=(2.73,3.02),ylim=(-.15,1.15));b.tidy(a,'World updates (millions)','Worlds with open memory-fed tape',yticks=[0,1]);b.text(a,2.77,.65,'0/20 at all 11 terminal samples',size=6.5)
    foot(f,1,'Same 20 S2 worlds as a; terminal persistence = 0/20\nEarlier samples combined: 1/20 worlds ever positive by 3M')
    d=read('results/s8086/tables/law_denominators.csv');a=panel(f,2,'Registered evolved law: untested',False)
    tab(a,['Family test','Candidates','Eligible','Unresolved'],[(r.family+(' excluding L1' if r.exclude_L1 else ''),r.unique_candidates,r.eligible,r.unresolved_extractions) for r in d.itertuples()],[30,19,15,18])
    b.text(a,0,15,'L2 support suspended: zero certified cases',size=6,color=b.RED)
    b.text(a,0,7,'Zero eligible cases do not mean zero capacity',size=5.5)
    foot(f,2,'c–e: candidate tapes from same 20 S2 worlds as a\nL1 frozen overlap / L2 broader pure-copy certificate; d = L2 ledger')
    d=read('results/s8086/tables/law_candidates.csv');q=d.query("family=='L2'");a=panel(f,3,'Registered exclusions and unresolved cases',False)
    counts=q.groupby(['status','reason'],dropna=False).size();rows=[]
    for (status,reason),n in counts.items():rows.append([str(status),str(reason).replace('unmodelled_dependency_opcode:07','Opcode 07 (POP ES)').replace('unmodelled_dependency_opcode:', 'Opcode ').replace('_',' '),n])
    tab(a,['Status','Reason','n'],rows,[27,46,9],size=5.2)
    foot(f,3,'L2 candidate tapes: 558 culture-negative, 55 excluded\n724 unresolved (708 at POP ES); total 1,337')
    e=read('results/s8086/exploratory/eligibility.csv');p=read('results/s8086/exploratory/law_assay_summary.csv');a=panel(f,4,'EXPLORATORY: admissions and exact-law panel',False)
    rows=[]
    for dd,q in p.groupby('d'):rows.append([dd,len(q),q.world.nunique(),f'{int(q.encounters.sum()):,}',str(q.status.unique()[0]).replace('complete-','')])
    tab(a,['d bytes','Genotypes','Worlds','Encounters','Mismatches'],[(r[0],r[1],r[2],r[3],'0') for r in rows],[12,17,13,25,15],size=5.2)
    b.text(a,0,19,'779 culture-positive world/tape records -> 600 admitted -> 32 selected',size=5.5)
    b.text(a,0,12,'Selected L16 genotypes: d4 has zero free coords; d16 zero erased coords',size=5.5)
    b.text(a,0,5,'32 genotypes / 20 worlds; 2,224,640 encounters',size=5.5)
    foot(f,4,'d = physical copy offset; 32 selected genotypes from admissions\nMismatch: predicted vs actual bytes, core and ancestry\nPost hoc; 590 admitted d16 +10 d4; not registered evidence')
    d=read('results/s8086/exploratory/first_family.csv');ef=read('results/s8086/exploratory/encounter_fates.csv')
    c=read('results/s8086/exploratory/first_family_copies.csv');a=panel(f,5,'EXPLORATORY 5F A4 x y: d4 physical copies')
    assert len(c)==640 and c.world.nunique()==10 and not c.original_partner_read.any()
    a.scatter(c.read_position,c.write_position,s=7,color=b.TEAL)
    a.set(xlim=(-1,33),ylim=(-1,33));b.tidy(a,'Physical read address','Physical write address',[0,16,31],[0,16,31])
    a.set_title('640 writes /10 first 5F A4 tapes; repeated pairs overlap',fontsize=5.5,pad=3)
    a.text(15,7,'Write = (read + 4) mod 32\nByte addresses; wraps at 32',fontsize=5.5)
    med=ef.groupby('isa').new_exact.median()
    foot(f,5,f'Snapshot fates: median new tiled offspring /512 encounters\nS2 20 worlds: {med["8086"]:g}; Z80 G 20 worlds: {med["Z80"]:g}; five times/world\nAcross soup/uniform partners and donor A/B roles; separate from map')
    save(f,9)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',default='1,2,3,4,5,6,7,8,9');args=ap.parse_args()
    b.fs.setup('nature');plt.rcParams.update({'font.size':6,'axes.labelsize':6,'xtick.labelsize':5.5,'ytick.labelsize':5.5})
    existing=json.loads((OUT/'ed_sources.json').read_text()) if (OUT/'ed_sources.json').exists() else {}
    for n in map(int,args.only.split(',')):globals()[f'ed{n}']()
    checks=json.loads((OUT/'figcheck.json').read_text());checks.update(b.REPORTS);(OUT/'figcheck.json').write_text(json.dumps(checks,indent=2)+'\n')
    for n,sources in SOURCES.items():existing[f'ed8_{n}']={'height_mm':HEIGHTS[n],'sources':sources}
    (OUT/'ed_sources.json').write_text(json.dumps(existing,indent=2)+'\n')
    lines=['# Extended Data v8','', 'Nine numbered displays follow the MAIN_v7 Extended Data map. Sources below resolve from `experiments/`. Figures use the existing palette, publication typography and frozen numerical tables. Full ledgers and proofs accompany the concise displays; no simulation was run.','', '**Unavailable measurements:** Pointer randomization was not measured and was not an S2 arm. No corresponding observation is plotted. The standard BFF harmless-bracket variant has no archived individuality-entropy panel in the source tables; it is shown only in the detection counts. All nine ED items have archived sources.','']
    for n in range(1,10):
        assert len(LEGENDS[n].split())<=250
        lines += [f'## Extended Data {n}. {TITLES[n]}','',f'[PDF](ed8_{n}.pdf) · [PNG](ed8_{n}.png)','',f'**Legend draft ({len(LEGENDS[n].split())} words).** '+LEGENDS[n],'','**Sources:**']
        for path in existing.get(f'ed8_{n}',{}).get('sources',{}):lines.append(f'- `{path}`')
        if n==2:
            pe=pd.read_csv(OUT/'data/parent_environment.csv');lines += ['','**Environment IDs for panel d:**']+[f'{i+1}. {s}' for i,s in enumerate(sorted(pe.environment.unique()))]
        lines.append('')
    (OUT/'ED_V8.md').write_text('\n'.join(lines))
    if any(b.REPORTS.values()):raise SystemExit('Fix ED figure layout before critique.')


if __name__=='__main__':main()

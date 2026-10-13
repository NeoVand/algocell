"""Render the nine SI_v8 Extended Data displays from archived evidence, CPU only.

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
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties
import math
import build_v8 as b

HERE=b.HERE
EXP=HERE.parents[1]
OUT=b.OUT
SOURCES={}
CURRENT=0
HEIGHTS={n:(290 if n==5 else 180 if n in [1,3] else 220) for n in range(1,10)}
TITLES={1:'Validation and correction impact',2:'Survival, carriage and erasure',3:'Inherited displacement and capacity',4:'Sampling, reset and classification',5:'Environmental competitions',6:'Machine laws and finite censuses',7:'BFF literal and environmental controls',8:'Founder genealogies',9:'Registered S2 and separate follow-ups'}
RING_CRITERION='Stable: upper 95% CI of changed-final-memory pair fraction <0.01%'
SITE_CRITERION='Site: ≥half identifiable tested values have excess carrier share ≥0.5'
FREE_DEFINITION='Free byte positions: arbitrary alternatives jointly inherited with causal roots'
GAIN_DEFINITION='Gain = mean (after−before)/(1−before) byte-match fraction'
LEGENDS={
1:'a, Validation comparisons identify their observational units and agreement criteria: instruction-vector states, ring conditions, isolated-assay metrics, claim verdicts and component outcomes. All 10,083 hardware divergences are expected. b, A ring condition is stable when the upper bound of its two-sided 95% Clopper–Pearson interval for the fraction of pairs with changed final memory is below 0.01%, using 100,000 encounter draws per condition. Flags indicate failure of that criterion. Historical core C0, corrected core C1 and core C2 identify emulator implementations, distinct from displacement experiment C1 and 6502 retention experiment C2. c, Nineteen of 22 I1-T2 claims retain every constituent verdict, including prior failures. d, Four of 36 components change; applicability changes are distinguished from newly met criteria. e, Counts of differing measurements use 417 isolated assays per core comparison; rows overlap. f, All ten Stage-L worlds at each tape length complete 10,000,000 updates and are scoreable from retained final bytes and journals, although their 7,500-update snapshots are missing. Applied-mutation and collision counts are unavailable for these worlds. Expected hardware divergences and stochastic rerun differences do not isolate a causal effect of core repair.',
2:'a, Historical serial transfer-eight outcomes over all identifiable mutant starts: lost, alive without the mutant marker window, or alive with it. Each serial population contains one descendant. b, Registered branching transfer-eight outcomes distinguish loss, survival with fewer than half of members carrying the marker, and survival with at least half carrying it. The marker is the centred cyclic five-byte window of the introduced mutant. A scored site requires at least half of identifiable tested values to have an excess carrier share of at least 0.5: the fraction of surviving mutant populations classified as carriers minus the corresponding fraction of surviving unmutated populations. Serial populations use the same criterion with one member. c, Complete serial site maps at transfers one and eight. d, Carrier lineages divided by all mutant starts for six L16 parents across partner environments; each design has 65,280 mutant starts and 256 unmutated controls. Benign zero opcodes execute; lethal zero opcodes halt. e, Identifiable alternative types and unmutated starts for the serial designs. f, One block-copy L32 lineage against all-zero byte partners, through two 128-execution-step encounters. At step six, a read from address four carrying byte-zero ancestry overwrites the final byte-eight marker descendant. Survival requires at least 75% immediate-parent byte match at the best cyclic shift; marker carriage and ancestry are separate measurements.',
3:'a, All 101 registered displacement experiment C1 settings in 22 Z80 parents, including 22 original controls. Free byte positions admit arbitrary alternatives jointly inherited with their causal roots while the core is fixed; markers coincide at eight coordinates with setting counts shown. b, All direct single-site, joint-payload and erased-coordinate challenges, separately for forward and reciprocal passages, total 1,067,608 cases without mismatch; each case uses 17 partners. c, Actual transfer-eight descendant bytes and original-parent ancestry for parallel byte-zero interventions 0x04 and 0x60. d, The exploratory Z80 L32 first-byte sweep partitions all 256 values into 157 culture-negative, 88 culture-positive but theorem-ineligible and 11 culture-positive eligible values. Culture positivity requires dimensionless generation-two gain of at least 0.3 on 16 uniform partners; gain averages the increase in parent-byte match divided by available match headroom. e, The exploratory 5e ed b0 motif-bearing cell cohort from stages G, I, K and L; extra fetched-site scores and the immediate-parent whole-tape-match gate deficit are distinct outcomes. f, Frozen eligibility accounting for those 2,347 cells, and zero mismatches among 25,856 registered descendant-transfer records. Exact bytes, fixed core and ancestry each use that serial denominator. Eligibility and sampled scores do not certify a universal inheritance law.',
4:'a, A separate random-register-reset cohort of 20 Z80 L16 worlds, with six observation updates. Confinement fractions use sampled heritable cells as denominator; two worlds are majority confined at 3,000,000 updates. b, Final-world modal-tape abundance versus modal identity among sampled heritable cells in the 170 worlds of panel c. c, Sampled-cell and world denominators by stage and tape length; confined and transmitter cells are subsets of the pooled cell rows. d, The Z80 L50 matched-pair invasion cohort comprises three worlds and 24 samples of 512 cells. Axes compare confinement among cells with and without the 20 f0 sequence marker; seed one at update zero has five of five marker cells confined and zero of 507 nonmarker cells confined. e, Exploratory order of first regenerator and transmitter detections at any positive estimated share. f, Complete sensitivity to majority and greater-than-10% class shares among heritable cells. Both-detected worlds equal the sum of the three mutually exclusive order columns. Regenerator and transmitter here mean confined site-score classes with at most two and at least five sites, respectively; they differ from competition byte classes. Observation order does not establish ancestry.',
5:'a–d, Historical core C0 and corrected Z80 core (C1) L32 competitions under benign and lethal zero-opcode chemistry with mutation on, through 300,000 updates. Thin curves show individual worlds; thick curves show within-start-group medians. Each environment has ten 50:50 worlds and five worlds per reciprocal 1% invasion. R and T are partial-copy and full-tape byte classes. e, Complete endpoint medians for 110 worlds per implementation, covering both tape lengths, chemistries, mutation settings and starts. Mutation-off endpoints are at 100,000 updates. T share is T/(R+T); core share counts 5e ed b0-bearing cells over 20,000 cells; tail types count distinct sequences at bytes four through L−1 per world. The corrected-core L32 benign T-at-1% median uses four defined ratios among five worlds, while core and tail medians include all five, including zeros. f, Exploratory historical-core L32 continuations report mean T share at start and 2,000 updates. The lethal L32 host control reports unchanged-host-byte fractions over 400 hosts times 64 soup-executor encounters per treatment. Detoxification replaces payload 0x00 and 0x76 with non-toxic bytes. Registered E1 under the corrected core, benign chemistry and mutation off meets its final-share interval in only four of ten worlds at each length, below seven required. Variable endpoints do not establish neutral drift.',
6:'a, Complete finite 6502 census accounting by tape length, BRK contract, execution-step budget and tiled-word period, including the four-billion-word period-four arms. Exact self-writers produce exact output into an all-zero partner. b, Complete finite 8086 census counts under the declared reset; nonzero words qualify by exact output into zero partners or at least 75% parent-byte match in at least eight of 16 uniform-partner encounters. c, The 296 constructed copier/contract configurations coincide at five coordinate groups. A payload position is carried when its parent byte remains at the same offspring position for all three sampled bodies and 64 partners per body; this is distinct from a mutant-value site score. d, S1 first-route counts for four 20-world 6502 arms; open and confined are fetch classifications, whereas loops overlap routes. e, The complete 6502 retention experiment C2 sampling ledger contains 942 roles from 80 S1 worlds. Its 42 eligible roles represent 41 tapes from 11 worlds; role-level world counts overlap. f, Predicted and measured free byte positions agree for all eligible roles. Exact-byte, fixed-core and ancestry checks each have zero mismatches over 91,428 direct cases and 731,424 serial records. Free positions admit arbitrary alternatives jointly inherited with causal roots. Pointer randomization is unmeasured and was not an 8086 S2 arm.',
7:'a, All archived BFF variants with modal detection, population detection and later collapse counts. Modal detection requires headroom-normalized generation-two parent-byte-match gain of at least 0.3; population detection requires at least half of sampled cells to satisfy the same heritability criterion. Collapse is terminal heritable share below 10% after population detection. Literal writing emits operands; wrap permits pointer wrap; harmless variants prevent unmatched brackets from halting. b, Complete paired fetch-fraction and conditional-entropy counts for first-modal tapes, one per detected world. All distinct observed pairs have explicit tape counts; values are rounded to five decimals. H is offspring entropy conditional on the fixed parent, measuring partner inflow over 256 uniform partners per tape. The standard harmless-bracket variant has no archived entropy panel. c, Literal-no-wrap repetition controls: open fetch can coexist with zero sampled partner inflow. d, Wrap-plus-literal halting controls retain all worlds in counts and final P-tape-share medians at 16,384 BFF population updates, called epochs in the archive. e, Paired first-versus-final world medians of entropy and fetch. f, Registered individuality results pair each pass fraction with its support and rejection thresholds. Z80 claims use 80 Stage G worlds, except the 67 loop-bearing finals; BFF Q4b pools three literal variants of 12 worlds each. Q4b is rejected and Q4a lies between decision boundaries. Entropy, confinement and dependence on initial partner contents remain separate measurements.',
8:'a, Z80 L16 discovery and N6 confirmation each comprise 20 worlds. Sixty-four ancestral lines are sampled from each world with confined founders: 17 discovery and 20 N6 worlds. The genealogy classifier requires at least 75% tape-byte copying in at least 32/64 uniform partners; its open/confined classes count any partner opcode or operand fetch in at least/fewer than 32/64 encounters. Open-first refers to the earliest copier on each line. b, First founders and later open-ancestry-to-confined-founder events have separate denominators. c, N6 founding-event categories distinguish rewrites from partial overwrites; recorded mechanisms and registered source classes are separate. d, One executor per N6 first founder is classified under whole-tape and executed-byte copying scopes. A copier reproduces at least 75% of scoped bytes in at least half of partner encounters. All 20 executors are whole-tape non-copiers, but only 11 are executed-byte non-copiers, below 12 required. The probability of all 20 being whole-tape non-copiers under independent draws from pre-founder soup composition is 0.062. e, Chain records and elapsed updates for all 20 first founders. f, Every registered N6 component includes its decision boundary and observation. N6-3 measures cells carrying two-byte words at founder-executed instruction starts divided by all soup cells; its median over 99 word/founder records is 0.00190 against a 0.00024 random-tape baseline, a 7.9-fold ratio. The 55 recorded point-mutation events on 20 pre-founder chains imply 0.0064 expected confined founders; zero of 20 founders are observed mutation completions. N6-4 remains not met; the five-of-20 point-mutation kill is not triggered.',
9:'a, All 20 registered coherent-fetch 8086 L16 S2 worlds: 14 first-copier detections and six right-censored nondetections at 3,000,000 updates, with open and confined fetch labels. b, Descriptive R2 persistence in the same worlds: zero open memory-fed tapes at all 11 terminal samples; earlier scheduled samples combined contain one ever-positive world. c–d, Registered law accounting from these S2 worlds retains culture-negative, excluded and unresolved candidate tapes. No case is certified eligible: L1/L2 remain untested and L2 support is suspended. e, Separate exploratory admissions and selected post hoc exact-law assays from the same 20 worlds: 779 culture-positive world/tape records lead to 600 admissions, comprising 590 d16 and ten d4 tapes, and then 32 selected genotypes. All 2,224,640 encounters check predicted bytes, core and ancestry without mismatch. Copy displacement d is the write-minus-read byte offset. Nonempty perturbations comprise d4 erasure and d16 single/joint payload; d4 joint payload and d16 erasure have zero applicable coordinates and no corresponding perturbations. Empty-assignment controls remain in encounter counts. f, Exploratory geometry for ten d4 genotypes with hexadecimal byte pattern 5F A4 u v, where u/v are variable bytes; repeated read/write pairs coincide. The separate post hoc snapshot-fate result pools five samples per world, soup/uniform partners and donor A/B roles; medians are totals per 512-encounter batch of non-donor 16-byte halves newly matching the donor’s two-byte tiling, across 20 S2 and 20 Z80 G L16 worlds. S2 uses three donor tiles and Z80 one. These follow-ups do not supply registered evolved-law evidence.'}


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
    a=b.axes(f,x+14,top-47,65,39) if plot else b.axes(f,x+1,top-55,82,51,True)
    a._ed_panel=i
    return a


def wrap(s,width,size=6.5,bold=False):
    """Wrap to physical millimetres without dropping a word or a byte token."""
    renderer=plt.gcf().canvas.get_renderer()
    prop=FontProperties(size=size,weight='bold' if bold else 'normal')
    limit=width*plt.gcf().dpi/25.4
    lines=[]
    for line in str(s).split('\n'):
        current=''
        for word in line.split(' '):
            candidate=(current+' '+word).strip()
            if current and renderer.get_text_width_height_descent(candidate,prop,False)[0]>limit:
                lines.append(current);current=word
            else:current=candidate
        lines.append(current)
    return '\n'.join(lines)


def foot(f,i,s,color=b.INK,width=82):
    x=3+92*(i%2);top=f._height_mm-6-70*(i//2)
    t=f.text(x/180,(top-64)/f._height_mm,wrap(s,width),fontsize=6.5,color=color,va='top',linespacing=1.25)
    t._ed_footer=i


def tab(a,headers,rows,widths=None,size=5.5,step=None):
    widths=widths or [82/len(headers)]*len(headers)
    xs=np.r_[0,np.cumsum(widths)]
    size=max(6.5,size);top=48;step=max(3.5,step or min(4.5,43/max(len(rows),1)))
    hs=[wrap(h,w-1,size,True) for h,w in zip(headers,widths)]
    for x,h in zip(xs,hs):a.text(x,top,h,fontsize=size,fontweight='bold',va='top',color=b.INK)
    y=top-max(h.count('\n')+1 for h in hs)*size*1.2*25.4/72-2
    for i,row in enumerate(rows):
        if row[0]=='__GROUP__':
            a.text(0,y,row[1],fontsize=size,fontweight='bold',va='top',color=b.INK)
            y-=4.5
            continue
        values=[wrap(v,w-1,size) for v,w in zip(row,widths)]
        for x,value in zip(xs,values):
            emph=str(value) in ['changed','not met → met']
            a.text(x,y,str(value),fontsize=size,color=b.RED if emph else b.INK,fontweight='bold' if emph else 'normal',va='top')
        y-=max(step,max(v.count('\n')+1 for v in values)*size*1.2*25.4/72+1.2)
    a._table_bottom=y+1.2


def note(a,y,s):
    s=wrap(s,a.get_xlim()[1],6.5)
    y=getattr(a,'_note_bottom',getattr(a,'_table_bottom',y+3.5))-3.5
    a.text(0,y,s,fontsize=6.5,color=b.INK,va='top')
    a._note_bottom=y-s.count('\n')*6.5*1.2*25.4/72


def frac(a,x='',y='Fraction'):
    a.set(ylim=(-.03,1.05));b.tidy(a,x,y,yticks=[0,.5,1])


def save(f,n):
    for item in f.findobj(Text):item.set_fontsize(max(6.5,item.get_fontsize()))
    # Off-axis tables may extend below the original layout box. Their complete
    # maps and bars must remain visible after packing the physical page.
    for a in f.axes:
        if not a.axison:
            for item in list(a.patches)+list(a.lines):item.set_clip_on(False)
    b.figcheck.MIN_PT=6
    # Attach each footer immediately below its own panel, then pack row bands
    # using rendered content bounds. Empty axes do not reserve blank space.
    f.canvas.draw();renderer=f.canvas.get_renderer();oldheight=f._height_mm
    def bounds(a):
        artists=[a] if a.axison else []
        artists+=list(a.texts)+list(a.patches)+list(a.images)+list(a.lines)
        if a.get_legend() is not None:artists.append(a.get_legend())
        if a.axison:artists+=list(a.get_xticklabels())+list(a.get_yticklabels())+[a.xaxis.label,a.yaxis.label,a.title]
        boxes=[v.get_window_extent(renderer) for v in artists if v.get_visible()]
        return [bb for bb in boxes if np.isfinite(bb.extents).all() and bb.height>0 and bb.width>0]
    for t in f.texts:
        if hasattr(t,'_ed_footer'):
            aa=[a for a in f.axes if getattr(a,'_ed_panel',None)==t._ed_footer]
            lo=min(bb.y0 for a in aa for bb in bounds(a))*25.4/f.dpi
            t.set_position((t.get_position()[0],(lo-2.5)/oldheight))
    f.canvas.draw();renderer=f.canvas.get_renderer()
    groups={}
    for a in f.axes:
        row=getattr(a,'_ed_panel',None)
        row=row//2 if row is not None else max(0,int((oldheight-6-(a.get_position().y1*oldheight))/70+.001))
        groups.setdefault(row,[]).append(a)
    for t in f.texts:
        row=t._ed_footer//2 if hasattr(t,'_ed_footer') else max(0,int((oldheight-6-t.get_position()[1]*oldheight)/70+.001))
        groups.setdefault(row,[]).append(t)
    for p in f.patches:
        row=max(0,int((oldheight-6-p.get_y()*oldheight)/70+.001));groups.setdefault(row,[]).append(p)
    extents={}
    for row,items in groups.items():
        boxes=[]
        for item in items:boxes+=bounds(item) if hasattr(item,'axison') else [item.get_window_extent(renderer)]
        extents[row]=(min(bb.y0 for bb in boxes)*25.4/f.dpi,max(bb.y1 for bb in boxes)*25.4/f.dpi)
    height=math.ceil(sum(hi-lo for lo,hi in extents.values())+6*(len(groups)-1)+8)
    cursor=height-4
    for row in sorted(groups):
        lo,hi=extents[row];delta=cursor-hi
        for item in groups[row]:
            if hasattr(item,'axison'):
                p=item.get_position();item.set_position([p.x0,(p.y0*oldheight+delta)/height,p.width,p.height*oldheight/height])
            elif isinstance(item,Text):
                x,y=item.get_position();item.set_position((x,(y*oldheight+delta)/height))
            else:item.set_y((item.get_y()*oldheight+delta)/height);item.set_height(item.get_height()*oldheight/height)
        cursor-=hi-lo+6
    HEIGHTS[n]=height
    f.set_size_inches(180*b.fs.MM,height*b.fs.MM);f._height_mm=height
    b.save(f,f'ed8_{n}')


def ed1():
    f=page(1);d=frozen('validation_fractions')
    a=panel(f,0,'Validation: distinct denominators',False)
    labs=['8088/hardware: identical vector states','Corrected/historical: below change-rate bound','Corrected/historical: identical assay metrics','Corrected/historical: unchanged claim verdicts','Corrected/historical: unchanged components']
    tab(a,['Comparison / agreement criterion','Met / tested units','Not met, n'],[(lab,f'{int(r.unchanged):,}/{int(r.total):,}',f'{int(r.changed):,}') for lab,r in zip(labs,d.itertuples())],[44,25,13],size=5.1)
    note(a,17,'All 10,083 hardware divergences were expected')
    foot(f,0,RING_CRITERION+'\n100,000 encounter draws/condition; units named in each row')
    d=read('results/impact/outcome.csv');a=panel(f,1,'Stability flags and assay differences',False)
    tab(a,['Core','Flagged / conditions','Assays differing / tested'],[(('Corrected' if r.core=='C1' else r.core),f'{r.flagged}/{r.decision_conditions}',f'{r.assay_differences}/{r.assay_combinations}') for r in d.itertuples()],[12,33,37])
    note(a,31,'Comparator: historical compatibility core C0')
    note(a,26,'Corrected core (C1): classic; core C2: Zilog NMOS')
    note(a,21,'Overall stability verdict: criterion not met for either core')
    foot(f,1,RING_CRITERION+'\nFlag = stability criterion not met; assays compare with core C0')
    d=read('results/tier2/T2_CLAIMS.csv');a=panel(f,2,'I1-T2: all 22 claim verdicts',False)
    half=(len(d)+1)//2;rows=[]
    for i in range(half):
        r=d.iloc[i];s=d.iloc[i+half] if i+half<len(d) else None
        rows.append([r['claim'],r.status,s['claim'] if s is not None else '',s.status if s is not None else ''])
    tab(a,['Claim','vs archive','Claim','vs archive'],rows,[13,29,13,27])
    foot(f,2,'DZ2 timing; DZ3 transmitter trend; L2 recovery sites\nStands = same as archive; E1 stability remains failed; 19/22 stand')
    d=read('results/tier2/T2_COMPONENTS.csv');q=d[d.published_verdict!=d.C1_verdict];a=panel(f,3,'Four changed components of 36',False)
    tab(a,['Component','Historical','Corrected core'],[(r.claim+(' L16' if 'L16' in r.registered_component else ' L20' if 'L20' in r.registered_component else ''),r.published_verdict.replace('n/a (no recovery)','n/a'),r.C1_verdict.replace('n/a (no recovery)','n/a')) for r in q.itertuples()],[28,27,27])
    note(a,20,'n/a = not applicable: no recovered lineage')
    note(a,14,'Component tests summarized in panel a; L = tape bytes')
    note(a,8,'DZ2/DZ3: criteria now met; L2: applicability changes')
    foot(f,3,'Historical = published archive; DZ timing/share; L2 sites\nComplete 36-component ledger retained in Source Data')
    d=read('results/impact/assay_columns.csv');a=panel(f,4,'Assay changes by measurement',False)
    names={'copy_into_neighbor_as_A':'Partner-copy fraction, parent A','copy_into_neighbor_as_B':'Partner-copy fraction, parent B','baseline_similarity':'Initial byte-match fraction','n_informative':'Informative partners, generation 1 (n)','n_informative2':'Informative partners, generation 2 (n)','copy_offset':'Cyclic copy offset (bytes)','self_preserved_as_A':'Parent-byte fraction preserved, A','self_preserved_as_B':'Parent-byte fraction preserved, B','offspring_within_q':'Offspring ≥75% parent-match fraction','gen2_score':'Generation-2 normalized gain','gen2_cond':'Generation-2 conditional gain','score':'Normalized culture gain','is_replicator':'Generation-2 gain ≥0.3 (yes/no)','faithful':'Faithful culture classification (yes/no)'}
    tab(a,['Metric','Differing\ncorrected core','Differing C2'],[(names.get(r.column,r.column.replace('_',' ')),r.c1_changed,r.c2_changed) for r in d.itertuples()],[48,17,17],size=5.2)
    foot(f,4,'Counts /417 assays vs core C0; rows overlap (not additive)\nA/B = parent role; copy = ≥75% immediate-parent byte match')
    d=read('results/tier2/T2_StageL_status.csv');a=panel(f,5,'Stage-L scoring and snapshot availability',False)
    rows=[]
    for L,q in d.groupby('L'):
        rows.append([L,len(q),int(q.scoreable.sum()),int(q.missing_steps.fillna('').astype(str).ne('').sum())])
    tab(a,['L bytes','Worlds, n','Scoreable from\nfinal data, n','Missing 7,500-update\nsnapshot, worlds'],rows,[12,14,25,31],size=5.2)
    note(a,31,'All worlds completed 10,000,000 updates')
    note(a,26,'Retained final bytes and journals support scoring')
    note(a,21,'Applied-mutation/collision counts unavailable for these worlds')
    foot(f,5,'Stage L: closure-recovery worlds, 10 / tape length\nThe missing observation is a world snapshot at 7,500 updates')
    for p in ['results/impact/supplement_manifest.json','results/tier2/stageL_diagnosis.json']:
        path=EXP/p;SOURCES[1][p]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(f,1)


def design_family(key):
    if '8080' in key:return 'Constructed 8080 controls'
    if key.startswith('genome') or key=='lethal16_s4009':return 'Evolved Z80 transmitters'
    if key.startswith('pusher'):return 'Z80 pushers'
    return 'Evolved Z80 closers'


def family_rows(d,top=43,span=58):
    families=[design_family(k) for k in d.key]
    boundaries=[i for i,g in enumerate(families) if i==0 or g!=families[i-1]]
    step=span/(len(d)+len(boundaries)-1)
    y=top
    for i,r in enumerate(d.itertuples()):
        if i in boundaries:
            yield y,families[i],None
            y-=step
        yield y,None,(i,r)
        y-=step


def partition(a,d,labels,control,sites,grouped=False):
    n=len(d);h=34/max(n,1);top=40
    b.text(a,0,48,'Design (L bytes)',size=5.5,bold=True);b.text(a,59,48,'Controls\nalive',size=5.5,bold=True);b.text(a,72,48,'Scored\nsites',size=5.5,bold=True)
    if grouped:
        for y,heading,entry in family_rows(d):
            if heading:a.text(0,y,heading,fontsize=5.2,fontweight='bold',va='center')
        positions={entry[0]:y for y,heading,entry in family_rows(d) if entry}
    for i,(idx,r) in enumerate(d.iterrows()):
        y=positions[i] if grouped else top-i*h
        a.text(1 if grouped else 0,y,labels[i],fontsize=5.1,va='center',color=b.INK)
        left=29
        for col,color in [('lost',b.old.LOST_C),('erased',b.old.ERASED_C),('retained',b.TEAL)]:
            v=float(r[col]);a.add_patch(Rectangle((left,y-.65),27*v,1.3,fc=color,ec='none'));left+=27*v
        a.text(59,y,control[i],fontsize=5.1,va='center');a.text(73,y,str(sites[i]),fontsize=5.1,va='center')
    bottom=min(positions.values())-4 if grouped else 4
    for x,lab in [(29,'0'),(42.5,'0.5'),(56,'1')]:a.text(x,bottom,lab,fontsize=6.5,ha='center',va='center')
    a.text(42.5,bottom-3.5,'Share of mutant starts',fontsize=6.5,ha='center',va='center')


def design(key):
    names={'ret16':'Return L16','ldir20':'Block-copy L20','ldir32':'Block-copy L32','jr50':'Open jump L50','lethal16_s4009':'Lethal T L16','genome16_s6006':'T L16 seed 6006','genome20_s6003':'T L20 seed 6003','genome20_s6004':'T L20 seed 6004','pusher32_8080':'Open 8080 pusher L32','closer32_8080_p1':'Copier L32, variant 1','closer32_8080_p2':'Copier L32, variant 2'}
    return names.get(key,'Pusher L'+key.removeprefix('pusher'))


def ed2():
    f=page(2);ss=frozen('serial_summary');sites=frozen('serial_sites')
    q=ss.copy();q['lost']=q.lost_g8;q['erased']=q.erased_g8;q['retained']=q.retained_g8
    a=panel(f,0,'All historical serial designs: transfer 8',False)
    partition(a,q,[design(r.key) for r in q.itertuples()],[f'{round(r.ctrl_alive_g8*256)}/256' for r in q.itertuples()],q.sites_g8.tolist(),grouped=True)
    foot(f,0,'Dark grey lost; light grey alive noncarrier; teal alive carrier\nCarrier = tape contains centred cyclic mutant 5-byte window\n'+SITE_CRITERION+'\nExcess = carriers/survivors (mutant) − carriers/survivors (WT)')
    d=frozen('branching').query("cohort=='Historical' and gen==8")
    a=panel(f,1,'Branching transfer-8 outcomes: U / K',False)
    partition(a,d,[design(r.key)+f' {r.arm}' for r in d.itertuples()],[f'{r.ctrl_alive_n}/{r.ctrl}' for r in d.itertuples()],d.sites.tolist())
    foot(f,1,'Grey: no surviving lineage; pale: alive, <50% members carry\nTeal: alive, ≥50% members carry\nCarriage = members containing mutant 5-byte window / all members\n'+SITE_CRITERION+'\nExcess as a; U = uniform-byte partners; K = evolved-soup partners\n32 values×4 starts/site; WT = unmutated control (64)')
    a=panel(f,2,'Complete serial site maps: 1 / 8',False)
    b.text(a,31,49,'Transfer: upper 1 / lower 8',size=6.5)
    for y,heading,entry in family_rows(ss,top=43,span=65):
        if heading:
            a.text(0,y,heading,fontsize=5.2,fontweight='bold',va='center');continue
        i,r=entry;L=int(r.L)
        a.text(1,y,design(r.key),fontsize=5.1,va='center')
        for gen,off in [(1,.85),(8,-.85)]:
            s=sites.query('key==@r.key and gen==@gen').set_index('pos').transmissible.reindex(range(L)).fillna(False)
            for j,v in enumerate(s):a.add_patch(Rectangle((31+j*49/L,y+off-.65),49/L,1.3,fc=b.TEAL if v else '#EEEEEE',ec='white',lw=.1))
    foot(f,2,'Upper: transfer 1; lower: transfer 8\nTeal = scored site; pale = tested, unscored site\n'+SITE_CRITERION+'\nSame excess definition as a; tracks span bytes 0 to L−1')
    d=frozen('parent_environment');pivot=d.pivot(index='parent',columns='environment',values='retained_g8')
    a=panel(f,3,'All partner-environment carriage tests')
    pos=a.get_position();a.set_position([126/180,pos.y0,51/180,pos.height])
    from matplotlib.colors import LinearSegmentedColormap
    a.imshow(pivot.values,vmin=0,vmax=1,cmap=LinearSegmentedColormap.from_list('carriage',['#F4F5F5',b.TEAL]),aspect='auto')
    environments={'random partners, benign':'U\nbenign','random partners, lethal':'U\nlethal','closed benign world':'Confined\nbenign\nsoup','lethal world':'Lethal\nsoup','open phase (t5000)':'Open\nsoup\n5k','random registers':'U\nrandom\nreset'}
    labels=[environments.get(s,s.replace(' partners,','').replace(' ','\n',1)) for s in pivot.columns]
    lengths=d.groupby('parent').tape.first().str.split().map(len)
    a.set_xticks(range(len(pivot.columns)),labels,fontsize=5);a.set_yticks(range(len(pivot)),[s.replace('lethal-tar closer 4009','Lethal chemistry T 4009').replace('block-copy R','Partial block-copy').replace('block-copy T','Full block-copy')+f' L{lengths[s]}' for s in pivot.index],fontsize=5.1)
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):a.text(j,i,f'{pivot.iloc[i,j]:.2f}',ha='center',va='center',fontsize=5.3,color='white' if pivot.iloc[i,j]>.6 else b.INK)
    b.tidy(a,'Partner source / zero-opcode chemistry / reset','')
    foot(f,3,'Transfer-8 carriers / all mutant starts; U = uniform-byte partners\nAll 6 parents L16: 16×255×16 = 65,280 starts/design; WT 256\nBenign zero executes; lethal zero halts; open soup at 5,000 updates')
    a=panel(f,4,'Identifiable alternatives and controls',False)
    for x,label in [(0,'Design'),(35,'Identifiable / tested\nalternatives'),(70,'WT starts')]:
        a.text(x,48,label,fontsize=5.4,fontweight='bold',va='center')
    for y,heading,entry in family_rows(ss):
        if heading:a.text(0,y,heading,fontsize=5.2,fontweight='bold',va='center');continue
        i,r=entry
        for x,value in [(1,design(r.key)),(35,f'{r.n_identifiable}/{r.n_mutants}'),(70,256)]:a.text(x,y,str(value),fontsize=5.1,va='center')
    foot(f,4,'Identifiable: marker window absent from WT; T = evolved transmitter\nAlternative types = L byte positions × tested values\n16 starts/value; pusher50/64 & jump50: 32 values, others 255')
    a=panel(f,5,'Ordered ancestry: two full encounters')
    d=frozen('erasure_ordered')
    for e,off in [(1,0),(2,128)]:
        q=d.query('encounter==@e');a.step(np.r_[off,q.step+off],np.r_[1 if e==1 else 0,q.marker_sources_remaining],where='post',color=b.RED,lw=1)
    a.axvline(128,color=b.RULE,ls='--',lw=.6);a.set(xlim=(-4,260),ylim=(-.04,1.1));b.tidy(a,'Execution steps across two encounters','Bytes with marker ancestry',[0,128,256],[0,1])
    a.text(10,.8,'Block-copy L32;\nbyte 8 marker = 0xA7',fontsize=6.5)
    a.text(26,.43,'1st encounter',fontsize=5.5);a.text(150,.43,'2nd encounter',fontsize=5.5)
    a.annotate('Step 6: last\nmarker ancestor lost',xy=(6,.02),xytext=(20,.22),fontsize=5.2,color=b.RED,arrowprops=dict(arrowstyle='-',color=b.RED,lw=.5))
    foot(f,5,'Counted unit: bytes with byte-8 ancestry; all-zero byte partners\nStep 6 reads address 4 (ancestry byte 0), writes 0x04 over 8\nAt execution step 128: offspring becomes parent; processor reset')
    save(f,2)


def ed3():
    f=page(3);d=frozen('capacity_edits');a=panel(f,0,'Registered: 101 settings (controls included)')
    a.plot([0,60],[0,60],color=b.RULE,lw=.6);a.scatter(d.predicted_free,d.measured_free,s=12,facecolors='none',edgecolors=b.TEAL,lw=.7);a.set(xlim=(-3,63),ylim=(-3,63));b.tidy(a,'Predicted number of free positions','Measured number of free positions',[0,20,40,60],[0,20,40,60])
    a.text(3,53,'Measured = predicted: 101/101 settings',fontsize=5.5)
    counts=[f'{int(r)}:{len(q)}' for r,q in d.groupby('predicted_free')]

    foot(f,0,'Free-position count: 0, 1, 4, 6, 12, 16, 28, 60\nSettings, respectively: 32,14,10,14,10,14,6,1\n'+FREE_DEFINITION+'\nDisplacement experiment C1; fixed core; 22 parents /101 settings\nIncludes 22 original controls; markers coincide; diagonal = equality')
    d=frozen('ed_joint_challenges').query("system=='Z80 C1'");a=panel(f,1,'Direct challenges: all passage directions',False)
    rows=[(f'{di} / {ki}',f'{int(q.cases.sum()):,}',f'{int(q.passed.sum()):,}') for (di,ki),q in d.groupby(['direction','kind'])]
    rows.append(('Total',f'{int(d.cases.sum()):,}',f'{int(d.passed.sum()):,}'))
    tab(a,['Edit / challenge','Intervention\ncases','Passed'],rows,[41,22,19])
    foot(f,1,'Z80 C1: 101 settings, 22 parents; actual transfer-8 descendants\nReciprocal = restore displacement; single = 1 site\nJoint = payload jointly varied; erased = lost-coordinate control\nPass: exact offspring, ancestry and fixed core\nIntervention counts; each tested against 17 partners')
    d=frozen('lineage_strips');serial=read('results/capacity/serial.csv.gz');a=panel(f,2,'Z80 C1, L32: lineage 0, transfer 8',False)
    for key,y in [('ldir32_v04',40),('ldir32_v60',17)]:
        q=d.query('edit_id==@key');r=q.iloc[0];b.text(a,0,y+6,f'Parent byte 0 = 0x{int(r.edited_byte):02X}; copy offset {int(r.displacement)} bytes',size=6.5,bold=True)
        for r in q.itertuples():a.add_patch(Rectangle((r.position*2.5,y),2.5,3,fc={'core':b.GREY,'erased':b.RED,'transmitted':b.TEAL}[r.state],ec='white',lw=.2))
        for pos in [0,4,16,31]:b.text(a,(pos+.5)*2.5,y-3,str(pos),size=5.5,ha='center')
        sample=serial.query('edit_id==@key and lineage==0 and generation==8 and direction=="forward"');assert len(sample)==1 and sample.exact.all() and sample.provenance_exact.all()
        haplotype=sample.offspring.iloc[0].split();assert len(haplotype)==32
        for j in range(2):a.text(0,y-7-j*3.5,f'{j*16:02d}–{j*16+15:02d}: '+' '.join(haplotype[j*16:(j+1)*16]).lower(),fontsize=6.5,fontfamily='DejaVu Sans Mono',va='center')
    b.key(a,0,-1,'Fixed core',b.GREY)
    b.key(a,23,-1,'Ancestry lost',b.RED)
    b.key(a,52,-1,'Ancestry retained',b.TEAL)
    foot(f,2,'Byte positions 0–31; each example is one wrapped descendant\nAncestry key refers to original-parent byte sources')
    d=frozen('dial_coordinates');a=panel(f,3,'Exploratory: all 256 first-byte values')
    color=np.array([0 if not r.culture_replicator else 2 if r.prop8_eligible else 1 for r in d.itertuples()]).reshape(16,16)
    from matplotlib.colors import ListedColormap
    a.imshow(color,cmap=ListedColormap(['#CFD4D7','#FFFFFF',b.TEAL]),vmin=0,vmax=2)
    for x in np.arange(-.5,16,.999999):a.axvline(x,color='#D0D5D8',lw=.15)
    for y in np.arange(-.5,16,.999999):a.axhline(y,color='#D0D5D8',lw=.15)
    a.set_xticks([0,4,8,12,15],list('048CF'));a.set_yticks([0,4,8,12,15],list('048CF'));b.tidy(a,'Low hex digit','High hex digit')
    a.set_title('Gain = mean (after − before)/(1 − before); 16 uniform partners\nMatch to original parent: partner before / offspring after; gen. 2',fontsize=6.5,pad=3)
    for yy,col,label in [(82,'#CFD4D7','157: culture-negative (gain <0.3)'),(79,'white','88: culture-positive (gain ≥0.3); prefix/offset/phase screen not met'),(76,b.TEAL,'11: culture-positive (gain ≥0.3); prefix/offset/phase screen met')]:
        f.patches.append(Rectangle((95/180,(yy-.7)/220),1.5/180,1.4/220,transform=f.transFigure,fc=col,ec=b.GREY,lw=.4))
        f.text(98/180,yy/220,label,fontsize=5.5,va='center',color=b.INK)
    d=frozen('z80_payload');a=panel(f,4,'Exploratory: sampled-score exceptions',False)
    tab(a,['Measurement','Cells'],[['Motif-bearing cells',len(d)],['Cells with dividing offset',int(d.eligible_dividing.sum())],['Cells with extra fetched-site scores',int(d.query('eligible_dividing').fetched_extra.gt(0).sum())],['Cells failing whole-tape match gate',int(d.query('eligible_dividing').residual.ne(0).sum())]],[62,20])
    note(a,20,'53 and 1 are disjoint cell subsets of the 2,343 cells')
    note(a,15,'Extra = fetched-site retention beyond unexecuted sites')
    note(a,10,'Gate deficit: mutant alleles retained at 4 byte positions')
    note(a,5,'Offspring <75% immediate-parent byte match at best cyclic shift')
    foot(f,4,f'Archived Z80 cohorts G/I/K/L; 5e ed b0 motif (family as c)\n2,347 cells from {d.world.nunique()} worlds; 4 nondividing excluded')
    d=read('results/capacity/candidate_eligibility.csv');a=panel(f,5,'Frozen candidate eligibility',False)
    q=d.groupby(['eligible','reason'],dropna=False).size().reset_index(name='n')
    tab(a,['Eligibility / reason','Candidates'],[(('Eligible' if r.eligible else str(r.reason).replace('_',' ')),r.n) for r in q.itertuples()],[65,17],size=5.3)
    note(a,31,'Same motif-cell candidates as e: d divides L, 4≤d≤L')
    note(a,26,'d = copy offset; L = tape length in bytes')
    note(a,18,'Displacement experiment C1: 0/25,856 transfer-record mismatches')
    foot(f,5,'101 registered Z80 settings ×16 lineages ×8 transfers ×2 directions\nExact bytes, fixed core and ancestry: each 0/25,856 mismatches')
    save(f,3)


def ed4():
    f=page(4);d=read('results/r4/r4_rep.csv');a=panel(f,0,'Random-register reset: Z80 L16, 20 worlds')
    for seed,q in d.groupby('seed'):a.plot(q.step/1e6,q.confined_given_heritable,color=b.TEAL,lw=.8,alpha=.8)
    q=d.groupby('step').confined_given_heritable.median();a.plot(q.index/1e6,q,color=b.INK,lw=1);a.set(xlim=(0,3.05));frac(a,'World updates (millions)','Confined / heritable')
    a.text(.7,.1,'18 zero trajectories overlap median',fontsize=5.5)
    foot(f,0,'At 3 million updates: 2/20 worlds majority confined\nThin = worlds; black median; cells confined / heritable cells\nConfined: no partner instruction-stream fetch in 16/16 traces')
    d=read('results/population/population_worlds.csv');a=panel(f,1,'Random cells versus the modal tape')
    a.scatter(d.modal_share,d.identical_to_modal_given_heritable,s=9,color=b.GREY,alpha=.7);a.set(xlim=(-.03,1.05));frac(a,'Cells identical to modal / all cells','Modal-identical / heritable cells')
    foot(f,1,f'One dot = final world sample; {len(d)} worlds in cohorts of c\nY: sampled heritable cells identical to modal / sampled heritable cells')
    d=read('results/population/classes_cells.csv');a=panel(f,2,'Class denominators by cohort',False)
    rows=[]
    for (stage,L),q in d.groupby(['stage','L']):rows.append([f'{stage}, L{L}',q.seed.nunique(),len(q),int(q.confined.sum()),int(q['class'].eq('transmitter').sum())])
    tab(a,['Cohort','Worlds','Cell rows','Confined\nrows','T\nrows'],rows,[28,13,14,16,11],size=5.2)
    foot(f,2,'Pooled sampled-cell rows; confined and T are subsets\nT = confined with ≥5 scored sites\nG benign Z80; I lethal Z80; K: L32 ring aligned with 16-bit address wrap\nL: closure-recovery extensions to 10 million updates\nM 8080 subset; L in labels = tape bytes (not cohort L)')
    d=read('results/marker_check/marker_check.csv');a=panel(f,3,'Sequence marker is not confinement')
    a.scatter(d.confined_given_no_marker,d.confined_given_marker,s=9,color=b.RED);a.plot([0,1],[0,1],color=b.RULE,lw=.6);a.set(xlim=(-.03,1.05));frac(a,'Confined / no marker','Confined / marker')
    a.annotate('Seed 1, update 0: 5/5 marker cells confined',xy=(0,1),xytext=(.05,.84),fontsize=5.1,arrowprops=dict(arrowstyle='-',color=b.GREY,lw=.5))
    a.text(.05,.74,'No marker in same sample: 0/507',fontsize=5.5)
    foot(f,3,f'Z80 L50 matched-pair invasion: {d.seed.nunique()} worlds, {len(d)} samples\nSequence marker: 20 f0 (hex bytes); positive/negative cells\n512 cells/sample; equal-fraction diagonal')
    d=read('results/ladder/population_groups.csv');q=d.query("threshold=='any'");a=panel(f,4,'Exploratory detection order: any detection',False)
    tab(a,['Cohort','World\ncount','R\nfirst','Same\nupdate','T\nfirst','R\nonly','T\nonly','None'],[[f'{r.stage}, L{r.L}',r.n,r['R before T'],r['same sample'],r['T before R'],r['R only'],r['T only'],r['neither']] for _,r in q.iterrows()],[18,8,9,9,9,10,10,9],size=5.2)
    foot(f,4,'R = confined ≤2 scored sites; T = confined ≥5 scored sites\nR<T: R observed first; Same: same sampled update; T<R: T first\nAny = positive estimated class share; n = worlds; not ancestry')
    a=panel(f,5,'Detection-order threshold sensitivity',False)
    rows=[]
    for _,r in d.query("threshold!='any'").iterrows():rows.append([f'{r.stage}, L{r.L} '+('>50%' if r.threshold=='majority' else '>10%'),r['R before T'],r['same sample'],r['T before R'],r['both_seen_n'],r.n])
    tab(a,['Threshold','R\nfirst','Same\nupdate','T\nfirst','Both\nseen','World\ncount'],rows,[32,10,10,10,10,10])
    foot(f,5,'R = confined ≤2 scored sites; T = confined ≥5 scored sites\nThreshold: R or T cells / heritable cells\nBoth detected = sum of the three order columns; n = worlds\nSite: ≥half tested byte values heritable, allele in ≥50% qualifying copies')
    save(f,4)


def ed5():
    f=page(5);d=frozen('competition_trajectories');starts={'mix50':('-',b.GREY),'R_into_T':('--',b.RED),'T_into_R':(':',b.TEAL)}
    for i,(cohort,tar) in enumerate([('Historical','benign'),('C1','benign'),('Historical','lethal'),('C1','lethal')]):
        a=panel(f,i,f'{"Corrected core" if cohort=="C1" else "Historical core C0"} / {tar}: L32, mutation on')
        q=d.query('cohort==@cohort and tar==@tar')
        for (start,seed),g in q.groupby(['start','seed']):a.plot(g.step/1000,g.T_share,color=starts[start][1],ls=starts[start][0],lw=.4,alpha=.3)
        for start,g in q.groupby('start'):
            m=g.groupby('step').T_share.median();ls,col=starts[start];a.plot(m.index/1000,m,color=col,ls=ls,lw=1.2)
        a.set(xlim=(0,305));frac(a,'World updates (thousands)','Full-tape T / (R + T)')
        handles=[b.Line2D([],[],color=col,ls=ls,lw=1.2,label=label) for (ls,col),label in zip(starts.values(),['Start: 50:50 (10)','Start: R = 1% (5)','Start: T = 1% (5)'])]
        a.legend(handles=handles,fontsize=5.1,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False,handlelength=2,columnspacing=.8)
        foot(f,i,'Parentheses count worlds. Thin: worlds; thick: medians\nR partial-copy / T full-tape\nBenign chemistry: zero executes; lethal chemistry: zero halts')
    archive=read('results/offset/offset_switch.csv');c1=read('results/tier2/scored/E/offset_switch.csv')
    assert len(archive)==len(c1)==110
    b.title(f,'e','Complete endpoints: 110 worlds/core; C1 Corrected, C0 Historical',2,144)
    a=b.axes(f,3,85,174,51,True);a._ed_panel=4;rows=[];previous_L=None
    for (L,tar,mut,start),q in archive.groupby(['L','tar','mut','start']):
        r=c1.query('L==@L and tar==@tar and mut==@mut and start==@start')
        if L!=previous_L:rows.append(['__GROUP__',f'L = {L} bytes']);previous_L=L
        star='*' if (L,tar,mut,start)==(32,'benign','on','T_into_R') else ''
        rows.append([L,tar,mut,{'mix50':'50:50','R_into_T':'R at 1%','T_into_R':'T at 1%'}[start],f'{len(r)}/{len(q)}',f'{r.T_share_end.median():.2f}'+star,f'{q.T_share_end.median():.2f}',f'{r.core_share_end.median():.2f}',f'{q.core_share_end.median():.2f}',f'{r.T_distinct_tails_end.median():.0f}',f'{q.T_distinct_tails_end.median():.0f}'])
    tab(a,['L\nbytes','Zero rule','Mutation','Start','Worlds\nCorr/H','C1','C0','C1','C0','C1','C0'],rows,[11,18,13,21,14,15,15,15,15,17,18],size=5.4)
    for x,label in [(92,'T share'),(122,'Core share'),(154.5,'Tail types')]:
        b.text(a,x,52,label,size=6.5,bold=True,ha='center')
    foot(f,4,'World medians; Corr = corrected core (C1); H = historical core C0. Benign zero executes; lethal zero halts.\nMutation off: 100,000; on: 300,000 updates. T share = T/(R+T); core share = 5e ed b0-bearing cells /20,000.\nTail types = distinct byte 4–(L−1) sequences/world. *L32 benign T-at-1%: T median n = 4/5 defined ratios;\ncore/tail medians n = 5/5, including zeros.',width=174)
    d=read('results/ladder/n2_groups.csv');a=b.axes(f,3,15,174,51,True);a._ed_panel=6
    b.title(f,'f','EXPLORATORY continuation and host controls; registered stability verdict',2,74)
    tab(a,['Source soup','Execution zero rule','Worlds','Mean T fraction, start','Mean T fraction, end'],[(r.composition,r.rule,r.worlds,f'{r.mean_T_share_start:.3f}',f'{r.mean_T_share_end:.3f}') for r in d.itertuples()],[34,35,20,42,43],size=6)
    note(a,25,'Continuation: historical core C0 Z80 L32; end = 2,000 updates; T fraction = T/(R+T).')
    toxin=read('results/toxin/causal.csv');read('results/n2/demography.csv')
    host=toxin.query('tar=="lethal" and variant in ["native","detox"]')
    native=host.query('variant=="native"').iloc[0];detox=host.query('variant=="detox"').iloc[0]
    note(a,19,'Host damage control: historical core C0 Z80 L32, lethal chemistry')
    note(a,14,f'Fraction leaving all host bytes unchanged: unmodified {native.intact:.4f}; detox {detox.intact:.4f}.\n25,600 encounters/condition (400 hosts ×64 executors).')
    note(a,10,'Detox replaces payload 0x00/0x76 with non-toxic bytes; host B faces executors from its own soup.')
    vals=[]
    for L in [16,32]:
        q=c1.query('L==@L and tar=="benign" and mut=="off" and start=="mix50"')
        vals.append(f'L{L}: {int(q.T_share_end.between(.3,.7).sum())}/10')
    note(a,4,'Registered E1 not met: '+', '.join(vals)+'; requires ≥7/10 worlds with final T/(R+T) in [0.3,0.7].')
    a.texts[-1].set_color(b.RED);a.texts[-1].set_fontweight('bold')
    foot(f,6,'E1: corrected Z80 core (C1), benign chemistry, mutation off, 50:50 starts; L16/L32.\nVariable endpoints do not establish neutral drift.',width=174)
    save(f,5)


def ed6():
    f=page(6);d=read('results/census6502/table_main.csv');a=panel(f,0,'6502 finite census: all completed arms',False)
    rows=[]
    for (L,brk,budget),q in d.groupby(['L','BRK','budget']):rows.append([f'{L}/{brk[:4]}/{budget}',','.join(map(str,q.k)),f'{int(q.done.sum()):,}',int(q['exact, zero partner'].sum())])
    tab(a,['L/BRK/budget','Periods k','Words tested','Exact\nself-writers'],rows,[30,13,25,14],size=5.1)
    foot(f,0,'L = tape bytes; budget = execution steps; k = tiled-word period\nfait = faithful BRK; halt = halt before BRK; exact output into zero B\nWords tested sums k; each L16 k4 arm exhausts 256 to the fourth power words')
    d=read('results/census8086/table_main.csv');a=panel(f,1,'8086 finite census: declared reset',False)
    tab(a,['L/budget/k','Words\ntested','Exact output\nzero B','≥75% match\nin ≥8/16'],[(f'{r.L}/{r.budget}/{r.k}',f'{r.words:,}',r.zero_exact_nonzero,r.random75_ge8_nonzero) for r in d.itertuples()],[24,19,18,21],size=5.2)
    foot(f,1,'L bytes; budget execution steps; k = word period; SP at ring end\nCounts of nonzero words: exact output into all-zero byte partner B\nor ≥75% parent-byte match in ≥8/16 uniform-partner encounters')
    d=frozen('law_6502_controls');a=panel(f,2,'Constructed 6502: sampled carriage')
    a.plot([0,28],[0,28],color=b.RULE,lw=.6);a.scatter(d.iloc[:,0],d.iloc[:,1],s=22,marker='s',facecolors='none',edgecolors=b.INK);a.set(xlim=(-1,29),ylim=(-1,29));b.tidy(a,'d - 6 (byte positions)','Carried payload byte positions',[0,14,28],[0,14,28])
    a.text(1,26,'Equality y = x; five coordinate groups',fontsize=5.5)
    foot(f,2,f'{int(d.cases.sum())} constructed copier/contract configurations; points coincide\nd − 6 / configurations: 2/96; 4/40; 10/96; 14/32; 26/32\nd = copy displacement in bytes\nCarried: same-position parent byte in all 3 bodies×64 partner tests')
    d=read('results/s6502/s1_worlds.csv');a=panel(f,3,'S1: complete four-arm route ledger',False)
    tab(a,['Arm L/budget','Worlds','Conf.','Open','Undet.','Loop\n(overlaps)'],[(f'{arm} {q.L.iloc[0]}/{q.budget.iloc[0]}',len(q),int(q.first_confined.sum()),int(q.first_open.sum()),int(q.t_rep.fillna(-1).lt(0).sum()),int(q.first_loop.sum())) for arm,q in d.groupby('arm')],[26,11,11,11,11,12],size=5.2)
    foot(f,3,'S1: 6502; Conf. = no partner fetch in 16/16 traces; Open = fetch ≥8/16\nUndet. = no heritable copier detected; Loop = repeated instruction address\nLoop overlaps routes; L bytes/budget steps; A–C faithful, D pre-BRK')
    d=read('results/evolved6502/eligibility.csv');a=panel(f,4,'Registered C2: complete sampling roles',False)
    labels={'final_random':'Final random cell','final_top3':'Final top-3 by abundance','first':'First detected tape'}
    tab(a,['Sampling role','Sampled','Eligible','Elig. worlds'],[(labels[role],len(q),int(q.eligible.sum()),q.query('eligible==1').world.nunique()) for role,q in d.groupby('role')],[33,17,16,16])
    q=d.query('eligible==1');note(a,26,f'Eligible: {len(q)} roles; {q.tape.nunique()} unique tapes; {q.world.nunique()} worlds')
    a.texts[-1].set_fontweight('bold')
    note(a,20,'6502 retention experiment C2: 80 S1 worlds; fixed-core certificate')
    note(a,14,'Pointer randomization: unmeasured; not an 8086 S2 arm')
    foot(f,4,'Eligible = certified fixed-core, pure forward-copy family\n942 roles; world counts overlap across roles (do not sum)')
    d=frozen('evolved6502_capacity');a=panel(f,5,'C2 exact capacities: all eligible roles')
    a.plot([0,10],[0,10],color=b.RULE,lw=.6);a.scatter(d.predicted_free,d.measured_free,s=13,marker='^',color='#C98E2F');a.set(xlim=(-1,11),ylim=(-1,11));b.tidy(a,'Predicted free byte positions','Measured free byte positions',[0,5,10],[0,5,10])
    a.text(.4,9.7,'Equality y = x; 42 role observations coincide',fontsize=5.3)
    foot(f,5,FREE_DEFINITION+'\nEach exact-byte/core/ancestry check: 0/91,428 direct-case mismatches\nand 0/731,424 serial-record mismatches; 42 roles /41 tapes /11 worlds')
    for p in ['results/literal_channel/direct_search.csv','results/literal_channel/finite_search.csv','results/literal_channel/CHECK.md','results/s6502/CORRECTION.md','results/census8086/reset_correction_counts.csv']:
        path=EXP/p;SOURCES[6][p]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(f,6)


def ed7():
    f=page(7);d=read('results/bff/runs.csv');extra=read('results/bff_stdnh/runs.csv');d=pd.concat([d,extra],ignore_index=True)
    a=panel(f,0,'BFF contracts: detection and persistence',False)
    variants={'std':'Standard','stdlit':'Literal, no wrap','stdnh':'Standard, harmless','wrap':'Pointer wrap','wraplit':'Wrap + literal','wraplitnh':'Wrap + literal, harmless','lit':'Literal, no wrap'}
    tab(a,['Variant','Worlds\nN','Modal\nworlds, n','Population\nworlds, n','Collapse\nn / pop. n'],[(variants[v],len(q),int(q.t_top.notna().sum()),int(q.t_her.notna().sum()),f'{int(q.collapsed.fillna(False).sum())}/{int(q.t_her.notna().sum())}') for v,q in d.groupby('variant')],[29,11,13,14,15],step=5)
    note(a,11,'Harmless: unmatched brackets do not halt; literal emits operands')
    note(a,5,'BFF = Brainfuck-family machine; wrap permits pointer wrap')
    foot(f,0,'Modal: generation-2 parent-byte-match gain ≥0.3 (headroom-normalized)\nPopulation: ≥50% cells heritable (same gain ≥0.3 criterion)\nLater collapse: terminal heritable-cell fraction <10% after population detection')
    h=read('results/biology/individuality/per_replicator.csv');q=h.query("machine=='bff' and which=='first'");a=panel(f,1,'First-modal tapes: fetch / inflow counts',False)
    palette={'std':b.GREY,'lit':b.INK,'wrap':'#C98E2F','wraplit':b.RED,'wraplitnh':b.TEAL}
    for x0,groups in [(0,['std','lit','wraplitnh']),(43,['wrap','wraplit'])]:
        for dx,label in [(0,'Fetch/256'),(17,'H (bits)'),(31,'Tapes, n')]:a.text(x0+dx,48,label,fontsize=5.2,fontweight='bold',va='center')
        y=44
        for group in groups:
            g=q.query('group==@group')
            y-=1.8
            label=wrap(f'{variants[group]} (n={len(g)})',39,6.5,True)
            a.text(x0,y,label,fontsize=6.5,fontweight='bold',va='top',color=palette[group]);y-=3.2*(label.count('\n')+1)+1.4
            counts=g.groupby(['entered_frac','H_bits']).size().reset_index(name='n')
            for r in counts.itertuples():
                for dx,value in [(0,f'{r.entered_frac:.5f}'),(17,f'{r.H_bits:.5f}'),(31,str(r.n))]:a.text(x0+dx,y,value,fontsize=5,va='center')
                y-=3.1
    foot(f,1,f'{len(q)} first-modal tapes, one/world; n counts each fetch/H pair\nFetch: encounters with any partner fetch /256; values rounded to 5 decimals\nH = H(offspring | fixed parent), bits; 256 uniform partners\nFive variants shown; Standard, harmless has no entropy archive')
    d2=read('results/bff_dials/d2_bandwidth.csv');a=panel(f,2,'Open fetch; zero inflow for r ≥ 2',False)
    tab(a,['Operand copies r','Worlds','H bits','Fetch/256'],[(r,len(q),f'{q.H_bits.median():.1f}',f'{q.entered_frac.median():.2f}') for r,q in d2.groupby('lit_rep')],[30,17,17,18])
    note(a,29,'Literal, no wrap; r = 1 baseline; r counts operand copies')
    note(a,23,'Literal P a b emits a,b repeated r times')
    foot(f,2,'World medians: conditional H and any-partner-fetch encounters/256\nr≥2: zero partner inflow with fetch in every encounter',b.TEAL)
    d1=read('results/bff_dials/d1_lethality.csv');a=panel(f,3,'Halting dial: all worlds included in counts',False)
    tab(a,['Halt p','Worlds\nN','P persistence\nworlds, n/N','Median final P share\nall worlds'],[(f'{p:g}',len(q),f'{int(q.persists.sum())}/{len(q)}',f'{q.share_last.median():.3f}') for p,q in d1.groupby('halt_p')],[12,13,24,33])
    note(a,10,'Background: Wrap + literal; p = 0 harmless, p = 1 halting')
    foot(f,3,'p = probability an unmatched bracket halts; P = literal-copy instruction\nPersistence: P-tape share ≥10% at 16,384 BFF population updates (epochs)')
    q=h.query("machine=='bff'");a=panel(f,4,'First / final inflow and fetch',False)
    rows=[]
    for g,r in q.groupby('group'):
        first=r.query("which=='first'");final=r.query("which=='final'")
        rows.append([variants[g],len(first),f'{first.H_bits.median():.2f} / {final.H_bits.median():.2f}',f'{first.entered_frac.median():.2f} / {final.entered_frac.median():.2f}'])
    tab(a,['Variant','Worlds\nn','H bits\nfirst / final','Fetch/256\nfirst / final'],rows,[29,10,22,21],size=5.2)
    note(a,15,'First = first-modal tape; final = terminal modal tape')
    foot(f,4,'Five variants shown; Standard, harmless has no entropy archive\nPaired world medians; fetch = any-partner-fetch encounters /256')
    q=read('results/biology/individuality/predictions.csv');a=panel(f,5,'Registered tests of partner inflow (H)',False)
    names={'Q1':'Z80 first H>1','Q2':'Z80 loop final H<0.5\n67 worlds: loop final tapes','Q3':'Z80 H_first − H_final ≥2','Q4a':'BFF standard first H<0.5','Q4b':'BFF pooled literal variants:\nfirst H>1'}
    tab(a,['Per-world criterion\nH in bits','Passing\nworlds, n/N','Support / reject\n% passing','Verdict'],[(r.prediction+' '+names[r.prediction],f'{r["count"]}/{r.n}',f'≥{r.threshold_met:.0%} / ≤{r.threshold_kill:.0%}',r.outcome.replace('between (neither met nor killed)','between').replace('KILL','rejected')) for _,r in q.iterrows()],[36,15,19,12],size=6,step=7.2)
    note(a,5,'Q4b: Literal, no wrap; Wrap + literal; Wrap + literal, harmless')
    note(a,0,'12 worlds each; 36 total; same names as a, b and e')
    foot(f,5,'Q1–3: Z80 Stage G (80 worlds); Q2: 67 worlds with loop-bearing final tapes\nQ4a: Standard BFF; Q4b: the three pooled variants named above\nSupport/reject use pass/world fraction; between = neither threshold')
    path=EXP/'results/bff_dials/FINDINGS.md';SOURCES[7][str(path.relative_to(EXP))]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(f,7)


def ed8():
    f=page(8);d=read('results/ladder/lod_groups.csv');a=panel(f,0,'Genealogy cohorts: worlds and nested lines',False)
    tab(a,['Cohort','Worlds','Confined\nworlds','Lines from\nconfined\nworlds','Open-first\nlines'],[(r.cohort,r.worlds,r.closed_worlds,r.sampled_lines,r.open_first_lines) for r in d.itertuples()],[23,12,13,18,16],size=5.1)
    note(a,31,'64 sampled-cell ancestries / confined world; world = simulation')
    note(a,26,'N3-v6: discovery; N6: confirmation; Z80 tape length L = 16 bytes')
    note(a,21,'Copier: ≥75% tape bytes copied in ≥32/64 uniform partners')
    note(a,16,'Counted trace = fetch of any partner opcode or operand byte')
    note(a,11,'Genealogy fetch class: open ≥32/64 counted traces; confined <32/64')
    note(a,6,'Open-first: earliest copier on each sampled line is classified open')
    foot(f,0,'Sampled lines: 64 ancestries per world with a confined founder\nWorlds column counts all worlds; genealogy classes use 64 partner traces')
    a=panel(f,1,'First founders versus later conversions',False)
    ds={name:read(f'results/{family}/founders2.csv') for name,family in [('N3-v6','lod'),('N6','lod_n6')]}
    tab(a,['Cohort','First founders,\nworlds','Later founder\nevents, n','Total founder\nevents'],[(name,int(q['first'].sum()),int((~q['first'].astype(bool)).sum()),len(q)) for name,q in ds.items()],[23,20,24,15],size=5.1)
    note(a,31,'20 worlds/cohort; at most one first confined founder/world')
    note(a,26,'Founder: first confined copier after latest open copier on sampled line')
    note(a,21,'Open ancestry to confined copier; classification as in panel a')
    note(a,16,'Later events follow the first founder in that world; events can repeat')
    note(a,11,'Later events can reuse existing confined-cell byte sources')
    foot(f,1,'First founders and later confined-founder events stay separate\nWorlds, ancestral lines and founding events are different units')
    d=read('results/lod_n6/founders2.csv');q=d[d['first'].astype(bool)];a=panel(f,2,'N6 first-founder mechanisms',False)
    tab(a,['Recorded mechanism','Founders /20'],[(kind,f'{count}/20') for kind,count in q.F_kind.value_counts().items()],[61,21])
    note(a,31,'novel = rewrite, <75% byte match to either parent (18)')
    note(a,26,'damage = partial overwrite, ≥75% parent byte match (2)')
    foot(f,2,'Separate N6-1 source rule: copy/rewrite 18/20, mutation 0/20\nRecorded event categories and causal source classes differ')
    d=read('results/lod_n6/founding.csv');a=panel(f,3,'N6 founding executors: two copier scopes',False)
    tab(a,['Executors, n','Whole-tape','Executed-byte'],[['Non-copier',int(d.executor_class.eq('non-copier').sum()),int(d.executor_exec_copier.lt(.5).sum())],['Copier',int(d.executor_class.ne('non-copier').sum()),int(d.executor_exec_copier.ge(.5).sum())]],[34,24,24])
    note(a,30,'Executed-byte: 11/20 non-copiers; requires ≥12/20; not met')
    a.texts[-1].set_color(b.RED);a.texts[-1].set_fontweight('bold')
    note(a,24,'Whole-tape: 20/20 non-copiers; requires ≥16/20')
    note(a,17,'Soup-abundance null: P(all 20 whole-tape non-copiers) = 0.062')
    note(a,12,'Null draws executors by pre-founder soup composition')
    foot(f,3,'One founding executor/first confined founder: 20 N6 worlds\nCopier: ≥75% scoped bytes copied in ≥50% partner encounters\nScopes: entire tape / only executed instruction-stream bytes')
    d=read('results/lod_n6/chain_v6b.csv');first=ds['N6'].query('first==True');q=first;assert len(q)==q.seed.nunique()==20;a=panel(f,4,'N6: all first-founder chain lengths')
    a.scatter(q.steps_elapsed,q.chain_records,s=9,facecolors='none',edgecolors=b.TEAL,linewidths=.55);b.tidy(a,'Updates from chain start','Contributing chain records')
    a.set_xlabel('Updates: chain start to first founder')
    foot(f,4,f'N6: one point per first founder, {len(q)} worlds\nRecords = contributing chain events; intervening whole-tape\ndonors are non-copiers (executed-byte function kept separate)')
    a=panel(f,5,'Registered N6: complete component ledger',False)
    p=EXP/'results/lod_n6/N6_SCORE.md';SOURCES[8][str(p.relative_to(EXP))]=hashlib.sha256(p.read_bytes()).hexdigest()
    parts=read('results/lod_n6/parts_v6.csv');first_parts=parts.merge(first[['seed','F_step']],on=['seed','F_step'])
    assert len(first_parts)==99 and abs(first_parts.share.median()-.0019)<1e-8
    assert abs(ds['N6'].query('first==True').soup_noncopier.prod()-.061811822140904575)<1e-12
    rows=[['N6-1','Copy/rewrite ≥16/20;\nmutation ≤2/20 founders','Copy/rewrite 18/20\nMutation 0/20','met'],['N6-2','Whole-tape non-copier\nchains ≥15/20','20/20','met'],['N6-3','Median cell carrier share\n≥5× random-tape baseline','Observed 0.00190\nBaseline 0.00024\nRatio 7.9×','met'],['N6-4','Non-copier executors\nwhole-tape ≥16/20;\nexecuted-byte ≥12/20','Whole 20/20\nExecuted-byte 11/20\nNull P = 0.062','not met'],['N6-5′','Expected mutation founders\n<0.05 in 20-world cohort','0.0064 expected\n0/20 observed','met']]
    tab(a,['ID','Requirement','Observation','Verdict'],rows,[11,34,25,12],size=6,step=9.5)
    for text in a.texts:
        if text.get_text()=='not met':text.set_color(b.RED);text.set_fontweight('bold')
    foot(f,5,'N6-3: cells carrying a word / all soup cells; 2-byte word at executed start\nMedian across 99 word/founder rows in pre-founder snapshots\nMutation-founded first founders: 0/20; N6 rejection threshold ≥5/20\nInput: 55 point-mutation events on 20 pre-founder chains')
    for path in ['results/lod_n6/REACH.md','results/lod_n6/REACH.csv','results/lod_n6/PARTS_FIRST.md','results/lod/OPEN_FIRST.md','results/lod_n6/OPEN_FIRST.md']:
        p=EXP/path;SOURCES[8][path]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(f,8)


def ed9():
    f=page(9);d=read('results/s8086/tables/worlds.csv');a=panel(f,0,'8086 S2: registered first-copier detection')
    for i,r in enumerate(d.itertuples()):
        t=r.censor_step if r.rep_right_censored else r.t_rep
        a.plot(t/1e6,i,marker='|' if r.rep_right_censored else 'o',color=b.GREY if r.rep_right_censored else b.TEAL if r.first_fetch=='confined' else b.RED,ms=3.5,ls='none')
    a.set(xlim=(-.1,3.15),ylim=(-1,20));a.set_yticks([0,5,10,15,19],[str(x) for x in d.seed.iloc[[0,5,10,15,19]]]);b.tidy(a,'Detection / censor time (million updates)','World seed')
    foot(f,0,'S2: coherent-fetch emergence; 20 worlds; L = 16 bytes\n14/20 detected; 6/20 censored at 3 million updates\nTeal dots: confined, 0/16 traces; red dots: open, ≥8/16 traces\nCounted trace = any partner opcode/operand byte fetch\nNo first cases at 1–7/16; grey ticks = right censor')
    p=read('results/s8086/tables/persistence.csv');a=panel(f,1,'R2: descriptive open-copier persistence')
    q=p.groupby('step').event_present.sum();a.plot(q.index/1e6,q,color=b.TEAL,lw=1,marker='o',ms=2);a.set(xlim=(2.73,3.02),ylim=(-.15,1.15));b.tidy(a,'World updates (millions)','Worlds with persistent open tape',yticks=[0,1]);b.text(a,2.77,.65,'0/20 at all 11 terminal samples',size=6.5)
    foot(f,1,'Same open fetch class as a; persistence also requires memory feeding\nSame 20 S2 worlds; terminal persistence = 0/20\nAll earlier scheduled samples combined: 1/20 worlds ever positive')
    d=read('results/s8086/tables/law_denominators.csv');a=panel(f,2,'Registered evolved law: untested',False)
    tab(a,['Family test','Candidate\ntapes','Eligible\ntapes','Unresolved\ntapes'],[(r.family+(' excluding L1' if r.exclude_L1 else ''),r.unique_candidates,r.eligible,r.unresolved_extractions) for r in d.itertuples()],[30,19,15,18])
    note(a,30,'L2 support suspended: zero certified cases')
    a.texts[-1].set_color(b.RED)
    note(a,24,'L1: frozen overlap; L2: broader pure-copy certificate')
    note(a,18,'Zero eligible cases: law tests untested; capacity undetermined')
    foot(f,2,'c–e: candidate tapes from same 20 S2 worlds as a\nL1 frozen overlap / L2 broader pure-copy certificate; d = L2 ledger')
    d=read('results/s8086/tables/law_candidates.csv');q=d.query("family=='L2'");a=panel(f,3,'Registered exclusions and unresolved cases',False)
    counts=q.groupby(['status','reason'],dropna=False).size();rows=[]
    for (status,reason),n in counts.items():rows.append([str(status),('Opcode '+str(reason).split(':')[-1].upper()+(' (POP ES)' if str(reason).endswith(':07') else '')) if str(reason).startswith('unmodelled_dependency_opcode:') else str(reason).replace('_',' '),n])
    tab(a,['Status','Reason','Candidate\ntapes, n'],rows,[22,44,16],size=6)
    note(a,1,'Total: 1,337 candidate tapes; 724 unresolved')
    foot(f,3,'L2 candidate tapes: 558 culture-negative, 55 excluded\n724 unresolved (708 at POP ES); total 1,337')
    e=read('results/s8086/exploratory/eligibility.csv');p=read('results/s8086/exploratory/law_assay_summary.csv');a=panel(f,4,'Post hoc: admissions and exact-law checks',False)
    rows=[]
    for dd,q in p.groupby('d'):rows.append([dd,len(q),q.world.nunique(),f'{int(q.encounters.sum()):,}',str(q.status.unique()[0]).replace('complete-','')])
    tab(a,['d bytes /\nedit','Genotypes','Worlds','Encounters\nwith controls','Byte/core/ancestry\nmismatches'],[(f'{r[0]} / '+('erasure' if r[0]==4 else 'single/joint\npayload'),r[1],r[2],r[3],'0') for r in rows],[12,17,13,23,17],size=5.1)
    note(a,34,'d = write-minus-read displacement mod 32, in bytes\nPaired address ring: donor A 0–15, partner B 16–31')
    note(a,30,'Same 20 S2 worlds: 779 culture-positive world/tape records')
    note(a,25,'Then 600 admitted records (590 d = 16; 10 d = 4); 32 selected genotypes')
    note(a,18,'d = 4 joint free-payload tests: 0 applicable positions/genotype (n = 10)')
    note(a,13,'d = 16 erasure tests: 0 applicable positions/genotype (n = 22)')
    note(a,6,'32 genotypes /20 worlds; 2,224,640 encounters; worlds overlap by d')
    foot(f,4,'Observed offspring bytes, fixed core and ancestry match copy-law predictions\nNonempty perturbations and empty-assignment controls included in table\nNo applicable coordinates above means no corresponding perturbations\nListed encounters include empty-assignment controls; check bytes/core/ancestry')
    d=read('results/s8086/exploratory/first_family.csv');ef=read('results/s8086/exploratory/encounter_fates.csv')
    c=read('results/s8086/exploratory/first_family_copies.csv');a=panel(f,5,'Exploratory byte-pattern copies: d = 4 bytes')
    assert len(c)==640 and c.world.nunique()==10 and not c.original_partner_read.any()
    a.scatter(c.read_position,c.write_position,s=7,color=b.TEAL)
    a.set(xlim=(-1,33),ylim=(-1,33));b.tidy(a,'Physical read address','Physical write address',[0,16,31],[0,16,31])
    pos=a.get_position();a.set_position([pos.x0,38/220,pos.width,25/220])
    a.set_title('Hex byte pattern 5F A4 u v; u/v are variable bytes\n640 writes /10 d = 4 genotypes in e; repeated pairs coincide',fontsize=5.1,pad=3)
    a.text(15,7,'Write = (read + d) mod 32\nd = 4-byte copy displacement',fontsize=5.3)
    med=ef.groupby('isa').new_exact.median()
    a=b.axes(f,95,3,82,24,True)
    a.text(0,23,'Separate snapshot fates: pooled median targets /512 encounters',fontsize=5.1,fontweight='bold',va='center')
    for x,label in [(0,'Cohort'),(59,'Median targets')]:a.text(x,19,label,fontsize=5.1,fontweight='bold',va='center')
    for y,isa,label in [(15,'8086','8086 S2: 20 L16 worlds; 3 donor motifs'),(11,'Z80','Z80 Stage G: 20 L16 worlds; 1 donor motif')]:
        a.text(0,y,label,fontsize=5.1,va='center');a.text(59,y,f'{med[isa]:g}',fontsize=5.1,va='center')
    a.text(0,6,'Target: new non-donor 16-byte half matching a donor 2-byte tiling',fontsize=5.1,va='center')
    a.text(0,1,'Pools 5 snapshots/world × both soup and uniform partners × donor A/B\nG = random-code cohort; motifs = distinct tiled 2-byte donor sequences',fontsize=5.1,va='center')
    save(f,9)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',default='1,2,3,4,5,6,7,8,9');args=ap.parse_args()
    b.fs.setup('nature');b.fs.EXP_PT=6.5;plt.rcParams.update({'font.size':6,'axes.labelsize':6,'xtick.labelsize':5.5,'ytick.labelsize':5.5})
    existing=json.loads((OUT/'ed_sources.json').read_text()) if (OUT/'ed_sources.json').exists() else {}
    for n in map(int,filter(None,args.only.split(','))):globals()[f'ed{n}']()
    checks=json.loads((OUT/'figcheck.json').read_text());checks.update(b.REPORTS);(OUT/'figcheck.json').write_text(json.dumps(checks,indent=2)+'\n')
    for n,sources in SOURCES.items():existing[f'ed8_{n}']={'height_mm':HEIGHTS[n],'sources':sources}
    (OUT/'ed_sources.json').write_text(json.dumps(existing,indent=2)+'\n')
    lines=['# Extended Data v8','', 'Nine numbered displays, titles and PDF targets match the SI_v8.md Extended Data display map. No title mismatch remains. Sources below resolve from `experiments/`. All quantities use frozen numerical tables. Full ledgers and proofs accompany the displays.','', '**Unavailable measurements:** Applied-mutation/collision counts are unavailable for the Stage-L worlds. Pointer randomization was not measured and was not an 8086 S2 arm. The standard BFF harmless-bracket variant has no archived individuality-entropy panel; it appears in detection counts. These absences are stated in ED1, ED6 and ED7, respectively.','']
    for n in range(1,10):
        assert len(LEGENDS[n].split())<=250
        lines += [f'## Extended Data {n}. {TITLES[n]}','',f'[PDF](ed8_{n}.pdf) · [PNG](ed8_{n}.png)','',f'**Legend ({len(LEGENDS[n].split())} words).** '+LEGENDS[n],'','**Sources:**']
        for path in existing.get(f'ed8_{n}',{}).get('sources',{}):lines.append(f'- `{path}`')
        if n==2:
            pe=pd.read_csv(OUT/'data/parent_environment.csv');lines += ['','**Environment IDs for panel d:**']+[f'{i+1}. {s}' for i,s in enumerate(sorted(pe.environment.unique()))]
        lines.append('')
    (OUT/'ED_V8.md').write_text('\n'.join(lines))
    if any(b.REPORTS.values()):raise SystemExit('Fix ED figure layout before critique.')


if __name__=='__main__':main()

"""Rebuild Figures 1-4 from make_figures/concept/figstyle and v6.2 tables.

All pages are 180 mm wide; PDF/SVG retain vector type, PNG is PDF-rendered
at 600 dpi. Every numerical annotation reads a generated CSV.
"""
from pathlib import Path
import argparse
import io
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mc
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import pymupdf as fitz

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import figstyle as fs
import concept as cp
import make_figures as old
import figcheck

OUT = HERE / 'out/v7'
D = OUT / 'data'
INK, GREY, RULE, TEAL, RED = old.INK, old.GREY, old.RULE, old.TEAL, old.RED
PURPLE = '#7B3294'
REPORTS = {}


def table(name):
    return pd.read_csv(D / (name + '.csv'), low_memory=False)


def method(name):
    return float(table('methods').set_index('name').loc[name,'value'])


def page(height):
    f = plt.figure(figsize=(180 * fs.MM, height * fs.MM))
    f._height_mm = height
    return f


def axes(f, x, y, w, h, off=False):
    a = f.add_axes([x/180, y/f._height_mm, w/180, h/f._height_mm])
    if off:
        a.set(xlim=(0, w), ylim=(0, h))
        a.set_axis_off()
        for spine in a.spines.values():spine.set_visible(False)
    return a


def text(a, x, y, s, size=6, color=INK, bold=False, **kw):
    return a.text(x, y, s, fontsize=max(6.5,size), color=color,
                  fontweight='bold' if bold else 'normal', va='center', **kw)


def title(f, letter, s, x, y, color=INK):
    f.text(x/180, y/f._height_mm, letter, fontsize=8, fontweight='bold', va='top', gid='panel-label')
    f.text((x+6)/180, (y-.2)/f._height_mm, s, fontsize=7, fontweight='bold', color=color, va='top')


def tidy(a, xlabel='', ylabel='', xticks=None, yticks=None):
    fs.tidy(a, xlabel, ylabel)
    if xticks is not None:
        a.set_xticks(xticks)
    if yticks is not None:
        a.set_yticks(yticks)


def key(a, x, y, label, color, shape='box', size=5.5):
    if shape == 'box':
        a.add_patch(Rectangle((x, y-.7), 2.4, 1.4, fc=color, ec='none'))
    elif shape == 'line':
        a.plot([x, x+3], [y, y], color=color, lw=1)
    else:
        a.plot(x+1.2, y, marker=shape, ls='none', color=color, ms=3.2)
    text(a, x+3.5, y, label, size=size)


def save(f, name, designer=None):
    f.canvas.draw()
    issues = figcheck.check(f)
    REPORTS[name] = issues
    figcheck.print_report(issues, name)
    f.savefig(OUT / (name+'.pdf'))
    f.savefig(OUT / (name+'.svg'))
    if designer is not None:
        src, clip, rect = designer
        pdf = fitz.open(OUT / (name+'.pdf'))
        x, y, w, h = rect
        dest = fitz.Rect(x*72/25.4, (f._height_mm-y-h)*72/25.4,
                         (x+w)*72/25.4, (f._height_mm-y)*72/25.4)
        pdf[0].draw_rect(dest, color=None, fill=(1,1,1), overlay=True)
        pdf[0].show_pdf_page(dest, src, 0, clip=clip, keep_proportion=False)
        tmp = OUT / (name+'_vector.pdf')
        pdf.save(tmp, garbage=4, deflate=True)
        pdf.close()
        tmp.replace(OUT / (name+'.pdf'))
        # Replace the Matplotlib preview by the same PDF's SVG content, clipped
        # to panel a. Editable SVG type from the chart panels is preserved.
        svg = ET.parse(OUT / (name+'.svg'))
        root = svg.getroot()
        ns = '{http://www.w3.org/2000/svg}'
        for parent in root.iter():
            for child in list(parent):
                if child.attrib.get('id') == 'designer-panel':
                    parent.remove(child)
        overlay = ET.fromstring(src[0].get_svg_image(text_as_path=False))
        overlay.attrib.update(x=str(dest.x0), y=str(dest.y0), width=str(dest.width),
                              height=str(dest.height), viewBox=f'{clip.x0} {clip.y0} {clip.width} {clip.height}',
                              preserveAspectRatio='none', overflow='hidden')
        root.append(overlay)
        svg.write(OUT / (name+'.svg'), encoding='utf-8', xml_declaration=True)
    subprocess.run(['pdftoppm', '-r', '600', '-png', '-singlefile',
                    str(OUT/(name+'.pdf')), str(OUT/name)], check=True, capture_output=True)
    plt.close(f)


def fig1():
    f = page(167)
    src = fitz.open(HERE / 'out/fig1.pdf')
    clip = fitz.Rect(0, 27, 353, 246.33)
    rect = (1, 80, 124.5, clip.height/clip.width*124.5)
    a = axes(f, *rect)
    a.set_gid('designer-panel')
    pix = src[0].get_pixmap(clip=clip, dpi=180)
    a.imshow(Image.open(io.BytesIO(pix.tobytes('png'))), aspect='auto')
    a.set_axis_off()
    title(f,'a','Writing code; evolving a return',2,162)
    title(f, 'b', 'Emergence from random code', 129, 162)
    a = axes(f, 140, 96, 37, 53)
    d = table('emergence_curves')
    for length, q in d.groupby('L'):
        a.step(q.step, q.emergence, where='post', color=old.L_COL[length], lw=.9, label=f'L = {length}')
    a.set(xscale='log', xlim=(40, 350000), ylim=(-.03, 1.04))
    fs.log10_ticks(a)
    tidy(a, 'World update', 'Cumulative fraction with emergence')
    a.legend(loc='lower center', bbox_to_anchor=(.46, 1.02), ncol=3, fontsize=5,
             handlelength=1.1, columnspacing=.7)
    f.text(134/180, 88/167, 'Z80 archive; L = tape bytes; n = 10 per L', fontsize=6, color=INK)
    f.text(134/180, 84/167, 'Event: a copy that itself reproduces', fontsize=6, color=INK)
    f.text(4/180,76/167,'Two observed designs, not one lineage. A|B marks the parent-partner boundary.',fontsize=6.5,color=INK)
    f.text(4/180,72/167,'LD loads the next two bytes into BC; PUSH writes BC at the stack pointer.',fontsize=6.5,color=INK)
    f.text(4/180,68/167,'Teal arc: written output (c0 ad below); red arc: conditional return at c0.',fontsize=6.5,color=INK)
    f.text(4/180,64/167,'Teal code / grey partner. e3 swaps stack memory. Thin arrows: execution.',fontsize=6.5,color=INK)
    title(f, 'c', 'One illustrative world: sequence similarity rises', 2, 59)
    d = table('world_trajectory')
    a = axes(f, 14, 18, 67, 33)
    a.plot(d.step, d.dominant_neighbourhood, color=TEAL, lw=.9)
    a.plot(d.step, d.zero_fraction, color=GREY, lw=.9)
    a.plot(d.step, d.distinct/d.cells, color=INK, lw=.8, ls='--')
    a.set(xscale='log', xlim=(d.step.min(), d.step.max()), ylim=(-.02,1.04))
    fs.log10_ticks(a)
    tidy(a, 'World update', 'Fraction', yticks=[0,.5,1])
    # Names in a compact inline key above the data, following the old Fig. 2.
    k = axes(f, 13, 52, 78, 6, True)
    key(k, 0, 3, 'Cells ≥75% match', TEAL, 'line')
    key(k, 31, 3, 'Zero-valued / bytes', GREY, 'line')
    key(k, 65, 3, 'Unique tapes / cells', INK, 'line');k.lines[-1].set_linestyle('--')
    jump=d.dominant_neighbourhood.diff().idxmax()
    text(a,600,.43,'Modal similarity rises',size=6.5,color=TEAL)
    a.annotate('',xy=(d.loc[jump,'step'],d.loc[jump,'dominant_neighbourhood']),xytext=(2000,.35),
               arrowprops=dict(arrowstyle='-',color=TEAL,lw=.6))
    f.text(5/180,9/167,'Match to each snapshot\'s modal tape; complete-tape diversity.',fontsize=6.5,color=INK)
    f.text(5/180,5/167,'One archive seed; separate from a. World update = encounter batch.',fontsize=6.5,color=INK)
    title(f, 'd', 'Five validation comparisons', 94, 59)
    a = axes(f, 95, 15, 83, 38, True)
    d = table('validation_fractions')
    key(a, 31, 36.5, 'Agreement', GREY)
    text(a,58,36.5,'Agree / tested',size=6.5,bold=True)
    for i, r in enumerate(d.itertuples()):
        y = 31-i*6.5
        label={'Instruction vectors':'8088 state matches','Ring encounters':'Ring-memory stability','Isolated assays':'Assay metrics identical','Claim verdicts':'Whole claims: pass/fail','Claim components':'Claim tests: pass/fail'}[r.level]
        text(a,29,y,label,size=6.5,ha='right')
        a.add_patch(Rectangle((31,y-1.4), 24*r.fraction, 2.8, fc=GREY, ec='none'))
        a.add_patch(Rectangle((31+24*r.fraction,y-1.4), 24*(1-r.fraction), 2.8, fc='#C98E2F' if i==0 else RED, ec='none'))
        text(a, 58, y, f'{r.unchanged:,} / {r.total:,}', size=5.3)
    f.text(95/180,12/167,f'Ring: {method("ring_confidence")*100:g}% upper change-rate bound <{method("ring_rate_bound")*100:g}%.',fontsize=6.5,color=INK)
    f.text(95/180,8/167,'8088: vs hardware; ochre = expected deviations.',fontsize=6.5,color=INK)
    f.text(95/180,4/167,'Other rows: original Z80 vs C1; red = flagged / changed.',fontsize=6.5,color=INK)
    save(f, 'fig1', (src, clip, rect))
    src.close()


def fig2():
    f = page(150)
    title(f, 'a', 'Branching populations, generation eight', 2, 146)
    title(f, 'b', 'Serial transfer of the same four designs', 111, 146)
    # The outcome bars and numeric side columns follow _sr_partition in v5.
    k = axes(f, 36, 133, 74, 6, True)
    key(k, 0, 3, 'Lost', old.LOST_C)
    key(k, 16, 3, 'Alive, erased', old.ERASED_C)
    key(k, 45, 3, 'Alive, carried', TEAL)
    a = axes(f, 35, 109, 48, 24)
    d = table('branching_endpoints').query("cohort=='C1'").set_index('key')
    rows = [('Open first replicator', GREY, [('pusher16','Pusher, L = 16')]),
            ('Closed regenerators', RED, [('ret16','Return closer, L = 16'),('ldir32','Block-copy tiling, L = 32')]),
            ('Closed transmitter', TEAL, [('genome20_s6003','Genome 20, world 3')])]
    y=0; ticks=[]; labs=[]
    c = axes(f, 86, 109, 23, 24, True)
    text(c, 0, 24, 'WT alive', size=6.5)
    text(c, 15, 24, 'Sites g8', size=6.5)
    for group, color, rr in rows:
        a.text(-.03, y-.18, group, transform=a.get_yaxis_transform(), ha='right', va='bottom',
               fontsize=6, fontweight='bold', color=color, gid='allow-outside')
        y += .6
        for keyname, label in rr:
            r=d.loc[keyname]; left=0
            for field, col in [('lost',old.LOST_C),('erased',old.ERASED_C),('retained',TEAL)]:
                a.barh(y,r[field],left=left,height=.62,color=col,lw=0)
                left += r[field]
            ticks.append(y);labs.append(label)
            cy = 24-(y+.6)/7.6*24
            text(c,0,cy,f'{int(r.ctrl_alive_n)}/{int(r.ctrl)}',size=5.7)
            text(c,15,cy,f'{int(r.sites)}/{int(r.L)}',size=5.7,color=TEAL if r.sites else INK)
            y += 1.1
        y += .55
    a.set(xlim=(0,1),ylim=(7,-.6))
    a.set_yticks(ticks,labs,fontsize=6)
    a.tick_params(axis='y',length=0);a.spines['left'].set_visible(False)
    tidy(a,'Share of all starting mutant populations',xticks=[0,.5,1])
    f.text(4/180,99/150,'Bars: all mutant starts. Carrying means ≥ half the population retains its marker.',fontsize=6,color=INK)
    f.text(4/180,95/150,'A site needs ≥ half its tested values to qualify; rare carriers can score zero sites.',fontsize=6,color=INK)
    f.text(4/180,91/150,'WT: 64 unmutated starts. Original and corrected Z80 agree; pusher retention prediction contradicted.',fontsize=6.5,color=INK)
    # Same curves/units as _sr_curves, with four explicit design identities.
    ss=table('serial_summary').set_index('key')
    a=axes(f,124,112,53,29)
    for keyname,label,col,marker in [('pusher16','Pusher 16',GREY,'o'),('ret16','Return 16',RED,'s'),
                                    ('ldir32','Block-copy 32',RED,'^'),('genome20_s6003','Genome 20, world 3',TEAL,'o')]:
        r=ss.loc[keyname]
        a.plot([1,2,4,8],[r[f'retained_g{g}'] for g in [1,2,4,8]],color=col,marker=marker,ms=3,lw=.9,
               mfc='white' if keyname=='ret16' else col,label=label)
    a.set(xscale='log',xlim=(.85,9.5),ylim=(-.02,.54))
    a.set_xticks([1,2,4,8],['1','2','4','8']);a.minorticks_off()
    tidy(a,'Serial transfers','Marker-carrying / all mutant starts')
    a.legend(loc='center right',bbox_to_anchor=(.99,.52),fontsize=6,handlelength=1.2)
    f.text(117/180,103/150,'Marker = aligned 5-byte mutant window; all starts pooled.',fontsize=6,color=INK)
    f.text(117/180,99/150,'255 values/site, 16 lines/value; original Z80 only.',fontsize=6,color=INK)
    f.text(117/180,95/150,'Three curves coincide near zero at transfer 8.',fontsize=6,color=INK)
    title(f,'c','Serial sites: one box per byte; each track scaled to its tape',2,85)
    a=axes(f,2,41,175,40,True)
    k=axes(f,36,76,140,6,True)
    key(k,0,3,'Byte executed',old.EXEC_C)
    key(k,35,3,'Never executed','white')
    k.patches[-1].set_edgecolor(RULE)
    key(k,71,3,'First copy only',old.SITE_C,'o')
    k.lines[-1].set_markerfacecolor('white')
    key(k,102,3,'Carried at transfer 8',old.SITE_C,'o')
    ss=table('serial_summary').set_index('key'); sites=table('serial_sites')
    rows=[('pusher50','Pusher, L = 50'),('ret16','Return closer, L = 16'),
          ('ldir32','Block-copy tiling, L = 32'),('lethal16_s4009','Lethal-tar closer, L = 16'),
          ('genome20_s6004','Genome 20, world 4'),('closer32_8080_p1','Constructed 8080, L = 32')]
    cols=[121,136,152,168]
    for x,lab in zip(cols,['Fetched\nbytes','Sites at\ntransfer 1','Sites at\ntransfer 8','Transfer-8\nnot fetched']):
        text(a,x,32,lab,size=5.2,ha='center',color=INK)
    for i,(keyname,label) in enumerate(rows):
        r=ss.loc[keyname]; L=int(r.L); ex=np.array([c=='1' for c in str(r.executed).zfill(L)])
        s1=sites.query('key==@keyname and gen==1').set_index('pos').transmissible.reindex(range(L)).fillna(False).to_numpy(bool)
        s8=sites.query('key==@keyname and gen==8').set_index('pos').transmissible.reindex(range(L)).fillna(False).to_numpy(bool)
        y=26-i*4.5;start=34;w=80/L
        text(a,start-2,y,label,ha='right',size=5.5)
        for j in range(L):
            a.add_patch(Rectangle((start+j*w,y-1.45),w,2.9,fc=old.EXEC_C if ex[j] else 'white',ec=RULE,lw=.3))
            if s1[j] or s8[j]:
                a.plot(start+(j+.5)*w,y,'o',ms=3.2,mec=old.SITE_C,mfc=old.SITE_C if s8[j] else 'white',mew=.7)
        for x,v in zip(cols,[f'{ex.sum()}/{L}',s1.sum(),s8.sum(),(s8&~ex).sum()]):
            text(a,x,y,str(v),size=5.7,ha='center')
    f.text(5/180,38/150,'Grey: opcode fetched in any of 64 random-partner executions of the unmutated tape (128 instructions each).',fontsize=6.5,color=INK)
    f.text(5/180,35/150,'Serial site: ≥ half surviving lines carry above background, for ≥ half identifiable alternative values.',fontsize=6.5,color=INK)
    title(f,'d','Block-copy regenerator from a-c: one erasure lineage',2,31)
    t=table('erasure_ordered');m=json.loads((D/'erasure_manifest.json').read_text())
    # The entire ordered trace is present as write coordinates, with its causal
    # marker count below. Every real write is plotted, no interpolated events.
    a=axes(f,15,11,61,14)
    for encounter,offset in [(1,0),(2,128)]:
        q=t.query("kind=='write' and encounter==@encounter")
        a.plot(q.step+offset,q.address,'.',color=GREY,ms=1.3,ls='none')
    q=t[(t.kind=='write')&(t.encounter==1)&(t.prior_source==m['position'])]
    r=q.iloc[0];a.plot(r.step,r.address,'o',ms=4,mfc='none',mec=RED,mew=.8)
    a.axvline(m['steps_per_encounter'],color=RULE,lw=.7,ls='--')
    a.axhline(m['L']-.5,color=RULE,lw=.5,ls=':')
    a.set(xlim=(-4,260),ylim=(-2,66));tidy(a,'Instruction step','Ring write address',[0,128,256],[0,32,63])
    f.text(15/180,26.7/150,f'Parent 0-{m["L"]-1}; partner {m["L"]}-{2*m["L"]-1}; wraps modulo {2*m["L"]}.',fontsize=6.5,color=INK)
    a=axes(f,88,11,35,14)
    for encounter,offset in [(1,0),(2,m['steps_per_encounter'])]:
        q=t[t.encounter==encounter]
        a.step(np.r_[offset,q.step+offset],np.r_[1 if encounter==1 else 0,q.marker_sources_remaining],where='post',color=RED,lw=.9)
    a.axvline(128,color=RULE,lw=.7,ls='--')
    a.set(xlim=(-4,260),ylim=(-.05,1.1));tidy(a,'Instruction step','Edited-byte descendants',[0,128,256],[0,1])
    k=axes(f,128,3,49,29,True)
    text(k,0,27,f'Initial label: edited byte {m["position"]} only (0x{m["marker"]:02X}).',size=6.5)
    text(k,0,23,f'Erasing write at instruction {int(r.step)}:',color=RED,size=6.5)
    text(k,0,19,f'source {int(r.source)} (0x{int(r.byte):02X}) to destination {int(r.address)} (0x{int(r.prior_byte):02X}).',color=RED,size=6.5)
    text(k,0,15,'Circled write removes last labelled byte.',size=6.5)
    text(k,0,11,f'{m["steps_per_encounter"]}: offspring becomes parent; CPU resets.',size=6.5)
    text(k,0,7,'Ancestry of one byte; b uses a 5-byte window.',size=6.5)
    text(k,0,3,'All-zero partner; clock spans both encounters.',size=6.5)
    f.text(5/180,2/150,'Open executes partner; closed stays in parent. Regenerator erases, transmitter carries variation. L = bytes; 8080 = Z80 subset.',fontsize=6.5,color=INK)
    save(f,'fig2')


def geometry(a):
    # concept.py's cells, arrows and brackets; each displayed cell aggregates
    # two bytes, keeping the mechanism abstract (a dozen units per geometry).
    g=table('geometry_units')
    a.set(xlim=(-.2,13.3),ylim=(-.5,11))
    initial=g[(g.d==12)&(g.unit<6)]
    text(a,0,10.65,'One address track: A at 0-11, partner B at 12-23',bold=True,size=6.5)
    for r in initial.itertuples():
        cp.cell(a,r.unit,8.4,f'K{r.source//2}' if r.state=='core' else chr(97+(r.source-4)//2),
                fill=cp.GREY_FILL if r.state=='core' else cp.TEAL_FILL,w=.95,h=.7,fs=6.5)
    for i in range(6,12):cp.cell(a,i,8.4,fill=cp.GREY_FILL,w=.95,h=.7)
    cp.arrow(a,(.45,9.8),(2.45,9.8),color=RED,lw=.85)
    text(a,3,9.8,'d=4: write 0 to 4 before reading 4',size=6.5,color=RED)
    cp.arrow(a,(.45,9.35),(6.45,9.35),color=TEAL,lw=.85)
    text(a,7,9.35,'d=12: write 0 to 12',size=6.5,color=TEAL)
    cp.bracket(a,0,1.95,8.15,tick=.15)
    text(a,0,7.6,'Protected core: 4 bytes',size=6)
    text(a,8.5,7.8,'B initially holds arbitrary bytes',size=6,ha='center')
    for d,y,label,col in [(4,5.2,'d = 4 bytes: payload erased',RED),
                          (12,2.0,'d = 12 bytes: 8 payload positions retained',TEAL)]:
        text(a,0,y+1.35,label,color=col,bold=True,size=6.5)
        q=g[(g.d==d)&(g.unit>=6)]
        for r in q.itertuples():
            fill=cp.TEAL_FILL if r.state=='payload' else cp.GREY_FILL
            cp.cell(a,r.unit,y, f'K{r.source//2}' if r.state=='core' else chr(97+(r.source-4)//2),fill=fill,w=.95,h=.75,fs=6.5)
        cp.bracket(a,6,7.95,y-.25,tick=.12)
        text(a,7,y-.5,'c=4',size=6,ha='center')
        if d==12:
            cp.bracket(a,8,11.95,y-.25,tick=.12)
            text(a,10,y-.5,'r=8',size=6,ha='center')
        cp.bracket(a,6,6+d/2-.05,y-.85,tick=.12)
        text(a,9 if d==12 else 8.8,y-1.35,'Retained prefix d; offspring B',size=6,ha='center')
        if d==4:
            cp.arrow(a,(6.4,y+1),(8.4,y+1),color=col,lw=.8)
            cp.arrow(a,(8.4,y+1),(10.4,y+1),color=col,lw=.8)
            text(a,0,y+.3,'First 4 reads: original core.\nLater reads: earlier writes.',size=6)
        else:
            text(a,0,y+.3,'All 12 parental bytes are read\nbefore being overwritten.',size=6)
    text(a,0,-.2,'Each box = 2 bytes; K = core; a-d = four payload blocks',size=6)
    a.set_axis_off()


def law(a, detail=False):
    pop=table('law_population').sort_values('cells',ascending=False)
    c1=table('law_z80_c1'); c2=table('law_6502_c2');ctrl=table('law_6502_controls')
    limit=10.8 if detail else 64
    a.plot([0,limit],[0,limit],color=RULE,lw=.6,zorder=0)
    if detail:
        pop=pop[pop.predicted<=10];c1=c1[c1.predicted_free<=10];c2=c2[c2.predicted_free<=10];ctrl=ctrl[ctrl.iloc[:,0]<=10]
    a.scatter(pop.predicted,pop.n_unexec,s=pop.cells*.18,facecolors='#D3D8DC',edgecolors=GREY,lw=.3,zorder=1)
    # Concentric exact-coordinate encodings avoid moving the true measurements.
    a.scatter(ctrl.iloc[:,0],ctrl.iloc[:,1],s=26,marker='s',facecolors='none',edgecolors=INK,lw=.7,zorder=2)
    a.scatter(c1.predicted_free,c1.measured_free,s=17,marker='o',facecolors='none',edgecolors=TEAL,lw=.8,zorder=3)
    a.scatter(c2.predicted_free,c2.measured_free,s=10,marker='^',facecolors='#C98E2F',edgecolors='white',lw=.25,zorder=4)
    deficits=pop[pop.predicted!=pop.n_unexec]
    a.scatter(deficits.predicted,deficits.n_unexec,s=13,marker='o',facecolors='none',edgecolors=RED,lw=.8,zorder=5)
    a.set(xlim=(-5 if not detail else -2.5,limit),ylim=(-5 if not detail else -2.5,limit),aspect='equal')
    tidy(a, '' if detail else 'Predicted d - core length (byte positions)',
         '' if detail else 'Measured payload positions (bytes)',
         [0,5,10] if detail else [0,20,40,60], [0,5,10] if detail else [0,20,40,60])


def fig3():
    f=page(166)
    title(f,'a','Forward overlap protects a prefix',2,162)
    geometry(axes(f,4,96,83,60,True))
    f.text(5/180,88/166,'r = d - c: r free byte positions; d copy offset; c core bytes.',fontsize=6,color=INK)
    f.text(5/180,84/166,'Read starts at 0, write at d; aligned forward copy covers both tapes.',fontsize=6,color=INK)
    title(f,'b','The law: prediction and measurement',97,162)
    a=axes(f,113,95,63,63);law(a)
    # Series are named in the plot; the pending processor consumes one key row.
    pop=table('law_population');c1=table('law_z80_c1');c2=table('law_6502_c2');ctrl=table('law_6502_controls')
    handles=[Line2D([],[],marker='o',ls='none',ms=4,mfc='#D3D8DC',mec=GREY,label=f'Z80 qualifying sites: {int(pop.cells.sum()):,} cells'),
             Line2D([],[],marker='o',ls='none',ms=3,mfc='none',mec=TEAL,label=f'Z80 exact positions: {int(c1.edits.sum())} edits'),
             Line2D([],[],marker='^',ls='none',ms=3,mfc='#C98E2F',mec='#C98E2F',label=f'6502 exact positions: {int(c2.roles.sum())} copiers'),
             Line2D([],[],marker='s',ls='none',ms=3,mfc='none',mec=INK,label=f'6502 sampled carried positions: {int(ctrl.cases.sum())} controls'),
             Line2D([],[],marker='o',ls='none',ms=3,mfc='none',mec=RED,label='Whole-tape match gate fails (one cell)'),
             Line2D([],[],marker='D',ls='none',ms=3,mfc='none',mec=GREY,label='8086: pending')]
    a.legend(handles=handles,loc='upper left',fontsize=6,handlelength=1,borderpad=.15)
    text(a,45,42,'y = x',size=5.7,color=GREY)
    # A real low-capacity enlargement in the unused below-diagonal quadrant.
    inset=axes(f,150,101,23,23);law(inset,True)
    inset.set_title('Same points: 0-10',fontsize=6,pad=2)
    inset.tick_params(labelsize=6)
    a.add_patch(Rectangle((0,0),10,10,fill=False,ec=RULE,lw=.6,ls='--'))
    text(a,12,7,f'<{method("match_gate")*100:g}% tape match:\n0 sampled sites;\n4 positions carried',size=6.5,color=RED)
    a.annotate('',xy=(4,.2),xytext=(12,4),arrowprops=dict(arrowstyle='-',color=RED,lw=.5))
    k=axes(f,104,73,74,13,True)
    text(k,0,10,'Disc area scales with cells; rings = series only',size=6.5,color=INK)
    for x,n in [(36,1),(46,100),(62,2000)]:
        k.scatter(x,5,s=n*.18,facecolors='#D3D8DC',edgecolors=GREY,lw=.3)
        text(k,x,0,str(n) if n<1000 else '2,000',size=5,ha='center')
    pop=table('law_population');c1=table('law_z80_c1');c2=table('law_6502_c2');ctrl=table('law_6502_controls')
    z=table('z80_payload').query('eligible_dividing==True');ev=table('evolved6502_capacity')
    f.text(5/180,79/166,'Exact: every alternative is copied with its causal source; core unchanged.',fontsize=6.5,color=INK)
    f.text(5/180,75/166,f'Sites: half values after ≥{method("match_gate")*100:g}% tape match; {z.world.nunique()} Z80 worlds, {ev.world.nunique()} 6502 worlds.',fontsize=6.5,color=INK)
    f.text(5/180,71/166,f'Dial: before/after = partner match to original parent; gain = (after-before)/(1-before), mean at before <{method("match_gate"):g}.',fontsize=6.5,color=INK)
    title(f,'c','One byte changes the inherited copying rule',2,68)
    a=axes(f,4,6,83,56,True)
    d=table('lineage_strips')
    text(a,0,54,'Z80 ldir32: byte 0 sets offset modulo the 64-byte ring.',size=6)
    colors={'core':GREY,'erased':RED,'transmitted':TEAL}
    for keyname,y in [('ldir32_v04',41),('ldir32_v60',20)]:
        q=d[d.edit_id==keyname].sort_values('position');r=q.iloc[0]
        label=(f'Displacement {int(r.displacement)} bytes: regenerates, erases payload' if r.displacement==4 else
               f'Edit 0x{int(r.original_byte):02X} to 0x{int(r.edited_byte):02X}: {int(r.edited_byte)} mod 64 = {int(r.displacement)} bytes\ntransmits {int((q.state=="transmitted").sum())} payload positions')
        text(a,0,y+6,label,size=6,bold=True,color=RED if r.displacement==4 else TEAL)
        for row in q.itertuples():
            a.add_patch(Rectangle((row.position*2.45,y-1.6),2.45,3.2,fc=colors[row.state],ec='white',lw=.25))
        text(a,4.9,y-4.5,'Core',size=5.5,ha='center')
        text(a,44,y-4.5,'Payload positions 4-31',size=5.5,ha='center')
        for pos in [0,4,16,31]:
            text(a,(pos+.5)*2.45,y-8,str(pos),size=5,ha='center',color=GREY)
    text(a,0,3,'Example offspring, lineage 0 at generation 8; teal = parental source.',size=6.5)
    k=axes(f,5,12,82,4,True)
    key(k,0,2,'Core preserved',GREY)
    key(k,27,2,'Erased',RED)
    key(k,47,2,'Inherited payload',TEAL)
    title(f,'d','All 256 values of the displacement byte',95,68)
    dial=table('dial_coordinates');a=axes(f,105,12,48,48)
    cmap=mc.LinearSegmentedColormap.from_list('payload',['#89C4C4',TEAL]);norm=mc.Normalize(0,28)
    for r in dial.itertuples():
        fill='#F0F1F2' if not r.culture_replicator else cmap(norm(r.predicted_payload)) if r.prop8_eligible else 'white'
        edge='white' if not r.culture_replicator else TEAL
        a.add_patch(Rectangle((r.value%16-.47,r.value//16-.47),.94,.94,fc=fill,ec=edge,lw=.3))
        if r.value in [4,96]:
            a.add_patch(Rectangle((r.value%16-.47,r.value//16-.47),.94,.94,fill=False,ec=INK,lw=1))
    a.set(xlim=(-.5,15.5),ylim=(15.5,-.5),aspect='equal')
    a.set_xticks(range(16),list('0123456789ABCDEF'));a.set_yticks(range(16),list('0123456789ABCDEF'))
    a.tick_params(length=0,pad=2)
    tidy(a,'Low hexadecimal digit','High hexadecimal digit')
    for sp in a.spines.values():sp.set_visible(False)
    k=axes(f,157,12,21,48,True)
    nw=int((~dial.culture_replicator).sum());ne=int(dial.prop8_eligible.sum());other=int(dial.culture_replicator.sum())-ne
    key(k,0,45,f'{nw} nonworking','#F0F1F2',size=5.1)
    key(k,0,38,f'{other} working,\noutside conditions','white',size=5.1);k.patches[-1].set_edgecolor(TEAL)
    key(k,0,28,f'{ne} eligible;\nshade = capacity',cmap(.6),size=5.1)
    text(k,0,21,'Free payload',size=5.3,bold=True)
    for i in range(100):
        k.add_patch(Rectangle((0,5+i*.12),3,.125,fc=cmap(i/99),ec='none'))
    text(k,4,5,'0',size=5.3);text(k,4,17,'28 positions',size=5.3)
    text(k,0,1,'Rims: 0x04 / 0x60',size=5.1)
    f.text(105/180,63/166,f'Working: gen-2 normalized match gain ≥ {method("gen2_minimum"):g}.',fontsize=6.5,color=INK)
    f.text(103/180,2/166,'Prediction requires 4 ≤ d ≤ 32, d dividing 32; aligned full-ring copy.',fontsize=6.5,color=INK)
    save(f,'fig3')


STARTS={'mix50':('50:50','-','o',10), 'R_into_T':('1% R in T','--','s',5), 'T_into_R':('1% T in R',':','^',5)}


def trajectories(a,tar,measure):
    d=table('competition_medians')
    color=TEAL if tar=='benign' else PURPLE
    for cohort in ['Historical','C1']:
        for start,(label,ls,marker,n) in STARTS.items():
            q=d[(d.cohort==cohort)&(d.tar==tar)&(d.start==start)].sort_values('step')
            a.plot(q.step/1000,q[measure],color=color,ls=ls,lw=.95 if cohort=='C1' else .75,
                   alpha=1 if cohort=='C1' else .5,marker=marker,ms=2.8,
                   markevery=[0,len(q)//4,len(q)//2,3*len(q)//4,len(q)-1],
                   mfc=color if cohort=='C1' else 'white',mec=color,mew=.55)
    a.set(xlim=(0,305),ylim=(-.03,1.05))
    tidy(a,'World updates (thousands)', 'T / (R + T)' if measure=='T_share' else 'Core-bearing / all cells',[0,150,300],[0,.5,1])
    if measure=='core_share':
        ends=table('competition_endpoints')
        lines=['Final T share, core share: C1 | archive']
        for start,(label,_,_,_) in STARTS.items():
            parts=[]
            for cohort in ['C1','Historical']:
                q=ends[(ends.cohort==cohort)&(ends.tar==tar)&(ends.start==start)]
                parts.append(f'{q.T_share_end.median():.2f}, {q.core_share_end.median():.2f}')
            lines.append(label+': '+' | '.join(parts))
        a.text(.035,.61,'\n'.join(lines),transform=a.transAxes,fontsize=6.5,va='top',color=INK,linespacing=1.15)


def endpoint(a,measure):
    d=table('endpoint_positions');d=d[d.measure==measure]
    for i,(cohort,tar) in enumerate([('Historical','benign'),('C1','benign'),('Historical','lethal'),('C1','lethal')]):
        color=TEAL if tar=='benign' else PURPLE
        q=d[(d.cohort==cohort)&(d.tar==tar)]
        for start,(_,_,marker,n) in STARTS.items():
            r=q[q.start==start]
            a.scatter(i+r.offset,r[measure],s=6.25,marker=marker,
                      facecolors=color if cohort=='C1' else 'white',edgecolors=color,lw=.55,zorder=3)
    a.set_xlim(-.5,3.5)
    if measure!='T_distinct_tails_end':a.set_ylim(-.03,1.07)
    else:a.set_ylim(-20,d[measure].max()*1.08)
    a.set_xticks(range(4),['Archive','C1','Archive','C1'])
    tidy(a,'',{'T_share_end':'Final T / (R + T)','core_share_end':'Core-bearing / all cells',
               'T_distinct_tails_end':'Unique T payloads / world'}[measure],yticks=[0,.5,1] if measure!='T_distinct_tails_end' else None)


def fig4():
    f=page(165)
    title(f,'a','Matched competitions: relative share and absolute persistence',2,161)
    k=axes(f,4,150,173,8,True)
    for x,(start,(label,ls,marker,n)) in zip([0,36,81],STARTS.items()):
        k.plot([x,x+5],[5,5],color=INK,lw=.9,ls=ls)
        k.plot(x+2.5,5,marker=marker,color=INK,ms=2.8,ls='none')
        text(k,x+7,5,f'{label} ({n} worlds)',size=5.5)
    for yy,col,fill,label in [(5,GREY,'white','Archive: light/open'),(1,INK,INK,'C1: dark/filled')]:
        k.plot([131,136],[yy,yy],color=col,lw=.9)
        k.plot(133.5,yy,'o',ms=3,mfc=fill,mec=col,ls='none')
        text(k,138,yy,label,size=6.5)
    f.text(15/180,147/165,'Z80; L = 32 bytes; 20,000 cells; mutation on. C1 corrects instruction flags/refresh; archive uses older semantics.',fontsize=6.5,color=INK)
    f.text(15/180,143/165,'Regenerator R preserves a short pattern; transmitter T preserves payload. Hex 5e ed b0 is the copy core. Curves: world medians.',fontsize=6.5,color=INK)
    for x,tar,col in [(15,'benign',TEAL),(103,'lethal',PURPLE)]:
        f.text((x+30)/180,137/165,f'{tar.capitalize()} tar: zero executes as '+('no operation' if tar=='benign' else 'halt'),fontsize=6.5,color=col,fontweight='bold',ha='center')
        trajectories(axes(f,x,108,72,25),tar,'T_share')
        trajectories(axes(f,x,73,72,25),tar,'core_share')
    f.text(15/180,63.5/165,'Core includes broken/other byte-0 variants; R+T may be smaller. Undefined ratios omitted from medians.',fontsize=6.5,color=INK)
    title(f,'b','Final world distributions at 300,000 world updates',2,59)
    f.text(101/180,57.8/165,'Open: archive / filled: C1 implementation',fontsize=6,color=INK)
    for x,measure in [(15,'T_share_end'),(73,'core_share_end'),(132,'T_distinct_tails_end')]:
        endpoint(axes(f,x,27,45,25),measure)
        if measure=='core_share_end':
            a=f.axes[-1]
            text(a,1.8,.22,'Core-loss world',size=5.1,color=GREY)
            points=table('endpoint_positions')
            zero=points[(points.measure=='core_share_end')&(points.cohort=='C1')&(points.tar=='benign')&(points.core_share_end==0)]
            assert len(zero)==1
            a.annotate('',xy=(1+zero.iloc[0].offset,0),xytext=(1.8,.16),arrowprops=dict(arrowstyle='-',color=GREY,lw=.5))
        f.text((x+10)/180,20/165,'Benign',fontsize=6,color=TEAL,ha='center')
        f.text((x+34)/180,20/165,'Lethal',fontsize=6,color=PURPLE,ha='center')
    f.text(15/180,15.5/165,'20 worlds/group; ratio has 19 in C1 benign (one zero R+T). Payload = distinct byte strings at positions 4-31.',fontsize=6,color=INK)
    # Increase page depth just for the dial while retaining 170-mm limit.
    # The table above is deliberately paired with its definitions on a second
    # registered strip below; no claim is represented by a blank panel.
    title(f,'c','Lethality dial from random code: L = 16; 10 worlds / p',2,12)
    d=table('lethality_summary')
    # Compact numeric columns reproduce the figure4v5 design without hiding
    # closure timing or the remaining C1 reversal in a separate page.
    # Freed header and plot spacing allows a fully defined dial in 40 mm.
    # Shift the entire a-b block upward after authoring, expanding the page to
    # the previous standard's maximum depth while reserving the bottom strip.
    shift,scale=28,.82
    f.set_size_inches(180*fs.MM,170*fs.MM)
    for ax in f.axes:
        pos=ax.get_position()
        ax.set_position([pos.x0,(pos.y0*165*scale+shift)/170,pos.width,pos.height*165*scale/170])
    for item in f.texts:
        x,y=item.get_position();item.set_position((x,(y*165*scale+shift)/170))
    f._height_mm=170
    # The prior temporary c title would sit at 40 mm; remove it and place the
    # final c heading directly above the table.
    for item in f.texts[-2:]:item.remove()
    title(f,'c','Random-code dial: L = 16; 10 worlds per implementation per p',2,37)
    a=axes(f,8,13,169,20,True)
    text(a,0,16,'p: halt per zero instruction',size=6.5,bold=True)
    for j,p in enumerate(sorted(d.p.unique())):
        text(a,60+j*19,16,f'{p:g}',size=6,ha='center',bold=True)
    for y,field,label in [(11,'open_first','Open-first (out of 10)'),(6,'T_majority','T-majority (out of 10)'),(1,'confinement_km','KM median to first confined copier (k updates)')]:
        text(a,0,y,label,size=5.3)
        for j,p in enumerate(sorted(d.p.unique())):
            c=d[(d.cohort=='C1')&(d.p==p)].iloc[0];h=d[(d.cohort=='Historical')&(d.p==p)].iloc[0]
            text(a,60+j*19,y,f'{c[field]:g} | {h[field]:g}',size=6.5,ha='center')
    f.text(126/180,36/170,'Each entry: C1 | archive',fontsize=6.5,color=INK)
    for y in [11.5,16.5,21.5,26.5,31.5]:
        f.add_artist(Line2D([8/180,177/180],[y/170,y/170],transform=f.transFigure,color=RULE,lw=.35))
    f.text(8/180,8/170,'Open-first copier executes partner; T-majority is > half of heritable cells at 300k updates. Counts may overlap.',fontsize=6.5,color=INK)
    f.text(8/180,4.5/170,'KM event = first confined copier (no partner fetch in 16 tests); median includes nondetections, censored at 300k.',fontsize=6.5,color=INK)
    f.text(8/180,1/170,'Heritable = offspring reproduces, distinct from motif counts above. Each zero instruction halts with probability p.',fontsize=6.5,color=INK)
    save(f,'fig4')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',default='1,2,3,4');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    fs.setup('nature')
    plt.rcParams.update({'font.size':6.5,'axes.labelsize':6.5,'xtick.labelsize':6,'ytick.labelsize':6})
    previous=json.loads((OUT/'figcheck.json').read_text()) if (OUT/'figcheck.json').exists() else {}
    for n in map(int,args.only.split(',')):
        globals()[f'fig{n}']()
    previous.update(REPORTS)
    (OUT/'figcheck.json').write_text(json.dumps(previous,indent=2)+'\n')
    if any(REPORTS.values()):raise SystemExit('Fix figure layout before critique.')


if __name__=='__main__':main()

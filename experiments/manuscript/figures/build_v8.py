"""Rebuild Figures 1-5 from make_figures/concept/figstyle and v6.2 tables.

All pages are 180 mm wide; PDF/SVG retain vector type, PNG is PDF-rendered
at 600 dpi. Every numerical annotation reads a generated CSV.
"""
from pathlib import Path
import argparse
import io
import json
import base64
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

OUT = HERE / 'out/v8'
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
    # Apply this to every export, including ED heatmaps: no SVG may embed
    # raster data. Keep relative links portable with the delivered directory.
    svg = ET.parse(OUT/(name+'.svg'))
    for i, item in enumerate(svg.getroot().iter('{http://www.w3.org/2000/svg}image')):
        for attr, value in list(item.attrib.items()):
            if attr.endswith('href') and value.startswith('data:image/'):
                assert value.startswith('data:image/png;'), value[:40]
                asset = f'{name}_render_{i+1}.png'
                (OUT/asset).write_bytes(base64.b64decode(value.split(',',1)[1]))
                item.set(attr, asset)
    svg.write(OUT/(name+'.svg'),encoding='utf-8',xml_declaration=True)
    subprocess.run(['pdftoppm', '-r', '600', '-png', '-singlefile',
                    str(OUT/(name+'.pdf')), str(OUT/name)], check=True, capture_output=True)
    plt.close(f)


def reflow(f, height, ymap):
    oldheight=f._height_mm
    for a in f.axes:
        p=a.get_position()
        a.set_position([p.x0,ymap(p.y0*oldheight)/height,p.width,p.height*oldheight/height])
    for t in f.texts:
        x,y=t.get_position();t.set_position((x,ymap(y*oldheight)/height))
    for item in f.artists:
        if isinstance(item,Line2D) and item.get_transform()==f.transFigure:
            x,y=item.get_data();item.set_data(x,[ymap(v*oldheight)/height for v in y])
    f.set_size_inches(180*fs.MM,height*fs.MM);f._height_mm=height


def fig1():
    f = page(146)
    src = fitz.open(HERE / 'out/fig1.pdf')
    clip = fitz.Rect(0, 27, 353, 246.33)
    rect = (1, 62, 124.5, clip.height/clip.width*124.5)
    a = axes(f, *rect)
    a.set_gid('designer-panel')
    pix = src[0].get_pixmap(clip=clip, dpi=180)
    a.imshow(Image.open(io.BytesIO(pix.tobytes('png'))), aspect='auto')
    a.set_axis_off()
    title(f,'a','Writing code; evolving a return',2,142)
    title(f, 'b', 'Emergence from random code', 129, 142)
    a = axes(f, 140, 78, 37, 53)
    d = table('emergence_curves')
    for length, q in d.groupby('L'):
        a.step(q.step, q.emergence, where='post', color=old.L_COL[length], lw=.9, label=f'L = {length}')
    a.set(xscale='log', xlim=(40, 350000), ylim=(-.03, 1.04))
    fs.log10_ticks(a)
    tidy(a, 'World update', 'Fraction with a heritable replicator')
    a.legend(loc='lower center', bbox_to_anchor=(.46, 1.02), ncol=3, fontsize=5,
             handlelength=1.1, columnspacing=.7)
    f.text(134/180, 70/146, 'Z80 archive; L = tape bytes; n = 10 per L', fontsize=6, color=INK)
    title(f, 'c', 'One Z80 world: similarity rises (L = 16)', 2, 59)
    d = table('world_trajectory')
    a = axes(f, 14, 16, 67, 33)
    a.plot(d.step, d.dominant_neighbourhood, color=TEAL, lw=.9)
    a.plot(d.step, d.zero_fraction, color=GREY, lw=.9)
    a.plot(d.step, d.distinct/d.cells, color=INK, lw=.8, ls='--')
    a.set(xscale='log', xlim=(d.step.min(), d.step.max()), ylim=(-.02,1.04))
    fs.log10_ticks(a)
    tidy(a, 'World update', 'Fraction', yticks=[0,.5,1])
    # Names in a compact inline key above the data, following the old Fig. 2.
    k = axes(f, 13, 50, 78, 6, True)
    key(k, 0, 3, 'Cells ≥75%\nmodal match', TEAL, 'line')
    key(k, 28, 3, 'Zero-valued bytes\n/ all bytes', GREY, 'line')
    key(k, 54, 3, 'Distinct tapes\n/ total cells', INK, 'line');k.lines[-1].set_linestyle('--')
    jump=d.dominant_neighbourhood.diff().idxmax()
    text(a,600,.43,'Modal similarity rises',size=6.5,color=TEAL)
    a.annotate('',xy=(d.loc[jump,'step'],d.loc[jump,'dominant_neighbourhood']),xytext=(2000,.35),
               arrowprops=dict(arrowstyle='-',color=TEAL,lw=.6))
    title(f, 'd', 'Five validation comparisons', 94, 57)
    a = axes(f, 95, 9, 83, 44, True)
    d = table('validation_fractions')
    key(a,0,42,'Same',GREY)
    key(a,20,42,'Expected difference','#C98E2F')
    key(a,54,42,'Flagged / changed',RED)
    text(a,0,36,'8088 vs hardware',size=6.5,bold=True)
    text(a,56,36,'Same / tested',size=6.5,bold=True)
    text(a,82,36,'Diff.',size=6.5,bold=True,ha='right')
    text(a,0,22,'Original Z80 vs C1',size=6.5,bold=True)
    for i,r in enumerate(d.itertuples()):
        y=30.5 if i==0 else 17-(i-1)*5
        label=['State agreement','Ring memory stable','Identical assay metrics','Same claim verdict','Same constituent test'][i]
        text(a,29,y,label,size=6.5,ha='right')
        a.add_patch(Rectangle((31,y-1.2),21*r.fraction,2.4,fc=GREY,ec='none'))
        a.add_patch(Rectangle((31+21*r.fraction,y-1.2),21*(1-r.fraction),2.4,fc='#C98E2F' if i==0 else RED,ec='none'))
        value=f'{r.unchanged:,}\n/ {r.total:,}' if i==0 else f'{r.unchanged:,} / {r.total:,}'
        text(a,56,y,value,size=6.5)
        text(a,82,y,f'{r.changed:,}'+('\nexpected' if i==0 else ''),size=6.5,ha='right')
        if i==0:
            text(a,31,26.4,f'{100*(1-r.fraction):.2f}% expected',size=6.5,color='#C98E2F')
            a.annotate('',xy=(52-.5*21*(1-r.fraction),y),xytext=(50,26.4),arrowprops=dict(arrowstyle='-',color='#C98E2F',lw=.6))
    f.text(5/180,5/146,'Seed 2002; match to the contemporaneous modal tape',fontsize=6.5,color=INK)
    f.text(95/180,5/146,'C1 = corrected Z80 instruction semantics',fontsize=6.5,color=INK)
    save(f, 'fig1', (src, clip, rect))
    src.close()


def fig2():
    f = page(150)
    title(f, 'a', 'Carriage can fail despite survival (generation 8)', 2, 146)
    title(f, 'b', 'Serial retention (archive)', 111, 146)
    # The outcome bars and numeric side columns follow _sr_partition in v5.
    k = axes(f, 2, 135, 106, 7, True)
    key(k, 0, 3, 'Lost under\ncopy criterion', old.LOST_C)
    key(k, 34, 3, 'Alive, <50%\ncarrying', old.ERASED_C)
    key(k, 70, 3, 'Alive, ≥50%\ncarrying', TEAL)
    a = axes(f, 35, 109, 48, 24)
    d = table('branching_endpoints').query("cohort=='C1'").set_index('key')
    rows = [('Open first replicator', GREY, [('pusher16','Pusher, L = 16')]),
            ('Closed regenerators', RED, [('ret16','Return closer, L = 16'),('ldir32','Block-copy tiling, L = 32')]),
            ('Closed transmitter', TEAL, [('genome20_s6003','Genome 20, world 3')])]
    y=0; ticks=[]; labs=[]
    c = axes(f, 86, 109, 23, 24, True)
    text(c, 4, 24, 'Controls alive\n(populations)', size=6.5, ha='center')
    text(c, 15, 24, 'Byte sites\nat g8', size=6.5)
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
    f.text(4/180,100/150,'Carrying: mutant five-byte marker window; % of population members',fontsize=6.5,color=INK)
    f.text(4/180,96/150,'Scored site in a,c: excess marker carriage ≥0.5 for ≥50% identifiable values',fontsize=6.5,color=INK)
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
    tidy(a,'Serial transfers','Marker-carrying lineages /\nall starting lineages')
    a.legend(loc='center right',bbox_to_anchor=(.99,.52),fontsize=6,handlelength=1.2)
    text(a,1.2,.105,'At transfer 8: pusher / return = 0;\nblock-copy = '+f'{ss.loc["ldir32","retained_g8"]:.5f}',size=6.5)
    title(f,'c','Serial sites: one box per byte; each track scaled to its tape',2,85)
    a=axes(f,2,41,175,40,True)
    k=axes(f,36,76,140,6,True)
    key(k,0,3,'Instruction-stream\nfetched',old.EXEC_C)
    key(k,37,3,'Not fetched','white')
    k.patches[-1].set_edgecolor(RULE)
    key(k,71,3,'Transfer 1, absent at 8',old.SITE_C,'o')
    k.lines[-1].set_markerfacecolor('white')
    key(k,118,3,'Marker site at 8',old.SITE_C,'o')
    ss=table('serial_summary').set_index('key'); sites=table('serial_sites')
    rows=[('pusher50','Pusher, L = 50'),('ret16','Return closer, L = 16'),
          ('ldir32','Block-copy tiling, L = 32'),('lethal16_s4009','Lethal-tar closer, L = 16'),
          ('genome20_s6004','Genome 20, world 4'),('closer32_8080_p1','Constructed 8080, L = 32')]
    cols=[121,136,152,168]
    for x,lab in zip(cols,['Fetched\nbytes','Scored sites\nat transfer 1','Scored sites\nat transfer 8','Transfer-8\nnot fetched']):
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
    text(a,28,9,'Erasure',size=6.5,color=RED)
    a.annotate('',xy=(r.step,r.address),xytext=(25,9),arrowprops=dict(arrowstyle='-',color=RED,lw=.5))
    a.axhline(m['L']-.5,color=RULE,lw=.5,ls=':')
    a.set(xlim=(-4,260),ylim=(-2,66));tidy(a,'Instruction step','Ring write address',[0,128,256],[0,32,63])
    f.text(15/180,26.7/150,f'Parent 0-{m["L"]-1} | partner {m["L"]}-{2*m["L"]-1}',fontsize=6.5,color=INK)
    a=axes(f,88,11,35,14)
    for encounter,offset in [(1,0),(2,m['steps_per_encounter'])]:
        q=t[t.encounter==encounter]
        a.step(np.r_[offset,q.step+offset],np.r_[1 if encounter==1 else 0,q.marker_sources_remaining],where='post',color=RED,lw=.9)
    a.axvline(128,color=RULE,lw=.7,ls='--')
    a.set(xlim=(-4,260),ylim=(-.05,1.1));tidy(a,'Instruction step','Ancestry-tagged\nbyte copies',[0,128,256],[0,1])
    k=axes(f,128,3,49,29,True)
    text(k,0,27,f'Ancestry label: byte {m["position"]} (0x{m["marker"]:02X})',size=6.5,bold=True)
    text(k,0,21,f'Erasure at instruction {int(r.step)}',color=RED,size=6.5,bold=True)
    text(k,0,15.5,'Read address 4; ancestry: byte 0\nWrites 0x04 over byte 8 (0xA7)',color=RED,size=6.5)
    text(k,0,10,'Labelled byte copies: 1 to 0',size=6.5)
    text(k,0,4,f'{m["steps_per_encounter"]}: offspring to parent; CPU reset',size=6.5)
    reflow(f,136,lambda y: y-14 if y>=90 else y-4 if y>=35 else y)
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
        text(a,9 if d==12 else 8.8,y-1.35,f'Retained prefix d = {d}; offspring B',size=6,ha='center')
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
         '' if detail else 'Reported position count (bytes)',
         [0,5,10] if detail else [0,20,40,60], [0,5,10] if detail else [0,20,40,60])


def fig3():
    f=page(166)
    title(f,'a','Forward overlap protects a prefix',2,162)
    geometry(axes(f,4,96,83,60,True))
    f.text(5/180,88/166,'r = d - c: r free byte positions; d copy offset; c core bytes.',fontsize=6,color=INK)
    title(f,'b','The law: prediction and measurement',97,162)
    a=axes(f,113,95,63,63);law(a)
    # Series are named in the plot; the untested evolved law is stated in the key.
    pop=table('law_population');c1=table('law_z80_c1');c2=table('law_6502_c2');ctrl=table('law_6502_controls')
    handles=[Line2D([],[],marker='o',ls='none',ms=4,mfc='#D3D8DC',mec=GREY,label=f'Z80 qualifying-site scores: {int(pop.cells.sum()):,} cells'),
             Line2D([],[],marker='o',ls='none',ms=3,mfc='none',mec=TEAL,label=f'Z80 exact positions: {int(c1.edits.sum())} settings'),
             Line2D([],[],marker='^',ls='none',ms=3,mfc='#C98E2F',mec='#C98E2F',label='6502 exact positions: 42 sampling roles\n(41 tapes, 11 worlds)'),
             Line2D([],[],marker='s',ls='none',ms=3,mfc='none',mec=INK,label=f'6502 sampled carried positions: {int(ctrl.cases.sum())} controls'),
             Line2D([],[],marker='o',ls='none',ms=3,mfc='none',mec=RED,label='Whole-tape match gate fails (one cell)'),
             Line2D([],[],marker='D',ls='none',ms=3,mfc='none',mec=GREY,label='Registered 8086 law test: untested\nFrozen extractor certified no eligible copier')]
    a.legend(handles=handles,loc='upper left',fontsize=5.2,handlelength=1,borderpad=.15)
    text(a,52,42,'y = x',size=5.7,color=GREY)
    # A real low-capacity enlargement in the unused below-diagonal quadrant.
    inset=axes(f,150,101,23,23);law(inset,True)
    inset.set_title('Same points: 0-10',fontsize=6,pad=2)
    inset.tick_params(labelsize=6)
    a.add_patch(Rectangle((0,0),10,10,fill=False,ec=RULE,lw=.6,ls='--'))
    text(a,10,7,'Assay = 0 sites\nCarried = 4',size=6.5,color=RED)
    a.annotate('',xy=(4,.2),xytext=(12,4),arrowprops=dict(arrowstyle='-',color=RED,lw=.5))
    k=axes(f,104,73,74,13,True)
    text(k,0,10,'Disc area scales with cells; rings = series only',size=6.5,color=INK)
    for x,n in [(36,1),(46,100),(62,2000)]:
        k.scatter(x,5,s=n*.18,facecolors='#D3D8DC',edgecolors=GREY,lw=.3)
        text(k,x,0,str(n) if n<1000 else '2,000',size=5,ha='center')
    pop=table('law_population');c1=table('law_z80_c1');c2=table('law_6502_c2');ctrl=table('law_6502_controls')
    z=table('z80_payload').query('eligible_dividing==True');ev=table('evolved6502_capacity')
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
    text(a,0,3,'Offspring: lineage 0, generation 8',size=6.5)
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
    key(k,0,38,f'{other} working,\noutside domain','white',size=5.1);k.patches[-1].set_edgecolor(TEAL)
    key(k,0,28,f'{ne} in law domain;\nshade = capacity',cmap(.6),size=5.1)
    text(k,0,21,'Free payload',size=5.3,bold=True)
    for i in range(100):
        k.add_patch(Rectangle((0,5+i*.12),3,.125,fc=cmap(i/99),ec='none'))
    text(k,4,5,'0',size=5.3);text(k,4,17,'28 positions',size=5.3)
    text(k,0,1,'Rims: 0x04 / 0x60',size=5.1)
    f.text(105/180,63/166,f'Working: normalized gen-2 partner/parent match gain ≥ {method("gen2_minimum"):g}\nLaw domain: 4≤d≤32; d divides 32; aligned full-ring copy',fontsize=6.5,color=INK,va='center')
    save(f,'fig3')


STARTS={'mix50':('50:50','-','o',10), 'R_into_T':('1% R in T','--','s',5), 'T_into_R':('1% T in R',':','^',5)}


def trajectories(a,tar,measure):
    d=table('competition_medians')
    color=TEAL if tar=='benign' else PURPLE
    for cohort in ['Historical','C1']:
        for start,(label,ls,marker,n) in STARTS.items():
            q=d[(d.cohort==cohort)&(d.tar==tar)&(d.start==start)].sort_values('step')
            a.plot(q.step/1000,q[measure],color=color,ls=ls,lw=.95 if cohort=='C1' else .75,
                   alpha=1 if cohort=='C1' else .7,marker=marker,ms=2.8,
                   markevery=([0,len(q)//4,len(q)//2,3*len(q)//4,len(q)-1] if cohort=='C1' else [len(q)//8,3*len(q)//8,5*len(q)//8,7*len(q)//8]),
                   mfc=color if cohort=='C1' else 'white',mec=color,mew=.55)
    a.set(xlim=(0,305),ylim=(-.03,1.05))
    tidy(a,'World updates (thousands)', 'T / (R + T)' if measure=='T_share' else '5E ED B0-bearing /\nall cells',[0,150,300],[0,.5,1])
    if measure=='core_share':
        ends=table('competition_endpoints')
        a.text(.035,.62,'Endpoint medians: C1 | archive',transform=a.transAxes,fontsize=6.5,va='center',color=INK)
        for x,label in [(.035,'Start'),(.36,'T share'),(.70,'Motif share')]:
            a.text(x,.48,label,transform=a.transAxes,fontsize=6.5,va='center',fontweight='bold',color=INK)
        for i,(start,(label,_,_,_)) in enumerate(STARTS.items()):
            q=[ends[(ends.cohort==cohort)&(ends.tar==tar)&(ends.start==start)] for cohort in ['C1','Historical']]
            y=.34-i*.14
            for x,value in [(.035,label),(.36,' | '.join(f'{g.T_share_end.median():.2f}' for g in q)),(.70,' | '.join(f'{g.core_share_end.median():.2f}' for g in q))]:
                a.text(x,y,value,transform=a.transAxes,fontsize=6.5,va='center',color=INK)



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
    tidy(a,'',{'T_share_end':'Final T / (R + T)','core_share_end':'5E ED B0-bearing /\nall cells',
               'T_distinct_tails_end':'Distinct T tails /\nworld (bytes 4-31)'}[measure],yticks=[0,.5,1] if measure!='T_distinct_tails_end' else None)


def fig4():
    f=page(165)
    title(f,'a','Z80 competitions: full-tape T share and byte-motif prevalence (L = 32)',2,161)
    k=axes(f,4,150,173,8,True)
    for x,(start,(label,ls,marker,n)) in zip([0,36,81],STARTS.items()):
        k.plot([x,x+5],[5,5],color=INK,lw=.9,ls=ls)
        k.plot(x+2.5,5,marker=marker,color=INK,ms=2.8,ls='none')
        text(k,x+7,5,f'{label} ({n} worlds)',size=5.5)
    for yy,col,fill,label in [(5,GREY,'white','Archive: light/open'),(1,INK,INK,'C1: dark/filled')]:
        k.plot([131,136],[yy,yy],color=col,lw=.9)
        k.plot(133.5,yy,'o',ms=3,mfc=fill,mec=col,ls='none')
        text(k,138,yy,label,size=6.5)
    f.text(15/180,143/165,'Partial-copy R | Full-tape T | World medians; C1 = corrected Z80 instructions',fontsize=6.5,color=INK)
    for x,tar,col in [(15,'benign',TEAL),(103,'lethal',PURPLE)]:
        f.text((x+30)/180,137/165,f'{tar.capitalize()} tar: zero executes as '+('no operation' if tar=='benign' else 'halt'),fontsize=6.5,color=col,fontweight='bold',ha='center')
        trajectories(axes(f,x,108,72,25),tar,'T_share')
        trajectories(axes(f,x,73,72,25),tar,'core_share')
    title(f,'b','Final world distributions at 300,000 world updates',2,59)
    f.text(101/180,57.8/165,'Open: archive / filled: C1 implementation',fontsize=6,color=INK)
    f.text(15/180,15.5/165,'20 worlds/group; C1 benign ratio n = 19 (one world has R + T = 0)',fontsize=6.5,color=INK)
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
    text(a,0,16,'p: probability a zero instruction halts',size=6.5,bold=True)
    for j,p in enumerate(sorted(d.p.unique())):
        text(a,60+j*19,16,f'{p:g}',size=6,ha='center',bold=True)
    for y,field,label in [(11,'open_first','First copier open (out of 10)'),(6,'T_majority','T >50% heritable cells at 300k (out of 10)'),(1,'confinement_km','Kaplan-Meier median to first confined copier\n(thousand world updates)')]:
        text(a,0,y,label,size=5.3)
        for j,p in enumerate(sorted(d.p.unique())):
            c=d[(d.cohort=='C1')&(d.p==p)].iloc[0];h=d[(d.cohort=='Historical')&(d.p==p)].iloc[0]
            text(a,60+j*19,y,f'{c[field]:g} | {h[field]:g}',size=6.5,ha='center')
    f.text(126/180,36/170,'Each entry: C1 | archive',fontsize=6.5,color=INK)
    for y in [11.5,16.5,21.5,26.5,31.5]:
        f.add_artist(Line2D([8/180,177/180],[y/170,y/170],transform=f.transFigure,color=RULE,lw=.35))
    f.text(8/180,8.5/170,'Open: partner instruction-stream fetch in ≥8/16 traces; confined: none in 16/16',fontsize=6.5,color=INK)
    reflow(f,163,lambda y:y-7)
    save(f,'fig4')


def fig5():
    f=page(162)
    title(f,'a','First-replicator route',2,148)
    a=axes(f,3,98,85,44,True)
    for x,label,col in [(0,'Open\npartner fetch',GREY),(27,'Confined\nno partner fetch',TEAL),(56,'Censored\nno detection','#E8E9EA')]:
        key(a,x,40,label,col)
    text(a,75,34,'Detected',size=6.5,bold=True,ha='center')
    routes=table('route_by_machine')
    for i,r in enumerate(routes.itertuples()):
        y=27-i*11
        text(a,0,y+1.5,r.machine,bold=True,size=7)
        text(a,0,y-2.2,r.cohort,size=6.5,color=GREY)
        left=29
        for field,col in [('open',GREY),('confined',TEAL),('censored','#E8E9EA')]:
            n=int(getattr(r,field));w=40*n/r.worlds
            if not n:continue
            a.add_patch(Rectangle((left,y-2.5),w,5,fc=col,ec='none'))
            text(a,left+w/2,y,str(n),size=7,bold=True,ha='center',color=INK if field=='censored' else 'white')
            left+=w
        text(a,75,y,f'{r.open+r.confined}/{r.worlds}',size=6.5,ha='center')
    text(a,29,0,'0',size=6.5,ha='center');text(a,49,0,'0.5',size=6.5,ha='center');text(a,69,0,'1',size=6.5,ha='center')
    text(a,49,-3,'Share of all worlds',size=6.5,ha='center')
    title(f,'b','Arrival: first detected heritable copier',94,148)
    a=axes(f,106,104,70,37)
    colors={'Z80 G':GREY,'6502 A':'#C98E2F','8086 S2':TEAL}
    d=table('arrival_curves')
    for name,q in d.groupby('machine',sort=False):
        a.step(q.step,q.detected,where='post',color=colors[name],lw=1.15)
    for r in table('arrival_censors').itertuples():
        a.plot(r.step,r.detected,marker='|',ms=6,color=colors[r.machine],ls='none',mew=1.1)
    a.set(xscale='log',xlim=(40,1.2e7),ylim=(-.02,1.08))
    fs.log10_ticks(a);tidy(a,'World updates','Fraction with\na detected copier',yticks=[0,.5,1])
    for name,x,y,label in [('Z80 G',3500,.90,'Z80 Stage G · 20/20'),('6502 A',3500,.62,'6502 S1 A · 20/20'),
                            ('8086 S2',4e3,.38,'8086 S2 · 14/20')]:
        text(a,x,y,label,size=6.5,bold=True,color=colors[name])
    text(a,1e5,.79,'6 censored',size=6.5,color=TEAL)
    a.annotate('',xy=(3e6,.7),xytext=(8e5,.78),arrowprops=dict(arrowstyle='-',color=TEAL,lw=.6))
    f.text(106/180,95/162,'16-byte tapes; 20 worlds/cohort; | censored',fontsize=6.5,color=INK)
    f.text(4/180,154/162,'Open: partner instruction-stream fetch in ≥8/16 encounters; confined: none (S1/S2)',fontsize=6.5,color=INK)
    title(f,'c','Literal-channel expectation and observed beginnings',2,90)
    a=axes(f,4,58,172,27,True)
    for x,label in [(0,'Machine'),(37,'Literal width\n(bytes / push)'),(76,'Census: exact pusher\n(opcode motif)'),(123,'First observed route')]:
        text(a,x,23,label,bold=True,size=6.5)
    matrix=table('literal_matrix')
    for i,r in enumerate(matrix.itertuples()):
        y=16-i*6.25
        if r.machine=='8086':
            a.add_patch(Rectangle((-1,y-3.3),172,6.6,fc='#EEF6F5',ec=TEAL,lw=.65))
        text(a,0,y,r.machine,bold=True,size=7)
        text(a,44,y,str(r.width),size=7,ha='center')
        text(a,76,y,r.pusher+(' · '+r.witness if pd.notna(r.witness) else ''),size=6.5)
        route=f'Open · {r.detected}/{r.detected}' if r.machine=='Z80' else f'Confined · {r.confined}/{r.detected}'
        if r.machine=='8086':route+=' (2 open)'
        text(a,123,y,route,size=6.5,bold=r.machine=='8086',color=TEAL if r.machine!='Z80' else GREY)
    f.text(4/180,55/162,'Census candidates; first routes from evolved worlds',fontsize=6.5,color=INK)
    title(f,'d','8086 convergence: the 5F A4 x y first-copier family (10/14; all ten confined)',2,52)
    a=axes(f,4,3,172,44,True)
    text(a,0,42,'World seed',bold=True,size=6.5)
    text(a,36,42,'5F A4 = POP DI; MOVSB',bold=True,size=6.5,color=TEAL)
    text(a,91,42,'x, y: variable bytes',size=6.5)
    text(a,130,42,'Detection\n(update)',bold=True,size=6.5,ha='center')
    text(a,153,42,'Mechanism\n(16/16 encounters)',bold=True,size=6.5,ha='center')
    start,w=35,5.4
    for j in range(16):text(a,start+(j+.5)*w,37,str(j),size=6.5,ha='center',color=GREY)
    d=table('first_copier_tiles')
    for i,(seed,q) in enumerate(d.groupby('seed',sort=True)):
        y=33-i*3.45
        text(a,0,y,str(seed),size=6.5)
        for r in q.itertuples():
            x=start+r.position*w
            shared=r.role=='shared'
            a.add_patch(Rectangle((x,y-1.35),w,2.7,fc='#D3E8E7' if shared else '#F0F1F2',ec='white',lw=.25))
            text(a,x+w/2,y,r.byte,size=6.5,ha='center',color=TEAL if shared else INK,bold=shared,fontfamily='DejaVu Sans Mono')
        r=q.iloc[0]
        text(a,130,y,f'{int(r.detection):,}',size=6.5,ha='center')
        text(a,153,y,'One-pass copy' if r.mechanism=='memory-fed one-pass copier' else 'Copy loop',size=6.5,ha='center',color=TEAL)
    for i in range(4):
        x=start+i*4*w
        # Tile separators stop at the strip edges and never cross byte labels.
        a.add_patch(Rectangle((x,.4),4*w,34.2,fill=False,ec=RULE,lw=.4))
    save(f,'fig5')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',default='1,2,3,4,5');args=ap.parse_args()
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

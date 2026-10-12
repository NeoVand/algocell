"""v6 data figures; all points read from prepare_v6.py generated tables.

No conceptual illustration is drawn. Fixed 180-mm pages and fixed-point type
are preserved; figcheck runs on EVERY panel and composed figure before saving.
"""
from pathlib import Path
import argparse
import json
import sys
import textwrap
import subprocess

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import figstyle as fs
import figcheck
from captions_v6 import captions
from matplotlib import pyplot as plt

OUT = HERE / 'out/v6'
D = OUT / 'data'
TEAL, RED, PURPLE = '#1E8A8A', '#E8431F', '#7B3294'
INK, GREY, LIGHT = '#1C2733', '#8A95A1', '#E3E5E8'
KEYS = ['pusher16','ret16','ldir32','genome20_s6003']
NAMES = {'pusher16':'Pusher', 'ret16':'Return R', 'ldir32':'Block-copy R', 'genome20_s6003':'Transmitter'}
STYLES = {'mix50':'-', 'R_into_T':'--', 'T_into_R':':'}
STARTS = {'mix50':'Equal start','R_into_T':'R invades T','T_into_R':'T invades R'}
REPORTS = {}


def table(n):return pd.read_csv(D / (n+'.csv'), low_memory=False)


class Panel:
    def __init__(self, fig, rect, letter, title):
        self.fig, self.rect = fig, rect
        self.canvas = self.ax(0,0,1,1)
        self.canvas.set_axis_off()
        fig.__dict__.setdefault('_v6_panels',[]).append(self)
        self.text(0.015,.98,letter,size=8,bold=True)
        self.text(.067,.978,title,size=7,bold=True)

    def ax(self,x,y,w,h):
        a,b,c,d=self.rect
        return self.fig.add_axes([a+x*c,b+y*d,w*c,h*d])

    def text(self,x,y,s,size=6.5,bold=False,**kw):
        return self.canvas.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',
                                color=INK,va='top',transform=self.canvas.transAxes,**kw)

    def foot(self,s):
        width=70 if self.rect[2]<.6 else 145
        lines=textwrap.wrap(s,width=width)
        self.text(.015,.09 if len(lines)>1 else .065,'\n'.join(lines),size=6)


def axis(ax,xlabel='',ylabel='',xticks=None,yticks=None):
    fs.tidy(ax,xlabel or None,ylabel or None)
    if xticks is not None:ax.set_xticks(xticks)
    if yticks is not None:ax.set_yticks(yticks)


def slot(p,description):
    p.text(.5,.56,description,ha='center',size=7)
    p.text(.5,.42,'Designer brief: DESIGNER_ROUND6',ha='center',size=6)


def fig1c(p):
    t=table('world_trajectory')
    ax=p.ax(.12,.25,.52,.57)
    ax.plot(t.step/1000,t.dominant_neighbourhood,color=TEAL,label='Dominant-tape neighbourhood')
    ax.plot(t.step/1000,t.zero_fraction,color=GREY,label='Zero-byte fraction')
    ax.set(xlim=(0,102),ylim=(-.025,1.04))
    axis(ax,'Step (thousands)','Fraction',[0,50,100],[0,.5,1])
    p.text(.12,.89,f"Illustrative world · seed {int(t.seed.iloc[0])} · {int(t.cells.iloc[0]):,} cells",size=6)
    p.text(.12,.13,'Teal: ≥12/16-byte modal match, cells / 20,000\nGrey: zero bytes / 320,000',size=5.7)
    b=table('branching').query("cohort=='Historical' and arm=='U' and gen==8")
    traces=table('reference_traces')
    p.text(.72,.85,'Reference assays',bold=True)
    for y,key in [(.69,'pusher16'),(.40,'ret16')]:
        r=b[b.key==key].iloc[0]
        p.text(.72,y,NAMES[key],bold=True)
        p.text(.72,y-.09,f"{int(r.ctrl_alive_n)}/{int(r.ctrl)} controls alive\n{int(r.sites)}/{int(r.L)} sites retained",size=6)
        q=traces[traces.key==key]
        if len(q):
            q=q.iloc[0]
            p.text(.72,y-.17,f"Entry: {int(q.entered_n)}/{int(q.encounters)}\nInflow: {max(0,q.H_bits):.2f} bits",size=5.7)
    p.foot('Historical; entry: n/256 encounters. Reference assays are separate.')


def fig1d(p):
    v=table('8086_validation').iloc[0]
    i=table('impact').query("core=='C1'").iloc[0]
    claims=table('claims');comp=table('components')
    rows=[('Instruction vectors',f"8088 raw pass: {int(v.pass_arch):,}/{int(v.tests):,}",
           f"{int(v.tests-v.pass_arch):,} expected divergences; queue untested"),
          ('Ring encounters',f"C1 unchanged: {int(i.unaffected)}/{int(i.decision_conditions)} conditions",f"{int(i.flagged)} flagged conditions"),
          ('Reproductive assays',f"C1 identical: {int(i.assay_identical)}/{int(i.assay_combinations)} cases",f"{int(i.assay_differences)} changed assay cases"),
          ('Ecological reruns',f"C1 verdicts retained: {int((claims.status=='stands').sum())}/{len(claims)}",f"Components unchanged: {int((comp.comparison=='same').sum())}/{len(comp)}"),
          ('8086 ecology','PENDING S2','No population outcome plotted')]
    for y,(label,a,b) in zip([.84,.68,.52,.36,.20],rows):
        p.text(.035,y,label,bold=True,size=6.5)
        p.text(.38,y,a,size=6.2)
        p.text(.38,y-.065,b,size=6)
    p.foot('Units differ between validation levels; 8086 ecology awaits S2.')


def fig2a(p):
    b=table('branching').query("arm=='U' and key in @KEYS")
    # Four small charts share limits; control denominator and allele denominator
    # stay separate. C1 overlays are dashed, so coincidence is visible.
    for j,key in enumerate(KEYS):
        x=.10+(j%2)*.47;y=.52 if j<2 else .20
        ax=p.ax(x,y,.36,.23)
        d=b[b.key==key]
        for cohort,ls in [('Historical','-'),('C1','none')]:
            q=d[d.cohort==cohort].sort_values('gen')
            for field,col in [('ctrl_alive',INK),('mut_alive',RED),('retained',TEAL)]:
                ax.plot(q.gen,q[field],color=col,ls=ls,lw=.9,
                        marker='o' if cohort=='C1' else None,ms=2.1,
                        mfc='white',mec=col,markevery=[1,3,7])
        r=d.query("gen==8 and cohort=='Historical'").iloc[0]
        p.text(x,y+.29,f"{NAMES[key]} · g8 {int(r.ctrl_alive_n)}/{int(r.ctrl)} controls",size=6,bold=True)
        p.text(x,y+.255,f"{int(r.n_identifiable)}×{int(r.reps)} mutant starts; g8 sites {int(r.sites)}/{int(r.L)}",size=5.7)
        ax.set(xlim=(.7,8.3),ylim=(-.04,1.06))
        axis(ax,'Generation' if j>=2 else '', 'Fraction of starts' if j%2==0 else '',[1,4,8],[0,1])
        if j<2:ax.tick_params(axis='x',labelbottom=False)
    p.text(.10,.93,'Black: controls alive   Red: mutants alive   Teal: carriage',size=6)
    p.canvas.plot([.07,.13],[.855,.855],transform=p.canvas.transAxes,color=INK,lw=.9)
    p.canvas.plot([.10],[.855],transform=p.canvas.transAxes,marker='o',mfc='white',mec=INK,ms=2.2,ls='none')
    p.text(.15,.87,'Historical lines; C1 circles coincide.',size=6)
    p.foot('X1-2 prediction failed: 0 scored sites (completed assay).')


def fig2b(p):
    d=table('retention_maps');s=table('retention_sites')
    cmap=mcolors.ListedColormap([LIGHT,RED,TEAL,'#56616F'])
    for j,key in enumerate(KEYS):
        q=d[d.key==key];L=int(q.pos.max()+1)
        image=np.full((256,L),3)
        for r in q.itertuples():image[r.val,r.pos]=r.outcome
        x=.07+j*.238
        ax=p.ax(x,.34,.195,.40)
        ax.imshow(image,origin='lower',aspect='auto',interpolation='nearest',cmap=cmap,vmin=0,vmax=3,
                  extent=(-.5,L-.5,-.5,255.5),rasterized=True)
        ax.set(xlim=(-.7,L-.3),ylim=(-2,268))
        core=s[(s.key==key)&s.executed.astype(bool)].pos
        ax.plot(core,[262]*len(core),'|',color=INK,ms=2.5,mew=.6,ls='none')
        axis(ax,'Position','Alternative' if j==0 else '',[0,L-1],[0,255] if j==0 else [])
        p.text(x,.85,NAMES[key],size=6,bold=True)
        corecarry=q[(q.outcome==2)&q.pos.isin(core)]
        p.text(x,.80,f"{len(q):,} alternatives",size=5.7)
        counts=corecarry.groupby('pos').size()
        summary='Core: none' if counts.empty else 'Core: '+', '.join(f'{int(pos)} ({int(n)})' for pos,n in counts.items())
        p.text(x,.22,summary,size=5.7)
    p.text(.07,.15,'Grey: 16 dead; red: survivors, none carrying; teal: ≥1 carrier\nDark grey: untested. Core: position (carrier alternatives).',size=5.7)
    p.foot('Historical serial g8; 16 lines/alternative. Black ticks: fetched sites.')


def fig2c(p):
    t=table('erasure_ordered')
    m=json.loads((D/'erasure_manifest.json').read_text())
    write=t[t.kind=='write']
    ax=p.ax(.13,.39,.83,.43)
    for encounter,offset in [(1,0),(2,128)]:
        q=write[write.encounter==encounter]
        ax.plot(q.step+offset,q.address,'.',color=GREY,ms=1.4,ls='none')
    erasure=write[(write.encounter==1)&(write.prior_source==m['position'])].iloc[0]
    ax.plot(erasure.step,erasure.address,'o',mfc='white',mec=RED,ms=4,mew=.9)
    ax.axvline(128,color=LIGHT,lw=.6)
    ax.axhline(32,color=LIGHT,lw=.5)
    ax.set(xlim=(-4,260),ylim=(-2,65))
    axis(ax,'','Write address',[0,64,128,192,256],[0,32,64])
    p.text(.13,.90,'ldir32, one lineage: 0–127 parent, 128–255 descendant',size=6)
    p.text(.13,.31,f"Circle: introduced site {m['position']} overwritten at step {int(erasure.step)}",size=6)
    low=p.ax(.13,.15,.83,.10)
    for encounter,offset in [(1,0),(2,128)]:
        q=t[t.encounter==encounter]
        low.step(np.r_[offset,q.step.to_numpy()+offset],np.r_[1 if encounter==1 else 0,q.marker_sources_remaining],where='post',color=RED)
    low.set(xlim=(-4,260),ylim=(-.1,1.15))
    axis(low,'Execution step','Marker bytes',[0,128,256],[0,1])
    p.foot(f"All {len(t):,} events archived; dots: writes; reset at 128.")


def fig2d(p):
    d=table('parent_environment')
    envs=list(d.environment.unique());parents=list(d.parent.unique())
    parentnames=['Pusher','Return R','Block-copy R','Block-copy T','Lethal closer','Self-initialising T']
    envnames=['Random\nbenign','Random\nlethal','Closed\nsoup','Lethal\nsoup','Open\nsoup','Random\nregisters']
    ax=p.ax(.21,.22,.75,.58)
    ax.set_axis_off();ax.set(xlim=(-.5,5.5),ylim=(5.5,-.5))
    for i,parent in enumerate(parents):
        for j,env in enumerate(envs):
            r=d[(d.parent==parent)&(d.environment==env)].iloc[0]
            fc=LIGHT if r.control_alive_n==0 else '#D8ECEC' if r.sites_g8>0 else '#FBE2DA'
            ax.add_patch(Rectangle((j-.46,i-.44),.92,.88,fc=fc,ec='white',lw=.5))
            label='×' if r.control_alive_n==0 else f"{int(r.control_alive_n)}/{int(r.control_n)}\n{int(r.sites_g8)}/16"
            ax.text(j,i,label,ha='center',va='center',fontsize=6)
    for i,n in enumerate(parentnames):p.text(.19,.80-(i+.5)*.58/6,n,ha='right',size=6)
    for j,n in enumerate(envnames):p.text(.21+(j+.5)*.75/6,.88,n,ha='center',size=5.7)
    p.text(.21,.15,'Each cell: controls alive / 256; retained sites / 16',size=6)
    p.foot('Grey ×: extinct; peach: 0 sites; teal: >0 sites. Soup partners; 16 lines/allele.')


def grouped_points(ax,d,x,y,color,marker,label):
    g=d.groupby([x,y]).size().reset_index(name='n')
    ax.scatter(g[x],g[y],s=18,facecolor='white',edgecolor=color,marker=marker,lw=.7,label=label,zorder=3)
    return g


def fig3b(p):
    z=table('z80_payload');z=z[z.eligible_dividing.astype(bool)]
    e=table('evolved6502_capacity');c=table('constructed6502');c=c[c.hypotheses.astype(bool)]
    p.text(.12,.90,f"Z80: {len(z):,} cells / {z.world.nunique()} worlds; evolved 6502: {len(e)} / {e.world.nunique()} worlds",size=5.7)
    p.text(.12,.85,f"Constructed 6502: {len(c)} cases. Coincident coordinates merged.",size=5.7)
    ax=p.ax(.13,.41,.48,.35)
    ax.plot([0,60],[0,60],color=GREY,lw=.5)
    grouped_points(ax,z,'predicted','n_unexec',TEAL,'o','Z80 sampled')
    grouped_points(ax,c,'d − 6','payload carried (sites ≥ 6)',GREY,'s','Constructed 6502')
    ax.scatter(e.predicted_free,e.measured_free,s=9,color=RED,marker='^',lw=.5,zorder=5)
    ax.set(xlim=(-3,63),ylim=(-3,63))
    axis(ax,'Predicted payload (positions)','Score or capacity (positions)',[0,30,60],[0,30,60])
    deficit=z[z.residual<0].iloc[0]
    ax.plot([deficit.predicted,18],[deficit.n_unexec,5],color=GREY,lw=.5)
    ax.text(20,5,'Gate deficit',fontsize=5.5,va='center')
    zoom=p.ax(.73,.43,.22,.31)
    zoom.plot([0,10],[0,10],color=GREY,lw=.5)
    # Exact coordinates preserved; larger open squares and smaller filled
    # triangles remain distinguishable where 6502 classes coincide.
    q=c[c['d − 6']<=10].drop_duplicates(['d − 6','payload carried (sites ≥ 6)'])
    zoom.scatter(q['d − 6'],q['payload carried (sites ≥ 6)'],s=22,facecolors='none',edgecolors=GREY,marker='s',lw=.6)
    q=e[e.predicted_free<=10].drop_duplicates(['predicted_free','measured_free'])
    zoom.scatter(q.predicted_free,q.measured_free,s=8,color=RED,marker='^',lw=.5)
    q=z[z.predicted<=10].drop_duplicates(['predicted','n_unexec'])
    zoom.scatter(q.predicted,q.n_unexec,s=32,facecolors='none',edgecolors=TEAL,marker='o',lw=.6)
    zoom.set(xlim=(-.7,10.7),ylim=(-.7,10.7))
    axis(zoom,'Predicted','',[0,10],[0,10])
    p.text(.73,.79,'0–10 detail',size=6)
    w=table('z80_worlds')
    low=p.ax(.13,.16,.48,.10)
    grouped_points(low,w,'predicted','residual',GREY,'o','World means')
    low.axhline(0,color=LIGHT,lw=.6)
    low.set(xlim=(-3,63),ylim=(-1.05,.2))
    axis(low,'','Mean residual',[0,30,60],[-1,0])
    p.text(.13,.30,'Z80 world mean: score − prediction (positions)',size=5.5)
    p.text(.70,.27,'8086: pending data',size=5.7,bold=True)
    p.text(.70,.21,f"Z80: {int((z.fetched_extra>0).sum())} cells with extra\nfetched sites; 4 excluded\n1 gate deficit",size=5.7)
    p.foot('Circles: Z80 scores; triangles: evolved 6502 capacity; squares: constructed carriage scores.')


def fig3c(p):
    e=table('capacity_edits');s=table('capacity_serial');a=table('allele_dial')
    p.text(.14,.90,f"{len(e)} edits / {e.key.nunique()} archived parents; 16 lineages/edit; g8 capacity",size=6)
    p.text(.14,.85,'Each symbol: unique capacity pair; coincident edits merged.',size=5.7)
    ax=p.ax(.14,.30,.78,.49)
    ax.plot([0,60],[0,60],color=GREY,lw=.5)
    grouped_points(ax,e,'predicted_free','measured_free',TEAL,'o','Forward C1 edits')
    rec=table('capacity_direct').query("direction=='reciprocal'")
    # Reciprocal coordinates are only plotted when exact direct assays verify them.
    ch=table('capacity_challenges').query("direction=='reciprocal'")
    assert (ch.passed==ch.cases).all()
    grouped_points(ax,rec,'predicted_free','measured_free',RED,'s','Reciprocal exact')
    ax.set(xlim=(-3,63),ylim=(-3,63))
    axis(ax,'Predicted d − 4 (positions)','Descendant capacity',[0,20,40,60],[0,20,40,60])
    p.text(.14,.17,'Circle: forward; square: reciprocal. Grey diagonal: equality.',size=6)
    p.foot('Registered C1; measured generation-eight capacities from 0 to 60 positions.')


def fig3d(p):
    c=table('capacity_challenges');e=table('evolved6502_challenges')
    p.text(.08,.88,'Exact successes / intervention cases',size=6,bold=True)
    p.text(.39,.81,'Z80 C1',bold=True);p.text(.70,.81,'Evolved 6502 C2',bold=True)
    for y,kind,label in [(.75,'single','All-value single'),(.62,'joint','Joint assignments'),(.49,'erased','Beyond protected prefix')]:
        p.text(.08,y,label,size=6)
        a=c[c.kind==kind];b=e[e.kind==kind]
        p.text(.39,y,f"{int(a.passed.sum()):,}/{int(a.cases.sum()):,}",size=6)
        p.text(.70,y,f"{int(b.passed.sum()):,}/{int(b.cases.sum()):,}",size=6)
    controls=table('constructed6502')
    eligible=controls.hypotheses.astype(bool)
    excluded=controls[~eligible]
    p.text(.08,.36,'Success: full offspring and byte ancestry match prediction.',size=5.7)
    p.text(.08,.26,'Constructed eligibility (separate, exploratory)',size=6,bold=True)
    p.text(.08,.18,f"{len(excluded):,} excluded cases; {len(controls):,} screened cases",size=6)
    p.foot('8086 joint analysis: no data yet; prospective study pending. See caption.')


def fig4a(p):
    t=table('competition_trajectories');end=table('competition_endpoints')
    missing=table('missing_trajectories')
    for j,(cohort,measure,ylabel) in enumerate([('Historical','T_share','T / (R + T)'),('C1','T_share','T / (R + T)'),('Historical','core_share','Core-bearing cells / 20,000'),('C1','core_share','Core-bearing cells / 20,000')]):
        x=.07+j*.237
        ax=p.ax(x,.28,.18,.49)
        q=t[t.cohort==cohort]
        for tar,col in [('benign',TEAL),('lethal',PURPLE)]:
            for start,ls in STYLES.items():
                d=q[(q.tar==tar)&(q.start==start)]
                if len(d):
                    med=d.groupby('step')[measure].median()
                    ax.plot(med.index/1000,med,color=col,ls=ls,lw=1)
                else:
                    d=end[(end.cohort==cohort)&(end.tar==tar)&(end.start==start)]
                    y=d.T_share_end.median() if measure=='T_share' else d.core_share_end.median()
                    xpos={'mix50':0,'R_into_T':1,'T_into_R':2}[start]+(-.13 if tar=='benign' else .13)
                    ax.plot([xpos],[y],marker={'mix50':'o','R_into_T':'s','T_into_R':'^'}[start],color=col,ls='none',mfc=col,ms=3)
        ax.set_ylim(-.04,1.05)
        if q.empty:
            ax.set_xlim(-.5,2.5)
            axis(ax,'All at 300,000 steps',ylabel,[0,1,2],[0,.5,1])
            ax.set_xticklabels(['Equal','R in T','T in R'])
            ax.text(.06,.65,'Trajectories unavailable',transform=ax.transAxes,fontsize=5.7,va='top')
        else:
            ax.set_xlim(-10,310)
            axis(ax,'Step (thousands)',ylabel,[0,150,300],[0,.5,1])
        p.text(x,.88,cohort+' endpoints' if q.empty else cohort,bold=True,size=6)
    p.text(.07,.13,'Benign: teal; lethal: purple. Equal: solid/circle; R invades: dash/square; T invades: dot/triangle.',size=6)
    p.foot('L32, mutation on; medians of 10 equal-start or 5 invasion worlds/chemistry/cohort. C1 offsets separate chemistries.')


def fig4b(p):
    d=table('competition_endpoints')
    for j,(measure,ylabel) in enumerate([('T_share_end','T / (R + T)'),('core_share_end','Core-bearing fraction'),('T_distinct_tails_end','Distinct transmitter tails')]):
        x=.08+j*.32
        ax=p.ax(x,.34,.24,.47)
        for i,(cohort,tar) in enumerate([('Historical','benign'),('C1','benign'),('Historical','lethal'),('C1','lethal')]):
            q=d[(d.cohort==cohort)&(d.tar==tar)]
            col=TEAL if tar=='benign' else PURPLE
            r=q[q[measure].notna()].sort_values([measure,'start','seed'])
            yrange=1.11 if j<2 else max(d[measure])*1.15
            placed=[]
            for row in r.to_dict('records'):
                value=row[measure]
                candidates=[0]+[s*v for v in [.11,.22,.33] for s in [-1,1]]
                costs=[sum((abs(value-y)<yrange*.045 and abs(off-x)<.10) for x,y in placed) for off in candidates]
                off=candidates[int(np.argmin(costs))];placed.append((off,value))
                marker={'mix50':'o','R_into_T':'s','T_into_R':'^'}[row['start']]
                ax.plot(i+off,value,marker=marker,ls='none',ms=2.6,mfc=col if cohort=='C1' else 'white',mec=col,mew=.6)
            ax.plot([i+.40,i+.52],[q[measure].median()]*2,color=INK,lw=.8)
        ax.set_xlim(-.6,3.65)
        if j<2:ax.set_ylim(-.04,1.07)
        else:ax.set_ylim(-150,max(d[measure])*1.08)
        axis(ax,'',ylabel)
        ax.set_xticks(range(4),['H\nbenign','C1\nbenign','H\nlethal','C1\nlethal'])
        p.text(x,.89,'20 worlds / group',size=6)
    nloss=int((d.core_share_end==0).sum())
    p.text(.08,.17,f"Circle: equal (10); square: R invades T (5); triangle: T invades R (5).",size=6)
    p.text(.08,.115,f"Missing ratios: × {nloss} C1 benign core-loss world, excluded from T median; retained in abundance/tails.",size=5.7)
    p.foot('Open: historical; filled: C1. Black bars at right: medians of defined values; horizontal offsets avoid collisions.')


def km(times,horizons):
    alive=1.;n=len(times)
    t=np.where(np.isfinite(times),times,horizons)
    event=np.isfinite(times)
    for time in sorted(set(t[event])):
        risk=int((t>=time).sum());deaths=int(((t==time)&event).sum())
        alive*=1-deaths/risk
        if alive<=.5:return time
    return np.nan


def fig4c(p):
    d=table('dial_worlds')
    probs=sorted(d.p.unique())
    for j,ylabel in enumerate(['Open first (worlds / 10)','T majority (worlds / 10)','Confinement median (k steps)']):
        x=.08+j*.32
        ax=p.ax(x,.32,.24,.50)
        for cohort,col,ls,mfc in [('Historical',GREY,'--','white'),('C1',PURPLE,'-',PURPLE)]:
            q=d[d.cohort==cohort]
            values=[]
            for prob in probs:
                a=q[q.p==prob]
                assert len(a)==10
                if j==0:values.append(int((a.first_open==True).sum()))
                elif j==1:values.append(int((a.frac_transmitter_of_heritable>.5).sum()))
                else:values.append(km(a.t_closed.to_numpy(),a.horizon.to_numpy())/1000)
            ax.plot(range(len(probs)),values,color=col,ls=ls,marker='o',mfc=mfc,mec=col,ms=4.2 if cohort=='Historical' else 2.1,lw=.9)
        ax.set_xlim(-.25,5.25)
        if j<2:ax.set_ylim(-.5,10.6)
        else:ax.set_ylim(-10,300)
        axis(ax,'Tested probability level',ylabel)
        ax.set_xticks(range(len(probs)),['0','.01','.03','.1','.3','1'])
        p.text(x,.90,'10 registered worlds / probability',size=6)
    comp=table('components')
    for j,claim in enumerate(['DZ1','DZ3','DZ2']):
        a=comp[comp.claim==claim]
        hist='PASS' if (a.published_verdict=='met').all() else 'FAILED component'
        c1='PASS' if (a.C1_verdict=='met').all() else 'FAILED component'
        p.text(.08+j*.32,.15,f"{claim}: historical {hist}; C1 {c1}",size=6)
    p.foot('Grey open/dashed: historical; purple: C1. KM: all 10 worlds, nondetections censored at 300k; p=.3 → 1 timing reversal.')


PANELS={'fig1c':fig1c,'fig1d':fig1d,'fig2a':fig2a,'fig2b':fig2b,'fig2c':fig2c,'fig2d':fig2d,
        'fig3b':fig3b,'fig3c':fig3c,'fig3d':fig3d,'fig4a':fig4a,'fig4b':fig4b,'fig4c':fig4c}
TITLES={'fig1c':'From world to assay','fig1d':'Validation has distinct levels',
        'fig2a':'Survival and carriage diverge','fig2b':'Full alternative maps',
        'fig2c':'Erasure is an ordered operation','fig2d':'Parent × environment',
        'fig3b':'Prediction across copiers','fig3c':'The inherited capacity switch',
        'fig3d':'Joint payload and boundaries','fig4a':'Matched chemistry, three starting mixtures',
        'fig4b':'Endpoints keep every world','fig4c':'Lethality dial: historical and C1'}


def save(fig,name):
    report=figcheck.check(fig)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for p in fig._v6_panels:
        bb=p.canvas.get_window_extent(renderer)
        for t in p.canvas.texts:
            tb=t.get_window_extent(renderer)
            if tb.x0<bb.x0-1 or tb.x1>bb.x1+1 or tb.y0<bb.y0-1 or tb.y1>bb.y1+1:
                report.append('panel text extends beyond reserved slot: '+t.get_text())
    for ax in fig.axes:
        if not ax.axison:continue
        for collection in ax.collections:
            offsets=np.asarray(collection.get_offsets())
            if len(offsets) and offsets.shape[1]==2:
                x0,x1=ax.get_xlim();y0,y1=ax.get_ylim()
                if ((offsets[:,0]<x0)|(offsets[:,0]>x1)|(offsets[:,1]<y0)|(offsets[:,1]>y1)).any():
                    report.append('scatter data clipped')
    REPORTS[name]=report
    figcheck.print_report(report,name)
    if report:
        plt.close(fig)
        return
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/(name+'.'+ext),dpi=600,bbox_inches=None,facecolor='white')
        if ext=='svg':
            path=OUT/(name+'.svg')
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    subprocess.run(['pdftoppm','-r','600','-png','-singlefile',str(OUT/(name+'.pdf')),str(OUT/name)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    plt.close(fig)


def main():
    # Keep the established entry point on the coordinator's current redesign.
    from build_v62 import main as redesign
    return redesign()


def legacy_main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',default='1,2,3,4');args=ap.parse_args()
    if args.only!='1,2,3,4' and (OUT/'figcheck.json').exists():
        REPORTS.update(json.loads((OUT/'figcheck.json').read_text()))
    fs.setup()
    plt.rcParams.update({'font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],'font.size':6.5,'axes.labelsize':6.5,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,'axes.titleweight':'normal'})
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        for number in args.only.split(','):
            for name,fn in PANELS.items():
                if name[3]==number:
                    fig=plt.figure(figsize=(180*fs.MM,72*fs.MM))
                    p=Panel(fig,(.025,.025,.95,.95),name[-1],TITLES[name]);fn(p);save(fig,name)
            height={'1':144,'2':154,'3':154,'4':170}[number]
            fig=plt.figure(figsize=(180*fs.MM,height*fs.MM))
            if number=='4':
                for j,letter in enumerate('abc'):
                    name='fig4'+letter
                    p=Panel(fig,(.01,.667-j*.333,.98,.327),letter,TITLES[name]);PANELS[name](p)
            else:
                for j,letter in enumerate('abcd'):
                    name='fig'+number+letter
                    rect=(.005+(j%2)*.5,.5 if j<2 else 0,.49,.49)
                    p=Panel(fig,rect,letter,TITLES.get(name,'Conceptual panel reserved'))
                    if name in PANELS:PANELS[name](p)
                    else:slot(p,{'fig1a':'Encounter and contract','fig1b':'Three mechanisms','fig3a':'Two copy geometries'}[name])
            save(fig,'fig'+number)
            text=captions()[int(number)]
            lines=[]
            for paragraph in text.split('\n\n'):
                lines+=textwrap.wrap(paragraph,width=135)+['']
            height=max(40,9+len(lines)*3.2)
            legend=plt.figure(figsize=(180*fs.MM,height*fs.MM))
            ax=legend.add_axes([.025,.02,.95,.96]);ax.set_axis_off()
            ax.text(0,1,'\n'.join(lines).rstrip(),transform=ax.transAxes,va='top',fontsize=6.5,linespacing=1.35)
            issues=figcheck.check(legend)
            REPORTS['fig'+number+'_legend']=issues
            figcheck.print_report(issues,'fig'+number+'_legend')
            if issues:raise RuntimeError('Caption layout failed')
            legend.savefig(OUT/('fig'+number+'_legend.png'),dpi=600,bbox_inches=None,facecolor='white')
            plt.close(legend)
    finally:
        (OUT/'figcheck.json').write_text(json.dumps(REPORTS,indent=2)+'\n')
    if any(REPORTS.values()):raise RuntimeError('figcheck must be clean before export')


if __name__=='__main__':main()

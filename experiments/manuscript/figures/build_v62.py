"""Coordinator-directed v6.2 data figures; every coordinate reads generated CSV.

Panels show; separate legends define units, protocols, symbols and exclusions.
180-mm vector pages, editable SVG text, 600-dpi PDF-rendered PNG; figcheck
must pass for every main panel, composition, legend and Extended Data build.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
import figstyle as fs
import figcheck
from matplotlib import pyplot as plt
from legends_v62 import legends,extended_legends

OUT=HERE/'out/v6'
D=OUT/'data'
TEAL,RED,PURPLE='#1E8A8A','#E8431F','#7B3294'
INK,GREY,LIGHT,PEACH='#222222','#8A95A1','#E3E5E8','#FBE2DA'
KEYS=['pusher16','ret16','ldir32','genome20_s6003']
NAMES={'pusher16':'Pusher','ret16':'Return R','ldir32':'Block-copy R','genome20_s6003':'Transmitter'}
STYLES={'mix50':'-','R_into_T':'--','T_into_R':':'}
MARKERS={'mix50':'o','R_into_T':'s','T_into_R':'^'}
REPORTS={}


def table(name):
    return pd.read_csv(D/(name+'.csv'),low_memory=False)


class Panel:
    def __init__(self,fig,rect,letter):
        self.fig,self.rect=fig,rect
        self.canvas=self.ax(0,0,1,1)
        self.canvas.set_axis_off()
        fig.__dict__.setdefault('_v6_panels',[]).append(self)
        self.text(.015,.985,letter,size=8,bold=True)

    def ax(self,x,y,w,h):
        a,b,c,d=self.rect
        return self.fig.add_axes([a+x*c,b+y*d,w*c,h*d])

    def text(self,x,y,text,size=6.5,bold=False,**kw):
        return self.canvas.text(x,y,text,transform=self.canvas.transAxes,
                                va='top',color=INK,fontsize=size,
                                fontweight='bold' if bold else 'normal',**kw)


def axis(ax,xlabel='',ylabel='',xticks=None,yticks=None):
    fs.tidy(ax,xlabel or None,ylabel or None)
    if xticks is not None:ax.set_xticks(xticks)
    if yticks is not None:ax.set_yticks(yticks)
    ax.tick_params(length=2,pad=2)


def empty(p):
    p.canvas.add_patch(Rectangle((.065,.12),.90,.79,transform=p.canvas.transAxes,
                                 facecolor='none',edgecolor=LIGHT,lw=.6))


def fig1c(p):
    t=table('world_trajectory')
    ax=p.ax(.14,.20,.82,.69)
    ax.plot(t.step/1000,t.dominant_neighbourhood,color=TEAL,lw=1.1)
    ax.plot(t.step/1000,t.zero_fraction,color=GREY,lw=1.1)
    ax.set(xlim=(0,137),ylim=(-.035,1.08))
    axis(ax,'Step (thousands)','Fraction',[0,50,100],[0,.5,1])
    ax.text(105,t.dominant_neighbourhood.iloc[-1],'Modal match',color=TEAL,fontsize=6,va='center')
    ax.text(105,t.zero_fraction.iloc[-1]+.045,'Zero bytes',color=GREY,fontsize=6,va='center')


def fig1d(p):
    d=table('validation_fractions')
    ax=p.ax(.34,.20,.62,.69)
    y=np.arange(len(d))
    ax.barh(y,d.fraction,color=GREY,height=.42)
    ax.barh(y,1-d.fraction,left=d.fraction,color=RED,height=.42)
    ax.set(xlim=(0,1.04),ylim=(len(d)-.4,-.6))
    axis(ax,'Fraction preserved','',[0,.5,1],y)
    ax.set_yticklabels(d.level)


def fig2a(p):
    b=table('branching').query("arm=='U' and key in @KEYS")
    for j,key in enumerate(KEYS):
        x=.10+(j%2)*.48;y=.57 if j<2 else .18
        ax=p.ax(x,y,.36,.27)
        for cohort in ('Historical','C1'):
            q=b[(b.key==key)&(b.cohort==cohort)].sort_values('gen')
            for field,color in [('ctrl_alive',INK),('mut_alive',RED),('retained',TEAL)]:
                ax.plot(q.gen,q[field],color=color,lw=.9 if cohort=='C1' else .7,
                        alpha=1 if cohort=='C1' else .28,ls='-' if cohort=='C1' else 'none',
                        marker=None if cohort=='C1' else 'o',ms=2.7,
                        mfc='white',mec=color,markevery=[1,3,7],zorder=3 if cohort=='Historical' else 2)
        p.text(x,y+.325,NAMES[key],size=6.2)
        ax.set(xlim=(.7,13 if j==0 else 8.4),ylim=(-.06,1.08))
        axis(ax,'Generation' if j>=2 else '', 'Fraction of starts' if j%2==0 else '',[1,4,8],[0,1])
        if j<2:ax.tick_params(axis='x',labelbottom=False)
        if j==0:
            q=b[(b.key==key)&(b.cohort=='C1')&(b.gen==8)].iloc[0]
            for field,label,color in [('ctrl_alive','controls',INK),('mut_alive','mutants alive',RED),('retained','carrying',TEAL)]:
                ax.text(8.5,max(.025,q[field]),label,fontsize=5.5,color=color,va='center')


def fig2b(p):
    d,s=table('retention_maps'),table('retention_sites')
    cmap=mcolors.ListedColormap([LIGHT,RED,TEAL,'#56616F'])
    for j,key in enumerate(KEYS):
        q=d[d.key==key];length=int(q.pos.max()+1)
        image=np.full((256,length),3)
        for row in q.itertuples():image[row.val,row.pos]=row.outcome
        x=.08+j*.235
        ax=p.ax(x,.32,.175,.55)
        ax.imshow(image,origin='lower',aspect='auto',interpolation='nearest',cmap=cmap,
                  vmin=0,vmax=3,extent=(-.5,length-.5,-.5,255.5),rasterized=True)
        ax.set(xlim=(-.5,length-.5),ylim=(-.5,255.5))
        axis(ax,NAMES[key]+'\nPosition','Alternative byte' if j==0 else '',[0,length-1],[0,255] if j==0 else [])
        core=s[(s.key==key)&s.executed.astype(bool)].pos.to_numpy()
        # Contiguous core runs, drawn below the axis in the x-axis transform.
        for run in np.split(core,np.where(np.diff(core)>1)[0]+1):
            if len(run):
                ax.plot([run[0]-.35,run[0]-.35,run[-1]+.35,run[-1]+.35],
                        [-.006,-.018,-.018,-.006],color=INK,lw=.6,
                        transform=ax.get_xaxis_transform(),clip_on=False)


def fig2c(p):
    t=table('erasure_ordered');m=json.loads((D/'erasure_manifest.json').read_text())
    write=t[t.kind=='write']
    ax=p.ax(.15,.21,.80,.68)
    for encounter,offset in [(1,0),(2,128)]:
        q=write[write.encounter==encounter]
        ax.plot(q.step+offset,q.address,'.',color=GREY,ms=1.6,ls='none')
    erasure=write[(write.encounter==1)&(write.prior_source==m['position'])].iloc[0]
    ax.plot(erasure.step,erasure.address,'o',mfc='none',mec=RED,ms=5,mew=1)
    ax.axvline(128,color=LIGHT,lw=.7)
    ax.axhline(32,color=LIGHT,lw=.5)
    ax.text(23,14,'mutation\noverwritten',color=RED,fontsize=6,va='center')
    ax.annotate('',xy=(erasure.step+2,erasure.address+1),xytext=(21,13),
                arrowprops=dict(arrowstyle='-',color=RED,lw=.6))
    ax.set(xlim=(-4,260),ylim=(-2,66))
    axis(ax,'Execution step','Write address',[0,128,256],[0,32,64])


def fig2d(p):
    d=table('parent_environment')
    parents=list(d.parent.unique());envs=list(d.environment.unique())
    names=['Pusher','Return R','Block-copy R','Block-copy T','Lethal closer','Self-initialising T']
    envnames=['Random\nbenign','Random\nlethal','Closed\nsoup','Lethal\nsoup','Open\nsoup','Random\nregisters']
    ax=p.ax(.30,.28,.66,.60)
    maximum=d.sites_g8.max()
    for i,parent in enumerate(parents):
        for j,env in enumerate(envs):
            row=d[(d.parent==parent)&(d.environment==env)].iloc[0]
            extinct=row.control_alive_n==0
            shade=.22+.78*row.sites_g8/maximum if row.sites_g8>0 else 0
            fc='none' if extinct else PEACH if row.sites_g8==0 else mcolors.to_hex(
                (1-shade)*np.array(mcolors.to_rgb('white'))+shade*np.array(mcolors.to_rgb(TEAL)))
            ax.add_patch(Rectangle((j-.44,i-.44),.88,.88,fc=fc,ec=LIGHT if extinct else 'white',lw=.45))
            if extinct:ax.text(j,i,'×',color=GREY,fontsize=8,ha='center',va='center')
    ax.set(xlim=(-.5,5.5),ylim=(5.5,-.5))
    axis(ax,'','',range(6),range(6))
    ax.set_xticklabels(envnames,fontsize=5.5)
    ax.set_yticklabels(names,fontsize=6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)


def fig3b(p):
    pop=table('law_population');c1=table('law_z80_c1');c2=table('law_6502_c2');controls=table('law_6502_controls')
    ax=p.ax(.17,.20,.79,.69)
    ax.plot([-5,65],[-5,65],color=LIGHT,lw=.75,zorder=0)
    # Draw large discs first so a smaller population coordinate stays visible
    # inside a larger one. The same area scale is used for the three-size key.
    pop=pop.sort_values('cells',ascending=False)
    population_area=.20
    population_colour='#CBD0D5'
    ax.scatter(pop.predicted,pop.n_unexec,s=pop.cells*population_area,
               facecolors=population_colour,edgecolors='#9AA5AE',lw=.20,zorder=1)
    # Equal data-coordinate translations preserve y - x at either panel size.
    # Screen-space (point) translations do not follow the unequal axis scales.
    def offset(delta):
        return Affine2D().translate(delta,delta)+ax.transData
    ax.scatter(controls['d − 6'],controls['payload carried (sites ≥ 6)'],s=16,facecolors='none',edgecolors=INK,marker='s',lw=.6,zorder=3,transform=offset(1))
    ax.scatter(c1.predicted_free,c1.measured_free,s=9,color=TEAL,marker='o',edgecolors='white',lw=.3,zorder=4,transform=offset(-1))
    ax.scatter(c2.predicted_free,c2.measured_free,s=9,color=TEAL,marker='^',edgecolors='white',lw=.3,zorder=5,transform=offset(2))
    # Highlight real score deficits without enlarging their count-scaled discs.
    deficits=pop[pop.predicted!=pop.n_unexec]
    ax.scatter(deficits.predicted,deficits.n_unexec,s=9,facecolors='none',
               edgecolors='#9AA5AE',lw=.6,zorder=2)
    ax.set(xlim=(-5,65),ylim=(-5,65))
    axis(ax,'Predicted free payload (positions)','Measured capacity / sampled\nretention score (positions)',[0,20,40,60],[0,20,40,60])
    handles=[Line2D([],[],marker='o',ms=4,mfc=population_colour,mec='#9AA5AE',mew=.2,ls='none',label='Z80 population'),
             Line2D([],[],marker='o',ms=3,color=TEAL,ls='none',label='Z80 C1'),
             Line2D([],[],marker='^',ms=3,color=TEAL,ls='none',label='6502 C2'),
             Line2D([],[],marker='s',ms=3,mfc='none',mec=INK,ls='none',label='6502 controls'),
             Line2D([],[],marker='D',ms=3,mfc='none',mec=GREY,ls='none',label='8086')]
    ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.02,.97),frameon=False,
              fontsize=5.7,labelspacing=.30,handletextpad=.5,borderpad=0,handlelength=1)
    ax.scatter([31,41,54],[9]*3,s=np.array([1,100,2000])*population_area,
               facecolors=population_colour,edgecolors='#9AA5AE',lw=.20,zorder=2)
    for x,label in zip([31,41,54],['1','100','2,000 cells']):
        ax.text(x,1,label,fontsize=5.5,ha='center',va='center')


def fig3c(p):
    d=table('lineage_strips')
    ax=p.ax(.17,.26,.79,.59)
    for y,key in [(1,'ldir32_v04'),(0,'ldir32_v60')]:
        q=d[d.edit_id==key]
        for row in q.itertuples():
            color={'core':GREY,'erased':RED,'transmitted':TEAL}[row.state]
            ax.add_patch(Rectangle((row.position-.47,y-.20),.94,.4,fc=color,ec='white',lw=.15))
    ax.set(xlim=(-.6,31.6),ylim=(-.55,1.55))
    axis(ax,'Offspring position','',[0,4,16,31],[1,0])
    ax.set_yticklabels(['d = 4','d = 32'])
    ax.tick_params(axis='y',length=0)
    ax.spines['left'].set_visible(False)
    original=int(d.original_byte.iloc[0]);edited=int(d[d.displacement==32].edited_byte.iloc[0])
    ax.annotate('',xy=(.5,.22),xytext=(.5,.78),arrowprops=dict(arrowstyle='->',lw=.65,color=INK))
    ax.text(2.3,.50,f'{original:02X} → {edited:02X}',fontsize=6.5,va='center')


def fig3d(p):
    d=table('dial_coordinates')
    eligible=d.culture_replicator & d.prop8_eligible
    formal=d[eligible]
    assert set(d.value)==set(range(256)) and len(d)==256
    assert formal.effective_d.between(4,32).all() and (32%formal.effective_d==0).all()
    assert (formal.predicted_payload==formal.effective_d-4).all()
    assert formal.predicted_payload.between(0,28).all()
    assert d.loc[~d.prop8_eligible,'predicted_payload'].isna().all()
    cmap=mcolors.LinearSegmentedColormap.from_list('free_payload',['#A7D5D5',TEAL])
    norm=mcolors.Normalize(vmin=0,vmax=28)
    # Keep the map and its colourbar square/registered at both export sizes.
    height=.68
    width=height*(p.rect[3]*p.fig.get_figheight())/(p.rect[2]*p.fig.get_figwidth())
    x=(1-width-.19)/2
    ax=p.ax(x,.22,width,height)
    for row in d.itertuples():
        if not row.culture_replicator:
            fill,edge,lw='#F1F2F3','white',.25
        elif row.prop8_eligible:
            fill,edge,lw=cmap(norm(row.predicted_payload)),'white',.25
        else:
            fill,edge,lw='white',TEAL,.40
        ax.add_patch(Rectangle((row.value%16-.45,row.value//16-.45),.9,.9,
                               facecolor=fill,edgecolor=edge,lw=lw))
    for value in (0x04,0x60):
        ax.add_patch(Rectangle((value%16-.45,value//16-.45),.9,.9,
                               facecolor='none',edgecolor='black',lw=.85))
    ax.set(xlim=(-.5,15.5),ylim=(15.5,-.5),aspect='equal')
    axis(ax,'Low nibble','High nibble',range(16),range(16))
    ax.set_xticklabels(list('0123456789ABCDEF'))
    ax.set_yticklabels(list('0123456789ABCDEF'))
    ax.tick_params(length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    cax=p.ax(x+width+.045,.31,.022,.48)
    colourbar=p.fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=[0,14,28])
    colourbar.set_label('free payload (positions)',fontsize=6.5,labelpad=3)
    colourbar.ax.tick_params(length=2,pad=2)
    colourbar.outline.set_visible(False)


def fig4a(p):
    d=table('competition_medians')
    ax=p.ax(.14,.20,.82,.69)
    for cohort,alpha,lw in [('Historical',.20,.65),('C1',1,1.1)]:
        for tar,color in [('benign',TEAL),('lethal',PURPLE)]:
            for start,ls in STYLES.items():
                q=d[(d.cohort==cohort)&(d.tar==tar)&(d.start==start)].sort_values('step')
                if len(q):ax.plot(q.step/1000,q.T_share,color=color,ls=ls,alpha=alpha,lw=lw)
    ax.set(xlim=(-4,354),ylim=(-.035,1.08))
    axis(ax,'Step (thousands)','T / (R + T)',[0,150,300],[0,.5,1])
    for tar,color in [('benign',TEAL),('lethal',PURPLE)]:
        q=d[(d.cohort=='C1')&(d.tar==tar)&(d.start=='mix50')]
        if q.empty:q=d[(d.cohort=='Historical')&(d.tar==tar)&(d.start=='mix50')]
        ax.text(310,q.sort_values('step').T_share.iloc[-1],tar,color=color,fontsize=6.5,va='center')


def endpoint_axis(ax,measure):
    d=table('endpoint_positions').query('measure==@measure')
    for i,(cohort,tar) in enumerate([('Historical','benign'),('C1','benign'),('Historical','lethal'),('C1','lethal')]):
        q=d[(d.cohort==cohort)&(d.tar==tar)]
        color=TEAL if tar=='benign' else PURPLE
        for start,marker in MARKERS.items():
            r=q[q.start==start]
            ax.scatter(i+r.offset,r[measure],s=10,marker=marker,facecolors=color if cohort=='C1' else 'white',
                       edgecolors=color,lw=.6,zorder=3)
    ax.set_xlim(-.45,3.45)
    if measure!='T_distinct_tails_end':ax.set_ylim(-.035,1.08)
    else:ax.set_ylim(-d[measure].max()*.04,d[measure].max()*1.08)
    axis(ax,'',{'T_share_end':'Final T / (R + T)','core_share_end':'Core-bearing fraction',
                'T_distinct_tails_end':'Distinct transmitter tails'}[measure],None,[0,.5,1] if measure!='T_distinct_tails_end' else None)
    ax.set_xticks([.5,2.5],['Benign','Lethal'])


def fig4b(p):
    ax=p.ax(.16,.22,.79,.65)
    endpoint_axis(ax,'T_share_end')


def lethality_axis(ax,measure):
    d=table('lethality_summary')
    probs=sorted(d.p.unique())
    for cohort,color,ls,fill,size in [('Historical',GREY,'--','white',4),('C1','black','-','black',2.5)]:
        q=d[d.cohort==cohort].sort_values('p')
        ax.plot(range(len(q)),q[measure],color=color,ls=ls,lw=.85,marker='o',
                ms=size,mfc=fill,mec=color,mew=.65)
    ax.set_xlim(-.25,5.25)
    ax.set_ylim((-.45,10.65) if measure!='confinement_km' else (-10,300))
    axis(ax,'',{'open_first':'Open-first worlds','T_majority':'Transmitter-majority worlds',
                'confinement_km':'Confinement median (k steps)'}[measure],None,[0,5,10] if measure!='confinement_km' else [0,150,300])
    ax.set_xticks(range(6),['0','.01','.03','.1','.3','1'])
    if measure=='confinement_km':ax.set_xlabel('Probability that executing a zero halts')


def fig4c(p):
    for j,measure in enumerate(['open_first','T_majority']):
        ax=p.ax(.12+j*.49,.28,.34,.59)
        lethality_axis(ax,measure)
    p.text(.53,.11,'Probability that executing a zero halts',ha='center',size=6.5)


def ed_erasure(p):
    t=table('erasure_ordered')
    ax=p.ax(.16,.22,.79,.65)
    for encounter,offset in [(1,0),(2,128)]:
        q=t[t.encounter==encounter]
        ax.step(np.r_[offset,q.step.to_numpy()+offset],np.r_[1 if encounter==1 else 0,q.marker_sources_remaining],
                where='post',color=RED,lw=1)
    ax.set(xlim=(-4,260),ylim=(-.08,1.12))
    axis(ax,'Execution step','Marker-derived bytes',[0,128,256],[0,1])


def ed_residual(p):
    d=table('z80_worlds')
    ax=p.ax(.16,.22,.79,.65)
    ax.axhline(0,color=LIGHT,lw=.6)
    ax.scatter(d.predicted,d.residual,s=12,color=GREY)
    ax.set(xlim=(-3,63),ylim=(-1.05,.2))
    axis(ax,'Predicted free payload (positions)','World mean score − prediction',[0,20,40,60],[-1,0])


def ed_law_detail(p):
    # Moved low-capacity magnification; no coordinate jitter.
    ax=p.ax(.16,.22,.79,.65)
    ax.plot([0,10],[0,10],color=LIGHT,lw=.6)
    for name,x,y,marker,color,fill,size in [
        ('law_6502_controls','d − 6','payload carried (sites ≥ 6)','s',INK,'none',30),
        ('law_population','predicted','n_unexec','o',GREY,'none',45),
        ('law_z80_c1','predicted_free','measured_free','o',TEAL,TEAL,12),
        ('law_6502_c2','predicted_free','measured_free','^',TEAL,TEAL,12)]:
        d=table(name).query(f'`{x}`<=10')
        ax.scatter(d[x],d[y],s=size,marker=marker,facecolors=fill,edgecolors=color,lw=.6)
    ax.set(xlim=(-.7,10.7),ylim=(-.7,10.7))
    axis(ax,'Predicted free payload (positions)','Capacity / retention score (positions)',[0,5,10],[0,5,10])


def ed_capacity(p):
    e=table('capacity_edits');r=table('capacity_direct').query("direction=='reciprocal'")
    ax=p.ax(.16,.22,.79,.65)
    ax.plot([0,60],[0,60],color=LIGHT,lw=.6)
    ax.scatter(e.predicted_free,e.measured_free,s=16,color=TEAL)
    ax.scatter(r.predicted_free,r.measured_free,s=35,facecolors='none',edgecolors=GREY,marker='s',lw=.6)
    ax.set(xlim=(-3,63),ylim=(-3,63))
    axis(ax,'Predicted free payload (positions)','Exact capacity (positions)',[0,20,40,60],[0,20,40,60])


def ed_reference(p):
    b=table('branching').query("cohort=='Historical' and arm=='U' and gen==8 and key in ['pusher16','ret16']")
    ax=p.ax(.20,.22,.74,.65)
    for i,(field,color,label) in enumerate([('ctrl_alive',GREY,'Controls'),('retained',TEAL,'Carrying')]):
        ax.plot(np.arange(len(b))+(i-.5)*.1,b[field],ls='none',marker='o',ms=4,color=color)
    ax.set(xlim=(-.4,1.7),ylim=(-.05,1.1))
    axis(ax,'','Fraction',[0,1],[0,.5,1]);ax.set_xticklabels([NAMES[k] for k in b.key])
    ax.text(1.18,1,'Controls',color=GREY,fontsize=6,va='center')
    ax.text(1.18,.04,'Carrying',color=TEAL,fontsize=6,va='center')


def ed_entry(p):
    d=table('reference_traces')
    ax=p.ax(.20,.22,.74,.65)
    ax.scatter(range(len(d)),d.entered_n/d.encounters,s=20,color=GREY)
    ax.set(xlim=(-.4,len(d)-.6),ylim=(-.05,1.1))
    axis(ax,'','Partner-entry fraction',range(len(d)),[0,.5,1]);ax.set_xticklabels([NAMES[k] for k in d.key])


def ed_inflow(p):
    d=table('reference_traces')
    ax=p.ax(.20,.22,.74,.65)
    ax.scatter(range(len(d)),d.H_bits.clip(lower=0),s=20,color=GREY)
    ax.set(xlim=(-.4,len(d)-.6),ylim=(-.1,max(1,d.H_bits.max())*1.15))
    axis(ax,'','Offspring entropy (bits)',range(len(d)));ax.set_xticklabels([NAMES[k] for k in d.key])


PANELS={'fig1c':fig1c,'fig1d':fig1d,'fig2a':fig2a,'fig2b':fig2b,'fig2c':fig2c,'fig2d':fig2d,
        'fig3b':fig3b,'fig3c':fig3c,'fig3d':fig3d,'fig4a':fig4a,'fig4b':fig4b,'fig4c':fig4c}


def save(fig,name):
    report=figcheck.check(fig)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for panel in getattr(fig,'_v6_panels',[]):
        box=panel.canvas.get_window_extent(renderer)
        for text in panel.canvas.texts:
            b=text.get_window_extent(renderer)
            if b.x0<box.x0-1 or b.x1>box.x1+1 or b.y0<box.y0-1 or b.y1>box.y1+1:
                report.append('Text outside panel: '+text.get_text())
    for ax in fig.axes:
        if not ax.axison:continue
        for collection in ax.collections:
            offsets=np.asarray(collection.get_offsets())
            if len(offsets) and offsets.shape[1]==2:
                x0,x1=ax.get_xlim();y0,y1=ax.get_ylim()
                if ((offsets[:,0]<x0)|(offsets[:,0]>x1)|(offsets[:,1]<y0)|(offsets[:,1]>y1)).any():
                    report.append('Scatter data outside limits')
    REPORTS[name]=report
    figcheck.print_report(report,name)
    if not report:
        for ext in ('pdf','svg'):
            fig.savefig(OUT/(name+'.'+ext),dpi=600,bbox_inches=None,facecolor='white',metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None})
            if ext=='svg':
                path=OUT/(name+'.svg')
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        subprocess.run(['pdftoppm','-r','600','-png','-singlefile',str(OUT/(name+'.pdf')),str(OUT/name)],
                       check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    plt.close(fig)


def composed(number):
    height={1:134,2:154,3:154,4:154}[number]
    fig=plt.figure(figsize=(180*fs.MM,height*fs.MM))
    if number==4:
        # A long main trajectory axis, then a world strip and paired dial axes.
        rects=[(.01,.505,.98,.49),(.01,.005,.40,.49),(.42,.005,.57,.49)]
        for letter,rect in zip('abc',rects):
            panel=Panel(fig,rect,letter);PANELS['fig4'+letter](panel)
    else:
        for j,letter in enumerate('abcd'):
            name='fig'+str(number)+letter
            panel=Panel(fig,(.005+(j%2)*.5,.5 if j<2 else 0,.49,.49),letter)
            PANELS.get(name,empty)(panel)
    save(fig,'fig'+str(number))


def extended(selected=None):
    groups={
        'ed_v6_reference':[ed_reference,ed_entry,ed_inflow],
        'ed_v6_erasure':[ed_erasure],
        'ed_v6_law':[ed_residual,ed_law_detail,ed_capacity],
        'ed_v6_competition':[lambda p:endpoint_axis(p.ax(.17,.22,.79,.65),'core_share_end'),
                             lambda p:endpoint_axis(p.ax(.17,.22,.79,.65),'T_distinct_tails_end'),
                             ed_core_trajectory],
        'ed_v6_confinement':[lambda p:lethality_axis(p.ax(.16,.22,.79,.65),'confinement_km')],
    }
    for name,functions in groups.items():
        if selected and name not in selected:continue
        fig=plt.figure(figsize=(180*fs.MM,(72 if len(functions)==1 else 144)*fs.MM))
        for i,fn in enumerate(functions):
            rect=(.01,.01,.98,.98) if len(functions)==1 else (.005+(i%2)*.5,.5 if i<2 else 0,.49,.49)
            fn(Panel(fig,rect,'abcd'[i]))
        save(fig,name)
    # The text table is deliberately retained as a standalone ED table.
    if selected and 'ed_v6_joint_table' not in selected:return
    d=table('ed_joint_challenges')
    rows=[]
    for system in ['Z80 C1','6502 C2']:
        for kind in ['single','joint','erased']:
            q=d[(d.system==system)&(d.kind==kind)]
            rows.append([system,{'single':'All-value single','joint':'Joint assignments','erased':'Beyond protected prefix'}[kind],
                         f'{int(q.passed.sum()):,}',f'{int(q.cases.sum()):,}'])
    fig=plt.figure(figsize=(180*fs.MM,72*fs.MM));p=Panel(fig,(.02,.02,.96,.96),'a')
    ax=p.ax(.10,.20,.80,.62);ax.set_axis_off()
    grid=ax.table(cellText=rows,colLabels=['System','Challenge','Exact successes','Cases'],
                  cellLoc='left',colLoc='left',loc='center',colWidths=[.17,.39,.24,.20])
    grid.auto_set_font_size(False);grid.set_fontsize(6.5);grid.scale(1,1.75)
    for (i,j),cell in grid.get_celld().items():
        cell.set_edgecolor('white');cell.set_facecolor(LIGHT if i==0 else 'white')
    save(fig,'ed_v6_joint_table')


def ed_core_trajectory(p):
    d=table('competition_medians')
    ax=p.ax(.17,.22,.79,.65)
    for cohort,alpha in [('Historical',.2),('C1',1)]:
        for tar,color in [('benign',TEAL),('lethal',PURPLE)]:
            for start,ls in STYLES.items():
                q=d[(d.cohort==cohort)&(d.tar==tar)&(d.start==start)]
                ax.plot(q.step/1000,q.core_share,color=color,ls=ls,alpha=alpha,lw=.8)
    ax.set(xlim=(-4,310),ylim=(-.04,1.08))
    axis(ax,'Step (thousands)','Core-bearing fraction',[0,150,300],[0,.5,1])


def render_legend(number,text):
    name='fig'+str(number) if isinstance(number,int) else number
    lines=[]
    for paragraph in text.split('\n\n'):
        lines+=textwrap.wrap(paragraph,width=135)+['']
    height=max(45,10+len(lines)*3.25)
    fig=plt.figure(figsize=(180*fs.MM,height*fs.MM))
    ax=fig.add_axes([.025,.02,.95,.96]);ax.set_axis_off()
    ax.text(0,1,'\n'.join(lines).rstrip(),transform=ax.transAxes,va='top',fontsize=6.5,linespacing=1.35)
    REPORTS[name+'_legend']=figcheck.check(fig)
    figcheck.print_report(REPORTS[name+'_legend'],name+'_legend')
    if not REPORTS[name+'_legend']:
        fig.savefig(OUT/(name+'_legend.png'),dpi=600,bbox_inches=None,facecolor='white')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',default='1,2,3,4');ap.add_argument('--skip-ed',action='store_true')
    ap.add_argument('--only-ed',nargs='+',choices=['reference','erasure','law','joint_table','competition','confinement'])
    args=ap.parse_args()
    fs.setup();plt.rcParams.update({'font.sans-serif':['Arial','Liberation Sans','DejaVu Sans'],
        'font.size':6.5,'axes.labelsize':6.5,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
        'svg.hashsalt':'algocell-v62'})
    OUT.mkdir(parents=True,exist_ok=True)
    if (args.only!='1,2,3,4' or args.only_ed) and (OUT/'figcheck_v62.json').exists():REPORTS.update(json.loads((OUT/'figcheck_v62.json').read_text()))
    ls=legends()
    (OUT/'LEGENDS_V62.md').write_text('\n\n'.join([*ls.values(),*extended_legends().values()])+'\n')
    try:
        for number in ([] if args.only_ed else map(int,args.only.split(','))):
            for name,fn in PANELS.items():
                if int(name[3])==number:
                    fig=plt.figure(figsize=(180*fs.MM,72*fs.MM));fn(Panel(fig,(.025,.025,.95,.95),name[-1]));save(fig,name)
            composed(number);render_legend(number,ls[number].replace('**',''))
        if not args.skip_ed:
            selected=['ed_v6_'+name for name in args.only_ed] if args.only_ed else None
            extended(selected)
            for name,text in extended_legends().items():
                if not selected or name in selected:render_legend(name,text.replace('**',''))
    finally:
        (OUT/'figcheck_v62.json').write_text(json.dumps(REPORTS,indent=2)+'\n')
    if any(REPORTS.values()):raise RuntimeError('figcheck must be clean before critique')


if __name__=='__main__':main()

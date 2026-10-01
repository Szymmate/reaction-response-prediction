"""Render report figures directly from the supplied numerical evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
INK='#142D43';TEAL='#007F82';GRAY='#7F8D98';LIGHT='#E8EEF2';ORANGE='#B56226';RED='#A84442'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,'axes.labelcolor':INK,
 'xtick.color':GRAY,'ytick.color':INK,'axes.spines.top':False,'axes.spines.right':False,
 'axes.spines.left':False,'axes.spines.bottom':False,'axes.titleweight':'bold','figure.facecolor':'white'})

def save(fig,name):
    destination = ROOT/"archive/historical_research/figures" if name == "unlabeled_comparison" else OUT
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination/f'{name}.png',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def box(ax,x,y,w,h,title,body,fill=LIGHT):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.025',facecolor=fill,edgecolor='none'))
    ax.text(x+w/2,y+h*.7,title,ha='center',va='center',weight='bold',fontsize=12)
    ax.text(x+w/2,y+h*.31,body,ha='center',va='center',fontsize=10,linespacing=1.5)

def arrow(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,linewidth=1.4,color=GRAY))

fig,ax=plt.subplots(figsize=(10.5,3.1));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,.01,.58,.26,.36,'Employee profile','Questionnaire + OCEAN\nTenure, job family, region')
box(ax,.01,.07,.26,.36,'Upcoming situation','Topic, intervention, domain\nWorkload, team, support, season')
box(ax,.37,.30,.25,.4,'Fitted pipeline','Join + clean + transform\nLightGBM + temperature')
box(ax,.72,.30,.26,.4,'Five probabilities','One distribution per situation\nDefault-intervention review',fill='#DDF0ED')
arrow(ax,(.285,.76),(.355,.57));arrow(ax,(.285,.25),(.355,.42));arrow(ax,(.635,.5),(.705,.5))
save(fig,'pipeline')

coverage=pd.read_csv(ROOT/'results/selection_audit/coverage.csv')
d=coverage[coverage['slice'].eq('context_domain')].set_index('group').loc[['formal_training','on_the_job','external_event']]
fig,ax=plt.subplots(figsize=(10.5,2.5))
y=np.arange(3);values=d.coverage.to_numpy()*100
ax.barh(y,values,color=[TEAL,GRAY,GRAY],height=.52)
for i,v in enumerate(values):ax.text(v+.16,i,f'{v:.2f}%',va='center',weight='bold')
ax.set(yticks=y,yticklabels=['Formal training','On the job','External event'],xlim=(0,8.3),xlabel='Situations with recorded feedback (%)')
ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
save(fig,'feedback_coverage')

metrics=pd.read_csv(ROOT/'results/model/metrics.csv');d=metrics[metrics.variant.eq('calibrated')].set_index('arm')
order=['prior','context','demographics','questionnaire','personality','personality_no_age','logistic','wellness','anonymous_context']
labels=['Class-frequency baseline','Context only','Context + demographics','+ Questionnaire','+ Personality (age retained)','SELECTED: no age / wellness','Logistic: personality inputs','+ Wellness','+ 50 anonymous features']
fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),sharey=True,gridspec_kw={'width_ratios':[1.35,1]})
colors=[TEAL if a=='personality_no_age' else ORANGE if a in ['wellness','anonymous_context'] else GRAY for a in order]
for ax,key,title in zip(axes,['log_loss','brier'],['Log loss','Brier score']):
    values=d.loc[order,key]
    span=values.max()-values.min()
    limits=(values.min()-.10*span,values.max()+.28*span)
    for i,(a,c) in enumerate(zip(order,colors)):
        v=d.loc[a,key];ax.scatter(v,i,s=65 if a=='personality_no_age' else 40,color=c,zorder=3)
        ax.text(v+.003 if key=='log_loss' else v+.0016,i,f'{v:.3f}',va='center',fontsize=10,color=c,weight='bold' if a=='personality_no_age' else 'normal')
    ax.set(xlim=limits,title=title+' - lower is better',yticks=np.arange(len(order)))
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
axes[0].set_yticklabels(labels);axes[0].invert_yaxis();axes[1].tick_params(axis='y',left=False,labelleft=False)
fig.tight_layout(w_pad=2);save(fig,'model_comparison')

ssl=json.loads((ROOT/'archive/historical_research/results/unlabeled_comparison.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(10.5,3.1),sharey=True)
methods=['Denoising autoencoder','SCARF representation','Soft pseudo-labels']
for ax,key,title in zip(axes,['matched_labeled_control','calibrated_supervised'],['Versus matched labeled control','Versus calibrated supervised control']):
    for i,r in enumerate(ssl['results']):
        result=r['contrasts'][key]['log_loss'];v=result['delta'];lo,hi=result['ci95']
        c=TEAL if hi<0 else GRAY
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=c,capsize=4,markersize=6)
        ax.text(.98,.91-i*.28,f'{v:+.4f}',transform=ax.transAxes,ha='right',color=c,fontsize=10)
    bounds=np.array([r['contrasts'][key]['log_loss']['ci95'] for r in ssl['results']])
    left=min(0,bounds.min());right=max(0,bounds.max());pad=(right-left)*.15
    ax.axvline(0,color=GRAY,lw=1,ls='--');ax.set(title=title,xlabel='Change in log loss (negative is better)',xlim=(left-pad,right+pad),yticks=np.arange(3))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.grid(axis='x',alpha=.12)
axes[0].set_yticklabels(methods);axes[0].invert_yaxis();axes[1].tick_params(axis='y',left=False,labelleft=False)
fig.tight_layout(w_pad=2);save(fig,'unlabeled_comparison')

bins=pd.read_csv(ROOT/'results/model/calibration_bins.csv')
fig,axes=plt.subplots(2,3,figsize=(10.5,6.1),constrained_layout=True)
classes=['completed_effective','completed_ineffective','partial','dropped_out','declined','top_label']
labels=['Completed, effective','Completed, ineffective','Partial','Dropped out','Declined','Most likely outcome']
for ax,name,label in zip(axes.flat,classes,labels):
    d=bins[bins.class_name.eq(name)];supported=d[(d.events>=100)&(d.people>=50)];sparse=d[~d.index.isin(supported.index)]
    ax.plot([0,1],[0,1],color='#BFC9CF',ls='--',lw=1)
    ax.errorbar(supported.predicted,supported.observed,yerr=np.maximum(np.vstack([supported.observed-supported.lower,supported.upper-supported.observed]),0),fmt='o-',color=TEAL,capsize=3,lw=1.5,markersize=4)
    ax.scatter(sparse.predicted,sparse.observed,s=25,facecolors='none',edgecolors=GRAY)
    ax.set(xlim=(0,1),ylim=(0,1),title=label,xticks=[0,.5,1],yticks=[0,.5,1])
    ax.tick_params(labelsize=9)
for ax in axes[1]:ax.set_xlabel('Predicted probability',fontsize=10)
for ax in axes[:,0]:ax.set_ylabel('Observed fraction',fontsize=10)
save(fig,'calibration')

d=pd.read_csv(ROOT/'results/revision/slices.csv')
fig,axes=plt.subplots(1,2,figsize=(10.5,3.1),gridspec_kw={'width_ratios':[1,1]})
for ax,slic,groups,labels in [(axes[0],'age',['<30','30–39','40–49','50+','missing'],['Under 30','30-39','40-49','50+','Age missing']),
                             (axes[1],'context_domain',['formal_training','on_the_job','external_event'],['Formal training','On the job','External event'])]:
    tab=d[d['slice'].eq(slic)].set_index('group').loc[groups]
    for i,(_,r) in enumerate(tab.iterrows()):
        c=ORANGE if groups[i]=='50+' else TEAL
        ax.errorbar(r.log_loss,i,xerr=[[r.log_loss-r.log_loss_lower],[r.log_loss_upper-r.log_loss]],fmt='o',color=c,capsize=3)
        ax.text(r.log_loss_upper+.008,i,f'{r.log_loss:.3f}  (n={int(r.events):,})',va='center',fontsize=9)
    ax.set(yticks=np.arange(len(groups)),yticklabels=labels,
           xlim=(min(1.25,tab.log_loss_lower.min()-.02),max(1.75,tab.log_loss_upper.max()+.22)),xlabel='Log loss - lower is better')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
fig.tight_layout(w_pad=2);save(fig,'cohort_performance')

policy=json.loads((ROOT/'results/model/policy.json').read_text())
fig,ax=plt.subplots(figsize=(10.5,1.6));v=policy['observed_flag_rate']*100;capacity=policy['capacity_limit']*100
ax.barh([0],[v],height=.35,color=ORANGE);ax.axvline(capacity,color=INK,lw=1.5,ls='--')
ax.text(v+1,0,f'{v:.1f}% flagged',va='center',weight='bold',color=ORANGE)
ax.text(capacity+.8,.27,f'{capacity:.0f}% assumed capacity',fontsize=10)
ax.set(xlim=(0,100),ylim=(-.4,.5),yticks=[],xlabel='Share of labeled development situations (%)');ax.grid(axis='x',alpha=.1)
save(fig,'review_capacity')

fig,ax=plt.subplots(figsize=(10.5,1.9));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
centers=[.11,.36,.61,.87]
for i,(x,title,body) in enumerate(zip(centers,['Before event','Event occurs','Weeks 4-8','After week 8'],['Freeze forecast\nand feature timestamps','Keep default\nrecommendation','Collect feedback\nacross the cohort','Check maturity, coverage\nand release criteria'])):
    box(ax,x-.10,.30,.20,.55,title,body,fill='#DDF0ED' if i==0 else LIGHT)
    if i<3:arrow(ax,(x+.115,.57),(centers[i+1]-.115,.57))
save(fig,'outcome_timeline')
print(f'Wrote {len(list(OUT.glob("*.png")))} PNG figures to {OUT} and the historical comparison PNG to the archive')

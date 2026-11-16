from pathlib import Path
import sys,csv,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from freeze_additional_experiment import REV,ROOT,read,sha,write,csvout
SKILL=Path('Z:/CodexHome/.codex/skills/nature-skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
sys.path.insert(0,str(REV/'wp6_evaluation'))
from statistics6 import losses,seed_interval
DATA=HERE/'source_data';QA=HERE/'qa';OUT=HERE/'exports'
for p in (DATA,QA,OUT):p.mkdir(parents=True,exist_ok=True)
SOURCES={};CHECKS=[]
BLUE='#28628A';ORANGE='#B96C30';GRAY='#71777D'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'savefig.facecolor':'white'})
def rows(path,name=None):
    SOURCES[path.relative_to(ROOT).as_posix()]=sha(path)
    if name:shutil.copy2(path,DATA/name)
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def save(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=QA/f'{name}.alignment.json',strict=True,tolerance_pt=1.5,gutter_tolerance_pt=1.5)
    fig.savefig(OUT/f'{name}.pdf')
    fig.savefig(OUT/f'{name}.svg')
    fig.savefig(OUT/f'{name}.png',dpi=600)
    plt.close(fig)
def standard(ax):
    ax.grid(axis='y',color='#E5E8EA',lw=.5);ax.set_axisbelow(True)
def main():
    # Verify frozen result artifacts before rendering their summaries.
    for lock in (HERE.parent/'sampling_complete.json',HERE.parent/'protocol_lock.json'):
        for f,h in read(lock)['hashes'].items():assert sha(ROOT/f)==h,f
    sec=rows(REV/'wp6_evaluation/reports/secondary_comparisons.csv','all_secondary_comparisons.csv')
    gn=sorted([r for r in sec if r['left']=='G' and r['right']=='N'],key=lambda r:int(r['minute']));assert len(gn)==4
    csvout(DATA/'F1_G_N.csv',gn)
    fig,ax=plt.subplots(figsize=(7.2,3.4));fig.subplots_adjust(left=.22,right=.96,bottom=.23,top=.88)
    ax.axvspan(-.005,.005,color='#EAF1F5',zorder=0);ax.axvline(0,color=GRAY,lw=.7)
    for i,r in enumerate(gn):
        x=float(r['difference']);lo=float(r['L']);hi=float(r['U']);primary=r['minute']=='20'
        ax.errorbar(x,3-i,xerr=[[x-lo],[hi-x]],fmt='o' if primary else 's',color=BLUE if primary else GRAY,capsize=3,ms=6 if primary else 4,lw=1.2)
    ax.set_yticks(range(4),['40 min (n = 2,118)','30 min (n = 5,221)','20 min (primary; n = 7,541)','10 min (n = 7,740)'])
    ax.set_ylim(-.6,3.6);ax.set_xlim(-.006,.006);ax.set_xticks([-.005,0,.005]);ax.set_xlabel('G − N log loss (nats); negative favors G')
    ax.set_title('Same-information representation comparison',pad=12)
    fig.text(.5,.04,'Points: 10-seed means; bars: conditional 95% intervals; shaded band: ±0.005.',ha='center',fontsize=7)
    save(fig,'F1_representation_effects')
    scope=rows(REV/'wp6_evaluation/reports/scope_contrasts.csv','all_original_scope_contrasts.csv')
    new=rows(HERE.parent/'reports/sampling_draw_means.csv','F2_new_draw_means.csv')
    rows(HERE.parent/'reports/sampling_cells.csv','F2_all_new_seed_values.csv')
    old3=rows(HERE.parent/'reports/sampling_original_reference.csv','F2_original_three_seeds.csv')
    fig,ax=plt.subplots(figsize=(7.2,3.7));fig.subplots_adjust(left=.12,right=.96,bottom=.25,top=.79)
    plotted=[]
    for target,color,marker in [('all',BLUE,'o'),('listed',ORANGE,'s')]:
        orig=next(r for r in scope if r['minute']=='20' and r['model']=='N' and r['contrast']=='listed_minus_broad_matched' and r['target']==target+'_holdout')
        values=[float(orig['difference']),np.mean([float(r['original_difference']) for r in old3 if r['target']==target])]+[float(next(r for r in new if r['draw']==f'draw_{i:02d}' and r['target']==target)['mean_difference']) for i in range(1,6)]
        ax.plot(range(2,7),values[2:],color=color,lw=1,alpha=.75)
        ax.scatter(range(7),values,color=color,marker=marker,s=30,label='All target (n = 7,541)' if target=='all' else 'Listed target (n = 1,502)',zorder=3)
        for i,val in enumerate(values):plotted.append({'target':target,'position':i,'mean_difference':val,'seeds':10 if i==0 else 3})
    ax.axhline(0,color=GRAY,lw=.8);ax.axvline(1.5,color=GRAY,lw=.7,ls=':');standard(ax)
    ax.set_xticks(range(7),['Original\n10 seeds','Original\n3 seeds','Draw 1','Draw 2','Draw 3','Draw 4','Draw 5'])
    ax.set_ylabel('Listed − matched log loss (nats)');ax.set_ylim(-.004,.006)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2)
    fig.text(.5,.95,'Training-scope effects across matched samples',ha='center',fontsize=10)
    fig.text(.5,.055,'New draws: three-seed means; points are not confidence intervals. Negative favors listed training.',ha='center',fontsize=7)
    csvout(DATA/'F2_plotted_values.csv',plotted);save(fig,'F2_scope_sampling')
    mainrows=rows(REV/'wp6_evaluation/reports/main_results.csv','all_main_results.csv')
    common=rows(REV/'wp6_evaluation/reports/common40_trajectory.csv','all_common40_results.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),sharey=True);fig.subplots_adjust(left=.09,right=.97,bottom=.22,top=.79,wspace=.18)
    trajectory=[]
    for j,ax in enumerate(axes):
        for model,color,marker in [('G',BLUE,'o'),('N',ORANGE,'s')]:
            series=sorted([r for r in (mainrows if j==0 else common) if r['scope']=='broad' and r['model']==model],key=lambda r:int(r['minute']))
            x=[int(r['minute']) for r in series];y=np.array([float(r['log_loss']) for r in series])
            lows=np.array([float(r['conditional_seed_CI_low'] if j==0 else r['L']) for r in series]);highs=np.array([float(r['conditional_seed_CI_high'] if j==0 else r['U']) for r in series])
            ax.errorbar(x,y,yerr=[y-lows,highs-y],color=color,marker=marker,lw=1.2,capsize=2,ms=4,label=model)
            for r in series:trajectory.append({'population':'eligible_at_cutoff' if j==0 else 'common40','model':model,'minute':r['minute'],'log_loss':r['log_loss'],'n':r['n'] if j==0 else r['n_common40']})
        ax.set_xticks([10,20,30,40]);ax.set_xlabel('Prediction cutoff (min)');ax.set_ylim(.33,.74);standard(ax)
        ax.set_title(('a  Eligible at each cutoff','b  Same 2,118 matches')[j],loc='left',pad=13)
    axes[0].set_ylabel('Log loss (nats)');axes[1].legend(loc='upper right')
    fig.text(.5,.04,'10-seed means and conditional 95% intervals; a separately trained model is used at each cutoff.',ha='center',fontsize=7)
    csvout(DATA/'F3_plotted_values.csv',trajectory);save(fig,'F3_common_cohort')
    degree=rows(HERE.parent/'reports/graph_degree_distribution.csv','all_graph_degree_distribution.csv')
    rows(HERE.parent/'reports/graph_summary.csv','all_graph_summary.csv')
    fig,ax=plt.subplots(figsize=(7.2,3.5));fig.subplots_adjust(left=.12,right=.96,bottom=.23,top=.80)
    colors=['#9CABB4','#28628A','#B96C30','#555B60'];plotted=[]
    for t,color,marker in zip([10,20,30,40],colors,['o','s','^','D']):
        a=sorted([r for r in degree if r['cohort']=='evaluation' and int(r['minute'])==t],key=lambda r:int(r['offdiagonal_degree']))
        assert len(a)==10 and abs(sum(float(r['fraction']) for r in a)-1)<1e-12
        ax.plot(range(10),[100*float(r['fraction']) for r in a],marker=marker,ms=4,lw=1.2,color=color,label=f'{t} min');plotted+=a
    ax.set_xticks(range(10));ax.set_xlim(-.25,9.25);ax.set_ylim(bottom=0);ax.set_xlabel('Neighbors other than self (bidirectional support)');ax.set_ylabel('Nodes (%)');standard(ax)
    ax.legend(ncol=4,loc='lower left',bbox_to_anchor=(0,1.01));fig.text(.5,.95,'Observed message-passing neighborhoods',ha='center',fontsize=10)
    fig.text(.5,.04,'All nodes retained; percentages use 10 × eligible matches at each cutoff. Self-loops excluded.',ha='center',fontsize=7)
    csvout(DATA/'F4_plotted_degrees.csv',plotted);save(fig,'F4_graph_degrees')
    attr=rows(HERE.parent/'reports/attrition.csv','F5_attrition.csv')
    fig,ax=plt.subplots(figsize=(7.2,4.5));fig.subplots_adjust(left=.02,right=.98,bottom=.08,top=.93);ax.set_xlim(0,10);ax.set_ylim(0,10);ax.axis('off')
    def box(x,y,w,h,text,fill='#F0F4F6'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',linewidth=.7,edgecolor=GRAY,facecolor=fill));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(x1,y1,x2,y2):ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops={'arrowstyle':'->','color':GRAY,'lw':.9})
    box(3.2,8.6,3.6,1,'Frozen candidates\n8,132 matches')
    box(7.5,6.9,2.25,1.1,'Excluded: 382\nShared series IDs','#F6F2EE');arrow(5,8.5,5,7.6);arrow(6.8,8.8,8.5,8.05)
    box(3.2,6.4,3.6,1.1,'After series exclusions\n7,750 matches')
    for i,t in enumerate([10,20,30,40]):
        a=[r for r in attr if int(r['minute'])==t]
        ended=int(next(r['n'] for r in a if 'already ended' in r['reason']));missing=int(next(r['n'] for r in a if 'Missing gold' in r['reason']));n=int(next(r['n'] for r in a if r['stage']=='accepted'))
        assert 7750-ended-missing==n
        x=.18+i*2.5
        arrow(5,6.3,x+1.05,5.55)
        box(x,3.2,2.1,2.2,f'{t} min\n\nEnded: {ended:,}\nMissing snapshot: {missing}')
        arrow(x+1.05,3.1,x+1.05,2.55);box(x,1.25,2.1,1.15,f'Eligible\n{n:,} matches','#E7F0F5')
    fig.text(.5,.96,'Common-input eligibility at each prediction cutoff',ha='center',fontsize=10)
    fig.text(.5,.035,'Each cutoff starts from the same 7,750 matches; the four columns are not successive exclusions.',ha='center',fontsize=7)
    save(fig,'F5_eligibility_flow')
    category_tables()
    write(HERE/'source_manifest.json',{'hashes':SOURCES,'checks':CHECKS,'backend':'Python/matplotlib'})
    print('Five figures and category tables created.')

def category_tables():
    old=rows(REV/'wp6_evaluation/reports/category_results.csv','supplement_all_category_results.csv')
    meta=read(REV/'wp6_evaluation/holdout_inputs/metadata.json');meta={r['match_id']:r for r in meta}
    with np.load(REV/'wp6_evaluation/holdout_inputs/labels_20m.npz') as a:ids=a['match_id'];y=a['labels']
    lock=read(REV/'wp6_evaluation/evaluation_lock.json');ph=read(REV/'wp6_evaluation/predictions_complete.json')['hashes']
    names=['G','N','gold_xp_lr','full_feature_lr'];metrics={};long=[]
    tiers=np.array([meta[int(i)]['provider_tier'] for i in ids])
    for name in names:
        cell=next(c for c in lock['models'] if c['scope']=='broad' and c['minute']==20 and c['model']==name)
        ps=[]
        for seed in cell['seeds']:
            path=REV/f'wp6_evaluation/predictions/broad/20m/{name}/seed{seed}.npz';h=sha(path);assert ph[path.relative_to(ROOT).as_posix()]==h;SOURCES[path.relative_to(ROOT).as_posix()]=h
            with np.load(path) as a:assert np.array_equal(a['match_id'],ids);ps.append(a['probabilities'])
        loss=losses(y,np.array(ps));metrics[name]=loss
        for group in ['listed','excluded','unmatched','professional','premium']:
            mask=np.isin(tiers,['professional','premium']) if group=='listed' else tiers==group
            values=loss[:,mask].mean(1);summ=seed_interval(values)
            if int(mask.sum())<200 or min(int(y[mask].sum()),int(mask.sum()-y[mask].sum()))<20:summ['L']=summ['U']=None
            long.append({'group':group,'model':name,'n':int(mask.sum()),'radiant_wins':int(y[mask].sum()),'log_loss':float(values.mean()),'seeds':len(values),'L':summ['L'],'U':summ['U']})
            if group in ('professional','premium','excluded','unmatched'):
                original=next(r for r in old if r['scope']=='broad' and r['minute']=='20' and r['model']==name and r['axis']=='provider_tier' and r['group']==group)
                assert abs(float(original['log_loss'])-values.mean())<1e-12
    csvout(DATA/'category_recomputed_20m.csv',long)
    main=[];latex=['\\begin{tabular}{lrrrrrr}','\\hline','Category & $n$ & R wins & G & N & Gold/XP LR & Full LR \\\\','\\hline']
    for group in ['listed','excluded','unmatched']:
        rs={r['model']:r for r in long if r['group']==group};n=rs['G']['n'];rw=rs['G']['radiant_wins'];v={m:rs[m]['log_loss'] for m in names}
        main.append({'group':group,'n':n,'radiant_wins':rw,**v})
        latex.append(f"{group.capitalize()} & {n} & {rw} & "+' & '.join(f'{v[m]:.4f}' for m in names)+' \\\\')
    assert sum(r['n'] for r in main)==7541
    csvout(DATA/'category_main_20m.csv',main);latex+=['\\hline','\\end{tabular}']
    (HERE/'category_main_20m.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    CHECKS.extend(['16 original category point estimates replayed exactly','listed recomputed from 1502 individual predictions','main groups partition all 7541 matches','premium n=3 retained without interval'])

if __name__=='__main__':main()

"""Single complete scoring pass after fixed-model, label-free predictions exist."""
from pipeline_common import *
import numpy as np
from statistics6 import losses,seed_interval,paired,block_sensitivity,time_blocks,equal_group_seed_values
from evaluation import probability_metrics


def main():
    lock=HERE6/'run_plan.json';check_lock(lock);spec=read(lock)
    complete=read(HERE6/'predictions_complete.json');assert complete['run_plan_sha256']==sha(lock)
    for p,h in complete['hashes'].items():assert sha(ROOT6/p)==h
    contract=read(HERE6/'holdout_inputs/contract.json')
    assert complete['input_contract_sha256']==sha(HERE6/'holdout_inputs/contract.json')
    for p,h in contract['hashes'].items():assert sha(HERE6/'holdout_inputs'/p)==h
    metadata=read(HERE6/'holdout_inputs/metadata.json');meta={int(r['match_id']):r for r in metadata}
    # Generalization refers to the entire audited development inventory, including old inspected test/external matches.
    exclusions=read(CODE/'manifests/exclusions.json')
    known_accounts=set(exclusions['account_hashes']);known_teams=set(exclusions['team_hashes'])
    def valid(values):return {hashlib.sha256(str(x).encode()).hexdigest() for x in values if isinstance(x,int) and 0<x<4294967295}
    def overlap(row,field,seen,expected):
        ids=valid(row[field])
        return 'seen_in_audited_development' if ids&seen else 'unseen_all_known_in_audited_inventory' if len(ids)==expected else 'unknown_identity'
    loaded={};main_rows=[];seed_rows=[];group_rows=[];trajectory=[]
    metric_names=('accuracy','brier','balanced_accuracy','f1','roc_auc','ece_10_equal_width_bins')
    common40=None
    for cutoff in (10,20,30,40):
        with np.load(HERE6/'holdout_inputs'/f'labels_{cutoff}m.npz') as a:current=set(map(int,a['match_id']))
        common40=current if common40 is None else common40&current
    for cell in spec['models']:
        scope,t,name=cell['scope'],cell['minute'],cell['model']
        with np.load(HERE6/'holdout_inputs'/f'labels_{t}m.npz') as a:ids=a['match_id'].copy();y=a['labels'].copy()
        ps=[]
        for seed in cell['seeds']:
            p=HERE6/'predictions'/scope/f'{t}m'/name/f'seed{seed}.npz'
            with np.load(p) as a:
                assert np.array_equal(ids,a['match_id']);ps.append(a['probabilities'].copy())
        ps=np.array(ps);ll=losses(y,ps);loaded[scope,t,name]=(ids,y,ps,ll,cell['seeds'])
        per=[]
        for seed,p,l in zip(cell['seeds'],ps,ll):
            metrics=probability_metrics(y,p);row={'scope':scope,'minute':t,'model':name,'seed':seed,'n':len(y),'log_loss':float(l.mean())}
            row.update({k:metrics[k] for k in metric_names});per.append(row);seed_rows.append(row)
        summary=seed_interval(ll.mean(1));r={'scope':scope,'minute':t,'model':name,'n':len(y),'radiant_wins':int(y.sum()),
            'log_loss':summary['mean'],'seed_sd':summary['seed_sd'],'conditional_seed_CI_low':summary['L'],
            'conditional_seed_CI_high':summary['U'],'seed_count':len(ps),'Hodge_label':'observable-feature-subset adaptation' if name=='hodge_adapted' else ''}
        for k in metric_names:
            vals=[v[k] for v in per];r[k]=float(np.mean(vals)) if all(v is not None and np.isfinite(v) for v in vals) else None
        main_rows.append(r)
        axes={
          'provider_tier':[meta[int(i)]['provider_tier'] for i in ids],
          'month':[meta[int(i)]['period'] for i in ids],
          'patch':[str(meta[int(i)]['patch']) for i in ids],
          'player_overlap_development':[overlap(meta[int(i)],'account_ids',known_accounts,10) for i in ids],
          'team_overlap_development':[overlap(meta[int(i)],'team_ids',known_teams,2) for i in ids]}
        for axis,values in axes.items():
            levels=sorted(set(values))
            if axis=='provider_tier':levels=['premium','professional','excluded','unmatched']
            if axis.endswith('_development'):levels=['seen_in_audited_development','unseen_all_known_in_audited_inventory','unknown_identity']
            for level in levels:
                mask=np.array([x==level for x in values]);n=int(mask.sum());wins=int(y[mask].sum());eligible=n>=200 and min(wins,n-wins)>=20
                row={'scope':scope,'minute':t,'model':name,'axis':axis,'group':level,'n':n,'radiant_wins':wins,'interval_eligible':eligible}
                if n:
                    s=seed_interval(ll[:,mask].mean(1));row.update(log_loss=s['mean'],seed_sd=s['seed_sd'],conditional_seed_CI_low=s['L'] if eligible else None,conditional_seed_CI_high=s['U'] if eligible else None)
                    metrics=[probability_metrics(y[mask],p[mask]) for p in ps]
                    for k in metric_names:
                        vals=[m.get(k) for m in metrics];row[k]=float(np.mean(vals)) if all(v is not None and np.isfinite(v) for v in vals) else None
                group_rows.append(row)
        mask=np.array([int(i) in common40 for i in ids]);s=seed_interval(ll[:,mask].mean(1)) if mask.any() else {'mean':None,'L':None,'U':None}
        trajectory.append({'scope':scope,'minute':t,'model':name,'n_common40':int(mask.sum()),'log_loss':s['mean'],'L':s['L'],'U':s['U']})
    # Primary result is fixed broad-pool G-N at20m; every other comparison remains descriptive.
    g=loaded['broad',20,'G'];n=loaded['broad',20,'N'];assert np.array_equal(g[1],n[1])
    primary=paired(g[1],g[2],n[2],g[0],n[0],g[4],n[4]);primary.update(scope='broad',minute=20,primary=True)
    comparisons=[];sensitivities=[];scope_rows=[];group_contrasts=[]
    for minute in (10,20,30,40):
        g=loaded['broad',minute,'G'];n=loaded['broad',minute,'N'];delta=g[3]-n[3]
        for left in ('G','N'):
            for right in ('N','flat_mlp','gold_xp_lr','full_feature_lr','random_forest','xgboost','lightgbm','hodge_adapted','empirical_prior','constant_half'):
                if left==right:continue
                a=loaded['broad',minute,left];b=loaded['broad',minute,right]
                assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
                if len(b[4])>1:assert a[4]==b[4]
                s=seed_interval((a[3]-b[3]).mean(1))
                comparisons.append({'minute':minute,'left':left,'right':right,'n':len(a[0]),'difference':s['mean'],'L':s['L'],'U':s['U'],
                    'primary':minute==20 and left=='G' and right=='N','interval':'conditional paired-seed95pct; secondary unadjusted; deterministic baseline broadcast without fake training repeats'})
        axes={'provider_tier':[meta[int(i)]['provider_tier'] for i in g[0]],
              'player_overlap_development':[overlap(meta[int(i)],'account_ids',known_accounts,10) for i in g[0]],
              'team_overlap_development':[overlap(meta[int(i)],'team_ids',known_teams,2) for i in g[0]]}
        for axis,values in axes.items():
            vals=np.array(values);ref='professional' if axis=='provider_tier' else 'seen_in_audited_development';refmask=vals==ref
            for level in sorted(set(values)):
                mask=vals==level;yy=g[1][mask];eligible=len(yy)>=200 and min(int(yy.sum()),len(yy)-int(yy.sum()))>=20
                d=delta[:,mask].mean(1);s=seed_interval(d)
                group_contrasts.append({'minute':minute,'axis':axis,'group':level,'contrast':'G_minus_N','n':len(yy),'difference':s['mean'],'L':s['L'] if eligible else None,'U':s['U'] if eligible else None})
                if level!=ref and refmask.any():
                    ry=g[1][refmask];ok=eligible and len(ry)>=200 and min(int(ry.sum()),len(ry)-int(ry.sum()))>=20
                    s=seed_interval(d-delta[:,refmask].mean(1))
                    group_contrasts.append({'minute':minute,'axis':axis,'group':level+' minus '+ref,'contrast':'difference_of_G_N_differences','n':len(yy),'reference_n':len(ry),'difference':s['mean'],'L':s['L'] if ok else None,'U':s['U'] if ok else None})
        months=[meta[int(i)]['period'] for i in g[0]];em=seed_interval(equal_group_seed_values(delta,months))
        sensitivities.append({'minute':minute,'scheme':'equal_month_conditional_seed','point':em['mean'],'L':em['L'],'U':em['U'],'n':len(g[0])})
        values=delta.mean(0);times=[meta[int(i)]['start_time'] for i in g[0]]
        allids=sorted(r['match_id'] for r in metadata if r['eligible_minutes']);alltimes=[meta[int(i)]['start_time'] for i in allids]
        schemes=[('match_iid_reference_only',list(map(int,g[0])),allids)]
        for days in (7,14,28):schemes.append((f'UTC_{days}day_blocks',time_blocks(times,days),sorted(set(time_blocks(alltimes,days)))))
        schemes.append(('league_clusters',[meta[int(i)]['leagueid'] for i in g[0]],sorted({meta[int(i)]['leagueid'] for i in allids})))
        for title,groups,universe in schemes:
            s=block_sensitivity(values,groups,universe)
            sensitivities.append({'minute':minute,'scheme':title,'point':s['point'][0],'L':s['L'][0] if s['L'] else None,'U':s['U'][0] if s['U'] else None,
                'nonempty_blocks':s['nonempty_blocks'],'universe_blocks':s['universe_blocks'],'replicates':s['replicates'],'status':s['status'],
                'inference':'sampling sensitivity conditional on fitted models; cross-block participant dependence may persist'})
        for model in ('G','N','xgboost'):
            a=loaded['listed',minute,model]
            for right in ('broad_matched','broad'):
                b=loaded[right,minute,model];assert np.array_equal(a[0],b[0]) and a[4]==b[4]
                for target,mask in [('all_holdout',np.ones(len(a[0]),bool)),('listed_holdout',np.array([meta[int(i)]['provider_tier'] in ('premium','professional') for i in a[0]]))]:
                    size=int(mask.sum());wins=int(a[1][mask].sum());eligible=size>=200 and min(wins,size-wins)>=20
                    row={'minute':minute,'model':model,'contrast':'listed_minus_'+right,'target':target,'n':size}
                    if size:
                        s=seed_interval((a[3][:,mask]-b[3][:,mask]).mean(1));row.update(difference=s['mean'],L=s['L'] if eligible else None,U=s['U'] if eligible else None)
                    scope_rows.append(row)
    for filename,rows in [('main_results.csv',main_rows),('per_seed_results.csv',seed_rows),('secondary_comparisons.csv',comparisons),
                         ('sampling_sensitivities.csv',sensitivities),('scope_contrasts.csv',scope_rows),('category_results.csv',group_rows),
                         ('common40_trajectory.csv',trajectory),('group_G_N_contrasts.csv',group_contrasts)]:csvout(filename,rows)
    write(HERE6/'reports/primary_result.json',primary)
    write(HERE6/'reports/holdout_complete.json',{'created_unix':time.time(),'run_plan_sha256':sha(lock),
        'prediction_manifest_sha256':sha(HERE6/'predictions_complete.json'),'main_result_rows':len(main_rows),'all_prespecified_cells':len(spec['models']),
        'primary_decision':primary['decisions']['0.005'],'independence_label':'replay on frozen matches; not a new independent evaluation; conditional-seed primary inference'})
    print('Complete fixed-model holdout analysis; primary result saved, all prespecified rows retained',flush=True)
if __name__=='__main__':main()

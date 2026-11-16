"""Generate publication tables from sealed reports; never fit or rescore a model."""
from pathlib import Path
import csv, hashlib, json, shutil, math

HERE = Path(__file__).resolve().parent
REV = HERE.parent
ROOT = REV.parent
TABLES = HERE / 'manuscript/tables'
EVIDENCE = HERE / 'evidence'

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def esc(x):
    return str(x).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

def write(name, text):
    (TABLES / name).write_text(text, encoding='utf-8')

def num(x, precision=4):
    return '--' if x in ('', None) else f'{float(x):.{precision}f}'

def ci(row, low='L', high='U', precision=5):
    return '--' if row[low] == '' else f'[{num(row[low],precision)}, {num(row[high],precision)}]'

def longtable(name, caption, columns, headers, rows, label):
    text = r'\small' + '\n' + r'\begin{longtable}{' + columns + '}\n'
    text += r'\caption{' + caption + r'}\label{' + label + r'}\\' + '\n'
    header = r'\toprule ' + ' & '.join(headers) + r'\\\midrule' + '\n'
    text += header + r'\endfirsthead' + '\n' + header + r'\endhead' + '\n'
    text += '\n'.join(' & '.join(row) + r'\\' for row in rows)
    text += '\n' + r'\bottomrule\end{longtable}\normalsize' + '\n'
    write(name, text)

def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    report=REPORT
    manifest=[]
    rows=read_csv(report/'main_results.csv')
    assert len(rows)==68
    models=['G','N','flat_mlp','gold_xp_lr','full_feature_lr','random_forest','xgboost','lightgbm','hodge_adapted','empirical_prior','constant_half']
    names=dict(zip(models,['G: bidirectional graph','N: node-and-edge MLP','Flat MLP','Gold/XP LR','Full-feature LR','Random forest','XGBoost','LightGBM','Hodge subset adaptation','Training prior','Constant 0.5']))
    index={(r['scope'],int(r['minute']),r['model']):r for r in rows}
    primary=read_json(report/'primary_result.json')
    macros={'PrimaryDelta':num(primary['mean'],6),'PrimaryLow':num(primary['L'],6),
            'PrimaryHigh':num(primary['U'],6),'ProbabilityRatio':num(primary['geometric_true_class_probability_ratio_left_over_right'],6),
            'PrimaryN':f"{primary['n_matches']:,}",
            'GraphLoss':num(index['broad',20,'G']['log_loss'],6),
            'NodeLoss':num(index['broad',20,'N']['log_loss'],6)}
    write('result_macros.tex','% Generated from frozen WP6 results.\n'+'\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    mainrows=[]
    for model in models:
        r=index['broad',20,model]
        mainrows.append([names[model]]+[num(index['broad',t,model]['log_loss']) for t in (10,20,30,40)]+[num(float(r['accuracy'])*100,2),num(r['brier'])])
    write('main_results.tex',r'''\begin{table*}[t]
\caption{Unused-holdout results for every broad-training model. Log loss uses natural logarithms; lower values indicate better probability predictions. Accuracy and Brier score refer to the prespecified 20-minute endpoint. Entries average metrics over ten final seeds for stochastic models; LR and constant predictors use one deterministic fit. All models share the eligible matches at each cutoff. Full metrics and seed variation appear in the Supplementary Information and source tables.}
\label{tab:main}\centering\small
\begin{tabular}{lrrrrrr}\toprule
Model & LL, 10 min & LL, 20 min & LL, 30 min & LL, 40 min & Acc., 20 min (\%) & Brier, 20 min\\\midrule
'''+ '\n'.join(' & '.join(r)+r'\\' for r in mainrows)+ '\n'+r'\bottomrule\end{tabular}\end{table*}'+'\n')
    coverage=read_csv(CODE/'manifests/pool_coverage.csv')
    cov={(r['pool'],int(r['minute']),r['split']):r for r in coverage}
    lines=[]
    for t in (10,20,30,40):
        h=index['broad',t,'G']
        lines.append([str(t),cov['broad',t,'train']['n'],cov['listed',t,'train']['n'],cov['broad',t,'validation']['n'],h['n'],num(100*int(h['radiant_wins'])/int(h['n']),2)])
    write('cohorts.tex',r'''\begin{table}[t]\caption{Prediction cohorts. Listed and broad-matched training sets have equal sizes at every UTC month. Holdout Radiant win rates describe the common eligible target.}\label{tab:cohorts}\centering\scriptsize
\begin{tabular}{rrrrrr}\toprule
& \multicolumn{2}{c}{Training} & & \multicolumn{2}{c}{Holdout}\\
Min. & Broad & Listed/matched & Valid. & $n$ & R win (\%)\\\midrule
'''+ '\n'.join(' & '.join(r)+r'\\' for r in lines)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n')
    paired=[r for r in read_csv(report/'secondary_comparisons.csv') if r['left']=='G' and r['right']=='N']
    write('paired.tex',r'''\begin{table}[t]\caption{Broad-training G$-$N log-loss differences. Intervals quantify paired training-seed variation conditional on fixed data and selected parameters. Twenty minutes is primary; other cutoffs are descriptive.}\label{tab:paired}\centering\small
\begin{tabular}{rrr}\toprule Min. & Difference & 95\% seed interval\\\midrule
'''+ '\n'.join(f"{r['minute']} & {num(r['difference'],6)} & {ci(r,precision=6)}"+r'\\' for r in paired)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n')
    scope=read_csv(report/'scope_contrasts.csv')
    srows=[]
    for r in scope:
        if r['minute']=='20' and r['contrast']=='listed_minus_broad_matched':
            srows.append([names[r['model']].split(':')[0], 'All' if r['target']=='all_holdout' else 'Listed',r['n'],num(r['difference'],6),ci(r,precision=6)])
    write('scope20.tex',r'''\begin{table*}[t]\caption{Training-scope contrasts at 20 minutes with monthly sample counts matched: listed minus broad-matched log loss on identical target matches. Negative values favor listed training. Each training pool has 24,685 matches; each model is independently tuned under the same policy. Intervals are descriptive, unadjusted paired-seed 95\% intervals.}\label{tab:scope}\centering\small
\begin{tabular}{llrrr}\toprule Model & Target & Target $n$ & Difference & 95\% seed interval\\\midrule
'''+ '\n'.join(' & '.join(r)+r'\\' for r in srows)+'\n'+r'\bottomrule\end{tabular}\end{table*}'+'\n')
    cats=read_csv(report/'category_results.csv')
    cindex={(r['scope'],r['minute'],r['model'],r['axis'],r['group']):r for r in cats}
    crows=[]
    for group in ['professional','premium','excluded','unmatched']:
        r=cindex['broad','20','G','provider_tier',group]
        crows.append([group,r['n'],num(100*int(r['radiant_wins'])/int(r['n']),2)]+[num(cindex['broad','20',m,'provider_tier',group]['log_loss']) for m in ['G','N','xgboost']])
    write('category20.tex',r'''\begin{table}[t]\caption{Provider-category results at 20 minutes, broad training. Labels describe a frozen provider directory. The premium group is retained as a point estimate and has no interval.}\label{tab:category}\centering\scriptsize
\begin{tabular}{lrrrrr}\toprule Label & $n$ & R win (\%) & G LL & N LL & XGB LL\\\midrule
'''+ '\n'.join(' & '.join(r)+r'\\' for r in crows)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n')
    ordered=sorted(rows,key=lambda r:({'broad':0,'listed':1,'broad_matched':2}[r['scope']],int(r['minute']),models.index(r['model'])))
    short={'broad':'B','listed':'L','broad_matched':'M'}
    names_short={'G':'G','N':'N','flat_mlp':'Flat MLP','gold_xp_lr':'Gold/XP LR','full_feature_lr':'Full LR','random_forest':'RF','xgboost':'XGB','lightgbm':'LGBM','hodge_adapted':'Hodge subset','empirical_prior':'Prior','constant_half':'0.5'}
    longtable('s_all_results.tex','All 68 model--pool--cutoff cells. B: broad; L: listed; M: broad-matched. Seed SD is training variation; a dash denotes a deterministic predictor.','llrrrrrr', ['Pool/model','Min.','$n$','LL','Seed SD','Seeds',r'Acc. (\%)','Brier'],[[short[r['scope']]+'/'+names_short[r['model']],r['minute'],r['n'],num(r['log_loss'],6),num(r['seed_sd'],6),r['seed_count'],num(100*float(r['accuracy']),2),num(r['brier'],6)] for r in ordered],'tab:all')
    longtable('s_other_metrics.tex','Additional prespecified metrics for every cell. Balanced accuracy and F1 use a 0.5 threshold. ECE uses ten equal-width probability bins.','llrrrr', ['Pool/model','Min.','Balanced acc.','F1','ROC AUC','ECE'],[[short[r['scope']]+'/'+names_short[r['model']],r['minute']]+[num(r[k],5) for k in ['balanced_accuracy','f1','roc_auc','ece_10_equal_width_bins']] for r in ordered],'tab:metrics')
    longtable('s_scope.tex',r'All training-scope contrasts. L$-$M controls monthly training counts; L$-$B includes a change in training-set size. Targets are identical within each contrast. Intervals are descriptive paired-seed 95\% intervals.','rllrrr', ['Min.','Model / contrast','Target','$n$','LL difference',r'95\% interval'], [[r['minute'],names_short[r['model']]+'/'+('L$-$M' if r['contrast']=='listed_minus_broad_matched' else 'L$-$B'),'All' if r['target']=='all_holdout' else 'Listed',r['n'],num(r['difference'],6),ci(r,precision=6)] for r in scope],'tab:allscope')
    sensitivity=read_csv(report/'sampling_sensitivities.csv')
    scheme={'equal_month_conditional_seed':'Equal-month / seeds','match_iid_reference_only':'Match IID reference','UTC_7day_blocks':'7-day blocks','UTC_14day_blocks':'14-day blocks','UTC_28day_blocks':'28-day blocks','league_clusters':'League clusters'}
    longtable('s_sampling.tex','Sampling and month-weighting sensitivity of broad-training G$-$N. Block intervals condition on fitted models. Fewer than 20 occupied blocks yields no interval. The equal-month row uses the training-seed interval.','rlrrr', ['Min.','Scheme','Occupied blocks','Difference',r'95\% interval'],[[r['minute'],scheme[r['scheme']],r['nonempty_blocks'] or '--',num(r['point'],6),ci(r,precision=6)] for r in sensitivity],'tab:sampling')
    common=read_csv(report/'common40_trajectory.csv')
    cidx={(r['scope'],int(r['minute']),r['model']):r for r in common}
    longtable('s_common.tex','All models evaluated at each cutoff on the same 2,118 matches eligible at 40 minutes. Entries are seed-mean log loss. Models are separately trained for each cutoff.','lrrrr',['Pool/model','10 min','20 min','30 min','40 min'],[[short[pool]+'/'+names_short[m]]+[num(cidx[pool,t,m]['log_loss'],6) for t in (10,20,30,40)] for pool in ['broad','listed','broad_matched'] for m in (models if pool=='broad' else ['G','N','xgboost'])],'tab:common')
    pars=read_csv(report/'resolution_parameters.csv')
    abbr={'branch_width':'width','learning_rate':'lr','weight_decay':'wd','batch_size':'batch','min_child_samples':'min child','min_samples_leaf':'min leaf','max_features':'max feat','n_estimators':'trees','colsample_bytree':'colsample','reg_lambda':'lambda','reg_alpha':'alpha','min_child_weight':'child wt','max_depth':'depth','num_leaves':'leaves'}
    prows=[]
    for r in pars:
        p=json.loads(r['params']);ps='; '.join(esc(abbr.get(k,k))+'='+(format(v,'.6g') if isinstance(v,float) else str(v)) for k,v in p.items())
        prows.append([short[r['scope']]+'/'+names_short[r['model']],r['minute'],r['selected_candidate'],num(r['selected_search_mean'],5),ps])
    longtable('s_parameters.tex','Final selected parameters from the 48-candidate searches. Candidate identifiers are zero-based. Search loss is measured on development validation data. Parameters are rounded for display; exact configurations and run-level results accompany the source tables.',r'>{\raggedright\arraybackslash}p{2.1cm}rrr>{\raggedright\arraybackslash}p{8.9cm}', ['Pool/model','Min.','ID','Search LL','Parameters'],prows,'tab:parameters')
    # Machine-readable summary for audits: source values, no independently rounded inputs.
    (TABLES/'numeric_contract.json').write_text(json.dumps({'primary':primary,'cells':len(rows),'vectors':len(read_csv(report/'per_seed_results.csv')),'macros':macros},indent=2),encoding='utf-8')
    print('Generated tables from scored reports;',len(rows),'cells.')

if __name__=='__main__':main()

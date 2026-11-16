"""Export current supplementary tables; retain provenance of separate experiments."""
from pipeline_common import *
def rows(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def table(out,name,caption,label,headers,values):
 spec='l'+'r'*(len(headers)-1)
 s='\\begin{table}[htbp]\n\\caption{'+caption+'}\\label{tab:'+label+'}\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{'+spec+'}\\toprule\n'
 s+=' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(map(str,r))+r'\\' for r in values)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n'
 (out/name).write_text(s,encoding='utf-8')
def export(run):
 out=run/'tables';data=CODE/'reference_tables';cats=rows(run/'reports/category_results.csv');values=[]
 for category,groups in [('listed',('professional','premium')),('excluded',('excluded',)),('unmatched',('unmatched',))]:
  cells={m:[r for r in cats if r['scope']=='broad' and r['minute']=='20' and r['axis']=='provider_tier' and r['model']==m and r['group'] in groups] for m in ('G','N')}
  n=sum(int(r['n']) for r in cells['G']);wins=sum(int(r['radiant_wins']) for r in cells['G'])
  values.append([category,n,f'{100*wins/n:.2f}',*[f"{sum(int(r['n'])*float(r['log_loss']) for r in cells[m])/n:.4f}" for m in ('G','N')]])
 original=(out/'category20.tex').read_text();(out/'s_category_original.tex').write_text(original.replace('tab:category','tab:category_original').replace('\\scriptsize','\\small'))
 table(out,'category20.tex','Twenty-minute broad-training category results; listed combines professional and premium labels.','category',['Category','$n$',r'R win (\%)','G LL','N LL'],values)
 draws=rows(data/'sampling_draw_means.csv')
 table(out,'s_repeat_draws.tex','Original supplementary N sampling experiment; three seeds per draw.','repeat_draws',['Draw','Full mean','Full SD','Listed mean','Listed SD'],[[i,*[f"{float(next(r for r in draws if r['draw']==f'draw_{i:02d}' and r['target']==t)[k]):.6f}" for t in ('all','listed') for k in ('mean_difference','seed_sd_difference')]] for i in range(1,6)])
 effects=rows(data/'baseline_effects.csv');intervals=rows(data/'baseline_sampling_intervals.csv');values=[]
 for r in effects:
  pair=[]
  for method in ('7_day','league'):
   x=next(x for x in intervals if x['minute']==r['minute'] and x['contrast']==r['contrast'] and x['method']==method);pair.append(f"[{float(x['L']):.6f}, {float(x['U']):.6f}]")
  seed='--' if not r['seed_CI_low'] else f"[{float(r['seed_CI_low']):.6f}, {float(r['seed_CI_high']):.6f}]"
  values.append([r['minute'],'G' if r['contrast'].startswith('G ') else 'Full LR',f"{float(r['mean_difference']):.6f}",seed,*pair])
 table(out,'s_extra_baselines.tex',r'Original supplementary contrasts relative to Gold/XP LR; 95\% intervals.','extra_baselines',['Min','Left','Difference','Seed interval','7-day interval','League interval'],values)
 costs=rows(data/'model_size_training_summary.csv')
 table(out,'s_all_costs.tex','Original final-fit parameter counts and recorded training times.','all_costs',['Min','Model','Parameters','Median seconds','Min seconds','Max seconds'],[[r['minute'],r['model'].replace('_','\\_'),r['parameters'],*[f"{float(r[k]):.2f}" for k in ('fit_seconds_median','fit_seconds_min','fit_seconds_max')]] for r in costs])
 timing=rows(data/'inference_timing_summary.csv')
 table(out,'s_timing_ranges.tex','Original synchronized GPU timings, five rounds.','timing_ranges',['Model','Batch','Median ms/batch','Min','Max','ms/match'],[[r['model'].replace('_','\\_'),r['batch_size'],*[f"{float(r[k]):.4f}" for k in ('median_ms_batch','min_ms_batch','max_ms_batch','median_ms_match')]] for r in timing])
 table(out,'cost20.tex','Original 20-minute resident-input GPU benchmark; medians in ms per batch.','cost',['Model','Parameters','Batch 1','Batch 128'],[[name,params,*[f"{float(next(r for r in timing if r['model']==key and r['batch_size']==b)['median_ms_batch']):.3f}" for b in ('1','128')]] for key,name,params in [('G','G','90,210'),('N','N','107,442'),('flat_mlp','Flat MLP','250,241')]])
 write(out/'table_sources.json',{'scored_reports':{p.name:sha(p) for p in (run/'reports').glob('*.csv')},'separate_original_experiments':{p.name:sha(p) for p in data.glob('*.csv')},'policy':'Separate supplementary experiments are frozen source-table exports, not recomputed from new holdout predictions. Recompute their analysis when changing models.'})

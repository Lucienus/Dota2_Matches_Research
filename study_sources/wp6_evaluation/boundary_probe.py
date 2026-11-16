"""Pre-holdout single-factor boundary diagnostic; never selects replacement models."""
from resolution_common import *
from engine import Data,execute
from statistics6 import seed_interval
import argparse,traceback

def freeze():
    p=HERE6/'boundary_probe_protocol.json'
    if p.exists():return
    jobs=[];plan=read(HERE6/'resolution_protocol.json')
    for scope,t,name,key,values,required in [('broad',20,'lightgbm','colsample_bytree',[.25,.125],True),
        ('broad_matched',20,'xgboost','colsample_bytree',[.0625,.03125],True),
        ('listed',40,'N','learning_rate',[.006,.012],True),('listed',40,'G','learning_rate',[.006,.012],False)]:
        cell=next(c for c in plan['cells'] if (c['scope'],c['minute'],c['model'])==(scope,t,name))
        source=final_folder(cell)/'selection.json';s=read(source)
        references=[]
        for seed in (110113,110117,110119):
            paths=[ROOT6/u['path'] for u in s['used_runs'] if u['stage']=='search' and u['candidate']==s['selected_candidate'] and u['seed']==seed]
            assert paths
            references.append({'seed':seed,'path':rel(paths[0]),'result_sha256':sha(paths[0]/'result.json')})
        jobs.append({'scope':scope,'minute':t,'model':name,'coordinate':key,'values':values,'required':required,
                     'params':s['params'],'selection_path':rel(source),'selection_sha256':sha(source),'references':references})
    write(p,{'version':'wp6-single-factor-diagnostic-v1','created_unix':time.time(),'jobs':jobs,'planned_fits':24,
      'search_seeds':[110113,110117,110119],'neural_epochs':450,'boosting_cap':10000,
      'decision':'For all required probes: complete early stopping and mean paired improvement (incumbent minus outward probe) <=0.0005. Otherwise retain closed gate for further development review.',
      'interval':'95pct paired-search-seed t, descriptive diagnostic only; not an equivalence test',
      'no_model_reselection':True,'primary_comparison_unchanged':True,'main_search_budget':48,'no_holdout_access':True})
    write(HERE6/'boundary_probe_lock.json',{'sources':{rel(q):sha(q) for q in [Path(__file__),p,HERE6/'resolution_lock.json']}})

def review():
    plan=read(HERE6/'boundary_probe_protocol.json');rows=[];passed=True;completed=0
    for job in plan['jobs']:
        for index,value in enumerate(job['values']):
            diffs=[];early=True
            for ref in job['references']:
                q=HERE6/'boundary_probe_runs'/job['scope']/f"{job['minute']}m"/job['model']/f"probe{index}_seed{ref['seed']}"
                if not (q/'result.json').exists():passed=False;continue
                assert sha(ROOT6/ref['path']/'result.json')==ref['result_sha256']
                diffs.append(read(ROOT6/ref['path']/'result.json')['validation_log_loss']-read(q/'result.json')['validation_log_loss'])
                early=early and stopped(q,job['model']);completed+=1
            full=len(diffs)==3;s=seed_interval(diffs) if diffs else {'mean':None,'L':None,'U':None}
            ok=full and early and s['mean']<=.0005
            if job['required'] and not ok:passed=False
            rows.append({'scope':job['scope'],'minute':job['minute'],'model':job['model'],'coordinate':job['coordinate'],'outward_value':value,
               'required_for_gate':job['required'],'completed_seeds':len(diffs),'mean_improvement':s['mean'],'L':s['L'],'U':s['U'],'early_stopped':early if full else None,'diagnostic_pass':ok})
    csvout('boundary_probe_results.csv',rows)
    write(HERE6/'reports/boundary_probe_review.json',{'status':'PASS' if passed and completed==24 else 'WAITING' if completed<24 else 'NEEDS_REVIEW',
        'completed_fits':completed,'planned_fits':24,'rows':rows,'protocol_sha256':sha(HERE6/'boundary_probe_protocol.json'),
        'main_models_unchanged':True,'no_holdout_evidence':True})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--review',action='store_true');ap.add_argument('--lane',choices=['cpu','gpu']);a=ap.parse_args()
    if a.freeze:freeze();return
    check_lock(HERE6/'resolution_lock.json');check_lock(HERE6/'boundary_probe_lock.json')
    if a.review:review();return
    assert a.lane
    try:
        plan=read(HERE6/'boundary_probe_protocol.json')
        for job in plan['jobs']:
            name=job['model']
            if (name in ('G','N'))!=(a.lane=='gpu'):continue
            assert sha(ROOT6/job['selection_path'])==job['selection_sha256']
            data=Data(job['minute'],'cuda' if a.lane=='gpu' else 'cpu',data_dir(job['scope']))
            for i,value in enumerate(job['values']):
                params={**job['params'],job['coordinate']:value}
                for seed in plan['search_seeds']:
                    write(HERE6/f'probe_status_{a.lane}.json',{'status':'running','scope':job['scope'],'minute':job['minute'],'model':name,'coordinate':job['coordinate'],'value':value,'seed':seed,'updated':time.time()})
                    out=HERE6/'boundary_probe_runs'/job['scope']/f"{job['minute']}m"/name/f'probe{i}_seed{seed}'
                    r=execute(data,name,params,seed,out,epochs=450,identity={'probe_lock_sha256':sha(HERE6/'boundary_probe_lock.json')})
                    print(a.lane,job['scope'],job['minute'],name,value,seed,r['validation_log_loss'],flush=True)
            del data
        write(HERE6/f'probe_status_{a.lane}.json',{'status':'complete','updated':time.time()})
    except Exception:
        write(HERE6/f'probe_status_{a.lane}.json',{'status':'failed','traceback':traceback.format_exc(),'updated':time.time()});raise
if __name__=='__main__':main()

from common4 import *
import argparse,time,traceback
import numpy as np
from engine import Data,execute

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lane',choices=['cpu','gpu'],required=True);a=ap.parse_args()
    protocol=read(HERE4/'search_protocol.json');lock=read(HERE4/'execution_lock.json')
    for p,h in lock['sources'].items():
        if sha(ROOT4/p)!=h:raise RuntimeError('Changed source '+p)
    contract=read(HERE4/'data/input_contract.json')
    for p,h in contract['hashes'].items():
        if sha(HERE4/'data'/p)!=h:raise RuntimeError('Changed data '+p)
    families=['G','N','flat_mlp'] if a.lane=='gpu' else ['gold_xp_lr','full_feature_lr','hodge_adapted','lightgbm','xgboost','random_forest']
    identity={'execution_lock_sha256':sha(HERE4/'execution_lock.json'),'protocol_sha256':sha(HERE4/'search_protocol.json'),'input_contract_sha256':sha(HERE4/'data/input_contract.json')}
    active=None
    try:
        for minute in protocol['minutes']:
            data=Data(minute,'cuda' if a.lane=='gpu' else 'cpu')
            for name in families:
                base=HERE4/'runs'/f'{minute}m'/name;stage1=[];candidates=protocol['candidates'][name]
                def fit(stage,cid,seed):
                    nonlocal active
                    active={'lane':a.lane,'minute':minute,'model':name,'stage':stage,'candidate':cid,'seed':seed}
                    write(HERE4/f'status_{a.lane}.json',{'status':'running',**active,'updated_unix':time.time()})
                    folder=base/stage/f'candidate_{cid:02d}_seed{seed}'
                    result=execute(data,name,candidates[cid],seed,folder,identity=identity)
                    print(a.lane,minute,name,stage,cid,seed,'val',round(result['validation_log_loss'],6),'sec',round(result['elapsed_seconds'],1),flush=True)
                    return result
                for cid in range(32):stage1.append(fit('search',cid,110113))
                ranked=sorted(range(32),key=lambda c:(stage1[c]['validation_log_loss'],c))
                aggregate=[]
                if name not in protocol['deterministic_families']:
                    for cid in ranked[:4]:
                        rs=[stage1[cid]]+[fit('search',cid,seed) for seed in (110117,110119)]
                        aggregate.append({'candidate_id':cid,'mean_validation_log_loss':float(np.mean([r['validation_log_loss'] for r in rs])),'search_seeds':[110113,110117,110119]})
                    selected=min(aggregate,key=lambda r:(r['mean_validation_log_loss'],r['candidate_id']))['candidate_id']
                else:selected=ranked[0]
                params=candidates[selected];boundary=[]
                for k,v in params.items():
                    values=[p[k] for p in candidates if isinstance(p[k],(float,int)) and not isinstance(p[k],bool)]
                    if isinstance(v,(float,int)) and len(set(values))>1 and v in (min(values),max(values)):boundary.append(k)
                write(base/'selection.json',{'selected_candidate':selected,'params':params,'first_seed_ranking':ranked,'repeated_search_scores':aggregate,'range_endpoint_coordinates':boundary,'holdout_allowed':False,'selection_metric':'validation_log_loss'})
                seeds=protocol['seeds']['final_seeds'] if name not in protocol['deterministic_families'] else [protocol['seeds']['final_seeds'][0]]
                final=[fit('final',selected,seed) for seed in seeds]
                write(base/'seed_summary.json',{'seeds':seeds,'mean_validation_log_loss':float(np.mean([r['validation_log_loss'] for r in final])),'sd_validation_log_loss':float(np.std([r['validation_log_loss'] for r in final],ddof=1)) if len(final)>1 else None,'deterministic_one_fit':len(final)==1,'boundary_flags':boundary,'population':'fixed development validation, not heldout inference'})
            if a.lane=='cpu':
                from lr_controls import run_controls
                run_controls(data,identity)
                for name in ('empirical_prior','constant_half'):
                    execute(data,name,{},110113,HERE4/'runs'/f'{minute}m'/name/'reference',identity=identity)
            del data
        write(HERE4/f'status_{a.lane}.json',{'status':'complete','updated_unix':time.time()})
    except Exception as e:
        write(HERE4/f'status_{a.lane}.json',{'status':'failed','active':active,'error':str(e),'traceback':traceback.format_exc(),'updated_unix':time.time()});raise
if __name__=='__main__':main()

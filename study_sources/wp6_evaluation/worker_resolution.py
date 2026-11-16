from resolution_common import *
from engine import Data,execute
import argparse,traceback

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lane',choices=['cpu','gpu'],required=True);lane=ap.parse_args().lane
    check_lock(HERE6/'resolution_lock.json');protocol=read(HERE6/'resolution_protocol.json')
    for p,h in read(WP4/'data/input_contract.json')['hashes'].items():assert sha(WP4/'data'/p)==h
    for p,h in read(WP5/'data/complete.json')['hashes'].items():assert sha(WP5/p)==h
    active={};data=None;cachekey=None
    try:
        for cell in protocol['cells']:
            scope,t,name=cell['scope'],cell['minute'],cell['model']
            if (name in ('G','N','flat_mlp'))!=(lane=='gpu'):continue
            folder=final_folder(cell)
            if (folder/'selection.json').exists():continue
            if cachekey!=(scope,t):
                if data is not None:del data
                data=Data(t,'cuda' if lane=='gpu' else 'cpu',data_dir(scope));cachekey=scope,t
            candidates=parameters(cell);used=[]
            def fit(stage,cid,seed):
                nonlocal active
                active={'scope':scope,'minute':t,'model':name,'stage':stage,'candidate':cid,'seed':seed}
                write(HERE6/f'resolution_status_{lane}.json',{'status':'running',**active,'updated':time.time()})
                earlier=old_run(cell,stage,cid,seed)
                if cid<40 and (earlier/'result.json').exists() and stopped(earlier,name):
                    used.append({'stage':stage,'candidate':cid,'seed':seed,'path':rel(earlier),'reused':True});return read(earlier/'result.json'),earlier
                params=dict(candidates[cid])
                if name in ('xgboost','lightgbm','hodge_adapted'):params['n_estimators']=protocol['boosting_max_estimators']
                out=folder/stage/f'candidate_{cid:02d}_seed{seed}'
                result=execute(data,name,params,seed,out,epochs=protocol['neural_max_epochs'],identity={'resolution_lock_sha256':sha(HERE6/'resolution_lock.json'),'source_data_directory':rel(data_dir(scope))})
                used.append({'stage':stage,'candidate':cid,'seed':seed,'path':rel(out),'reused':False})
                print(lane,scope,t,name,stage,cid,seed,round(result['validation_log_loss'],7),'early_stopped',stopped(out,name),flush=True)
                return result,out
            initial=[fit('search',i,110113)[0] for i in range(48)]
            ranking=sorted(range(48),key=lambda i:(initial[i]['validation_log_loss'],i));scores=[];det=name.endswith('_lr')
            for cid in ranking[:1 if det else 4]:
                rs=[initial[cid]] if det else [initial[cid]]+[fit('search',cid,s)[0] for s in (110117,110119)]
                scores.append({'candidate':cid,'mean_log_loss':float(np.mean([r['validation_log_loss'] for r in rs]))})
            selected=min(scores,key=lambda r:(r['mean_log_loss'],r['candidate']))['candidate']
            seeds=protocol['final_seeds'][:1] if det else protocol['final_seeds'];finals=[fit('final',selected,s) for s in seeds]
            effective=dict(candidates[selected])
            if name in ('xgboost','lightgbm','hodge_adapted'):effective['n_estimators']=protocol['boosting_max_estimators']
            write(folder/'selection.json',{'scope':scope,'minute':t,'model':name,'selected_candidate':selected,'params':effective,
                  'first_seed_ranking':ranking,'repeated_search_scores':scores,'final_paths':[rel(p) for r,p in finals],
                  'final_seeds':seeds,'final_validation_loss_mean':float(np.mean([r['validation_log_loss'] for r,p in finals])),
                  'final_all_early_stopped':all(stopped(p,name) for r,p in finals),'used_runs':used,'holdout_allowed':False})
        write(HERE6/f'resolution_status_{lane}.json',{'status':'complete','updated':time.time()})
    except Exception as e:
        write(HERE6/f'resolution_status_{lane}.json',{'status':'failed','active':active,'error_type':type(e).__name__,'traceback':traceback.format_exc(),'updated':time.time()});raise
if __name__=='__main__':main()

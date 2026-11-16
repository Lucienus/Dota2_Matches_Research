from common5 import *
import time,traceback,argparse
import numpy as np
from engine import Data,execute

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lane',choices=['gpu','cpu'],required=True);lane=ap.parse_args().lane
    protocol=read(HERE5/'protocol.json');search=read(WP4/'search_protocol.json');lock=read(HERE5/'execution_lock.json')
    for p,h in lock['sources'].items():assert sha(ROOT4/p)==h,p
    while not (HERE5/'data/complete.json').exists():
        write(HERE5/f'status_{lane}.json',{'status':'waiting_for_data','updated':time.time()});time.sleep(15)
    complete=read(HERE5/'data/complete.json')
    for p,h in complete['hashes'].items():assert sha(HERE5/p)==h,p
    families=['G','N'] if lane=='gpu' else [protocol['tree_selection']['selected']]
    identity={'protocol_sha256':sha(HERE5/'protocol.json'),'data_contract_sha256':sha(HERE5/'data/complete.json'),'execution_lock_sha256':sha(HERE5/'execution_lock.json')}
    active={}
    try:
        for minute in protocol['minutes']:
            for scope in ('listed','broad_matched'):
                data=Data(minute,'cuda' if lane=='gpu' else 'cpu',HERE5/'data'/scope)
                for name in families:
                    base=HERE5/'runs'/scope/f'{minute}m'/name;candidates=search['candidates'][name]
                    def fit(stage,cid,seed):
                        nonlocal active
                        active={'scope':scope,'minute':minute,'model':name,'stage':stage,'candidate':cid,'seed':seed}
                        write(HERE5/f'status_{lane}.json',{'status':'running',**active,'updated':time.time()})
                        r=execute(data,name,candidates[cid],seed,base/stage/f'candidate_{cid:02d}_seed{seed}',identity=identity)
                        print(lane,scope,minute,name,stage,cid,seed,round(r['validation_log_loss'],6),flush=True);return r
                    initial=[fit('search',i,110113) for i in range(32)];ranking=sorted(range(32),key=lambda i:(initial[i]['validation_log_loss'],i));scores=[]
                    for cid in ranking[:4]:
                        rs=[initial[cid]]+[fit('search',cid,s) for s in (110117,110119)];scores.append((float(np.mean([r['validation_log_loss'] for r in rs])),cid))
                    chosen=min(scores)[1]
                    endpoints=[]
                    for k,v in candidates[chosen].items():
                        values=[p[k] for p in candidates if isinstance(p[k],(float,int))]
                        if isinstance(v,(float,int)) and len(set(values))>1 and v in (min(values),max(values)):endpoints.append(k)
                    write(base/'selection.json',{'selected_candidate':chosen,'params':candidates[chosen],'first_seed_ranking':ranking,'repeated_scores':scores,'range_endpoints':endpoints,'target':'same WP4 development validation','holdout_allowed':False})
                    final=[fit('final',chosen,s) for s in search['seeds']['final_seeds']]
                    write(base/'complete.json',{'fits':50,'seeds':search['seeds']['final_seeds'],'mean_validation_log_loss':float(np.mean([r['validation_log_loss'] for r in final]))})
                del data
        write(HERE5/f'status_{lane}.json',{'status':'complete','updated':time.time()})
    except Exception as e:
        write(HERE5/f'status_{lane}.json',{'status':'failed','active':active,'error':str(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()

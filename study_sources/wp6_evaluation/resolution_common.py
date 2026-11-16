from common6 import *
import numpy as np

def old_run(cell,stage,cid,seed):
    leaf=f'{stage}/candidate_{cid:02d}_seed{seed}'
    latest=HERE6/'extension_runs'/cell['scope']/f"{cell['minute']}m"/cell['model']/leaf
    original=ROOT6/cell['old_base']/leaf
    return latest if (latest/'result.json').exists() else original

def stopped(folder,name):
    r=read(folder/'result.json')
    if name in ('G','N','flat_mlp'):
        return len(r['history'])-r['best_epoch']>=15
    if name in ('lightgbm','hodge_adapted','xgboost'):
        curve=read(folder/'learning_curve.json');sequences=[z for d in curve.values() if isinstance(d,dict) for z in d.values() if isinstance(z,list)]
        if len(sequences)!=1:raise RuntimeError('Ambiguous validation learning curve')
        a=np.asarray(sequences[0]);return len(a)-1-int(np.argmin(a))>=50
    return True

def parameters(cell):
    old=read(WP4/'search_protocol.json')['candidates'][cell['model']]
    return old+cell['additional_candidates']+cell['extra_candidates']

def final_folder(cell):return HERE6/'resolution_runs'/cell['scope']/f"{cell['minute']}m"/cell['model']

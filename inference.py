"""Inference only. No labels, fitting, calibration, early stopping, or model selection."""
from pipeline_common import *
from common4 import *
import numpy as np
import torch,joblib
from inputs import HeroVocabulary,flatten
from engine import BranchModel,FlatMLP,FIELDS

def load_arrays(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}

def infer(arrays,train_path,model_path,name,minute,device='cpu'):
    with np.load(train_path,allow_pickle=False) as a:vocab=HeroVocabulary().fit(a['heroes'])
    b={k:v for k,v in arrays.items() if k not in ('outcome','labels')}
    b['heroes']=vocab.transform(b['heroes']);b['lengths']=np.full(len(b['match_id']),minute+1,dtype=np.int64)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    if name in ('G','N','flat_mlp'):
        checkpoint=torch.load(model_path,map_location='cpu',weights_only=False);spec=checkpoint['spec'];params=spec['params']
        assert checkpoint['hero_ids']==vocab.ids and checkpoint['num_heroes']==vocab.size and spec['minute']==minute and spec['model']==name
        model=FlatMLP(vocab.size,minute,width=params['branch_width'],dropout=params['dropout']) if name=='flat_mlp' else BranchModel(vocab.size,name,params['branch_width'],params['dropout'])
        model.load_state_dict(checkpoint['state_dict']);model.to(device).eval();p=[]
        with torch.no_grad():
            for start in range(0,len(b['match_id']),512):
                batch={k:torch.as_tensor(b[k][start:start+512],device=device) for k in FIELDS}
                p.append(model(batch).sigmoid().cpu().numpy())
        probabilities=np.concatenate(p)
    else:
        model=joblib.load(model_path)
        if name=='gold_xp_lr':x=b['sequence'][:,-1,:2].astype(float)
        elif name=='hodge_adapted':x=b['hodge_observable'].astype(float)
        elif name in ('empirical_prior','constant_half'):x=np.zeros((len(b['match_id']),1))
        else:x=flatten({k:torch.as_tensor(b[k]) for k in FIELDS},vocab.size).numpy().astype(float)
        probabilities=model.predict_proba(x)[:,1]
    assert np.isfinite(probabilities).all() and np.all((probabilities>=0)&(probabilities<=1))
    return probabilities

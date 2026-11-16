from common4 import *
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import time,copy,random,threading,warnings
import numpy as np
import torch,joblib,psutil
from torch import nn
from controlled_models import ControlledModel,ControlledAttention,FlatMLP
from inputs import HeroVocabulary,flatten
from baselines import make_baseline,fit_baseline,ContinuousScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

NEURAL=('G','N','flat_mlp')
FIELDS=('nodes','availability','heroes','edges','roles','sequence','lengths')

class BranchModel(ControlledModel):
    """WP4 width search changes relation branch only; shared branches stay width64."""
    def __init__(self,num_heroes,branch,branch_width,dropout):
        super().__init__(num_heroes,branch=branch,width=64,dropout=dropout)
        if branch=='G':
            self.g1=ControlledAttention(43,branch_width)
            self.g2=ControlledAttention(branch_width,32)
        else:self.non_graph=nn.Sequential(nn.Linear(630,branch_width),nn.ReLU(),nn.Dropout(dropout),nn.Linear(branch_width,64))

def logloss(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7);y=np.asarray(y,dtype=float)
    return float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))

class Data:
    def __init__(self,minute,device='cpu',directory=None):
        self.minute=minute;self.path=Path(directory or HERE4/'data');self.arrays={}
        for split in ('train','validation'):
            with np.load(self.path/f'{split}_{minute}m.npz',allow_pickle=False) as a:self.arrays[split]={k:a[k] for k in a.files}
            b=self.arrays[split];b['lengths']=np.full(len(b['outcome']),minute+1,dtype=np.int64)
        assert not set(self.arrays['train']['match_id'])&set(self.arrays['validation']['match_id'])
        self.vocab=HeroVocabulary().fit(self.arrays['train']['heroes']);self.batches={};self.flat={}
        for split,b in self.arrays.items():
            b['heroes']=self.vocab.transform(b['heroes'])
            self.batches[split]={k:torch.as_tensor(b[k],device=device) for k in FIELDS}
        self.device=device
    def features(self,split,name):
        b=self.arrays[split]
        if name=='gold_xp_lr':return b['sequence'][:,-1,:2].astype(float)
        if name=='hodge_adapted':return b['hodge_observable'].astype(float)
        if name in ('constant_half','empirical_prior'):return np.zeros((len(b['outcome']),1))
        if split not in self.flat:self.flat[split]=flatten({k:v.cpu() for k,v in self.batches[split].items()},self.vocab.size).numpy().astype(float)
        return self.flat[split]

def subset(batch,indices):return {k:v[indices] for k,v in batch.items()}
def atomic_torch(obj,path):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)

def execute(data,name,params,seed,folder,epochs=150,patience=15,identity=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    spec={'model':name,'minute':data.minute,'params':params,'seed':seed,'epochs':epochs,'patience':patience,'identity':identity,'device':str(data.device)}
    if (folder/'run_config.json').exists():
        if read(folder/'run_config.json')!=spec:raise ValueError('Resume identity mismatch')
    else:write(folder/'run_config.json',spec)
    if (folder/'result.json').exists():return read(folder/'result.json')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter();peak=[psutil.Process().memory_info().rss];stop=threading.Event()
    def poll():
        while not stop.wait(.25):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    watcher=threading.Thread(target=poll,daemon=True);watcher.start()
    try:
        y=data.arrays['train']['outcome'];vy=data.arrays['validation']['outcome'];history=[];best_epoch=None;count=None;previous_seconds=0;iteration=None
        if name in NEURAL:
            if str(data.device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            bw=params['branch_width'];model=FlatMLP(data.vocab.size,data.minute,width=bw,dropout=params['dropout']) if name=='flat_mlp' else BranchModel(data.vocab.size,name,bw,params['dropout'])
            model.to(data.device);optimizer=torch.optim.AdamW(model.parameters(),lr=params['learning_rate'],weight_decay=params['weight_decay'])
            batch_size=params['batch_size'];yt=torch.as_tensor(y,device=data.device,dtype=torch.float32);vyt=torch.as_tensor(vy,device=data.device,dtype=torch.float32)
            rng=np.random.default_rng(seed);best=float('inf');bad=0;first_epoch=0;best_state=None
            resume=folder/'resume.pt'
            if resume.exists():
                c=torch.load(resume,map_location=data.device,weights_only=False);model.load_state_dict(c['model']);optimizer.load_state_dict(c['optimizer']);best=c['best'];bad=c['bad'];first_epoch=c['epoch']+1;best_epoch=c['best_epoch'];best_state=c['best_state'];history=c['history'];rng.bit_generator.state=c['numpy_rng'];torch.set_rng_state(c['cpu_rng'].cpu())
                if str(data.device).startswith('cuda'):torch.cuda.set_rng_state(c['cuda_rng'].cpu())
                previous_seconds=c['elapsed_seconds']
            def predict(split):
                model.eval();out=[]
                with torch.no_grad():
                    for offset in range(0,len(data.arrays[split]['outcome']),512):out.append(model(subset(data.batches[split],slice(offset,offset+512))).sigmoid().cpu().numpy())
                return np.concatenate(out)
            for epoch in range(first_epoch,epochs):
                if bad>=patience:break
                model.train();order=rng.permutation(len(y));online=0.;epoch_start=time.perf_counter()
                for off in range(0,len(order),batch_size):
                    ix=torch.as_tensor(order[off:off+batch_size],device=data.device);optimizer.zero_grad(set_to_none=True)
                    logits=model(subset(data.batches['train'],ix));loss=nn.functional.binary_cross_entropy_with_logits(logits,yt[ix])
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();online+=float(loss.detach())*len(ix)
                vp=predict('validation');vl=logloss(vy,vp)
                history.append({'epoch':epoch+1,'training_online_logit_loss':online/len(y),'validation_log_loss':vl,'seconds':time.perf_counter()-epoch_start})
                if vl<best:best=vl;bad=0;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                else:bad+=1
                atomic_torch({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'best_state':best_state,'best':best,'bad':bad,'epoch':epoch,'best_epoch':best_epoch,'history':history,'numpy_rng':rng.bit_generator.state,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if str(data.device).startswith('cuda') else None,'elapsed_seconds':previous_seconds+time.perf_counter()-started},resume)
                write(folder/'progress.json',{'epoch':epoch+1,'best_epoch':best_epoch,'best_validation_log_loss':best,'patience_used':bad,'elapsed_seconds':previous_seconds+time.perf_counter()-started})
            if best_state is None:raise RuntimeError('No valid checkpoint')
            model.load_state_dict(best_state);tp=predict('train');vp=predict('validation');count=sum(p.numel() for p in model.parameters())
            atomic_torch({'state_dict':best_state,'spec':spec,'num_heroes':data.vocab.size,'hero_ids':data.vocab.ids},folder/'model.pt')
            peak_gpu=int(torch.cuda.max_memory_allocated()) if str(data.device).startswith('cuda') else 0
        else:
            x=data.features('train',name);vx=data.features('validation',name);model=make_baseline(name,data.minute,seed,**params)
            if name in ('lightgbm','hodge_adapted'):model.set_params(deterministic=True,force_col_wise=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');fit_baseline(model,name,x,y,vx,vy)
            warning_records=[{'category':w.category.__name__,'message':str(w.message)} for w in caught]
            write(folder/'warnings.json',warning_records)
            if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError('LR did not converge; candidate retained for repair, not silently ranked')
            tp=model.predict_proba(x)[:,1];vp=model.predict_proba(vx)[:,1];joblib.dump(model,folder/'model.joblib');peak_gpu=0
            iteration=int(model[-1].n_iter_[0]) if name.endswith('_lr') else int(getattr(model,'best_iteration_',0) or getattr(model,'best_iteration',0) or params.get('n_estimators',0))
            write(folder/'learning_curve.json',getattr(model,'evals_result_',{} ) if name!='xgboost' else model.evals_result())
        if not (np.isfinite(tp).all() and np.isfinite(vp).all()):raise FloatingPointError('Invalid predictions')
        for split,p in [('train',tp),('validation',vp)]:np.savez_compressed(folder/f'{split}_predictions.npz',match_id=data.arrays[split]['match_id'],labels=data.arrays[split]['outcome'],probabilities=p)
        result={'model':name,'seed':seed,'minute':data.minute,'params':params,'n_train':len(y),'n_validation':len(vy),'training_log_loss':logloss(y,tp),'validation_log_loss':logloss(vy,vp),'training_accuracy':float(np.mean((tp>=.5)==y)),'validation_accuracy':float(np.mean((vp>=.5)==vy)),'best_epoch':best_epoch,'iterations':iteration,'parameter_count':count,'history':history,'elapsed_seconds':previous_seconds+time.perf_counter()-started,'peak_process_RSS_bytes':max(peak[0],psutil.Process().memory_info().rss),'peak_GPU_allocated_bytes':peak_gpu,'training_steps':sum((len(y)+params.get('batch_size',len(y))-1)//params.get('batch_size',len(y)) for _ in history),'mode':'development_validation_not_holdout'}
        write(folder/'result.json',result);return result
    finally:stop.set();watcher.join()

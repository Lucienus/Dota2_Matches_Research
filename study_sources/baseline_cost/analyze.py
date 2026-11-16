from pathlib import Path
import os,sys,time,json,csv,platform
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from freeze_additional_experiment import REV,ROOT,read,sha,write,csvout
sys.path.insert(0,str(REV/'wp4_training'))
from engine import BranchModel,FlatMLP,FIELDS
sys.path.insert(0,str(REV/'wp6_evaluation'))
from statistics6 import losses,seed_interval,block_sensitivity,time_blocks
from inputs import HeroVocabulary
import numpy as np
import torch

SOURCES={}
def track(p,expected=None):
    h=sha(p);SOURCES[p.relative_to(ROOT).as_posix()]=h
    if expected is not None:assert h==expected,p
    return p
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def csvread(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    lock=read(track(REV/'wp6_evaluation/evaluation_lock.json'))
    contract=read(track(REV/'wp6_evaluation/holdout_inputs/contract.json'))
    ph=read(track(REV/'wp6_evaluation/predictions_complete.json'))['hashes']
    def pred(model,t,seed):
        p=REV/f'wp6_evaluation/predictions/broad/{t}m/{model}/seed{seed}.npz'
        return load(track(p,ph[p.relative_to(ROOT).as_posix()]))
    cells={(c['minute'],c['model']):c for c in lock['models'] if c['scope']=='broad'}
    def locked(p):return track(p,lock['sources'].get(p.relative_to(ROOT).as_posix()))
    metadata=read(track(REV/'wp6_evaluation/holdout_inputs/metadata.json',contract['hashes']['metadata.json']))
    meta={r['match_id']:r for r in metadata}
    union=set()
    for t in (10,20,30,40):
        p=REV/f'wp6_evaluation/holdout_inputs/labels_{t}m.npz';a=load(track(p,contract['hashes'][p.name]));union.update(a['match_id'].tolist())
    universes={'7_day':sorted(set(time_blocks([meta[i]['start_time'] for i in union],7))),
               'league':sorted({meta[i]['leagueid'] for i in union})}
    effects=[];seedrows=[];intervals=[]
    for t in (10,20,30,40):
        a=load(REV/f'wp6_evaluation/holdout_inputs/labels_{t}m.npz');ids=a['match_id'];y=a['labels']
        computed={}
        for model in ('G','gold_xp_lr','full_feature_lr'):
            ps=[]
            for seed in cells[t,model]['seeds']:
                p=pred(model,t,seed);assert np.array_equal(p['match_id'],ids);ps.append(p['probabilities'])
            computed[model]=losses(y,np.stack(ps))
        for left in ('G','full_feature_lr'):
            d=computed[left]-computed['gold_xp_lr']
            s=seed_interval(d.mean(1));name=left+' minus Gold/XP LR'
            effects.append({'minute':t,'contrast':name,'n':len(y),'mean_difference':s['mean'],
                            'training_repeats':len(d),'seed_sd':s['seed_sd'],'seed_CI_low':s['L'],'seed_CI_high':s['U']})
            for seed,value in zip(cells[t,left]['seeds'],d.mean(1)):seedrows.append({'minute':t,'contrast':name,'seed':seed,'difference':float(value)})
            groups={'7_day':time_blocks([meta[int(i)]['start_time'] for i in ids],7),'league':[meta[int(i)]['leagueid'] for i in ids]}
            for method,g in groups.items():
                r=block_sensitivity(d.mean(0),g,universe=universes[method],replicates=20000,seed=731)
                assert abs(r['point'][0]-s['mean'])<1e-12
                intervals.append({'minute':t,'contrast':name,'method':method,'difference':s['mean'],'blocks':r['nonempty_blocks'],
                                  'L':r['L'][0] if r['L'] else None,'U':r['U'][0] if r['U'] else None,'replicates':r['replicates']})
    csvout(HERE/'baseline_effects.csv',effects);csvout(HERE/'baseline_seed_differences.csv',seedrows);csvout(HERE/'baseline_sampling_intervals.csv',intervals)
    # Numerical cross-check with independently frozen mean-loss tables.
    original=csvread(track(REV/'wp6_evaluation/reports/main_results.csv'))
    means={(int(r['minute']),r['model']):float(r['log_loss']) for r in original if r['scope']=='broad'}
    for r in effects:
        m=r['contrast'].split(' minus ')[0]
        assert abs(r['mean_difference']-(means[r['minute'],m]-means[r['minute'],'gold_xp_lr']))<1e-12
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    def model_for(t,name):
        path=ROOT/cells[t,name]['paths'][0];ck=torch.load(locked(path/'model.pt'),map_location='cpu',weights_only=False)
        param=ck['spec']['params'];v=ck['num_heroes']
        model=FlatMLP(v,t,param['branch_width'],param['dropout']) if name=='flat_mlp' else BranchModel(v,name,param['branch_width'],param['dropout'])
        model.load_state_dict(ck['state_dict']);model.eval();return model,ck
    size=[];training=[]
    for t in (10,20,30,40):
        for name in ('G','N','flat_mlp'):
            model,ck=model_for(t,name);count=sum(p.numel() for p in model.parameters())
            values=[];epochs=[]
            for seed,p in zip(cells[t,name]['seeds'],cells[t,name]['paths']):
                r=read(locked(ROOT/p/'result.json'));assert r['parameter_count']==count
                values.append(r['elapsed_seconds']);epochs.append(len(r['history']))
                training.append({'minute':t,'model':name,'seed':seed,'seconds':r['elapsed_seconds'],'epochs':len(r['history']),'best_epoch':r['best_epoch'],'source':p})
            size.append({'minute':t,'model':name,'parameters':count,'float32_parameter_bytes':4*count,
                         'branch_width':ck['spec']['params']['branch_width'],'dropout':ck['spec']['params']['dropout'],
                         'fit_seconds_median':float(np.median(values)),'fit_seconds_min':min(values),'fit_seconds_max':max(values),
                         'epochs_median':float(np.median(epochs)),'training_fit_count':len(values)})
    csvout(HERE/'model_size_training_summary.csv',size);csvout(HERE/'final_training_records.csv',training)
    records=csvread(track(REV/'wp6_evaluation/reports/resolution_fit_records.csv'));seen={};groups={}
    for r in records:
        if r['scope']!='broad' or r['model'] not in ('G','N','flat_mlp'):continue
        key=r['path'];assert key not in seen;seen[key]=True
        k=(r['minute'],r['model'],r['stage']);g=groups.setdefault(k,{'minute':int(r['minute']),'model':r['model'],'stage':r['stage'],'fits':0,'fit_seconds_sum':0.,'reused_fits':0})
        g['fits']+=1;g['fit_seconds_sum']+=float(r['elapsed_seconds']);g['reused_fits']+=r['reused'].lower()=='true'
    csvout(HERE/'selected_search_recipe_costs.csv',groups.values())
    if not torch.cuda.is_available():raise RuntimeError('GPU benchmark requires CUDA')
    path=REV/'wp6_evaluation/holdout_inputs/broad_20m.npz';a=load(track(path,contract['hashes'][path.name]))
    train=REV/'wp4_training/data/train_20m.npz';locked(train)
    vocab=HeroVocabulary().fit(load(train)['heroes'])
    a['heroes']=vocab.transform(a['heroes']);a['lengths']=np.full(len(a['match_id']),21,dtype=np.int64)
    models={};batches={}
    for name in ('G','N','flat_mlp'):
        m,ck=model_for(20,name);assert ck['hero_ids']==vocab.ids;models[name]=m.cuda()
    for batchsize in (1,128):batches[batchsize]={k:torch.as_tensor(a[k][:batchsize],device='cuda') for k in FIELDS}
    timings=[];names=list(models)
    with torch.no_grad():
        for b in (1,128):
            for name in names:
                for _ in range(20):models[name](batches[b])
            torch.cuda.synchronize()
            for round_id in range(5):
                order=names[round_id%3:]+names[:round_id%3]
                for position,name in enumerate(order):
                    torch.cuda.synchronize();start=time.perf_counter()
                    for _ in range(100):out=models[name](batches[b])
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    assert torch.isfinite(out).all()
                    timings.append({'model':name,'batch_size':b,'round':round_id+1,'position':position,'calls':100,
                                    'ms_per_batch':elapsed*1000/100,'ms_per_match':elapsed*1000/100/b})
    csvout(HERE/'inference_timing_rounds.csv',timings)
    summary=[]
    for b in (1,128):
        for name in names:
            v=[r['ms_per_batch'] for r in timings if r['batch_size']==b and r['model']==name]
            summary.append({'model':name,'batch_size':b,'median_ms_batch':float(np.median(v)),'min_ms_batch':min(v),'max_ms_batch':max(v),'median_ms_match':float(np.median(v))/b})
    csvout(HERE/'inference_timing_summary.csv',summary)
    write(HERE/'benchmark_environment.json',{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,'platform':platform.platform(),
                                         'dtype':'float32','batch_sizes':[1,128],'warmup':20,'rounds':5,'calls_per_round':100,'timing':'synchronized perf_counter; resident tensors; no feature extraction or transfer','match_ids':a['match_id'][:128].tolist()})
    assert len(effects)==8 and len(intervals)==16 and len(size)==12 and len(timings)==30
    for f in [HERE/'analyze.py',HERE/'ANALYSIS_PLAN_ZH.md',REV/'wp6_evaluation/statistics6.py',REV/'wp4_training/engine.py',REV/'wp3_models/controlled_models.py']:track(f)
    write(HERE/'validation.json',{'status':'PASS','baseline_cells':8,'block_intervals':16,'parameter_checks':120,'timing_rounds':30,'mean_loss_table_crosscheck':True,'sources':SOURCES})
    print('PASS baseline comparisons, parameter checks and GPU timing',flush=True)

if __name__=='__main__':main()

"""Frozen supplementary sampling pipeline: prepare, validate, GPU fit, infer, summarize."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys, time, threading, traceback, msvcrt, gc
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime, timezone
import numpy as np
from freeze_additional_experiment import HERE, REV, ROOT, read, sha, csvout
sys.path.insert(0,str(REV/'wp4_training'))
from engine import Data, execute, logloss, write
sys.path.insert(0,str(REV/'wp6_evaluation'))
from predict_holdout import infer, load_arrays
from feature_extractor import timestamped
import torch

STATE={'stage':'starting','completed_fits':0,'total_fits':15}
STOP=threading.Event()

def verify(lock):
    for p,h in read(lock)['hashes'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('Source changed: '+p)

def heartbeat():
    while not STOP.is_set():
        snapshot=dict(STATE,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat())
        folder=STATE.get('active_folder')
        if folder and (HERE/folder/'progress.json').exists():
            try:snapshot['training']=read(HERE/folder/'progress.json')
            except (OSError,ValueError):pass
        write(HERE/'training_progress.json',snapshot)
        body=['# 新增抽样实验进度','',f"状态：{snapshot['stage']}",f"已完成拟合：{snapshot['completed_fits']} / 15",f"更新时间（UTC）：{snapshot['updated_utc']}",f"进程：{os.getpid()}",f"GPU：{snapshot.get('gpu','检测中')}"]
        for k in ('preparation_records','draw','seed','training','error'):
            if k in snapshot:body.append(f'{k}：{snapshot[k]}')
        body+=['','本文件由运行进程更新。更新时间停止变化时需核对进程，不能仅凭状态文字判断仍在运行。','原WP6结果与论文未修改。阶段complete并生成sampling_complete.json才表示本次新增实验完成。']
        temp=HERE/'TRAINING_PROGRESS.md.tmp';temp.write_text('\n\n'.join(body)+'\n',encoding='utf-8');temp.replace(HERE/'TRAINING_PROGRESS.md')
        if STOP.wait(10):break

def purchase_record(entry):
    mid,path,expected=entry
    import json,hashlib
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected,mid
    d=json.loads(raw);assert int(d['match_id'])==mid
    out=[]
    for p in sorted(d['players'],key=lambda x:x['player_slot']):
        stages=[set(),set(),set()]
        for e in timestamped(p.get('purchase_log'),min(d['duration'],1800)):
            if isinstance(e.get('key'),str):stages[0 if e['time']<=720 else 1 if e['time']<=1200 else 2].add(e['key'].removeprefix('item_'))
        prefix=sorted({e['key'].removeprefix('item_') for e in timestamped(p.get('purchase_log'),1200) if isinstance(e.get('key'),str)})
        out.append({'hero':str(p['hero_id']),'stages':[sorted(s) for s in stages],'prefix':prefix})
    return mid,out

def prior_for(ids,records):
    counters=defaultdict(lambda:[Counter(),Counter(),Counter()])
    for mid in ids:
        for p in records[mid]:
            for j,items in enumerate(p['stages']):counters[p['hero']][j].update(items)
    return {'items':{h:[[k for k,v in sorted(c.items(),key=lambda x:(-x[1],x[0]))[:3]] for c in stages] for h,stages in counters.items()},'training_ids':ids}

def transform(base,ids,prior,records):
    lookup={int(i):j for j,i in enumerate(base['match_id'])}
    positions=[lookup[i] for i in ids]
    a={k:v[positions].copy() for k,v in base.items()}
    for j,mid in enumerate(ids):
        for h,p in enumerate(records[mid]):
            a['nodes'][j,h,10:13]=[len(set(p['prefix'])&set(top))/len(top) if top else 0 for top in prior['items'].get(p['hero'],[[],[],[]])]
    return a

def prepare(ids):
    complete=HERE/'data/complete.json'
    if complete.exists():verify(complete);return
    STATE['stage']='preparing_training_only_priors'
    dev=read(REV/'wp4_training/data/split_manifest.json')
    meta={r['match_id']:r for split in ('train','validation') for r in dev[split]}
    broad=read(REV/'wp5_scope/pool_ids.json')['20']['broad']
    entries=[(i,ROOT/'Dota2_matches'/meta[i]['path'],meta[i]['sha256']) for i in broad+ids['validation']]
    raw=read(REV/'wp6_evaluation/raw_manifest.json')['hashes']
    entries += [(i,REV/f'wp6_evaluation/raw_holdout/{i}.json',raw[f'{i}.json']) for i in ids['evaluation']]
    assert len({i for i,_,_ in entries})==len(entries)
    records={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,(mid,r) in enumerate(pool.map(purchase_record,entries),1):
            records[mid]=r
            if n%1000==0:STATE['preparation_records']=f'{n}/{len(entries)}';print('Purchase records',n,flush=True)
    bases={'train':load_arrays(REV/'wp4_training/data/train_20m.npz'),
           'validation':load_arrays(REV/'wp4_training/data/validation_20m.npz'),
           'evaluation':load_arrays(REV/'wp6_evaluation/holdout_inputs/broad_20m.npz')}
    # Exact replay against independently frozen prior/feature artifacts before new fits.
    checks=[]
    for scope,trainids in [('listed',ids['listed_reference']),('broad_matched',ids['original_matched_reference'])]:
        prior=prior_for(trainids,records)
        old=read(REV/f'wp5_scope/data/{scope}/prior_20m.json')
        assert prior==old,scope+' prior replay'
        for split,desired in [('train',trainids),('validation',ids['validation']),('evaluation',ids['evaluation'])]:
            a=transform(bases[split],desired,prior,records)
            path=REV/f'wp6_evaluation/holdout_inputs/{scope}_20m.npz' if split=='evaluation' else REV/f'wp5_scope/data/{scope}/{split}_20m.npz'
            expected=load_arrays(path)
            assert set(a)==set(expected)
            assert all(np.array_equal(a[k],expected[k]) for k in a),(scope,split)
            checks.append({'scope':scope,'split':split,'all_arrays_exact':True,'n':len(desired)})
    hashes={}
    for draw,trainids in ids['new_matched_draws'].items():
        folder=HERE/'data'/draw;folder.mkdir(parents=True,exist_ok=True)
        prior=prior_for(trainids,records);write(folder/'prior_20m.json',prior)
        for split,desired in [('train',trainids),('validation',ids['validation']),('evaluation',ids['evaluation'])]:
            a=transform(bases[split],desired,prior,records)
            assert np.array_equal(a['match_id'],desired)
            assert all(np.isfinite(v).all() for v in a.values())
            if split=='evaluation':assert not {'outcome','labels'}&set(a)
            np.savez_compressed(folder/f'{split}_20m.npz',**a)
        for p in folder.iterdir():hashes[p.relative_to(ROOT).as_posix()]=sha(p)
        print('Prepared',draw,flush=True)
    write(HERE/'reports/preparation_validation.json',{'status':'PASS','exact_replays':checks,'training_only_priors':True,'raw_records_hash_checked':len(entries)})
    write(complete,{'hashes':hashes,'protocol_lock_sha256':sha(HERE/'protocol_lock.json'),'execution_lock_sha256':sha(HERE/'training_execution_lock.json')})

def summarize(ids,protocol):
    labels=load_arrays(REV/'wp6_evaluation/holdout_inputs/labels_20m.npz')
    assert np.array_equal(labels['match_id'],ids['evaluation'])
    metadata={r['match_id']:r for r in read(REV/'wp6_evaluation/holdout_inputs/metadata.json')}
    masks={'all':np.ones(len(ids['evaluation']),bool),'listed':np.array([metadata[i]['provider_tier'] in ('professional','premium') for i in ids['evaluation']])}
    assert masks['listed'].sum()==1502
    refs={}
    for scope in ('listed','broad_matched'):
        for seed in protocol['training_seeds']:
            path=REV/f'wp6_evaluation/predictions/{scope}/20m/N/seed{seed}.npz'
            manifest=read(REV/'wp6_evaluation/predictions_complete.json')['hashes']
            assert sha(path)==manifest[path.relative_to(ROOT).as_posix()]
            a=load_arrays(path);assert np.array_equal(a['match_id'],ids['evaluation'])
            refs[scope,seed]={t:logloss(labels['labels'][mask],a['probabilities'][mask]) for t,mask in masks.items()}
    cells=[];means=[];original=[];interactions=[]
    for draw in ids['new_matched_draws']:
        for seed in protocol['training_seeds']:
            a=load_arrays(HERE/f'runs/{draw}/seed{seed}/evaluation_predictions.npz')
            assert np.array_equal(a['match_id'],ids['evaluation'])
            deltas={}
            for t,mask in masks.items():
                loss=logloss(labels['labels'][mask],a['probabilities'][mask]);reference=refs['listed',seed][t]
                delta=reference-loss;deltas[t]=delta
                cells.append({'draw':draw,'seed':seed,'target':t,'n':int(mask.sum()),'listed_loss':reference,'matched_loss':loss,'difference':delta})
            interactions.append({'draw':draw,'seed':seed,'listed_target_minus_all_target_difference':deltas['listed']-deltas['all']})
        for t in masks:
            subset=[r for r in cells if r['draw']==draw and r['target']==t]
            means.append({'draw':draw,'target':t,'mean_difference':float(np.mean([r['difference'] for r in subset])),
                          'seed_sd_difference':float(np.std([r['difference'] for r in subset],ddof=1)),
                          'mean_matched_loss':float(np.mean([r['matched_loss'] for r in subset])),
                          'seed_sd_matched_loss':float(np.std([r['matched_loss'] for r in subset],ddof=1))})
    summary=[]
    for t in masks:
        vals=np.array([r['mean_difference'] for r in means if r['target']==t]);losses=np.array([r['mean_matched_loss'] for r in means if r['target']==t])
        summary.append({'target':t,'draws':5,'mean_difference':float(vals.mean()),'draw_mean_sd':float(vals.std(ddof=1)),
                        'min':float(vals.min()),'max':float(vals.max()),'positive':int((vals>0).sum()),'negative':int((vals<0).sum()),'zero':int((vals==0).sum()),
                        'matched_loss_mean':float(losses.mean()),'matched_loss_draw_mean_sd':float(losses.std(ddof=1))})
        for seed in protocol['training_seeds']:original.append({'target':t,'seed':seed,'original_difference':refs['listed',seed][t]-refs['broad_matched',seed][t]})
    assert len(cells)==30 and len(means)==10 and len(interactions)==15
    for name,data in [('sampling_cells',cells),('sampling_draw_means',means),('sampling_summary',summary),('sampling_original_reference',original),('sampling_target_contrasts',interactions)]:csvout(HERE/f'reports/{name}.csv',data)
    lines=['# 新增抽样结果','', '补充稳健性分析；固定每个范围已选参数，5次抽样×3个种子。以下标准差来自五个三种子均值，不是联合置信区间。','', '|目标|平均listed−matched|抽样均值标准差|最小值|最大值|正/负/零抽样数|','|---|---:|---:|---:|---:|---|']
    for r in summary:lines.append(f"|{r['target']}|{r['mean_difference']:.6f}|{r['draw_mean_sd']:.6f}|{r['min']:.6f}|{r['max']:.6f}|{r['positive']}/{r['negative']}/{r['zero']}|")
    (HERE/'SAMPLING_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    lockfile=(HERE/'training_process.lock').open('a+b');lockfile.seek(0)
    try:msvcrt.locking(lockfile.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('Another sampling pipeline owns the lock')
    verify(HERE/'protocol_lock.json');verify(HERE/'training_execution_lock.json')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback')
    STATE['gpu']=torch.cuda.get_device_name(0)
    protocol=read(HERE/'experiment_protocol.json');ids=read(HERE/'frozen_sampling_ids.json')
    write(HERE/'training_environment.json',{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':STATE['gpu'],'pid':os.getpid()})
    monitor=threading.Thread(target=heartbeat,daemon=True);monitor.start()
    try:
        prepare(ids)
        for draw in ids['new_matched_draws']:
            directory=HERE/'data'/draw
            identity={'draw':draw,'protocol_lock_sha256':sha(HERE/'protocol_lock.json'),'execution_lock_sha256':sha(HERE/'training_execution_lock.json'),
                      'data_hashes':{p.name:sha(p) for p in directory.iterdir()}}
            data=Data(20,'cuda',directory)
            for seed in protocol['training_seeds']:
                STATE.update(stage='training',draw=draw,seed=seed,active_folder=f'runs/{draw}/seed{seed}')
                folder=HERE/STATE['active_folder']
                r=execute(data,'N',protocol['fixed_params']['broad_matched'],seed,folder,epochs=450,patience=15,identity=identity)
                if len(r['history'])-r['best_epoch']<15:raise RuntimeError(f'{draw}/{seed} reached cap without stopping; retain result and inspect')
                assert r['n_train']==24685 and r['n_validation']==7200
                STATE['completed_fits']+=1
                write(folder/'fit_validated.json',{'result_sha256':sha(folder/'result.json'),'model_sha256':sha(folder/'model.pt'),'early_stopped':True})
            del data;gc.collect();torch.cuda.empty_cache()
        STATE['stage']='predicting_all_fixed_models'
        verify(HERE/'protocol_lock.json');verify(HERE/'training_execution_lock.json');verify(HERE/'data/complete.json')
        for draw in ids['new_matched_draws']:
            for seed in protocol['training_seeds']:
                folder=HERE/f'runs/{draw}/seed{seed}';directory=HERE/'data'/draw
                arrays=load_arrays(directory/'evaluation_20m.npz')
                probabilities=infer(arrays,directory/'train_20m.npz',folder/'model.pt','N',20,'cuda')
                np.savez_compressed(folder/'evaluation_predictions.npz',match_id=arrays['match_id'],probabilities=probabilities)
        STATE['stage']='summarizing';summarize(ids,protocol)
        outputs=[p for p in (HERE/'runs').rglob('*') if p.is_file() and p.name in ('model.pt','result.json','run_config.json','fit_validated.json','evaluation_predictions.npz')]
        outputs+=list((HERE/'reports').glob('sampling_*.csv'))+[HERE/'SAMPLING_RESULTS_ZH.md']
        write(HERE/'sampling_complete.json',{'status':'PASS','new_fits':15,'protocol_lock_sha256':sha(HERE/'protocol_lock.json'),'execution_lock_sha256':sha(HERE/'training_execution_lock.json'),
                                            'hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in outputs}})
        STATE['stage']='complete'
    except BaseException as e:
        STATE.update(stage='needs_attention',error=repr(e));traceback.print_exc();raise
    finally:
        STOP.set();monitor.join();STOP.clear();STOP.set()
        # Publish one final snapshot without keeping the worker alive.
        write(HERE/'training_progress.json',dict(STATE,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat()))
        (HERE/'TRAINING_PROGRESS.md').write_text('# 新增抽样实验进度\n\n'+str(STATE)+'\n\n详见training_progress.json与日志。\n',encoding='utf-8')
        lockfile.close()

if __name__=='__main__':main()

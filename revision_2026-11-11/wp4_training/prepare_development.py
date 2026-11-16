"""Clean original train/validation anchors; prepare all four shared model inputs."""
from common4 import *
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import Counter,defaultdict
from inputs import extract,hodge_observable
from feature_extractor import timestamped

MINUTES=(10,20,30,40)
def one(row):
    p=PROJECT4/row['path'];raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Raw hash mismatch')
    d=json.loads(raw);samples={};failures=[]
    for t in MINUTES:
        if t not in row['eligible_minutes']:continue
        try:
            s=extract(d,t,{})
            s.update(hodge_observable=hodge_observable(d,t),outcome=np.float32(d['radiant_win']),match_id=np.int64(row['match_id']))
            if not all(np.isfinite(v).all() for v in s.values()):raise ValueError('Nonfinite input')
            samples[t]=s
        except (ValueError,KeyError,IndexError,TypeError) as e:failures.append({'match_id':row['match_id'],'minute':t,'reason':str(e)})
    # Training prior acquisitions fit from the cleaned training fold, independent of y.
    prior=[];purchases={t:[] for t in samples}
    for player in sorted(d['players'],key=lambda p:p['player_slot']):
        hero=str(player['hero_id']);stages=[set(),set(),set()]
        for ev in timestamped(player.get('purchase_log'),min(d['duration'],1800)):
            if isinstance(ev.get('key'),str):stages[0 if ev['time']<=720 else 1 if ev['time']<=1200 else 2].add(ev['key'].removeprefix('item_'))
        prior.append((hero,[sorted(x) for x in stages]))
        for t in samples:purchases[t].append((hero,sorted({e['key'].removeprefix('item_') for e in timestamped(player.get('purchase_log'),60*t) if isinstance(e.get('key'),str)})))
    return row,samples,failures,prior,purchases

def main():
    out=HERE4/'data'
    if (out/'input_contract.json').exists():print('Preserved completed data');return
    if out.exists():raise RuntimeError('Partial data directory exists; preserve it before deliberate rebuild')
    out.mkdir(parents=True)
    rows=[json.loads(s) for s in (HERE4.parent/'wp2_diagnostics/outputs/audited_matches.jsonl').read_text(encoding='utf-8').splitlines()]
    original={s:[r for r in rows if r['cohort']=='historical' and r['original_split']==s] for s in ('train','validation')}
    boundary=min(r['start_time'] for r in original['validation'])
    shared={r['series_id'] for r in original['train'] if r['series_id'] and r['series_id']>0}&{r['series_id'] for r in original['validation'] if r['series_id'] and r['series_id']>0}
    excluded=[];clean={}
    for split,rs in original.items():
        clean[split]=[]
        for r in rs:
            reason=[]
            if r['series_id'] in shared:reason.append('cross_boundary_series')
            if split=='train' and r['end_time']>=boundary:reason.append('outcome_unavailable_at_validation_start')
            if r['end_time']>=1772668800:reason.append('ends_after_2026_03_05_UTC')
            if reason:excluded.append({'match_id':r['match_id'],'split':split,'reasons':reason})
            else:clean[split].append(r)
    write(out/'split_manifest.json',{'boundary_start':boundary,'shared_series':sorted(shared),'removed':excluded,'train':clean['train'],'validation':clean['validation']})
    outputs={s:{t:[] for t in MINUTES} for s in clean};pr={t:defaultdict(lambda:[Counter(),Counter(),Counter()]) for t in MINUTES};failures=[]
    sequence=clean['train']+clean['validation']
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,(r,samples,failed,prior,purchases) in enumerate(pool.map(one,sequence,chunksize=12),1):
            failures.extend(failed);split=r['original_split']
            for t,s in samples.items():
                outputs[split][t].append((s,purchases[t]))
                if split=='train':
                    for hero,stages in prior:
                        for k,items in enumerate(stages):pr[t][hero][k].update(items)
            if i%1000==0:print('Extracted',i,'/',len(sequence),flush=True)
    counts={};hashes={}
    for t in MINUTES:
        items={h:[[k for k,_ in sorted(c.items(),key=lambda v:(-v[1],v[0]))[:3]] for c in stages] for h,stages in pr[t].items()}
        write(out/f'purchase_prior_{t}m.json',{'items':items,'training_match_ids':[int(s['match_id']) for s,p in outputs['train'][t]],'method':'training_only_top3_per_hero_phase_same_as_WP3'})
        for split in clean:
            samples=[]
            for s,purchases in outputs[split][t]:
                for i,(hero,bought) in enumerate(purchases):
                    s['nodes'][i,10:13]=[len(set(bought)&set(top))/len(top) if top else 0. for top in items.get(hero,[[],[],[]])]
                samples.append(s)
            path=out/f'{split}_{t}m.npz';np.savez_compressed(path,**{k:np.stack([s[k] for s in samples]) for k in samples[0]})
            counts[f'{split}_{t}m']=len(samples);hashes[path.name]=sha(path)
    write(out/'eligibility_failures.json',failures)
    write(out/'input_contract.json',{'schema':'wp4-prefix-v1','counts':counts,'hashes':hashes,'split_manifest_sha256':sha(out/'split_manifest.json'),'prior_pool':'separate per-minute eligible cleaned training pool','outcome_inspected':'historical training and validation only','prior_hashes':{f'{t}m':sha(out/f'purchase_prior_{t}m.json') for t in MINUTES},'source_hashes':{p.name:sha(p) for p in [Path(__file__),WP3/'inputs.py',PROJECT4/'feature_extractor.py',PROJECT4/'v4_objectives.py']}})
    print('DATA COMPLETE',counts,'eligibility failures',len(failures),flush=True)
if __name__=='__main__':main()

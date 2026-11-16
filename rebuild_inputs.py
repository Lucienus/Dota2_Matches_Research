"""Rebuild a frozen training cell from exact raw snapshots, in recorded ID order."""
from pathlib import Path
import argparse,sys,json,hashlib
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'revision_2026-11-11/wp4_training'))
import prepare_development as impl
import numpy as np
def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-dir',type=Path,required=True);p.add_argument('--scope',choices=['broad','listed','broad_matched'],required=True);p.add_argument('--minute',type=int,choices=[10,20,30,40],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Output must be new; partial outputs are retained for inspection')
    records=json.loads((ROOT/'manifests/development_raw.json').read_text());byid={r['match_id']:r for rs in records.values() for r in rs}
    registry=json.loads((ROOT/'manifests/frozen_inputs.json').read_text());ids={s:registry[f'{a.scope}/{s}_{a.minute}m.npz']['ids'] for s in ('train','validation')}
    impl.PROJECT4=a.raw_dir.resolve();buffers={};counters=defaultdict(lambda:[Counter(),Counter(),Counter()])
    for split in ids:
        buffers[split]=[]
        for n,mid in enumerate(ids[split],1):
            row=dict(byid[mid],path=f'{mid}.json',eligible_minutes=[a.minute])
            _,samples,failures,prior,purchases=impl.one(row)
            if failures or a.minute not in samples:raise ValueError(f'Frozen eligibility changed for {mid}; no replacement')
            buffers[split].append((samples[a.minute],purchases[a.minute]))
            if split=='train':
                for hero,stages in prior:
                    for k,items in enumerate(stages):counters[hero][k].update(items)
            if n%1000==0:print(split,n,flush=True)
    prior={h:[[k for k,_ in sorted(c.items(),key=lambda v:(-v[1],v[0]))[:3]] for c in stages] for h,stages in counters.items()}
    a.out.mkdir(parents=True);report={'scope':a.scope,'minute':a.minute,'hashes':{},'matches_reference':{}}
    for split,rs in buffers.items():
        samples=[]
        for s,purchases in rs:
            for i,(hero,bought) in enumerate(purchases):s['nodes'][i,10:13]=[len(set(bought)&set(top))/len(top) if top else 0. for top in prior.get(hero,[[],[],[]])]
            samples.append(s)
        file=a.out/f'{split}_{a.minute}m.npz';np.savez_compressed(file,**{k:np.stack([s[k] for s in samples]) for k in samples[0]})
        digest=hashlib.sha256(file.read_bytes()).hexdigest();report['hashes'][file.name]=digest;report['matches_reference'][split]=digest==registry[f'{a.scope}/{file.name}']['sha256']
    (a.out/'purchase_prior.json').write_text(json.dumps(prior),encoding='utf-8');(a.out/'reconstruction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report));return 0 if all(report['matches_reference'].values()) else 2
if __name__=='__main__':sys.exit(main())

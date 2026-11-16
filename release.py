"""Portable release entry: verification, input audit and selected-recipe retraining."""
from pathlib import Path
import argparse,json,hashlib,sys,csv
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('verify')
    for action in ('check-inputs','train'):
        p=sub.add_parser(action);p.add_argument('--data',type=Path,required=True);p.add_argument('--scope',choices=['broad','listed','broad_matched'],required=True);p.add_argument('--minute',type=int,choices=[10,20,30,40],required=True)
        if action=='train':
            p.add_argument('--model',required=True);p.add_argument('--seed',type=int,default=111001);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.action=='verify':
        manifest=read(ROOT/'checksums.json');bad=[n for n,h in manifest.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
        print(json.dumps({'verified_files':len(manifest),'failures':bad}));return int(bool(bad))
    expected=read(ROOT/'manifests/frozen_inputs.json');bad=[]
    for split in ('train','validation'):
        p=a.data/f'{split}_{a.minute}m.npz';key=f'{a.scope}/{p.name}'
        if not p.is_file():bad.append({'file':str(p),'status':'missing'})
        elif sha(p)!=expected[key]['sha256']:bad.append({'file':str(p),'status':'hash_mismatch'})
    if bad:print(json.dumps({'inputs':'not_verified','details':bad}));return 2
    if a.action=='check-inputs':print('Verified both frozen input files');return 0
    if a.out.exists():raise ValueError('Choose a new output directory; existing runs are preserved')
    prot=read(ROOT/'revision_2026-11-11/wp6_evaluation/resolution_protocol.json')
    if a.seed not in prot['final_seeds']:raise ValueError('Seed is not one of the frozen final seeds')
    rows=[r for r in read(ROOT/'manifests/model_plan.json')['models'] if r['scope']==a.scope and r['minute']==a.minute and r['model']==a.model]
    if len(rows)!=1:raise ValueError('No unique selected recipe for this cell')
    params=rows[0]['params_by_seed'][str(a.seed)]
    if a.seed not in rows[0]['seeds']:raise ValueError('Seed not part of this final model cell')
    sys.path.insert(0,str(ROOT/'revision_2026-11-11/wp4_training'))
    from engine import Data,execute
    result=execute(Data(a.minute,a.device,a.data),a.model,params,a.seed,a.out,epochs=prot['neural_max_epochs'],identity={'release':'access-reproduction-2026-11-16','scope':a.scope,'frozen_inputs_verified':True})
    print(json.dumps({'result':str(a.out/'result.json'),'validation_log_loss':result['validation_log_loss']}));return 0
if __name__=='__main__':sys.exit(main())

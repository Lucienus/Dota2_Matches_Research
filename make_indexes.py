"""Build local path indexes; neither data nor checkpoints enter the release."""
from pipeline_common import *
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--models-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise ValueError('Choose a new index directory')
 train={};models={};missing=[]
 for c in read(CODE/'manifests/model_plan.json')['models']:
  s,t,m=c['scope'],c['minute'],c['model'];base=a.data_root/s/f'{t}m'
  train[f'{s}/{t}']={'train':str((base/f'train_{t}m.npz').resolve()),'prior':str((base/'purchase_prior.json').resolve())}
  for seed in c['seeds']:models[f'{s}/{t}/{m}/{seed}']=str((a.models_root/s/f'{t}m'/m/f'seed{seed}'/('model.pt' if m in ('G','N','flat_mlp') else 'model.joblib')).resolve())
 for path in [v for r in train.values() for v in r.values()]+list(models.values()):
  if not Path(path).is_file():missing.append(path)
 write(a.out/'training_index.json',train);write(a.out/'checkpoint_index.json',models);write(a.out/'missing_files.json',missing)
 print(json.dumps({'training_cells':len(train),'model_files':len(models),'missing':len(missing),'indexes':str(a.out)}))
 if missing:sys.exit(2)
if __name__=='__main__':main()

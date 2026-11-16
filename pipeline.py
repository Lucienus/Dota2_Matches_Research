"""Explicit, isolated holdout reconstruction, inference, scoring and table entry."""
from pipeline_common import *
import argparse,shutil
import numpy as np
def training(index,scope,t):
 r=index[f'{scope}/{t}'];p=Path(r['train'])
 expected=read(CODE/'manifests/frozen_inputs.json')[f'{scope}/train_{t}m.npz']['sha256']
 if sha(p)!=expected:raise ValueError('Training input mismatch: '+str(p))
 return r
def rebuild(a):
 from inputs import extract,hodge_observable
 from feature_extractor import timestamped
 out=a.run/'holdout_inputs'
 if out.exists():raise ValueError('Choose a new run; existing inputs are preserved')
 candidate=read(CODE/'revision_2026-11-11/wp6_evaluation/candidate_manifest.json')
 raw_manifest=read(CODE/'revision_2026-11-11/wp6_evaluation/raw_manifest.json');ex=read(CODE/'manifests/exclusions.json')
 excluded=set(ex['match_ids']);series=set(ex['series_ids']);index=read(a.training_index)
 tiers={int(r['leagueid']):r.get('tier') or 'unmatched' for r in read(CODE/'Dota2_matches/reports/cohort_scope/opendota_leagues.json')}
 buffers={(s,t):[] for s in ('broad','listed','broad_matched') for t in (10,20,30,40)};priors={}
 for s,t in buffers:
  r=training(index,s,t);p=read(Path(r['prior']));priors[s,t]=p if 'items' in p else {'items':p}
  canonical=hashlib.sha256(json.dumps(priors[s,t]['items'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
  if canonical!=read(CODE/'manifests/prior_hashes.json')[f'{s}/{t}']:raise ValueError('Wrong training prior')
 labels={t:[] for t in (10,20,30,40)};metadata=[];failures=[]
 for i,row in enumerate(candidate['matches']):
  mid=int(row['match_id']);assert mid not in excluded;name=f'{mid}.json'
  if name not in raw_manifest['hashes']:
   assert str(mid) in raw_manifest['unavailable'];failures.append({'match_id':mid,'minute':'all','reason':'API_404_or410'});continue
  path=a.raw/name
  if sha(path)!=raw_manifest['hashes'][name]:raise ValueError('Raw hash mismatch: '+name)
  d=read(path);assert all(int(d.get(k,0))==int(row[k]) for k in ('match_id','start_time','leagueid','game_mode','lobby_type'))
  if not isinstance(d.get('radiant_win'),bool):failures.append({'match_id':mid,'minute':'all','reason':'invalid_binary_outcome'});continue
  if d.get('series_id') in series:failures.append({'match_id':mid,'minute':'all','reason':'series_seen_in_audited_development_inventory'});continue
  accepted=[]
  for t in labels:
   try:
    sample=extract(d,t,{});sample.update(hodge_observable=hodge_observable(d,t),match_id=np.int64(mid))
    if not all(np.isfinite(v).all() for v in sample.values()):raise ValueError('Nonfinite inputs')
    scoped={}
    for scope in ('broad','listed','broad_matched'):
     b={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in sample.items()}
     for player_index,player in enumerate(sorted(d['players'],key=lambda x:x['player_slot'])):
      purchases={e['key'].removeprefix('item_') for e in timestamped(player.get('purchase_log'),t*60) if isinstance(e.get('key'),str)}
      stages=priors[scope,t]['items'].get(str(player['hero_id']),[[],[],[]])
      b['nodes'][player_index,10:13]=[len(purchases.intersection(items))/len(items) if items else 0. for items in stages]
     scoped[scope]=b
    for s,b in scoped.items():b.update(hodge_observable=sample['hodge_observable'],match_id=sample['match_id']);buffers[s,t].append(b)
    labels[t].append((mid,int(d['radiant_win'])));accepted.append(t)
   except (ValueError,KeyError,TypeError,IndexError) as e:failures.append({'match_id':mid,'minute':t,'reason':type(e).__name__+': '+str(e)})
  metadata.append({**row,'duration':d['duration'],'series_id':d.get('series_id'),'patch':d.get('patch'),'provider_tier':tiers.get(int(d['leagueid']),'unmatched'),'account_ids':[p.get('account_id') for p in d.get('players',[])],'team_ids':[d.get('radiant_team_id'),d.get('dire_team_id')],'eligible_minutes':accepted})
  if (i+1)%500==0:print('Reconstructed',i+1,flush=True)
 out.mkdir(parents=True);hashes={}
 for (s,t),samples in buffers.items():
  p=out/f'{s}_{t}m.npz';np.savez_compressed(p,**{k:np.stack([r[k] for r in samples]) for k in samples[0]});hashes[p.name]=sha(p)
 for t,rs in labels.items():
  p=out/f'labels_{t}m.npz';np.savez_compressed(p,match_id=np.array([r[0] for r in rs],dtype=np.int64),labels=np.array([r[1] for r in rs],dtype=np.int64));hashes[p.name]=sha(p)
 for name,value in [('metadata.json',metadata),('eligibility_failures.json',failures)]:write(out/name,value);hashes[name]=sha(out/name)
 expected=read(CODE/'manifests/reference_holdout_contract.json')['hashes'];checks={n:h==expected[n] for n,h in hashes.items() if n.endswith('.npz')}
 plan=read(CODE/'manifests/model_plan.json');plan['sources']={str(p.relative_to(CODE)).replace('\\','/'):sha(p) for p in CODE.rglob('*.py') if 'outputs' not in p.parts and '__pycache__' not in p.parts}
 write(a.run/'run_plan.json',plan);write(out/'contract.json',{'run_plan_sha256':sha(a.run/'run_plan.json'),'hashes':hashes,'n_per_minute':{t:len(r) for t,r in labels.items()},'reference_array_checks':checks,'labels_separate':True})
 if not all(checks.values()):raise ValueError('Reconstruction differs from reference; report retained')
 print('All 16 reconstructed array hashes match reference',flush=True)
def predict(a):
 from inference import infer,load_arrays
 plan=read(a.run/'run_plan.json');check_lock(a.run/'run_plan.json');contract=read(a.run/'holdout_inputs/contract.json');index=read(a.training_index);models=read(a.checkpoints);done={}
 for n,h in contract['hashes'].items():assert sha(a.run/'holdout_inputs'/n)==h
 for c in plan['models']:
  s,t,m=c['scope'],c['minute'],c['model'];arrays=load_arrays(a.run/'holdout_inputs'/f'{s}_{t}m.npz');assert not {'outcome','labels'}&arrays.keys();r=training(index,s,t)
  for seed,ref in zip(c['seeds'],c['reference_checkpoints']):
   model=Path(models[f'{s}/{t}/{m}/{seed}']);digest=sha(model)
   if digest!=ref['sha256'] and not a.allow_retrained:raise ValueError('Checkpoint differs; use --allow-retrained for a separately labelled rerun')
   p=infer(arrays,Path(r['train']),model,m,t,a.device if m in ('G','N','flat_mlp') else 'cpu');out=a.run/'predictions'/s/f'{t}m'/m/f'seed{seed}.npz';out.parent.mkdir(parents=True,exist_ok=True)
   if out.exists():raise ValueError('Existing prediction preserved: '+str(out))
   np.savez_compressed(out,match_id=arrays['match_id'],probabilities=p);done[out.relative_to(a.run).as_posix()]=sha(out)
  print('Predicted',s,t,m,flush=True)
 write(a.run/'predictions_complete.json',{'run_plan_sha256':sha(a.run/'run_plan.json'),'input_contract_sha256':sha(a.run/'holdout_inputs/contract.json'),'hashes':done,'allow_retrained':a.allow_retrained,'labels_read_for_inference':False})
def summarize(a):
 import pipeline_common as context,summary_engine as engine
 context.HERE6=a.run;context.ROOT6=a.run;engine.HERE6=a.run;engine.ROOT6=a.run;engine.main()
def tables(a):
 import table_engine as e
 e.CODE=CODE;e.REPORT=a.run/'reports';e.TABLES=a.run/'tables';e.EVIDENCE=a.run/'table_sources'
 shutil.copy2(CODE/'revision_2026-11-11/wp6_evaluation/reports/resolution_parameters.csv',e.REPORT/'resolution_parameters.csv');e.main()
 from supplement_tables import export
 export(a.run)
def main():
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='action',required=True)
 for action in ('rebuild-holdout','predict','summarize','tables'):
  q=sp.add_parser(action);q.add_argument('--run',type=Path,required=True)
  if action in ('rebuild-holdout','predict'):q.add_argument('--training-index',type=Path,required=True)
  if action=='rebuild-holdout':q.add_argument('--raw',type=Path,required=True)
  if action=='predict':q.add_argument('--checkpoints',type=Path,required=True);q.add_argument('--device',choices=['cpu','cuda'],default='cpu');q.add_argument('--allow-retrained',action='store_true')
 a=p.parse_args();a.run=a.run.resolve();{'rebuild-holdout':rebuild,'predict':predict,'summarize':summarize,'tables':tables}[a.action](a)
if __name__=='__main__':main()

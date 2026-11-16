from common5 import *
import numpy as np
from collections import Counter

def main():
    if (HERE5/'protocol.json').exists():raise RuntimeError('Protocol already frozen')
    search=read(WP4/'search_protocol.json');manifest=read(WP4/'data/split_manifest.json');inventory={r['match_id']:r for s in ('train','validation') for r in manifest[s]}
    scores=[]
    for name in ('random_forest','xgboost','lightgbm'):
        s=read(WP4/f'runs/20m/{name}/selection.json');score=next(r['mean_validation_log_loss'] for r in s['repeated_search_scores'] if r['candidate_id']==s['selected_candidate']);scores.append((score,name))
    tree=min(scores)[1];pools={};matching=[]
    for t in (20,10,30,40):
        with np.load(WP4/f'data/train_{t}m.npz') as a:broad=[int(i) for i in a['match_id']]
        with np.load(WP4/f'data/validation_{t}m.npz') as a:val=[int(i) for i in a['match_id']]
        listed=[i for i in broad if inventory[i]['frozen_tier'] in ('professional','premium')]
        counts=Counter(inventory[i]['month'] for i in listed);matched=[]
        for month,n in sorted(counts.items()):
            candidates=[i for i in broad if inventory[i]['month']==month]
            chosen=sorted(candidates,key=lambda i:hashlib.sha256(f'wp5-scope-v1:{month}:{i}'.encode()).hexdigest())[:n]
            assert len(chosen)==n;matched+=chosen
            matching.append({'minute':t,'month':month,'listed_n':n,'matched_n':len(chosen),'broad_n':len(candidates),'listed_matched_overlap':len(set(chosen)&set(listed))})
        pools[str(t)]={'broad':broad,'listed':listed,'broad_matched':matched,'validation':val}
    protocol={'version':'wp5-scope-v1','declared_date':'2026-11-13','models':['G','N',tree],'tree_selection':{'rule':'minimum WP4 20min repeated search validation loss; lexicographic name tie break','scores':scores,'selected':tree},'minutes':[20,10,30,40],'training_pools':['broad','listed','broad_matched'],'listed_definition':'frozen provider tier professional OR premium; not independent skill verification','allowed_game_modes':sorted({r['game_mode'] for r in manifest['train']+manifest['validation']}),'eligibility':'exact WP4 per-minute common eligibility; same allowed game modes in every pool','target':'same complete WP4 validation IDs for every pool; listed validation subset secondary; not unused holdout','matching':'per UTC month counts; SHA256 wp5-scope-v1:<month>:<id>; no outcome matching; overlap allowed','separate_preprocessing_and_tuning':True,'broad_reuse':'exact WP4 inputs/preprocessing/search/target; reuse all 50 fits per model/time','new_fits':1200,'reused_fits':600,'scope_difference':'listed minus broad_matched; listed minus broad secondary','seed_policy':search['seeds'],'candidate_source_sha256':sha(WP4/'search_protocol.json'),'subgroups':['provider tier','month','patch','seen/unseen/unknown player','seen/unseen/unknown team'],'overlap_reference':'training pool for development validation only; validation already used for tuning, not unseen relative to all development','intervals':'paired seed t across final fits, conditional on fixed validation; n>=200 and both classes>=20; no p values; no match-iid bootstrap','official_unseen_generalization':'deferred to unused holdout; this phase describes validation transfer conditional on fitted pool','cautions':['monthly matching does not balance all competitive strength or game-mode confounders','WP4 boundary flags remain open; no heldout scoring enabled'],'manifest_sha256':sha(WP4/'data/split_manifest.json')}
    write(HERE5/'protocol.json',protocol);write(HERE5/'pool_ids.json',pools);csvout('monthly_matching.csv',matching)
    write(HERE5/'source_lock.json',{'WP4_execution_lock':sha(WP4/'execution_lock.json'),'protocol':sha(HERE5/'protocol.json'),'pool_ids':sha(HERE5/'pool_ids.json')})
    print('Frozen tree',tree,'new fits 1200; reused 600',flush=True)
if __name__=='__main__':main()

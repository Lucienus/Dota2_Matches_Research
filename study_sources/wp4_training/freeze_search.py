from common4 import *
import numpy as np
import platform,importlib.metadata

def main():
    path=HERE4/'search_protocol.json'
    if path.exists():print('Preserved frozen search');return
    rng=np.random.default_rng(110111)
    def draw(space):return {k:v[int(rng.integers(len(v)))] for k,v in space.items()}
    def unique(space):
        result=[];seen=set()
        while len(result)<32:
            p=draw(space);key=json.dumps(p,sort_keys=True)
            if key not in seen:seen.add(key);result.append(p)
        return result
    neural=dict(learning_rate=[.0001,.0003,.001,.003],weight_decay=[0.,.00001,.0001,.001,.01],dropout=[0.,.1,.2,.4],branch_width=[32,64,128],batch_size=[128,256,512])
    lr=[{'C':float(c)} for c in np.geomspace(1e-8,1e2,32)]
    forest=dict(n_estimators=[300,600],max_depth=[None,8,16,32],min_samples_leaf=[1,2,5,10,20],max_features=['sqrt',.3,.7,1.])
    boosting=dict(n_estimators=[2000],learning_rate=[.01,.03,.1],max_depth=[3,5,8],min_child_weight=[1,5,20],subsample=[.7,.85,1.],colsample_bytree=[.7,.85,1.],reg_lambda=[.1,1.,10.,100.],reg_alpha=[0.,.1,1.])
    light=dict(n_estimators=[2000],learning_rate=[.01,.03,.1],num_leaves=[7,15,31,63],max_depth=[-1,4,8],min_child_samples=[10,30,100],subsample=[.7,.85,1.],subsample_freq=[1],colsample_bytree=[.7,.85,1.],reg_lambda=[0.,1.,10.,100.],reg_alpha=[0.,.1,1.],max_bin=[63,127,255])
    ng=unique(neural);lg=unique(light)
    candidates={'G':ng,'N':ng,'flat_mlp':unique(neural),'gold_xp_lr':lr,'full_feature_lr':lr,'random_forest':unique(forest),'xgboost':unique(boosting),'lightgbm':lg,'hodge_adapted':lg}
    wp1=read(HERE4.parent/'wp1_protocol/protocol.json')
    protocol={'version':'wp4-search-v1','frozen_before_any_new_validation_scores':True,'date':'2026-11-11','candidate_generation_seed':110111,
        'minutes':[20,10,30,40],'candidates':candidates,'spaces':{'neural':neural,'LR_C':[1e-8,1e2],'RF':forest,'XGB':boosting,'LGBM':light},
        'seeds':wp1['tuning'],'selection':'mean validation raw natural log loss of 3 search seeds among first-seed top4; LR deterministic choose among32; ascending candidate ID tie break',
        'fixed_neural':{'shared_temporal_width':64,'epochs':150,'patience':15,'optimizer':'AdamW','gradient_clip':5.,'dtype':'float32','aux_loss':False,'input_scaling':'WP3 fixed original numeric scales; no fitted extra neural normalization'},
        'fixed_LR':{'max_iter':5000,'tol':1e-6,'solver':'lbfgs','numeric_only_scaling':True},'boosting_patience':50,
        'failed_fit_policy':'retain failure; stop family selection until resolved under same candidate and seed; no silent omission',
        'boundary_check':'flag selected numeric coordinate on sampled range endpoint; finite capacity categories reported separately; validation only plan amendment before heldout',
        'budget_curve':'candidate index and cumulative wall time vs best first-stage validation loss; additional-seed selection separate',
        'deterministic_families':['gold_xp_lr','full_feature_lr'],'stochastic_families':['G','N','flat_mlp','random_forest','xgboost','lightgbm','hodge_adapted'],
        'planned_fit_count':{'stage1':1152,'search_repeats':224,'final_fits':288,'constants':8,'total_excluding_LR_feature_controls':1672},
        'LR_feature_controls':'selected full LR C held fixed: full, without heroes, without edges, gold/XP-only; training/validation same; deterministic; no test',
        'formal_holdout_scoring_enabled':False,'wp1_sha256':sha(HERE4.parent/'wp1_protocol/protocol.json'),'wp3_manifest_sha256':sha(WP3/'manifest.json'),
        'environment':{'python':platform.python_version(),**{n:importlib.metadata.version(n) for n in ['numpy','torch','scikit-learn','xgboost','lightgbm','joblib']}}}
    write(path,protocol);print('Frozen 9 families x 32 candidates x 4 times; 1672 planned fits')
if __name__=='__main__':main()

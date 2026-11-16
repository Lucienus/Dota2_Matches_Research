from runtime import *
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class ContinuousScaler(BaseEstimator,TransformerMixin):
    """Scale numerical columns; leave availability and hero one-hot indicators unchanged."""
    def __init__(self,indices):self.indices=indices
    def fit(self,x,y=None):self.scaler_=StandardScaler().fit(x[:,self.indices]);return self
    def transform(self,x):
        z=np.asarray(x,dtype=np.float64).copy();z[:,self.indices]=self.scaler_.transform(z[:,self.indices]);return z

class Prior(BaseEstimator):
    def __init__(self,half=False):self.half=half
    def fit(self,x,y):
        y=np.asarray(y)
        if len(y)==0 or not np.isin(y,[0,1]).all():raise ValueError('Binary training labels required')
        self.p_=.5 if self.half else float(y.mean());self.classes_=np.array([0,1]);return self
    def predict_proba(self,x):return np.tile([1-self.p_,self.p_],(len(x),1))

def make_baseline(name,minute,seed=17,**params):
    if name in ('empirical_prior','constant_half'):return Prior(name=='constant_half')
    if name in ('gold_xp_lr','full_feature_lr'):
        continuous=list(range(2)) if name=='gold_xp_lr' else list(range(200))+list(range(270,470+4*(minute+1)))
        fixed=dict(C=.001,max_iter=5000,tol=1e-6,solver='lbfgs',random_state=seed);fixed.update(params)
        return make_pipeline(ContinuousScaler(continuous),LogisticRegression(**fixed))
    if name=='random_forest':
        fixed=dict(n_estimators=300,min_samples_leaf=5,max_features='sqrt',n_jobs=4,random_state=seed);fixed.update(params);return RandomForestClassifier(**fixed)
    if name in ('lightgbm','hodge_adapted'):
        from lightgbm import LGBMClassifier
        fixed=dict(n_estimators=2000,num_leaves=31,learning_rate=.03,min_child_samples=20,verbosity=-1,n_jobs=4,random_state=seed);fixed.update(params);return LGBMClassifier(**fixed)
    if name=='xgboost':
        from xgboost import XGBClassifier
        fixed=dict(n_estimators=2000,max_depth=4,learning_rate=.03,tree_method='hist',eval_metric='logloss',early_stopping_rounds=50,n_jobs=4,random_state=seed);fixed.update(params);return XGBClassifier(**fixed)
    raise ValueError(name)

def fit_baseline(model,name,x,y,vx,vy):
    if name in ('lightgbm','hodge_adapted'):
        from lightgbm import early_stopping
        return model.fit(x,y,eval_set=[(vx,vy)],eval_metric='binary_logloss',callbacks=[early_stopping(50,verbose=False)])
    if name=='xgboost':return model.fit(x,y,eval_set=[(vx,vy)],verbose=False)
    return model.fit(x,y)

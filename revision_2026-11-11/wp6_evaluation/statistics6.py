"""Prespecified numerical analysis, with no file access or model selection."""
import numpy as np
from scipy.stats import t

MARGINS=(.0025,.005,.01)
def losses(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if y.ndim!=1 or p.ndim!=2 or p.shape[1]!=len(y) or not len(y):raise ValueError('Expected nonempty y[N], p[seeds,N]')
    if not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid labels/probabilities')
    q=np.clip(p,1e-7,1-1e-7)
    return -y[None,:]*np.log(q)-(1-y[None,:])*np.log1p(-q)

def seed_interval(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Invalid replicate values')
    out={'mean':float(a.mean()),'seed_count':len(a),'seed_sd':None,'L':None,'U':None,'df':None}
    if len(a)>1:
        sd=float(a.std(ddof=1));half=float(t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)))
        out.update(seed_sd=sd,L=float(a.mean()-half),U=float(a.mean()+half),df=len(a)-1)
    return out

def decision(lo,hi,margin):
    if lo is None or hi is None:return 'interval_unavailable'
    if not np.isfinite([lo,hi,margin]).all() or lo>hi or margin<=0:raise ValueError('Invalid interval/margin')
    if hi < -margin:return 'G_practical_gain'
    if lo > margin:return 'N_practical_gain'
    if lo > -margin and hi < margin:return 'operational_equivalence'
    return 'unresolved'

def paired(y,left,right,left_ids,right_ids,left_seeds,right_seeds,margin=.005):
    if not np.array_equal(left_ids,right_ids) or list(left_seeds)!=list(right_seeds):raise ValueError('Paired IDs/seeds differ')
    d=losses(y,left)-losses(y,right)
    summary=seed_interval(d.mean(axis=1))
    summary.update(difference='left_minus_right_log_loss',n_matches=len(y),seed_values=d.mean(axis=1).tolist(),
      geometric_true_class_probability_ratio_left_over_right=float(np.exp(-summary['mean'])),
      conditional_on_fixed_data_and_selected_parameters=True,includes_population_uncertainty=False,
      decisions={str(m):decision(summary['L'],summary['U'],m) for m in MARGINS},primary_margin=margin,
      leave_one_seed_out_means=[float(np.delete(d.mean(1),i).mean()) for i in range(len(d))] if len(d)>1 else [])
    return summary

def block_sensitivity(values,groups,universe=None,replicates=20000,seed=731,minimum_groups=20):
    """Resample nonoverlapping whole blocks; use ratio of loss sums to match counts.

    Pass the same ordered universe at every cutoff to share block draws across
    models/cutoffs. Empty groups at a cutoff contribute zero, not fictitious matches.
    This describes sampling sensitivity; it does not eliminate cross-block players.
    """
    x=np.asarray(values,dtype=float)
    if x.ndim==1:x=x[:,None]
    g=list(groups)
    if len(g)!=len(x) or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid block inputs')
    levels=list(universe) if universe is not None else sorted(set(g))
    if len(levels)!=len(set(levels)) or not set(g)<=set(levels):raise ValueError('Invalid block universe')
    lookup={v:i for i,v in enumerate(levels)};ix=np.array([lookup[v] for v in g]);counts=np.bincount(ix,minlength=len(levels))
    sums=np.zeros((len(levels),x.shape[1]));np.add.at(sums,ix,x)
    occupied=int((counts>0).sum())
    result={'point':x.mean(0).tolist(),'nonempty_blocks':occupied,'universe_blocks':len(levels),
            'L':None,'U':None,'replicates':0,'invalid_empty_draws':0,'method':'paired whole-block ratio bootstrap percentile95'}
    if occupied<minimum_groups:
        result['status']='insufficient_nonempty_blocks';return result
    rng=np.random.default_rng(seed);draws=[];invalid=0
    for start in range(0,replicates,256):
        weights=rng.multinomial(len(levels),np.full(len(levels),1/len(levels)),size=min(256,replicates-start))
        denominator=weights@counts;valid=denominator>0;invalid+=int((~valid).sum())
        draws.append((weights[valid]@sums)/denominator[valid,None])
    distribution=np.concatenate(draws)
    result.update(L=np.quantile(distribution,.025,axis=0).tolist(),U=np.quantile(distribution,.975,axis=0).tolist(),
                  replicates=len(distribution),invalid_empty_draws=invalid,status='descriptive_sensitivity')
    return result

def time_blocks(start_times,days,origin=1772668800):
    stamps=np.asarray(start_times,dtype=np.int64)
    if np.any(stamps<origin):raise ValueError('Before frozen holdout start')
    return ((stamps-origin)//(days*86400)).tolist()

def equal_group_seed_values(loss_matrix,groups):
    """Equal-month weighting: average each observed month's mean, then months."""
    a=np.asarray(loss_matrix,dtype=float);g=np.asarray(groups)
    if a.ndim!=2 or len(g)!=a.shape[1]:raise ValueError('Shape mismatch')
    return np.stack([a[:,g==v].mean(1) for v in np.unique(g)]).mean(0)

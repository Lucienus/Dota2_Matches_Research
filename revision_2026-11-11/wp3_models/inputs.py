"""Versioned optional-field masks and explicitly adapted Hodge window."""
from runtime import *
import copy
import numpy as np
from feature_extractor import value_at,timestamped,hero_metadata
from v4_objectives import TowerCorrectedExtractor

OPTIONAL=('lh_t','dn_t','obs_log','sen_log','runes_log','kills_log','purchase_log')
def extract(raw,minute,prior):
    d=copy.deepcopy(raw);mask=np.ones((10,7),dtype=np.float32)
    players=sorted(d['players'],key=lambda p:p['player_slot'])
    for i,p in enumerate(players):
        for j,k in enumerate(OPTIONAL):
            if k.endswith('_t'):
                try:value_at(p,k,minute*60)
                except ValueError:mask[i,j]=0
            else:
                value=p.get(k)
                if value is None:mask[i,j]=0;p[k]=[]
                elif not isinstance(value,list) or any(not isinstance(e,dict) or not isinstance(e.get('time'),(int,float)) or not np.isfinite(e['time']) for e in value):
                    raise ValueError('Malformed optional log '+k)
    result=TowerCorrectedExtractor(d,minute,prior).extract_inputs()
    result['availability']=mask
    return result

class HeroVocabulary:
    """Zero is the predeclared unknown token; fit only on training hero IDs."""
    def fit(self,heroes):
        self.ids=sorted(set(np.asarray(heroes).ravel().tolist()));self.mapping={h:i+1 for i,h in enumerate(self.ids)};return self
    def transform(self,heroes):return np.asarray([[self.mapping.get(int(h),0) for h in row] for row in heroes],dtype=np.int64)
    @property
    def size(self):return len(self.ids)+1

def hodge_window(team_values,minute):
    """[t-5..t, Radiant/Dire, metrics] -> five timestamps x (R,D,R-D,dR,dD).
    Missing original metrics must not be silently fabricated.
    """
    a=np.asarray(team_values,dtype=np.float64)
    if a.ndim!=3 or a.shape[:2]!=(6,2) or not np.isfinite(a).all():raise ValueError('Expected finite (6,2,metrics)')
    r,d=a[1:,0],a[1:,1];dr,dd=np.diff(a[:,0],axis=0),np.diff(a[:,1],axis=0)
    return np.concatenate([np.stack([r,d,r-d,dr,dd],axis=-1).ravel(),np.arange(minute-4,minute+1)])

def hodge_observable(raw,minute):
    """Observable-subset adaptation: enemy-hero kills, last hits, XP; no gold proxy."""
    players=sorted(raw['players'],key=lambda p:p['player_slot']);meta=hero_metadata();values=[]
    for t in range(minute-5,minute+1):
        teams=[]
        for side in (0,1):
            enemy={meta[str(p['hero_id'])]['name'] for p in players[(1-side)*5:(2-side)*5]}
            kills=lh=xp=0.
            for p in players[side*5:(side+1)*5]:
                if not isinstance(p.get('kills_log'),list):raise ValueError('Hodge subset requires observed kill log')
                kills+=sum(e.get('key') in enemy for e in timestamped(p['kills_log'],t*60))
                lh+=value_at(p,'lh_t',t*60);xp+=value_at(p,'xp_t',t*60)
            teams.append([kills,lh,xp])
        values.append(teams)
    return hodge_window(values,minute)

def flatten(batch,num_heroes):
    import torch
    # Fixed-width one-hot includes the declared UNK column.
    hero=torch.nn.functional.one_hot(batch['heroes'],num_heroes).float().flatten(1)
    return torch.cat([batch['nodes'].flatten(1),batch['availability'].flatten(1),batch['edges'].flatten(1),batch['sequence'].flatten(1),hero],1)

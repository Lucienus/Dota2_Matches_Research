"""Read-only audits of frozen cohorts and actual bidirectional attention support."""
from freeze_additional_experiment import HERE, REV, ROOT, read, write, sha, csvout
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import json
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(REV/'wp3_models'))
from controlled_models import ControlledAttention
from inputs import extract
import torch

OUT = HERE/'reports'
CHECKS = []
SOURCES = {}

def check(name, condition):
    CHECKS.append({'check':name, 'passed':bool(condition)})
    if not condition:
        raise AssertionError(name)

def source(path, expected=None):
    h = sha(path)
    SOURCES[Path(path).relative_to(ROOT).as_posix()] = h
    if expected is not None:
        check('hash:'+str(Path(path).relative_to(ROOT)), h==expected)
    return path

def utc(t):
    return datetime.fromtimestamp(float(t), timezone.utc).isoformat()

def support(edges):
    incoming = np.any(np.abs(edges)>0, axis=-1)
    adjacency = incoming | incoming.transpose(0,2,1)
    offdiag = adjacency & ~np.eye(10, dtype=bool)[None]
    return incoming, offdiag, offdiag | np.eye(10,dtype=bool)[None]

def audit_model_support():
    # Empty graph, one directed event, and a dense graph distinguish conventions.
    edges = np.zeros((3,10,10,2), np.float32)
    edges[1,1,7,0] = 1
    edges[2,:,:,0] = 1
    _, off, mask = support(edges)
    check('synthetic:empty graph has no off-diagonal edges',not off[0].any())
    check('synthetic:one directed event creates exactly two directed support entries',int(off[1].sum())==2)
    check('synthetic:dense graph has 90 off-diagonal support entries',int(off[2].sum())==90)
    check('synthetic:reverse event visible to both endpoints',mask[1,1,7] and mask[1,7,1])
    layer = ControlledAttention(3,4,heads=2,direction='bidirectional',weighting='none').eval()
    with torch.no_grad():
        _, weights = layer(torch.zeros((3,10,3)),torch.tensor(edges),torch.ones((3,10)),True)
    check('synthetic:actual attention mask matches audit',np.array_equal(weights.numpy()[...,0]>0,mask))
    check('synthetic:attention normalized over source nodes',np.allclose(weights.numpy().sum(2),1))

def profile(cohort, minute, rows, summaries, profiles, cross):
    n = len(rows)
    wins = sum(r['radiant_win'] for r in rows)
    listed = sum(r['frozen_tier'] in ('professional','premium') for r in rows)
    known = sum(r['frozen_tier']!='unmatched' for r in rows)
    summaries.append({'cohort':cohort,'minute':minute,'n':n,'radiant_wins':wins,'radiant_win_rate':wins/n,
                      'listed_n':listed,'listed_fraction':listed/n,'known_provider_fraction':known/n,
                      'league_count':len({r['league_id'] for r in rows}),
                      'match_start_min_utc':utc(min(r['start_time'] for r in rows)),
                      'match_start_max_utc':utc(max(r['start_time'] for r in rows))})
    axes = ('month','game_mode','lobby_type','frozen_tier','patch','league_id','region','cluster','skill','rank_tier','avg_rank_tier','player_rank_tier_count')
    for axis in axes:
        groups = defaultdict(list)
        for r in rows:
            value = r.get(axis)
            groups['MISSING' if value is None else str(value)].append(r)
        check(f'{cohort}:{minute}:{axis}:denominators',sum(len(rs) for rs in groups.values())==n)
        for label, rs in sorted(groups.items()):
            profiles.append({'cohort':cohort,'minute':minute,'axis':axis,'value':label,'n':len(rs),
                             'fraction_of_cohort':len(rs)/n,'radiant_wins':sum(r['radiant_win'] for r in rs),
                             'radiant_win_rate':sum(r['radiant_win'] for r in rs)/len(rs)})
    groups = defaultdict(list)
    for r in rows:
        groups[(r['month'],r['game_mode'],r['frozen_tier'])].append(r)
    for (month,mode,tier),rs in sorted(groups.items()):
        cross.append({'cohort':cohort,'minute':minute,'month':month,'game_mode':mode,'provider_tier':tier,
                      'n':len(rs),'radiant_wins':sum(r['radiant_win'] for r in rs)})

def graphs(cohort, minute, ids, edges, summaries, histograms, matches):
    n = len(ids)
    check(f'{cohort}:{minute}:graph shape',edges.shape==(n,10,10,2))
    check(f'{cohort}:{minute}:finite nonnegative edges',np.isfinite(edges).all() and (edges>=0).all())
    incoming, off, mask = support(edges)
    degree = off.sum(2)
    density = off.sum((1,2))/90
    side = np.arange(10)//5
    cross = side[:,None] != side[None,:]
    same = ~cross & ~np.eye(10,dtype=bool)
    crossdensity = (off & cross[None]).sum((1,2))/50
    samedensity = (off & same[None]).sum((1,2))/40
    twohop = ((mask.astype(np.int16) @ mask.astype(np.int16))>0) & ~np.eye(10,dtype=bool)[None]
    check(f'{cohort}:{minute}:undirected support symmetry',np.array_equal(off,off.transpose(0,2,1)))
    check(f'{cohort}:{minute}:all support self loops',np.diagonal(mask,axis1=1,axis2=2).all())
    check(f'{cohort}:{minute}:degree density identity',np.allclose(degree.mean(1)/9,density))
    check(f'{cohort}:{minute}:twohop contains onehop',np.all(twohop|~off))
    check(f'{cohort}:{minute}:cross/same density identity',np.allclose(density,(50*crossdensity+40*samedensity)/90))
    layer = ControlledAttention(3,4,heads=2).eval()
    with torch.no_grad():
        _, a = layer(torch.zeros((min(n,5),10,3)),torch.tensor(edges[:5]),torch.ones((min(n,5),10)),True)
    check(f'{cohort}:{minute}:actual attention supports',np.array_equal(a.numpy()[...,0]>0,mask[:5]))
    q = np.quantile(density,[.05,.5,.95])
    summaries.append({'cohort':cohort,'minute':minute,'matches':n,'nodes':10*n,
                      'mean_offdiagonal_degree':float(degree.mean()),'mean_neighbors_including_self':float(degree.mean()+1),
                      'self_only_nodes':int((degree==0).sum()),'self_only_node_fraction':float((degree==0).mean()),
                      'matches_with_self_only_node_fraction':float((degree==0).any(1).mean()),
                      'density_mean':float(density.mean()),'density_p05':float(q[0]),'density_median':float(q[1]),'density_p95':float(q[2]),
                      'cross_team_density_mean':float(crossdensity.mean()),'same_team_density_mean':float(samedensity.mean()),
                      'matches_with_same_team_edges':int((samedensity>0).sum()),
                      'fraction_density_at_least_0_9':float((density>=.9).mean()),
                      'raw_diagonal_nonzero_entries':int(np.diagonal(incoming,axis1=1,axis2=2).sum()),
                      'mean_twohop_other_nodes':float(twohop.sum(2).mean())})
    for d in range(10):
        histograms.append({'cohort':cohort,'minute':minute,'offdiagonal_degree':d,'node_count':int((degree==d).sum()),'fraction':float((degree==d).mean())})
    check(f'{cohort}:{minute}:degree histogram sums to 10n',sum(int((degree==d).sum()) for d in range(10))==10*n)
    for i,mid in enumerate(ids):
        matches.append({'cohort':cohort,'minute':minute,'match_id':int(mid),'density':float(density[i]),
                        'cross_team_density':float(crossdensity[i]),'same_team_density':float(samedensity[i]),
                        'self_only_nodes':int((degree[i]==0).sum()),'mean_offdiag_degree':float(degree[i].mean()),
                        'mean_twohop_other_nodes':float(twohop[i].sum(1).mean())})

def main():
    for p,h in read(HERE/'protocol_lock.json')['hashes'].items():
        source(ROOT/p,h)
    audit_model_support()
    dev = read(REV/'wp4_training/data/split_manifest.json')
    meta = {r['match_id']:r for split in ('train','validation') for r in dev[split]}
    poolids = read(REV/'wp5_scope/pool_ids.json')
    contract = read(source(REV/'wp4_training/data/input_contract.json'))
    holdcontract = read(source(REV/'wp6_evaluation/holdout_inputs/contract.json'))
    holdmeta = read(source(REV/'wp6_evaluation/holdout_inputs/metadata.json',holdcontract['hashes']['metadata.json']))
    holdmeta = {r['match_id']:r for r in holdmeta}
    rawmanifest = read(source(REV/'wp6_evaluation/raw_manifest.json',holdcontract['raw_manifest_sha256']))
    candidate = read(source(REV/'wp6_evaluation/candidate_manifest.json',rawmanifest['candidate_manifest_sha256']))
    frozen_leagues = read(source(ROOT/'Dota2_matches/reports/cohort_scope/opendota_leagues.json'))
    tiers = {int(r['leagueid']):r.get('tier') or 'unmatched' for r in frozen_leagues}
    for r in meta.values():
        check(f"tier mapping development:{r['match_id']}",r['frozen_tier']==tiers.get(r['league_id'],'unmatched'))
    rawmeta = {}
    for index,r in enumerate(candidate['matches'],1):
        mid=int(r['match_id']);p=REV/'wp6_evaluation/raw_holdout'/f'{mid}.json'
        d=read(source(p,rawmanifest['hashes'][p.name]))
        check(f'candidate metadata:{mid}',all(d[k]==r[k] for k in ('match_id','start_time','leagueid','game_mode','lobby_type')))
        item={**r,'league_id':d['leagueid'],'month':utc(d['start_time'])[:7], 'patch':d.get('patch'),
              'frozen_tier':tiers.get(d['leagueid'],'unmatched'),'radiant_win':int(d['radiant_win']),
              **{k:d.get(k) for k in ('region','cluster','skill','rank_tier','avg_rank_tier')},
              'player_rank_tier_count':sum(p.get('rank_tier') is not None for p in d['players'])}
        rawmeta[mid]=item
        if mid in holdmeta:
            check(f'holdout frozen category:{mid}',holdmeta[mid]['provider_tier']==item['frozen_tier'])
        if index%1500==0: print('Raw metadata and hash audit',index,'/',len(candidate['matches']),flush=True)
    summaries=[];profiles=[];cross=[];gs=[];hist=[];matches=[];selection=[];subgroups=[]
    for t in (10,20,30,40):
        for split in ('train','validation'):
            path=REV/f'wp4_training/data/{split}_{t}m.npz'
            source(path,contract['hashes'][path.name])
            with np.load(path,allow_pickle=False) as a:
                ids=a['match_id'];edges=a['edges'];labels=a['outcome']
            check(f'{split}:{t}:unique IDs',len(set(ids.tolist()))==len(ids))
            check(f'{split}:{t}:label metadata agreement',all(meta[int(i)]['radiant_win']==int(y) for i,y in zip(ids,labels)))
            scopes=('broad','listed','broad_matched') if split=='train' else ('validation',)
            for scope in scopes:
                desired=poolids[str(t)][scope]
                ix={int(mid):j for j,mid in enumerate(ids)}
                positions=[ix[mid] for mid in desired]
                rows=[meta[i] for i in desired]
                name=f'train_{scope}' if split=='train' else 'validation'
                profile(name,t,rows,summaries,profiles,cross)
                graphs(name,t,desired,edges[positions],gs,hist,matches)
                if split=='train' and scope!='broad':
                    p=REV/f'wp5_scope/data/{scope}/train_{t}m.npz'
                    old={k.replace('\\','/'):v for k,v in read(REV/'wp5_scope/data/complete.json')['hashes'].items()}
                    source(p,old[str(p.relative_to(REV/'wp5_scope')).replace('\\','/')])
                    with np.load(p) as a:
                        check(f'{scope}:{t}:scope graph invariant',np.array_equal(a['match_id'],desired) and np.array_equal(a['edges'],edges[positions]))
            for i in (0,len(ids)//2,len(ids)-1):
                r=meta[int(ids[i])];p=ROOT/'Dota2_matches'/r['path'];d=read(source(p,r['sha256']))
                check(f'{split}:{t}:raw edge replay:{ids[i]}',np.array_equal(extract(d,t,{})['edges'],edges[i]))
        path=REV/f'wp6_evaluation/holdout_inputs/broad_{t}m.npz'
        source(path,holdcontract['hashes'][path.name])
        with np.load(path) as a:ids=a['match_id'];edges=a['edges']
        lp=REV/f'wp6_evaluation/holdout_inputs/labels_{t}m.npz';source(lp,holdcontract['hashes'][lp.name])
        with np.load(lp) as a:
            check(f'holdout:{t}:label IDs',np.array_equal(ids,a['match_id']))
            check(f'holdout:{t}:raw labels agree',all(rawmeta[int(i)]['radiant_win']==int(y) for i,y in zip(ids,a['labels'])))
        rows=[rawmeta[int(i)] for i in ids]
        check(f'holdout:{t}:n',len(ids)==holdcontract['n_per_minute'][str(t)])
        check(f'holdout:{t}:no development overlap',not set(ids.tolist())&set(meta))
        profile('evaluation',t,rows,summaries,profiles,cross)
        graphs('evaluation',t,ids,edges,gs,hist,matches)
        for tier in ('professional','premium','excluded','unmatched','listed'):
            ix=[j for j,r in enumerate(rows) if (r['frozen_tier'] in ('professional','premium') if tier=='listed' else r['frozen_tier']==tier)]
            if ix:
                graphs('evaluation_'+tier,t,ids[ix],edges[ix],subgroups,[],[])
        for scope in ('listed','broad_matched'):
            p=REV/f'wp6_evaluation/holdout_inputs/{scope}_{t}m.npz';source(p,holdcontract['hashes'][p.name])
            with np.load(p) as a:
                check(f'holdout:{scope}:{t}:same graphs',np.array_equal(a['match_id'],ids) and np.array_equal(a['edges'],edges))
        for i in (0,len(ids)//2,len(ids)-1):
            d=read(REV/f'wp6_evaluation/raw_holdout/{ids[i]}.json')
            check(f'holdout:{t}:raw edge replay:{ids[i]}',np.array_equal(extract(d,t,{})['edges'],edges[i]))
        for scope in ('broad','listed','broad_matched'):
            ids0=poolids[str(t)][scope]
            cm=[meta[mid] for mid in ids0 if meta[mid]['game_mode']==2]
            selection.append({'cohort':'train_'+scope,'minute':t,'all_n':len(ids0),'captains_mode_n':len(cm),
                              'captains_mode_listed_n':sum(r['frozen_tier'] in ('professional','premium') for r in cm),
                              'captains_mode_listed_fraction':sum(r['frozen_tier'] in ('professional','premium') for r in cm)/len(cm)})
        print('Cohort and graph audit completed at minute',t,flush=True)
    csvout(OUT/'cohort_summary.csv',summaries);csvout(OUT/'cohort_profiles.csv',profiles)
    csvout(OUT/'month_mode_tier_cross.csv',cross);csvout(OUT/'mode_restricted_composition.csv',selection)
    csvout(OUT/'graph_summary.csv',gs);csvout(OUT/'graph_degree_distribution.csv',hist)
    csvout(OUT/'graph_per_match.csv',matches);csvout(OUT/'graph_evaluation_categories.csv',subgroups)
    failures=read(source(REV/'wp6_evaluation/holdout_inputs/eligibility_failures.json',holdcontract['hashes']['eligibility_failures.json']))
    attrition=[]
    common=Counter(r['reason'] for r in failures if r['minute']=='all')
    for t in (10,20,30,40):
        local=Counter(r['reason'] for r in failures if r['minute']==t)
        accepted=holdcontract['n_per_minute'][str(t)]
        check(f'attrition:{t}',len(candidate['matches'])-sum(common.values())-sum(local.values())==accepted)
        for stage, groups in [('common_exclusion',common),('cutoff_exclusion',local)]:
            for reason,n in groups.items():attrition.append({'minute':t,'stage':stage,'reason':reason,'n':n})
        attrition.append({'minute':t,'stage':'accepted','reason':'common input eligible','n':accepted})
    csvout(OUT/'attrition.csv',attrition)
    requestpath=source(REV/'wp6_evaluation/requests.jsonl')
    req=[json.loads(line) for line in requestpath.read_text().splitlines() if line.strip()]
    good=[r for r in req if r.get('status')==200 and r.get('sha256')]
    check('requests cover all raw hashes',{str(r['match_id'])+'.json':r['sha256'] for r in good}==rawmanifest['hashes'])
    dates=[]
    for p in sorted((REV/'wp6_evaluation/listings').glob('*.json')):
        d=read(source(p,candidate['listing_hashes'][p.name] if p.name in candidate['listing_hashes'] else None))
        dates.append({'event':'metadata query','file':str(p.relative_to(ROOT)),'at_utc':utc(d['retrieved_unix']),'n':len(d['rows'])})
    dates.extend([{'event':'first successful detail request','at_utc':utc(min(r['at_unix'] for r in good)),'n':len(good)},
                  {'event':'last successful detail request','at_utc':utc(max(r['at_unix'] for r in good)),'n':len(good)},
                  {'event':'candidate freeze','at_utc':utc(candidate['created_unix'])},
                  {'event':'raw manifest sealed','at_utc':utc(rawmanifest['created_unix'])}])
    for stem in ('opendota_leagues','lobby_type','game_mode'):
        p=ROOT/f'Dota2_matches/reports/cohort_scope/{stem}_source.json';d=read(source(p))
        source(ROOT/f'Dota2_matches/reports/cohort_scope/{stem}.json',d['sha256'])
        dates.append({'event':stem+' snapshot','at_utc':d['retrieved_at'],'url':d['url']})
    csvout(OUT/'acquisition_timeline.csv',dates)
    actual_tiers={r['value']:r['n'] for r in profiles if r['cohort']=='evaluation' and r['minute']==20 and r['axis']=='frozen_tier'}
    check('20 minute target fixed categories',actual_tiers=={'professional':1499,'premium':3,'excluded':5550,'unmatched':489})
    # Cross-check existing independent pool reports on their shared axes.
    import csv
    with (REV/'wp5_scope/reports/training_pool_profiles.csv').open(encoding='utf-8-sig') as f:old=list(csv.DictReader(f))
    lookup={(r['minute'],r['cohort'].removeprefix('train_'),r['axis'],r['value']):r for r in profiles if r['cohort'].startswith('train_')}
    checked=0
    for r in old:
        axis={'provider_tier':'frozen_tier','tier':'frozen_tier'}.get(r['axis'],r['axis'])
        key=(int(r['minute']),r['scope'],axis,r['group'])
        if key in lookup:
            check('existing profile:'+str(key),int(r['n'])==lookup[key]['n'] and int(r['radiant_wins'])==lookup[key]['radiant_wins']);checked+=1
    check('existing profile comparison executed',checked>100)
    write(OUT/'audit_validation.json',{'status':'PASS','checks':len(CHECKS),'failed':0,'details':CHECKS,
                                      'raw_holdout_hashes_verified':len(rawmanifest['hashes']),
                                      'raw_edge_replays':36,'historical_raw_policy':'full historical raw inventory not rehashed; sealed arrays and metadata verified, 24 raw cutoff replays',
                                      'historical_acquisition_dates':'not recoverable from currently retained complete request logs',
                                      'graph_definition':'abs(edges).sum(-1)>0 OR transpose, plus self loops; offdiag degree excludes self; density denominator 90, cross-team 50, same-team 40',
                                      'profiles_existing_rows_verified':checked})
    for p in (HERE/'audit_data_graph.py',HERE/'freeze_additional_experiment.py'):
        source(p)
    write(OUT/'audit_sources.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'hashes':SOURCES})
    print('PASS',len(CHECKS),'checks;',len(rawmanifest['hashes']),'raw evaluation files verified',flush=True)

if __name__=='__main__':main()

"""Freeze a supplementary sampling-sensitivity design; never fits a model."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import csv
import numpy as np

HERE = Path(__file__).resolve().parent
REV = HERE.parent
ROOT = REV.parent

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def csvout(p, rows):
    rows = list(rows)
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)

def main():
    if (HERE/'protocol_lock.json').exists():
        for p, h in read(HERE/'protocol_lock.json')['hashes'].items():
            assert sha(ROOT/p) == h, p
        print('Existing supplementary protocol verified; no overwrites.')
        return
    manifest = read(REV/'wp4_training/data/split_manifest.json')
    meta = {int(r['match_id']): r for r in manifest['train']}
    pools = read(REV/'wp5_scope/pool_ids.json')['20']
    broad, listed = pools['broad'], pools['listed']
    with np.load(REV/'wp4_training/data/train_20m.npz') as a:
        assert broad == a['match_id'].tolist()
    assert listed == [i for i in broad if meta[i]['frozen_tier'] in ('professional', 'premium')]
    counts = Counter(meta[i]['month'] for i in listed)
    seeds = [111001, 111017, 111031]  # First three, not selected by performance.
    draws, checks = {}, []
    forbidden = set(pools['validation'])
    with np.load(REV/'wp6_evaluation/holdout_inputs/labels_20m.npz') as a:
        target = a['match_id'].tolist()
    forbidden.update(target)
    assert not set(broad) & forbidden
    for draw in range(1, 6):
        selected = []
        for month, n in sorted(counts.items()):
            candidates = [i for i in broad if meta[i]['month'] == month]
            key = lambda i: (hashlib.sha256(f'access-scope-repeat-v1:{draw:02d}:{month}:{i}'.encode()).hexdigest(), i)
            chosen = sorted(candidates, key=key)[:n]
            assert len(chosen) == n
            selected.extend(chosen)
            checks.append({'draw': draw, 'month': month, 'broad_n': len(candidates), 'listed_n': n,
                           'draw_n': len(chosen), 'overlap_listed': len(set(chosen)&set(listed))})
        assert len(selected) == len(set(selected)) == 24685
        assert not set(selected) & forbidden
        draws[f'draw_{draw:02d}'] = selected
    assert len({tuple(v) for v in draws.values()}) == 5
    lock = read(REV/'wp6_evaluation/evaluation_lock.json')
    recipes, reused = {}, {}
    sources = [REV/'wp4_training/data/split_manifest.json', REV/'wp5_scope/pool_ids.json',
               REV/'wp4_training/data/train_20m.npz', REV/'wp4_training/data/validation_20m.npz',
               REV/'wp4_training/data/input_contract.json', REV/'wp6_evaluation/evaluation_lock.json',
               REV/'wp6_evaluation/holdout_inputs/contract.json', REV/'wp6_evaluation/holdout_inputs/metadata.json',
               REV/'wp6_evaluation/holdout_inputs/labels_20m.npz', REV/'wp3_models/controlled_models.py',
               REV/'wp3_models/inputs.py', REV/'wp4_training/engine.py', HERE/'freeze_additional_experiment.py',
               ROOT/'Dota2_matches/reports/cohort_scope/opendota_leagues.json']
    for scope in ('listed', 'broad_matched'):
        cell = next(c for c in lock['models'] if c['scope']==scope and c['model']=='N' and c['minute']==20)
        assert cell['seeds'][:3] == seeds
        paths = cell['paths'][:3]
        configurations = [read(ROOT/p/'run_config.json') for p in paths]
        recipes[scope] = configurations[0]['params']
        assert all(c['params']==recipes[scope] for c in configurations)
        reused[scope] = paths
        for p in paths:
            for name in ('run_config.json', 'result.json', 'model.pt'):
                sources.append(ROOT/p/name)
            result = read(ROOT/p/'result.json')
            assert len(result['history'])-result['best_epoch'] >= 15
    ids = {'listed_reference': listed, 'original_matched_reference': pools['broad_matched'],
           'validation': pools['validation'], 'evaluation': target, 'new_matched_draws': draws}
    write(HERE/'frozen_sampling_ids.json', ids)
    csvout(HERE/'reports/new_draw_monthly_matching.csv', checks)
    overlap = []
    sets = {'listed': set(listed), 'original_matched': set(pools['broad_matched']),
            **{k:set(v) for k,v in draws.items()}}
    for i, a in enumerate(sets):
        for b in list(sets)[i+1:]:
            overlap.append({'pool_a': a, 'pool_b': b, 'intersection':len(sets[a]&sets[b]),
                            'jaccard': len(sets[a]&sets[b])/len(sets[a]|sets[b])})
    csvout(HERE/'reports/new_draw_overlap.csv', overlap)
    protocol = {
        'version':'access-scope-repeat-v1', 'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'frozen_before_new_training; original WP6 results already known',
        'purpose':'sensitivity of the fixed selected training recipes to a new monthly matched sample',
        'model':'N', 'minute':20, 'draw_count':5, 'training_seeds':seeds,
        'new_fit_count':15, 'listed_reference_reused_fits':3, 'original_matched_reference_reused_fits':3,
        'sampling':'ascending SHA256 access-scope-repeat-v1:<draw:02d>:<UTC month>:<match_id>, integer ID tie-break; without replacement within each month',
        'sampling_uses_outcomes':False, 'overlap_between_draws_and_listed_allowed':True,
        'fixed_params':recipes, 'reused_paths':reused,
        'retune_per_draw':False,
        'estimand_boundary':'fixed per-scope selected recipes; parameters differ across scopes. Does not estimate reoptimized-pipeline variability or a pure composition effect with identical hyperparameters.',
        'preprocessing':'fit purchase priors and hero vocabulary separately from each sampled training pool; re-extract pool-dependent purchase features for train, validation and evaluation; never reuse original matched priors for a new draw',
        'training':{'device':'cuda','max_epochs':450,'early_stopping_patience':15,'validation':'original common WP4 validation IDs, n=7200',
                    'checkpoint':'minimum development-validation log loss; same frozen engine and deterministic policy'},
        'failures':'resume same draw and seed; retain failed/capped fits and stop acceptance of a complete result. No replacing draws or dropping runs according to scores. Any budget extension requires a new recorded protocol version.',
        'targets':{'all':7541,'listed':1502,'definition':'unchanged WP6 20-minute IDs; professional/premium from same frozen league metadata'},
        'metric':'natural-log binary log loss, probabilities clipped at [1e-7,1-1e-7]; compute per model seed, no prediction ensembling',
        'difference':'D[d,s,target] = loss(listed reference,s,target) - loss(new matched draw d,s,target)',
        'reporting':[
            'Publish all 5x3 paired differences on both targets and their direct listed-target minus all-target contrasts.',
            'For each draw report mean across three seeds and within-draw sample SD; across five draw means report mean, sample SD, min/max and counts of positive/negative/zero means.',
            'Report matched-model loss dispersion separately to describe sampling sensitivity without the fixed listed reference.',
            'Across-draw SD is the observed sensitivity of three-seed averages, not a noise-free composition variance component.',
            'Seeds are crossed with draws and the same listed reference is reused: do not treat 15 contrasts as independent replicates; do not construct a naive t interval on 15 cells.',
            'No new confirmatory hypothesis test, p-value, equivalence decision, or CI claiming total population uncertainty. Original WP6 primary inference stays unchanged.',
            'For both targets show all draw means and direction counts; do not stop early, select a favorable draw or discard a direction-changing draw.',
            'Original matched pool is a separately labelled reference, never counted as a sixth new random draw.'
        ],
        'future_worker_requirements':['verify source lock before training','training-only prior hash and IDs per draw','validation and target IDs identical across draws','GPU availability recorded; no silent CPU fallback','fit identity includes draw/seed/config/prior/data hashes','resume checks exact identity','no training started by this freezer'],
        'sampling_ids_sha256':sha(HERE/'frozen_sampling_ids.json')
    }
    write(HERE/'experiment_protocol.json', protocol)
    sources += [HERE/'frozen_sampling_ids.json', HERE/'experiment_protocol.json']
    hashes = {p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    for p,h in hashes.items():
        if p in lock['sources']:
            assert h == lock['sources'][p], p
    write(HERE/'protocol_lock.json', {'version':protocol['version'], 'hashes':hashes,
                                      'note':'New supplementary protocol, independent of original WP6 lock; do not rewrite frozen WP6 sources.'})
    print('Frozen five draws x three seeds; 15 future fits; no training started.')

if __name__ == '__main__':
    main()

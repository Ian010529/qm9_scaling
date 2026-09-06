"""Generate the entire run matrix without observing target values."""
from copy import deepcopy
from itertools import product
import numpy as np
from .io import digest


def family(route):
    return {'count_xgb': 'morgan_xgb', 'wide_count_xgb': 'morgan_xgb', 'morgan_mlp': 'molformer_mlp'}.get(route, route)


def candidates(cfg, route):
    key = family(route)
    if key == 'molformer_ridge':
        return [{'alpha': a} for a in cfg['search_spaces'][key]['alpha']]
    default = cfg['defaults'][key]
    rng = np.random.default_rng(cfg['hpo']['seed'] + int(digest(route)[:8], 16))
    result = [deepcopy(default)]
    while len(result) < cfg['hpo']['n_candidates']:
        p = {**default, **{k: v[int(rng.integers(len(v)))] for k, v in cfg['search_spaces'][key].items()}}
        if p not in result:
            result.append(p)
    return result


def make_job(stage, route, task, n, s, t, split='random', arm='primary', **kwargs):
    obj = dict(stage=stage, route=route, task=task, n=int(n), subset_seed=int(s), training_seed=int(t), split=split, arm=arm, **kwargs)
    obj['id'] = f'{stage}-{route}-{task}-n{n}-s{s}-t{t}-{digest(obj)[:10]}'
    return obj


def matrix(cfg, pools=None):
    jobs = []
    pairs = list(zip(cfg['subset_seeds'], cfg['training_seeds'], strict=True))
    routes = cfg['primary_routes']
    hp = cfg['hpo']
    for route in routes + ['morgan_mlp', 'molformer_ridge']:
        tasks = cfg['tasks'] if route in routes else cfg['targets']
        for task, (trial, params), n, (s, t) in product(tasks, enumerate(candidates(cfg, route)), hp['anchors'], zip(hp['subset_seeds'], hp['training_seeds'], strict=True)):
            jobs.append(make_job('tune', route, task, n, s, t, arm='hpo', trial=trial, params=params))
    for r, task, n, (s, t) in product(routes, cfg['tasks'], cfg['train_sizes'], pairs):
        stage = 'calibration' if n <= cfg['calibration_max_n'] else 'confirmation'
        jobs.append(make_job(stage, r, task, n, s, t, arm='residual' if task == 'G_residual' else 'primary'))
    c = cfg['controls']
    for r, task, n, (s, t) in product(c['routes'], cfg['targets'], c['sizes'], pairs[:c['repetitions']]):
        jobs.append(make_job('controls', r, task, n, s, t, arm='representation'))
    c = cfg['capacity']
    for r, task, n, (s, t), m in product(c['routes'], cfg['targets'], c['sizes'], pairs[:c['repetitions']], c['multipliers']):
        jobs.append(make_job('capacity', r, task, n, s, t, arm='capacity', width_multiplier=m))
    c = cfg['budget']
    for r, task, n, (s, t), m in product(routes, cfg['tasks'], c['sizes'], pairs[:c['repetitions']], c['multipliers']):
        jobs.append(make_job('budget', r, task, n, s, t, arm='training_budget', budget_multiplier=m))
    c = cfg['capacity']
    for r, task, n, (s, t), m in product(c['routes'], cfg['targets'], c['budget_sizes'], pairs[:c['repetitions']], c['budget_multipliers']):
        jobs.append(make_job('budget', r, task, n, s, t, arm='large_capacity_budget', width_multiplier=max(c['multipliers']), budget_multiplier=m))
    c = cfg['validation_budget']
    for r, task, n, (s, t), limited in product(routes, cfg['targets'], c['sizes'], pairs[:c['repetitions']], [False, True]):
        jobs.append(make_job('validation_budget', r, task, n, s, t, arm='validation_budget', preset=True, limited_validation=limited))
    for task, n, (s, t) in product(['G', 'G_residual'], cfg['budget']['sizes'], pairs[:5]):
        jobs.append(make_job('aggregation', 'dmpnn', task, n, s, t, arm='aggregation', aggregation='mean'))
    for r, task, n, s, t in product(routes, cfg['targets'], cfg['budget']['sizes'], cfg['subset_seeds'][:3], cfg['training_seeds'][:3]):
        if (s, t) not in pairs:
            jobs.append(make_job('seed_variance', r, task, n, s, t, arm='crossed_seeds'))
    c = cfg['robustness']
    for split in c['splits']:
        cap = (pools or {}).get(split, max(c['sizes']))
        sizes = sorted({min(n, cap) for n in c['sizes']})
        for r, task, n, (s, t) in product(routes, cfg['targets'], sizes, pairs[:c['repetitions']]):
            jobs.append(make_job('robustness', r, task, n, s, t, split=split, arm='split', preset=True))
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Run ID collision')
    return jobs


def parameters(cfg, item, selected):
    route = item['route']
    if item['stage'] == 'tune':
        p = deepcopy(item['params'])
    elif item.get('preset'):
        p = deepcopy(cfg['defaults'][family(route)])
    else:
        key = 'morgan_xgb' if route in ['count_xgb', 'wide_count_xgb'] else route
        p = deepcopy(selected[f'{key}/{item["task"]}']['params'])
    if 'width_multiplier' in item:
        p['hidden'] = max(1, round(p['hidden'] * item['width_multiplier']))
    if route == 'dmpnn':
        p['aggregation'] = item.get('aggregation', 'sum' if item['task'].startswith('G') else 'mean')
    return p

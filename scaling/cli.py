"""Phase-gated execution: prepare -> preflight -> tune -> lock -> calibrate -> forecast -> seal -> confirm -> evaluate -> analyze."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import fcntl
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
from .io import ROOT, check_file, code_manifest, digest, read_json, sha256, versions, write_json
from .design import matrix, parameters
from .data import Workspace, prepare
from .train import metrics, neural_model, batch_input, predict_saved, run_one
from .laws import forecast


def complete(work, item):
    folder = work.folder/'runs'/item['id']
    record = read_json(folder/'complete.json')
    if record['job_sha256'] != digest(item) or record['protocol_sha256'] != digest(work.cfg): raise ValueError('Run identity mismatch')
    for rel, value in record['files'].items(): check_file(folder/rel,value)
    result = read_json(folder/'result.json')
    if result['job'] != item: raise ValueError('Result is for a different job')
    return result


def assert_hyper_lock(work):
    lock = read_json(work.folder/'hyper_lock.json')
    if lock['manifest'] != sha256(work.folder/'manifest.json'): raise ValueError('Hyperparameter data/protocol changed')
    check_file(work.folder/'selected.json',lock['selected'])
    return read_json(work.folder/'selected.json')


def assert_seal(work):
    seal = read_json(work.folder/'seal.json')
    for name, value in seal['files'].items(): check_file(work.folder/name,value)
    if seal['code'] != code_manifest(): raise ValueError('Code changed after forecast sealing')
    return seal


def preflight_key(device):
    return f'preflight-{digest([device,versions()])[:16]}.json'


def preflight(work, device):
    import torch
    required = {'rdkit':read_json(work.root/'features/morgan.json')['versions']['rdkit'],
                'xgboost':read_json(work.root/'features/morgan.json')['versions']['xgboost'],
                'chemprop':'2.2.1', 'torch':'2.8.0'}
    actual = versions()
    for name,value in required.items():
        if actual[name] is None or actual[name].split('+')[0] != value.split('+')[0]:
            raise RuntimeError(f'{name}: need {value}, found {actual[name]}; use the pinned research environment')
    for name in ['rdkit','numpy','pandas']:
        if actual[name] != work.manifest['versions'][name]: raise ValueError('Preparation and training environments differ for '+name)
    graphs = work.features('dmpnn')
    p = {**work.cfg['defaults']['dmpnn'],'aggregation':'sum'}
    torch.manual_seed(101)
    model = neural_model('dmpnn',0,p).to(device)
    batch = batch_input(graphs,np.arange(8),'dmpnn',{},device)
    y = model(batch)
    if y.shape != (8,1) or not torch.isfinite(y).all(): raise ValueError('Chemprop forward failed')
    y.square().mean().backward()
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None): raise ValueError('Chemprop gradients failed')
    normalized = {k:(v.split('+')[0] if v is not None else None) for k,v in actual.items()}
    env_path = work.folder/'environment.json'
    if env_path.exists() and read_json(env_path) != normalized: raise ValueError('Cannot replace the frozen environment')
    if not env_path.exists(): write_json(env_path,normalized)
    write_json(work.folder/preflight_key(device),dict(passed=True,manifest=sha256(work.folder/'manifest.json'),versions=actual,device=device,
                 graph_forward_backward=True,note='Synthetic objective only; this is not a QM9 result'))


def lock_hyperparameters(work):
    if (work.folder/'hyper_lock.json').exists(): assert_hyper_lock(work); return
    groups = defaultdict(list)
    for item in work.jobs:
        if item['stage'] != 'tune': continue
        result = complete(work,item); value = result['metrics']['tune']
        score = float(np.log(max(value['MAE'],1e-15)/max(value['median_baseline_MAE'],1e-15)))
        groups[(item['route'],item['task'],item['trial'])].append((score,item))
    finalists = defaultdict(list)
    expected = len(work.cfg['hpo']['anchors'])*len(work.cfg['hpo']['subset_seeds'])
    for (route,task,trial),values in groups.items():
        if len(values) != expected: raise ValueError('Incomplete HPO candidate')
        finalists[f'{route}/{task}'].append(dict(trial=trial,score=float(np.mean([v[0] for v in values])),params=values[0][1]['params'],runs=[v[1]['id'] for v in values]))
    selected = {k:min(v,key=lambda x:(x['score'],x['trial'])) for k,v in finalists.items()}
    write_json(work.folder/'selected.json',selected)
    write_json(work.folder/'hyper_lock.json',dict(manifest=sha256(work.folder/'manifest.json'),selected=sha256(work.folder/'selected.json'),candidate_scores=dict(finalists)))


def train_stage(work, args):
    record = read_json(work.folder/preflight_key(args.device))
    if not record['passed'] or record['manifest'] != sha256(work.folder/'manifest.json'): raise ValueError('Preflight is missing or stale')
    if args.threads < 1 or (args.limit is not None and args.limit < 1): raise ValueError('threads and limit must be positive')
    selected = None if args.stage == 'tune' else assert_hyper_lock(work)
    if args.stage not in ['tune','calibration']: assert_seal(work)
    if (work.folder/'seal.json').exists() and args.stage in ['tune','calibration']:
        # Completed runs may be verified; no new/replacement discovery runs after sealing.
        for item in work.jobs:
            if item['stage']==args.stage: complete(work,item)
        return
    shard,total = map(int,args.shard.split('/'))
    if total < 1 or not 0 <= shard < total: raise ValueError('Shard must be zero-based i/k, 0 <= i < k')
    items = [j for j in work.jobs if j['stage']==args.stage]
    done = 0
    for i,item in enumerate(items):
        if i % total != shard: continue
        p = parameters(work.cfg,item,selected)
        out = work.folder/'runs'/item['id']
        lock_path = work.folder/'locks'/f'{item["id"]}.lock'; lock_path.parent.mkdir(exist_ok=True)
        with lock_path.open('a') as stream:
            try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: continue
            if (out/'complete.json').exists():
                complete(work,item)
                if read_json(out/'complete.json')['parameters_sha256'] != digest(p): raise ValueError('Completed run uses different parameters')
                continue
            if out.exists():
                if not args.retry_failed: raise FileExistsError(f'Incomplete run: {out}; inspect it and rerun with --retry-failed')
                archive = work.folder/'failed'/f'{item["id"]}-{time.time_ns()}'; archive.parent.mkdir(exist_ok=True)
                shutil.move(str(out),str(archive))
            try:
                run_one(work,item,p,out,args.device,args.threads)
            except Exception:
                write_json(out/'failure.json',dict(job=item,traceback=traceback.format_exc()))
                raise
            done += 1
            print(f'Completed {item["id"]}',flush=True)
            if args.limit is not None and done >= args.limit: break


def make_forecasts(work):
    assert_hyper_lock(work)
    if (work.folder/'forecasts.json').exists(): raise FileExistsError('Forecasts already frozen; never overwrite')
    if (work.folder/'seal.json').exists(): raise ValueError('Already sealed')
    items = [j for j in work.jobs if j['stage']=='calibration']
    groups = defaultdict(list)
    for item in items: complete(work,item); groups[(item['route'],item['task'])].append(item)
    all_forecasts = {}; inputs = {}
    for (route,task),values in groups.items():
        sizes = sorted({v['n'] for v in values}); seeds = work.cfg['subset_seeds']
        lookup = {(v['subset_seed'],v['n']):v for v in values}
        cube = []; expected_ids = None
        for seed in seeds:
            trajectory = []
            for n in sizes:
                item = lookup[seed,n]; path = work.folder/'runs'/item['id']/'law.npz'
                with np.load(path,allow_pickle=False) as f:
                    if expected_ids is None: expected_ids = f['mol_id'].copy()
                    if not np.array_equal(expected_ids,f['mol_id']): raise ValueError('Law evaluation molecules changed across sizes')
                    trajectory.append(np.abs(f['y_pred']-f['y_true']))
                inputs[item['id']] = sha256(path)
            cube.append(trajectory)
        future = [n for n in work.cfg['train_sizes'] if n > work.cfg['calibration_max_n']]
        key = f'{route}/{task}'
        all_forecasts[key] = forecast(sizes,np.array(cube),future,work.cfg['analysis'])
        print('Fitted and forecast',key,flush=True)
    write_json(work.folder/'forecast_inputs.json',inputs)
    write_json(work.folder/'forecasts.json',all_forecasts)


def seal(work):
    assert_hyper_lock(work)
    if (work.folder/'seal.json').exists(): assert_seal(work); return
    forecasts = read_json(work.folder/'forecasts.json')
    expected = {f'{r}/{t}' for r in work.cfg['primary_routes'] for t in work.cfg['tasks']}
    if set(forecasts) != expected: raise ValueError('Missing forecast family')
    for item in work.jobs:
        if item['stage'] == 'calibration': complete(work,item)
        elif item['stage'] != 'tune' and (work.folder/'runs'/item['id']/'complete.json').exists():
            raise ValueError('A confirmatory run exists before sealing')
    for run,value in read_json(work.folder/'forecast_inputs.json').items(): check_file(work.folder/'runs'/run/'law.npz',value)
    files = ['manifest.json','environment.json','selected.json','hyper_lock.json','forecasts.json','forecast_inputs.json']
    write_json(work.folder/'seal.json',dict(files={f:sha256(work.folder/f) for f in files},code=code_manifest(),created_unix=time.time(),
                meaning='Procedural preregistration lock; public raw labels are not cryptographically concealed'))


def evaluate(work,args):
    assert_seal(work); selected = assert_hyper_lock(work)
    for item in work.jobs:
        result = complete(work,item)
        expected = parameters(work.cfg,item,selected if item['stage']!='tune' else None)
        if result['parameters'] != expected: raise ValueError('Evaluation parameter drift')
    shard,total = map(int,args.shard.split('/'))
    if total < 1 or not 0 <= shard < total: raise ValueError('Invalid shard')
    items = [j for j in work.jobs if j['stage']!='tune']
    for i,item in enumerate(items):
        if i % total != shard: continue
        out = work.folder/'runs'/item['id']
        if (out/'evaluation.json').exists():
            old=read_json(out/'evaluation.json')
            if old['seal_sha256'] != sha256(work.folder/'seal.json'): raise ValueError('Evaluation belongs to another seal')
            check_file(out/'test.npz',old['predictions_sha256']); continue
        indices = work.rows(item)['test']; target = 'G' if item['task']=='G_residual' else item['task']
        true = work.frame[target].to_numpy(dtype=float)[indices]
        started = time.perf_counter(); pred = predict_saved(work,item,out,'test',args.device); seconds=time.perf_counter()-started
        with np.load(out/'transform.npz') as f:
            median = np.full_like(true,f['median']); composition=work.composition[indices]@f['baseline_coef']
        np.savez_compressed(out/'test.npz',mol_id=work.frame.mol_id.to_numpy(dtype=str)[indices],y_true=true,y_pred=pred,median_pred=median,composition_pred=composition)
        write_json(out/'evaluation.json',dict(metrics=metrics(true,pred),median_baseline=metrics(true,median),composition_baseline=metrics(true,composition),
                    prediction_seconds=seconds,predictions_sha256=sha256(out/'test.npz'),seal_sha256=sha256(work.folder/'seal.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT); parser.add_argument('--study',type=Path)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('plan'); sub.add_parser('prepare'); sub.add_parser('lock'); sub.add_parser('forecast'); sub.add_parser('seal'); sub.add_parser('status'); sub.add_parser('analyze')
    pf=sub.add_parser('preflight'); pf.add_argument('--device',default='cpu')
    run=sub.add_parser('run'); run.add_argument('--stage',required=True,choices=['tune','calibration','confirmation','controls','capacity','budget','validation_budget','aggregation','seed_variance','robustness'])
    run.add_argument('--device',default='cpu'); run.add_argument('--threads',type=int,default=4); run.add_argument('--shard',default='0/1'); run.add_argument('--limit',type=int); run.add_argument('--retry-failed',action='store_true')
    ev=sub.add_parser('evaluate'); ev.add_argument('--device',default='cpu'); ev.add_argument('--shard',default='0/1')
    args=parser.parse_args(); folder=args.study or args.root/'studies/scaling-law-v1'; cfg=read_json(ROOT/'config/scaling_law_v1.json')
    if args.command=='plan':
        jobs=matrix(cfg); print(dict(Counter(j['stage'] for j in jobs))); print('Nominal total:',len(jobs)); return
    if args.command=='prepare': prepare(args.root,folder,cfg); return
    work=Workspace(args.root,folder)
    if args.command=='preflight': preflight(work,args.device)
    elif args.command=='run': train_stage(work,args)
    elif args.command=='lock': lock_hyperparameters(work)
    elif args.command=='forecast': make_forecasts(work)
    elif args.command=='seal': seal(work)
    elif args.command=='evaluate': evaluate(work,args)
    elif args.command=='analyze':
        from .report import analyze
        analyze(work)
    elif args.command=='status':
        counts=Counter(j['stage'] for j in work.jobs if (folder/'runs'/j['id']/'complete.json').exists())
        print('Completion markers (not an integrity audit):',dict(counts)); print('Expected:',dict(Counter(j['stage'] for j in work.jobs)))


if __name__=='__main__': main()

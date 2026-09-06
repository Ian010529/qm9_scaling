"""Train-only transforms and shared, restartable single-run model adapters."""
from __future__ import annotations
import math
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
from .design import family
from .io import digest, read_json, sha256, versions, write_json


def target_transform(y, composition, train, residual=False):
    """All fitted quantities depend exclusively on this run's training rows."""
    coef = np.linalg.lstsq(composition[train], y[train], rcond=None)[0]
    offset_coef = coef if residual else np.zeros(composition.shape[1])
    delta = y[train] - composition[train] @ offset_coef
    scale = float(delta.std())
    return dict(coef=offset_coef, baseline_coef=coef, mean=np.array(float(delta.mean())),
                scale=np.array(scale if scale > 1e-12 else 1.), median=np.array(float(np.median(y[train]))))


def metrics(y, pred):
    err = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    if err.shape != np.asarray(y).shape or not np.isfinite(err).all(): raise ValueError('Invalid predictions')
    return dict(n=len(err), MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt(np.mean(err**2))))


def neural_model(route, input_dim, p):
    import torch
    if route == 'dmpnn':
        from chemprop import nn, models
        mp = nn.BondMessagePassing(d_v=72, d_e=14, d_h=p['hidden'], depth=p['depth'], dropout=p['dropout'])
        agg = nn.SumAggregation() if p['aggregation'] == 'sum' else nn.MeanAggregation()
        head = nn.RegressionFFN(input_dim=p['hidden'], hidden_dim=p['hidden'], n_layers=p['layers'], dropout=p['dropout'], n_tasks=1)
        return models.MPNN(mp, agg, head, batch_norm=False)
    layers = []
    dim = input_dim
    for _ in range(p['layers']):
        layers += [torch.nn.Linear(dim, p['hidden']), torch.nn.ReLU(), torch.nn.Dropout(p['dropout'])]
        dim = p['hidden']
    return torch.nn.Sequential(*layers, torch.nn.Linear(dim, 1))


def batch_input(x, indices, route, transform, device):
    import torch
    if route == 'dmpnn':
        from chemprop.data.collate import BatchMolGraph
        batch = BatchMolGraph([x[int(i)] for i in indices])
        batch.to(device)  # Chemprop mutates in place; do not use the return value.
        return batch
    arr = (np.asarray(x[indices], dtype=np.float32) - transform['x_mean']) / transform['x_scale']
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def neural_predict(model, x, indices, route, transform, device, batch_size):
    import torch
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = batch_input(x, indices[start:start+batch_size], route, transform, device)
            result.append(model(batch).detach().cpu().numpy().reshape(-1))
    return np.concatenate(result).astype(float)


def fit_neural(x, y, train, stop, route, p, transform, cfg, seed, device, out, budget=1):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('Requested CUDA is unavailable')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = neural_model(route, 0 if route == 'dmpnn' else x.shape[1], p).to(device)
    if device.startswith('cuda'): torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg['lr_factor'], patience=cfg['lr_patience'], min_lr=cfg['min_lr'], threshold=cfg['min_delta'], threshold_mode='abs')
    batch_size = cfg['batch_size']; per_epoch = math.ceil(len(train)/batch_size)
    limit = int(budget * max(cfg['minimum_max_updates'], cfg['max_epochs']*per_epoch))
    interval = max(per_epoch, cfg['eval_interval_min'])
    patience = int(budget*cfg['patience'])
    rng = np.random.default_rng(seed)
    step = 0; best = float('inf'); reference = float('inf'); stale = 0; best_step = 0; presented = 0
    history = []; loss_sum = 0.; loss_count = 0; stop_reason = 'update_limit'
    while step < limit:
        order = rng.permutation(train)
        for start in range(0, len(order), batch_size):
            indices = order[start:start+batch_size]
            model.train(); optimizer.zero_grad(set_to_none=True)
            pred = model(batch_input(x, indices, route, transform, device)).reshape(-1)
            labels = torch.as_tensor(y[indices], dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(pred, labels)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip']); optimizer.step()
            step += 1; presented += len(indices); loss_sum += float(loss.detach())*len(indices); loss_count += len(indices)
            if step % interval == 0 or step == limit:
                vp = neural_predict(model, x, stop, route, transform, device, batch_size)
                mae = float(np.abs(vp-y[stop]).mean())
                if not np.isfinite(mae): raise FloatingPointError('Nonfinite validation metric')
                if mae < best:
                    best, best_step = mae, step
                    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, out/'model.pt')
                if mae < reference - cfg['min_delta']: reference, stale = mae, 0
                else: stale += 1
                scheduler.step(mae)
                history.append(dict(step=step, examples_seen=presented, train_MSE=loss_sum/loss_count, stop_MAE=mae, lr=optimizer.param_groups[0]['lr']))
                loss_sum = 0.; loss_count = 0
                if step >= cfg['minimum_updates'] and stale >= patience:
                    stop_reason = 'early_stopping'; break
            if step >= limit: break
        if stop_reason == 'early_stopping': break
    model.load_state_dict(torch.load(out/'model.pt', map_location=device, weights_only=True))
    info = dict(best_step=best_step, updates=step, update_limit=limit, examples_seen=presented,
                stop_reason=stop_reason, trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                gpu_peak_bytes=int(torch.cuda.max_memory_allocated(device)) if device.startswith('cuda') else None)
    return model, history, info


def predict_saved(work, item, out, split, device='cpu'):
    import torch
    meta = read_json(Path(out)/'result.json')
    with np.load(Path(out)/'transform.npz', allow_pickle=False) as f: transform = {k:f[k] for k in f.files}
    indices = work.rows(item)[split]
    x = work.features(item['route']); route = item['route']
    if route.endswith('xgb'):
        from xgboost import XGBRegressor
        model = XGBRegressor(); model.load_model(Path(out)/'model.ubj')
        scaled = model.predict(np.asarray(x[indices]))
    elif route == 'molformer_ridge':
        with np.load(Path(out)/'model.npz') as f:
            values = (np.asarray(x[indices], dtype=float)-transform['x_mean'])/transform['x_scale']
            scaled = values @ f['coef'] + f['intercept']
    else:
        model = neural_model(route, 0 if route == 'dmpnn' else x.shape[1], meta['parameters']).to(device)
        model.load_state_dict(torch.load(Path(out)/'model.pt', map_location=device, weights_only=True))
        scaled = neural_predict(model, x, indices, route, transform, device, work.cfg['training']['batch_size'])
    return np.asarray(scaled, dtype=float)*transform['scale'] + transform['mean'] + work.composition[indices]@transform['coef']


def run_one(work, item, p, out, device='cpu', threads=4):
    """Called by the phase-gated CLI. Writes completion last; failed runs are never reused."""
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(threads)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); rows = work.rows(item)
    raw_target = 'G' if item['task'] == 'G_residual' else item['task']
    y_raw = work.frame[raw_target].to_numpy(dtype=float)
    transform = target_transform(y_raw, work.composition, rows['train'], item['task'] == 'G_residual')
    y = (y_raw - work.composition@transform['coef'] - transform['mean'])/transform['scale']
    x = work.features(item['route']); route = item['route']
    if route in ['molformer_mlp', 'molformer_ridge', 'morgan_mlp']:
        values = np.asarray(x[rows['train']], dtype=np.float64)
        transform['x_mean'] = values.mean(axis=0).astype('float32')
        std = values.std(axis=0); std[std < 1e-8] = 1.
        transform['x_scale'] = std.astype('float32'); del values
    else:
        transform['x_mean'] = np.array(0., dtype='float32'); transform['x_scale'] = np.array(1., dtype='float32')
    np.savez(out/'transform.npz', **transform)
    prep_seconds = time.perf_counter()-started; fit_start = time.perf_counter()
    history = []; info = {}; budget = item.get('budget_multiplier', 1)
    with threadpool_limits(limits=threads):
        if route.endswith('xgb'):
            from xgboost import XGBRegressor
            settings = dict(p); settings['n_estimators'] *= budget; settings['early_stopping_rounds'] *= budget
            model = XGBRegressor(**settings, objective='reg:squarederror', eval_metric='mae', tree_method='hist', n_jobs=threads, random_state=item['training_seed'])
            model.fit(np.asarray(x[rows['train']]), y[rows['train']], eval_set=[(np.asarray(x[rows['stop']]), y[rows['stop']])], verbose=False)
            model.save_model(out/'model.ubj'); history = model.evals_result()
            info = dict(best_step=int(model.best_iteration), updates=len(history['validation_0']['mae']), update_limit=settings['n_estimators'], stop_reason='early_stopping' if len(history['validation_0']['mae']) < settings['n_estimators'] else 'tree_limit', trainable_parameters=None, device='cpu')
        elif route == 'molformer_ridge':
            from sklearn.linear_model import Ridge
            values = (np.asarray(x[rows['train']], dtype=float)-transform['x_mean'])/transform['x_scale']
            model = Ridge(alpha=p['alpha']*len(rows['train']))
            model.fit(values, y[rows['train']]); np.savez(out/'model.npz', coef=model.coef_, intercept=model.intercept_)
            info = dict(trainable_parameters=int(len(model.coef_)+1), objective='mean squared error + alpha * squared coefficient norm', stop_reason='closed_form', device='cpu')
        else:
            model, history, info = fit_neural(x,y,rows['train'],rows['stop'],route,p,transform,work.cfg['training'],item['training_seed'],device,out,budget)
    if device.startswith('cuda') and not route.endswith('xgb') and route != 'molformer_ridge': torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter()-fit_start
    result = dict(job=item, parameters=p, protocol_sha256=digest(work.cfg), versions=versions(),
                  hardware=dict(platform=platform.platform(), torch_device=device, threads=threads, cuda_name=torch.cuda.get_device_name(device) if device.startswith('cuda') else None),
                  preprocessing_seconds=prep_seconds, fit_seconds=fit_seconds, training=info, unit='Debye' if raw_target=='mu' else 'Hartree', metrics={})
    write_json(out/'history.json', history); write_json(out/'result.json', result)
    # Reload the persisted model, not the in-memory object, for reported predictions.
    for split in ['train','stop'] + (['tune'] if item['stage']=='tune' else ['law']):
        pred = predict_saved(work,item,out,split,device); true = y_raw[rows[split]]
        met = metrics(true,pred)
        median_pred = np.full_like(true, transform['median'])
        comp_pred = work.composition[rows[split]] @ transform['baseline_coef']
        met['median_baseline_MAE'] = metrics(true,median_pred)['MAE']
        met['composition_baseline_MAE'] = metrics(true,comp_pred)['MAE']
        result['metrics'][split] = met
        np.savez_compressed(out/f'{split}.npz', mol_id=work.frame.mol_id.to_numpy(dtype=str)[rows[split]], y_true=true, y_pred=pred,
                            median_pred=median_pred, composition_pred=comp_pred)
    write_json(out/'result.json', result)
    files = {f.name:sha256(f) for f in out.iterdir() if f.is_file()}
    write_json(out/'complete.json', dict(job_sha256=digest(item), parameters_sha256=digest(p), protocol_sha256=digest(work.cfg), files=files))
    return result

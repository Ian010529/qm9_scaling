"""Prespecified curve competition, extrapolation and paired statistical tools."""
from __future__ import annotations
from itertools import product
import numpy as np
from scipy.optimize import least_squares

PARAMETERS = {'constant': 1, 'power': 2, 'floor_power': 3, 'floor_exp': 3, 'broken_2000': 4, 'broken_5000': 4}


def predict(n, fit):
    x = np.asarray(n, dtype=float)/1000.
    p = np.asarray(fit['parameters']); form = fit['form']
    if form == 'constant': value = np.full_like(x, p[0])
    elif form == 'power': value = p[0]*x**(-p[1])
    elif form == 'floor_power': value = p[0] + p[1]*x**(-p[2])
    elif form == 'floor_exp': value = p[0] + p[1]*np.exp(-p[2]*x)
    else:
        knot = float(form.split('_')[1])/1000.
        value = p[0] + p[1]*x**(-p[2])*np.where(x > knot, (x/knot)**(p[2]-p[3]), 1.)
    return value*fit.get('scale', 1.)


def fit_curve(n, y, form):
    n, y = np.asarray(n, dtype=float), np.asarray(y, dtype=float)
    if len(n) < PARAMETERS[form]+1 or (n <= 0).any() or (y <= 0).any() or not np.isfinite(y).all():
        raise ValueError('Insufficient/invalid positive curve observations')
    if form.startswith('broken') and n.max() <= float(form.split('_')[1]):
        base = fit_curve(n,y,'floor_power')
        base['form'] = form
        base['parameters'].append(base['parameters'][-1])
        base['unobserved_post_knot_slope_tied'] = True
        return base
    scale = float(np.median(y)); target = y/scale; x = n/1000.
    if form == 'constant':
        pars = [float(np.exp(np.log(target).mean()))]; boundary = False
    elif form == 'power':
        slope = float(np.clip(-np.polyfit(np.log(x),np.log(target),1)[0],0,3))
        pars = [float(np.exp(np.mean(np.log(target)+slope*np.log(x)))), slope]
        boundary = slope < 1e-5 or slope > 3-1e-5
    else:
        upper_floor = float(target.min()*.999999)
        lower = [0, 1e-12, 0] + ([0] if form.startswith('broken') else [])
        upper = [upper_floor, 1e5, 3] + ([3] if form.startswith('broken') else [])
        solutions = []
        for floor, slope in product([0., .5, .9], [.15, .5, 1.]):
            initial = [floor*upper_floor, max(float(target.max()-floor*upper_floor),1e-3), slope]
            if form.startswith('broken'): initial.append(slope)
            def residual(p):
                guess = predict(n,dict(form=form,parameters=p,scale=1.))
                return np.log(np.maximum(guess,1e-15))-np.log(target)
            result = least_squares(residual, initial, bounds=(lower,upper), max_nfev=1000)
            if result.success and np.isfinite(result.cost): solutions.append(result)
        if not solutions: raise RuntimeError(f'All fits failed: {form}')
        best = min(solutions,key=lambda r:r.cost); pars = best.x.tolist()
        boundary = bool(np.any(best.x[2:] < 1e-5) or np.any(best.x[2:] > 3-1e-5))
    fit = dict(form=form,parameters=pars,scale=scale,slope_boundary=boundary)
    fit['log_RMSE'] = float(np.sqrt(np.mean((np.log(predict(n,fit))-np.log(y))**2)))
    return fit


def choose_form(n, y, cfg):
    n, y = np.asarray(n), np.asarray(y)
    scores = {}
    for form in cfg['forms']:
        errors = []
        for origin in cfg['rolling_origins']:
            train = n <= origin[0]; valid = np.isin(n, origin[1:])
            if valid.sum() == 0 or train.sum() < PARAMETERS[form]+1: continue
            try:
                fit = fit_curve(n[train],y[train],form)
            except (ValueError, RuntimeError):
                errors = None
                break
            errors.extend((np.log(predict(n[valid],fit))-np.log(y[valid])).tolist())
        if errors is None:
            scores[form] = None
            continue
        if not errors: raise ValueError('Rolling-origin validation has no held-out observations')
        scores[form] = float(np.sqrt(np.mean(np.square(errors))))
    valid_scores = [v for v in scores.values() if v is not None]
    if not valid_scores: raise ValueError('All candidate forms failed')
    best = min(valid_scores); tolerance = max(cfg.get('tie_absolute',1e-6), best*cfg['tie_relative'])
    eligible = [f for f in cfg['forms'] if scores[f] is not None and scores[f] <= best+tolerance]
    selected = min(eligible,key=lambda f:(PARAMETERS[f],cfg['forms'].index(f)))
    return selected,scores


def forecast(n, absolute_errors, future_n, cfg):
    """Crossed bootstrap: whole nested run trajectories and shared evaluation molecules.

    absolute_errors: [repetition, training size, evaluation molecule]. All sizes share IDs.
    The selected form is fixed for coefficient intervals; selection stability is reported separately.
    """
    errors = np.asarray(absolute_errors, dtype=float)
    if errors.ndim != 3 or errors.shape[1] != len(n) or not np.isfinite(errors).all(): raise ValueError('Invalid curve tensor')
    mean = errors.mean(axis=(0,2)); form, scores = choose_form(n,mean,cfg)
    fits = {}
    for f in cfg['forms']:
        try: fits[f] = fit_curve(n,mean,f)
        except (ValueError,RuntimeError) as error: fits[f] = {'fit_failure':str(error)}
    if 'fit_failure' in fits[form]:
        raise RuntimeError('Selected form failed on the full calibration range; preserve the failure record')
    fitted = fits[form]; expected = predict(future_n,fitted)
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    draws = []; pars = []; selections = {f:0 for f in cfg['forms']}
    r,k,m = errors.shape; failures = 0; selection_failures = 0
    for b in range(cfg['bootstrap_repetitions']):
        rr = rng.integers(r,size=r); mm = rng.integers(m,size=m)
        curve = errors[rr][:,:,mm].mean(axis=(0,2))
        try:
            fit = fit_curve(n,curve,form)
            draws.append(predict(future_n,fit))
            physical = np.array(fit['parameters'],dtype=float)
            if form == 'constant': physical[0] *= fit['scale']
            elif form == 'power': physical[0] *= fit['scale']
            else: physical[:2] *= fit['scale']
            pars.append(physical)
        except (ValueError,RuntimeError):
            failures += 1
            continue
        if b < min(100,cfg['bootstrap_repetitions']):
            try: selections[choose_form(n,curve,cfg)[0]] += 1
            except (ValueError,RuntimeError): selection_failures += 1
    reliable = len(draws) >= .95*cfg['bootstrap_repetitions']
    if not draws:
        draws = [expected]
        physical = np.array(fitted['parameters'],dtype=float)
        physical[:1 if form in ['constant','power'] else 2] *= fitted['scale']
        pars = [physical]
    draws = np.asarray(draws); alpha = cfg['alpha']
    pointwise = np.quantile(draws,[alpha/2,1-alpha/2],axis=0)
    simultaneous_radius = float(np.quantile(np.max(np.abs(np.log(draws/expected)),axis=1),1-alpha))
    return dict(selected_form=form,rolling_log_RMSE=scores,fit=fitted,candidate_fits=fits,
                calibration_n=list(map(int,n)),calibration_mean_MAE=mean.tolist(),future_n=list(map(int,future_n)),
                predictions=expected.tolist(),pointwise_95_interval=pointwise.tolist(),
                simultaneous_95_band=[(expected*np.exp(-simultaneous_radius)).tolist(),(expected*np.exp(simultaneous_radius)).tolist()],
                parameter_95_interval=np.quantile(np.asarray(pars),[alpha/2,1-alpha/2],axis=0).tolist(),
                bootstrap_failures=failures,bootstrap_reliable=reliable,
                selection_failures=selection_failures,selection_counts_first_100_bootstraps=selections,
                persistence_prediction=float(mean[-1]),interval_meaning='conditional fitted-mean uncertainty, not a guaranteed predictive interval')


def confirm(frozen, observed, cfg):
    actual = np.asarray(observed,dtype=float); estimate = np.asarray(frozen['predictions'])
    if actual.shape != estimate.shape or (actual <= 0).any(): raise ValueError('Invalid confirmation values')
    rel = np.abs(estimate-actual)/actual
    persistence = float(np.mean(np.abs(frozen['persistence_prediction']-actual)/actual))
    improvement = 1-float(rel.mean())/persistence if persistence > 1e-12 else 0.
    good = bool(frozen.get('bootstrap_reliable',True) and rel.mean() <= cfg['mean_relative_error_max'] and rel.max() <= cfg['worst_relative_error_max'] and improvement >= cfg['improvement_over_persistence_min'] and not frozen['fit']['slope_boundary'])
    kind = 'power_law' if frozen['selected_form'] in ['power','floor_power'] else 'piecewise_power' if frozen['selected_form'].startswith('broken') else 'alternative_shape'
    return dict(observed_MAE=actual.tolist(),relative_errors=rel.tolist(),mean_relative_error=float(rel.mean()),worst_relative_error=float(rel.max()),
                persistence_mean_relative_error=persistence,improvement_over_persistence=improvement,
                predictive_criteria_passed=good,selected_shape=kind,
                conclusion=('supported_' + kind) if good else 'insufficient_predictive_evidence')


def exact_sign_flip(differences):
    d = np.asarray(differences,dtype=float)
    if len(d) > 16: raise ValueError('Use at most 16 independent repetition blocks')
    if not np.isfinite(d).all(): raise ValueError('Nonfinite paired difference')
    observed = abs(float(d.mean()))
    values = [abs(float((d*np.asarray(sign)).mean())) for sign in product([-1,1],repeat=len(d))]
    return float(np.mean(np.asarray(values) >= observed-1e-15))


def holm(pvalues):
    p = np.asarray(pvalues,dtype=float)
    order = np.argsort(p); result = np.empty_like(p)
    result[order] = np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return result


def fit_joint(n, parameters, errors):
    """Prespecified separable N/P law; use measured trainable parameter counts."""
    n,p,y = map(lambda v:np.asarray(v,dtype=float),(n,parameters,errors))
    if len(y) < 8 or np.unique(p).size < 3 or (y <= 0).any(): raise ValueError('Insufficient joint-scaling grid')
    ns,ps,ys = 1000.,float(np.median(p)),float(np.median(y))
    x,z,target = n/ns,p/ps,y/ys
    def evaluate(v): return v[0]+v[1]*x**(-v[3])+v[2]*z**(-v[4])
    solutions = []
    for a,b in product([.2,.5,1.],[.2,.5,1.]):
        solution = least_squares(lambda v:np.log(evaluate(v))-np.log(target),[.01,1.,.1,a,b],bounds=([0,1e-12,1e-12,0,0],[float(target.min()*.999),1e5,1e5,3,3]),max_nfev=3000)
        if solution.success: solutions.append(solution)
    if not solutions: raise RuntimeError('Joint scaling fit did not converge')
    best = min(solutions,key=lambda v:v.cost)
    return dict(parameters=best.x.tolist(),n_scale=ns,p_scale=ps,y_scale=ys,log_RMSE=float(np.sqrt(2*best.cost/len(y))))


def predict_joint(n,p,fit):
    c,a,b,alpha,beta = fit['parameters']
    return fit['y_scale']*(c+a*(np.asarray(n)/fit['n_scale'])**(-alpha)+b*(np.asarray(p)/fit['p_scale'])**(-beta))

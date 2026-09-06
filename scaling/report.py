"""Regenerate manuscript tables, predictions audits and figures from sealed runs."""
from itertools import combinations
import numpy as np
import pandas as pd
from .io import check_file, read_json, sha256, write_json
from .laws import confirm, exact_sign_flip, holm, fit_curve, fit_joint, predict_joint


def analyze(work):
    from .cli import assert_seal, assert_hyper_lock, complete
    assert_seal(work); assert_hyper_lock(work)
    rows = []
    for item in work.jobs:
        if item['stage']=='tune': continue
        result = complete(work,item); folder=work.folder/'runs'/item['id']
        evaluation = read_json(folder/'evaluation.json')
        if evaluation['seal_sha256'] != sha256(work.folder/'seal.json'): raise ValueError('Evaluation belongs to another seal')
        check_file(folder/'test.npz',evaluation['predictions_sha256'])
        row = {k:v for k,v in item.items() if k not in ['params']}
        row.update({k:v for k,v in evaluation['metrics'].items() if k != 'n'}); row['n_test']=evaluation['metrics']['n']; row['law_MAE']=result['metrics']['law']['MAE']
        row['train_MAE']=result['metrics']['train']['MAE']; row['stop_MAE']=result['metrics']['stop']['MAE']
        row['median_baseline_MAE']=evaluation['median_baseline']['MAE']; row['composition_baseline_MAE']=evaluation['composition_baseline']['MAE']
        row['trainable_parameters']=result['training']['trainable_parameters']; row['fit_seconds']=result['fit_seconds']
        row['stop_reason']=result['training']['stop_reason']; row['unit']=result['unit']
        rows.append(row)
    frame=pd.DataFrame(rows); out=work.folder/'analysis'; out.mkdir(exist_ok=True)
    frame.to_csv(out/'all_runs.csv',index=False)
    main=frame[frame.stage.isin(['calibration','confirmation'])]
    main.groupby(['route','task','n'],sort=True).agg(mean_MAE=('MAE','mean'),std_MAE=('MAE','std'),mean_RMSE=('RMSE','mean'),repetitions=('MAE','size')).to_csv(out/'learning_curves.csv')
    frozen=read_json(work.folder/'forecasts.json'); confirmation={}; descriptive={}
    for key,forecast in frozen.items():
        route,task=key.split('/'); group=main[(main.route==route)&(main.task==task)]
        means=group.groupby('n')[['law_MAE','MAE']].mean()
        validation=confirm(forecast,means.loc[forecast['future_n'],'law_MAE'].to_numpy(),work.cfg['analysis'])
        test=confirm(forecast,means.loc[forecast['future_n'],'MAE'].to_numpy(),work.cfg['analysis'])
        confirmation[key]=dict(validation=validation,independent_test=test,
                              replicated=validation['predictive_criteria_passed'] and test['predictive_criteria_passed'])
        descriptive[key]=fit_curve(means.index.to_numpy(),means.MAE.to_numpy(),forecast['selected_form'])
    write_json(out/'forecast_confirmation.json',confirmation)
    write_json(out/'descriptive_full_range_fits.json',descriptive)
    tests=[]; nstar=work.cfg['analysis']['primary_comparison_n']
    for task in work.cfg['targets']:
        for a,b in combinations(work.cfg['primary_routes'],2):
            aa=main[(main.route==a)&(main.task==task)&(main.n==nstar)].set_index('subset_seed').MAE
            bb=main[(main.route==b)&(main.task==task)&(main.n==nstar)].set_index('subset_seed').MAE
            difference=aa-bb
            tests.append(dict(task=task,contrast=f'{a} - {b}',N=nstar,mean_difference=float(difference.mean()),p=exact_sign_flip(difference.to_numpy())))
    for route in work.cfg['primary_routes']:
        a=main[(main.route==route)&(main.task=='G_residual')&(main.n==nstar)].set_index('subset_seed').MAE
        b=main[(main.route==route)&(main.task=='G')&(main.n==nstar)].set_index('subset_seed').MAE
        difference=a-b
        tests.append(dict(task='G',contrast=f'{route}: residual - direct',N=nstar,mean_difference=float(difference.mean()),p=exact_sign_flip(difference.to_numpy())))
    adjusted=holm([t['p'] for t in tests])
    for t,p in zip(tests,adjusted,strict=True): t['holm_p']=float(p)
    pd.DataFrame(tests).to_csv(out/'paired_primary_tests.csv',index=False)
    for name,stages in [('representation_controls',['controls']),('capacity_controls',['capacity']),('budget_controls',['budget']),('validation_budget',['validation_budget']),('seed_variance',['seed_variance']),('split_robustness',['robustness']),('aggregation',['aggregation'])]:
        frame[frame.stage.isin(stages)].to_csv(out/f'{name}.csv',index=False)
    # Capacity forecasts use only small N and widths <= 2; width=4 and large N are held out.
    joint={}
    for route in work.cfg['capacity']['routes']:
        for task in work.cfg['targets']:
            base=main[(main.route==route)&(main.task==task)&main.n.isin(work.cfg['capacity']['sizes'])&main.subset_seed.isin(work.cfg['subset_seeds'][:3])].copy()
            base['width_multiplier']=1.
            extra=frame[(frame.stage=='capacity')&(frame.route==route)&(frame.task==task)]
            grid=pd.concat([base,extra]).groupby(['n','width_multiplier'],as_index=False).agg(P=('trainable_parameters','mean'),MAE=('MAE','mean'),law_MAE=('law_MAE','mean'))
            train=(grid.n<=work.cfg['calibration_max_n'])&(grid.width_multiplier<=2)
            try:
                fit=fit_joint(grid.loc[train,'n'],grid.loc[train,'P'],grid.loc[train,'law_MAE'])
                held=grid.loc[~train].copy(); held['prediction']=predict_joint(held.n,held.P,fit)
                held['relative_error']=(held.prediction-held.law_MAE).abs()/held.law_MAE
                joint[f'{route}/{task}']=dict(fit=fit,heldout_mean_relative_error=float(held.relative_error.mean()),heldout_max_relative_error=float(held.relative_error.max()),
                    interpretation='joint N/trainable-P diagnostic; not a FLOP law, not total pretrained encoder parameter scaling')
                held.to_csv(out/f'joint_{route}_{task}.csv',index=False)
            except (ValueError,RuntimeError) as error:
                joint[f'{route}/{task}']=dict(status='not_identifiable',reason=str(error))
    write_json(out/'joint_scaling.json',joint)
    # Whole-trajectory bootstrap for crossover bands. These are descriptive, not extra hypothesis tests.
    crossovers={}; rng=np.random.default_rng(work.cfg['analysis']['bootstrap_seed'])
    for task in work.cfg['targets']:
        for a,b in combinations(work.cfg['primary_routes'],2):
            left=main[(main.route==a)&(main.task==task)].pivot(index='subset_seed',columns='n',values='MAE').sort_index()
            right=main[(main.route==b)&(main.task==task)].pivot(index='subset_seed',columns='n',values='MAE').reindex(index=left.index,columns=left.columns)
            d=(left-right).to_numpy(); point=d.mean(axis=0)
            bs=d[rng.integers(len(d),size=(1000,len(d)))].mean(axis=1)
            sd=bs.std(axis=0); sd=np.maximum(sd,1e-15)
            q=float(np.quantile(np.max(np.abs((bs-point)/sd),axis=1),.95)); lo,hi=point-q*sd,point+q*sd
            sign=np.where(lo>0,1,np.where(hi<0,-1,0)); n=left.columns.to_numpy(); intervals=[]
            for i in range(1,len(n)-1):
                prior=np.flatnonzero(sign[:i]!=0)
                if sign[i]!=0 and sign[i+1]==sign[i] and len(prior) and sign[prior[-1]]!=sign[i]: intervals.append([int(n[prior[-1]]),int(n[i])])
            crossovers[f'{task}/{a}-vs-{b}']=dict(n=n.tolist(),mean_difference=point.tolist(),simultaneous_95_lower=lo.tolist(),simultaneous_95_upper=hi.tolist(),observed_crossing_intervals=intervals)
    write_json(out/'crossovers.json',crossovers)
    # Finite-test-set diagnostic, not a population noise floor and never a trained predictor.
    with np.load(work.root/'features/morgan.npz') as f: packed=f['packed']
    example=next(j for j in work.jobs if j['stage']=='confirmation' and j['task']=='G')
    test_rows=work.rows(example)['test']; _,inverse=np.unique(packed[test_rows],axis=0,return_inverse=True)
    bounds={}
    for task in work.cfg['targets']:
        y=work.frame[task].to_numpy()[test_rows]; group_median=pd.Series(y).groupby(inverse).transform('median').to_numpy()
        bounds[task]=float(np.abs(y-group_median).mean())
    write_json(out/'fingerprint_test_diagnostic.json',dict(empirical_equal_input_MAE_lower_bound=bounds,
        meaning='finite test-set diagnostic for raw-target binary Morgan only; not a population Bayes error estimate'))
    from .diagnostics import diagnostics
    diagnostics(work,frame,out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for task in work.cfg['tasks']:
        fig,ax=plt.subplots(figsize=(7,5))
        for route in work.cfg['primary_routes']:
            g=main[(main.task==task)&(main.route==route)].groupby('n').MAE.agg(['mean','std'])
            ax.plot(g.index,g['mean'],marker='o',label=route)
            ax.fill_between(g.index,np.maximum(g['mean']-g['std'],1e-15),g['mean']+g['std'],alpha=.15)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Training molecules'); ax.set_ylabel(f'Test MAE ({"Debye" if task=="mu" else "Hartree"})')
        ax.set_title(f'{task}: mean and one standard deviation across repetitions'); ax.legend(); fig.tight_layout()
        fig.savefig(out/f'learning_curve_{task}.png',dpi=180); fig.savefig(out/f'learning_curve_{task}.svg'); plt.close(fig)
    write_json(out/'publication_audit.json',dict(planned_runs=len(work.jobs),evaluated_runs=len(frame),all_non_tuning_runs_evaluated=len(frame)==sum(j['stage']!='tune' for j in work.jobs),
        protocol_sha256=sha256(work.folder/'protocol.json'),seal_sha256=sha256(work.folder/'seal.json'),
        claims_must_follow_forecast_and_control_results=True,negative_findings_retained=True,
        scientific_publication_or_positive_law_not_guaranteed=True))

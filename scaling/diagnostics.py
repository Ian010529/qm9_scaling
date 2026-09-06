"""Prespecified explanatory contrasts. These do not select or replace primary runs."""
import numpy as np
import pandas as pd
from .io import write_json


def diagnostics(work, frame, out):
    main=frame[frame.stage.isin(['calibration','confirmation'])]
    keys=['route','task','n','subset_seed','training_seed']
    reference={tuple(row[k] for k in keys):row for _,row in main.iterrows()}
    contrasts=[]
    for _,row in frame[frame.stage.isin(['capacity','budget','aggregation'])].iterrows():
        key=tuple(row[k] for k in keys); base=reference.get(key)
        if row.get('arm')=='large_capacity_budget':
            candidates=frame[(frame.stage=='capacity')&(frame.route==row.route)&(frame.task==row.task)&(frame.n==row.n)&(frame.subset_seed==row.subset_seed)&(frame.training_seed==row.training_seed)&(frame.width_multiplier==row.width_multiplier)]
            base=candidates.iloc[0] if len(candidates)==1 else None
        if base is None: continue
        contrasts.append(dict(stage=row.stage,arm=row.get('arm'),route=row.route,task=row.task,n=int(row.n),subset_seed=int(row.subset_seed),
                              variant_run=row.id,reference_run=base.id,reference_MAE=float(base.MAE),variant_MAE=float(row.MAE),
                              relative_improvement=float(1-row.MAE/base.MAE),fit_seconds_ratio=float(row.fit_seconds/base.fit_seconds) if base.fit_seconds>0 else None))
    pd.DataFrame(contrasts).to_csv(out/'paired_capacity_budget_aggregation.csv',index=False)
    flags=[]
    budget=[x for x in contrasts if x['stage']=='budget']
    if budget:
        for key,g in pd.DataFrame(budget).groupby(['arm','route','task','n']):
            gain=float(g.relative_improvement.mean())
            flags.append(dict(arm=key[0],route=key[1],task=key[2],n=int(key[3]),mean_relative_improvement=gain,
                training_budget_sensitive=gain>.05,meaning='5% is a prespecified diagnostic threshold, not a universal convergence guarantee'))
    write_json(out/'optimization_diagnostics.json',flags)
    # Balanced 3x3 subset/initialization grid. With one observation per cell, interaction
    # and residual stochasticity cannot be separately estimated.
    crossed=pd.concat([main,frame[frame.stage=='seed_variance']]); components=[]
    for route in work.cfg['primary_routes']:
        for task in work.cfg['targets']:
            for n in work.cfg['budget']['sizes']:
                g=crossed[(crossed.route==route)&(crossed.task==task)&(crossed.n==n)&crossed.subset_seed.isin(work.cfg['subset_seeds'][:3])&crossed.training_seed.isin(work.cfg['training_seeds'][:3])]
                table=g.pivot(index='subset_seed',columns='training_seed',values='MAE')
                if table.shape!=(3,3) or table.isna().any().any(): continue
                a=table.to_numpy(); grand=a.mean(); r=a.mean(axis=1); c=a.mean(axis=0)
                msr=3*np.sum((r-grand)**2)/2; msc=3*np.sum((c-grand)**2)/2
                interaction=float(np.sum((a-r[:,None]-c[None,:]+grand)**2)/4)
                vr=float((msr-interaction)/3); vc=float((msc-interaction)/3)
                components.append(dict(route=route,task=task,n=n,subset_variance_untruncated=vr,initialization_variance_untruncated=vc,
                    subset_variance_nonnegative=max(0.,vr),initialization_variance_nonnegative=max(0.,vc),interaction_plus_residual=interaction))
    pd.DataFrame(components).to_csv(out/'variance_components.csv',index=False)
    exclusions=pd.read_csv(work.root/'data/processed/exclusions.csv')
    duplicates=exclusions[exclusions.reason=='duplicate_nonisomeric_smiles']
    raw=pd.read_csv(work.root/'data/raw/qm9_smiles.csv').set_index('mol_id',verify_integrity=True)
    audit=[]
    for row in duplicates.itertuples():
        item=dict(removed_id=row.mol_id,retained_id=row.retained_id)
        for task,source in [('mu','mu'),('homo','homo'),('lumo','lumo'),('G','g298')]:
            item[f'{task}_removed_minus_retained']=float(raw.loc[row.mol_id,source]-raw.loc[row.retained_id,source])
        audit.append(item)
    pd.DataFrame(audit).to_csv(out/'duplicate_label_audit.csv',index=False)
    write_json(out/'duplicate_label_audit.json',dict(removed_rows=len(audit),policy='Audit only; neither representatives nor labels are changed after evaluation'))

"""Synthetic software tests only. These are not empirical QM9 findings."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import numpy as np
import pandas as pd
import pytest
from scaling.design import candidates, matrix, parameters
from scaling.io import ROOT, check_file, digest, partition_check, read_json, sha256, write_json
from scaling.laws import fit_curve,predict,choose_form,confirm,exact_sign_flip,holm,fit_joint,predict_joint
from scaling.train import target_transform,run_one,predict_saved,neural_model,batch_input

CFG=read_json(ROOT/'config/scaling_law_v1.json')


def test_matrix_and_unique_identity():
    jobs=matrix(CFG)
    assert len(jobs)==6439
    assert len({j['id'] for j in jobs})==len(jobs)
    assert len([j for j in jobs if j['stage']=='tune'])==1008
    assert len([j for j in jobs if j['arm']=='primary'])==1800


def test_original_targets_models_and_sizes_preserved():
    assert CFG['targets']==['mu','homo','lumo','G']
    assert CFG['primary_routes']==['morgan_xgb','dmpnn','molformer_mlp']
    assert set([500,1000,2000,5000,10000,20000,50000,100000]) <= set(CFG['train_sizes'])
    assert len([n for n in CFG['train_sizes'] if n>20000])==4


def test_robustness_does_not_inherit_main_tuning():
    for item in matrix(CFG):
        if item['stage']=='robustness':
            assert item['preset']
            assert parameters(CFG,item,None)
    assert 'random' in CFG['robustness']['splits']


def test_hpo_reproducible_and_unique():
    for route in CFG['primary_routes']+['morgan_mlp','molformer_ridge']:
        a,b=candidates(CFG,route),candidates(CFG,route)
        assert a==b and len({digest(v) for v in a})==len(a)


def test_partition_overlap_and_duplicates_fail():
    ids=np.array(['a','b','c'])
    partition_check(ids,{'train':['a'],'val':['b'],'test':['c']})
    for splits in [{'train':['a','b'],'test':['b','c']},{'train':['a','a'],'test':['b','c']},{'train':['a'],'test':['b']}]:
        with pytest.raises(ValueError): partition_check(ids,splits)


def test_mutated_file_and_lfs_fail(tmp_path):
    p=tmp_path/'x';p.write_bytes(b'original');h=sha256(p);check_file(p,h)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError):check_file(p,h)
    p.write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError):check_file(p,sha256(p))


def test_strict_json(tmp_path):
    with pytest.raises(ValueError):write_json(tmp_path/'bad.json',{'x':float('nan')})
    p=tmp_path/'bad.json';p.write_text('{"x":NaN}')
    with pytest.raises(ValueError):read_json(p)


def test_target_scaler_and_residual_use_train_only():
    rng=np.random.default_rng(7);c=np.c_[np.ones(50),rng.integers(0,6,(50,5))].astype(float)
    y=c@np.array([-3.,-40.,-55.,-70.,-99.,-1.])+rng.normal(size=50)*.1
    for residual in [False,True]:
        a=target_transform(y,c,np.arange(20),residual)
        changed=y.copy();changed[20:]+=1e8
        b=target_transform(changed,c,np.arange(20),residual)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        scaled=(y-c@a['coef']-a['mean'])/a['scale']
        np.testing.assert_allclose(scaled*a['scale']+a['mean']+c@a['coef'],y,atol=1e-12)


@pytest.mark.parametrize('form',['constant','power','floor_power'])
def test_known_laws_and_selection(form):
    n=np.array([v for v in CFG['train_sizes'] if v<=20000])
    if form=='constant':y=np.full(len(n),.02)
    elif form=='power':y=.02*(n/1000)**-.4
    else:y=.001+.02*(n/1000)**-.4
    f=fit_curve(n,y,form)
    np.testing.assert_allclose(predict(n,f),y,rtol=1e-5,atol=1e-9)
    assert choose_form(n,y,CFG['analysis'])[0]==form


def test_confirmation_rejects_bad_extrapolation():
    f=dict(predictions=[.1,.08],persistence_prediction=.2,selected_form='power',fit={'slope_boundary':False})
    assert confirm(f,[.1,.08],CFG['analysis'])['predictive_criteria_passed']
    assert not confirm(f,[.2,.2],CFG['analysis'])['predictive_criteria_passed']


def test_exact_permutation_and_holm():
    assert exact_sign_flip([0,0,0])==1.
    assert exact_sign_flip([1]*10)==pytest.approx(2/1024)
    np.testing.assert_allclose(holm([.001,.02,.03]),[.003,.04,.04])


def test_joint_law_recovers_predictions():
    n,p=np.meshgrid([500,1000,2000,5000,10000,20000],[10000,40000,160000]);n=n.ravel();p=p.ravel()
    y=.002+.02*(n/1000)**-.4+.003*(p/40000)**-.3
    f=fit_joint(n,p,y)
    np.testing.assert_allclose(predict_joint(n,p,f),y,rtol=1e-4)


def synthetic_workspace(route,task='mu'):
    rng=np.random.default_rng(12);x=rng.normal(size=(120,12)).astype('float32')
    comp=np.c_[np.ones(120),rng.integers(0,8,(120,5))].astype(float)
    y=2*x[:,0]+.3*x[:,1]+rng.normal(size=120)*.01
    if task=='G_residual':y=y+comp@np.array([0,-40,-55,-70,-90,-1])
    frame=pd.DataFrame({'mol_id':[f'm{i}' for i in range(120)],'mu':y,'G':y})
    cfg=deepcopy(CFG);cfg['training'].update(max_epochs=2,minimum_max_updates=120,minimum_updates=20,eval_interval_min=10,patience=8,batch_size=32)
    indices={'train':np.arange(70),'stop':np.arange(70,90),'tune':np.arange(90,100),'law':np.arange(100,110),'test':np.arange(110,120)}
    return SimpleNamespace(cfg=cfg,frame=frame,composition=comp,features=lambda _:x,rows=lambda _:indices)


@pytest.mark.parametrize('route',['morgan_xgb','molformer_ridge','molformer_mlp','morgan_mlp'])
def test_model_adapters_save_reload_and_units(tmp_path,route):
    w=synthetic_workspace(route)
    p=deepcopy(CFG['defaults']['morgan_xgb' if route=='morgan_xgb' else 'molformer_ridge' if route=='molformer_ridge' else 'molformer_mlp'])
    if route=='morgan_xgb':p.update(n_estimators=100,early_stopping_rounds=10,learning_rate=.1,max_depth=3)
    elif route=='molformer_ridge':p['alpha']=.001
    else:p.update(hidden=24,layers=1,lr=.01,dropout=0.)
    item=dict(route=route,task='mu',stage='calibration',training_seed=101)
    out=tmp_path/route;result=run_one(w,item,p,out,threads=1)
    a=predict_saved(w,item,out,'test');b=predict_saved(w,item,out,'test')
    np.testing.assert_array_equal(a,b)
    assert np.isfinite(a).all() and result['unit']=='Debye'
    assert (out/'complete.json').exists()
    assert result['metrics']['law']['MAE'] < result['metrics']['law']['median_baseline_MAE']


def test_residual_adapter_reports_original_energy_units(tmp_path):
    w=synthetic_workspace('molformer_ridge','G_residual')
    item=dict(route='molformer_ridge',task='G_residual',stage='calibration',training_seed=101)
    result=run_one(w,item,{'alpha':.001},tmp_path/'residual',threads=1)
    assert result['unit']=='Hartree'
    pred=predict_saved(w,item,tmp_path/'residual','test')
    assert np.mean(np.abs(pred-w.frame.G.to_numpy()[w.rows(item)['test']])) < 1.


def test_missing_seal_fails_closed(tmp_path):
    from scaling.cli import assert_seal
    with pytest.raises(FileNotFoundError): assert_seal(SimpleNamespace(folder=tmp_path))


@pytest.mark.skipif(importlib.util.find_spec('chemprop') is None,reason='Chemprop not installed in this test environment')
def test_real_chemprop_forward_backward():
    import torch
    from rdkit import Chem
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.data.collate import BatchMolGraph
    fg=SimpleMoleculeMolGraphFeaturizer();batch=BatchMolGraph([fg(Chem.MolFromSmiles(s)) for s in ['C','CC','CCO']])
    p={**CFG['defaults']['dmpnn'],'hidden':32,'aggregation':'sum'}
    model=neural_model('dmpnn',0,p);pred=model(batch)
    assert pred.shape==(3,1) and torch.isfinite(pred).all()
    pred.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_crossed_bootstrap_forecast_and_confirmation():
    from scaling.laws import forecast
    cfg=deepcopy(CFG['analysis']);cfg['bootstrap_repetitions']=6
    n=np.array([v for v in CFG['train_sizes'] if v<=20000]);future=np.array([30000,50000,75000,100000])
    rng=np.random.default_rng(23)
    curve=.03*(n/1000)**-.45
    e=curve[None,:,None]*rng.uniform(.97,1.03,(4,1,25))
    result=forecast(n,e,future,cfg)
    assert result['bootstrap_reliable']
    assert result['selected_form']=='power'
    assert np.asarray(result['simultaneous_95_band']).shape==(2,4)
    assert confirm(result,.03*(future/1000)**-.45,cfg)['predictive_criteria_passed']


def test_report_end_to_end_preserves_training_size(tmp_path,monkeypatch):
    """Synthetic prediction records test reporting, not training or scientific performance."""
    from itertools import product
    from scaling.report import analyze
    import scaling.cli as cli
    cfg=deepcopy(CFG);cfg['subset_seeds']=[11,22];cfg['training_seeds']=[101,102]
    folder=tmp_path/'study';folder.mkdir();root=tmp_path/'repo';(root/'features').mkdir(parents=True)
    np.savez_compressed(root/'features/morgan.npz',packed=np.array([[1,0],[1,0],[0,1],[0,2]],dtype='uint8'))
    write_json(folder/'seal.json',{'synthetic':True});write_json(folder/'protocol.json',cfg)
    jobs=[];results={};forecasts={}
    for route,task,n,s in product(cfg['primary_routes'],cfg['tasks'],cfg['train_sizes'],cfg['subset_seeds']):
        jobs.append(dict(id=f'main-{route}-{task}-{n}-{s}',stage='calibration' if n<=20000 else 'confirmation',route=route,task=task,n=n,subset_seed=s,training_seed=101 if s==11 else 102,split='random',arm='primary'))
    for route,task,n,s,w in product(cfg['capacity']['routes'],cfg['targets'],cfg['capacity']['sizes'],cfg['subset_seeds'],cfg['capacity']['multipliers']):
        jobs.append(dict(id=f'cap-{route}-{task}-{n}-{s}-{w}',stage='capacity',route=route,task=task,n=n,subset_seed=s,training_seed=101 if s==11 else 102,split='random',arm='capacity',width_multiplier=w))
    frame=pd.DataFrame({'mol_id':['m0','m1','m2','m3'],**{t:np.arange(4,dtype=float) for t in cfg['targets']}})
    (root/'data/processed').mkdir(parents=True);(root/'data/raw').mkdir(parents=True)
    pd.DataFrame(columns=['mol_id','reason','retained_id']).to_csv(root/'data/processed/exclusions.csv',index=False)
    frame.rename(columns={'G':'g298'}).to_csv(root/'data/raw/qm9_smiles.csv',index=False)
    for j in jobs:
        directory=folder/'runs'/j['id'];directory.mkdir(parents=True)
        n=j['n'];p=10000*j.get('width_multiplier',1.)**2
        error=.001+.02*(n/1000)**-.4+.003*(p/10000)**-.3
        np.savez_compressed(directory/'test.npz',y_true=np.arange(4.),y_pred=np.arange(4.)+error)
        write_json(directory/'evaluation.json',dict(seal_sha256=sha256(folder/'seal.json'),predictions_sha256=sha256(directory/'test.npz'),metrics={'n':4,'MAE':error,'RMSE':error},median_baseline={'MAE':1.},composition_baseline={'MAE':.1}))
        results[j['id']]=dict(metrics={k:{'MAE':error} for k in ['train','stop','law']},training={'trainable_parameters':p,'stop_reason':'synthetic'},fit_seconds=0.,unit='Debye' if j['task']=='mu' else 'Hartree')
    future=[n for n in cfg['train_sizes'] if n>20000]
    for r,t in product(cfg['primary_routes'],cfg['tasks']):
        forecasts[f'{r}/{t}']=dict(selected_form='floor_power',predictions=(.004+.02*(np.array(future)/1000)**-.4).tolist(),future_n=future,persistence_prediction=float(.004+.02*20**-.4),fit={'slope_boundary':False},bootstrap_reliable=True)
    write_json(folder/'forecasts.json',forecasts)
    monkeypatch.setattr(cli,'assert_seal',lambda w:None);monkeypatch.setattr(cli,'assert_hyper_lock',lambda w:None)
    monkeypatch.setattr(cli,'complete',lambda w,j:results[j['id']])
    work=SimpleNamespace(cfg=cfg,folder=folder,root=root,jobs=jobs,frame=frame,rows=lambda _:dict(test=np.arange(4)))
    analyze(work)
    table=pd.read_csv(folder/'analysis/all_runs.csv')
    assert table.n.max()==100000 and table.n_test.eq(4).all()
    assert len(pd.read_csv(folder/'analysis/paired_primary_tests.csv'))==15
    assert read_json(folder/'analysis/publication_audit.json')['all_non_tuning_runs_evaluated']
    assert len(list((folder/'analysis').glob('learning_curve_*.png')))==5

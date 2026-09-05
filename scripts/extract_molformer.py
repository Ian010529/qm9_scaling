"""Frozen MoLFormer embeddings with explicit revision, FP32 and resumable checkpoints."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from common import ROOT, config, dataset, sha256, versions, write_json

COMMIT='7b12d946c181a37f6012b9dc3b002275de070314'


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu')
    p.add_argument('--model-path',type=Path)
    p.add_argument('--revision',default=COMMIT)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-batches',type=int)
    args=p.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.threads);torch.manual_seed(0)
    torch.set_float32_matmul_precision('highest')
    frame=dataset();cfg=config()['molformer'];out=ROOT/'features';out.mkdir(exist_ok=True)
    final=out/'molformer.npy';partial=out/'molformer.partial.npy';progress=out/'molformer_progress.json'
    if final.exists(): raise FileExistsError('Final embeddings already exist')
    source=str(args.model_path) if args.model_path else cfg['model_id']
    kw={'trust_remote_code':True}
    if args.model_path: kw['local_files_only']=True
    else: kw['revision']=args.revision
    tokenizer=AutoTokenizer.from_pretrained(source,**kw)
    model,loading=AutoModel.from_pretrained(source,deterministic_eval=True,output_loading_info=True,**kw)
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise RuntimeError(f'Incomplete model weights: {loading}')
    model.requires_grad_(False);model.eval().to(args.device)
    dim=model.config.hidden_size
    metadata={'dataset_sha256':sha256(ROOT/'data/processed/qm9.csv.gz'),'model_id':cfg['model_id'],
              'revision':args.revision,'rows':len(frame),'dimension':dim,'pooling':'pooler_output',
              'precision':'float32','encoder_seed':0,'deterministic_eval':True,
              'batch_size':args.batch_size,'versions':versions()}
    if args.model_path:
        metadata['model_files_sha256']={f.name:sha256(f) for f in sorted(args.model_path.iterdir())
                                        if f.is_file() and (f.suffix in ['.py','.json','.safetensors'])}
    else: metadata['resolved_commit']=getattr(model.config,'_commit_hash',None)
    def encode(smiles):
        inputs=tokenizer(smiles,padding=True,truncation=False,return_tensors='pt')
        if inputs['input_ids'].shape[1]>model.config.max_position_embeddings: raise ValueError('Overlong SMILES; no truncation')
        if tokenizer.unk_token_id is not None and (inputs['input_ids']==tokenizer.unk_token_id).any(): raise ValueError('Unknown token')
        result=model(**{k:v.to(args.device) for k,v in inputs.items()}).pooler_output
        if not torch.isfinite(result).all(): raise ValueError('Nonfinite embeddings')
        return result.cpu().numpy().astype('float32',copy=False)
    with torch.inference_mode():
        sample=frame.smiles.iloc[:8].tolist();v1=encode(sample);v2=encode(sample)
        repeat_diff=float(np.abs(v1-v2).max())
        if repeat_diff!=0: raise ValueError(f'Non-deterministic repeated evaluation: {repeat_diff}')
    start=0;prior_seconds=0
    if partial.exists():
        if not args.resume or not progress.exists(): raise FileExistsError('Use --resume for matching incomplete extraction')
        old=json.loads(progress.read_text())
        for key,value in metadata.items():
            if old['metadata'].get(key)!=value: raise ValueError(f'Resume metadata mismatch: {key}')
        start=old['completed_rows'];prior_seconds=old['elapsed_seconds']
        vectors=np.lib.format.open_memmap(partial,mode='r+')
        if vectors.shape!=(len(frame),dim): raise ValueError('Partial shape mismatch')
    else:
        vectors=np.lib.format.open_memmap(partial,mode='w+',dtype='float32',shape=(len(frame),dim))
    started=time.perf_counter();batches=0;last=start
    def checkpoint():
        vectors.flush()
        elapsed=prior_seconds+time.perf_counter()-started
        write_json(progress,{'status':'partial','completed_rows':last,'elapsed_seconds':elapsed,'metadata':metadata})
        print(json.dumps({'completed_rows':last,'total_rows':len(frame),'elapsed_seconds':round(elapsed,2)}),flush=True)
    with torch.inference_mode():
        for i in range(start,len(frame),args.batch_size):
            smiles=frame.smiles.iloc[i:i+args.batch_size].tolist()
            vectors[i:i+len(smiles)]=encode(smiles);last=i+len(smiles);batches+=1
            if batches%16==0: checkpoint()
            if args.stop_after_batches and batches>=args.stop_after_batches:
                checkpoint();print('Partial checkpoint saved; resume to finish',flush=True);return
    checkpoint()
    # Verify molecules spanning the dataset in a new batch (different padding/order).
    indices=np.unique(np.linspace(0,len(frame)-1,32,dtype=int))
    with torch.inference_mode(): check=encode(frame.smiles.iloc[indices].tolist())
    diff=float(np.max(np.abs(check-vectors[indices])))
    if not np.allclose(check,vectors[indices],atol=1e-5,rtol=1e-4): raise ValueError(f'Batch/order consistency failure: {diff}')
    for i in range(0,len(frame),4096):
        if not np.isfinite(vectors[i:i+4096]).all(): raise ValueError('Invalid saved vectors')
    vectors.flush();del vectors
    partial.replace(final)
    np.save(out/'molformer_ids.npy',frame.mol_id.to_numpy(dtype=str),allow_pickle=False)
    metadata.update({'features_sha256':sha256(final),'id_sha256':sha256(out/'molformer_ids.npy'),
                     'elapsed_seconds':prior_seconds+time.perf_counter()-started,'loading_info':loading,
                     'verification':{'all_rows_finite':True,'same_batch_repeat_max_abs_difference':repeat_diff,
                                      'cross_batch_sample_count':len(indices),'cross_batch_max_abs_difference':diff}})
    write_json(out/'molformer.json',metadata)
    write_json(progress,{'status':'complete','completed_rows':len(frame),'metadata':metadata})
    print('Complete',len(frame),dim,flush=True)


if __name__=='__main__': main()

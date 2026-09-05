"""Load aligned inputs for one of the already-frozen training subsets.

No normalization is fitted here. Fit any input/target scaler on train only.
"""
import json
import numpy as np
from common import ROOT, dataset, load_split, row_indices, sha256
from graph_cache import GraphCache


def load_run(size=1000, subset_seed=11, target='homo'):
    frame=dataset()
    dataset_hash=sha256(ROOT/'data/processed/qm9.csv.gz')
    split_ids=load_split(subset_seed,size)
    rows={s:row_indices(frame,ids) for s,ids in split_ids.items()}
    with np.load(ROOT/'features/morgan.npz') as f:
        if not np.array_equal(f['mol_id'],frame.mol_id.to_numpy(dtype=str)):
            raise ValueError('Morgan IDs do not match the data')
        morgan=f['packed']
    graphs=GraphCache()
    morgan_meta=json.loads((ROOT/'features/morgan.json').read_text())
    if morgan_meta['dataset_sha256']!=dataset_hash or graphs.meta['dataset_sha256']!=dataset_hash:
        raise ValueError('Stale Morgan features or graph cache')
    if not np.array_equal(graphs.mol_id,frame.mol_id.to_numpy(dtype=str)):
        raise ValueError('Graph IDs do not match the data')
    embedding=np.load(ROOT/'features/molformer.npy',mmap_mode='r')
    embedding_ids=np.load(ROOT/'features/molformer_ids.npy')
    if not np.array_equal(embedding_ids,frame.mol_id.to_numpy(dtype=str)):
        raise ValueError('MoLFormer IDs do not match the data')
    meta=json.loads((ROOT/'features/molformer.json').read_text())
    if meta['dataset_sha256']!=dataset_hash:
        raise ValueError('Stale embeddings')
    return {
        'ids':split_ids,'rows':rows,'targets':{s:frame[target].to_numpy()[i] for s,i in rows.items()},
        'morgan':{s:np.unpackbits(morgan[i],axis=1,bitorder='little',count=2048) for s,i in rows.items()},
        'graphs':graphs,
        'molformer':{s:embedding[i] for s,i in rows.items()},
    }


if __name__=='__main__':
    data=load_run()
    for s in ['train','val','test']:
        print(s,len(data['ids'][s]),data['morgan'][s].shape,data['molformer'][s].shape)
    g=data['graphs'][str(data['ids']['train'][0])]
    print('First training graph:',g.V.shape,g.E.shape)

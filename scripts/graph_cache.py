"""Generate/load lossless Chemprop 2.2.1 graph inputs for the fixed dataset."""
from pathlib import Path
import argparse
import json
import numpy as np
from rdkit import Chem
from common import ROOT, config, dataset, sha256, versions, write_json


class GraphCache:
    def __init__(self, folder=None):
        self.folder=Path(folder) if folder else ROOT/'features/dmpnn'
        self.meta=json.loads((self.folder/'manifest.json').read_text())
        for name in ['V','E','edge_index','rev_edge_index','atom_ptr','edge_ptr','mol_id']:
            setattr(self,name,np.load(self.folder/f'{name}.npy',mmap_mode='r',allow_pickle=False))
        self.lookup={str(m):i for i,m in enumerate(self.mol_id)}
    def __len__(self): return len(self.mol_id)
    def __getitem__(self,index):
        from chemprop.data.molgraph import MolGraph
        if isinstance(index,str): index=self.lookup[index]
        a,b=self.atom_ptr[index:index+2];e,f=self.edge_ptr[index:index+2]
        return MolGraph(self.V[a:b],self.E[e:f],self.edge_index[:,e:f],self.rev_edge_index[e:f])


def main():
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    df=dataset();out=ROOT/'features/dmpnn';fg=SimpleMoleculeMolGraphFeaturizer()
    if not args.verify_only:
        if out.exists(): raise FileExistsError('Graph output exists; use --verify-only or a new folder')
        out.mkdir(parents=True)
        ap=np.zeros(len(df)+1,dtype=np.int64);ep=np.zeros_like(ap)
        for i,s in enumerate(df.smiles):
            mol=Chem.MolFromSmiles(s)
            if mol is None: raise ValueError(f'Invalid SMILES {df.mol_id.iloc[i]}')
            ap[i+1]=ap[i]+mol.GetNumAtoms();ep[i+1]=ep[i]+2*mol.GetNumBonds()
        av,be=fg.shape
        shape={'V':(int(ap[-1]),av),'E':(int(ep[-1]),be),'edge_index':(2,int(ep[-1])),'rev_edge_index':(int(ep[-1]),)}
        arrays={name:np.lib.format.open_memmap(out/f'{name}.npy',mode='w+',dtype=np.float32 if name in ['V','E'] else np.int32,shape=dims) for name,dims in shape.items()}
        for i,s in enumerate(df.smiles):
            graph=fg(Chem.MolFromSmiles(s));a,b=ap[i:i+2];e,f=ep[i:i+2]
            arrays['V'][a:b]=graph.V;arrays['E'][e:f]=graph.E
            arrays['edge_index'][:,e:f]=graph.edge_index;arrays['rev_edge_index'][e:f]=graph.rev_edge_index
            if (i+1)%20000==0: print('Cached graphs',i+1,flush=True)
        for ar in arrays.values(): ar.flush()
        del arrays
        for name,arr in [('atom_ptr',ap),('edge_ptr',ep),('mol_id',df.mol_id.to_numpy(dtype=str))]: np.save(out/f'{name}.npy',arr,allow_pickle=False)
        write_json(out/'manifest.json',{'rows':len(df),'shapes':shape,'featurizer':'chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer',
            'atom_mode':'MultiHotAtomFeaturizer.v2','extra_features':False,'node_dimension':av,'edge_dimension':be,
            'index_convention':'edge_index and rev_edge_index are LOCAL to each molecule; atom_ptr and edge_ptr slice flattened arrays',
            'precision':'float32 features; int32 graph indices; int64 offsets','dataset_sha256':sha256(ROOT/'data/processed/qm9.csv.gz'),
            'versions':versions(),'files_sha256':{p.name:sha256(p) for p in sorted(out.glob('*.npy'))}})
    cache=GraphCache(out)
    assert cache.meta['dataset_sha256']==sha256(ROOT/'data/processed/qm9.csv.gz')
    assert np.array_equal(cache.mol_id,df.mol_id.to_numpy(dtype=str))
    checks=0;zero_bond=0
    for i,s in enumerate(df.smiles):
        g=cache[i];expected=fg(Chem.MolFromSmiles(s))
        for x,y in zip(g,expected):
            if not np.array_equal(x,y): raise AssertionError(f'Graph differs for {df.mol_id.iloc[i]}')
        n=len(g.V);ne=len(g.E)
        assert np.isfinite(g.V).all() and np.isfinite(g.E).all()
        if ne:
            assert g.edge_index.min()>=0 and g.edge_index.max()<n
            assert np.array_equal(g.rev_edge_index[g.rev_edge_index],np.arange(ne))
            assert np.array_equal(g.edge_index[:,g.rev_edge_index],g.edge_index[::-1])
        else: zero_bond+=1
        checks+=1
        if checks%30000==0: print('Verified graphs',checks,flush=True)
    # Exercise the official collator and a directed message-passing forward pass.
    import torch
    from chemprop.data.collate import BatchMolGraph
    from chemprop.nn import BondMessagePassing
    torch.manual_seed(101)
    batch=BatchMolGraph([cache[i] for i in [0,1,2,100,1000]])
    mp=BondMessagePassing(d_v=fg.shape[0],d_e=fg.shape[1],d_h=32,depth=2)
    mp.eval()
    with torch.inference_mode(): output=mp(batch)
    assert torch.isfinite(output).all()
    verification={'all_graphs_equal_to_official_featurizer':True,'graphs_checked':checks,'zero_bond_graphs':zero_bond,
                  'official_batch_and_message_passing_forward':'passed; randomly initialized diagnostic only, not trained molecular embeddings',
                  'forward_shape':list(output.shape)}
    write_json(out/'verification.json',verification);print(json.dumps(verification,indent=2))


if __name__=='__main__': main()

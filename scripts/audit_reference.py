"""Compare each previous molecule with canonicalized uploaded reference SMILES."""
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from common import ROOT, config, sha256, write_json, write_csv_gz
from download import exclusion_ids


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--previous-project',type=Path,required=True)
    args=p.parse_args()
    ref=pd.read_csv(ROOT/'data/raw/qm9_smiles.csv').set_index('mol_id',verify_integrity=True)
    prev=pd.read_csv(args.previous_project/'data/processed/qm9.csv.gz').set_index('mol_id',verify_integrity=True)
    exclusions=pd.read_csv(args.previous_project/'data/processed/exclusions.csv').set_index('mol_id',verify_integrity=True)
    with np.load(args.previous_project/'features/morgan.npz') as f:
        oldbits=f['packed'];oldids=pd.Index(f['mol_id'])
    with zipfile.ZipFile(ROOT/'data/raw/qm9.zip') as z:
        with z.open('gdb9.sdf.csv') as f: labels=pd.read_csv(f).set_index('mol_id',verify_integrity=True)
    target_checks={}
    for t in ['mu','homo','lumo','g298']:
        delta=np.abs(ref.loc[labels.index,t].to_numpy()-labels[t].to_numpy())
        target_checks[t]={'changed':int((delta>1e-12).sum()),'max_absolute_difference':float(delta.max())}
    gen=rdFingerprintGenerator.GetMorganGenerator(**config()['morgan'])
    official=exclusion_ids(ROOT/'data/raw/uncharacterized.txt')
    rows=[]
    for i,row in enumerate(ref.reset_index().itertuples(index=False)):
        mol_id=row.mol_id
        number=int(mol_id.removeprefix('gdb_'))
        r={'mol_id':mol_id,'qm9_index':number,'official_excluded':number in official,
           'previous_status':'retained' if mol_id in prev.index else exclusions.loc[mol_id,'reason'],
           'reference_valid':False,'old_smiles':prev.loc[mol_id,'smiles'] if mol_id in prev.index else None,
           'reference_smiles':None,'structure_changed':None,'fingerprint_changed':None}
        with rdBase.BlockLogs(): mol=Chem.MolFromSmiles(str(row.smiles))
        if mol is not None:
            mol=Chem.RemoveHs(mol)
            canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False)
            r['reference_valid']=True;r['reference_smiles']=canonical
            if mol_id in prev.index:
                r['structure_changed']=canonical!=r['old_smiles']
                fp=np.packbits(gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(canonical)),bitorder='little')
                r['fingerprint_changed']=not np.array_equal(fp,oldbits[oldids.get_loc(mol_id)])
        rows.append(r)
        if (i+1)%30000==0: print('Audited',i+1,flush=True)
    result=pd.DataFrame(rows)
    parsed=result.loc[~result.official_excluded & result.reference_valid].sort_values('qm9_index')
    duplicate=parsed.duplicated('reference_smiles',keep='first')
    kept=set(parsed.loc[~duplicate,'mol_id'])
    result['retained_in_new_version']=result.mol_id.isin(kept)
    was_invalid=result.previous_status=='invalid_structure'
    was_kept=result.previous_status=='retained'
    changed=result.structure_changed.eq(True)
    bit_changed=result.fingerprint_changed.eq(True)
    summary={'reference_rows':len(ref),'reference_sha256':sha256(ROOT/'data/raw/qm9_smiles.csv'),
        'reference_ids_equal_raw_ids':set(ref.index)==set(labels.index),'target_checks':target_checks,
        'previous_retained':int(was_kept.sum()),'previous_invalid':int(was_invalid.sum()),
        'previous_invalid_parseable_now':int((was_invalid&result.reference_valid).sum()),
        'previous_invalid_retained_now':int((was_invalid&result.retained_in_new_version).sum()),
        'previous_retained_same_structure':int((was_kept&result.structure_changed.eq(False)).sum()),
        'previous_retained_changed_structure':int(changed.sum()),
        'previous_retained_changed_fingerprint':int(bit_changed.sum()),
        'new_nonexcluded_parse_failures':int((~result.official_excluded&~result.reference_valid).sum()),
        'new_duplicates':int(duplicate.sum()),'new_retained':len(kept),
        'added_ids':len(kept-set(prev.index)),'removed_ids':len(set(prev.index)-kept),
        'changed_structure_ids_retained_now':int((changed&result.retained_in_new_version).sum()),
        'comparison_scope':'Canonical non-isomeric SMILES and Morgan(radius=2, bits=2048); stereochemical differences are outside this protocol'}
    (ROOT/'reports').mkdir(exist_ok=True)
    write_csv_gz(result,ROOT/'reports/reference_comparison.csv.gz')
    write_csv_gz(result.loc[changed|was_invalid],ROOT/'reports/affected_molecules.csv.gz')
    write_json(ROOT/'reports/reference_audit.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()

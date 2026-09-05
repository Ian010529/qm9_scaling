"""Prepare QM9 from reference SMILES; use the SDF mirror only to check labels."""
import argparse
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from common import ROOT, TARGETS, config, sha256, versions, write_json, write_csv_gz
from download import exclusion_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles-csv', type=Path, default=ROOT / 'data/raw/qm9_smiles.csv')
    args = parser.parse_args()
    dest = ROOT / 'data/processed'
    if (dest / 'qm9.csv.gz').exists():
        raise FileExistsError('Processed data exists; create a new data version to change it')
    supplied = pd.read_csv(args.smiles_csv).set_index('mol_id', verify_integrity=True)
    required = ['smiles', 'mu', 'homo', 'lumo', 'g298']
    if not set(required) <= set(supplied.columns):
        raise ValueError('Missing required SMILES CSV columns')
    with zipfile.ZipFile(ROOT / 'data/raw/qm9.zip') as z:
        with z.open('gdb9.sdf.csv') as f:
            labels = pd.read_csv(f).set_index('mol_id', verify_integrity=True)
    if len(supplied) != 133885 or set(supplied.index) != set(labels.index):
        raise ValueError('Expected same 133885 original IDs in both sources')
    delta = np.abs(supplied.loc[labels.index, required[1:]].to_numpy(float)-labels[required[1:]].to_numpy(float))
    if not np.isfinite(delta).all() or (delta > 1e-12).any():
        raise ValueError('Reference labels or units differ')
    excluded = exclusion_ids(ROOT / 'data/raw/uncharacterized.txt')
    rows, removals = [], []
    counts = {'raw_records': len(supplied), 'uncharacterized': 0, 'invalid_structure': 0, 'missing_target': 0}
    for row in supplied.reset_index().itertuples(index=False):
        mol_id = row.mol_id
        number = int(mol_id.removeprefix('gdb_'))
        if number in excluded:
            counts['uncharacterized'] += 1
            removals.append({'mol_id': mol_id, 'reason': 'uncharacterized'})
            continue
        try:
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(str(row.smiles))
                if mol is None:
                    raise ValueError('Reference SMILES cannot be parsed')
                mol = Chem.RemoveHs(mol)
                iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                if Chem.MolFromSmiles(smi) is None:
                    raise ValueError('Canonical SMILES roundtrip failed')
        except Exception as error:
            counts['invalid_structure'] += 1
            removals.append({'mol_id': mol_id, 'reason': 'invalid_structure', 'detail': str(error)})
            continue
        target = np.array([row.mu, row.homo, row.lumo, row.g298], dtype=float)
        if not np.isfinite(target).all():
            counts['missing_target'] += 1
            removals.append({'mol_id': mol_id, 'reason': 'missing_target'})
            continue
        rows.append({'mol_id': mol_id, 'qm9_index': number, 'smiles': smi, 'smiles_isomeric': iso,
                     **dict(zip(TARGETS, target))})
        if len(rows) % 30000 == 0:
            print(f'Parsed {len(rows):,} valid reference SMILES', flush=True)
    if counts['uncharacterized'] != 3054:
        raise ValueError('Unexpected official exclusion count')
    frame = pd.DataFrame(rows).sort_values('qm9_index').reset_index(drop=True)
    duplicate = frame.duplicated('smiles', keep='first')
    representatives = frame.drop_duplicates('smiles').set_index('smiles').mol_id
    for row in frame.loc[duplicate].itertuples():
        removals.append({'mol_id': row.mol_id, 'reason': 'duplicate_nonisomeric_smiles',
                         'retained_id': representatives.loc[row.smiles]})
    counts['duplicates_removed'] = int(duplicate.sum())
    frame = frame.loc[~duplicate].reset_index(drop=True)
    counts['retained'] = len(frame)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / 'qm9.csv.gz'
    write_csv_gz(frame, out)
    pd.DataFrame(removals).to_csv(dest / 'exclusions.csv', index=False)
    cfg = config()
    write_json(ROOT / 'reports/data_audit.json', {
        'counts': counts, 'units': cfg['units'], 'G_definition': 'g298; total molecular Gibbs free energy at 298.15 K',
        'smiles_policy': cfg['smiles_policy'], 'duplicate_policy': cfg['duplicate_policy'],
        'structure_source': 'Uploaded qm9.csv reference SMILES; IDs and four targets checked against downloaded QM9 mirror',
        'label_max_absolute_difference': float(delta.max()), 'input_information': '2D connectivity only; no SDF-derived topology or coordinates',
        'supplied_smiles_sha256': sha256(args.smiles_csv), 'dataset_sha256': sha256(out),
        'versions': versions(), 'sources': {n: sha256(ROOT / 'data/raw' / n) for n in ['qm9.zip', 'uncharacterized.txt', 'qm9_smiles.csv']}})
    print(counts)


if __name__ == '__main__':
    main()

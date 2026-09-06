"""Build research splits and extra label-free inputs without modifying v2 data."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from .design import matrix
from .io import check_file, code_manifest, digest, partition_check, read_json, sha256, versions, write_json


def divide_validation(pool, val, test, ratios):
    a, b = [int(len(val) * x / sum(ratios)) for x in ratios[:2]]
    parts = dict(pool=pool, stop=val[:a], tune=val[a:a+b], law=val[a+b:], test=test)
    if any(len(x) == 0 for x in parts.values()):
        raise ValueError('A required partition is empty')
    return parts


def random_split(ids, seed, cfg):
    order = np.random.default_rng(seed).permutation(ids)
    nt, npool = int(.1 * len(ids)), int(.8 * len(ids))
    return divide_validation(order[len(ids)-npool:], order[nt:len(ids)-npool], order[:nt], cfg['validation_counts'])


def scaffold_split(frame, cfg):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    groups = defaultdict(list)
    for row in frame.itertuples():
        key = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(row.smiles), includeChirality=False)
        groups[key or '<ACYCLIC>'].append(str(row.mol_id))
    seed = cfg['robustness']['scaffold_seed']
    keys = sorted(groups, key=lambda k: (-len(groups[k]), digest([seed, k])))
    target = np.array([.8, .1, .1]) * len(frame)
    counts = np.zeros(3)
    bins = [[], [], []]
    assignment = {}
    for key in keys:
        scores = []
        for k in range(3):
            trial = counts.copy()
            trial[k] += len(groups[key])
            scores.append(float(np.sum(((trial-target)/target)**2)))
        k = int(np.argmin(scores))
        bins[k].extend(groups[key]); counts[k] += len(groups[key]); assignment[key] = k
    rng = np.random.default_rng(seed)
    pool, val, test = [rng.permutation(np.asarray(x, dtype=str)) for x in bins]
    if min(len(pool), len(val), len(test)) < 1000:
        raise ValueError('Scaffold split has fewer than 1000 molecules in a partition; protocol amendment required, no silent fallback')
    parts = divide_validation(pool, val, test, cfg['validation_counts'])
    report = dict(groups=len(groups), acyclic_group_size=len(groups.get('<ACYCLIC>', [])),
                  largest_group=max(map(len, groups.values())), counts={k: len(v) for k,v in parts.items()},
                  group_assignment=assignment, empty_scaffold_policy='one indivisible group')
    return parts, report


def source_assets(root, cfg, frame):
    assets = {'data/processed/qm9.csv.gz': cfg['dataset_sha256']}
    for rel in ['data/processed/exclusions.csv','data/raw/qm9_smiles.csv']:
        assets[rel] = sha256(root/rel)
    base = read_json(root/'splits/manifest.json')
    if base['dataset_sha256'] != cfg['dataset_sha256']:
        raise ValueError('Wrong base split dataset')
    assets['splits/manifest.json'] = sha256(root/'splits/manifest.json')
    assets.update({f'splits/{k}': v for k,v in base['files_sha256'].items()})
    for rel in ['features/morgan.json', 'features/molformer.json', 'features/dmpnn/manifest.json']:
        meta = read_json(root/rel)
        if meta['dataset_sha256'] != cfg['dataset_sha256']:
            raise ValueError(f'Stale feature metadata: {rel}')
        assets[rel] = sha256(root/rel)
        if rel.endswith('morgan.json'):
            if meta['parameters'] != dict(radius=2, fpSize=2048, includeChirality=False):
                raise ValueError('Unexpected binary Morgan parameters')
            assets['features/morgan.npz'] = meta['features_sha256']
        elif rel.endswith('molformer.json'):
            if meta['revision'] != '7b12d946c181a37f6012b9dc3b002275de070314' or meta['dimension'] != 768:
                raise ValueError('Unexpected MoLFormer checkpoint or width')
            assets['features/molformer.npy'] = meta['features_sha256']
            assets['features/molformer_ids.npy'] = meta['id_sha256']
        else:
            assets.update({f'features/dmpnn/{k}': v for k,v in meta['files_sha256'].items()})
    for rel, value in assets.items():
        check_file(root/rel, value)
    ids = frame.mol_id.to_numpy(dtype=str)
    with np.load(root/'features/morgan.npz', allow_pickle=False) as f:
        if not np.array_equal(f['mol_id'], ids): raise ValueError('Morgan ID mismatch')
    for rel in ['features/molformer_ids.npy', 'features/dmpnn/mol_id.npy']:
        if not np.array_equal(np.load(root/rel, allow_pickle=False), ids): raise ValueError(f'ID mismatch: {rel}')
    return assets


def prepare(root, folder, cfg):
    root, folder = Path(root), Path(folder)
    if folder.exists(): raise FileExistsError('Study exists; never overwrite a prepared study')
    check_file(root/'data/processed/qm9.csv.gz', cfg['dataset_sha256'])
    frame = pd.read_csv(root/'data/processed/qm9.csv.gz')
    if len(frame) != cfg['dataset_rows'] or not frame.mol_id.is_unique or not frame.smiles.is_unique:
        raise ValueError('Unexpected data population')
    if not np.isfinite(frame[cfg['targets']].to_numpy()).all(): raise ValueError('Invalid target')
    assets = source_assets(root, cfg, frame)
    expected_rdkit = read_json(root/'features/morgan.json')['versions']['rdkit']
    if versions()['rdkit'] != expected_rdkit:
        raise RuntimeError(f'Prepare derived features with RDKit {expected_rdkit}, not {versions()["rdkit"]}')
    work = folder.with_name(folder.name + '.partial')
    work.mkdir(parents=True, exist_ok=False)
    ids = frame.mol_id.to_numpy(dtype=str)
    with np.load(root/'splits/fixed.npz', allow_pickle=False) as base:
        parts = divide_validation(base['pool'], base['val'], base['test'], cfg['validation_counts'])
    splits = {'random': parts}
    for name in cfg['robustness']['splits']:
        if name == 'random': continue
        if name.startswith('random_'):
            splits[name] = random_split(ids, int(name.split('_')[1]), cfg)
        else:
            splits[name], report = scaffold_split(frame, cfg)
            write_json(work/'scaffold_audit.json', report)
    seeds = cfg['subset_seeds'] + cfg['hpo']['subset_seeds']
    for name, parts in splits.items():
        partition_check(ids, parts)
        seqs = {f'seq_{s}': np.random.default_rng(s).permutation(parts['pool']) for s in seeds}
        if name == 'random':
            for s in cfg['subset_seeds'][:5]:
                if not np.array_equal(seqs[f'seq_{s}'], np.load(root/f'splits/sequence_{s}.npy', allow_pickle=False)):
                    raise ValueError('Existing nested sequence changed')
        np.savez_compressed(work/f'{name}.npz', **parts, **seqs)
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    composition = np.zeros((len(frame), 6), dtype=np.float64)
    composition[:, 0] = 1.0
    counts = {bits: np.lib.format.open_memmap(work/f'count_{bits}.npy', mode='w+', dtype='uint8', shape=(len(frame), bits)) for bits in [2048,8192]}
    generators = {b: rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=b, includeChirality=False) for b in counts}
    for i, smi in enumerate(frame.smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: raise ValueError('Unparseable retained molecule')
        atoms = [a.GetAtomicNum() for a in Chem.AddHs(mol).GetAtoms()]
        if not set(atoms) <= {1,6,7,8,9}: raise ValueError('Unexpected element in QM9')
        composition[i, 1:] = [atoms.count(z) for z in [1,6,7,8,9]]
        for bits in counts:
            arr = generators[bits].GetCountFingerprintAsNumPy(mol)
            if arr.max() > 255: raise OverflowError('Count fingerprint exceeds uint8')
            counts[bits][i] = arr
    for arr in counts.values(): arr.flush()
    del counts
    np.save(work/'composition.npy', composition, allow_pickle=False)
    np.save(work/'mol_id.npy', ids, allow_pickle=False)
    jobs = matrix(cfg, {k: len(v['pool']) for k,v in splits.items()})
    write_json(work/'protocol.json', cfg); write_json(work/'matrix.json', jobs)
    local = {p.name: sha256(p) for p in work.iterdir() if p.is_file()}
    write_json(work/'manifest.json', dict(assets=assets, files=local, code=code_manifest(), versions=versions(),
               split_counts={k: {s:len(v) for s,v in p.items()} for k,p in splits.items()}))
    work.rename(folder)


class Workspace:
    def __init__(self, root, folder):
        self.root, self.folder = Path(root), Path(folder)
        self.manifest = read_json(self.folder/'manifest.json')
        if self.manifest['code'] != code_manifest(): raise ValueError('Research code/config changed; start a versioned amendment')
        for rel, value in self.manifest['files'].items(): check_file(self.folder/rel, value)
        for rel, value in self.manifest['assets'].items(): check_file(self.root/rel, value)
        environment = self.folder/'environment.json'
        if environment.exists():
            expected = read_json(environment)
            actual = {k:(v.split('+')[0] if v is not None else None) for k,v in versions().items()}
            if actual != expected: raise ValueError('Runtime versions differ from frozen training environment')
        self.cfg = read_json(self.folder/'protocol.json')
        self.jobs = read_json(self.folder/'matrix.json')
        self.frame = pd.read_csv(self.root/'data/processed/qm9.csv.gz')
        self.index = pd.Index(self.frame.mol_id)
        self.composition = np.load(self.folder/'composition.npy', mmap_mode='r')
        self._features = {}
        self._splits = {}

    def rows(self, item):
        name = item['split']
        if name not in self._splits:
            with np.load(self.folder/f'{name}.npz', allow_pickle=False) as f:
                self._splits[name] = {k:f[k] for k in f.files}
        parts = self._splits[name]
        seq = parts[f'seq_{item["subset_seed"]}']
        if item['n'] > len(seq) or item['n'] < 1: raise ValueError('Invalid training size')
        if len(np.unique(seq)) != len(seq) or set(seq) != set(parts['pool']): raise ValueError('Invalid nested sequence')
        partition_check(self.index.to_numpy(), {k:v for k,v in parts.items() if not k.startswith('seq_')})
        ids = {k:v for k,v in parts.items() if k in ['stop','tune','law','test']}
        ids['train'] = seq[:item['n']]
        if item.get('limited_validation'):
            c = self.cfg['validation_budget']; nval = min(c['maximum'], max(1, int(np.ceil(item['n']*c['fraction']))))
            ids['stop'] = ids['stop'][:nval]
        rows = {k:self.index.get_indexer(v) for k,v in ids.items()}
        if any((x < 0).any() for x in rows.values()): raise ValueError('Unknown molecule ID')
        return rows

    def features(self, route):
        if route in self._features: return self._features[route]
        if route in ['morgan_xgb', 'morgan_mlp']:
            with np.load(self.root/'features/morgan.npz', allow_pickle=False) as f:
                arr = np.unpackbits(f['packed'], axis=1, bitorder='little', count=2048)
        elif route in ['count_xgb', 'wide_count_xgb']:
            arr = np.load(self.folder/f'count_{8192 if route == "wide_count_xgb" else 2048}.npy', mmap_mode='r')
        elif route.startswith('molformer'):
            arr = np.load(self.root/'features/molformer.npy', mmap_mode='r')
        elif route == 'dmpnn':
            arr = Graphs(self.root/'features/dmpnn')
        else: raise ValueError(f'Unknown route: {route}')
        self._features[route] = arr
        return arr


class Graphs:
    def __init__(self, folder):
        for n in ['V', 'E', 'edge_index', 'rev_edge_index', 'atom_ptr', 'edge_ptr']:
            setattr(self, n, np.load(Path(folder)/f'{n}.npy', mmap_mode='r'))
    def __getitem__(self, i):
        from chemprop.data.molgraph import MolGraph
        a,b = self.atom_ptr[i:i+2]; e,f = self.edge_ptr[i:i+2]
        return MolGraph(self.V[a:b], self.E[e:f], self.edge_index[:,e:f], self.rev_edge_index[e:f])

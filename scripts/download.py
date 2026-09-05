"""Download public QM9 mirror and original exclusion list; validate before rename."""
import subprocess
import zipfile
import pandas as pd

from common import ROOT, sha256, write_json

SOURCES = {
    "qm9.zip": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip",
    "uncharacterized.txt": "https://ndownloader.figshare.com/files/3195404",
    "qm9_smiles.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
}


def exclusion_ids(path):
    # Data rows begin with a positive original QM9 integer ID; header does not.
    ids = [int(p[0]) for line in path.read_text().splitlines()
           if (p := line.split()) and p[0].isdigit() and 1 <= int(p[0]) <= 133885]
    if len(ids) != 3054 or len(set(ids)) != 3054:
        raise ValueError(f"Expected 3054 unique excluded IDs, got {len(ids)}")
    return set(ids)


def validate(name, path):
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            if not {"gdb9.sdf", "gdb9.sdf.csv", "QM9_README"}.issubset(z.namelist()):
                raise ValueError("Unexpected QM9 archive contents")
            if z.testzip() is not None:
                raise ValueError("Archive CRC check failed")
    elif name.endswith('.csv'):
        frame = pd.read_csv(path)
        if len(frame) != 133885 or not frame.mol_id.is_unique or not {'mol_id', 'smiles', 'mu', 'homo', 'lumo', 'g298'} <= set(frame.columns):
            raise ValueError('Unexpected QM9 SMILES CSV contents')
    else:
        exclusion_ids(path)


def main():
    folder = ROOT / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in SOURCES.items():
        dest = folder / name
        try:
            validate(name, dest)
        except (OSError, ValueError, zipfile.BadZipFile):
            temp = folder / (name + ".part")
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--connect-timeout", "15", "--max-time", "180",
                            "--output", str(temp), url], check=True, timeout=190)
            validate(name, temp)
            temp.replace(dest)
        manifest[name] = {"url": url, "bytes": dest.stat().st_size, "sha256": sha256(dest)}
        print(f"Verified {name}: {dest.stat().st_size:,} bytes", flush=True)
    write_json(folder / "manifest.json", manifest)


if __name__ == "__main__":
    main()

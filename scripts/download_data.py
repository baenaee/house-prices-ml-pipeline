"""Download the Kaggle House Prices competition data into the repository root.

The data are not in this repository: the competition rules (section 7.B) forbid redistributing
them. Anyone who has a Kaggle account and has accepted the competition rules can download them:

    1. Accept the rules once: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules
    2. Create an API token (kaggle.com -> Settings -> API -> Create New Token) and either save
       kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY in the environment.
    3. pip install kaggle
    4. python scripts/download_data.py

The script checks each file's SHA-256 against the copies the experiments were run on, so a
reproduction uses byte-identical inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "house-prices-advanced-regression-techniques"
ROOT = Path(__file__).resolve().parents[1]

# SHA-256 of the files used for every registered experiment (see docs/AUDIT_2026-11-11.md).
EXPECTED = {
    "train.csv": "1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41",
    "test.csv": "8fdd3d829d4d986b58f845c9553b225e67dd8383624d90fb6ca1d4bed5798c1e",
    "sample_submission.csv": "f19e99699b925aa3085c8d2ae43c62f15fbcd35e4a953b98025f8392c3ffa807",
    "data_description.txt": "356a4670aa688afa8ba44b1365b633fbbac2dae535fe4638532630607d64070d",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(dest: Path) -> list[str]:
    """Return the names of files that are missing or differ from the expected copies."""
    return [n for n, h in EXPECTED.items() if not (dest / n).exists() or sha256(dest / n) != h]


def kaggle_cmd() -> list[str]:
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    sys.exit("The Kaggle CLI was not found. Install it with:  pip install kaggle")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=ROOT, help="folder to put the files in (default: repo root)")
    ap.add_argument("--force", action="store_true", help="download even if the files are already present")
    args = ap.parse_args()
    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if not args.force and not check(dest):
        print(f"All four data files already present and verified in {dest}")
        return 0

    zip_path = dest / f"{COMPETITION}.zip"
    print(f"Downloading {COMPETITION} with the Kaggle CLI ...")
    result = subprocess.run(kaggle_cmd() + ["competitions", "download", "-c", COMPETITION, "-p", str(dest), "--force"])
    if result.returncode != 0 or not zip_path.exists():
        print("Download failed. Check that you accepted the competition rules on kaggle.com and that your "
              "API token is set (~/.kaggle/kaggle.json, or KAGGLE_USERNAME and KAGGLE_KEY).", file=sys.stderr)
        return 1
    with zipfile.ZipFile(zip_path) as zf:
        for name in EXPECTED:
            zf.extract(name, dest)
    zip_path.unlink()

    bad = check(dest)
    for name in EXPECTED:
        status = "MISMATCH" if name in bad else "ok"
        print(f"  {name:24s} {status}")
    if bad:
        print("Warning: some files differ from the versions the experiments used; registered scores may not "
              "reproduce exactly.", file=sys.stderr)
        return 2
    print("Data ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

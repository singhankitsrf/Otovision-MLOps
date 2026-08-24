from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(root: Path) -> pd.DataFrame:
    records = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            record = {"path": str(path), "label": class_dir.name}
            try:
                with Image.open(path) as img:
                    record.update(width=img.width, height=img.height, mode=img.mode, readable=True)
            except Exception as exc:
                record.update(width=None, height=None, mode=None, readable=False, read_error=repr(exc))
            record["sha256"] = sha256_file(path)
            records.append(record)
    return pd.DataFrame(records)


def assign_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    duplicate_sizes = df.groupby("sha256")["sha256"].transform("size")
    df["duplicate_group_size"] = duplicate_sizes
    df["is_exact_duplicate"] = duplicate_sizes > 1
    df["is_canonical"] = ~df.duplicated("sha256", keep="first")
    df["split"] = "duplicate_excluded"

    canonical = df[df["is_canonical"] & df["readable"]].copy()
    train, temp = train_test_split(
        canonical,
        test_size=0.30,
        random_state=seed,
        stratify=canonical["label"],
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=seed,
        stratify=temp["label"],
    )
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    df.loc[~df["readable"], "split"] = "unreadable"
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/dataset_manifest.csv"))
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    df = scan_dataset(args.data_root)
    if df.empty:
        raise SystemExit(f"No supported image files found below {args.data_root}")
    df = assign_splits(df, args.split_seed)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.manifest, index=False)

    canonical = df[df["is_canonical"] & df["readable"]]
    summary = canonical.groupby(["label", "split"]).size().unstack(fill_value=0)
    print(summary)
    print(f"\nTotal source files: {len(df)}")
    print(f"Exact duplicate files beyond canonical copies: {(~df['is_canonical']).sum()}")
    print(f"Modeling files: {len(canonical)}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()

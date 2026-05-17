"""
Create one managed per-residue embedding directory for host benchmarks.

Existing strict_vh_v1 embeddings are linked into the unified directory. Sequence
CSVs are then checked by hp_<md5-prefix> id; unresolved rows are written to
sequences.missing.csv so only those need fresh ESMC extraction.

Run:
  python p2psiglip_db/data/prepare_unified_host_embeddings.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "runs" / "strict_vh_v1" / "embeddings_perres"
DEFAULT_OUT = ROOT / "runs" / "strict_vh_v1" / "embeddings_unified"
DEFAULT_SEQUENCES = [
    ROOT / "data" / "datasets" / "strict_vh_v1" / "sequences_for_embed.csv",
    ROOT / "data" / "datasets" / "bench_host_corpus" / "sequences_hp.csv",
    ROOT / "data" / "datasets" / "bench_rf2_ppi" / "sequences_hp.csv",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_sequence_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id") or cols.get("protein_id") or df.columns[0]
    seq_col = cols.get("sequence") or cols.get("seq") or df.columns[1]
    out = df[[id_col, seq_col]].copy()
    out.columns = ["id", "sequence"]
    out["id"] = out["id"].astype(str)
    out["sequence"] = out["sequence"].astype(str).str.strip().str.upper()
    out["source_csv"] = rel(path)
    return out


def link_file(source: Path, target: Path) -> bool:
    if target.exists() or target.is_symlink():
        return False
    target.symlink_to(source.resolve())
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified host embedding directory.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sequences", type=Path, nargs="*", default=DEFAULT_SEQUENCES)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    linked_base = 0
    for path in sorted(args.base_dir.glob("*.npy")):
        if link_file(path, args.out / path.name):
            linked_base += 1

    frames = [load_sequence_csv(path) for path in args.sequences if path.exists()]
    required = pd.concat(frames, ignore_index=True).drop_duplicates(["id", "sequence"])
    required = required.sort_values(["id", "source_csv"])

    present = []
    missing = []
    for _, row in required.iterrows():
        target = args.out / f"{row['id']}.npy"
        if target.exists() or target.is_symlink():
            present.append(row)
        else:
            missing.append(row)

    missing_df = pd.DataFrame(missing, columns=required.columns) if missing else required.iloc[0:0]
    present_df = pd.DataFrame(present, columns=required.columns) if present else required.iloc[0:0]
    missing_path = args.out / "sequences.missing.csv"
    present_path = args.out / "sequences.present.csv"
    missing_df.to_csv(missing_path, index=False)
    present_df.to_csv(present_path, index=False)

    print(f"base embeddings linked : {linked_base}")
    print(f"required sequences     : {len(required)}")
    print(f"present embeddings     : {len(present_df)}")
    print(f"missing embeddings     : {len(missing_df)} -> {missing_path}")


if __name__ == "__main__":
    main()

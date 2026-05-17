"""
Reuse pre-computed ESMC embeddings: scan a sequences CSV and, for each entry,
try (in order):
  1. Direct ID match: <src>/<id>.npy exists → symlink as <dst>/<id>.npy
  2. (Optional) Sequence MD5 match via --md5-map:
       compute md5(sequence); look up fpid in the map; if <src>/<fpid>.npy
       exists → symlink it as <dst>/<id>.npy
Whatever can't be resolved goes into <dst>/sequences.missing.csv so the caller
can run get_embeddings.py only on the leftovers.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd


def md5_seq(seq: str) -> str:
    return hashlib.md5(seq.strip().upper().encode("utf-8")).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", required=True,
                   help="CSV with id,sequence columns")
    p.add_argument("--src", required=True,
                   help="Existing embeddings dir to source <id>.npy from")
    p.add_argument("--dst", required=True,
                   help="Destination embeddings dir (will be created)")
    p.add_argument("--md5-map", default=None,
                   help="Optional CSV with protein_md5,fpid columns (e.g. P2PSigLip_proteins_total.csv); "
                        "enables sequence-based fallback when direct ID match fails")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        sys.exit(f"--src {src} is not a directory")
    dst.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.sequences)
    cols = {c.lower(): c for c in df.columns}
    id_col = cols.get("id") or cols.get("fpid_1") or df.columns[0]
    seq_col = cols.get("sequence") or cols.get("seq") or df.columns[1]

    # Optional MD5 lookup
    md5_to_fpid = {}
    if args.md5_map:
        map_df = pd.read_csv(args.md5_map, usecols=["protein_md5", "fpid"])
        md5_to_fpid = dict(zip(map_df["protein_md5"].astype(str),
                               map_df["fpid"].astype(str)))
        print(f"  loaded {len(md5_to_fpid):,} md5→fpid mappings", file=sys.stderr)

    n_total = len(df)
    already_present = 0
    linked_direct = 0
    linked_md5 = 0
    missing_rows = []

    for _, row in df.iterrows():
        pid = str(row[id_col])
        target = dst / f"{pid}.npy"
        if target.exists() or target.is_symlink():
            already_present += 1
            continue

        # Try direct ID match
        source = src / f"{pid}.npy"
        if source.exists():
            target.symlink_to(source.resolve())
            linked_direct += 1
            continue

        # Try MD5 match
        if md5_to_fpid:
            seq = str(row[seq_col])
            fp = md5_to_fpid.get(md5_seq(seq))
            if fp:
                source = src / f"{fp}.npy"
                if source.exists():
                    target.symlink_to(source.resolve())
                    linked_md5 += 1
                    continue

        missing_rows.append(row)

    missing_df = pd.DataFrame(missing_rows, columns=df.columns) if missing_rows else df.iloc[0:0]
    missing_path = dst / "sequences.missing.csv"
    missing_df.to_csv(missing_path, index=False)

    print(f"  sequences total       : {n_total}")
    print(f"  already in dst        : {already_present}")
    print(f"  linked by direct ID   : {linked_direct}")
    print(f"  linked by sequence md5: {linked_md5}")
    print(f"  missing (need embed)  : {len(missing_df)}  → {missing_path}")


if __name__ == "__main__":
    main()

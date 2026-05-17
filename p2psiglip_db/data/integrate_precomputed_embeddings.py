"""
Remap precomputed embeddings (embed/<plm>/{embeddings.npy, protein_ids.txt})
into the strict_c3_v1 fpid space, writing one .npy per fpid.

Each fpid in proteins.csv carries a `;`-joined list of upstream `original_ids`.
For each PLM we:
  1. Load (embeddings.npy, protein_ids.txt) and build a dict orig_id -> row_idx.
  2. For each fpid, look up the FIRST original_id present in that dict and copy
     embeddings[row_idx] to <output_dir>/<fpid>.npy.
  3. Emit a `_missing_<plm>.csv` (id,sequence) with all uncovered fpids so they
     can be patch-extracted with `get_embeddings.py --plm <name>`.

Usage:
  python p2psiglip_db/data/integrate_precomputed_embeddings.py \\
      --proteins data/datasets/strict_c3_v1/proteins.csv \\
      --sequences data/datasets/strict_c3_v1/sequences.csv \\
      --embed-dir embed/esm2_650m \\
      --out-dir runs/strict_c3_v1/embeddings_esm2 \\
      --missing-csv runs/strict_c3_v1/_missing_esm2.csv

Run once per PLM (esmc_300m / esm2_650m / prott5 / saprot_650m_af2).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--proteins", required=True,
                   help="strict_c3_v1 proteins.csv (must have fpid + original_ids columns)")
    p.add_argument("--sequences", required=True,
                   help="strict_c3_v1 sequences.csv (id,sequence) for missing-list lookup")
    p.add_argument("--embed-dir", required=True,
                   help="dir containing embeddings.npy + protein_ids.txt + metadata.json")
    p.add_argument("--out-dir", required=True,
                   help="output dir for per-fpid .npy files")
    p.add_argument("--missing-csv", required=True,
                   help="path to write the missing-fpid sequences CSV")
    return p.parse_args()


def main():
    args = parse_args()
    embed_dir = Path(args.embed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load precomputed matrix + ID order
    emb = np.load(embed_dir / "embeddings.npy", mmap_mode="r")
    ids = (embed_dir / "protein_ids.txt").read_text().strip().split("\n")
    assert len(ids) == emb.shape[0], f"ID/row mismatch: {len(ids)} vs {emb.shape[0]}"
    orig_to_row = {pid: i for i, pid in enumerate(ids)}
    plm_dim = emb.shape[1]
    print(f"[integrate] loaded {emb.shape} from {embed_dir}")
    if (embed_dir / "metadata.json").exists():
        meta = json.loads((embed_dir / "metadata.json").read_text())
        print(f"[integrate] model: {meta.get('model_name')}, dim: {meta.get('embedding_dim')}, "
              f"input_kind: {meta.get('input_kind')}")

    # 2. Load fpid -> original_ids
    df = pd.read_csv(args.proteins, usecols=["fpid", "original_ids"])
    df["fpid"] = df["fpid"].astype(str)

    def split_origs(s):
        if pd.isna(s):
            return []
        return [x.strip() for x in str(s).split(";") if x.strip()]

    n_covered = 0
    missing_fpids = []
    for fpid, origs_str in tqdm(zip(df["fpid"], df["original_ids"]),
                                total=len(df), desc="[integrate] mapping fpids"):
        origs = split_origs(origs_str)
        chosen_row = None
        for o in origs:
            if o in orig_to_row:
                chosen_row = orig_to_row[o]
                break
        if chosen_row is None:
            missing_fpids.append(fpid)
            continue
        # fpid filename is sanitized the same way create_ppi_h5_esm.py expects
        safe_id = str(fpid).replace("/", "_").replace("\\", "_")
        np.save(out_dir / f"{safe_id}.npy",
                emb[chosen_row].astype(np.float32))
        n_covered += 1

    print(f"[integrate] covered {n_covered:,} / {len(df):,} fpids "
          f"({100 * n_covered / len(df):.1f}%); missing {len(missing_fpids):,}")

    # 3. Write missing.csv (id,sequence) for patch extraction.
    # Exclude any fpid that already has a .npy on disk (e.g. from a prior
    # partial extraction run or a previous integrate pass on the same dir).
    seqs = pd.read_csv(args.sequences)
    seqs["id"] = seqs["id"].astype(str)
    on_disk = {p.stem for p in out_dir.glob("*.npy")}
    truly_missing = [f for f in missing_fpids if f not in on_disk]
    missing_set = set(truly_missing)
    missing_df = seqs[seqs["id"].isin(missing_set)].copy()
    missing_df.to_csv(args.missing_csv, index=False)
    print(f"[integrate] {len(on_disk):,} npy files now on disk in {out_dir}")
    print(f"[integrate] wrote {len(missing_df):,} truly-missing seqs to {args.missing_csv} "
          f"(integrate-uncovered={len(missing_fpids):,}, already-on-disk={len(missing_fpids)-len(truly_missing):,})")
    if len(missing_df) != len(truly_missing):
        print(f"[integrate] WARNING: {len(truly_missing) - len(missing_df)} truly-missing fpids "
              f"not present in sequences.csv (cannot patch-extract)")


if __name__ == "__main__":
    main()

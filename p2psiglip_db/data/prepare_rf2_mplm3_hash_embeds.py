"""Prepare hash-named embedding cache inputs for the RF2 MPLM3 run.

Embedding files are keyed by MD5(normalized amino-acid sequence), not by source
protein IDs. This makes the embedding cache sequence-unique across strict_c3
FP IDs, RF2 hp IDs, and any future aliases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=REPO / "data/datasets/rf2_holdout_c3_v1_trainplusval/train_plus_val_rf2_homology_endpoint_clean_id40_cov80.csv",
    )
    parser.add_argument(
        "--rf2-pairs-csv",
        type=Path,
        default=REPO / "data/datasets/bench_rf2_ppi/pairs_1to10_hp.csv",
    )
    parser.add_argument(
        "--train-sequences-csv",
        type=Path,
        default=REPO / "data/datasets/rf2_holdout_c3_v1_trainplusval/sequences_train_plus_val.csv",
    )
    parser.add_argument(
        "--rf2-sequences-csv",
        type=Path,
        default=REPO / "data/datasets/bench_rf2_ppi/sequences_hp.csv",
    )
    parser.add_argument("--embed-root", type=Path, default=REPO / "data/embeds")
    parser.add_argument(
        "--legacy-train-esmc",
        type=Path,
        default=REPO / "data/datasets/strict_c3_v1/embed_perres",
    )
    parser.add_argument(
        "--legacy-train-prot5-3di",
        type=Path,
        default=REPO / "data/datasets/strict_c3_v1/embed_perres_prostt5_3di",
    )
    parser.add_argument(
        "--legacy-train-esm2",
        type=Path,
        default=REPO / "runs/rf2ppi_holdout_mplm3/embeddings/train_esm2_perres",
    )
    parser.add_argument(
        "--legacy-rf2-esmc",
        type=Path,
        default=REPO / "runs/strict_vh_v1/rf2ppi_embeddings/esmc",
    )
    parser.add_argument(
        "--legacy-rf2-esm2",
        type=Path,
        default=REPO / "runs/strict_vh_v1/rf2ppi_embeddings/esm2",
    )
    parser.add_argument(
        "--legacy-rf2-prot5-3di",
        type=Path,
        default=REPO / "runs/rf2ppi_holdout_mplm3/embeddings/rf2_prostt5_3di",
    )
    return parser.parse_args()


def sequence_md5(sequence: str) -> str:
    normalized = "".join(str(sequence).upper().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def load_sequences(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or "sequence" not in df.columns:
        raise SystemExit(f"{path}: expected id and sequence columns")
    out = df[["id", "sequence"]].copy()
    out["id"] = out["id"].astype(str)
    out["sequence"] = out["sequence"].astype(str).str.upper().str.strip()
    out["sequence_md5"] = out["sequence"].map(sequence_md5)
    out["source"] = source
    return out


def hardlink_or_copy(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return True
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    return True


def link_legacy_embeddings(
    id_to_hash: dict[str, str],
    source_dir: Path,
    dest_dir: Path,
) -> tuple[int, int]:
    linked = 0
    missing = 0
    for source_id, md5 in id_to_hash.items():
        src = source_dir / f"{source_id}.npy"
        dest = dest_dir / f"{md5}.npy"
        if hardlink_or_copy(src, dest):
            linked += 1
        else:
            missing += 1
    return linked, missing


def write_hashed_pairs(
    pairs_csv: Path,
    out_csv: Path,
    id_to_hash: dict[str, str],
    id_cols: tuple[str, str],
    out_cols: tuple[str, str],
) -> int:
    df = pd.read_csv(pairs_csv)
    left, right = id_cols
    out_left, out_right = out_cols
    missing = sorted(
        (set(df[left].astype(str)) | set(df[right].astype(str))) - set(id_to_hash)
    )
    if missing:
        raise SystemExit(f"{pairs_csv}: {len(missing)} IDs missing hash mapping, first={missing[:5]}")
    out = df.copy()
    out[out_left] = out[left].astype(str).map(id_to_hash)
    out[out_right] = out[right].astype(str).map(id_to_hash)
    if out_left != left:
        out = out.drop(columns=[left])
    if out_right != right:
        out = out.drop(columns=[right])
    front = [out_left, out_right, "label"]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return len(out)


def write_missing_sequences(
    unique_sequences: pd.DataFrame,
    embed_dir: Path,
    out_csv: Path,
) -> int:
    missing = unique_sequences[
        ~unique_sequences["id"].map(lambda md5: (embed_dir / f"{md5}.npy").exists())
    ].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(out_csv, index=False)
    return len(missing)


def main() -> None:
    args = parse_args()
    embed_root = args.embed_root
    esmc_dir = embed_root / "esmc"
    esm2_dir = embed_root / "esm2"
    prot5_dir = embed_root / "prot5_3di"
    manifest_dir = embed_root / "manifests" / "rf2_holdout_mplm3"
    for path in [esmc_dir, esm2_dir, prot5_dir, manifest_dir]:
        path.mkdir(parents=True, exist_ok=True)

    train_seq = load_sequences(args.train_sequences_csv, "train_plus_val")
    rf2_seq = load_sequences(args.rf2_sequences_csv, "rf2")
    id_map = pd.concat([train_seq, rf2_seq], ignore_index=True)
    id_map = id_map.drop_duplicates(subset=["id"], keep="first")
    id_to_hash = dict(zip(id_map["id"], id_map["sequence_md5"]))

    unique_sequences = (
        id_map[["sequence_md5", "sequence"]]
        .drop_duplicates(subset=["sequence_md5"])
        .rename(columns={"sequence_md5": "id"})
        .sort_values("id")
        .reset_index(drop=True)
    )

    id_map_out = manifest_dir / "id_to_sequence_md5.csv"
    unique_out = manifest_dir / "sequences_by_md5.csv"
    train_hash_csv = manifest_dir / "train_plus_val_hash.csv"
    rf2_hash_csv = manifest_dir / "rf2_pairs_1to10_hp_hash.csv"
    id_map.to_csv(id_map_out, index=False)
    unique_sequences.to_csv(unique_out, index=False)

    train_rows = write_hashed_pairs(
        args.train_csv,
        train_hash_csv,
        id_to_hash,
        ("fpid_1", "fpid_2"),
        ("fpid_1", "fpid_2"),
    )
    rf2_rows = write_hashed_pairs(
        args.rf2_pairs_csv,
        rf2_hash_csv,
        id_to_hash,
        ("ID_1", "ID_2"),
        ("ID_1", "ID_2"),
    )

    train_ids = set(train_seq["id"])
    rf2_ids = set(rf2_seq["id"])
    train_id_to_hash = {pid: id_to_hash[pid] for pid in train_ids}
    rf2_id_to_hash = {pid: id_to_hash[pid] for pid in rf2_ids}

    link_stats = {
        "esmc_train": link_legacy_embeddings(train_id_to_hash, args.legacy_train_esmc, esmc_dir),
        "esmc_rf2": link_legacy_embeddings(rf2_id_to_hash, args.legacy_rf2_esmc, esmc_dir),
        "esm2_train_partial": link_legacy_embeddings(train_id_to_hash, args.legacy_train_esm2, esm2_dir),
        "esm2_rf2": link_legacy_embeddings(rf2_id_to_hash, args.legacy_rf2_esm2, esm2_dir),
        "prot5_3di_train": link_legacy_embeddings(train_id_to_hash, args.legacy_train_prot5_3di, prot5_dir),
        "prot5_3di_rf2_partial": link_legacy_embeddings(rf2_id_to_hash, args.legacy_rf2_prot5_3di, prot5_dir),
    }

    missing = {
        "esmc": write_missing_sequences(unique_sequences, esmc_dir, manifest_dir / "missing_esmc.csv"),
        "esm2": write_missing_sequences(unique_sequences, esm2_dir, manifest_dir / "missing_esm2.csv"),
        "prot5_3di": write_missing_sequences(unique_sequences, prot5_dir, manifest_dir / "missing_prot5_3di.csv"),
    }

    summary = {
        "method": "sequence-unique embedding cache keyed by MD5(normalized amino-acid sequence)",
        "embed_root": str(embed_root),
        "embed_dirs": {
            "esmc": str(esmc_dir),
            "esm2": str(esm2_dir),
            "prot5_3di": str(prot5_dir),
        },
        "id_map_csv": str(id_map_out),
        "unique_sequences_csv": str(unique_out),
        "train_hash_csv": str(train_hash_csv),
        "rf2_hash_csv": str(rf2_hash_csv),
        "input_ids": int(len(id_map)),
        "unique_sequences": int(len(unique_sequences)),
        "train_pairs": int(train_rows),
        "rf2_pairs": int(rf2_rows),
        "link_stats": {
            key: {"linked_or_existing": value[0], "missing_source_id_files": value[1]}
            for key, value in link_stats.items()
        },
        "missing_after_link": missing,
    }
    (manifest_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

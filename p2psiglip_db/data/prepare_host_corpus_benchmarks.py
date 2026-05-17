"""
Convert normalized host_corpus records into hp-id benchmark CSVs.

The V3 host model stores per-residue embeddings under ids derived from sequence
MD5: hp_<first16>. This script rewrites host_corpus pair endpoints to that
canonical id so benchmark files can share one embedding directory.

Outputs under data/datasets/bench_host_corpus:
  - sequences_hp.csv
  - virahinter_highconfidence_train_hp.csv
  - virahinter_highconfidence_val_hp.csv
  - virahinter_highconfidence_test_hp.csv
  - virahinter_highconfidence_val_lt_3000_hp.csv
  - hvidb_all_positives_hp.csv
  - SUMMARY.json

Run:
  python p2psiglip_db/data/prepare_host_corpus_benchmarks.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = ROOT / "data" / "external" / "host_corpus" / "pairs_with_sequences.parquet"
DEFAULT_OUT = ROOT / "data" / "datasets" / "bench_host_corpus"


def md5_seq(seq: str) -> str:
    return hashlib.md5(seq.strip().upper().encode("utf-8")).hexdigest()


def hp_id(seq: str) -> str:
    return f"hp_{md5_seq(seq)[:16]}"


def make_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ID_1": df["host_sequence"].map(hp_id),
            "ID_2": df["pathogen_sequence"].map(hp_id),
            "label": df["label"].astype(int),
            "dataset": df["dataset"],
            "split": df["split"],
            "source_database": df["source_database"],
            "host_id": df["host_id"],
            "pathogen_id": df["pathogen_id"],
            "original_pair_id": df["original_pair_id"],
        }
    )
    return out[out["ID_1"] != out["ID_2"]].drop_duplicates(["ID_1", "ID_2", "label"]).copy()


def make_sequence_table(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for role, id_col, seq_col in [
        ("host", "host_id", "host_sequence"),
        ("pathogen", "pathogen_id", "pathogen_sequence"),
    ]:
        part = df[[id_col, seq_col, "dataset"]].copy()
        part.columns = ["original_id", "sequence", "dataset"]
        part["role"] = role
        parts.append(part)

    seqs = pd.concat(parts, ignore_index=True)
    seqs["sequence"] = seqs["sequence"].astype(str).str.strip().str.upper()
    seqs["id"] = seqs["sequence"].map(hp_id)
    seqs["sequence_md5"] = seqs["sequence"].map(md5_seq)
    grouped = (
        seqs.groupby(["id", "sequence", "sequence_md5"], dropna=False)
        .agg(
            roles=("role", lambda values: ";".join(sorted(set(map(str, values))))),
            original_ids=("original_id", lambda values: ";".join(sorted(set(map(str, values))))),
            datasets=("dataset", lambda values: ";".join(sorted(set(map(str, values))))),
        )
        .reset_index()
    )
    return grouped[["id", "sequence", "sequence_md5", "roles", "original_ids", "datasets"]]


def write_subset(name: str, df: pd.DataFrame, out_dir: Path, summary: dict) -> None:
    table = make_pair_table(df)
    path = out_dir / f"{name}.csv"
    table.to_csv(path, index=False)
    summary[name] = {
        "path": str(path.relative_to(ROOT)),
        "rows": int(len(table)),
        "positives": int((table["label"] == 1).sum()),
        "negatives": int((table["label"] == 0).sum()),
        "unique_proteins": int(len(set(table["ID_1"]) | set(table["ID_2"]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare hp-id host-corpus benchmarks.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.pairs)
    sequences = make_sequence_table(df)
    sequences.to_csv(args.out / "sequences_hp.csv", index=False)

    summary = {
        "source": str(args.pairs.relative_to(ROOT) if args.pairs.is_relative_to(ROOT) else args.pairs),
        "sequences_hp": int(len(sequences)),
    }
    write_subset(
        "virahinter_highconfidence_train_hp",
        df[(df["dataset"] == "virahinter") & (df["split"] == "highconfidence_train")],
        args.out,
        summary,
    )
    write_subset(
        "virahinter_highconfidence_val_hp",
        df[(df["dataset"] == "virahinter") & (df["split"] == "highconfidence_val")],
        args.out,
        summary,
    )
    write_subset(
        "virahinter_highconfidence_test_hp",
        df[(df["dataset"] == "virahinter") & (df["split"] == "highconfidence_test")],
        args.out,
        summary,
    )
    write_subset(
        "virahinter_highconfidence_val_lt_3000_hp",
        df[(df["dataset"] == "virahinter") & (df["split"] == "highconfidence_val_lt_3000")],
        args.out,
        summary,
    )
    write_subset(
        "hvidb_all_positives_hp",
        df[(df["dataset"] == "hvidb") & (df["split"] == "all")],
        args.out,
        summary,
    )
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Build a hash-ID C3 split from the merged PPI tables and a fixed test CSV.

The fixed test set is treated as the model-selection set. Candidate train pairs
come from ``data/merged/pairs.csv`` after mapping FP IDs to sequence MD5 IDs.
By default, only label=1 rows are kept because SigLIP-style training creates
negatives from in-batch off-diagonal pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from p2psiglip_db.data.split_utils import (
    normalize_sequence,
    pair_columns,
    pair_key,
    pair_summary,
    parse_cluster_tsv,
    parse_csv_values,
    resolve_mmseqs,
    run_mmseqs_easy_cluster,
    write_fasta,
    write_hash_dataset_readme,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-proteins", type=Path, default=REPO / "data/merged/proteins.csv")
    parser.add_argument("--merged-pairs", type=Path, default=REPO / "data/merged/pairs.csv")
    parser.add_argument(
        "--tier-include",
        default=None,
        help=(
            "Comma-separated PPI_Tier values to keep before C3 filtering. "
            "Use with data/merged/interactions.csv, e.g. diamond,gold."
        ),
    )
    parser.add_argument("--test-csv", type=Path, default=REPO / "data/datasets/p2psiglip_hash_v1/test.csv")
    parser.add_argument(
        "--extra-sequences-csv",
        type=Path,
        default=REPO / "data/datasets/p2psiglip_hash_v1/sequences.csv",
        help="Sequence table used only to recover fixed-test hashes absent from data/merged.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument(
        "--all-labels",
        action="store_true",
        help="Keep label=0 rows too. Do not use with SigLIP train_loss unless you really intend it.",
    )
    parser.add_argument("--identity", type=float, default=0.4)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 8))
    parser.add_argument("--mmseqs", default=None)
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--reuse-cluster", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write hash-ID intermediate files under OUT/_work and stop before clustering/splitting.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Collection name written into SUMMARY.json. Defaults to OUT directory name.",
    )
    parser.add_argument(
        "--write-val-alias",
        action="store_true",
        help="Also write val.csv as an alias of the fixed test CSV.",
    )
    parser.add_argument(
        "--no-val-alias",
        action="store_true",
        help="Deprecated compatibility flag; val.csv is not written unless --write-val-alias is set.",
    )
    return parser.parse_args()


def read_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    left, right = pair_columns(list(df.columns))
    out = df.copy()
    out["ID_1"] = df[left].astype(str)
    out["ID_2"] = df[right].astype(str)
    out["label"] = df["label"].astype(int) if "label" in df.columns else 1
    front = ["ID_1", "ID_2", "label"]
    drop_source_pair_cols = {left, right} - set(front)
    rest = [c for c in out.columns if c not in front and c not in drop_source_pair_cols]
    return out[front + rest]


def load_proteins(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    need = ["protein_md5", "fpid", "sequence"]
    df = pd.read_csv(path, usecols=need)
    for col in need:
        df[col] = df[col].astype(str)
    df["sequence"] = df["sequence"].map(normalize_sequence)
    df = df[(df["protein_md5"] != "") & (df["fpid"] != "") & (df["sequence"] != "")]
    fpid_to_md5 = dict(zip(df["fpid"], df["protein_md5"]))
    seqs = (
        df[["protein_md5", "sequence"]]
        .drop_duplicates("protein_md5")
        .rename(columns={"protein_md5": "id"})
        .sort_values("id")
        .reset_index(drop=True)
    )
    return seqs, fpid_to_md5


def recover_extra_test_sequences(
    seqs: pd.DataFrame,
    test_ids: set[str],
    extra_sequences_csv: Path | None,
) -> pd.DataFrame:
    present = set(seqs["id"].astype(str))
    missing = test_ids - present
    if not missing or extra_sequences_csv is None or not extra_sequences_csv.exists():
        return seqs

    extra = pd.read_csv(extra_sequences_csv, usecols=["id", "sequence"])
    extra["id"] = extra["id"].astype(str)
    extra = extra[extra["id"].isin(missing)].copy()
    extra["sequence"] = extra["sequence"].map(normalize_sequence)
    extra = extra[extra["sequence"] != ""]
    if len(extra):
        seqs = (
            pd.concat([seqs, extra[["id", "sequence"]]], ignore_index=True)
            .drop_duplicates("id")
            .sort_values("id")
            .reset_index(drop=True)
        )
    return seqs


def write_hash_pairs(
    source_csv: Path,
    out_csv: Path,
    fpid_to_md5: dict[str, str],
    chunk_size: int,
    keep_all_labels: bool,
    tier_include: set[str] | None = None,
) -> dict[str, object]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    first = True
    rows = 0
    skipped_label0 = 0
    skipped_tier = 0
    missing_rows = 0
    labels: Counter[int] = Counter()
    tiers: Counter[str] = Counter()
    used_ids: set[str] = set()

    for chunk in pd.read_csv(source_csv, chunksize=chunk_size):
        left, right = pair_columns(list(chunk.columns))
        chunk[left] = chunk[left].astype(str)
        chunk[right] = chunk[right].astype(str)
        if "label" not in chunk.columns:
            chunk["label"] = 1
        chunk["label"] = chunk["label"].astype(int)
        if tier_include is not None:
            if "PPI_Tier" not in chunk.columns:
                raise SystemExit("--tier-include requires a source CSV with a PPI_Tier column")
            chunk["PPI_Tier"] = chunk["PPI_Tier"].astype(str)
            before = len(chunk)
            chunk = chunk[chunk["PPI_Tier"].isin(tier_include)].copy()
            skipped_tier += before - len(chunk)
        if not keep_all_labels:
            before = len(chunk)
            chunk = chunk[chunk["label"] == 1].copy()
            skipped_label0 += before - len(chunk)
        if chunk.empty:
            continue

        mapped_left = chunk[left].map(fpid_to_md5)
        mapped_right = chunk[right].map(fpid_to_md5)
        ok = mapped_left.notna() & mapped_right.notna()
        missing_rows += int((~ok).sum())
        if not ok.all():
            chunk = chunk.loc[ok].copy()
            mapped_left = mapped_left.loc[ok]
            mapped_right = mapped_right.loc[ok]
        if chunk.empty:
            continue

        out = pd.DataFrame(
            {
                "ID_1": mapped_left.astype(str).to_numpy(),
                "ID_2": mapped_right.astype(str).to_numpy(),
                "label": chunk["label"].astype(int).to_numpy(),
            }
        )
        out.to_csv(out_csv, mode="w" if first else "a", header=first, index=False)
        first = False
        rows += len(out)
        labels.update(out["label"].astype(int).tolist())
        if "PPI_Tier" in chunk.columns:
            tiers.update(chunk["PPI_Tier"].astype(str).tolist())
        used_ids.update(out["ID_1"].astype(str))
        used_ids.update(out["ID_2"].astype(str))

    if first:
        raise SystemExit(f"{source_csv}: no mapped rows were written")
    return {
        "hash_pair_csv": str(out_csv),
        "rows": int(rows),
        "label_counts": {str(k): int(v) for k, v in sorted(labels.items())},
        "unique_pair_ids": int(len(used_ids)),
        "skipped_label0_rows": int(skipped_label0),
        "tier_include": None if tier_include is None else sorted(tier_include),
        "tier_counts": {str(k): int(v) for k, v in sorted(tiers.items())},
        "skipped_tier_rows": int(skipped_tier),
        "missing_mapping_rows": int(missing_rows),
    }


def filter_train(
    hash_pairs_csv: Path,
    out_train_csv: Path,
    tainted_ids: set[str],
    test_pairs: set[str],
    chunk_size: int,
) -> dict[str, object]:
    first = True
    rows_in = 0
    rows_after_cluster = 0
    rows_out = 0
    labels: Counter[int] = Counter()
    used_ids: set[str] = set()

    for chunk in pd.read_csv(hash_pairs_csv, chunksize=chunk_size):
        rows_in += len(chunk)
        chunk["ID_1"] = chunk["ID_1"].astype(str)
        chunk["ID_2"] = chunk["ID_2"].astype(str)
        keep = ~chunk["ID_1"].isin(tainted_ids) & ~chunk["ID_2"].isin(tainted_ids)
        chunk = chunk.loc[keep, ["ID_1", "ID_2", "label"]].copy()
        rows_after_cluster += len(chunk)
        if chunk.empty:
            continue
        keys = pair_key(chunk["ID_1"], chunk["ID_2"])
        chunk = chunk.loc[~keys.isin(test_pairs)].copy()
        if chunk.empty:
            continue
        chunk.to_csv(out_train_csv, mode="w" if first else "a", header=first, index=False)
        first = False
        rows_out += len(chunk)
        labels.update(chunk["label"].astype(int).tolist())
        used_ids.update(chunk["ID_1"].astype(str))
        used_ids.update(chunk["ID_2"].astype(str))

    if first:
        raise SystemExit("C3 filtering removed all train rows")
    return {
        "pairs_in": int(rows_in),
        "pairs_after_c3_cluster_filter": int(rows_after_cluster),
        "pairs_train_out": int(rows_out),
        "train_label_counts": {str(k): int(v) for k, v in sorted(labels.items())},
        "train_unique_ids": int(len(used_ids)),
        "used_train_ids": used_ids,
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    tier_include = parse_csv_values(args.tier_include)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    test_df = read_test(args.test_csv)
    test_ids = set(test_df["ID_1"].astype(str)) | set(test_df["ID_2"].astype(str))
    test_pairs = set(pair_key(test_df["ID_1"], test_df["ID_2"]))
    test_work_csv = work_dir / "test_hash.csv"
    test_df.to_csv(test_work_csv, index=False)

    hash_pairs_csv = work_dir / "pairs_hash.csv"
    seqs_work_csv = work_dir / "sequences_hash_all.csv"
    proteins_work_csv = work_dir / "proteins_hash_all.csv"

    prep_summary: dict[str, object]
    if args.reuse_prepared and hash_pairs_csv.exists() and seqs_work_csv.exists():
        print(f"reusing prepared files under {work_dir}", flush=True)
        seqs = pd.read_csv(seqs_work_csv)
        prepare_summary_json = work_dir / "prepare_summary.json"
        if prepare_summary_json.exists():
            prep_summary = json.loads(prepare_summary_json.read_text(encoding="utf-8"))
            prep_summary["reused"] = True
        else:
            prep_summary = {"hash_pair_csv": str(hash_pairs_csv), "reused": True}
    else:
        print(f"loading protein map: {args.merged_proteins}", flush=True)
        seqs, fpid_to_md5 = load_proteins(args.merged_proteins)
        seqs = recover_extra_test_sequences(seqs, test_ids, args.extra_sequences_csv)
        seqs.to_csv(seqs_work_csv, index=False)
        seqs.rename(columns={"id": "fpid"}).to_csv(proteins_work_csv, index=False)
        print(f"writing hash-ID candidate pairs: {hash_pairs_csv}", flush=True)
        prep_summary = write_hash_pairs(
            args.merged_pairs,
            hash_pairs_csv,
            fpid_to_md5,
            args.chunk_size,
            keep_all_labels=args.all_labels,
            tier_include=tier_include,
        )
        prep_summary["sequence_rows_all"] = int(len(seqs))
        prep_summary["fixed_test_rows"] = int(len(test_df))
        prep_summary["fixed_test_unique_ids"] = int(len(test_ids))

    write_summary(work_dir / "prepare_summary.json", prep_summary)
    if args.prepare_only:
        print(json.dumps(prep_summary, indent=2), flush=True)
        print(f"prepared hash-ID inputs under {work_dir}", flush=True)
        return

    cluster_dir = args.out_dir / f"homology_cluster_all_id{int(args.identity * 100)}_cov{int(args.coverage * 100)}"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    fasta = cluster_dir / "cluster_input.fasta"
    write_fasta(seqs, fasta)
    mmseqs_bin = resolve_mmseqs(args.mmseqs)
    cluster_tsv = run_mmseqs_easy_cluster(
        fasta=fasta,
        out_dir=cluster_dir,
        identity=args.identity,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        threads=args.threads,
        mmseqs_bin=mmseqs_bin,
        reuse_cluster=args.reuse_cluster,
    )
    cluster_map = parse_cluster_tsv(cluster_tsv)
    cluster_map_csv = cluster_dir / "cluster_map.csv"
    cluster_map.to_csv(cluster_map_csv, index=False)
    cluster_summary = {
        "sequences_csv": str(seqs_work_csv),
        "train_csv": None,
        "mmseqs": mmseqs_bin,
        "identity": args.identity,
        "coverage": args.coverage,
        "cov_mode": args.cov_mode,
        "threads": args.threads,
        "input_sequences": int(len(seqs)),
        "cluster_map_rows": int(len(cluster_map)),
        "clusters": int(cluster_map["cluster"].nunique()),
        "cluster_tsv": str(cluster_tsv),
        "cluster_map_csv": str(cluster_map_csv),
    }
    write_summary(cluster_dir / "SUMMARY.json", cluster_summary)

    test_clusters = set(cluster_map.loc[cluster_map["id"].isin(test_ids), "cluster"].astype(str))
    missing_test_cluster = sorted(test_ids - set(cluster_map["id"].astype(str)))
    tainted_ids = set(cluster_map.loc[cluster_map["cluster"].isin(test_clusters), "id"].astype(str))
    print(
        f"C3 tainted clusters={len(test_clusters):,}, tainted ids={len(tainted_ids):,}, "
        f"missing fixed-test ids in clusters={len(missing_test_cluster):,}",
        flush=True,
    )

    train_csv = args.out_dir / "train.csv"
    if train_csv.exists():
        train_csv.unlink()
    filter_summary = filter_train(
        hash_pairs_csv=hash_pairs_csv,
        out_train_csv=train_csv,
        tainted_ids=tainted_ids,
        test_pairs=test_pairs,
        chunk_size=args.chunk_size,
    )
    used_ids = set(filter_summary.pop("used_train_ids")) | test_ids

    test_out = args.out_dir / "test.csv"
    test_df.to_csv(test_out, index=False)
    write_val_alias = args.write_val_alias and not args.no_val_alias
    if write_val_alias:
        test_df.to_csv(args.out_dir / "val.csv", index=False)

    seqs_used = seqs[seqs["id"].astype(str).isin(used_ids)].copy()
    seqs_used = seqs_used.sort_values("id").reset_index(drop=True)
    seqs_used.to_csv(args.out_dir / "sequences.csv", index=False)

    fpid_map = pd.read_csv(args.merged_proteins, usecols=["fpid", "protein_md5", "sequence"])
    fpid_map["fpid"] = fpid_map["fpid"].astype(str)
    fpid_map["protein_md5"] = fpid_map["protein_md5"].astype(str)
    fpid_map["sequence"] = fpid_map["sequence"].map(normalize_sequence)
    fpid_map = fpid_map.rename(columns={"protein_md5": "sequence_md5"})
    fpid_map.drop(columns=["sequence"]).to_csv(
        args.out_dir / "fpid_to_sequence_md5.csv",
        index=False,
    )
    id_map = fpid_map[fpid_map["sequence_md5"].isin(used_ids)].copy()
    id_map["source_dataset"] = "merged"
    id_map = id_map.rename(columns={"fpid": "source_id"})
    mapped_md5s = set(id_map["sequence_md5"].astype(str))
    missing_id_map = seqs_used[~seqs_used["id"].isin(mapped_md5s)].copy()
    if len(missing_id_map):
        missing_id_map["source_dataset"] = "fixed_test"
        missing_id_map["source_id"] = missing_id_map["id"]
        missing_id_map = missing_id_map.rename(columns={"id": "sequence_md5"})
        id_map = pd.concat(
            [
                id_map[["source_dataset", "source_id", "sequence_md5", "sequence"]],
                missing_id_map[["source_dataset", "source_id", "sequence_md5", "sequence"]],
            ],
            ignore_index=True,
        )
    else:
        id_map = id_map[["source_dataset", "source_id", "sequence_md5", "sequence"]]
    id_map = id_map.drop_duplicates(["source_dataset", "source_id", "sequence_md5"]).sort_values(
        ["source_dataset", "source_id"]
    )
    id_map.to_csv(
        args.out_dir / "id_map.csv",
        index=False,
    )

    pairs_summary = {
        "train.csv": pair_summary(train_csv, args.merged_pairs),
        "test.csv": pair_summary(test_out, args.test_csv),
    }
    pairs_summary["train.csv"].update(
        {
            "candidate_hash_pair_csv": str(hash_pairs_csv),
            "pairs_in_before_c3": filter_summary["pairs_in"],
            "pairs_after_c3_cluster_filter": filter_summary["pairs_after_c3_cluster_filter"],
            "tainted_clusters": int(len(test_clusters)),
            "tainted_ids": int(len(tainted_ids)),
        }
    )
    if write_val_alias:
        pairs_summary["val.csv"] = pair_summary(args.out_dir / "val.csv", args.test_csv)

    summary = {
        "collection": args.collection_name or args.out_dir.name,
        "output_dir": str(args.out_dir),
        "id_namespace": "sequence_md5",
        "hash_method": "MD5(uppercase amino-acid sequence with whitespace removed)",
        "has_val": bool(write_val_alias),
        "sequence_sources": [
            {"path": str(args.merged_proteins), "source_dataset": "merged"},
            {"path": str(args.extra_sequences_csv), "source_dataset": "fixed_test_recovery"},
        ],
        "unique_sequences": int(len(seqs_used)),
        "id_map_rows": int(len(id_map)),
        "primary_test": "test.csv",
        "pairs": pairs_summary,
        "tier_include": None if tier_include is None else sorted(tier_include),
        "c3_filter": {
            "fixed_test_csv": str(args.test_csv),
            "train_positive_only": not args.all_labels,
            "identity": args.identity,
            "coverage": args.coverage,
            "cov_mode": args.cov_mode,
            "cluster_map_csv": str(cluster_map_csv),
            "clusters": int(cluster_map["cluster"].nunique()),
            "cluster_map_rows": int(len(cluster_map)),
            "fixed_test_rows": int(len(test_df)),
            "fixed_test_unique_ids": int(len(test_ids)),
            "tainted_clusters": int(len(test_clusters)),
            "tainted_ids": int(len(tainted_ids)),
            "missing_fixed_test_ids_in_clusters": missing_test_cluster[:20],
            "missing_fixed_test_ids_in_clusters_count": int(len(missing_test_cluster)),
            "pairs_in_before_c3": filter_summary["pairs_in"],
            "pairs_after_c3_cluster_filter": filter_summary["pairs_after_c3_cluster_filter"],
            "pairs_train_out": filter_summary["pairs_train_out"],
        },
        "prepare": prep_summary,
        "cluster_summary": cluster_summary,
        "elapsed_seconds": round(time.time() - t0, 1),
        "test_csv": str(args.test_csv),
        "cluster_map_csv": str(cluster_map_csv),
    }
    write_summary(args.out_dir / "SUMMARY.json", summary)
    write_hash_dataset_readme(args.out_dir, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

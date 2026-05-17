"""Build RF2-holdout train+filtered-val splits.

The existing RF2-holdout train split is already filtered against the fixed
RF2-PPI test proteins. This script applies the same endpoint-level MMseqs
cluster rule to the original strict_c3 validation split, then appends the kept
validation pairs to the clean train split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from p2psiglip_db.data.split_utils import resolve_mmseqs, run_mmseqs_easy_cluster, write_fasta, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-train-csv",
        type=Path,
        default=REPO / "data/datasets/rf2_holdout_c3_v1/train_rf2_homology_endpoint_clean_id40_cov80.csv",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=REPO / "data/datasets/strict_c3_v1/val.csv",
    )
    parser.add_argument(
        "--strict-sequences-csv",
        type=Path,
        default=REPO / "data/datasets/strict_c3_v1/sequences.csv",
    )
    parser.add_argument(
        "--rf2-pairs-csv",
        type=Path,
        default=REPO / "data/datasets/bench_rf2_ppi/pairs_1to10_hp.csv",
    )
    parser.add_argument(
        "--rf2-sequences-csv",
        type=Path,
        default=REPO / "data/datasets/bench_rf2_ppi/sequences_hp.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "data/datasets/rf2_holdout_c3_v1_trainplusval",
    )
    parser.add_argument("--identity", type=float, default=0.4)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--mmseqs", default=None)
    parser.add_argument(
        "--reuse-cluster",
        action="store_true",
        help="reuse an existing MMseqs cluster TSV under out-dir when present",
    )
    return parser.parse_args()


def normalize_pair_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = df.copy()
    if "fpid_1" in df.columns and "fpid_2" in df.columns:
        pass
    elif "ID_1" in df.columns and "ID_2" in df.columns:
        df = df.rename(columns={"ID_1": "fpid_1", "ID_2": "fpid_2"})
    elif "id_1" in df.columns and "id_2" in df.columns:
        df = df.rename(columns={"id_1": "fpid_1", "id_2": "fpid_2"})
    else:
        raise SystemExit(f"{path}: expected fpid_1/fpid_2, ID_1/ID_2, or id_1/id_2 columns")
    if "label" not in df.columns:
        raise SystemExit(f"{path}: missing label column")
    df["fpid_1"] = df["fpid_1"].astype(str)
    df["fpid_2"] = df["fpid_2"].astype(str)
    df["label"] = df["label"].astype(int)
    return df[["fpid_1", "fpid_2", "label"]]


def load_sequences(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "id" not in df.columns or "sequence" not in df.columns:
        raise SystemExit(f"{path}: expected id and sequence columns")
    return dict(zip(df["id"].astype(str), df["sequence"].astype(str).str.upper().str.strip()))


def run_mmseqs_cluster(
    records: list[tuple[str, str]],
    work_dir: Path,
    mmseqs_bin: str,
    identity: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    reuse_cluster: bool,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta = work_dir / "val_rf2_union.fasta"
    write_fasta(pd.DataFrame(records, columns=["id", "sequence"]), fasta)
    return run_mmseqs_easy_cluster(
        fasta=fasta,
        out_dir=work_dir,
        identity=identity,
        coverage=coverage,
        cov_mode=cov_mode,
        threads=threads,
        mmseqs_bin=mmseqs_bin,
        reuse_cluster=reuse_cluster,
    )


def parse_cluster_tsv(cluster_tsv: Path) -> tuple[set[str], dict[str, str], int, int]:
    member_to_rep: dict[str, str] = {}
    rep_to_sources: dict[str, set[str]] = {}
    with cluster_tsv.open() as handle:
        for line in handle:
            rep, member = line.rstrip("\n").split("\t")
            member_to_rep[member] = rep
            rep_to_sources.setdefault(rep, set()).add(member.split("::", 1)[0])

    rf2_reps = {rep for rep, sources in rep_to_sources.items() if "rf2" in sources}
    homologous_val_ids = {
        member.split("::", 1)[1]
        for member, rep in member_to_rep.items()
        if member.startswith("val::") and rep in rf2_reps
    }
    return homologous_val_ids, member_to_rep, len(rep_to_sources), len(rf2_reps)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_train = normalize_pair_columns(pd.read_csv(args.clean_train_csv), args.clean_train_csv)
    val = normalize_pair_columns(pd.read_csv(args.val_csv), args.val_csv)
    strict_seqs = load_sequences(args.strict_sequences_csv)
    rf2_seqs = load_sequences(args.rf2_sequences_csv)

    rf2_pairs = pd.read_csv(args.rf2_pairs_csv)
    if not {"ID_1", "ID_2", "label"}.issubset(rf2_pairs.columns):
        raise SystemExit(f"{args.rf2_pairs_csv}: expected ID_1, ID_2, and label columns")
    rf2_test_ids = set(rf2_pairs["ID_1"].astype(str)) | set(rf2_pairs["ID_2"].astype(str))
    val_ids = set(val["fpid_1"]) | set(val["fpid_2"])

    missing_val_seq = sorted(pid for pid in val_ids if pid not in strict_seqs)
    missing_rf2_seq = sorted(pid for pid in rf2_test_ids if pid not in rf2_seqs)
    if missing_val_seq or missing_rf2_seq:
        raise SystemExit(
            f"missing sequences: val={len(missing_val_seq)} rf2={len(missing_rf2_seq)} "
            f"first_val={missing_val_seq[:5]} first_rf2={missing_rf2_seq[:5]}"
        )

    records = [(f"val::{pid}", strict_seqs[pid]) for pid in sorted(val_ids)]
    records.extend((f"rf2::{pid}", rf2_seqs[pid]) for pid in sorted(rf2_test_ids))
    filter_dir = out_dir / "homology_filter_val_vs_rf2_id40_cov80"
    cluster_tsv = run_mmseqs_cluster(
        records,
        filter_dir,
        resolve_mmseqs(args.mmseqs),
        args.identity,
        args.coverage,
        args.cov_mode,
        args.threads,
        args.reuse_cluster,
    )
    homologous_val_ids, member_to_rep, n_clusters, n_rf2_clusters = parse_cluster_tsv(cluster_tsv)

    keep_mask = ~val["fpid_1"].isin(homologous_val_ids) & ~val["fpid_2"].isin(homologous_val_ids)
    val_clean = val.loc[keep_mask].copy()
    val_removed = val.loc[~keep_mask].copy()
    train_plus_val = pd.concat([clean_train, val_clean], ignore_index=True)

    train_plus_ids = set(train_plus_val["fpid_1"]) | set(train_plus_val["fpid_2"])
    train_plus_sequences = pd.DataFrame(
        [{"id": pid, "sequence": strict_seqs[pid]} for pid in sorted(train_plus_ids)]
    )

    homologous_rows = []
    for pid in sorted(homologous_val_ids):
        tag = f"val::{pid}"
        homologous_rows.append(
            {
                "fpid": pid,
                "cluster_rep": member_to_rep.get(tag),
            }
        )

    out_val_clean = out_dir / "val_rf2_homology_endpoint_clean_id40_cov80.csv"
    out_val_removed = out_dir / "val_rf2_homology_endpoint_removed_id40_cov80.csv"
    out_train_plus_val = out_dir / "train_plus_val_rf2_homology_endpoint_clean_id40_cov80.csv"
    out_sequences = out_dir / "sequences_train_plus_val.csv"
    out_homologous = out_dir / "val_proteins_homologous_to_rf2_id40_cov80.csv"

    val_clean.to_csv(out_val_clean, index=False)
    val_removed.to_csv(out_val_removed, index=False)
    train_plus_val.to_csv(out_train_plus_val, index=False)
    train_plus_sequences.to_csv(out_sequences, index=False)
    pd.DataFrame(homologous_rows).to_csv(out_homologous, index=False)

    summary = {
        "method": "append original strict_c3_v1 val after endpoint-level RF2 MMseqs cluster filtering",
        "source_clean_train_csv": str(args.clean_train_csv),
        "source_val_csv": str(args.val_csv),
        "rf2_test_pairs_csv_unchanged": str(args.rf2_pairs_csv),
        "rf2_sequences_csv": str(args.rf2_sequences_csv),
        "mmseqs_cluster_tsv": str(cluster_tsv),
        "identity": args.identity,
        "coverage": args.coverage,
        "coverage_mode": args.cov_mode,
        "rf2_test_pairs": int(len(rf2_pairs)),
        "rf2_test_positives": int(rf2_pairs["label"].astype(int).sum()),
        "rf2_test_negatives": int((rf2_pairs["label"].astype(int) == 0).sum()),
        "rf2_test_proteins": int(len(rf2_test_ids)),
        "val_pairs_original": int(len(val)),
        "val_proteins_original": int(len(val_ids)),
        "val_proteins_homologous_to_rf2": int(len(homologous_val_ids)),
        "val_pairs_kept": int(len(val_clean)),
        "val_pairs_removed": int(len(val_removed)),
        "clean_train_pairs": int(len(clean_train)),
        "train_plus_val_pairs": int(len(train_plus_val)),
        "train_plus_val_proteins": int(len(train_plus_ids)),
        "mmseqs_clusters_total": int(n_clusters),
        "clusters_containing_rf2": int(n_rf2_clusters),
        "outputs": {
            "filtered_val_csv": str(out_val_clean),
            "removed_val_csv": str(out_val_removed),
            "train_plus_val_csv": str(out_train_plus_val),
            "train_plus_val_sequences_csv": str(out_sequences),
            "homologous_val_proteins_csv": str(out_homologous),
        },
    }
    write_summary(out_dir / "SUMMARY.json", summary)
    (out_dir / "README.md").write_text(
        "# rf2_holdout_c3_v1_trainplusval\n\n"
        "RF2-PPI test is fixed at `data/datasets/bench_rf2_ppi/pairs_1to10_hp.csv`.\n"
        "The original strict_c3 validation split is filtered against RF2 test proteins "
        "by MMseqs2 cluster membership at 40% identity / 80% coverage / cov-mode 0, "
        "then appended to the existing RF2-clean training split.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

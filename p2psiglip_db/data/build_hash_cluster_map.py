"""Build an id -> MMseqs cluster map for hash-based sequence datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from p2psiglip_db.data.split_utils import (
    load_hash_sequences,
    pair_ids,
    parse_cluster_tsv,
    resolve_mmseqs,
    run_mmseqs_easy_cluster,
    write_fasta,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences-csv", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, default=None,
                        help="optional pair CSV; restrict clustering to IDs used by this split")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--identity", type=float, default=0.4)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--mmseqs", default=None)
    parser.add_argument("--reuse-cluster", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    keep_ids = pair_ids(args.train_csv) if args.train_csv else None
    seqs = load_hash_sequences(args.sequences_csv, keep_ids)
    if seqs.empty:
        raise SystemExit("no sequences to cluster")
    fasta = args.out_dir / "cluster_input.fasta"
    write_fasta(seqs, fasta)
    mmseqs_bin = resolve_mmseqs(args.mmseqs)
    print(f"clustering {len(seqs):,} sequences with {mmseqs_bin}", flush=True)
    cluster_tsv = run_mmseqs_easy_cluster(
        fasta=fasta,
        out_dir=args.out_dir,
        identity=args.identity,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        threads=args.threads,
        mmseqs_bin=mmseqs_bin,
        reuse_cluster=args.reuse_cluster,
    )
    cluster_map = parse_cluster_tsv(cluster_tsv)
    cluster_map_path = args.out_dir / "cluster_map.csv"
    cluster_map.to_csv(cluster_map_path, index=False)
    summary = {
        "sequences_csv": str(args.sequences_csv),
        "train_csv": None if args.train_csv is None else str(args.train_csv),
        "mmseqs": mmseqs_bin,
        "identity": args.identity,
        "coverage": args.coverage,
        "cov_mode": args.cov_mode,
        "threads": args.threads,
        "input_sequences": int(len(seqs)),
        "cluster_map_rows": int(len(cluster_map)),
        "clusters": int(cluster_map["cluster"].nunique()),
        "cluster_tsv": str(cluster_tsv),
        "cluster_map_csv": str(cluster_map_path),
    }
    write_summary(args.out_dir / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

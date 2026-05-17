"""Validate canonical hash-ID training split collections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from p2psiglip_db.data.split_utils import (
    split_csvs,
    validate_hash_pair_csv,
    validate_hash_sequences,
)


DEFAULT_DIRS = (
    "p2psiglip_hash_v1",
    "virahinter_hp_hash_v1",
    "rf2ppi_holdout_hash_v1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO / "data/datasets")
    parser.add_argument("--embed-root", type=Path, default=REPO / "data/embeds")
    parser.add_argument(
        "--collections",
        default=",".join(DEFAULT_DIRS),
        help="Comma-separated collection directory names under --dataset-root.",
    )
    parser.add_argument(
        "--report-embeds",
        default="",
        help="Comma-separated PLMs to report missing embeddings for without failing.",
    )
    parser.add_argument(
        "--require-embeds",
        default="",
        help="Comma-separated PLMs that must have every sequence embedded.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Rows per chunk when validating large pair CSVs.",
    )
    return parser.parse_args()


def parse_list(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def embed_coverage(sequence_ids: set[str], embed_root: Path, plms: tuple[str, ...]) -> dict[str, object]:
    coverage = {}
    for plm in plms:
        embed_dir = embed_root / plm
        missing = sorted(pid for pid in sequence_ids if not (embed_dir / f"{pid}.npy").exists())
        coverage[plm] = {
            "embed_dir": str(embed_dir),
            "present": int(len(sequence_ids) - len(missing)),
            "missing": int(len(missing)),
            "first_missing": missing[:10],
        }
    return coverage


def main() -> None:
    args = parse_args()
    collection_names = parse_list(args.collections)
    report_plms = parse_list(args.report_embeds)
    require_plms = parse_list(args.require_embeds)
    all_plms = tuple(dict.fromkeys((*report_plms, *require_plms)))

    summaries = {}
    strict_failures = []
    for name in collection_names:
        collection_dir = args.dataset_root / name
        if not collection_dir.exists():
            raise SystemExit(f"missing collection dir: {collection_dir}")
        seq, sequence_ids = validate_hash_sequences(collection_dir / "sequences.csv")
        pair_summaries = {}
        for csv_path in split_csvs(collection_dir):
            pair_summaries[csv_path.name] = validate_hash_pair_csv(csv_path, sequence_ids, args.chunk_size)

        coverage = embed_coverage(sequence_ids, args.embed_root, all_plms) if all_plms else {}
        for plm in require_plms:
            missing = coverage[plm]["missing"]
            if missing:
                strict_failures.append(f"{name}:{plm} missing {missing}")

        summaries[name] = {
            "sequence_rows": int(len(seq)),
            "pair_csvs": pair_summaries,
            "embed_coverage": coverage,
        }

    print(json.dumps(summaries, indent=2))
    if strict_failures:
        raise SystemExit("embedding coverage failures: " + "; ".join(strict_failures))


if __name__ == "__main__":
    main()

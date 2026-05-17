"""Build canonical hash-ID training split collections.

The output pair CSVs all use ID_1, ID_2, label where IDs are
MD5(normalized amino-acid sequence). Those IDs match files in:

  data/embeds/{esmc,esm2,prot5_3di}/{sequence_md5}.npy
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from p2psiglip_db.data.split_utils import (
    label_counts_to_json,
    merge_sequence_sources,
    pair_columns,
    write_hash_dataset_readme,
    write_summary,
)


DEFAULT_COLLECTIONS = ("p2psiglip", "virahinter", "rf2ppi")


@dataclass(frozen=True)
class PairSpec:
    source: Path
    output_name: str


@dataclass(frozen=True)
class CollectionSpec:
    key: str
    output_dir: Path
    sequence_sources: tuple[tuple[Path, str], ...]
    pairs: tuple[PairSpec, ...]
    primary_test: str
    has_val: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collections",
        default="all",
        help="Comma-separated subset: all,p2psiglip,virahinter,rf2ppi",
    )
    parser.add_argument("--dataset-root", type=Path, default=REPO / "data/datasets")
    parser.add_argument("--embed-root", type=Path, default=REPO / "data/embeds")
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=REPO / "data/embeds/manifests/training_splits_v1",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Rows per chunk when rewriting large pair CSVs.",
    )
    parser.add_argument(
        "--no-link-legacy-embeds",
        action="store_true",
        help="Only write split CSVs/manifests; do not hardlink/copy legacy embeddings.",
    )
    return parser.parse_args()


def write_hash_pairs(
    source_csv: Path,
    output_csv: Path,
    id_to_hash: dict[str, str],
    chunk_size: int,
) -> dict[str, object]:
    if not source_csv.exists():
        raise SystemExit(f"missing pair CSV: {source_csv}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    labels: Counter = Counter()
    unique_ids: set[str] = set()
    first = True
    pair_cols: tuple[str, str] | None = None

    for chunk in pd.read_csv(source_csv, chunksize=chunk_size):
        if pair_cols is None:
            pair_cols = pair_columns(chunk.columns)
        left, right = pair_cols
        chunk[left] = chunk[left].astype(str)
        chunk[right] = chunk[right].astype(str)

        missing = sorted((set(chunk[left]) | set(chunk[right])) - set(id_to_hash))
        if missing:
            raise SystemExit(f"{source_csv}: {len(missing)} IDs missing sequence mapping, first={missing[:5]}")

        out = chunk.copy()
        out["ID_1"] = chunk[left].map(id_to_hash)
        out["ID_2"] = chunk[right].map(id_to_hash)
        if "label" not in out.columns:
            out["label"] = 1

        front = ["ID_1", "ID_2", "label"]
        drop_source_pair_cols = {left, right} - set(front)
        rest = [c for c in out.columns if c not in front and c not in drop_source_pair_cols]
        out = out[front + rest]

        out.to_csv(output_csv, mode="w" if first else "a", index=False, header=first)
        first = False

        total_rows += len(out)
        labels.update(out["label"].astype(int).tolist())
        unique_ids.update(out["ID_1"].astype(str))
        unique_ids.update(out["ID_2"].astype(str))

    if first:
        raise SystemExit(f"{source_csv}: no rows found")

    return {
        "source": str(source_csv),
        "output": str(output_csv),
        "rows": int(total_rows),
        "label_counts": label_counts_to_json(labels),
        "unique_pair_ids": int(len(unique_ids)),
    }


def copy_primary_test(collection_dir: Path, primary_test: str) -> None:
    source = collection_dir / primary_test
    target = collection_dir / "test.csv"
    if source == target:
        return
    if target.exists():
        target.unlink()
    shutil.copy2(source, target)


def hardlink_or_copy(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return True
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return True


def legacy_embed_dirs() -> dict[str, tuple[Path, ...]]:
    return {
        "esmc": (
            REPO / "data/datasets/strict_c3_v1/embed_perres",
            REPO / "runs/strict_vh_v1/rf2ppi_embeddings/esmc",
        ),
        "esm2": (
            REPO / "runs/rf2ppi_holdout_mplm3/embeddings/train_esm2_perres",
            REPO / "runs/strict_vh_v1/rf2ppi_embeddings/esm2",
            REPO / "runs/strict_vh_v1/multiplm/embeddings_esm2",
        ),
        "prot5_3di": (
            REPO / "data/datasets/strict_c3_v1/embed_perres_prostt5_3di",
        ),
    }


def link_legacy_embeddings(
    all_id_map: pd.DataFrame,
    embed_root: Path,
) -> dict[str, object]:
    id_to_hash = dict(zip(all_id_map["source_id"], all_id_map["sequence_md5"]))
    stats: dict[str, object] = {}
    for plm, dirs in legacy_embed_dirs().items():
        dest_dir = embed_root / plm
        dest_dir.mkdir(parents=True, exist_ok=True)
        plm_stats = []
        for source_dir in dirs:
            linked_or_existing = 0
            missing_source_id_files = 0
            if not source_dir.exists():
                plm_stats.append(
                    {
                        "source_dir": str(source_dir),
                        "exists": False,
                        "linked_or_existing": 0,
                        "missing_source_id_files": len(id_to_hash),
                    }
                )
                continue
            for source_id, md5 in id_to_hash.items():
                if hardlink_or_copy(source_dir / f"{source_id}.npy", dest_dir / f"{md5}.npy"):
                    linked_or_existing += 1
                else:
                    missing_source_id_files += 1
            plm_stats.append(
                {
                    "source_dir": str(source_dir),
                    "exists": True,
                    "linked_or_existing": int(linked_or_existing),
                    "missing_source_id_files": int(missing_source_id_files),
                }
            )
        stats[plm] = plm_stats
    return stats


def write_missing_sequences(unique_sequences: pd.DataFrame, embed_root: Path, manifest_root: Path) -> dict[str, int]:
    manifest_root.mkdir(parents=True, exist_ok=True)
    missing_counts: dict[str, int] = {}
    for plm in ("esmc", "esm2", "prot5_3di"):
        embed_dir = embed_root / plm
        missing = unique_sequences[
            ~unique_sequences["id"].map(lambda md5: (embed_dir / f"{md5}.npy").exists())
        ].copy()
        missing.to_csv(manifest_root / f"missing_{plm}.csv", index=False)
        missing_counts[plm] = int(len(missing))
    return missing_counts


def specs(dataset_root: Path) -> dict[str, CollectionSpec]:
    strict_c3 = dataset_root / "strict_c3_v1"
    host = dataset_root / "bench_host_corpus"
    rf2 = dataset_root / "bench_rf2_ppi"
    rf2_train = dataset_root / "rf2_holdout_c3_v1_trainplusval"

    p2p_size_tests = tuple(
        PairSpec(path, path.name)
        for path in sorted(strict_c3.glob("test_*.csv"))
    )

    return {
        "p2psiglip": CollectionSpec(
            key="p2psiglip",
            output_dir=dataset_root / "p2psiglip_hash_v1",
            sequence_sources=((strict_c3 / "sequences.csv", "strict_c3_v1"),),
            pairs=(
                PairSpec(strict_c3 / "train.csv", "train.csv"),
                PairSpec(strict_c3 / "val.csv", "val.csv"),
                PairSpec(strict_c3 / "P2PSigLip_benchmark_test.csv", "test.csv"),
                *p2p_size_tests,
            ),
            primary_test="test.csv",
            has_val=True,
        ),
        "virahinter": CollectionSpec(
            key="virahinter",
            output_dir=dataset_root / "virahinter_hp_hash_v1",
            sequence_sources=((host / "sequences_hp.csv", "bench_host_corpus"),),
            pairs=(
                PairSpec(host / "host_v3_train_pos_hp.csv", "train.csv"),
                PairSpec(host / "virahinter_highconfidence_val_pos_hp.csv", "val.csv"),
                PairSpec(host / "virahinter_highconfidence_val_hp.csv", "metric_val.csv"),
                PairSpec(host / "virahinter_highconfidence_test_hp.csv", "test.csv"),
                PairSpec(host / "host_v3_train_pos_hp.csv", "host_v3_train_pos_hp.csv"),
                PairSpec(host / "host_v3_train_pos_minimal_hp.csv", "host_v3_train_pos_minimal_hp.csv"),
                PairSpec(host / "hvidb_all_positives_hp.csv", "hvidb_all_positives_hp.csv"),
                PairSpec(host / "virahinter_highconfidence_train_hp.csv", "virahinter_highconfidence_train_hp.csv"),
                PairSpec(host / "virahinter_highconfidence_train_pos_hp.csv", "virahinter_highconfidence_train_pos_hp.csv"),
                PairSpec(host / "virahinter_highconfidence_val_hp.csv", "virahinter_highconfidence_val_hp.csv"),
                PairSpec(host / "virahinter_highconfidence_val_lt_3000_hp.csv", "virahinter_highconfidence_val_lt_3000_hp.csv"),
                PairSpec(host / "virahinter_highconfidence_val_pos_hp.csv", "virahinter_highconfidence_val_pos_hp.csv"),
                PairSpec(host / "virahinter_highconfidence_test_hp.csv", "virahinter_highconfidence_test_hp.csv"),
            ),
            primary_test="test.csv",
            has_val=True,
        ),
        "rf2ppi": CollectionSpec(
            key="rf2ppi",
            output_dir=dataset_root / "rf2ppi_holdout_hash_v1",
            sequence_sources=(
                (rf2_train / "sequences_train_plus_val.csv", "rf2_holdout_c3_v1_trainplusval"),
                (rf2 / "sequences_hp.csv", "bench_rf2_ppi"),
            ),
            pairs=(
                PairSpec(
                    rf2_train / "train_plus_val_rf2_homology_endpoint_clean_id40_cov80.csv",
                    "train.csv",
                ),
                PairSpec(rf2 / "pairs_1to10_hp.csv", "test_1to10.csv"),
                PairSpec(rf2 / "pairs_1to1000_hp.csv", "test_1to1000.csv"),
            ),
            primary_test="test_1to10.csv",
            has_val=False,
        ),
    }


def requested_collections(value: str) -> tuple[str, ...]:
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    if not requested or requested == ("all",):
        return DEFAULT_COLLECTIONS
    unknown = sorted(set(requested) - set(DEFAULT_COLLECTIONS))
    if unknown:
        raise SystemExit(f"unknown collections: {unknown}")
    return requested


def build_collection(spec: CollectionSpec, chunk_size: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    id_map, unique_sequences, id_to_hash = merge_sequence_sources(spec.sequence_sources)

    id_map.to_csv(spec.output_dir / "id_map.csv", index=False)
    unique_sequences.to_csv(spec.output_dir / "sequences.csv", index=False)

    pair_summaries = {}
    for pair in spec.pairs:
        pair_summaries[pair.output_name] = write_hash_pairs(
            pair.source,
            spec.output_dir / pair.output_name,
            id_to_hash,
            chunk_size,
        )

    copy_primary_test(spec.output_dir, spec.primary_test)

    summary = {
        "collection": spec.key,
        "output_dir": str(spec.output_dir),
        "id_namespace": "sequence_md5",
        "hash_method": "MD5(uppercase amino-acid sequence with whitespace removed)",
        "has_val": spec.has_val,
        "sequence_sources": [
            {"path": str(path), "source_dataset": source}
            for path, source in spec.sequence_sources
        ],
        "unique_sequences": int(len(unique_sequences)),
        "id_map_rows": int(len(id_map)),
        "primary_test": "test.csv",
        "pairs": pair_summaries,
    }
    write_summary(spec.output_dir / "SUMMARY.json", summary)
    write_hash_dataset_readme(spec.output_dir, summary)
    return id_map, unique_sequences, summary


def main() -> None:
    args = parse_args()
    selected = requested_collections(args.collections)
    all_specs = specs(args.dataset_root)

    all_id_maps = []
    all_sequences = []
    collection_summaries = {}
    for key in selected:
        id_map, unique_sequences, summary = build_collection(all_specs[key], args.chunk_size)
        all_id_maps.append(id_map)
        all_sequences.append(unique_sequences)
        collection_summaries[key] = summary

    manifest_root = args.manifest_root
    manifest_root.mkdir(parents=True, exist_ok=True)
    all_id_map = (
        pd.concat(all_id_maps, ignore_index=True)
        .drop_duplicates(["source_dataset", "source_id", "sequence_md5"])
        .sort_values(["source_dataset", "source_id"])
        .reset_index(drop=True)
    )
    all_unique_sequences = (
        pd.concat(all_sequences, ignore_index=True)
        .drop_duplicates("id")
        .sort_values("id")
        .reset_index(drop=True)
    )
    all_id_map.to_csv(manifest_root / "id_to_sequence_md5.csv", index=False)
    all_unique_sequences.to_csv(manifest_root / "sequences_by_md5.csv", index=False)

    link_stats = {}
    if not args.no_link_legacy_embeds:
        link_stats = link_legacy_embeddings(all_id_map, args.embed_root)
    missing_after_link = write_missing_sequences(all_unique_sequences, args.embed_root, manifest_root)

    summary = {
        "collections": list(selected),
        "dataset_root": str(args.dataset_root),
        "embed_root": str(args.embed_root),
        "manifest_root": str(manifest_root),
        "unique_sequences": int(len(all_unique_sequences)),
        "id_map_rows": int(len(all_id_map)),
        "collection_summaries": collection_summaries,
        "legacy_link_stats": link_stats,
        "missing_after_link": missing_after_link,
    }
    write_summary(manifest_root / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

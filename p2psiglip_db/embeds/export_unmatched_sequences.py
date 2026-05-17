"""Export project sequences without an AFDB/UniProt exact match."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from p2psiglip_db.embeds.copy_afdb_structures import discover_inputs, load_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export sequence CSV for records with matched=0 in the UniProt mapping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mapping-csv", type=Path, default=Path("data/embeds/manifests/strucs/full_sequence_uniprot_ids.csv"))
    parser.add_argument("-i", "--input", action="append", type=Path)
    parser.add_argument("--datasets-root", type=Path, default=Path("data/datasets"))
    parser.add_argument("--dataset-glob", default="*hash_v1/sequences.csv")
    parser.add_argument("--out-csv", type=Path, default=Path("data/embeds/manifests/strucs/minifold_unmatched_sequences.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("data/embeds/manifests/strucs/minifold_unmatched_sequences_summary.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.mapping_csv.is_file():
        raise FileNotFoundError(args.mapping_csv)

    inputs = args.input or discover_inputs(args.datasets_root, args.dataset_glob)
    if not inputs:
        raise SystemExit("no input sequence files found")

    mapping = pd.read_csv(args.mapping_csv)
    required = {"sequence_md5", "matched"}
    missing = required.difference(mapping.columns)
    if missing:
        raise ValueError(f"{args.mapping_csv} missing columns: {sorted(missing)}")

    unmatched = set(mapping.loc[mapping["matched"].astype(int) == 0, "sequence_md5"].astype(str))
    targets = load_targets(inputs)

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for target in targets:
        if target.sequence_md5 not in unmatched or target.sequence_md5 in seen:
            continue
        seen.add(target.sequence_md5)
        rows.append(
            {
                "id": target.sequence_md5,
                "sequence": target.sequence,
                "sequence_md5": target.sequence_md5,
                "length": target.length,
                "target_id": target.target_id,
                "source_files": target.source,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "sequence", "sequence_md5", "length", "target_id", "source_files"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "mapping_csv": str(args.mapping_csv),
        "inputs": [str(p) for p in inputs],
        "unmatched_in_mapping": len(unmatched),
        "exported_rows": len(rows),
        "out_csv": str(args.out_csv),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

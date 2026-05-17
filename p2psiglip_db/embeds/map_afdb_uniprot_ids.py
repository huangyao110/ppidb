"""Map local project sequences to AFDB/UniProt accessions.

The project datasets often use sequence hashes as ids. This script writes a
single CSV that maps each local sequence to a UniProt accession when it can be
found in a local AlphaFold DB FASTA by exact sequence match.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from p2psiglip_db.embeds.copy_afdb_structures import (
    Target,
    discover_inputs,
    iter_fasta,
    load_targets,
    parse_afdb_accession,
    sequence_md5,
)


def unique_targets_by_sequence(targets: list[Target]) -> list[Target]:
    unique: dict[str, Target] = {}
    for target in targets:
        unique.setdefault(target.sequence_md5, target)
    return list(unique.values())


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_id",
        "sequence_md5",
        "length",
        "source_files",
        "matched",
        "uniprot_id",
        "afdb_header",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a CSV mapping local sequences to AFDB/UniProt ids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", action="append", type=Path)
    parser.add_argument("--datasets-root", type=Path, default=Path("data/datasets"))
    parser.add_argument("--dataset-glob", default="*hash_v1/sequences.csv")
    parser.add_argument("--afdb-fasta", type=Path, default=Path("/media/zlab/ZhaoLab_27/afdb/monoer/sequences.fasta"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/embeds/manifests/strucs/full_sequence_uniprot_ids.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("data/embeds/manifests/strucs/full_sequence_uniprot_ids_summary.json"))
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument("--max-fasta-records", type=int, default=None)
    parser.add_argument("--max-targets", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inputs = args.input
    if not inputs:
        inputs = discover_inputs(args.datasets_root, args.dataset_glob)
        print("auto-discovered inputs:\n  " + "\n  ".join(str(p) for p in inputs), flush=True)
    if not inputs:
        raise SystemExit("no input sequence files found")
    if not args.afdb_fasta.is_file():
        raise FileNotFoundError(args.afdb_fasta)

    targets = load_targets(inputs, max_targets=args.max_targets)
    unique_targets = unique_targets_by_sequence(targets)
    source_by_md5: dict[str, set[str]] = defaultdict(set)
    ids_by_md5: dict[str, set[str]] = defaultdict(set)
    lengths_by_md5: dict[str, int] = {}
    for target in targets:
        source_by_md5[target.sequence_md5].add(target.source)
        ids_by_md5[target.sequence_md5].add(target.target_id)
        lengths_by_md5[target.sequence_md5] = target.length

    needed = {target.sequence_md5 for target in unique_targets}
    needed_lengths = {target.length for target in unique_targets}
    matched: dict[str, tuple[str, str]] = {}

    print(f"loaded target rows: {len(targets):,}", flush=True)
    print(f"unique target sequences: {len(unique_targets):,}", flush=True)
    start = time.time()
    scanned = 0
    for header, seq in iter_fasta(args.afdb_fasta):
        if args.max_fasta_records is not None and scanned >= args.max_fasta_records:
            break
        scanned += 1
        if len(seq) in needed_lengths:
            md5 = sequence_md5(seq)
            if md5 in needed and md5 not in matched:
                accession = parse_afdb_accession(header) or ""
                matched[md5] = (accession, header)
                if len(matched) == len(needed):
                    break
        if args.progress_every and scanned % args.progress_every == 0:
            elapsed = time.time() - start
            print(
                f"scanned AFDB FASTA records: {scanned:,}; "
                f"matched: {len(matched):,}/{len(needed):,}; "
                f"elapsed: {elapsed / 60:.1f} min",
                flush=True,
            )

    rows: list[dict[str, object]] = []
    for md5 in sorted(needed):
        accession, header = matched.get(md5, ("", ""))
        rows.append(
            {
                "target_id": ";".join(sorted(ids_by_md5[md5])),
                "sequence_md5": md5,
                "length": lengths_by_md5[md5],
                "source_files": ";".join(sorted(source_by_md5[md5])),
                "matched": int(bool(accession)),
                "uniprot_id": accession,
                "afdb_header": header,
            }
        )

    write_rows(args.out_csv, rows)
    summary = {
        "inputs": [str(p) for p in inputs],
        "afdb_fasta": str(args.afdb_fasta),
        "out_csv": str(args.out_csv),
        "target_rows": len(targets),
        "unique_target_sequences": len(unique_targets),
        "scanned_afdb_records": scanned,
        "matched": len(matched),
        "unmatched": len(unique_targets) - len(matched),
        "elapsed_sec": time.time() - start,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

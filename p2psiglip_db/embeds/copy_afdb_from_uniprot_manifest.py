"""Copy AFDB mmCIF files using a sequence-to-UniProt mapping CSV."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

import pandas as pd


AFDB_CIF_RE = re.compile(r"^AF-(?P<accession>.+)-F(?P<fragment>\d+)-model_(?P<version>[^.]+)\.cif\.gz$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy AFDB cif.gz files for matched UniProt accessions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mapping-csv", type=Path, default=Path("data/embeds/manifests/strucs/full_sequence_uniprot_ids.csv"))
    parser.add_argument("--afdb-dir", type=Path, default=Path("/media/zlab/ZhaoLab_27/afdb/monoer"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/embeds/strucs/afdb_matched"))
    parser.add_argument("--manifest", type=Path, default=Path("data/embeds/manifests/strucs/afdb_matched_cif_manifest.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("data/embeds/manifests/strucs/afdb_matched_cif_summary.json"))
    parser.add_argument("--model-version", default="v6")
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def iter_structure_roots(afdb_dir: Path) -> list[Path]:
    return [root for root in (afdb_dir, afdb_dir / "cifgz") if root.is_dir()]


def build_structure_index(afdb_dir: Path, accessions: set[str], model_version: str) -> dict[str, Path]:
    """Index local AFDB CIF files once, preferring F1 files for the requested model version."""
    index: dict[str, Path] = {}
    fallback: dict[str, Path] = {}
    start = time.time()
    scanned = 0
    for root in iter_structure_roots(afdb_dir):
        for path in root.glob("AF-*-F*-model_*.cif.gz"):
            scanned += 1
            match = AFDB_CIF_RE.match(path.name)
            if match is None:
                continue
            accession = match.group("accession").upper()
            if accession not in accessions:
                continue
            fragment = match.group("fragment")
            version = match.group("version")
            if accession not in fallback:
                fallback[accession] = path
            if fragment == "1" and version == model_version:
                index[accession] = path

    for accession, path in fallback.items():
        index.setdefault(accession, path)

    print(
        json.dumps(
            {
                "indexed_structure_files": scanned,
                "requested_accessions": len(accessions),
                "found_accessions": len(index),
                "elapsed_sec": round(time.time() - start, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return index


def copy_file_atomic(source: Path, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        if tmp.exists():
            tmp.unlink()
        shutil.copyfile(source, tmp)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def existing_copy_is_complete(source: Path, dest: Path) -> bool:
    try:
        return dest.is_file() and dest.stat().st_size == source.stat().st_size
    except OSError:
        return False


def run_copy_task(task: dict[str, object], overwrite: bool, dry_run: bool) -> dict[str, object]:
    source = task["source"]
    dest = Path(str(task["output_path"]))
    status = "missing_source"
    error = ""

    try:
        if source is None:
            status = "missing_source"
        else:
            source_path = Path(str(source))
            if dest.exists() and not overwrite and existing_copy_is_complete(source_path, dest):
                status = "exists"
            else:
                status = "dry_run" if dry_run else "copied"
                if not dry_run:
                    copy_file_atomic(source_path, dest)
    except Exception as exc:  # Keep the manifest useful if one source file is unreadable.
        status = "copy_error"
        error = repr(exc)

    task["status"] = status
    task["error"] = error
    return task


def main() -> None:
    args = build_parser().parse_args()
    if not args.mapping_csv.is_file():
        raise FileNotFoundError(args.mapping_csv)

    df = pd.read_csv(args.mapping_csv)
    required = {"sequence_md5", "matched", "uniprot_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{args.mapping_csv} missing columns: {sorted(missing)}")

    matched = df[(df["matched"].astype(int) == 1) & df["uniprot_id"].notna() & (df["uniprot_id"].astype(str) != "")]
    accessions = {str(value).upper() for value in matched["uniprot_id"].tolist()}
    structure_index = build_structure_index(args.afdb_dir, accessions, args.model_version)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, object]] = []
    for row in matched.itertuples(index=False):
        row_dict = row._asdict()
        sequence_md5 = str(row_dict["sequence_md5"])
        accession = str(row_dict["uniprot_id"]).upper()
        source = structure_index.get(accession)
        dest = args.out_dir / f"{sequence_md5}.cif.gz"
        tasks.append(
            {
                "sequence_md5": sequence_md5,
                "uniprot_id": accession,
                "length": row_dict.get("length", ""),
                "target_id": row_dict.get("target_id", ""),
                "source_structure": "" if source is None else str(source),
                "source": source,
                "output_path": str(dest),
                "status": "",
                "error": "",
            }
        )
    tasks.sort(key=lambda item: (item["source"] is None, "" if item["source"] is None else Path(str(item["source"])).name))

    rows: list[dict[str, object]] = []
    counts = {"copied": 0, "dry_run": 0, "exists": 0, "missing_source": 0, "copy_error": 0}
    start = time.time()

    def record_result(result: dict[str, object], processed: int) -> None:
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1
        result.pop("source", None)
        rows.append(result)
        if args.progress_every > 0 and processed % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "matched_rows": int(len(matched)),
                        **counts,
                        "elapsed_sec": round(time.time() - start, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    workers = max(1, int(args.workers))
    if workers == 1:
        for processed, task in enumerate(tasks, start=1):
            record_result(run_copy_task(task, args.overwrite, args.dry_run), processed)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_copy_task, task, args.overwrite, args.dry_run) for task in tasks]
            for processed, future in enumerate(as_completed(futures), start=1):
                record_result(future.result(), processed)

    with args.manifest.open("w", newline="") as handle:
        fieldnames = [
            "sequence_md5",
            "uniprot_id",
            "length",
            "target_id",
            "source_structure",
            "output_path",
            "status",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "mapping_csv": str(args.mapping_csv),
        "afdb_dir": str(args.afdb_dir),
        "out_dir": str(args.out_dir),
        "matched_rows": int(len(matched)),
        "copied": counts.get("copied", 0),
        "dry_run": counts.get("dry_run", 0),
        "exists": counts.get("exists", 0),
        "missing_source": counts.get("missing_source", 0),
        "copy_error": counts.get("copy_error", 0),
        "workers": workers,
        "manifest": str(args.manifest),
        "dry_run": args.dry_run,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Download AlphaFold DB mmCIF files by UniProt accession.

The public AlphaFold DB individual-file endpoint serves uncompressed ``.cif``
files. This utility stores them as ``sequence_md5.cif.gz`` so they live beside
the locally copied AFDB bulk structures.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


USER_AGENT = "P2PSigLip-AFDB-Downloader/1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download AFDB CIF files for UniProt IDs in a manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/embeds/manifests/strucs/afdb_matched_cif_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/embeds/strucs/afdb_matched"))
    parser.add_argument("--download-manifest", type=Path, default=Path("data/embeds/manifests/strucs/afdb_downloaded_cif_manifest.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("data/embeds/manifests/strucs/afdb_downloaded_cif_summary.json"))
    parser.add_argument("--model-version", default="v6")
    parser.add_argument("--status", action="append", default=["missing_source"], help="Input manifest status to download. Repeatable.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def afdb_cif_url(accession: str, model_version: str) -> str:
    return f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_{model_version}.cif"


def request_url(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def existing_file_ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def write_gzip_atomic(data: bytes, dest: Path) -> None:
    tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        if tmp.exists():
            tmp.unlink()
        with gzip.open(tmp, "wb") as handle:
            handle.write(data)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def download_one(task: dict[str, object], model_version: str, timeout: float, retries: int, overwrite: bool, dry_run: bool) -> dict[str, object]:
    accession = str(task["uniprot_id"]).upper()
    dest = Path(str(task["output_path"]))
    url = afdb_cif_url(accession, model_version)
    task["url"] = url
    task["status"] = ""
    task["bytes"] = 0
    task["error"] = ""

    if existing_file_ok(dest) and not overwrite:
        task["status"] = "exists"
        task["bytes"] = dest.stat().st_size
        return task

    if dry_run:
        task["status"] = "dry_run"
        return task

    last_error = ""
    for attempt in range(retries + 1):
        try:
            data = request_url(url, timeout)
            write_gzip_atomic(data, dest)
            task["status"] = "downloaded"
            task["bytes"] = dest.stat().st_size
            return task
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                task["status"] = "not_available"
                task["error"] = f"HTTP {exc.code}"
                return task
            last_error = f"HTTP {exc.code}"
        except Exception as exc:
            last_error = repr(exc)
        if attempt < retries:
            time.sleep(min(2.0 * (attempt + 1), 10.0))

    task["status"] = "download_error"
    task["error"] = last_error
    return task


def main() -> None:
    args = build_parser().parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    df = pd.read_csv(args.manifest)
    required = {"sequence_md5", "uniprot_id", "status"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{args.manifest} missing columns: {sorted(missing)}")

    statuses = {str(status) for status in args.status}
    candidates = df[df["status"].astype(str).isin(statuses)].copy()
    candidates = candidates.drop_duplicates(subset=["sequence_md5", "uniprot_id"])
    if args.limit is not None:
        candidates = candidates.head(args.limit)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.download_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        row_dict = row._asdict()
        sequence_md5 = str(row_dict["sequence_md5"])
        tasks.append(
            {
                "sequence_md5": sequence_md5,
                "uniprot_id": str(row_dict["uniprot_id"]).upper(),
                "length": row_dict.get("length", ""),
                "target_id": row_dict.get("target_id", sequence_md5),
                "output_path": str(args.out_dir / f"{sequence_md5}.cif.gz"),
                "input_status": row_dict.get("status", ""),
            }
        )

    counts: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    start = time.time()

    def record(result: dict[str, object], processed: int) -> None:
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1
        rows.append(result)
        if args.progress_every > 0 and processed % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "total": len(tasks),
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
            record(download_one(task, args.model_version, args.timeout, args.retries, args.overwrite, args.dry_run), processed)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(download_one, task, args.model_version, args.timeout, args.retries, args.overwrite, args.dry_run)
                for task in tasks
            ]
            for processed, future in enumerate(as_completed(futures), start=1):
                record(future.result(), processed)

    fieldnames = [
        "sequence_md5",
        "uniprot_id",
        "length",
        "target_id",
        "output_path",
        "input_status",
        "url",
        "status",
        "bytes",
        "error",
    ]
    with args.download_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "download_manifest": str(args.download_manifest),
        "candidate_rows": len(tasks),
        "model_version": args.model_version,
        "workers": workers,
        "dry_run": args.dry_run,
        **counts,
        "elapsed_sec": round(time.time() - start, 3),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

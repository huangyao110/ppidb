"""Extract Foldseek 3Di sequences from available project structures."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from Bio import SeqIO


MD5_RE = re.compile(r"[0-9a-f]{32}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch Foldseek 3Di extraction from sequence_structure_sources.tsv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-tsv", type=Path, default=Path("data/embeds/manifests/strucs/sequence_structure_sources.tsv"))
    p.add_argument("--sequence-csv", action="append", type=Path, default=None)
    p.add_argument("--foldseek-bin", type=Path, default=Path("external/foldseek/bin/foldseek"))
    p.add_argument("--out-fasta", type=Path, default=Path("data/embeds/manifests/strucs/structure_3di_full.fasta"))
    p.add_argument("--manifest-csv", type=Path, default=Path("data/embeds/manifests/strucs/structure_3di_full_manifest.csv"))
    p.add_argument("--work-dir", type=Path, default=Path("data/embeds/manifests/strucs/foldseek_3di_work"))
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def discover_sequence_csvs() -> list[Path]:
    return sorted(Path("data/datasets").glob("*hash_v1/sequences.csv"))


def load_sequences(paths: list[Path]) -> dict[str, str]:
    seqs: dict[str, str] = {}
    for path in paths:
        df = pd.read_csv(path)
        if "id" not in df.columns or "sequence" not in df.columns:
            continue
        for row in df[["id", "sequence"]].itertuples(index=False):
            seqs.setdefault(str(row.id), str(row.sequence).strip())
    return seqs


def read_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["sequence_md5", "status"], chunksize=100000):
        ok = chunk[chunk["status"].astype(str).eq("ok")]
        done.update(ok["sequence_md5"].astype(str))
    return done


def parse_fasta_by_md5(path: Path, aa_lengths: dict[str, int]) -> dict[str, tuple[str, str]]:
    by_id: dict[str, list[tuple[str, str]]] = {}
    for record in SeqIO.parse(path, "fasta"):
        header = str(record.description)
        match = MD5_RE.search(header)
        if match is None:
            continue
        md5 = match.group(0)
        by_id.setdefault(md5, []).append((str(record.seq).lower(), header))

    selected: dict[str, tuple[str, str]] = {}
    for md5, entries in by_id.items():
        target_len = aa_lengths.get(md5, 0)
        seq, header = min(entries, key=lambda item: (abs(len(item[0]) - target_len), -len(item[0])))
        selected[md5] = (seq, header)
    return selected


def append_fasta(path: Path, records: dict[str, str]) -> None:
    with path.open("a") as handle:
        for md5, seq in records.items():
            handle.write(f">{md5}\n{seq}\n")


def append_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    exists = path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence_md5",
                "status",
                "structure_path",
                "structure_source",
                "aa_length",
                "tdi_length",
                "foldseek_header",
                "error",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_foldseek(foldseek: Path, paths: list[Path], work_dir: Path, threads: int) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_tsv = work_dir / "structures.tsv"
    input_tsv.write_text("\n".join(str(p) for p in paths) + "\n")
    db = work_dir / "db"
    ss_fasta = work_dir / "ss.fasta"
    subprocess.run(
        [str(foldseek), "createdb", str(input_tsv), str(db), "--threads", str(threads), "-v", "2"],
        check=True,
    )
    subprocess.run([str(foldseek), "lndb", str(db) + "_h", str(db) + "_ss_h", "-v", "1"], check=True)
    subprocess.run([str(foldseek), "convert2fasta", str(db) + "_ss", str(ss_fasta), "-v", "1"], check=True)
    return ss_fasta


def main() -> None:
    args = build_parser()
    if not args.foldseek_bin.is_file():
        raise FileNotFoundError(args.foldseek_bin)
    if not args.source_tsv.is_file():
        raise FileNotFoundError(args.source_tsv)

    sequence_csvs = args.sequence_csv or discover_sequence_csvs()
    seqs = load_sequences(sequence_csvs)
    print(f"loaded sequences: {len(seqs):,} from {len(sequence_csvs)} files", flush=True)

    src = pd.read_csv(args.source_tsv, sep="\t")
    src["sequence_md5"] = src["sequence_md5"].astype(str)
    available = src[src["structure_status"].astype(str).eq("available")].copy()
    available = available[available["sequence_md5"].isin(seqs)]
    available = available[available["structure_path"].astype(str).map(lambda p: Path(p).is_file())]
    available = available.sort_values("sequence_md5").reset_index(drop=True)
    if args.limit is not None:
        available = available.head(args.limit)

    if args.resume:
        done = read_done(args.manifest_csv)
        available = available[~available["sequence_md5"].isin(done)].reset_index(drop=True)
        mode = "append"
    else:
        done = set()
        args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
        args.out_fasta.write_text("")
        if args.manifest_csv.exists():
            args.manifest_csv.unlink()
        mode = "overwrite"

    print(f"mode={mode}; to extract: {len(available):,}; already done: {len(done):,}", flush=True)
    aa_lengths = {md5: len(seq) for md5, seq in seqs.items()}

    total_ok = 0
    total_missing = 0
    for start in range(0, len(available), args.batch_size):
        batch = available.iloc[start : start + args.batch_size].copy()
        paths = [Path(p) for p in batch["structure_path"].astype(str)]
        print(f"[batch {start // args.batch_size + 1}] structures {start:,}-{start + len(batch):,}", flush=True)
        rows: list[dict[str, object]] = []
        try:
            ss_fasta = run_foldseek(args.foldseek_bin, paths, args.work_dir, args.threads)
            extracted = parse_fasta_by_md5(ss_fasta, aa_lengths)
        except subprocess.CalledProcessError as exc:
            extracted = {}
            batch_error = str(exc)
        else:
            batch_error = ""

        fasta_records: dict[str, str] = {}
        for row in batch.itertuples(index=False):
            row_dict = row._asdict()
            md5 = str(row_dict["sequence_md5"])
            if md5 in extracted:
                tdi, header = extracted[md5]
                fasta_records[md5] = tdi
                status = "ok"
                error = ""
                total_ok += 1
            else:
                tdi, header = "", ""
                status = "missing_3di"
                error = batch_error
                total_missing += 1
            rows.append(
                {
                    "sequence_md5": md5,
                    "status": status,
                    "structure_path": row_dict.get("structure_path", ""),
                    "structure_source": row_dict.get("structure_source", ""),
                    "aa_length": aa_lengths.get(md5, ""),
                    "tdi_length": len(tdi),
                    "foldseek_header": header,
                    "error": error,
                }
            )
        append_fasta(args.out_fasta, fasta_records)
        append_manifest(args.manifest_csv, rows)
        print(f"  ok={total_ok:,} missing={total_missing:,}", flush=True)

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    print(f"done -> {args.out_fasta}; manifest -> {args.manifest_csv}", flush=True)


if __name__ == "__main__":
    main()

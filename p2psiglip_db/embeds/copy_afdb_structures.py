"""Copy AFDB monomer structures for project sequences.

This utility links local sequence tables to AlphaFold DB monomer mmCIF files.
It first tries direct UniProt/AFDB accession matching from the sequence id, then
optionally streams the AFDB FASTA and performs exact sequence matching by MD5.

Typical usage from the repository root:

    python -m p2psiglip_db.embeds.copy_afdb_structures \
      --afdb-dir /media/zlab/ZhaoLab_27/afdb/monoer \
      --out-dir data/embeds/strucs

By default, inputs are discovered from ``data/datasets/*hash_v1/sequences.csv``.
Use ``--input`` one or more times to target explicit CSV/FASTA files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd


AFDB_ACCESSION_RE = re.compile(r"AF-([A-Z0-9]+)-F\d+", re.IGNORECASE)
UA_RE = re.compile(r"\bUA=([A-Z0-9]+)\b", re.IGNORECASE)
SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Target:
    target_id: str
    sequence: str
    sequence_md5: str
    source: str

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.target_id, self.sequence_md5)


def normalise_sequence(seq: str) -> str:
    """Return the canonical sequence string used for exact matching."""
    seq = re.sub(r"\s+", "", str(seq)).upper()
    return seq.replace("*", "")


def sequence_md5(seq: str) -> str:
    return hashlib.md5(normalise_sequence(seq).encode("utf-8")).hexdigest()


def safe_name(name: str) -> str:
    name = SAFE_CHARS_RE.sub("_", str(name).strip())
    return name.strip("._") or "sequence"


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    chunks: list[str] = []
    with open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, normalise_sequence("".join(chunks))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        yield header, normalise_sequence("".join(chunks))


def infer_pair_columns(df: pd.DataFrame) -> tuple[str, str, str | None, str | None] | None:
    seq1_candidates = ["seq1", "sequence1", "sequence_a", "seq_a", "a_sequence"]
    seq2_candidates = ["seq2", "sequence2", "sequence_b", "seq_b", "b_sequence"]
    id1_candidates = ["id1", "protein1", "protein_a", "a_id", "0"]
    id2_candidates = ["id2", "protein2", "protein_b", "b_id", "1"]

    seq1_col = next((c for c in seq1_candidates if c in df.columns), None)
    seq2_col = next((c for c in seq2_candidates if c in df.columns), None)
    if seq1_col is None or seq2_col is None:
        return None

    id1_col = next((c for c in id1_candidates if c in df.columns), None)
    id2_col = next((c for c in id2_candidates if c in df.columns), None)
    return seq1_col, seq2_col, id1_col, id2_col


def iter_sequence_table(path: Path) -> Iterator[tuple[str, str]]:
    suffixes = [s.lower() for s in path.suffixes]
    if any(s in {".fa", ".faa", ".fasta", ".fna"} for s in suffixes):
        for header, seq in iter_fasta(path):
            target_id = header.split()[0]
            yield target_id, seq
        return

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if {"id", "sequence"}.issubset(df.columns):
        for row in df[["id", "sequence"]].itertuples(index=False):
            yield str(row.id), str(row.sequence)
        return

    pair_cols = infer_pair_columns(df)
    if pair_cols is not None:
        seq1_col, seq2_col, id1_col, id2_col = pair_cols
        for _, row in df.iterrows():
            seq1 = normalise_sequence(row[seq1_col])
            seq2 = normalise_sequence(row[seq2_col])
            id1 = str(row[id1_col]) if id1_col is not None else sequence_md5(seq1)
            id2 = str(row[id2_col]) if id2_col is not None else sequence_md5(seq2)
            yield id1, seq1
            yield id2, seq2
        return

    if len(df.columns) >= 2:
        id_col, seq_col = df.columns[:2]
        for row in df[[id_col, seq_col]].itertuples(index=False):
            yield str(row[0]), str(row[1])
        return

    raise ValueError(f"{path}: expected id/sequence CSV, pair CSV, or FASTA")


def discover_inputs(datasets_root: Path, dataset_glob: str) -> list[Path]:
    return sorted(p for p in datasets_root.glob(dataset_glob) if p.is_file())


def load_targets(paths: Iterable[Path], max_targets: int | None = None) -> list[Target]:
    targets: list[Target] = []
    seen: set[tuple[str, str]] = set()
    conflicting_ids: dict[str, set[str]] = defaultdict(set)

    for path in paths:
        for target_id, seq in iter_sequence_table(path):
            seq = normalise_sequence(seq)
            if not target_id or not seq:
                continue
            md5 = sequence_md5(seq)
            key = (str(target_id), md5)
            if key in seen:
                continue
            seen.add(key)
            conflicting_ids[str(target_id)].add(md5)
            targets.append(Target(str(target_id), seq, md5, str(path)))
            if max_targets is not None and len(targets) >= max_targets:
                break
        if max_targets is not None and len(targets) >= max_targets:
            break

    conflicts = {k: v for k, v in conflicting_ids.items() if len(v) > 1}
    if conflicts:
        print(
            f"warning: {len(conflicts):,} ids map to multiple sequences; "
            "manifest rows keep source+id+sequence_md5 distinct",
            file=sys.stderr,
            flush=True,
        )

    return targets


def parse_afdb_accession(text: str) -> str | None:
    text = str(text).strip()
    if not text:
        return None
    match = AFDB_ACCESSION_RE.search(text)
    if match:
        return match.group(1).upper()
    match = UA_RE.search(text)
    if match:
        return match.group(1).upper()

    token = text.split()[0]
    if "|" in token:
        parts = [p for p in token.split("|") if p]
        for part in parts:
            if re.fullmatch(r"[A-Z0-9]{6,10}(?:-\d+)?", part, re.IGNORECASE):
                token = part
                break
    token = token.strip()
    if token.startswith("AF-"):
        match = AFDB_ACCESSION_RE.search(token)
        return match.group(1).upper() if match else None
    if re.fullmatch(r"[A-Z0-9]{6,10}(?:-\d+)?", token, re.IGNORECASE):
        return token.split("-")[0].upper()
    return None


def structure_path_for_accession(afdb_dir: Path, accession: str, model_version: str) -> Path | None:
    accession = accession.upper()
    exact_name = f"AF-{accession}-F1-model_{model_version}.cif.gz"
    candidates = [
        afdb_dir / exact_name,
        afdb_dir / "cifgz" / exact_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    patterns = [
        f"AF-{accession}-F*-model_{model_version}.cif.gz",
        f"AF-{accession}-F*-model_*.cif.gz",
    ]
    for root in [afdb_dir, afdb_dir / "cifgz"]:
        if not root.is_dir():
            continue
        for pattern in patterns:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    return None


def destination_for(
    target: Target,
    accession: str,
    source_path: Path,
    out_dir: Path,
    name_mode: str,
    decompress: bool,
) -> Path:
    if name_mode == "id":
        stem = safe_name(target.target_id)
    elif name_mode == "accession":
        stem = safe_name(accession)
    elif name_mode == "sequence_md5":
        stem = target.sequence_md5
    else:
        raise ValueError(f"unknown name mode: {name_mode}")

    if decompress:
        suffix = ".cif"
    elif "".join(source_path.suffixes[-2:]) == ".cif.gz":
        suffix = ".cif.gz"
    else:
        suffix = source_path.suffix
    return out_dir / f"{stem}{suffix}"


def copy_structure(source: Path, dest: Path, overwrite: bool, decompress: bool, dry_run: bool) -> str:
    if dest.exists() and not overwrite:
        return "exists"
    if dry_run:
        return "dry_run"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if decompress:
        with gzip.open(source, "rb") as src_handle, dest.open("wb") as dst_handle:
            shutil.copyfileobj(src_handle, dst_handle)
    else:
        shutil.copy2(source, dest)
    return "copied"


def manifest_row(
    target: Target,
    match_type: str,
    accession: str,
    source_structure: Path,
    output_path: Path,
    status: str,
    note: str = "",
) -> dict[str, str | int]:
    return {
        "target_id": target.target_id,
        "target_sequence_md5": target.sequence_md5,
        "target_length": target.length,
        "target_source": target.source,
        "match_type": match_type,
        "afdb_accession": accession,
        "source_structure": str(source_structure),
        "output_path": str(output_path),
        "status": status,
        "note": note,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy AFDB monomer structures for local project sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        action="append",
        type=Path,
        help="Input id/sequence CSV, pair CSV with seq1/seq2, or FASTA. Can be repeated.",
    )
    parser.add_argument("--datasets-root", type=Path, default=Path("data/datasets"))
    parser.add_argument(
        "--dataset-glob",
        default="*hash_v1/sequences.csv",
        help="Used only when --input is omitted.",
    )
    parser.add_argument("--afdb-dir", type=Path, default=Path("/media/zlab/ZhaoLab_27/afdb/monoer"))
    parser.add_argument(
        "--afdb-fasta",
        type=Path,
        default=None,
        help="AFDB sequence FASTA. Defaults to <afdb-dir>/sequences.fasta.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/embeds/strucs"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/embeds/manifests/strucs/afdb_monoer_structures_manifest.csv"),
    )
    parser.add_argument(
        "--missing",
        type=Path,
        default=Path("data/embeds/manifests/strucs/afdb_monoer_structures_missing.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/embeds/manifests/strucs/afdb_monoer_structures_summary.json"),
    )
    parser.add_argument("--model-version", default="v6")
    parser.add_argument(
        "--name-mode",
        choices=["sequence_md5", "id", "accession"],
        default="sequence_md5",
        help="How copied structures are named in --out-dir.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--decompress", action="store_true", help="Write .cif instead of copying .cif.gz")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-id-match", action="store_true", help="Disable direct id/accession matching.")
    parser.add_argument(
        "--no-sequence-match",
        action="store_true",
        help="Disable sequence-MD5 matching through AFDB FASTA.",
    )
    parser.add_argument("--max-targets", type=int, default=None, help="Debug limit on loaded target rows.")
    parser.add_argument(
        "--max-fasta-records",
        type=int,
        default=None,
        help="Debug limit on scanned AFDB FASTA records.",
    )
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    afdb_fasta = args.afdb_fasta or (args.afdb_dir / "sequences.fasta")

    inputs = args.input
    if not inputs:
        inputs = discover_inputs(args.datasets_root, args.dataset_glob)
        print(
            "auto-discovered inputs:\n  " + "\n  ".join(str(p) for p in inputs),
            flush=True,
        )
    if not inputs:
        raise SystemExit("no input sequence files found; pass --input or adjust --dataset-glob")

    targets = load_targets(inputs, max_targets=args.max_targets)
    if not targets:
        raise SystemExit("no target sequences loaded")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loaded targets: {len(targets):,}", flush=True)
    print(f"unique target sequences: {len({t.sequence_md5 for t in targets}):,}", flush=True)
    print(f"output directory: {args.out_dir}", flush=True)

    copied_rows: list[dict[str, str | int]] = []
    done_keys: set[tuple[str, str, str]] = set()
    copied_paths: set[Path] = set()

    def record_copy(target: Target, accession: str, source: Path, match_type: str, note: str = "") -> None:
        dest = destination_for(
            target=target,
            accession=accession,
            source_path=source,
            out_dir=args.out_dir,
            name_mode=args.name_mode,
            decompress=args.decompress,
        )
        if dest in copied_paths and dest.exists() and not args.overwrite:
            status = "exists"
        else:
            status = copy_structure(source, dest, args.overwrite, args.decompress, args.dry_run)
            copied_paths.add(dest)
        copied_rows.append(manifest_row(target, match_type, accession, source, dest, status, note=note))
        done_keys.add(target.key)

    if not args.no_id_match:
        for target in targets:
            accession = parse_afdb_accession(target.target_id)
            if accession is None:
                continue
            source = structure_path_for_accession(args.afdb_dir, accession, args.model_version)
            if source is None:
                continue
            record_copy(target, accession, source, "id")
        print(f"matched by id/accession: {len(done_keys):,}", flush=True)

    if not args.no_sequence_match:
        if not afdb_fasta.is_file():
            raise FileNotFoundError(f"AFDB FASTA not found: {afdb_fasta}")

        hash_to_targets: dict[str, list[Target]] = defaultdict(list)
        for target in targets:
            if target.key not in done_keys:
                hash_to_targets[target.sequence_md5].append(target)
        needed_hashes = set(hash_to_targets)
        needed_lengths = {t.length for target_list in hash_to_targets.values() for t in target_list}

        print(f"sequence-match remaining sequences: {len(needed_hashes):,}", flush=True)
        if needed_hashes:
            start = time.time()
            scanned = 0
            for header, seq in iter_fasta(afdb_fasta):
                if args.max_fasta_records is not None and scanned >= args.max_fasta_records:
                    break
                scanned += 1
                if len(seq) in needed_lengths:
                    md5 = sequence_md5(seq)
                    if md5 in needed_hashes:
                        accession = parse_afdb_accession(header)
                        if accession is not None:
                            source = structure_path_for_accession(args.afdb_dir, accession, args.model_version)
                            if source is not None:
                                for target in hash_to_targets[md5]:
                                    if target.key not in done_keys:
                                        record_copy(target, accession, source, "sequence")
                                needed_hashes.discard(md5)
                                if not needed_hashes:
                                    break
                if args.progress_every and scanned % args.progress_every == 0:
                    elapsed = time.time() - start
                    print(
                        f"scanned AFDB FASTA records: {scanned:,}; "
                        f"remaining sequence hashes: {len(needed_hashes):,}; "
                        f"elapsed: {elapsed / 60:.1f} min",
                        flush=True,
                    )
            print(
                f"finished AFDB FASTA scan: scanned={scanned:,}, "
                f"remaining sequence hashes={len(needed_hashes):,}",
                flush=True,
            )

    missing_rows: list[dict[str, str | int]] = []
    for target in targets:
        if target.key in done_keys:
            continue
        missing_rows.append(
            {
                "target_id": target.target_id,
                "target_sequence_md5": target.sequence_md5,
                "target_length": target.length,
                "target_source": target.source,
                "note": "no AFDB structure matched by enabled methods",
            }
        )

    manifest_fields = [
        "target_id",
        "target_sequence_md5",
        "target_length",
        "target_source",
        "match_type",
        "afdb_accession",
        "source_structure",
        "output_path",
        "status",
        "note",
    ]
    missing_fields = [
        "target_id",
        "target_sequence_md5",
        "target_length",
        "target_source",
        "note",
    ]
    write_csv(args.manifest, copied_rows, manifest_fields)
    write_csv(args.missing, missing_rows, missing_fields)

    summary = {
        "inputs": [str(p) for p in inputs],
        "afdb_dir": str(args.afdb_dir),
        "afdb_fasta": str(afdb_fasta),
        "out_dir": str(args.out_dir),
        "targets": len(targets),
        "unique_target_sequences": len({t.sequence_md5 for t in targets}),
        "matched_targets": len(done_keys),
        "missing_targets": len(missing_rows),
        "manifest": str(args.manifest),
        "missing": str(args.missing),
        "name_mode": args.name_mode,
        "dry_run": args.dry_run,
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

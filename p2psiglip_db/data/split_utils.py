"""Shared helpers for hash-ID dataset splits.

All canonical split datasets should use the same contract:

- pair CSV first columns are ``ID_1,ID_2,label``;
- IDs are ``MD5(normalized amino-acid sequence)``;
- ``sequences.csv`` contains ``id,sequence`` for every pair endpoint;
- ``id_map.csv`` maps source IDs to sequence MD5 IDs;
- ``SUMMARY.json`` records sources, pair counts, and C3 settings when used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MMSEQS = (
    "/home/zlab/miniconda3/envs/lucafold/lib/python3.11/"
    "site-packages/pymmseqs/bin/mmseqs"
)
HEX32 = re.compile(r"^[0-9a-f]{32}$")
PAIR_COLUMN_CANDIDATES = (
    ("ID_1", "ID_2"),
    ("fpid_1", "fpid_2"),
    ("FPid_1", "FPid_2"),
    ("id_1", "id_2"),
)
CANONICAL_PAIR_FRONT = ["ID_1", "ID_2", "label"]


def normalize_sequence(sequence: object) -> str:
    return "".join(str(sequence).upper().split())


def sequence_md5(sequence: object) -> str:
    return hashlib.md5(normalize_sequence(sequence).encode("utf-8")).hexdigest()


def parse_csv_values(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {part.strip() for part in value.split(",") if part.strip()}
    return items or None


def label_counts_to_json(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda item: str(item[0]))}


def pair_columns(columns: Iterable[str]) -> tuple[str, str]:
    colset = set(columns)
    for left, right in PAIR_COLUMN_CANDIDATES:
        if left in colset and right in colset:
            return left, right
    raise SystemExit(
        "expected pair columns "
        + " or ".join(f"{left}/{right}" for left, right in PAIR_COLUMN_CANDIDATES)
        + f"; got {list(columns)}"
    )


def pair_key(left: pd.Series, right: pd.Series) -> pd.Series:
    left = left.astype(str)
    right = right.astype(str)
    return left.where(left <= right, right) + "|" + right.where(left <= right, left)


def pair_ids(path: Path, chunk_size: int = 500_000) -> set[str]:
    ids: set[str] = set()
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        left, right = pair_columns(chunk.columns)
        ids.update(chunk[left].astype(str))
        ids.update(chunk[right].astype(str))
    return ids


def pair_summary(path: Path, source: Path | str | None = None, chunk_size: int = 500_000) -> dict[str, object]:
    rows = 0
    labels: Counter[int] = Counter()
    unique_ids: set[str] = set()
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        left, right = pair_columns(chunk.columns)
        rows += len(chunk)
        if "label" in chunk.columns:
            labels.update(chunk["label"].astype(int).tolist())
        unique_ids.update(chunk[left].astype(str))
        unique_ids.update(chunk[right].astype(str))
    return {
        "source": None if source is None else str(source),
        "output": str(path),
        "rows": int(rows),
        "label_counts": label_counts_to_json(labels),
        "unique_pair_ids": int(len(unique_ids)),
    }


def read_sequence_source(path: Path, source_dataset: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing sequence file: {path}")
    df = pd.read_csv(path)
    if "id" not in df.columns or "sequence" not in df.columns:
        raise SystemExit(f"{path}: expected id and sequence columns")
    out = df[["id", "sequence"]].copy()
    out = out.dropna(subset=["id", "sequence"])
    out["source_id"] = out["id"].astype(str)
    out["sequence"] = out["sequence"].map(normalize_sequence)
    out = out[out["sequence"] != ""].copy()
    out["sequence_md5"] = out["sequence"].map(sequence_md5)
    out["source_dataset"] = source_dataset
    return out[["source_dataset", "source_id", "sequence_md5", "sequence"]]


def merge_sequence_sources(sources: Iterable[tuple[Path, str]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    frames = [read_sequence_source(path, source) for path, source in sources]
    id_map = pd.concat(frames, ignore_index=True)
    conflicting_ids = id_map.groupby("source_id")["sequence_md5"].nunique()
    conflicting_ids = conflicting_ids.loc[conflicting_ids > 1]
    if not conflicting_ids.empty:
        examples = [str(idx) for idx in conflicting_ids.head(10).index.tolist()]
        raise SystemExit(f"source IDs with multiple sequences: {examples}")

    unique_sequences = (
        id_map[["sequence_md5", "sequence"]]
        .drop_duplicates("sequence_md5")
        .rename(columns={"sequence_md5": "id"})
        .sort_values("id")
        .reset_index(drop=True)
    )
    id_map = (
        id_map.drop_duplicates(["source_dataset", "source_id", "sequence_md5"])
        .sort_values(["source_dataset", "source_id"])
        .reset_index(drop=True)
    )
    id_to_hash = dict(zip(id_map["source_id"], id_map["sequence_md5"]))
    return id_map, unique_sequences, id_to_hash


def load_hash_sequences(path: Path, keep_ids: set[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or "sequence" not in df.columns:
        raise SystemExit(f"{path}: expected id and sequence columns")
    df = df[["id", "sequence"]].copy()
    df["id"] = df["id"].astype(str)
    df["sequence"] = df["sequence"].map(normalize_sequence)
    df = df.drop_duplicates("id")
    if keep_ids is not None:
        df = df[df["id"].isin(keep_ids)].copy()
        missing = keep_ids - set(df["id"])
        if missing:
            print(f"WARNING: {len(missing):,} requested IDs absent from sequences CSV")
    return df[df["sequence"] != ""].copy()


def write_fasta(seqs: pd.DataFrame, path: Path, id_col: str = "id", sequence_col: str = "sequence") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in seqs[[id_col, sequence_col]].itertuples(index=False):
            handle.write(f">{row[0]}\n{row[1]}\n")


def resolve_mmseqs(user_value: str | None = None) -> str:
    candidates = [user_value, shutil.which("mmseqs"), DEFAULT_MMSEQS]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise SystemExit("mmseqs not found; pass --mmseqs or install pymmseqs/mmseqs2")


def run_mmseqs_easy_cluster(
    fasta: Path,
    out_dir: Path,
    identity: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    mmseqs_bin: str,
    reuse_cluster: bool = False,
) -> Path:
    prefix = out_dir / "mmseqs_cluster"
    cluster_tsv = Path(f"{prefix}_cluster.tsv")
    if reuse_cluster and cluster_tsv.exists():
        return cluster_tsv
    tmp = out_dir / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    cmd = [
        mmseqs_bin,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(tmp),
        "--min-seq-id",
        str(identity),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
        "-v",
        "1",
    ]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if not cluster_tsv.exists():
        raise SystemExit(f"missing expected cluster TSV: {cluster_tsv}")
    shutil.rmtree(tmp, ignore_errors=True)
    return cluster_tsv


def parse_cluster_tsv(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rep, member = line.rstrip("\n").split("\t")
            rows.append((member, rep))
    return pd.DataFrame(rows, columns=["id", "cluster"]).drop_duplicates("id")


def split_csvs(collection_dir: Path) -> list[Path]:
    preferred = ["train.csv", "val.csv", "test.csv"]
    paths = [collection_dir / name for name in preferred if (collection_dir / name).exists()]
    paths.extend(sorted(collection_dir.glob("test_*.csv")))
    paths.extend(
        path
        for path in sorted(collection_dir.glob("*.csv"))
        if path.name not in {"sequences.csv", "id_map.csv", "fpid_to_sequence_md5.csv"}
    )
    out: list[Path] = []
    seen = set()
    for path in paths:
        if path.name not in seen:
            seen.add(path.name)
            out.append(path)
    return out


def validate_hash_sequences(path: Path) -> tuple[pd.DataFrame, set[str]]:
    if not path.exists():
        raise SystemExit(f"missing sequences.csv: {path}")
    seq = pd.read_csv(path)
    required = {"id", "sequence"}
    if not required.issubset(seq.columns):
        raise SystemExit(f"{path}: expected columns {sorted(required)}")
    seq = seq[["id", "sequence"]].copy()
    seq["id"] = seq["id"].astype(str)
    seq["sequence"] = seq["sequence"].map(normalize_sequence)
    duplicate_ids = int(seq["id"].duplicated().sum())
    if duplicate_ids:
        raise SystemExit(f"{path}: duplicate sequence IDs: {duplicate_ids}")
    bad_hashes = seq.loc[~seq["id"].map(lambda value: bool(HEX32.match(value)))]
    if len(bad_hashes):
        raise SystemExit(f"{path}: non-MD5 IDs found, first={bad_hashes['id'].head().tolist()}")
    mismatch = seq.loc[seq["id"] != seq["sequence"].map(sequence_md5)]
    if len(mismatch):
        raise SystemExit(f"{path}: IDs do not match sequence MD5, first={mismatch['id'].head().tolist()}")
    return seq, set(seq["id"])


def validate_hash_pair_csv(path: Path, sequence_ids: set[str], chunk_size: int = 500_000) -> dict[str, object]:
    rows = 0
    unique_pair_ids: set[str] = set()
    label_counts: Counter[int] = Counter()
    saw_chunk = False
    for df in pd.read_csv(path, chunksize=chunk_size):
        saw_chunk = True
        front = list(df.columns[:3])
        if front != CANONICAL_PAIR_FRONT:
            raise SystemExit(f"{path}: first columns must be ID_1,ID_2,label; got {front}")
        df["ID_1"] = df["ID_1"].astype(str)
        df["ID_2"] = df["ID_2"].astype(str)
        chunk_ids = set(df["ID_1"]) | set(df["ID_2"])
        bad_ids = [pid for pid in chunk_ids if not HEX32.match(pid)]
        if bad_ids:
            raise SystemExit(f"{path}: non-MD5 pair IDs, first={bad_ids[:5]}")
        missing_ids = sorted(chunk_ids - sequence_ids)
        if missing_ids:
            raise SystemExit(f"{path}: {len(missing_ids)} pair IDs absent from sequences.csv, first={missing_ids[:5]}")
        rows += len(df)
        unique_pair_ids.update(chunk_ids)
        label_counts.update(df["label"].astype(int).tolist())
    if not saw_chunk:
        raise SystemExit(f"{path}: no rows found")
    return {
        "rows": int(rows),
        "label_counts": label_counts_to_json(label_counts),
        "unique_pair_ids": int(len(unique_pair_ids)),
    }


def write_hash_dataset_readme(out_dir: Path, summary: dict[str, object]) -> None:
    tier = summary.get("tier_include")
    notes = []
    if tier:
        notes.append(f"- Candidate tier filter: `{tier}`.")
    if summary.get("cluster_map_csv"):
        notes.append(f"- Homology cluster map: `{summary['cluster_map_csv']}`.")
    if summary.get("test_csv"):
        notes.append(f"- Fixed test set: `{summary['test_csv']}`.")
    notes_text = "\n".join(notes) if notes else "- No additional generation notes recorded."

    text = f"""# {out_dir.name}

Canonical split collection with sequence-MD5 protein IDs.

Pair CSV contract:

- `ID_1,ID_2,label` are the first three columns.
- `ID_1` and `ID_2` are MD5 hashes of normalized amino-acid sequences.
- `sequences.csv` contains every endpoint used by train/val/test pair CSVs.
- `id_map.csv` maps source IDs to sequence-MD5 IDs.
- Embeddings are read from `data/embeds/{{plm}}/{{ID}}.npy`.

Primary files:

- `train.csv`
- `val.csv` when present
- `test.csv`
- `sequences.csv`
- `id_map.csv`
- `SUMMARY.json`

Generation notes:

{notes_text}

C3 rule when `cluster_map_csv` is present: any candidate train pair is removed
if either endpoint belongs to an MMseqs cluster touched by a fixed-test protein.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

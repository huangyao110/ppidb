"""
Unify host-virus data from data/external/hp_corpus and ViraHinter.

Outputs under data/external/host_corpus by default:
  - pairs_all.parquet             full standardized table
  - pairs_with_sequences.parquet  rows with both host/pathogen sequences
  - pairs_index.csv               lightweight table without sequence columns
  - sequences.csv                 unique protein sequences by role/id/md5
  - SUMMARY.json                  counts and coverage
  - README.md                     schema notes

Run:
  python p2psiglip_db/data/prepare_host_pathogen_corpus.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HP_ROOT = ROOT / "data" / "external" / "hp_corpus"
DEFAULT_VIRAHINTER_ROOT = ROOT / "data" / "external" / "virahinter" / "raw"
DEFAULT_OUT = ROOT / "data" / "external" / "host_corpus"

PAIR_COLUMNS = [
    "record_uid",
    "dataset",
    "source_table",
    "split",
    "source_database",
    "confidence",
    "label",
    "host_taxon",
    "pathogen_taxon",
    "host_id",
    "pathogen_id",
    "host_sequence",
    "pathogen_sequence",
    "host_len",
    "pathogen_len",
    "pathogen_family",
    "original_pair_id",
]

INDEX_COLUMNS = [c for c in PAIR_COLUMNS if c not in {"host_sequence", "pathogen_sequence"}]


def clean_sequence(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    seq = "".join(str(value).split()).upper().rstrip("*")
    return seq or None


def clean_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def sequence_len(value: object) -> int | None:
    return len(value) if isinstance(value, str) and value else None


def confidence_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value).strip() or None


def md5_or_none(seq: str | None) -> str | None:
    return hashlib.md5(seq.encode()).hexdigest() if seq else None


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def source_name_from_path(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_high_confidence", "_medium_confidence"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def confidence_from_path(path: Path) -> str | None:
    name = path.stem.lower()
    if "high_confidence" in name or "highconfidence" in name:
        return "high"
    if "medium_confidence" in name or "mediumconfidence" in name:
        return "medium"
    return None


def load_hvidb(hp_root: Path) -> pd.DataFrame:
    pairs_path = hp_root / "hvidb" / "pairs_clean.csv"
    seq_path = hp_root / "hvidb" / "sequences.csv"
    if not pairs_path.exists():
        return pd.DataFrame(columns=PAIR_COLUMNS)

    pairs = pd.read_csv(pairs_path)
    seqs = pd.read_csv(seq_path) if seq_path.exists() else pd.DataFrame(columns=["id", "sequence"])
    seq_map = dict(zip(seqs["id"].astype(str), seqs["sequence"].map(clean_sequence)))

    df = pd.DataFrame(
        {
            "dataset": "hvidb",
            "source_table": rel(pairs_path),
            "split": "all",
            "source_database": "HVIDB",
            "confidence": pd.NA,
            "label": 1,
            "host_taxon": "Homo sapiens",
            "pathogen_taxon": "virus",
            "host_id": pairs["host"].map(clean_text),
            "pathogen_id": pairs["virus"].map(clean_text),
            "pathogen_family": pairs.get("family", pd.Series(pd.NA, index=pairs.index)).map(clean_text),
        }
    )
    df["host_sequence"] = df["host_id"].map(seq_map)
    df["pathogen_sequence"] = df["pathogen_id"].map(seq_map)
    df["host_len"] = df["host_sequence"].map(sequence_len)
    df["pathogen_len"] = df["pathogen_sequence"].map(sequence_len)
    df["original_pair_id"] = df["host_id"].fillna("") + "_" + df["pathogen_id"].fillna("")
    df["record_uid"] = (
        "hvidb:"
        + df["original_pair_id"]
        + ":"
        + df.index.astype(str)
    )
    return df[PAIR_COLUMNS]


def split_name(path: Path, split_root: Path) -> str:
    rel_path = path.relative_to(split_root)
    if rel_path.parent.name == "highconfidence_data":
        return f"highconfidence_{path.stem}"
    return path.stem


def load_virahinter_split_file(path: Path, split_root: Path) -> pd.DataFrame:
    src = pd.read_parquet(path)
    label_col = "label" if "label" in src.columns else "REG_LABEL" if "REG_LABEL" in src.columns else None
    split = split_name(path, split_root)

    df = pd.DataFrame(
        {
            "dataset": "virahinter",
            "source_table": rel(path),
            "split": split,
            "source_database": "ViraHinter",
            "confidence": src.get("confidence", pd.Series(pd.NA, index=src.index)).map(confidence_value),
            "label": src[label_col] if label_col else 1,
            "host_taxon": "Homo sapiens",
            "pathogen_taxon": "virus",
            "host_id": src["UNIPROT_HUMAN"].map(clean_text),
            "pathogen_id": src["UNIPROT_VIRUS"].map(clean_text),
            "host_sequence": src["SEQUENCE_HUMAN"].map(clean_sequence),
            "pathogen_sequence": src["SEQUENCE_VIRUS"].map(clean_sequence),
            "pathogen_family": pd.NA,
            "original_pair_id": src.get("id", src.get("ID", pd.Series(pd.NA, index=src.index))).map(clean_text),
        }
    )
    df["host_len"] = df["host_sequence"].map(sequence_len)
    df["pathogen_len"] = df["pathogen_sequence"].map(sequence_len)
    df["record_uid"] = "virahinter:split:" + split + ":" + src.index.astype(str)
    return df[PAIR_COLUMNS]


def load_virahinter_raw_file(path: Path) -> pd.DataFrame:
    src = pd.read_parquet(path)
    confidence = src["confidence"].map(confidence_value) if "confidence" in src.columns else confidence_from_path(path)
    source_database = (
        src["Source Database"].map(clean_text)
        if "Source Database" in src.columns
        else src["source"].map(clean_text)
        if "source" in src.columns
        else source_name_from_path(path).upper()
    )

    pair_id = src.get("id", src.get("pair_key", src.get("interaction_id", src.get("ID"))))
    if pair_id is None:
        pair_id = src["UNIPROT_HUMAN"].astype(str) + "_" + src["UNIPROT_VIRUS"].astype(str)

    df = pd.DataFrame(
        {
            "dataset": "virahinter",
            "source_table": rel(path),
            "split": "raw",
            "source_database": source_database,
            "confidence": confidence,
            "label": 1,
            "host_taxon": "Homo sapiens",
            "pathogen_taxon": "virus",
            "host_id": src["UNIPROT_HUMAN"].map(clean_text),
            "pathogen_id": src["UNIPROT_VIRUS"].map(clean_text),
            "host_sequence": src["SEQUENCE_HUMAN"].map(clean_sequence),
            "pathogen_sequence": src["SEQUENCE_VIRUS"].map(clean_sequence),
            "host_len": src["SEQUENCE_HUMAN"].map(lambda seq: sequence_len(clean_sequence(seq))),
            "pathogen_len": src["SEQUENCE_VIRUS"].map(lambda seq: sequence_len(clean_sequence(seq))),
            "pathogen_family": pd.NA,
            "original_pair_id": pair_id.map(clean_text),
        }
    )
    df["record_uid"] = "virahinter:raw:" + path.stem + ":" + src.index.astype(str)
    return df[PAIR_COLUMNS]


def load_virahinter(virahinter_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    raw_root = virahinter_root / "data" / "raw"
    for path in sorted(raw_root.glob("*.parquet")):
        frames.append(load_virahinter_raw_file(path))

    split_root = virahinter_root / "data" / "split"
    for path in sorted(split_root.rglob("*.parquet")):
        frames.append(load_virahinter_split_file(path, split_root))

    if not frames:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    return pd.concat(frames, ignore_index=True)[PAIR_COLUMNS]


def build_sequence_table(pairs: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for role, id_col, seq_col in [
        ("host", "host_id", "host_sequence"),
        ("pathogen", "pathogen_id", "pathogen_sequence"),
    ]:
        part = pairs[[id_col, seq_col, "dataset"]].copy()
        part.columns = ["protein_id", "sequence", "dataset"]
        part["role"] = role
        parts.append(part)

    seqs = pd.concat(parts, ignore_index=True)
    seqs = seqs.dropna(subset=["protein_id", "sequence"])
    seqs["sequence_md5"] = seqs["sequence"].map(md5_or_none)
    seqs["length"] = seqs["sequence"].map(sequence_len)
    grouped = (
        seqs.groupby(["role", "protein_id", "sequence_md5", "sequence", "length"], dropna=False)["dataset"]
        .agg(lambda values: ";".join(sorted(set(map(str, values)))))
        .reset_index()
        .rename(columns={"dataset": "datasets"})
    )
    return grouped[["role", "protein_id", "sequence_md5", "length", "datasets", "sequence"]]


def nested_counts(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    if df.empty:
        return []
    return (
        df.groupby(columns, dropna=False)
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )


def write_readme(out: Path) -> None:
    text = """# Host Corpus

This directory contains a normalized host-virus corpus built from `data/external/hp_corpus` and `data/external/virahinter`.

## Files

- `pairs_all.parquet`: all standardized rows, including raw ViraHinter records and prepared splits.
- `pairs_with_sequences.parquet`: subset of rows where both host and pathogen sequences are present.
- `pairs_index.csv`: lightweight view without sequence columns for quick inspection.
- `sequences.csv`: de-duplicated protein sequences by role, identifier, and sequence MD5.
- `SUMMARY.json`: row counts, label counts, and sequence coverage.

## Pair Schema

Common fields are `dataset`, `source_table`, `split`, `source_database`, `confidence`, `label`,
`host_id`, `pathogen_id`, `host_sequence`, `pathogen_sequence`, `host_len`, `pathogen_len`,
`pathogen_family`, `has_host_sequence`, and `has_pathogen_sequence`. HVIDB pairs are positive-only
(`label=1`). ViraHinter split files keep their provided `label` or `REG_LABEL`; raw ViraHinter
files are positive evidence rows.

## Regeneration

Run from the repository root:

```bash
python p2psiglip_db/data/prepare_host_pathogen_corpus.py
```
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def write_outputs(pairs: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pairs = pairs[PAIR_COLUMNS].copy()
    pairs["has_host_sequence"] = pairs["host_sequence"].notna()
    pairs["has_pathogen_sequence"] = pairs["pathogen_sequence"].notna()

    with_sequences = pairs[pairs["has_host_sequence"] & pairs["has_pathogen_sequence"]].copy()
    sequences = build_sequence_table(with_sequences)
    index = pairs[INDEX_COLUMNS + ["has_host_sequence", "has_pathogen_sequence"]].copy()

    pairs.to_parquet(out / "pairs_all.parquet", index=False)
    with_sequences.to_parquet(out / "pairs_with_sequences.parquet", index=False)
    index.to_csv(out / "pairs_index.csv", index=False)
    sequences.to_csv(out / "sequences.csv", index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows_total": int(len(pairs)),
        "rows_with_sequences": int(len(with_sequences)),
        "rows_missing_host_sequence": int((~pairs["has_host_sequence"]).sum()),
        "rows_missing_pathogen_sequence": int((~pairs["has_pathogen_sequence"]).sum()),
        "unique_host_ids": int(pairs["host_id"].nunique(dropna=True)),
        "unique_pathogen_ids": int(pairs["pathogen_id"].nunique(dropna=True)),
        "unique_sequences": int(len(sequences)),
        "by_dataset": nested_counts(pairs, ["dataset"]),
        "by_dataset_split_label": nested_counts(pairs, ["dataset", "split", "label"]),
        "by_dataset_source": nested_counts(pairs, ["dataset", "source_database"]),
        "outputs": {
            "pairs_all": rel(out / "pairs_all.parquet"),
            "pairs_with_sequences": rel(out / "pairs_with_sequences.parquet"),
            "pairs_index": rel(out / "pairs_index.csv"),
            "sequences": rel(out / "sequences.csv"),
        },
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unify host-virus external datasets.")
    parser.add_argument("--hp-root", type=Path, default=DEFAULT_HP_ROOT)
    parser.add_argument("--virahinter-root", type=Path, default=DEFAULT_VIRAHINTER_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = [
        load_hvidb(args.hp_root),
        load_virahinter(args.virahinter_root),
    ]
    pairs = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if pairs.empty:
        raise SystemExit("No host-virus records found.")

    write_outputs(pairs, args.out)
    print(f"wrote {len(pairs):,} rows to {args.out}")


if __name__ == "__main__":
    main()

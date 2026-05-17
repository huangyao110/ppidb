"""Validate the hard API contract for ``data/merged`` tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from p2psiglip_db.data.merged_contract import (
    BOOLEAN_TEXT,
    EVIDENCE_TAGS,
    EVIDENCE_TYPES,
    EXPECTED_SNAPSHOT,
    FPID_RE,
    INTERACTIONS_COLUMNS,
    INTERACTIONS_FILE,
    LABELS,
    PAIRS_COLUMNS,
    PAIRS_FILE,
    PPI_SOURCES,
    PPI_TIERS,
    PROTEIN_MD5_RE,
    PROTEINS_COLUMNS,
    PROTEINS_FILE,
    SEQUENCE_RE,
    SEQUENCES_COLUMNS,
    SEQUENCES_FILE,
    count_tokens,
    require_columns,
    sequence_md5,
    split_tokens,
)


REPO = Path(__file__).resolve().parents[2]


class ContractError(RuntimeError):
    """Raised when a merged database table violates the public contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-root", type=Path, default=REPO / "data/merged")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only check file existence and headers. Use full validation before publishing data.",
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="Skip locked row-count and SHA256 checks. Use only while drafting a new database contract.",
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def fail(message: str) -> None:
    raise ContractError(message)


def read_header(path: Path) -> list[str]:
    if not path.exists():
        fail(f"missing required merged table: {path}")
    return list(pd.read_csv(path, nrows=0).columns)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_snapshot(paths: dict[str, Path]) -> None:
    for file_name, expected in EXPECTED_SNAPSHOT.items():
        path = paths[file_name]
        row_count = sum(1 for _ in path.open("rb")) - 1
        if row_count != expected["rows"]:
            fail(f"{file_name}: expected {expected['rows']:,} rows, got {row_count:,}")
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            fail(
                f"{file_name}: SHA256 drift. Expected {expected['sha256']}, got {digest}. "
                "If this is an intentional database refresh, update merged_contract.EXPECTED_SNAPSHOT."
            )


def ensure_no_duplicates(values: pd.Series, name: str) -> None:
    dup = values.loc[values.duplicated()]
    if not dup.empty:
        fail(f"{name}: duplicate values, first={dup.iloc[0]!r}")


def validate_proteins(path: Path) -> pd.DataFrame:
    require_columns(read_header(path), PROTEINS_COLUMNS, PROTEINS_FILE)
    proteins = pd.read_csv(path, dtype=str, keep_default_na=False)

    if proteins.empty:
        fail("proteins.csv must not be empty")
    if proteins.isna().any().any():
        fail("proteins.csv must not contain NA values")

    bad_fpid = proteins.loc[~proteins["fpid"].str.match(FPID_RE)]
    if not bad_fpid.empty:
        fail(f"proteins.csv: invalid fpid {bad_fpid['fpid'].iloc[0]!r}")

    fp_numbers = proteins["fpid"].str[2:].astype(int)
    expected_numbers = pd.Series(range(1, len(proteins) + 1), index=proteins.index)
    if not fp_numbers.equals(expected_numbers):
        fail("proteins.csv: fpid values must be contiguous and row-ordered from FP0000001")

    bad_md5 = proteins.loc[~proteins["protein_md5"].str.match(PROTEIN_MD5_RE)]
    if not bad_md5.empty:
        fail(f"proteins.csv: invalid protein_md5 {bad_md5['protein_md5'].iloc[0]!r}")

    bad_sequence = proteins.loc[~proteins["sequence"].str.match(SEQUENCE_RE)]
    if not bad_sequence.empty:
        fail(f"proteins.csv: sequence must be non-empty uppercase letters, fpid={bad_sequence['fpid'].iloc[0]}")

    computed_md5 = proteins["sequence"].map(sequence_md5)
    mismatch_md5 = proteins.loc[proteins["protein_md5"] != computed_md5]
    if not mismatch_md5.empty:
        fail(f"proteins.csv: protein_md5 mismatch for {mismatch_md5['fpid'].iloc[0]}")

    length = pd.to_numeric(proteins["length"], errors="coerce")
    if length.isna().any():
        fail("proteins.csv: length must be integer text")
    mismatch_length = proteins.loc[length.astype(int) != proteins["sequence"].str.len()]
    if not mismatch_length.empty:
        fail(f"proteins.csv: length mismatch for {mismatch_length['fpid'].iloc[0]}")

    hydrophobicity = pd.to_numeric(proteins["hydrophobicity"], errors="coerce")
    bad_hydrophobicity = proteins.loc[proteins["hydrophobicity"].ne("") & hydrophobicity.isna()]
    if not bad_hydrophobicity.empty:
        fail(f"proteins.csv: hydrophobicity must be numeric or empty for {bad_hydrophobicity['fpid'].iloc[0]}")

    bad_bool = proteins.loc[~proteins["is_canonical"].isin(BOOLEAN_TEXT)]
    if not bad_bool.empty:
        fail(f"proteins.csv: is_canonical must be exactly True/False for {bad_bool['fpid'].iloc[0]}")

    missing_original = proteins.loc[proteins["original_ids"].str.len() == 0]
    if not missing_original.empty:
        fail(f"proteins.csv: original_ids must be non-empty for {missing_original['fpid'].iloc[0]}")

    ensure_no_duplicates(proteins["fpid"], "proteins.csv fpid")
    ensure_no_duplicates(proteins["protein_md5"], "proteins.csv protein_md5")
    ensure_no_duplicates(proteins["sequence"], "proteins.csv sequence")
    return proteins


def validate_sequences(path: Path, proteins: pd.DataFrame) -> None:
    require_columns(read_header(path), SEQUENCES_COLUMNS, SEQUENCES_FILE)
    sequences = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = proteins.loc[:, ["fpid", "sequence"]].rename(columns={"fpid": "id"})
    if len(sequences) != len(expected):
        fail(f"sequences.csv: expected {len(expected):,} rows, got {len(sequences):,}")
    if not sequences.equals(expected):
        fail("sequences.csv must exactly equal proteins.csv projected to id,sequence in fpid order")


def check_token_set(series: pd.Series, allowed: set[str], table: str, column: str) -> None:
    for value in series:
        unknown = split_tokens(value) - allowed
        if unknown:
            fail(f"{table}: unknown {column} token(s) {sorted(unknown)} in {value!r}")


def validate_interaction_chunk(chunk: pd.DataFrame, fpid_set: set[str], offset: int) -> None:
    row_label = f"interactions.csv row {offset + 2}"
    if not chunk["FPid_1"].str.match(FPID_RE).all() or not chunk["FPid_2"].str.match(FPID_RE).all():
        fail(f"{row_label}: FPid_1/FPid_2 must match FP0000001 format")
    if not (chunk["FPid_1"] < chunk["FPid_2"]).all():
        fail(f"{row_label}: pair endpoints must be unordered-canonical with FPid_1 < FPid_2")
    if not chunk["FPid_1"].isin(fpid_set).all() or not chunk["FPid_2"].isin(fpid_set).all():
        fail(f"{row_label}: pair endpoint absent from proteins.csv")

    labels = pd.to_numeric(chunk["label"], errors="coerce")
    if labels.isna().any() or not labels.isin(LABELS).all():
        fail(f"{row_label}: label must be 0 or 1")

    n_sources = pd.to_numeric(chunk["n_sources"], errors="coerce")
    if n_sources.isna().any() or (n_sources.astype(int) < 1).any():
        fail(f"{row_label}: n_sources must be a positive integer")
    source_counts = chunk["PPI_Source"].map(count_tokens)
    if not (n_sources.astype(int) == source_counts).all():
        fail(f"{row_label}: n_sources must equal distinct semicolon-delimited PPI_Source token count")

    if not chunk["Evidence_Type"].isin(EVIDENCE_TYPES).all():
        bad = chunk.loc[~chunk["Evidence_Type"].isin(EVIDENCE_TYPES), "Evidence_Type"].iloc[0]
        fail(f"{row_label}: unknown Evidence_Type {bad!r}")
    if not chunk["PPI_Tier"].isin(PPI_TIERS).all():
        bad = chunk.loc[~chunk["PPI_Tier"].isin(PPI_TIERS), "PPI_Tier"].iloc[0]
        fail(f"{row_label}: unknown PPI_Tier {bad!r}")
    expected_zh = chunk["PPI_Tier"].map(PPI_TIERS)
    if not (chunk["PPI_Tier_ZH"] == expected_zh).all():
        fail(f"{row_label}: PPI_Tier_ZH must match PPI_Tier")

    check_token_set(chunk["PPI_Source"], PPI_SOURCES, "interactions.csv", "PPI_Source")
    check_token_set(chunk["Evidence_Tags"], EVIDENCE_TAGS, "interactions.csv", "Evidence_Tags")

    negative = labels.astype(int).eq(0)
    if negative.any():
        neg_rows = chunk.loc[negative]
        if not (
            neg_rows["Evidence_Type"].eq("negative_synthetic")
            & neg_rows["PPI_Tier"].eq("negative_synthetic")
        ).all():
            fail(f"{row_label}: label=0 rows must be negative_synthetic evidence and tier")


def validate_interactions(path: Path, fpid_set: set[str], chunk_size: int) -> tuple[int, int]:
    require_columns(read_header(path), INTERACTIONS_COLUMNS, INTERACTIONS_FILE)
    rows = 0
    positives = 0
    seen_pairs: set[str] = set()
    for chunk in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunk_size):
        validate_interaction_chunk(chunk, fpid_set, rows)
        pair_keys = chunk["FPid_1"] + "|" + chunk["FPid_2"]
        duplicate_in_chunk = pair_keys.loc[pair_keys.duplicated()]
        if not duplicate_in_chunk.empty:
            fail(f"interactions.csv: duplicate pair {duplicate_in_chunk.iloc[0]!r}")
        duplicate_seen = next((key for key in pair_keys if key in seen_pairs), None)
        if duplicate_seen is not None:
            fail(f"interactions.csv: duplicate pair {duplicate_seen!r}")
        seen_pairs.update(pair_keys)
        labels = chunk["label"].astype(int)
        positives += int(labels.sum())
        rows += len(chunk)
    if rows == 0:
        fail("interactions.csv must not be empty")
    return rows, positives


def validate_pairs(path: Path, interactions_path: Path, fpid_set: set[str], chunk_size: int) -> int:
    require_columns(read_header(path), PAIRS_COLUMNS, PAIRS_FILE)
    rows = 0
    pair_iter = pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunk_size)
    interaction_iter = pd.read_csv(
        interactions_path,
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
        usecols=["FPid_1", "FPid_2", "label"],
    )
    for pair_chunk, interaction_chunk in zip(pair_iter, interaction_iter, strict=True):
        expected = interaction_chunk.rename(columns={"FPid_1": "fpid_1", "FPid_2": "fpid_2"})
        if not pair_chunk.equals(expected):
            fail(f"pairs.csv row {rows + 2}: must exactly equal interactions.csv projected to fpid_1,fpid_2,label")
        if not pair_chunk["fpid_1"].isin(fpid_set).all() or not pair_chunk["fpid_2"].isin(fpid_set).all():
            fail(f"pairs.csv row {rows + 2}: pair endpoint absent from proteins.csv")
        rows += len(pair_chunk)
    return rows


def main() -> None:
    args = parse_args()
    root = args.merged_root
    paths = {
        PROTEINS_FILE: root / PROTEINS_FILE,
        SEQUENCES_FILE: root / SEQUENCES_FILE,
        INTERACTIONS_FILE: root / INTERACTIONS_FILE,
        PAIRS_FILE: root / PAIRS_FILE,
    }

    require_columns(read_header(paths[PROTEINS_FILE]), PROTEINS_COLUMNS, PROTEINS_FILE)
    require_columns(read_header(paths[SEQUENCES_FILE]), SEQUENCES_COLUMNS, SEQUENCES_FILE)
    require_columns(read_header(paths[INTERACTIONS_FILE]), INTERACTIONS_COLUMNS, INTERACTIONS_FILE)
    require_columns(read_header(paths[PAIRS_FILE]), PAIRS_COLUMNS, PAIRS_FILE)
    if args.quick:
        print(json.dumps({"status": "ok", "mode": "quick", "merged_root": str(root)}, indent=2))
        return
    if not args.skip_snapshot:
        validate_snapshot(paths)

    proteins = validate_proteins(paths[PROTEINS_FILE])
    fpid_set = set(proteins["fpid"])
    validate_sequences(paths[SEQUENCES_FILE], proteins)
    interaction_rows, positives = validate_interactions(paths[INTERACTIONS_FILE], fpid_set, args.chunk_size)
    pair_rows = validate_pairs(paths[PAIRS_FILE], paths[INTERACTIONS_FILE], fpid_set, args.chunk_size)
    if pair_rows != interaction_rows:
        fail(f"pairs.csv row count {pair_rows:,} must equal interactions.csv row count {interaction_rows:,}")

    report = {
        "status": "ok",
        "merged_root": str(root),
        "proteins": int(len(proteins)),
        "interactions": int(interaction_rows),
        "pairs": int(pair_rows),
        "positives": int(positives),
        "negatives": int(interaction_rows - positives),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

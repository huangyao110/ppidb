"""
Prepare a positive-only host-virus training CSV for V3 fine-tuning.

The ViraHinter high-confidence validation and test CSVs are treated as fixed
holdouts. Training candidates come from current host_corpus positives and any
pair whose hp-id endpoint pair appears in the holdouts is excluded. To match
ViraHinter's split isolation more closely, train pathogen sequences are also
filtered by MMseqs homology against validation/test pathogen sequences; host
endpoint overlap is allowed.

Outputs under data/datasets/bench_host_corpus:
  - host_v3_train_pos_hp.csv
  - host_v3_train_pos_minimal_hp.csv
  - host_v3_train_summary.json

Run:
  python p2psiglip_db/data/prepare_host_v3_train_dataset.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = ROOT / "data" / "external" / "host_corpus" / "pairs_with_sequences.parquet"
DEFAULT_OUT = ROOT / "data" / "datasets" / "bench_host_corpus"
DEFAULT_VAL = DEFAULT_OUT / "virahinter_highconfidence_val_hp.csv"
DEFAULT_TEST = DEFAULT_OUT / "virahinter_highconfidence_test_hp.csv"
DEFAULT_VAL_LT = DEFAULT_OUT / "virahinter_highconfidence_val_lt_3000_hp.csv"
DEFAULT_SEQUENCES = DEFAULT_OUT / "sequences_hp.csv"

TRAIN_SPLITS = {
    ("hvidb", "all"),
    ("virahinter", "raw"),
    ("virahinter", "highconfidence_train"),
    ("virahinter", "mediumconfidence_train"),
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def md5_seq(seq: str) -> str:
    return hashlib.md5(seq.strip().upper().encode("utf-8")).hexdigest()


def hp_id(seq: str) -> str:
    return f"hp_{md5_seq(seq)[:16]}"


def pair_key(id1: pd.Series, id2: pd.Series) -> pd.Series:
    return id1.astype(str) + "\t" + id2.astype(str)


def make_hp_pairs(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ID_1": df["host_sequence"].map(hp_id),
            "ID_2": df["pathogen_sequence"].map(hp_id),
            "label": df["label"].astype(int),
            "dataset": df["dataset"].astype(str),
            "split": df["split"].astype(str),
            "source_database": df["source_database"].astype(str),
            "confidence": df["confidence"].astype(str),
            "host_id": df["host_id"].astype(str),
            "pathogen_id": df["pathogen_id"].astype(str),
            "host_len": df["host_len"],
            "pathogen_len": df["pathogen_len"],
            "pathogen_family": df["pathogen_family"].astype(str),
            "original_pair_id": df["original_pair_id"].astype(str),
            "_pathogen_sequence": df["pathogen_sequence"].astype(str).str.strip().str.upper(),
        }
    )
    out = out[out["ID_1"] != out["ID_2"]].copy()
    out["pair_key"] = pair_key(out["ID_1"], out["ID_2"])
    return out


def load_holdout_info(paths: list[Path]) -> tuple[set[str], set[str], dict[str, dict[str, int | str]]]:
    keys: set[str] = set()
    pathogen_ids: set[str] = set()
    summary: dict[str, dict[str, int | str]] = {}
    for path in paths:
        df = pd.read_csv(path)
        df["pair_key"] = pair_key(df["ID_1"], df["ID_2"])
        keys.update(df["pair_key"].tolist())
        pathogen_ids.update(df["ID_2"].astype(str).tolist())
        summary[path.name] = {
            "path": rel(path),
            "rows": int(len(df)),
            "positives": int((df["label"].astype(int) == 1).sum()) if "label" in df.columns else 0,
            "negatives": int((df["label"].astype(int) == 0).sum()) if "label" in df.columns else 0,
            "unique_pair_keys": int(df["pair_key"].nunique()),
            "unique_pathogen_endpoints": int(df["ID_2"].astype(str).nunique()),
        }
    return keys, pathogen_ids, summary


def counts_by(df: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    if df.empty:
        return []
    return (
        df.groupby(columns, dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
        .to_dict(orient="records")
    )


def wrap_fasta_sequence(sequence: str) -> str:
    return "\n".join(sequence[start : start + 80] for start in range(0, len(sequence), 80))


def write_fasta(records: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for protein_id, sequence in sorted(records.items()):
            handle.write(f">{protein_id}\n{wrap_fasta_sequence(sequence)}\n")


def load_sequence_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, usecols=["id", "sequence"])
    return dict(zip(df["id"].astype(str), df["sequence"].astype(str).str.strip().str.upper()))


def run_pathogen_homology_filter(
    candidates: pd.DataFrame,
    holdout_pathogen_ids: set[str],
    sequence_by_id: dict[str, str],
    out_dir: Path,
    *,
    min_seq_id: float,
    coverage: float,
    sensitivity: float,
    max_seqs: int,
) -> tuple[set[str], dict[str, object]]:
    from pymmseqs.commands import easy_search

    homology_dir = out_dir / "host_v3_train_homology_filter"
    homology_dir.mkdir(parents=True, exist_ok=True)

    holdout_records = {
        pid: sequence_by_id[pid]
        for pid in sorted(holdout_pathogen_ids)
        if pid in sequence_by_id and sequence_by_id[pid]
    }
    target_records = (
        candidates[["ID_2", "_pathogen_sequence"]]
        .drop_duplicates("ID_2")
        .set_index("ID_2")["_pathogen_sequence"]
        .to_dict()
    )

    query_fasta = homology_dir / "holdout_pathogens.faa"
    target_fasta = homology_dir / "candidate_train_pathogens.faa"
    hits_tsv = homology_dir / "holdout_vs_train_pathogens.tsv"
    tmp_dir = homology_dir / "tmp"
    write_fasta(holdout_records, query_fasta)
    write_fasta(target_records, target_fasta)

    easy_search(
        query_fasta=query_fasta,
        target_fasta_or_db=target_fasta,
        alignment_file=hits_tsv,
        tmp_dir=tmp_dir,
        s=sensitivity,
        min_seq_id=min_seq_id,
        c=coverage,
        max_seqs=max_seqs,
        search_type=1,
        format_output="query,target,fident,alnlen,qcov,tcov,evalue,bits",
    )

    columns = ["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"]
    if hits_tsv.exists() and hits_tsv.stat().st_size > 0:
        hits = pd.read_csv(hits_tsv, sep="\t")
        if list(hits.columns) != columns:
            hits = pd.read_csv(hits_tsv, sep="\t", names=columns)
        hits["fident"] = pd.to_numeric(hits["fident"], errors="coerce")
        hits["bits"] = pd.to_numeric(hits["bits"], errors="coerce")
        hits = hits[hits["fident"] >= min_seq_id].copy()
    else:
        hits = pd.DataFrame(columns=columns)

    homologous_train_pathogens = set(hits["target"].astype(str))
    best_hits_path = homology_dir / "best_hits_by_train_pathogen.csv"
    if len(hits):
        best = hits.sort_values(["target", "fident", "bits"], ascending=[True, False, False])
        best = best.drop_duplicates("target", keep="first")
        best.to_csv(best_hits_path, index=False)
    else:
        best_hits_path.write_text("query,target,fident,alnlen,qcov,tcov,evalue,bits\n", encoding="utf-8")

    summary = {
        "method": "pymmseqs.easy_search",
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "sensitivity": sensitivity,
        "max_seqs": max_seqs,
        "query_fasta": rel(query_fasta),
        "target_fasta": rel(target_fasta),
        "hits_tsv": rel(hits_tsv),
        "best_hits_by_train_pathogen_csv": rel(best_hits_path),
        "holdout_pathogens_requested": len(holdout_pathogen_ids),
        "holdout_pathogens_with_sequence": len(holdout_records),
        "candidate_train_pathogens": len(target_records),
        "hit_rows": int(len(hits)),
        "homologous_train_pathogens": len(homologous_train_pathogens),
        "queries_with_hit": int(hits["query"].nunique()) if len(hits) else 0,
        "max_fident": float(hits["fident"].max()) if len(hits) else None,
        "median_best_fident": float(
            hits.sort_values(["target", "fident"], ascending=[True, False])
            .drop_duplicates("target")["fident"]
            .median()
        )
        if len(hits)
        else None,
    }
    (homology_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return homologous_train_pathogens, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V3 host-virus positive training CSV.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sequences", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--holdout", type=Path, nargs="*", default=[DEFAULT_VAL, DEFAULT_TEST])
    parser.add_argument("--include-val-lt-3000", action="store_true")
    parser.add_argument("--skip-homology-filter", action="store_true")
    parser.add_argument("--min-seq-id", type=float, default=0.6)
    parser.add_argument("--coverage", type=float, default=0.0)
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--max-seqs", type=int, default=100000)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    holdout_paths = list(args.holdout)
    if args.include_val_lt_3000:
        holdout_paths.append(DEFAULT_VAL_LT)
    holdout_keys, holdout_pathogen_ids, holdout_summary = load_holdout_info(holdout_paths)

    raw = pd.read_parquet(args.pairs)
    split_mask = raw[["dataset", "split"]].apply(tuple, axis=1).isin(TRAIN_SPLITS)
    candidates_raw = raw[split_mask & (raw["label"].astype(int) == 1)].copy()
    candidates = make_hp_pairs(candidates_raw)

    before_dedupe = len(candidates)
    pair_overlap_mask = candidates["pair_key"].isin(holdout_keys)
    if args.skip_homology_filter:
        sequence_by_id = {}
        homology_summary = {"method": "skipped"}
        homologous_train_pathogens: set[str] = set()
    else:
        sequence_by_id = load_sequence_map(args.sequences)
        homologous_train_pathogens, homology_summary = run_pathogen_homology_filter(
            candidates,
            holdout_pathogen_ids,
            sequence_by_id,
            args.out,
            min_seq_id=args.min_seq_id,
            coverage=args.coverage,
            sensitivity=args.sensitivity,
            max_seqs=args.max_seqs,
        )
    pathogen_overlap_mask = candidates["ID_2"].isin(homologous_train_pathogens)
    excluded_pair_overlap = candidates[pair_overlap_mask].copy()
    excluded_pathogen_overlap = candidates[pathogen_overlap_mask].copy()
    exclude_mask = pair_overlap_mask | pathogen_overlap_mask
    excluded_overlap = candidates[exclude_mask].copy()
    candidates = candidates[~exclude_mask].copy()
    after_holdout = len(candidates)

    candidates = candidates.sort_values(
        ["dataset", "split", "source_database", "ID_1", "ID_2", "original_pair_id"]
    )
    train = candidates.drop_duplicates(["ID_1", "ID_2"], keep="first").copy()
    duplicate_rows_removed = after_holdout - len(train)
    train = train.drop(columns=["pair_key", "_pathogen_sequence"])

    out_csv = args.out / "host_v3_train_pos_hp.csv"
    minimal_csv = args.out / "host_v3_train_pos_minimal_hp.csv"
    train.to_csv(out_csv, index=False)
    train[["ID_1", "ID_2"]].to_csv(minimal_csv, index=False)

    val_keys, val_pathogens, _ = load_holdout_info([DEFAULT_VAL]) if DEFAULT_VAL.exists() else (set(), set(), {})
    test_keys, test_pathogens, _ = load_holdout_info([DEFAULT_TEST]) if DEFAULT_TEST.exists() else (set(), set(), {})
    train_keys = set(pair_key(train["ID_1"], train["ID_2"]))
    train_pathogens = set(train["ID_2"].astype(str))
    summary = {
        "source_pairs": rel(args.pairs),
        "train_csv": rel(out_csv),
        "train_minimal_csv": rel(minimal_csv),
        "format": "positive-only V3/SigLIP pair CSV; use ID_1 as host side and ID_2 as virus/pathogen side",
        "isolation_policy": "ViraHinter-aligned: fixed val/test holdouts; exclude exact pair overlaps and exclude train rows whose pathogen sequence has a MMseqs hit to val/test pathogen sequences. Host endpoint overlap is allowed.",
        "holdouts_unchanged": holdout_summary,
        "homology_filter": homology_summary,
        "candidate_positive_rows_before_dedupe": int(before_dedupe),
        "excluded_rows_exact_holdout_pair": int(len(excluded_pair_overlap)),
        "excluded_rows_homologous_holdout_pathogen": int(len(excluded_pathogen_overlap)),
        "excluded_rows_total_isolation": int(len(excluded_overlap)),
        "rows_after_holdout_exclusion": int(after_holdout),
        "duplicate_rows_removed_by_hp_pair": int(duplicate_rows_removed),
        "train_rows": int(len(train)),
        "train_unique_proteins": int(len(set(train["ID_1"]) | set(train["ID_2"]))),
        "train_unique_hosts": int(train["ID_1"].nunique()),
        "train_unique_pathogens": int(train["ID_2"].nunique()),
        "train_vs_val_exact_pair_overlap": int(len(train_keys & val_keys)),
        "train_vs_test_exact_pair_overlap": int(len(train_keys & test_keys)),
        "train_vs_val_pathogen_endpoint_overlap": int(len(train_pathogens & val_pathogens)),
        "train_vs_test_pathogen_endpoint_overlap": int(len(train_pathogens & test_pathogens)),
        "candidate_counts_by_dataset_split": counts_by(candidates, ["dataset", "split"]),
        "train_counts_by_dataset_split": counts_by(train, ["dataset", "split"]),
        "train_counts_by_source_database": counts_by(train, ["dataset", "source_database"]),
        "excluded_holdout_pair_overlap_counts": counts_by(excluded_pair_overlap, ["dataset", "split", "source_database"]),
        "excluded_homologous_holdout_pathogen_counts": counts_by(
            excluded_pathogen_overlap, ["dataset", "split", "source_database"]
        ),
        "excluded_total_isolation_counts": counts_by(excluded_overlap, ["dataset", "split", "source_database"]),
    }
    summary_path = args.out / "host_v3_train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

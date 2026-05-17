"""Create C3-filtered hash-ID train splits from PPIDB.

The command accepts user-supplied validation/test pair CSVs or samples them
from ``data/merged/interactions.csv``. Only positive validation/test endpoints
taint C3 clusters; output training pairs are always positive-only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from p2psiglip_db.data.split_utils import (
    CANONICAL_PAIR_FRONT,
    HEX32,
    label_counts_to_json,
    normalize_sequence,
    pair_columns,
    pair_key,
    parse_cluster_tsv,
    resolve_mmseqs,
    run_mmseqs_easy_cluster,
    sequence_md5,
    validate_hash_pair_csv,
    validate_hash_sequences,
    write_fasta,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MERGED_ROOT = REPO / "data/merged"
DEFAULT_NEGATIVE_SOURCE_PRIORITY = ("Negatome", "BERNETT_neg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train_pos.csv split from PPIDB with C3 filtering against "
            "provided or generated validation/test positives."
        )
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--merged-root", type=Path, default=DEFAULT_MERGED_ROOT)
    parser.add_argument("--merged-proteins", type=Path, default=None)
    parser.add_argument("--merged-interactions", type=Path, default=None)
    parser.add_argument(
        "--sequences-csv",
        type=Path,
        default=None,
        help="Required when --val-csv or --test-csv is provided; columns must be id,sequence.",
    )
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument(
        "--val-size",
        type=int,
        default=0,
        help="Positive validation rows to sample from PPIDB and append to val.csv.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=0,
        help="Positive test rows to sample from PPIDB and append to test.csv.",
    )
    parser.add_argument(
        "--val-neg-size",
        type=int,
        default=0,
        help="Negative validation rows to sample from PPIDB and append to val.csv.",
    )
    parser.add_argument(
        "--test-neg-size",
        type=int,
        default=0,
        help="Negative test rows to sample from PPIDB and append to test.csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identity", type=float, default=0.4)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 8))
    parser.add_argument("--mmseqs", default=None)
    parser.add_argument("--reuse-cluster", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument(
        "--negative-source-priority",
        default=",".join(DEFAULT_NEGATIVE_SOURCE_PRIORITY),
        help="Comma-separated label=0 PPI_Source tokens to prefer before generic negatives.",
    )
    return parser.parse_args()


def _positive_int(value: int, name: str) -> None:
    if value < 0:
        raise SystemExit(f"{name} must be >= 0")


def _pair_key_values(left: str, right: str) -> str:
    left = str(left)
    right = str(right)
    return f"{left}|{right}" if left <= right else f"{right}|{left}"


def _canonical_pair_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[CANONICAL_PAIR_FRONT].copy()
    out["ID_1"] = out["ID_1"].astype(str)
    out["ID_2"] = out["ID_2"].astype(str)
    swap = out["ID_1"] > out["ID_2"]
    if bool(swap.any()):
        left = out.loc[swap, "ID_1"].copy()
        out.loc[swap, "ID_1"] = out.loc[swap, "ID_2"].to_numpy()
        out.loc[swap, "ID_2"] = left.to_numpy()
    out["label"] = out["label"].astype(int)
    return out


def _validate_pair_labels(path: Path, df: pd.DataFrame) -> None:
    labels = set(df["label"].astype(int).unique().tolist())
    bad = labels - {0, 1}
    if bad:
        raise SystemExit(f"{path}: label must be 0 or 1; got {sorted(bad)}")


def _read_external_pairs(path: Path, sequence_ids: set[str], chunk_size: int) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing pair CSV: {path}")
    header = list(pd.read_csv(path, nrows=0).columns)
    if header != CANONICAL_PAIR_FRONT:
        raise SystemExit(f"{path}: columns must be exactly {CANONICAL_PAIR_FRONT}; got {header}")
    validate_hash_pair_csv(path, sequence_ids=sequence_ids, chunk_size=chunk_size)
    df = pd.read_csv(path)
    _validate_pair_labels(path, df)
    df = _canonical_pair_frame(df)
    duplicated = pair_key(df["ID_1"], df["ID_2"]).duplicated()
    if bool(duplicated.any()):
        examples = df.loc[duplicated, ["ID_1", "ID_2"]].head(5).to_dict("records")
        raise SystemExit(f"{path}: duplicate unordered pairs, first={examples}")
    return df


def _concat_pair_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=CANONICAL_PAIR_FRONT)
    out = pd.concat(non_empty, ignore_index=True)
    return _canonical_pair_frame(out)


def _pair_keys(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    return set(pair_key(df["ID_1"], df["ID_2"]).astype(str))


def _check_no_overlap(named_frames: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for name, df in named_frames.items():
        for key in _pair_keys(df):
            prior = seen.get(key)
            if prior is not None:
                raise SystemExit(f"pair overlap between {prior} and {name}: {key}")
            seen[key] = name


def _load_external_sequences(path: Path | None) -> tuple[pd.DataFrame, set[str]]:
    if path is None:
        return pd.DataFrame(columns=["id", "sequence"]), set()
    return validate_hash_sequences(path)


def _load_merged_proteins(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing merged proteins CSV: {path}")
    df = pd.read_csv(path, usecols=["protein_md5", "fpid", "sequence"])
    df["protein_md5"] = df["protein_md5"].astype(str)
    df["fpid"] = df["fpid"].astype(str)
    df["sequence"] = df["sequence"].map(normalize_sequence)
    df = df[(df["protein_md5"] != "") & (df["fpid"] != "") & (df["sequence"] != "")]
    mismatched = df.loc[df["protein_md5"] != df["sequence"].map(sequence_md5), "protein_md5"].head(5)
    if len(mismatched):
        raise SystemExit(f"{path}: protein_md5 does not match sequence MD5, first={mismatched.tolist()}")
    duplicated_fpid = int(df["fpid"].duplicated().sum())
    if duplicated_fpid:
        raise SystemExit(f"{path}: duplicate fpid values: {duplicated_fpid}")
    fpid_to_md5 = dict(zip(df["fpid"], df["protein_md5"]))
    seqs = (
        df[["protein_md5", "sequence"]]
        .drop_duplicates("protein_md5")
        .rename(columns={"protein_md5": "id"})
        .sort_values("id")
        .reset_index(drop=True)
    )
    return seqs, fpid_to_md5


def _merge_sequence_tables(merged: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([merged[["id", "sequence"]], external[["id", "sequence"]]], ignore_index=True)
    combined["id"] = combined["id"].astype(str)
    combined["sequence"] = combined["sequence"].map(normalize_sequence)
    conflicting = combined.groupby("id")["sequence"].nunique()
    conflicting = conflicting.loc[conflicting > 1]
    if not conflicting.empty:
        examples = conflicting.head(5).index.tolist()
        raise SystemExit(f"sequence ID conflicts between merged and external sequences: {examples}")
    return combined.drop_duplicates("id").sort_values("id").reset_index(drop=True)


def _require_hash_id(value: str, source: Path) -> None:
    if not HEX32.match(value):
        raise SystemExit(f"{source}: mapped non-MD5 ID: {value}")


def _reservoir_add(
    sample: list[dict[str, object]],
    seen: int,
    row: dict[str, object],
    capacity: int,
    rng: random.Random,
) -> int:
    seen += 1
    if capacity <= 0:
        return seen
    if len(sample) < capacity:
        sample.append(row)
    else:
        idx = rng.randrange(seen)
        if idx < capacity:
            sample[idx] = row
    return seen


def _read_interaction_header(path: Path) -> tuple[list[str], str, str]:
    if not path.exists():
        raise SystemExit(f"missing merged interactions CSV: {path}")
    header = list(pd.read_csv(path, nrows=0).columns)
    left, right = pair_columns(header)
    if "label" not in header:
        raise SystemExit(f"{path}: expected label column")
    return header, left, right


def _source_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    normalized = str(value).replace(",", ";")
    return {token.strip().lower() for token in normalized.split(";") if token.strip()}


def _source_matches_priority(value: object, priority: set[str]) -> bool:
    if not priority:
        return False
    return bool(_source_tokens(value) & priority)


def _iter_mapped_interaction_chunks(
    path: Path,
    fpid_to_md5: dict[str, str],
    chunk_size: int,
) -> Iterable[pd.DataFrame]:
    header, left, right = _read_interaction_header(path)
    usecols = [left, right, "label"]
    if "PPI_Source" in header:
        usecols.append("PPI_Source")
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size):
        chunk[left] = chunk[left].astype(str)
        chunk[right] = chunk[right].astype(str)
        chunk["label"] = pd.to_numeric(chunk["label"], errors="coerce")
        chunk = chunk[chunk["label"].isin([0, 1])].copy()
        if chunk.empty:
            continue
        mapped_left = chunk[left].map(fpid_to_md5)
        mapped_right = chunk[right].map(fpid_to_md5)
        ok = mapped_left.notna() & mapped_right.notna()
        if not bool(ok.any()):
            continue
        out = pd.DataFrame(
            {
                "ID_1": mapped_left.loc[ok].astype(str).to_numpy(),
                "ID_2": mapped_right.loc[ok].astype(str).to_numpy(),
                "label": chunk.loc[ok, "label"].astype(int).to_numpy(),
            }
        )
        if "PPI_Source" in chunk.columns:
            out["PPI_Source"] = chunk.loc[ok, "PPI_Source"].astype(str).to_numpy()
        out = _canonical_pair_frame(out.assign(label=out["label"]))
        if "PPI_Source" in chunk.columns:
            out["PPI_Source"] = chunk.loc[ok, "PPI_Source"].astype(str).to_numpy()
        yield out


def _sample_merged_pairs(
    path: Path,
    fpid_to_md5: dict[str, str],
    label: int,
    count: int,
    exclude_keys: set[str],
    rng: random.Random,
    chunk_size: int,
    source_priority: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if count <= 0:
        return pd.DataFrame(columns=CANONICAL_PAIR_FRONT), {"requested": 0, "eligible": 0, "sampled": 0}

    def scan(priority_mode: bool | None, capacity: int, extra_exclude: set[str]) -> tuple[list[dict[str, object]], int]:
        sample: list[dict[str, object]] = []
        seen = 0
        for chunk in _iter_mapped_interaction_chunks(path, fpid_to_md5, chunk_size):
            chunk = chunk[chunk["label"] == label].copy()
            if chunk.empty:
                continue
            if priority_mode is not None:
                has_priority = chunk.get("PPI_Source", pd.Series("", index=chunk.index)).map(
                    lambda value: _source_matches_priority(value, source_priority or set())
                )
                chunk = chunk[has_priority] if priority_mode else chunk[~has_priority]
                if chunk.empty:
                    continue
            keys = pair_key(chunk["ID_1"], chunk["ID_2"])
            chunk = chunk.loc[~keys.isin(exclude_keys | extra_exclude), CANONICAL_PAIR_FRONT].copy()
            if chunk.empty:
                continue
            for row in chunk.itertuples(index=False):
                pair = {"ID_1": row.ID_1, "ID_2": row.ID_2, "label": int(row.label)}
                _require_hash_id(pair["ID_1"], path)
                _require_hash_id(pair["ID_2"], path)
                seen = _reservoir_add(sample, seen, pair, capacity, rng)
        return sample, seen

    report = {"requested": int(count), "eligible": 0, "sampled": 0}
    if label == 0 and source_priority:
        priority_sample, priority_seen = scan(True, count, set())
        report["priority_eligible"] = int(priority_seen)
        report["priority_sampled"] = int(len(priority_sample))
        if len(priority_sample) < count:
            selected_keys = {_pair_key_values(row["ID_1"], row["ID_2"]) for row in priority_sample}
            fill_sample, fill_seen = scan(False, count - len(priority_sample), selected_keys)
            report["fallback_eligible"] = int(fill_seen)
            sample = priority_sample + fill_sample
        else:
            report["fallback_eligible"] = 0
            sample = priority_sample
        report["eligible"] = int(report["priority_eligible"] + report["fallback_eligible"])
    else:
        sample, eligible = scan(None, count, set())
        report["eligible"] = int(eligible)

    if len(sample) < count:
        raise SystemExit(f"requested {count} label={label} rows but only sampled {len(sample)}")
    rng.shuffle(sample)
    df = pd.DataFrame(sample, columns=CANONICAL_PAIR_FRONT)
    report["sampled"] = int(len(df))
    return _canonical_pair_frame(df), report


def _split_sampled_rows(rows: pd.DataFrame, val_count: int, test_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty:
        empty = pd.DataFrame(columns=CANONICAL_PAIR_FRONT)
        return empty.copy(), empty.copy()
    val = rows.iloc[:val_count].copy()
    test = rows.iloc[val_count : val_count + test_count].copy()
    return val.reset_index(drop=True), test.reset_index(drop=True)


def _write_pair_csv(path: Path, df: pd.DataFrame) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _canonical_pair_frame(df)
    df.to_csv(path, index=False)
    labels = Counter(df["label"].astype(int).tolist())
    ids = set(df["ID_1"].astype(str)) | set(df["ID_2"].astype(str))
    return {
        "path": str(path),
        "rows": int(len(df)),
        "label_counts": label_counts_to_json(labels),
        "unique_ids": int(len(ids)),
    }


def _build_c3_filter(
    all_sequences: pd.DataFrame,
    positive_holdout_ids: set[str],
    out_dir: Path,
    identity: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    mmseqs: str | None,
    reuse_cluster: bool,
) -> tuple[set[str], pd.DataFrame, dict[str, object]]:
    cluster_map_path = out_dir / "holdout_cluster_map.csv"
    if not positive_holdout_ids:
        empty = pd.DataFrame(columns=["id", "cluster", "is_positive_holdout", "is_tainted"])
        empty.to_csv(cluster_map_path, index=False)
        return set(), empty, {
            "skipped": True,
            "reason": "no positive holdout endpoints",
            "positive_holdout_ids": 0,
            "tainted_clusters": 0,
            "tainted_ids": 0,
            "cluster_map_csv": str(cluster_map_path),
        }

    work_dir = out_dir / "_work" / "c3"
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta = work_dir / "cluster_input.fasta"
    write_fasta(all_sequences, fasta)
    mmseqs_bin = resolve_mmseqs(mmseqs)
    cluster_tsv = run_mmseqs_easy_cluster(
        fasta=fasta,
        out_dir=work_dir,
        identity=identity,
        coverage=coverage,
        cov_mode=cov_mode,
        threads=threads,
        mmseqs_bin=mmseqs_bin,
        reuse_cluster=reuse_cluster,
    )
    cluster_map = parse_cluster_tsv(cluster_tsv)
    missing = sorted(positive_holdout_ids - set(cluster_map["id"].astype(str)))
    if missing:
        raise SystemExit(f"positive holdout IDs missing from MMseqs clusters, first={missing[:5]}")
    tainted_clusters = set(
        cluster_map.loc[cluster_map["id"].isin(positive_holdout_ids), "cluster"].astype(str)
    )
    cluster_map["is_positive_holdout"] = cluster_map["id"].isin(positive_holdout_ids)
    cluster_map["is_tainted"] = cluster_map["cluster"].isin(tainted_clusters)
    tainted_ids = set(cluster_map.loc[cluster_map["is_tainted"], "id"].astype(str))
    cluster_map = cluster_map.sort_values(["cluster", "id"]).reset_index(drop=True)
    cluster_map.to_csv(cluster_map_path, index=False)
    return tainted_ids, cluster_map, {
        "skipped": False,
        "mmseqs": mmseqs_bin,
        "identity": identity,
        "coverage": coverage,
        "cov_mode": cov_mode,
        "threads": threads,
        "input_sequences": int(len(all_sequences)),
        "positive_holdout_ids": int(len(positive_holdout_ids)),
        "clusters": int(cluster_map["cluster"].nunique()),
        "cluster_map_rows": int(len(cluster_map)),
        "tainted_clusters": int(len(tainted_clusters)),
        "tainted_ids": int(len(tainted_ids)),
        "cluster_tsv": str(cluster_tsv),
        "cluster_map_csv": str(cluster_map_path),
    }


def _write_train_pos(
    interactions_path: Path,
    out_csv: Path,
    fpid_to_md5: dict[str, str],
    tainted_ids: set[str],
    exclude_pair_keys: set[str],
    chunk_size: int,
) -> tuple[set[str], dict[str, object]]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()

    first = True
    rows_in_positive = 0
    rows_after_c3 = 0
    rows_out = 0
    used_ids: set[str] = set()

    for chunk in _iter_mapped_interaction_chunks(interactions_path, fpid_to_md5, chunk_size):
        chunk = chunk[chunk["label"] == 1].copy()
        if chunk.empty:
            continue
        rows_in_positive += len(chunk)
        keep = ~chunk["ID_1"].isin(tainted_ids) & ~chunk["ID_2"].isin(tainted_ids)
        chunk = chunk.loc[keep, CANONICAL_PAIR_FRONT].copy()
        rows_after_c3 += len(chunk)
        if chunk.empty:
            continue
        keys = pair_key(chunk["ID_1"], chunk["ID_2"])
        chunk = chunk.loc[~keys.isin(exclude_pair_keys), CANONICAL_PAIR_FRONT].copy()
        if chunk.empty:
            continue
        chunk.to_csv(out_csv, mode="w" if first else "a", header=first, index=False)
        first = False
        rows_out += len(chunk)
        used_ids.update(chunk["ID_1"].astype(str))
        used_ids.update(chunk["ID_2"].astype(str))

    if first:
        raise SystemExit("C3 filtering removed all positive train rows")
    return used_ids, {
        "path": str(out_csv),
        "positive_rows_in": int(rows_in_positive),
        "rows_after_c3": int(rows_after_c3),
        "rows_out": int(rows_out),
        "unique_ids": int(len(used_ids)),
    }


def _write_sequences(path: Path, all_sequences: pd.DataFrame, used_ids: set[str]) -> dict[str, object]:
    seqs = all_sequences[all_sequences["id"].isin(used_ids)].copy()
    missing = sorted(used_ids - set(seqs["id"].astype(str)))
    if missing:
        raise SystemExit(f"missing sequences for output IDs, first={missing[:5]}")
    seqs = seqs.sort_values("id").reset_index(drop=True)
    seqs.to_csv(path, index=False)
    return {"path": str(path), "rows": int(len(seqs))}


def main() -> None:
    args = parse_args()
    t0 = time.time()
    for name in ("val_size", "test_size", "val_neg_size", "test_neg_size"):
        _positive_int(getattr(args, name), f"--{name.replace('_', '-')}")

    merged_proteins = args.merged_proteins or args.merged_root / "proteins.csv"
    merged_interactions = args.merged_interactions or args.merged_root / "interactions.csv"
    provided_any = args.val_csv is not None or args.test_csv is not None
    generated_any = any([args.val_size, args.test_size, args.val_neg_size, args.test_neg_size])
    if not provided_any and not generated_any:
        raise SystemExit("provide --val-csv/--test-csv or request --val-size/--test-size/negative sizes")
    if provided_any and args.sequences_csv is None:
        raise SystemExit("--sequences-csv is required when --val-csv or --test-csv is provided")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    negative_priority = {
        item.strip().lower()
        for item in str(args.negative_source_priority).split(",")
        if item.strip()
    }

    print(f"loading merged proteins: {merged_proteins}", flush=True)
    merged_sequences, fpid_to_md5 = _load_merged_proteins(merged_proteins)
    external_sequences, external_ids = _load_external_sequences(args.sequences_csv)

    val_external = (
        _read_external_pairs(args.val_csv, external_ids, args.chunk_size)
        if args.val_csv is not None
        else pd.DataFrame(columns=CANONICAL_PAIR_FRONT)
    )
    test_external = (
        _read_external_pairs(args.test_csv, external_ids, args.chunk_size)
        if args.test_csv is not None
        else pd.DataFrame(columns=CANONICAL_PAIR_FRONT)
    )
    _check_no_overlap({"val_csv": val_external, "test_csv": test_external})

    exclude_keys = _pair_keys(val_external) | _pair_keys(test_external)
    print(f"sampling generated holdouts from: {merged_interactions}", flush=True)
    positive_total = args.val_size + args.test_size
    positive_sample, positive_report = _sample_merged_pairs(
        merged_interactions,
        fpid_to_md5,
        label=1,
        count=positive_total,
        exclude_keys=exclude_keys,
        rng=rng,
        chunk_size=args.chunk_size,
    )
    val_generated_pos, test_generated_pos = _split_sampled_rows(
        positive_sample,
        args.val_size,
        args.test_size,
    )
    exclude_keys |= _pair_keys(val_generated_pos) | _pair_keys(test_generated_pos)

    negative_total = args.val_neg_size + args.test_neg_size
    negative_sample, negative_report = _sample_merged_pairs(
        merged_interactions,
        fpid_to_md5,
        label=0,
        count=negative_total,
        exclude_keys=exclude_keys,
        rng=rng,
        chunk_size=args.chunk_size,
        source_priority=negative_priority,
    )
    val_generated_neg, test_generated_neg = _split_sampled_rows(
        negative_sample,
        args.val_neg_size,
        args.test_neg_size,
    )

    val_df = _concat_pair_frames([val_external, val_generated_pos, val_generated_neg])
    test_df = _concat_pair_frames([test_external, test_generated_pos, test_generated_neg])
    _check_no_overlap({"val": val_df, "test": test_df})
    holdout_pair_keys = _pair_keys(val_df) | _pair_keys(test_df)

    all_sequences = _merge_sequence_tables(merged_sequences, external_sequences)
    positive_holdout = _concat_pair_frames(
        [val_df[val_df["label"] == 1], test_df[test_df["label"] == 1]]
    )
    positive_holdout_ids = set(positive_holdout["ID_1"].astype(str)) | set(
        positive_holdout["ID_2"].astype(str)
    )

    print(
        f"C3 positive holdout ids={len(positive_holdout_ids):,}; "
        f"val rows={len(val_df):,}; test rows={len(test_df):,}",
        flush=True,
    )
    tainted_ids, cluster_map, c3_report = _build_c3_filter(
        all_sequences=all_sequences,
        positive_holdout_ids=positive_holdout_ids,
        out_dir=args.out_dir,
        identity=args.identity,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        threads=args.threads,
        mmseqs=args.mmseqs,
        reuse_cluster=args.reuse_cluster,
    )

    train_ids, train_report = _write_train_pos(
        interactions_path=merged_interactions,
        out_csv=args.out_dir / "train_pos.csv",
        fpid_to_md5=fpid_to_md5,
        tainted_ids=tainted_ids,
        exclude_pair_keys=holdout_pair_keys,
        chunk_size=args.chunk_size,
    )

    outputs: dict[str, object] = {"train_pos.csv": train_report}
    used_ids = set(train_ids)
    if not val_df.empty:
        outputs["val.csv"] = _write_pair_csv(args.out_dir / "val.csv", val_df)
        used_ids.update(val_df["ID_1"].astype(str))
        used_ids.update(val_df["ID_2"].astype(str))
    if not test_df.empty:
        outputs["test.csv"] = _write_pair_csv(args.out_dir / "test.csv", test_df)
        used_ids.update(test_df["ID_1"].astype(str))
        used_ids.update(test_df["ID_2"].astype(str))
    outputs["sequences.csv"] = _write_sequences(args.out_dir / "sequences.csv", all_sequences, used_ids)

    report = {
        "status": "ok",
        "output_dir": str(args.out_dir),
        "id_namespace": "sequence_md5",
        "pair_columns": CANONICAL_PAIR_FRONT,
        "train_positive_only": True,
        "merged_proteins": str(merged_proteins),
        "merged_interactions": str(merged_interactions),
        "external_sequences_csv": None if args.sequences_csv is None else str(args.sequences_csv),
        "external_val_csv": None if args.val_csv is None else str(args.val_csv),
        "external_test_csv": None if args.test_csv is None else str(args.test_csv),
        "seed": args.seed,
        "generated_positive_sampling": positive_report,
        "generated_negative_sampling": {
            **negative_report,
            "source_priority": sorted(negative_priority),
        },
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "positive_holdout_ids": int(len(positive_holdout_ids)),
        "c3": c3_report,
        "outputs": outputs,
        "holdout_cluster_map_rows": int(len(cluster_map)),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    report_path = args.out_dir / "split_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()


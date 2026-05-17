"""
Deduplicate runs/merged/interactions_merged.csv by unordered pair (fpid_1, fpid_2).
Multi-threaded via polars (uses all cores).

For each unique pair:
  - resolved_label = 1 if ANY source reports label=1 ("positive wins" — synthetic
    negatives that accidentally land on a curated positive should not override it).
  - Provenance fields are aggregated: PPI_Source / Experimental_Method /
    Evidence_Type / Seq_Source / original_id1 / original_id2 are joined as
    semicolon-separated unique strings.
  - n_sources = number of distinct PPI_Source values.

Outputs:
  - runs/merged/interactions_merged_dedup.csv  (one row per unique pair)
  - runs/merged/pairs_merged_dedup.csv         (fpid_1, fpid_2, label) for H5 build
  - runs/merged/dedup_report.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "merged"


def main() -> None:
    print(f"polars threads: {pl.thread_pool_size()}")
    t0 = time.time()
    print("[load] interactions_merged.csv (lazy) ...")
    lf = pl.scan_csv(OUT / "interactions_merged.csv")

    # Canonicalise pair: sort fpid_1, fpid_2 so (A,B) and (B,A) collapse.
    # Also swap original_id1/2 so the order matches.
    lf = lf.with_columns(
        swap=pl.col("FPid_1") > pl.col("FPid_2"),
    ).with_columns([
        pl.when(pl.col("swap")).then(pl.col("FPid_2")).otherwise(pl.col("FPid_1")).alias("fp1"),
        pl.when(pl.col("swap")).then(pl.col("FPid_1")).otherwise(pl.col("FPid_2")).alias("fp2"),
        pl.when(pl.col("swap")).then(pl.col("original_id2")).otherwise(pl.col("original_id1")).alias("oid1"),
        pl.when(pl.col("swap")).then(pl.col("original_id1")).otherwise(pl.col("original_id2")).alias("oid2"),
    ]).drop(["FPid_1", "FPid_2", "original_id1", "original_id2", "swap"]) \
      .rename({"fp1": "FPid_1", "fp2": "FPid_2", "oid1": "original_id1", "oid2": "original_id2"})

    # GroupBy + aggregate. polars unique-list-then-join is efficient.
    print("[dedup] groupby + aggregate ...")
    g = (
        lf.group_by(["FPid_1", "FPid_2"])
        .agg([
            pl.col("label").max().alias("label"),                       # positive wins
            pl.col("label").n_unique().alias("labels_seen"),            # for conflict counting
            pl.col("PPI_Source").n_unique().alias("n_sources"),
            pl.col("PPI_Source").unique().sort().str.join(";").alias("PPI_Source"),
            pl.col("Seq_Source").unique().sort().str.join(";").alias("Seq_Source"),
            pl.col("Experimental_Method").unique().sort().str.join(";").alias("Experimental_Method"),
            pl.col("Evidence_Type").str.split(";").explode().unique().sort().str.join(";").alias("Evidence_Type"),
            pl.col("original_id1").unique().sort().str.join(";").alias("original_id1"),
            pl.col("original_id2").unique().sort().str.join(";").alias("original_id2"),
        ])
    )
    df = g.collect(streaming=False)
    t1 = time.time()
    print(f"[dedup] {len(df):,} unique pairs (took {t1 - t0:.1f}s)")

    n_conflicts = int(df.filter(pl.col("labels_seen") >= 2).height)
    print(f"[dedup] label conflicts resolved (positive wins): {n_conflicts:,}")

    df = df.drop("labels_seen").select(
        ["FPid_1", "FPid_2", "original_id1", "original_id2",
         "PPI_Source", "Seq_Source", "label",
         "Experimental_Method", "Evidence_Type", "n_sources"]
    )

    out_inter = OUT / "interactions_merged_dedup.csv"
    out_pairs = OUT / "pairs_merged_dedup.csv"
    print(f"[write] {out_inter}")
    df.write_csv(out_inter)
    print(f"[write] {out_pairs}")
    df.select(["FPid_1", "FPid_2", "label"]).rename(
        {"FPid_1": "fpid_1", "FPid_2": "fpid_2"}).write_csv(out_pairs)

    # Report
    n_src_dist = df.group_by("n_sources").len().sort("n_sources").to_dict(as_series=False)
    label_dist = df.group_by("label").len().sort("label").to_dict(as_series=False)
    rep = {
        "input_rows": int(pl.scan_csv(OUT / "interactions_merged.csv").select(pl.len()).collect().item()),
        "unique_pairs": int(df.height),
        "label_conflicts_resolved_positive_wins": n_conflicts,
        "n_sources_distribution": dict(zip(n_src_dist["n_sources"], n_src_dist["len"])),
        "label_distribution": dict(zip(label_dist["label"], label_dist["len"])),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    rep["duplicate_rows_removed"] = rep["input_rows"] - rep["unique_pairs"]
    (OUT / "dedup_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"[write] {OUT / 'dedup_report.json'}")
    print(f"\nDONE in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()

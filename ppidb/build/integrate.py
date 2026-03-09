"""
Core integration pipeline: parse all raw files → merge → deduplicate → output.

Usage
-----
    python -m ppidb.build.integrate \
        --raw-dir ./raw_data \
        --out-dir ./integrated \
        --taxon 9606

Output files
------------
    integrated/
        ppi_positive.parquet   — all positive interactions (merged, deduplicated)
        ppi_negative.parquet   — Negatome non-interactions
        ppi_all.parquet        — positive + negative with interaction_type label
        stats.json             — per-database and overall statistics
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable

import polars as pl

from .parsers import (
    parse_mitab,
    parse_biogrid,
    parse_string,
    parse_hippie,
    parse_bioplex,
    parse_corum,
    parse_negatome,
    parse_phosphositeplus,
    parse_omnipath,
    parse_complex_portal,
    parse_signor,
    parse_huri,
)

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = {
    "uniprot_a":         pl.Utf8,
    "uniprot_b":         pl.Utf8,
    "source_dbs":        pl.Utf8,   # pipe-separated list after merge
    "n_sources":         pl.Int32,
    "taxon_a":           pl.Utf8,
    "taxon_b":           pl.Utf8,
    "detection_methods": pl.Utf8,
    "interaction_type":  pl.Utf8,   # 'positive' | 'negative'
    "throughput_type":   pl.Utf8,   # 'LTP' | 'HTP' | 'both' | 'no_exp' | 'negative_sample'
}

# ── Database registry ─────────────────────────────────────────────────────────
# Each entry: (glob_pattern, parser_fn, extra_kwargs)
# The glob is relative to raw_dir.

DB_REGISTRY: list[tuple[str, Callable, dict]] = [
    # MITAB-based (IntAct, MINT, Reactome, InnateDB, VirHostNet)
    ("intact/intact.txt",                    lambda p, **kw: parse_mitab(p, "IntAct",       **kw), {"taxon_filter": "9606"}),
    ("mint_micluster.txt",                   lambda p, **kw: parse_mitab(p, "MINT",          **kw), {"taxon_filter": "9606"}),
    ("intact-micluster.txt",                 lambda p, **kw: parse_mitab(p, "MINT",          **kw), {"taxon_filter": "9606"}),
    ("reactome_human_interactions.txt",      lambda p, **kw: parse_mitab(p, "Reactome",      **kw), {"taxon_filter": "9606"}),
    ("innatedb_ppi.mitab.txt",               lambda p, **kw: parse_mitab(p, "InnateDB",      **kw), {"taxon_filter": "9606"}),
    ("virhostnet_all.mitab.txt",             lambda p, **kw: parse_mitab(p, "VirHostNet",    **kw), {}),
    # BioGRID
    ("BIOGRID-ALL-*.tab3.txt",               parse_biogrid,                                         {"taxon_filter": "9606"}),
    # STRING
    ("string_9606_links_full.txt",           lambda p, **kw: parse_string(p, **kw),                 {"min_score": 700}),
    # HIPPIE
    ("hippie_current.txt",                   parse_hippie,                                          {"min_score": 0.63}),
    # BioPlex
    ("bioplex_293t.tsv",                     parse_bioplex,                                         {}),
    ("bioplex_jurkat.tsv",                   parse_bioplex,                                         {}),
    # CORUM
    ("allComplexes*.txt",                    lambda p, **kw: parse_corum(p, **kw),                  {"taxon_filter": "9606"}),
    ("corum_allComplexes.txt",               lambda p, **kw: parse_corum(p, **kw),                  {"taxon_filter": "9606"}),
    # Negatome
    ("negatome_combined_stringent.txt",      parse_negatome,                                        {}),
    # PhosphoSitePlus
    ("phosphosite_kinase_substrate.txt",     lambda p, **kw: parse_phosphositeplus(p, **kw),        {"taxon_filter": "human"}),
    # OmniPath
    ("omnipath_omnipath.tsv",                parse_omnipath,                                        {}),
    ("omnipath_undirected.tsv",              parse_omnipath,                                        {}),
    # Complex Portal
    ("complex_portal_human.tsv",             parse_complex_portal,                                  {}),
    # SIGNOR
    ("signor_human.tsv",                     parse_signor,                                          {}),
    # HuRI
    ("huri.tsv",                             parse_huri,                                            {}),
]


# ── Per-database parsing ──────────────────────────────────────────────────────

def parse_all(raw_dir: Path) -> tuple[list[pl.DataFrame], dict]:
    """
    Scan raw_dir for known files and parse each one.

    Returns
    -------
    dfs : list of DataFrames (one per matched file)
    stats : dict with per-database row counts and timing
    """
    dfs: list[pl.DataFrame] = []
    stats: dict[str, dict] = {}

    for pattern, parser_fn, kwargs in DB_REGISTRY:
        matches = sorted(raw_dir.glob(pattern))
        if not matches:
            logger.debug(f"No files matched: {pattern}")
            continue

        for path in matches:
            db_label = _infer_db_label(path)
            logger.info(f"Parsing {db_label}: {path.name}")
            t0 = time.time()
            try:
                df = parser_fn(path, **kwargs)
                elapsed = time.time() - t0
                n = len(df)
                logger.info(f"  → {n:,} interactions in {elapsed:.1f}s")
                stats[db_label] = {"rows": n, "file": str(path.name), "elapsed_s": round(elapsed, 2)}
                if n > 0:
                    dfs.append(df)
            except Exception as e:
                logger.error(f"  ERROR parsing {path.name}: {e}")
                stats[db_label] = {"rows": 0, "file": str(path.name), "error": str(e)}

    return dfs, stats


def _infer_db_label(path: Path) -> str:
    """Derive a short DB label from filename."""
    name = path.stem.lower()
    for db in ["biogrid", "intact", "mint", "string", "hippie", "reactome",
               "signor", "bioplex", "innatedb", "phosphosite", "omnipath",
               "complex_portal", "corum", "negatome", "virhostnet", "huri",
               "dip", "hprd"]:
        if db in name:
            return db.upper()
    return path.stem


# ── Merge & deduplicate ───────────────────────────────────────────────────────

def merge_and_deduplicate(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """
    Concatenate all per-database DataFrames, then for each unique
    (uniprot_a, uniprot_b) pair:
      - merge source_dbs into a pipe-separated string
      - count n_sources
      - merge detection_methods (unique, pipe-separated)
      - keep interaction_type (positive > negative)
      - keep throughput_type (LTP > HTP > both > no_exp > negative_sample)

    Canonical ordering: uniprot_a < uniprot_b (alphabetically).
    """
    if not dfs:
        return pl.DataFrame(schema=SCHEMA)

    combined = pl.concat(dfs, how="diagonal_relaxed")

    # Ensure canonical ordering
    combined = combined.with_columns([
        pl.when(pl.col("uniprot_a") > pl.col("uniprot_b"))
          .then(pl.col("uniprot_b"))
          .otherwise(pl.col("uniprot_a"))
          .alias("uniprot_a"),
        pl.when(pl.col("uniprot_a") > pl.col("uniprot_b"))
          .then(pl.col("uniprot_a"))
          .otherwise(pl.col("uniprot_b"))
          .alias("uniprot_b"),
    ])

    # Group by protein pair
    merged = (
        combined
        .group_by(["uniprot_a", "uniprot_b"])
        .agg([
            # Merge source databases
            pl.col("source_dbs")
              .str.split("|")
              .explode()
              .unique()
              .sort()
              .str.join("|")
              .alias("source_dbs"),
            # Count unique sources
            pl.col("source_dbs")
              .str.split("|")
              .explode()
              .n_unique()
              .alias("n_sources"),
            # Merge detection methods (unique, truncated)
            pl.col("detection_methods")
              .str.split("|")
              .explode()
              .filter(pl.col("detection_methods") != "")
              .unique()
              .sort()
              .str.join("|")
              .alias("detection_methods"),
            # Taxon: take first non-empty
            pl.col("taxon_a").filter(pl.col("taxon_a") != "").first().alias("taxon_a"),
            pl.col("taxon_b").filter(pl.col("taxon_b") != "").first().alias("taxon_b"),
            # interaction_type: positive wins over negative
            pl.when(pl.col("interaction_type").str.contains("positive").any())
              .then(pl.lit("positive"))
              .otherwise(pl.lit("negative"))
              .first()
              .alias("interaction_type"),
            # throughput_type: priority order
            pl.col("throughput_type").first().alias("throughput_type"),
        ])
    )

    # Apply throughput priority: LTP > HTP > both > no_exp > negative_sample
    # Re-derive from merged source list
    merged = _resolve_throughput(merged, combined)

    return merged.sort(["uniprot_a", "uniprot_b"])


def _resolve_throughput(merged: pl.DataFrame, combined: pl.DataFrame) -> pl.DataFrame:
    """
    For each pair, determine the best throughput label from all contributing rows.
    Priority: LTP > HTP > both > no_exp > negative_sample
    """
    PRIORITY = {"LTP": 4, "HTP": 3, "both": 2, "no_exp": 1, "negative_sample": 0}
    PRIORITY_INV = {v: k for k, v in PRIORITY.items()}

    # Map string labels → integer scores using pl.when/then chains (no deprecated default=)
    score_expr = pl.lit(0).alias("tp_score")
    for label, score in PRIORITY.items():
        score_expr = (
            pl.when(pl.col("throughput_type") == label)
              .then(pl.lit(score))
              .otherwise(score_expr)
        )

    tp_best = (
        combined
        .select(["uniprot_a", "uniprot_b", "throughput_type"])
        .with_columns(score_expr.alias("tp_score"))
        .group_by(["uniprot_a", "uniprot_b"])
        .agg(pl.col("tp_score").max().alias("tp_score_max"))
    )

    # Map integer scores back → string labels
    label_expr = pl.lit("no_exp").alias("throughput_type")
    for score, label in PRIORITY_INV.items():
        label_expr = (
            pl.when(pl.col("tp_score_max") == score)
              .then(pl.lit(label))
              .otherwise(label_expr)
        )

    tp_best = (
        tp_best
        .with_columns(label_expr.alias("throughput_type"))
        .drop("tp_score_max")
    )

    merged = merged.drop("throughput_type").join(
        tp_best, on=["uniprot_a", "uniprot_b"], how="left"
    )
    return merged


# ── Quality filters ───────────────────────────────────────────────────────────

def apply_quality_filters(
    df: pl.DataFrame,
    min_sources: int = 1,
    taxon: str | None = "9606",
    remove_self_loops: bool = True,
) -> pl.DataFrame:
    """
    Apply quality filters to the merged interaction table.

    Parameters
    ----------
    df : pl.DataFrame
    min_sources : int
        Minimum number of source databases (default 1 = keep all).
    taxon : str | None
        If set, keep only pairs where both proteins are from this taxon.
        Set to None to keep all species.
    remove_self_loops : bool
        Remove interactions where uniprot_a == uniprot_b.
    """
    if remove_self_loops:
        df = df.filter(pl.col("uniprot_a") != pl.col("uniprot_b"))

    if min_sources > 1:
        df = df.filter(pl.col("n_sources") >= min_sources)

    if taxon:
        df = df.filter(
            (pl.col("taxon_a").is_in([taxon, ""])) &
            (pl.col("taxon_b").is_in([taxon, ""]))
        )

    return df


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(
    merged: pl.DataFrame,
    per_db_stats: dict,
) -> dict:
    """Compute summary statistics for the integrated dataset."""
    pos = merged.filter(pl.col("interaction_type") == "positive")
    neg = merged.filter(pl.col("interaction_type") == "negative")

    # Source coverage histogram
    src_hist = (
        pos.group_by("n_sources")
           .agg(pl.len().alias("count"))
           .sort("n_sources")
           .to_dicts()
    )

    # Throughput breakdown
    tp_counts = (
        pos.group_by("throughput_type")
           .agg(pl.len().alias("count"))
           .sort("throughput_type")
           .to_dicts()
    )

    return {
        "total_positive": len(pos),
        "total_negative": len(neg),
        "unique_proteins": len(
            set(pos["uniprot_a"].to_list()) | set(pos["uniprot_b"].to_list())
        ),
        "source_coverage_histogram": src_hist,
        "throughput_breakdown": tp_counts,
        "per_database": per_db_stats,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_integration(
    raw_dir: Path,
    out_dir: Path,
    taxon: str = "9606",
    min_sources: int = 1,
    string_min_score: int = 700,
    hippie_min_score: float = 0.63,
) -> dict:
    """
    Full integration pipeline.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw downloaded files.
    out_dir : Path
        Output directory for integrated files.
    taxon : str
        NCBI taxon ID to filter (default '9606' = human).
    min_sources : int
        Minimum number of databases an interaction must appear in.
    string_min_score : int
        STRING combined score threshold (0–1000).
    hippie_min_score : float
        HIPPIE confidence score threshold (0–1).

    Returns
    -------
    dict
        Summary statistics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ppidb integration pipeline")
    logger.info(f"  raw_dir    : {raw_dir}")
    logger.info(f"  out_dir    : {out_dir}")
    logger.info(f"  taxon      : {taxon}")
    logger.info(f"  min_sources: {min_sources}")
    logger.info("=" * 70)

    # 1. Parse all databases
    dfs, per_db_stats = parse_all(raw_dir)
    if not dfs:
        logger.error("No data parsed. Check raw_dir and file names.")
        return {}

    # 2. Merge and deduplicate
    logger.info("\nMerging and deduplicating...")
    merged = merge_and_deduplicate(dfs)
    logger.info(f"  Raw merged: {len(merged):,} pairs")

    # 3. Quality filters
    merged = apply_quality_filters(merged, min_sources=min_sources, taxon=taxon)
    logger.info(f"  After filters: {len(merged):,} pairs")

    # 4. Split positive / negative
    pos = merged.filter(pl.col("interaction_type") == "positive")
    neg = merged.filter(pl.col("interaction_type") == "negative")
    logger.info(f"  Positive: {len(pos):,}  |  Negative: {len(neg):,}")

    # 5. Write outputs
    pos.write_parquet(out_dir / "ppi_positive.parquet")
    neg.write_parquet(out_dir / "ppi_negative.parquet")
    merged.write_parquet(out_dir / "ppi_all.parquet")
    logger.info(f"  Written: ppi_positive.parquet, ppi_negative.parquet, ppi_all.parquet")

    # Also write TSV for easy inspection
    pos.write_csv(out_dir / "ppi_positive.tsv", separator="\t")
    neg.write_csv(out_dir / "ppi_negative.tsv", separator="\t")

    # 6. Statistics
    stats = compute_stats(merged, per_db_stats)
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Written: stats.json")

    _print_summary(stats)
    return stats


def _print_summary(stats: dict) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("INTEGRATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Positive interactions : {stats['total_positive']:>10,}")
    logger.info(f"  Negative interactions : {stats['total_negative']:>10,}")
    logger.info(f"  Unique proteins       : {stats['unique_proteins']:>10,}")
    logger.info("\n  Per-database row counts:")
    for db, info in sorted(stats["per_database"].items()):
        status = f"{info.get('rows', 0):>8,} rows"
        if "error" in info:
            status = f"  ERROR: {info['error']}"
        logger.info(f"    {db:<25} {status}")
    logger.info("\n  Source coverage (n_sources → n_pairs):")
    for row in stats.get("source_coverage_histogram", []):
        logger.info(f"    {row['n_sources']:>3} source(s): {row['count']:>8,} pairs")
    logger.info("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Integrate PPI databases into a unified parquet dataset."
    )
    parser.add_argument("--raw-dir",   required=True, type=Path, help="Raw data directory")
    parser.add_argument("--out-dir",   required=True, type=Path, help="Output directory")
    parser.add_argument("--taxon",     default="9606",  help="NCBI taxon filter (default: 9606)")
    parser.add_argument("--min-sources", default=1, type=int,
                        help="Min databases an interaction must appear in (default: 1)")
    parser.add_argument("--string-min-score", default=700, type=int,
                        help="STRING combined score threshold (default: 700)")
    parser.add_argument("--hippie-min-score", default=0.63, type=float,
                        help="HIPPIE confidence threshold (default: 0.63)")
    args = parser.parse_args()

    run_integration(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        taxon=args.taxon,
        min_sources=args.min_sources,
        string_min_score=args.string_min_score,
        hippie_min_score=args.hippie_min_score,
    )


if __name__ == "__main__":
    main()

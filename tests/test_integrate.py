"""
Integration tests for ppidb.build.integrate.
Tests the merge/dedup/filter pipeline using synthetic DataFrames.
No file I/O or network access required.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from ppidb.build.integrate import (
    merge_and_deduplicate,
    apply_quality_filters,
    compute_stats,
    run_integration,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_df(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal ppidb-schema DataFrame from a list of dicts."""
    defaults = {
        "source_dbs": "TestDB",
        "taxon_a": "9606",
        "taxon_b": "9606",
        "detection_methods": "two hybrid",
        "interaction_type": "positive",
        "throughput_type": "LTP",
        "n_sources": 1,
    }
    full_rows = [{**defaults, **r} for r in rows]
    return pl.DataFrame(full_rows)


# ── merge_and_deduplicate ─────────────────────────────────────────────────────

class TestMergeAndDeduplicate:

    def test_single_df_passthrough(self):
        df = make_df([{"uniprot_a": "P12345", "uniprot_b": "Q67890"}])
        merged = merge_and_deduplicate([df])
        assert len(merged) == 1

    def test_deduplication_across_dbs(self):
        """Same pair from two databases → 1 row, n_sources=2."""
        df1 = make_df([{"uniprot_a": "P12345", "uniprot_b": "Q67890", "source_dbs": "IntAct"}])
        df2 = make_df([{"uniprot_a": "P12345", "uniprot_b": "Q67890", "source_dbs": "BioGRID"}])
        merged = merge_and_deduplicate([df1, df2])
        assert len(merged) == 1
        assert merged["n_sources"][0] == 2

    def test_source_dbs_merged(self):
        df1 = make_df([{"uniprot_a": "P12345", "uniprot_b": "Q67890", "source_dbs": "IntAct"}])
        df2 = make_df([{"uniprot_a": "P12345", "uniprot_b": "Q67890", "source_dbs": "BioGRID"}])
        merged = merge_and_deduplicate([df1, df2])
        src = merged["source_dbs"][0]
        assert "IntAct" in src
        assert "BioGRID" in src

    def test_canonical_ordering_enforced(self):
        """Input with uniprot_a > uniprot_b should be reordered."""
        df = make_df([{"uniprot_a": "Q67890", "uniprot_b": "P12345"}])
        merged = merge_and_deduplicate([df])
        assert merged["uniprot_a"][0] == "P12345"
        assert merged["uniprot_b"][0] == "Q67890"

    def test_different_pairs_kept_separate(self):
        df = make_df([
            {"uniprot_a": "P12345", "uniprot_b": "Q67890"},
            {"uniprot_a": "A11111", "uniprot_b": "B22222"},
        ])
        merged = merge_and_deduplicate([df])
        assert len(merged) == 2

    def test_positive_wins_over_negative(self):
        """If a pair appears as positive in one DB and negative in another,
        the merged result should be positive."""
        df_pos = make_df([{
            "uniprot_a": "P12345", "uniprot_b": "Q67890",
            "source_dbs": "IntAct", "interaction_type": "positive",
        }])
        df_neg = make_df([{
            "uniprot_a": "P12345", "uniprot_b": "Q67890",
            "source_dbs": "Negatome", "interaction_type": "negative",
        }])
        merged = merge_and_deduplicate([df_pos, df_neg])
        assert len(merged) == 1
        assert merged["interaction_type"][0] == "positive"

    def test_throughput_priority_ltp_over_htp(self):
        df_htp = make_df([{
            "uniprot_a": "P12345", "uniprot_b": "Q67890",
            "source_dbs": "HuRI", "throughput_type": "HTP",
        }])
        df_ltp = make_df([{
            "uniprot_a": "P12345", "uniprot_b": "Q67890",
            "source_dbs": "IntAct", "throughput_type": "LTP",
        }])
        merged = merge_and_deduplicate([df_htp, df_ltp])
        assert merged["throughput_type"][0] == "LTP"

    def test_empty_list_returns_empty_df(self):
        merged = merge_and_deduplicate([])
        assert len(merged) == 0
        assert "uniprot_a" in merged.columns

    def test_n_sources_counts_unique_dbs(self):
        """Three rows from same DB should count as n_sources=1."""
        rows = [
            {"uniprot_a": "P12345", "uniprot_b": "Q67890", "source_dbs": "IntAct"},
            {"uniprot_a": "P12345", "uniprot_b": "Q67890", "source_dbs": "IntAct"},
            {"uniprot_a": "P12345", "uniprot_b": "Q67890", "source_dbs": "IntAct"},
        ]
        df = make_df(rows)
        merged = merge_and_deduplicate([df])
        assert merged["n_sources"][0] == 1

    def test_output_sorted(self):
        df = make_df([
            {"uniprot_a": "Z99999", "uniprot_b": "Z99998"},
            {"uniprot_a": "A00001", "uniprot_b": "A00002"},
        ])
        merged = merge_and_deduplicate([df])
        assert merged["uniprot_a"][0] == "A00001"
        assert merged["uniprot_a"][1] == "Z99998"


# ── apply_quality_filters ─────────────────────────────────────────────────────

class TestApplyQualityFilters:

    def test_min_sources_filter(self):
        df = make_df([
            {"uniprot_a": "P12345", "uniprot_b": "Q67890", "n_sources": 1},
            {"uniprot_a": "A11111", "uniprot_b": "B22222", "n_sources": 3},
        ])
        filtered = apply_quality_filters(df, min_sources=2)
        assert len(filtered) == 1
        assert filtered["uniprot_a"][0] == "A11111"

    def test_taxon_filter_human_only(self):
        df = make_df([
            {"uniprot_a": "P12345", "uniprot_b": "Q67890", "taxon_a": "9606", "taxon_b": "9606"},
            {"uniprot_a": "A11111", "uniprot_b": "B22222", "taxon_a": "10090", "taxon_b": "10090"},
        ])
        filtered = apply_quality_filters(df, taxon="9606")
        assert len(filtered) == 1

    def test_taxon_filter_none_keeps_all(self):
        df = make_df([
            {"uniprot_a": "P12345", "uniprot_b": "Q67890", "taxon_a": "9606", "taxon_b": "9606"},
            {"uniprot_a": "A11111", "uniprot_b": "B22222", "taxon_a": "10090", "taxon_b": "10090"},
        ])
        filtered = apply_quality_filters(df, taxon=None)
        assert len(filtered) == 2

    def test_self_loop_removal(self):
        df = make_df([
            {"uniprot_a": "P12345", "uniprot_b": "P12345"},
            {"uniprot_a": "P12345", "uniprot_b": "Q67890"},
        ])
        filtered = apply_quality_filters(df, remove_self_loops=True)
        assert len(filtered) == 1

    def test_self_loop_kept_when_disabled(self):
        df = make_df([
            {"uniprot_a": "P12345", "uniprot_b": "P12345"},
        ])
        filtered = apply_quality_filters(df, remove_self_loops=False)
        assert len(filtered) == 1

    def test_empty_taxon_passes_filter(self):
        """Rows with empty taxon should pass the human filter (unknown species)."""
        df = make_df([
            {"uniprot_a": "P12345", "uniprot_b": "Q67890", "taxon_a": "", "taxon_b": ""},
        ])
        filtered = apply_quality_filters(df, taxon="9606")
        assert len(filtered) == 1

    def test_combined_filters(self):
        df = make_df([
            {"uniprot_a": "P12345", "uniprot_b": "Q67890",
             "n_sources": 3, "taxon_a": "9606", "taxon_b": "9606"},
            {"uniprot_a": "A11111", "uniprot_b": "B22222",
             "n_sources": 1, "taxon_a": "9606", "taxon_b": "9606"},
            {"uniprot_a": "C33333", "uniprot_b": "D44444",
             "n_sources": 3, "taxon_a": "10090", "taxon_b": "10090"},
        ])
        filtered = apply_quality_filters(df, min_sources=2, taxon="9606")
        assert len(filtered) == 1
        assert filtered["uniprot_a"][0] == "P12345"


# ── compute_stats ─────────────────────────────────────────────────────────────

class TestComputeStats:

    def test_basic_counts(self):
        df = make_df([
            {"uniprot_a": "P12345", "uniprot_b": "Q67890",
             "interaction_type": "positive", "n_sources": 2},
            {"uniprot_a": "A11111", "uniprot_b": "B22222",
             "interaction_type": "negative", "n_sources": 1},
        ])
        stats = compute_stats(df, per_db_stats={})
        assert stats["total_positive"] == 1
        assert stats["total_negative"] == 1

    def test_unique_proteins(self):
        df = make_df([
            {"uniprot_a": "P12345", "uniprot_b": "Q67890", "interaction_type": "positive"},
            {"uniprot_a": "P12345", "uniprot_b": "A11111", "interaction_type": "positive"},
        ])
        stats = compute_stats(df, per_db_stats={})
        # P12345, Q67890, A11111 = 3 unique proteins
        assert stats["unique_proteins"] == 3

    def test_source_coverage_histogram(self):
        df = make_df([
            {"uniprot_a": "P12345", "uniprot_b": "Q67890",
             "interaction_type": "positive", "n_sources": 1},
            {"uniprot_a": "A11111", "uniprot_b": "B22222",
             "interaction_type": "positive", "n_sources": 3},
        ])
        stats = compute_stats(df, per_db_stats={})
        hist = {row["n_sources"]: row["count"]
                for row in stats["source_coverage_histogram"]}
        assert hist[1] == 1
        assert hist[3] == 1

    def test_per_db_stats_passthrough(self):
        df = make_df([{"uniprot_a": "P12345", "uniprot_b": "Q67890"}])
        per_db = {"IntAct": {"rows": 100, "file": "intact.txt"}}
        stats = compute_stats(df, per_db_stats=per_db)
        assert stats["per_database"]["IntAct"]["rows"] == 100


# ── run_integration (end-to-end with real files) ──────────────────────────────

class TestRunIntegration:

    def _write_negatome(self, raw_dir: Path) -> None:
        (raw_dir / "negatome_combined_stringent.txt").write_text(
            "P12345\tQ11111\n"
            "Q67890\tA22222\n",
            encoding="utf-8",
        )

    def _write_huri(self, raw_dir: Path) -> None:
        # HuRI: two pairs — P12345↔Q67890 and A11111↔B22222
        (raw_dir / "huri.tsv").write_text(
            "P12345\tQ67890\n"
            "A11111\tB22222\n",
            encoding="utf-8",
        )

    def _write_hippie(self, raw_dir: Path) -> None:
        # HIPPIE: P12345↔Q67890 (shared with HuRI) + C33333↔D44444 (unique to HIPPIE)
        (raw_dir / "hippie_current.txt").write_text(
            "P12345\t1\tQ67890\t2\t0.80\tsource1\n"
            "C33333\t3\tD44444\t4\t0.75\tsource2\n",
            encoding="utf-8",
        )

    def test_basic_pipeline(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        self._write_huri(raw_dir)
        self._write_hippie(raw_dir)

        stats = run_integration(raw_dir=raw_dir, out_dir=out_dir, taxon=None)

        assert (out_dir / "ppi_positive.parquet").exists()
        assert (out_dir / "ppi_negative.parquet").exists()
        assert (out_dir / "ppi_all.parquet").exists()
        assert (out_dir / "stats.json").exists()
        assert stats["total_positive"] > 0

    def test_negative_interactions_written(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        self._write_negatome(raw_dir)

        stats = run_integration(raw_dir=raw_dir, out_dir=out_dir, taxon=None)
        assert stats["total_negative"] > 0

    def test_deduplication_across_files(self, tmp_path):
        """Same pair in HuRI and HIPPIE → 1 row with n_sources=2."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        # Both files contain P12345 ↔ Q67890
        self._write_huri(raw_dir)
        self._write_hippie(raw_dir)

        run_integration(raw_dir=raw_dir, out_dir=out_dir, taxon=None)
        pos = pl.read_parquet(out_dir / "ppi_positive.parquet")
        pair = pos.filter(
            (pl.col("uniprot_a") == "P12345") & (pl.col("uniprot_b") == "Q67890")
        )
        assert len(pair) == 1
        assert pair["n_sources"][0] == 2

    def test_min_sources_filter(self, tmp_path):
        """With min_sources=2, only the pair in both HuRI and HIPPIE survives."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        self._write_huri(raw_dir)
        self._write_hippie(raw_dir)

        stats = run_integration(
            raw_dir=raw_dir, out_dir=out_dir, taxon=None, min_sources=2
        )
        pos = pl.read_parquet(out_dir / "ppi_positive.parquet")
        # Only P12345-Q67890 appears in both files
        assert len(pos) == 1
        assert pos["uniprot_a"][0] == "P12345"

    def test_stats_json_valid(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        self._write_huri(raw_dir)

        run_integration(raw_dir=raw_dir, out_dir=out_dir, taxon=None)
        with open(out_dir / "stats.json") as f:
            stats = json.load(f)
        assert "total_positive" in stats
        assert "unique_proteins" in stats
        assert "per_database" in stats

    def test_empty_raw_dir_returns_empty(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        # No files → should return empty dict (no crash)
        stats = run_integration(raw_dir=raw_dir, out_dir=out_dir, taxon=None)
        assert stats == {}

    def test_output_schema_columns(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        self._write_huri(raw_dir)

        run_integration(raw_dir=raw_dir, out_dir=out_dir, taxon=None)
        pos = pl.read_parquet(out_dir / "ppi_positive.parquet")
        required_cols = {
            "uniprot_a", "uniprot_b", "source_dbs", "n_sources",
            "interaction_type", "throughput_type",
        }
        assert required_cols.issubset(set(pos.columns))

    def test_no_self_loops_in_output(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "out"
        raw_dir.mkdir()
        # Write a file with a self-loop
        (raw_dir / "huri.tsv").write_text(
            "P12345\tP12345\n"
            "P12345\tQ67890\n",
            encoding="utf-8",
        )
        run_integration(raw_dir=raw_dir, out_dir=out_dir, taxon=None)
        pos = pl.read_parquet(out_dir / "ppi_positive.parquet")
        self_loops = pos.filter(pl.col("uniprot_a") == pl.col("uniprot_b"))
        assert len(self_loops) == 0

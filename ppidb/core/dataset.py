"""
PPIDataset — the central data container for ppidb.

Design principles:
  - Polars LazyFrame backend for memory-efficient operations on 14M+ pairs
  - Fluent API: all filter/transform methods return a new PPIDataset (immutable)
  - Lazy evaluation: computation deferred until .collect() or iteration
  - The .filter accessor provides a composable filter chain
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional, Union

import polars as pl

from ppidb.core.pair import PPIPair


# ── Schema constants ──────────────────────────────────────────────────────────
REQUIRED_COLUMNS = {
    "uniprot_a", "uniprot_b", "source_dbs", "n_sources",
    "taxon_a", "taxon_b", "detection_methods",
    "interaction_type", "throughput_type",
}

TAXON_NAMES = {
    "9606": "Human", "559292": "S. cerevisiae", "10090": "Mouse",
    "7227": "D. melanogaster", "3702": "A. thaliana", "6239": "C. elegans",
    "284812": "S. pombe", "2697049": "SARS-CoV-2", "10116": "Rat",
}


class PPIDataset:
    """
    Central container for PPI data.

    Parameters
    ----------
    data : pl.LazyFrame | pl.DataFrame | str | Path
        Source data. Accepts a Polars frame or a path to a .parquet / .csv file.

    Examples
    --------
    >>> ds = PPIDataset.load("ppi_final_uniprot_v6.parquet")
    >>> ds
    PPIDataset(14_486_613 pairs, 39 sources)

    >>> # Fluent filter chain
    >>> subset = ds.filter.by_species("9606").filter.by_throughput("LTP").filter.by_min_sources(2)
    >>> len(subset)
    ...

    >>> # Iterate as PPIPair objects
    >>> for pair in subset.head(10):
    ...     print(pair)
    """

    def __init__(self, data: Union[pl.LazyFrame, pl.DataFrame, str, Path]):
        if isinstance(data, (str, Path)):
            data = self._read_file(Path(data))
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        self._lf: pl.LazyFrame = data
        self._filter_accessor: Optional["FilterAccessor"] = None

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PPIDataset":
        """Load from a .parquet or .csv file."""
        return cls(Path(path))

    @staticmethod
    def _read_file(path: Path) -> pl.LazyFrame:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pl.scan_parquet(path)
        elif suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            return pl.scan_csv(path, separator=sep)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .parquet or .csv/.tsv")

    # ── Filter accessor ───────────────────────────────────────────────────────

    @property
    def filter(self) -> "FilterAccessor":
        """
        Composable filter accessor.

        Usage:
            ds.filter.by_species("9606")
            ds.filter.by_throughput("LTP").filter.by_min_sources(2)
        """
        from ppidb.filter.accessor import FilterAccessor
        return FilterAccessor(self)

    # ── Core operations ───────────────────────────────────────────────────────

    def _apply(self, expr: pl.Expr) -> "PPIDataset":
        """Apply a Polars filter expression, returning a new PPIDataset."""
        return PPIDataset(self._lf.filter(expr))

    def _apply_lf(self, lf: pl.LazyFrame) -> "PPIDataset":
        """Wrap a transformed LazyFrame as a new PPIDataset."""
        return PPIDataset(lf)

    def collect(self) -> pl.DataFrame:
        """Materialize the lazy frame into a DataFrame."""
        return self._lf.collect()

    def head(self, n: int = 10) -> "PPIDataset":
        """Return first n rows."""
        return PPIDataset(self._lf.head(n))

    def sample(self, n: int, seed: int = 42) -> "PPIDataset":
        """Random sample of n pairs."""
        return PPIDataset(self._lf.collect().sample(n=n, seed=seed).lazy())

    def __len__(self) -> int:
        return self._lf.select(pl.len()).collect().item()

    def __iter__(self) -> Iterator[PPIPair]:
        for row in self._lf.collect().to_dicts():
            yield PPIPair.from_dict(row)

    def __repr__(self) -> str:
        try:
            n = len(self)
            sources = self._lf.select(
                pl.col("source_dbs").str.split("|").explode().n_unique()
            ).collect().item()
            return f"PPIDataset({n:,} pairs, {sources} sources)"
        except Exception:
            return "PPIDataset(<unevaluated>)"

    # ── Protein accessors ─────────────────────────────────────────────────────

    def proteins(self) -> list[str]:
        """Return sorted list of all unique UniProt accessions."""
        df = self._lf.select(
            pl.concat([pl.col("uniprot_a"), pl.col("uniprot_b")]).unique()
        ).collect()
        return sorted(df.to_series().to_list())

    def n_proteins(self) -> int:
        """Number of unique proteins."""
        return len(self.proteins())

    # ── Export ────────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pl.DataFrame:
        """Collect and return as Polars DataFrame."""
        return self.collect()

    def to_pandas(self):
        """Collect and return as pandas DataFrame."""
        return self.collect().to_pandas()

    def save(self, path: Union[str, Path], format: str = "parquet") -> None:
        """
        Save dataset to disk.

        Parameters
        ----------
        path : str | Path
        format : 'parquet' | 'csv' | 'tsv'
        """
        path = Path(path)
        df = self.collect()
        if format == "parquet":
            df.write_parquet(path)
        elif format == "csv":
            df.write_csv(path)
        elif format == "tsv":
            df.write_csv(path, separator="\t")
        else:
            raise ValueError(f"Unknown format: {format}")
        print(f"Saved {len(df):,} pairs to {path}")

    # ── Summary statistics ────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a summary statistics dict."""
        df = self.collect()
        n = len(df)

        species_counts = (
            df.group_by("taxon_a").agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(5)
            .to_dicts()
        )

        source_counts = {}
        for row in df["source_dbs"].to_list():
            if row:
                for s in row.split("|"):
                    source_counts[s.strip()] = source_counts.get(s.strip(), 0) + 1
        top_sources = sorted(source_counts.items(), key=lambda x: -x[1])[:5]

        throughput_counts = df["throughput_type"].value_counts().to_dicts()
        interaction_counts = df["interaction_type"].value_counts().to_dicts()

        return {
            "n_pairs": n,
            "n_proteins": self.n_proteins(),
            "n_sources": len(source_counts),
            "top_species": [
                {**r, "name": TAXON_NAMES.get(str(r["taxon_a"]), r["taxon_a"])}
                for r in species_counts
            ],
            "top_sources": [{"source": k, "count": v} for k, v in top_sources],
            "throughput_distribution": throughput_counts,
            "interaction_type_distribution": interaction_counts,
            "n_sources_stats": {
                "mean": df["n_sources"].mean(),
                "median": df["n_sources"].median(),
                "max": df["n_sources"].max(),
            },
        }

    def describe(self) -> None:
        """Print a human-readable summary."""
        s = self.summary()
        print(f"{'='*50}")
        print(f"  PPIDataset Summary")
        print(f"{'='*50}")
        print(f"  Pairs:    {s['n_pairs']:>12,}")
        print(f"  Proteins: {s['n_proteins']:>12,}")
        print(f"  Sources:  {s['n_sources']:>12,}")
        print(f"\n  Top species:")
        for sp in s["top_species"]:
            print(f"    {sp.get('name', sp['taxon_a']):<25} {sp['count']:>10,}")
        print(f"\n  Top databases:")
        for src in s["top_sources"]:
            print(f"    {src['source']:<25} {src['count']:>10,}")
        print(f"\n  Throughput type:")
        for t in s["throughput_distribution"]:
            print(f"    {t['throughput_type']:<20} {t['count']:>10,}")
        print(f"\n  Interaction type:")
        for t in s["interaction_type_distribution"]:
            print(f"    {t['interaction_type']:<20} {t['count']:>10,}")
        print(f"{'='*50}")

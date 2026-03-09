"""
FilterAccessor — composable, chainable filter interface for PPIDataset.

All methods return a new PPIDataset, enabling fluent chaining:

    ds.filter.by_species("9606")
             .filter.by_throughput("LTP")
             .filter.by_min_sources(2)
             .filter.positives_only()

Design note: FilterAccessor is a thin proxy that delegates to PPIDataset._apply().
It never stores state — each call produces a fresh PPIDataset with the filter applied.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Union, List

import polars as pl

if TYPE_CHECKING:
    from ppidb.core.dataset import PPIDataset


class FilterAccessor:
    """
    Composable filter accessor attached to a PPIDataset via the `.filter` property.

    All methods return a new PPIDataset (immutable, lazy).
    """

    def __init__(self, dataset: "PPIDataset"):
        self._ds = dataset

    # ── Species ───────────────────────────────────────────────────────────────

    def by_species(
        self,
        taxon: Union[str, int, List[Union[str, int]]],
        require_both: bool = True,
    ) -> "PPIDataset":
        """
        Filter by NCBI taxonomy ID.

        Parameters
        ----------
        taxon : str | int | list
            Taxonomy ID(s). Common values: '9606' (human), '10090' (mouse),
            '559292' (S. cerevisiae), '7227' (D. melanogaster).
        require_both : bool
            If True (default), both proteins must belong to the taxon.
            If False, at least one protein must match.

        Examples
        --------
        >>> ds.filter.by_species("9606")           # human only
        >>> ds.filter.by_species(["9606", "10090"]) # human or mouse
        >>> ds.filter.by_species("9606", require_both=False)  # at least one human
        """
        if not isinstance(taxon, list):
            taxon = [taxon]
        taxon_strs = [str(t) for t in taxon]

        if require_both:
            expr = pl.col("taxon_a").is_in(taxon_strs) & pl.col("taxon_b").is_in(taxon_strs)
        else:
            expr = pl.col("taxon_a").is_in(taxon_strs) | pl.col("taxon_b").is_in(taxon_strs)

        return self._ds._apply(expr)

    # ── Throughput ────────────────────────────────────────────────────────────

    def by_throughput(
        self,
        throughput: Union[str, List[str]],
    ) -> "PPIDataset":
        """
        Filter by experimental throughput type.

        Parameters
        ----------
        throughput : str | list
            One or more of: 'LTP', 'HTP', 'both', 'no_exp', 'negative_sample'.
            Shorthand: 'experimental' = ['LTP', 'HTP', 'both']

        Examples
        --------
        >>> ds.filter.by_throughput("LTP")
        >>> ds.filter.by_throughput(["LTP", "both"])   # LTP-validated pairs
        >>> ds.filter.by_throughput("experimental")    # any experimental evidence
        """
        SHORTHANDS = {
            "experimental": ["LTP", "HTP", "both"],
            "ltp_validated": ["LTP", "both"],
        }
        if isinstance(throughput, str) and throughput in SHORTHANDS:
            throughput = SHORTHANDS[throughput]
        if not isinstance(throughput, list):
            throughput = [throughput]
        throughput = [t.upper() if t not in ("no_exp", "negative_sample") else t
                      for t in throughput]
        return self._ds._apply(pl.col("throughput_type").is_in(throughput))

    # ── Evidence strength ─────────────────────────────────────────────────────

    def by_min_sources(self, n: int) -> "PPIDataset":
        """
        Keep only pairs supported by at least n independent databases.

        Parameters
        ----------
        n : int
            Minimum number of supporting sources.
            Recommended: n=2 for high-confidence, n=4 for very high confidence.

        Examples
        --------
        >>> ds.filter.by_min_sources(2)   # supported by ≥2 databases
        """
        return self._ds._apply(pl.col("n_sources") >= n)

    def by_max_sources(self, n: int) -> "PPIDataset":
        """Keep only pairs supported by at most n databases."""
        return self._ds._apply(pl.col("n_sources") <= n)

    def by_sources_range(self, min_n: int, max_n: int) -> "PPIDataset":
        """Keep pairs with n_sources in [min_n, max_n]."""
        return self._ds._apply(
            (pl.col("n_sources") >= min_n) & (pl.col("n_sources") <= max_n)
        )

    # ── Source database ───────────────────────────────────────────────────────

    def by_database(
        self,
        databases: Union[str, List[str]],
        mode: str = "any",
    ) -> "PPIDataset":
        """
        Filter by source database(s).

        Parameters
        ----------
        databases : str | list
            Database name(s), e.g. 'BioGRID', ['IntAct', 'BioGRID'].
        mode : 'any' | 'all'
            'any': pair must appear in at least one of the databases.
            'all': pair must appear in all specified databases.

        Examples
        --------
        >>> ds.filter.by_database("BioGRID")
        >>> ds.filter.by_database(["IntAct", "BioGRID"], mode="all")
        """
        if isinstance(databases, str):
            databases = [databases]

        if mode == "any":
            expr = pl.lit(False)
            for db in databases:
                expr = expr | pl.col("source_dbs").str.contains(db)
        elif mode == "all":
            expr = pl.lit(True)
            for db in databases:
                expr = expr & pl.col("source_dbs").str.contains(db)
        else:
            raise ValueError(f"mode must be 'any' or 'all', got '{mode}'")

        return self._ds._apply(expr)

    # ── Detection method ──────────────────────────────────────────────────────

    def by_method(
        self,
        methods: Union[str, List[str]],
        mode: str = "any",
    ) -> "PPIDataset":
        """
        Filter by experimental detection method.

        Parameters
        ----------
        methods : str | list
            Method name(s), e.g. 'two hybrid', 'affinity chromatography technology'.
            Shorthand groups:
              'biophysical'  → SPR, ITC, NMR, X-ray, FRET, BRET
              'coip'         → co-IP variants
              'y2h'          → two-hybrid variants
              'apms'         → AP-MS variants
        mode : 'any' | 'all'

        Examples
        --------
        >>> ds.filter.by_method("two hybrid")
        >>> ds.filter.by_method("biophysical")
        >>> ds.filter.by_method(["pull down", "two hybrid"], mode="any")
        """
        METHOD_GROUPS = {
            "biophysical": [
                "surface plasmon resonance", "isothermal titration calorimetry",
                "nuclear magnetic resonance", "x-ray crystallography",
                "fluorescent resonance energy transfer",
                "bioluminescence resonance energy transfer",
                "fluorescence polarization spectroscopy",
                "3D electron microscopy",
            ],
            "coip": [
                "anti tag coimmunoprecipitation", "anti bait coimmunoprecipitation",
                "coimmunoprecipitation", "tandem affinity purification",
                "affinity chromatography technology",
            ],
            "y2h": [
                "two hybrid", "Two-hybrid", "validated two hybrid",
                "yeast two-hybrid", "two hybrid array",
                "two hybrid prey pooling approach", "two hybrid pooling approach",
            ],
            "apms": [
                "AP-MS (293T)", "AP-MS (HCT116)",
                "affinity chromatography technology",
                "tandem affinity purification",
            ],
        }

        if isinstance(methods, str) and methods in METHOD_GROUPS:
            methods = METHOD_GROUPS[methods]
        if isinstance(methods, str):
            methods = [methods]

        if mode == "any":
            expr = pl.lit(False)
            for m in methods:
                expr = expr | pl.col("detection_methods").str.contains(m, literal=True)
        elif mode == "all":
            expr = pl.lit(True)
            for m in methods:
                expr = expr & pl.col("detection_methods").str.contains(m, literal=True)
        else:
            raise ValueError(f"mode must be 'any' or 'all', got '{mode}'")

        return self._ds._apply(expr)

    # ── Interaction type ──────────────────────────────────────────────────────

    def positives_only(self) -> "PPIDataset":
        """Keep only positive (interacting) pairs."""
        return self._ds._apply(pl.col("interaction_type") == "positive")

    def negatives_only(self) -> "PPIDataset":
        """Keep only negative (non-interacting) pairs from Negatome."""
        return self._ds._apply(pl.col("interaction_type") == "negative")

    # ── Protein list ──────────────────────────────────────────────────────────

    def by_proteins(
        self,
        proteins: List[str],
        mode: str = "any",
    ) -> "PPIDataset":
        """
        Filter to pairs involving specific proteins.

        Parameters
        ----------
        proteins : list[str]
            UniProt accessions of interest.
        mode : 'any' | 'both'
            'any': at least one protein in the pair is in the list.
            'both': both proteins must be in the list.

        Examples
        --------
        >>> ds.filter.by_proteins(["P04637", "P53350"])  # pairs involving TP53 or PLK1
        >>> ds.filter.by_proteins(my_gene_list, mode="both")  # intra-list pairs only
        """
        protein_set = set(proteins)
        if mode == "any":
            expr = (
                pl.col("uniprot_a").is_in(protein_set) |
                pl.col("uniprot_b").is_in(protein_set)
            )
        elif mode == "both":
            expr = (
                pl.col("uniprot_a").is_in(protein_set) &
                pl.col("uniprot_b").is_in(protein_set)
            )
        else:
            raise ValueError(f"mode must be 'any' or 'both', got '{mode}'")
        return self._ds._apply(expr)

    # ── Convenience presets ───────────────────────────────────────────────────

    def high_confidence(self, min_sources: int = 2) -> "PPIDataset":
        """
        Preset: human, positive, LTP-validated, ≥min_sources.
        A sensible default for ML training.

        Equivalent to:
            .by_species("9606")
            .positives_only()
            .by_throughput("ltp_validated")
            .by_min_sources(min_sources)
        """
        return (
            self._ds
            .filter.by_species("9606")
            .filter.positives_only()
            .filter.by_throughput("ltp_validated")
            .filter.by_min_sources(min_sources)
        )

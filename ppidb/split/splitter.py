"""
Splitter — train/val/test splitting for PPI datasets.

Strategies:
1. similarity_split  — sequence-similarity-aware split: proteins in test have
                       < identity_threshold sequence identity to any train protein
                       (prevents data leakage from homologous proteins)
2. greedy_c3_split   — iteratively adds the highest-degree protein (within the
                       remaining graph) to the test pool until the target protein
                       fraction is reached.
3. community_c3_split — detects communities (Louvain algorithm) in the PPI
                        graph and assigns whole communities to the test pool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import polars as pl

from ppidb.core.dataset import PPIDataset


@dataclass
class SplitResult:
    """
    Container for a train/val/test split.

    Attributes
    ----------
    train, val, test : PPIDataset
        The three folds. val may be None if not requested.
    metadata : dict
        Split parameters for reproducibility.
    """
    train: PPIDataset
    test: PPIDataset
    val: Optional[PPIDataset] = None
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        val_str = f", val={len(self.val):,}" if self.val else ""
        return (
            f"SplitResult(train={len(self.train):,}"
            f"{val_str}, test={len(self.test):,})"
        )

    def save(self, path: Union[str, Path]) -> None:
        """
        Save split as a directory with parquet files + metadata JSON.

        Structure:
            path/
              train.parquet
              val.parquet      (if val exists)
              test.parquet
              metadata.json
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.train.save(path / "train.parquet")
        self.test.save(path / "test.parquet")
        if self.val:
            self.val.save(path / "val.parquet")
        with open(path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Split saved to {path}/")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SplitResult":
        """Load a previously saved split."""
        path = Path(path)
        train = PPIDataset.load(path / "train.parquet")
        test = PPIDataset.load(path / "test.parquet")
        val = PPIDataset.load(path / "val.parquet") if (path / "val.parquet").exists() else None
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        return cls(train=train, val=val, test=test, metadata=metadata)

    def summary(self) -> None:
        """Print split statistics."""
        total = len(self.train) + len(self.test) + (len(self.val) if self.val else 0)
        print(f"{'='*45}")
        print(f"  Split Summary  [{self.metadata.get('strategy', 'unknown')}]")
        print(f"{'='*45}")
        print(f"  Train: {len(self.train):>10,}  ({len(self.train)/total*100:.1f}%)")
        if self.val:
            print(f"  Val:   {len(self.val):>10,}  ({len(self.val)/total*100:.1f}%)")
        print(f"  Test:  {len(self.test):>10,}  ({len(self.test)/total*100:.1f}%)")
        print(f"  Total: {total:>10,}")
        if "leakage_pairs" in self.metadata:
            lk = self.metadata["leakage_pairs"]
            if lk > 0:
                print(f"\n  WARNING: {lk} pairs appear in both train and test!")
        print(f"{'='*45}")


class Splitter:
    """
    Splits a PPIDataset into train/val/test folds using C3 strategies.

    Parameters
    ----------
    dataset : PPIDataset
        The dataset to split (should be filtered to desired subset first).
    """

    def __init__(self, dataset: PPIDataset):
        self._ds = dataset

    def similarity_split(
        self,
        identity_threshold: float = 0.3,
        test_frac: float = 0.1,
        val_frac: float = 0.1,
        seed: int = 42,
        sequence_dict: Optional[Dict[str, str]] = None,
    ) -> SplitResult:
        """
        Sequence-similarity-aware split (leakage-free, C3).

        Proteins in test/val have < identity_threshold sequence identity
        to any protein in train. This prevents data leakage from homologs.

        Algorithm:
          1. Cluster proteins by sequence identity using MMseqs2-style greedy
             clustering (or a pre-computed cluster assignment)
          2. Assign clusters to folds (not individual proteins)
          3. Assign pairs based on cluster membership

        Parameters
        ----------
        identity_threshold : float
            Maximum allowed sequence identity between train and test proteins.
            PPIRef uses 0.3 (30%). Common choices: 0.3, 0.5.
        test_frac : float
            Approximate fraction of pairs for test.
        val_frac : float
            Approximate fraction of pairs for val.
        seed : int
            Random seed.
        sequence_dict : dict[str, str] | None
            Pre-fetched {uniprot_id: sequence} dict. If None, sequences will
            be fetched from UniProt automatically (requires internet).

        Notes
        -----
        This is the most rigorous strategy but requires sequence data.
        Use SequenceFetcher to obtain sequences first:

            from ppidb.sequence import SequenceFetcher
            seqs = SequenceFetcher().fetch(ds.proteins(), as_dict=True)
            split = Splitter(ds).similarity_split(
                identity_threshold=0.3,
                sequence_dict=seqs
            )
        """
        if sequence_dict is None:
            raise ValueError(
                "sequence_dict is required for similarity_split. "
                "Use SequenceFetcher to fetch sequences first:\n"
                "  from ppidb.sequence import SequenceFetcher\n"
                "  seqs = SequenceFetcher().fetch(ds.proteins(), as_dict=True)\n"
                "  split = Splitter(ds).similarity_split(sequence_dict=seqs)"
            )

        try:
            from ppidb.split._clustering import cluster_by_identity
        except ImportError:
            raise ImportError(
                "Similarity split requires MMseqs2. Install with: conda install -c bioconda mmseqs2"
            )

        rng = np.random.default_rng(seed)
        df = self._ds.collect()

        # Cluster proteins
        clusters = cluster_by_identity(sequence_dict, identity_threshold)
        cluster_ids = sorted(set(clusters.values()))
        rng.shuffle(cluster_ids)

        n_clusters = len(cluster_ids)
        n_test_cl = max(1, int(n_clusters * test_frac))
        n_val_cl = max(1, int(n_clusters * val_frac))

        test_clusters = set(cluster_ids[:n_test_cl])
        val_clusters = set(cluster_ids[n_test_cl: n_test_cl + n_val_cl])

        def assign_fold(row):
            ca = clusters.get(row["uniprot_a"])
            cb = clusters.get(row["uniprot_b"])
            if ca in test_clusters and cb in test_clusters:
                return "test"
            elif ca in val_clusters and cb in val_clusters:
                return "val"
            elif ca not in test_clusters and ca not in val_clusters \
                    and cb not in test_clusters and cb not in val_clusters:
                return "train"
            else:
                return "discard"

        folds = [assign_fold(r) for r in df.select(["uniprot_a", "uniprot_b"]).to_dicts()]
        df = df.with_columns(pl.Series("_fold", folds))

        df_train = df.filter(pl.col("_fold") == "train").drop("_fold")
        df_val = df.filter(pl.col("_fold") == "val").drop("_fold")
        df_test = df.filter(pl.col("_fold") == "test").drop("_fold")
        n_discarded = sum(1 for f in folds if f == "discard")

        metadata = {
            "strategy": "similarity",
            "identity_threshold": identity_threshold,
            "test_frac": test_frac, "val_frac": val_frac,
            "seed": seed,
            "n_clusters": n_clusters,
            "n_total": len(df),
            "n_train": len(df_train), "n_val": len(df_val), "n_test": len(df_test),
            "n_discarded": n_discarded,
        }

        return SplitResult(
            train=PPIDataset(df_train.lazy()),
            val=PPIDataset(df_val.lazy()) if val_frac > 0 else None,
            test=PPIDataset(df_test.lazy()),
            metadata=metadata,
        )

    def greedy_c3_split(
        self,
        test_protein_frac: float = 0.2,
        val_protein_frac: float = 0.0,
        seed: int = 42,
        sequence_dict: Optional[Dict[str, str]] = None,
        identity_threshold: Optional[float] = None,
    ) -> SplitResult:
        """
        C3-maximizing greedy split.

        Selects a dense subgraph as the test pool by iteratively adding the
        protein with the most connections into the current test pool.
        Guarantees C3=100% (strict protein-level disjointness).

        See ``ppidb.split.c1c2c3.greedy_c3_split`` for full documentation.

        Parameters
        ----------
        test_protein_frac : float
            Fraction of proteins to place in the test pool.
        val_protein_frac : float
            Fraction of proteins for the validation pool (0 = no val).
        seed : int
            Random seed for tie-breaking.
        sequence_dict : dict[str, str] | None
            Optional protein sequences for stricter sequence-similarity-aware C3.
        identity_threshold : float | None
            Maximum allowed sequence identity between train and test proteins.
        """
        from ppidb.split.c1c2c3 import greedy_c3_split as _greedy
        return _greedy(
            self._ds, test_protein_frac, val_protein_frac, seed,
            sequence_dict=sequence_dict, identity_threshold=identity_threshold
        )

    def community_c3_split(
        self,
        test_protein_frac: float = 0.2,
        val_protein_frac: float = 0.0,
        resolution: float = 1.0,
        seed: int = 42,
        sequence_dict: Optional[Dict[str, str]] = None,
        identity_threshold: Optional[float] = None,
    ) -> SplitResult:
        """
        C3-maximizing community split.

        Detects Louvain communities in the PPI graph and assigns whole
        communities to the test pool, exploiting the modular structure of
        real PPI networks to maximise internal test edges.
        Guarantees C3=100% (strict protein-level disjointness).

        Requires ``igraph``: ``pip install igraph``.

        See ``ppidb.split.c1c2c3.community_c3_split`` for full documentation.

        Parameters
        ----------
        test_protein_frac : float
            Target fraction of proteins in the test pool.
        val_protein_frac : float
            Target fraction of proteins in the validation pool.
        resolution : float
            Louvain resolution parameter (default 1.0).
        seed : int
            Random seed for Louvain.
        sequence_dict : dict[str, str] | None
            Optional protein sequences for stricter sequence-similarity-aware C3.
        identity_threshold : float | None
            Maximum allowed sequence identity between train and test proteins.
        """
        from ppidb.split.c1c2c3 import community_c3_split as _community
        return _community(
            self._ds, test_protein_frac, val_protein_frac, resolution, seed,
            sequence_dict=sequence_dict, identity_threshold=identity_threshold
        )

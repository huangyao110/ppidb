"""
NegativeSampler — generate negative (non-interacting) protein pairs.

Three strategies:

1. negatome   — Use experimentally validated non-interacting pairs from Negatome
                (highest quality, but limited to ~3K pairs)

2. random     — Random protein pairs not in the positive set
                (fast, scalable, but may include unknown true positives)

3. subcellular — Random pairs from proteins in DIFFERENT subcellular compartments
                (biologically informed: proteins that never co-localize
                are unlikely to interact; reduces false negatives)

All strategies:
  - Guarantee no overlap with the positive set
  - Return a PPIDataset with interaction_type='negative'
  - Support ratio control (e.g., 1:1 positive:negative)
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import polars as pl

from ppidb.core.dataset import PPIDataset


class NegativeSampler:
    """
    Generate negative PPI samples.

    Parameters
    ----------
    positive_dataset : PPIDataset
        The positive PPI dataset to sample against.
        Negatives are guaranteed not to appear in this set.

    Examples
    --------
    >>> sampler = NegativeSampler(ds.filter.high_confidence())

    >>> # Use Negatome validated negatives
    >>> negs = sampler.from_negatome(full_dataset)

    >>> # Random negatives at 1:1 ratio
    >>> negs = sampler.random_sample(ratio=1.0, seed=42)

    >>> # Subcellular-aware negatives
    >>> negs = sampler.subcellular_sample(ratio=1.0, compartment_map=my_map)
    """

    def __init__(self, positive_dataset: PPIDataset):
        self._pos = positive_dataset
        self._pos_df: Optional[pl.DataFrame] = None
        self._pos_set: Optional[set] = None

    def _get_pos_set(self) -> set:
        """Lazily compute the set of positive pair IDs."""
        if self._pos_set is None:
            df = self._pos.collect()
            self._pos_set = set(
                a + "__" + b
                for a, b in zip(df["uniprot_a"].to_list(), df["uniprot_b"].to_list())
            )
            # Also add reversed pairs (since pairs are canonical)
            self._pos_set |= set(
                b + "__" + a
                for a, b in zip(df["uniprot_a"].to_list(), df["uniprot_b"].to_list())
            )
        return self._pos_set

    # ── Strategy 1: Negatome ──────────────────────────────────────────────────

    def from_negatome(self, full_dataset: PPIDataset) -> PPIDataset:
        """
        Extract experimentally validated negative pairs from the full dataset.

        These are pairs from Negatome (interaction_type='negative') that are
        already in the database. Highest quality but limited in number (~3K).

        Parameters
        ----------
        full_dataset : PPIDataset
            The complete v6 dataset (which contains Negatome negatives).

        Returns
        -------
        PPIDataset
            Negatome negative pairs.

        Notes
        -----
        Negatome pairs are already guaranteed not to be in the positive set
        (they were separated during v6 construction).
        """
        negs = full_dataset.filter.negatives_only()
        n = len(negs)
        print(f"Negatome negatives: {n:,} pairs")
        return negs

    # ── Strategy 2: Random sampling ───────────────────────────────────────────

    def random_sample(
        self,
        ratio: float = 1.0,
        n: Optional[int] = None,
        seed: int = 42,
        max_attempts_multiplier: int = 10,
    ) -> PPIDataset:
        """
        Generate random negative pairs by sampling protein pairs not in the
        positive set.

        Parameters
        ----------
        ratio : float
            Ratio of negatives to positives (e.g., 1.0 = equal numbers).
            Ignored if n is specified.
        n : int | None
            Exact number of negative pairs to generate.
        seed : int
            Random seed.
        max_attempts_multiplier : int
            How many extra candidates to generate to account for collisions
            with the positive set.

        Returns
        -------
        PPIDataset
            Randomly sampled negative pairs.

        Notes
        -----
        For large positive sets (>1M pairs), the probability of random collision
        is low (~0.01% for human proteome of ~20K proteins). However, some
        sampled pairs may be true positives not yet discovered experimentally.
        This is an inherent limitation of random negative sampling.
        """
        rng = np.random.default_rng(seed)
        pos_df = self._pos.collect()
        pos_set = self._get_pos_set()

        # Determine target count
        n_target = n if n is not None else int(len(pos_df) * ratio)

        # Get protein universe from positive set
        proteins = sorted(set(
            pos_df["uniprot_a"].to_list() + pos_df["uniprot_b"].to_list()
        ))
        proteins = np.array(proteins)
        n_prot = len(proteins)

        print(f"Sampling {n_target:,} random negatives from {n_prot:,} proteins...")

        negatives = []
        n_attempts = 0
        max_attempts = n_target * max_attempts_multiplier

        while len(negatives) < n_target and n_attempts < max_attempts:
            # Batch sample for efficiency
            batch_size = min((n_target - len(negatives)) * 3, 100_000)
            idx_a = rng.integers(0, n_prot, size=batch_size)
            idx_b = rng.integers(0, n_prot, size=batch_size)

            for ia, ib in zip(idx_a, idx_b):
                if ia == ib:
                    continue
                a, b = proteins[ia], proteins[ib]
                # Canonical ordering
                if a > b:
                    a, b = b, a
                pair_id = a + "__" + b
                if pair_id not in pos_set:
                    negatives.append({
                        "uniprot_a": a,
                        "uniprot_b": b,
                        "source_dbs": "random_negative",
                        "n_sources": 0,
                        "taxon_a": None,
                        "taxon_b": None,
                        "detection_methods": "random_sampling",
                        "interaction_type": "negative",
                        "throughput_type": "negative_sample",
                    })
                    pos_set.add(pair_id)  # prevent duplicates
                    if len(negatives) >= n_target:
                        break
            n_attempts += batch_size

        if len(negatives) < n_target:
            warnings.warn(
                f"Could only generate {len(negatives):,} negatives "
                f"(requested {n_target:,}). Protein universe may be too small."
            )

        print(f"Generated {len(negatives):,} random negative pairs.")
        if not negatives:
            # Return empty dataset with correct schema
            schema = {
                "uniprot_a": pl.Utf8, "uniprot_b": pl.Utf8,
                "source_dbs": pl.Utf8, "n_sources": pl.Int64,
                "taxon_a": pl.Utf8, "taxon_b": pl.Utf8,
                "detection_methods": pl.Utf8,
                "interaction_type": pl.Utf8, "throughput_type": pl.Utf8,
            }
            return PPIDataset(pl.DataFrame(schema=schema).lazy())
        df = pl.DataFrame(negatives)
        return PPIDataset(df.lazy())

    # ── Strategy 3: Subcellular-aware sampling ────────────────────────────────

    def subcellular_sample(
        self,
        ratio: float = 1.0,
        n: Optional[int] = None,
        compartment_map: Optional[dict] = None,
        seed: int = 42,
    ) -> PPIDataset:
        """
        Generate biologically-informed negatives by sampling pairs from
        proteins in DIFFERENT subcellular compartments.

        Proteins that never co-localize are unlikely to interact, making
        these negatives more realistic than purely random sampling.

        Parameters
        ----------
        ratio : float
            Ratio of negatives to positives.
        n : int | None
            Exact number of negatives.
        compartment_map : dict[str, str] | None
            {uniprot_id: compartment_name} mapping.
            If None, will attempt to fetch from UniProt automatically.
        seed : int
            Random seed.

        Returns
        -------
        PPIDataset
            Subcellular-aware negative pairs.

        Notes
        -----
        Compartment data can be fetched via:
            from ppidb.sequence import SequenceFetcher
            compartments = SequenceFetcher().fetch_compartments(ds.proteins())
        """
        if compartment_map is None:
            raise ValueError(
                "compartment_map is required. Fetch it with:\n"
                "  from ppidb.sequence import SequenceFetcher\n"
                "  compartments = SequenceFetcher().fetch_compartments(ds.proteins())"
            )

        rng = np.random.default_rng(seed)
        pos_df = self._pos.collect()
        pos_set = self._get_pos_set()

        n_target = n if n is not None else int(len(pos_df) * ratio)

        # Group proteins by compartment
        compartment_to_proteins: dict[str, list] = {}
        for pid, comp in compartment_map.items():
            compartment_to_proteins.setdefault(comp, []).append(pid)

        compartments = [c for c, ps in compartment_to_proteins.items() if len(ps) >= 1]
        if len(compartments) < 2:
            raise ValueError("Need at least 2 compartments for subcellular sampling.")

        print(f"Sampling {n_target:,} subcellular negatives from "
              f"{len(compartments)} compartments...")

        negatives = []
        max_attempts = n_target * 20

        for _ in range(max_attempts):
            if len(negatives) >= n_target:
                break
            # Pick two different compartments
            c1, c2 = rng.choice(compartments, size=2, replace=False)
            a = rng.choice(compartment_to_proteins[c1])
            b = rng.choice(compartment_to_proteins[c2])
            if a == b:
                continue
            if a > b:
                a, b = b, a
            pair_id = a + "__" + b
            if pair_id not in pos_set:
                negatives.append({
                    "uniprot_a": a,
                    "uniprot_b": b,
                    "source_dbs": "subcellular_negative",
                    "n_sources": 0,
                    "taxon_a": compartment_map.get(a, {}).get("taxon") if isinstance(
                        compartment_map.get(a), dict) else None,
                    "taxon_b": None,
                    "detection_methods": f"subcellular_sampling:{c1}_vs_{c2}",
                    "interaction_type": "negative",
                    "throughput_type": "negative_sample",
                })
                pos_set.add(pair_id)

        if len(negatives) < n_target:
            warnings.warn(
                f"Could only generate {len(negatives):,} subcellular negatives "
                f"(requested {n_target:,})."
            )

        print(f"Generated {len(negatives):,} subcellular negative pairs.")
        df = pl.DataFrame(negatives)
        return PPIDataset(df.lazy())

    # ── Combine positives and negatives ───────────────────────────────────────

    @staticmethod
    def combine(
        positives: PPIDataset,
        negatives: PPIDataset,
        shuffle: bool = True,
        seed: int = 42,
    ) -> PPIDataset:
        """
        Combine positive and negative datasets into a single labeled dataset.

        Parameters
        ----------
        positives : PPIDataset
        negatives : PPIDataset
        shuffle : bool
            Whether to shuffle the combined dataset.
        seed : int

        Returns
        -------
        PPIDataset
            Combined dataset with interaction_type column as label.
        """
        pos_df = positives.collect()
        neg_df = negatives.collect()
        combined = pl.concat([pos_df, neg_df], how="diagonal_relaxed")
        if shuffle:
            combined = combined.sample(fraction=1.0, shuffle=True, seed=seed)
        n_pos = len(pos_df)
        n_neg = len(neg_df)
        print(f"Combined dataset: {n_pos:,} positives + {n_neg:,} negatives "
              f"= {len(combined):,} total (ratio {n_pos/n_neg:.2f}:1)")
        return PPIDataset(combined.lazy())

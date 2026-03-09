"""
C1/C2/C3 evaluation framework for PPI prediction benchmarking.

The C1/C2/C3 classification was introduced by Park & Marcotte (2012,
Nature Methods) to quantify data leakage in PPI benchmarks. It partitions
a test set based on whether each protein in a pair was seen during training:

  C1 (Both-Seen):    both proteins in the test pair appeared in training
  C2 (One-Seen):     exactly one protein in the test pair appeared in training
  C3 (Neither-Seen): neither protein in the test pair appeared in training

This framework is now standard for rigorous PPI evaluation [23, 46].
Bernett et al. (2024) showed that random splits yield >90% C1 pairs,
causing severely inflated performance estimates [46].

C3-Maximizing Split Strategies
-------------------------------
The cold_split in splitter.py already guarantees C3=100% by randomly
assigning proteins to train/test pools. However, random assignment yields
sparse test sets because random protein subsets have few internal edges.

Two C3-maximizing strategies select a *dense* subgraph as the test pool,
maximising the number of test pairs while keeping C3=100%:

  greedy_c3_split:    iteratively adds the highest-degree protein (within
                      the remaining graph) to the test pool until the
                      target protein fraction is reached.

  community_c3_split: detects communities (Louvain algorithm) in the PPI
                      graph and assigns whole communities to the test pool,
                      exploiting the fact that real PPI networks are modular
                      and communities are internally dense.

Both strategies guarantee strict protein-level disjointness (C3=100%).

References
----------
Park & Marcotte (2012). Flaws in evaluation schemes for pair-input
    computational predictions. Nature Methods, 9, 1134-1136.
Bernett et al. (2024). Cracking the black box of deep sequence-based
    protein-protein interaction prediction. Briefings in Bioinformatics.
Lv et al. (2021). Learning unknown from correlations: GNN for
    inter-novel-protein interaction prediction. IJCAI 2021.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import polars as pl

from ppidb.core.dataset import PPIDataset


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class C1C2C3Stats:
    """
    Statistics describing the C1/C2/C3 composition of a test set.

    Attributes
    ----------
    c1_count : int
        Number of test pairs where BOTH proteins appeared in training (Both-Seen).
    c2_count : int
        Number of test pairs where EXACTLY ONE protein appeared in training (One-Seen).
    c3_count : int
        Number of test pairs where NEITHER protein appeared in training (Neither-Seen).
    total_count : int
        Total number of test pairs.
    c1_pct, c2_pct, c3_pct : float
        Percentage of each category.
    n_train_proteins : int
        Number of unique proteins in the training set.
    n_test_proteins : int
        Number of unique proteins in the test set.
    n_overlap_proteins : int
        Number of proteins appearing in both train and test.
    """
    c1_count: int
    c2_count: int
    c3_count: int
    total_count: int
    c1_pct: float
    c2_pct: float
    c3_pct: float
    n_train_proteins: int
    n_test_proteins: int
    n_overlap_proteins: int

    def __str__(self) -> str:
        bar_width = 30
        def bar(pct: float) -> str:
            filled = int(round(pct / 100 * bar_width))
            return "█" * filled + "░" * (bar_width - filled)

        lines = [
            "C1/C2/C3 Evaluation Statistics",
            "=" * 50,
            f"  C1 (Both-Seen):    {self.c1_count:>8,}  {self.c1_pct:>5.1f}%  {bar(self.c1_pct)}",
            f"  C2 (One-Seen):     {self.c2_count:>8,}  {self.c2_pct:>5.1f}%  {bar(self.c2_pct)}",
            f"  C3 (Neither-Seen): {self.c3_count:>8,}  {self.c3_pct:>5.1f}%  {bar(self.c3_pct)}",
            f"  Total:             {self.total_count:>8,}",
            "",
            f"  Train proteins:    {self.n_train_proteins:>8,}",
            f"  Test proteins:     {self.n_test_proteins:>8,}",
            f"  Overlap proteins:  {self.n_overlap_proteins:>8,}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def is_leaky(self, c1_threshold: float = 0.5) -> bool:
        """
        Return True if C1 fraction exceeds threshold.

        A high C1 fraction (>50%) indicates significant data leakage.
        Random splits typically yield C1 > 90% [46].
        """
        return self.c1_pct > (c1_threshold * 100)


@dataclass
class C1C2C3Split:
    """
    A test set partitioned into C1, C2, and C3 subsets.

    Attributes
    ----------
    c1 : PPIDataset
        Both-Seen pairs: both proteins seen in training.
    c2 : PPIDataset
        One-Seen pairs: exactly one protein seen in training.
    c3 : PPIDataset
        Neither-Seen pairs: no protein seen in training.
    stats : C1C2C3Stats
        Summary statistics.
    """
    c1: PPIDataset
    c2: PPIDataset
    c3: PPIDataset
    stats: C1C2C3Stats

    def __repr__(self) -> str:
        return (
            f"C1C2C3Split("
            f"C1={len(self.c1):,} [{self.stats.c1_pct:.1f}%], "
            f"C2={len(self.c2):,} [{self.stats.c2_pct:.1f}%], "
            f"C3={len(self.c3):,} [{self.stats.c3_pct:.1f}%])"
        )

    def summary(self) -> None:
        """Print formatted statistics."""
        print(self.stats)


# ── Core functions ────────────────────────────────────────────────────────────

def get_train_proteins(train: PPIDataset) -> Set[str]:
    """
    Extract the set of all unique proteins that appear in the training set.

    Parameters
    ----------
    train : PPIDataset
        Training dataset.

    Returns
    -------
    set[str]
        UniProt accessions of all proteins seen during training.
    """
    df = train.collect()
    return set(df["uniprot_a"].to_list()) | set(df["uniprot_b"].to_list())


def classify_pairs(
    test_df: pl.DataFrame,
    train_proteins: Set[str],
    sequence_dict: Optional[Dict[str, str]] = None,
    identity_threshold: Optional[float] = None,
) -> pl.Series:
    """
    Classify each test pair as C1, C2, or C3.

    Parameters
    ----------
    test_df : pl.DataFrame
        Test set DataFrame with columns ``uniprot_a`` and ``uniprot_b``.
    train_proteins : set[str]
        Set of protein accessions seen in training.
    sequence_dict : dict[str, str] | None
        Optional protein sequence mapping used for stricter sequence-similarity
        aware C1/C2/C3 classification.
    identity_threshold : float | None
        Maximum allowed sequence identity for a protein to be considered novel
        w.r.t. train proteins. Used only when ``sequence_dict`` is provided.

    Returns
    -------
    pl.Series
        String Series with values "C1", "C2", or "C3" for each row.
    """
    strict_seq_mode = sequence_dict is not None and identity_threshold is not None
    if strict_seq_mode:
        required = set(test_df["uniprot_a"].to_list()) | set(test_df["uniprot_b"].to_list()) | train_proteins
        missing = sorted([p for p in required if p not in sequence_dict])
        if missing:
            raise ValueError(
                "sequence_dict is missing sequences for proteins required by strict C1/C2/C3 "
                f"classification (showing first 10): {missing[:10]}"
            )

        train_sequences = {p: sequence_dict[p] for p in train_proteins}
        sim_cache: Dict[str, float] = {}

        def _seq_identity(a: str, b: str) -> float:
            k = 5
            if len(a) < k or len(b) < k:
                return 0.0
            kmers_a = set(a[i:i + k] for i in range(len(a) - k + 1))
            kmers_b = set(b[i:i + k] for i in range(len(b) - k + 1))
            if not kmers_a or not kmers_b:
                return 0.0
            return len(kmers_a & kmers_b) / min(len(kmers_a), len(kmers_b))

        def _max_seq_sim_to_train(protein_id: str) -> float:
            if protein_id in sim_cache:
                return sim_cache[protein_id]
            seq = sequence_dict[protein_id]
            max_sim = 0.0
            for train_seq in train_sequences.values():
                sim = _seq_identity(seq, train_seq)
                if sim > max_sim:
                    max_sim = sim
                if max_sim > identity_threshold:
                    break
            sim_cache[protein_id] = max_sim
            return max_sim

    labels = []
    for row in test_df.select(["uniprot_a", "uniprot_b"]).iter_rows():
        a_seen = row[0] in train_proteins
        b_seen = row[1] in train_proteins
        if not strict_seq_mode:
            if a_seen and b_seen:
                labels.append("C1")
            elif a_seen or b_seen:
                labels.append("C2")
            else:
                labels.append("C3")
            continue

        a_train_like = a_seen or (_max_seq_sim_to_train(row[0]) > identity_threshold)
        b_train_like = b_seen or (_max_seq_sim_to_train(row[1]) > identity_threshold)

        if a_train_like and b_train_like:
            labels.append("C1")
        elif a_train_like or b_train_like:
            labels.append("C2")
        else:
            labels.append("C3")
    return pl.Series("c_label", labels)


def compute_c1c2c3_stats(
    train: PPIDataset,
    test: PPIDataset,
    train_proteins: Optional[Set[str]] = None,
    sequence_dict: Optional[Dict[str, str]] = None,
    identity_threshold: Optional[float] = None,
) -> C1C2C3Stats:
    """
    Compute C1/C2/C3 statistics for a given train/test split.

    Parameters
    ----------
    train : PPIDataset
        Training set.
    test : PPIDataset
        Test set to analyse.
    train_proteins : set[str] | None
        Pre-computed set of training proteins (avoids recomputation when
        calling this function multiple times with the same training set).
    sequence_dict : dict[str, str] | None
        Optional protein sequences for stricter sequence-similarity-aware
        classification.
    identity_threshold : float | None
        Sequence identity threshold used with ``sequence_dict``.

    Returns
    -------
    C1C2C3Stats
        Detailed statistics object.

    Examples
    --------
    >>> from ppidb.split import Splitter
    >>> from ppidb.split.c1c2c3 import compute_c1c2c3_stats
    >>> split = Splitter(ds).random_split()
    >>> stats = compute_c1c2c3_stats(split.train, split.test)
    >>> print(stats)
    """
    if train_proteins is None:
        train_proteins = get_train_proteins(train)

    test_df = test.collect()
    labels = classify_pairs(
        test_df,
        train_proteins,
        sequence_dict=sequence_dict,
        identity_threshold=identity_threshold,
    )

    c1 = (labels == "C1").sum()
    c2 = (labels == "C2").sum()
    c3 = (labels == "C3").sum()
    total = len(test_df)

    test_proteins = set(test_df["uniprot_a"].to_list()) | set(test_df["uniprot_b"].to_list())
    overlap = train_proteins & test_proteins

    pct = lambda n: 100.0 * n / total if total > 0 else 0.0

    return C1C2C3Stats(
        c1_count=c1,
        c2_count=c2,
        c3_count=c3,
        total_count=total,
        c1_pct=pct(c1),
        c2_pct=pct(c2),
        c3_pct=pct(c3),
        n_train_proteins=len(train_proteins),
        n_test_proteins=len(test_proteins),
        n_overlap_proteins=len(overlap),
    )


def split_test_by_c1c2c3(
    train: PPIDataset,
    test: PPIDataset,
    train_proteins: Optional[Set[str]] = None,
    sequence_dict: Optional[Dict[str, str]] = None,
    identity_threshold: Optional[float] = None,
) -> C1C2C3Split:
    """
    Partition a test set into C1, C2, and C3 subsets.

    This is the primary function for C1/C2/C3 evaluation. After obtaining
    a train/test split (via any strategy), call this function to decompose
    the test set for fine-grained evaluation.

    Parameters
    ----------
    train : PPIDataset
        Training set (used to determine which proteins are "seen").
    test : PPIDataset
        Test set to partition.
    train_proteins : set[str] | None
        Pre-computed training protein set (optional, for efficiency).
    sequence_dict : dict[str, str] | None
        Optional protein sequences for stricter sequence-similarity-aware
        classification.
    identity_threshold : float | None
        Sequence identity threshold used with ``sequence_dict``.

    Returns
    -------
    C1C2C3Split
        Object containing c1, c2, c3 sub-datasets and summary statistics.

    Examples
    --------
    >>> from ppidb.split import Splitter
    >>> from ppidb.split.c1c2c3 import split_test_by_c1c2c3
    >>>
    >>> # Random split — expect high C1 (data leakage)
    >>> split = Splitter(ds).random_split(train=0.8, val=0.0, test=0.2)
    >>> c123 = split_test_by_c1c2c3(split.train, split.test)
    >>> c123.summary()
    >>>
    >>> # Cold split — expect high C3 (no leakage)
    >>> split = Splitter(ds).cold_split(test_frac=0.2)
    >>> c123 = split_test_by_c1c2c3(split.train, split.test)
    >>> c123.summary()
    """
    if train_proteins is None:
        train_proteins = get_train_proteins(train)

    test_df = test.collect()
    labels = classify_pairs(
        test_df,
        train_proteins,
        sequence_dict=sequence_dict,
        identity_threshold=identity_threshold,
    )
    labeled_df = test_df.with_columns(labels)

    c1_df = labeled_df.filter(pl.col("c_label") == "C1").drop("c_label")
    c2_df = labeled_df.filter(pl.col("c_label") == "C2").drop("c_label")
    c3_df = labeled_df.filter(pl.col("c_label") == "C3").drop("c_label")

    total = len(test_df)
    pct = lambda n: 100.0 * n / total if total > 0 else 0.0

    test_proteins = set(test_df["uniprot_a"].to_list()) | set(test_df["uniprot_b"].to_list())

    stats = C1C2C3Stats(
        c1_count=len(c1_df),
        c2_count=len(c2_df),
        c3_count=len(c3_df),
        total_count=total,
        c1_pct=pct(len(c1_df)),
        c2_pct=pct(len(c2_df)),
        c3_pct=pct(len(c3_df)),
        n_train_proteins=len(train_proteins),
        n_test_proteins=len(test_proteins),
        n_overlap_proteins=len(train_proteins & test_proteins),
    )

    return C1C2C3Split(
        c1=PPIDataset(c1_df.lazy()),
        c2=PPIDataset(c2_df.lazy()),
        c3=PPIDataset(c3_df.lazy()),
        stats=stats,
    )


def compare_splits(
    splits: dict[str, tuple[PPIDataset, PPIDataset]],
) -> pl.DataFrame:
    """
    Compare C1/C2/C3 composition across multiple split strategies.

    Parameters
    ----------
    splits : dict[str, (train, test)]
        Mapping of strategy name to (train, test) PPIDataset pairs.

    Returns
    -------
    pl.DataFrame
        Comparison table with columns:
        strategy, c1_count, c2_count, c3_count, c1_pct, c2_pct, c3_pct,
        n_train_proteins, n_test_proteins, n_overlap_proteins.

    Examples
    --------
    >>> from ppidb.split import Splitter
    >>> from ppidb.split.c1c2c3 import compare_splits
    >>>
    >>> splitter = Splitter(ds)
    >>> random_split = splitter.random_split(val=0.0)
    >>> cold_split   = splitter.cold_split()
    >>>
    >>> table = compare_splits({
    ...     "random": (random_split.train, random_split.test),
    ...     "cold":   (cold_split.train,   cold_split.test),
    ... })
    >>> print(table)
    """
    rows = []
    for name, (train, test) in splits.items():
        stats = compute_c1c2c3_stats(train, test)
        rows.append({
            "strategy": name,
            "c1_count": stats.c1_count,
            "c2_count": stats.c2_count,
            "c3_count": stats.c3_count,
            "c1_pct": round(stats.c1_pct, 1),
            "c2_pct": round(stats.c2_pct, 1),
            "c3_pct": round(stats.c3_pct, 1),
            "n_train_proteins": stats.n_train_proteins,
            "n_test_proteins": stats.n_test_proteins,
            "n_overlap_proteins": stats.n_overlap_proteins,
        })
    return pl.DataFrame(rows)


# ── C3-Maximizing Split Strategies ───────────────────────────────────────────

def _build_adjacency(df: pl.DataFrame) -> Dict[str, Set[str]]:
    """Build an undirected adjacency dict from a PPI DataFrame."""
    adj: Dict[str, Set[str]] = {}
    for a, b in df.select(["uniprot_a", "uniprot_b"]).iter_rows():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def _count_internal_edges(proteins: Set[str], adj: Dict[str, Set[str]]) -> int:
    """Count edges where both endpoints are in *proteins*."""
    count = 0
    for p in proteins:
        for nb in adj.get(p, set()):
            if nb in proteins and nb > p:   # count each edge once
                count += 1
    return count


def greedy_c3_split(
    dataset: PPIDataset,
    test_protein_frac: float = 0.2,
    val_protein_frac: float = 0.0,
    seed: int = 42,
    sequence_dict: Optional[Dict[str, str]] = None,
    identity_threshold: Optional[float] = None,
) -> "SplitResult":
    """
    C3-maximizing greedy split: select a dense subgraph as the test pool.

    Algorithm
    ---------
    1. Build the PPI graph from *dataset*.
    2. Iteratively add the protein with the highest degree **within the
       current test pool's neighbourhood** to the test pool, until the
       pool reaches ``test_protein_frac`` of all proteins.
       Ties are broken by total degree, then alphabetically.
    3. Assign pairs:
       - test  = both proteins in test pool
       - val   = both proteins in val pool  (if val_protein_frac > 0)
       - train = both proteins in train pool
       - discard = cross-pool pairs

    The greedy strategy exploits the fact that high-degree proteins tend to
    cluster together (hubs and their neighbours form dense subgraphs), so
    selecting them first maximises internal test edges.

    Guarantee
    ---------
    C3 = 100% by construction: train and test protein sets are disjoint.

    Parameters
    ----------
    dataset : PPIDataset
        Full dataset to split.
    test_protein_frac : float
        Fraction of *proteins* (not pairs) to place in the test pool.
        Actual test pair count depends on network topology.
    val_protein_frac : float
        Fraction of proteins for the validation pool (default 0 = no val).
    seed : int
        Random seed used only for tie-breaking and val pool selection.

    Returns
    -------
    SplitResult
        metadata includes ``n_test_proteins``, ``n_test_pairs``,
        ``n_discarded``, and ``strategy = "greedy_c3"``.

    Examples
    --------
    >>> from ppidb.split.c1c2c3 import greedy_c3_split
    >>> split = greedy_c3_split(ds, test_protein_frac=0.2)
    >>> split.summary()
    >>> # Verify C3=100%
    >>> from ppidb.split.c1c2c3 import compute_c1c2c3_stats
    >>> stats = compute_c1c2c3_stats(split.train, split.test)
    >>> assert stats.c3_pct == 100.0
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    df = dataset.collect()
    adj = _build_adjacency(df)

    all_proteins = sorted(adj.keys())
    n_prot = len(all_proteins)
    n_test_target = max(1, int(n_prot * test_protein_frac))
    n_val_target = max(0, int(n_prot * val_protein_frac))

    # ── Greedy selection ──────────────────────────────────────────────────────
    # Start from the highest-degree protein, then greedily add the neighbour
    # with the most connections INTO the current test pool (maximises density).
    remaining = set(all_proteins)
    test_pool: Set[str] = set()

    # Seed: pick the highest-degree protein (random tie-break)
    degrees = {p: len(adj[p]) for p in remaining}
    max_deg = max(degrees.values())
    candidates = sorted([p for p, d in degrees.items() if d == max_deg])
    seed_protein = rng.choice(candidates)
    test_pool.add(seed_protein)
    remaining.remove(seed_protein)

    while len(test_pool) < n_test_target and remaining:
        # Score each remaining protein by how many edges it would add
        # to the current test pool (= number of its neighbours already in pool)
        best_score = -1
        best_candidates: List[str] = []
        for p in remaining:
            score = sum(1 for nb in adj.get(p, set()) if nb in test_pool)
            if score > best_score:
                best_score = score
                best_candidates = [p]
            elif score == best_score:
                best_candidates.append(p)

        # Tie-break by total degree, then alphabetically
        best_candidates.sort(key=lambda p: (-degrees[p], p))
        chosen = best_candidates[0]
        test_pool.add(chosen)
        remaining.remove(chosen)

    # ── Val pool: greedy from remaining ──────────────────────────────────────
    val_pool: Set[str] = set()
    if n_val_target > 0:
        # Same greedy logic on the remaining proteins
        remaining_list = sorted(remaining)
        if remaining_list:
            seed_val = rng.choice(remaining_list)
            val_pool.add(seed_val)
            remaining.remove(seed_val)
            while len(val_pool) < n_val_target and remaining:
                best_score = -1
                best_cands: List[str] = []
                for p in remaining:
                    score = sum(1 for nb in adj.get(p, set()) if nb in val_pool)
                    if score > best_score:
                        best_score = score
                        best_cands = [p]
                    elif score == best_score:
                        best_cands.append(p)
                best_cands.sort(key=lambda p: (-degrees[p], p))
                chosen = best_cands[0]
                val_pool.add(chosen)
                remaining.remove(chosen)

    train_pool = remaining  # everything not in test or val

    # ── Strict C3: Enforce sequence similarity constraint ─────────────────────
    if sequence_dict is not None and identity_threshold is not None:
        try:
            from ppidb.split._clustering import cluster_by_identity
        except ImportError:
            raise ImportError("MMseqs2 required for strict C3 split. Install with conda install -c bioconda mmseqs2")

        # We need to ensure that NO protein in test_pool has > identity_threshold similarity
        # to ANY protein in train_pool.
        # Since greedy_c3 picks test_pool first, we filter train_pool.
        # But wait, standard similarity_split clusters FIRST.
        # Here we already picked test_pool based on graph topology.
        # So we must prune train_pool: remove any protein that is too similar to ANY test protein.
        
        # This can be expensive O(N*M). Better approach:
        # Use MMseqs2 to find all pairs (a, b) with sim > threshold.
        # If a in test_pool and b in train_pool, remove b from train_pool.
        
        # Actually, let's use the cluster_by_identity helper? No, that clusters globally.
        # We need "search" functionality.
        # Let's fallback to a simpler heuristic if we can't do full all-vs-all efficiently:
        # Just use the existing cluster_by_identity to get clusters.
        # If any protein in a cluster is in test_pool, ALL proteins in that cluster
        # MUST NOT be in train_pool (they can be in test or discarded).
        
        clusters = cluster_by_identity(sequence_dict, identity_threshold)
        
        # Identify clusters present in test_pool
        test_clusters = {clusters[p] for p in test_pool if p in clusters}
        
        # Remove any train protein that belongs to a test cluster
        train_pool_strict = {p for p in train_pool if clusters.get(p) not in test_clusters}
        
        removed = len(train_pool) - len(train_pool_strict)
        if removed > 0:
            warnings.warn(
                f"Strict C3: Removed {removed} proteins from train_pool because they "
                f"share >{identity_threshold} identity with test proteins."
            )
        train_pool = train_pool_strict

    # ── Assign pairs ──────────────────────────────────────────────────────────
    folds = []
    for a, b in df.select(["uniprot_a", "uniprot_b"]).iter_rows():
        a_test = a in test_pool
        b_test = b in test_pool
        a_val = a in val_pool
        b_val = b in val_pool
        a_train = a in train_pool
        b_train = b in train_pool
        if a_test and b_test:
            folds.append("test")
        elif a_val and b_val:
            folds.append("val")
        elif a_train and b_train:
            folds.append("train")
        else:
            folds.append("discard")

    df = df.with_columns(pl.Series("_fold", folds))
    df_train = df.filter(pl.col("_fold") == "train").drop("_fold")
    df_val   = df.filter(pl.col("_fold") == "val").drop("_fold")
    df_test  = df.filter(pl.col("_fold") == "test").drop("_fold")
    n_discarded = sum(1 for f in folds if f == "discard")

    if n_discarded > 0:
        warnings.warn(
            f"greedy_c3_split discarded {n_discarded:,} cross-pool pairs "
            f"({n_discarded / len(df) * 100:.1f}% of total)."
        )

    n_test_internal = _count_internal_edges(test_pool, adj)

    from ppidb.split.splitter import SplitResult
    return SplitResult(
        train=PPIDataset(df_train.lazy()),
        val=PPIDataset(df_val.lazy()) if val_protein_frac > 0 else None,
        test=PPIDataset(df_test.lazy()),
        metadata={
            "strategy": "greedy_c3",
            "test_protein_frac": test_protein_frac,
            "val_protein_frac": val_protein_frac,
            "seed": seed,
            "n_total": len(df),
            "n_train": len(df_train),
            "n_val": len(df_val),
            "n_test": len(df_test),
            "n_discarded": n_discarded,
            "n_train_proteins": len(train_pool),
            "n_val_proteins": len(val_pool),
            "n_test_proteins": len(test_pool),
            "n_test_internal_edges": n_test_internal,
            "leakage_pairs": 0,
        },
    )


def community_c3_split(
    dataset: PPIDataset,
    test_protein_frac: float = 0.2,
    val_protein_frac: float = 0.0,
    resolution: float = 1.0,
    seed: int = 42,
    sequence_dict: Optional[Dict[str, str]] = None,
    identity_threshold: Optional[float] = None,
) -> "SplitResult":
    """
    C3-maximizing community split: assign whole Louvain communities to test.

    Algorithm
    ---------
    1. Build the PPI graph and detect communities with the Louvain algorithm
       (via ``igraph``).
    2. Sort communities by size (descending). Greedily assign communities to
       the test pool until the pool reaches ``test_protein_frac`` of all
       proteins.
    3. Assign pairs exactly as in ``greedy_c3_split``.

    Rationale
    ---------
    Real PPI networks are highly modular: proteins in the same biological
    pathway or complex interact densely with each other and sparsely with
    the rest of the network. Assigning whole communities to the test pool
    therefore maximises internal test edges while minimising cross-pool
    edges (which are discarded).

    Guarantee
    ---------
    C3 = 100% by construction.

    Parameters
    ----------
    dataset : PPIDataset
        Full dataset to split.
    test_protein_frac : float
        Target fraction of proteins in the test pool.
    val_protein_frac : float
        Target fraction of proteins in the validation pool.
    resolution : float
        Louvain resolution parameter. Higher values → more, smaller
        communities. Default 1.0 is the standard Louvain objective.
    seed : int
        Random seed for Louvain (reproducibility).

    Returns
    -------
    SplitResult
        metadata includes ``n_communities``, ``n_test_communities``,
        ``n_test_proteins``, ``n_test_pairs``, and
        ``strategy = "community_c3"``.

    Raises
    ------
    ImportError
        If ``igraph`` is not installed. Install with ``pip install igraph``.

    Examples
    --------
    >>> from ppidb.split.c1c2c3 import community_c3_split
    >>> split = community_c3_split(ds, test_protein_frac=0.2)
    >>> split.summary()
    """
    try:
        import igraph as ig
    except ImportError:
        raise ImportError(
            "community_c3_split requires igraph. Install with:\n"
            "  pip install igraph"
        )

    import numpy as np

    rng = np.random.default_rng(seed)
    df = dataset.collect()
    adj = _build_adjacency(df)

    all_proteins = sorted(adj.keys())
    n_prot = len(all_proteins)
    n_test_target = max(1, int(n_prot * test_protein_frac))
    n_val_target = max(0, int(n_prot * val_protein_frac))

    # ── Build igraph Graph ────────────────────────────────────────────────────
    prot_to_idx = {p: i for i, p in enumerate(all_proteins)}
    edges_idx = [
        (prot_to_idx[a], prot_to_idx[b])
        for a, b in df.select(["uniprot_a", "uniprot_b"]).iter_rows()
        if a in prot_to_idx and b in prot_to_idx
    ]
    g = ig.Graph(n=n_prot, edges=edges_idx)
    g.vs["name"] = all_proteins

    # ── Louvain community detection ───────────────────────────────────────────
    partition = g.community_multilevel(
        weights=None,
        resolution=resolution,
    )
    # partition.membership[i] = community id for protein i
    communities: Dict[int, List[str]] = {}
    for i, comm_id in enumerate(partition.membership):
        communities.setdefault(comm_id, []).append(all_proteins[i])

    # Sort communities by size descending, shuffle ties with rng
    comm_list = sorted(communities.values(), key=lambda c: (-len(c), rng.random()))

    # ── Assign communities to test pool ───────────────────────────────────────
    test_pool: Set[str] = set()
    test_communities_used = 0
    for comm in comm_list:
        if len(test_pool) >= n_test_target:
            break
        test_pool.update(comm)
        test_communities_used += 1

    remaining = set(all_proteins) - test_pool

    # ── Assign communities to val pool ────────────────────────────────────────
    val_pool: Set[str] = set()
    if n_val_target > 0:
        # Build communities from remaining proteins
        remaining_comms = [
            [p for p in comm if p in remaining]
            for comm in comm_list
            if any(p in remaining for p in comm)
        ]
        remaining_comms = [c for c in remaining_comms if c]
        remaining_comms.sort(key=lambda c: (-len(c), rng.random()))
        for comm in remaining_comms:
            if len(val_pool) >= n_val_target:
                break
            val_pool.update(comm)

    train_pool = remaining - val_pool

    # ── Strict C3: Enforce sequence similarity constraint ─────────────────────
    if sequence_dict is not None and identity_threshold is not None:
        try:
            from ppidb.split._clustering import cluster_by_identity
        except ImportError:
            raise ImportError("MMseqs2 required for strict C3 split. Install with conda install -c bioconda mmseqs2")

        clusters = cluster_by_identity(sequence_dict, identity_threshold)
        
        # Identify clusters present in test_pool
        test_clusters = {clusters[p] for p in test_pool if p in clusters}
        
        # Remove any train protein that belongs to a test cluster
        train_pool_strict = {p for p in train_pool if clusters.get(p) not in test_clusters}
        
        removed = len(train_pool) - len(train_pool_strict)
        if removed > 0:
            warnings.warn(
                f"Strict C3: Removed {removed} proteins from train_pool because they "
                f"share >{identity_threshold} identity with test proteins."
            )
        train_pool = train_pool_strict

    # ── Assign pairs ──────────────────────────────────────────────────────────
    folds = []
    for a, b in df.select(["uniprot_a", "uniprot_b"]).iter_rows():
        if a in test_pool and b in test_pool:
            folds.append("test")
        elif a in val_pool and b in val_pool:
            folds.append("val")
        elif a in train_pool and b in train_pool:
            folds.append("train")
        else:
            folds.append("discard")

    df = df.with_columns(pl.Series("_fold", folds))
    df_train = df.filter(pl.col("_fold") == "train").drop("_fold")
    df_val   = df.filter(pl.col("_fold") == "val").drop("_fold")
    df_test  = df.filter(pl.col("_fold") == "test").drop("_fold")
    n_discarded = sum(1 for f in folds if f == "discard")

    if n_discarded > 0:
        warnings.warn(
            f"community_c3_split discarded {n_discarded:,} cross-pool pairs "
            f"({n_discarded / len(df) * 100:.1f}% of total)."
        )

    n_test_internal = _count_internal_edges(test_pool, adj)

    from ppidb.split.splitter import SplitResult
    return SplitResult(
        train=PPIDataset(df_train.lazy()),
        val=PPIDataset(df_val.lazy()) if val_protein_frac > 0 else None,
        test=PPIDataset(df_test.lazy()),
        metadata={
            "strategy": "community_c3",
            "test_protein_frac": test_protein_frac,
            "val_protein_frac": val_protein_frac,
            "resolution": resolution,
            "seed": seed,
            "n_total": len(df),
            "n_train": len(df_train),
            "n_val": len(df_val),
            "n_test": len(df_test),
            "n_discarded": n_discarded,
            "n_communities": len(communities),
            "n_test_communities": test_communities_used,
            "n_train_proteins": len(train_pool),
            "n_val_proteins": len(val_pool),
            "n_test_proteins": len(test_pool),
            "n_test_internal_edges": n_test_internal,
            "leakage_pairs": 0,
        },
    )

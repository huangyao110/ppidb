"""
Sequence clustering helper for similarity_split.
Uses MMseqs2 (if available) or a pure-Python fallback with pairwise identity.
"""

from __future__ import annotations
import os
import subprocess
import tempfile
from typing import Dict


def cluster_by_identity(
    sequences: Dict[str, str],
    identity_threshold: float = 0.3,
) -> Dict[str, str]:
    """
    Cluster proteins by sequence identity.

    Returns a dict mapping {protein_id: cluster_representative_id}.

    Tries MMseqs2 first; falls back to greedy pairwise clustering.
    """
    try:
        return _cluster_mmseqs2(sequences, identity_threshold)
    except (FileNotFoundError, subprocess.CalledProcessError):
        import warnings
        warnings.warn(
            "MMseqs2 not found. Falling back to greedy pairwise clustering "
            "(slow for large datasets). Install MMseqs2 for better performance."
        )
        return _cluster_greedy(sequences, identity_threshold)


def _cluster_mmseqs2(
    sequences: Dict[str, str],
    identity_threshold: float,
) -> Dict[str, str]:
    """Cluster using MMseqs2 easy-cluster."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "input.fasta")
        out_prefix = os.path.join(tmpdir, "clusters")

        # Write FASTA
        with open(fasta_path, "w") as f:
            for pid, seq in sequences.items():
                f.write(f">{pid}\n{seq}\n")

        # Run MMseqs2
        subprocess.run([
            "mmseqs", "easy-cluster",
            fasta_path, out_prefix, tmpdir,
            "--min-seq-id", str(identity_threshold),
            "--cov-mode", "0",
            "-c", "0.8",
            "--cluster-mode", "1",
            "-v", "0",
        ], check=True, capture_output=True)

        # Parse cluster TSV: rep_id \t member_id
        clusters = {}
        tsv_path = out_prefix + "_cluster.tsv"
        with open(tsv_path) as f:
            for line in f:
                rep, member = line.strip().split("\t")
                clusters[member] = rep

    return clusters


def _cluster_greedy(
    sequences: Dict[str, str],
    identity_threshold: float,
) -> Dict[str, str]:
    """
    Greedy pairwise clustering fallback.
    O(n^2) — only suitable for small datasets (<10K proteins).
    """
    def seq_identity(a: str, b: str) -> float:
        """Simple k-mer based identity estimate."""
        k = 5
        if len(a) < k or len(b) < k:
            return 0.0
        kmers_a = set(a[i:i+k] for i in range(len(a) - k + 1))
        kmers_b = set(b[i:i+k] for i in range(len(b) - k + 1))
        if not kmers_a or not kmers_b:
            return 0.0
        return len(kmers_a & kmers_b) / min(len(kmers_a), len(kmers_b))

    ids = list(sequences.keys())
    clusters: Dict[str, str] = {}
    representatives: list[str] = []

    for pid in ids:
        assigned = False
        for rep in representatives:
            if seq_identity(sequences[pid], sequences[rep]) >= identity_threshold:
                clusters[pid] = rep
                assigned = True
                break
        if not assigned:
            representatives.append(pid)
            clusters[pid] = pid

    return clusters

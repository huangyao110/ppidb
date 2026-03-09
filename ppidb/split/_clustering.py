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
    from tqdm import tqdm

    def get_kmers(seq: str, k: int = 5) -> set:
        if len(seq) < k:
            return set()
        return set(seq[i:i+k] for i in range(len(seq) - k + 1))

    # Pre-compute kmers for all sequences to speed up comparison
    seq_kmers = {pid: get_kmers(seq) for pid, seq in sequences.items()}
    
    def seq_identity(pid_a: str, pid_b: str) -> float:
        """Simple k-mer based identity estimate using pre-computed sets."""
        kmers_a = seq_kmers[pid_a]
        kmers_b = seq_kmers[pid_b]
        if not kmers_a or not kmers_b:
            return 0.0
        return len(kmers_a & kmers_b) / min(len(kmers_a), len(kmers_b))

    ids = list(sequences.keys())
    # Sort by length descending (greedy heuristic: cluster around longest sequences first)
    # This often produces better clusters and is slightly more stable
    ids.sort(key=lambda x: len(sequences[x]), reverse=True)
    
    clusters: Dict[str, str] = {}
    representatives: list[str] = []

    print(f"Clustering {len(ids)} sequences (greedy strategy)...")
    for pid in tqdm(ids, desc="Clustering"):
        assigned = False
        # Check against existing representatives
        for rep in representatives:
            if seq_identity(pid, rep) >= identity_threshold:
                clusters[pid] = rep
                assigned = True
                break
        
        # If no match, this protein becomes a new representative
        if not assigned:
            representatives.append(pid)
            clusters[pid] = pid

    return clusters

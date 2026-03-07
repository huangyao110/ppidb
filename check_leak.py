#!/usr/bin/env python3
"""
check_leak.py - Check for data leakage in PPI datasets.

This script allows you to:
1. Load a PPI dataset (Parquet format).
2. Perform a train/test split using C3 strategies (similarity, greedy_c3, community_c3).
3. Evaluate data leakage using the C1/C2/C3 framework.
4. Optionally use sequence similarity (strict mode) for more rigorous leakage detection.

Examples:
    # C3 split with strict evaluation (requires sequences)
    python check_leak.py data.parquet --strict --identity 0.3
    
    # Greedy C3 split
    python check_leak.py data.parquet --strategy greedy_c3
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from ppidb.core.dataset import PPIDataset
    from ppidb.split import Splitter
    from ppidb.split.c1c2c3 import compute_c1c2c3_stats
    from ppidb.sequence import SequenceFetcher
except ImportError:
    print("Error: ppidb package not found. Please install it first (pip install -e .)")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Check for data leakage in PPI datasets."
    )
    
    # Data arguments
    parser.add_argument("dataset", type=str, help="Path to the input PPI dataset (Parquet file)")
    parser.add_argument("--filter-conf", action="store_true", help="Apply high-confidence filter before splitting")
    
    # Split arguments
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["similarity", "greedy_c3", "community_c3"],
        default="similarity",
        help="Split strategy to use (default: similarity)"
    )
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fraction of data/proteins for test set (default: 0.2)")
    parser.add_argument("--val-frac", type=float, default=0.0, help="Fraction of data/proteins for validation set (default: 0.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    # Similarity specific arguments
    parser.add_argument("--identity", type=float, default=0.3, help="Sequence identity threshold for C3 split/strict eval (default: 0.3)")
    
    # Evaluation arguments
    parser.add_argument("--strict", action="store_true", help="Enable strict C1/C2/C3 evaluation")
    parser.add_argument("--leak-threshold", type=float, default=0.5, help="Threshold for C1 fraction to consider 'leaky' (default: 0.5)")

    args = parser.parse_args()

    # 1. Load dataset
    print(f"Loading dataset from {args.dataset}...")
    try:
        ds = PPIDataset.load(args.dataset)
        print(f"Loaded {len(ds):,} pairs.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if args.filter_conf:
        print("Applying high-confidence filter...")
        ds = ds.filter.high_confidence()
        print(f"Remaining pairs: {len(ds):,}")

    if len(ds) == 0:
        print("Error: Dataset is empty.")
        sys.exit(1)

    # 2. Prepare sequences (required for similarity split OR strict evaluation)
    seqs = None
    if args.strategy == "similarity" or args.strict:
        print("Fetching sequences (required for similarity split or strict evaluation)...")
        try:
            fetcher = SequenceFetcher()
            # Fetch for all proteins in the dataset
            proteins = ds.proteins()
            seqs = fetcher.fetch(proteins, as_dict=True)
            print(f"Fetched {len(seqs):,} sequences.")
        except Exception as e:
            print(f"Error fetching sequences: {e}")
            if args.strategy == "similarity":
                print("Similarity split requires sequences. Aborting.")
                sys.exit(1)
            else:
                print("Proceeding without strict evaluation.")
                args.strict = False

    # 3. Perform Split
    print(f"\nPerforming {args.strategy} split...")
    splitter = Splitter(ds)
    
    try:
        if args.strategy == "similarity":
            split = splitter.similarity_split(
                identity_threshold=args.identity,
                test_frac=args.test_frac,
                val_frac=args.val_frac,
                seed=args.seed,
                sequence_dict=seqs
            )
        elif args.strategy == "greedy_c3":
            split = splitter.greedy_c3_split(
                test_protein_frac=args.test_frac,
                val_protein_frac=args.val_frac,
                seed=args.seed
            )
        elif args.strategy == "community_c3":
            split = splitter.community_c3_split(
                test_protein_frac=args.test_frac,
                val_protein_frac=args.val_frac,
                seed=args.seed
            )
    except Exception as e:
        print(f"Error during splitting: {e}")
        sys.exit(1)

    split.summary()

    # 4. Evaluate Leakage
    print("\nEvaluating C1/C2/C3 Leakage...")
    if args.strict:
        print(f"Mode: STRICT (Sequence identity threshold: {args.identity})")
        eval_seqs = seqs
        eval_identity = args.identity
    else:
        print("Mode: STANDARD (Protein ID overlap only)")
        eval_seqs = None
        eval_identity = None

    stats = compute_c1c2c3_stats(
        split.train,
        split.test,
        sequence_dict=eval_seqs,
        identity_threshold=eval_identity
    )
    
    print(stats)

    is_leaky = stats.is_leaky(c1_threshold=args.leak_threshold)
    print("\n" + "="*50)
    if is_leaky:
        print(f"❌ LEAKAGE DETECTED (C1 > {args.leak_threshold*100:.0f}%)")
        print("Recommendation: Use a stricter split strategy or lower the identity threshold.")
    else:
        print(f"✅ NO SIGNIFICANT LEAKAGE (C1 <= {args.leak_threshold*100:.0f}%)")
    print("="*50)


if __name__ == "__main__":
    main()

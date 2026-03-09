
from ppidb import PPIDataset
from ppidb.sequence import SequenceFetcher
from ppidb.negative import NegativeSampler
from ppidb.split import Splitter
from ppidb.split.c1c2c3 import compute_c1c2c3_stats, split_test_by_c1c2c3
import os
import sys
import io

# Force UTF-8 output for Windows terminals
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("## 1. Quick Start: Loading Data")
# Load the full database
if not os.path.exists("ppidb_interaction.parquet"):
    print("Error: ppidb_interaction.parquet not found.")
    exit(1)

ds = PPIDataset.load("ppidb_interaction.parquet")
# Inspect
ds.describe()

print("\n## 2. Filtering")
# Filter by species (Human)
human = ds.filter.by_species("9606")
print(f"Human interactions: {len(human)}")

# Chained filtering
subset = (
    ds.filter.by_species("9606")
      .filter.positives_only()
      .filter.by_throughput("LTP")
      .filter.by_min_sources(2)
      .filter.by_database("BioGRID")
)
print(f"Filtered subset size: {len(subset)}")

print("\n## 3. Sequence Retrieval (Local)")
fetcher = SequenceFetcher()
# Load sequences from local parquet file to avoid web fetching
if os.path.exists("ppidb_protein.parquet"):
    fetcher.load_local_parquet("ppidb_protein.parquet")
else:
    print("Warning: ppidb_protein.parquet not found. Skipping local load.")

# Fetch sequences for a small subset of proteins
if len(subset) > 0:
    sample_proteins = subset.proteins()[:5]
    seqs = fetcher.fetch(sample_proteins, as_dict=True)

    print(f"Fetched {len(seqs)} sequences.")
    for pid, seq in seqs.items():
        print(f"{pid}: {seq[:20]}...")
else:
    print("Subset is empty, skipping sequence fetch.")

print("\n## 4. Negative Sampling")
# Use a smaller sample for quick testing
# Note: sample returns a new dataset
positives = ds.filter.high_confidence().sample(1000, seed=42)
sampler = NegativeSampler(positives)

# Strategy: Random negatives
negs = sampler.random_sample(ratio=1.0, seed=42)
labeled = NegativeSampler.combine(positives, negs, shuffle=True)

print(f"Labeled dataset size: {len(labeled)}")
labeled.describe()

print("\n## 5. Splitting (Greedy C3) & Leakage Evaluation")
# Ensure we have sequences for the proteins in our labeled sample
all_proteins = labeled.proteins()
seqs = fetcher.fetch(all_proteins, as_dict=True)

splitter = Splitter(labeled)

# Using Greedy C3 split
# Reduced fraction for testing speed if needed, but 0.2 is standard
split = splitter.greedy_c3_split(
    test_protein_frac=0.2, 
    seed=42,
    sequence_dict=seqs,
    identity_threshold=0.3
)

print(f"Train size: {len(split.train)}")
print(f"Test size: {len(split.test)}")

# Verify leakage
stats = compute_c1c2c3_stats(split.train, split.test)
print(stats)

print("\n## 6. Full ML Workflow Integration Test")
print("Starting full workflow test...")

# 1. Load and filter (taking a small sample for quick testing)
ds = PPIDataset.load("ppidb_interaction.parquet")
positives = ds.filter.high_confidence().sample(2000, seed=42)

# 2. Generate negatives
negs = NegativeSampler(positives).random_sample(ratio=1.0, seed=42)
labeled = NegativeSampler.combine(positives, negs)

# 3. Load sequences
fetcher = SequenceFetcher()
if os.path.exists("ppidb_protein.parquet"):
    fetcher.load_local_parquet("ppidb_protein.parquet")
all_proteins = labeled.proteins()
seqs = fetcher.fetch(all_proteins, as_dict=True)

# 4. Split (C3-maximizing)
split = Splitter(labeled).greedy_c3_split(
    test_protein_frac=0.2, 
    seed=42, 
    sequence_dict=seqs,
    identity_threshold=0.3
)

# 5. Verify no leakage
stats = compute_c1c2c3_stats(split.train, split.test)
print(stats)

# 6. Fetch sequences for model input
train_seqs = fetcher.fetch(split.train.proteins(), as_dict=True)
test_seqs  = fetcher.fetch(split.test.proteins(),  as_dict=True)

print(f"Workflow completed. Train sequences: {len(train_seqs)}, Test sequences: {len(test_seqs)}")

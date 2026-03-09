# ppidb

A Python toolkit for working with large-scale protein-protein interaction (PPI) databases. Designed for machine learning research on PPI prediction, with built-in support for data filtering, negative sampling, train/test splitting, and rigorous leakage evaluation.

## Installation

### Option 1: Local Installation (Recommended for Linux/Mac)

```bash
pip install -e .
```

Optional dependencies:

```bash
pip install pandas          # for .to_pandas() export
pip install igraph          # for community_c3_split (Louvain community detection)
conda install -c bioconda mmseqs2  # for similarity_split (Recommended for large datasets)
```

### Option 2: Docker (Recommended for Windows)

To ensure all dependencies (including `mmseqs2`) are available and configured correctly, especially on Windows:

```bash
docker build -t ppidb .
docker run -it -v $(pwd):/app ppidb
```

---

## Data

Download the dataset from Baidu Netdisk:
- **Link**: [ppidb_20260309.zip](https://pan.baidu.com/s/1EjwC7ymNVnj2cM0bDkZydQ?pwd=pxke)
- **Access Code**: `pxke`
- **Contents**:
    - `ppidb_interaction.parquet`: The unified PPI interaction dataset (~7.6M pairs).
    - `ppidb_protein.parquet`: Protein sequences (use this locally to avoid fetching from UniProt).

**File Integrity (SHA256)**:
- `ppidb_interaction.parquet`: `831a72e7c47b074ea78c5eb535582d5e7b8f4b9f5c3f8778f1eb65b0dc623cd5`
- `ppidb_protein.parquet`: `f1b52bc99995098cdd060241e961290df9db534d8aa2a581f361b9921986915e`

ppidb is designed to work with a unified PPI parquet file (e.g. `ppidb_interaction.parquet`) that aggregates interactions from multiple public databases.

### Supported source databases

| Database | Type | Notes |
|---|---|---|
| BioGRID | Physical & genetic interactions | Broad organism coverage |
| IntAct | Physical interactions | EMBL-EBI curated |
| STRING | Functional associations | Includes computational predictions |
| MINT | Physical interactions | Manually curated |
| DIP | Physical interactions | Experimentally determined |
| HPRD | Human protein interactions | Human-specific |
| Reactome | Pathway-based interactions | Reaction participants |
| HIPPIE | Human integrated PPI | Confidence-scored |
| Negatome | Non-interacting pairs | Experimentally validated negatives |

### Data schema

Each row in the dataset represents one protein pair:

| Column | Type | Description |
|---|---|---|
| `uniprot_a` | str | UniProt accession of protein A (canonical, alphabetically first) |
| `uniprot_b` | str | UniProt accession of protein B |
| `source_dbs` | str | Pipe-separated list of databases reporting this pair (e.g. `BioGRID\|IntAct`) |
| `n_sources` | int | Number of independent databases supporting this pair |
| `taxon_a` | str | NCBI taxonomy ID of protein A (e.g. `9606` for human) |
| `taxon_b` | str | NCBI taxonomy ID of protein B |
| `detection_methods` | str | Pipe-separated experimental detection methods |
| `interaction_type` | str | `positive` (interacting) or `negative` (Negatome non-interacting) |
| `throughput_type` | str | `LTP` (low-throughput), `HTP` (high-throughput), `both`, `no_exp`, `negative_sample` |

---

## Quick Start

```python
from ppidb import PPIDataset

# Load the full database
ds = PPIDataset.load("ppidb_interaction.parquet")
# PPIDataset(7_669_733 pairs, 39 sources)

# Inspect
ds.describe()
```

---

## Filtering

All filter methods return a new `PPIDataset` (immutable, lazy). Calls can be chained freely.

```python
# Filter by species (NCBI taxonomy ID)
human = ds.filter.by_species("9606")           # human only (both proteins)
cross = ds.filter.by_species("9606", require_both=False)  # at least one human

# Common taxonomy IDs
# 9606    Human
# 10090   Mouse
# 559292  S. cerevisiae
# 7227    D. melanogaster
# 6239    C. elegans
# 3702    A. thaliana

# Filter by throughput type
ltp  = ds.filter.by_throughput("LTP")                  # low-throughput only
exp  = ds.filter.by_throughput("experimental")          # LTP + HTP + both
hc   = ds.filter.by_throughput(["LTP", "both"])         # LTP-validated

# Filter by evidence strength (number of supporting databases)
ds.filter.by_min_sources(2)    # supported by ≥2 databases (recommended)
ds.filter.by_min_sources(4)    # very high confidence

# Filter by source database
ds.filter.by_database("BioGRID")
ds.filter.by_database(["IntAct", "BioGRID"], mode="all")   # in both

# Filter by detection method
ds.filter.by_method("two hybrid")
ds.filter.by_method("biophysical")   # SPR, ITC, NMR, X-ray, FRET, BRET
ds.filter.by_method("coip")          # co-immunoprecipitation variants
ds.filter.by_method("apms")          # AP-MS variants

# Keep only positive or negative pairs
ds.filter.positives_only()
ds.filter.negatives_only()   # Negatome validated non-interacting pairs

# Filter to specific proteins
ds.filter.by_proteins(["P04637", "P53350"])              # pairs involving TP53 or PLK1
ds.filter.by_proteins(my_gene_list, mode="both")         # intra-list pairs only

# Convenience preset: human, positive, LTP-validated, ≥2 sources
ds.filter.high_confidence()
ds.filter.high_confidence(min_sources=4)   # stricter
```

### Chaining filters

```python
subset = (
    ds.filter.by_species("9606")
      .filter.positives_only()
      .filter.by_throughput("LTP")
      .filter.by_min_sources(2)
      .filter.by_database("BioGRID")
)
print(len(subset))
```

---

## Splitting

```python
from ppidb.split import Splitter

splitter = Splitter(ds.filter.high_confidence())
```

### Strategy 1: Similarity-aware split (Default)

Proteins in test/val have < `identity_threshold` sequence identity to any train protein. Prevents leakage from homologs.

```python
from ppidb.sequence import SequenceFetcher
fetcher = SequenceFetcher()
fetcher.load_local_parquet("ppidb_protein.parquet")
seqs = fetcher.fetch(ds.proteins(), as_dict=True)

split = splitter.similarity_split(
    identity_threshold=0.3,
    test_frac=0.1,
    sequence_dict=seqs,
)
```

Requires: `conda install -c bioconda mmseqs2`

### Strategy 2: Greedy C3 split (Sequence-constrained)

Selects a **dense subgraph** as the test pool by iteratively adding the protein with the most connections into the current test pool.
Ensures **strict sequence similarity** separation: no protein in the test pool has > `identity_threshold` similarity to any protein in the train pool.

```python
split = splitter.greedy_c3_split(
    test_protein_frac=0.2, 
    seed=42,
    sequence_dict=seqs,       # Required for strict similarity check
    identity_threshold=0.3
)
```

Produces far more test pairs than simple cold split while maintaining strict leakage control.

### Strategy 3: Community C3 split (Sequence-constrained)

Detects Louvain communities in the PPI graph (via `igraph`) and assigns whole communities to the test pool.
Also enforces **strict sequence similarity** separation.

```python
split = splitter.community_c3_split(
    test_protein_frac=0.2,
    resolution=1.0,   # Louvain resolution; higher = more, smaller communities
    seed=42,
    sequence_dict=seqs,
    identity_threshold=0.3
)
```

Requires: `pip install igraph`

### Saving and loading splits

```python
split.save("my_split/")
# Saves: train.parquet, val.parquet, test.parquet, metadata.json

from ppidb.split import SplitResult
split = SplitResult.load("my_split/")
```

---

## C1/C2/C3 Leakage Evaluation

The C1/C2/C3 framework (Park & Marcotte, 2012) quantifies data leakage in PPI benchmarks by classifying each test pair based on whether its proteins were seen during training:

| Category | Definition | Implication |
|---|---|---|
| **C1** (Both-Seen) | Both proteins appeared in training | High leakage — model may memorize protein embeddings |
| **C2** (One-Seen) | Exactly one protein appeared in training | Partial leakage |
| **C3** (Neither-Seen) | Neither protein appeared in training | No leakage — true generalization test |

Random splits typically yield >90% C1 pairs, causing severely inflated performance estimates.

```python
from ppidb.split.c1c2c3 import compute_c1c2c3_stats, split_test_by_c1c2c3, compare_splits

# Compute stats for a single split
stats = compute_c1c2c3_stats(split.train, split.test)
print(stats)
# C1/C2/C3 Evaluation Statistics
# ==================================================
#   C1 (Both-Seen):       12,345   91.2%  ████████████████████████████░░
#   C2 (One-Seen):         1,100    8.1%  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░
#   C3 (Neither-Seen):        90    0.7%  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
#   Total:                13,535

stats.is_leaky()          # True if C1% > 50%
stats.is_leaky(c1_threshold=0.9)

# Partition the test set into C1/C2/C3 subsets for fine-grained evaluation
c123 = split_test_by_c1c2c3(split.train, split.test)
c123.c1   # PPIDataset of Both-Seen pairs
c123.c2   # PPIDataset of One-Seen pairs
c123.c3   # PPIDataset of Neither-Seen pairs

# Compare multiple strategies side by side
table = compare_splits({
    "random":       (rand.train,      rand.test),
    "cold":         (cold.train,      cold.test),
    "greedy_c3":    (greedy.train,    greedy.test),
    "community_c3": (community.train, community.test),
})
print(table)
```

### Strict C1/C2/C3 Evaluation (Sequence Similarity Aware)

Standard C1/C2/C3 classification only checks if a protein ID appears in the training set. However, a protein in the test set might be a close homolog of a training protein, leading to leakage even if the IDs are different.

To address this, you can enable **strict evaluation** by providing protein sequences and an identity threshold. In this mode:

*   **Train-like protein**: A protein is considered "seen" if it appears in the training set OR if it has > `identity_threshold` sequence identity to *any* training protein.
*   **C1/C2/C3** are then classified based on the count of train-like proteins in the pair (2, 1, or 0).

```python
from ppidb.sequence import SequenceFetcher

# 1. Fetch sequences for all proteins involved
all_proteins = split.train.proteins() + split.test.proteins()
fetcher = SequenceFetcher()
fetcher.load_local_parquet("ppidb_protein.parquet")
seqs = fetcher.fetch(all_proteins, as_dict=True)

# 2. Compute strict statistics
# Proteins with >30% identity to any train protein are treated as "seen"
stats = compute_c1c2c3_stats(
    split.train,
    split.test,
    sequence_dict=seqs,
    identity_threshold=0.3
)
print(stats)

# 3. Strict partition
c123_strict = split_test_by_c1c2c3(
    split.train,
    split.test,
    sequence_dict=seqs,
    identity_threshold=0.3
)
```

---

## Negative Sampling

```python
from ppidb.negative import NegativeSampler

sampler = NegativeSampler(ds.filter.high_confidence())

# Strategy 1: Negatome validated negatives (highest quality, ~3K pairs)
negs = sampler.from_negatome(ds)

# Strategy 2: Random negatives (scalable)
negs = sampler.random_sample(ratio=1.0, seed=42)   # 1:1 positive:negative

# Strategy 3: Subcellular-aware negatives (biologically informed)
compartments = SequenceFetcher().fetch_compartments(ds.proteins())
negs = sampler.subcellular_sample(ratio=1.0, compartment_map=compartments)

# Combine positives and negatives into a labeled dataset
from ppidb.negative import NegativeSampler
labeled = NegativeSampler.combine(positives, negs, shuffle=True)
# interaction_type column: 'positive' / 'negative'
```

---

## Sequence Retrieval

```python
from ppidb.sequence import SequenceFetcher

fetcher = SequenceFetcher()   # caches to ~/.ppidb/sequence_cache/ by default

# Load sequences from local parquet file (if available)
# This avoids fetching from UniProt for proteins present in the file
fetcher.load_local_parquet("ppidb_protein.parquet")

# Fetch as dict (Parallel download with 10 threads)
# If found in local parquet or cache, returns immediately
seqs = fetcher.fetch(["P04637", "P53350"], as_dict=True)
seqs["P04637"][:20]   # 'MEEPQSDPSVEPPLSQETF'

# Automatic fallback
# If batch download fails, it automatically retries with Biopython or single requests
# Failed IDs are logged to failed_fetch.txt

# Fetch all proteins in a dataset and save as FASTA
fetcher.fetch(ds.proteins(), output_fasta="proteins.fasta")

# Fetch subcellular compartments (for subcellular negative sampling)
compartments = fetcher.fetch_compartments(ds.proteins())
# {'P04637': 'Nucleus', 'P53350': 'Cytoplasm', ...}
```

Sequences are cached locally after the first download. Isoform suffixes (e.g. `P04637-2`) are stripped automatically.

---

## Typical ML Workflow

```python
from ppidb import PPIDataset
from ppidb.split import Splitter
from ppidb.split.c1c2c3 import compute_c1c2c3_stats, split_test_by_c1c2c3
from ppidb.negative import NegativeSampler
from ppidb.sequence import SequenceFetcher

# 1. Load and filter
ds = PPIDataset.load("ppidb_interaction.parquet")
positives = ds.filter.high_confidence()

# 2. Generate negatives
negs = NegativeSampler(positives).random_sample(ratio=1.0, seed=42)
labeled = NegativeSampler.combine(positives, negs)

# 3. Split (C3-maximizing for rigorous evaluation)
split = Splitter(labeled).greedy_c3_split(test_protein_frac=0.2, seed=42)

# 4. Verify no leakage
stats = compute_c1c2c3_stats(split.train, split.test)
print(stats)
assert stats.c3_pct == 100.0

# 5. Fine-grained test evaluation
c123 = split_test_by_c1c2c3(split.train, split.test)
# Evaluate your model separately on c123.c1, c123.c2, c123.c3

# 6. Fetch sequences for model input
fetcher = SequenceFetcher()
fetcher.load_local_parquet("ppidb_protein.parquet")  # Pre-load local sequences
train_seqs = fetcher.fetch(split.train.proteins(), as_dict=True)
test_seqs  = fetcher.fetch(split.test.proteins(),  as_dict=True)

# 7. Save split for reproducibility
split.save("splits/greedy_c3_seed42/")
```

---

## Visualization

If `igraph` and `matplotlib` are installed, you can visualize interaction subnetworks directly.

```python
import matplotlib.pyplot as plt

# Visualize interactions for a specific protein (e.g., TP53)
# Limit to first 20 neighbors for clarity
subset = ds.filter.by_proteins(["P04637"]).head(20)

# Plot the ego network
subset.plot_network(
    layout="fruchterman_reingold",
    node_color="skyblue",
    node_size=500,
    with_labels=True,
    title="TP53 Interaction Subnetwork"
)
plt.show()
```

---

## Module Reference

| Module | Class / Function | Description |
|---|---|---|
| `ppidb` | `PPIDataset` | Central data container (Polars LazyFrame backend) |
| `ppidb` | `PPIPair` | Single PPI pair object |
| `ppidb.filter` | `FilterAccessor` | Composable filter chain (via `ds.filter.*`) |
| `ppidb.split` | `Splitter` | Train/val/test splitting strategies |
| `ppidb.split` | `SplitResult` | Split container with `.train`, `.val`, `.test` |
| `ppidb.split.c1c2c3` | `compute_c1c2c3_stats` | Compute C1/C2/C3 leakage statistics |
| `ppidb.split.c1c2c3` | `split_test_by_c1c2c3` | Partition test set into C1/C2/C3 subsets |
| `ppidb.split.c1c2c3` | `compare_splits` | Compare leakage across multiple strategies |
| `ppidb.split.c1c2c3` | `greedy_c3_split` | C3-maximizing greedy split (standalone) |
| `ppidb.split.c1c2c3` | `community_c3_split` | C3-maximizing community split (standalone) |
| `ppidb.negative` | `NegativeSampler` | Generate negative PPI pairs |
| `ppidb.sequence` | `SequenceFetcher` | Batch sequence retrieval from UniProt |

---

## References

- Park & Marcotte (2012). Flaws in evaluation schemes for pair-input computational predictions. *Nature Methods*, 9, 1134–1136.
- Bernett et al. (2024). Cracking the black box of deep sequence-based protein-protein interaction prediction. *Briefings in Bioinformatics*.
- Hamp & Rost (2015). Evolutionary profiles improve protein-protein interaction prediction from sequence. *Bioinformatics*, 31(12), 1945–1950.

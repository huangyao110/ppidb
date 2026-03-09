"""
ppidb — A toolkit for working with large-scale PPI databases.

Design philosophy (inspired by PPIRef):
  - Data-centric: the PPIDataset is the single source of truth
  - Composable filters: chain .filter() calls like a fluent API
  - ML-ready: built-in train/val/test splitting with leakage detection
  - Reproducible: all operations are deterministic given a random seed
  - Fast: Polars backend for lazy evaluation on 7.6M+ pairs

Quick start:
    from ppidb import PPIDataset

    ds = PPIDataset.load("/path/to/ppi_final_uniprot_v6.parquet")

    # Fluent filtering
    human_ltp = (
        ds.filter.by_species("9606")
          .filter.by_throughput("LTP")
          .filter.by_min_sources(2)
    )

    # Split
    from ppidb.split import Splitter
    splits = Splitter(human_ltp).random_split(train=0.8, val=0.1, test=0.1)

    # Negatives
    from ppidb.negative import NegativeSampler
    negatives = NegativeSampler(human_ltp).random_sample(ratio=1.0)

    # Sequences
    from ppidb.sequence import SequenceFetcher
    fasta = SequenceFetcher().fetch(human_ltp.proteins())
"""

from ppidb.core.dataset import PPIDataset
from ppidb.core.pair import PPIPair

__version__ = "0.1.0"
__all__ = ["PPIDataset", "PPIPair"]

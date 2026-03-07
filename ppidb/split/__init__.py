from ppidb.split.splitter import Splitter, SplitResult
from ppidb.split.c1c2c3 import (
    C1C2C3Stats,
    C1C2C3Split,
    get_train_proteins,
    classify_pairs,
    compute_c1c2c3_stats,
    split_test_by_c1c2c3,
    compare_splits,
    greedy_c3_split,
    community_c3_split,
)

__all__ = [
    "Splitter",
    "SplitResult",
    # C1/C2/C3 evaluation
    "C1C2C3Stats",
    "C1C2C3Split",
    "get_train_proteins",
    "classify_pairs",
    "compute_c1c2c3_stats",
    "split_test_by_c1c2c3",
    "compare_splits",
    # C3-maximizing split strategies
    "greedy_c3_split",
    "community_c3_split",
]

"""Legacy dataset-prep entry point.

The old multi-command CLI in this file produced fpid-based split layouts that
did not match the current hash-ID dataset contract. It was intentionally
removed so new datasets are built through one canonical path.

Use:

- ``p2psiglip_db/data/build_merged_hash_c3_split.py`` for fixed-test C3 splits from
  ``data/merged``.
- ``p2psiglip_db/data/build_training_split_collections.py`` for the standard
  p2psiglip / virahinter / rf2ppi hash-ID collections.
- ``p2psiglip_db/data/build_hash_cluster_map.py`` when only an MMseqs cluster map is
  needed.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_args", nargs="*")
    parser.parse_args()
    raise SystemExit(
        "p2psiglip_db/data/dataset_prep.py is retired. Use "
        "build_merged_hash_c3_split.py or build_training_split_collections.py "
        "so every dataset has the same hash-ID structure."
    )


if __name__ == "__main__":
    main()

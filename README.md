# P2PSigLip DB

Database-building utilities for P2PSigLip protein-protein interaction workflows.
The repository keeps source and embedding preparation code in package form and
exposes one command runner, `ppidb.py`, for common database operations.

## Install

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## CLI

Run from the repository root:

```bash
python ppidb.py --help
python ppidb.py commands
python ppidb.py download-data --url gs://<bucket>/<archive>.tar.gz
python ppidb.py merge
python ppidb.py dedup
python ppidb.py split-c3 --help
python ppidb.py validate-merged --quick
python ppidb.py validate --dataset-root data/datasets
```

After editable installation, the same entry point is available as:

```bash
ppidb --help
python -m p2psiglip_db --help
```

Embedding extractors are dispatched through the same runner:

```bash
python ppidb.py embed --list
python ppidb.py embed esmc -i data/merged/sequences.csv -o data/embeds/esmc/
```

Structure helper commands live under the `structure` namespace:

```bash
python ppidb.py structure --help
python ppidb.py structure extract-3di --help
```

## C3 Splits

Use `split-c3` to build positive-only CLIP training data while keeping train
sequences out of C3 clusters touched by validation/test positives:

```bash
python ppidb.py split-c3 \
  --sequences-csv data/datasets/p2psiglip_hash_v1/sequences.csv \
  --test-csv data/datasets/p2psiglip_hash_v1/test.csv \
  --out-dir data/datasets/my_split \
  --identity 0.4 --coverage 0.8 --cov-mode 0
```

The input pair CSV must be exactly `ID_1,ID_2,label`, where IDs are
`md5(sequence)`. Outputs are `train_pos.csv`, optional `val.csv`/`test.csv`,
`sequences.csv`, `holdout_cluster_map.csv`, and `split_report.json`. To sample
holdouts from PPIDB instead of providing files, use `--val-size`, `--test-size`,
`--val-neg-size`, or `--test-neg-size`.

## Data Layout

The build flow is:

```text
data/external/  ->  data/merged/  ->  data/datasets/
```

Large upstream sources and generated outputs are intentionally gitignored. Keep
only code, README files, manifests, and small metadata under version control.

## Download Data

Published database archives can be installed into the local `data/` layout with:

```bash
python ppidb.py download-data --url https://storage.googleapis.com/<bucket>/<archive>.tar.gz
```

`gs://bucket/object` URLs are also accepted for public Google Cloud Storage
objects. The command downloads, safely extracts into `data/`, and then runs
`validate-merged` unless `--no-validate` is provided. Use `PPIDB_DATA_URL` and
`PPIDB_DATA_SHA256` to make the command one-line in production:

```bash
PPIDB_DATA_URL=gs://<bucket>/<archive>.tar.gz python ppidb.py download-data
```

## Merged Database Contract

`data/merged` is treated as a public API surface. Its table names, column order,
ID formats, evidence enums, pair construction rules, row counts, and file hashes are locked in
`p2psiglip_db.data.merged_contract`. Before publishing regenerated merged data,
run:

```bash
python ppidb.py validate-merged --merged-root data/merged
```

Do not rename columns, change `FP0000001`-style IDs, reorder pair endpoints, change snapshot bytes, or
introduce new evidence/source labels without updating the contract and API
consumers in the same change.

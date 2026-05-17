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
python ppidb.py merge
python ppidb.py dedup
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

## Data Layout

The build flow is:

```text
data/external/  ->  data/merged/  ->  data/datasets/
```

Large upstream sources and generated outputs are intentionally gitignored. Keep
only code, README files, manifests, and small metadata under version control.

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

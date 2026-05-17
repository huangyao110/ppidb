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

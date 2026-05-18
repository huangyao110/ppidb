# Repository Guidelines

## Project Structure & Module Organization

This repository builds and maintains a self-contained PPI database for P2PSigLip workflows. The main data flow is `data/external/` raw upstream sources, `data/merged/` integrated master tables, and `data/datasets/` training-ready split collections. Treat `data/external/` as immutable input and regenerate downstream artifacts with scripts.

`p2psiglip_db/data/` is the database construction area. Use it for source merging, evidence normalization, deduplication, merged-contract validation, and legacy dataset builders. `p2psiglip_db/split/` owns reusable split commands such as C3 train/val/test generation. `p2psiglip_db/embeds/` keeps PLM-specific embedding extractors and structure/FASTA helpers used by the database build pipeline.

## Build, Test, and Development Commands

Install dependencies from the repository root:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Merge external sources into master tables:

```bash
python ppidb.py merge
python ppidb.py dedup
```

Validate dataset collections:

```bash
python ppidb.py download-data --url gs://<bucket>/<archive>.tar.gz
python ppidb.py split-c3 --merged data/merged --sequences-csv data/datasets/<name>/sequences.csv --test-csv data/datasets/<name>/test.csv --out-dir data/datasets/<new_name>
python ppidb.py struct -i data/merged/sequences.csv --out-dir data/embeds/strucs/simplefold_100M --limit 100
python ppidb.py 3di foldseek -i data/embeds/manifests/strucs/sequence_structure_sources.tsv --sequence-csv data/merged/sequences.csv -o data/embeds/manifests/strucs/structure_3di_full.fasta
python ppidb.py validate-merged --merged-root data/merged
python ppidb.py validate --dataset-root data/datasets
```

List available database commands with `python ppidb.py commands`. For quick syntax checks, run `python -m compileall p2psiglip_db ppidb.py`.

`data/merged` is an API contract. Keep `proteins.csv`, `sequences.csv`, `interactions.csv`, and `pairs.csv` column names, column order, ID formats, evidence enums, unordered-pair rules, row counts, and SHA256 snapshot hashes aligned with `p2psiglip_db/data/merged_contract.py`. Any database refresh must pass `python ppidb.py validate-merged`.

Published Google Cloud data archives should be installed only through `python ppidb.py download-data`; it extracts safely into `data/` and validates the merged database contract by default.

## Coding Style & Naming Conventions

Use Python 3.11+, 4-space indentation, and clear `snake_case` names for functions, variables, and scripts. Keep command-line options descriptive and consistent with existing `argparse` patterns. Prefer helpers in `p2psiglip_db/data/split_utils.py` or `p2psiglip_db/embeds/io.py` before adding duplicate parsing, validation, or embedding I/O logic.

## Testing Guidelines

There is no formal coverage gate. After code changes, run `python -m compileall p2psiglip_db ppidb.py`. For data-pipeline changes, run the smallest reproducible command on a sample or limited input, then report input paths, output paths, row counts, and missing embeddings. For merged-database changes, run `python ppidb.py validate-merged`; for split logic, run `python ppidb.py split-c3 --help` plus a small smoke run, then validate any generated collection before training.

## Commit & Pull Request Guidelines

Use short imperative, sentence-case commit subjects, for example `Add hash split validation`. Keep commits focused and avoid mixing code changes with regenerated large artifacts. Pull requests should describe the pipeline stage changed, list validation commands, name affected scripts and data directories, and call out any intentionally added reports or metadata files.

## Security & Configuration Tips

Run scripts from the repository root so imports and relative paths resolve. Do not commit large generated CSV, FASTA, HDF5, parquet, or embedding arrays unless explicitly requested. Control embedding GPU selection with `CUDA_VISIBLE_DEVICES`, and keep upstream version metadata in source manifests or README files.

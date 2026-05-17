# `data/` — P2PSigLip data root

Three subdirectories, each with its own README:

| Subdir | Purpose | Size | What lives here |
|---|---|---:|---|
| **[`external/`](external/)** | Raw upstream sources | 1.1 GB | The 7 unmodified third-party PPI databases we pulled from (`bernett_gold`, `dscript`, `pinder`, `ppidb`, `ppiref`, `manifests`, `p2psiglip`). Never edited; gitignored. |
| **[`merged/`](merged/README.md)** | The integrated master database | 4.1 GB | The output of `p2psiglip_db/data/merge_external_sources.py` + `p2psiglip_db/data/dedup_merged.py`: 356,373 deduplicated proteins and 8,717,404 unique pairs across **15 PPI sources** with full provenance (`PPI_Source`, `Experimental_Method`, `Evidence_Type`, `n_sources`, `label`). |
| **[`datasets/`](datasets/)** | Training-ready datasets | 303 MB | One subdir per curated split. Currently: `strict_c3_v1/` — high-quality C3-clean training set (1,019,250 positives) with original P2PSigLip val/test held fixed. |

## Provenance flow

```
external/*  ──┐
              │  p2psiglip_db/data/merge_external_sources.py    (MD5-dedup + per-source ingesters)
              ▼
        merged/_pre-dedup       ──┐
                                  │  p2psiglip_db/data/dedup_merged.py    (polars 32-thread groupby)
                                  ▼
        merged/{interactions,pairs}.csv  ──┐
                                           │  p2psiglip_db/data/build_merged_hash_c3_split.py
                                           │  p2psiglip_db/data/build_training_split_collections.py
                                           ▼
                                 datasets/<name>/
```

Every artifact under `merged/` and `datasets/` is reproducible from `external/` via the scripts in [`tools/`](../tools/).

## Versioning

- `data/external/`: the upstream releases. Their version pins are recorded in each source's `*.json` manifest (e.g. `bernett_gold/figshare_article_21591618_v4.json`, `ppiref/zenodo_record_14845086_v9.json`, PINDER index date `2023-11`).
- `data/merged/`: tagged by the date of the merge run. The current snapshot was produced on 2026-04-28.
- `data/datasets/strict_c3_v1/`: `v1` corresponds to the merge run above + the strict-filter recipe (`--drop-htp-pure --drop-noexp-pure --drop-negative-synthetic`, length 5..1000, MMseqs2 40% id / 80% cov clustering for the C3 split). Bumping any of these knobs creates a new `v2/` etc.

## How NOT to use this folder

- **Don't edit `external/` files in place.** Treat them as immutable upstream releases.
- **Don't commit large CSVs / FASTAs / parquets.** `.gitignore` excludes the bulk content; only READMEs, JSON reports, and figure PNGs end up tracked.
- **Don't rerun merge / split inside this folder.** Tools live in [`tools/`](../tools/) and write outputs into the right subdirs already.

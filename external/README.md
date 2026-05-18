# External Tool Checkouts

This directory is for local third-party structure/PLM tooling used by PPIDB
wrappers. The contents are intentionally gitignored because these are upstream
repositories, binaries, caches, or large model-adjacent assets.

Expected local layout:

- `external/minifold/` contains MiniFold with `predict.py`.
- `external/ml-simplefold/` contains Apple's SimpleFold source tree with
  `src/simplefold/cli.py`.
- `external/ProSST/` contains the ProSST repository used by
  `ppidb.py embed prosst_2048`.
- `external/foldseek/bin/foldseek` is the Foldseek binary used for 3Di
  extraction.

The command wrappers prefer these project-local paths but still allow explicit
overrides, for example:

```bash
python ppidb.py struct --backend minifold --minifold-repo external/minifold --help
python ppidb.py struct simplefold --simplefold-repo external/ml-simplefold --help
python ppidb.py 3di foldseek --foldseek-bin external/foldseek/bin/foldseek --help
```

Install each tool's Python dependencies in the runtime environment used for the
command, or pass the matching `--*-python` option when a separate environment is
needed.

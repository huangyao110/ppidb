"""Command-line entry point for the P2PSigLip database package."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence


DATA_COMMANDS: dict[str, str] = {
    "canonical-fasta": "p2psiglip_db.data.build_canonical_fasta",
    "cluster-map": "p2psiglip_db.data.build_hash_cluster_map",
    "collections": "p2psiglip_db.data.build_training_split_collections",
    "dedup": "p2psiglip_db.data.dedup_merged",
    "download-data": "p2psiglip_db.data.download_database",
    "fix-ppidb-evidence": "p2psiglip_db.data.fix_ppidb_evidence_labels",
    "3di": "p2psiglip_db.embeds.get_3di",
    "get-3di": "p2psiglip_db.embeds.get_3di",
    "h5": "p2psiglip_db.data.create_ppi_h5_esm",
    "host-benchmarks": "p2psiglip_db.data.prepare_host_corpus_benchmarks",
    "host-corpus": "p2psiglip_db.data.prepare_host_pathogen_corpus",
    "host-train": "p2psiglip_db.data.prepare_host_v3_train_dataset",
    "integrate-embeddings": "p2psiglip_db.data.integrate_precomputed_embeddings",
    "link-embeddings": "p2psiglip_db.data.link_existing_embeddings",
    "merge": "p2psiglip_db.data.merge_external_sources",
    "merged-c3": "p2psiglip_db.data.build_merged_hash_c3_split",
    "negatives": "p2psiglip_db.data.build_explicit_pair_negatives",
    "normalize-evidence": "p2psiglip_db.data.normalize_evidence_labels",
    "rf2-benchmark": "p2psiglip_db.data.prepare_rf2_ppi_benchmark",
    "rf2-embeds": "p2psiglip_db.data.prepare_rf2_mplm3_hash_embeds",
    "rf2-interface-tiers": "p2psiglip_db.data.prepare_rf2_ppi_interface_tiers",
    "rf2-train-val": "p2psiglip_db.data.build_rf2_train_plus_filtered_val",
    "split-c3": "p2psiglip_db.split.c3",
    "unified-host-embeds": "p2psiglip_db.data.prepare_unified_host_embeddings",
    "validate": "p2psiglip_db.data.validate_training_split_collections",
    "validate-merged": "p2psiglip_db.data.validate_merged_contract",
}

DATA_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "canonical-fasta": "write canonical FASTA files from hash-ID sequence tables",
    "cluster-map": "build an MMseqs cluster map for hash-ID sequences",
    "collections": "build canonical train/val/test hash-ID dataset collections",
    "dedup": "deduplicate merged interaction pairs after source merging",
    "download-data": "download a published data archive, extract into data/, and validate it",
    "fix-ppidb-evidence": "repair PPIDB evidence labels from original PPIDB metadata",
    "3di": "generate 3Di FASTA with foldseek, ProstT5, or the convolutional head",
    "get-3di": "alias of 3di",
    "h5": "build an HDF5 PPI dataset from pair CSVs and embedding arrays",
    "host-benchmarks": "prepare host/pathogen benchmark split files",
    "host-corpus": "prepare the host/pathogen corpus",
    "host-train": "prepare host/pathogen training data",
    "integrate-embeddings": "copy or register precomputed embedding arrays into the project layout",
    "link-embeddings": "link existing embedding arrays into data/embeds",
    "merge": "merge data/external source files into data/merged master tables",
    "merged-c3": "legacy fixed-test C3 split builder from data/merged",
    "negatives": "build explicit pair-level negative examples",
    "normalize-evidence": "normalize evidence labels and PPI tiers in merged tables",
    "rf2-benchmark": "prepare RF2 PPI benchmark files",
    "rf2-embeds": "prepare RF2 MPLM3 hash embedding inputs",
    "rf2-interface-tiers": "prepare RF2 interface tier annotations",
    "rf2-train-val": "build RF2 train plus filtered validation files",
    "split-c3": "build positive train_pos.csv after C3 filtering against val/test positives",
    "unified-host-embeds": "prepare unified host/pathogen embedding inputs",
    "validate": "validate generated data/datasets split collections",
    "validate-merged": "validate the locked data/merged public API contract",
}

STRUCTURE_COMMANDS: dict[str, str] = {
    "build-source-tsv": "p2psiglip_db.embeds.build_structure_source_tsv",
    "copy-afdb": "p2psiglip_db.embeds.copy_afdb_structures",
    "copy-afdb-manifest": "p2psiglip_db.embeds.copy_afdb_from_uniprot_manifest",
    "download-afdb-manifest": "p2psiglip_db.embeds.download_afdb_from_uniprot_manifest",
    "export-unmatched": "p2psiglip_db.embeds.export_unmatched_sequences",
    "extract-3di": "p2psiglip_db.embeds.extract_structure_3di",
    "3di": "p2psiglip_db.embeds.get_3di",
    "map-afdb": "p2psiglip_db.embeds.map_afdb_uniprot_ids",
    "minifold": "p2psiglip_db.embeds.minifold_predict",
    "predict-3di": "p2psiglip_db.embeds.predict_3di",
    "predict-3di-aa": "p2psiglip_db.embeds.predict_3di_from_aa",
    "simplefold": "p2psiglip_db.embeds.simplefold_predict",
}

STRUCTURE_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "build-source-tsv": "build sequence_structure_sources.tsv from AFDB/MiniFold/SimpleFold outputs",
    "copy-afdb": "copy matching AFDB monomer structures for project sequences",
    "copy-afdb-manifest": "copy AFDB structures using a UniProt manifest",
    "download-afdb-manifest": "download AFDB structures listed in a UniProt manifest",
    "export-unmatched": "export sequences still missing usable structures",
    "extract-3di": "extract Foldseek 3Di sequences from available structures",
    "3di": "unified 3Di FASTA interface: foldseek, ProstT5, or convolutional head",
    "map-afdb": "map project sequence IDs to AFDB/UniProt accessions",
    "minifold": "low-level MiniFold prediction wrapper; prefer top-level struct for new runs",
    "predict-3di": "predict 3Di tokens with the ProstT5 3Di predictor",
    "predict-3di-aa": "predict 3Di tokens directly from amino-acid sequences",
    "simplefold": "low-level SimpleFold prediction wrapper; prefer top-level struct for new runs",
}

STRUCT_BACKENDS: dict[str, str] = {
    "simplefold": "p2psiglip_db.embeds.simplefold_predict",
    "minifold": "p2psiglip_db.embeds.minifold_predict",
}


def _run_module(module_name: str, argv: Sequence[str], display_name: str) -> int:
    old_argv = sys.argv[:]
    sys.argv = [display_name, *argv]
    try:
        runpy.run_module(module_name, run_name="__main__", alter_sys=False)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(exc.code, file=sys.stderr)
        return 1
    finally:
        sys.argv = old_argv
    return 0


def _print_table(items: dict[str, str], *, prefix: str = "", indent: str = "  ") -> None:
    width = max(len(name) for name in items) if items else 0
    for name in sorted(items):
        label = f"{prefix}{name}"
        print(f"{indent}{label:<{width + len(prefix)}}  {items[name]}")


def _print_help() -> None:
    print(
        """PPIDB command runner.

Run database construction, validation, dataset splitting, embeddings, and
structure prediction from one entry point. Commands should be run from the
repository root so relative paths like data/merged resolve correctly.

Usage:
  ppidb.py <command> [args...]
  ppidb.py embed [--list | <plm> args... | --plm <plm> args...]
  ppidb.py struct [--backend simplefold|minifold] [args...]
  ppidb.py structure <command> [args...]
  ppidb.py commands

Primary workflows:
  ppidb.py download-data --url gs://<bucket>/<archive>.tar.gz
      Install a published data archive into data/ and validate data/merged.

  ppidb.py validate-merged --merged-root data/merged
      Check the locked public database contract: headers, IDs, enums, rows,
      pair rules, and snapshot SHA256 values.

  ppidb.py split-c3 --merged data/merged
      --test-csv <test.csv> --sequences-csv <sequences.csv>
      --out-dir data/datasets/<split>
      Build positive-only train_pos.csv after C3 filtering against val/test
      positive endpoints.

  ppidb.py embed esm2 -i data/merged/sequences.csv -o data/embeds/esm2 --pool residue
      Extract PLM embeddings. Most extractors support --pool mean|max|cls|residue.

  ppidb.py 3di foldseek -i data/embeds/manifests/strucs/sequence_structure_sources.tsv
      --sequence-csv data/merged/sequences.csv -o data/embeds/manifests/strucs/structure_3di_full.fasta
      Generate lowercase 3Di FASTA with foldseek, ProstT5 generation, or a
      ProstT5+CNN head.

  ppidb.py struct -i data/merged/sequences.csv -o data/embeds/strucs/simplefold_100M
      Predict monomer structures. Defaults to SimpleFold; use --backend minifold
      for MiniFold.

Important command groups:
  merge / dedup / normalize-evidence
      Rebuild data/merged from raw external sources.
  split-c3 / collections / validate
      Build and validate training-ready data/datasets collections.
  embed
      Dispatch to PLM extractors: esmc, esm2, prott5, prostt5, prostt5_3di,
      saprot, profam, prosst_2048.
  3di / get-3di
      Produce 3Di FASTA for downstream saprot/prostt5_3di embedding.
  struct
      User-facing structure prediction CLI.
  structure
      Lower-level structure utilities: AFDB copy/download, 3Di extraction,
      source manifests, and backend-specific wrappers.

More help:
  ppidb.py commands
  ppidb.py <command> --help
  ppidb.py embed --list
  ppidb.py embed <plm> --help
  ppidb.py struct --help
  ppidb.py struct simplefold --help
  ppidb.py structure <command> --help
"""
    )


def _print_commands() -> None:
    print("Data and dataset commands:")
    _print_table(DATA_COMMAND_DESCRIPTIONS)
    print("\nEmbedding command:")
    print("  embed             run PLM extractors; use `ppidb.py embed --list`")
    print("\nStructure prediction command:")
    print("  struct            predict monomer structures with SimpleFold or MiniFold")
    print("\nLow-level structure utility commands:")
    _print_table(STRUCTURE_COMMAND_DESCRIPTIONS, prefix="structure ")


def _run_embed(argv: Sequence[str]) -> int:
    if argv and argv[0] not in {"-h", "--help", "--list"} and not argv[0].startswith("-"):
        argv = ["--plm", argv[0], *argv[1:]]
    return _run_module("p2psiglip_db.data.get_embeddings", argv, "ppidb embed")


def _print_struct_help() -> None:
    print(
        """Usage:
  ppidb.py struct [--backend simplefold|minifold] [args...]
  ppidb.py struct simplefold [args...]
  ppidb.py struct minifold [args...]

Predict monomer structures from sequence CSV/FASTA inputs. Default backend is
simplefold. Backend-specific args are forwarded unchanged.

Inputs:
  -i/--input may be repeated and accepts CSV files with id,sequence columns or
  FASTA files. If no input is provided, the backend wrappers can auto-discover
  sequence files under data/datasets using --datasets-root and --dataset-glob.

Outputs:
  SimpleFold writes predictions_<model>/ plus simplefold_predictions_manifest.csv.
  MiniFold writes predictions_minifold/ plus minifold_predictions_manifest.csv.
  Structure filenames are keyed by sequence MD5 where possible.

Common options forwarded to both backends:
  -i/--input <path>      sequence CSV/FASTA input; repeat for multiple files
  --out-dir <path>       output directory for structures and manifests
  --limit <n>            cap selected sequences for smoke tests or batches
  --min-len/--max-len    length filters before prediction
  --dry-run              load/select sequences without invoking the backend

Examples:
  ppidb.py struct -i data/merged/sequences.csv --out-dir data/embeds/strucs/simplefold_100M --limit 100
  ppidb.py struct --backend minifold -i data/merged/sequences.csv --out-dir data/embeds/strucs/minifold_48L --limit 100
  ppidb.py struct simplefold --dry-run --limit 10

Backend help:
  ppidb.py struct simplefold --help
  ppidb.py struct minifold --help
"""
    )


def _run_struct(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_struct_help()
        return 0

    backend = "simplefold"
    if args[0] in STRUCT_BACKENDS:
        backend = args[0]
        args = args[1:]
    elif args[0] in {"-b", "--backend"}:
        if len(args) < 2:
            print("missing value for --backend", file=sys.stderr)
            return 2
        backend = args[1]
        args = args[2:]
    elif args[0].startswith("--backend="):
        backend = args[0].split("=", 1)[1]
        args = args[1:]

    module_name = STRUCT_BACKENDS.get(backend)
    if module_name is None:
        print(f"Unknown struct backend: {backend}", file=sys.stderr)
        print("Available backends: " + ", ".join(sorted(STRUCT_BACKENDS)), file=sys.stderr)
        return 2
    return _run_module(module_name, args, f"ppidb struct {backend}")


def _run_structure(argv: Sequence[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            """Usage:
  ppidb.py structure <command> [args...]

Lower-level structure utilities. Use top-level `ppidb.py struct` for routine
SimpleFold/MiniFold structure prediction. Use this namespace for AFDB structure
copy/download, 3Di extraction, source manifests, and backend-specific tools.

Available structure commands:
"""
        )
        _print_table(STRUCTURE_COMMAND_DESCRIPTIONS)
        print(
            """
Examples:
  ppidb.py structure copy-afdb --help
  ppidb.py structure build-source-tsv --help
  ppidb.py structure extract-3di --help
  ppidb.py structure simplefold --help
"""
        )
        return 0

    command, *command_args = argv
    module_name = STRUCTURE_COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown structure command: {command}", file=sys.stderr)
        print("Run `ppidb.py structure --help` for available commands.", file=sys.stderr)
        return 2
    return _run_module(module_name, command_args, f"ppidb structure {command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0
    if args[0] in {"commands", "list"}:
        _print_commands()
        return 0
    if args[0] == "embed":
        return _run_embed(args[1:])
    if args[0] in {"struct", "predict-struct", "generate-struct"}:
        return _run_struct(args[1:])
    if args[0] == "structure":
        return _run_structure(args[1:])

    command, *command_args = args
    module_name = DATA_COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run `ppidb.py commands` for available commands.", file=sys.stderr)
        return 2
    return _run_module(module_name, command_args, f"ppidb {command}")

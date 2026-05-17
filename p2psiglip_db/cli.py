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
    "unified-host-embeds": "p2psiglip_db.data.prepare_unified_host_embeddings",
    "validate": "p2psiglip_db.data.validate_training_split_collections",
    "validate-merged": "p2psiglip_db.data.validate_merged_contract",
}

STRUCTURE_COMMANDS: dict[str, str] = {
    "build-source-tsv": "p2psiglip_db.embeds.build_structure_source_tsv",
    "copy-afdb": "p2psiglip_db.embeds.copy_afdb_structures",
    "copy-afdb-manifest": "p2psiglip_db.embeds.copy_afdb_from_uniprot_manifest",
    "download-afdb-manifest": "p2psiglip_db.embeds.download_afdb_from_uniprot_manifest",
    "export-unmatched": "p2psiglip_db.embeds.export_unmatched_sequences",
    "extract-3di": "p2psiglip_db.embeds.extract_structure_3di",
    "map-afdb": "p2psiglip_db.embeds.map_afdb_uniprot_ids",
    "minifold": "p2psiglip_db.embeds.minifold_predict",
    "predict-3di": "p2psiglip_db.embeds.predict_3di",
    "predict-3di-aa": "p2psiglip_db.embeds.predict_3di_from_aa",
    "simplefold": "p2psiglip_db.embeds.simplefold_predict",
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


def _print_help() -> None:
    print(
        """P2PSigLip database command runner.

Usage:
  ppidb.py <command> [args...]
  ppidb.py embed [--list | <plm> args... | --plm <plm> args...]
  ppidb.py structure <command> [args...]
  ppidb.py commands

Common commands:
  merge                 merge data/external sources into master tables
  dedup                 deduplicate merged interaction pairs
  download-data         download and extract the published data archive
  collections           build canonical hash-ID split collections
  validate              validate generated dataset collections
  validate-merged       validate data/merged API contract
  embed                 run a PLM embedding extractor
  h5                    build a relational PPI HDF5 file

Run command-specific help with:
  ppidb.py <command> --help
  ppidb.py structure <command> --help
"""
    )


def _print_commands() -> None:
    print("Data commands:")
    for name in sorted(DATA_COMMANDS):
        print(f"  {name}")
    print("\nStructure commands:")
    for name in sorted(STRUCTURE_COMMANDS):
        print(f"  structure {name}")
    print("\nEmbedding command:")
    print("  embed")


def _run_embed(argv: Sequence[str]) -> int:
    if argv and argv[0] not in {"-h", "--help", "--list"} and not argv[0].startswith("-"):
        argv = ["--plm", argv[0], *argv[1:]]
    return _run_module("p2psiglip_db.data.get_embeddings", argv, "ppidb embed")


def _run_structure(argv: Sequence[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: ppidb.py structure <command> [args...]\n")
        print("Available structure commands:")
        for name in sorted(STRUCTURE_COMMANDS):
            print(f"  {name}")
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
    if args[0] == "structure":
        return _run_structure(args[1:])

    command, *command_args = args
    module_name = DATA_COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run `ppidb.py commands` for available commands.", file=sys.stderr)
        return 2
    return _run_module(module_name, command_args, f"ppidb {command}")

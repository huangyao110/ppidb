"""Unified 3Di sequence generation interface.

Supported methods:

- ``foldseek``: extract 3Di from existing structure files with Foldseek.
- ``prostt5`` / ``prot5-3di``: predict 3Di from amino-acid sequences using the
  ProstT5 encoder-decoder generator.
- ``conv`` / ``cnn``: predict 3Di from amino-acid sequences using ProstT5
  encoder embeddings plus the convolutional 3Di head.

All methods write a FASTA whose record IDs match the input sequence IDs where
possible and whose sequences are lowercase 3Di letters accepted by downstream
SaProt and ProstT5(3Di) embedding extractors.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Sequence


METHOD_ALIASES = {
    "foldseek": "foldseek",
    "struct": "foldseek",
    "structure": "foldseek",
    "prostt5": "prostt5",
    "prostt5-3di": "prostt5",
    "prot5": "prostt5",
    "prot5-3di": "prostt5",
    "prott5": "prostt5",
    "prott5-3di": "prostt5",
    "prostt5-generate": "prostt5",
    "generate": "prostt5",
    "conv": "conv",
    "cnn": "conv",
    "prostt5-cnn": "conv",
}

METHOD_MODULES = {
    "foldseek": "p2psiglip_db.embeds.extract_structure_3di",
    "prostt5": "p2psiglip_db.embeds.predict_3di",
    "conv": "p2psiglip_db.embeds.predict_3di_from_aa",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate PPIDB-compatible 3Di FASTA files with one of three methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Structure -> 3Di using Foldseek and an existing structure source manifest.
  ppidb.py 3di foldseek -i data/embeds/manifests/strucs/sequence_structure_sources.tsv \\
      --sequence-csv data/merged/sequences.csv \\
      -o data/embeds/manifests/strucs/structure_3di_full.fasta

  # Foldseek also accepts a single PDB/mmCIF file or a directory of structures.
  ppidb.py 3di foldseek -i data/embeds/strucs/simplefold_100M \\
      -o data/embeds/manifests/strucs/structure_3di_full.fasta

  # AA -> 3Di using ProstT5 encoder-decoder generation.
  ppidb.py 3di prostt5 -i data/merged/sequences.csv \\
      -o data/embeds/manifests/3di/prostt5_3di.fasta

  # AA -> 3Di using ProstT5 encoder + convolutional head.
  ppidb.py 3di conv -i data/merged/sequences.csv \\
      -o data/embeds/manifests/3di/conv_3di.fasta

Method notes:
  foldseek  reads a source TSV, one structure file, or a structure directory.
  prostt5   uses Rostlab/ProstT5 generate; slower but direct.
  conv      uses Rostlab/ProstT5 encoder plus the downloaded CNN head.

Use --dry-run to print the forwarded backend command without running it.
""",
    )
    parser.add_argument(
        "method",
        nargs="?",
        metavar="{foldseek,prostt5,conv}",
        default=None,
        help="3Di method. Aliases: struct=foldseek, prot5-3di=prostt5, cnn=conv.",
    )
    parser.add_argument(
        "--method",
        dest="method_flag",
        metavar="{foldseek,prostt5,conv}",
        default=None,
        help="Alternative way to choose the method.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help=(
            "Input path. For foldseek this is sequence_structure_sources.tsv, "
            "one PDB/mmCIF file, or a directory of structures; "
            "for prostt5/conv this is an id,sequence CSV or FASTA."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output 3Di FASTA path.",
    )
    parser.add_argument(
        "--sequence-csv",
        action="append",
        type=Path,
        default=None,
        help="Foldseek only: id,sequence CSV(s) used to filter structures and validate lengths.",
    )
    parser.add_argument(
        "--manifest",
        "--manifest-csv",
        dest="manifest_csv",
        type=Path,
        default=None,
        help="Foldseek only: output extraction manifest CSV.",
    )
    parser.add_argument(
        "--foldseek-bin",
        type=Path,
        default=Path("external/foldseek/bin/foldseek"),
        help="Foldseek binary for method=foldseek.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Foldseek only: temporary work directory.",
    )
    parser.add_argument("--threads", type=int, default=None, help="Foldseek thread count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Backend batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Limit records for a smoke run.")
    parser.add_argument("--resume", action="store_true", help="Foldseek only: append and skip completed records.")
    parser.add_argument("--model", default=None, help="ProstT5 method: Hugging Face model name.")
    parser.add_argument("--max-len", type=int, default=None, help="AA length cap for ProstT5 generation.")
    parser.add_argument("--beams", type=int, default=None, help="ProstT5 generation beam count.")
    parser.add_argument("--model-cache", type=Path, default=None, help="Conv method: Hugging Face cache directory.")
    parser.add_argument("--cnn-cache", type=Path, default=None, help="Conv method: CNN head cache directory.")
    parser.add_argument("--max-residues", type=int, default=None, help="Conv method: residue budget per batch.")
    parser.add_argument("--max-batch", type=int, default=None, help="Conv method: max sequences per batch.")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Conv method: long sequence threshold.")
    parser.add_argument("--case", choices=("lower", "upper"), default=None, help="Conv method: output 3Di case.")
    parser.add_argument("--full-precision", action="store_true", help="Conv method: disable fp16.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the backend module and argv without running it.",
    )
    return parser


def normalize_method(method: str | None) -> str:
    if method is None:
        raise SystemExit("choose a method: foldseek, prostt5, or conv")
    normalized = METHOD_ALIASES.get(method)
    if normalized is None:
        raise SystemExit(f"unknown 3Di method {method!r}; choose foldseek, prostt5, or conv")
    return normalized


def add_path(argv: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def add_value(argv: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def backend_argv(method: str, args: argparse.Namespace, extra: Sequence[str]) -> list[str]:
    argv: list[str] = []
    if method == "foldseek":
        add_path(argv, "--source-tsv", args.input)
        add_path(argv, "--out-fasta", args.output)
        add_path(argv, "--manifest-csv", args.manifest_csv)
        add_path(argv, "--foldseek-bin", args.foldseek_bin)
        add_path(argv, "--work-dir", args.work_dir)
        for sequence_csv in args.sequence_csv or []:
            add_path(argv, "--sequence-csv", sequence_csv)
        add_value(argv, "--threads", args.threads)
        add_value(argv, "--batch-size", args.batch_size)
        add_value(argv, "--limit", args.limit)
        if args.resume:
            argv.append("--resume")
    elif method == "prostt5":
        if args.input is None or args.output is None:
            raise SystemExit("method=prostt5 requires -i/--input and -o/--output")
        add_path(argv, "-i", args.input)
        add_path(argv, "-o", args.output)
        add_value(argv, "--model", args.model)
        add_value(argv, "--batch_size", args.batch_size)
        add_value(argv, "--max_len", args.max_len)
        add_value(argv, "--beams", args.beams)
    elif method == "conv":
        if args.input is None or args.output is None:
            raise SystemExit("method=conv requires -i/--input and -o/--output")
        add_path(argv, "--input", args.input)
        add_path(argv, "--output", args.output)
        add_path(argv, "--model-cache", args.model_cache)
        add_path(argv, "--cnn-cache", args.cnn_cache)
        add_value(argv, "--max-residues", args.max_residues)
        add_value(argv, "--max-batch", args.max_batch)
        add_value(argv, "--max-seq-len", args.max_seq_len)
        add_value(argv, "--case", args.case)
        if args.full_precision:
            argv.append("--full-precision")
    else:
        raise AssertionError(method)
    argv.extend(extra)
    return argv


def run_module(module_name: str, argv: Sequence[str], display_name: str) -> int:
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


def main() -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args()
    method = normalize_method(args.method_flag or args.method)
    module_name = METHOD_MODULES[method]
    argv = backend_argv(method, args, extra)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "method": method,
                    "module": module_name,
                    "argv": argv,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    return run_module(module_name, argv, f"ppidb 3di {method}")


if __name__ == "__main__":
    raise SystemExit(main())

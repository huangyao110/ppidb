"""Batch MiniFold structure prediction for project sequences.

This wrapper mirrors ``simplefold_predict.py`` but invokes MiniFold's
``predict.py`` entrypoint. MiniFold writes PDB files named from the FASTA
record id, so this wrapper uses the project-wide sequence MD5 as the FASTA id.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from p2psiglip_db.embeds.copy_afdb_structures import (
    Target,
    discover_inputs,
    load_targets,
)


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def default_cache() -> Path:
    env_cache = os.environ.get("MINIFOLD_CACHE")
    if env_cache:
        return Path(env_cache)
    existing = Path("/home/zlab/esmdock/minifold_cache")
    if existing.exists():
        return existing
    return Path("data/embeds/strucs/minifold_cache")


def existing_prediction(pred_dir: Path, sequence_md5: str) -> bool:
    return (pred_dir / f"{sequence_md5}.pdb").is_file()


def sequence_for_minifold(sequence: str) -> str:
    """MiniFold PDB conversion supports standard residues plus X."""
    return "".join(res if res in STANDARD_AA else "X" for res in str(sequence).upper())


def write_fasta(path: Path, targets: list[Target]) -> None:
    with path.open("w") as handle:
        for target in targets:
            handle.write(f">{target.sequence_md5}\n{sequence_for_minifold(target.sequence)}\n")


def choose_targets(
    targets: list[Target],
    pred_dir: Path,
    min_len: int,
    max_len: int | None,
    limit: int | None,
) -> list[Target]:
    selected: list[Target] = []
    seen_seq: set[str] = set()
    for target in targets:
        if target.sequence_md5 in seen_seq:
            continue
        seen_seq.add(target.sequence_md5)
        if target.length < min_len:
            continue
        if max_len is not None and target.length > max_len:
            continue
        if existing_prediction(pred_dir, target.sequence_md5):
            continue
        selected.append(target)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def batched(items: list[Target], sequences_per_call: int) -> list[list[Target]]:
    if sequences_per_call <= 0:
        return [items]
    return [items[i : i + sequences_per_call] for i in range(0, len(items), sequences_per_call)]


def clean_work_dir(work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_md5",
        "target_id",
        "length",
        "target_source",
        "prediction_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MiniFold on local sequence CSV/FASTA files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", action="append", type=Path)
    parser.add_argument("--datasets-root", type=Path, default=Path("data/datasets"))
    parser.add_argument("--dataset-glob", default="*hash_v1/sequences.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("data/embeds/strucs/minifold_48L"))
    parser.add_argument("--minifold-repo", type=Path, default=Path("external/minifold"))
    parser.add_argument("--minifold-python", type=Path, default=Path("/home/zlab/miniconda3/envs/minifold/bin/python"))
    parser.add_argument("--cache", type=Path, default=default_cache())
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-size", choices=["48L", "12L"], default="48L")
    parser.add_argument("--token-per-batch", type=int, default=2048)
    parser.add_argument("--sequences-per-call", type=int, default=0, help="0 means one MiniFold call for all selected sequences.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--kernels", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    predict_py = args.minifold_repo / "predict.py"
    if not predict_py.is_file():
        raise FileNotFoundError(predict_py)
    if not args.minifold_python.is_file():
        raise FileNotFoundError(args.minifold_python)

    inputs = args.input
    if not inputs:
        inputs = discover_inputs(args.datasets_root, args.dataset_glob)
        print("auto-discovered inputs:\n  " + "\n  ".join(str(p) for p in inputs), flush=True)
    if not inputs:
        raise SystemExit("no input sequence files found")

    targets = load_targets(inputs)
    pred_dir = args.out_dir / "predictions_minifold"
    selected = choose_targets(
        targets=targets,
        pred_dir=pred_dir,
        min_len=args.min_len,
        max_len=args.max_len,
        limit=args.limit,
    )
    print(f"loaded targets: {len(targets):,}", flush=True)
    print(f"selected for MiniFold: {len(selected):,}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    work_dir = args.out_dir / "work"
    manifest_path = args.manifest or (args.out_dir / "minifold_predictions_manifest.csv")
    summary_path = args.summary_json or (args.out_dir / "minifold_predictions_summary.json")

    batches = batched(selected, args.sequences_per_call)
    if args.dry_run:
        print("dry-run only; not invoking MiniFold", flush=True)
        print(f"batches: {len(batches):,}", flush=True)
        return

    rows: list[dict[str, object]] = []
    start = time.time()

    env = os.environ.copy()
    repo_path = str(args.minifold_repo.resolve())
    env["PYTHONPATH"] = repo_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["MINIFOLD_CACHE"] = str(args.cache.resolve())

    for batch_idx, batch in enumerate(batches, start=1):
        clean_work_dir(work_dir)
        fasta_path = work_dir / f"batch_{batch_idx:06d}.fasta"
        batch_out = work_dir / f"batch_{batch_idx:06d}_out"
        write_fasta(fasta_path, batch)

        cmd = [
            str(args.minifold_python),
            str(predict_py),
            str(fasta_path),
            "--out_dir",
            str(batch_out),
            "--cache",
            str(args.cache),
            "--model_size",
            args.model_size,
            "--token_per_batch",
            str(args.token_per_batch),
        ]
        if args.checkpoint is not None:
            cmd.extend(["--checkpoint", str(args.checkpoint)])
        if args.compile:
            cmd.append("--compile")
        if args.kernels:
            cmd.append("--kernels")

        print(
            f"[batch {batch_idx}/{len(batches)}] running MiniFold on {len(batch):,} sequences",
            flush=True,
        )
        subprocess.run(cmd, check=True, env=env)

        raw_pred_dir = batch_out / f"minifold_results_{fasta_path.stem}"
        for target in batch:
            raw_path = raw_pred_dir / f"{target.sequence_md5}.pdb"
            final_path = pred_dir / f"{target.sequence_md5}.pdb"
            if not raw_path.is_file():
                print(f"warning: missing MiniFold output for {target.sequence_md5}: {raw_path}", flush=True)
                continue
            shutil.move(str(raw_path), str(final_path))
            rows.append(
                {
                    "sequence_md5": target.sequence_md5,
                    "target_id": target.target_id,
                    "length": target.length,
                    "target_source": target.source,
                    "prediction_path": str(final_path),
                }
            )
        write_manifest(manifest_path, rows)

    clean_work_dir(work_dir)
    summary = {
        "inputs": [str(p) for p in inputs],
        "out_dir": str(args.out_dir),
        "prediction_dir": str(pred_dir),
        "minifold_repo": str(args.minifold_repo),
        "model_size": args.model_size,
        "cache": str(args.cache),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "token_per_batch": args.token_per_batch,
        "loaded_targets": len(targets),
        "selected_targets": len(selected),
        "sanitized_targets": sum(sequence_for_minifold(target.sequence) != target.sequence for target in selected),
        "predicted_targets": len(rows),
        "batches": len(batches),
        "elapsed_sec": time.time() - start,
        "manifest": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

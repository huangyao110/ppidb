"""Batch SimpleFold structure prediction for project sequences.

This wrapper converts local CSV/FASTA sequence inputs into one FASTA per unique
sequence, then invokes Apple's SimpleFold CLI. Outputs are named by sequence
MD5, matching the structure/embedding convention used in this repository.

Example:

    python -m p2psiglip_db.embeds.simplefold_predict \
      --input data/datasets/virahinter_hp_hash_v1/sequences.csv \
      --out-dir data/embeds/strucs/simplefold_100M \
      --simplefold-model simplefold_100M \
      --num-steps 500 --tau 0.01
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from p2psiglip_db.embeds.copy_afdb_structures import (
    Target,
    discover_inputs,
    load_targets,
)


def existing_prediction(pred_dir: Path, sequence_md5: str, output_format: str, nsample: int) -> bool:
    suffix = ".cif" if output_format == "mmcif" else ".pdb"
    return all((pred_dir / f"{sequence_md5}_sampled_{i}{suffix}").is_file() for i in range(nsample))


def write_fasta(path: Path, target: Target) -> None:
    path.write_text(f">A|{target.target_id}\n{target.sequence}\n")


def choose_targets(
    targets: list[Target],
    pred_dir: Path,
    output_format: str,
    nsample: int,
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
        if existing_prediction(pred_dir, target.sequence_md5, output_format, nsample):
            continue
        selected.append(target)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def batched(items: list[Target], batch_size: int) -> list[list[Target]]:
    if batch_size <= 0:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def clean_work_inputs(work_dir: Path) -> None:
    for name in ["fasta_batch", "structures", "records"]:
        path = work_dir / name
        if path.exists():
            shutil.rmtree(path)
    manifest = work_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def ensure_shared_boltz_cache(out_dir: Path, ckpt_dir: Path) -> None:
    """Make SimpleFold's hard-coded output_dir/cache reusable across runs."""
    shared_cache = ckpt_dir / "boltz_cache"
    local_cache = out_dir / "cache"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if local_cache.is_symlink():
        if local_cache.resolve() == shared_cache.resolve():
            shared_cache.mkdir(parents=True, exist_ok=True)
            return
        local_cache.unlink()

    if local_cache.exists() and not shared_cache.exists():
        local_cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(local_cache), str(shared_cache))
    elif local_cache.exists():
        for item in local_cache.iterdir():
            dest = shared_cache / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        shutil.rmtree(local_cache)

    shared_cache.mkdir(parents=True, exist_ok=True)
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    if not local_cache.exists():
        local_cache.symlink_to(shared_cache.resolve(), target_is_directory=True)


def default_simplefold_bin() -> Path | None:
    env_bin = os.environ.get("SIMPLEFOLD_BIN")
    if env_bin:
        return Path(env_bin)
    repo_cli = Path("external/ml-simplefold/src/simplefold/cli.py")
    if repo_cli.is_file():
        return None
    legacy = Path("/home/zlab/miniconda3/envs/lucafold/bin/simplefold")
    if legacy.is_file():
        return legacy
    found = shutil.which("simplefold")
    if found:
        return Path(found)
    return None


def default_simplefold_python() -> Path:
    env_python = os.environ.get("SIMPLEFOLD_PYTHON")
    if env_python:
        return Path(env_python)
    legacy = Path("/home/zlab/miniconda3/envs/lucafold/bin/python")
    if legacy.is_file():
        return legacy
    return Path(sys.executable)


def simplefold_command(args) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    if args.simplefold_bin is not None:
        return [str(args.simplefold_bin)], env

    src_dir = args.simplefold_repo / "src"
    cli_py = src_dir / "simplefold" / "cli.py"
    if not cli_py.is_file():
        raise FileNotFoundError(
            f"SimpleFold CLI not found. Pass --simplefold-bin or set --simplefold-repo correctly: {cli_py}"
        )
    env["PYTHONPATH"] = str(src_dir.resolve()) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return [str(args.simplefold_python), "-c", "from simplefold.cli import main; main()"], env


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
        description="Run Apple's SimpleFold on local sequence CSV/FASTA files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", action="append", type=Path)
    parser.add_argument("--datasets-root", type=Path, default=Path("data/datasets"))
    parser.add_argument("--dataset-glob", default="*hash_v1/sequences.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("data/embeds/strucs/simplefold_100M"))
    parser.add_argument("--simplefold-bin", type=Path, default=default_simplefold_bin())
    parser.add_argument("--simplefold-repo", type=Path, default=Path("external/ml-simplefold"))
    parser.add_argument("--simplefold-python", type=Path, default=default_simplefold_python())
    parser.add_argument("--simplefold-model", default="simplefold_100M")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("data/embeds/strucs/simplefold_checkpoints"))
    parser.add_argument("--backend", choices=["torch", "mlx"], default="torch")
    parser.add_argument("--output-format", choices=["mmcif", "pdb"], default="mmcif")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--nsample-per-protein", type=int, default=1)
    parser.add_argument("--plddt", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=0, help="0 means run all selected sequences in one SimpleFold call.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-len", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    inputs = args.input
    if not inputs:
        inputs = discover_inputs(args.datasets_root, args.dataset_glob)
        print("auto-discovered inputs:\n  " + "\n  ".join(str(p) for p in inputs), flush=True)
    if not inputs:
        raise SystemExit("no input sequence files found")

    targets = load_targets(inputs)
    pred_dir = args.out_dir / f"predictions_{args.simplefold_model}"
    selected = choose_targets(
        targets=targets,
        pred_dir=pred_dir,
        output_format=args.output_format,
        nsample=args.nsample_per_protein,
        min_len=args.min_len,
        max_len=args.max_len,
        limit=args.limit,
    )
    print(f"loaded targets: {len(targets):,}", flush=True)
    print(f"selected for SimpleFold: {len(selected):,}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ensure_shared_boltz_cache(args.out_dir, args.ckpt_dir)
    manifest_path = args.manifest or (args.out_dir / "simplefold_predictions_manifest.csv")
    summary_path = args.summary_json or (args.out_dir / "simplefold_predictions_summary.json")

    if args.dry_run:
        print("dry-run only; not invoking SimpleFold", flush=True)
        batches = batched(selected, args.batch_size)
        print(f"batches: {len(batches):,}", flush=True)
        return

    rows: list[dict[str, object]] = []
    suffix = ".cif" if args.output_format == "mmcif" else ".pdb"
    batches = batched(selected, args.batch_size)
    start = time.time()

    for batch_idx, batch in enumerate(batches, start=1):
        clean_work_inputs(args.out_dir)
        fasta_dir = args.out_dir / "fasta_batch"
        fasta_dir.mkdir(parents=True, exist_ok=True)
        for target in batch:
            write_fasta(fasta_dir / f"{target.sequence_md5}.fasta", target)

        launcher, env = simplefold_command(args)
        cmd = [
            *launcher,
            "--simplefold_model",
            args.simplefold_model,
            "--ckpt_dir",
            str(args.ckpt_dir),
            "--output_dir",
            str(args.out_dir),
            "--num_steps",
            str(args.num_steps),
            "--tau",
            str(args.tau),
            "--nsample_per_protein",
            str(args.nsample_per_protein),
            "--fasta_path",
            str(fasta_dir),
            "--output_format",
            args.output_format,
            "--backend",
            args.backend,
            "--seed",
            str(args.seed),
        ]
        if args.plddt:
            cmd.append("--plddt")

        print(
            f"[batch {batch_idx}/{len(batches)}] running SimpleFold on {len(batch):,} sequences",
            flush=True,
        )
        subprocess.run(cmd, check=True, env=env)

        for target in batch:
            pred_path = pred_dir / f"{target.sequence_md5}_sampled_0{suffix}"
            rows.append(
                {
                    "sequence_md5": target.sequence_md5,
                    "target_id": target.target_id,
                    "length": target.length,
                    "target_source": target.source,
                    "prediction_path": str(pred_path),
                }
            )
        write_manifest(manifest_path, rows)

    summary = {
        "inputs": [str(p) for p in inputs],
        "out_dir": str(args.out_dir),
        "prediction_dir": str(pred_dir),
        "simplefold_bin": None if args.simplefold_bin is None else str(args.simplefold_bin),
        "simplefold_repo": str(args.simplefold_repo),
        "simplefold_python": str(args.simplefold_python),
        "simplefold_model": args.simplefold_model,
        "backend": args.backend,
        "output_format": args.output_format,
        "num_steps": args.num_steps,
        "tau": args.tau,
        "nsample_per_protein": args.nsample_per_protein,
        "loaded_targets": len(targets),
        "selected_targets": len(selected),
        "batches": len(batches),
        "elapsed_sec": time.time() - start,
        "manifest": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

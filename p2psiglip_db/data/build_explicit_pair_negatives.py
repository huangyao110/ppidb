"""Build labeled positive/negative pair CSVs for CLIP/SigLIP training.

The output keeps the high-throughput two-tower assumption: negatives are
explicit pair labels used by the loss, not architecture changes.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def read_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ID_1" not in df.columns:
        for left, right in (("fpid_1", "fpid_2"), ("id_1", "id_2")):
            if left in df.columns and right in df.columns:
                df = df.rename(columns={left: "ID_1", right: "ID_2"})
                break
    if "ID_1" not in df.columns or "ID_2" not in df.columns:
        raise SystemExit(f"{path}: missing ID_1/ID_2 columns")
    if "label" not in df.columns:
        df["label"] = 1
    df = df.copy()
    df["ID_1"] = df["ID_1"].astype(str)
    df["ID_2"] = df["ID_2"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def undirected_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def positive_exclusion(paths: list[Path]) -> set[tuple[str, str]]:
    excluded: set[tuple[str, str]] = set()
    for path in paths:
        df = read_pairs(path)
        pos = df[df["label"] == 1]
        excluded.update(undirected_key(a, b) for a, b in zip(pos["ID_1"], pos["ID_2"]) if a != b)
    return excluded


def endpoint_sampler(positives: pd.DataFrame, strategy: str, rng: np.random.Generator):
    proteins = np.array(sorted(set(positives["ID_1"]) | set(positives["ID_2"])), dtype=object)
    if strategy == "uniform":
        probs = None
    elif strategy == "degree":
        counts = Counter(positives["ID_1"].tolist() + positives["ID_2"].tolist())
        weights = np.array([counts[p] for p in proteins], dtype=np.float64)
        probs = weights / weights.sum()
    else:
        raise ValueError(strategy)

    def sample_pair() -> tuple[str, str]:
        idx = rng.choice(len(proteins), size=2, replace=True, p=probs)
        return str(proteins[int(idx[0])]), str(proteins[int(idx[1])])

    return sample_pair, len(proteins)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--exclude-csv", action="append", default=[], type=Path,
                        help="extra CSV whose label=1 undirected pairs must not be sampled as negatives")
    parser.add_argument("--neg-per-pos", type=float, default=10.0)
    parser.add_argument("--max-negatives", type=int, default=None)
    parser.add_argument("--strategy", choices=("degree", "uniform"), default="degree")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-positives", action="store_true",
                        help="write negatives only")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    src = read_pairs(args.positives_csv)
    positives = src[src["label"] == 1][["ID_1", "ID_2"]].drop_duplicates().copy()
    positives = positives[positives["ID_1"] != positives["ID_2"]].copy()
    if positives.empty:
        raise SystemExit(f"{args.positives_csv}: no positive pairs")

    exclude_paths = [args.positives_csv] + list(args.exclude_csv)
    excluded = positive_exclusion(exclude_paths)
    sample_pair, n_proteins = endpoint_sampler(positives, args.strategy, rng)

    target_negatives = int(round(len(positives) * args.neg_per_pos))
    if args.max_negatives is not None:
        target_negatives = min(target_negatives, int(args.max_negatives))
    if target_negatives <= 0 and args.no_positives:
        raise SystemExit("nothing to write: no positives and target negatives <= 0")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    seen_negatives: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(100_000, target_negatives * 200)

    with args.out_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID_1", "ID_2", "label", "category", "source_pair"])
        rows = 0
        if not args.no_positives:
            for a, b in zip(positives["ID_1"], positives["ID_2"]):
                writer.writerow([a, b, 1, "positive", f"pos_{a}_{b}"])
                rows += 1
        while len(seen_negatives) < target_negatives:
            attempts += 1
            if attempts > max_attempts:
                raise SystemExit(
                    f"stopped after {attempts:,} attempts; sampled {len(seen_negatives):,} "
                    f"of {target_negatives:,} negatives"
                )
            a, b = sample_pair()
            if a == b:
                continue
            key = undirected_key(a, b)
            if key in excluded or key in seen_negatives:
                continue
            seen_negatives.add(key)
            writer.writerow([a, b, 0, f"negative_{args.strategy}", f"neg_{args.strategy}_{a}_{b}"])
            rows += 1
            if len(seen_negatives) % 500_000 == 0:
                print(f"sampled {len(seen_negatives):,}/{target_negatives:,} negatives", flush=True)

    print(
        f"wrote {args.out_csv} rows={rows:,} positives={0 if args.no_positives else len(positives):,} "
        f"negatives={len(seen_negatives):,} proteins={n_proteins:,} strategy={args.strategy} "
        f"attempts={attempts:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()

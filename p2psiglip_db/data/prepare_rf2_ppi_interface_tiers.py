"""
Prepare RF2-PPI interface-size tier benchmarks.

The Science RF2-PPI paper evaluates PDB-derived human PPIs stratified by
interface size against the same random negative controls used for the main
benchmark. This script converts those tiered positives and official negatives
to local hp_<sequence-md5> ids while preserving the original rows.

Run:
  python p2psiglip_db/data/prepare_rf2_ppi_interface_tiers.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from p2psiglip_db.data.prepare_rf2_ppi_benchmark import fetch_uniprot, md5_seq, parse_pair


DEFAULT_PARTITIONS = ROOT / "data" / "external" / "rf2_ppi" / "benchmarks" / "pairs_partitioned_by_interface_sizes.tsv"
DEFAULT_CONTROLS = ROOT / "data" / "external" / "rf2_ppi" / "benchmarks" / "positives_and_negatives.tsv"
DEFAULT_OUT = ROOT / "data" / "datasets" / "bench_rf2_ppi"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def tier_name(category: str) -> str:
    text = str(category).lower()
    if "weak" in text or "small" in text:
        return "weak"
    if "strong" in text or "large" in text:
        return "strong"
    if "medium" in text:
        return "medium"
    raise ValueError(f"unknown tier category: {category}")


def build_partition_positives(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    pair_col = "Protein pair" if "Protein pair" in raw.columns else "Protein pairs"
    parsed = raw[pair_col].map(parse_pair)
    out = pd.DataFrame(
        {
            "ID_1": [p[0] for p in parsed],
            "ID_2": [p[1] for p in parsed],
            "label": 1,
            "category": raw["Category"].astype(str),
            "tier": raw["Category"].map(tier_name),
            "source_pair": raw[pair_col].astype(str),
        }
    )
    return out


def build_negative_controls(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    neg = raw[raw["Category"] == "negative"].copy()
    parsed = neg["Protein pairs"].map(parse_pair)
    return pd.DataFrame(
        {
            "ID_1": [p[0] for p in parsed],
            "ID_2": [p[1] for p in parsed],
            "label": 0,
            "category": "negative",
            "tier": "negative",
            "source_pair": neg["Protein pairs"].astype(str),
        }
    )


def hp_id(sequence: str) -> str:
    return f"hp_{md5_seq(sequence)[:16]}"


def to_hp_preserve_rows(
    pairs: pd.DataFrame, sequences: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seqs = sequences.copy()
    seqs["id"] = seqs["id"].astype(str)
    seqs["sequence"] = seqs["sequence"].astype(str).str.strip().str.upper()
    seqs = seqs.dropna(subset=["id", "sequence"]).drop_duplicates("id", keep="first")
    seqs["sequence_md5"] = seqs["sequence"].map(md5_seq)
    seqs["hp_id"] = seqs["sequence"].map(hp_id)

    id_map = dict(zip(seqs["id"], seqs["hp_id"]))
    mapped = pairs.copy()
    mapped["uniprot_ID_1"] = mapped["ID_1"]
    mapped["uniprot_ID_2"] = mapped["ID_2"]
    mapped["ID_1"] = mapped["ID_1"].map(id_map)
    mapped["ID_2"] = mapped["ID_2"].map(id_map)
    missing = mapped[mapped["ID_1"].isna() | mapped["ID_2"].isna()].copy()
    mapped = mapped.dropna(subset=["ID_1", "ID_2"]).copy()

    sequences_hp = (
        seqs.groupby(["hp_id", "sequence_md5", "sequence"], dropna=False)["id"]
        .agg(lambda values: ";".join(sorted(set(map(str, values)))))
        .reset_index()
        .rename(columns={"hp_id": "id", "id": "original_ids"})
    )
    sequences_hp = sequences_hp[["id", "sequence", "sequence_md5", "original_ids"]].copy()
    uniprot_to_hp = seqs[["id", "hp_id", "sequence_md5", "sequence"]].rename(columns={"id": "uniprot_id"})
    return mapped, missing, sequences_hp, uniprot_to_hp


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RF2-PPI weak/medium/strong tier CSVs.")
    parser.add_argument("--partitions", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sequence-cache", type=Path, default=DEFAULT_OUT / "sequences_interface_tiers.csv")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    positives = build_partition_positives(args.partitions)
    negatives = build_negative_controls(args.controls)
    all_pairs = pd.concat([positives, negatives], ignore_index=True)

    accessions = sorted(set(all_pairs["ID_1"]) | set(all_pairs["ID_2"]))
    if not args.sequence_cache.exists() and (args.out / "sequences.csv").exists():
        base = pd.read_csv(args.out / "sequences.csv")
        base.to_csv(args.sequence_cache, index=False)
    sequences = fetch_uniprot(accessions, args.sequence_cache)
    mapped, missing, sequences_hp, uniprot_to_hp = to_hp_preserve_rows(all_pairs, sequences)

    sequences_hp.to_csv(args.out / "sequences_interface_tiers_hp.csv", index=False)
    uniprot_to_hp.to_csv(args.out / "uniprot_to_hp_interface_tiers.csv", index=False)
    mapped.to_csv(args.out / "pairs_interface_tiers_hp.csv", index=False)
    missing.to_csv(args.out / "pairs_interface_tiers_missing.csv", index=False)

    tier_summary = {}
    for tier in ("weak", "medium", "strong"):
        tier_pairs = pd.concat([mapped[mapped["tier"] == tier], mapped[mapped["label"] == 0]], ignore_index=True)
        out_path = args.out / f"pairs_interface_{tier}_hp.csv"
        tier_pairs.to_csv(out_path, index=False)
        labels = tier_pairs["label"].astype(int)
        tier_summary[tier] = {
            "csv": rel(out_path),
            "rows": int(len(tier_pairs)),
            "positives": int(labels.sum()),
            "negatives": int((labels == 0).sum()),
        }

    summary = {
        "partitions": rel(args.partitions),
        "controls": rel(args.controls),
        "sequence_cache": rel(args.sequence_cache),
        "unique_accessions": len(accessions),
        "sequences_found": int(len(sequences)),
        "sequences_missing": int(len(set(accessions) - set(sequences["id"].astype(str)))),
        "mapped_rows": int(len(mapped)),
        "missing_rows": int(len(missing)),
        "all_pairs_csv": rel(args.out / "pairs_interface_tiers_hp.csv"),
        "sequences_hp_csv": rel(args.out / "sequences_interface_tiers_hp.csv"),
        "tiers": tier_summary,
    }
    (args.out / "SUMMARY_interface_tiers.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

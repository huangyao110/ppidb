"""Normalize merged evidence labels into clean single-label classes.

This keeps provenance in ``Evidence_Tags`` and rewrites ``Evidence_Type`` as a
single clean class suitable for filtering and notebook summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from p2psiglip_db.data.merged_contract import INTERACTIONS_COLUMNS, order_interactions
from p2psiglip_db.data.split_utils import pair_key


AA_RE = re.compile(r"[^A-Za-z]")
TAG_ORDER = {
    "structural": 0,
    "HTP": 1,
    "LTP": 2,
    "complex_curation": 3,
    "mixed": 4,
    "no_exp": 5,
    "negative_synthetic": 6,
}
TIER_ZH = {
    "diamond": "钻石",
    "gold": "黄金",
    "silver": "白银",
    "bronze": "青铜",
    "negative_synthetic": "负样本",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Merged interaction CSV(s) to normalize.")
    parser.add_argument("--merged-proteins", type=Path, default=REPO / "data/merged/proteins.csv")
    parser.add_argument("--ppidb-proteins", type=Path, default=REPO / "data/external/ppidb/ppidb_protein.parquet")
    parser.add_argument("--ppidb-interactions", type=Path, default=REPO / "data/external/ppidb/ppidb_interaction.parquet")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Output path for a single input when not using --in-place.")
    parser.add_argument("--report", type=Path, default=REPO / "data/merged/reports/evidence_normalize_report.json")
    return parser.parse_args()


def md5_of(sequence: object) -> str:
    seq_norm = AA_RE.sub("", str(sequence)).upper().rstrip("*")
    return hashlib.md5(seq_norm.encode("utf-8")).hexdigest()


def tokenise(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {tok.strip() for tok in str(value).split(";") if tok.strip()}


def join_tags(tokens: set[str]) -> str:
    return ";".join(sorted(tokens, key=lambda tok: (TAG_ORDER.get(tok, 99), tok)))


def build_ppidb_complex_curation_keys(args: argparse.Namespace) -> set[str]:
    merged = pd.read_csv(args.merged_proteins, usecols=["fpid", "protein_md5"])
    md5_to_fpid = dict(zip(merged["protein_md5"].astype(str), merged["fpid"].astype(str)))

    ppidb_proteins = pd.read_parquet(args.ppidb_proteins, columns=["id", "sequence"])
    ppidb_proteins["protein_md5"] = ppidb_proteins["sequence"].map(md5_of)
    ppidb_proteins["fpid"] = ppidb_proteins["protein_md5"].map(md5_to_fpid)
    id_to_fpid = dict(
        zip(
            ppidb_proteins.loc[ppidb_proteins["fpid"].notna(), "id"].astype(str),
            ppidb_proteins.loc[ppidb_proteins["fpid"].notna(), "fpid"].astype(str),
        )
    )

    ppidb_interactions = pd.read_parquet(
        args.ppidb_interactions,
        columns=["uniprot_a", "uniprot_b", "detection_methods"],
    )
    mask = ppidb_interactions["detection_methods"].fillna("").astype(str).str.contains(
        "complex curation", regex=False
    )
    ppidb_interactions = ppidb_interactions.loc[mask].copy()
    f1 = ppidb_interactions["uniprot_a"].astype(str).map(id_to_fpid)
    f2 = ppidb_interactions["uniprot_b"].astype(str).map(id_to_fpid)
    ok = f1.notna() & f2.notna()
    return set(pair_key(f1.loc[ok], f2.loc[ok]).astype(str))


def clean_class(tokens: set[str], label: int) -> str:
    if label == 0:
        return "negative_synthetic"

    positive_tokens = set(tokens)
    positive_tokens.discard("negative_synthetic")

    if "structural" in positive_tokens:
        return "structural"
    if "HTP" in positive_tokens and "LTP" in positive_tokens:
        return "HTP_LTP"
    if "LTP" in positive_tokens:
        return "LTP"
    if "complex_curation" in positive_tokens:
        return "complex_curation"
    if "HTP" in positive_tokens:
        return "HTP"
    if "mixed" in positive_tokens:
        return "mixed"
    if "no_exp" in positive_tokens:
        return "no_exp"
    if "negative_synthetic" in tokens:
        return "negative_synthetic"
    return "unknown"


def ppi_tier(evidence_type: str, tokens: set[str], label: int, n_sources: int) -> str:
    """Assign a conservative evidence-strength tier for PPI positives."""
    if label == 0 or evidence_type == "negative_synthetic":
        return "negative_synthetic"

    if evidence_type == "structural":
        return "diamond"
    if evidence_type in {"LTP", "HTP_LTP"} and n_sources >= 2:
        return "diamond"

    if evidence_type in {"LTP", "HTP_LTP"}:
        return "gold"
    if evidence_type == "HTP" and n_sources >= 2:
        return "gold"
    if evidence_type == "mixed" and n_sources >= 3:
        return "gold"

    if evidence_type == "HTP":
        return "silver"
    if evidence_type == "mixed" and n_sources >= 2:
        return "silver"
    if evidence_type == "complex_curation" and n_sources >= 2:
        return "silver"

    return "bronze"


def normalize_file(
    input_path: Path,
    output_path: Path,
    complex_curation_keys: set[str],
    chunk_size: int,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    stats: Counter = Counter()
    first = True
    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        required = {"FPid_1", "FPid_2", "PPI_Source", "Evidence_Type", "label"}
        missing = required - set(chunk.columns)
        if missing:
            raise SystemExit(f"{input_path}: missing required columns {sorted(missing)}")

        tag_source_col = "Evidence_Tags" if "Evidence_Tags" in chunk.columns else "Evidence_Type"
        keys = pair_key(chunk["FPid_1"], chunk["FPid_2"])
        n_source_values = (
            chunk["n_sources"].fillna(1).astype(int)
            if "n_sources" in chunk.columns
            else pd.Series([1] * len(chunk), index=chunk.index)
        )
        evidence_tags: list[str] = []
        evidence_type: list[str] = []
        ppi_tiers: list[str] = []
        ppi_tiers_zh: list[str] = []
        for raw_tags, source, label, key, n_sources in zip(
            chunk[tag_source_col],
            chunk["PPI_Source"],
            chunk["label"],
            keys,
            n_source_values,
        ):
            tokens = tokenise(raw_tags)
            if "PPIDB" in tokenise(source) and str(key) in complex_curation_keys:
                tokens.add("complex_curation")
                stats["rows_with_complex_curation_tag"] += 1
            tag_value = join_tags(tokens)
            cls = clean_class(tokens, int(label))
            tier = ppi_tier(cls, tokens, int(label), int(n_sources))
            evidence_tags.append(tag_value)
            evidence_type.append(cls)
            ppi_tiers.append(tier)
            ppi_tiers_zh.append(TIER_ZH[tier])
            stats[f"class_{cls}"] += 1
            stats[f"tier_{tier}"] += 1

        stats["rows_read"] += len(chunk)
        chunk["Evidence_Type"] = evidence_type
        if "Evidence_Tags" in chunk.columns:
            chunk["Evidence_Tags"] = evidence_tags
        else:
            insert_at = chunk.columns.get_loc("Evidence_Type") + 1
            chunk.insert(insert_at, "Evidence_Tags", evidence_tags)
        if "PPI_Tier" in chunk.columns:
            chunk["PPI_Tier"] = ppi_tiers
        else:
            insert_at = chunk.columns.get_loc("Evidence_Tags") + 1
            chunk.insert(insert_at, "PPI_Tier", ppi_tiers)
        if "PPI_Tier_ZH" in chunk.columns:
            chunk["PPI_Tier_ZH"] = ppi_tiers_zh
        else:
            insert_at = chunk.columns.get_loc("PPI_Tier") + 1
            chunk.insert(insert_at, "PPI_Tier_ZH", ppi_tiers_zh)
        if set(INTERACTIONS_COLUMNS).issubset(chunk.columns):
            chunk = order_interactions(chunk)
        chunk.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
        first = False

    return {
        "input": str(input_path),
        "output": str(output_path),
        **{key: int(value) for key, value in sorted(stats.items())},
    }


def main() -> None:
    args = parse_args()
    if args.output is not None and (args.in_place or len(args.inputs) != 1):
        raise SystemExit("--output is only valid with exactly one input and without --in-place")
    if not args.in_place and args.output is None:
        raise SystemExit("provide --in-place or --output")

    complex_curation_keys = build_ppidb_complex_curation_keys(args)
    reports = []
    for input_path in args.inputs:
        input_path = input_path.resolve()
        if args.in_place:
            tmp_path = input_path.with_name(input_path.name + ".evidence_normalize.tmp")
            report = normalize_file(input_path, tmp_path, complex_curation_keys, args.chunk_size)
            os.replace(tmp_path, input_path)
            report["replaced_in_place"] = str(input_path)
        else:
            report = normalize_file(input_path, args.output.resolve(), complex_curation_keys, args.chunk_size)
        reports.append(report)

    payload = {
        "complex_curation_pair_keys": len(complex_curation_keys),
        "class_priority": [
            "negative_synthetic when label=0",
            "structural",
            "HTP_LTP",
            "LTP",
            "complex_curation",
            "HTP",
            "mixed",
            "no_exp",
            "unknown",
        ],
        "files": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

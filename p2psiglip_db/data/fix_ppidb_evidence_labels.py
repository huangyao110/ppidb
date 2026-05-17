"""Repair PPIDB evidence labels in merged interaction CSV files.

Earlier merged tables mapped PPIDB ``throughput_type=both`` and
``throughput_type=no_exp`` to the same ``Evidence_Type=mixed`` token. This
script reconstructs PPIDB pair evidence from the original PPIDB parquet files
and rewrites only the affected ``Evidence_Type`` cells:

  - both -> HTP;LTP
  - no_exp -> no_exp

Other source-level evidence tokens are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from p2psiglip_db.data.split_utils import pair_key


AA_RE = re.compile(r"[^A-Za-z]")
EVIDENCE_ORDER = {
    "HTP": 0,
    "LTP": 1,
    "mixed": 2,
    "no_exp": 3,
    "structural": 4,
    "negative_synthetic": 5,
}
NON_PPIDB_MIXED_SOURCES = {
    "HINT",
    "BIOGRID",
    "MINT",
    "PLM_interact",
    "BERNETT_pos",
    "DSCRIPT_human_train",
    "DSCRIPT_human_test",
    "DSCRIPT_fly",
    "DSCRIPT_mouse",
    "DSCRIPT_worm",
    "DSCRIPT_yeast",
    "DSCRIPT_ecoli",
}
THROUGHPUT_TOKENS = {
    "LTP": ("LTP",),
    "HTP": ("HTP",),
    "both": ("HTP", "LTP"),
    "no_exp": ("no_exp",),
    "negative_sample": ("negative_synthetic",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Merged interaction CSV(s) to repair.")
    parser.add_argument("--merged-proteins", type=Path, default=REPO / "data/merged/proteins.csv")
    parser.add_argument("--ppidb-proteins", type=Path, default=REPO / "data/external/ppidb/ppidb_protein.parquet")
    parser.add_argument("--ppidb-interactions", type=Path, default=REPO / "data/external/ppidb/ppidb_interaction.parquet")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--in-place", action="store_true", help="Replace each input file atomically.")
    parser.add_argument("--output", type=Path, default=None, help="Output path for a single input when not using --in-place.")
    parser.add_argument("--report", type=Path, default=REPO / "data/merged/reports/ppidb_evidence_fix_report.json")
    return parser.parse_args()


def md5_of(sequence: object) -> str:
    seq_norm = AA_RE.sub("", str(sequence)).upper().rstrip("*")
    return hashlib.md5(seq_norm.encode("utf-8")).hexdigest()


def tokenise(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {tok.strip() for tok in str(value).split(";") if tok.strip()}


def join_tokens(tokens: set[str]) -> str:
    return ";".join(sorted(tokens, key=lambda tok: (EVIDENCE_ORDER.get(tok, 99), tok)))


def build_ppidb_pair_evidence(args: argparse.Namespace) -> dict[str, tuple[str, ...]]:
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
        columns=["uniprot_a", "uniprot_b", "throughput_type"],
    )
    f1 = ppidb_interactions["uniprot_a"].astype(str).map(id_to_fpid)
    f2 = ppidb_interactions["uniprot_b"].astype(str).map(id_to_fpid)
    ok = f1.notna() & f2.notna()
    ppidb_interactions = ppidb_interactions.loc[ok].copy()
    ppidb_interactions["_pair_key"] = pair_key(f1.loc[ok], f2.loc[ok])

    pair_to_tokens: dict[str, set[str]] = defaultdict(set)
    for key, throughput in zip(
        ppidb_interactions["_pair_key"].astype(str),
        ppidb_interactions["throughput_type"].astype(str),
    ):
        pair_to_tokens[key].update(THROUGHPUT_TOKENS.get(throughput, ("unknown",)))
    return {key: tuple(sorted(tokens, key=lambda tok: (EVIDENCE_ORDER.get(tok, 99), tok)))
            for key, tokens in pair_to_tokens.items()}


def fix_evidence_value(
    evidence: object,
    source: object,
    key: str,
    ppidb_pair_evidence: dict[str, tuple[str, ...]],
    stats: Counter,
) -> str:
    sources = tokenise(source)
    tokens = tokenise(evidence)
    if "PPIDB" not in sources:
        return join_tokens(tokens)
    ppidb_tokens = ppidb_pair_evidence.get(key)
    if not ppidb_tokens:
        stats["ppidb_rows_without_raw_mapping"] += 1
        return join_tokens(tokens)

    stats["ppidb_rows_seen"] += 1
    before = join_tokens(tokens)

    non_ppidb_sources = sources - {"PPIDB"}
    keep_mixed = bool(non_ppidb_sources & NON_PPIDB_MIXED_SOURCES)
    if "mixed" in tokens and not keep_mixed:
        tokens.remove("mixed")
        stats["mixed_removed"] += 1

    tokens.update(ppidb_tokens)
    if "no_exp" in ppidb_tokens:
        stats["no_exp_added"] += 1
    if "HTP" in ppidb_tokens and "LTP" in ppidb_tokens:
        stats["both_expanded_to_htp_ltp"] += 1

    after = join_tokens(tokens)
    if after != before:
        stats["rows_changed"] += 1
    return after


def repair_file(
    input_path: Path,
    output_path: Path,
    ppidb_pair_evidence: dict[str, tuple[str, ...]],
    chunk_size: int,
) -> dict[str, object]:
    stats: Counter = Counter()
    first = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        required = {"FPid_1", "FPid_2", "PPI_Source", "Evidence_Type"}
        missing = required - set(chunk.columns)
        if missing:
            raise SystemExit(f"{input_path}: missing required columns {sorted(missing)}")
        keys = pair_key(chunk["FPid_1"], chunk["FPid_2"])
        fixed = [
            fix_evidence_value(evidence, source, key, ppidb_pair_evidence, stats)
            for evidence, source, key in zip(chunk["Evidence_Type"], chunk["PPI_Source"], keys)
        ]
        stats["rows_read"] += len(chunk)
        chunk["Evidence_Type"] = fixed
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

    ppidb_pair_evidence = build_ppidb_pair_evidence(args)
    reports = []
    for input_path in args.inputs:
        input_path = input_path.resolve()
        if args.in_place:
            tmp_path = input_path.with_name(input_path.name + ".ppidb_evidence_fix.tmp")
            report = repair_file(input_path, tmp_path, ppidb_pair_evidence, args.chunk_size)
            os.replace(tmp_path, input_path)
            report["replaced_in_place"] = str(input_path)
        else:
            report = repair_file(input_path, args.output.resolve(), ppidb_pair_evidence, args.chunk_size)
        reports.append(report)

    payload = {
        "ppidb_pair_evidence_rows": len(ppidb_pair_evidence),
        "throughput_mapping": THROUGHPUT_TOKENS,
        "files": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

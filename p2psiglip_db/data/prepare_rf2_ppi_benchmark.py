"""
Prepare RF2-PPI benchmark files for local V3 evaluation.

Inputs:
  data/external/rf2_ppi/benchmarks/positives_and_negatives.tsv

Outputs:
  data/datasets/bench_rf2_ppi/
    sequences.csv                 UniProt accession -> sequence
    sequences_hp.csv              hp_<sequence-md5-prefix> -> sequence
    uniprot_to_hp.csv             UniProt accession -> hp id mapping
    pairs_1to10.csv               official RF2 positives/negatives
    pairs_1to10_hp.csv            official RF2 pairs converted to hp ids
    pairs_1to1000_hp.csv          sampled 1:1000 benchmark from mapped hp pool
    positives_hp.csv              positives only for VS/retrieval eval
    strict_vh_md5_map.csv         md5 -> hp_id for linking existing embeddings
    SUMMARY.json

Run:
  python p2psiglip_db/data/prepare_rf2_ppi_benchmark.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RF2 = ROOT / "data" / "external" / "rf2_ppi" / "benchmarks" / "positives_and_negatives.tsv"
DEFAULT_STRICT = ROOT / "data" / "datasets" / "strict_vh_v1" / "proteins.csv"
DEFAULT_OUT = ROOT / "data" / "datasets" / "bench_rf2_ppi"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/accessions"


def md5_seq(seq: str) -> str:
    return hashlib.md5(seq.strip().upper().encode("utf-8")).hexdigest()


def parse_pair(pair: str) -> tuple[str, str]:
    a, b = str(pair).split("_", 1)
    return a.strip(), b.strip()


def parse_fasta(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cur_id = None
    cur_seq: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                out[cur_id] = "".join(cur_seq)
            head = line[1:]
            tokens = head.split("|")
            cur_id = tokens[1] if len(tokens) >= 3 else head.split()[0]
            cur_seq = []
        else:
            cur_seq.append(line)
    if cur_id is not None:
        out[cur_id] = "".join(cur_seq)
    return out


def fetch_uniprot(accessions: list[str], cache_csv: Path, batch_size: int = 200) -> pd.DataFrame:
    if cache_csv.exists():
        cached = pd.read_csv(cache_csv)
        have = set(cached["id"].astype(str))
    else:
        cached = pd.DataFrame(columns=["id", "sequence"])
        have = set()

    missing = [acc for acc in accessions if acc not in have]
    new_records: dict[str, str] = {}
    for start in range(0, len(missing), batch_size):
        chunk = missing[start : start + batch_size]
        params = urllib.parse.urlencode({"accessions": ",".join(chunk), "format": "fasta", "size": 500})
        url = f"{UNIPROT_URL}?{params}"
        with urllib.request.urlopen(url, timeout=120) as resp:
            new_records.update(parse_fasta(resp.read().decode("utf-8")))
        print(f"fetched {min(start + batch_size, len(missing)):,}/{len(missing):,} missing UniProt sequences", flush=True)
        time.sleep(0.1)

    if new_records:
        new_df = pd.DataFrame({"id": list(new_records), "sequence": list(new_records.values())})
        cached = pd.concat([cached, new_df], ignore_index=True)
        cached = cached.drop_duplicates("id", keep="first").sort_values("id")
        cached.to_csv(cache_csv, index=False)

    return cached[cached["id"].isin(accessions)].copy()


def build_pairs(rf2_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(rf2_path, sep="\t")
    parsed = raw["Protein pairs"].map(parse_pair)
    pairs = pd.DataFrame(
        {
            "ID_1": [p[0] for p in parsed],
            "ID_2": [p[1] for p in parsed],
            "label": raw["Category"].map({"positive": 1, "negative": 0}).astype(int),
            "category": raw["Category"],
            "source_pair": raw["Protein pairs"],
        }
    )
    return pairs


def sample_1to1000(pairs: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    positives = pairs[pairs["label"] == 1].copy()
    protein_ids = sorted(set(pairs["ID_1"]) | set(pairs["ID_2"]))
    positive_keys = {tuple(sorted((a, b))) for a, b in zip(positives["ID_1"], positives["ID_2"])}
    official_negative_keys = {
        tuple(sorted((a, b))) for a, b in zip(pairs.loc[pairs["label"] == 0, "ID_1"], pairs.loc[pairs["label"] == 0, "ID_2"])
    }

    negative_keys = set(official_negative_keys)
    target_negatives = len(positives) * 1000
    attempts = 0
    max_attempts = target_negatives * 20
    while len(negative_keys) < target_negatives and attempts < max_attempts:
        a, b = rng.sample(protein_ids, 2)
        key = tuple(sorted((a, b)))
        if key not in positive_keys:
            negative_keys.add(key)
        attempts += 1

    if len(negative_keys) < target_negatives:
        raise RuntimeError(f"sampled only {len(negative_keys):,}/{target_negatives:,} negatives")

    selected = sorted(negative_keys)[:target_negatives]
    negatives = pd.DataFrame(
        {
            "ID_1": [a for a, _ in selected],
            "ID_2": [b for _, b in selected],
            "label": 0,
            "category": "negative_sampled_1to1000",
            "source_pair": [f"{a}_{b}" for a, b in selected],
        }
    )
    return pd.concat([positives, negatives], ignore_index=True)


def sequence_to_hp_id(seq: str) -> str:
    return f"hp_{md5_seq(seq)[:16]}"


def to_hp_tables(sequences: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seqs = sequences.copy()
    seqs["sequence"] = seqs["sequence"].astype(str).str.strip().str.upper()
    seqs["sequence_md5"] = seqs["sequence"].map(md5_seq)
    seqs["hp_id"] = seqs["sequence_md5"].map(lambda md5: f"hp_{md5[:16]}")

    uniprot_to_hp = seqs[["id", "hp_id", "sequence_md5", "sequence"]].rename(columns={"id": "uniprot_id"})
    grouped = (
        uniprot_to_hp.groupby(["hp_id", "sequence_md5", "sequence"], dropna=False)["uniprot_id"]
        .agg(lambda values: ";".join(sorted(set(map(str, values)))))
        .reset_index()
        .rename(columns={"hp_id": "id", "uniprot_id": "original_ids"})
    )
    sequences_hp = grouped[["id", "sequence", "sequence_md5", "original_ids"]].copy()

    id_map = dict(zip(uniprot_to_hp["uniprot_id"], uniprot_to_hp["hp_id"]))
    mapped = pairs.copy()
    mapped["ID_1"] = mapped["ID_1"].map(id_map)
    mapped["ID_2"] = mapped["ID_2"].map(id_map)
    mapped = mapped.dropna(subset=["ID_1", "ID_2"]).copy()
    mapped = mapped[mapped["ID_1"] != mapped["ID_2"]].drop_duplicates(["ID_1", "ID_2", "label"])
    return sequences_hp, uniprot_to_hp, mapped


def write_strict_md5_map(strict_proteins: Path, out_path: Path) -> None:
    strict = pd.read_csv(strict_proteins, usecols=["id", "md5"])
    strict = strict.rename(columns={"id": "fpid", "md5": "protein_md5"})
    strict[["protein_md5", "fpid"]].to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RF2-PPI benchmark CSVs.")
    parser.add_argument("--rf2", type=Path, default=DEFAULT_RF2)
    parser.add_argument("--strict-proteins", type=Path, default=DEFAULT_STRICT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(args.rf2)
    accessions = sorted(set(pairs["ID_1"]) | set(pairs["ID_2"]))
    sequences = fetch_uniprot(accessions, args.out / "sequences.csv")
    sequences_hp, uniprot_to_hp, pairs_hp = to_hp_tables(sequences, pairs)

    pairs_1to10 = pairs.copy()
    pairs_1to10_hp = pairs_hp.copy()
    pairs_1to1000_hp = sample_1to1000(pairs_1to10_hp, args.seed)
    positives_hp = pairs_1to10_hp[pairs_1to10_hp["label"] == 1].copy()

    sequences_hp.to_csv(args.out / "sequences_hp.csv", index=False)
    uniprot_to_hp.to_csv(args.out / "uniprot_to_hp.csv", index=False)
    pairs_1to10.to_csv(args.out / "pairs_1to10.csv", index=False)
    pairs_1to10_hp.to_csv(args.out / "pairs_1to10_hp.csv", index=False)
    pairs_1to1000_hp.to_csv(args.out / "pairs_1to1000_hp.csv", index=False)
    positives_hp.to_csv(args.out / "positives_hp.csv", index=False)
    write_strict_md5_map(args.strict_proteins, args.out / "strict_vh_md5_map.csv")

    summary = {
        "rf2_source": str(args.rf2.relative_to(ROOT) if args.rf2.is_relative_to(ROOT) else args.rf2),
        "unique_proteins_in_pairs": len(accessions),
        "sequences_found": int(len(sequences)),
        "sequences_missing": int(len(set(accessions) - set(sequences["id"].astype(str)))),
        "hp_sequences": int(len(sequences_hp)),
        "pairs_1to10": {
            "rows": int(len(pairs_1to10)),
            "positives": int((pairs_1to10["label"] == 1).sum()),
            "negatives": int((pairs_1to10["label"] == 0).sum()),
        },
        "pairs_1to10_hp": {
            "rows": int(len(pairs_1to10_hp)),
            "positives": int((pairs_1to10_hp["label"] == 1).sum()),
            "negatives": int((pairs_1to10_hp["label"] == 0).sum()),
        },
        "pairs_1to1000_hp": {
            "rows": int(len(pairs_1to1000_hp)),
            "positives": int((pairs_1to1000_hp["label"] == 1).sum()),
            "negatives": int((pairs_1to1000_hp["label"] == 0).sum()),
            "seed": args.seed,
        },
    }
    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

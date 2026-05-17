"""Build a per-sequence structure source table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record sequence and structure provenance for every project sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mapping-csv", type=Path, default=Path("data/embeds/manifests/strucs/full_sequence_uniprot_ids.csv"))
    parser.add_argument("--afdb-local-manifest", type=Path, default=Path("data/embeds/manifests/strucs/afdb_matched_cif_manifest.csv"))
    parser.add_argument("--afdb-download-manifest", type=Path, default=Path("data/embeds/manifests/strucs/afdb_downloaded_cif_manifest.csv"))
    parser.add_argument(
        "--minifold-manifest",
        action="append",
        type=Path,
        default=None,
        help="MiniFold manifest to use for provenance. Can be passed more than once.",
    )
    parser.add_argument("--afdb-dir", type=Path, default=Path("data/embeds/strucs/afdb_matched"))
    parser.add_argument("--minifold-dir", type=Path, default=Path("data/embeds/strucs/minifold_48L_unmatched_hash_v1/predictions_minifold"))
    parser.add_argument("--out-tsv", type=Path, default=Path("data/embeds/manifests/strucs/sequence_structure_sources.tsv"))
    parser.add_argument("--summary-json", type=Path, default=Path("data/embeds/manifests/strucs/sequence_structure_sources_summary.json"))
    return parser


def load_manifest(path: Path, status_col: str = "status") -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if status_col not in df.columns:
        df[status_col] = ""
    return df


def main() -> None:
    args = build_parser().parse_args()
    if not args.mapping_csv.is_file():
        raise FileNotFoundError(args.mapping_csv)

    mapping = pd.read_csv(args.mapping_csv).copy()
    mapping["sequence_md5"] = mapping["sequence_md5"].astype(str)

    local = load_manifest(args.afdb_local_manifest)
    local_ok: dict[str, str] = {}
    if not local.empty:
        for row in local.itertuples(index=False):
            row_dict = row._asdict()
            if str(row_dict.get("status", "")) in {"copied", "exists"}:
                local_ok[str(row_dict["sequence_md5"])] = str(row_dict.get("output_path", ""))

    downloaded = load_manifest(args.afdb_download_manifest)
    download_status: dict[str, tuple[str, str]] = {}
    if not downloaded.empty:
        for row in downloaded.itertuples(index=False):
            row_dict = row._asdict()
            download_status[str(row_dict["sequence_md5"])] = (
                str(row_dict.get("status", "")),
                str(row_dict.get("output_path", "")),
            )

    minifold_manifests = args.minifold_manifest or [
        Path("data/embeds/strucs/minifold_48L_unmatched_hash_v1/minifold_predictions_manifest.csv")
    ]
    minifold_ok: dict[str, tuple[str, str]] = {}
    for manifest_path in minifold_manifests:
        minifold = pd.read_csv(manifest_path) if manifest_path.is_file() else pd.DataFrame()
        if minifold.empty:
            continue
        for row in minifold.itertuples(index=False):
            row_dict = row._asdict()
            path = Path(str(row_dict.get("prediction_path", "")))
            if path.is_file():
                minifold_ok[str(row_dict["sequence_md5"])] = (str(path), str(manifest_path))

    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for row in mapping.itertuples(index=False):
        row_dict = row._asdict()
        md5 = str(row_dict["sequence_md5"])
        matched = int(row_dict["matched"])
        afdb_path = args.afdb_dir / f"{md5}.cif.gz"
        minifold_path = args.minifold_dir / f"{md5}.pdb"

        structure_source = "none"
        structure_status = "missing"
        structure_path = ""
        source_manifest = ""

        if md5 in local_ok and afdb_path.is_file():
            structure_source = "afdb_local"
            structure_status = "available"
            structure_path = str(afdb_path)
            source_manifest = str(args.afdb_local_manifest)
        elif md5 in download_status and download_status[md5][0] in {"downloaded", "exists"} and afdb_path.is_file():
            structure_source = "afdb_download"
            structure_status = "available"
            structure_path = str(afdb_path)
            source_manifest = str(args.afdb_download_manifest)
        elif matched == 1 and afdb_path.is_file():
            structure_source = "afdb_download"
            structure_status = "available"
            structure_path = str(afdb_path)
            source_manifest = "file_exists_without_final_manifest"
        elif md5 in minifold_ok:
            structure_source = "minifold"
            structure_status = "available"
            structure_path, source_manifest = minifold_ok[md5]
        elif minifold_path.is_file():
            structure_source = "minifold"
            structure_status = "available"
            structure_path = str(minifold_path)
            source_manifest = "file_exists_without_final_manifest"
        elif matched == 1:
            structure_source = "afdb_pending_or_unavailable"
            structure_status = "pending"
        else:
            structure_source = "minifold_pending"
            structure_status = "pending"

        counts[structure_source] = counts.get(structure_source, 0) + 1
        rows.append(
            {
                "sequence_md5": md5,
                "target_id": row_dict.get("target_id", ""),
                "length": row_dict.get("length", ""),
                "sequence_source_files": row_dict.get("source_files", ""),
                "uniprot_matched": matched,
                "uniprot_id": row_dict.get("uniprot_id", ""),
                "afdb_header": row_dict.get("afdb_header", ""),
                "structure_source": structure_source,
                "structure_status": structure_status,
                "structure_path": structure_path,
                "source_manifest": source_manifest,
            }
        )

    out = pd.DataFrame(rows)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_tsv, sep="\t", index=False)

    summary = {
        "mapping_csv": str(args.mapping_csv),
        "out_tsv": str(args.out_tsv),
        "rows": len(out),
        "counts": counts,
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

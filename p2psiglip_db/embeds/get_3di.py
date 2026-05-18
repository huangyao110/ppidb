"""Generate 3Di FASTA with Foldseek, ProstT5, or ProstT5 plus CNN."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from urllib import request

import pandas as pd
from Bio import SeqIO

from p2psiglip_db.embeds.io import load_input_dataframe


warnings.filterwarnings("ignore")

DEFAULT_FOLDSEEK_SOURCE = Path("data/embeds/manifests/strucs/sequence_structure_sources.tsv")
DEFAULT_FOLDSEEK_FASTA = Path("data/embeds/manifests/strucs/structure_3di_full.fasta")
DEFAULT_FOLDSEEK_MANIFEST = Path("data/embeds/manifests/strucs/structure_3di_full_manifest.csv")
DEFAULT_FOLDSEEK_WORK = Path("data/embeds/manifests/strucs/foldseek_3di_work")
DEFAULT_FOLDSEEK_BIN = Path("external/foldseek/bin/foldseek")
DEFAULT_PROSTT5_MODEL = "Rostlab/ProstT5"
CNN_WEIGHTS_URL = "https://github.com/mheinzinger/ProstT5/raw/main/cnn_chkpnt/model.pt"

STRUCTURE_SUFFIXES = (".pdb", ".pdb.gz", ".cif", ".cif.gz", ".mmcif", ".mmcif.gz")
MD5_RE = re.compile(r"[0-9a-f]{32}")
RARE_AA_RE = re.compile(r"[UZOB]")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

AA2FOLD_TOKEN_ID = 149
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 2
THREE_DI_TOKEN_IDS = {
    "a": 128,
    "l": 129,
    "g": 130,
    "v": 131,
    "s": 132,
    "r": 133,
    "e": 134,
    "d": 135,
    "t": 136,
    "i": 137,
    "p": 138,
    "k": 139,
    "f": 140,
    "q": 141,
    "n": 142,
    "y": 143,
    "m": 144,
    "h": 145,
    "w": 146,
    "c": 147,
}
ID_TO_3DI = {v: k for k, v in THREE_DI_TOKEN_IDS.items()}
SS_MAPPING = "ACDEFGHIKLMNPQRSTVWY"
AA_LETTER_TO_ID = {
    "A": 3,
    "L": 4,
    "G": 5,
    "V": 6,
    "S": 7,
    "R": 8,
    "E": 9,
    "D": 10,
    "T": 11,
    "I": 12,
    "P": 13,
    "K": 14,
    "F": 15,
    "Q": 16,
    "N": 17,
    "Y": 18,
    "M": 19,
    "H": 20,
    "W": 21,
    "C": 22,
    "X": UNK_TOKEN_ID,
}
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
    "conv": "conv",
    "cnn": "conv",
    "prostt5-cnn": "conv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate PPIDB-compatible 3Di FASTA files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  ppidb.py 3di foldseek -i data/embeds/strucs/simplefold_100M \\
      -o data/embeds/manifests/strucs/structure_3di_full.fasta
  ppidb.py 3di foldseek -i data/embeds/manifests/strucs/sequence_structure_sources.tsv \\
      --sequence-csv data/merged/sequences.csv \\
      -o data/embeds/manifests/strucs/structure_3di_full.fasta
  ppidb.py 3di prostt5 -i data/merged/sequences.csv \\
      -o data/embeds/manifests/3di/prostt5_3di.fasta
  ppidb.py 3di conv -i data/merged/sequences.csv \\
      -o data/embeds/manifests/3di/conv_3di.fasta

Methods:
  foldseek  structure file, structure directory, or sequence_structure_sources.tsv
  prostt5   amino-acid CSV/FASTA to 3Di with ProstT5 generate
  conv      amino-acid CSV/FASTA to 3Di with ProstT5 encoder plus CNN head
""",
    )
    parser.add_argument("method", nargs="?", metavar="{foldseek,prostt5,conv}")
    parser.add_argument("-i", "--input", type=Path, help="Input path for the selected method.")
    parser.add_argument("-o", "--output", type=Path, help="Output 3Di FASTA.")
    parser.add_argument("--limit", type=int, help="Limit records for a smoke run.")
    parser.add_argument("--batch-size", type=int, help="Batch size for foldseek/prostt5.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved method and paths only.")

    parser.add_argument("--sequence-csv", action="append", type=Path, help="Foldseek: id,sequence CSV/FASTA.")
    parser.add_argument("--manifest", "--manifest-csv", dest="manifest", type=Path, help="Foldseek manifest CSV.")
    parser.add_argument("--foldseek-bin", type=Path, default=DEFAULT_FOLDSEEK_BIN)
    parser.add_argument("--work-dir", type=Path, help="Foldseek temporary work directory.")
    parser.add_argument("--threads", type=int, default=16, help="Foldseek thread count.")
    parser.add_argument("--resume", action="store_true", help="Foldseek: append and skip successful rows.")

    parser.add_argument("--model", default=DEFAULT_PROSTT5_MODEL, help="ProstT5 Hugging Face model name.")
    parser.add_argument("--max-len", type=int, default=1024, help="ProstT5 AA length cap.")
    parser.add_argument("--beams", type=int, default=1, help="ProstT5 generation beams.")

    parser.add_argument("--model-cache", type=Path, default=Path.home() / ".cache" / "huggingface")
    parser.add_argument("--cnn-cache", type=Path, default=Path.home() / ".cache" / "prostt5_cnn")
    parser.add_argument("--max-residues", type=int, default=0, help="Conv residue budget per batch.")
    parser.add_argument("--max-batch", type=int, default=500, help="Conv max sequences per batch.")
    parser.add_argument("--max-seq-len", type=int, default=1000, help="Conv long sequence threshold.")
    parser.add_argument("--case", choices=("lower", "upper"), default="lower", help="Conv output case.")
    parser.add_argument("--full-precision", action="store_true", help="Conv: disable fp16 on GPU/MPS.")
    return parser


def normalize_method(method: str | None) -> str:
    if method is None:
        raise SystemExit("choose a 3Di method: foldseek, prostt5, or conv")
    resolved = METHOD_ALIASES.get(method)
    if resolved is None:
        raise SystemExit(f"unknown 3Di method {method!r}; choose foldseek, prostt5, or conv")
    return resolved


def require_path(path: Path | None, flag: str) -> Path:
    if path is None:
        raise SystemExit(f"{flag} is required for this method")
    return path


def write_fasta(records: dict[str, str], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record_id, seq in records.items():
            handle.write(f">{record_id}\n{seq}\n")
    return len(records)


def is_structure_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in STRUCTURE_SUFFIXES)


def structure_id(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in STRUCTURE_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def source_from_structures(paths: list[Path], source: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_md5": structure_id(path),
                "structure_status": "available",
                "structure_path": str(path),
                "structure_source": source,
            }
            for path in paths
        ],
        columns=["sequence_md5", "structure_status", "structure_path", "structure_source"],
    )


def read_structure_source(path: Path) -> tuple[pd.DataFrame, bool]:
    if path.is_dir():
        structures = sorted(p for p in path.rglob("*") if p.is_file() and is_structure_file(p))
        return source_from_structures(structures, "directory"), True
    if path.is_file() and is_structure_file(path):
        return source_from_structures([path], "file"), True
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    return pd.read_csv(path, sep=sep), False


def load_sequence_map(paths: list[Path]) -> dict[str, str]:
    seqs: dict[str, str] = {}
    for path in paths:
        df = load_input_dataframe(path)
        for row in df[["id", "sequence"]].itertuples(index=False):
            seqs.setdefault(str(row.id), str(row.sequence).strip())
    return seqs


def discover_sequence_csvs() -> list[Path]:
    return sorted(Path("data/datasets").glob("*hash_v1/sequences.csv"))


def read_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["sequence_md5", "status"], chunksize=100000):
        done.update(chunk.loc[chunk["status"].astype(str).eq("ok"), "sequence_md5"].astype(str))
    return done


def run_foldseek_binary(foldseek: Path, paths: list[Path], work_dir: Path, threads: int) -> Path:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_tsv = work_dir / "structures.tsv"
    input_tsv.write_text("\n".join(str(path) for path in paths) + "\n")
    db = work_dir / "db"
    ss_fasta = work_dir / "ss.fasta"
    subprocess.run([str(foldseek), "createdb", str(input_tsv), str(db), "--threads", str(threads), "-v", "2"], check=True)
    subprocess.run([str(foldseek), "lndb", f"{db}_h", f"{db}_ss_h", "-v", "1"], check=True)
    subprocess.run([str(foldseek), "convert2fasta", f"{db}_ss", str(ss_fasta), "-v", "1"], check=True)
    return ss_fasta


def foldseek_header_ids(header: str) -> list[str]:
    record_id = header.split()[0] if header.split() else header
    candidates = [record_id, structure_id(Path(record_id)), Path(record_id).stem]
    match = MD5_RE.search(header)
    if match is not None:
        candidates.insert(0, match.group(0))
    return list(dict.fromkeys(candidates))


def parse_foldseek_fasta(path: Path, aa_lengths: dict[str, int]) -> dict[str, tuple[str, str]]:
    by_id: dict[str, list[tuple[str, str]]] = {}
    for record in SeqIO.parse(path, "fasta"):
        header = str(record.description)
        for record_id in foldseek_header_ids(header):
            if record_id in aa_lengths:
                by_id.setdefault(record_id, []).append((str(record.seq).lower(), header))
                break
    return {
        record_id: min(entries, key=lambda item: (abs(len(item[0]) - aa_lengths.get(record_id, 0)), -len(item[0])))
        for record_id, entries in by_id.items()
    }


def append_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    exists = path.is_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sequence_md5", "status", "structure_path", "structure_source", "aa_length", "tdi_length", "foldseek_header", "error"]
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def append_fasta(path: Path, records: dict[str, str]) -> None:
    with path.open("a") as handle:
        for record_id, seq in records.items():
            handle.write(f">{record_id}\n{seq}\n")


def run_foldseek(args: argparse.Namespace) -> int:
    source_path = args.input or DEFAULT_FOLDSEEK_SOURCE
    output = args.output or DEFAULT_FOLDSEEK_FASTA
    manifest = args.manifest or DEFAULT_FOLDSEEK_MANIFEST
    work_dir = args.work_dir or DEFAULT_FOLDSEEK_WORK
    if not args.foldseek_bin.is_file():
        raise FileNotFoundError(args.foldseek_bin)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    src, direct_input = read_structure_source(source_path)
    sequence_paths = args.sequence_csv or ([] if direct_input else discover_sequence_csvs())
    seqs = load_sequence_map(sequence_paths)
    if direct_input and not seqs:
        seqs = {str(record_id): "" for record_id in src["sequence_md5"].astype(str)}

    required = {"sequence_md5", "structure_status", "structure_path", "structure_source"}
    missing = required - set(src.columns)
    if missing:
        raise ValueError(f"{source_path}: missing columns {sorted(missing)}")

    src["sequence_md5"] = src["sequence_md5"].astype(str)
    available = src[src["structure_status"].astype(str).eq("available")].copy()
    available = available[available["sequence_md5"].isin(seqs)]
    available = available[available["structure_path"].astype(str).map(lambda p: Path(p).is_file())]
    available = available.sort_values("sequence_md5").reset_index(drop=True)
    if args.limit is not None:
        available = available.head(args.limit)

    if args.resume:
        done = read_done(manifest)
        available = available[~available["sequence_md5"].isin(done)].reset_index(drop=True)
        mode = "append"
    else:
        done = set()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("")
        if manifest.exists():
            manifest.unlink()
        mode = "overwrite"

    print(f"loaded sequences: {len(seqs):,} from {len(sequence_paths)} files", flush=True)
    print(f"mode={mode}; to extract: {len(available):,}; already done: {len(done):,}", flush=True)
    aa_lengths = {record_id: len(seq) for record_id, seq in seqs.items()}
    batch_size = args.batch_size or 1000
    total_ok = total_missing = 0
    for start in range(0, len(available), batch_size):
        batch = available.iloc[start : start + batch_size].copy()
        print(f"[batch {start // batch_size + 1}] structures {start:,}-{start + len(batch):,}", flush=True)
        try:
            ss_fasta = run_foldseek_binary(args.foldseek_bin, [Path(p) for p in batch["structure_path"].astype(str)], work_dir, args.threads)
            extracted = parse_foldseek_fasta(ss_fasta, aa_lengths)
            batch_error = ""
        except subprocess.CalledProcessError as exc:
            extracted = {}
            batch_error = str(exc)

        fasta_records: dict[str, str] = {}
        rows: list[dict[str, object]] = []
        for row in batch.itertuples(index=False):
            row_dict = row._asdict()
            record_id = str(row_dict["sequence_md5"])
            tdi, header = extracted.get(record_id, ("", ""))
            status = "ok" if tdi else "missing_3di"
            total_ok += int(status == "ok")
            total_missing += int(status != "ok")
            if tdi:
                fasta_records[record_id] = tdi
            rows.append(
                {
                    "sequence_md5": record_id,
                    "status": status,
                    "structure_path": row_dict.get("structure_path", ""),
                    "structure_source": row_dict.get("structure_source", ""),
                    "aa_length": aa_lengths.get(record_id, ""),
                    "tdi_length": len(tdi),
                    "foldseek_header": header,
                    "error": "" if tdi else batch_error,
                }
            )
        append_fasta(output, fasta_records)
        append_manifest(manifest, rows)
        print(f"  ok={total_ok:,} missing={total_missing:,}", flush=True)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    print(f"done -> {output}; manifest -> {manifest}", flush=True)
    return 0


def encode_prostt5_aa(sp, seq: str, max_len: int) -> list[int]:
    seq = RARE_AA_RE.sub("X", seq.upper())[:max_len]
    return [AA2FOLD_TOKEN_ID] + sp.EncodeAsIds(" ".join(seq)) + [EOS_TOKEN_ID]


def ids_to_3di(token_ids) -> str:
    return "".join(ID_TO_3DI[int(token_id)] for token_id in token_ids if int(token_id) in ID_TO_3DI)


def generation_kwargs(beams: int) -> dict[str, object]:
    if beams == 1:
        return {"do_sample": False, "num_beams": 1, "early_stopping": False}
    return {
        "do_sample": True,
        "num_beams": beams,
        "top_p": 0.95,
        "temperature": 1.2,
        "top_k": 6,
        "repetition_penalty": 1.2,
        "early_stopping": True,
    }


def run_prostt5(args: argparse.Namespace) -> int:
    input_path = require_path(args.input, "-i/--input")
    output = require_path(args.output, "-o/--output")

    import sentencepiece as spm
    import torch
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    from transformers import T5ForConditionalGeneration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.model} (T5ForConditionalGeneration) ...", flush=True)
    snapshot_dir = snapshot_download(args.model)
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(snapshot_dir, "spiece.model"))
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    if device.type == "cuda":
        model = model.half()

    df = load_input_dataframe(input_path)
    if args.limit is not None:
        df = df.head(args.limit)
    df["len"] = df["sequence"].str.len()
    df = df.sort_values("len", ascending=False).reset_index(drop=True)
    print(f"input: {len(df):,} sequences; max_len={df['len'].max() if len(df) else 0}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size or 8
    gen_kwargs = generation_kwargs(args.beams)
    started = time.time()
    n_done = 0
    with output.open("w") as fout, torch.no_grad():
        for start in tqdm(range(0, len(df), batch_size), desc="predict 3Di"):
            batch = df.iloc[start : start + batch_size]
            encoded = [encode_prostt5_aa(sp, seq, args.max_len) for seq in batch["sequence"].astype(str)]
            aa_lens = [len(ids) - 2 for ids in encoded]
            max_tokens = max(len(ids) for ids in encoded)
            input_ids = torch.full((len(encoded), max_tokens), PAD_TOKEN_ID, dtype=torch.long, device=device)
            attn_mask = torch.zeros((len(encoded), max_tokens), dtype=torch.long, device=device)
            for i, ids in enumerate(encoded):
                input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                attn_mask[i, : len(ids)] = 1
            output_ids = model.generate(input_ids=input_ids, attention_mask=attn_mask, max_length=max(aa_lens) + 8, min_length=2, **gen_kwargs)
            for record_id, aa_len, row in zip(batch["id"].astype(str), aa_lens, output_ids):
                tdi = ids_to_3di(row.tolist())[:aa_len].ljust(aa_len, "d")
                fout.write(f">{record_id}\n{tdi}\n")
            n_done += len(batch)
    elapsed = time.time() - started
    print(f"done: {n_done:,} sequences in {elapsed / 60:.1f} min -> {output}", flush=True)
    return 0


def normalize_aa(seq: str) -> str:
    return "".join(char if char in STANDARD_AA else "X" for char in seq.upper())


def encode_conv_aa(seq: str) -> list[int]:
    return [AA2FOLD_TOKEN_ID] + [AA_LETTER_TO_ID.get(char, UNK_TOKEN_ID) for char in seq] + [EOS_TOKEN_ID]


def pick_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with request.urlopen(req) as resp, dest.open("wb") as handle:
        shutil.copyfileobj(resp, handle)


def load_conv_head(torch, nn, cache_dir: Path, device):
    class CNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),
                nn.ReLU(),
                nn.Dropout(0.0),
                nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0)),
            )

        def forward(self, x):
            return self.classifier(x.permute(0, 2, 1).unsqueeze(dim=-1)).squeeze(dim=-1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / "prostt5_cnn.pt"
    if not checkpoint.exists():
        print(f"downloading CNN head -> {checkpoint}", flush=True)
        download_file(CNN_WEIGHTS_URL, checkpoint)
    model = CNN()
    model.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
    return model.eval().to(device)


def run_conv(args: argparse.Namespace) -> int:
    input_path = require_path(args.input, "-i/--input")
    output = require_path(args.output, "-o/--output")

    import numpy as np
    import torch
    import torch.nn as nn
    from transformers import T5EncoderModel

    device = pick_device(torch)
    half = (not args.full_precision) and device.type in {"cuda", "mps"}
    max_residues = args.max_residues
    if max_residues <= 0:
        max_residues = int(4000 * torch.cuda.get_device_properties(0).total_memory / (16 * 1024**3)) if device.type == "cuda" else 4000

    print(f"device: {device}; max_residues={max_residues}", flush=True)
    model = T5EncoderModel.from_pretrained(DEFAULT_PROSTT5_MODEL, cache_dir=str(args.model_cache)).to(device).eval()
    predictor = load_conv_head(torch, nn, args.cnn_cache, device)
    if half:
        model = model.half()
        predictor = predictor.half()
    else:
        model = model.to(torch.float32)
        predictor = predictor.to(torch.float32)

    df = load_input_dataframe(input_path)
    if args.limit is not None:
        df = df.head(args.limit)
    items = sorted(
        ((str(row.id), normalize_aa(str(row.sequence))) for row in df[["id", "sequence"]].itertuples(index=False)),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    predictions: dict[str, str] = {}
    failed: list[tuple[str, int]] = []
    batch: list[tuple[str, str]] = []
    started = time.time()

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        ids, seqs = zip(*batch)
        lens = [len(seq) for seq in seqs]
        encoded = [encode_conv_aa(seq) for seq in seqs]
        max_tokens = max(len(tokens) for tokens in encoded)
        input_ids = torch.full((len(encoded), max_tokens), PAD_TOKEN_ID, dtype=torch.long, device=device)
        attn_mask = torch.zeros((len(encoded), max_tokens), dtype=torch.long, device=device)
        for i, tokens in enumerate(encoded):
            input_ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            attn_mask[i, : len(tokens)] = 1
        try:
            with torch.no_grad():
                hidden = model(input_ids, attention_mask=attn_mask).last_hidden_state.detach()
        except RuntimeError as exc:
            print(f"runtime error on batch of {len(batch)}: {exc}", flush=True)
            failed.extend((record_id, length) for record_id, length in zip(ids, lens))
            batch = []
            return
        for i, length in enumerate(lens):
            attn_mask[i, length + 1] = 0
        hidden = (hidden * attn_mask.unsqueeze(-1))[:, 1:]
        pred = torch.max(predictor(hidden), dim=1)[1].detach().cpu().numpy().astype(np.int8)
        for i, record_id in enumerate(ids):
            text = "".join(SS_MAPPING[idx] for idx in pred[i, : lens[i]])
            predictions[record_id] = text.lower() if args.case == "lower" else text
        batch = []

    for idx, item in enumerate(items, 1):
        batch.append(item)
        residues = sum(len(seq) for _, seq in batch)
        if len(batch) >= args.max_batch or residues >= max_residues or idx == len(items) or len(item[1]) > args.max_seq_len:
            flush()
            if idx == len(items) or idx % max(1, len(items) // 50) == 0:
                rate = idx / max(1e-6, time.time() - started)
                print(f"  [{idx}/{len(items)}] {rate:.1f} prot/s, predicted {len(predictions)}, failed {len(failed)}", flush=True)

    written = write_fasta(predictions, output)
    if failed:
        failed_path = output.with_suffix(".failed.tsv")
        failed_path.write_text("".join(f"{record_id}\t{length}\n" for record_id, length in failed))
        print(f"failed list -> {failed_path}", flush=True)
    print(f"wrote {written} sequences to {output}", flush=True)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    method = normalize_method(args.method)
    if args.dry_run:
        input_path = (args.input or DEFAULT_FOLDSEEK_SOURCE) if method == "foldseek" else args.input
        output_path = (args.output or DEFAULT_FOLDSEEK_FASTA) if method == "foldseek" else args.output
        payload = {
            "method": method,
            "input": str(input_path) if input_path is not None else None,
            "output": str(output_path) if output_path is not None else None,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if method == "foldseek":
        return run_foldseek(args)
    if method == "prostt5":
        return run_prostt5(args)
    if method == "conv":
        return run_conv(args)
    raise AssertionError(method)


if __name__ == "__main__":
    raise SystemExit(main())

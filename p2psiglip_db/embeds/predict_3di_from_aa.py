"""Predict Foldseek 3Di sequences from amino acid sequences using ProstT5.

Adapted from https://github.com/mheinzinger/ProstT5/blob/main/scripts/predict_3Di_encoderOnly.py
(Heinzinger et al. NAR Genomics & Bioinformatics 2024). Translates AA sequences to 3Di via:
  ProstT5 T5 encoder -> CNN decoder (downloaded once from upstream repo)

Output: FASTA with 3Di letters (lower-case, foldseek/saprot convention) — same protein IDs as input.

Usage:
  python scripts/predict_3di_from_aa.py \
    --input data/integrated_splits_v3/proteins.fasta \
    --output data/integrated_splits_v3/proteins_3di.fasta \
    --model-cache /home/zlab/.cache/huggingface
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from urllib import request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from transformers import T5EncoderModel


def read_simple_fasta(path):
    """Return {id: sequence} dict from a FASTA *or* an id/sequence CSV."""
    p = Path(path)
    if p.suffix.lower() in {".fasta", ".fa", ".faa", ".fna"}:
        return {str(r.id): str(r.seq).upper().strip()
                for r in SeqIO.parse(p, "fasta")}
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    if "id" not in df.columns or "sequence" not in df.columns:
        raise ValueError(f"{path}: expected id+sequence columns or FASTA, got {list(df.columns)}")
    return dict(zip(df["id"].astype(str),
                    df["sequence"].astype(str).str.upper().str.strip()))


def write_fasta_util(seqs, path, line_width=10**9):
    """Write {id: sequence} dict as FASTA."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for sid, s in seqs.items():
            if line_width >= len(s):
                f.write(f">{sid}\n{s}\n")
            else:
                f.write(f">{sid}\n")
                for i in range(0, len(s), line_width):
                    f.write(s[i:i+line_width] + "\n")
            n += 1
    return n


CNN_WEIGHTS_URL = "https://github.com/mheinzinger/ProstT5/raw/main/cnn_chkpnt/model.pt"
SS_MAPPING = "ACDEFGHIKLMNPQRSTVWY"  # 20 classes, mapped to AA-equivalent letters
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
AA2FOLD_TOKEN_ID = 149
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 2
_AA_LETTER_TO_ID = {
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


class CNN(nn.Module):
    """Two-conv 3Di classifier head on top of ProstT5 embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Conv2d(32, 20, kernel_size=(7, 1), padding=(3, 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F=1024) -> (B, F, L, 1) -> (B, 20, L, 1) -> (B, 20, L)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        y = self.classifier(x)
        return y.squeeze(dim=-1)


def device_pick() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}")
    req = request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with request.urlopen(req) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def load_predictor(weights_dir: Path, device: torch.device) -> nn.Module:
    weights_dir.mkdir(parents=True, exist_ok=True)
    ckpt = weights_dir / "prostt5_cnn.pt"
    if not ckpt.exists():
        download(CNN_WEIGHTS_URL, ckpt)
    model = CNN()
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    return model.eval().to(device)


def normalize(seq: str) -> str:
    """Replace non-standard AAs with X (per ProstT5 protocol)."""
    return "".join(c if c in STANDARD_AA else "X" for c in seq.upper())


def encode_aa2fold(seq: str) -> list[int]:
    """Tokenize AA sequence for ProstT5 AA->3Di without sentencepiece."""
    return [AA2FOLD_TOKEN_ID] + [_AA_LETTER_TO_ID.get(c, UNK_TOKEN_ID) for c in seq] + [EOS_TOKEN_ID]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="Input AA FASTA")
    ap.add_argument("--output", required=True, type=Path, help="Output 3Di FASTA")
    ap.add_argument("--model-cache", type=Path, default=Path.home() / ".cache" / "huggingface",
                    help="HuggingFace cache dir (defaults to ~/.cache/huggingface)")
    ap.add_argument("--cnn-cache", type=Path, default=Path.home() / ".cache" / "prostt5_cnn",
                    help="Where to cache the CNN decoder weights")
    ap.add_argument("--max-residues", type=int, default=0,
                    help="Max residues per batch. 0 = auto (4000 per 16GB of GPU memory)")
    ap.add_argument("--max-batch", type=int, default=500, help="Max sequences per batch")
    ap.add_argument("--max-seq-len", type=int, default=1000,
                    help="Sequences longer than this are processed alone")
    ap.add_argument("--full-precision", action="store_true",
                    help="Disable fp16 (default fp16 on GPU/MPS)")
    ap.add_argument("--case", choices=("lower", "upper"), default="lower",
                    help="3Di output case (foldseek/saprot expect lower; default: lower)")
    args = ap.parse_args()

    device = device_pick()
    print(f"device: {device}")
    half = (not args.full_precision) and device.type in {"cuda", "mps"}

    # Auto-pick max_residues from GPU memory if not user-specified.
    max_residues = args.max_residues
    if max_residues <= 0:
        if device.type == "cuda":
            total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            max_residues = int(4000 * total_gib / 16)
            print(f"auto max_residues={max_residues} (scaled from 4000 per 16GB on {total_gib:.0f}GB GPU)")
        else:
            max_residues = 4000
    lowercase = args.case == "lower"

    print(f"loading ProstT5 (Rostlab/ProstT5) into cache {args.model_cache}")
    model = T5EncoderModel.from_pretrained(
        "Rostlab/ProstT5", cache_dir=str(args.model_cache)
    ).to(device).eval()
    predictor = load_predictor(args.cnn_cache, device)
    if half:
        model = model.half()
        predictor = predictor.half()
        print("running fp16")
    else:
        model = model.to(torch.float32)
        predictor = predictor.to(torch.float32)
        print("running fp32")

    seqs = read_simple_fasta(args.input)
    print(f"loaded {len(seqs):,} sequences from {args.input}")
    items = sorted(seqs.items(), key=lambda kv: len(kv[1]), reverse=True)
    avg_len = sum(len(s) for _, s in items) / max(1, len(items))
    n_long = sum(1 for _, s in items if len(s) > args.max_seq_len)
    print(f"avg length {avg_len:.1f}, long ones (>{args.max_seq_len}): {n_long}")

    predictions: dict[str, str] = {}
    failed: list[tuple[str, int]] = []
    batch: list[tuple[str, str, int]] = []
    start = time.time()

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        ids, seqs_batch, lens = zip(*batch)
        encoded = [encode_aa2fold(seq) for seq in seqs_batch]
        max_tokens = max(len(tokens) for tokens in encoded)
        input_ids = torch.full((len(encoded), max_tokens), PAD_TOKEN_ID, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(encoded), max_tokens), dtype=torch.long, device=device)
        for i, tokens in enumerate(encoded):
            input_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            attention_mask[i, :len(tokens)] = 1
        try:
            with torch.no_grad():
                out = model(input_ids, attention_mask=attention_mask)
        except RuntimeError as e:
            print(f"OOM/runtime error on batch of {len(batch)} (longest={lens[0]}): {e}")
            for sid, _, slen in batch:
                failed.append((sid, slen))
            batch = []
            return

        # mask out the trailing special token, account for the prefix at index 0
        for i, slen in enumerate(lens):
            attention_mask[i, slen + 1] = 0
        h = out.last_hidden_state.detach() * attention_mask.unsqueeze(-1)
        h = h[:, 1:]  # strip leading <AA2fold> token

        logits = predictor(h)
        # Argmax over the 20 3Di-class dim (dim=1). Result shape: (B, L).
        pred = torch.max(logits, dim=1)[1].detach().cpu().numpy().astype(np.int8)
        for i, sid in enumerate(ids):
            slen = lens[i]
            seq3di = pred[i, :slen]
            text = "".join(SS_MAPPING[c] for c in seq3di)
            if lowercase:
                text = text.lower()
            assert len(text) == slen, f"length mismatch {sid}: got {len(text)}, expected {slen}"
            predictions[sid] = text
        batch = []

    for idx, (sid, seq) in enumerate(items, 1):
        normalized = normalize(seq)
        slen = len(normalized)
        batch.append((sid, normalized, slen))

        n_res = sum(s for _, _, s in batch) + slen
        if (
            len(batch) >= args.max_batch
            or n_res >= max_residues
            or idx == len(items)
            or slen > args.max_seq_len
        ):
            flush()
            if idx % max(1, len(items) // 50) == 0 or idx == len(items):
                rate = idx / max(1e-6, time.time() - start)
                print(f"  [{idx}/{len(items)}] {rate:.1f} prot/s, predicted {len(predictions)}, failed {len(failed)}")

    elapsed = time.time() - start
    print(f"\n=== STATS ===")
    print(f"predicted: {len(predictions)} / {len(items)}  ({len(failed)} failed)")
    print(f"total time: {elapsed:.1f}s ({elapsed/max(1,len(predictions)):.4f}s/prot)")

    n_written = write_fasta_util(predictions, args.output, line_width=10**9)
    print(f"wrote {n_written} sequences to {args.output}")
    if failed:
        fp = args.output.with_suffix(".failed.tsv")
        with fp.open("w") as f:
            for sid, slen in failed:
                f.write(f"{sid}\t{slen}\n")
        print(f"failed list -> {fp}")


if __name__ == "__main__":
    main()

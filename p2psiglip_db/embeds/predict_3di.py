"""
ProstT5 AA→3Di structural-token predictor.

For each input AA sequence, runs the ProstT5 encoder-decoder generate to
predict a 3Di structural-token sequence of the same length, then writes a
FASTA where each record's id matches the input id and the body is the
predicted 3Di letters (lowercase).

Output FASTA format (matches `embed/_universe_3di.fasta` for downstream
SaProt / ProstT5(3Di) encoding):
    >FP0xxxxxxx
    dvpvvvvvvvvvvvpvvvlvvvlvlvaafkfwfk...

Generation kwargs follow Heinzinger et al.'s canonical recipe (mheinzinger/ProstT5
scripts/translate.py). For batches, sequences are length-bucketed to minimise
padding; KV-cache is enabled by default (transformers >= 4.30).
"""
import argparse
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "Rostlab/ProstT5"

RARE_AA_RE = re.compile(r"[UZOB]")

# Token IDs (from added_tokens.json + sentencepiece vocab)
AA2FOLD_TOKEN_ID = 149
FOLD2AA_TOKEN_ID = 148
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 0

# 3Di letter -> token id (added_tokens.json). These are the *valid* output tokens
# for AA→3Di generation.
THREE_DI_LETTERS = "acdefghiklmnpqrstvwy"
THREE_DI_TOKEN_IDS = {
    "a": 128, "l": 129, "g": 130, "v": 131, "s": 132, "r": 133, "e": 134,
    "d": 135, "t": 136, "i": 137, "p": 138, "k": 139, "f": 140, "q": 141,
    "n": 142, "y": 143, "m": 144, "h": 145, "w": 146, "c": 147,
}
ID_TO_3DI = {v: k for k, v in THREE_DI_TOKEN_IDS.items()}


def parse_arguments():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=str, required=True,
                   help="CSV (id,sequence) of AA sequences to translate")
    p.add_argument("-o", "--output", type=str, required=True,
                   help="output FASTA path (will be overwritten)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1024,
                   help="cap on AA length; longer seqs get truncated")
    p.add_argument("--beams", type=int, default=1,
                   help="num_beams for generation. 1=greedy (~2-3x faster, default); "
                        "3=Heinzinger canonical recipe (slower, marginally better quality)")
    return p.parse_args()


class AASeqDataset(Dataset):
    def __init__(self, ids, seqs):
        self.ids = ids
        self.seqs = seqs
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, idx):
        return self.ids[idx], self.seqs[idx]


def _encode_aa(sp, seq, max_len):
    """Tokenize an AA sequence as ProstT5 input: [<AA2fold>, sp(AA), </s>]."""
    s = RARE_AA_RE.sub("X", seq.upper())[: max_len]
    spaced = " ".join(list(s))
    body = sp.EncodeAsIds(spaced)
    return [AA2FOLD_TOKEN_ID] + body + [EOS_TOKEN_ID]


def _ids_to_3di(token_ids):
    """Convert generated decoder token IDs to a string of lowercase 3Di letters.

    Drops anything that isn't a valid 3Di token (start tokens, eos, pad,
    occasional special-id leakage from the decoder).
    """
    out = []
    for t in token_ids:
        t = int(t)
        if t in ID_TO_3DI:
            out.append(ID_TO_3DI[t])
    return "".join(out)


def main():
    args = parse_arguments()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer (sentencepiece) + model (full T5 with decoder)
    print(f"loading {args.model} (T5ForConditionalGeneration) ...", flush=True)
    snapshot_dir = snapshot_download(args.model)
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(snapshot_dir, "spiece.model"))
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()
    if device.type == "cuda":
        model = model.half()

    # Load AA sequences
    df = pd.read_csv(args.input)
    df.columns = [c.strip() for c in df.columns]
    df["id"] = df["id"].astype(str)
    df["sequence"] = df["sequence"].astype(str).str.strip()
    # Sort by length descending — long sequences first surface OOM fast.
    df["len"] = df["sequence"].str.len()
    df = df.sort_values("len", ascending=False).reset_index(drop=True)
    print(f"input: {len(df):,} sequences (lens p10/p50/p90/max="
          f"{df['len'].quantile([.1,.5,.9]).values.tolist()}/{df['len'].max()})", flush=True)

    dataset = AASeqDataset(df["id"].tolist(), df["sequence"].tolist())
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: (list(zip(*b))[0], list(zip(*b))[1]),
    )

    # Generation kwargs. For feature-extraction downstream we don't need optimal
    # 3Di sampling, just deterministic+consistent → greedy default. For matching
    # the Heinzinger paper recipe set --beams 3.
    if args.beams == 1:
        GEN_KW = {
            "do_sample": False,
            "num_beams": 1,
            "early_stopping": False,  # not used in greedy
        }
    else:
        GEN_KW = {
            "do_sample": True,
            "num_beams": args.beams,
            "top_p": 0.95,
            "temperature": 1.2,
            "top_k": 6,
            "repetition_penalty": 1.2,
            "early_stopping": True,
        }
    print(f"generation kwargs: {GEN_KW}", flush=True)

    n_done = 0
    t0 = time.time()
    with out_path.open("w") as fout, torch.no_grad():
        for batch_ids, batch_seqs in tqdm(dataloader, desc="predict 3Di"):
            id_lists = [_encode_aa(sp, s, args.max_len) for s in batch_seqs]
            seq_lens_in = [len(x) - 2 for x in id_lists]  # AA-only length (no <AA2fold>, no </s>)

            L = max(len(x) for x in id_lists)
            input_ids = torch.full((len(id_lists), L), PAD_TOKEN_ID, dtype=torch.long)
            attn_mask = torch.zeros((len(id_lists), L), dtype=torch.long)
            for i, ids_i in enumerate(id_lists):
                input_ids[i, :len(ids_i)] = torch.tensor(ids_i, dtype=torch.long)
                attn_mask[i, :len(ids_i)] = 1
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            # Generate one 3Di token per AA position.
            # Add slack (+8) on max_length to allow control + EOS + minor over-generation.
            target_len = max(seq_lens_in) + 8
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_length=target_len,
                min_length=2,
                **GEN_KW,
            )

            # output is (B, gen_L). Decode each row to a 3Di string, truncating to
            # the original AA length so the FASTA is 1:1 with AA.
            for prot_id, aa_len, row in zip(batch_ids, seq_lens_in, output):
                tdi = _ids_to_3di(row.tolist())
                tdi = tdi[:aa_len]  # keep 1:1 mapping with AA
                # Pad with 'd' if generation came up short (rare, but defensive)
                if len(tdi) < aa_len:
                    tdi = tdi + "d" * (aa_len - len(tdi))
                fout.write(f">{prot_id}\n{tdi}\n")
            n_done += len(batch_ids)

    elapsed = time.time() - t0
    print(f"done: {n_done:,} sequences in {elapsed/60:.1f} min "
          f"({n_done/elapsed:.2f} seq/s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

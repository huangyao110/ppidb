"""
SaProt-650M-AF2 (westlake-repl/SaProt_650M_AF2) feature extractor.

SaProt operates on a "structural alphabet" where each residue's token is the
concatenation of the AA letter (uppercase) and the 3Di letter (lowercase).
This script:
  1. Reads AA sequences from a CSV (id,sequence)
  2. Reads matching 3Di sequences from a FASTA (lowercase letters, 1:1 length)
  3. Builds the per-residue pair tokens and runs the SaProt encoder
  4. Saves either mean-pooled (1280,) fp32 embeddings or per-residue
     (L,1280) fp16 embeddings with CLS/EOS stripped
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from p2psiglip_db.embeds.io import ProteinDataset, atomic_save_npy, pair_collate, safe_id

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "westlake-repl/SaProt_650M_AF2"


def parse_arguments():
    p = argparse.ArgumentParser(description="SaProt-650M feature extractor")
    p.add_argument("-i", "--input", type=str, required=True,
                   help="AA CSV: id,sequence")
    p.add_argument("--tdi-fasta", type=str, required=True,
                   help="3Di FASTA (lowercase, 1:1 length with each AA seq)")
    p.add_argument("-o", "--output_dir", type=str, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--per-residue", action="store_true",
                   help="save per-residue (L, 1280) fp16 instead of mean-pooled (1280,) fp32")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute and replace existing .npy files")
    return p.parse_args()


def _build_pair_seq(aa: str, tdi: str, max_len: int) -> str:
    """Build SaProt input string: per-residue '<AA_upper><tdi_lower>'."""
    aa = aa.upper()
    tdi = tdi.lower()
    L = min(len(aa), len(tdi), max_len)
    return "".join(aa[i] + tdi[i] for i in range(L))


def main():
    args = parse_arguments()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(device)
    model.eval()
    if device.type == "cuda":
        model = model.half()

    # Load AA sequences
    aa_df = pd.read_csv(args.input)
    aa_df.columns = [c.strip() for c in aa_df.columns]
    aa_df["id"] = aa_df["id"].astype(str)
    aa_df["sequence"] = aa_df["sequence"].astype(str).str.strip()
    aa_map = dict(zip(aa_df["id"], aa_df["sequence"]))

    # Load 3Di sequences
    tdi_map = {}
    for r in SeqIO.parse(args.tdi_fasta, "fasta"):
        tdi_map[str(r.id)] = str(r.seq).lower().strip()

    # Build paired tokens for ids that appear in both, skipping completed files
    # so long extraction jobs can be resumed.
    ids, paired = [], []
    skipped = 0
    done = 0
    for pid, aa in aa_map.items():
        if (out_dir / f"{safe_id(pid)}.npy").exists() and not args.overwrite:
            done += 1
            continue
        if pid not in tdi_map:
            skipped += 1
            continue
        tdi = tdi_map[pid]
        if len(aa) != len(tdi):
            print(f"⚠ length mismatch for {pid}: aa={len(aa)} tdi={len(tdi)} — truncating", flush=True)
        ids.append(pid)
        paired.append(_build_pair_seq(aa, tdi, args.max_len))
    if skipped:
        print(f"⚠ {skipped:,} ids in {args.input} have no 3Di in {args.tdi_fasta} (skipped)", flush=True)
    if done:
        print(f"skip existing embeddings: {done:,}", flush=True)

    # Length sort desc
    order = sorted(range(len(paired)), key=lambda i: -len(paired[i]))
    ids = [ids[i] for i in order]
    paired = [paired[i] for i in order]
    print(f"to-encode: {len(ids):,} sequences", flush=True)

    dataset = ProteinDataset(ids, paired)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=pair_collate)

    t0 = time.time()
    with torch.no_grad():
        for batch_ids, batch_seqs in tqdm(dataloader, desc="encode SaProt"):
            enc = tokenizer(batch_seqs, padding=True, truncation=True,
                            max_length=args.max_len + 2,  # +2 for CLS/EOS
                            return_tensors="pt").to(device)
            out = model(**enc).last_hidden_state  # (B, L, 1280)

            seq_lens = enc.attention_mask.sum(dim=1)
            if args.per_residue:
                out_cpu = out.float().cpu()
                for i, prot_id in enumerate(batch_ids):
                    Li = int(seq_lens[i])
                    res = out_cpu[i, 1:Li - 1].numpy().astype(np.float16)
                    atomic_save_npy(out_dir / f"{safe_id(prot_id)}.npy", res)
            else:
                # Drop CLS (pos 0) and EOS (last attended pos per row)
                mask = enc.attention_mask.unsqueeze(-1).float()
                mask[:, 0, :] = 0.0
                for i, Li in enumerate(seq_lens.tolist()):
                    mask[i, Li - 1, 0] = 0.0
                pooled = (out.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                embs = pooled.cpu().numpy()

                for prot_id, vec in zip(batch_ids, embs):
                    atomic_save_npy(out_dir / f"{safe_id(prot_id)}.npy", vec.astype(np.float32))

    print(f"SaProt 编码完成: {len(ids):,} in {(time.time()-t0)/60:.1f} min, dim=1280", flush=True)


if __name__ == "__main__":
    main()

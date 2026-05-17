"""
ESM2-650M (facebook/esm2_t33_650M_UR50D) feature extractor.

Input CSV (id,sequence) or FASTA. By default writes one mean-pooled
(1280,)-shape .npy per protein; pass --per-residue to write (L,1280) fp16
arrays with BOS/EOS stripped for ColBERT-style late interaction.
"""
import time
from pathlib import Path

import torch
import numpy as np
import argparse
import warnings
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from multiprocessing import Process, set_start_method
from tqdm import tqdm

from p2psiglip_db.embeds.io import (
    ProteinDataset,
    atomic_save_npy,
    filter_existing_outputs,
    load_input_dataframe,
    pair_collate,
    safe_id,
    sort_by_sequence_length,
    split_dataframe_by_workers,
)

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"


def parse_arguments():
    p = argparse.ArgumentParser(description="ESM2-650M 特征提取")
    p.add_argument("-i", "--input", type=str, required=True,
                   help="输入：CSV (id,sequence) 或 FASTA (.fasta/.fa/.faa/.fna)")
    p.add_argument("-o", "--output_dir", type=str, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--batch_size", type=int, default=4,
                   help="batch_size (默认 4，ESM2-650M 在 1024-tok 序列下显存敏感)")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--per-residue", action="store_true",
                   help="保存 per-residue 嵌入 (L,1280) fp16 而非 mean-pooled (1280,) fp32")
    return p.parse_args()


def run_worker(gpu_id, subset_df, args):
    device = torch.device(f"cuda:{gpu_id}")
    tag = f"GPU-{gpu_id}"
    print(f"[{tag}] loading {args.model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()

    dataset = ProteinDataset(subset_df['id'].tolist(), subset_df['sequence'].tolist())
    # batch via tokenizer padding inside the loop (variable lengths)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=pair_collate)

    print(f"[{tag}] extracting 1280-D features ...", flush=True)
    with torch.no_grad():
        for batch_ids, batch_seqs in tqdm(dataloader, desc=tag, position=gpu_id):
            seqs = [s[: args.max_len] for s in batch_seqs]
            enc = tokenizer(seqs, padding=True, truncation=True,
                            max_length=args.max_len + 2,  # +2 for CLS/EOS
                            return_tensors="pt").to(device)
            out = model(**enc).last_hidden_state           # (B, L, 1280)
            mask = enc.attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            # ESM2 special tokens: <cls> at pos 0, <eos> at last real position.
            # Zero them out before mean-pool by removing pos 0 and the last attended pos per row.
            mask[:, 0, :] = 0.0
            seq_lens = enc.attention_mask.sum(dim=1)        # (B,)
            for i, L in enumerate(seq_lens.tolist()):
                mask[i, L - 1, 0] = 0.0
            if args.per_residue:
                embs = []
                for i, L in enumerate(seq_lens.tolist()):
                    embs.append(out[i, 1:L - 1, :].detach().cpu().numpy())
            else:
                pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                embs = pooled.cpu().numpy()                    # (B, 1280)

            for prot_id, vec in zip(batch_ids, embs):
                atomic_save_npy(
                    Path(args.output_dir) / f"{safe_id(prot_id)}.npy",
                    vec.astype(np.float16 if args.per_residue else np.float32),
                )


if __name__ == "__main__":
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_arguments()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = filter_existing_outputs(load_input_dataframe(args.input), out_dir)
    print(f"ESM2 remaining sequences: {len(df):,}", flush=True)
    if len(df) == 0:
        print("ESM2 特征提取完成，维度: 1280")
        raise SystemExit(0)
    df = sort_by_sequence_length(df, ascending=False)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise SystemExit("ESM2 extraction requires at least one CUDA GPU")
    chunks = split_dataframe_by_workers(df, num_gpus)
    procs = []
    for i in range(num_gpus):
        p = Process(target=run_worker, args=(i, chunks[i], args))
        p.start()
        procs.append(p)
        time.sleep(5)
    for p in procs:
        p.join()
    failed = [p.exitcode for p in procs if p.exitcode != 0]
    if failed:
        raise SystemExit(f"ESM2 worker failures: exitcodes={failed}")
    print("ESM2 特征提取完成，维度: 1280")

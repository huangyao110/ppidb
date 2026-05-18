"""
ESM2-650M (facebook/esm2_t33_650M_UR50D) feature extractor.

Input CSV (id,sequence) or FASTA. By default writes one mean-pooled
(1280,)-shape .npy per protein; pass --per-residue to write (L,1280) fp16
arrays with BOS/EOS stripped for ColBERT-style late interaction.
"""
import time
from pathlib import Path

import torch
import argparse
import warnings
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from multiprocessing import Process, set_start_method
from tqdm import tqdm

from p2psiglip_db.embeds.io import (
    POOL_CHOICES,
    ProteinDataset,
    atomic_save_npy,
    filter_existing_outputs,
    load_input_dataframe,
    normalize_pool_mode,
    pair_collate,
    pooled_array,
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
    p.add_argument("--pool", choices=POOL_CHOICES, default=None,
                   help="输出池化模式: mean, max, cls, residue")
    p.add_argument("--per-residue", action="store_true",
                   help="兼容旧参数；等同于 --pool residue")
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
            # ESM2 token layout: <cls>, residues, <eos>, pads.
            seq_lens = enc.attention_mask.sum(dim=1)        # (B,)
            out_cpu = out.detach().float().cpu().numpy()

            for i, prot_id in enumerate(batch_ids):
                L = int(seq_lens[i])
                vec = pooled_array(
                    out_cpu[i, 1:L - 1, :],
                    args.pool,
                    cls_embedding=out_cpu[i, 0, :],
                )
                atomic_save_npy(
                    Path(args.output_dir) / f"{safe_id(prot_id)}.npy",
                    vec,
                )


if __name__ == "__main__":
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_arguments()
    args.pool = normalize_pool_mode(args.pool, default="mean", per_residue=args.per_residue)
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

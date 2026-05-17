"""
ProtT5-XL (Rostlab/prot_t5_xl_uniref50) feature extractor.

Same I/O contract as get_embeddings_esmc.py: input CSV (id,sequence) or FASTA,
output one (1024,)-shape .npy per protein, mean-pooled with the trailing </s>
position dropped. Pass --per-residue to write (L,1024) fp16 arrays with the
trailing </s> position dropped for ColBERT-style late interaction.

Differs from get_embeddings_prostt5.py only in: (a) the model name, and
(b) NO `<AA2fold>` control-token prefix (ProtT5 has no such token).
"""
import os
import re
import time
from pathlib import Path

import torch
import numpy as np
import argparse
import warnings
import sentencepiece as spm
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from transformers import T5EncoderModel
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

DEFAULT_MODEL = "Rostlab/prot_t5_xl_uniref50"

RARE_AA_RE = re.compile(r"[UZOB]")

EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 0


def parse_arguments():
    p = argparse.ArgumentParser(description="ProtT5-XL 特征提取 (AA encoder)")
    p.add_argument("-i", "--input", type=str, required=True,
                   help="输入：CSV (id,sequence) 或 FASTA (.fasta/.fa/.faa/.fna)")
    p.add_argument("-o", "--output_dir", type=str, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--per-residue", action="store_true",
                   help="保存 per-residue 嵌入 (L,1024) fp16 而非 mean-pooled (1024,) fp32")
    return p.parse_args()


def _encode_one(sp, seq, max_len):
    s = RARE_AA_RE.sub("X", seq.upper())[: max_len]
    spaced = " ".join(list(s))
    body = sp.EncodeAsIds(spaced)
    return body + [EOS_TOKEN_ID]


def run_worker(gpu_id, subset_df, args):
    device = torch.device(f"cuda:{gpu_id}")
    tag = f"GPU-{gpu_id}"
    print(f"[{tag}] loading {args.model} ...", flush=True)
    snapshot_dir = snapshot_download(args.model)
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(snapshot_dir, "spiece.model"))
    model = T5EncoderModel.from_pretrained(args.model).to(device)
    model.eval()
    if device.type == "cuda":
        model = model.half()

    dataset = ProteinDataset(subset_df['id'].tolist(), subset_df['sequence'].tolist())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=pair_collate)

    print(f"[{tag}] extracting 1024-D features ...", flush=True)
    with torch.no_grad():
        for batch_ids, batch_seqs in tqdm(dataloader, desc=tag, position=gpu_id):
            id_lists = [_encode_one(sp, s, args.max_len) for s in batch_seqs]
            L = max(len(x) for x in id_lists)
            input_ids = torch.full((len(id_lists), L), PAD_TOKEN_ID, dtype=torch.long)
            attn_mask = torch.zeros((len(id_lists), L), dtype=torch.long)
            for i, ids_i in enumerate(id_lists):
                input_ids[i, :len(ids_i)] = torch.tensor(ids_i, dtype=torch.long)
                attn_mask[i, :len(ids_i)] = 1
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state

            mask = attn_mask.unsqueeze(-1).float()
            seq_lens = attn_mask.sum(dim=1)
            for i, Li in enumerate(seq_lens.tolist()):
                mask[i, Li - 1, 0] = 0.0  # drop </s>
            if args.per_residue:
                embs = []
                for i, Li in enumerate(seq_lens.tolist()):
                    embs.append(out[i, :Li - 1, :].detach().float().cpu().numpy())
            else:
                pooled = (out.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                embs = pooled.cpu().numpy()

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
    print(f"ProtT5 remaining sequences: {len(df):,}", flush=True)
    df = sort_by_sequence_length(df, ascending=False)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise SystemExit("ProtT5 extraction requires at least one CUDA GPU")
    chunks = split_dataframe_by_workers(df, num_gpus)
    procs = []
    for i in range(num_gpus):
        p = Process(target=run_worker, args=(i, chunks[i], args))
        p.start()
        procs.append(p)
        time.sleep(5)
    for p in procs:
        p.join()
    print("ProtT5 特征提取完成，维度: 1024")

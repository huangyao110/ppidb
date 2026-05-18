"""
ProstT5 (Rostlab/ProstT5) feature extractor.

ProstT5 is a T5 encoder–decoder model trained jointly on AA sequences and
3Di structural tokens. For pure-AA embedding extraction we only need the
encoder; the per-residue 1024-D last_hidden_state is mean-pooled.

Output: one (1024,)-shape .npy per protein, mean-pooled with the leading
CLS-like position and the trailing EOS removed.
"""
import os
import re
import time
from pathlib import Path

import torch
import argparse
import warnings
from torch.utils.data import DataLoader
from multiprocessing import Process, set_start_method
from tqdm import tqdm

from p2psiglip_db.embeds.io import (
    POOL_CHOICES,
    ProteinDataset,
    atomic_save_npy,
    load_input_dataframe,
    normalize_pool_mode,
    pair_collate,
    pooled_array,
    safe_id,
    sort_by_sequence_length,
    split_dataframe_by_workers,
)

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "Rostlab/ProstT5"

# ProstT5 / ProtT5 expect rare AAs (U, Z, O, B) replaced with X.
RARE_AA_RE = re.compile(r"[UZOB]")

# Token IDs (from added_tokens.json + sentencepiece model). transformers 5.x's
# auto-tokenizer-conversion misreads the spiece.model as tiktoken, so we
# bypass it and tokenize directly via sentencepiece + a few hardcoded specials.
AA2FOLD_TOKEN_ID = 149
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 0


def parse_arguments():
    p = argparse.ArgumentParser(description="ProstT5 特征提取 (AA encoder)")
    p.add_argument("-i", "--input", type=str, required=True,
                   help="输入：CSV (id,sequence) 或 FASTA (.fasta/.fa/.faa/.fna)")
    p.add_argument("-o", "--output_dir", type=str, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--pool", choices=POOL_CHOICES, default=None,
                   help="输出池化模式: mean, max, cls, residue")
    p.add_argument("--per-residue", action="store_true",
                   help="兼容参数；等同于 --pool residue")
    return p.parse_args()


def _encode_one(sp, seq: str, max_len: int) -> list[int]:
    """Tokenize a single AA sequence to ProstT5 input ids: [<AA2fold>, AA_1, ..., AA_n, </s>]."""
    s = RARE_AA_RE.sub("X", seq.upper())[: max_len]
    spaced = " ".join(list(s))
    body = sp.EncodeAsIds(spaced)
    return [AA2FOLD_TOKEN_ID] + body + [EOS_TOKEN_ID]


def run_worker(gpu_id, subset_df, args):
    device = torch.device(f"cuda:{gpu_id}")
    tag = f"GPU-{gpu_id}"
    print(f"[{tag}] loading {args.model} ...", flush=True)
    import sentencepiece as spm
    from huggingface_hub import snapshot_download
    from transformers import T5EncoderModel

    snapshot_dir = snapshot_download(args.model)
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(snapshot_dir, "spiece.model"))
    model = T5EncoderModel.from_pretrained(args.model).to(device)
    model.eval()
    if device.type == "cuda":
        model = model.half()

    ids = subset_df['id'].tolist()
    seqs = subset_df['sequence'].tolist()
    dataset = ProteinDataset(ids, seqs)
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

            out = model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state  # (B, L, 1024)

            # Token layout per row: [<AA2fold>, AA_1, ..., AA_n, </s>, <pad>...]
            seq_lens = attn_mask.sum(dim=1)
            out_cpu = out.detach().float().cpu().numpy()

            for i, prot_id in enumerate(batch_ids):
                Li = int(seq_lens[i])
                vec = pooled_array(
                    out_cpu[i, 1:Li - 1, :],
                    args.pool,
                    cls_embedding=out_cpu[i, 0, :],
                    cls_name="<AA2fold>",
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

    df = sort_by_sequence_length(load_input_dataframe(args.input), ascending=False)

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise SystemExit("ProstT5 extraction requires at least one CUDA GPU")
    chunks = split_dataframe_by_workers(df, num_gpus)
    procs = []
    for i in range(num_gpus):
        p = Process(target=run_worker, args=(i, chunks[i], args))
        p.start()
        procs.append(p)
        time.sleep(5)
    for p in procs:
        p.join()
    print("ProstT5 特征提取完成，维度: 1024")

"""
ProstT5 3Di-mode encoder feature extractor.

Reads a 3Di FASTA (lowercase letters from {a,c,d,e,f,g,h,i,k,l,m,n,p,q,r,s,t,v,w,y})
and encodes each sequence via the ProstT5 *encoder* with the `<fold2AA>` control
token, mean-pooling residue states (excluding the prefix and </s>) into a (1024,)
embedding saved as `<safe_id>.npy`.

This matches the user's pre-computed `embed/prostt5_3di/` so that hybrid
(precomputed + patch) yields a homogeneous PLM column.
"""
import argparse
import os
import time
import warnings
from pathlib import Path

import torch
from Bio import SeqIO
from torch.utils.data import DataLoader
from tqdm import tqdm

from p2psiglip_db.embeds.io import (
    POOL_CHOICES,
    ProteinDataset,
    atomic_save_npy,
    normalize_pool_mode,
    pair_collate,
    pooled_array,
    safe_id,
)

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "Rostlab/ProstT5"

FOLD2AA_TOKEN_ID = 148
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 0


def parse_arguments():
    p = argparse.ArgumentParser(description="ProstT5 3Di-mode encoder extractor")
    p.add_argument("-i", "--input", type=str, required=True,
                   help="3Di FASTA file (lowercase letters)")
    p.add_argument("-o", "--output_dir", type=str, required=True)
    p.add_argument("--pool", choices=POOL_CHOICES, default=None,
                   help="output pooling mode: mean, max, cls, residue")
    p.add_argument("--per-residue", action="store_true",
                   help="legacy alias for --pool residue")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1024)
    return p.parse_args()


_TDI_LETTER_TO_ID = {
    'a': 128, 'l': 129, 'g': 130, 'v': 131, 's': 132, 'r': 133, 'e': 134,
    'd': 135, 't': 136, 'i': 137, 'p': 138, 'k': 139, 'f': 140, 'q': 141,
    'n': 142, 'y': 143, 'm': 144, 'h': 145, 'w': 146, 'c': 147,
}
_UNK_TOKEN_ID = 2


def _encode_3di(sp, tdi_seq, max_len):
    """Tokenize a 3Di sequence as ProstT5 input: [<fold2AA>, 3Di_ids, </s>].
    3Di letters (a,c,d,e,f,g,h,i,k,l,m,n,p,q,r,s,t,v,w,y) are mapped via the
    explicit added-token IDs (128..147) — sentencepiece's base vocab is
    uppercase-AA-only and would emit <unk> for every 3Di letter."""
    s = tdi_seq.lower()[: max_len]
    body = [_TDI_LETTER_TO_ID.get(c, _UNK_TOKEN_ID) for c in s]
    return [FOLD2AA_TOKEN_ID] + body + [EOS_TOKEN_ID]


def main():
    args = parse_arguments()
    args.pool = normalize_pool_mode(args.pool, default="mean", per_residue=args.per_residue)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.model} encoder ...", flush=True)
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

    # Load 3Di FASTA
    ids, seqs = [], []
    for r in SeqIO.parse(args.input, "fasta"):
        ids.append(str(r.id))
        seqs.append(str(r.seq).lower().strip())
    # length-sort descending to surface OOM fast
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    ids = [ids[i] for i in order]
    seqs = [seqs[i] for i in order]
    print(f"input: {len(ids):,} 3Di seqs (max len {max(map(len, seqs))})", flush=True)

    dataset = ProteinDataset(ids, seqs)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=pair_collate)

    t0 = time.time()
    with torch.no_grad():
        for batch_ids, batch_seqs in tqdm(dataloader, desc="encode 3Di"):
            id_lists = [_encode_3di(sp, s, args.max_len) for s in batch_seqs]
            L = max(len(x) for x in id_lists)
            input_ids = torch.full((len(id_lists), L), PAD_TOKEN_ID, dtype=torch.long)
            attn_mask = torch.zeros((len(id_lists), L), dtype=torch.long)
            for i, ids_i in enumerate(id_lists):
                input_ids[i, :len(ids_i)] = torch.tensor(ids_i, dtype=torch.long)
                attn_mask[i, :len(ids_i)] = 1
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state

            # Strip <fold2AA> prefix (pos 0) and </s> trailing token per row.
            seq_lens = attn_mask.sum(dim=1)
            out_cpu = out.detach().float().cpu().numpy()
            for i, prot_id in enumerate(batch_ids):
                Li = int(seq_lens[i])
                vec = pooled_array(
                    out_cpu[i, 1:Li - 1, :],
                    args.pool,
                    cls_embedding=out_cpu[i, 0, :],
                    cls_name="<fold2AA>",
                )
                atomic_save_npy(out_dir / f"{safe_id(prot_id)}.npy", vec)

    print(f"ProstT5(3Di) 编码完成: {len(ids):,} in {(time.time()-t0)/60:.1f} min, dim=1024", flush=True)


if __name__ == "__main__":
    main()

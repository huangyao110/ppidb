"""
ProFam-1 feature extractor.

Input CSV (id,sequence) or FASTA. By default writes per-residue arrays
(L,D) fp16 for ColBERT-style late interaction; pass --mean-pool to write
one (D,) fp32 vector per protein.

ProFam is an autoregressive protein family LM. For a single sequence without
family context, we mirror ProFam's no-context scoring input:
    [start-of-document] [RAW] <sequence> [SEP]
and save the final hidden states at the residue-token positions.
"""
import argparse
import time
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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
)

warnings.filterwarnings("ignore")


def parse_arguments():
    p = argparse.ArgumentParser(description="ProFam-1 feature extractor")
    p.add_argument("-i", "--input", type=str, required=True,
                   help="Input CSV (id,sequence) or FASTA (.fasta/.fa/.faa/.fna)")
    p.add_argument("-o", "--output_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional ProFam checkpoint .ckpt path")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                   default="bfloat16")
    p.add_argument("--attn-implementation", type=str, default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--device", type=str, default=None,
                   help="Device override, e.g. cuda, cuda:0, or cpu")
    p.add_argument("--no-auto-download", action="store_true",
                   help="Do not auto-download the default ProFam-1 checkpoint")
    p.add_argument("--pool", choices=POOL_CHOICES, default=None,
                   help="Output pooling mode: mean, max, cls, residue. Default is residue for ProFam")
    p.add_argument("--mean-pool", action="store_true",
                   help="Legacy alias for --pool mean")
    return p.parse_args()


def clean_sequence(seq, max_len):
    seq = str(seq).upper().replace("-", "").replace(".", "")
    seq = seq.replace("U", "C").replace("O", "K")
    return seq[:max_len]


def load_profam(args):
    try:
        from profam import ProFam
    except ImportError as exc:
        raise SystemExit(
            "ProFam is not installed in this environment. Install with: "
            "pip install profam"
        ) from exc

    kwargs = {
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "auto_download": not args.no_auto_download,
    }
    if args.checkpoint:
        kwargs["checkpoint"] = args.checkpoint
    api = ProFam(**kwargs)
    model = api._model
    model.eval()
    return model


def build_input_ids(tokenizer, seqs, device):
    # Token ids follow ProFam's no-context scoring convention.
    start_id = tokenizer.convert_tokens_to_ids("[start-of-document]")
    raw_id = tokenizer.convert_tokens_to_ids("[RAW]")
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    encoded = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in seqs]
    lengths = [len(x) for x in encoded]
    max_tokens = max(lengths) + 3
    input_ids = torch.full((len(seqs), max_tokens), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), max_tokens), dtype=torch.long)
    for i, ids in enumerate(encoded):
        row = [start_id, raw_id, *ids, sep_id]
        L = len(row)
        input_ids[i, :L] = torch.tensor(row, dtype=torch.long)
        attention_mask[i, :L] = 1
    return input_ids.to(device), attention_mask.to(device), lengths


def main():
    args = parse_arguments()
    args.pool = normalize_pool_mode(args.pool, default="residue", mean_pool=args.mean_pool)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = filter_existing_outputs(load_input_dataframe(args.input), out_dir)
    print(f"ProFam remaining sequences: {len(df):,}", flush=True)
    if len(df) == 0:
        print("ProFam feature extraction complete")
        return

    df["sequence"] = df["sequence"].map(lambda s: clean_sequence(s, args.max_len))
    df = df[df["sequence"].str.len() > 0].copy()
    df = sort_by_sequence_length(df, ascending=False)

    print("loading ProFam-1 ...", flush=True)
    model = load_profam(args)
    device = model.device
    tokenizer = model.tokenizer
    hidden_dim = int(model.model.config.hidden_size)
    print(f"ProFam loaded on {device}, hidden_dim={hidden_dim}", flush=True)

    dataset = ProteinDataset(df["id"].tolist(), df["sequence"].tolist())
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pair_collate,
    )

    t0 = time.time()
    with torch.no_grad():
        for batch_ids, batch_seqs in tqdm(dataloader, desc="encode ProFam"):
            input_ids, attention_mask, lengths = build_input_ids(
                tokenizer, batch_seqs, device
            )
            outputs = model.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state.float().cpu().numpy()
            for i, prot_id in enumerate(batch_ids):
                L = lengths[i]
                res = hidden[i, 2:2 + L]
                out_path = out_dir / f"{safe_id(prot_id)}.npy"
                atomic_save_npy(
                    out_path,
                    pooled_array(
                        res,
                        args.pool,
                        cls_embedding=hidden[i, 0],
                        cls_name="[start-of-document]",
                    ),
                )

    mode = args.pool
    print(
        f"ProFam extraction complete: {len(df):,} in "
        f"{(time.time() - t0) / 60:.1f} min, dim={hidden_dim}, mode={mode}",
        flush=True,
    )


if __name__ == "__main__":
    main()

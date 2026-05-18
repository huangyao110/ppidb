import time
from pathlib import Path

import torch
import argparse
import warnings
from torch.utils.data import DataLoader
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from multiprocessing import Process, set_start_method
from tqdm import tqdm

from p2psiglip_db.embeds.io import (
    POOL_CHOICES,
    ProteinDataset,
    atomic_save_npy,
    load_input_dataframe,
    normalize_pool_mode,
    pooled_array,
    safe_id,
    sort_by_sequence_length,
    split_dataframe_by_workers,
)

# 忽略警告
warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser(description="ESMC-300M 特征提取 (双GPU并行)")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="输入：CSV (含 id,sequence 两列) 或 FASTA (.fasta/.fa/.faa/.fna)")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="输出目录")
    # ESMC 默认模型名
    parser.add_argument("--model", type=str, default="esmc_300m", help="ESMC 模型名")
    parser.add_argument("--batch_size", type=int, default=128, help="ESMC 较大，建议 Batch Size 设小一点")
    parser.add_argument("--max_len", type=int, default=1024, help="最大截断长度")
    parser.add_argument("--pool", choices=POOL_CHOICES, default=None,
                        help="输出池化模式: mean, max, cls, residue")
    parser.add_argument("--per-residue", action="store_true",
                        help="兼容旧参数；等同于 --pool residue")
    return parser.parse_args()

# ==============================================================================
# GPU 工作进程 (针对 ESMC 优化)
# ==============================================================================
def run_worker(gpu_id, subset_df, args):
    device = torch.device(f"cuda:{gpu_id}")
    process_name = f"GPU-{gpu_id}"
    
    print(f"[{process_name}] 正在加载 ESMC-300M 模型...")
    # 使用官方 SDK 加载模型
    model = ESMC.from_pretrained(args.model).to(device)
    model.eval()

    dataset = ProteinDataset(subset_df['id'].tolist(), subset_df['sequence'].tolist())
    # 注意：ESMC SDK 处理 Batch 较为特殊，这里我们手动循环或使用简单的 DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[{process_name}] 开始提取 960 维特征...")
    
    with torch.no_grad():
        for batch_ids, batch_seqs in tqdm(dataloader, desc=process_name, position=gpu_id):
            for i, seq in enumerate(batch_seqs):
                prot_id = batch_ids[i]
                
                # 1. 构造 ESMC 输入对象
                protein = ESMProtein(sequence=seq[:args.max_len])
                
                # 2. 编码与推理
                protein_tensor = model.encode(protein).to(device)
                # 获取 logits 和 embeddings (sequence=True 表示保留残基级特征)
                output = model.logits(
                    protein_tensor, 
                    LogitsConfig(sequence=True, return_embeddings=True)
                )
                
                # 3. 提取 Embedding (形状通常为 [1, L+2, 960])
                # ESMC 包含开头结尾特殊 token，我们取中间部分
                # .cpu() 转换到 CPU 处理以释放显存
                residue_embeds = output.embeddings.cpu().numpy()[0]

                # 去掉 BOS 和 EOS (第一行和最后一行)
                core_residues = residue_embeds[1:-1, :]
                cls_embedding = residue_embeds[0, :]

                # 4. 保存：residue (L, 960) fp16 或 pooled (960,) fp32
                out_path = Path(args.output_dir) / f"{safe_id(prot_id)}.npy"
                atomic_save_npy(
                    out_path,
                    pooled_array(core_residues, args.pool, cls_embedding=cls_embedding),
                )

if __name__ == "__main__":
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    args = parse_arguments()
    args.pool = normalize_pool_mode(args.pool, default="mean", per_residue=args.per_residue)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_input_dataframe(args.input)
    # 按长度排序可以防止 batch 内 padding 过多
    df = sort_by_sequence_length(df, ascending=False)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise SystemExit("ESMC extraction requires at least one CUDA GPU")
    chunks = split_dataframe_by_workers(df, num_gpus)
    processes = []
    
    for i in range(num_gpus):
        p = Process(target=run_worker, args=(i, chunks[i], args))
        p.start()
        processes.append(p)
        time.sleep(5) # 预留模型加载时间

    for p in processes:
        p.join()
    print("ESMC 特征提取完成，维度: 960")

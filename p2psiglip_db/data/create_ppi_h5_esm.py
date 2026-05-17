import numpy as np
from tqdm import tqdm
from pathlib import Path
import h5py
import pandas as pd
import argparse

def create_ppi_relational_hdf5(ppi_df_path, embeddings_dir, output_file):
    print("--- 开始创建蛋白质-蛋白质关系型 HDF5 文件 (Fixed Size Mode) ---")
    
    # 1. 读取交互对CSV文件
    try:
        ppi_df = pd.read_csv(ppi_df_path)
        # 兼容截图中的列名大小写问题，统一处理
        ppi_df.columns = [c.strip() for c in ppi_df.columns]
        if 'ID_1' not in ppi_df.columns or 'ID_2' not in ppi_df.columns:
            # 尝试查找常见变体
            if 'fpid_1' in ppi_df.columns: ppi_df.rename(columns={'fpid_1': 'ID_1', 'fpid_2': 'ID_2'}, inplace=True)
            else: raise ValueError("CSV文件必须包含 'ID_1' 和 'ID_2' 列。")
    except Exception as e:
        print(f"读取 CSV 错误: {e}")
        return

    # 2. 提取并排序 ID
    all_ids = pd.concat([ppi_df['ID_1'], ppi_df['ID_2']]).dropna().astype(str)
    unique_protein_ids = sorted(all_ids.unique())
    print(f"统计: {len(unique_protein_ids)} 个唯一蛋白, {len(ppi_df)} 对交互。")

    protein_id_to_idx = {pid: i for i, pid in enumerate(unique_protein_ids)}
    embeddings_path_root = Path(embeddings_dir)
    embedding_files = {p.stem: p for p in embeddings_path_root.glob('*.npy')}

    # 3. 获取维度信息
    try:
        # 随机读取一个文件获取维度，预期是 (480,)
        sample_id = unique_protein_ids[0]
        if sample_id in embedding_files:
            sample_emb = np.load(embedding_files[sample_id])
            if sample_emb.ndim != 1:
                # 防御性编程：如果你之前的步骤没跑好，这里可能会报错
                raise ValueError(f"Embedding 维度错误: 预期是一维向量 (480,), 实际是 {sample_emb.shape}")
            embedding_dim = sample_emb.shape[0]
        else:
            print("警告: 无法找到第一个ID的嵌入文件，无法确定维度。")
            return
    except Exception as e:
        print(f"检查维度失败: {e}")
        return

    with h5py.File(output_file, 'w') as f:
        # --- A. 存储 Embedding (使用定长 Dataset，极大提升性能) ---
        print(f"正在写入嵌入矩阵 (Shape: {len(unique_protein_ids)} x {embedding_dim})...")

        # 创建定长数据集
        protein_dset = f.create_dataset('unique_protein_embeddings',
                                      shape=(len(unique_protein_ids), embedding_dim),
                                      dtype='float32')
        # valid_mask[i] = True iff fpid i has a real embedding on disk (False = zero-filled)
        valid_dset = f.create_dataset('valid_mask',
                                      shape=(len(unique_protein_ids),),
                                      dtype='bool')
        f.attrs['embedding_dim'] = embedding_dim

        missing_count = 0
        for i, pid in enumerate(tqdm(unique_protein_ids, desc="写入数据")):
            if pid in embedding_files:
                data = np.load(embedding_files[pid]).astype(np.float32)
                protein_dset[i] = data
                valid_dset[i] = True
            else:
                # 缺失填充0，并标记为 invalid（多-PLM 场景下 dataloader 会跳过这些）
                protein_dset[i] = np.zeros(embedding_dim, dtype='float32')
                valid_dset[i] = False
                missing_count += 1

        if missing_count > 0:
            print(f"警告: 有 {missing_count} 个蛋白缺少嵌入文件（已填充0, valid_mask=False）。")

        # --- B. 存储索引 ---
        print("正在写入交互索引...")
        pairs_indices = []
        for _, row in tqdm(ppi_df.iterrows(), total=len(ppi_df), desc="映射索引"):
            pid1, pid2 = str(row['ID_1']), str(row['ID_2'])
            if pid1 in protein_id_to_idx and pid2 in protein_id_to_idx:
                pairs_indices.append([protein_id_to_idx[pid1], protein_id_to_idx[pid2]])
        
        f.create_dataset('pairs_index', data=np.array(pairs_indices, dtype=np.int32))
        print(f"完成！保存至 {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ppi_df_path', type=str, required=True)
    parser.add_argument('--embeddings_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='./data/ppi_relational.h5')
    args = parser.parse_args()
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    create_ppi_relational_hdf5(args.ppi_df_path, args.embeddings_dir, args.output_file)
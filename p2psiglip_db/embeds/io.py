"""Shared I/O helpers for PLM embedding extractors."""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from Bio import SeqIO
from torch.utils.data import Dataset


FASTA_SUFFIXES = {".fasta", ".fa", ".faa", ".fna"}
POOL_CHOICES = ("mean", "max", "cls", "residue", "per-residue")


class ProteinDataset(Dataset):
    def __init__(self, ids: Iterable[str], seqs: Iterable[str]):
        self.ids = list(ids)
        self.seqs = list(seqs)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.ids[idx], self.seqs[idx]


def pair_collate(batch):
    ids, seqs = zip(*batch)
    return list(ids), list(seqs)


def safe_id(protein_id: object) -> str:
    return str(protein_id).replace("/", "_").replace("\\", "_")


def embedding_path(output_dir: str | Path, protein_id: object) -> Path:
    return Path(output_dir) / f"{safe_id(protein_id)}.npy"


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, array)
    tmp.replace(path)


def normalize_pool_mode(
    pool: str | None,
    *,
    default: str,
    per_residue: bool = False,
    mean_pool: bool = False,
) -> str:
    """Normalize new ``--pool`` and legacy pooling flags into one mode."""
    mode = pool or default
    if mode == "per-residue":
        mode = "residue"
    if per_residue:
        mode = "residue"
    if mean_pool:
        mode = "mean"
    if mode not in {"mean", "max", "cls", "residue"}:
        raise ValueError(f"unsupported pool mode: {pool}")
    return mode


def pooled_array(
    residue_embeddings: np.ndarray,
    pool: str,
    *,
    cls_embedding: np.ndarray | None = None,
    cls_name: str = "CLS/BOS",
) -> np.ndarray:
    """Return residue-level or protein-level embedding with stable dtypes."""
    if residue_embeddings.ndim != 2:
        raise ValueError(f"expected residue embedding array with shape (L,D), got {residue_embeddings.shape}")
    if residue_embeddings.shape[0] == 0:
        raise ValueError("cannot pool an empty residue embedding")
    if pool == "residue":
        return residue_embeddings.astype(np.float16)
    if pool == "mean":
        return residue_embeddings.mean(axis=0).astype(np.float32)
    if pool == "max":
        return residue_embeddings.max(axis=0).astype(np.float32)
    if pool == "cls":
        if cls_embedding is None:
            raise ValueError(f"pool=cls requires a {cls_name} embedding for this extractor")
        return np.asarray(cls_embedding).astype(np.float32)
    raise ValueError(f"unsupported pool mode: {pool}")


def load_csv_to_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    if "id" not in df.columns or "sequence" not in df.columns:
        raise ValueError(f"{path}: expected id and sequence columns")
    df = df.copy()
    df["id"] = df["id"].astype(str)
    df["sequence"] = df["sequence"].astype(str).str.strip()
    return df


def load_fasta_to_dataframe(path: str | Path) -> pd.DataFrame:
    ids, seqs, seen = [], [], set()
    for record in SeqIO.parse(str(path), "fasta"):
        rid = str(record.id)
        if rid in seen:
            warnings.warn(f"duplicate id in FASTA, keeping first: {rid}")
            continue
        seen.add(rid)
        ids.append(rid)
        seqs.append(str(record.seq).upper().strip())
    return pd.DataFrame({"id": ids, "sequence": seqs})


def load_input_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in FASTA_SUFFIXES:
        return load_fasta_to_dataframe(path)
    return load_csv_to_dataframe(path)


def filter_existing_outputs(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    if overwrite:
        return df.copy()
    output_dir = Path(output_dir)
    return df[~df["id"].map(lambda x: embedding_path(output_dir, x).exists())].copy()


def sort_by_sequence_length(df: pd.DataFrame, *, ascending: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["len"] = out["sequence"].str.len()
    return out.sort_values("len", ascending=ascending).reset_index(drop=True)


def split_dataframe_by_workers(df: pd.DataFrame, workers: int) -> list[pd.DataFrame]:
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    return [chunk.copy() for chunk in np.array_split(df, workers)]

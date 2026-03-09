"""
SequenceFetcher — batch retrieval of protein sequences from UniProt.

Features:
  - Batch API calls for efficiency
  - Local disk cache to avoid redundant downloads
  - Automatic retry with exponential backoff via urllib3
  - Connection pooling for stability
  - Isoform handling (strips isoform suffix, e.g. P12345-2 → P12345)
  - Output as FASTA file or dict {uniprot_id: sequence}
  - Optional subcellular compartment fetching for NegativeSampler
"""

from __future__ import annotations

import concurrent.futures
import json
import warnings
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Union

import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# UniProt REST API endpoint
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/accessions"
BATCH_SIZE = 100     # 降低单批次数量，避免 URL 过长或负载过大
MAX_WORKERS = 10     # 降低并发数，50并发极易被 UniProt 封禁/限流

class SequenceFetcher:
    """
    Fetch protein sequences and metadata from UniProt.

    Parameters
    ----------
    cache_dir : str | Path | None
        Directory to cache downloaded sequences.
        Default: ~/.ppidb/sequence_cache/
        Set to None to disable caching.
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".ppidb" / "sequence_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "sequences.json"
        self._cache_lock = Lock()
        self._cache: Dict[str, str] = self._load_cache()
        
        # 初始化带连接池和自动重试机制的 Session
        self._session = self._create_robust_session()

    def _create_robust_session(self) -> requests.Session:
        """创建一个稳健的 HTTP Session, 包含自动重试和连接池功能"""
        session = requests.Session()
        # 遇到 429(限流) 或 5xx(服务器错误) 时自动指数退避重试
        retry = Retry(
            total=5,
            backoff_factor=1.0, 
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(
            max_retries=retry, 
            pool_connections=MAX_WORKERS, 
            pool_maxsize=MAX_WORKERS
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    # ── Cache management ──────────────────────────────────────────────────────

    def _load_cache(self) -> Dict[str, str]:
        if self._cache_file.exists():
            with open(self._cache_file) as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        with self._cache_lock:
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f)

    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache = {}
            if self._cache_file.exists():
                self._cache_file.unlink()
        print("Cache cleared.")

    def load_local_parquet(
        self,
        parquet_path: Union[str, Path],
        id_col: str = "uniprot_id",
        seq_col: str = "sequence",
        verbose: bool = True,
    ) -> int:
        """
        Load sequences from a local parquet file into the cache.
        
        Parameters
        ----------
        parquet_path : str | Path
            Path to the parquet file.
        id_col : str
            Column name for protein IDs (default: "uniprot_id").
        seq_col : str
            Column name for sequences (default: "sequence").
        verbose : bool
            Whether to print loading status.
            
        Returns
        -------
        int
            Number of sequences loaded.
        """
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            if verbose:
                print(f"Local sequence file not found: {parquet_path}")
            return 0
            
        try:
            df = pl.read_parquet(parquet_path)
            
            # Check columns
            if id_col not in df.columns:
                # Try to guess ID column if not found (e.g. "id", "accession", "uniprot")
                candidates = ["id", "accession", "uniprot", "protein_id"]
                for c in candidates:
                    if c in df.columns:
                        id_col = c
                        break
            
            if seq_col not in df.columns:
                # Try to guess sequence column (e.g. "seq", "sequence_str")
                candidates = ["seq", "sequence_str", "aa_seq"]
                for c in candidates:
                    if c in df.columns:
                        seq_col = c
                        break
            
            if id_col not in df.columns or seq_col not in df.columns:
                if verbose:
                    print(f"Error: Columns '{id_col}' or '{seq_col}' not found in {parquet_path}")
                    print(f"Available columns: {df.columns}")
                return 0
                
            # Load into dict
            local_seqs = dict(zip(df[id_col].to_list(), df[seq_col].to_list()))
            count = len(local_seqs)
            
            with self._cache_lock:
                self._cache.update(local_seqs)
                
            if verbose:
                print(f"Loaded {count:,} sequences from {parquet_path}")
                
            return count
            
        except Exception as e:
            if verbose:
                print(f"Failed to load local parquet {parquet_path}: {e}")
            return 0

    # ── Core fetch ────────────────────────────────────────────────────────────

    def fetch(
        self,
        proteins: List[str],
        as_dict: bool = False,
        output_fasta: Optional[Union[str, Path]] = None,
        include_isoforms: bool = False,
        verbose: bool = True,
    ) -> Union[Dict[str, str], str]:
        """
        Fetch sequences for a list of UniProt accessions.
        (接口和参数完全保持不变)
        """
        # Normalize IDs
        if not include_isoforms:
            proteins = [p.split("-")[0] for p in proteins]
        proteins = sorted(set(proteins))

        # Split into cached and uncached
        with self._cache_lock:
            cached = {p: self._cache[p] for p in proteins if p in self._cache}
        uncached = [p for p in proteins if p not in self._cache]

        if verbose:
            print(f"Fetching sequences: {len(proteins):,} total, "
                  f"{len(cached):,} cached, {len(uncached):,} to download")

        # Fetch uncached in batches (Parallel)
        if uncached:
            fetched = {}
            failed = []
            
            batches = [uncached[i: i + BATCH_SIZE] for i in range(0, len(uncached), BATCH_SIZE)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_batch = {executor.submit(self._fetch_batch, batch): batch for batch in batches}
                
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_res, batch_failed = future.result()
                    fetched.update(batch_res)
                    failed.extend(batch_failed)
                    
                    completed_count += 1
                    if verbose:
                        print(f"  Batch {completed_count}/{len(batches)}: "
                              f"OK ({len(batch_res)} fetched, {len(batch_failed)} failed)")

            # 单个重试 fallback (更加稳健的方式)
            if failed:
                if verbose:
                    print(f"Attempting to recover {len(failed)} failed IDs individually...")
                recovered, still_failed = self._fetch_individual_fallback(failed, verbose=verbose)
                fetched.update(recovered)
                failed = still_failed
                if verbose:
                    print(f"Recovered: {len(recovered)}, Still failed: {len(failed)}")

            # Update cache
            with self._cache_lock:
                self._cache.update(fetched)
                self._save_cache()

            if failed:
                warnings.warn(
                    f"{len(failed)} proteins could not be fetched: "
                    f"{failed[:5]}{'...' if len(failed) > 5 else ''}"
                )
                with open("failed_fetch.txt", "a") as f:
                    for pid in failed:
                        f.write(f"{pid}\n")
            
            cached.update(fetched)

        all_seqs = cached

        if verbose:
            print(f"Total sequences retrieved: {len(all_seqs):,} / {len(proteins):,}")

        if output_fasta:
            fasta_str = self._to_fasta(all_seqs)
            with open(output_fasta, "w") as f:
                f.write(fasta_str)
            print(f"FASTA saved to {output_fasta}")

        if as_dict:
            return all_seqs
        return self._to_fasta(all_seqs)

    def _fetch_batch(
        self,
        accessions: List[str],
    ) -> tuple[Dict[str, str], List[str]]:
        """使用底层自带重试机制的 Session 请求批次"""
        try:
            response = self._session.get(
                UNIPROT_API,
                params={
                    "accessions": ",".join(accessions),
                    "fields": "accession,sequence",
                    "format": "json",
                    "size": len(accessions),
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            fetched = {}
            found_ids = set()
            for entry in data.get("results", []):
                acc = entry["primaryAccession"]
                seq = entry.get("sequence", {}).get("value", "")
                if seq:
                    fetched[acc] = seq
                    found_ids.add(acc)

            failed = [a for a in accessions if a not in found_ids]
            return fetched, failed

        except requests.exceptions.RequestException:
            # 如果重试多次依然失败，直接交给单条 fallback 处理
            return {}, accessions

    def _fetch_individual_fallback(self, accessions: List[str], verbose: bool = True) -> tuple[Dict[str, str], List[str]]:
        """
        放弃不稳定的 curl 和 biopython, 直接使用带重试的 Session 请求 FASTA 文本流。
        这是最纯粹、最不容易出错的方式。
        """
        fetched = {}
        failed = []

        def _fetch_one(acc):
            url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
            try:
                # 即使是单个请求，也享受自动重试和防限流机制
                resp = self._session.get(url, timeout=15)
                resp.raise_for_status()
                lines = resp.text.splitlines()
                if not lines:
                    return acc, None
                seq = "".join(line.strip() for line in lines if not line.startswith(">"))
                return acc, seq
            except requests.exceptions.RequestException:
                return acc, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_acc = {executor.submit(_fetch_one, acc): acc for acc in accessions}
            
            iterator = concurrent.futures.as_completed(future_to_acc)
            if verbose:
                iterator = tqdm(iterator, total=len(accessions), desc="Fallback fetch", unit="seq")
                
            for future in iterator:
                acc, seq = future.result()
                if seq:
                    fetched[acc] = seq
                else:
                    failed.append(acc)
        
        return fetched, failed

    # ── Subcellular compartments ──────────────────────────────────────────────

    def fetch_compartments(
        self,
        proteins: List[str],
        verbose: bool = True,
    ) -> Dict[str, str]:
        """(接口保持不变)"""
        proteins = sorted(set(p.split("-")[0] for p in proteins))
        compartment_cache_file = self.cache_dir / "compartments.json"

        comp_cache = {}
        if compartment_cache_file.exists():
            with open(compartment_cache_file) as f:
                comp_cache = json.load(f)

        uncached = [p for p in proteins if p not in comp_cache]

        if verbose:
            print(f"Fetching compartments: {len(proteins):,} total, "
                  f"{len(comp_cache):,} cached, {len(uncached):,} to download")

        if uncached:
            fetched_batch = {}
            failed_batch = []
            
            batches = [uncached[i: i + BATCH_SIZE] for i in range(0, len(uncached), BATCH_SIZE)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_batch = {executor.submit(self._fetch_compartments_batch, batch): batch for batch in batches}
                
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_res, failed = future.result()
                    fetched_batch.update(batch_res)
                    failed_batch.extend(failed)
                    
                    completed_count += 1
                    if verbose:
                        print(f"  Batch {completed_count}/{len(batches)}: "
                              f"OK ({len(batch_res)} fetched, {len(failed)} failed)")
            
            comp_cache.update(fetched_batch)
            with open(compartment_cache_file, "w") as f:
                json.dump(comp_cache, f)
            
            if failed_batch:
                warnings.warn(f"{len(failed_batch)} compartments could not be fetched.")
                with open("failed_fetch_compartments.txt", "a") as f:
                    for pid in failed_batch:
                        f.write(f"{pid}\n")

        return {p: comp_cache.get(p, "Unknown") for p in proteins}

    def _fetch_compartments_batch(self, accessions: List[str]) -> tuple[Dict[str, str], List[str]]:
        try:
            response = self._session.get(
                UNIPROT_API,
                params={
                    "accessions": ",".join(accessions),
                    "fields": "accession,cc_subcellular_location",
                    "format": "json",
                    "size": len(accessions),
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            result = {}
            found_ids = set()
            for entry in data.get("results", []):
                acc = entry["primaryAccession"]
                found_ids.add(acc)
                locs = entry.get("comments", [])
                compartment = "Unknown"
                for loc in locs:
                    if loc.get("commentType") == "SUBCELLULAR LOCATION":
                        subcell = loc.get("subcellularLocations", [])
                        if subcell:
                            loc_val = subcell[0].get("location", {}).get("value", "Unknown")
                            compartment = _simplify_compartment(loc_val)
                            break
                result[acc] = compartment
            
            failed = [a for a in accessions if a not in found_ids]
            return result, failed

        except requests.exceptions.RequestException:
            return {}, accessions

    # ── FASTA utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _to_fasta(sequences: Dict[str, str]) -> str:
        lines = []
        for pid, seq in sorted(sequences.items()):
            lines.append(f">{pid}")
            for i in range(0, len(seq), 60):
                lines.append(seq[i:i+60])
        return "\n".join(lines) + "\n"

    @staticmethod
    def parse_fasta(fasta_path: Union[str, Path]) -> Dict[str, str]:
        sequences = {}
        current_id = None
        current_seq = []
        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
        if current_id:
            sequences[current_id] = "".join(current_seq)
        return sequences

# ── Compartment simplification ────────────────────────────────────────────────

_COMPARTMENT_MAP = {
    "Nucleus": ["nucleus", "nucleoplasm", "nucleolus", "chromatin", "nuclear"],
    "Cytoplasm": ["cytoplasm", "cytosol", "cytoskeleton"],
    "Mitochondrion": ["mitochondri"],
    "Endoplasmic reticulum": ["endoplasmic reticulum", " er "],
    "Golgi apparatus": ["golgi"],
    "Cell membrane": ["cell membrane", "plasma membrane", "membrane"],
    "Extracellular": ["extracellular", "secreted"],
    "Lysosome": ["lysosom"],
    "Peroxisome": ["peroxisom"],
    "Endosome": ["endosom"],
}

def _simplify_compartment(loc: str) -> str:
    loc_lower = loc.lower()
    for compartment, keywords in _COMPARTMENT_MAP.items():
        if any(kw in loc_lower for kw in keywords):
            return compartment
    return loc
"""
ProSST-2048 feature extractor.

This script reads hashed protein sequences and matching structure files,
quantizes each structure with the ProSST structure tokenizer, runs the
ProSST Transformer, and saves per-residue arrays as fp16 .npy files.

The output format follows the other P2PSigLip embedding extractors:
    <output_dir>/<sequence_md5>.npy  # shape (min(L, max_len), 768)
"""
import argparse
import csv
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from p2psiglip_db.embeds.io import atomic_save_npy, safe_id

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "AI4Protein/ProSST-2048"
DEFAULT_SEQUENCE_CSV = "data/embeds/manifests/strucs/structure_available_sequences.csv"
DEFAULT_STRUCTURE_MANIFEST = "data/embeds/manifests/strucs/structure_3di_full_manifest.csv"


@dataclass
class ProteinRecord:
    protein_id: str
    sequence: str
    structure_path: str
    structure_source: str
    length: int


def parse_arguments():
    parser = argparse.ArgumentParser(description="ProSST per-residue feature extractor")
    parser.add_argument("--input", default=DEFAULT_SEQUENCE_CSV,
                        help="CSV with id,sequence columns")
    parser.add_argument("--structure-manifest", default=DEFAULT_STRUCTURE_MANIFEST,
                        help="CSV with sequence_md5,status,structure_path columns")
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--manifest-output", default=None,
                        help="Append extraction status rows to this CSV")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--structure-vocab-size", type=int, default=2048)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Transformer batch size after structure tokenization")
    parser.add_argument("--chunk-size", type=int, default=64,
                        help="Number of structures sent to the structure tokenizer per chunk")
    parser.add_argument("--chunk-token-budget", type=int, default=8192,
                        help="Approximate total residue budget per structure-tokenizer chunk")
    parser.add_argument("--max-batch-nodes", type=int, default=10000,
                        help="ProSST structure-tokenizer node budget")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--sst-timeout", type=float, default=90.0,
                        help="Base seconds to wait for one structure-tokenizer chunk before splitting/restarting")
    parser.add_argument("--sst-timeout-per-1k-residues", type=float, default=15.0,
                        help="Additional timeout seconds per 1,000 residues in the tokenizer chunk")
    parser.add_argument("--sst-worker-max-chunks", type=int, default=200,
                        help="Restart the structure-tokenizer worker after this many successful chunks; 0 disables")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-dtype", choices=["float32", "float16", "bfloat16"],
                        default="float16")
    parser.add_argument("--sort-by-length", choices=["asc", "desc", "none"],
                        default="asc")
    parser.add_argument("--include-sources", default=None,
                        help="Comma-separated structure_source values to include, e.g. afdb_local,afdb_download")
    parser.add_argument("--exclude-sources", default=None,
                        help="Comma-separated structure_source values to exclude, e.g. minifold")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional smoke-test limit after filtering existing files")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_prosst_importable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    prosst_root = repo_root / "external" / "ProSST"
    if not prosst_root.exists():
        raise SystemExit(f"ProSST repo not found: {prosst_root}")
    sys.path.insert(0, str(prosst_root))


def load_records(args, output_dir: Path) -> list[ProteinRecord]:
    seq_df = pd.read_csv(args.input)
    seq_df.columns = [c.strip() for c in seq_df.columns]
    seq_df["id"] = seq_df["id"].astype(str)
    seq_df["sequence"] = seq_df["sequence"].astype(str).str.strip()

    struct_df = pd.read_csv(args.structure_manifest)
    struct_df.columns = [c.strip() for c in struct_df.columns]
    struct_df["sequence_md5"] = struct_df["sequence_md5"].astype(str)
    struct_df = struct_df[struct_df["status"].astype(str).str.lower().eq("ok")].copy()

    merged = seq_df.merge(
        struct_df[["sequence_md5", "structure_path", "structure_source", "aa_length", "tdi_length"]],
        left_on="id",
        right_on="sequence_md5",
        how="inner",
    )
    merged["length"] = merged["sequence"].str.len()

    include_sources = parse_source_list(args.include_sources)
    exclude_sources = parse_source_list(args.exclude_sources)
    if include_sources:
        merged = merged[merged["structure_source"].astype(str).isin(include_sources)].copy()
    if exclude_sources:
        merged = merged[~merged["structure_source"].astype(str).isin(exclude_sources)].copy()

    if not args.overwrite:
        merged = merged[
            ~merged["id"].map(lambda x: (output_dir / f"{safe_id(x)}.npy").exists())
        ].copy()

    if args.sort_by_length == "asc":
        merged = merged.sort_values("length", ascending=True)
    elif args.sort_by_length == "desc":
        merged = merged.sort_values("length", ascending=False)

    if args.limit is not None:
        merged = merged.head(args.limit)

    records = []
    for row in merged.itertuples(index=False):
        records.append(
            ProteinRecord(
                protein_id=str(row.id),
                sequence=str(row.sequence).upper(),
                structure_path=str(row.structure_path),
                structure_source=str(row.structure_source),
                length=int(row.length),
            )
        )
    return records


def parse_source_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def chunks(items: list[ProteinRecord], size: int, token_budget: int) -> list[list[ProteinRecord]]:
    result = []
    current = []
    current_tokens = 0
    for item in items:
        item_tokens = max(1, item.length)
        would_exceed_count = len(current) >= size
        would_exceed_tokens = (
            token_budget > 0
            and current
            and current_tokens + item_tokens > token_budget
        )
        if would_exceed_count or would_exceed_tokens:
            result.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += item_tokens
    if current:
        result.append(current)
    return result


def append_manifest(manifest_path: Path | None, rows: list[dict]) -> None:
    if manifest_path is None or not rows:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    fieldnames = [
        "id",
        "status",
        "sequence_len",
        "saved_len",
        "hidden_dim",
        "structure_path",
        "structure_source",
        "embedding_path",
        "error",
    ]
    with manifest_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def manifest_error_row(record: ProteinRecord, error: str) -> dict:
    return {
        "id": record.protein_id,
        "status": "error",
        "sequence_len": len(record.sequence),
        "saved_len": 0,
        "hidden_dim": 0,
        "structure_path": record.structure_path,
        "structure_source": record.structure_source,
        "embedding_path": "",
        "error": error,
    }


def load_model(args, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model, trust_remote_code=True)
    model.eval().to(device)
    if device.type == "cuda":
        if args.model_dtype == "float16":
            model = model.half()
        elif args.model_dtype == "bfloat16":
            model = model.bfloat16()
    return tokenizer, model


def make_ss_input_ids(
    sst_sequences: list[list[int]],
    sequence_lengths: list[int],
    max_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    ss_input_ids = torch.zeros((len(sst_sequences), max_tokens), dtype=torch.long)
    for i, (sst, length) in enumerate(zip(sst_sequences, sequence_lengths)):
        ids = [1, *[int(x) + 3 for x in sst[:length]], 2]
        ss_input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return ss_input_ids.to(device)


@torch.no_grad()
def encode_and_save(
    tokenizer,
    model,
    output_dir: Path,
    records: list[ProteinRecord],
    sst_sequences: list[list[int]],
    args,
    device: torch.device,
) -> list[dict]:
    rows = []
    hidden_dim = int(model.config.hidden_size)
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start:start + args.batch_size]
        batch_sst = sst_sequences[start:start + args.batch_size]
        lengths = [
            min(len(r.sequence), len(sst), args.max_len)
            for r, sst in zip(batch_records, batch_sst)
        ]
        seqs = [r.sequence[:length] for r, length in zip(batch_records, lengths)]
        encoded = tokenizer(
            seqs,
            padding=True,
            truncation=True,
            max_length=args.max_len + 2,
            return_tensors="pt",
        ).to(device)
        ss_input_ids = make_ss_input_ids(
            batch_sst,
            sequence_lengths=lengths,
            max_tokens=encoded["input_ids"].shape[1],
            device=device,
        )
        outputs = model.prosst(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            ss_input_ids=ss_input_ids,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state.float().cpu()
        for i, (record, length) in enumerate(zip(batch_records, lengths)):
            out_path = output_dir / f"{safe_id(record.protein_id)}.npy"
            array = hidden[i, 1:1 + length].numpy().astype(np.float16)
            atomic_save_npy(out_path, array)
            rows.append(
                {
                    "id": record.protein_id,
                    "status": "ok",
                    "sequence_len": len(record.sequence),
                    "saved_len": int(array.shape[0]),
                    "hidden_dim": hidden_dim,
                    "structure_path": record.structure_path,
                    "structure_source": record.structure_source,
                    "embedding_path": str(out_path),
                    "error": "",
                }
            )
    return rows


def predict_sst_for_chunk(predictor, records: list[ProteinRecord], args) -> tuple[dict[str, list[int]], list[dict]]:
    key = f"{args.structure_vocab_size}_sst_seq"
    structure_paths = [r.structure_path for r in records]
    result_by_name = {}
    error_rows = []
    try:
        results = predictor.predict_from_pdb(structure_paths)
    except Exception as exc:
        if len(records) == 1:
            record = records[0]
            return {}, [manifest_error_row(record, str(exc))]
        sst_by_id = {}
        for record in records:
            sub_sst, sub_errors = predict_sst_for_chunk(predictor, [record], args)
            sst_by_id.update(sub_sst)
            error_rows.extend(sub_errors)
        return sst_by_id, error_rows

    for result in results:
        result_by_name[Path(result["name"]).name] = result

    sst_by_id = {}
    for record in records:
        result = result_by_name.get(Path(record.structure_path).name)
        if result is None or key not in result:
            error_rows.append(
                {
                    "id": record.protein_id,
                    "status": "error",
                    "sequence_len": len(record.sequence),
                    "saved_len": 0,
                    "hidden_dim": 0,
                    "structure_path": record.structure_path,
                    "structure_source": record.structure_source,
                    "embedding_path": "",
                    "error": f"missing {key} from ProSST tokenizer result",
                }
            )
            continue
        sst_by_id[record.protein_id] = result[key]
    return sst_by_id, error_rows


def _sst_worker_main(in_queue, out_queue, args) -> None:
    if hasattr(os, "setsid"):
        os.setsid()
    try:
        torch.multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    ensure_prosst_importable()
    from prosst.structure.get_sst_seq import SSTPredictor

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    predictor = SSTPredictor(
        structure_vocab_size=args.structure_vocab_size,
        max_batch_nodes=args.max_batch_nodes,
        num_processes=args.num_processes,
        num_threads=args.num_threads,
        device=str(device),
    )
    while True:
        item = in_queue.get()
        if item is None:
            return
        request_id, records = item
        try:
            sst_by_id, error_rows = predict_sst_for_chunk(predictor, records, args)
            out_queue.put((request_id, sst_by_id, error_rows, None))
        except BaseException as exc:
            out_queue.put((request_id, {}, [], repr(exc)))


class SSTChunkRunner:
    def __init__(self, args) -> None:
        self.args = args
        self.ctx = mp.get_context("spawn")
        self.in_queue = None
        self.out_queue = None
        self.process = None
        self.request_id = 0
        self.successful_chunks = 0

    def start(self) -> None:
        self.close(kill=True)
        self.in_queue = self.ctx.Queue(maxsize=1)
        self.out_queue = self.ctx.Queue(maxsize=1)
        self.process = self.ctx.Process(
            target=_sst_worker_main,
            args=(self.in_queue, self.out_queue, self.args),
            daemon=False,
        )
        self.process.start()
        self.successful_chunks = 0

    def _terminate_process_group(self) -> None:
        if self.process is None or not self.process.is_alive():
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except Exception:
            self.process.terminate()
        self.process.join(timeout=10)
        if self.process.is_alive():
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except Exception:
                self.process.kill()
            self.process.join(timeout=10)

    def close(self, *, kill: bool = False) -> None:
        if self.process is not None:
            if kill:
                self._terminate_process_group()
            elif self.process.is_alive() and self.in_queue is not None:
                try:
                    self.in_queue.put_nowait(None)
                except Exception:
                    pass
                self.process.join(timeout=10)
                if self.process.is_alive():
                    self._terminate_process_group()
            else:
                self.process.join(timeout=1)
        for q in (self.in_queue, self.out_queue):
            if q is not None:
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass
        self.process = None
        self.in_queue = None
        self.out_queue = None

    def _restart_if_needed(self) -> None:
        if self.process is None or not self.process.is_alive():
            self.start()
            return
        max_chunks = int(self.args.sst_worker_max_chunks)
        if max_chunks > 0 and self.successful_chunks >= max_chunks:
            self.start()

    def run_once(self, records: list[ProteinRecord]) -> tuple[dict[str, list[int]], list[dict], str | None]:
        self._restart_if_needed()
        self.request_id += 1
        request_id = self.request_id
        assert self.in_queue is not None and self.out_queue is not None
        self.in_queue.put((request_id, records))
        token_count = sum(max(1, min(record.length, self.args.max_len)) for record in records)
        timeout = None
        if self.args.sst_timeout > 0:
            timeout = float(self.args.sst_timeout) + (
                float(self.args.sst_timeout_per_1k_residues) * token_count / 1000.0
            )
        deadline = None if timeout is None else time.time() + timeout
        try:
            while True:
                wait = 1.0
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise queue.Empty
                    wait = min(wait, remaining)
                try:
                    got_id, sst_by_id, error_rows, worker_error = self.out_queue.get(timeout=wait)
                    break
                except queue.Empty:
                    if deadline is not None and time.time() >= deadline:
                        raise
                    if self.process is not None and not self.process.is_alive():
                        code = self.process.exitcode
                        self.close(kill=True)
                        return {}, [], f"structure tokenizer worker exited early with code {code}"
        except queue.Empty:
            self.close(kill=True)
            label = "disabled" if timeout is None else f"{timeout:.1f}s"
            return {}, [], f"structure tokenizer timeout after {label}"

        if got_id != request_id:
            self.close(kill=True)
            return {}, [], f"structure tokenizer protocol error: expected request {request_id}, got {got_id}"
        if worker_error:
            self.close(kill=True)
            return {}, [], f"structure tokenizer worker error: {worker_error}"
        self.successful_chunks += 1
        return sst_by_id, error_rows, None

    def run_resilient(self, records: list[ProteinRecord], *, depth: int = 0) -> tuple[dict[str, list[int]], list[dict]]:
        sst_by_id, error_rows, error = self.run_once(records)
        if error is None:
            return sst_by_id, error_rows
        if len(records) == 1:
            return {}, [manifest_error_row(records[0], error)]

        mid = len(records) // 2
        print(
            f"ProSST tokenizer stalled on chunk size={len(records)} depth={depth}; "
            f"splitting into {mid}+{len(records) - mid}: {error}",
            flush=True,
        )
        left_sst, left_errors = self.run_resilient(records[:mid], depth=depth + 1)
        right_sst, right_errors = self.run_resilient(records[mid:], depth=depth + 1)
        left_sst.update(right_sst)
        return left_sst, left_errors + right_errors


def main():
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_output) if args.manifest_output else None

    ensure_prosst_importable()
    from prosst.structure.get_sst_seq import SSTPredictor

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    records = load_records(args, output_dir)
    print(f"ProSST pending records: {len(records):,}", flush=True)
    if not records:
        print("ProSST extraction complete: no pending records", flush=True)
        return

    print(f"Loading {args.model} on {device} ...", flush=True)
    tokenizer, model = load_model(args, device)
    print(
        f"ProSST model loaded: hidden_dim={model.config.hidden_size}, "
        f"dtype={args.model_dtype}, max_len={args.max_len}",
        flush=True,
    )

    total_ok = 0
    total_error = 0
    start_time = time.time()
    record_chunks = chunks(records, args.chunk_size, args.chunk_token_budget)
    progress = tqdm(record_chunks, total=len(record_chunks), desc="ProSST chunks")
    print(f"Starting isolated ProSST structure-tokenizer worker on {device} ...", flush=True)
    sst_runner = SSTChunkRunner(args)
    try:
        for chunk_records in progress:
            sst_by_id, error_rows = sst_runner.run_resilient(chunk_records)
            if error_rows:
                total_error += len(error_rows)
                append_manifest(manifest_path, error_rows)

            valid_records = []
            valid_sst = []
            for record in chunk_records:
                sst = sst_by_id.get(record.protein_id)
                if sst is None:
                    continue
                length = min(len(record.sequence), len(sst), args.max_len)
                if length <= 0:
                    row = manifest_error_row(
                        record,
                        f"invalid sequence/structure length: seq={len(record.sequence)} sst={len(sst)}",
                    )
                    total_error += 1
                    append_manifest(manifest_path, [row])
                    continue
                valid_records.append(record)
                valid_sst.append(sst)

            if valid_records:
                ok_rows = encode_and_save(
                    tokenizer=tokenizer,
                    model=model,
                    output_dir=output_dir,
                    records=valid_records,
                    sst_sequences=valid_sst,
                    args=args,
                    device=device,
                )
                total_ok += len(ok_rows)
                append_manifest(manifest_path, ok_rows)

            elapsed = max(time.time() - start_time, 1.0)
            progress.set_postfix(ok=total_ok, errors=total_error, rate=f"{total_ok / elapsed:.2f}/s")
    finally:
        sst_runner.close()

    minutes = (time.time() - start_time) / 60.0
    print(
        f"ProSST extraction complete: ok={total_ok:,}, errors={total_error:,}, "
        f"time={minutes:.1f} min, output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

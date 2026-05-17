"""
Merge data/external/* sources into the existing P2PSigLip master tables.

Outputs in runs/merged/:
  - proteins_merged.csv     (extended P2PSigLip_proteins_total.csv schema)
  - interactions_merged.csv (extended P2PSigLip_interactions_total.csv +
                             label, Experimental_Method, Evidence_Type)
  - sequences_merged.csv    (id, sequence) — drop-in for get_embeddings.py --plm <name>
  - pairs_merged.csv        (fpid_1, fpid_2, label) — drop-in for create_ppi_h5_esm.py
  - merge_report.json       (per-source stats)

Run:  python p2psiglip_db/data/merge_external_sources.py
"""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external"
OUT = ROOT / "runs" / "merged"
UNIPROT_CACHE = OUT / "_uniprot_cache.fasta"
OUT.mkdir(parents=True, exist_ok=True)

# Aminocacid characters we keep when normalising sequences (drop X for MD5? no — keep as-is per source).
AA_RE = re.compile(r"[^A-Za-z]")

# Source-default classification, used when per-pair info is absent.
SOURCE_DEFAULTS = {
    # existing master sources, backfilled
    "HINT":         ("HINT meta-curated (HTP/LTP flag lost in master)", "mixed"),
    "BIOGRID":      ("BIOGRID curated (PSI-MI detection code lost)", "mixed"),
    "MINT":         ("MINT curated (PSI-MI detection code lost)", "mixed"),
    "PLM_interact": ("STRING v12 experimentally-verified PPIs (PLM-interact training data)", "mixed"),
    "PPI3D":        ("PDB-derived 3D interfaces (clustered)", "structural"),
    "PepBDB":       ("PDB-derived peptide-protein crystal complexes", "structural"),
    "FoldBench":    ("PDB biological assemblies (structure-prediction benchmark)", "structural"),
    "PLMDA_PPI":    ("PDB-derived structural PPIs (PLMDA training data)", "structural"),
    # new sources
    "BERNETT_pos":  ("HIPPIE v2.3 curated PPIs (Bernett gold standard, KaHIP-split)", "mixed"),
    "BERNETT_neg":  ("Random shuffle from human proteome (Bernett gold standard)", "negative_synthetic"),
    "PINDER":       ("PDB dimers (holo crystal/cryo-EM) — PINDER (Bushuiev et al. 2024)", "structural"),
    "PPIDB":        ("Multi-source curated (per-pair detection_methods + throughput_type)", "mixed"),
    "SKEMPI2":      ("PDB crystal complexes + ITC/SPR ΔΔG measurements (SKEMPI 2.0)", "structural"),
    "PPIREF_10A_clust03": ("PDB-derived 3D interfaces (PPIRef, 30%-similarity clustered)", "structural"),
}
for sp in ["DSCRIPT_human_train", "DSCRIPT_human_test", "DSCRIPT_fly", "DSCRIPT_mouse",
           "DSCRIPT_worm", "DSCRIPT_yeast", "DSCRIPT_ecoli"]:
    SOURCE_DEFAULTS[sp] = ("STRING experimentally-verified subset (D-SCRIPT training data)", "mixed")


def md5_of(seq: str) -> str:
    seq_norm = AA_RE.sub("", seq).upper().rstrip("*")
    return hashlib.md5(seq_norm.encode()).hexdigest()


def parse_fasta(path: Path) -> dict[str, str]:
    """Single-line and multi-line FASTA. Returns {id: sequence}."""
    out: dict[str, str] = {}
    cur_id, cur_seq = None, []
    fh = gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)
    with fh as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    out[cur_id] = "".join(cur_seq)
                # Header parsing: >sp|Q12913|REG_HUMAN -> Q12913, fall back to first token.
                head = line[1:].strip()
                tokens = head.split("|")
                cur_id = tokens[1] if len(tokens) >= 3 else head.split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            out[cur_id] = "".join(cur_seq)
    return out


# ----------------------------------------------------------------------------
# Master loader + identity registry
# ----------------------------------------------------------------------------
class Registry:
    """Holds all known proteins and the canonical fpid each maps to."""

    def __init__(self) -> None:
        self.md5_to_fpid: dict[str, str] = {}
        self.id_to_fpid: dict[str, str] = {}      # source_id (UniProt/Ensembl/PDB chain) → fpid
        self.fpid_to_orig_ids: dict[str, set[str]] = {}
        self.proteins: dict[str, dict] = {}        # fpid → row dict (the master schema)
        self.next_fpid_num: int = 1

    def load_master(self) -> None:
        path = EXT / "p2psiglip" / "P2PSigLip_proteins_total.csv"
        print(f"[master] loading {path} ...")
        df = pd.read_csv(path, low_memory=False)
        print(f"[master] {len(df):,} proteins")
        for _, row in df.iterrows():
            fpid = row["fpid"]
            md5h = row["protein_md5"]
            self.md5_to_fpid[md5h] = fpid
            self.proteins[fpid] = {
                "protein_md5":   md5h,
                "fpid":          fpid,
                "sequence":      row["sequence"],
                "length":        row["length"],
                "hydrophobicity": row["hydrophobicity"],
                "is_canonical":  row["is_canonical"],
                "original_ids":  row["original_ids"] if pd.notna(row["original_ids"]) else "",
            }
            ids = set()
            if pd.notna(row["original_ids"]):
                for tok in str(row["original_ids"]).replace(",", ";").split(";"):
                    tok = tok.strip()
                    if tok:
                        ids.add(tok)
                        self.id_to_fpid[tok] = fpid
            self.fpid_to_orig_ids[fpid] = ids
        # Determine next FP number
        nums = [int(f[2:]) for f in self.proteins if f.startswith("FP") and f[2:].isdigit()]
        self.next_fpid_num = (max(nums) + 1) if nums else 1
        print(f"[master] indexed {len(self.id_to_fpid):,} original_ids → fpid;"
              f" next FP num = {self.next_fpid_num}")

    def add(self, source_id: str, sequence: str) -> tuple[str, bool]:
        """Resolve source_id to an fpid; mint a new one if its sequence is unseen.
        Returns (fpid, is_new)."""
        if not sequence:
            raise ValueError(f"empty sequence for id={source_id}")
        # Direct id hit (don't even need MD5 if we already mapped this id)
        if source_id in self.id_to_fpid:
            return self.id_to_fpid[source_id], False
        md5h = md5_of(sequence)
        if md5h in self.md5_to_fpid:
            fpid = self.md5_to_fpid[md5h]
            self.id_to_fpid[source_id] = fpid
            self.fpid_to_orig_ids.setdefault(fpid, set()).add(source_id)
            self._update_orig_ids_str(fpid, source_id)
            return fpid, False
        # Mint
        fpid = f"FP{self.next_fpid_num:07d}"
        self.next_fpid_num += 1
        seq_norm = AA_RE.sub("", sequence).upper().rstrip("*")
        self.md5_to_fpid[md5h] = fpid
        self.id_to_fpid[source_id] = fpid
        self.fpid_to_orig_ids[fpid] = {source_id}
        self.proteins[fpid] = {
            "protein_md5":    md5h,
            "fpid":           fpid,
            "sequence":       seq_norm,
            "length":         len(seq_norm),
            "hydrophobicity": float("nan"),
            "is_canonical":   False,
            "original_ids":   source_id,
        }
        return fpid, True

    def _update_orig_ids_str(self, fpid: str, new_id: str) -> None:
        """Append new_id to a fpid's original_ids string if not already present."""
        cur = self.proteins[fpid].get("original_ids", "") or ""
        toks = [t.strip() for t in cur.replace(",", ";").split(";") if t.strip()]
        if new_id not in toks:
            toks.append(new_id)
            self.proteins[fpid]["original_ids"] = ";".join(toks)


# ----------------------------------------------------------------------------
# UniProt REST fetcher (with on-disk cache)
# ----------------------------------------------------------------------------
class UniProtFetcher:
    BATCH = 200
    URL = "https://rest.uniprot.org/uniprotkb/accessions"

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            self.cache = parse_fasta(cache_path)
            print(f"[uniprot] loaded {len(self.cache):,} cached sequences from {cache_path}")

    def fetch(self, accessions: set[str]) -> dict[str, str]:
        missing = [a for a in accessions if a not in self.cache]
        if not missing:
            return {a: self.cache[a] for a in accessions if a in self.cache}
        print(f"[uniprot] fetching {len(missing):,} missing accessions in batches of {self.BATCH}")
        new = {}
        for i in range(0, len(missing), self.BATCH):
            chunk = missing[i:i + self.BATCH]
            url = f"{self.URL}?accessions={','.join(chunk)}&format=fasta&size=500"
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    fasta_text = resp.read().decode("utf-8")
            except Exception as e:
                print(f"  batch {i // self.BATCH}: error {e}; skipping")
                continue
            cur_id, cur_seq = None, []
            for line in fasta_text.splitlines():
                if line.startswith(">"):
                    if cur_id:
                        new[cur_id] = "".join(cur_seq)
                    head = line[1:].strip()
                    tokens = head.split("|")
                    cur_id = tokens[1] if len(tokens) >= 3 else head.split()[0]
                    cur_seq = []
                else:
                    cur_seq.append(line.strip())
            if cur_id:
                new[cur_id] = "".join(cur_seq)
            if (i // self.BATCH) % 10 == 0:
                print(f"  fetched {len(new):,} so far")
            time.sleep(0.1)  # be polite
        self.cache.update(new)
        # Persist cache
        with open(self.cache_path, "w") as f:
            for acc, seq in self.cache.items():
                f.write(f">{acc}\n{seq}\n")
        print(f"[uniprot] cached total {len(self.cache):,} (new this run: {len(new):,})")
        return {a: self.cache[a] for a in accessions if a in self.cache}


# ----------------------------------------------------------------------------
# Per-source ingesters
# Each one returns: list of dict rows for interactions_merged
# Each one mutates the registry to add proteins.
# ----------------------------------------------------------------------------
def ingest_ppidb(reg: Registry, report: dict) -> list[dict]:
    print("\n=== PPIDB ===")
    pp = pd.read_parquet(EXT / "ppidb" / "ppidb_protein.parquet")
    print(f"[ppidb] proteins: {len(pp):,}")
    n_new = n_reused = 0
    for _, r in pp.iterrows():
        seq = r["sequence"]
        if not isinstance(seq, str) or not seq:
            continue
        _, is_new = reg.add(str(r["id"]), seq)
        if is_new:
            n_new += 1
        else:
            n_reused += 1
    print(f"[ppidb] proteins new={n_new:,}  reused={n_reused:,}")

    pi = pd.read_parquet(EXT / "ppidb" / "ppidb_interaction.parquet")
    print(f"[ppidb] interactions: {len(pi):,}")

    # Map throughput_type → Evidence_Type.
    #
    # PPIDB's "both" means the pair has both HTP and LTP support; it should not
    # be collapsed with "no_exp". Keep "no_exp" separate so downstream clean
    # training subsets can exclude unknown-throughput pairs without losing the
    # HTP+LTP-supported pairs.
    et_map = {"LTP": "LTP", "HTP": "HTP", "both": "HTP;LTP",
              "no_exp": "no_exp", "negative_sample": "negative_synthetic"}

    rows = []
    skipped = 0
    for r in pi.itertuples(index=False):
        u1, u2 = r.uniprot_a, r.uniprot_b
        f1 = reg.id_to_fpid.get(u1)
        f2 = reg.id_to_fpid.get(u2)
        if f1 is None or f2 is None:
            skipped += 1
            continue
        label = 0 if r.interaction_type == "negative" else 1
        evi = et_map.get(r.throughput_type, "mixed")
        meth = r.detection_methods if isinstance(r.detection_methods, str) and r.detection_methods.strip() \
               else SOURCE_DEFAULTS["PPIDB"][0]
        rows.append({
            "FPid_1": f1, "FPid_2": f2,
            "original_id1": u1, "original_id2": u2,
            "PPI_Source": "PPIDB",
            "Seq_Source": "UniProt",
            "label": label,
            "Experimental_Method": meth,
            "Evidence_Type": evi,
        })
    report["PPIDB"] = {
        "proteins_in": len(pp), "proteins_new": n_new, "proteins_reused": n_reused,
        "pairs_in": len(pi), "pairs_kept": len(rows), "pairs_skipped": skipped,
    }
    print(f"[ppidb] pairs kept={len(rows):,} skipped={skipped:,}")
    return rows


def ingest_bernett(reg: Registry, report: dict) -> list[dict]:
    print("\n=== Bernett gold ===")
    bg = EXT / "bernett_gold"
    fasta = parse_fasta(bg / "human_swissprot_oneliner.fasta")
    print(f"[bernett] FASTA: {len(fasta):,} entries")
    n_new = n_reused = 0
    for uid, seq in fasta.items():
        _, is_new = reg.add(uid, seq)
        n_new += int(is_new); n_reused += int(not is_new)
    print(f"[bernett] proteins new={n_new:,}  reused={n_reused:,}")

    rows = []
    counts = {"BERNETT_pos": 0, "BERNETT_neg": 0}
    skipped = 0
    for label, suffix, tag in [(1, "pos", "BERNETT_pos"), (0, "neg", "BERNETT_neg")]:
        for fold in (0, 1, 2):
            fp = bg / f"Intra{fold}_{suffix}_rr.txt"
            if not fp.exists():
                continue
            with open(fp) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    u1, u2 = parts[0], parts[1]
                    f1 = reg.id_to_fpid.get(u1); f2 = reg.id_to_fpid.get(u2)
                    if f1 is None or f2 is None:
                        skipped += 1
                        continue
                    meth, evi = SOURCE_DEFAULTS[tag]
                    rows.append({
                        "FPid_1": f1, "FPid_2": f2,
                        "original_id1": u1, "original_id2": u2,
                        "PPI_Source": tag, "Seq_Source": "UniProt",
                        "label": label, "Experimental_Method": meth, "Evidence_Type": evi,
                    })
                    counts[tag] += 1
    report["Bernett"] = {
        "proteins_new": n_new, "proteins_reused": n_reused,
        "pairs_pos": counts["BERNETT_pos"], "pairs_neg": counts["BERNETT_neg"],
        "pairs_skipped": skipped,
    }
    print(f"[bernett] pos={counts['BERNETT_pos']:,} neg={counts['BERNETT_neg']:,} skipped={skipped:,}")
    return rows


def ingest_dscript(reg: Registry, report: dict) -> list[dict]:
    print("\n=== DScript ===")
    ds = EXT / "dscript"
    species_map = {
        "human_train": "DSCRIPT_human_train",
        "human_test":  "DSCRIPT_human_test",
        "fly_test":    "DSCRIPT_fly",
        "mouse_test":  "DSCRIPT_mouse",
        "worm_test":   "DSCRIPT_worm",
        "yeast_test":  "DSCRIPT_yeast",
        "ecoli_test":  "DSCRIPT_ecoli",
    }
    fasta_map = {
        "human": "human.fasta", "fly": "fly.fasta", "mouse": "mouse.fasta",
        "worm": "worm.fasta", "yeast": "yeast.fasta", "ecoli": "ecoli.fasta",
    }

    # Load all FASTAs, merging into a single id → sequence (D-SCRIPT IDs are organism-prefixed → globally unique)
    all_seqs: dict[str, str] = {}
    for short, fn in fasta_map.items():
        seqs = parse_fasta(ds / "seqs" / fn)
        # D-SCRIPT FASTA headers like >9606.ENSP00000000233 — already organism-prefixed, no extra processing
        # But parse_fasta handles `|`-style first; for these headers (no `|`), it falls back to first token. Good.
        all_seqs.update(seqs)
    print(f"[dscript] FASTA total: {len(all_seqs):,} entries")
    n_new = n_reused = 0
    for sid, seq in all_seqs.items():
        if not seq:
            continue
        _, is_new = reg.add(sid, seq)
        n_new += int(is_new); n_reused += int(not is_new)
    print(f"[dscript] proteins new={n_new:,}  reused={n_reused:,}")

    rows = []
    skipped = 0
    by_tag = {}
    for stem, tag in species_map.items():
        fp = ds / "pairs" / f"{stem}.tsv"
        if not fp.exists():
            print(f"[dscript] missing {fp}"); continue
        n_local = 0
        with open(fp) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                u1, u2 = parts[0], parts[1]
                lbl = int(parts[2]) if len(parts) >= 3 and parts[2] in ("0", "1") else 1
                f1 = reg.id_to_fpid.get(u1); f2 = reg.id_to_fpid.get(u2)
                if f1 is None or f2 is None:
                    skipped += 1; continue
                meth, evi = SOURCE_DEFAULTS[tag]
                # DScript negatives are random-shuffle synthetic (per the D-SCRIPT paper).
                if lbl == 0:
                    meth = "Random shuffle synthetic negatives (D-SCRIPT)"
                    evi = "negative_synthetic"
                rows.append({
                    "FPid_1": f1, "FPid_2": f2,
                    "original_id1": u1, "original_id2": u2,
                    "PPI_Source": tag, "Seq_Source": "Ensembl",
                    "label": lbl, "Experimental_Method": meth, "Evidence_Type": evi,
                })
                n_local += 1
        by_tag[tag] = n_local
    report["DScript"] = {
        "proteins_new": n_new, "proteins_reused": n_reused,
        "pairs_per_tag": by_tag, "pairs_skipped": skipped,
    }
    print(f"[dscript] pairs total={sum(by_tag.values()):,} skipped={skipped:,}")
    return rows


PINDER_ID_RE = re.compile(r"^([0-9a-zA-Z]{4})__([^_]+)_([^-]+)--([0-9a-zA-Z]{4})__([^_]+)_([^-]+)$")


def ingest_pinder(reg: Registry, report: dict) -> list[dict]:
    """Use the Synthyra HF re-distribution (sequences inline) — no UniProt API needed."""
    print("\n=== PINDER (Synthyra HF) ===")
    from datasets import load_dataset
    cols = ["id", "receptor_sequence", "ligand_sequence"]
    print("[pinder] downloading Synthyra/PINDER (~129 MB) ...")
    ds = load_dataset("Synthyra/PINDER", split="train")
    print(f"[pinder] rows: {len(ds):,}")

    # Build per-pair STRUCTURE METHOD lookup from PPIRef raw_stats (pdb_id-keyed)
    method_by_pdb: dict[str, str] = {}
    raw_path = EXT / "ppiref" / "ppi_10A_stats" / "raw_stats.csv"
    if raw_path.exists():
        print(f"[pinder] joining method from {raw_path}")
        rs = pd.read_csv(raw_path, usecols=["PATH", "STRUCTURE METHOD"])
        for path, meth in zip(rs["PATH"].astype(str), rs["STRUCTURE METHOD"].astype(str)):
            pdb_id = Path(path).stem.split("_")[0].lower()
            method_by_pdb.setdefault(pdb_id, meth)
        print(f"[pinder] method lookup table: {len(method_by_pdb):,} pdb_ids")

    rows = []
    n_new = n_reused = 0
    skipped = 0
    for ex in ds:
        rid = ex["id"]
        rseq = ex.get("receptor_sequence") or ""
        lseq = ex.get("ligand_sequence") or ""
        if not rseq or not lseq:
            skipped += 1; continue
        m = PINDER_ID_RE.match(rid)
        if not m:
            skipped += 1; continue
        pdb_id, chain_R, uni_R, _pdb2, chain_L, uni_L = m.groups()
        pdb_id_lower = pdb_id.lower()
        # Resolve / mint each side via MD5; prefer UniProt as the source_id key
        sid_R = uni_R if uni_R != "UNDEFINED" else f"{pdb_id_lower}_{chain_R}"
        sid_L = uni_L if uni_L != "UNDEFINED" else f"{pdb_id_lower}_{chain_L}"
        f1, is_new1 = reg.add(sid_R, rseq); n_new += int(is_new1); n_reused += int(not is_new1)
        f2, is_new2 = reg.add(sid_L, lseq); n_new += int(is_new2); n_reused += int(not is_new2)
        meth = method_by_pdb.get(pdb_id_lower, "x-ray diffraction (PDB-derived; PINDER)")
        rows.append({
            "FPid_1": f1, "FPid_2": f2,
            "original_id1": sid_R, "original_id2": sid_L,
            "PPI_Source": "PINDER", "Seq_Source": "Synthyra/PINDER",
            "label": 1, "Experimental_Method": meth, "Evidence_Type": "structural",
        })
    report["PINDER"] = {
        "pairs_in": len(ds), "pairs_kept": len(rows), "pairs_skipped": skipped,
        "proteins_new": n_new, "proteins_reused": n_reused,
    }
    print(f"[pinder] pairs kept={len(rows):,} skipped={skipped:,}; proteins new={n_new:,} reused={n_reused:,}")
    return rows


def ingest_skempi2(reg: Registry, report: dict) -> list[dict]:
    print("\n=== SKEMPI2 ===")
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import IUPACData
    _three_to_one = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}
    def three_to_one(rn: str) -> str:
        return _three_to_one[rn.upper()]

    pdb_dir = EXT / "ppiref" / "skempi2" / "ppi_cleaned"
    pdb_files = sorted(pdb_dir.glob("**/*.pdb"))
    print(f"[skempi2] PDB files: {len(pdb_files):,}")

    parser = PDBParser(QUIET=True)
    rows = []
    n_new = n_reused = 0
    n_pairs_emitted = 0
    n_files_skipped = 0
    for fp in pdb_files:
        stem = fp.stem  # e.g. 1WQJ_I_B
        parts = stem.split("_")
        pdb_id = parts[0]
        chain_letters = parts[1:]
        try:
            structure = parser.get_structure(stem, fp)
        except Exception:
            n_files_skipped += 1; continue
        # Extract per-chain CA-derived sequence
        chain_seqs: dict[str, str] = {}
        for model in structure:
            for chain in model:
                cid = chain.id
                if cid not in chain_letters:
                    continue
                seq_parts = []
                for res in chain:
                    if not is_aa(res, standard=True):
                        continue
                    try:
                        seq_parts.append(three_to_one(res.get_resname()))
                    except Exception:
                        continue
                if seq_parts:
                    chain_seqs[cid] = "".join(seq_parts)
            break  # only first model
        if len(chain_seqs) < 2:
            n_files_skipped += 1; continue
        # Map chain → fpid
        chain_fpid: dict[str, str] = {}
        for cid, seq in chain_seqs.items():
            sid = f"{pdb_id}_{cid}"
            fpid, is_new = reg.add(sid, seq)
            chain_fpid[cid] = fpid
            n_new += int(is_new); n_reused += int(not is_new)
        # All pairwise
        for c1, c2 in itertools.combinations(chain_letters, 2):
            f1 = chain_fpid.get(c1); f2 = chain_fpid.get(c2)
            if f1 is None or f2 is None:
                continue
            meth, evi = SOURCE_DEFAULTS["SKEMPI2"]
            rows.append({
                "FPid_1": f1, "FPid_2": f2,
                "original_id1": f"{pdb_id}_{c1}", "original_id2": f"{pdb_id}_{c2}",
                "PPI_Source": "SKEMPI2", "Seq_Source": "PDB",
                "label": 1, "Experimental_Method": meth, "Evidence_Type": evi,
            })
            n_pairs_emitted += 1
    report["SKEMPI2"] = {
        "pdb_files": len(pdb_files), "files_skipped": n_files_skipped,
        "proteins_new": n_new, "proteins_reused": n_reused, "pairs_kept": n_pairs_emitted,
    }
    print(f"[skempi2] pairs={n_pairs_emitted:,}  proteins new={n_new:,} reused={n_reused:,}")
    return rows


def ingest_ppiref_clust03(reg: Registry, report: dict) -> list[dict]:
    print("\n=== PPIREF clust03 ===")
    split_path = EXT / "ppiref" / "PPIRef_repo" / "ppiref" / "data" / "splits" / "ppiref_10A_filtered_clustered_03.json"
    if not split_path.exists():
        print(f"[ppiref_clust03] missing {split_path}; skipping")
        report["PPIREF_10A_clust03"] = {"error": "split json missing"}
        return []
    js = json.loads(split_path.read_text())
    ppi_ids: list[str] = []
    folds = js.get("folds", {})
    for v in folds.values():
        if isinstance(v, list):
            ppi_ids.extend(v)
    print(f"[ppiref_clust03] PPI IDs in split json: {len(ppi_ids):,}")

    # Build (pdb_id_lower, sorted-chain-pair) → (uniprot_R, uniprot_L) from PINDER
    pinder = pd.read_parquet(EXT / "pinder" / "index.parquet",
                             columns=["pdb_id", "chain_R", "chain_L", "uniprot_R", "uniprot_L"])
    chains_to_uni: dict[tuple, tuple] = {}
    for r in pinder.itertuples(index=False):
        # Normalise chain ids: PINDER uses A1/A2 etc., we strip digits to A/A
        cR = re.sub(r"\d+$", "", str(r.chain_R))
        cL = re.sub(r"\d+$", "", str(r.chain_L))
        key = (str(r.pdb_id).lower(), tuple(sorted([cR, cL])))
        chains_to_uni.setdefault(key, (str(r.uniprot_R), str(r.uniprot_L)))
    print(f"[ppiref_clust03] PINDER chain-pair map size: {len(chains_to_uni):,}")

    rows = []
    n_resolved = n_skipped = 0
    for ppi_id in ppi_ids:
        # parse e.g. 4q2p_A_B  or 6q2a_F_O  or 1aoh_A_B
        parts = ppi_id.split("_")
        pdb_id = parts[0].lower()
        chains = parts[1:]
        for c1, c2 in itertools.combinations(chains, 2):
            key = (pdb_id, tuple(sorted([c1, c2])))
            uni = chains_to_uni.get(key)
            if uni is None:
                n_skipped += 1; continue
            u1, u2 = uni
            f1 = reg.id_to_fpid.get(u1); f2 = reg.id_to_fpid.get(u2)
            if f1 is None or f2 is None:
                n_skipped += 1; continue
            meth, evi = SOURCE_DEFAULTS["PPIREF_10A_clust03"]
            rows.append({
                "FPid_1": f1, "FPid_2": f2,
                "original_id1": f"{pdb_id}_{c1}:{u1}", "original_id2": f"{pdb_id}_{c2}:{u2}",
                "PPI_Source": "PPIREF_10A_clust03", "Seq_Source": "Pinder/UniProt",
                "label": 1, "Experimental_Method": meth, "Evidence_Type": evi,
            })
            n_resolved += 1
    report["PPIREF_10A_clust03"] = {
        "ppi_ids_in": len(ppi_ids), "pairs_resolved": n_resolved, "pairs_skipped": n_skipped,
    }
    print(f"[ppiref_clust03] pairs={n_resolved:,} skipped={n_skipped:,}")
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    report: dict = {}
    reg = Registry()
    reg.load_master()
    n_master_proteins_initial = len(reg.proteins)

    # Backfill master interactions with label / method / evidence defaults
    print(f"\n[master] loading interactions ...")
    master_inter = pd.read_csv(EXT / "p2psiglip" / "P2PSigLip_interactions_total.csv", low_memory=False)
    print(f"[master] interactions: {len(master_inter):,}")
    master_inter["label"] = 1
    master_inter["Experimental_Method"] = master_inter["PPI_Source"].map(
        {k: v[0] for k, v in SOURCE_DEFAULTS.items()})
    master_inter["Evidence_Type"] = master_inter["PPI_Source"].map(
        {k: v[1] for k, v in SOURCE_DEFAULTS.items()})

    # Order: PPIDB first (adds 99k UniProt sequences) → bernett → dscript → pinder → skempi2 → ppiref_clust03
    new_pair_dfs = []
    new_pair_dfs.append(pd.DataFrame(ingest_ppidb(reg, report)))
    new_pair_dfs.append(pd.DataFrame(ingest_bernett(reg, report)))
    new_pair_dfs.append(pd.DataFrame(ingest_dscript(reg, report)))
    new_pair_dfs.append(pd.DataFrame(ingest_pinder(reg, report)))
    new_pair_dfs.append(pd.DataFrame(ingest_skempi2(reg, report)))
    new_pair_dfs.append(pd.DataFrame(ingest_ppiref_clust03(reg, report)))

    new_inter = pd.concat([d for d in new_pair_dfs if not d.empty], ignore_index=True)
    print(f"\n[merge] new interaction rows: {len(new_inter):,}")
    interactions = pd.concat([master_inter, new_inter], ignore_index=True)
    print(f"[merge] total interactions: {len(interactions):,}")

    # Build proteins_merged from registry
    proteins_rows = list(reg.proteins.values())
    proteins = pd.DataFrame(proteins_rows)
    print(f"[merge] total proteins: {len(proteins):,} (added {len(proteins) - n_master_proteins_initial:,})")

    # Write outputs
    print(f"\n[write] -> {OUT}")
    proteins.to_csv(OUT / "proteins_merged.csv", index=False)
    interactions.to_csv(OUT / "interactions_merged.csv", index=False)
    proteins[["fpid", "sequence"]].rename(columns={"fpid": "id"}).to_csv(
        OUT / "sequences_merged.csv", index=False)
    interactions[["FPid_1", "FPid_2", "label"]].rename(
        columns={"FPid_1": "fpid_1", "FPid_2": "fpid_2"}).to_csv(
        OUT / "pairs_merged.csv", index=False)

    # Crosstab summaries for the report
    report["_crosstab_PPI_Source_x_Evidence_Type"] = pd.crosstab(
        interactions["PPI_Source"], interactions["Evidence_Type"], dropna=False).to_dict()
    report["_crosstab_PPI_Source_x_label"] = pd.crosstab(
        interactions["PPI_Source"], interactions["label"], dropna=False).to_dict()
    report["_totals"] = {
        "proteins_master_initial": n_master_proteins_initial,
        "proteins_total": len(proteins),
        "interactions_master_initial": len(master_inter),
        "interactions_new": len(new_inter),
        "interactions_total": len(interactions),
    }

    with open(OUT / "merge_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[write] merge_report.json")

    print("\nDONE.")


if __name__ == "__main__":
    main()

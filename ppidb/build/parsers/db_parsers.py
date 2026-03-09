"""
Database-specific parsers for non-MITAB formats.
Each function returns a pl.DataFrame in ppidb standard schema.
"""
from __future__ import annotations

import gzip
import re
from itertools import combinations
from pathlib import Path

import polars as pl

from .mitab import parse_mitab, _SCHEMA

# ── BioGRID (tab3 format) ─────────────────────────────────────────────────────

def parse_biogrid(path: Path, taxon_filter: str = "9606") -> pl.DataFrame:
    """
    Parse BioGRID tab3 format.
    Columns of interest (0-indexed):
      7  = Entrez Gene Interactor A
      8  = Entrez Gene Interactor B
      23 = SWISS-PROT Accessions Interactor A
      24 = SWISS-PROT Accessions Interactor B
      11 = Experimental System
      12 = Experimental System Type (physical / genetic)
      15 = Organism ID Interactor A
      16 = Organism ID Interactor B
      14 = Source Database
    """
    rows = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split("\t")
            if header is None:
                header = cols
                continue
            if len(cols) < 25:
                continue

            taxon_a = cols[15].strip()
            taxon_b = cols[16].strip()
            if taxon_filter and (taxon_a != taxon_filter or taxon_b != taxon_filter):
                continue

            # Prefer UniProt (SWISS-PROT) IDs
            uniprot_a = cols[23].strip().split("|")[0] if cols[23].strip() not in ("-", "") else None
            uniprot_b = cols[24].strip().split("|")[0] if cols[24].strip() not in ("-", "") else None
            if not uniprot_a or not uniprot_b:
                continue
            if uniprot_a == uniprot_b:
                continue

            exp_system = cols[11].strip()
            exp_type   = cols[12].strip().lower()  # 'physical' or 'genetic'
            if exp_type == "genetic":
                continue  # keep only physical interactions

            throughput = _biogrid_throughput(exp_system)

            if uniprot_a > uniprot_b:
                uniprot_a, uniprot_b = uniprot_b, uniprot_a
                taxon_a, taxon_b = taxon_b, taxon_a

            rows.append({
                "uniprot_a": uniprot_a,
                "uniprot_b": uniprot_b,
                "source_dbs": "BioGRID",
                "taxon_a": taxon_a,
                "taxon_b": taxon_b,
                "detection_methods": exp_system,
                "interaction_type": "positive",
                "throughput_type": throughput,
            })

    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows).unique(subset=["uniprot_a", "uniprot_b"])
    return df.with_columns(pl.lit(1).alias("n_sources"))


def _biogrid_throughput(exp_system: str) -> str:
    htp = {
        "Two-hybrid array", "Protein-peptide", "Protein-RNA",
        "Affinity Capture-MS", "Co-fractionation",
        "PCA", "FRET", "Two-hybrid",
    }
    ltp = {
        "Co-immunoprecipitation", "Co-crystal Structure",
        "Biochemical Activity", "Far Western", "FRET",
        "Reconstituted Complex", "Co-purification",
        "Proximity Label-MS",
    }
    if exp_system in htp and exp_system in ltp:
        return "both"
    if exp_system in htp:
        return "HTP"
    if exp_system in ltp:
        return "LTP"
    return "no_exp"


# ── STRING ────────────────────────────────────────────────────────────────────

def parse_string(
    links_path: Path,
    aliases_path: Path | None = None,
    min_score: int = 700,
    taxon: str = "9606",
) -> pl.DataFrame:
    """
    Parse STRING protein.links.full file.
    Filters by combined_score >= min_score (default 700 = high confidence).
    Maps STRING IDs to UniProt via aliases file if provided.

    STRING full columns:
      protein1, protein2, neighborhood, neighborhood_transferred,
      fusion, cooccurence, coexpression, coexpression_transferred,
      experiments, experiments_transferred, database, database_transferred,
      textmining, textmining_transferred, combined_score
    """
    # Build STRING→UniProt mapping
    id_map: dict[str, str] = {}
    if aliases_path and aliases_path.exists():
        opener = gzip.open if str(aliases_path).endswith(".gz") else open
        with opener(aliases_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3 and "UniProt_AC" in parts[2]:
                    string_id = parts[0]
                    uniprot   = parts[1]
                    id_map[string_id] = uniprot

    rows = []
    opener = gzip.open if str(links_path).endswith(".gz") else open
    with opener(links_path, "rt", encoding="utf-8", errors="replace") as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            cols = line.split()
            if header is None:
                header = cols
                continue
            if len(cols) < 15:
                continue

            try:
                score = int(cols[14])
            except ValueError:
                continue
            if score < min_score:
                continue

            sid_a, sid_b = cols[0], cols[1]
            # Map to UniProt
            uniprot_a = id_map.get(sid_a, sid_a.replace(f"{taxon}.", ""))
            uniprot_b = id_map.get(sid_b, sid_b.replace(f"{taxon}.", ""))

            # Only keep proper UniProt-like IDs
            if not re.match(r"[A-Z][0-9][A-Z0-9]{3}[0-9]", uniprot_a):
                continue
            if not re.match(r"[A-Z][0-9][A-Z0-9]{3}[0-9]", uniprot_b):
                continue
            if uniprot_a == uniprot_b:
                continue

            # Infer evidence type from score columns
            exp_score = int(cols[8]) if len(cols) > 8 else 0
            throughput = "HTP" if exp_score >= 400 else "no_exp"

            if uniprot_a > uniprot_b:
                uniprot_a, uniprot_b = uniprot_b, uniprot_a

            rows.append({
                "uniprot_a": uniprot_a,
                "uniprot_b": uniprot_b,
                "source_dbs": "STRING",
                "taxon_a": taxon,
                "taxon_b": taxon,
                "detection_methods": f"combined_score:{score}",
                "interaction_type": "positive",
                "throughput_type": throughput,
            })

    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows).unique(subset=["uniprot_a", "uniprot_b"])
    return df.with_columns(pl.lit(1).alias("n_sources"))


# ── HIPPIE ────────────────────────────────────────────────────────────────────

def parse_hippie(path: Path, min_score: float = 0.63) -> pl.DataFrame:
    """
    Parse HIPPIE current.txt.
    Columns: UniProt_A, Entrez_A, UniProt_B, Entrez_B, score, sources
    Filters by confidence score >= min_score (0.63 = medium confidence).
    """
    rows = []
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("UniProt") or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            uniprot_a = cols[0].strip()
            uniprot_b = cols[2].strip()
            try:
                score = float(cols[4].strip())
            except ValueError:
                continue
            if score < min_score:
                continue
            if not uniprot_a or not uniprot_b or uniprot_a == uniprot_b:
                continue
            if uniprot_a > uniprot_b:
                uniprot_a, uniprot_b = uniprot_b, uniprot_a
            rows.append({
                "uniprot_a": uniprot_a,
                "uniprot_b": uniprot_b,
                "source_dbs": "HIPPIE",
                "taxon_a": "9606",
                "taxon_b": "9606",
                "detection_methods": "",
                "interaction_type": "positive",
                "throughput_type": "no_exp",
            })
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows).unique(subset=["uniprot_a", "uniprot_b"])
    return df.with_columns(pl.lit(1).alias("n_sources"))


# ── BioPlex ───────────────────────────────────────────────────────────────────

def parse_bioplex(path: Path) -> pl.DataFrame:
    """
    Parse BioPlex TSV.
    Columns: GeneA, GeneB, UniprotA, UniprotB, SymbolA, SymbolB, pNI, pInt
    """
    rows = []
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        header = None
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if header is None:
                header = cols
                continue
            if len(cols) < 4:
                continue
            uniprot_a = cols[2].strip()
            uniprot_b = cols[3].strip()
            if not uniprot_a or not uniprot_b or uniprot_a == uniprot_b:
                continue
            if uniprot_a > uniprot_b:
                uniprot_a, uniprot_b = uniprot_b, uniprot_a
            rows.append({
                "uniprot_a": uniprot_a,
                "uniprot_b": uniprot_b,
                "source_dbs": "BioPlex",
                "taxon_a": "9606",
                "taxon_b": "9606",
                "detection_methods": "affinity chromatography technology",
                "interaction_type": "positive",
                "throughput_type": "HTP",
            })
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows).unique(subset=["uniprot_a", "uniprot_b"])
    return df.with_columns(pl.lit(1).alias("n_sources"))


# ── CORUM (protein complexes → pairwise interactions) ─────────────────────────

def parse_corum(path: Path, taxon_filter: str = "9606") -> pl.DataFrame:
    """
    Parse CORUM allComplexes.txt.
    Expands each complex into all pairwise protein interactions.
    Relevant columns: ComplexID, ComplexName, Organism, subunits(UniProt IDs)
    """
    TAXON_MAP = {
        "Human": "9606", "Mouse": "10090", "Rat": "10116",
        "Bovine": "9913", "Rabbit": "9986",
    }
    rows = []
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        header = None
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if header is None:
                header = [c.strip() for c in cols]
                continue
            if len(cols) < 6:
                continue
            try:
                organism = cols[2].strip()
                taxon = TAXON_MAP.get(organism, "")
                if taxon_filter and taxon != taxon_filter:
                    continue
                # subunits(UniProt IDs) column — find by header
                subunit_col = header.index("subunits(UniProt IDs)") if "subunits(UniProt IDs)" in header else 5
                subunits_raw = cols[subunit_col].strip()
                subunits = [s.strip() for s in subunits_raw.split(";") if s.strip()]
                # Expand to pairwise
                for a, b in combinations(sorted(subunits), 2):
                    if a == b:
                        continue
                    rows.append({
                        "uniprot_a": a,
                        "uniprot_b": b,
                        "source_dbs": "CORUM",
                        "taxon_a": taxon,
                        "taxon_b": taxon,
                        "detection_methods": "co-complex",
                        "interaction_type": "positive",
                        "throughput_type": "LTP",
                    })
            except (IndexError, ValueError):
                continue

    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows).unique(subset=["uniprot_a", "uniprot_b"])
    return df.with_columns(pl.lit(1).alias("n_sources"))


# ── Negatome ──────────────────────────────────────────────────────────────────

def parse_negatome(path: Path) -> pl.DataFrame:
    """
    Parse Negatome combined_stringent.txt.
    Format: UniProt_A <tab> UniProt_B (two columns, no header).
    """
    rows = []
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            a, b = cols[0].strip(), cols[1].strip()
            if not a or not b or a == b:
                continue
            if a > b:
                a, b = b, a
            rows.append({
                "uniprot_a": a,
                "uniprot_b": b,
                "source_dbs": "Negatome",
                "taxon_a": "",
                "taxon_b": "",
                "detection_methods": "literature_curated",
                "interaction_type": "negative",
                "throughput_type": "negative_sample",
            })
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows).unique(subset=["uniprot_a", "uniprot_b"])
    return df.with_columns(pl.lit(1).alias("n_sources"))


# ── PhosphoSitePlus (kinase-substrate) ────────────────────────────────────────

def parse_phosphositeplus(path: Path, taxon_filter: str = "human") -> pl.DataFrame:
    """
    Parse PhosphoSitePlus Kinase_Substrate_Dataset.
    Columns: KINASE, KIN_ACC_ID, KIN_ORGANISM, SUBSTRATE, SUB_ACC_ID,
             SUB_ORGANISM, SUB_MOD_RSD, SITE_GRP_ID, SITE_+/-7_AA,
             DOMAIN, IN_VIVO_RXN, IN_VITRO_RXN, CST_Catalog#
    """
    rows = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line:
                continue
            cols = line.split("\t")
            if header is None:
                header = cols
                continue
            if len(cols) < 6:
                continue
            kin_org = cols[2].strip().lower()
            sub_org = cols[5].strip().lower()
            if taxon_filter and (taxon_filter not in kin_org or taxon_filter not in sub_org):
                continue
            uniprot_a = cols[1].strip()  # KIN_ACC_ID
            uniprot_b = cols[4].strip()  # SUB_ACC_ID
            if not uniprot_a or not uniprot_b or uniprot_a == uniprot_b:
                continue
            if uniprot_a > uniprot_b:
                uniprot_a, uniprot_b = uniprot_b, uniprot_a
            rows.append({
                "uniprot_a": uniprot_a,
                "uniprot_b": uniprot_b,
                "source_dbs": "PhosphoSitePlus",
                "taxon_a": "9606",
                "taxon_b": "9606",
                "detection_methods": "kinase_substrate",
                "interaction_type": "positive",
                "throughput_type": "LTP",
            })
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows).unique(subset=["uniprot_a", "uniprot_b"])
    return df.with_columns(pl.lit(1).alias("n_sources"))


# ── OmniPath ──────────────────────────────────────────────────────────────────

def parse_omnipath(path: Path) -> pl.DataFrame:
    """
    Parse OmniPath TSV (from REST API).
    Columns: source, target, source_genesymbol, target_genesymbol,
             is_directed, is_stimulation, is_inhibition,
             consensus_direction, consensus_stimulation, consensus_inhibition,
             sources, references, curation_effort
    """
    rows = []
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        header = None
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if header is None:
                header = cols
                continue
            if len(cols) < 2:
                continue
            uniprot_a = cols[0].strip()
            uniprot_b = cols[1].strip()
            if not uniprot_a or not uniprot_b or uniprot_a == uniprot_b:
                continue
            sources = cols[10].strip() if len(cols) > 10 else ""
            if uniprot_a > uniprot_b:
                uniprot_a, uniprot_b = uniprot_b, uniprot_a
            rows.append({
                "uniprot_a": uniprot_a,
                "uniprot_b": uniprot_b,
                "source_dbs": "OmniPath",
                "taxon_a": "9606",
                "taxon_b": "9606",
                "detection_methods": sources[:200] if sources else "",
                "interaction_type": "positive",
                "throughput_type": "no_exp",
            })
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows).unique(subset=["uniprot_a", "uniprot_b"])
    return df.with_columns(pl.lit(1).alias("n_sources"))


# ── Complex Portal ────────────────────────────────────────────────────────────

def parse_complex_portal(path: Path) -> pl.DataFrame:
    """
    Parse Complex Portal TSV (9606.tsv).
    Columns: Complex ac, Recommended name, Aliases for complex,
             Taxonomy identifier, Identifiers (and stoichiometry) of molecules
             in complex, Evidence Code, Experimental evidence, ...
    Expands each complex into pairwise UniProt interactions.
    """
    rows = []
    _uniprot_re = re.compile(r"uniprotkb:([A-Z][0-9][A-Z0-9]{3}[0-9])")
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        header = None
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if header is None:
                header = cols
                continue
            if len(cols) < 5:
                continue
            taxon = cols[3].strip()
            members_raw = cols[4]
            members = _uniprot_re.findall(members_raw)
            members = sorted(set(members))
            for a, b in combinations(members, 2):
                if a == b:
                    continue
                rows.append({
                    "uniprot_a": a,
                    "uniprot_b": b,
                    "source_dbs": "ComplexPortal",
                    "taxon_a": taxon,
                    "taxon_b": taxon,
                    "detection_methods": "co-complex",
                    "interaction_type": "positive",
                    "throughput_type": "LTP",
                })
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows).unique(subset=["uniprot_a", "uniprot_b"])
    return df.with_columns(pl.lit(1).alias("n_sources"))


# ── SIGNOR ────────────────────────────────────────────────────────────────────

def parse_signor(path: Path) -> pl.DataFrame:
    """
    Parse SIGNOR TSV (human causal interactions).
    Columns: ENTITYA, TYPEA, IDA, DATABASEA, ENTITYB, TYPEB, IDB, DATABASEB,
             EFFECT, MECHANISM, RESIDUE, SEQUENCE, TAX_ID, CELL_DATA,
             TISSUE_DATA, MODULATOR_COMPLEX, TARGET_COMPLEX,
             MODIFICATIONA, MODASEQA, MODIFICATIONB, MODASEQB,
             PMID, DIRECT, ANNOTATOR, SENTENCE, SIGNOR_ID
    """
    rows = []
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        header = None
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if header is None:
                header = cols
                continue
            if len(cols) < 8:
                continue
            type_a = cols[1].strip()
            type_b = cols[5].strip()
            # Keep only protein-protein interactions
            if type_a != "protein" or type_b != "protein":
                continue
            db_a = cols[3].strip().lower()
            db_b = cols[7].strip().lower()
            if "uniprot" not in db_a or "uniprot" not in db_b:
                continue
            uniprot_a = cols[2].strip()
            uniprot_b = cols[6].strip()
            if not uniprot_a or not uniprot_b or uniprot_a == uniprot_b:
                continue
            mechanism = cols[9].strip() if len(cols) > 9 else ""
            if uniprot_a > uniprot_b:
                uniprot_a, uniprot_b = uniprot_b, uniprot_a
            rows.append({
                "uniprot_a": uniprot_a,
                "uniprot_b": uniprot_b,
                "source_dbs": "SIGNOR",
                "taxon_a": "9606",
                "taxon_b": "9606",
                "detection_methods": mechanism,
                "interaction_type": "positive",
                "throughput_type": "LTP",
            })
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows).unique(subset=["uniprot_a", "uniprot_b"])
    return df.with_columns(pl.lit(1).alias("n_sources"))


# ── HuRI ──────────────────────────────────────────────────────────────────────

def parse_huri(path: Path) -> pl.DataFrame:
    """
    Parse HuRI TSV (two UniProt columns, no header).
    Format: UniProt_A <tab> UniProt_B
    """
    rows = []
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            a, b = cols[0].strip(), cols[1].strip()
            if not a or not b or a == b:
                continue
            if a > b:
                a, b = b, a
            rows.append({
                "uniprot_a": a,
                "uniprot_b": b,
                "source_dbs": "HuRI",
                "taxon_a": "9606",
                "taxon_b": "9606",
                "detection_methods": "two hybrid",
                "interaction_type": "positive",
                "throughput_type": "HTP",
            })
    if not rows:
        return pl.DataFrame(schema=_SCHEMA)
    df = pl.DataFrame(rows).unique(subset=["uniprot_a", "uniprot_b"])
    return df.with_columns(pl.lit(1).alias("n_sources"))

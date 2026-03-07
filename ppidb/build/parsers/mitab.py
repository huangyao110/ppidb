"""
PSI-MITAB 2.5/2.7 parser — converts raw MITAB files to standard ppidb schema.

PSI-MITAB column layout (2.5, 0-indexed):
  0  ID(s) interactor A
  1  ID(s) interactor B
  2  Alt. ID(s) interactor A
  3  Alt. ID(s) interactor B
  4  Alias(es) interactor A
  5  Alias(es) interactor B
  6  Interaction detection method(s)
  7  Publication 1st author(s)
  8  Publication Identifier(s)
  9  Taxon interactor A
  10 Taxon interactor B
  11 Interaction type(s)
  12 Source database(s)
  13 Interaction identifier(s)
  14 Confidence value(s)
  ... (2.7 adds more columns)

Output schema (ppidb standard):
  uniprot_a, uniprot_b, source_dbs, taxon_a, taxon_b,
  detection_methods, interaction_type, throughput_type, n_sources
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Iterator

import polars as pl


# ── UniProt ID extraction ─────────────────────────────────────────────────────

_UNIPROT_RE = re.compile(
    r"uniprotkb:([A-Z][0-9][A-Z0-9]{3}[0-9](?:-\d+)?)", re.IGNORECASE
)
_TAXON_RE = re.compile(r"taxid:(\d+)", re.IGNORECASE)
_METHOD_RE = re.compile(r'psi-mi:"MI:\d+"(?:\(([^)]+)\))?', re.IGNORECASE)
_ITYPE_RE  = re.compile(r'psi-mi:"MI:\d+"(?:\(([^)]+)\))?', re.IGNORECASE)


def _extract_uniprot(field: str) -> str | None:
    """Extract first UniProt accession from a MITAB ID field."""
    if not field or field in ("-", ""):
        return None
    m = _UNIPROT_RE.search(field)
    if m:
        acc = m.group(1)
        # Strip isoform suffix for canonical ID
        return acc.split("-")[0]
    return None


def _extract_taxon(field: str) -> str | None:
    """Extract NCBI taxon ID from taxid:XXXX(name) field."""
    if not field or field in ("-", ""):
        return None
    m = _TAXON_RE.search(field)
    return m.group(1) if m else None


def _extract_methods(field: str) -> str:
    """Extract method names from PSI-MI term fields, pipe-separated."""
    if not field or field in ("-", ""):
        return ""
    names = _METHOD_RE.findall(field)
    return "|".join(n for n in names if n)


def _infer_throughput(methods: str, source_db: str) -> str:
    """
    Infer throughput type from detection methods and source database.
    Returns: 'LTP', 'HTP', 'both', or 'no_exp'
    """
    m = methods.lower()
    htp_keywords = [
        "two hybrid array", "protein array", "mass spectrometry",
        "affinity chromatography", "tandem affinity", "ap-ms",
        "pull down", "proximity ligation", "chip",
    ]
    ltp_keywords = [
        "two hybrid", "coimmunoprecipitation", "co-ip",
        "surface plasmon", "isothermal titration", "nmr",
        "x-ray crystallography", "fluorescence", "bioluminescence",
        "biochemical", "in vitro", "electrophoretic mobility",
    ]
    is_htp = any(k in m for k in htp_keywords)
    is_ltp = any(k in m for k in ltp_keywords)
    if is_htp and is_ltp:
        return "both"
    if is_htp:
        return "HTP"
    if is_ltp:
        return "LTP"
    return "no_exp"


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_mitab(
    path: Path,
    source_db: str,
    taxon_filter: str | None = None,
    skip_header: bool = True,
) -> pl.DataFrame:
    """
    Parse a PSI-MITAB 2.5/2.7 file into ppidb standard schema.

    Parameters
    ----------
    path : Path
        Path to MITAB file (.txt or .gz).
    source_db : str
        Name of the source database (e.g. 'IntAct', 'MINT').
    taxon_filter : str | None
        If set, keep only pairs where both proteins have this taxon ID.
        E.g. '9606' for human-only.
    skip_header : bool
        Skip the first line if it's a header.

    Returns
    -------
    pl.DataFrame
        Standardized PPI DataFrame.
    """
    rows = []
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if skip_header and i == 0:
                continue
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 11:
                continue

            uniprot_a = _extract_uniprot(cols[0]) or _extract_uniprot(cols[2])
            uniprot_b = _extract_uniprot(cols[1]) or _extract_uniprot(cols[3])

            if not uniprot_a or not uniprot_b:
                continue
            if uniprot_a == uniprot_b:
                continue  # skip self-interactions

            taxon_a = _extract_taxon(cols[9])
            taxon_b = _extract_taxon(cols[10])

            if taxon_filter:
                if taxon_a != taxon_filter or taxon_b != taxon_filter:
                    continue

            methods = _extract_methods(cols[6]) if len(cols) > 6 else ""
            throughput = _infer_throughput(methods, source_db)

            # Canonical ordering: alphabetically sort the pair
            if uniprot_a > uniprot_b:
                uniprot_a, uniprot_b = uniprot_b, uniprot_a
                taxon_a, taxon_b = taxon_b, taxon_a

            rows.append({
                "uniprot_a": uniprot_a,
                "uniprot_b": uniprot_b,
                "source_dbs": source_db,
                "taxon_a": taxon_a or "",
                "taxon_b": taxon_b or "",
                "detection_methods": methods,
                "interaction_type": "positive",
                "throughput_type": throughput,
            })

    if not rows:
        return pl.DataFrame(schema=_SCHEMA)

    df = pl.DataFrame(rows)
    # Deduplicate within this source
    df = df.unique(subset=["uniprot_a", "uniprot_b"])
    df = df.with_columns(pl.lit(1).alias("n_sources"))
    return df


_SCHEMA = {
    "uniprot_a": pl.Utf8,
    "uniprot_b": pl.Utf8,
    "source_dbs": pl.Utf8,
    "taxon_a": pl.Utf8,
    "taxon_b": pl.Utf8,
    "detection_methods": pl.Utf8,
    "interaction_type": pl.Utf8,
    "throughput_type": pl.Utf8,
    "n_sources": pl.Int32,
}

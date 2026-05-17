"""Hard contract for the public ``data/merged`` database.

This module is intentionally strict. The files under ``data/merged`` are API
inputs, so changing column names, column order, ID namespaces, or evidence
labels requires updating this contract and the downstream consumers together.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import pandas as pd


PROTEINS_FILE = "proteins.csv"
SEQUENCES_FILE = "sequences.csv"
INTERACTIONS_FILE = "interactions.csv"
PAIRS_FILE = "pairs.csv"

EXPECTED_SNAPSHOT = {
    PROTEINS_FILE: {
        "rows": 356_373,
        "sha256": "e315cc0e3ac86596f613b75277068fc2ce79550a19a442e8099234f9f27629e5",
    },
    SEQUENCES_FILE: {
        "rows": 356_373,
        "sha256": "83bc28b50631c248125461d9f6f3a20b5c250ffcd3dde17dd24482cfbe03200e",
    },
    INTERACTIONS_FILE: {
        "rows": 8_717_404,
        "sha256": "ea6c7de251ef94de5853e229d1131826369d826bc2863281dd945a97aac130b1",
    },
    PAIRS_FILE: {
        "rows": 8_717_404,
        "sha256": "a27aaae821ae2142ed1deaaaf9ad7368283adcb9d3e1fb0558910ba2c839fb0c",
    },
}

PROTEINS_COLUMNS = (
    "protein_md5",
    "fpid",
    "sequence",
    "length",
    "hydrophobicity",
    "is_canonical",
    "original_ids",
)
SEQUENCES_COLUMNS = ("id", "sequence")
INTERACTIONS_COLUMNS = (
    "FPid_1",
    "FPid_2",
    "original_id1",
    "original_id2",
    "PPI_Source",
    "Seq_Source",
    "label",
    "Experimental_Method",
    "Evidence_Type",
    "Evidence_Tags",
    "PPI_Tier",
    "PPI_Tier_ZH",
    "n_sources",
)
PAIRS_COLUMNS = ("fpid_1", "fpid_2", "label")

FPID_RE = re.compile(r"^FP\d{7}$")
PROTEIN_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
SEQUENCE_RE = re.compile(r"^[A-Z]+$")
AA_RE = re.compile(r"[^A-Za-z]")

LABELS = {0, 1}
BOOLEAN_TEXT = {"True", "False"}
EVIDENCE_TYPES = {
    "no_exp",
    "LTP",
    "HTP",
    "negative_synthetic",
    "mixed",
    "HTP_LTP",
    "structural",
    "complex_curation",
}
EVIDENCE_TAGS = {
    "HTP",
    "LTP",
    "mixed",
    "no_exp",
    "structural",
    "negative_synthetic",
    "complex_curation",
}
PPI_TIERS = {
    "diamond": "钻石",
    "gold": "黄金",
    "silver": "白银",
    "bronze": "青铜",
    "negative_synthetic": "负样本",
}
PPI_SOURCES = {
    "BERNETT_neg",
    "BERNETT_pos",
    "BIOGRID",
    "DSCRIPT_ecoli",
    "DSCRIPT_fly",
    "DSCRIPT_human_test",
    "DSCRIPT_human_train",
    "DSCRIPT_mouse",
    "DSCRIPT_worm",
    "DSCRIPT_yeast",
    "FoldBench",
    "HINT",
    "MINT",
    "PINDER",
    "PLMDA_PPI",
    "PLM_interact",
    "PPIDB",
    "PPI3D",
    "PPIREF_10A_clust03",
    "PepBDB",
    "SKEMPI2",
}

CONSTRUCTION_RULES = (
    "proteins.csv is one row per normalized amino-acid sequence",
    "protein_md5 is md5(uppercase letters only, trailing stop removed)",
    "fpid is the only public protein ID namespace and must be FP followed by 7 digits",
    "fpid values are contiguous, increasing, and never reused for a different sequence",
    "sequences.csv is exactly proteins.csv projected to id,sequence",
    "interactions.csv is one row per unordered fpid pair",
    "FPid_1 must be lexicographically smaller than FPid_2",
    "self-pairs are forbidden",
    "if positive and synthetic-negative evidence collide, label=1 wins",
    "Evidence_Type and PPI_Tier must use the fixed enums in this module",
    "pairs.csv is exactly interactions.csv projected to fpid_1,fpid_2,label",
    "published data/merged CSV bytes must match EXPECTED_SNAPSHOT unless the contract is intentionally revised",
)


def normalize_sequence(sequence: object) -> str:
    return AA_RE.sub("", str(sequence)).upper().rstrip("*")


def sequence_md5(sequence: object) -> str:
    return hashlib.md5(normalize_sequence(sequence).encode("utf-8")).hexdigest()


def split_tokens(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {token.strip() for token in str(value).split(";") if token.strip()}


def count_tokens(value: object) -> int:
    return len(split_tokens(value))


def require_columns(actual: Iterable[str], expected: tuple[str, ...], table_name: str) -> None:
    actual_tuple = tuple(actual)
    if actual_tuple != expected:
        raise ValueError(
            f"{table_name}: header drift. Expected {list(expected)}, got {list(actual_tuple)}"
        )


def order_proteins(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, list(PROTEINS_COLUMNS)]


def order_sequences(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, list(SEQUENCES_COLUMNS)]


def order_interactions(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, list(INTERACTIONS_COLUMNS)]


def order_pairs(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, list(PAIRS_COLUMNS)]

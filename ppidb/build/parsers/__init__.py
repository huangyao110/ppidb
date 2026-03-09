"""Parsers for individual PPI databases."""
from .mitab import parse_mitab
from .db_parsers import (
    parse_biogrid,
    parse_string,
    parse_hippie,
    parse_bioplex,
    parse_corum,
    parse_negatome,
    parse_phosphositeplus,
    parse_omnipath,
    parse_complex_portal,
    parse_signor,
    parse_huri,
)

__all__ = [
    "parse_mitab",
    "parse_biogrid",
    "parse_string",
    "parse_hippie",
    "parse_bioplex",
    "parse_corum",
    "parse_negatome",
    "parse_phosphositeplus",
    "parse_omnipath",
    "parse_complex_portal",
    "parse_signor",
    "parse_huri",
]

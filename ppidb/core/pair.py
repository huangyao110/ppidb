"""
PPIPair — lightweight dataclass representing a single protein-protein interaction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PPIPair:
    """
    Immutable representation of a single PPI record.

    Attributes
    ----------
    uniprot_a : str
        UniProt accession of protein A (canonical, lexicographically smaller).
    uniprot_b : str
        UniProt accession of protein B.
    source_dbs : tuple[str, ...]
        Databases reporting this interaction (pipe-separated in raw data).
    n_sources : int
        Number of independent databases supporting this interaction.
    taxon_a : str | None
        NCBI taxonomy ID of protein A.
    taxon_b : str | None
        NCBI taxonomy ID of protein B.
    detection_methods : tuple[str, ...]
        Experimental detection methods (pipe-separated in raw data).
    interaction_type : str
        'positive' or 'negative'.
    throughput_type : str
        'LTP', 'HTP', 'both', 'no_exp', or 'negative_sample'.
    """

    uniprot_a: str
    uniprot_b: str
    source_dbs: tuple[str, ...] = field(default_factory=tuple)
    n_sources: int = 1
    taxon_a: Optional[str] = None
    taxon_b: Optional[str] = None
    detection_methods: tuple[str, ...] = field(default_factory=tuple)
    interaction_type: str = "positive"
    throughput_type: str = "no_exp"

    def __post_init__(self):
        # Enforce canonical ordering: smaller ID always in position A
        if self.uniprot_a > self.uniprot_b:
            a, b = self.uniprot_a, self.uniprot_b
            object.__setattr__(self, "uniprot_a", b)
            object.__setattr__(self, "uniprot_b", a)

    @property
    def pair_id(self) -> str:
        """Canonical string identifier: 'A__B'."""
        return f"{self.uniprot_a}__{self.uniprot_b}"

    @property
    def is_positive(self) -> bool:
        return self.interaction_type == "positive"

    @property
    def is_negative(self) -> bool:
        return self.interaction_type == "negative"

    @property
    def is_human(self) -> bool:
        return self.taxon_a == "9606" and self.taxon_b == "9606"

    @property
    def is_ltp(self) -> bool:
        """Low-throughput experimental evidence."""
        return self.throughput_type in ("LTP", "both")

    @property
    def is_htp(self) -> bool:
        """High-throughput experimental evidence."""
        return self.throughput_type in ("HTP", "both")

    @classmethod
    def from_dict(cls, d: dict) -> "PPIPair":
        """Construct from a raw row dict (e.g. from Polars .to_dicts())."""

        def _split_pipe(val) -> tuple[str, ...]:
            if not val or val == "None":
                return ()
            return tuple(v.strip() for v in str(val).split("|") if v.strip())

        return cls(
            uniprot_a=str(d["uniprot_a"]),
            uniprot_b=str(d["uniprot_b"]),
            source_dbs=_split_pipe(d.get("source_dbs")),
            n_sources=int(d.get("n_sources", 1)),
            taxon_a=str(d["taxon_a"]) if d.get("taxon_a") else None,
            taxon_b=str(d["taxon_b"]) if d.get("taxon_b") else None,
            detection_methods=_split_pipe(d.get("detection_methods")),
            interaction_type=str(d.get("interaction_type", "positive")),
            throughput_type=str(d.get("throughput_type", "no_exp")),
        )

    def __repr__(self) -> str:
        return (
            f"PPIPair({self.uniprot_a} <-> {self.uniprot_b} | "
            f"sources={self.n_sources} | {self.throughput_type} | {self.interaction_type})"
        )

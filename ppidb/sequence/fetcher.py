"""
SequenceFetcher — batch retrieval of protein sequences from UniProt.

Features:
  - Batch API calls (200 proteins per request) for efficiency
  - Local disk cache to avoid redundant downloads
  - Automatic retry with exponential backoff
  - Isoform handling (strips isoform suffix, e.g. P12345-2 → P12345)
  - Output as FASTA file or dict {uniprot_id: sequence}
  - Optional subcellular compartment fetching for NegativeSampler
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests


# UniProt REST API endpoint
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/accessions"
UNIPROT_FIELDS = "accession,sequence,organism_id,subcellular_location"
BATCH_SIZE = 200
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds


class SequenceFetcher:
    """
    Fetch protein sequences and metadata from UniProt.

    Parameters
    ----------
    cache_dir : str | Path | None
        Directory to cache downloaded sequences.
        Default: ~/.ppidb/sequence_cache/
        Set to None to disable caching.

    Examples
    --------
    >>> fetcher = SequenceFetcher()

    >>> # Fetch sequences as dict
    >>> seqs = fetcher.fetch(["P04637", "P53350", "O15111"], as_dict=True)
    >>> seqs["P04637"][:20]
    'MEEPQSDPSVEPPLSQETF'

    >>> # Fetch and save as FASTA
    >>> fetcher.fetch(ds.proteins(), output_fasta="proteins.fasta")

    >>> # Fetch subcellular compartments
    >>> compartments = fetcher.fetch_compartments(["P04637", "P53350"])
    """

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".ppidb" / "sequence_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "sequences.json"
        self._cache: Dict[str, str] = self._load_cache()

    # ── Cache management ──────────────────────────────────────────────────────

    def _load_cache(self) -> Dict[str, str]:
        if self._cache_file.exists():
            with open(self._cache_file) as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        with open(self._cache_file, "w") as f:
            json.dump(self._cache, f)

    def cache_size(self) -> int:
        """Number of sequences in local cache."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Clear the local sequence cache."""
        self._cache = {}
        if self._cache_file.exists():
            self._cache_file.unlink()
        print("Cache cleared.")

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

        Parameters
        ----------
        proteins : list[str]
            UniProt accession IDs (e.g. ['P04637', 'P53350']).
            Isoform suffixes (e.g. 'P04637-2') are handled automatically.
        as_dict : bool
            If True, return {uniprot_id: sequence} dict.
            If False (default), return FASTA-formatted string.
        output_fasta : str | Path | None
            If provided, save FASTA to this file.
        include_isoforms : bool
            If True, keep isoform IDs as-is. If False (default), strip
            isoform suffix and use canonical sequence.
        verbose : bool
            Print progress.

        Returns
        -------
        dict[str, str] | str
            Sequences as dict or FASTA string.
        """
        # Normalize IDs
        if not include_isoforms:
            proteins = [p.split("-")[0] for p in proteins]
        proteins = sorted(set(proteins))

        # Split into cached and uncached
        cached = {p: self._cache[p] for p in proteins if p in self._cache}
        uncached = [p for p in proteins if p not in self._cache]

        if verbose:
            print(f"Fetching sequences: {len(proteins):,} total, "
                  f"{len(cached):,} cached, {len(uncached):,} to download")

        # Fetch uncached in batches
        fetched = {}
        failed = []
        for i in range(0, len(uncached), BATCH_SIZE):
            batch = uncached[i: i + BATCH_SIZE]
            if verbose:
                print(f"  Batch {i//BATCH_SIZE + 1}/{(len(uncached)-1)//BATCH_SIZE + 1}: "
                      f"{len(batch)} proteins...", end=" ", flush=True)
            result, batch_failed = self._fetch_batch(batch)
            fetched.update(result)
            failed.extend(batch_failed)
            if verbose:
                print(f"OK ({len(result)} fetched, {len(batch_failed)} failed)")

        # Update cache
        self._cache.update(fetched)
        self._save_cache()

        if failed:
            warnings.warn(
                f"{len(failed)} proteins could not be fetched: "
                f"{failed[:5]}{'...' if len(failed) > 5 else ''}"
            )

        # Combine all sequences
        all_seqs = {**cached, **fetched}

        if verbose:
            print(f"Total sequences retrieved: {len(all_seqs):,} / {len(proteins):,}")

        # Output
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
        """Fetch a batch of sequences from UniProt REST API."""
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(
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

            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    warnings.warn(f"Failed to fetch batch after {MAX_RETRIES} attempts: {e}")
                    return {}, accessions

        return {}, accessions

    # ── Subcellular compartments ──────────────────────────────────────────────

    def fetch_compartments(
        self,
        proteins: List[str],
        verbose: bool = True,
    ) -> Dict[str, str]:
        """
        Fetch subcellular localization for proteins from UniProt.

        Returns {uniprot_id: compartment_name} mapping.
        Used by NegativeSampler.subcellular_sample().

        Parameters
        ----------
        proteins : list[str]
            UniProt accession IDs.

        Returns
        -------
        dict[str, str]
            {uniprot_id: primary_compartment}
        """
        proteins = sorted(set(p.split("-")[0] for p in proteins))
        compartment_cache_file = self.cache_dir / "compartments.json"

        # Load compartment cache
        comp_cache = {}
        if compartment_cache_file.exists():
            with open(compartment_cache_file) as f:
                comp_cache = json.load(f)

        uncached = [p for p in proteins if p not in comp_cache]

        if verbose:
            print(f"Fetching compartments: {len(proteins):,} total, "
                  f"{len(comp_cache):,} cached, {len(uncached):,} to download")

        for i in range(0, len(uncached), BATCH_SIZE):
            batch = uncached[i: i + BATCH_SIZE]
            if verbose:
                print(f"  Batch {i//BATCH_SIZE + 1}/{(len(uncached)-1)//BATCH_SIZE + 1}...",
                      end=" ", flush=True)
            result = self._fetch_compartments_batch(batch)
            comp_cache.update(result)
            if verbose:
                print(f"OK ({len(result)} fetched)")

        # Save cache
        with open(compartment_cache_file, "w") as f:
            json.dump(comp_cache, f)

        return {p: comp_cache.get(p, "Unknown") for p in proteins}

    def _fetch_compartments_batch(self, accessions: List[str]) -> Dict[str, str]:
        """Fetch subcellular location for a batch."""
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(
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
                for entry in data.get("results", []):
                    acc = entry["primaryAccession"]
                    locs = entry.get("comments", [])
                    compartment = "Unknown"
                    for loc in locs:
                        if loc.get("commentType") == "SUBCELLULAR LOCATION":
                            subcell = loc.get("subcellularLocations", [])
                            if subcell:
                                loc_val = subcell[0].get("location", {}).get("value", "Unknown")
                                # Simplify to major compartment
                                compartment = _simplify_compartment(loc_val)
                                break
                    result[acc] = compartment
                return result

            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                else:
                    return {}
        return {}

    # ── FASTA utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _to_fasta(sequences: Dict[str, str]) -> str:
        """Convert sequence dict to FASTA string."""
        lines = []
        for pid, seq in sorted(sequences.items()):
            lines.append(f">{pid}")
            # Wrap at 60 chars
            for i in range(0, len(seq), 60):
                lines.append(seq[i:i+60])
        return "\n".join(lines) + "\n"

    @staticmethod
    def parse_fasta(fasta_path: Union[str, Path]) -> Dict[str, str]:
        """Parse a FASTA file into {id: sequence} dict."""
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
    return loc  # Return as-is if no match

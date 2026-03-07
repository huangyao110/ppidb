"""OmniPath downloader — https://omnipathdb.org
Uses the OmniPath REST API to download human PPI interactions.
OmniPath integrates 160+ resources; we download the 'omnipath' dataset
(experimentally validated, directed interactions) and the undirected
'undirected_ppi' dataset.
"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

BASE = "https://omnipathdb.org"

QUERIES = {
    "omnipath": (
        f"{BASE}/interactions"
        "?datasets=omnipath"
        "&fields=sources,references,curation_effort"
        "&format=tsv"
        "&organisms=9606"
    ),
    "undirected": (
        f"{BASE}/interactions"
        "?datasets=omnipath,pathwayextra,kinaseextra,ligrecextra"
        "&directed=0"
        "&format=tsv"
        "&organisms=9606"
    ),
}


class OmniPathDownloader(BaseDownloader):
    DB_NAME = "OmniPath"

    def download(self) -> DownloadResult:
        last_out = None
        for name, url in QUERIES.items():
            out = self.raw_dir / f"omnipath_{name}.tsv"
            if self._already_downloaded(out):
                last_out = out
                continue
            try:
                self._fetch(url, out, desc=f"OmniPath {name}")
                last_out = out
            except Exception as e:
                return self._fail(f"{name}: {e}", url=url)
        if last_out:
            return self._ok(last_out, url=QUERIES["omnipath"])
        return self._fail("No files downloaded")

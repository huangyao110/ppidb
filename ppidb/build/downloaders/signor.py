"""SIGNOR downloader — https://signor.uniroma2.it
Downloads human causal interactions in TSV format (v4.0, 2026).
"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

URL = "https://signor.uniroma2.it/download_entity.php?format=tsv&organism=9606"


class SIGNORDownloader(BaseDownloader):
    DB_NAME = "SIGNOR"

    def download(self) -> DownloadResult:
        out = self.raw_dir / "signor_human.tsv"
        if self._already_downloaded(out):
            return self._ok(out, url=URL)
        try:
            self._fetch(URL, out, desc="SIGNOR human TSV")
            return self._ok(out, url=URL)
        except Exception as e:
            return self._fail(str(e), url=URL)

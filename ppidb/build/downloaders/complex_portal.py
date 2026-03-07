"""Complex Portal downloader — https://www.ebi.ac.uk/complexportal
Downloads human complex TSV from EBI FTP.
"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

# Human complexes TSV (confirmed by HEAD test: 200 OK, 5.3 MB)
URL = "https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/9606.tsv"


class ComplexPortalDownloader(BaseDownloader):
    DB_NAME = "ComplexPortal"

    def download(self) -> DownloadResult:
        out = self.raw_dir / "complex_portal_human.tsv"
        if self._already_downloaded(out):
            return self._ok(out, url=URL)
        try:
            self._fetch(URL, out, desc="Complex Portal human TSV")
            return self._ok(out, url=URL)
        except Exception as e:
            return self._fail(str(e), url=URL)

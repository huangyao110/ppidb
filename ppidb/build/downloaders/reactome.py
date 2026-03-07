"""Reactome downloader — https://reactome.org/download/current/"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

# PSI-MITAB format, human interactions
URL = (
    "https://reactome.org/download/current/interactors/"
    "reactome.homo_sapiens.interactions.psi-mitab.txt"
)


class ReactomeDownloader(BaseDownloader):
    DB_NAME = "Reactome"

    def download(self) -> DownloadResult:
        out = self.raw_dir / "reactome_human_interactions.txt"
        if self._already_downloaded(out):
            return self._ok(out, url=URL)
        try:
            self._fetch(URL, out, desc="Reactome human MITAB")
            return self._ok(out, url=URL)
        except Exception as e:
            return self._fail(str(e), url=URL)

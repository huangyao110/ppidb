"""CORUM downloader — https://mips.helmholtz-muenchen.de/corum/
Downloads allComplexes.txt (mammalian protein complexes).
Note: correct domain is helmholtz-muenchen.de (not helmholtz-munich.de).
"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

# Confirmed correct domain from official CORUM website
URL = "https://mips.helmholtz-muenchen.de/corum/download/releases/current/allComplexes.txt.zip"
URL_JSON = "https://mips.helmholtz-muenchen.de/corum/download/releases/current/allComplexes.json.zip"


class CORUMDownloader(BaseDownloader):
    DB_NAME = "CORUM"

    def download(self) -> DownloadResult:
        zip_path = self.raw_dir / "corum_allComplexes.zip"
        out = self.raw_dir / "corum_allComplexes.txt"
        if self._already_downloaded(out):
            return self._ok(out, url=URL)
        try:
            if not self._already_downloaded(zip_path):
                self._fetch(URL, zip_path, desc="CORUM allComplexes")
            self._extract_zip(zip_path, self.raw_dir)
            txts = list(self.raw_dir.glob("allComplexes*.txt"))
            out = txts[0] if txts else out
            return self._ok(out, url=URL)
        except Exception as e:
            return self._fail(str(e), url=URL)

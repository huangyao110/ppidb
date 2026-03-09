"""IntAct downloader — https://ftp.ebi.ac.uk/pub/databases/intact/"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

# Full PSI-MITAB 2.7 dump (all species, ~1.3 GB compressed)
URL = "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact.zip"


class IntActDownloader(BaseDownloader):
    DB_NAME = "IntAct"

    def download(self) -> DownloadResult:
        zip_path = self.raw_dir / "intact.zip"
        if self._already_downloaded(zip_path):
            return self._ok(zip_path, url=URL)
        try:
            self._fetch(URL, zip_path, desc="IntAct MITAB")
            self._extract_zip(zip_path, self.raw_dir)
            out = self.raw_dir / "intact.txt"
            return self._ok(out, url=URL)
        except Exception as e:
            return self._fail(str(e), url=URL)

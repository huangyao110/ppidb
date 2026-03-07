"""BioGRID downloader — https://downloads.thebiogrid.org"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

# BioGRID latest-release index always redirects to the current version zip
URL = "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/BIOGRID-ALL-LATEST.tab3.zip"


class BioGRIDDownloader(BaseDownloader):
    DB_NAME = "BioGRID"

    def download(self) -> DownloadResult:
        zip_path = self.raw_dir / "biogrid_all.tab3.zip"
        if self._already_downloaded(zip_path):
            return self._ok(zip_path, url=URL)
        try:
            self._fetch(URL, zip_path, desc="BioGRID ALL tab3")
            # Extract the single .txt file inside
            out_dir = self._extract_zip(zip_path, self.raw_dir)
            # Find extracted file
            txts = list(self.raw_dir.glob("BIOGRID-ALL-*.tab3.txt"))
            out = txts[0] if txts else zip_path
            return self._ok(out, url=URL)
        except Exception as e:
            return self._fail(str(e), url=URL)

"""InnateDB downloader — https://www.innatedb.com"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

URL = "https://www.innatedb.com/download/interactions/innatedb_ppi.mitab.gz"


class InnateDBDownloader(BaseDownloader):
    DB_NAME = "InnateDB"

    def download(self) -> DownloadResult:
        gz_path = self.raw_dir / "innatedb_ppi.mitab.gz"
        out = self.raw_dir / "innatedb_ppi.mitab.txt"
        if self._already_downloaded(out):
            return self._ok(out, url=URL)
        try:
            if not self._already_downloaded(gz_path):
                self._fetch(URL, gz_path, desc="InnateDB MITAB")
            self._extract_gz(gz_path, out)
            return self._ok(out, url=URL)
        except Exception as e:
            return self._fail(str(e), url=URL)

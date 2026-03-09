"""HIPPIE downloader — http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

URL = "http://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/hippie_current.txt"


class HIPPIEDownloader(BaseDownloader):
    DB_NAME = "HIPPIE"

    def download(self) -> DownloadResult:
        out = self.raw_dir / "hippie_current.txt"
        if self._already_downloaded(out):
            return self._ok(out, url=URL)
        try:
            self._fetch(URL, out, desc="HIPPIE current")
            return self._ok(out, url=URL)
        except Exception as e:
            return self._fail(str(e), url=URL)

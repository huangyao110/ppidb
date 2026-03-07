"""DIP downloader — https://dip.doe-mbi.ucla.edu
DIP requires free registration. The server is often slow/unreachable.
We attempt the download and provide clear instructions on failure.
"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

# DIP full dataset in PSI-MITAB format (requires free registration)
URL = "https://dip.doe-mbi.ucla.edu/dip/Download.cgi?SM=7"
# Alternative: species-specific human file
URL_HUMAN = "https://dip.doe-mbi.ucla.edu/dip/Download.cgi?SM=7&TX=9606&type=txt"


class DIPDownloader(BaseDownloader):
    DB_NAME = "DIP"

    def download(self) -> DownloadResult:
        out = self.raw_dir / "dip_full.txt"
        if self._already_downloaded(out):
            return self._ok(out, url=URL)
        try:
            self._fetch(URL_HUMAN, out, desc="DIP human MITAB")
            if out.exists() and out.stat().st_size > 10_000:
                return self._ok(out, url=URL_HUMAN)
            return self._skip(
                "DIP requires free registration at https://dip.doe-mbi.ucla.edu. "
                "After login, download 'Hsapi20...' file and place at raw/dip/dip_full.txt"
            )
        except Exception as e:
            return self._skip(
                f"DIP server unreachable ({e}). "
                "Register at https://dip.doe-mbi.ucla.edu and download manually."
            )

"""VirHostNet downloader — https://virhostnet.prabi.fr
VirHostNet 3.0 distributes data via PSICQUIC REST API and NDEx platform.
We query the PSICQUIC endpoint for all virus-host interactions.
"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

# PSICQUIC REST endpoint for VirHostNet
PSICQUIC_URL = (
    "http://www.ebi.ac.uk/Tools/webservices/psicquic/virhostnet/webservices/"
    "current/search/query/*?format=tab25"
)
# Alternative: query all interactions in MITAB 2.5 format
PSICQUIC_ALT = (
    "https://virhostnet.prabi.fr/psicquic/webservices/current/search/query/*"
    "?format=tab25"
)


class VirHostNetDownloader(BaseDownloader):
    DB_NAME = "VirHostNet"

    def download(self) -> DownloadResult:
        out = self.raw_dir / "virhostnet_all.mitab.txt"
        if self._already_downloaded(out):
            return self._ok(out, url=PSICQUIC_ALT)
        for url in [PSICQUIC_ALT, PSICQUIC_URL]:
            try:
                self._fetch(url, out, desc="VirHostNet PSICQUIC")
                if out.stat().st_size > 1000:
                    return self._ok(out, url=url)
            except Exception:
                continue
        return self._fail(
            "VirHostNet PSICQUIC endpoint unreachable. "
            "Data also available via NDEx: https://www.ndexbio.org (search VirHostNet).",
            url=PSICQUIC_ALT,
        )

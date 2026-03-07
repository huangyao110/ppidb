"""Negatome downloader — https://mips.helmholtz-muenchen.de/proj/ppi/negatome/
Downloads the combined_stringent non-interacting protein pairs.
Note: correct domain is helmholtz-muenchen.de.
"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

BASE = "https://mips.helmholtz-muenchen.de/proj/ppi/negatome"
URLS = {
    "combined_stringent": f"{BASE}/combined_stringent.txt",
    "manual_stringent":   f"{BASE}/manual_stringent.txt",
    "pdb_stringent":      f"{BASE}/pdb_stringent.txt",
}


class NegatomeDownloader(BaseDownloader):
    DB_NAME = "Negatome"

    def download(self) -> DownloadResult:
        last_out = None
        for name, url in URLS.items():
            out = self.raw_dir / f"negatome_{name}.txt"
            if self._already_downloaded(out):
                last_out = out
                continue
            try:
                self._fetch(url, out, desc=f"Negatome {name}")
                last_out = out
            except Exception as e:
                # Negatome server is sometimes slow; log and continue
                import logging
                logging.getLogger(__name__).warning(
                    f"[Negatome] {name} failed: {e}"
                )
        if last_out:
            return self._ok(last_out, url=URLS["combined_stringent"])
        return self._fail(
            "All Negatome files failed. Server may be down. "
            "Try manually: https://mips.helmholtz-muenchen.de/proj/ppi/negatome/",
            url=URLS["combined_stringent"],
        )

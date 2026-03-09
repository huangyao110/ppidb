"""BioPlex downloader — https://bioplex.hms.harvard.edu
Downloads BioPlex 3.0 (293T cells, 10K ORFs) and Jurkat network.
"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

URLS = {
    "293T": "https://bioplex.hms.harvard.edu/data/BioPlex_293T_Network_10K_Dec_2019.tsv",
    "Jurkat": "https://bioplex.hms.harvard.edu/data/BioPlex_Jurkat_Network_Oct_2019.tsv",
}


class BioPlexDownloader(BaseDownloader):
    DB_NAME = "BioPlex"

    def download(self) -> DownloadResult:
        results = []
        last_out = None
        for cell_line, url in URLS.items():
            out = self.raw_dir / f"bioplex_{cell_line.lower()}.tsv"
            if self._already_downloaded(out):
                results.append(out)
                last_out = out
                continue
            try:
                self._fetch(url, out, desc=f"BioPlex {cell_line}")
                results.append(out)
                last_out = out
            except Exception as e:
                return self._fail(f"{cell_line}: {e}", url=url)

        if last_out:
            return self._ok(last_out, url=URLS["293T"])
        return self._fail("No files downloaded")

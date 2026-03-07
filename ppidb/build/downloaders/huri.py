"""HuRI downloader — https://interactome-atlas.org
HuRI (Human Reference Interactome) requires user registration to download.
This downloader provides instructions and attempts to fetch the published
Supplementary Data from the Nature 2020 paper as a fallback.
"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

# Published supplementary data from Luck et al. Nature 2020
# (HuRI dataset, 64,006 interactions — publicly available via Zenodo/Figshare)
ZENODO_URL = "https://zenodo.org/record/3978997/files/HuRI.tsv"
OFFICIAL_URL = "https://interactome-atlas.org/download"


class HuRIDownloader(BaseDownloader):
    DB_NAME = "HuRI"

    def download(self) -> DownloadResult:
        out = self.raw_dir / "huri.tsv"
        if self._already_downloaded(out):
            return self._ok(out, url=ZENODO_URL)
        # Try Zenodo mirror first (no login required)
        try:
            self._fetch(ZENODO_URL, out, desc="HuRI via Zenodo")
            if out.exists() and out.stat().st_size > 10_000:
                return self._ok(out, url=ZENODO_URL)
        except Exception:
            pass
        # If Zenodo fails, report skip with instructions
        return self._skip(
            "HuRI requires registration at https://interactome-atlas.org/download "
            "OR download from Zenodo: https://zenodo.org/record/3978997 "
            "(Luck et al. Nature 2020, Supplementary Data). "
            "Place the file at: raw/huri/huri.tsv"
        )

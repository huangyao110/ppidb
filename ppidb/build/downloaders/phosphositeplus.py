"""PhosphoSitePlus downloader — https://www.phosphosite.org
Downloads the Kinase-Substrate dataset (academic use, free).
Note: PhosphoSitePlus requires accepting a license; the kinase-substrate
file is freely downloadable without login for academic use.
"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

# Kinase-Substrate dataset (contains kinase→substrate PPI relationships)
URL_KS = "https://www.phosphosite.org/downloads/Kinase_Substrate_Dataset.gz"
# Regulatory sites (phosphorylation events with upstream kinase)
URL_REG = "https://www.phosphosite.org/downloads/Regulatory_sites.gz"


class PhosphoSitePlusDownloader(BaseDownloader):
    DB_NAME = "PhosphoSitePlus"

    def download(self) -> DownloadResult:
        gz_ks  = self.raw_dir / "phosphosite_kinase_substrate.gz"
        out_ks = self.raw_dir / "phosphosite_kinase_substrate.txt"
        if self._already_downloaded(out_ks):
            return self._ok(out_ks, url=URL_KS)
        try:
            if not self._already_downloaded(gz_ks):
                self._fetch(URL_KS, gz_ks, desc="PhosphoSitePlus KS dataset")
            self._extract_gz(gz_ks, out_ks)
            return self._ok(out_ks, url=URL_KS)
        except Exception as e:
            return self._fail(str(e), url=URL_KS)

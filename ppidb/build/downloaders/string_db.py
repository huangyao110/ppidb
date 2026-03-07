"""STRING downloader — https://stringdb-downloads.org
Downloads human protein links (full detail) v12.0.
For all species, change TAXON to None and use the all-organisms file.
"""
from __future__ import annotations
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

# Human-only full detail file (includes experimental/coexpression/etc scores)
TAXON = "9606"
VERSION = "12.0"
URL_FULL = (
    f"https://stringdb-downloads.org/download/"
    f"protein.links.full.v{VERSION}/{TAXON}.protein.links.full.v{VERSION}.txt.gz"
)
# Also download protein info for ID mapping (STRING ID → gene name)
URL_INFO = (
    f"https://stringdb-downloads.org/download/"
    f"protein.info.v{VERSION}/{TAXON}.protein.info.v{VERSION}.txt.gz"
)
# UniProt ID mapping
URL_ALIASES = (
    f"https://stringdb-downloads.org/download/"
    f"protein.aliases.v{VERSION}/{TAXON}.protein.aliases.v{VERSION}.txt.gz"
)


class STRINGDownloader(BaseDownloader):
    DB_NAME = "STRING"

    def download(self) -> DownloadResult:
        gz_links = self.raw_dir / f"string_{TAXON}_links_full.txt.gz"
        gz_info  = self.raw_dir / f"string_{TAXON}_info.txt.gz"
        gz_alias = self.raw_dir / f"string_{TAXON}_aliases.txt.gz"

        try:
            if not self._already_downloaded(gz_links):
                self._fetch(URL_FULL, gz_links, desc="STRING links full")
            if not self._already_downloaded(gz_info):
                self._fetch(URL_INFO, gz_info, desc="STRING protein info")
            if not self._already_downloaded(gz_alias):
                self._fetch(URL_ALIASES, gz_alias, desc="STRING aliases")

            # Decompress links file (large — keep gz for info/aliases)
            links_txt = self.raw_dir / f"string_{TAXON}_links_full.txt"
            if not links_txt.exists():
                self._extract_gz(gz_links, links_txt)

            return self._ok(links_txt, url=URL_FULL)
        except Exception as e:
            return self._fail(str(e), url=URL_FULL)

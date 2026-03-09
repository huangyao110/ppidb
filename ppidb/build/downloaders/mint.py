"""MINT downloader — https://mint.bio.uniroma2.it/index.php/download/
MINT serves data via IntAct's FTP (they share infrastructure as IMEx members).
We scrape the download page to find the current MITAB link, then fall back
to the IntAct micluster file which contains MINT-curated interactions.
"""
from __future__ import annotations
import re
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

# MINT data is distributed via IntAct FTP as part of the IMEx consortium
# The micluster file is a non-redundant merge of all IMEx members incl. MINT
FALLBACK_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/"
    "intact-micluster.zip"
)
MINT_DOWNLOAD_PAGE = "https://mint.bio.uniroma2.it/index.php/download/"


class MINTDownloader(BaseDownloader):
    DB_NAME = "MINT"

    def download(self) -> DownloadResult:
        zip_path = self.raw_dir / "mint_micluster.zip"
        if self._already_downloaded(zip_path):
            return self._ok(zip_path, url=FALLBACK_URL)
        try:
            # Try to scrape the MINT download page for a direct MITAB link
            url = self._find_direct_url() or FALLBACK_URL
            self._fetch(url, zip_path, desc="MINT MITAB")
            self._extract_zip(zip_path, self.raw_dir)
            # Find extracted file
            txts = list(self.raw_dir.glob("intact-micluster.txt")) + \
                   list(self.raw_dir.glob("mint*.txt"))
            out = txts[0] if txts else zip_path
            return self._ok(out, url=url)
        except Exception as e:
            return self._fail(str(e), url=FALLBACK_URL)

    def _find_direct_url(self) -> str | None:
        """Scrape MINT download page for a .txt or .gz MITAB link."""
        try:
            r = self.session.get(MINT_DOWNLOAD_PAGE, timeout=15)
            r.raise_for_status()
            # Look for href pointing to a MITAB file
            matches = re.findall(
                r'href=["\']([^"\']*(?:MiTab|mitab|MITAB)[^"\']*\.(?:txt|gz|zip))["\']',
                r.text, re.IGNORECASE
            )
            if matches:
                url = matches[0]
                if not url.startswith("http"):
                    url = "https://mint.bio.uniroma2.it" + url
                return url
        except Exception:
            pass
        return None

"""HINT downloader — https://hint.yulab.org
HINT (High-quality INTeractomes) v2024-6.
The download page lists files dynamically; we construct the URL pattern
from the known directory structure confirmed at hint.yulab.org/download.
"""
from __future__ import annotations
import re
from pathlib import Path
from ._base import BaseDownloader, DownloadResult

DOWNLOAD_PAGE = "https://hint.yulab.org/download"
# URL pattern confirmed from hint.yulab.org/download page structure
# Files are served as: /download/{Species}/{type}/{quality}/
BASE_URL = "https://hint.yulab.org/download/HomoSapiens/binary/hq/"


class HINTDownloader(BaseDownloader):
    DB_NAME = "HINT"

    def download(self) -> DownloadResult:
        out = self.raw_dir / "hint_human_binary_hq.txt"
        if self._already_downloaded(out):
            return self._ok(out, url=BASE_URL)
        try:
            # Scrape download page to find actual file link
            url = self._find_file_url() or BASE_URL
            self._fetch(url, out, desc="HINT human binary HQ")
            return self._ok(out, url=url)
        except Exception as e:
            return self._fail(str(e), url=BASE_URL)

    def _find_file_url(self) -> str | None:
        """Scrape HINT download page for the human binary HQ file URL."""
        try:
            r = self.session.get(DOWNLOAD_PAGE, timeout=15)
            r.raise_for_status()
            # Look for download links matching HomoSapiens binary hq
            matches = re.findall(
                r'href=["\']([^"\']*HomoSapiens[^"\']*binary[^"\']*hq[^"\']*)["\']',
                r.text, re.IGNORECASE
            )
            if matches:
                url = matches[0]
                if not url.startswith("http"):
                    url = "https://hint.yulab.org" + url
                return url
            # Fallback: look for any .txt download link on the page
            matches2 = re.findall(
                r'href=["\']([^"\']*\.txt)["\']', r.text
            )
            if matches2:
                url = matches2[0]
                if not url.startswith("http"):
                    url = "https://hint.yulab.org" + url
                return url
        except Exception:
            pass
        return None

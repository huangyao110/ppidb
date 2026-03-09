"""
Base downloader class with retry, resume, progress display.
All database-specific downloaders inherit from BaseDownloader.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ppidb-crawler/1.0; "
        "+https://github.com/ppidb; research use only)"
    )
}
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0   # seconds, doubles each retry
CHUNK_SIZE = 1 << 20  # 1 MB


class DownloadResult:
    """Result of a single database download attempt."""

    def __init__(
        self,
        db_name: str,
        success: bool,
        output_path: Optional[Path] = None,
        n_rows: Optional[int] = None,
        skip_reason: Optional[str] = None,
        error: Optional[str] = None,
        url: Optional[str] = None,
        file_size_mb: Optional[float] = None,
    ):
        self.db_name = db_name
        self.success = success
        self.output_path = output_path
        self.n_rows = n_rows
        self.skip_reason = skip_reason   # set when skipped (not an error)
        self.error = error               # set on failure
        self.url = url
        self.file_size_mb = file_size_mb

    @property
    def skipped(self) -> bool:
        return self.skip_reason is not None

    def __repr__(self) -> str:
        if self.success:
            return (
                f"DownloadResult({self.db_name}: OK, "
                f"rows={self.n_rows}, {self.file_size_mb:.1f} MB)"
            )
        elif self.skipped:
            return f"DownloadResult({self.db_name}: SKIPPED — {self.skip_reason})"
        else:
            return f"DownloadResult({self.db_name}: FAILED — {self.error})"


class BaseDownloader:
    """
    Base class for all PPI database downloaders.

    Subclasses must implement:
        download(raw_dir) -> DownloadResult

    Parameters
    ----------
    raw_dir : Path
        Directory where raw downloaded files are stored.
    force : bool
        Re-download even if file already exists.
    timeout : int
        HTTP request timeout in seconds.
    """

    #: Override in subclass
    DB_NAME: str = "unknown"

    def __init__(
        self,
        raw_dir: Path,
        force: bool = False,
        timeout: int = 120,
    ):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ── Public interface ──────────────────────────────────────────────────────

    def download(self) -> DownloadResult:
        """Download and return DownloadResult. Override in subclass."""
        raise NotImplementedError

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fetch(
        self,
        url: str,
        dest: Path,
        desc: str = "",
    ) -> Path:
        """
        Download url → dest with resume support, retry, and progress logging.
        Returns dest path.
        Raises requests.HTTPError on non-200 after retries.
        """
        # Resume: check existing partial file
        existing_size = dest.stat().st_size if dest.exists() else 0

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                headers = dict(HEADERS)
                if existing_size > 0:
                    headers["Range"] = f"bytes={existing_size}-"

                r = self.session.get(
                    url, headers=headers, stream=True,
                    timeout=self.timeout, allow_redirects=True,
                )

                # 416 = range not satisfiable → file already complete
                if r.status_code == 416:
                    logger.info(f"[{self.DB_NAME}] {desc} already complete (416)")
                    return dest

                r.raise_for_status()

                total = int(r.headers.get("Content-Length", 0)) + existing_size
                mode = "ab" if existing_size > 0 and r.status_code == 206 else "wb"
                if mode == "wb":
                    existing_size = 0

                downloaded = existing_size
                with open(dest, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                size_mb = dest.stat().st_size / 1e6
                logger.info(
                    f"[{self.DB_NAME}] {desc} → {dest.name} "
                    f"({size_mb:.1f} MB)"
                )
                return dest

            except (requests.RequestException, OSError) as e:
                logger.warning(
                    f"[{self.DB_NAME}] attempt {attempt}/{MAX_RETRIES} failed: {e}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * (2 ** (attempt - 1)))
                else:
                    raise

    def _already_downloaded(self, path: Path) -> bool:
        """Return True if file exists and force=False."""
        if path.exists() and path.stat().st_size > 0 and not self.force:
            logger.info(f"[{self.DB_NAME}] {path.name} already exists, skipping download")
            return True
        return False

    def _extract_gz(self, gz_path: Path, out_path: Optional[Path] = None) -> Path:
        """Decompress .gz file. Returns path to decompressed file."""
        if out_path is None:
            out_path = gz_path.with_suffix("")
        with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        logger.info(f"[{self.DB_NAME}] Extracted {gz_path.name} → {out_path.name}")
        return out_path

    def _extract_zip(self, zip_path: Path, out_dir: Optional[Path] = None) -> Path:
        """Extract first file from zip. Returns path to extracted file."""
        if out_dir is None:
            out_dir = zip_path.parent
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            logger.info(f"[{self.DB_NAME}] ZIP contains: {names}")
            zf.extractall(out_dir)
        return out_dir

    def _ok(self, path: Path, n_rows: int = 0, url: str = "") -> DownloadResult:
        size_mb = path.stat().st_size / 1e6 if path.exists() else 0.0
        return DownloadResult(
            db_name=self.DB_NAME,
            success=True,
            output_path=path,
            n_rows=n_rows,
            url=url,
            file_size_mb=size_mb,
        )

    def _skip(self, reason: str) -> DownloadResult:
        return DownloadResult(
            db_name=self.DB_NAME,
            success=False,
            skip_reason=reason,
        )

    def _fail(self, error: str, url: str = "") -> DownloadResult:
        return DownloadResult(
            db_name=self.DB_NAME,
            success=False,
            error=error,
            url=url,
        )

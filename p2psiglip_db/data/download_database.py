"""Download and extract the published database archive into ``data/``."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from p2psiglip_db.data.validate_merged_contract import main as validate_merged_main


REPO = Path(__file__).resolve().parents[2]
URL_ENV = "PPIDB_DATA_URL"
SHA_ENV = "PPIDB_DATA_SHA256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get(URL_ENV),
        help=f"Archive URL. Supports https://... and public gs://bucket/object. Defaults to ${URL_ENV}.",
    )
    parser.add_argument(
        "--sha256",
        default=os.environ.get(SHA_ENV),
        help=f"Optional archive SHA256. Defaults to ${SHA_ENV}.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Use an already-downloaded archive instead of fetching --url.",
    )
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--cache-dir", type=Path, default=REPO / ".cache" / "ppidb")
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted files.")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the downloaded archive in --cache-dir.")
    parser.add_argument("--no-validate", action="store_true", help="Skip validate-merged after extraction.")
    return parser.parse_args()


def normalize_url(url: str) -> str:
    if url.startswith("gs://"):
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc or not parsed.path:
            raise SystemExit(f"invalid GCS URL: {url}")
        bucket = parsed.netloc
        object_name = parsed.path.lstrip("/")
        return f"https://storage.googleapis.com/{bucket}/{urllib.parse.quote(object_name)}"
    return url


def archive_name(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    return name or "ppidb-data-archive"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, cache_dir: Path) -> Path:
    resolved_url = normalize_url(url)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / archive_name(resolved_url)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    print(f"[download] {url}")
    print(f"[download] -> {dest}")
    with urllib.request.urlopen(resolved_url, timeout=60) as response, tmp.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last_report = time.time()
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
            done += len(block)
            now = time.time()
            if now - last_report >= 5:
                if total:
                    print(f"[download] {done / 1024**2:.1f} / {total / 1024**2:.1f} MiB")
                else:
                    print(f"[download] {done / 1024**2:.1f} MiB")
                last_report = now
    tmp.replace(dest)
    return dest


def safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise SystemExit(f"unsafe archive path: {member_name}")
    return target


def member_names(archive: Path) -> list[str]:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            return [info.filename for info in zf.infolist() if not info.is_dir()]
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            return [member.name for member in tf.getmembers() if member.isfile()]
    raise SystemExit(f"unsupported archive type: {archive}")


def extraction_root(data_root: Path, names: list[str]) -> Path:
    first_parts = {Path(name).parts[0] for name in names if Path(name).parts}
    if first_parts == {"data"}:
        return data_root.parent
    return data_root


def copy_from_staging(staging_root: Path, output_root: Path, force: bool) -> int:
    copied = 0
    for src in sorted(path for path in staging_root.rglob("*") if path.is_file()):
        rel = src.relative_to(staging_root)
        dest = output_root / rel
        if dest.exists() and not force:
            raise SystemExit(f"{dest} already exists; rerun with --force to overwrite")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    return copied


def extract_archive(archive: Path, data_root: Path, force: bool) -> int:
    names = member_names(archive)
    if not names:
        raise SystemExit(f"archive contains no files: {archive}")
    final_root = extraction_root(data_root.resolve(), names)

    with tempfile.TemporaryDirectory(prefix="ppidb_extract_") as tmp_dir:
        staging = Path(tmp_dir)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    target = safe_target(staging, info.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        else:
            with tarfile.open(archive) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    target = safe_target(staging, member.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    with extracted, target.open("wb") as dst:
                        shutil.copyfileobj(extracted, dst)

        copied = copy_from_staging(staging, final_root, force)
    print(f"[extract] copied {copied:,} files into {final_root}")
    return copied


def run_validate_merged(data_root: Path) -> None:
    import sys

    old_argv = sys.argv[:]
    sys.argv = [
        "ppidb validate-merged",
        "--merged-root",
        str(data_root / "merged"),
    ]
    try:
        validate_merged_main()
    finally:
        sys.argv = old_argv


def main() -> None:
    args = parse_args()
    if args.archive is None and not args.url:
        raise SystemExit(f"provide --url or set ${URL_ENV}")

    archive = args.archive.resolve() if args.archive else download(args.url, args.cache_dir)
    if not archive.exists():
        raise SystemExit(f"archive does not exist: {archive}")

    if args.sha256:
        actual = sha256_file(archive)
        if actual.lower() != args.sha256.lower():
            raise SystemExit(f"archive SHA256 mismatch: expected {args.sha256}, got {actual}")
        print(f"[sha256] ok {actual}")

    extract_archive(archive, args.data_root, args.force)
    if not args.no_validate:
        run_validate_merged(args.data_root)

    if args.archive is None and not args.keep_archive:
        archive.unlink(missing_ok=True)
    print("[done] data archive installed")


if __name__ == "__main__":
    main()

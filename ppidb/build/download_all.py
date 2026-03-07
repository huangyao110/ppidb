"""
Master download orchestrator for all PPI databases.

Usage
-----
    # Download everything
    python -m ppidb.build.download_all --output-dir ./raw_data

    # Download specific databases only
    python -m ppidb.build.download_all --output-dir ./raw_data \
        --databases BioGRID IntAct STRING

    # Force re-download even if files exist
    python -m ppidb.build.download_all --output-dir ./raw_data --force

    # Dry-run: show what would be downloaded
    python -m ppidb.build.download_all --output-dir ./raw_data --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

from .downloaders import (
    BioGRIDDownloader,
    IntActDownloader,
    STRINGDownloader,
    MINTDownloader,
    HIPPIEDownloader,
    ReactomeDownloader,
    SIGNORDownloader,
    BioPlexDownloader,
    InnateDBDownloader,
    PhosphoSitePlusDownloader,
    OmniPathDownloader,
    ComplexPortalDownloader,
    HINTDownloader,
    CORUMDownloader,
    NegatomeDownloader,
    VirHostNetDownloader,
    HuRIDownloader,
    DIPDownloader,
)
from .downloaders._base import BaseDownloader, DownloadResult

logger = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────
# Ordered by: reliability (most stable first), then size (small first)

DOWNLOADER_REGISTRY: list[tuple[str, type[BaseDownloader], str]] = [
    # (name, class, tier)
    # Tier 1: Stable FTP/HTTP, no auth required
    ("ComplexPortal",    ComplexPortalDownloader,    "tier1"),
    ("Reactome",         ReactomeDownloader,          "tier1"),
    ("SIGNOR",           SIGNORDownloader,            "tier1"),
    ("OmniPath",         OmniPathDownloader,          "tier1"),
    ("HIPPIE",           HIPPIEDownloader,            "tier1"),
    ("BioPlex",          BioPlexDownloader,           "tier1"),
    ("BioGRID",          BioGRIDDownloader,           "tier1"),
    ("IntAct",           IntActDownloader,            "tier1"),
    ("STRING",           STRINGDownloader,            "tier1"),
    ("InnateDB",         InnateDBDownloader,          "tier1"),
    ("PhosphoSitePlus",  PhosphoSitePlusDownloader,   "tier1"),
    # Tier 2: Stable but larger or slower
    ("MINT",             MINTDownloader,              "tier2"),
    ("HINT",             HINTDownloader,              "tier2"),
    # Tier 3: Unstable servers or require scraping
    ("CORUM",            CORUMDownloader,             "tier3"),
    ("Negatome",         NegatomeDownloader,          "tier3"),
    ("VirHostNet",       VirHostNetDownloader,        "tier3"),
    # Tier 4: Require registration / manual download
    ("HuRI",             HuRIDownloader,              "tier4"),
    ("DIP",              DIPDownloader,               "tier4"),
]

ALL_DB_NAMES = [name for name, _, _ in DOWNLOADER_REGISTRY]


# ── Core runner ───────────────────────────────────────────────────────────────

def run_downloads(
    output_dir: Path,
    databases: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    delay: float = 1.0,
) -> list[DownloadResult]:
    """
    Download all (or selected) PPI databases.

    Parameters
    ----------
    output_dir : Path
        Root directory for raw files. Each DB gets a subdirectory.
    databases : list[str] | None
        Subset of database names to download. None = all.
    force : bool
        Re-download even if files already exist.
    dry_run : bool
        Print what would be downloaded without actually downloading.
    delay : float
        Seconds to wait between downloads (be polite to servers).

    Returns
    -------
    list[DownloadResult]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter registry to requested databases
    registry = DOWNLOADER_REGISTRY
    if databases:
        names_lower = {d.lower() for d in databases}
        registry = [
            (name, cls, tier)
            for name, cls, tier in DOWNLOADER_REGISTRY
            if name.lower() in names_lower
        ]
        unknown = names_lower - {name.lower() for name, _, _ in registry}
        if unknown:
            logger.warning(f"Unknown databases (will skip): {', '.join(sorted(unknown))}")
            logger.warning(f"Available: {', '.join(ALL_DB_NAMES)}")

    if dry_run:
        logger.info("DRY RUN — no files will be downloaded")
        for name, cls, tier in registry:
            logger.info(f"  [{tier}] {name}")
        return []

    results: list[DownloadResult] = []
    n = len(registry)

    for i, (name, cls, tier) in enumerate(registry, 1):
        logger.info(f"\n[{i}/{n}] {name} ({tier})")
        logger.info("─" * 50)

        # Each DB gets its own subdirectory
        db_dir = output_dir / name.lower().replace(" ", "_")
        db_dir.mkdir(parents=True, exist_ok=True)

        downloader = cls(raw_dir=db_dir, force=force)
        result = downloader.download()
        results.append(result)

        if result.success:
            size_mb = result.file_size_mb
            logger.info(f"  OK  {result.path.name}  ({size_mb:.1f} MB)")
        elif result.skipped:
            logger.warning(f"  SKIP  {result.message}")
        else:
            logger.error(f"  FAIL  {result.message}")

        # Polite delay between requests (skip for last item)
        if i < n and delay > 0:
            time.sleep(delay)

    _log_summary(results)
    return results


# ── Summary & report ──────────────────────────────────────────────────────────

def _log_summary(results: list[DownloadResult]) -> None:
    success = [r for r in results if r.success]
    skipped = [r for r in results if r.skipped]
    failed  = [r for r in results if not r.success and not r.skipped]

    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total    : {len(results)}")
    logger.info(f"  Success  : {len(success)}")
    logger.info(f"  Skipped  : {len(skipped)}")
    logger.info(f"  Failed   : {len(failed)}")

    if failed:
        logger.info("\n  Failed databases:")
        for r in failed:
            logger.info(f"    - {r.db_name}: {r.message}")

    if skipped:
        logger.info("\n  Skipped databases (manual download required):")
        for r in skipped:
            logger.info(f"    - {r.db_name}: {r.message[:120]}")

    logger.info("=" * 60)


def write_download_report(
    results: list[DownloadResult],
    output_path: Path,
) -> None:
    """Write a markdown download report."""
    success = [r for r in results if r.success]
    skipped = [r for r in results if r.skipped]
    failed  = [r for r in results if not r.success and not r.skipped]

    lines = [
        "# PPI Database Download Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"| Status  | Count |",
        f"|---------|-------|",
        f"| Success | {len(success)} |",
        f"| Skipped | {len(skipped)} |",
        f"| Failed  | {len(failed)} |",
        f"| **Total** | **{len(results)}** |",
        "",
    ]

    if success:
        lines += [
            "## Successfully Downloaded",
            "",
            "| Database | File | Size (MB) | URL |",
            "|----------|------|-----------|-----|",
        ]
        for r in sorted(success, key=lambda x: x.db_name):
            fname = r.path.name if r.path else "—"
            size  = f"{r.file_size_mb:.1f}" if r.path and r.path.exists() else "—"
            url   = r.url or "—"
            lines.append(f"| {r.db_name} | `{fname}` | {size} | {url} |")
        lines.append("")

    if skipped:
        lines += [
            "## Skipped (Manual Download Required)",
            "",
            "| Database | Reason |",
            "|----------|--------|",
        ]
        for r in sorted(skipped, key=lambda x: x.db_name):
            msg = r.message.replace("|", "\\|")[:200]
            lines.append(f"| {r.db_name} | {msg} |")
        lines.append("")

    if failed:
        lines += [
            "## Failed",
            "",
            "| Database | Error |",
            "|----------|-------|",
        ]
        for r in sorted(failed, key=lambda x: x.db_name):
            msg = r.message.replace("|", "\\|")[:200]
            lines.append(f"| {r.db_name} | {msg} |")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Download report written to: {output_path}")


def write_download_manifest(
    results: list[DownloadResult],
    output_path: Path,
) -> None:
    """Write a JSON manifest of all downloaded files (for the integration step)."""
    manifest = {
        "generated": datetime.now().isoformat(),
        "databases": {},
    }
    for r in results:
        manifest["databases"][r.db_name] = {
            "success": r.success,
            "skipped": r.skipped,
            "path": str(r.path) if r.path else None,
            "url": r.url,
            "message": r.message,
            "size_mb": round(r.file_size_mb, 2) if r.path and r.path.exists() else None,
        }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Download manifest written to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Download all PPI databases for ppidb.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available databases:\n  " + "\n  ".join(ALL_DB_NAMES),
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, type=Path,
        help="Root directory for raw downloaded files",
    )
    parser.add_argument(
        "--databases", "-d", nargs="+", metavar="DB",
        help="Specific databases to download (default: all)",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between downloads (default: 1.0)",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Write markdown download report to this path",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Write JSON manifest to this path",
    )
    args = parser.parse_args()

    results = run_downloads(
        output_dir=args.output_dir,
        databases=args.databases,
        force=args.force,
        dry_run=args.dry_run,
        delay=args.delay,
    )

    if results:
        report_path = args.report or (args.output_dir / "download_report.md")
        manifest_path = args.manifest or (args.output_dir / "download_manifest.json")
        write_download_report(results, report_path)
        write_download_manifest(results, manifest_path)

    # Exit with error code if any downloads failed
    failed = [r for r in results if not r.success and not r.skipped]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

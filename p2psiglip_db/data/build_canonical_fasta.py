"""Pick the longest isoform per gene and write a single-record-per-gene FASTA.

The genome-scan top-K TSVs identify proteins by their *gene* IDs (AT5G16970,
Solyc10g055010), but Ensembl-style FASTAs are per-transcript with versioned
gene/transcript IDs (AT5G16970.1, Solyc12g009745.1.1). For BBH ortholog
mapping we want exactly one representative sequence per gene.

Usage:
    python p2psiglip_db/data/build_canonical_fasta.py <input.fa> <output.fa>
        [--gene-id-strip-version]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


GENE_RE = re.compile(r"gene:(\S+)")


def stream_fasta(path: Path):
    cur_id = None
    cur_header = None
    cur_seq = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur_id is not None:
                    yield cur_id, cur_header, "".join(cur_seq)
                cur_header = line[1:].rstrip()
                cur_id = cur_header.split()[0]
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id is not None:
            yield cur_id, cur_header, "".join(cur_seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_fa", type=Path)
    ap.add_argument("output_fa", type=Path)
    ap.add_argument(
        "--gene-id-strip-version",
        action="store_true",
        help="Strip a trailing .N from the gene ID (Ensembl Plants Solyc has e.g. Solyc12g009745.1).",
    )
    args = ap.parse_args()

    # gene_id -> (length, transcript_id, sequence)
    best: dict[str, tuple[int, str, str]] = {}
    n_seen = 0
    for tid, header, seq in stream_fasta(args.input_fa):
        m = GENE_RE.search(header)
        if not m:
            continue
        gid = m.group(1)
        if args.gene_id_strip_version:
            gid = re.sub(r"\.\d+$", "", gid)
        n_seen += 1
        if gid not in best or len(seq) > best[gid][0]:
            best[gid] = (len(seq), tid, seq)

    args.output_fa.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_fa, "w") as out:
        for gid in sorted(best):
            length, tid, seq = best[gid]
            out.write(f">{gid} transcript:{tid} length:{length}\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i : i + 60] + "\n")

    print(
        f"{args.input_fa.name}: {n_seen} transcripts -> {len(best)} unique genes "
        f"(written to {args.output_fa})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

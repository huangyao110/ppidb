"""
Per-PLM embedding extractor dispatcher.

Forwards all flags after --plm to the chosen submodule under p2psiglip_db.embeds,
which is invoked as `python -m p2psiglip_db.embeds.<plm>`. The submodule's own
argparse handles the rest. Subprocess invocation (rather than in-process call)
preserves the multiprocessing.spawn semantics each PLM extractor depends on
for one-process-per-GPU parallelism.

Usage:
    python p2psiglip_db/data/get_embeddings.py --plm esmc        -i seqs.csv  -o embed/esmc/ --pool mean
    python p2psiglip_db/data/get_embeddings.py --plm prostt5_3di -i 3di.fasta -o embed/prostt5_3di/
    python p2psiglip_db/data/get_embeddings.py --plm saprot      -i aa.csv --tdi-fasta 3di.fasta -o embed/saprot/
    python p2psiglip_db/data/get_embeddings.py --plm profam      -i seqs.csv  -o embed/profam/
    python p2psiglip_db/data/get_embeddings.py --list            # show available PLMs

For per-PLM flag help:
    python p2psiglip_db/data/get_embeddings.py --plm <name> --help

Most extractors support --pool {mean,max,cls,residue}. Pooled outputs are
(D,) fp32 arrays; residue outputs are (L,D) fp16 arrays. ProtT5 does not expose
a CLS/BOS token, so use mean, max, or residue there.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from p2psiglip_db.embeds import PLMS

PLM_MODULES = {
    "prosst_2048": "prosst",
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    p.add_argument('-p', '--plm', choices=PLMS,
                   help='Which PLM extractor to run')
    p.add_argument('--list', action='store_true',
                   help='List available PLMs and exit')
    p.add_argument('-h', '--help', action='store_true',
                   help='Show this help (or --plm <name> --help for per-PLM flags)')
    args, remaining = p.parse_known_args()

    if args.list:
        for name in PLMS:
            print(name)
        return 0
    if args.help and not args.plm:
        p.print_help()
        return 0
    if not args.plm:
        p.error('--plm is required (or --list / --help)')

    module_name = PLM_MODULES.get(args.plm, args.plm)
    cmd = [sys.executable, '-m', f'p2psiglip_db.embeds.{module_name}', *remaining]
    if args.help:
        cmd.append('--help')
    return subprocess.run(cmd).returncode


if __name__ == '__main__':
    sys.exit(main())

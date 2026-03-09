"""
ppidb CLI — command-line interface for the ppidb toolkit.

Usage:
    ppidb query   --db ppi_v6.parquet --species 9606 --throughput LTP --min-sources 2
    ppidb split   --db ppi_v6.parquet --strategy cold --test 0.1 --val 0.1 --output ./split/
    ppidb negatives --db ppi_v6.parquet --strategy random --ratio 1.0 --output negatives.parquet
    ppidb fetch-sequences --db ppi_v6.parquet --output proteins.fasta
    ppidb describe --db ppi_v6.parquet
"""

import sys
import click

from ppidb.core.dataset import PPIDataset


# ── Main group ────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="ppidb")
def cli():
    """
    ppidb — toolkit for working with large-scale PPI databases.

    \b
    Quick start:
        ppidb describe --db ppi_v6.parquet
        ppidb query --db ppi_v6.parquet --species 9606 --throughput LTP --output human_ltp.parquet
        ppidb split --db human_ltp.parquet --strategy cold --output ./split/
    """
    pass


# ── describe ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--db", required=True, type=click.Path(exists=True),
              help="Path to PPI database (.parquet or .csv)")
def describe(db):
    """Print summary statistics of the database."""
    ds = PPIDataset.load(db)
    ds.describe()


# ── query ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--db", required=True, type=click.Path(exists=True),
              help="Path to PPI database (.parquet or .csv)")
@click.option("--species", "-s", multiple=True, default=None,
              help="NCBI taxon ID(s). E.g. --species 9606 --species 10090")
@click.option("--throughput", "-t", default=None,
              type=click.Choice(["LTP", "HTP", "both", "no_exp", "experimental", "ltp_validated"],
                                case_sensitive=False),
              help="Throughput type filter.")
@click.option("--min-sources", "-m", default=None, type=int,
              help="Minimum number of supporting databases.")
@click.option("--database", "-d", multiple=True, default=None,
              help="Filter by source database name(s). E.g. --database BioGRID")
@click.option("--interaction-type", default=None,
              type=click.Choice(["positive", "negative"]),
              help="Filter by interaction type.")
@click.option("--proteins", default=None, type=click.Path(exists=True),
              help="Path to a text file with UniProt IDs (one per line) to filter by.")
@click.option("--output", "-o", default=None,
              help="Output file path. Default: print stats only.")
@click.option("--format", "fmt", default="parquet",
              type=click.Choice(["parquet", "csv", "tsv"]),
              help="Output format (default: parquet).")
@click.option("--preset", default=None,
              type=click.Choice(["high_confidence"]),
              help="Use a preset filter combination.")
def query(db, species, throughput, min_sources, database, interaction_type,
          proteins, output, fmt, preset):
    """
    Query and filter the PPI database.

    \b
    Examples:
        # Human LTP pairs with ≥2 sources
        ppidb query --db ppi_v6.parquet --species 9606 --throughput LTP --min-sources 2

        # High-confidence preset
        ppidb query --db ppi_v6.parquet --preset high_confidence --output hc.parquet

        # Pairs involving specific proteins
        ppidb query --db ppi_v6.parquet --proteins my_genes.txt --output subset.parquet
    """
    click.echo(f"Loading {db}...")
    ds = PPIDataset.load(db)
    click.echo(f"Loaded: {ds}")

    # Apply preset
    if preset == "high_confidence":
        ds = ds.filter.high_confidence(min_sources=min_sources or 2)
        click.echo(f"Applied preset 'high_confidence': {ds}")
    else:
        # Apply individual filters
        if species:
            ds = ds.filter.by_species(list(species))
            click.echo(f"After species filter: {ds}")

        if throughput:
            ds = ds.filter.by_throughput(throughput)
            click.echo(f"After throughput filter: {ds}")

        if min_sources is not None:
            ds = ds.filter.by_min_sources(min_sources)
            click.echo(f"After min-sources filter: {ds}")

        if database:
            ds = ds.filter.by_database(list(database))
            click.echo(f"After database filter: {ds}")

        if interaction_type:
            if interaction_type == "positive":
                ds = ds.filter.positives_only()
            else:
                ds = ds.filter.negatives_only()
            click.echo(f"After interaction-type filter: {ds}")

        if proteins:
            with open(proteins) as f:
                protein_list = [line.strip() for line in f if line.strip()]
            ds = ds.filter.by_proteins(protein_list)
            click.echo(f"After protein filter: {ds}")

    if output:
        ds.save(output, format=fmt)
    else:
        click.echo(f"\nResult: {ds}")
        click.echo("(Use --output to save results)")


# ── split ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--db", required=True, type=click.Path(exists=True),
              help="Path to PPI database (.parquet or .csv)")
@click.option("--strategy", default="similarity",
              type=click.Choice(["similarity", "greedy_c3", "community_c3"]),
              help="Split strategy (default: similarity).")
@click.option("--train", default=0.8, type=float, help="Train fraction (default: 0.8)")
@click.option("--val", default=0.1, type=float, help="Val fraction (default: 0.1)")
@click.option("--test", default=0.1, type=float, help="Test fraction (default: 0.1)")
@click.option("--seed", default=42, type=int, help="Random seed (default: 42)")
@click.option("--identity-threshold", default=0.3, type=float,
              help="Sequence identity threshold for similarity split (default: 0.3)")
@click.option("--output", "-o", required=True,
              help="Output directory for split files.")
def split(db, strategy, train, val, test, seed, identity_threshold, output):
    """
    Split the PPI database into train/val/test folds using C3 strategies.

    \b
    Examples:
        ppidb split --db human_ltp.parquet --strategy similarity --test 0.1 --output ./split/
        ppidb split --db human_ltp.parquet --strategy greedy_c3 --test 0.1 --output ./split/
    """
    from ppidb.split import Splitter
    from ppidb.sequence import SequenceFetcher

    click.echo(f"Loading {db}...")
    ds = PPIDataset.load(db)
    click.echo(f"Loaded: {ds}")

    seqs = None
    if strategy == "similarity":
        click.echo("Fetching sequences (required for similarity split)...")
        fetcher = SequenceFetcher()
        seqs = fetcher.fetch(ds.proteins(), as_dict=True)
        click.echo(f"Fetched {len(seqs):,} sequences.")

    splitter = Splitter(ds)
    click.echo(f"Performing {strategy} split...")

    if strategy == "similarity":
        result = splitter.similarity_split(
            identity_threshold=identity_threshold,
            test_frac=test,
            val_frac=val,
            seed=seed,
            sequence_dict=seqs
        )
    elif strategy == "greedy_c3":
        result = splitter.greedy_c3_split(
            test_protein_frac=test,
            val_protein_frac=val,
            seed=seed
        )
    elif strategy == "community_c3":
        result = splitter.community_c3_split(
            test_protein_frac=test,
            val_protein_frac=val,
            seed=seed
        )

    result.summary()
    result.save(output)


# ── negatives ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--db", required=True, type=click.Path(exists=True),
              help="Path to POSITIVE PPI database (.parquet or .csv)")
@click.option("--full-db", default=None, type=click.Path(exists=True),
              help="Full database (required for --strategy negatome)")
@click.option("--strategy", default="random",
              type=click.Choice(["random", "negatome"]),
              help="Negative sampling strategy (default: random).")
@click.option("--ratio", default=1.0, type=float,
              help="Negative:positive ratio (default: 1.0)")
@click.option("--seed", default=42, type=int, help="Random seed (default: 42)")
@click.option("--output", "-o", required=True,
              help="Output file for negative pairs.")
@click.option("--format", "fmt", default="parquet",
              type=click.Choice(["parquet", "csv", "tsv"]))
@click.option("--combine", is_flag=True, default=False,
              help="Combine positives and negatives into one output file.")
def negatives(db, full_db, strategy, ratio, seed, output, fmt, combine):
    """
    Generate negative (non-interacting) protein pairs.

    \b
    Examples:
        ppidb negatives --db human_ltp.parquet --strategy random --ratio 1.0 --output neg.parquet
        ppidb negatives --db human_ltp.parquet --strategy negatome --full-db ppi_v6.parquet --output neg.parquet
    """
    from ppidb.negative import NegativeSampler

    click.echo(f"Loading positive dataset: {db}...")
    pos_ds = PPIDataset.load(db)
    click.echo(f"Positives: {pos_ds}")

    sampler = NegativeSampler(pos_ds)

    if strategy == "negatome":
        if not full_db:
            click.echo("ERROR: --full-db is required for negatome strategy.", err=True)
            sys.exit(1)
        full_ds = PPIDataset.load(full_db)
        neg_ds = sampler.from_negatome(full_ds)
    elif strategy == "random":
        neg_ds = sampler.random_sample(ratio=ratio, seed=seed)

    if combine:
        combined = NegativeSampler.combine(pos_ds, neg_ds)
        combined.save(output, format=fmt)
    else:
        neg_ds.save(output, format=fmt)


# ── fetch-sequences ───────────────────────────────────────────────────────────

@cli.command("fetch-sequences")
@click.option("--db", required=True, type=click.Path(exists=True),
              help="Path to PPI database (.parquet or .csv)")
@click.option("--output", "-o", required=True,
              help="Output FASTA file path.")
@click.option("--cache-dir", default=None,
              help="Cache directory for sequences (default: ~/.ppidb/sequence_cache/)")
@click.option("--compartments-output", default=None,
              help="Optional output TSV path for subcellular compartments.")
def fetch_sequences(db, output, cache_dir, compartments_output):
    """
    Fetch protein sequences from UniProt for all proteins in the database.

    \b
    Examples:
        ppidb fetch-sequences --db human_ltp.parquet --output proteins.fasta
        ppidb fetch-sequences --db human_ltp.parquet --output proteins.fasta --compartments-output compartments.tsv
    """
    from ppidb.sequence import SequenceFetcher

    click.echo(f"Loading {db}...")
    ds = PPIDataset.load(db)
    proteins = ds.proteins()
    click.echo(f"Found {len(proteins):,} unique proteins.")

    fetcher = SequenceFetcher(cache_dir=cache_dir)
    fetcher.fetch(proteins, output_fasta=output)
    click.echo(f"Done. Sequences saved to {output}")

    if compartments_output:
        click.echo("Fetching subcellular compartments...")
        compartments = fetcher.fetch_compartments(proteins, verbose=True)
        with open(compartments_output, "w") as f:
            f.write("protein\tcompartment\n")
            for protein in sorted(compartments):
                f.write(f"{protein}\t{compartments[protein]}\n")
        click.echo(f"Done. Compartments saved to {compartments_output}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()

"""
Unit tests for ppidb parsers.
All tests use synthetic in-memory data — no network access required.
"""
from __future__ import annotations

import gzip
import textwrap
from pathlib import Path

import polars as pl
import pytest

from ppidb.build.parsers.mitab import parse_mitab, _extract_uniprot, _extract_taxon
from ppidb.build.parsers.db_parsers import (
    parse_biogrid,
    parse_hippie,
    parse_bioplex,
    parse_corum,
    parse_negatome,
    parse_signor,
    parse_huri,
    parse_omnipath,
    parse_complex_portal,
    parse_phosphositeplus,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def write_tmp(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def write_tmp_gz(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write(textwrap.dedent(content))
    return p


# ── MITAB field extractors ────────────────────────────────────────────────────

class TestExtractUniprot:
    def test_standard_accession(self):
        assert _extract_uniprot("uniprotkb:P12345") == "P12345"

    def test_case_insensitive(self):
        assert _extract_uniprot("UniProtKB:P12345") == "P12345"

    def test_isoform_stripped(self):
        assert _extract_uniprot("uniprotkb:P12345-2") == "P12345"

    def test_multiple_ids_first_wins(self):
        assert _extract_uniprot("uniprotkb:P12345|uniprotkb:Q67890") == "P12345"

    def test_non_uniprot_returns_none(self):
        assert _extract_uniprot("entrez gene:1234") is None

    def test_dash_returns_none(self):
        assert _extract_uniprot("-") is None

    def test_empty_returns_none(self):
        assert _extract_uniprot("") is None


class TestExtractTaxon:
    def test_human(self):
        assert _extract_taxon("taxid:9606(Homo sapiens)") == "9606"

    def test_bare_taxid(self):
        assert _extract_taxon("taxid:10090") == "10090"

    def test_dash_returns_none(self):
        assert _extract_taxon("-") is None

    def test_empty_returns_none(self):
        assert _extract_taxon("") is None


# ── MITAB parser ──────────────────────────────────────────────────────────────

MITAB_HEADER = (
    "ID(s) interactor A\tID(s) interactor B\t"
    "Alt. ID(s) interactor A\tAlt. ID(s) interactor B\t"
    "Alias(es) interactor A\tAlias(es) interactor B\t"
    "Interaction detection method(s)\tPublication 1st author(s)\t"
    "Publication Identifier(s)\tTaxon interactor A\tTaxon interactor B\t"
    "Interaction type(s)\tSource database(s)\tInteraction identifier(s)\t"
    "Confidence value(s)\n"
)

def _mitab_row(uid_a, uid_b, taxon_a="taxid:9606", taxon_b="taxid:9606",
               method='psi-mi:"MI:0018"(two hybrid)'):
    return (
        f"uniprotkb:{uid_a}\tuniprotkb:{uid_b}\t-\t-\t-\t-\t"
        f"{method}\t-\t-\t{taxon_a}\t{taxon_b}\t"
        f'psi-mi:"MI:0915"(physical association)\t-\t-\t-\n'
    )


class TestParseMITAB:
    def test_basic_parse(self, tmp_path):
        content = MITAB_HEADER + _mitab_row("P12345", "Q67890")
        p = write_tmp(tmp_path, "test.txt", content)
        df = parse_mitab(p, "TestDB")
        assert len(df) == 1
        assert set(df["uniprot_a"].to_list()) | set(df["uniprot_b"].to_list()) == {"P12345", "Q67890"}

    def test_canonical_ordering(self, tmp_path):
        # Q > P alphabetically, so after parsing uniprot_a should be P
        content = MITAB_HEADER + _mitab_row("Q67890", "P12345")
        p = write_tmp(tmp_path, "test.txt", content)
        df = parse_mitab(p, "TestDB")
        assert df["uniprot_a"][0] == "P12345"
        assert df["uniprot_b"][0] == "Q67890"

    def test_self_interaction_removed(self, tmp_path):
        content = MITAB_HEADER + _mitab_row("P12345", "P12345")
        p = write_tmp(tmp_path, "test.txt", content)
        df = parse_mitab(p, "TestDB")
        assert len(df) == 0

    def test_taxon_filter(self, tmp_path):
        content = (
            MITAB_HEADER
            + _mitab_row("P12345", "Q67890", taxon_a="taxid:9606", taxon_b="taxid:9606")
            + _mitab_row("P11111", "Q22222", taxon_a="taxid:10090", taxon_b="taxid:10090")
        )
        p = write_tmp(tmp_path, "test.txt", content)
        df = parse_mitab(p, "TestDB", taxon_filter="9606")
        assert len(df) == 1
        assert "P12345" in df["uniprot_a"].to_list() + df["uniprot_b"].to_list()

    def test_non_uniprot_ids_skipped(self, tmp_path):
        content = (
            MITAB_HEADER
            + "entrez gene:1234\tentrez gene:5678\t-\t-\t-\t-\t-\t-\t-\t"
            "taxid:9606\ttaxid:9606\t-\t-\t-\t-\n"
        )
        p = write_tmp(tmp_path, "test.txt", content)
        df = parse_mitab(p, "TestDB")
        assert len(df) == 0

    def test_deduplication(self, tmp_path):
        row = _mitab_row("P12345", "Q67890")
        content = MITAB_HEADER + row + row  # duplicate
        p = write_tmp(tmp_path, "test.txt", content)
        df = parse_mitab(p, "TestDB")
        assert len(df) == 1

    def test_gz_file(self, tmp_path):
        content = MITAB_HEADER + _mitab_row("P12345", "Q67890")
        p = write_tmp_gz(tmp_path, "test.txt.gz", content)
        df = parse_mitab(p, "TestDB")
        assert len(df) == 1

    def test_source_db_label(self, tmp_path):
        content = MITAB_HEADER + _mitab_row("P12345", "Q67890")
        p = write_tmp(tmp_path, "test.txt", content)
        df = parse_mitab(p, "IntAct")
        assert df["source_dbs"][0] == "IntAct"

    def test_empty_file_returns_empty_df(self, tmp_path):
        p = write_tmp(tmp_path, "empty.txt", MITAB_HEADER)
        df = parse_mitab(p, "TestDB")
        assert len(df) == 0
        assert "uniprot_a" in df.columns


# ── BioGRID parser ────────────────────────────────────────────────────────────

def _biogrid_header():
    # Minimal tab3 header with required columns
    cols = ["#BioGRID Interaction ID"] * 25
    cols[7]  = "Entrez Gene Interactor A"
    cols[8]  = "Entrez Gene Interactor B"
    cols[11] = "Experimental System"
    cols[12] = "Experimental System Type"
    cols[15] = "Organism ID Interactor A"
    cols[16] = "Organism ID Interactor B"
    cols[23] = "SWISS-PROT Accessions Interactor A"
    cols[24] = "SWISS-PROT Accessions Interactor B"
    return "\t".join(cols) + "\n"


def _biogrid_row(uid_a, uid_b, taxon="9606", exp_type="physical", exp_sys="Two-hybrid"):
    cols = ["-"] * 25
    cols[7]  = "1"
    cols[8]  = "2"
    cols[11] = exp_sys
    cols[12] = exp_type
    cols[15] = taxon
    cols[16] = taxon
    cols[23] = uid_a
    cols[24] = uid_b
    return "\t".join(cols) + "\n"


class TestParseBioGRID:
    def test_basic(self, tmp_path):
        content = _biogrid_header() + _biogrid_row("P12345", "Q67890")
        p = write_tmp(tmp_path, "biogrid.txt", content)
        df = parse_biogrid(p)
        assert len(df) == 1

    def test_genetic_interactions_excluded(self, tmp_path):
        content = _biogrid_header() + _biogrid_row("P12345", "Q67890", exp_type="genetic")
        p = write_tmp(tmp_path, "biogrid.txt", content)
        df = parse_biogrid(p)
        assert len(df) == 0

    def test_taxon_filter(self, tmp_path):
        content = (
            _biogrid_header()
            + _biogrid_row("P12345", "Q67890", taxon="9606")
            + _biogrid_row("P11111", "Q22222", taxon="10090")
        )
        p = write_tmp(tmp_path, "biogrid.txt", content)
        df = parse_biogrid(p, taxon_filter="9606")
        assert len(df) == 1

    def test_canonical_ordering(self, tmp_path):
        content = _biogrid_header() + _biogrid_row("Q67890", "P12345")
        p = write_tmp(tmp_path, "biogrid.txt", content)
        df = parse_biogrid(p)
        assert df["uniprot_a"][0] == "P12345"
        assert df["uniprot_b"][0] == "Q67890"

    def test_self_loop_removed(self, tmp_path):
        content = _biogrid_header() + _biogrid_row("P12345", "P12345")
        p = write_tmp(tmp_path, "biogrid.txt", content)
        df = parse_biogrid(p)
        assert len(df) == 0


# ── HIPPIE parser ─────────────────────────────────────────────────────────────

class TestParseHIPPIE:
    def test_basic(self, tmp_path):
        content = "P12345\t1\tQ67890\t2\t0.80\tsource1\n"
        p = write_tmp(tmp_path, "hippie.txt", content)
        df = parse_hippie(p, min_score=0.63)
        assert len(df) == 1

    def test_score_filter(self, tmp_path):
        content = (
            "P12345\t1\tQ67890\t2\t0.80\tsource1\n"
            "P11111\t3\tQ22222\t4\t0.40\tsource2\n"
        )
        p = write_tmp(tmp_path, "hippie.txt", content)
        df = parse_hippie(p, min_score=0.63)
        assert len(df) == 1

    def test_header_skipped(self, tmp_path):
        content = "UniProt_A\tEntrez_A\tUniProt_B\tEntrez_B\tscore\tsources\nP12345\t1\tQ67890\t2\t0.80\ts\n"
        p = write_tmp(tmp_path, "hippie.txt", content)
        df = parse_hippie(p, min_score=0.63)
        assert len(df) == 1

    def test_taxon_set_to_human(self, tmp_path):
        content = "P12345\t1\tQ67890\t2\t0.80\tsource1\n"
        p = write_tmp(tmp_path, "hippie.txt", content)
        df = parse_hippie(p)
        assert df["taxon_a"][0] == "9606"


# ── BioPlex parser ────────────────────────────────────────────────────────────

class TestParseBioPlex:
    def test_basic(self, tmp_path):
        content = "GeneA\tGeneB\tUniprotA\tUniprotB\tSymbolA\tSymbolB\tpNI\tpInt\n"
        content += "1\t2\tP12345\tQ67890\tGENEA\tGENEB\t0.01\t0.99\n"
        p = write_tmp(tmp_path, "bioplex.tsv", content)
        df = parse_bioplex(p)
        assert len(df) == 1
        assert df["throughput_type"][0] == "HTP"

    def test_canonical_ordering(self, tmp_path):
        content = "GeneA\tGeneB\tUniprotA\tUniprotB\tSymbolA\tSymbolB\tpNI\tpInt\n"
        content += "1\t2\tQ67890\tP12345\tGENEB\tGENEA\t0.01\t0.99\n"
        p = write_tmp(tmp_path, "bioplex.tsv", content)
        df = parse_bioplex(p)
        assert df["uniprot_a"][0] == "P12345"


# ── CORUM parser ──────────────────────────────────────────────────────────────

class TestParseCORUM:
    def test_pairwise_expansion(self, tmp_path):
        # 3 subunits → 3 pairs
        header = "ComplexID\tComplexName\tOrganism\tSynonyms\tsubunits(UniProt IDs)\tPubMed\n"
        row    = "1\tTestComplex\tHuman\t-\tP12345;Q67890;A11111\t12345678\n"
        p = write_tmp(tmp_path, "allComplexes.txt", header + row)
        df = parse_corum(p, taxon_filter="9606")
        assert len(df) == 3  # C(3,2) = 3 pairs

    def test_non_human_filtered(self, tmp_path):
        header = "ComplexID\tComplexName\tOrganism\tSynonyms\tsubunits(UniProt IDs)\tPubMed\n"
        row    = "1\tTestComplex\tMouse\t-\tP12345;Q67890\t12345678\n"
        p = write_tmp(tmp_path, "allComplexes.txt", header + row)
        df = parse_corum(p, taxon_filter="9606")
        assert len(df) == 0

    def test_single_subunit_no_pairs(self, tmp_path):
        header = "ComplexID\tComplexName\tOrganism\tSynonyms\tsubunits(UniProt IDs)\tPubMed\n"
        row    = "1\tTestComplex\tHuman\t-\tP12345\t12345678\n"
        p = write_tmp(tmp_path, "allComplexes.txt", header + row)
        df = parse_corum(p, taxon_filter="9606")
        assert len(df) == 0

    def test_interaction_type_positive(self, tmp_path):
        header = "ComplexID\tComplexName\tOrganism\tSynonyms\tsubunits(UniProt IDs)\tPubMed\n"
        row    = "1\tTestComplex\tHuman\t-\tP12345;Q67890\t12345678\n"
        p = write_tmp(tmp_path, "allComplexes.txt", header + row)
        df = parse_corum(p, taxon_filter="9606")
        assert df["interaction_type"][0] == "positive"


# ── Negatome parser ───────────────────────────────────────────────────────────

class TestParseNegatome:
    def test_basic(self, tmp_path):
        content = "P12345\tQ67890\n"
        p = write_tmp(tmp_path, "negatome.txt", content)
        df = parse_negatome(p)
        assert len(df) == 1
        assert df["interaction_type"][0] == "negative"

    def test_comment_lines_skipped(self, tmp_path):
        content = "# comment\nP12345\tQ67890\n"
        p = write_tmp(tmp_path, "negatome.txt", content)
        df = parse_negatome(p)
        assert len(df) == 1

    def test_canonical_ordering(self, tmp_path):
        content = "Q67890\tP12345\n"
        p = write_tmp(tmp_path, "negatome.txt", content)
        df = parse_negatome(p)
        assert df["uniprot_a"][0] == "P12345"
        assert df["uniprot_b"][0] == "Q67890"

    def test_throughput_negative_sample(self, tmp_path):
        content = "P12345\tQ67890\n"
        p = write_tmp(tmp_path, "negatome.txt", content)
        df = parse_negatome(p)
        assert df["throughput_type"][0] == "negative_sample"


# ── SIGNOR parser ─────────────────────────────────────────────────────────────

def _signor_header():
    return (
        "ENTITYA\tTYPEA\tIDA\tDATABASEA\t"
        "ENTITYB\tTYPEB\tIDB\tDATABASEB\t"
        "EFFECT\tMECHANISM\tRESIDUE\tSEQUENCE\t"
        "TAX_ID\tCELL_DATA\tTISSUE_DATA\t"
        "MODULATOR_COMPLEX\tTARGET_COMPLEX\t"
        "MODIFICATIONA\tMODASEQA\tMODIFICATIONB\tMODASEQB\t"
        "PMID\tDIRECT\tANNOTATOR\tSENTENCE\tSIGNOR_ID\n"
    )


def _signor_row(uid_a, uid_b, type_a="protein", type_b="protein",
                db_a="uniprot", db_b="uniprot"):
    return (
        f"GENEA\t{type_a}\t{uid_a}\t{db_a}\t"
        f"GENEB\t{type_b}\t{uid_b}\t{db_b}\t"
        f"up-regulates\tphosphorylation\t-\t-\t"
        f"9606\t-\t-\t-\t-\t-\t-\t-\t-\t"
        f"12345678\tYES\tannotator\tsentence\tSIGNOR-1\n"
    )


class TestParseSIGNOR:
    def test_basic(self, tmp_path):
        content = _signor_header() + _signor_row("P12345", "Q67890")
        p = write_tmp(tmp_path, "signor.tsv", content)
        df = parse_signor(p)
        assert len(df) == 1

    def test_non_protein_excluded(self, tmp_path):
        content = _signor_header() + _signor_row("P12345", "Q67890", type_b="chemical")
        p = write_tmp(tmp_path, "signor.tsv", content)
        df = parse_signor(p)
        assert len(df) == 0

    def test_non_uniprot_db_excluded(self, tmp_path):
        content = _signor_header() + _signor_row("P12345", "Q67890", db_b="pubchem")
        p = write_tmp(tmp_path, "signor.tsv", content)
        df = parse_signor(p)
        assert len(df) == 0

    def test_throughput_ltp(self, tmp_path):
        content = _signor_header() + _signor_row("P12345", "Q67890")
        p = write_tmp(tmp_path, "signor.tsv", content)
        df = parse_signor(p)
        assert df["throughput_type"][0] == "LTP"


# ── HuRI parser ───────────────────────────────────────────────────────────────

class TestParseHuRI:
    def test_basic(self, tmp_path):
        content = "P12345\tQ67890\n"
        p = write_tmp(tmp_path, "huri.tsv", content)
        df = parse_huri(p)
        assert len(df) == 1
        assert df["throughput_type"][0] == "HTP"

    def test_self_loop_removed(self, tmp_path):
        content = "P12345\tP12345\n"
        p = write_tmp(tmp_path, "huri.tsv", content)
        df = parse_huri(p)
        assert len(df) == 0

    def test_canonical_ordering(self, tmp_path):
        content = "Q67890\tP12345\n"
        p = write_tmp(tmp_path, "huri.tsv", content)
        df = parse_huri(p)
        assert df["uniprot_a"][0] == "P12345"


# ── OmniPath parser ───────────────────────────────────────────────────────────

class TestParseOmniPath:
    def test_basic(self, tmp_path):
        header = "source\ttarget\tsource_genesymbol\ttarget_genesymbol\t"
        header += "is_directed\tis_stimulation\tis_inhibition\t"
        header += "consensus_direction\tconsensus_stimulation\tconsensus_inhibition\t"
        header += "sources\treferences\tcuration_effort\n"
        row = "P12345\tQ67890\tGENEA\tGENEB\t1\t1\t0\t1\t1\t0\tSignor\t12345\t5\n"
        p = write_tmp(tmp_path, "omnipath.tsv", header + row)
        df = parse_omnipath(p)
        assert len(df) == 1

    def test_self_loop_removed(self, tmp_path):
        header = "source\ttarget\tsource_genesymbol\ttarget_genesymbol\t"
        header += "is_directed\tis_stimulation\tis_inhibition\t"
        header += "consensus_direction\tconsensus_stimulation\tconsensus_inhibition\t"
        header += "sources\treferences\tcuration_effort\n"
        row = "P12345\tP12345\tGENEA\tGENEA\t1\t1\t0\t1\t1\t0\tSignor\t12345\t5\n"
        p = write_tmp(tmp_path, "omnipath.tsv", header + row)
        df = parse_omnipath(p)
        assert len(df) == 0


# ── Complex Portal parser ─────────────────────────────────────────────────────

class TestParseComplexPortal:
    def test_pairwise_expansion(self, tmp_path):
        header = "Complex ac\tRecommended name\tAliases\tTaxonomy identifier\t"
        header += "Identifiers (and stoichiometry) of molecules in complex\t"
        header += "Evidence Code\tExperimental evidence\n"
        # 3 UniProt members → 3 pairs
        row = (
            "CPX-1\tTestComplex\t-\t9606\t"
            "uniprotkb:P12345(1)|uniprotkb:Q67890(1)|uniprotkb:A11111(1)\t"
            "ECO:0000353\t-\n"
        )
        p = write_tmp(tmp_path, "complex_portal.tsv", header + row)
        df = parse_complex_portal(p)
        assert len(df) == 3

    def test_non_uniprot_members_ignored(self, tmp_path):
        header = "Complex ac\tRecommended name\tAliases\tTaxonomy identifier\t"
        header += "Identifiers (and stoichiometry) of molecules in complex\t"
        header += "Evidence Code\tExperimental evidence\n"
        row = (
            "CPX-1\tTestComplex\t-\t9606\t"
            "chebi:12345(1)|uniprotkb:P12345(1)\t"
            "ECO:0000353\t-\n"
        )
        p = write_tmp(tmp_path, "complex_portal.tsv", header + row)
        df = parse_complex_portal(p)
        # Only 1 UniProt member → no pairs
        assert len(df) == 0


# ── PhosphoSitePlus parser ────────────────────────────────────────────────────

class TestParsePhosphoSitePlus:
    def test_basic(self, tmp_path):
        header = (
            "KINASE\tKIN_ACC_ID\tKIN_ORGANISM\t"
            "SUBSTRATE\tSUB_ACC_ID\tSUB_ORGANISM\t"
            "SUB_MOD_RSD\tSITE_GRP_ID\tSITE_+/-7_AA\t"
            "DOMAIN\tIN_VIVO_RXN\tIN_VITRO_RXN\tCST_Catalog#\n"
        )
        row = "EGFR\tP00533\thuman\tSHC1\tP29353\thuman\tY1068\t1\tABCDEFG\t-\tX\t-\t-\n"
        p = write_tmp(tmp_path, "psp.txt", header + row)
        df = parse_phosphositeplus(p, taxon_filter="human")
        assert len(df) == 1
        assert df["throughput_type"][0] == "LTP"

    def test_non_human_filtered(self, tmp_path):
        header = (
            "KINASE\tKIN_ACC_ID\tKIN_ORGANISM\t"
            "SUBSTRATE\tSUB_ACC_ID\tSUB_ORGANISM\t"
            "SUB_MOD_RSD\tSITE_GRP_ID\tSITE_+/-7_AA\t"
            "DOMAIN\tIN_VIVO_RXN\tIN_VITRO_RXN\tCST_Catalog#\n"
        )
        row = "Egfr\tP00533\tmouse\tShc1\tP29353\tmouse\tY1068\t1\tABCDEFG\t-\tX\t-\t-\n"
        p = write_tmp(tmp_path, "psp.txt", header + row)
        df = parse_phosphositeplus(p, taxon_filter="human")
        assert len(df) == 0

    def test_gz_file(self, tmp_path):
        header = (
            "KINASE\tKIN_ACC_ID\tKIN_ORGANISM\t"
            "SUBSTRATE\tSUB_ACC_ID\tSUB_ORGANISM\t"
            "SUB_MOD_RSD\tSITE_GRP_ID\tSITE_+/-7_AA\t"
            "DOMAIN\tIN_VIVO_RXN\tIN_VITRO_RXN\tCST_Catalog#\n"
        )
        row = "EGFR\tP00533\thuman\tSHC1\tP29353\thuman\tY1068\t1\tABCDEFG\t-\tX\t-\t-\n"
        p = write_tmp_gz(tmp_path, "psp.txt.gz", header + row)
        df = parse_phosphositeplus(p, taxon_filter="human")
        assert len(df) == 1

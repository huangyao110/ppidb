"""Tests for ppidb.core"""

import pytest
import polars as pl
from ppidb.core.pair import PPIPair
from ppidb.core.dataset import PPIDataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Small in-memory PPI DataFrame for testing."""
    return pl.DataFrame({
        "uniprot_a": ["P04637", "P53350", "O15111", "P04637", "A0A000"],
        "uniprot_b": ["P53350", "O15111", "P04637", "O15111", "B0B000"],
        "source_dbs": ["BioGRID|IntAct", "BioGRID", "IntAct", "STRING", "Negatome"],
        "n_sources": [2, 1, 1, 1, 1],
        "taxon_a": ["9606", "9606", "9606", "9606", None],
        "taxon_b": ["9606", "9606", "9606", "9606", None],
        "detection_methods": [
            "two hybrid|pull down",
            "affinity chromatography technology",
            "anti tag coimmunoprecipitation",
            "combined_score:800",
            "literature_curated_negative",
        ],
        "interaction_type": ["positive", "positive", "positive", "positive", "negative"],
        "throughput_type": ["LTP", "LTP", "LTP", "no_exp", "negative_sample"],
    })


@pytest.fixture
def sample_ds(sample_df):
    return PPIDataset(sample_df)


# ── PPIPair tests ─────────────────────────────────────────────────────────────

class TestPPIPair:
    def test_canonical_ordering(self):
        """Smaller UniProt ID should always be in position A."""
        pair = PPIPair(uniprot_a="Z99999", uniprot_b="A00001")
        assert pair.uniprot_a == "A00001"
        assert pair.uniprot_b == "Z99999"

    def test_pair_id(self):
        pair = PPIPair(uniprot_a="A00001", uniprot_b="Z99999")
        assert pair.pair_id == "A00001__Z99999"

    def test_is_human(self):
        pair = PPIPair(uniprot_a="P04637", uniprot_b="P53350", taxon_a="9606", taxon_b="9606")
        assert pair.is_human

        pair_mixed = PPIPair(uniprot_a="P04637", uniprot_b="P53350", taxon_a="9606", taxon_b="10090")
        assert not pair_mixed.is_human

    def test_is_ltp(self):
        ltp = PPIPair(uniprot_a="A", uniprot_b="B", throughput_type="LTP")
        both = PPIPair(uniprot_a="A", uniprot_b="B", throughput_type="both")
        htp = PPIPair(uniprot_a="A", uniprot_b="B", throughput_type="HTP")
        assert ltp.is_ltp
        assert both.is_ltp
        assert not htp.is_ltp

    def test_from_dict(self):
        d = {
            "uniprot_a": "P04637", "uniprot_b": "P53350",
            "source_dbs": "BioGRID|IntAct", "n_sources": 2,
            "taxon_a": "9606", "taxon_b": "9606",
            "detection_methods": "two hybrid",
            "interaction_type": "positive",
            "throughput_type": "LTP",
        }
        pair = PPIPair.from_dict(d)
        assert pair.uniprot_a == "P04637"
        assert pair.n_sources == 2
        assert "BioGRID" in pair.source_dbs
        assert pair.is_ltp

    def test_immutable(self):
        pair = PPIPair(uniprot_a="A", uniprot_b="B")
        with pytest.raises((AttributeError, TypeError)):
            pair.uniprot_a = "C"


# ── PPIDataset tests ──────────────────────────────────────────────────────────

class TestPPIDataset:
    def test_len(self, sample_ds):
        assert len(sample_ds) == 5

    def test_repr(self, sample_ds):
        r = repr(sample_ds)
        assert "PPIDataset" in r
        assert "5" in r

    def test_collect(self, sample_ds):
        df = sample_ds.collect()
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5

    def test_head(self, sample_ds):
        head = sample_ds.head(3)
        assert len(head) == 3

    def test_proteins(self, sample_ds):
        proteins = sample_ds.proteins()
        assert isinstance(proteins, list)
        assert "P04637" in proteins
        assert len(proteins) == len(set(proteins))  # no duplicates

    def test_to_pandas(self, sample_ds):
        import pandas as pd
        pdf = sample_ds.to_pandas()
        assert isinstance(pdf, pd.DataFrame)
        assert len(pdf) == 5

    def test_iter(self, sample_ds):
        pairs = list(sample_ds.filter.positives_only())
        assert all(isinstance(p, PPIPair) for p in pairs)
        assert all(p.is_positive for p in pairs)


# ── FilterAccessor tests ──────────────────────────────────────────────────────

class TestFilterAccessor:
    def test_by_species(self, sample_ds):
        human = sample_ds.filter.by_species("9606")
        # All rows with taxon_a=9606 AND taxon_b=9606
        assert len(human) == 4

    def test_by_throughput_ltp(self, sample_ds):
        ltp = sample_ds.filter.by_throughput("LTP")
        assert len(ltp) == 3

    def test_by_throughput_experimental(self, sample_ds):
        exp = sample_ds.filter.by_throughput("experimental")
        assert len(exp) == 3  # LTP only in this fixture

    def test_by_min_sources(self, sample_ds):
        multi = sample_ds.filter.by_min_sources(2)
        assert len(multi) == 1  # only BioGRID|IntAct pair has n_sources=2

    def test_positives_only(self, sample_ds):
        pos = sample_ds.filter.positives_only()
        assert len(pos) == 4

    def test_negatives_only(self, sample_ds):
        neg = sample_ds.filter.negatives_only()
        assert len(neg) == 1

    def test_by_database(self, sample_ds):
        biogrid = sample_ds.filter.by_database("BioGRID")
        assert len(biogrid) == 2

    def test_by_proteins(self, sample_ds):
        subset = sample_ds.filter.by_proteins(["P04637"])
        # P04637 appears in 3 pairs
        assert len(subset) == 3

    def test_chain(self, sample_ds):
        """Test fluent chaining."""
        result = (
            sample_ds
            .filter.by_species("9606")
            .filter.by_throughput("LTP")
            .filter.positives_only()
        )
        assert len(result) == 3

    def test_high_confidence_preset(self, sample_ds):
        hc = sample_ds.filter.high_confidence(min_sources=2)
        # human + positive + LTP + n_sources>=2
        assert len(hc) == 1


# ── Shared large fixture ──────────────────────────────────────────────────────

@pytest.fixture
def large_ds():
    """Larger dataset for split/negative testing."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = 1000
    proteins = [f"P{i:05d}" for i in range(100)]
    pairs = set()
    rows = []
    while len(rows) < n:
        a, b = rng.choice(proteins, 2, replace=False)
        if a > b:
            a, b = b, a
        if (a, b) not in pairs:
            pairs.add((a, b))
            rows.append({
                "uniprot_a": a, "uniprot_b": b,
                "source_dbs": "BioGRID", "n_sources": 1,
                "taxon_a": "9606", "taxon_b": "9606",
                "detection_methods": "two hybrid",
                "interaction_type": "positive",
                "throughput_type": "LTP",
            })
    return PPIDataset(pl.DataFrame(rows))


# ── NegativeSampler tests ─────────────────────────────────────────────────────

class TestNegativeSampler:
    def test_random_sample(self, large_ds):
        from ppidb.negative import NegativeSampler
        pos = large_ds.filter.positives_only()
        sampler = NegativeSampler(pos)
        negs = sampler.random_sample(n=50, seed=42)
        neg_df = negs.collect()
        assert len(neg_df) == 50
        assert all(neg_df["interaction_type"] == "negative")

    def test_no_overlap_with_positives(self, large_ds):
        from ppidb.negative import NegativeSampler
        pos = large_ds.filter.positives_only()
        sampler = NegativeSampler(pos)
        negs = sampler.random_sample(n=50, seed=42)

        pos_ids = set(
            a + "__" + b
            for a, b in zip(
                pos.collect()["uniprot_a"].to_list(),
                pos.collect()["uniprot_b"].to_list()
            )
        )
        neg_df = negs.collect()
        for row in neg_df.to_dicts():
            pair_id = row["uniprot_a"] + "__" + row["uniprot_b"]
            assert pair_id not in pos_ids, f"Negative pair {pair_id} found in positives!"

    def test_combine(self, large_ds):
        from ppidb.negative import NegativeSampler
        pos = large_ds.filter.positives_only()
        sampler = NegativeSampler(pos)
        negs = sampler.random_sample(n=50, seed=42)
        combined = NegativeSampler.combine(pos, negs, shuffle=True, seed=42)
        assert len(combined) == len(pos) + len(negs)


# ── Splitter tests ────────────────────────────────────────────────────────────

class TestSplitter:
    def test_random_split_sizes(self, large_ds):
        from ppidb.split import Splitter
        result = Splitter(large_ds).random_split(train=0.8, val=0.1, test=0.1, seed=42)
        total = len(result.train) + len(result.val) + len(result.test)
        assert total == len(large_ds)
        assert abs(len(result.train) / total - 0.8) < 0.02

    def test_cold_split_no_protein_overlap(self, large_ds):
        from ppidb.split import Splitter
        result = Splitter(large_ds).cold_split(test_frac=0.1, val_frac=0.1, seed=42)

        train_proteins = set(
            result.train.collect()["uniprot_a"].to_list() +
            result.train.collect()["uniprot_b"].to_list()
        )
        test_proteins = set(
            result.test.collect()["uniprot_a"].to_list() +
            result.test.collect()["uniprot_b"].to_list()
        )
        # No protein should appear in both train and test
        overlap = train_proteins & test_proteins
        assert len(overlap) == 0, f"Protein overlap found: {overlap}"

    def test_split_result_save_load(self, large_ds, tmp_path):
        from ppidb.split import Splitter, SplitResult
        result = Splitter(large_ds).random_split(seed=42)
        result.save(tmp_path / "split")
        loaded = SplitResult.load(tmp_path / "split")
        assert len(loaded.train) == len(result.train)
        assert len(loaded.test) == len(result.test)
        assert loaded.metadata["strategy"] == "random"


class TestC1C2C3StrictSeqSim:
    def test_classify_pairs_strict_seqsim(self):
        from ppidb.split.c1c2c3 import classify_pairs

        train_proteins = {"A", "B"}
        test_df = pl.DataFrame({
            "uniprot_a": ["A", "A", "Y", "X", "X"],
            "uniprot_b": ["B", "Y", "Z", "Y", "A"],
        })
        sequence_dict = {
            "A": "AAAAAAAAAA",
            "B": "CCCCCCCCCC",
            "X": "AAAAAAAACC",
            "Y": "GGGGGGGGGG",
            "Z": "TTTTTTTTTT",
        }

        labels = classify_pairs(
            test_df,
            train_proteins,
            sequence_dict=sequence_dict,
            identity_threshold=0.3,
        ).to_list()

        assert labels == ["C1", "C2", "C3", "C2", "C1"]

    def test_classify_pairs_strict_seqsim_missing_sequence(self):
        from ppidb.split.c1c2c3 import classify_pairs

        train_proteins = {"A", "B"}
        test_df = pl.DataFrame({
            "uniprot_a": ["X"],
            "uniprot_b": ["Y"],
        })
        sequence_dict = {
            "A": "AAAAAAAAAA",
            "B": "CCCCCCCCCC",
            "X": "AAAAAAAACC",
        }

        with pytest.raises(ValueError):
            classify_pairs(
                test_df,
                train_proteins,
                sequence_dict=sequence_dict,
                identity_threshold=0.3,
            )

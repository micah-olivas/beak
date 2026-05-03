"""Regression tests for `annotate_temperature`.

Locks the contract that surfaces the taxonomy view's growth-temp
histogram: organism strings as UniProt returns them ("Escherichia
coli", "Thermus thermophilus (strain HB8)") must resolve to a numeric
growth_temp via the Enqvist dataset, with a genus-level fallback when
the species name isn't in the lookup.
"""

import pandas as pd
import pytest

from beak.temperature import (
    _to_genus,
    _to_genus_species,
    annotate_temperature,
)


class TestNormalization:
    def test_titlecase_with_space(self):
        assert _to_genus_species("Escherichia coli") == "escherichia_coli"
        assert _to_genus("Escherichia coli") == "escherichia"

    def test_strain_qualifier_in_parens_is_stripped(self):
        assert (
            _to_genus_species("Escherichia coli (strain K12)")
            == "escherichia_coli"
        )

    def test_trailing_strain_token_is_dropped(self):
        # Only genus + species are kept; trailing strain identifiers don't
        # appear in the Enqvist key, so they shouldn't break the merge.
        assert (
            _to_genus_species("Mycobacterium tuberculosis H37Rv")
            == "mycobacterium_tuberculosis"
        )

    def test_nan_and_empty(self):
        assert _to_genus_species(None) is None
        assert _to_genus_species("") is None
        assert _to_genus(None) is None

    def test_single_token(self):
        # No species — return just the genus.
        assert _to_genus_species("Escherichia") == "escherichia"
        assert _to_genus("Escherichia") == "escherichia"


class TestAnnotateTemperature:
    @pytest.fixture
    def organism_df(self):
        return pd.DataFrame({
            "sequence_id": ["s1", "s2", "s3", "s4", "s5"],
            "organism": [
                "Escherichia coli",                  # well-known mesophile
                "Thermus thermophilus (strain HB8)", # thermophile w/ strain
                "Pyrococcus furiosus",               # hyperthermophile
                "Banana republic phantasmus",        # nonsense → NaN
                None,                                # missing organism
            ],
        })

    def test_titlecase_uniprot_names_now_resolve(self, organism_df):
        # The bug: an exact merge between "Escherichia coli" and
        # "escherichia_coli" matched zero rows, so the taxonomy view's
        # growth-temp histogram silently rendered as empty.
        out = annotate_temperature(organism_df, organism_col="organism")
        assert "growth_temp" in out.columns
        ec = out.loc[out["organism"] == "Escherichia coli", "growth_temp"]
        assert ec.notna().all(), "E. coli should resolve to a real number"
        assert 20 <= ec.iloc[0] <= 45  # mesophile range

    def test_strain_qualifier_does_not_block_match(self, organism_df):
        out = annotate_temperature(organism_df, organism_col="organism")
        tt = out.loc[
            out["organism"] == "Thermus thermophilus (strain HB8)",
            "growth_temp",
        ]
        assert tt.notna().all()
        assert tt.iloc[0] >= 60  # thermophile

    def test_unknown_organism_is_nan_not_error(self, organism_df):
        out = annotate_temperature(organism_df, organism_col="organism")
        nonsense = out.loc[
            out["organism"] == "Banana republic phantasmus", "growth_temp"
        ]
        assert nonsense.isna().all()

    def test_missing_organism_is_nan(self, organism_df):
        out = annotate_temperature(organism_df, organism_col="organism")
        assert out.loc[out["organism"].isna(), "growth_temp"].isna().all()

    def test_temp_source_labels_match_resolution_path(self, organism_df):
        out = annotate_temperature(organism_df, organism_col="organism")
        # E. coli is in the species-level table → temp_source = 'species'.
        sources = dict(zip(out["organism"], out["temp_source"]))
        assert sources["Escherichia coli"] == "species"
        # Nonsense organism: NaN.
        assert pd.isna(sources["Banana republic phantasmus"])

    def test_genus_fallback_resolves_species_not_in_dataset(self):
        # Pick a real genus where many species are in Enqvist but invent
        # a species name. The fallback should still produce a number.
        df = pd.DataFrame({"organism": ["Escherichia notarealspecies"]})
        out = annotate_temperature(df, organism_col="organism")
        v = out["growth_temp"].iloc[0]
        assert pd.notna(v)
        assert out["temp_source"].iloc[0] == "genus"
        # E. coli relatives are mesophilic; sanity-bound the fallback.
        assert 15 <= v <= 50

    def test_input_dataframe_is_not_mutated(self, organism_df):
        before = organism_df.copy()
        annotate_temperature(organism_df, organism_col="organism")
        pd.testing.assert_frame_equal(organism_df, before)

    def test_organism_col_must_exist(self):
        df = pd.DataFrame({"organism": ["Escherichia coli"]})
        with pytest.raises(KeyError):
            annotate_temperature(df, organism_col="missing")

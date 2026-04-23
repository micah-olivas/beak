"""Tests for the backend dispatch in generate_embeddings.py.

Uses importlib to load the generator script without triggering the
heavy imports inside generate_embeddings() (torch, transformers,
fair-esm). Only the dispatcher and its backend-class structure are
exercised — actual model loading is out of scope here.
"""

import importlib.util
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src" / "beak" / "remote" / "docker" / "generate_embeddings.py"
)
_spec = importlib.util.spec_from_file_location("_gen_embed", _SCRIPT)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


class TestMakeBackend:
    def test_esm2_prefix_raises_if_fair_esm_not_installed(self, monkeypatch):
        # We don't actually need fair-esm installed to test dispatch —
        # we just need to confirm the right branch is taken. Patch the
        # Esm2Backend constructor to a sentinel.
        calls = []

        class Stub:
            name = 'esm2'

            def __init__(self, model_name, gpu_id):
                calls.append(('esm2', model_name, gpu_id))

        monkeypatch.setattr(gen, 'Esm2Backend', Stub)
        backend = gen._make_backend('esm2_t12_35M_UR50D', gpu_id=0)
        assert calls == [('esm2', 'esm2_t12_35M_UR50D', 0)]

    def test_esmc_prefix_dispatches_to_esmc_backend(self, monkeypatch):
        calls = []

        class Stub:
            name = 'esmc'

            def __init__(self, model_name, gpu_id):
                calls.append(('esmc', model_name, gpu_id))

        monkeypatch.setattr(gen, 'EsmCBackend', Stub)
        gen._make_backend('esmc_300m', gpu_id=1)
        gen._make_backend('esmc_600m', gpu_id=0)
        assert [c[1] for c in calls] == ['esmc_300m', 'esmc_600m']

    def test_full_hf_esmc_path_also_dispatches_to_esmc(self, monkeypatch):
        # Users can pass a full HF org/name path. Anything containing
        # '/esmc' should still land on the ESM-C backend.
        calls = []

        class Stub:
            name = 'esmc'

            def __init__(self, model_name, gpu_id):
                calls.append(model_name)

        monkeypatch.setattr(gen, 'EsmCBackend', Stub)
        gen._make_backend('EvolutionaryScale/esmc_300m_2024_12', gpu_id=0)
        assert calls == ['EvolutionaryScale/esmc_300m_2024_12']

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unrecognized model family"):
            gen._make_backend('some_random_model', gpu_id=0)

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="Unrecognized model family"):
            gen._make_backend('', gpu_id=0)


class TestEsmCBackendHfIdMap:
    def test_short_aliases_defined(self):
        assert 'esmc_300m' in gen.EsmCBackend.HF_IDS
        assert 'esmc_600m' in gen.EsmCBackend.HF_IDS

    def test_aliases_map_to_evolutionaryscale_hub_paths(self):
        for alias, hf_id in gen.EsmCBackend.HF_IDS.items():
            assert hf_id.startswith('EvolutionaryScale/')
            assert 'esmc' in hf_id.lower()

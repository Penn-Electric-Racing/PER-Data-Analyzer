import builtins
import zlib

import numpy as np
import pytest

from perda.constants import SENTENCE_EMBEDDING_DIMENSION
from perda.core_data_structures.data_instance import DataInstance
from perda.core_data_structures.semantic_embedding import load_encoder
from perda.core_data_structures.single_run_data import SingleRunData
from perda.units import Timescale


@pytest.fixture
def di_simple():
    """Short, clean DataInstance with timestamps [0,1,2,3] and linear values."""
    return DataInstance(
        timestamp_np=np.array([0, 1, 2, 3], dtype=np.int64),
        value_np=np.array([0.0, 1.0, 2.0, 3.0]),
        label="simple",
        var_id=1,
        cpp_name="test.simple",
    )


@pytest.fixture
def srd_basic():
    """Minimal but complete SingleRunData with two variables."""
    di_a = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([1.0, 2.0, 3.0]),
        label="var_a",
        var_id=1,
        cpp_name="cpp.var_a",
    )
    di_b = DataInstance(
        timestamp_np=np.array([0, 1, 2], dtype=np.int64),
        value_np=np.array([4.0, 5.0, 6.0]),
        label="var_b",
        var_id=2,
        cpp_name="cpp.var_b",
    )
    return SingleRunData(
        id_to_instance={1: di_a, 2: di_b},
        cpp_name_to_id={"cpp.var_a": 1, "cpp.var_b": 2},
        id_to_cpp_name={1: "cpp.var_a", 2: "cpp.var_b"},
        id_to_descript={1: "Variable A description", 2: "Variable B description"},
        total_data_points=6,
        data_start_time=0,
        data_end_time=2,
        timestamp_unit=Timescale.MS,
    )


@pytest.fixture
def fake_encoder(monkeypatch):
    """Replace the sentence model with a bag-of-words hashing encoder.

    Yields the list of text batches it was asked to encode.
    """
    batches = []

    def encode(texts):
        batches.append(list(texts))
        vectors = np.zeros((len(texts), SENTENCE_EMBEDDING_DIMENSION))
        for row, text in enumerate(texts):
            for word in text.lower().split():
                vectors[
                    row, zlib.crc32(word.encode()) % SENTENCE_EMBEDDING_DIMENSION
                ] += 1
        return vectors

    monkeypatch.setattr(
        "perda.core_data_structures.semantic_embedding.load_encoder",
        lambda model_id: encode,
    )
    yield batches


@pytest.fixture
def no_sentence_transformers(monkeypatch):
    """Make importing sentence-transformers fail, as in a base install.

    Clears the encoder cache on both sides to reset between tests.
    """
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    load_encoder.cache_clear()
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    yield
    load_encoder.cache_clear()


@pytest.fixture
def failing_encoder(monkeypatch):
    """Make the encoder loader raise, as when a model download fails."""

    def blow_up(model_id):
        raise RuntimeError("model load failure")

    load_encoder.cache_clear()
    monkeypatch.setattr(
        "perda.core_data_structures.semantic_embedding.load_encoder", blow_up
    )
    yield
    load_encoder.cache_clear()

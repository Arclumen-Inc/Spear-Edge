"""Tests for spear_edge.ml.preprocess (Phase A contract)."""

import numpy as np
import pytest

from spear_edge.ml.preprocess import (
    CURRENT_PREPROCESS_SCHEMA,
    apply_model_preprocess_v1,
    ml_features_metadata,
    spec_ml_to_single_bchw,
    validate_spec_2d,
)


def test_schema_constant():
    assert CURRENT_PREPROCESS_SCHEMA == "spear_ml_spec_v1"


def test_ml_features_metadata_has_schema():
    meta = ml_features_metadata()
    assert meta["preprocess_schema"] == CURRENT_PREPROCESS_SCHEMA
    assert meta["spectrogram_shape"] == [512, 512]
    assert meta["dtype"] == "float32"


def test_validate_spec_2d_ok():
    x = np.zeros((512, 512), dtype=np.float32)
    y = validate_spec_2d(x)
    assert y.shape == (512, 512)


def test_validate_spec_2d_wrong_shape():
    with pytest.raises(ValueError, match="Expected shape"):
        validate_spec_2d(np.zeros((64, 64), dtype=np.float32))


def test_spec_ml_to_single_bchw_layouts():
    x2 = np.random.randn(512, 512).astype(np.float32)
    y = spec_ml_to_single_bchw(x2)
    assert y.shape == (1, 1, 512, 512)

    x3 = x2[np.newaxis, ...]
    y3 = spec_ml_to_single_bchw(x3)
    assert y3.shape == (1, 1, 512, 512)


def test_apply_model_preprocess_v1_identity_shape():
    x = np.ones((512, 512), dtype=np.float64)
    z = apply_model_preprocess_v1(x)
    assert z.dtype == np.float32
    assert z.shape == (512, 512)

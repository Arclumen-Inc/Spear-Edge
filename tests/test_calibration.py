import numpy as np

from spear_edge.ml.calibration import (
    distribution_entropy_natural,
    fit_temperature,
    softmax_rows,
)


def test_softmax_rows_sums_to_one():
    x = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float32)
    p = softmax_rows(x)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_fit_temperature_reduces_nll_on_peaked_logits():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((200, 5)).astype(np.float32) * 0.5
    labels = rng.integers(0, 5, size=(200,), dtype=np.int64)
    # Make correct class slightly higher on average
    for i in range(200):
        logits[i, labels[i]] += 2.0
    T, m = fit_temperature(logits, labels)
    assert T > 0
    assert m["nll_after"] <= m["nll_before"] + 1e-6


def test_entropy_uniform():
    p = np.ones(4, dtype=np.float32) / 4.0
    h = distribution_entropy_natural(p)
    assert abs(h - np.log(4)) < 1e-5

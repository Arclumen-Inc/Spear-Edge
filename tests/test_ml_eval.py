"""Tests for spear_edge.ml.eval."""

import numpy as np

from spear_edge.ml.eval import compute_per_class_metrics


def test_compute_per_class_metrics_perfect():
    cm = np.eye(3, dtype=np.int64)
    m = compute_per_class_metrics(cm)
    assert m["macro_f1"] == 1.0
    assert m["macro_precision"] == 1.0
    assert m["macro_recall"] == 1.0

"""Phase 0: required ML assets exist (weights are optional / gitignored)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "spear_edge" / "ml" / "models"


def test_class_labels_present():
    assert (MODELS / "class_labels.json").is_file()


def test_dummy_generator_scripts_present():
    assert (MODELS / "make_dummy_pytorch_model.py").is_file()
    assert (MODELS / "make_dummy_model.py").is_file()


@pytest.mark.skipif(
    not (MODELS / "rf_classifier.pth").is_file(),
    reason="No local rf_classifier.pth (expected after Phase 0 cleanup)",
)
def test_optional_main_weights_load_only_if_present():
    """If operator restored weights, file should be readable (smoke)."""
    p = MODELS / "rf_classifier.pth"
    assert p.stat().st_size > 1000

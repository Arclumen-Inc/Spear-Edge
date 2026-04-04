"""
Single preprocessing contract for RF spectrogram classification.

All training, fine-tuning, and inference must use this module so inputs match
live captures. Spectrograms are produced by compute_spectrogram_chunked() in
spear_edge/core/capture/spectrogram.py (512x512, float32, median noise floor
subtracted in dB space).

Schema bumps require retraining; document in docs/ml-pipeline.md.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# Bump when preprocessing or capture contract changes (invalidates old .pth).
SPEAR_ML_PREPROCESS_SCHEMA_V1 = "spear_ml_spec_v1"
CURRENT_PREPROCESS_SCHEMA = SPEAR_ML_PREPROCESS_SCHEMA_V1

ML_SPEC_SHAPE: Tuple[int, int] = (512, 512)


def ml_features_metadata() -> Dict[str, Any]:
    """Metadata written to capture.json under ml_features."""
    return {
        "spectrogram_shape": [ML_SPEC_SHAPE[0], ML_SPEC_SHAPE[1]],
        "dtype": "float32",
        "normalized": "noise_floor",
        "preprocess_schema": CURRENT_PREPROCESS_SCHEMA,
    }


def validate_spec_2d(spec: np.ndarray) -> np.ndarray:
    """
    Ensure a spectrogram slice is float32 (512, 512) with finite values.
    """
    x = np.asarray(spec, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(
            f"Expected 2D spectrogram (H, W), got shape {x.shape} "
            f"(preprocess_schema={CURRENT_PREPROCESS_SCHEMA})"
        )
    if x.shape != ML_SPEC_SHAPE:
        raise ValueError(
            f"Expected shape {ML_SPEC_SHAPE}, got {x.shape} "
            f"(preprocess_schema={CURRENT_PREPROCESS_SCHEMA})"
        )
    if not np.isfinite(x).all():
        raise ValueError("Spectrogram contains non-finite values")
    return x


def apply_model_preprocess_v1(spec_2d: np.ndarray) -> np.ndarray:
    """
    spe_ml_spec_v1: capture pipeline already applies noise-floor normalization.
    No additional per-sample min-max (that caused train/serve skew).
    """
    return validate_spec_2d(spec_2d)


def spec_ml_to_bchw(spec_ml: np.ndarray) -> np.ndarray:
    """
    Convert user-provided spectrogram layout to (batch, 1, 512, 512) float32.

    Accepts (512, 512), (1, 512, 512), (N, 512, 512), or (N, 1, 512, 512) /
    (N, H, W, 1) style layouts consistent with infer_pytorch historical API.
    """
    x = np.asarray(spec_ml, dtype=np.float32)

    if x.ndim == 2:
        z = apply_model_preprocess_v1(x)
        return z[None, None, :, :]

    if x.ndim == 3:
        if x.shape[0] == 1:
            z = apply_model_preprocess_v1(x[0])
            return z[None, None, :, :]
        if x.shape[1:] != ML_SPEC_SHAPE:
            raise ValueError(
                f"Expected (N, 512, 512), got {x.shape} "
                f"(preprocess_schema={CURRENT_PREPROCESS_SCHEMA})"
            )
        out = np.stack([apply_model_preprocess_v1(x[i]) for i in range(x.shape[0])])
        return out[:, None, :, :]

    if x.ndim == 4:
        if x.shape[1] != 1:
            x = x.transpose(0, 3, 1, 2)
        if x.shape[1] != 1 or x.shape[2:] != ML_SPEC_SHAPE:
            raise ValueError(
                f"Expected (N, 1, 512, 512) or (N, H, W, 1), got {x.shape} "
                f"after layout normalize (preprocess_schema={CURRENT_PREPROCESS_SCHEMA})"
            )
        batch = x.shape[0]
        out = np.stack([apply_model_preprocess_v1(x[i, 0]) for i in range(batch)])
        return out[:, None, :, :]

    raise ValueError(f"Unsupported spectrogram ndim={x.ndim}, shape={x.shape}")


def spec_ml_to_single_bchw(spec_ml: np.ndarray) -> np.ndarray:
    """Return (1, 1, 512, 512) for a single example (classify path)."""
    bchw = spec_ml_to_bchw(spec_ml)
    if bchw.shape[0] != 1:
        raise ValueError(f"Expected single example, got batch {bchw.shape[0]}")
    return bchw

"""
Temperature scaling for classifier calibration (post-hoc on validation logits).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _log_softmax_stable(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=1, keepdims=True)
    ex = np.exp(logits - m)
    return (logits - m) - np.log(np.sum(ex, axis=1, keepdims=True) + 1e-12)


def nll_mean_stable(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    t = max(float(temperature), 1e-6)
    ls = _log_softmax_stable(logits / t)
    n = logits.shape[0]
    return float(-np.mean(ls[np.arange(n), labels.astype(np.int64)]))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    t_min: float = 0.05,
    t_max: float = 15.0,
    steps: int = 80,
) -> Tuple[float, Dict[str, float]]:
    """
    Grid search temperature T>0 minimizing validation NLL (1D, robust, no extra deps).
    """
    if logits.size == 0 or len(labels) == 0:
        return 1.0, {"nll_before": 0.0, "nll_after": 0.0}
    Ts = np.geomspace(t_min, t_max, num=steps)
    nll0 = nll_mean_stable(logits, labels, 1.0)
    best_t, best_nll = 1.0, nll0
    for t in Ts:
        nll = nll_mean_stable(logits, labels, float(t))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t, {"nll_before": nll0, "nll_after": best_nll}


def collect_val_logits(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    parts_l = []
    parts_y = []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            out = model(data)
            parts_l.append(out.cpu().numpy())
            parts_y.append(target.numpy())
    if not parts_l:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(parts_l, axis=0), np.concatenate(parts_y, axis=0)


def save_calibration_json(
    path: Path,
    temperature: float,
    metrics: Dict[str, float],
    preprocess_schema: str,
) -> None:
    doc = {
        "schema": "spear_edge.calibration.v1",
        "temperature": float(temperature),
        "metrics": metrics,
        "preprocess_schema": preprocess_schema,
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def load_temperature_from_json(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("temperature", 1.0))
    except Exception:
        return None


def distribution_entropy_natural(probs: np.ndarray) -> float:
    """Shannon entropy in nats for a single probability vector."""
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))

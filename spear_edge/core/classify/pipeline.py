# spear_edge/core/classify/pipeline.py

from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Dict, Any

class ClassifierPipeline:
    """
    RF signal classification pipeline.
    Stage 1: coarse RF family classification.
    """

    def __init__(self):
        self.enabled = True
        self.model = None  # loaded lazily

    def classify_capture(self, capture: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point called after a capture completes.
        """
        if not self.enabled:
            return self._noop_result(capture)

        iq_path = Path(capture["iq_path"])
        meta = capture.get("meta", {})

        iq = self._load_iq(iq_path)

        features = self._extract_features(iq, meta)

        label, confidence, scores = self._infer(features)

        return {
            "schema": "spear.edge.classify.v1",
            "freq_hz": meta.get("freq_hz"),
            "primary_label": label,
            "confidence": confidence,
            "top_k": scores,
            "features_used": list(features.keys()),
            "model": {
                "name": "rf_family_stub",
                "version": "0.1"
            }
        }

    # --------------------------------------------------

    def _load_iq(self, path: Path) -> np.ndarray:
        return np.fromfile(path, dtype=np.complex64)

    def _extract_features(self, iq: np.ndarray, meta: dict) -> Dict[str, np.ndarray]:
        """
        Deterministic RF features (NO normalization tricks).
        """
        features = {}

        # Instantaneous frequency variance (FHSS explodes here)
        phase = np.unwrap(np.angle(iq))
        inst_freq = np.diff(phase)
        features["if_variance"] = np.array([np.var(inst_freq)], dtype=np.float32)

        # Simple power stats
        mag2 = iq.real**2 + iq.imag**2
        features["rms_power"] = np.array([np.mean(mag2)], dtype=np.float32)
        features["crest_factor"] = np.array(
            [np.sqrt(np.max(mag2)) / (np.sqrt(np.mean(mag2)) + 1e-9)],
            dtype=np.float32,
        )

        return features

    def _infer(self, features: Dict[str, np.ndarray]):
        """
        STUB inference logic.
        Replace with ML model later.
        """
        if features["if_variance"][0] > 1.0:
            return "fhss_like", 0.9, [
                ("fhss_like", 0.9),
                ("wideband_ofdm_like", 0.08),
                ("noise", 0.02),
            ]

        return "unknown", 0.5, [
            ("unknown", 0.5),
            ("noise", 0.3),
            ("analog_fm_voice", 0.2),
        ]

    def _noop_result(self, capture):
        return {
            "schema": "spear.edge.classify.v1",
            "primary_label": "disabled",
            "confidence": 0.0,
        }

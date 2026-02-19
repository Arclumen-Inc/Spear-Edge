import numpy as np

class StubRFClassifier:
    """
    Temporary classifier used to test the pipeline.
    Replace with ONNX model later.
    """
    def classify(self, spec_ml: np.ndarray) -> dict:
        # Dummy logic just for plumbing test
        return {
            "label": "unknown",
            "confidence": 0.0,
            "topk": [{"label": "unknown", "p": 1.0}],
            "model": "stub"
        }

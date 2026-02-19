# spear_edge/ml/__init__.py
from .infer_stub import StubRFClassifier

# Try to import ONNX classifier
try:
    from .infer_onnx import ONNXRfClassifier
    __all__ = ["StubRFClassifier", "ONNXRfClassifier"]
except ImportError:
    __all__ = ["StubRFClassifier"]
    ONNXRfClassifier = None

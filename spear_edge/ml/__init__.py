# spear_edge/ml/__init__.py
from .infer_stub import StubRFClassifier

# Try to import PyTorch classifier (GPU-accelerated)
try:
    from .infer_pytorch import PyTorchRfClassifier, RFClassifier
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    PyTorchRfClassifier = None
    RFClassifier = None

# Try to import ONNX classifier
try:
    from .infer_onnx import ONNXRfClassifier
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ONNXRfClassifier = None

# Build __all__ list
__all__ = ["StubRFClassifier"]
if PYTORCH_AVAILABLE:
    __all__.extend(["PyTorchRfClassifier", "RFClassifier"])
if ONNX_AVAILABLE:
    __all__.append("ONNXRfClassifier")

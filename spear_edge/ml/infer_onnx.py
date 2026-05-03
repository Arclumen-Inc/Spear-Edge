# spear_edge/ml/infer_onnx.py
"""
ONNX Runtime-based RF signal classifier.
Uses TensorRT execution provider when available for best performance on Jetson.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from spear_edge.ml.calibration import distribution_entropy_natural, load_temperature_from_json
from spear_edge.ml.preprocess import spec_ml_to_single_bchw

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None


class ONNXRfClassifier:
    """
    RF signal classifier using ONNX Runtime.
    
    Supports:
    - TensorRT execution provider (best performance on Jetson)
    - CUDA execution provider (fallback)
    - CPU execution provider (final fallback)
    
    Input: 512x512 float32 spectrogram (noise-floor normalized)
    Output: Classification with confidence scores
    """
    
    def __init__(self, model_path: str, class_labels_path: Optional[str] = None):
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "onnxruntime not installed. "
                "Install with: pip install onnxruntime (Jetson GPU builds: see docs/INSTALLATION.md)"
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Class labels (same JSON contract as PyTorch path)
        if class_labels_path is None:
            class_labels_path = Path(__file__).parent / "models" / "class_labels.json"
        else:
            class_labels_path = Path(class_labels_path)
        self.class_labels: Optional[Dict[str, Any]] = None
        self.index_to_class: Dict[str, str] = {}
        self.num_classes: int = 0
        if class_labels_path.exists():
            try:
                with open(class_labels_path, "r", encoding="utf-8") as f:
                    self.class_labels = json.load(f)
                self.index_to_class = self.class_labels.get("index_to_class", {}) or {}
                self.num_classes = int(self.class_labels.get("num_classes", len(self.index_to_class)))
                print(f"[ONNX] Loaded class labels from {class_labels_path} ({self.num_classes} classes)")
            except Exception as e:
                print(f"[ONNX] Warning: failed to load class labels: {e}")

        cal_p = model_path.with_suffix(".calibration.json")
        t_cal = load_temperature_from_json(cal_p)
        self.calibration_temperature = float(t_cal) if t_cal is not None else 1.0
        if t_cal is not None:
            print(f"[ONNX] Loaded calibration temperature {t_cal:.4f} from {cal_p.name}")

        # Store model path for API access
        self.model_path = str(model_path)
        
        # Build provider list (try TensorRT first, then CUDA, then CPU)
        providers = []
        available_providers = ort.get_available_providers()
        
        print(f"[ONNX] Available providers: {available_providers}")
        
        # Try TensorRT execution provider (best performance on Jetson)
        if 'TensorrtExecutionProvider' in available_providers:
            try:
                providers.append(('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_max_workspace_size': 2**30,  # 1GB workspace
                    'trt_fp16_enable': True,  # Use FP16 for better performance
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': str(model_path.parent / '.trt_cache'),
                }))
                print("[ONNX] TensorRT execution provider configured")
            except Exception as e:
                print(f"[ONNX] TensorRT provider configuration failed: {e}")
        else:
            print("[ONNX] TensorRT provider not available")
        
        # CUDA provider (good fallback, requires onnxruntime-gpu or Jetson build)
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
            print("[ONNX] CUDA execution provider available")
        else:
            print("[ONNX] CUDA provider not available (CPU-only onnxruntime installed)")
            print("[ONNX] For GPU support on Jetson, install Jetson-specific onnxruntime-gpu")
        
        # CPU provider (always last fallback)
        providers.append("CPUExecutionProvider")
        
        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
        print(f"[ONNX] Model loaded: {model_path}")
        print(f"[ONNX] Active providers: {self.session.get_providers()}")
        print(f"[ONNX] Input: {self.input_name} {self.input_shape}")
        print(f"[ONNX] Output: {self.output_name} {self.output_shape}")
        out_dim = self.output_shape[-1] if self.output_shape and self.output_shape[-1] is not None else None
        if out_dim and not self.num_classes:
            self.num_classes = int(out_dim)
        elif self.session.get_outputs()[0].shape:
            try:
                flat = int(np.prod([d for d in self.output_shape if isinstance(d, int)]))
                if flat and not self.num_classes:
                    self.num_classes = flat
            except Exception:
                pass
    
    def classify(self, spec_ml: np.ndarray) -> Dict[str, Any]:
        """
        Classify a spectrogram.
        
        Args:
            spec_ml: Spectrogram array, shape (512, 512) or (time, freq)
                    dtype float32, noise-floor normalized
        
        Returns:
            Dictionary with:
            - label: Predicted class label
            - confidence: Confidence score [0.0, 1.0]
            - topk: List of top-k predictions
            - model: "onnx"
        """
        x = spec_ml_to_single_bchw(spec_ml)
        
        # Run inference
        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: x}
            )
        except Exception as e:
            print(f"[ONNX] Inference error: {e}")
            raise
        
        logits = outputs[0].astype(np.float64, copy=False)
        if logits.ndim > 1:
            logits = logits.reshape(-1)

        t_cal = max(float(self.calibration_temperature), 1e-6)
        scaled = logits / t_cal
        m = float(np.max(scaled))
        exp_logits = np.exp(scaled - m)
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        class_id = str(idx)
        if class_id in self.index_to_class:
            label = self.index_to_class[class_id]
            class_info = None
            if self.class_labels and "class_mapping" in self.class_labels:
                class_info = self.class_labels["class_mapping"].get(class_id, {})
        else:
            label = f"class_{idx}"
            class_info = None

        signal_type = None
        device_name = None
        if class_info:
            signal_type = class_info.get("signal_type")
            device_name = class_info.get("name", label)

        top_k = min(5, len(probs))
        top_indices = np.argsort(probs)[-top_k:][::-1]
        topk: List[Dict[str, Any]] = []
        for i in top_indices:
            top_class_id = str(int(i))
            if top_class_id in self.index_to_class:
                top_label = self.index_to_class[top_class_id]
                top_info = None
                if self.class_labels and "class_mapping" in self.class_labels:
                    top_info = self.class_labels["class_mapping"].get(top_class_id, {})
                topk.append(
                    {
                        "label": top_label,
                        "name": top_info.get("name", top_label) if top_info else top_label,
                        "signal_type": top_info.get("signal_type") if top_info else None,
                        "p": float(probs[int(i)]),
                    }
                )
            else:
                topk.append(
                    {
                        "label": f"class_{int(i)}",
                        "name": f"class_{int(i)}",
                        "signal_type": None,
                        "p": float(probs[int(i)]),
                    }
                )

        uncertain = False
        uncertain_reasons: List[str] = []
        min_c = os.environ.get("SPEAR_ML_UNCERTAIN_MIN_CONFIDENCE")
        if min_c is not None:
            try:
                if confidence < float(min_c):
                    uncertain = True
                    uncertain_reasons.append("below_min_confidence")
            except ValueError:
                pass
        min_ent = os.environ.get("SPEAR_ML_UNCERTAIN_MIN_ENTROPY_NATS")
        if min_ent is not None:
            try:
                if distribution_entropy_natural(probs) > float(min_ent):
                    uncertain = True
                    uncertain_reasons.append("above_entropy_threshold")
            except ValueError:
                pass

        result: Dict[str, Any] = {
            "label": label,
            "confidence": confidence,
            "topk": topk,
            "model": "onnx",
            "device": "onnxruntime",
            "uncertain": uncertain,
            "calibration_temperature": float(self.calibration_temperature),
        }
        if uncertain_reasons:
            result["uncertain_reasons"] = uncertain_reasons
        if device_name:
            result["device_name"] = device_name
        if signal_type:
            result["signal_type"] = signal_type
        if class_info:
            result["description"] = class_info.get("description")

        return result


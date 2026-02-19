# spear_edge/ml/infer_onnx.py
"""
ONNX Runtime-based RF signal classifier.
Uses TensorRT execution provider when available for best performance on Jetson.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

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
    
    def __init__(self, model_path: str):
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "onnxruntime-gpu not installed. "
                "Install with: pip3 install onnxruntime-gpu"
            )
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
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
        
        # CPU provider (always available, slowest but works)
        providers.append('CPUExecutionProvider')
        print("[ONNX] Using CPU execution provider (fallback)")
        
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
        # Ensure correct dtype
        x = spec_ml.astype(np.float32)
        
        # Handle different input shapes
        if x.ndim == 2:
            # (512, 512) -> (1, 1, 512, 512) for CNN
            x = x[None, None, :, :]
        elif x.ndim == 3:
            # (1, 512, 512) -> (1, 1, 512, 512)
            if x.shape[0] == 1:
                x = x[None, :, :, :]
            else:
                x = x[:, None, :, :]
        elif x.ndim == 4:
            # Already (batch, channels, H, W) - verify channels
            if x.shape[1] != 1:
                # Assume (batch, H, W, channels) -> transpose
                x = x.transpose(0, 3, 1, 2)
        
        # Validate shape
        if x.shape[1] != 1 or x.shape[2] != 512 or x.shape[3] != 512:
            raise ValueError(
                f"Expected input shape (batch, 1, 512, 512), got {x.shape}"
            )
        
        # Run inference
        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: x}
            )
        except Exception as e:
            print(f"[ONNX] Inference error: {e}")
            raise
        
        # Get output
        out = outputs[0]
        
        # Handle different output shapes
        if out.ndim > 1:
            out = out.flatten()
        
        # Apply softmax if output is logits (not probabilities)
        # Check if values are in logit range (typically > 1.0 or sum != 1.0)
        if np.max(out) > 1.0 or abs(np.sum(out) - 1.0) > 0.1:
            # Apply softmax
            exp_out = np.exp(out - np.max(out))  # Numerical stability
            out = exp_out / np.sum(exp_out)
        
        # Get top prediction
        idx = int(np.argmax(out))
        confidence = float(out[idx])
        
        # Get top-k (top 5)
        top_k = min(5, len(out))
        top_indices = np.argsort(out)[-top_k:][::-1]
        
        return {
            "label": f"class_{idx}",
            "confidence": confidence,
            "topk": [
                {"label": f"class_{i}", "p": float(out[i])}
                for i in top_indices
            ],
            "model": "onnx"
        }


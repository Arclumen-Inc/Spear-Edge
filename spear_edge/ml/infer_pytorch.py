# spear_edge/ml/infer_pytorch.py
"""
PyTorch-based RF signal classifier with GPU support.
Uses CUDA when available for accelerated inference.
"""

import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class RFClassifier(nn.Module):
    """
    CNN-based RF signal classifier for hierarchical device/protocol identification.
    Architecture from ML_INFERENCE_PLAN.txt Section 3.3, extended for 30-40 classes.
    
    Input: (batch, 1, 512, 512) float32 spectrogram
    Output: (batch, num_classes) logits
    """
    
    def __init__(self, num_classes: int = 23):
        super().__init__()
        # Feature extraction - deeper network for better feature learning
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  # Additional layer for more capacity
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
        # Classification head - use adaptive pooling to handle any input size
        # After 4 pooling ops: 512 -> 256 -> 128 -> 64 -> 32
        # But use adaptive pooling to be safe
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Fixed size: 256 * 8 * 8 = 16384
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)  # Larger FC layer for more classes
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        # Use adaptive pooling to ensure consistent size
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 256 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class PyTorchRfClassifier:
    """
    RF signal classifier using PyTorch with GPU acceleration.
    Supports hierarchical classification: device/protocol identification.
    
    Supports:
    - CUDA execution (GPU-accelerated inference)
    - CPU execution (fallback)
    - Hierarchical labels (device + signal_type)
    - Class label mapping from class_labels.json
    
    Input: 512x512 float32 spectrogram (noise-floor normalized)
    Output: Classification with confidence scores, device name, signal type
    """
    
    def __init__(self, model_path: Optional[str] = None, num_classes: int = 23, 
                 class_labels_path: Optional[str] = None, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch not installed. "
                "Install with: pip3 install torch"
            )
        
        # Load class labels mapping
        if class_labels_path is None:
            class_labels_path = Path(__file__).parent / "models" / "class_labels.json"
        else:
            class_labels_path = Path(class_labels_path)
        
        self.class_labels = None
        self.index_to_class = {}
        self.class_to_index = {}
        if class_labels_path.exists():
            try:
                with open(class_labels_path, 'r') as f:
                    self.class_labels = json.load(f)
                self.index_to_class = self.class_labels.get("index_to_class", {})
                self.class_to_index = self.class_labels.get("class_to_index", {})
                # Update num_classes from labels file if available
                if "num_classes" in self.class_labels:
                    num_classes = self.class_labels["num_classes"]
                print(f"[PyTorch] Loaded class labels from: {class_labels_path}")
                print(f"[PyTorch] Number of classes: {num_classes}")
            except Exception as e:
                print(f"[PyTorch] Warning: Failed to load class labels: {e}")
                print(f"[PyTorch] Using default class mapping (class_0, class_1, ...)")
        else:
            print(f"[PyTorch] Class labels file not found: {class_labels_path}")
            print(f"[PyTorch] Using default class mapping (class_0, class_1, ...)")
        
        # Determine device (GPU if available, else CPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"[PyTorch] Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"[PyTorch] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[PyTorch] CUDA version: {torch.version.cuda}")
        
        # Load or create model
        # Create model on CPU first (more reliable for loading)
        if model_path and Path(model_path).exists():
            print(f"[PyTorch] Loading model from: {model_path}")
            try:
                # Load to CPU first
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Full checkpoint with metadata
                    self.num_classes = checkpoint.get('num_classes', num_classes)
                    self.model = RFClassifier(num_classes=self.num_classes)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"[PyTorch] Loaded checkpoint with {self.num_classes} classes")
                elif isinstance(checkpoint, dict) and any(k.startswith('conv') or k.startswith('fc') for k in checkpoint.keys()):
                    # State dict only
                    self.num_classes = num_classes
                    self.model = RFClassifier(num_classes=self.num_classes)
                    self.model.load_state_dict(checkpoint)
                    print(f"[PyTorch] Loaded state dict")
                else:
                    # Assume it's a full model
                    self.model = checkpoint
                    self.num_classes = num_classes
                    print(f"[PyTorch] Loaded full model")
            except Exception as e:
                print(f"[PyTorch] Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                print(f"[PyTorch] Creating new model with {num_classes} classes")
                self.model = RFClassifier(num_classes=self.num_classes)
        else:
            if model_path:
                print(f"[PyTorch] Model file not found: {model_path}")
            print(f"[PyTorch] Creating new model with {num_classes} classes")
            self.model = RFClassifier(num_classes=num_classes)
        
        # Set to eval mode first
        self.model.eval()
        
        # Move model to device (try GPU, fallback to CPU if needed)
        try:
            if self.device.type == "cuda" and torch.cuda.is_available():
                # Try moving to GPU
                self.model = self.model.to(self.device)
                # Test GPU with a small tensor to verify it works
                test_tensor = torch.randn(1, 1, 64, 64).to(self.device)
                _ = self.model(test_tensor)
                print(f"[PyTorch] Model successfully moved to GPU")
            else:
                # Use CPU
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                print(f"[PyTorch] Using CPU (GPU not available or failed)")
        except RuntimeError as e:
            print(f"[PyTorch] Warning: GPU operation failed: {e}")
            print(f"[PyTorch] Falling back to CPU")
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
        
        # Get model info
        self.num_classes = self.num_classes if hasattr(self, 'num_classes') else num_classes
        print(f"[PyTorch] Model initialized: {self.num_classes} classes")
        print(f"[PyTorch] Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
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
            - model: "pytorch"
            - device: "cuda" or "cpu"
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
        
        # Convert to tensor and move to device
        x_tensor = torch.from_numpy(x).to(self.device)
        
        # Run inference
        try:
            with torch.no_grad():
                outputs = self.model(x_tensor)
        except Exception as e:
            print(f"[PyTorch] Inference error: {e}")
            raise
        
        # Get output (logits)
        logits = outputs.cpu().numpy()
        
        # Handle batch dimension
        if logits.ndim > 1:
            logits = logits[0]  # Take first batch item
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        # Get top prediction
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        
        # Map index to class label
        class_id = str(idx)
        if class_id in self.index_to_class:
            label = self.index_to_class[class_id]
            # Get full class info if available
            class_info = None
            if self.class_labels and "class_mapping" in self.class_labels:
                class_info = self.class_labels["class_mapping"].get(class_id, {})
        else:
            label = f"class_{idx}"
            class_info = None
        
        # Get signal type from class info
        signal_type = None
        device_name = None
        if class_info:
            signal_type = class_info.get("signal_type")
            device_name = class_info.get("name", label)
        
        # Get top-k (top 5)
        top_k = min(5, len(probs))
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        # Build top-k with labels
        topk = []
        for i in top_indices:
            top_class_id = str(i)
            if top_class_id in self.index_to_class:
                top_label = self.index_to_class[top_class_id]
                top_info = None
                if self.class_labels and "class_mapping" in self.class_labels:
                    top_info = self.class_labels["class_mapping"].get(top_class_id, {})
                topk.append({
                    "label": top_label,
                    "name": top_info.get("name", top_label) if top_info else top_label,
                    "signal_type": top_info.get("signal_type") if top_info else None,
                    "p": float(probs[i])
                })
            else:
                topk.append({
                    "label": f"class_{i}",
                    "name": f"class_{i}",
                    "signal_type": None,
                    "p": float(probs[i])
                })
        
        result = {
            "label": label,
            "confidence": confidence,
            "topk": topk,
            "model": "pytorch",
            "device": self.device.type
        }
        
        # Add hierarchical information if available
        if device_name:
            result["device_name"] = device_name
        if signal_type:
            result["signal_type"] = signal_type
        if class_info:
            result["description"] = class_info.get("description")
        
        return result
    
    def save_model(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
        }, path)
        print(f"[PyTorch] Model saved to: {path}")

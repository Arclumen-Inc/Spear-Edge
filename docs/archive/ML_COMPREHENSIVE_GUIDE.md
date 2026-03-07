# SPEAR-Edge ML System: Comprehensive Guide

**Last Updated**: 2025-03-02  
**Version**: 1.0

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [How ML Inference Works](#how-ml-inference-works)
4. [How Training Works](#how-training-works)
5. [Data Preparation](#data-preparation)
6. [Adding New Classes and Training Data](#adding-new-classes-and-training-data)
7. [Model Management](#model-management)
8. [Integration with Capture System](#integration-with-capture-system)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## Overview

### What is SPEAR-Edge ML?

SPEAR-Edge ML is a **hierarchical RF signal classification system** that identifies devices and protocols from captured RF signals. It uses deep learning (PyTorch) to analyze spectrograms and classify signals into specific device types (e.g., "ELRS", "DJI Mini 4 Pro") with confidence scores.

### Key Capabilities

- **Device/Protocol Identification**: Classifies RF signals into 23+ device/protocol classes
- **Hierarchical Classification**: Provides both device name and signal type (e.g., "FHSS Control", "Digital Video")
- **GPU Acceleration**: Uses CUDA for fast inference on Jetson Orin Nano
- **Confidence Scoring**: Returns probability scores for top-k predictions
- **Extensible**: Easy to add new classes through fine-tuning

### Current Status

- ✅ **Phase 1 (Classification)**: Infrastructure complete, training in progress
- 📋 **Phase 2 (Detection)**: Planned for future (standalone operation)

---

## System Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    SPEAR-Edge ML System                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │   Capture    │──────▶│ Spectrogram │                   │
│  │   Manager    │      │  Generator   │                   │
│  └──────────────┘      └──────┬───────┘                   │
│                                │                            │
│                                ▼                            │
│                        ┌──────────────┐                    │
│                        │   PyTorch     │                    │
│                        │  Classifier   │                    │
│                        │   (GPU/CUDA)  │                    │
│                        └──────┬───────┘                    │
│                                │                            │
│                                ▼                            │
│                        ┌──────────────┐                    │
│                        │ Classification│                    │
│                        │    Results    │                    │
│                        └──────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
spear_edge/ml/
├── __init__.py                    # Module exports
├── infer_pytorch.py               # PyTorch classifier (main)
├── infer_onnx.py                  # ONNX classifier (optional)
├── infer_stub.py                 # Stub classifier (fallback)
└── models/
    ├── class_labels.json          # Class definitions and mapping
    ├── rf_classifier.pth          # Trained model (if available)
    └── rf_classifier_dummy.pth   # Dummy model (for testing)

scripts/
├── train_rf_classifier.py         # Main training script
├── prepare_training_dataset.py    # Data preparation pipeline
├── fine_tune_new_class.py         # Fine-tuning for new classes
└── download_rfuav_dataset.py      # RFUAV dataset downloader

data/
├── dataset/                       # Prepared training dataset
│   ├── train/                     # Training samples
│   ├── val/                       # Validation samples
│   ├── test/                      # Test samples
│   └── manifest.json              # Dataset manifest
└── artifacts/captures/             # Capture artifacts (for training)
```

---

## How ML Inference Works

### Overview

When a capture is performed, the ML system:

1. **Generates Spectrogram**: Converts IQ samples to 512×512 spectrogram
2. **Loads Model**: Loads PyTorch model (GPU-accelerated)
3. **Runs Inference**: Classifies spectrogram into device/protocol classes
4. **Returns Results**: Provides top-k predictions with confidence scores

### Step-by-Step Process

#### 1. Spectrogram Generation

**Location**: `spear_edge/core/capture/spectrogram.py`

```python
# From IQ samples (complex64) to spectrogram (512×512 float32)
spec_ml = compute_spectrogram_chunked(
    iq_path=iq_path,
    sample_rate_sps=10_000_000,
    fft_size=1024,
    hop_size=256,
    chunk_size_samples=5_000_000
)
```

**Process**:
- Reads IQ file in chunks (memory-efficient)
- Computes FFT with Hanning window
- Converts power to dB scale
- Downsamples to 512×512 (time and frequency)
- Normalizes to noise floor (median subtraction)

**Output Format**:
- Shape: `(512, 512)` - time bins × frequency bins
- dtype: `float32`
- Units: Relative dB (noise-floor normalized)
- Size: ~1 MB per spectrogram

#### 2. Model Loading

**Location**: `spear_edge/ml/infer_pytorch.py`

```python
classifier = PyTorchRfClassifier(
    model_path="spear_edge/ml/models/rf_classifier.pth",
    num_classes=23,
    device="cuda"  # or "cpu"
)
```

**Process**:
- Loads PyTorch model checkpoint
- Loads class labels from `class_labels.json`
- Moves model to GPU (if CUDA available)
- Sets model to evaluation mode

**Model Architecture**:
```
Input: (1, 1, 512, 512) float32 spectrogram
  ↓
Conv2d(1→32) + ReLU + MaxPool
  ↓
Conv2d(32→64) + ReLU + MaxPool
  ↓
Conv2d(64→128) + ReLU + MaxPool
  ↓
Conv2d(128→256) + ReLU + MaxPool
  ↓
AdaptiveAvgPool2d(8×8)
  ↓
Flatten → FC(16384→1024) + ReLU + Dropout
  ↓
FC(1024→512) + ReLU + Dropout
  ↓
FC(512→23) → Logits
  ↓
Softmax → Probabilities
```

#### 3. Inference

```python
result = classifier.classify(spec_ml)
```

**Process**:
- Converts numpy array to PyTorch tensor
- Moves tensor to GPU
- Runs forward pass (no gradients)
- Applies softmax to get probabilities
- Returns top-k predictions

**Output Format**:
```python
{
    "label": "elrs",                    # Predicted class ID
    "confidence": 0.95,                  # Confidence score [0-1]
    "device_name": "ExpressLRS",        # Human-readable name
    "signal_type": "fhss_control",      # Signal category
    "description": "ExpressLRS control link",
    "topk": [                           # Top-5 predictions
        {"label": "elrs", "name": "ExpressLRS", "p": 0.95},
        {"label": "frsky", "name": "FrSky", "p": 0.03},
        ...
    ],
    "model": "pytorch",
    "device": "cuda"
}
```

#### 4. Integration with Capture System

**Location**: `spear_edge/core/capture/capture_manager.py`

```python
# After capture completes
if triage.get("signal_present") and not triage.get("likely_noise"):
    classification = self.classifier.classify(spec_ml)
    # Results saved to capture.json
```

**When Classification Runs**:
- Only if signal is present (not noise)
- Only if signal is not likely noise
- Runs automatically after capture completes
- Results stored in `capture.json` metadata

---

## How Training Works

### Overview

Training the ML model involves:

1. **Data Preparation**: Organize spectrograms into train/val/test splits
2. **Model Initialization**: Create or load base model
3. **Training Loop**: Iterate over batches, compute loss, update weights
4. **Validation**: Evaluate on validation set
5. **Model Saving**: Save best model checkpoint

### Training Script

**Location**: `scripts/train_rf_classifier.py`

**Usage**:
```bash
python3 scripts/train_rf_classifier.py \
    --dataset-dir data/dataset \
    --output-dir spear_edge/ml/models \
    --batch-size 2 \
    --epochs 50 \
    --device cuda \
    --learning-rate 0.001
```

### Training Process

#### 1. Dataset Loading

```python
train_dataset = SpectrogramDataset(
    dataset_dir=Path("data/dataset"),
    split="train",
    transform=RandomFlip()  # Data augmentation
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)
```

**Dataset Structure**:
```
data/dataset/
├── train/
│   ├── elrs/
│   │   ├── sample_000000.npy
│   │   ├── sample_000001.npy
│   │   └── ...
│   ├── dji_mini_4_pro/
│   │   └── ...
│   └── ...
├── val/
│   └── (same structure)
├── test/
│   └── (same structure)
└── manifest.json
```

#### 2. Model Initialization

```python
model = RFClassifier(num_classes=23)
model = model.to(device)  # Move to GPU

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

#### 3. Training Loop

```python
for epoch in range(epochs):
    model.train()  # Set to training mode
    
    for batch_idx, (spectrograms, labels) in enumerate(train_loader):
        # Move to device
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step()
```

#### 4. Validation

```python
model.eval()  # Set to evaluation mode
with torch.no_grad():
    for spectrograms, labels in val_loader:
        outputs = model(spectrograms)
        # Compute accuracy, loss, etc.
```

#### 5. Model Saving

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': 23,
    'epoch': epoch,
    'loss': loss,
    'accuracy': accuracy
}, 'spear_edge/ml/models/rf_classifier.pth')
```

### Training Parameters

**Recommended Settings (Jetson Orin Nano)**:
- **Batch Size**: 2-4 (limited by GPU memory)
- **Learning Rate**: 0.001 (Adam optimizer)
- **Epochs**: 50-100 (depending on dataset size)
- **Learning Rate Schedule**: CosineAnnealingLR
- **Data Augmentation**: Random horizontal flip

**CUDA Memory Optimization**:
```python
# Set before training
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

### Training Time Estimates

**On Jetson Orin Nano**:
- Small dataset (1000 samples): ~2-4 hours
- Medium dataset (10,000 samples): ~10-20 hours
- Large dataset (100,000 samples): ~50-100 hours

**On Desktop GPU (RTX 3080)**:
- Small dataset: ~10-20 minutes
- Medium dataset: ~1-2 hours
- Large dataset: ~5-10 hours

---

## Data Preparation

### Overview

Data preparation converts raw captures and external datasets into a unified training format.

### Data Sources

1. **SPEAR-Edge Captures**: Manual or automatic captures from Edge system
2. **RFUAV Dataset**: Public drone detection dataset from Hugging Face
3. **Custom Captures**: User-provided captures

### Data Preparation Script

**Location**: `scripts/prepare_training_dataset.py`

**Usage**:
```bash
python3 scripts/prepare_training_dataset.py \
    --rfuav-dir /path/to/RFUAV/Dataset \
    --spear-dir data/dataset_raw \
    --output-dir data/dataset \
    --class-labels spear_edge/ml/models/class_labels.json
```

### Process

#### 1. Load Class Labels

```python
class_labels = load_class_labels("spear_edge/ml/models/class_labels.json")
# Contains: class_mapping, class_to_index, index_to_class
```

#### 2. Process RFUAV Dataset

```python
rfuav_samples = process_rfuav_dataset(
    rfuav_dir=Path("/path/to/RFUAV"),
    class_labels=class_labels,
    output_base=Path("data/dataset")
)
```

**Process**:
- Converts RFUAV images (JPG/PNG) to spectrograms (512×512 float32)
- Maps RFUAV drone names to internal class IDs
- Normalizes to noise floor
- Saves as `.npy` files

#### 3. Process SPEAR-Edge Captures

```python
spear_samples = process_spear_captures(
    spear_dir=Path("data/dataset_raw"),
    class_labels=class_labels
)
```

**Process**:
- Reads `.npy` spectrograms from captures
- Extracts class labels from `capture.json`
- Validates format (512×512, float32)
- Maps to class IDs

#### 4. Organize Dataset

```python
organize_dataset(
    all_samples={...},
    output_dir=Path("data/dataset"),
    class_labels=class_labels,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

**Process**:
- Validates all spectrograms
- Splits into train/val/test (80/10/10)
- Creates directory structure
- Generates `manifest.json`

### Dataset Format

**Spectrogram Format**:
- File: `.npy` (NumPy binary)
- Shape: `(512, 512)` - time × frequency
- dtype: `float32`
- Units: Relative dB (noise-floor normalized)
- Size: ~1 MB per file

**Directory Structure**:
```
data/dataset/
├── train/
│   ├── elrs/
│   │   ├── sample_000000.npy
│   │   └── ...
│   └── dji_mini_4_pro/
│       └── ...
├── val/
│   └── (same structure)
├── test/
│   └── (same structure)
└── manifest.json
```

**Manifest Format**:
```json
{
  "version": "1.0",
  "num_classes": 23,
  "splits": {
    "train": [
      {
        "path": "train/elrs/sample_000000.npy",
        "class_id": "elrs",
        "class_index": 0
      },
      ...
    ],
    "val": [...],
    "test": [...]
  }
}
```

---

## Adding New Classes and Training Data

### Overview

To add a new device/protocol class:

1. **Add Class Definition**: Update `class_labels.json`
2. **Collect Training Data**: Capture examples of the new signal
3. **Prepare Data**: Run data preparation script
4. **Fine-Tune Model**: Use fine-tuning script or full retraining

### Step 1: Add Class Definition

**Edit**: `spear_edge/ml/models/class_labels.json`

```json
{
  "num_classes": 24,  // Increment from 23
  "class_mapping": {
    "23": {  // New class index
      "id": "new_protocol",
      "name": "New Protocol",
      "signal_type": "fhss_control",
      "description": "Description of new protocol"
    }
  },
  "class_to_index": {
    "new_protocol": 23
  },
  "index_to_class": {
    "23": "new_protocol"
  }
}
```

### Step 2: Collect Training Data

**Option A: Manual Captures**
```bash
# Use Edge UI or API to capture signals
# Captures saved to: data/artifacts/captures/
```

**Option B: External Dataset**
```bash
# Download or prepare external dataset
# Convert to SPEAR-Edge format (512×512 float32)
```

**Minimum Requirements**:
- **Training samples**: 50-100 per class (minimum)
- **Recommended**: 200-500 per class
- **Validation samples**: 10-20% of training
- **Test samples**: 10-20% of training

### Step 3: Prepare Data

```bash
# Add new captures to data/dataset_raw/
# Organize by class (if possible)

python3 scripts/prepare_training_dataset.py \
    --spear-dir data/dataset_raw \
    --output-dir data/dataset \
    --class-labels spear_edge/ml/models/class_labels.json
```

### Step 4: Fine-Tune Model

**Location**: `scripts/fine_tune_new_class.py`

**Usage**:
```bash
python3 scripts/fine_tune_new_class.py \
    --base-model spear_edge/ml/models/rf_classifier.pth \
    --dataset-dir data/dataset \
    --output-model spear_edge/ml/models/rf_classifier_finetuned.pth \
    --new-classes new_protocol \
    --epochs 20 \
    --freeze-features  # Freeze feature extraction layers
```

**Process**:
1. Loads existing model
2. Extends classification head for new classes
3. Optionally freezes feature extraction layers
4. Trains only on new class data (or all data)
5. Saves fine-tuned model

**Alternative: Full Retraining**
```bash
# Retrain entire model with all classes
python3 scripts/train_rf_classifier.py \
    --dataset-dir data/dataset \
    --output-dir spear_edge/ml/models \
    --batch-size 2 \
    --epochs 50 \
    --device cuda
```

### Fine-Tuning vs Full Retraining

**Fine-Tuning** (Recommended for new classes):
- ✅ Faster (trains only new layers)
- ✅ Preserves existing knowledge
- ✅ Requires less data
- ⚠️ May not learn complex new patterns

**Full Retraining**:
- ✅ Better performance (if enough data)
- ✅ Learns all relationships
- ⚠️ Slower (trains entire model)
- ⚠️ Requires data for all classes

---

## Model Management

### Model Files

**Trained Model**: `spear_edge/ml/models/rf_classifier.pth`
- PyTorch checkpoint format
- Contains: `model_state_dict`, `num_classes`, metadata
- Size: ~10-50 MB (depending on architecture)

**Dummy Model**: `spear_edge/ml/models/rf_classifier_dummy.pth`
- Untrained model (for testing)
- Same architecture, random weights
- Used when trained model unavailable

**Class Labels**: `spear_edge/ml/models/class_labels.json`
- Class definitions and mappings
- Must match model's `num_classes`

### Model Loading Priority

The capture manager loads models in this order:

1. **PyTorch Model** (`rf_classifier.pth`) - GPU-accelerated
2. **Dummy Model** (`rf_classifier_dummy.pth`) - Fallback
3. **Stub Classifier** - Always returns "unknown"

### Model Deployment

**To Deploy New Model**:
```bash
# Copy trained model to Jetson
scp rf_classifier.pth user@jetson:/home/spear/spear-edgev1_0/spear_edge/ml/models/

# Update class_labels.json if needed
scp class_labels.json user@jetson:/home/spear/spear-edgev1_0/spear_edge/ml/models/

# Restart Edge application
sudo systemctl restart spear-edge  # or however you run Edge
```

### Model Versioning

**Recommended Naming**:
```
rf_classifier_v1.0.pth      # Version 1.0 (23 classes)
rf_classifier_v1.1.pth      # Version 1.1 (24 classes, added new protocol)
rf_classifier_v2.0.pth      # Version 2.0 (major architecture change)
```

**Backup Strategy**:
- Keep previous model versions
- Document model changes in commit messages
- Tag releases with model versions

---

## Integration with Capture System

### Automatic Classification

**When Classification Runs**:
- After capture completes
- Only if signal is present (not noise)
- Only if signal is not likely noise
- Runs automatically (no user intervention)

**Location**: `spear_edge/core/capture/capture_manager.py`

```python
# After spectrogram generation
if triage.get("signal_present") and not triage.get("likely_noise"):
    if self.classifier is not None:
        classification = self.classifier.classify(spec_ml)
        # Results saved to capture.json
```

### Classification Results Storage

**Location**: `capture.json` in capture directory

```json
{
  "classification": {
    "label": "elrs",
    "confidence": 0.95,
    "device_name": "ExpressLRS",
    "signal_type": "fhss_control",
    "description": "ExpressLRS control link",
    "topk": [
      {"label": "elrs", "name": "ExpressLRS", "p": 0.95},
      {"label": "frsky", "name": "FrSky", "p": 0.03},
      ...
    ],
    "model": "pytorch",
    "device": "cuda"
  }
}
```

### Manual Classification

**Using Python**:
```python
from spear_edge.ml.infer_pytorch import PyTorchRfClassifier
import numpy as np

# Load classifier
classifier = PyTorchRfClassifier("spear_edge/ml/models/rf_classifier.pth")

# Load spectrogram
spec = np.load("path/to/spectrogram.npy")

# Classify
result = classifier.classify(spec)
print(f"Predicted: {result['device_name']} ({result['confidence']:.2%})")
```

**Using API** (if implemented):
```bash
curl -X POST http://localhost:8080/api/ml/classify \
    -H "Content-Type: application/json" \
    -d '{"spectrogram_path": "path/to/spectrogram.npy"}'
```

---

## Performance Optimization

### GPU Acceleration

**CUDA Setup**:
- PyTorch with CUDA support (installed)
- cuDNN 8.9.5 (for Jetson)
- CUDA memory optimization configured

**Performance**:
- **GPU Inference**: ~10-50 ms per spectrogram
- **CPU Inference**: ~100-500 ms per spectrogram
- **Speedup**: ~10× faster on GPU

### Memory Optimization

**CUDA Memory Config**:
```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
```

**Batch Size** (for training):
- Jetson Orin Nano: 2-4
- Desktop GPU: 16-32

**Model Size**:
- Current architecture: ~10-50 MB
- Can be reduced with quantization (future)

### Inference Optimization

**Optimizations Applied**:
- ✅ GPU acceleration (CUDA)
- ✅ Batch processing (if multiple spectrograms)
- ✅ No gradient computation (`torch.no_grad()`)
- ✅ Model in eval mode

**Future Optimizations**:
- Model quantization (INT8)
- TensorRT optimization
- ONNX Runtime with TensorRT

---

## Troubleshooting

### Common Issues

#### 1. Model Not Found

**Error**: `FileNotFoundError: rf_classifier.pth not found`

**Solution**:
- Check model path: `spear_edge/ml/models/rf_classifier.pth`
- Use dummy model for testing: `rf_classifier_dummy.pth`
- Train model if needed

#### 2. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size: `--batch-size 1`
- Clear CUDA cache: `torch.cuda.empty_cache()`
- Use CPU: `device="cpu"`

#### 3. Class Mismatch

**Error**: `RuntimeError: size mismatch for fc3.weight`

**Solution**:
- Check `num_classes` in model matches `class_labels.json`
- Retrain model with correct number of classes

#### 4. Low Confidence Scores

**Symptoms**: All predictions have low confidence (<0.5)

**Possible Causes**:
- Model not trained or poorly trained
- Spectrogram format mismatch
- Signal not in training data

**Solution**:
- Retrain model with better data
- Verify spectrogram format (512×512, float32)
- Add more training examples

#### 5. Classification Always Returns "unknown"

**Possible Causes**:
- Model not loaded (using stub classifier)
- Model path incorrect
- Model file corrupted

**Solution**:
- Check capture manager logs for model loading errors
- Verify model file exists and is valid
- Check `class_labels.json` matches model

### Debugging Tips

**Enable Verbose Logging**:
```python
# In capture_manager.py
print(f"[CAPTURE] Classifier: {self.classifier}")
print(f"[CAPTURE] Classification: {classification}")
```

**Test Classifier Directly**:
```python
from spear_edge.ml.infer_pytorch import PyTorchRfClassifier
import numpy as np

classifier = PyTorchRfClassifier("spear_edge/ml/models/rf_classifier.pth")
spec = np.load("test_spectrogram.npy")
result = classifier.classify(spec)
print(result)
```

**Check GPU Availability**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

---

## Best Practices

### Data Collection

1. **Diversity**: Collect samples under various conditions
   - Different power levels
   - Different distances
   - Different environments
   - Different times of day

2. **Quality**: Ensure high-quality captures
   - Good SNR (>15 dB)
   - No clipping
   - Proper gain settings
   - Full signal bandwidth

3. **Balance**: Balance classes in dataset
   - Similar number of samples per class
   - Avoid class imbalance (>10:1 ratio)

### Training

1. **Validation**: Always use validation set
   - Monitor validation loss
   - Stop early if overfitting
   - Use best validation model

2. **Augmentation**: Use data augmentation
   - Random horizontal flip
   - Noise injection (future)
   - Time/frequency shifts (future)

3. **Hyperparameters**: Tune carefully
   - Learning rate: 0.001-0.0001
   - Batch size: As large as memory allows
   - Epochs: Until validation loss plateaus

### Model Deployment

1. **Testing**: Test before deployment
   - Validate on test set
   - Test on real captures
   - Verify performance metrics

2. **Versioning**: Version models
   - Tag model versions
   - Document changes
   - Keep backups

3. **Monitoring**: Monitor in production
   - Track classification accuracy
   - Monitor confidence scores
   - Log failures

### Performance

1. **GPU Usage**: Always use GPU when available
   - Faster inference
   - Better throughput
   - Lower CPU usage

2. **Batch Processing**: Process multiple spectrograms together
   - More efficient GPU usage
   - Faster overall processing

3. **Memory Management**: Manage GPU memory
   - Clear cache periodically
   - Use appropriate batch sizes
   - Monitor memory usage

---

## Quick Reference

### Training a New Model

```bash
# 1. Prepare data
python3 scripts/prepare_training_dataset.py \
    --spear-dir data/dataset_raw \
    --output-dir data/dataset

# 2. Train model
python3 scripts/train_rf_classifier.py \
    --dataset-dir data/dataset \
    --output-dir spear_edge/ml/models \
    --batch-size 2 \
    --epochs 50 \
    --device cuda

# 3. Test model
python3 -c "
from spear_edge.ml.infer_pytorch import PyTorchRfClassifier
import numpy as np
c = PyTorchRfClassifier('spear_edge/ml/models/rf_classifier.pth')
spec = np.load('test.npy')
print(c.classify(spec))
"
```

### Adding a New Class

```bash
# 1. Update class_labels.json (add new class)

# 2. Collect training data (captures)

# 3. Prepare dataset
python3 scripts/prepare_training_dataset.py \
    --spear-dir data/dataset_raw \
    --output-dir data/dataset

# 4. Fine-tune model
python3 scripts/fine_tune_new_class.py \
    --base-model spear_edge/ml/models/rf_classifier.pth \
    --dataset-dir data/dataset \
    --output-model spear_edge/ml/models/rf_classifier.pth \
    --new-classes new_class_name \
    --epochs 20
```

### Testing Classification

```python
from spear_edge.ml.infer_pytorch import PyTorchRfClassifier
import numpy as np

# Load classifier
classifier = PyTorchRfClassifier("spear_edge/ml/models/rf_classifier.pth")

# Load spectrogram
spec = np.load("data/artifacts/captures/.../features/spectrogram.npy")

# Classify
result = classifier.classify(spec)

# Print results
print(f"Predicted: {result['device_name']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Signal Type: {result['signal_type']}")
print("\nTop-5 Predictions:")
for pred in result['topk']:
    print(f"  {pred['name']}: {pred['p']:.2%}")
```

---

## Related Documents

- `docs/ML_ROADMAP.md` - ML implementation roadmap
- `docs/FINE_TUNING_GUIDE.md` - Fine-tuning workflow
- `docs/ML_DETECTION_PHASE.md` - Future detection phase
- `technical_data_package/ML_INFERENCE_PLAN.txt` - Original ML plan

---

## Appendix

### Class List (Current)

**FHSS Control** (6 classes):
- elrs, crossfire, frsky, flysky, ghost, redpine

**Digital Video** (10 classes):
- dji_mini_4_pro, dji_avata, dji_fpv, dji_mini_3, dji_air_3, dji_mavic_3, walksnail, hdzero, dji_o3, dji_o4

**Analog Video** (1 class):
- analog_fpv

**Control Protocols** (3 classes):
- mavlink, mavlink2, lora_telemetry

**Voice/PTT** (2 classes):
- voice_analog, voice_digital

**Unknown** (1 class):
- unknown

**Total**: 23 classes

---

**End of Document**

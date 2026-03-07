# Training Next Steps

**Last Updated**: 2025-03-01  
**Status**: Ready to prepare dataset and train

---

## Current Status

✅ **RFUAV Dataset**: Downloaded (5,679 images, 37 classes)  
✅ **Data Preparation Script**: Ready (`scripts/prepare_training_dataset.py`)  
⏳ **Training Script**: Needs to be created  
⏳ **Model Training**: Pending

---

## Next Steps

### Step 1: Prepare Training Dataset (5-10 minutes)

Combine RFUAV dataset with SPEAR-Edge captures:

```bash
python3 scripts/prepare_training_dataset.py \
    --rfuav-dir data/rfuav \
    --spear-dir data/dataset_raw \
    --output-dir data/dataset \
    --class-labels spear_edge/ml/models/class_labels.json
```

**What this does**:
- Converts RFUAV images to spectrogram arrays (512x512)
- Maps RFUAV drone names to our class labels
- Combines with SPEAR-Edge captures (if any)
- Splits into train/val/test (80/10/10)
- Creates `manifest.json` with dataset metadata

**Output**: `data/dataset/` with organized train/val/test splits

---

### Step 2: Create Training Script

Need to create `scripts/train_rf_classifier.py` with:
- PyTorch DataLoader for spectrograms
- Training loop (loss, optimizer, metrics)
- Validation loop
- Model checkpointing
- Export to `.pth` format

**Estimated time to create**: 30-60 minutes

---

### Step 3: Train Model

#### Option A: Train on Jetson (Recommended for initial test)

**Time Estimate**: 18-30 minutes for 50 epochs

**Pros**:
- ✅ Already set up with CUDA/PyTorch
- ✅ Can test immediately after training
- ✅ No data transfer needed
- ✅ Fast enough for initial model

**Cons**:
- ⚠️ Slower than desktop GPU (5-10x)
- ⚠️ Uses Jetson resources (may impact other tasks)
- ⚠️ Limited batch size (16 vs 32+ on desktop)

**Command**:
```bash
python3 scripts/train_rf_classifier.py \
    --dataset-dir data/dataset \
    --output-dir spear_edge/ml/models \
    --batch-size 16 \
    --epochs 50 \
    --device cuda
```

#### Option B: Train on Development Machine (Recommended for production)

**Time Estimate**: 2-5 minutes for 50 epochs (on RTX 3060/4090)

**Pros**:
- ✅ Much faster (5-10x speedup)
- ✅ Larger batch sizes (32-64)
- ✅ Doesn't tie up Jetson
- ✅ Better for experimentation

**Cons**:
- ⚠️ Need to transfer model to Jetson
- ⚠️ Need development machine with GPU

**Command** (on dev machine):
```bash
python3 scripts/train_rf_classifier.py \
    --dataset-dir /path/to/dataset \
    --output-dir models \
    --batch-size 32 \
    --epochs 50 \
    --device cuda
```

---

## Training Time Estimates

### Jetson Orin Nano

| Configuration | Time per Epoch | Total (50 epochs) |
|--------------|----------------|-------------------|
| Batch size 16 | ~20-25 seconds | **18-25 minutes** |
| Batch size 8  | ~35-40 seconds | **30-35 minutes** |

**Factors affecting time**:
- Data loading speed (SSD vs eMMC)
- GPU utilization
- Augmentation overhead
- Model complexity

### Desktop GPU (RTX 3060/4090)

| Configuration | Time per Epoch | Total (50 epochs) |
|--------------|----------------|-------------------|
| Batch size 32 | ~2-4 seconds | **2-5 minutes** |
| Batch size 64 | ~1-2 seconds | **1-3 minutes** |

---

## Recommendation

### For Initial Model (First Training)
**Train on Jetson** - Fast enough (18-30 min), no transfer needed, can test immediately

### For Production/Iteration
**Train on Development Machine** - Much faster, better for experimentation and hyperparameter tuning

### Hybrid Approach
1. Train initial model on Jetson (verify pipeline works)
2. Train production model on dev machine (faster iteration)
3. Transfer trained model to Jetson for deployment

---

## Training Configuration

### Model Architecture
- **Input**: 512x512 spectrogram (float32)
- **Architecture**: 4-layer CNN (32→64→128→256 channels)
- **Parameters**: ~4.3M
- **Output**: 23 classes (hierarchical classification)

### Hyperparameters
- **Batch size**: 16 (Jetson) or 32 (Desktop)
- **Learning rate**: 1e-3 with cosine annealing
- **Epochs**: 50-100 (early stopping)
- **Optimizer**: AdamW
- **Loss**: CrossEntropyLoss
- **Weight decay**: 1e-4
- **Dropout**: 0.5

### Data Augmentation
- Random horizontal flip (50%)
- Small rotations (±5 degrees)
- Noise injection (optional)
- Time/frequency shifts (optional)

---

## After Training

1. **Validate Model**:
   ```bash
   python3 scripts/validate_model.py \
       --model-path spear_edge/ml/models/rf_classifier.pth \
       --test-dir data/dataset/test
   ```

2. **Test Inference**:
   ```bash
   python3 -c "
   from spear_edge.ml.infer_pytorch import PyTorchRfClassifier
   import numpy as np
   
   classifier = PyTorchRfClassifier('spear_edge/ml/models/rf_classifier.pth')
   spec = np.load('data/dataset/test/elrs/sample_000000.npy')
   result = classifier.classify(spec)
   print(result)
   "
   ```

3. **Deploy**: Model automatically used by capture manager

---

## Files Needed

### Existing ✅
- `scripts/prepare_training_dataset.py` - Dataset preparation
- `spear_edge/ml/infer_pytorch.py` - Model architecture
- `spear_edge/ml/models/class_labels.json` - Class mappings

### Need to Create ⏳
- `scripts/train_rf_classifier.py` - Training script
- `scripts/validate_model.py` - Model validation (optional)

---

## Quick Start (After Training Script Created)

```bash
# 1. Prepare dataset
python3 scripts/prepare_training_dataset.py \
    --rfuav-dir data/rfuav \
    --output-dir data/dataset

# 2. Train model (on Jetson)
python3 scripts/train_rf_classifier.py \
    --dataset-dir data/dataset \
    --output-dir spear_edge/ml/models \
    --batch-size 16 \
    --epochs 50

# 3. Test
python3 -c "from spear_edge.ml.infer_pytorch import PyTorchRfClassifier; print('Model loaded!')"
```

---

## Notes

- **Jetson training is feasible**: 18-30 minutes is reasonable for initial model
- **Desktop training is faster**: Use for production/iteration
- **Model size**: ~17MB (PyTorch .pth format)
- **Inference speed**: ~17ms per spectrogram on Jetson GPU (already tested)

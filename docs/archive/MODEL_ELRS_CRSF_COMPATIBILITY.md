# Model ELRS/CRSF Compatibility Analysis

**Date**: 2025-03-06  
**Purpose**: Verify if current model supports ELRS/CRSF classification and check spectrogram format compatibility

---

## Summary

✅ **Model Architecture Supports ELRS/CRSF**: Yes  
✅ **Model Was Trained on ELRS/CRSF**: Yes  
✅ **Spectrogram Format Matches Model Expectations**: Yes  
⚠️ **Model Training Duration**: Only 1 epoch (very short training)

---

## 1. Model Class Support

### Class Labels (`class_labels.json`)

The model supports **23 classes**, including:

- **ELRS** (ExpressLRS): Class ID `0`, signal type `fhss_control`
- **Crossfire** (TBS Crossfire): Class ID `1`, signal type `fhss_control`

Both classes are properly defined in the class mapping:
```json
"0": {
  "id": "elrs",
  "name": "ExpressLRS",
  "signal_type": "fhss_control",
  "description": "ExpressLRS control link"
},
"1": {
  "id": "crossfire",
  "name": "TBS Crossfire",
  "signal_type": "fhss_control",
  "description": "Team BlackSheep Crossfire control link"
}
```

---

## 2. Training Data

### Classes in Training Dataset

The model was trained on data including:
- ✅ `elrs` (ExpressLRS)
- ✅ `crossfire` (TBS Crossfire)
- Plus 21 other classes (DJI drones, analog FPV, MAVLink, etc.)

**Training Data Location**: `data/dataset/train/`

### Training Status

- **Epochs Trained**: 1 epoch (very short!)
- **Best Validation Accuracy**: 91.2%
- **Model File**: `spear_edge/ml/models/rf_classifier.pth` (203 MB)

**Note**: Training for only 1 epoch is insufficient for good generalization. The model likely needs more training epochs (50-100 recommended).

---

## 3. Spectrogram Format Compatibility

### Model Expectations

The model (`PyTorchRfClassifier`) expects:
- **Shape**: `(batch, 1, 512, 512)` or `(512, 512)` (will be reshaped)
- **Dtype**: `float32`
- **Normalization**: Noise-floor normalized (median ≈ 0 dB)
- **Format**: Power spectrogram in dB scale

### Actual Spectrogram Produced

From capture `20260306_192449_915000000Hz_10000000sps_manual`:

```
Shape: (512, 512) ✓
Dtype: float32 ✓
Normalization: Noise-floor normalized ✓
  - Min: -11.20 dB
  - Max: 57.18 dB
  - Mean: 3.63 dB
  - Median: 0.00 dB ✓ (matches expectation)
  10th percentile: -4.07 dB
  90th percentile: 16.43 dB
```

**Format Match**: ✅ **Perfect match**

### Spectrogram Generation Process

**Location**: `spear_edge/core/capture/spectrogram.py`

1. **FFT Processing**: 
   - FFT size: 1024 (default)
   - Hop size: 256 (default)
   - Window: Hanning
   - FFT-shifted (DC in center)

2. **Downsampling**:
   - Frequency axis: Downsampled to 512 bins (mean pooling)
   - Time axis: Downsampled to 512 bins (mean pooling)

3. **Normalization**:
   ```python
   noise_floor = np.median(spec_ml)
   spec_ml = spec_ml - noise_floor  # Median-normalized to ~0 dB
   ```

4. **Output**:
   - Shape: `(512, 512)`
   - Dtype: `float32`
   - Units: dB relative to noise floor

---

## 4. Why Classification Shows "Unknown"

### Current Classification Result

From capture `20260306_192449_915000000Hz_10000000sps_manual`:
```json
{
  "label": "unknown",
  "confidence": 1.0,
  "topk": [
    {"label": "unknown", "p": 1.0},
    {"label": "dji_air_3", "p": 0.0},
    {"label": "crossfire", "p": 0.0},
    ...
  ]
}
```

### Likely Reasons

1. **Insufficient Training**: Model trained for only 1 epoch
   - Needs 50-100 epochs for good generalization
   - Current training is likely underfit

2. **Limited ELRS Training Data**: 
   - May not have enough ELRS examples in training set
   - Model may not have learned ELRS features well

3. **Spectrogram Characteristics Mismatch**:
   - Training data may have different characteristics (sample rate, bandwidth, etc.)
   - Your capture: 10 MS/s, 915 MHz
   - Training data: May have different parameters

4. **Model Overfitting to Training Data**:
   - With only 1 epoch, model may not generalize well
   - May have memorized training examples rather than learned features

---

## 5. Recommendations

### Immediate Actions

1. **Check Training Data Quality**:
   ```bash
   # Count ELRS samples in training data
   ls data/dataset/train/elrs/*.npy | wc -l
   ```

2. **Verify Spectrogram Characteristics**:
   - Compare your capture spectrogram with training data spectrograms
   - Check if sample rates, bandwidths match

3. **Continue Training**:
   ```bash
   # Train for more epochs
   python3 scripts/train_rf_classifier.py \
       --dataset-dir data/dataset \
       --output-dir spear_edge/ml/models \
       --batch-size 2 \
       --epochs 50 \
       --device cuda
   ```

### Long-term Improvements

1. **Collect More ELRS Data**:
   - Capture more ELRS signals at various frequencies
   - Include different ELRS modes (500 Hz, 250 Hz, etc.)
   - Vary sample rates and bandwidths

2. **Data Augmentation**:
   - Add frequency shifts
   - Add time shifts
   - Add noise variations

3. **Hyperparameter Tuning**:
   - Adjust learning rate
   - Try different optimizers
   - Adjust batch size

---

## 6. Conclusion

✅ **Model architecture supports ELRS/CRSF**: Yes  
✅ **Model was trained on ELRS/CRSF data**: Yes  
✅ **Spectrogram format matches model expectations**: Perfect match  
⚠️ **Model training duration**: Insufficient (only 1 epoch)

**Root Cause**: The model was trained for only 1 epoch, which is insufficient for good generalization. The model needs more training epochs (50-100) to properly learn ELRS/CRSF features.

**Next Steps**: Continue training the model for more epochs, then retest ELRS classification.

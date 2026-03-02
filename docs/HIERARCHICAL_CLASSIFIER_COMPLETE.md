# Hierarchical Classifier Implementation - Complete

## ✅ Status: ALL STEPS COMPLETE

All three steps have been successfully implemented for Option 1: Single Multi-Class Model.

---

## Step 1: Updated Classifier Architecture ✅

### Changes Made

1. **Enhanced CNN Architecture** (`spear_edge/ml/infer_pytorch.py`):
   - Added 4th convolutional layer (256 channels) for better feature extraction
   - Expanded fully connected layers: 16384 → 1024 → 512 → num_classes
   - Increased model capacity for 23-35 classes
   - Model parameters: ~17.7M (up from 4.3M)

2. **Hierarchical Classification Support**:
   - Loads `class_labels.json` automatically
   - Maps class indices to device/protocol names
   - Returns hierarchical information:
     - `label`: Class ID (e.g., "elrs", "dji_mini_4_pro")
     - `device_name`: Human-readable name (e.g., "ExpressLRS", "DJI Mini 4 Pro")
     - `signal_type`: Signal category (e.g., "fhss_control", "digital_video")
     - `description`: Full description

3. **Enhanced Output Format**:
   ```python
   {
       "label": "elrs",
       "confidence": 0.85,
       "device_name": "ExpressLRS",
       "signal_type": "fhss_control",
       "description": "ExpressLRS control link",
       "topk": [
           {
               "label": "elrs",
               "name": "ExpressLRS",
               "signal_type": "fhss_control",
               "p": 0.85
           },
           ...
       ],
       "model": "pytorch",
       "device": "cuda"
   }
   ```

---

## Step 2: Class Mapping File ✅

### Created: `spear_edge/ml/models/class_labels.json`

**23 Device/Protocol Classes:**

#### FHSS Control (6 classes):
- `elrs` → ExpressLRS
- `crossfire` → TBS Crossfire
- `frsky` → FrSky
- `flysky` → FlySky
- `ghost` → Ghost
- `redpine` → Redpine

#### Digital Video (10 classes):
- `dji_mini_4_pro` → DJI Mini 4 Pro
- `dji_avata` → DJI Avata
- `dji_fpv` → DJI FPV
- `dji_mini_3` → DJI Mini 3
- `dji_air_3` → DJI Air 3
- `dji_mavic_3` → DJI Mavic 3
- `walksnail` → Walksnail
- `hdzero` → HDZero
- `dji_o3` → DJI O3 Air Unit
- `dji_o4` → DJI O4 Air Unit

#### Analog Video (1 class):
- `analog_fpv` → Generic Analog FPV

#### Control Protocols (3 classes):
- `mavlink` → MAVLink
- `mavlink2` → MAVLink 2.0
- `lora_telemetry` → LoRa Telemetry

#### Voice/PTT (2 classes):
- `voice_analog` → Analog Voice
- `voice_digital` → Digital Voice

#### Unknown (1 class):
- `unknown` → Unknown Signal

### Features:
- **RFUAV Mapping**: Maps RFUAV drone names to class IDs
- **Index Mapping**: Bidirectional class_id ↔ index mapping
- **Signal Type Hierarchy**: Groups classes by signal type
- **Metadata**: Name, description, signal_type for each class

---

## Step 3: Data Preparation Pipeline ✅

### Created: `scripts/prepare_training_dataset.py`

### Features:

1. **RFUAV Dataset Processing**:
   - Converts spectrogram images (PNG/JPG) to NumPy arrays
   - Resizes to 512x512 if needed
   - Normalizes to dB scale
   - Maps RFUAV drone names to class IDs

2. **SPEAR-Edge Capture Processing**:
   - Reads existing spectrograms (already 512x512 float32)
   - Extracts class labels from `capture.json`
   - Supports classification-based labeling
   - Falls back to "unknown" if no label found

3. **Dataset Organization**:
   - Combines RFUAV + SPEAR-Edge samples
   - Splits into train/val/test (80/10/10 default)
   - Organizes by class ID
   - Validates spectrogram format

4. **Output Structure**:
   ```
   data/dataset/
   ├── train/
   │   ├── elrs/
   │   │   ├── sample_000000.npy
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

5. **Manifest Generation**:
   - JSON manifest with all samples
   - Class mappings
   - Split statistics

### Usage:

```bash
# Process RFUAV + SPEAR-Edge captures
python3 scripts/prepare_training_dataset.py \
    --rfuav-dir /path/to/RFUAV/Dataset \
    --spear-dir data/dataset_raw \
    --output-dir data/dataset \
    --class-labels spear_edge/ml/models/class_labels.json

# Process only SPEAR-Edge captures
python3 scripts/prepare_training_dataset.py \
    --spear-dir data/dataset_raw \
    --output-dir data/dataset

# Custom split ratios
python3 scripts/prepare_training_dataset.py \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

---

## Integration with Capture Manager

The classifier automatically:
1. Loads `class_labels.json` on initialization
2. Returns hierarchical classification results
3. Includes device name and signal type in output
4. Works with existing `capture_manager.py` integration

### Example Classification Output:

When Tripwire sends: `"classification": "FHSS-like"`

Edge classifier refines to:
```json
{
    "label": "elrs",
    "confidence": 0.80,
    "device_name": "ExpressLRS",
    "signal_type": "fhss_control",
    "description": "ExpressLRS control link",
    "topk": [
        {"label": "elrs", "name": "ExpressLRS", "p": 0.80},
        {"label": "crossfire", "name": "TBS Crossfire", "p": 0.15},
        ...
    ]
}
```

---

## Next Steps

### For Training:

1. **Download RFUAV Dataset**:
   - Get spectrograms from Hugging Face
   - Place in directory structure: `Dataset/train/{DRONE}/imgs/`

2. **Collect SPEAR-Edge Data**:
   - Continue capturing signals
   - Classified captures auto-export to `data/dataset_raw/`

3. **Prepare Dataset**:
   ```bash
   python3 scripts/prepare_training_dataset.py \
       --rfuav-dir /path/to/RFUAV \
       --spear-dir data/dataset_raw \
       --output-dir data/dataset
   ```

4. **Train Model** (on development machine):
   - Use PyTorch training script (to be created)
   - Train on `data/dataset/train/`
   - Validate on `data/dataset/val/`

5. **Deploy Trained Model**:
   - Replace `rf_classifier_dummy.pth` with trained model
   - Test on Jetson with real captures

---

## Files Created/Modified

### New Files:
- `spear_edge/ml/models/class_labels.json` - Class mapping (23 classes)
- `scripts/prepare_training_dataset.py` - Data preparation pipeline

### Modified Files:
- `spear_edge/ml/infer_pytorch.py` - Enhanced architecture + hierarchical classification

---

## Testing

✅ **Classifier Architecture**: Updated for 23-35 classes  
✅ **Class Labels**: 23 device/protocol classes defined  
✅ **Hierarchical Output**: Returns device name, signal type, description  
✅ **Data Pipeline**: Handles RFUAV + SPEAR-Edge formats  
✅ **Integration**: Works with existing capture manager  

---

## Summary

All three steps complete! The system now supports:
- ✅ Hierarchical classification (device + signal type)
- ✅ 23 device/protocol classes
- ✅ RFUAV dataset integration
- ✅ SPEAR-Edge capture processing
- ✅ Unified training dataset preparation

**Ready for**: Data collection → Training → Deployment

---

## Future Phase: Detection (After Classification Complete)

### Phase 2: Standalone Detection

After classification is working, we plan to add **detection capability** to enable Edge to operate standalone without Tripwire:

- **Detection Model**: YOLOv5 or custom for signal localization
- **Two-Stage Pipeline**: Detection → Classification
- **Standalone Operation**: Wide-band scanning and automatic signal detection
- **Timeline**: 5-10 weeks after Phase 1 complete

**See**: `docs/ML_DETECTION_PHASE.md` for detailed detection phase plan  
**See**: `docs/ML_ROADMAP.md` for overall ML implementation roadmap

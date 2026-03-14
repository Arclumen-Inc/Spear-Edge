# SPEAR-Edge ML Dashboard

## Overview

The ML Dashboard is a dedicated web interface for managing machine learning models and capture labeling. It provides tools for viewing captures, editing labels, exporting models for training, and importing new models.

## Access

Navigate to: `http://localhost:8080/ml`

Or click the "ML Dashboard" link in the main UI header.

## Features

### Capture Management

#### View Captures
- Grid view of all captures with thumbnails
- Shows: timestamp, frequency, label, confidence, source
- Thumbnail spectrograms for visual identification

#### Filter Captures
- **Search**: Filter by capture directory name
- **Label Filter**: Filter by classification label
- **Source Filter**: Filter by capture source (manual/armed/tasked)

#### Edit Labels
- **Single Edit**: Click dropdown on any capture card to change label
- **Batch Edit**: Select multiple captures and update all at once
- Labels are validated against `class_labels.json`
- Updates both `capture.json` and `dataset_raw` copy (if exists)

#### Batch Operations
- **Update Labels**: Change labels for multiple selected captures
- **Export Selected**: Export selected captures for training (coming soon)
- **Delete Selected**: Delete multiple captures at once

### Model Management

#### Quick Train
- **Fine-tune model on 1-2 captures** (5-15 minutes)
- Select 1-2 labeled captures
- Choose target label
- Set number of epochs (5-30, default: 15)
- Real-time progress tracking
- Only trains classification head (fast fine-tuning)
- Saves new model automatically

#### Current Model Info
- Displays active model type (PyTorch/ONNX/Stub)
- Shows number of classes
- Shows model file path
- Indicates if model is active

#### Export Model
- Exports current model as ZIP file
- Includes:
  - Model file (`.pth` or `.onnx`)
  - `class_labels.json`
  - `model_metadata.json` (version, classes, export timestamp)
- Download via browser

#### Import Model
- Upload ZIP file containing:
  - Model file (`.pth` or `.onnx`)
  - `class_labels.json` (optional, updates if present)
- Validates model format
- Places files in `spear_edge/ml/models/`
- **Note**: Application restart required to activate new model

#### Test Model
- Test model on selected capture
- Shows classification result (label, confidence)
- Useful for validating model performance

#### Available Models
- Lists all models in `spear_edge/ml/models/`
- Shows model type, size, modification date
- Highlights currently active model

### Statistics Dashboard

#### Capture Statistics
- **Total Captures**: Total number of captures
- **Labeled**: Number of captures with labels
- **Unlabeled**: Number of captures without labels

#### Label Distribution
- Bar chart showing count per label
- Sorted by frequency (most common first)
- Helps identify dataset balance

## API Endpoints

All endpoints are under `/api/ml`:

### Captures
- `GET /api/ml/captures` - List captures (with filtering)
- `GET /api/ml/captures/{capture_dir}` - Get capture details
- `GET /api/ml/captures/{capture_dir}/thumbnail` - Get thumbnail image
- `POST /api/ml/captures/{capture_dir}/label` - Update capture label
- `POST /api/ml/captures/batch-label` - Batch update labels
- `POST /api/ml/captures/{capture_dir}/delete` - Delete capture
- `POST /api/ml/captures/batch-delete` - Batch delete captures

### Models
- `GET /api/ml/models` - List available models
- `GET /api/ml/models/current` - Get current active model info
- `POST /api/ml/models/export` - Export current model (downloads ZIP)
- `POST /api/ml/models/import` - Import new model (upload ZIP)
- `POST /api/ml/models/test` - Test model on capture
- `POST /api/ml/train/quick` - Start quick training
- `GET /api/ml/train/status/{job_id}` - Get training status
- `POST /api/ml/train/cancel/{job_id}` - Cancel training

### Metadata
- `GET /api/ml/class-labels` - Get class labels mapping
- `GET /api/ml/stats` - Get ML statistics

## Usage Examples

### Changing a Capture Label

1. Navigate to ML Dashboard
2. Find capture in grid
3. Click dropdown on capture card
4. Select new label (e.g., "elrs" → "voice_analog")
5. Label updates immediately

### Batch Labeling

1. Select multiple captures (checkboxes)
2. Click "Update Selected Labels"
3. Enter label name
4. All selected captures updated

### Exporting Model for Training

1. Click "Export Current Model"
2. ZIP file downloads automatically
3. Contains:
   - Model file
   - Class labels
   - Metadata
4. Transfer to training machine
5. Use for fine-tuning or evaluation

### Importing New Model

1. Train model on larger machine
2. Create ZIP with:
   - Model file (`.pth` or `.onnx`)
   - `class_labels.json`
3. Click "Import New Model"
4. Select ZIP file
5. Model imported to `spear_edge/ml/models/`
6. **Restart application** to activate

### Testing Model

1. Select a single capture
2. Click "Test Model on Selected"
3. View classification result
4. Compare with existing label

### Quick Training

1. Select 1-2 labeled captures
2. Choose target label from dropdown
3. Set epochs (default: 15, range: 5-30)
4. Click "Start Quick Train"
5. Monitor progress in real-time
6. Training completes in 5-15 minutes
7. New model saved automatically
8. **Restart application** to activate new model

**Note**: Quick training uses fine-tuning (only trains classification head), making it fast and suitable for on-device training on Jetson.

## File Structure

### Capture Directory
```
data/artifacts/captures/
└── YYYYMMDD_HHMMSS_FREQHz_.../
    ├── capture.json          # Metadata (includes classification)
    ├── thumbnails/
    │   └── spectrogram.png   # Thumbnail for UI
    └── ...
```

### Model Export ZIP
```
spear_edge_model_<name>_<timestamp>.zip
├── rf_classifier.pth         # Model file
├── class_labels.json         # Class labels
└── model_metadata.json       # Export metadata
```

## Class Labels

Labels are defined in `spear_edge/ml/models/class_labels.json`:

```json
{
  "class_mapping": {
    "0": {
      "id": "elrs",
      "name": "ExpressLRS",
      "signal_type": "fhss_control"
    },
    ...
  }
}
```

Valid labels are validated against this file when editing.

## Notes

- **Model Activation**: Imported models require application restart
- **Label Validation**: Only labels in `class_labels.json` are accepted
- **Thumbnails**: Generated automatically during capture
- **Statistics**: Auto-refresh every 30 seconds
- **Batch Operations**: Can process multiple captures efficiently

## Troubleshooting

### No Captures Showing
- Check `data/artifacts/captures/` directory exists
- Verify captures have `capture.json` files
- Check browser console for errors

### Model Import Fails
- Verify ZIP contains `.pth` or `.onnx` file
- Check file size (may be too large)
- Verify model format is correct

### Label Update Fails
- Verify label exists in `class_labels.json`
- Check capture directory exists
- Verify write permissions

### Thumbnails Not Showing
- Check `thumbnails/spectrogram.png` exists in capture directory
- Verify file permissions
- Check browser console for 404 errors

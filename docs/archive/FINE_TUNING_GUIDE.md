# Fine-Tuning Guide: Adding New Classes to RF Classifier

## Overview

The fine-tuning script (`scripts/fine_tune_new_class.py`) allows you to quickly add new device/protocol classes to an existing trained model without full retraining.

**Time Estimate**: 15-60 minutes (vs 2-8 hours for full retraining)

## When to Use Fine-Tuning

✅ **Use fine-tuning when:**
- Adding 1-3 new classes
- You have 20-100 samples of the new class
- You want to preserve existing model performance
- Quick deployment is needed

❌ **Use full retraining when:**
- Adding 5+ new classes at once
- Major dataset changes
- Quarterly maintenance
- Performance degradation

## Prerequisites

1. **Existing trained model**: `spear_edge/ml/models/rf_classifier.pth`
2. **New class data**: Captures in `data/dataset_raw/` or `data/dataset/train/{new_class_id}/`
3. **Updated class_labels.json**: New class must be added to class mapping

## Step-by-Step Workflow

### Step 1: Collect and Label Data

After recovering a drone or identifying a new signal:

```bash
# Your captures are auto-exported to data/dataset_raw/
# Label them by moving to class directory or using JSON mapping
```

### Step 2: Add Class to class_labels.json

Edit `spear_edge/ml/models/class_labels.json`:

```json
{
  "22": {
    "id": "custom_fpv_drone_x",
    "name": "Custom FPV Drone X",
    "signal_type": "digital_video",
    "description": "Custom FPV drone identified on 2025-01-15"
  }
}
```

Update `class_to_index` and `index_to_class` mappings.

### Step 3: Prepare Dataset

```bash
python3 scripts/prepare_training_dataset.py \
    --spear-dir data/dataset_raw \
    --output-dir data/dataset \
    --class-labels spear_edge/ml/models/class_labels.json
```

### Step 4: Run Fine-Tuning

```bash
python3 scripts/fine_tune_new_class.py \
    --model-path spear_edge/ml/models/rf_classifier.pth \
    --dataset-dir data/dataset \
    --new-class-id custom_fpv_drone_x \
    --new-class-name "Custom FPV Drone X" \
    --output-path spear_edge/ml/models/rf_classifier_v2.pth \
    --epochs 20 \
    --batch-size 16 \
    --learning-rate 1e-4
```

### Step 5: Deploy Updated Model

```bash
# Copy to Jetson (if training on dev machine)
scp spear_edge/ml/models/rf_classifier_v2.pth jetson:/path/to/spear-edgev1_0/spear_edge/ml/models/

# Or replace existing model
mv spear_edge/ml/models/rf_classifier_v2.pth spear_edge/ml/models/rf_classifier.pth
```

## Fine-Tuning Parameters

### Required Arguments
- `--model-path`: Path to existing trained model
- `--dataset-dir`: Path to prepared dataset directory
- `--new-class-id`: Class ID from class_labels.json
- `--output-path`: Where to save fine-tuned model

### Optional Arguments
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 16)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--include-old-data`: Include some old class data to prevent forgetting (default: True)
- `--old-data-ratio`: Ratio of old data to include (default: 0.3)

## How It Works

1. **Loads existing model** with N classes
2. **Extends classification head** to N+1 classes
3. **Copies all weights** except final layer (randomly initialized)
4. **Freezes feature extraction** (conv layers)
5. **Trains only classification head** on new data + some old data
6. **Saves fine-tuned model** with extended classes

## Tips for Best Results

### Data Requirements
- **Minimum**: 10-20 samples (works but limited)
- **Recommended**: 50-100 samples
- **Ideal**: 100+ samples

### Training Tips
1. **Lower learning rate**: Use 1e-4 or 1e-5 (10x smaller than initial training)
2. **Include old data**: Prevents catastrophic forgetting
3. **Monitor validation**: Stop if accuracy plateaus
4. **Fewer epochs**: 5-20 epochs usually sufficient

### Troubleshooting

**Low accuracy on new class:**
- Increase number of samples
- Increase epochs
- Lower learning rate further
- Check data quality

**Forgetting old classes:**
- Increase `--old-data-ratio` (try 0.5)
- Include more diverse old class samples
- Consider full retraining if severe

**Out of memory:**
- Reduce `--batch-size` (try 8 or 4)
- Use CPU instead of GPU

## Example Timeline

**Scenario**: Unknown signal collected 30 times, drone recovered and identified

- **10:00 AM** - Drone recovered, identified
- **10:05 AM** - Label 30 captures
- **10:07 AM** - Update class_labels.json
- **10:12 AM** - Prepare dataset
- **10:15 AM** - Start fine-tuning
- **10:45 AM** - Fine-tuning complete (30 min)
- **10:47 AM** - Deploy to Jetson
- **10:50 AM** - Testing new classification

**Total time: ~50 minutes**

## Next Steps

After fine-tuning:
1. Test model on validation set
2. Deploy to Jetson
3. Monitor classification performance
4. Collect more data if needed
5. Consider full retraining after several additions

# RFUAV Dataset Download Guide

**Dataset**: [kitofrank/RFUAV on Hugging Face](https://huggingface.co/datasets/kitofrank/RFUAV)  
**Purpose**: Training data for RF signal classification (drone identification)

---

## Dataset Overview

- **Format**: ImageFolder (spectrogram images)
- **Classes**: 35 drone types
- **Splits**: 
  - Train: 5,300 samples
  - Validation: 5,100 samples
- **Size**: ~10GB (spectrograms), 1.3TB (full with raw data)
- **License**: Apache 2.0

---

## Quick Start

### Option 1: Download Using Script (Recommended)

```bash
# Install Hugging Face datasets library (if not already installed)
pip3 install datasets

# Download train split (full dataset)
python3 scripts/download_rfuav_dataset.py \
    --output-dir data/rfuav \
    --split train

# Download both splits
python3 scripts/download_rfuav_dataset.py \
    --output-dir data/rfuav \
    --split all

# Download limited samples (for testing)
python3 scripts/download_rfuav_dataset.py \
    --output-dir data/rfuav \
    --split train \
    --max-samples-per-class 50
```

### Option 2: Manual Download via Python

```python
from datasets import load_dataset

# Load train split
dataset = load_dataset("kitofrank/RFUAV", split="train")

# Access samples
for sample in dataset:
    image = sample["image"]
    label = sample["label"]
    # Process image...
```

### Option 3: Using Hugging Face CLI

```bash
# Install Hugging Face CLI
pip install huggingface_hub[cli]

# Download dataset
huggingface-cli download kitofrank/RFUAV --local-dir data/rfuav
```

---

## Dataset Structure (After Download)

```
data/rfuav/
├── train/
│   ├── AVATA/
│   │   └── imgs/
│   │       ├── sample_000000.jpg
│   │       └── ...
│   ├── MINI4/
│   │   └── imgs/
│   │       └── ...
│   └── ...
└── validation/
    └── (same structure)
```

---

## Integration with SPEAR-Edge

After downloading, use the data preparation script:

```bash
python3 scripts/prepare_training_dataset.py \
    --rfuav-dir data/rfuav \
    --spear-dir data/dataset_raw \
    --output-dir data/dataset \
    --class-labels spear_edge/ml/models/class_labels.json
```

This will:
1. Convert RFUAV images to NumPy arrays (512x512 float32)
2. Map RFUAV class names to SPEAR-Edge class IDs
3. Combine with SPEAR-Edge captures
4. Organize into train/val/test splits

---

## Class Mapping

RFUAV has 35 classes. The data preparation script maps them to SPEAR-Edge classes:

- **AVATA** → `dji_avata`
- **MINI4** → `dji_mini_4_pro`
- **MINI3** → `dji_mini_3`
- **AIR3** → `dji_air_3`
- **MAVIC3** → `dji_mavic_3`
- **FPV** → `dji_fpv`
- **O3** → `dji_o3`
- **O4** → `dji_o4`
- (Other classes may map to "unknown" or need manual mapping)

See `spear_edge/ml/models/class_labels.json` for full mapping.

---

## Download Options

### Full Dataset
- **Size**: ~10GB
- **Time**: 30-60 minutes (depending on connection)
- **Use case**: Full training

### Limited Samples (Testing)
- **Size**: ~500MB-1GB
- **Time**: 5-10 minutes
- **Use case**: Testing pipeline, quick validation

```bash
# Download 50 samples per class for testing
python3 scripts/download_rfuav_dataset.py \
    --output-dir data/rfuav_test \
    --split train \
    --max-samples-per-class 50
```

---

## Requirements

- **Python 3.10+**
- **Hugging Face datasets library**: `pip install datasets`
- **Pillow**: `pip install pillow` (for image handling)
- **Internet connection**: For downloading from Hugging Face
- **Disk space**: ~15GB free (for download + processing)

---

## Troubleshooting

### Issue: "datasets library not installed"
```bash
pip3 install datasets
```

### Issue: "Connection timeout"
- Check internet connection
- Try again (Hugging Face may be busy)
- Use `--cache-dir` to specify cache location

### Issue: "Out of disk space"
- Download only train split: `--split train`
- Limit samples: `--max-samples-per-class 100`
- Use different output directory with more space

### Issue: "Permission denied"
- Check write permissions on output directory
- Use `sudo` if needed (not recommended)

---

## Notes

- **First download**: May take longer (downloading and caching)
- **Subsequent downloads**: Faster (uses cache)
- **Disk space**: Full dataset requires ~15GB (download + processing)
- **Network**: Stable connection recommended for large downloads

---

## References

- **Hugging Face Dataset**: https://huggingface.co/datasets/kitofrank/RFUAV
- **GitHub Repository**: https://github.com/kitoweeknd/RFUAV/
- **Paper**: https://arxiv.org/abs/2503.09033

#!/usr/bin/env python3
"""
Data preparation pipeline for RF signal classification training.
Combines RFUAV dataset and SPEAR-Edge captures into unified training dataset.

Usage:
    python3 scripts/prepare_training_dataset.py \
        --rfuav-dir /path/to/RFUAV/Dataset \
        --spear-dir data/dataset_raw \
        --output-dir data/dataset \
        --class-labels spear_edge/ml/models/class_labels.json
"""

import argparse
import hashlib
import json
import shutil
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from PIL import Image
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_class_labels(class_labels_path: Path) -> Dict:
    """Load class labels mapping from JSON file."""
    with open(class_labels_path, 'r') as f:
        labels = json.load(f)
    return labels


def convert_image_to_spectrogram(img_path: Path) -> Optional[np.ndarray]:
    """
    Convert RFUAV spectrogram image to numpy array.
    RFUAV images are typically grayscale spectrograms.
    """
    try:
        img = Image.open(img_path)
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to 512x512 if needed
        if img.size != (512, 512):
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1] then convert to dB scale (approximate)
        # RFUAV images are typically in [0, 255] range
        img_array = img_array / 255.0
        
        # Convert to dB scale (approximate, assuming image represents power)
        # This is a rough conversion - actual RFUAV format may vary
        img_array = 20 * np.log10(img_array + 1e-10)  # Add small epsilon to avoid log(0)
        
        # Normalize to noise floor (subtract minimum, typical for RF spectrograms)
        img_array = img_array - np.min(img_array)
        
        return img_array
    except Exception as e:
        print(f"[ERROR] Failed to convert image {img_path}: {e}")
        return None


def process_rfuav_dataset(rfuav_dir: Path, class_labels: Dict, output_base: Path) -> Dict[str, List[Path]]:
    """
    Process RFUAV dataset and convert to SPEAR-Edge format.
    
    RFUAV structure (numeric IDs):
    Dataset/
    ├── train/
    │   ├── 0/
    │   │   └── imgs/
    │   ├── 1/
    │   │   └── imgs/
    │   └── ...
    └── valid/
        └── (same structure)
    
    Returns: Dict mapping class_id to list of spectrogram paths
    """
    project_root = Path(__file__).resolve().parent.parent
    rfuav_cfg = project_root / "spear_edge" / "ml" / "config" / "rfuav_class_map.json"
    extra_dir_map: Dict = {}
    if rfuav_cfg.exists():
        try:
            with open(rfuav_cfg, "r", encoding="utf-8") as f:
                extra_dir_map = json.load(f).get("rfuav_dir_to_class") or {}
        except Exception as e:
            print(f"[RFUAV] Could not load {rfuav_cfg}: {e}")
    rfuav_mapping = {**class_labels.get("rfuav_mapping", {}), **extra_dir_map}
    class_to_index = class_labels.get("class_to_index", {})
    index_to_class = class_labels.get("index_to_class", {})
    
    # Create mapping from RFUAV numeric ID to our class_id
    # RFUAV has 37 classes (0-36), we map them to our 23 classes
    # For now, map numeric IDs directly to our class indices where possible
    # This is a heuristic - may need adjustment based on actual RFUAV class meanings
    rfuav_numeric_mapping = {}
    
    # Try to map numeric IDs to our class indices
    # Since we don't know the exact RFUAV class meanings, we'll use a simple mapping:
    # Map RFUAV IDs 0-22 to our class indices 0-22 (if they exist)
    # Map others to "unknown" (class index 22)
    for rfuav_id in range(37):  # RFUAV has 37 classes
        rfuav_id_str = str(rfuav_id)
        if rfuav_id_str in index_to_class:
            # Direct mapping: RFUAV ID matches our class index
            class_id = index_to_class[rfuav_id_str]
            rfuav_numeric_mapping[rfuav_id_str] = class_id
        else:
            # Map to unknown
            rfuav_numeric_mapping[rfuav_id_str] = "unknown"
    
    processed_samples = {}  # class_id -> list of paths
    
    if not rfuav_dir.exists():
        print(f"[WARN] RFUAV directory not found: {rfuav_dir}")
        return processed_samples
    
    print(f"[RFUAV] Processing dataset from: {rfuav_dir}")
    
    # Process both train and valid splits
    for split in ["train", "valid"]:
        split_dir = rfuav_dir / split
        if not split_dir.exists():
            print(f"[RFUAV] Split directory not found: {split_dir}")
            continue
        
        print(f"[RFUAV] Processing {split} split...")
        
        # Iterate through drone type directories
        for drone_dir in split_dir.iterdir():
            if not drone_dir.is_dir():
                continue
            
            drone_name = drone_dir.name
            
            # Try string mapping first (for named directories like "AVATA")
            class_id = rfuav_mapping.get(drone_name)
            
            # If no string mapping, try numeric mapping
            if not class_id:
                class_id = rfuav_numeric_mapping.get(drone_name)
            
            if not class_id:
                print(f"[RFUAV] No mapping for drone type: {drone_name}, skipping")
                continue
            
            if class_id not in class_to_index:
                print(f"[RFUAV] Class {class_id} not in class_to_index, skipping")
                continue
            
            # Find images directory
            imgs_dir = drone_dir / "imgs"
            if not imgs_dir.exists():
                print(f"[RFUAV] Images directory not found: {imgs_dir}")
                continue
            
            # Process all images
            image_files = list(imgs_dir.glob("*.png")) + list(imgs_dir.glob("*.jpg"))
            
            if class_id not in processed_samples:
                processed_samples[class_id] = []
            
            print(f"[RFUAV] Processing {len(image_files)} images for {drone_name} -> {class_id}")
            
            for img_path in image_files:
                # Convert image to spectrogram array
                spec_array = convert_image_to_spectrogram(img_path)
                
                if spec_array is None:
                    continue
                
                # Save as .npy file in temporary location
                temp_dir = output_base / "temp_rfuav" / class_id
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                spec_path = temp_dir / f"{img_path.stem}.npy"
                np.save(spec_path, spec_array)
                
                processed_samples[class_id].append(spec_path)
    
    print(f"[RFUAV] Processed {sum(len(v) for v in processed_samples.values())} samples")
    return processed_samples


def process_spear_captures(spear_dir: Path, class_labels: Dict) -> Dict[str, List[Path]]:
    """
    Process SPEAR-Edge captures from data/dataset_raw/.
    
    SPEAR-Edge structure:
    data/dataset_raw/
    ├── {capture_dir_1}/
    │   ├── spectrogram.npy
    │   └── capture.json
    └── {capture_dir_2}/
        └── ...
    
    Returns: Dict mapping class_id to list of spectrogram paths
    """
    processed_samples = {}
    class_to_index = class_labels.get("class_to_index", {})
    
    if not spear_dir.exists():
        print(f"[SPEAR] Directory not found: {spear_dir}")
        return processed_samples
    
    print(f"[SPEAR] Processing captures from: {spear_dir}")
    
    # Iterate through capture directories
    for capture_dir in spear_dir.iterdir():
        if not capture_dir.is_dir():
            continue
        
        spec_path = capture_dir / "spectrogram.npy"
        json_path = capture_dir / "capture.json"
        
        if not spec_path.exists():
            print(f"[SPEAR] Spectrogram not found in {capture_dir}, skipping")
            continue
        
        # Try to get class label from capture.json
        class_id = None
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    capture_data = json.load(f)
                
                # Check classification result
                classification = capture_data.get("classification", {})
                if classification:
                    label = classification.get("label", "")
                    # Map label to class_id
                    if label.startswith("class_"):
                        idx = label.replace("class_", "")
                        # Reverse lookup from index_to_class
                        index_to_class = class_labels.get("index_to_class", {})
                        class_id = index_to_class.get(idx)
                    else:
                        # Direct class_id lookup
                        class_id = class_to_index.get(label)
                
                # If no classification, try to infer from metadata
                if not class_id:
                    # Check tripwire classification hint
                    meta = capture_data.get("meta", {})
                    tripwire_class = meta.get("classification")
                    if tripwire_class:
                        # Map tripwire classification to class_id
                        # This is a heuristic - may need adjustment
                        tripwire_to_class = {
                            "fhss_like": "elrs",  # Default to ELRS for FHSS
                            "drone_control": "dji_mini_4_pro",  # Default drone
                        }
                        class_id = tripwire_to_class.get(tripwire_class.lower())
            except Exception as e:
                print(f"[SPEAR] Error reading {json_path}: {e}")
        
        # If still no class_id, use "unknown"
        if not class_id:
            class_id = "unknown"
        
        if class_id not in class_to_index:
            print(f"[SPEAR] Class {class_id} not in class_to_index, using 'unknown'")
            class_id = "unknown"
        
        if class_id not in processed_samples:
            processed_samples[class_id] = []
        
        processed_samples[class_id].append(spec_path)
    
    print(f"[SPEAR] Processed {sum(len(v) for v in processed_samples.values())} samples")
    return processed_samples


def validate_spectrogram(spec_path: Path) -> bool:
    """Validate spectrogram file."""
    try:
        spec = np.load(spec_path)
        if spec.shape != (512, 512):
            print(f"[VALIDATE] Wrong shape {spec.shape} in {spec_path}")
            return False
        if spec.dtype != np.float32:
            print(f"[VALIDATE] Wrong dtype {spec.dtype} in {spec_path}")
            return False
        if not np.isfinite(spec).all():
            print(f"[VALIDATE] Non-finite values in {spec_path}")
            return False
        return True
    except Exception as e:
        print(f"[VALIDATE] Error loading {spec_path}: {e}")
        return False


def _capture_group_key(path: Path) -> str:
    """Group SPEAR samples by capture directory when capture.json sits next to spectrogram."""
    par = path.parent
    if (par / "capture.json").is_file():
        return f"capture:{par.name}"
    return f"file:{path.resolve()}"


def _split_paths_random(
    valid_samples: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List[Path]]:
    vs = list(valid_samples)
    random.shuffle(vs)
    n_total = len(vs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    return {
        "train": vs[:n_train],
        "val": vs[n_train : n_train + n_val],
        "test": vs[n_train + n_val :],
    }


def _split_paths_grouped(
    valid_samples: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    group_by: str,
) -> Dict[str, List[Path]]:
    if group_by != "capture":
        return _split_paths_random(valid_samples, train_ratio, val_ratio, test_ratio)

    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in valid_samples:
        groups[_capture_group_key(p)].append(p)
    keys = list(groups.keys())
    random.shuffle(keys)
    n_g = len(keys)
    if n_g <= 1:
        return _split_paths_random(valid_samples, train_ratio, val_ratio, test_ratio)

    n_train_g = max(1, int(round(n_g * train_ratio)))
    n_val_g = max(1, int(round(n_g * val_ratio))) if n_g > 2 else 0
    n_test_g = max(1, n_g - n_train_g - n_val_g)
    while n_train_g + n_val_g + n_test_g > n_g:
        if n_val_g > 1:
            n_val_g -= 1
        elif n_test_g > 1:
            n_test_g -= 1
        else:
            n_train_g = n_g - n_val_g - n_test_g
            break
    train_keys = keys[:n_train_g]
    val_keys = keys[n_train_g : n_train_g + n_val_g]
    test_keys = keys[n_train_g + n_val_g : n_train_g + n_val_g + n_test_g]

    out = {"train": [], "val": [], "test": []}
    for k in train_keys:
        out["train"].extend(groups[k])
    for k in val_keys:
        out["val"].extend(groups[k])
    for k in test_keys:
        out["test"].extend(groups[k])
    return out


def organize_dataset(
    all_samples: Dict[str, List[Path]],
    output_dir: Path,
    class_labels: Dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    group_by: str = "none",
    class_labels_path: Optional[Path] = None,
) -> None:
    """
    Organize samples into train/val/test splits.
    
    Structure:
    data/dataset/
    ├── train/
    │   ├── elrs/
    │   │   ├── sample_001.npy
    │   │   └── ...
    │   └── dji_mini_4_pro/
    │       └── ...
    ├── val/
    │   └── (same structure)
    └── test/
        └── (same structure)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio
    }
    
    class_to_index = class_labels.get("class_to_index", {})
    try:
        from spear_edge.ml.preprocess import CURRENT_PREPROCESS_SCHEMA
    except ImportError:
        CURRENT_PREPROCESS_SCHEMA = "spear_ml_spec_v1"

    labels_fingerprint = ""
    if class_labels_path and class_labels_path.exists():
        h = hashlib.sha256()
        h.update(class_labels_path.read_bytes())
        labels_fingerprint = h.hexdigest()[:16]

    manifest = {
        "version": "2.0",
        "preprocess_schema": CURRENT_PREPROCESS_SCHEMA,
        "group_by": group_by,
        "class_labels_sha256_16": labels_fingerprint,
        "num_classes": len(class_to_index),
        "splits": {},
    }
    
    print(f"[ORGANIZE] Organizing dataset into {output_dir}")
    
    for class_id, sample_paths in all_samples.items():
        if not sample_paths:
            continue
        
        # Validate and filter samples
        valid_samples = [p for p in sample_paths if validate_spectrogram(p)]
        
        if not valid_samples:
            print(f"[ORGANIZE] No valid samples for {class_id}, skipping")
            continue
        
        split_samples = _split_paths_grouped(
            valid_samples, train_ratio, val_ratio, test_ratio, group_by
        )
        n_train = len(split_samples["train"])
        n_val = len(split_samples["val"])
        n_test = len(split_samples["test"])
        print(f"[ORGANIZE] {class_id}: {n_train} train, {n_val} val, {n_test} test (group_by={group_by})")
        
        # Copy samples to organized structure
        for split_name, samples in split_samples.items():
            if not samples:
                continue
            
            split_dir = output_dir / split_name / class_id
            split_dir.mkdir(parents=True, exist_ok=True)
            
            split_manifest = []
            
            for i, src_path in enumerate(samples):
                dst_path = split_dir / f"sample_{i:06d}.npy"
                shutil.copy2(src_path, dst_path)
                
                split_manifest.append({
                    "path": str(dst_path.relative_to(output_dir)),
                    "class_id": class_id,
                    "class_index": class_to_index.get(class_id, -1)
                })
            
            if split_name not in manifest["splits"]:
                manifest["splits"][split_name] = []
            manifest["splits"][split_name].extend(split_manifest)
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"[ORGANIZE] Dataset organized. Manifest saved to: {manifest_path}")
    
    # Print summary
    print("\n[SUMMARY] Dataset Statistics:")
    for split_name in ["train", "val", "test"]:
        split_samples = manifest["splits"].get(split_name, [])
        class_counts = {}
        for sample in split_samples:
            class_id = sample["class_id"]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        print(f"  {split_name}: {len(split_samples)} samples, {len(class_counts)} classes")
        for class_id, count in sorted(class_counts.items()):
            print(f"    {class_id}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training dataset from RFUAV and SPEAR-Edge captures"
    )
    parser.add_argument(
        "--rfuav-dir",
        type=Path,
        help="Path to RFUAV Dataset directory (optional)"
    )
    parser.add_argument(
        "--spear-dir",
        type=Path,
        default=Path("data/dataset_raw"),
        help="Path to SPEAR-Edge captures directory (default: data/dataset_raw)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dataset"),
        help="Output directory for organized dataset (default: data/dataset)"
    )
    parser.add_argument(
        "--class-labels",
        type=Path,
        default=Path("spear_edge/ml/models/class_labels.json"),
        help="Path to class_labels.json (default: spear_edge/ml/models/class_labels.json)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=["none", "capture"],
        default="none",
        help="Split mode: 'capture' keeps all files from one SPEAR capture in one split (when capture.json is present).",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load class labels
    if not args.class_labels.exists():
        print(f"[ERROR] Class labels file not found: {args.class_labels}")
        return 1
    
    class_labels = load_class_labels(args.class_labels)
    print(f"[LOAD] Loaded class labels: {class_labels['num_classes']} classes")
    
    # Process datasets
    all_samples = {}
    
    # Process RFUAV dataset
    if args.rfuav_dir:
        rfuav_samples = process_rfuav_dataset(args.rfuav_dir, class_labels, args.output_dir)
        for class_id, samples in rfuav_samples.items():
            if class_id not in all_samples:
                all_samples[class_id] = []
            all_samples[class_id].extend(samples)
    
    # Process SPEAR-Edge captures
    spear_samples = process_spear_captures(args.spear_dir, class_labels)
    for class_id, samples in spear_samples.items():
        if class_id not in all_samples:
            all_samples[class_id] = []
        all_samples[class_id].extend(samples)
    
    if not all_samples:
        print("[ERROR] No samples found. Check input directories.")
        return 1
    
    # Organize into train/val/test
    organize_dataset(
        all_samples,
        args.output_dir,
        class_labels,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        group_by=args.group_by,
        class_labels_path=args.class_labels,
    )
    
    # Cleanup temporary RFUAV directory
    temp_dir = args.output_dir / "temp_rfuav"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"[CLEANUP] Removed temporary directory: {temp_dir}")
    
    print("\n[SUCCESS] Dataset preparation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Download RFUAV dataset from Hugging Face.

The RFUAV dataset contains spectrogram images for drone identification.
This script downloads the dataset and organizes it for use with
SPEAR-Edge training pipeline.

Dataset: https://huggingface.co/datasets/kitofrank/RFUAV
- Format: imagefolder
- Classes: 35 drone types
- Splits: train (5.3k), validation (5.1k)
- Size: ~10GB (spectrograms), 1.3TB (full with raw data)

Usage:
    python3 scripts/download_rfuav_dataset.py \
        --output-dir data/rfuav \
        --split train \
        --max-samples-per-class 100
"""

import argparse
import sys
from pathlib import Path
import shutil
from tqdm import tqdm

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("[ERROR] Hugging Face datasets library not installed")
    print("[ERROR] Install with: pip install datasets")
    sys.exit(1)


def download_rfuav_dataset(
    output_dir: Path,
    split: str = "train",
    max_samples_per_class: int = None,
    cache_dir: Path = None
):
    """
    Download RFUAV dataset from Hugging Face.
    
    Args:
        output_dir: Directory to save downloaded images
        split: Dataset split ("train" or "validation")
        max_samples_per_class: Limit samples per class (None = all)
        cache_dir: Cache directory for Hugging Face (optional)
    """
    print(f"[RFUAV] Downloading {split} split from Hugging Face...")
    print(f"[RFUAV] Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from Hugging Face
    print("[RFUAV] Loading dataset (this may take a while on first run)...")
    try:
        dataset = load_dataset(
            "kitofrank/RFUAV",
            split=split,
            cache_dir=str(cache_dir) if cache_dir else None
        )
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        print("[ERROR] Make sure you have internet connection and Hugging Face access")
        return False
    
    print(f"[RFUAV] Dataset loaded: {len(dataset)} samples")
    print(f"[RFUAV] Features: {dataset.features}")
    
    # Organize by class
    class_counts = {}
    class_dirs = {}
    
    print("[RFUAV] Organizing images by class...")
    
    # Create progress bar
    pbar = tqdm(total=len(dataset), desc="Downloading", unit="img", ncols=80)
    
    for idx, sample in enumerate(dataset):
        # Get class label
        label = sample.get("label")
        if label is None:
            pbar.update(1)
            continue
        
        # Get class name (handle different label formats)
        if isinstance(label, dict):
            class_name = label.get("class", str(label))
        else:
            class_name = str(label)
        
        # Create class directory
        if class_name not in class_dirs:
            class_dir = output_dir / class_name / "imgs"
            class_dir.mkdir(parents=True, exist_ok=True)
            class_dirs[class_name] = class_dir
            class_counts[class_name] = 0
        
        # Check sample limit
        if max_samples_per_class and class_counts[class_name] >= max_samples_per_class:
            pbar.update(1)
            continue
        
        # Get image
        image = sample.get("image")
        if image is None:
            pbar.update(1)
            continue
        
        # Save image
        image_path = class_dirs[class_name] / f"sample_{class_counts[class_name]:06d}.jpg"
        try:
            image.save(image_path)
            class_counts[class_name] += 1
        except Exception as e:
            pbar.update(1)
            continue
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'saved': sum(class_counts.values()),
            'classes': len(class_dirs)
        })
    
    pbar.close()
    
    # Print summary
    print("\n[RFUAV] Download Summary:")
    print("=" * 60)
    total_samples = sum(class_counts.values())
    print(f"Total samples downloaded: {total_samples}")
    print(f"Number of classes: {len(class_counts)}")
    print("\nSamples per class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")
    print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download RFUAV dataset from Hugging Face"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/rfuav"),
        help="Output directory for downloaded dataset (default: data/rfuav)"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "all"],
        default="train",
        help="Dataset split to download (default: train)"
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Maximum samples per class (default: all)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for Hugging Face (optional)"
    )
    
    args = parser.parse_args()
    
    if not DATASETS_AVAILABLE:
        print("[ERROR] Hugging Face datasets library required")
        print("[ERROR] Install with: pip install datasets")
        return 1
    
    # Download requested splits
    splits = []
    if args.split == "all":
        splits = ["train", "validation"]
    else:
        splits = [args.split]
    
    success = True
    for split in splits:
        split_output_dir = args.output_dir / split
        print(f"\n[RFUAV] Downloading {split} split...")
        if not download_rfuav_dataset(
            split_output_dir,
            split,
            args.max_samples_per_class,
            args.cache_dir
        ):
            success = False
    
    if success:
        print("\n[SUCCESS] RFUAV dataset download complete!")
        print(f"[INFO] Dataset location: {args.output_dir}")
        print("\n[INFO] Next steps:")
        print("  1. Review downloaded images")
        print("  2. Run data preparation script:")
        print(f"     python3 scripts/prepare_training_dataset.py \\")
        print(f"         --rfuav-dir {args.output_dir} \\")
        print(f"         --spear-dir data/dataset_raw \\")
        print(f"         --output-dir data/dataset")
        return 0
    else:
        print("\n[ERROR] Dataset download failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

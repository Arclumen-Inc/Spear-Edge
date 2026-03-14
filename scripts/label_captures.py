#!/usr/bin/env python3
"""
Label raw captures and add them to training dataset.

Usage:
    # Label a single capture as ELRS:
    python3 scripts/label_captures.py --capture 20260314_142444_915000000Hz_2000000sps_manual --label elrs
    
    # Label multiple captures:
    python3 scripts/label_captures.py --captures-file captures_to_label.txt
    
    # List unlabeled captures:
    python3 scripts/label_captures.py --list-unlabeled
    
    # Auto-label by frequency (interactive confirmation):
    python3 scripts/label_captures.py --auto-label
"""

import argparse
import json
import shutil
import numpy as np
from pathlib import Path
import sys

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "dataset_raw"
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
CLASS_LABELS_PATH = PROJECT_ROOT / "spear_edge" / "ml" / "models" / "class_labels.json"


def load_class_labels():
    """Load class labels."""
    with open(CLASS_LABELS_PATH) as f:
        return json.load(f)


def get_capture_info(capture_name: str) -> dict:
    """Parse capture name to extract frequency and sample rate."""
    parts = capture_name.split("_")
    info = {"name": capture_name}
    
    for part in parts:
        if part.endswith("Hz"):
            info["freq_hz"] = int(part[:-2])
            info["freq_mhz"] = info["freq_hz"] / 1e6
        elif part.endswith("sps"):
            info["sample_rate"] = int(part[:-3])
    
    return info


def suggest_label(freq_hz: int) -> str:
    """Suggest a label based on frequency."""
    freq_mhz = freq_hz / 1e6
    
    # Common frequency ranges
    if 860 <= freq_mhz <= 940:
        return "elrs"  # 900 MHz band - likely ELRS
    elif 2400 <= freq_mhz <= 2500:
        return "elrs"  # 2.4 GHz - could be ELRS or other
    elif 5650 <= freq_mhz <= 5950:
        return "analog_fpv"  # 5.8 GHz - likely VTX
    elif 5100 <= freq_mhz <= 5400:
        return "dji_o3"  # 5.2 GHz - could be DJI
    else:
        return "unknown"


def find_spectrogram(capture_dir: Path) -> Path | None:
    """Find spectrogram .npy file in capture directory."""
    # Check for spectrogram in various locations
    for pattern in ["spectrogram*.npy", "thumbnails/spectrogram*.npy", "*.npy"]:
        files = list(capture_dir.glob(pattern))
        if files:
            return files[0]
    return None


def add_to_dataset(capture_name: str, label: str, split: str = "train"):
    """Add a capture's spectrogram to the training dataset."""
    capture_dir = RAW_DIR / capture_name
    
    if not capture_dir.exists():
        print(f"[ERROR] Capture not found: {capture_dir}")
        return False
    
    # Find spectrogram
    spec_path = find_spectrogram(capture_dir)
    if spec_path is None:
        # Try to load from capture.json and generate
        print(f"[WARN] No spectrogram found in {capture_name}, checking for .npy files...")
        npy_files = list(capture_dir.rglob("*.npy"))
        if npy_files:
            spec_path = npy_files[0]
        else:
            print(f"[ERROR] No .npy spectrogram found in {capture_name}")
            return False
    
    # Create output directory
    output_dir = DATASET_DIR / split / label
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy spectrogram with unique name
    output_name = f"{capture_name}.npy"
    output_path = output_dir / output_name
    
    if output_path.exists():
        print(f"[SKIP] Already exists: {output_path}")
        return True
    
    # Load and verify spectrogram
    try:
        spec = np.load(spec_path)
        if spec.shape != (512, 512):
            print(f"[WARN] Resizing spectrogram from {spec.shape} to (512, 512)")
            # Simple resize using averaging
            from scipy.ndimage import zoom
            zoom_factors = (512 / spec.shape[0], 512 / spec.shape[1])
            spec = zoom(spec, zoom_factors, order=1)
        
        # Save to dataset
        np.save(output_path, spec.astype(np.float32))
        print(f"[OK] Added {capture_name} -> {label}/{output_name}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to process {capture_name}: {e}")
        return False


def list_unlabeled():
    """List captures that haven't been added to the dataset."""
    # Get all raw captures
    raw_captures = set(d.name for d in RAW_DIR.iterdir() if d.is_dir())
    
    # Get all labeled captures (from all splits)
    labeled = set()
    for split in ["train", "val", "test"]:
        split_dir = DATASET_DIR / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    for npy_file in class_dir.glob("*.npy"):
                        # Extract original capture name from filename
                        name = npy_file.stem
                        labeled.add(name)
    
    # Find unlabeled
    unlabeled = raw_captures - labeled
    
    print(f"\n=== Unlabeled Captures ({len(unlabeled)}) ===\n")
    
    for name in sorted(unlabeled):
        info = get_capture_info(name)
        freq_mhz = info.get("freq_mhz", 0)
        suggested = suggest_label(info.get("freq_hz", 0))
        print(f"  {name}")
        print(f"    Freq: {freq_mhz:.1f} MHz -> Suggested: {suggested}")
        print()


def auto_label_interactive():
    """Interactively label captures based on frequency."""
    class_labels = load_class_labels()
    valid_classes = list(class_labels.get("class_to_index", {}).keys())
    
    # Find unlabeled captures
    raw_captures = set(d.name for d in RAW_DIR.iterdir() if d.is_dir())
    labeled = set()
    for split in ["train", "val", "test"]:
        split_dir = DATASET_DIR / split
        if split_dir.exists():
            for class_dir in split_dir.iterdir():
                if class_dir.is_dir():
                    for npy_file in class_dir.glob("*.npy"):
                        labeled.add(npy_file.stem)
    
    unlabeled = sorted(raw_captures - labeled)
    
    if not unlabeled:
        print("No unlabeled captures found!")
        return
    
    print(f"\nFound {len(unlabeled)} unlabeled captures.")
    print(f"Valid labels: {', '.join(valid_classes)}")
    print("\nFor each capture, enter:")
    print("  - Label name (e.g., 'elrs', 'analog_fpv')")
    print("  - 'y' or Enter to accept suggestion")
    print("  - 's' to skip")
    print("  - 'q' to quit")
    print()
    
    added = 0
    skipped = 0
    
    for name in unlabeled:
        info = get_capture_info(name)
        freq_mhz = info.get("freq_mhz", 0)
        suggested = suggest_label(info.get("freq_hz", 0))
        
        print(f"\n{name}")
        print(f"  Freq: {freq_mhz:.1f} MHz")
        
        response = input(f"  Label [{suggested}]: ").strip().lower()
        
        if response == 'q':
            print("\nQuitting...")
            break
        elif response == 's':
            skipped += 1
            continue
        elif response in ('y', ''):
            label = suggested
        elif response in valid_classes:
            label = response
        else:
            print(f"  Invalid label '{response}', skipping...")
            skipped += 1
            continue
        
        if add_to_dataset(name, label, split="train"):
            added += 1
    
    print(f"\nDone! Added: {added}, Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Label captures for training")
    parser.add_argument("--capture", type=str, help="Single capture name to label")
    parser.add_argument("--label", type=str, help="Label for the capture")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--list-unlabeled", action="store_true", help="List unlabeled captures")
    parser.add_argument("--auto-label", action="store_true", help="Interactive auto-labeling")
    
    args = parser.parse_args()
    
    if args.list_unlabeled:
        list_unlabeled()
    elif args.auto_label:
        auto_label_interactive()
    elif args.capture and args.label:
        add_to_dataset(args.capture, args.label, args.split)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python3 scripts/label_captures.py --list-unlabeled")
        print("  python3 scripts/label_captures.py --auto-label")
        print("  python3 scripts/label_captures.py --capture 20260314_142444_915000000Hz_2000000sps_manual --label elrs")


if __name__ == "__main__":
    main()

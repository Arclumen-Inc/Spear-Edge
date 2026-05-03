#!/usr/bin/env python3
"""
Train RF signal classifier using PyTorch.

Usage:
    python3 scripts/train_rf_classifier.py \
        --dataset-dir data/dataset \
        --output-dir spear_edge/ml/models \
        --batch-size 16 \
        --epochs 50 \
        --device cuda
"""

import argparse
import json
import platform
import sys
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import time

# Fix CUDA memory allocator issues on Jetson
if torch.cuda.is_available():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from spear_edge.ml.eval import (
    compute_per_class_metrics,
    dataset_fingerprint,
    evaluate_with_confusion,
    sha256_file,
)
from spear_edge.ml.infer_pytorch import RFClassifier
from spear_edge.ml.calibration import (
    collect_val_logits,
    fit_temperature,
    save_calibration_json,
)
from spear_edge.ml.preprocess import CURRENT_PREPROCESS_SCHEMA, apply_model_preprocess_v1


class SpectrogramDataset(Dataset):
    """Dataset for loading spectrograms from organized directory structure."""
    
    def __init__(self, dataset_dir: Path, split: str = "train", transform=None):
        self.dataset_dir = dataset_dir
        self.split = split
        self.transform = transform
        
        # Load manifest if available
        manifest_path = dataset_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Get samples for this split
            self.samples = manifest.get("splits", {}).get(split, [])
        else:
            # Fallback: scan directory structure
            self.samples = []
            split_dir = dataset_dir / split
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if not class_dir.is_dir():
                        continue
                    class_id = class_dir.name
                    for spec_file in class_dir.glob("*.npy"):
                        self.samples.append({
                            "path": str(spec_file.relative_to(dataset_dir)),
                            "class_id": class_id,
                            "class_index": -1  # Will be set from class_labels
                        })
        
        # Load class labels to get class_index
        class_labels_path = project_root / "spear_edge" / "ml" / "models" / "class_labels.json"
        if class_labels_path.exists():
            with open(class_labels_path, 'r') as f:
                class_labels = json.load(f)
            class_to_index = class_labels.get("class_to_index", {})
            for sample in self.samples:
                class_id = sample["class_id"]
                sample["class_index"] = class_to_index.get(class_id, -1)
        
        # Filter out invalid samples
        self.samples = [s for s in self.samples if s["class_index"] >= 0]
        
        print(f"[DATASET] Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        spec_path = self.dataset_dir / sample["path"]
        
        # Load spectrogram (spear_ml_spec_v1: same normalization as live capture — no per-sample min-max)
        raw = np.load(spec_path)
        if raw.ndim == 2:
            spec2 = apply_model_preprocess_v1(raw)
        elif raw.ndim == 3 and raw.shape[0] == 1:
            spec2 = apply_model_preprocess_v1(raw[0])
        else:
            raise ValueError(f"Unexpected spectrogram shape: {raw.shape}")
        spec = spec2[np.newaxis, :, :]
        
        # Apply transform (augmentation)
        if self.transform:
            spec = self.transform(spec)
        
        # Convert to tensor
        spec_tensor = torch.from_numpy(spec).float()
        label = torch.tensor(sample["class_index"], dtype=torch.long)
        
        return spec_tensor, label


class RandomFlip:
    """Random horizontal flip augmentation."""
    def __call__(self, x):
        if np.random.rand() < 0.5:
            return np.flip(x, axis=2).copy()  # Flip along frequency axis, make copy for PyTorch
        return x


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    if len(dataloader) == 0:
        return 0.0, 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", ncols=80)
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Clear CUDA cache periodically to avoid memory fragmentation
        if device.type == "cuda" and (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    if len(dataloader) == 0:
        return 0.0, 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", ncols=80)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(dataloader):.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train RF signal classifier")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to prepared dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("spear_edge/ml/models"),
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16 for Jetson)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (auto-detected from dataset if not specified)"
    )
    parser.add_argument(
        "--class-labels",
        type=Path,
        default=Path("spear_edge/ml/models/class_labels.json"),
        help="Path to class_labels.json"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--max-batch-gpu",
        type=int,
        default=2,
        help="Clamp batch size to this value on CUDA (Jetson safety). Default: 2",
    )
    parser.add_argument(
        "--no-gpu-batch-clamp",
        action="store_true",
        help="Do not reduce batch size on CUDA (useful on desktop GPUs)",
    )
    parser.add_argument(
        "--min-val-accuracy",
        type=float,
        default=60.0,
        help="Minimum val accuracy %% to mark validation artifact as passing (default: 60)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default 0 — avoids RAM/CMA fragmentation on Jetson; try 2-4 on desktop)",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="DataLoader pin_memory (mainly useful with --num-workers > 0 on desktop CUDA; costs RAM on Orin)",
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        args.device = "cpu"
    
    device = torch.device(args.device)
    print(f"[TRAIN] Using device: {device}")
    
    # For CUDA, set device explicitly and configure memory
    if device.type == "cuda":
        torch.cuda.set_device(0)
        if (
            not args.no_gpu_batch_clamp
            and args.batch_size > args.max_batch_gpu
        ):
            print(
                f"[WARN] Reducing batch size from {args.batch_size} to {args.max_batch_gpu} "
                f"(CUDA); use --no-gpu-batch-clamp for full batch on desktop GPUs"
            )
            args.batch_size = args.max_batch_gpu
    
    # Load class labels to determine num_classes
    class_labels: Dict = {}
    if args.class_labels.exists():
        with open(args.class_labels, 'r') as f:
            class_labels = json.load(f)
        num_classes = class_labels.get("num_classes", 23)
    else:
        num_classes = args.num_classes or 23
    
    if args.num_classes:
        num_classes = args.num_classes
    
    print(f"[TRAIN] Number of classes: {num_classes}")
    pin_mem = bool(args.pin_memory)
    print(
        f"[TRAIN] DataLoader num_workers={args.num_workers}, pin_memory={pin_mem} "
        f"(defaults are Jetson-safe; on desktop try --num-workers 4 --pin-memory)"
    )
    
    # Create datasets
    train_dataset = SpectrogramDataset(
        args.dataset_dir,
        split="train",
        transform=RandomFlip()  # Simple augmentation
    )
    val_dataset = SpectrogramDataset(
        args.dataset_dir,
        split="val",
        transform=None  # No augmentation for validation
    )
    test_dataset = SpectrogramDataset(
        args.dataset_dir,
        split="test",
        transform=None,
    )
    
    # Create data loaders
    _dl_common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_mem,
    }
    if args.num_workers > 0:
        _dl_common["persistent_workers"] = True
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **_dl_common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **_dl_common,
    )
    test_loader = None
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            **_dl_common,
        )
    
    # Create model
    model = RFClassifier(num_classes=num_classes)
    # For CUDA, initialize on device to avoid memory issues
    if device.type == "cuda":
        # Move parameters individually first, then move model
        for param in model.parameters():
            param.data = param.data.to(device)
    model = model.to(device)
    print(f"[TRAIN] Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = float("-inf")
    if args.resume and args.resume.exists():
        print(f"[TRAIN] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"[TRAIN] Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
    
    # Training loop
    print(f"\n[TRAIN] Starting training for {args.epochs} epochs...")
    print(
        f"[TRAIN] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, "
        f"Test samples: {len(test_dataset)}"
    )
    print(f"[TRAIN] Batch size: {args.batch_size}, Learning rate: {args.learning_rate}")
    print("-" * 80)
    
    train_history = []
    val_history = []
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        train_history.append({"loss": train_loss, "acc": train_acc})
        val_history.append({"loss": val_loss, "acc": val_acc})
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        improved = val_acc > best_val_acc if len(val_dataset) > 0 else False
        save_last_no_val = len(val_dataset) == 0 and epoch == args.epochs - 1
        if improved or save_last_no_val:
            if improved:
                best_val_acc = val_acc
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model_path = args.output_dir / "rf_classifier.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc if best_val_acc != float("-inf") else val_acc,
                'num_classes': num_classes,
                'train_history': train_history,
                'val_history': val_history,
                'preprocess_schema': CURRENT_PREPROCESS_SCHEMA,
            }, model_path)
            tag = f"val acc: {val_acc:.2f}%" if len(val_dataset) > 0 else "no val split — last epoch"
            print(f"  ✓ Saved model ({tag}) to {model_path}")
        
        print("-" * 80)
    
    # Final summary
    print(f"\n[TRAIN] Training complete!")
    if best_val_acc == float("-inf"):
        print("[TRAIN] Best validation accuracy: n/a (no val data)")
    else:
        print(f"[TRAIN] Best validation accuracy: {best_val_acc:.2f}%")
    model_path = args.output_dir / "rf_classifier.pth"
    print(f"[TRAIN] Model saved to: {model_path}")

    class_to_index = class_labels.get("class_to_index", {}) if class_labels else {}
    index_to_class = {int(v): k for k, v in class_to_index.items()}
    validation_path = model_path.with_suffix(".validation.json")
    modelcard_path = model_path.with_suffix(".modelcard.json")

    validation: Dict = {
        "model_path": str(model_path),
        "created_at_unix": time.time(),
        "validated": False,
        "reason": "no_model_checkpoint",
        "thresholds": {"min_val_accuracy_pct": float(args.min_val_accuracy)},
        "metrics": {},
    }

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        criterion_eval = nn.CrossEntropyLoss()

        if len(val_dataset) > 0 and len(val_loader) > 0:
            val_loss, val_acc, confusion = evaluate_with_confusion(
                model, val_loader, criterion_eval, device, num_classes=num_classes
            )
            pcm = compute_per_class_metrics(confusion)
            passed = float(val_acc) >= float(args.min_val_accuracy)
            metrics = {
                "val_accuracy_pct": float(val_acc),
                "val_loss": float(val_loss),
                "macro_f1": pcm["macro_f1"],
                "macro_precision": pcm["macro_precision"],
                "macro_recall": pcm["macro_recall"],
                "confusion_matrix": confusion.tolist(),
                "per_class": {
                    index_to_class.get(int(k), f"class_{k}"): v
                    for k, v in pcm["per_class"].items()
                },
                "num_val_batches": len(val_loader),
            }
            calib_path = model_path.with_suffix(".calibration.json")
            try:
                vl, yl = collect_val_logits(model, val_loader, device)
                if len(yl) > 0:
                    T_cal, cal_metrics = fit_temperature(vl, yl)
                    save_calibration_json(
                        calib_path, T_cal, cal_metrics, CURRENT_PREPROCESS_SCHEMA
                    )
                    metrics["calibration_temperature"] = T_cal
                    metrics["calibration_val_nll_before"] = cal_metrics.get("nll_before")
                    metrics["calibration_val_nll_after"] = cal_metrics.get("nll_after")
                    print(f"[TRAIN] Wrote {calib_path} (temperature={T_cal:.4f})")
            except Exception as cal_ex:
                print(f"[TRAIN] Calibration fit skipped: {cal_ex}")
            if test_loader is not None:
                t_loss, t_acc, t_conf = evaluate_with_confusion(
                    model, test_loader, criterion_eval, device, num_classes=num_classes
                )
                t_pcm = compute_per_class_metrics(t_conf)
                metrics["test_accuracy_pct"] = float(t_acc)
                metrics["test_loss"] = float(t_loss)
                metrics["test_macro_f1"] = t_pcm["macro_f1"]
                metrics["test_confusion_matrix"] = t_conf.tolist()
            validation = {
                "model_path": str(model_path),
                "created_at_unix": time.time(),
                "validated": bool(passed),
                "reason": "ok" if passed else "val_accuracy_below_threshold",
                "thresholds": {"min_val_accuracy_pct": float(args.min_val_accuracy)},
                "metrics": metrics,
            }
        else:
            validation = {
                "model_path": str(model_path),
                "created_at_unix": time.time(),
                "validated": False,
                "reason": "no_validation_data",
                "thresholds": {"min_val_accuracy_pct": float(args.min_val_accuracy)},
                "metrics": {},
            }

        modelcard = {
            "schema": "spear_edge.modelcard.v1",
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "base_model_path": None,
            "base_model_sha256": None,
            "training_type": "full",
            "preprocess_schema": CURRENT_PREPROCESS_SCHEMA,
            "num_classes": int(num_classes),
            "dataset": dataset_fingerprint(args.dataset_dir),
            "training": {
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
                "device": str(device),
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": str(torch.version.cuda),
            },
            "validation": validation,
        }
        calib_file = model_path.with_suffix(".calibration.json")
        if calib_file.exists():
            modelcard["calibration"] = {"path": str(calib_file)}
        validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
        modelcard_path.write_text(json.dumps(modelcard, indent=2), encoding="utf-8")
        print(f"[TRAIN] Wrote {validation_path}")
        print(f"[TRAIN] Wrote {modelcard_path}")
    else:
        print("[TRAIN] No checkpoint written (val never improved); skipping validation/modelcard")

    return 0


if __name__ == "__main__":
    sys.exit(main())

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

from spear_edge.ml.infer_pytorch import RFClassifier


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
        
        # Load spectrogram
        spec = np.load(spec_path).astype(np.float32)
        
        # Ensure correct shape: (512, 512) -> (1, 512, 512)
        if spec.ndim == 2:
            spec = spec[np.newaxis, :, :]
        elif spec.ndim == 3 and spec.shape[0] == 1:
            pass  # Already correct
        else:
            raise ValueError(f"Unexpected spectrogram shape: {spec.shape}")
        
        # Normalize to [0, 1] range (assuming dB scale)
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max > spec_min:
            spec = (spec - spec_min) / (spec_max - spec_min)
        
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


class RandomRotation:
    """Small random rotation augmentation."""
    def __init__(self, max_angle=5):
        self.max_angle = max_angle
    
    def __call__(self, x):
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        # Simple rotation using scipy or numpy (simplified)
        # For now, just return original (can add proper rotation later)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
        # Reduce batch size for Jetson if too large (Jetson Orin Nano has limited GPU memory)
        if args.batch_size > 2:
            print(f"[WARN] Batch size {args.batch_size} may be too large for Jetson GPU")
            print(f"[WARN] Reducing to 2 for stability (use gradient accumulation for larger effective batch)")
            args.batch_size = 2
    
    # Load class labels to determine num_classes
    if args.class_labels.exists():
        with open(args.class_labels, 'r') as f:
            class_labels = json.load(f)
        num_classes = class_labels.get("num_classes", 23)
    else:
        num_classes = args.num_classes or 23
    
    if args.num_classes:
        num_classes = args.num_classes
    
    print(f"[TRAIN] Number of classes: {num_classes}")
    
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
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
    best_val_acc = 0.0
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
    print(f"[TRAIN] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
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
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model_path = args.output_dir / "rf_classifier.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'num_classes': num_classes,
                'train_history': train_history,
                'val_history': val_history
            }, model_path)
            print(f"  ✓ Saved best model (val acc: {val_acc:.2f}%) to {model_path}")
        
        print("-" * 80)
    
    # Final summary
    print(f"\n[TRAIN] Training complete!")
    print(f"[TRAIN] Best validation accuracy: {best_val_acc:.2f}%")
    print(f"[TRAIN] Model saved to: {args.output_dir / 'rf_classifier.pth'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

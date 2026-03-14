#!/usr/bin/env python3
"""
Fine-tune existing RF classifier to add a new class.

This script allows you to add a new device/protocol class to an existing
trained model without full retraining. It uses transfer learning by:
1. Loading existing model
2. Extending classification head for new class
3. Freezing feature extraction layers
4. Fine-tuning only the classification head

Usage:
    python3 scripts/fine_tune_new_class.py \
        --model-path spear_edge/ml/models/rf_classifier.pth \
        --dataset-dir data/dataset \
        --new-class-id custom_fpv_drone_x \
        --new-class-name "Custom FPV Drone X" \
        --output-path spear_edge/ml/models/rf_classifier_v2.pth \
        --epochs 20 \
        --batch-size 16 \
        --learning-rate 1e-4
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from spear_edge.ml.infer_pytorch import RFClassifier


class SpectrogramDataset(Dataset):
    """Dataset for loading spectrogram .npy files."""
    
    def __init__(self, class_dir: Path, class_index: int, transform=None):
        self.class_dir = class_dir
        self.class_index = class_index
        self.transform = transform
        
        # Find all .npy files
        self.samples = list(class_dir.glob("*.npy"))
        
        if not self.samples:
            raise ValueError(f"No .npy files found in {class_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load spectrogram
        spec_path = self.samples[idx]
        spec = np.load(spec_path).astype(np.float32)
        
        # Ensure correct shape (512, 512)
        if spec.shape != (512, 512):
            raise ValueError(f"Expected shape (512, 512), got {spec.shape} for {spec_path}")
        
        # Add channel dimension: (512, 512) -> (1, 512, 512)
        spec = spec[None, :, :]
        
        # Apply transforms if provided
        if self.transform:
            spec = self.transform(spec)
        
        return torch.from_numpy(spec), self.class_index


def load_class_labels(class_labels_path: Path) -> dict:
    """Load class labels mapping."""
    with open(class_labels_path, 'r') as f:
        return json.load(f)


def create_extended_model(old_model_path: Path, old_num_classes: int, new_num_classes: int) -> RFClassifier:
    """
    Create new model with extended classification head.
    Copies all weights except final layer.
    """
    print(f"[MODEL] Creating extended model: {old_num_classes} -> {new_num_classes} classes")
    
    # Load old model
    checkpoint = torch.load(old_model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        old_state = checkpoint['model_state_dict']
    else:
        old_state = checkpoint
    
    # Create new model with more classes
    new_model = RFClassifier(num_classes=new_num_classes)
    new_state = new_model.state_dict()
    
    # Copy all weights except final classification layer (fc3)
    copied = 0
    skipped = 0
    for key in old_state:
        if key in new_state:
            if 'fc3' in key:
                # Skip final layer - will be randomly initialized
                skipped += 1
                print(f"[MODEL] Skipping {key} (new layer)")
            else:
                # Copy weight
                if old_state[key].shape == new_state[key].shape:
                    new_state[key] = old_state[key]
                    copied += 1
                else:
                    print(f"[MODEL] Shape mismatch for {key}: {old_state[key].shape} vs {new_state[key].shape}")
                    skipped += 1
    
    new_model.load_state_dict(new_state)
    print(f"[MODEL] Copied {copied} layers, skipped {skipped} layers")
    
    return new_model


def freeze_feature_extraction(model: RFClassifier):
    """Freeze convolutional layers, keep classification head trainable."""
    frozen = 0
    trainable = 0
    
    for name, param in model.named_parameters():
        if 'conv' in name or 'fc1' in name or 'fc2' in name:
            # Freeze feature extraction
            param.requires_grad = False
            frozen += 1
        else:
            # Keep classification head trainable
            param.requires_grad = True
            trainable += 1
    
    print(f"[MODEL] Frozen {frozen} parameters, {trainable} trainable")
    return model


def create_dataloaders(
    dataset_dir: Path,
    class_labels: dict,
    new_class_id: str,
    batch_size: int = 16,
    include_old_data: bool = True,
    old_data_ratio: float = 0.3
):
    """Create data loaders for fine-tuning."""
    class_to_index = class_labels.get("class_to_index", {})
    
    # Get new class index
    if new_class_id not in class_to_index:
        raise ValueError(f"Class {new_class_id} not found in class_labels.json")
    
    new_class_index = class_to_index[new_class_id]
    
    # Load new class data
    new_class_dir = dataset_dir / "train" / new_class_id
    if not new_class_dir.exists():
        raise ValueError(f"New class directory not found: {new_class_dir}")
    
    new_dataset = SpectrogramDataset(new_class_dir, new_class_index)
    print(f"[DATA] New class samples: {len(new_dataset)}")
    
    # Optionally include some old data to prevent catastrophic forgetting
    if include_old_data:
        old_datasets = []
        for class_id, class_idx in class_to_index.items():
            if class_id == new_class_id:
                continue
            
            class_dir = dataset_dir / "train" / class_id
            if class_dir.exists():
                class_dataset = SpectrogramDataset(class_dir, class_idx)
                # Sample subset of old data
                n_samples = max(1, int(len(class_dataset) * old_data_ratio))
                indices = random.sample(range(len(class_dataset)), min(n_samples, len(class_dataset)))
                subset = torch.utils.data.Subset(class_dataset, indices)
                old_datasets.append(subset)
        
        if old_datasets:
            old_dataset = ConcatDataset(old_datasets)
            print(f"[DATA] Old class samples: {len(old_dataset)}")
            
            # Combine new and old data
            combined_dataset = ConcatDataset([new_dataset, old_dataset])
        else:
            combined_dataset = new_dataset
    else:
        combined_dataset = new_dataset
    
    # Create data loader
    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=False
    )
    
    # Validation loader (if available)
    val_dir = dataset_dir / "val" / new_class_id
    val_dataset = None
    if val_dir.exists():
        val_dataset = SpectrogramDataset(val_dir, new_class_index)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        print(f"[DATA] Validation samples: {len(val_dataset)}")
    else:
        val_loader = None
        print(f"[DATA] No validation data found for new class")
    
    return dataloader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune RF classifier to add new class"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to existing trained model (.pth file)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dataset"),
        help="Path to dataset directory (default: data/dataset)"
    )
    parser.add_argument(
        "--new-class-id",
        type=str,
        required=True,
        help="New class ID (e.g., 'custom_fpv_drone_x')"
    )
    parser.add_argument(
        "--new-class-name",
        type=str,
        help="Human-readable name for new class (optional)"
    )
    parser.add_argument(
        "--class-labels",
        type=Path,
        default=Path("spear_edge/ml/models/class_labels.json"),
        help="Path to class_labels.json"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output path for fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--include-old-data",
        action="store_true",
        default=True,
        help="Include some old class data to prevent forgetting"
    )
    parser.add_argument(
        "--old-data-ratio",
        type=float,
        default=0.3,
        help="Ratio of old data to include (default: 0.3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load class labels
    if not args.class_labels.exists():
        print(f"[ERROR] Class labels file not found: {args.class_labels}")
        return 1
    
    class_labels = load_class_labels(args.class_labels)
    class_to_index = class_labels.get("class_to_index", {})
    
    # Verify new class exists
    if args.new_class_id not in class_to_index:
        print(f"[ERROR] Class {args.new_class_id} not found in class_labels.json")
        print(f"[ERROR] Available classes: {list(class_to_index.keys())}")
        return 1
    
    # Load existing model
    if not args.model_path.exists():
        print(f"[ERROR] Model file not found: {args.model_path}")
        return 1
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'num_classes' in checkpoint:
        old_num_classes = checkpoint['num_classes']
    else:
        # Try to infer from model
        print("[WARN] Could not determine num_classes from checkpoint, assuming 23")
        old_num_classes = 23
    
    new_num_classes = len(class_to_index)
    
    # Check if we're adding a new class or fine-tuning existing
    if new_num_classes > old_num_classes:
        # Adding new class - extend model
        print(f"[INFO] Extending model from {old_num_classes} to {new_num_classes} classes")
        model = create_extended_model(args.model_path, old_num_classes, new_num_classes)
    elif new_num_classes == old_num_classes:
        # Fine-tuning existing class - load model as-is
        print(f"[INFO] Fine-tuning existing model with {old_num_classes} classes (class: {args.new_class_id})")
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model = RFClassifier(num_classes=old_num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = RFClassifier(num_classes=old_num_classes)
            model.load_state_dict(checkpoint)
    else:
        print(f"[ERROR] New number of classes ({new_num_classes}) cannot be less than old ({old_num_classes})")
        return 1
    
    # Freeze feature extraction
    model = freeze_feature_extraction(model)
    
    # Move to device
    device = torch.device(args.device)
    model = model.to(device)
    print(f"[INFO] Using device: {device}")
    
    # Create data loaders
    try:
        train_loader, val_loader = create_dataloaders(
            args.dataset_dir,
            class_labels,
            args.new_class_id,
            args.batch_size,
            args.include_old_data,
            args.old_data_ratio
        )
    except Exception as e:
        print(f"[ERROR] Failed to create data loaders: {e}")
        return 1
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # Training loop
    print(f"\n[INFO] Starting fine-tuning for {args.epochs} epochs...")
    best_val_acc = 0.0
    train_history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        val_loss = None
        val_acc = None
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record history
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr
        })
        
        # Save full checkpoint (with optimizer states, scheduler, history)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'num_classes': old_num_classes if new_num_classes == old_num_classes else new_num_classes,
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': current_lr,
            'new_class_id': args.new_class_id,
            'train_history': train_history,
            'fine_tuned_from': str(args.model_path),
            'fine_tune_config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'include_old_data': args.include_old_data,
                'old_data_ratio': args.old_data_ratio,
            }
        }
        
        if val_loader:
            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint['best_val_acc'] = best_val_acc
                torch.save(checkpoint, args.output_path)
                print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            # Save model every epoch if no validation
            torch.save(checkpoint, args.output_path)
            print(f"  ✓ Saved model checkpoint (epoch {epoch})")
    
    print(f"\n[SUCCESS] Fine-tuning complete!")
    print(f"  Model saved to: {args.output_path}")
    if val_loader:
        print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

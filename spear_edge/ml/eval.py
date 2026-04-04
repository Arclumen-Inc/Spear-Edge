"""
Shared evaluation utilities for RF classifier training and fine-tuning.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dataset_fingerprint(dataset_dir: Path) -> Dict[str, Any]:
    npy_files = sorted(dataset_dir.rglob("*.npy"))
    hasher = hashlib.sha256()
    for p in npy_files:
        rel = str(p.relative_to(dataset_dir))
        st = p.stat()
        hasher.update(f"{rel}:{st.st_size}:{int(st.st_mtime)}".encode("utf-8"))
    return {
        "root": str(dataset_dir),
        "npy_count": len(npy_files),
        "fingerprint_sha256": hasher.hexdigest(),
    }


def compute_per_class_metrics(confusion: np.ndarray) -> Dict[str, Any]:
    per_class: Dict[str, Any] = {}
    n = confusion.shape[0]
    for i in range(n):
        tp = float(confusion[i, i])
        fp = float(confusion[:, i].sum() - tp)
        fn = float(confusion[i, :].sum() - tp)
        support = int(confusion[i, :].sum())
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[str(i)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0
    macro_precision = float(np.mean([v["precision"] for v in per_class.values()])) if per_class else 0.0
    macro_recall = float(np.mean([v["recall"] for v in per_class.values()])) if per_class else 0.0
    return {
        "per_class": per_class,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


def evaluate_with_confusion(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    if len(dataloader) == 0:
        return 0.0, 0.0, confusion
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            t_np = target.cpu().numpy()
            p_np = pred.cpu().numpy()
            for t, p in zip(t_np, p_np):
                if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
                    confusion[int(t), int(p)] += 1
    avg_loss = total_loss / len(dataloader)
    accuracy = (100.0 * correct / total) if total > 0 else 0.0
    return avg_loss, accuracy, confusion

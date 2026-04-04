#!/usr/bin/env python3
"""Fit temperature scaling on validation split for an existing checkpoint (no retrain)."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from spear_edge.ml.calibration import (  # noqa: E402
    collect_val_logits,
    fit_temperature,
    save_calibration_json,
)
from spear_edge.ml.infer_pytorch import RFClassifier  # noqa: E402
from spear_edge.ml.preprocess import CURRENT_PREPROCESS_SCHEMA  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_rf_classifier", project_root / "scripts" / "train_rf_classifier.py"
)
_train = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_train)
SpectrogramDataset = _train.SpectrogramDataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="Default: <checkpoint>.calibration.json")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    ncls = int(ckpt.get("num_classes", 23))
    model = RFClassifier(num_classes=ncls)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    val_ds = SpectrogramDataset(args.dataset_dir, split="val", transform=None)
    if len(val_ds) == 0:
        print("[ERROR] No validation samples in dataset")
        return 1
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    vl, yl = collect_val_logits(model, loader, device)
    T, metrics = fit_temperature(vl, yl)
    out = args.output or args.checkpoint.with_suffix(".calibration.json")
    save_calibration_json(out, T, metrics, CURRENT_PREPROCESS_SCHEMA)
    print(json.dumps({"temperature": T, "metrics": metrics, "path": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

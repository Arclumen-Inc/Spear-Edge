#!/usr/bin/env python3
"""Export RFClassifier checkpoint to ONNX (logits output, NCHW 1x1x512x512)."""

import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from spear_edge.ml.infer_pytorch import RFClassifier  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("spear_edge/ml/models/rf_classifier.onnx"),
    )
    p.add_argument("--opset", type=int, default=14)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ncls = int(ckpt.get("num_classes", 23))
    model = RFClassifier(num_classes=ncls)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    dummy = torch.randn(1, 1, 512, 512)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["input"],
        output_names=["logits"],
        opset_version=int(args.opset),
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"[ONNX] Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

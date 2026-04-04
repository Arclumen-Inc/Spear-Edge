"""PT vs ONNX logits parity for RFClassifier (optional deps)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from spear_edge.ml.infer_pytorch import RFClassifier  # noqa: E402


def test_export_and_onnx_matches_pytorch_logits(tmp_path):
    model = RFClassifier(num_classes=5)
    model.eval()
    x = torch.randn(1, 1, 512, 512, dtype=torch.float32)
    with torch.no_grad():
        ref = model(x).numpy()

    onnx_p = tmp_path / "m.onnx"
    torch.onnx.export(
        model,
        x,
        str(onnx_p),
        input_names=["input"],
        output_names=["logits"],
        opset_version=14,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    sess = ort.InferenceSession(
        str(onnx_p), providers=["CPUExecutionProvider"]
    )
    out = sess.run(
        [sess.get_outputs()[0].name],
        {sess.get_inputs()[0].name: x.numpy()},
    )[0]

    np.testing.assert_allclose(ref, out, rtol=1e-4, atol=1e-4)

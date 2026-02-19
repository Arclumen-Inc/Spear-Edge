import torch
import torch.nn as nn

class Dummy(nn.Module):
    def forward(self, x):
        return torch.softmax(torch.randn(x.shape[0], 5), dim=1)

model = Dummy()
x = torch.randn(1, 1, 512, 512)

torch.onnx.export(
    model, x, "spear_dummy.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=12,
)

print("Saved spear_dummy.onnx")

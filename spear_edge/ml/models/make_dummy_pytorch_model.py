#!/usr/bin/env python3
"""
Create a dummy PyTorch model for testing the classifier.
This model will produce random but consistent predictions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from spear_edge.ml.infer_pytorch import RFClassifier
import torch

def create_dummy_model(output_path: str = "spear_edge/ml/models/rf_classifier_dummy.pth", num_classes: int = 5):
    """Create and save a dummy PyTorch model."""
    print(f"Creating dummy PyTorch model with {num_classes} classes...")
    
    # Create model
    model = RFClassifier(num_classes=num_classes)
    
    # Initialize with small random weights (for testing)
    for param in model.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=0.01)
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    
    print(f"✓ Dummy model saved to: {output_path}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test the model
    print("\nTesting model...")
    model.eval()
    dummy_input = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0][:3].tolist()}")
    print("✓ Model test successful!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create dummy PyTorch model")
    parser.add_argument("--output", default="spear_edge/ml/models/rf_classifier_dummy.pth",
                       help="Output path for model")
    parser.add_argument("--num-classes", type=int, default=5,
                       help="Number of output classes")
    args = parser.parse_args()
    
    create_dummy_model(args.output, args.num_classes)

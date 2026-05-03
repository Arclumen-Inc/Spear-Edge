#!/bin/bash
# test_onnx_installation.sh - Test ONNX Runtime installation on Jetson

echo "=========================================="
echo "ONNX Runtime Installation Test"
echo "=========================================="
echo ""

# Test 1: Import ONNX Runtime
echo "Test 1: Importing ONNX Runtime..."
python3 << 'PYEOF'
import sys
try:
    import onnxruntime as ort
    print(f"✅ ONNX Runtime imported successfully")
    print(f"   Version: {ort.__version__}")
except ImportError as e:
    print(f"❌ Failed to import ONNX Runtime: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Installation test failed at import step"
    exit 1
fi

echo ""

# Test 2: Check providers
echo "Test 2: Checking execution providers..."
python3 << 'PYEOF'
import onnxruntime as ort

print("Available Execution Providers:")
providers = ort.get_available_providers()
for i, provider in enumerate(providers, 1):
    print(f"  {i}. {provider}")

print()
if 'TensorrtExecutionProvider' in providers:
    print("🎉 SUCCESS: TensorRT provider available!")
    print("   This provides the best performance on Jetson.")
elif 'CUDAExecutionProvider' in providers:
    print("✅ SUCCESS: CUDA provider available!")
    print("   Good performance with GPU acceleration.")
elif 'CPUExecutionProvider' in providers:
    print("⚠️  WARNING: Only CPU provider available.")
    print("   GPU acceleration not working.")
else:
    print("❌ ERROR: No providers available!")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Provider test failed"
    exit 1
fi

echo ""

# Test 3: Test with dummy model
echo "Test 3: Testing with dummy model..."
cd ~/spear-edgev1_0 2>/dev/null || cd /home/spear/spear-edgev1_0

if [ -f "spear_edge/ml/models/spear_dummy.onnx" ]; then
    python3 << 'PYEOF'
import onnxruntime as ort
import numpy as np
import sys

model_path = 'spear_edge/ml/models/spear_dummy.onnx'
try:
    sess = ort.InferenceSession(model_path)
    print(f"✅ Model loaded: {model_path}")
    print(f"   Input: {sess.get_inputs()[0].name} {sess.get_inputs()[0].shape}")
    print(f"   Output: {sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}")
    print(f"   Active providers: {sess.get_providers()}")
    
    # Test inference with dummy data
    input_name = sess.get_inputs()[0].name
    dummy_input = np.random.randn(1, 1, 512, 512).astype(np.float32)
    output = sess.run(None, {input_name: dummy_input})
    print(f"✅ Inference test: SUCCESS")
    print(f"   Output shape: {output[0].shape}")
except Exception as e:
    print(f"⚠️  Model test failed: {e}")
    import traceback
    traceback.print_exc()
PYEOF
else
    echo "⚠️  Dummy model not found, skipping model test"
    echo "   (This is OK - model test is optional)"
fi

echo ""
echo "=========================================="
echo "Installation test complete!"
echo "=========================================="

#!/bin/bash
# ensure_onnx_build_survives.sh - Ensure ONNX build survives SSH disconnection

echo "=========================================="
echo "Ensuring ONNX Build Survives SSH Disconnect"
echo "=========================================="
echo ""

cd ~/onnxruntime

# Check if build is already running
BUILD_PID=$(pgrep -f "build.py.*onnxruntime" | head -1)

if [ -n "$BUILD_PID" ]; then
    echo "✅ Build is already running (PID: $BUILD_PID)"
    echo ""
    echo "To ensure it survives SSH disconnection:"
    echo "1. The process is already backgrounded"
    echo "2. Use 'disown' to detach it from this shell"
    echo ""
    echo "Run this command to detach:"
    echo "  disown -h $BUILD_PID 2>/dev/null || echo 'Process may already be detached'"
    echo ""
    echo "Or check status anytime with:"
    echo "  ~/spear-edgev1_0/scripts/check_onnx_build_status.sh"
    echo ""
    
    # Try to disown it
    disown -h $BUILD_PID 2>/dev/null && echo "✅ Process detached (will survive SSH disconnect)" || echo "⚠️  Could not disown (may already be detached)"
    
else
    echo "❌ Build process not found"
    echo ""
    echo "Starting build with nohup (will survive SSH disconnect)..."
    echo ""
    
    # Set environment
    export CUDA_HOME=/usr/local/cuda
    export CUDNN_HOME=/usr/lib/aarch64-linux-gnu
    export TENSORRT_HOME=/usr/src/tensorrt
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
    
    # Start build with nohup
    nohup ./build.sh \
        --config Release \
        --build_shared_lib \
        --parallel $(nproc) \
        --use_tensorrt \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/lib/aarch64-linux-gnu \
        --tensorrt_home /usr/src/tensorrt \
        --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 CMAKE_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu CMAKE_INCLUDE_PATH=/usr/include/aarch64-linux-gnu \
        --build_wheel \
        --skip_tests \
        --skip_submodule_sync > build.log 2>&1 &
    
    NEW_PID=$!
    echo "✅ Build started with nohup (PID: $NEW_PID)"
    echo "   Log file: ~/onnxruntime/build.log"
    echo "   Monitor with: tail -f ~/onnxruntime/build.log"
fi

echo ""
echo "=========================================="
echo "Build will continue even if SSH disconnects!"
echo "=========================================="

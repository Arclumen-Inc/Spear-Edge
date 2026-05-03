#!/bin/bash
# restart_onnx_build_cuda.sh - Restart ONNX build with CUDA only (no TensorRT)

echo "=========================================="
echo "Restarting ONNX Runtime Build (CUDA Only)"
echo "=========================================="
echo ""

cd ~/onnxruntime

# Kill any existing build
pkill -f "build.py.*onnxruntime" 2>/dev/null
sleep 2

# Set environment
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/usr/lib/aarch64-linux-gnu
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

# Start CUDA-only build (no TensorRT to avoid compilation errors)
echo "Starting build with CUDA support (no TensorRT)..."
echo "This avoids the abseil template compilation errors."
echo ""

nohup ./build.sh \
    --config Release \
    --build_shared_lib \
    --parallel 4 \
    --use_cuda \
    --cuda_home /usr/local/cuda \
    --cudnn_home /usr/lib/aarch64-linux-gnu \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=87 \
    --build_wheel \
    --skip_tests \
    --skip_submodule_sync > build_cuda.log 2>&1 &

NEW_PID=$!
echo "✅ Build started (PID: $NEW_PID)"
echo "   Log: ~/onnxruntime/build_cuda.log"
echo "   Monitor: tail -f ~/onnxruntime/build_cuda.log"
echo ""
echo "This build uses CUDA (good performance) but avoids TensorRT"
echo "to prevent the template compilation errors we encountered."
echo ""
echo "=========================================="

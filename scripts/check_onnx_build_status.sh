#!/bin/bash
# check_onnx_build_status.sh - Quick status check for ONNX build

echo "=========================================="
echo "ONNX Runtime Build Status Check"
echo "=========================================="
echo ""

# Check if build is running
BUILD_PID=$(pgrep -f "build.py.*onnxruntime" | head -1)

if [ -n "$BUILD_PID" ]; then
    echo "✅ Build is RUNNING (PID: $BUILD_PID)"
    echo ""
    
    # Show process info
    ps -p $BUILD_PID -o pid,etime,pcpu,pmem,cmd | tail -1
    echo ""
    
    # Check build directory size
    if [ -d ~/onnxruntime/build/Linux ]; then
        SIZE=$(du -sh ~/onnxruntime/build/Linux 2>/dev/null | cut -f1)
        echo "Build directory size: $SIZE"
    fi
    
    # Check for wheel
    WHEEL=$(find ~/onnxruntime/build -name "*.whl" 2>/dev/null | head -1)
    if [ -n "$WHEEL" ]; then
        echo "✅ BUILD COMPLETE! Wheel file: $WHEEL"
        echo ""
        echo "To install, run:"
        echo "  pip3 install $WHEEL"
    else
        echo "⏳ Build in progress (no wheel file yet)"
    fi
else
    echo "❌ Build process NOT running"
    echo ""
    
    # Check if wheel exists (maybe it completed)
    WHEEL=$(find ~/onnxruntime/build -name "*.whl" 2>/dev/null | head -1)
    if [ -n "$WHEEL" ]; then
        echo "✅ Build completed! Wheel file found: $WHEEL"
        echo ""
        echo "To install, run:"
        echo "  pip3 install $WHEEL"
    else
        echo "Build may have stopped or failed."
        echo "Check logs: ~/onnxruntime/build.log"
    fi
fi

echo ""
echo "=========================================="

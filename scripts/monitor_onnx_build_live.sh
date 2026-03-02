#!/bin/bash
# monitor_onnx_build_live.sh - Live monitoring of ONNX build with tail -f

echo "=========================================="
echo "ONNX Runtime Build Live Monitor"
echo "=========================================="
echo ""
echo "Press Ctrl+C to exit (build will continue)"
echo ""

cd ~/onnxruntime

# Check if build.log exists
if [ -f "build.log" ]; then
    echo "Following build.log..."
    echo ""
    tail -f build.log
else
    echo "No build.log found. Checking build process..."
    echo ""
    
    BUILD_PID=$(pgrep -f "build.py.*onnxruntime" | head -1)
    if [ -n "$BUILD_PID" ]; then
        echo "✅ Build is running (PID: $BUILD_PID)"
        echo ""
        echo "To see live output, check:"
        echo "  ps aux | grep $BUILD_PID"
        echo ""
        echo "Or check build directory size:"
        du -sh build/Linux
    else
        echo "❌ Build process not found"
    fi
fi

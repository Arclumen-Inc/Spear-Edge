#!/bin/bash
# monitor_onnx_build.sh - Monitor ONNX Runtime build progress

echo "=========================================="
echo "ONNX Runtime Build Monitor"
echo "=========================================="
echo ""

cd ~/onnxruntime || exit 1

# Check if build is running
if pgrep -f "build.py.*onnxruntime" > /dev/null; then
    echo "✅ Build process is running"
    echo ""
    
    # Show build process info
    echo "Build processes:"
    ps aux | grep -E "(build.py|cmake|make)" | grep -v grep | awk '{print $2, $11, $12, $13, $14, $15}' | head -5
    echo ""
    
    # Check build directory size
    if [ -d "build/Linux" ]; then
        SIZE=$(du -sh build/Linux 2>/dev/null | cut -f1)
        echo "Build directory size: $SIZE"
    fi
    
    # Check for wheel file
    WHEEL=$(find build -name "*.whl" 2>/dev/null | head -1)
    if [ -n "$WHEEL" ]; then
        echo "✅ Wheel file found: $WHEEL"
        ls -lh "$WHEEL"
    else
        echo "⏳ Wheel file not created yet (build in progress)"
    fi
    
    # Check for recent build activity
    echo ""
    echo "Recent build activity (last 5 modified files):"
    find build -type f -mmin -10 2>/dev/null | head -5 || echo "No recent activity"
    
else
    echo "❌ Build process not running"
    echo ""
    echo "Checking for completed build..."
    
    WHEEL=$(find build -name "*.whl" 2>/dev/null | head -1)
    if [ -n "$WHEEL" ]; then
        echo "✅ Build completed! Wheel file: $WHEEL"
        ls -lh "$WHEEL"
        echo ""
        echo "To install, run:"
        echo "  pip3 install $WHEEL"
    else
        echo "❌ No wheel file found. Build may have failed."
        echo ""
        echo "Check build logs:"
        find build -name "*.log" -mtime -1 2>/dev/null | head -5
    fi
fi

echo ""
echo "=========================================="

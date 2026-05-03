#!/bin/bash
# Monitor RFUAV dataset download progress

DOWNLOAD_DIR="${1:-data/rfuav/train}"

echo "Monitoring RFUAV download progress..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    if [ -d "$DOWNLOAD_DIR" ]; then
        IMG_COUNT=$(find "$DOWNLOAD_DIR" -name "*.jpg" 2>/dev/null | wc -l)
        SIZE=$(du -sh "$DOWNLOAD_DIR" 2>/dev/null | awk '{print $1}')
        CLASS_COUNT=$(find "$DOWNLOAD_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        
        echo -ne "\r[$(date +%H:%M:%S)] Images: $IMG_COUNT | Size: $SIZE | Classes: $CLASS_COUNT"
    else
        echo -ne "\r[$(date +%H:%M:%S)] Loading dataset..."
    fi
    
    sleep 2
done

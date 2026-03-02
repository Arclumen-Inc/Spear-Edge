# RFUAV Download Status

**Started**: 2025-03-01  
**Status**: In Progress

## Download Information

- **Dataset**: kitofrank/RFUAV from Hugging Face
- **Split**: train (5,679 samples)
- **Output**: `data/rfuav/train/`
- **Expected Size**: ~5GB
- **Expected Time**: 30-60 minutes (depending on connection)

## Monitoring Progress

### Option 1: Monitor Script (Recommended)
```bash
./scripts/monitor_rfuav_download.sh
```

This will show real-time updates:
```
[10:57:30] Images: 1234 | Size: 1.2G | Classes: 37
```

### Option 2: Manual Check
```bash
# Count images
find data/rfuav/train -name "*.jpg" | wc -l

# Check size
du -sh data/rfuav

# Check classes
ls data/rfuav/train/
```

### Option 3: Check Process
```bash
ps aux | grep download_rfuav
```

## Progress Bar

The download script uses `tqdm` for progress visualization. The progress bar shows:
- Total images processed
- Images saved
- Number of classes
- Estimated time remaining

## Download Phases

1. **Initial Load** (2-5 minutes)
   - Downloading dataset metadata from Hugging Face
   - Caching dataset structure
   - No images yet - this is normal

2. **Image Download** (30-60 minutes)
   - Progress bar visible
   - Images being saved
   - Size increasing

3. **Completion**
   - ~5,679 images downloaded
   - ~5GB total size
   - 37 class directories

## Troubleshooting

### Download seems stuck
- Check internet connection
- Verify process is running: `ps aux | grep download_rfuav`
- Check disk space: `df -h .`

### Slow download
- Normal for large datasets
- Hugging Face may throttle downloads
- Can resume if interrupted (uses cache)

### Out of space
- Check available space: `df -h .`
- Need ~15GB free (download + processing)
- Can download to external drive if needed

## After Download Completes

1. **Verify download**:
   ```bash
   find data/rfuav/train -name "*.jpg" | wc -l
   # Should show ~5,679 images
   ```

2. **Prepare dataset**:
   ```bash
   python3 scripts/prepare_training_dataset.py \
       --rfuav-dir data/rfuav \
       --spear-dir data/dataset_raw \
       --output-dir data/dataset
   ```

3. **Ready for training**!

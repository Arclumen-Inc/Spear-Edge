# ML Detection Phase - Future Enhancement

**Status**: Planned for Phase 2 (after classification is complete)  
**Date**: 2025-03-01  
**Purpose**: Enable standalone operation without Tripwire dependency

---

## Overview

Currently, SPEAR-Edge relies on Tripwire for signal detection (coarse detection), then refines with classification. Adding detection capability will allow Edge to operate **standalone** - scanning wide frequency ranges and detecting drone signals independently.

## Current Architecture (Classification Only)

```
Tripwire → Detection (coarse) → Edge → Classification (fine-grained)
         "FHSS-like signal"          "ELRS with 80% confidence"
```

## Future Architecture (Detection + Classification)

```
Edge → Detection → Classification
     "Signal at 915 MHz" → "ELRS with 80% confidence"
```

---

## Detection vs Classification

### Detection (Phase 2)
- **Question**: "Where is the drone signal?"
- **Task**: Locate signals in spectrograms (frequency, time coordinates)
- **Model**: YOLOv5 or similar object detection model
- **Input**: Full spectrogram (512x512 or larger)
- **Output**: Bounding boxes/regions with confidence scores
  ```json
  {
    "detections": [
      {
        "freq_start_hz": 915000000,
        "freq_end_hz": 917000000,
        "time_start_s": 2.3,
        "time_end_s": 4.7,
        "confidence": 0.92,
        "bbox": [x1, y1, x2, y2]  // Spectrogram coordinates
      }
    ]
  }
  ```

### Classification (Phase 1 - Current)
- **Question**: "What type of drone is this?"
- **Task**: Identify device/protocol from spectrogram
- **Model**: CNN-based classifier (RFClassifier)
- **Input**: 512x512 spectrogram (focused region)
- **Output**: Class label with confidence
  ```json
  {
    "label": "elrs",
    "confidence": 0.80,
    "device_name": "ExpressLRS",
    "signal_type": "fhss_control"
  }
  ```

---

## Two-Stage Pipeline (RFUAV Approach)

Based on [RFUAV repository](https://github.com/kitoweeknd/RFUAV/), the recommended approach is:

### Stage 1: Detection
- Scan wide frequency ranges
- Detect signal regions in spectrograms
- Localize signals (frequency, time)
- Filter by confidence threshold

### Stage 2: Classification
- Extract detected regions
- Classify each detected signal
- Return device/protocol identification

### Combined Output
```json
{
  "detections": [
    {
      "region": {
        "freq_start_hz": 915000000,
        "freq_end_hz": 917000000,
        "time_start_s": 2.3,
        "time_end_s": 4.7
      },
      "classification": {
        "label": "elrs",
        "confidence": 0.80,
        "device_name": "ExpressLRS"
      }
    }
  ]
}
```

---

## Implementation Plan

### Phase 1: Classification (Current) ✅
- [x] Hierarchical classifier architecture
- [x] 23 device/protocol classes
- [x] GPU-accelerated inference
- [x] Fine-tuning capability
- [ ] Initial model training
- [ ] Production deployment

### Phase 2: Detection (Future)

#### 2.1 Detection Model Development
- [ ] Research YOLOv5 adaptation for spectrograms
- [ ] Design detection architecture
- [ ] Create detection dataset (annotate spectrograms with bounding boxes)
- [ ] Train detection model
- [ ] Optimize for Jetson (TensorRT, quantization)

#### 2.2 Integration with Classification
- [ ] Create two-stage pipeline
- [ ] Detection → Classification workflow
- [ ] Region extraction from detections
- [ ] Combined inference pipeline

#### 2.3 Standalone Operation Mode
- [ ] Wide-band scanning mode
- [ ] Automatic signal detection
- [ ] Detection-triggered captures
- [ ] UI integration for detection visualization

#### 2.4 Performance Optimization
- [ ] Batch processing for multiple detections
- [ ] GPU acceleration for detection
- [ ] Real-time detection on live FFT stream
- [ ] Memory optimization for large spectrograms

---

## Detection Model Architecture

### Option 1: YOLOv5 (RFUAV Approach)
- **Pros**: Proven for object detection, RFUAV provides reference
- **Cons**: May need adaptation for spectrograms
- **Input**: Full spectrogram (512x512 or larger)
- **Output**: Bounding boxes with confidence scores

### Option 2: Custom CNN Detection Head
- **Pros**: Can share feature extraction with classification model
- **Cons**: More custom development
- **Architecture**: Shared CNN backbone + detection head

### Option 3: Hybrid Approach
- Use existing RFClassifier feature extraction
- Add detection head (bounding box regression)
- Multi-task learning (detection + classification)

---

## Dataset Requirements

### Detection Dataset
- **Format**: Spectrograms with bounding box annotations
- **Annotations**: 
  - Frequency range (Hz)
  - Time range (seconds)
  - Signal type (for training)
- **Sources**:
  - RFUAV dataset (if detection annotations available)
  - SPEAR-Edge captures (manual annotation)
  - Synthetic data generation

### Annotation Format
```json
{
  "spectrogram_path": "data/dataset/train/elrs/sample_001.npy",
  "annotations": [
    {
      "bbox": [100, 200, 150, 250],  // [x1, y1, x2, y2] in spectrogram coords
      "freq_start_hz": 915000000,
      "freq_end_hz": 917000000,
      "time_start_s": 2.3,
      "time_end_s": 4.7,
      "class": "elrs",
      "confidence": 1.0
    }
  ]
}
```

---

## Integration Points

### With Existing Classification
```python
# Two-stage pipeline
detections = detection_model.detect(spectrogram)
for detection in detections:
    region = extract_region(spectrogram, detection.bbox)
    classification = classification_model.classify(region)
    detection.classification = classification
```

### With Capture System
- Detection triggers automatic captures
- Similar to Tripwire integration
- Detection confidence threshold
- Cooldown periods

### With Live FFT Stream
- Real-time detection on waterfall
- Overlay detection boxes on UI
- Alert operator on high-confidence detections

---

## Benefits of Standalone Operation

1. **Independence**: No Tripwire dependency
2. **Wide-band scanning**: Detect signals across full spectrum
3. **Autonomous operation**: Fully self-contained system
4. **Flexibility**: Can still use Tripwire when available
5. **Redundancy**: Multiple detection methods

---

## Challenges

1. **Computational cost**: Detection is more expensive than classification
2. **Real-time performance**: Must detect fast enough for live operation
3. **False positives**: Need robust filtering
4. **Dataset annotation**: Time-consuming manual work
5. **Model complexity**: Two-stage pipeline adds complexity

---

## Timeline Estimate

**After Classification Phase 1 Complete:**
- Detection model development: 2-4 weeks
- Dataset annotation: 1-2 weeks
- Training and optimization: 1-2 weeks
- Integration and testing: 1-2 weeks
- **Total**: 5-10 weeks

---

## References

- [RFUAV Repository](https://github.com/kitoweeknd/RFUAV/) - Detection and classification models
- RFUAV Section 2.4: Custom Detection Models (YOLOv5)
- RFUAV Section 2.5: Two-Stage Detection and Classification
- Current ML Plan: `technical_data_package/ML_INFERENCE_PLAN.txt`

---

## Notes

- Detection is **optional** - classification can work standalone with manual captures
- Detection enables **autonomous operation** - key for standalone use cases
- Two-stage approach (detect → classify) is industry standard (RFUAV)
- Can leverage RFUAV detection code as reference
- Consider starting with simpler detection (energy-based) before ML detection

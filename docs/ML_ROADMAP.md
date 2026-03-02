# ML Implementation Roadmap

**Last Updated**: 2025-03-01  
**Status**: Phase 1 (Classification) in progress

---

## Overview

SPEAR-Edge ML implementation is planned in two phases:
1. **Phase 1: Classification** (Current) - Device/protocol identification
2. **Phase 2: Detection** (Future) - Standalone signal detection

---

## Phase 1: Classification (Current Focus)

### Status: ✅ Infrastructure Complete, ⏳ Training Pending

### Completed ✅
- [x] PyTorch GPU classifier architecture
- [x] Hierarchical classification (23 device/protocol classes)
- [x] Class labels mapping (`class_labels.json`)
- [x] Data preparation pipeline (`prepare_training_dataset.py`)
- [x] Fine-tuning script (`fine_tune_new_class.py`)
- [x] GPU acceleration (CUDA support)
- [x] Integration with capture manager

### In Progress ⏳
- [ ] RFUAV dataset download
- [ ] Training dataset preparation
- [ ] Initial model training (on development machine)
- [ ] Model validation and testing

### Pending 📋
- [ ] Production model deployment
- [ ] Performance optimization
- [ ] Real-world testing with captures
- [ ] Documentation and user guides

### Timeline
- **Infrastructure**: Complete ✅
- **Data Collection**: In progress
- **Training**: Pending
- **Deployment**: Pending

---

## Phase 2: Detection (Future)

### Status: 📋 Planned

### Goals
- Enable standalone operation (no Tripwire dependency)
- Wide-band signal detection
- Automatic signal localization
- Two-stage pipeline: Detection → Classification

### Requirements
- Detection model (YOLOv5 or custom)
- Detection dataset with annotations
- Integration with classification pipeline
- Real-time detection on live FFT stream

### Timeline Estimate
- **Start**: After Phase 1 complete
- **Duration**: 5-10 weeks
- **Dependencies**: Phase 1 classification model

### Detailed Plan
See: `docs/ML_DETECTION_PHASE.md`

---

## Current Architecture

```
Tripwire (optional) → Edge Classification
                    ↓
              "ELRS with 80% confidence"
```

## Future Architecture (Phase 2)

```
Edge Detection → Edge Classification
     ↓                    ↓
"Signal at 915 MHz" → "ELRS with 80% confidence"
```

---

## Implementation Checklist

### Phase 1: Classification
- [x] Classifier architecture
- [x] Class labels (23 classes)
- [x] Data preparation pipeline
- [x] Fine-tuning capability
- [ ] RFUAV dataset integration
- [ ] Initial model training
- [ ] Model deployment
- [ ] Production testing

### Phase 2: Detection
- [ ] Detection model architecture
- [ ] Detection dataset creation
- [ ] Detection model training
- [ ] Two-stage pipeline integration
- [ ] Standalone operation mode
- [ ] Real-time detection
- [ ] UI integration

---

## Key Milestones

### Phase 1 Milestones
1. ✅ **Infrastructure Complete** (2025-03-01)
   - Classifier, data pipeline, fine-tuning ready
2. ⏳ **Dataset Ready** (In progress)
   - RFUAV + SPEAR-Edge captures prepared
3. 📋 **Model Trained** (Pending)
   - Initial 23-class model trained and validated
4. 📋 **Production Deployed** (Pending)
   - Model deployed to Jetson, tested in field

### Phase 2 Milestones
1. 📋 **Detection Model Designed** (Future)
2. 📋 **Detection Dataset Annotated** (Future)
3. 📋 **Two-Stage Pipeline Working** (Future)
4. 📋 **Standalone Operation** (Future)

---

## Dependencies

### Phase 1 Dependencies
- ✅ PyTorch with CUDA (installed)
- ✅ GPU classifier (implemented)
- ⏳ Training dataset (in progress)
- 📋 Development machine with GPU (for training)

### Phase 2 Dependencies
- 📋 Phase 1 complete
- 📋 Detection model architecture
- 📋 Detection dataset
- 📋 Additional GPU memory (for detection)

---

## Notes

- **Phase 1 is priority**: Classification enables core functionality
- **Phase 2 is enhancement**: Detection enables standalone operation
- **Both phases are independent**: Can use classification without detection
- **RFUAV reference**: Use RFUAV code as reference for Phase 2

---

## Related Documents

- `docs/ML_DETECTION_PHASE.md` - Detailed detection phase plan
- `docs/FINE_TUNING_GUIDE.md` - Fine-tuning workflow
- `docs/HIERARCHICAL_CLASSIFIER_COMPLETE.md` - Phase 1 completion summary
- `technical_data_package/ML_INFERENCE_PLAN.txt` - Original ML plan

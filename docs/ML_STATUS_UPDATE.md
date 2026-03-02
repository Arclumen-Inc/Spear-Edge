# ML Plan Status Update - Post CUDA Installation

## Current State (Updated)

### Infrastructure ✅
- ✅ **PyTorch 2.4.0 with CUDA** - **NEWLY INSTALLED AND WORKING!**
- ✅ ONNX Runtime installed (CPU provider only)
- ✅ ONNX classifier implementation (infer_onnx.py)
- ✅ Capture pipeline generates ML-ready spectrograms (512x512 float32)
- ✅ Dataset export system (data/dataset_raw/)
- ✅ Classification integration in capture_manager.py
- ✅ Fallback chain: ONNX → Orchestrator → Stub

### Current Limitations (Updated)
- ⚠️ **ONNX Runtime: CPU-only** (no CUDA/TensorRT providers available)
- ⚠️ **No trained model** (using dummy model)
- ✅ **NEW: PyTorch CUDA available** - Can use for GPU inference!

## Key Opportunity

Since **PyTorch with CUDA is now working**, we have a new option:

### Option 1: Use PyTorch Directly for GPU Inference
**Advantages:**
- ✅ GPU acceleration available NOW
- ✅ No need for ONNX conversion
- ✅ Direct model training and inference
- ✅ Better integration with PyTorch ecosystem

**Implementation:**
- Create `spear_edge/ml/infer_pytorch.py` 
- Use PyTorch model directly on GPU
- Can train and deploy without ONNX conversion step

### Option 2: Enable ONNX Runtime GPU
**Advantages:**
- ✅ Keep existing ONNX infrastructure
- ✅ Model portability (ONNX is framework-agnostic)
- ✅ TensorRT optimization possible

**Challenges:**
- Need Jetson-specific onnxruntime-gpu build
- May require additional setup

## Recommended Next Steps

### Immediate (This Session)
1. **Create PyTorch-based classifier** (`infer_pytorch.py`)
   - Use GPU for inference
   - Match ONNX classifier interface
   - Test with dummy model

2. **Benchmark GPU vs CPU inference**
   - Compare PyTorch GPU vs ONNX CPU
   - Measure inference time
   - Verify performance improvement

### Short Term (Next Phase)
3. **Data Collection & Preparation**
   - Review existing captures in `data/dataset_raw/`
   - Create `scripts/prepare_training_dataset.py`
   - Organize data for training

4. **Model Development**
   - Design CNN architecture (per ML plan)
   - Create training script
   - Train initial model on development machine

### Medium Term
5. **Deploy Trained Model**
   - Transfer model to Jetson
   - Test with real captures
   - Monitor performance

## ML Plan Reference

Full plan available at:
- `technical_data_package/ML_INFERENCE_PLAN.txt`
- `docs/ML_INFERENCE_PLAN.txt`

## Current ML Infrastructure Files

```
spear_edge/ml/
├── infer_onnx.py          ✅ ONNX classifier (CPU-only currently)
├── infer_stub.py          ✅ Stub classifier (fallback)
├── models/
│   ├── spear_dummy.onnx   ✅ Dummy ONNX model
│   └── (rf_classifier.onnx - to be created)
└── __init__.py            ✅ Module initialization
```

## Next Action Items

1. **Create PyTorch GPU Classifier** (Recommended)
   - Leverage newly working PyTorch CUDA
   - Get GPU acceleration immediately
   - Can train models directly on Jetson if needed

2. **Test GPU Inference Performance**
   - Benchmark against CPU ONNX
   - Verify meets <50ms target (per ML plan)

3. **Review Data Collection Status**
   - Check `data/dataset_raw/` for existing captures
   - Assess data quality and quantity

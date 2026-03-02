# TorchSig Status and Integration Plan

## What is TorchSig?

**TorchSig** is a PyTorch library specifically designed for RF signal classification and processing. It provides:
- Pre-trained models for RF signal classification
- Dataset utilities for RF signals
- Signal processing transforms
- Model architectures optimized for spectrograms

## Current Status

### Installation Status
- ❌ **Not installed** - TorchSig is not available via pip for ARM64/Jetson
- ⚠️ **May need source installation** - May require building from GitHub

### Project References
- ✅ Mentioned in `requirements-ml-dev.txt`
- ✅ Documentation exists: `docs/CAPTURE_ARTIFACTS_ANALYSIS.md`
- ✅ Converter utility exists: `tests/torchsig_converter.py`
- ✅ Format compatibility analyzed (512x512 spectrograms are compatible)

## Integration Options

### Option 1: Install TorchSig from Source (If Available)
```bash
pip install git+https://github.com/RI-SE/torchsig.git
```

### Option 2: Use TorchSig Models/Architectures (Without Full Library)
- Extract model architectures from TorchSig
- Use PyTorch directly with similar architectures
- Train custom models using TorchSig-inspired designs

### Option 3: Skip TorchSig, Use Custom PyTorch Models
- Since PyTorch CUDA is working, we can build custom models
- Follow ML plan architecture (CNN for spectrograms)
- Train directly on Jetson or development machine

## Recommendation

Given that:
1. ✅ PyTorch with CUDA is working
2. ✅ Spectrogram format is ready (512x512 float32)
3. ⚠️ TorchSig may not be easily installable on Jetson ARM64

**Recommended Approach:**
- **For Training**: Use TorchSig on development machine (x86_64) if needed
- **For Inference**: Use PyTorch directly on Jetson with custom models
- **Alternative**: Build TorchSig from source if it's critical

## Next Steps

1. **Try installing from GitHub source** (if compatible)
2. **If that fails**: Proceed with custom PyTorch models per ML plan
3. **Use TorchSig architectures as reference** for model design

## ML Plan Alignment

The ML plan recommends:
- **Framework**: PyTorch ✅ (now working with CUDA)
- **Architecture**: CNN for spectrograms
- **Training**: On development machine
- **Deployment**: ONNX or PyTorch on Jetson

TorchSig would be **nice to have** but not **required** - we can build equivalent models with PyTorch.

# cuDNN 8 Download Instructions

## ⚠️ Important: You Need cuDNN 8, Not cuDNN 9!

The two links you provided are for **cuDNN 9**, but PyTorch requires **cuDNN 8**. 

## Correct Download Links for cuDNN 8

Based on available versions, here are the cuDNN 8 download options:

### Option A: cuDNN 8.9.5.30 (Recommended - Latest 8.x)
```bash
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-aarch64/cudnn-linux-aarch64-8.9.5.30_cuda12-archive.tar.xz
```

### Option B: cuDNN 8.9.0.131 (Alternative)
```bash
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-aarch64/cudnn-linux-aarch64-8.9.0.131_cuda12-archive.tar.xz
```

## Why Not cuDNN 9?

- ❌ Your links point to cuDNN 9.19.1.2
- ❌ PyTorch wheels for JetPack 6 are built against cuDNN 8
- ❌ cuDNN 9 has different version symbols that PyTorch can't use
- ✅ cuDNN 8 will work with your PyTorch installation

## Manual Download (If wget doesn't work)

If the direct wget links require authentication:

1. Visit: https://developer.nvidia.com/cudnn
2. Sign in with your NVIDIA Developer account
3. Navigate to "Download cuDNN"
4. Look for **cuDNN 8.x** versions (NOT 9.x)
5. Select:
   - **Version**: 8.9.5 or 8.9.0
   - **Platform**: Linux
   - **Architecture**: aarch64 (ARM64)
   - **CUDA Version**: 12.x
6. Download the `.tar.xz` archive (not the `.deb` package)

## After Download

Once you have the cuDNN 8 archive, run:
```bash
./install_cudnn8.sh
```

The script will guide you through the installation.

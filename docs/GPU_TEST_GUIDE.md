# GPU Stress Test Script Guide

## Quick Start

### Basic Test (Quick Verification)
```bash
python3 test_gpu.py
```
Runs basic GPU functionality tests - takes ~5 seconds

### Full Benchmark
```bash
python3 test_gpu.py --benchmark
```
Runs performance benchmarks with detailed metrics

### Stress Test (Recommended for Stability Testing)
```bash
# 5 minute stress test
python3 test_gpu.py --stress 300

# 1 hour stress test
python3 test_gpu.py --stress 3600
```

### Memory Test
```bash
python3 test_gpu.py --memory
```
Tests GPU memory allocation and deallocation

### Combined Tests
```bash
# Full suite with 10-minute stress test
python3 test_gpu.py --benchmark --memory --stress 600
```

## Test Options

| Option | Description | Example |
|--------|-------------|---------|
| `--stress N` | Run continuous stress test for N seconds | `--stress 300` |
| `--benchmark` | Run performance benchmarks | `--benchmark` |
| `--size N` | Matrix size for tests (default: 2048) | `--size 4096` |
| `--iterations N` | Benchmark iterations (default: 10) | `--iterations 20` |
| `--memory` | Run memory allocation tests | `--memory` |

## What the Script Tests

1. **GPU Information**
   - CUDA availability
   - GPU name and specs
   - Memory capacity
   - Compute capability

2. **Basic Operations**
   - Tensor creation on GPU
   - Matrix multiplication
   - Element-wise operations

3. **Performance Benchmark**
   - Matrix multiplication speed (GFLOPS)
   - Convolution operations
   - Average operation times

4. **Stress Test**
   - Continuous GPU computation
   - Memory usage monitoring
   - Error detection
   - Performance stability

5. **Memory Test**
   - Progressive memory allocation
   - Memory cleanup verification
   - Maximum allocation capacity

## Example Output

```
============================================================
  GPU Information
============================================================
CUDA Available: True
CUDA Version: 12.2
cuDNN Version: 8905
Number of GPUs: 1

GPU 0: Orin
  Compute Capability: 8.7
  Total Memory: 7.44 GB
  Multiprocessors: 8

============================================================
  Basic Operations Test
============================================================
✓ Tensors created: torch.Size([1000, 1000])
✓ Matrix multiplication: 84.23 ms
✓ Element-wise ops: 175.67 ms
```

## Recommended Stress Test Durations

- **Quick test**: 60 seconds (`--stress 60`)
- **Standard test**: 5-10 minutes (`--stress 300` or `--stress 600`)
- **Extended test**: 30-60 minutes (`--stress 1800` or `--stress 3600`)
- **Overnight test**: Several hours (`--stress 28800` for 8 hours)

## Monitoring During Stress Test

The script will print progress every 50 iterations:
```
Iteration   50 | Time:  12.5s | Memory: 2.34GB / 2.50GB
Iteration  100 | Time:  25.1s | Memory: 2.34GB / 2.50GB
```

## Notes

- The `NvMapMemAllocInternalTagged` warnings are normal on Jetson devices and can be ignored
- Press `Ctrl+C` to stop stress test early
- Monitor system temperature during extended stress tests
- The script automatically handles memory cleanup

## Troubleshooting

If tests fail:
1. Check GPU is not being used by another process
2. Verify PyTorch CUDA installation: `python3 -c "import torch; print(torch.cuda.is_available())"`
3. Check GPU memory: `nvidia-smi`
4. Try smaller matrix size: `--size 1024`

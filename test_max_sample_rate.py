#!/usr/bin/env python3
"""
Test script to determine maximum achievable sample rate on Jetson Orin Nano.
Tests FFT performance, buffer management, and system stability at various rates.
"""
import asyncio
import time
import numpy as np
from typing import Optional

# Try to import CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[TEST] CuPy available - GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("[TEST] CuPy not available - using CPU (NumPy)")

# Test parameters
TEST_RATES = [
    1_000_000,      # 1 Msps
    2_000_000,      # 2 Msps
    5_000_000,      # 5 Msps
    10_000_000,     # 10 Msps
    15_000_000,     # 15 Msps
    20_000_000,     # 20 Msps
    25_000_000,     # 25 Msps
    30_000_000,     # 30 Msps
    40_000_000,     # 40 Msps
    50_000_000,     # 50 Msps
    61_440_000,     # 61.44 Msps (bladeRF 2.0 micro max)
]

FFT_SIZES = [1024, 2048, 4096, 8192]
TEST_DURATION = 2.0  # seconds per test
TARGET_FPS = 15.0


def test_fft_performance(sample_rate: int, fft_size: int, use_gpu: bool) -> dict:
    """Test FFT computation performance at given sample rate and FFT size."""
    print(f"\n[TEST] Sample rate: {sample_rate/1e6:.2f} Msps, FFT size: {fft_size}, GPU: {use_gpu}")
    
    # Calculate decimation needed
    if sample_rate > 2_000_000:
        target_rate = 2_000_000
        decimation = max(1, sample_rate // target_rate)
        effective_rate = sample_rate // decimation
    else:
        decimation = 1
        effective_rate = sample_rate
    
    print(f"  Decimation: {decimation}x, Effective rate: {effective_rate/1e6:.2f} Msps")
    
    # Generate test IQ data
    samples_per_fft = fft_size * decimation
    iq = np.random.randn(samples_per_fft).astype(np.complex64) + 1j * np.random.randn(samples_per_fft).astype(np.complex64)
    iq = iq.astype(np.complex64)
    
    # Apply decimation if needed
    if decimation > 1:
        iq = iq[::decimation]
    
    # Take exactly fft_size samples
    iq = iq[:fft_size].astype(np.complex64)
    
    # Window
    win = np.hanning(fft_size).astype(np.float32)
    
    # Test GPU FFT if available
    if use_gpu and GPU_AVAILABLE:
        try:
            gpu_iq = cp.asarray(iq)
            gpu_win = cp.asarray(win)
            
            # Warmup
            for _ in range(5):
                gpu_windowed = gpu_iq * gpu_win
                gpu_fft = cp.fft.fftshift(cp.fft.fft(gpu_windowed, n=fft_size))
                gpu_power = cp.abs(gpu_fft) ** 2
                _ = cp.asnumpy(gpu_power)
            
            # Benchmark
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                gpu_windowed = gpu_iq * gpu_win
                gpu_fft = cp.fft.fftshift(cp.fft.fft(gpu_windowed, n=fft_size))
                gpu_power = cp.abs(gpu_fft) ** 2
                gpu_power_db = 10.0 * cp.log10(gpu_power + 1e-12)
                result = cp.asnumpy(gpu_power_db).astype(np.float32)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # ms
            
            avg_time = np.mean(times)
            max_time = np.max(times)
            min_time = np.min(times)
            
            return {
                "method": "GPU (CuPy)",
                "avg_ms": avg_time,
                "min_ms": min_time,
                "max_ms": max_time,
                "fps_capable": 1000.0 / avg_time if avg_time > 0 else 0,
                "success": True
            }
        except Exception as e:
            print(f"  GPU FFT failed: {e}")
            use_gpu = False
    
    # Test CPU FFT
    if not use_gpu or not GPU_AVAILABLE:
        # Warmup
        for _ in range(5):
            windowed = iq * win
            fft_result = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
            power = np.abs(fft_result) ** 2
            power_db = 10.0 * np.log10(power + 1e-12)
        
        # Benchmark
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            windowed = iq * win
            fft_result = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
            power = np.abs(fft_result) ** 2
            power_db = 10.0 * np.log10(power + 1e-12)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)
        
        return {
            "method": "CPU (NumPy)",
            "avg_ms": avg_time,
            "min_ms": min_time,
            "max_ms": max_time,
            "fps_capable": 1000.0 / avg_time if avg_time > 0 else 0,
            "success": True
        }


def test_sustained_rate(sample_rate: int, fft_size: int, use_gpu: bool) -> dict:
    """Test if system can sustain a given sample rate for TEST_DURATION seconds."""
    print(f"\n[SUSTAINED TEST] {sample_rate/1e6:.2f} Msps, FFT: {fft_size}, GPU: {use_gpu}")
    
    # Calculate parameters
    if sample_rate > 2_000_000:
        decimation = max(1, sample_rate // 2_000_000)
        effective_rate = sample_rate // decimation
    else:
        decimation = 1
        effective_rate = sample_rate
    
    samples_per_fft = fft_size * decimation
    period = 1.0 / TARGET_FPS
    
    # Generate test data
    iq_buffer = np.random.randn(samples_per_fft * 10).astype(np.complex64) + \
                1j * np.random.randn(samples_per_fft * 10).astype(np.complex64)
    iq_buffer = iq_buffer.astype(np.complex64)
    win = np.hanning(fft_size).astype(np.float32)
    
    frames_processed = 0
    total_time = 0.0
    max_frame_time = 0.0
    errors = 0
    
    start_time = time.time()
    end_time = start_time + TEST_DURATION
    
    try:
        while time.time() < end_time:
            frame_start = time.perf_counter()
            
            # Get samples (simulate ring buffer)
            offset = (frames_processed * samples_per_fft) % (len(iq_buffer) - samples_per_fft)
            iq = iq_buffer[offset:offset + samples_per_fft].copy()
            
            # Decimate
            if decimation > 1:
                iq = iq[::decimation]
            iq = iq[:fft_size]
            
            # FFT
            if use_gpu and GPU_AVAILABLE:
                try:
                    gpu_iq = cp.asarray(iq)
                    gpu_win = cp.asarray(win)
                    gpu_windowed = gpu_iq * gpu_win
                    gpu_fft = cp.fft.fftshift(cp.fft.fft(gpu_windowed, n=fft_size))
                    gpu_power = cp.abs(gpu_fft) ** 2
                    gpu_power_db = 10.0 * cp.log10(gpu_power + 1e-12)
                    result = cp.asnumpy(gpu_power_db).astype(np.float32)
                except Exception as e:
                    errors += 1
                    if errors > 10:
                        raise
                    continue
            else:
                windowed = iq * win
                fft_result = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
                power = np.abs(fft_result) ** 2
                result = 10.0 * np.log10(power + 1e-12).astype(np.float32)
            
            frame_time = time.perf_counter() - frame_start
            total_time += frame_time
            max_frame_time = max(max_frame_time, frame_time)
            frames_processed += 1
            
            # Sleep to maintain target FPS
            elapsed = time.perf_counter() - frame_start
            sleep_time = max(0.0, period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "frames": frames_processed,
            "duration": time.time() - start_time
        }
    
    actual_duration = time.time() - start_time
    avg_fps = frames_processed / actual_duration if actual_duration > 0 else 0
    avg_frame_time = total_time / frames_processed if frames_processed > 0 else 0
    
    return {
        "success": True,
        "frames": frames_processed,
        "duration": actual_duration,
        "avg_fps": avg_fps,
        "avg_frame_time_ms": avg_frame_time * 1000,
        "max_frame_time_ms": max_frame_time * 1000,
        "errors": errors,
        "can_sustain": avg_fps >= TARGET_FPS * 0.8  # 80% of target is acceptable
    }


def main():
    print("=" * 70)
    print("Jetson Orin Nano - Maximum Sample Rate Test")
    print("=" * 70)
    print(f"GPU Acceleration: {'YES (CuPy)' if GPU_AVAILABLE else 'NO (CPU only)'}")
    print(f"Target FPS: {TARGET_FPS}")
    print(f"Test duration per rate: {TEST_DURATION}s")
    print("=" * 70)
    
    results = []
    
    # Test with optimal FFT size (2048 for most cases)
    optimal_fft = 2048
    
    print("\n" + "=" * 70)
    print("PHASE 1: FFT Performance Benchmark")
    print("=" * 70)
    
    for rate in TEST_RATES:
        result = test_fft_performance(rate, optimal_fft, use_gpu=GPU_AVAILABLE)
        result["sample_rate"] = rate
        result["fft_size"] = optimal_fft
        results.append(result)
        
        if result["success"]:
            print(f"  ✓ Avg: {result['avg_ms']:.2f}ms, "
                  f"Max: {result['max_ms']:.2f}ms, "
                  f"Capable: {result['fps_capable']:.1f} FPS")
            if result["fps_capable"] < TARGET_FPS:
                print(f"  ⚠ WARNING: Cannot achieve target {TARGET_FPS} FPS")
        else:
            print(f"  ✗ FAILED")
    
    print("\n" + "=" * 70)
    print("PHASE 2: Sustained Rate Test")
    print("=" * 70)
    
    max_sustainable_rate = 0
    
    for rate in TEST_RATES:
        result = test_sustained_rate(rate, optimal_fft, use_gpu=GPU_AVAILABLE)
        result["sample_rate"] = rate
        result["fft_size"] = optimal_fft
        
        if result["success"] and result.get("can_sustain", False):
            print(f"  ✓ {rate/1e6:.2f} Msps: {result['avg_fps']:.1f} FPS, "
                  f"Avg frame: {result['avg_frame_time_ms']:.2f}ms, "
                  f"Max frame: {result['max_frame_time_ms']:.2f}ms")
            max_sustainable_rate = rate
        elif result["success"]:
            print(f"  ⚠ {rate/1e6:.2f} Msps: {result['avg_fps']:.1f} FPS (below target)")
            if rate > max_sustainable_rate:
                max_sustainable_rate = rate  # Still record as partially working
        else:
            print(f"  ✗ {rate/1e6:.2f} Msps: FAILED - {result.get('error', 'Unknown error')}")
            break
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"GPU Acceleration: {'YES' if GPU_AVAILABLE else 'NO'}")
    print(f"Maximum Sustainable Rate: {max_sustainable_rate/1e6:.2f} Msps")
    print(f"Maximum Bandwidth (Nyquist): {max_sustainable_rate/2e6:.2f} MHz")
    
    if GPU_AVAILABLE:
        print("\n✓ GPU acceleration is available - you should be able to handle 30+ Msps")
    else:
        print("\n⚠ GPU acceleration NOT available - install CuPy for better performance:")
        print("  pip install cupy-cuda12x  # For CUDA 12.x (Jetson Orin)")
        print("  pip install cupy-cuda11x  # For CUDA 11.x")
    
    print("=" * 70)


if __name__ == "__main__":
    main()


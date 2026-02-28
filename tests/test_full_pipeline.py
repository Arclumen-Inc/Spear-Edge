#!/usr/bin/env python3
"""
Full pipeline test - simulates actual SDR data flow to find real bottlenecks.
Tests: Data generation → Ring buffer → Decimation → FFT → Frame creation
"""
import asyncio
import time
import numpy as np
from collections import deque

# Try to import CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Simulate ring buffer
class SimpleRingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.complex64)
        self.write_idx = 0
        self.size = 0
    
    def push(self, samples):
        n = samples.size
        if n == 0:
            return
        if n >= self.capacity:
            self.buffer[:] = samples[-self.capacity:]
            self.write_idx = 0
            self.size = self.capacity
            return
        
        end = (self.write_idx + n) % self.capacity
        if end < self.write_idx:
            split = self.capacity - self.write_idx
            self.buffer[self.write_idx:] = samples[:split]
            self.buffer[:end] = samples[split:]
        else:
            self.buffer[self.write_idx:end] = samples
        
        self.write_idx = end
        self.size = min(self.size + n, self.capacity)
    
    def pop(self, n):
        if self.size < n:
            return None
        # Simple pop from start
        result = self.buffer[:n].copy()
        self.buffer[:-n] = self.buffer[n:]
        self.size -= n
        return result


async def test_full_pipeline(sample_rate: int, fft_size: int, chunk_size: int, duration: float = 3.0):
    """Test full pipeline: data generation → ring buffer → decimation → FFT"""
    print(f"\n[PIPELINE TEST] {sample_rate/1e6:.2f} Msps, FFT: {fft_size}, Chunk: {chunk_size}")
    
    # Calculate decimation
    if sample_rate > 2_000_000:
        decimation = max(1, sample_rate // 2_000_000)
        effective_rate = sample_rate // decimation
    else:
        decimation = 1
        effective_rate = sample_rate
    
    # Ring buffer: 200ms at sample rate
    ring_size = int(sample_rate * 0.2)
    ring = SimpleRingBuffer(ring_size)
    
    # FFT parameters
    samples_per_fft = fft_size * decimation
    target_fps = 15.0
    period = 1.0 / target_fps
    
    # Pre-allocate
    win = np.hanning(fft_size).astype(np.float32)
    
    # Statistics
    frames_processed = 0
    buffer_underruns = 0
    max_frame_time = 0.0
    total_fft_time = 0.0
    total_decimation_time = 0.0
    total_buffer_time = 0.0
    
    # Producer: Simulate SDR reading
    async def producer():
        nonlocal buffer_underruns
        samples_generated = 0
        target_samples = int(sample_rate * duration)
        
        while samples_generated < target_samples:
            # Generate chunk of samples (simulate SDR read)
            chunk = np.random.randn(chunk_size).astype(np.complex64) + \
                   1j * np.random.randn(chunk_size).astype(np.complex64)
            chunk = chunk.astype(np.complex64)
            
            # Push to ring buffer
            ring.push(chunk)
            
            samples_generated += chunk_size
            
            # Simulate USB/PCIe delay
            if sample_rate > 20_000_000:
                await asyncio.sleep(0)  # No delay for very high rates
            elif sample_rate > 5_000_000:
                await asyncio.sleep(0.0001)  # 100us
            else:
                await asyncio.sleep(0.001)  # 1ms
    
    # Consumer: FFT processing
    async def consumer():
        nonlocal frames_processed, buffer_underruns, max_frame_time
        nonlocal total_fft_time, total_decimation_time, total_buffer_time
        
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            frame_start = time.perf_counter()
            
            # Get samples from ring buffer
            t0 = time.perf_counter()
            iq = ring.pop(samples_per_fft)
            t1 = time.perf_counter()
            total_buffer_time += (t1 - t0)
            
            if iq is None or iq.size < samples_per_fft:
                buffer_underruns += 1
                await asyncio.sleep(0.01)
                continue
            
            # Decimation
            t0 = time.perf_counter()
            if decimation > 1:
                iq = iq[::decimation].astype(np.complex64)
            iq = iq[:fft_size]
            t1 = time.perf_counter()
            total_decimation_time += (t1 - t0)
            
            # FFT
            t0 = time.perf_counter()
            if GPU_AVAILABLE:
                try:
                    gpu_iq = cp.asarray(iq)
                    gpu_win = cp.asarray(win)
                    gpu_windowed = gpu_iq * gpu_win
                    gpu_fft = cp.fft.fftshift(cp.fft.fft(gpu_windowed, n=fft_size))
                    gpu_power = cp.abs(gpu_fft) ** 2
                    power_db = cp.asnumpy(10.0 * cp.log10(gpu_power + 1e-12)).astype(np.float32)
                except:
                    # Fallback to CPU
                    windowed = iq * win
                    fft_result = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
                    power = np.abs(fft_result) ** 2
                    power_db = 10.0 * np.log10(power + 1e-12).astype(np.float32)
            else:
                windowed = iq * win
                fft_result = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
                power = np.abs(fft_result) ** 2
                power_db = 10.0 * np.log10(power + 1e-12).astype(np.float32)
            t1 = time.perf_counter()
            total_fft_time += (t1 - t0)
            
            frame_time = time.perf_counter() - frame_start
            max_frame_time = max(max_frame_time, frame_time)
            frames_processed += 1
            
            # Maintain target FPS
            elapsed = time.perf_counter() - frame_start
            sleep_time = max(0.0, period - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
    
    # Run both tasks
    start = time.time()
    try:
        await asyncio.gather(producer(), consumer(), return_exceptions=True)
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return None
    
    actual_duration = time.time() - start
    avg_fps = frames_processed / actual_duration if actual_duration > 0 else 0
    
    # Calculate percentages
    total_time = total_buffer_time + total_decimation_time + total_fft_time
    buffer_pct = (total_buffer_time / total_time * 100) if total_time > 0 else 0
    decimation_pct = (total_decimation_time / total_time * 100) if total_time > 0 else 0
    fft_pct = (total_fft_time / total_time * 100) if total_time > 0 else 0
    
    result = {
        "success": buffer_underruns < frames_processed * 0.1,  # <10% underruns
        "sample_rate": sample_rate,
        "frames": frames_processed,
        "duration": actual_duration,
        "avg_fps": avg_fps,
        "max_frame_time_ms": max_frame_time * 1000,
        "buffer_underruns": buffer_underruns,
        "underrun_rate": buffer_underruns / frames_processed if frames_processed > 0 else 1.0,
        "buffer_time_pct": buffer_pct,
        "decimation_time_pct": decimation_pct,
        "fft_time_pct": fft_pct,
        "can_sustain": avg_fps >= target_fps * 0.8 and buffer_underruns < frames_processed * 0.1
    }
    
    status = "✓" if result["can_sustain"] else "⚠" if result["success"] else "✗"
    print(f"  {status} FPS: {avg_fps:.1f}, Underruns: {buffer_underruns} ({result['underrun_rate']*100:.1f}%), "
          f"Max frame: {max_frame_time*1000:.2f}ms")
    print(f"    Time breakdown: Buffer {buffer_pct:.1f}%, Decimation {decimation_pct:.1f}%, FFT {fft_pct:.1f}%")
    
    return result


async def main():
    print("=" * 70)
    print("Full Pipeline Test - Real Bottleneck Detection")
    print("=" * 70)
    print(f"GPU: {'YES' if GPU_AVAILABLE else 'NO'}")
    print("=" * 70)
    
    test_rates = [2_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 61_440_000]
    fft_size = 2048
    
    results = []
    max_sustainable = 0
    
    for rate in test_rates:
        # Adaptive chunk sizing
        if rate > 20_000_000:
            chunk_size = 262144  # 256k
        elif rate > 5_000_000:
            chunk_size = 131072  # 128k
        else:
            chunk_size = 65536   # 64k
        
        result = await test_full_pipeline(rate, fft_size, chunk_size, duration=2.0)
        if result:
            results.append(result)
            if result["can_sustain"]:
                max_sustainable = rate
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Maximum Sustainable Rate: {max_sustainable/1e6:.2f} Msps")
    print(f"Maximum Bandwidth: {max_sustainable/2e6:.2f} MHz")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())


#!/usr/bin/env python3
"""
Stress test for BladeRFNativeDevice.
Tests multiple reads, different sample rates, and stability.
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/spear/spear-edgev1_0')

from spear_edge.core.sdr.bladerf_native import BladeRFNativeDevice
from spear_edge.core.sdr.base import GainMode

def test_multiple_reads():
    """Test multiple consecutive reads for stability."""
    print("\n[STRESS] Testing multiple consecutive reads...")
    sdr = BladeRFNativeDevice()
    sdr.tune(915_000_000, 10_000_000, 8_000_000)
    time.sleep(0.5)
    
    read_sizes = [8192, 16384, 32768, 65536]
    total_samples = 0
    successful_reads = 0
    failed_reads = 0
    
    start_time = time.time()
    for i in range(20):
        for size in read_sizes:
            samples = sdr.read_samples(size)
            if len(samples) == size:
                successful_reads += 1
                total_samples += len(samples)
            else:
                failed_reads += 1
                print(f"[STRESS] Failed read: expected {size}, got {len(samples)}")
    
    elapsed = time.time() - start_time
    throughput = (total_samples * 8) / (elapsed * 1024 * 1024)  # MB/s
    
    print(f"[STRESS] ✓ Total reads: {successful_reads + failed_reads}")
    print(f"[STRESS] ✓ Successful: {successful_reads}")
    print(f"[STRESS] ✓ Failed: {failed_reads}")
    print(f"[STRESS] ✓ Total samples: {total_samples:,}")
    print(f"[STRESS] ✓ Throughput: {throughput:.2f} MB/s")
    print(f"[STRESS] ✓ Success rate: {successful_reads/(successful_reads+failed_reads)*100:.1f}%")
    
    sdr.close()
    return failed_reads == 0

def test_different_sample_rates():
    """Test different sample rates."""
    print("\n[STRESS] Testing different sample rates...")
    sdr = BladeRFNativeDevice()
    
    rates = [5_000_000, 10_000_000, 20_000_000]
    all_passed = True
    
    for rate in rates:
        print(f"[STRESS] Testing {rate/1e6:.1f} MS/s...")
        sdr.tune(915_000_000, rate, rate * 0.8)
        time.sleep(0.5)
        
        samples = sdr.read_samples(8192)
        if len(samples) == 8192:
            print(f"[STRESS] ✓ {rate/1e6:.1f} MS/s: OK")
        else:
            print(f"[STRESS] ✗ {rate/1e6:.1f} MS/s: Failed (got {len(samples)} samples)")
            all_passed = False
    
    sdr.close()
    return all_passed

def test_different_frequencies():
    """Test different frequencies."""
    print("\n[STRESS] Testing different frequencies...")
    sdr = BladeRFNativeDevice()
    
    freqs = [433_000_000, 915_000_000, 2400_000_000]
    all_passed = True
    
    for freq in freqs:
        print(f"[STRESS] Testing {freq/1e6:.1f} MHz...")
        sdr.tune(freq, 10_000_000, 8_000_000)
        time.sleep(0.5)
        
        samples = sdr.read_samples(8192)
        if len(samples) == 8192:
            print(f"[STRESS] ✓ {freq/1e6:.1f} MHz: OK")
        else:
            print(f"[STRESS] ✗ {freq/1e6:.1f} MHz: Failed (got {len(samples)} samples)")
            all_passed = False
    
    sdr.close()
    return all_passed

def test_health_tracking():
    """Test health tracking over time."""
    print("\n[STRESS] Testing health tracking...")
    sdr = BladeRFNativeDevice()
    sdr.tune(915_000_000, 10_000_000, 8_000_000)
    time.sleep(0.5)
    
    # Read samples for a few seconds
    for _ in range(50):
        sdr.read_samples(8192)
        time.sleep(0.1)
    
    health = sdr.get_health()
    print(f"[STRESS] ✓ Status: {health['status']}")
    print(f"[STRESS] ✓ Success rate: {health['success_rate_pct']:.1f}%")
    print(f"[STRESS] ✓ Throughput: {health['throughput_mbps']:.2f} MB/s")
    print(f"[STRESS] ✓ Samples/sec: {health['samples_per_sec']:.2f} MS/s")
    print(f"[STRESS] ✓ Avg read time: {health['avg_read_time_ms']:.2f} ms")
    print(f"[STRESS] ✓ Errors: {health['errors']}")
    print(f"[STRESS] ✓ Timeouts: {health['timeouts']}")
    
    sdr.close()
    return health['success_rate_pct'] >= 95.0

def main():
    print("=" * 60)
    print("BladeRF Native Device Stress Test")
    print("=" * 60)
    
    tests = [
        ("Multiple Reads", test_multiple_reads),
        ("Different Sample Rates", test_different_sample_rates),
        ("Different Frequencies", test_different_frequencies),
        ("Health Tracking", test_health_tracking),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[STRESS] ✗ {name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Stress Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:25s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All stress tests passed!")
        return 0
    else:
        print("\n[WARNING] Some stress tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

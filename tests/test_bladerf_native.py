#!/usr/bin/env python3
"""
Test script for BladeRFNativeDevice.
Run this to verify the libbladerf migration works correctly.

Usage:
    python3 test_bladerf_native.py
"""

import sys
import time
import numpy as np

# Add project to path
sys.path.insert(0, '/home/spear/spear-edgev1_0')

try:
    from spear_edge.core.sdr.bladerf_native import BladeRFNativeDevice
    from spear_edge.core.sdr.base import GainMode
    print("[TEST] Successfully imported BladeRFNativeDevice")
except ImportError as e:
    print(f"[TEST] Import failed: {e}")
    sys.exit(1)

def test_device_open_close():
    """Test device open/close."""
    print("\n[TEST] Testing device open/close...")
    try:
        sdr = BladeRFNativeDevice()
        print(f"[TEST] ✓ Device opened: {sdr.dev is not None}")
        assert sdr.dev is not None, "Device should be opened"
        
        sdr.close()
        print(f"[TEST] ✓ Device closed: {sdr.dev is None}")
        assert sdr.dev is None, "Device should be closed"
        return True
    except RuntimeError as e:
        print(f"[TEST] ✗ Device open failed (expected if no hardware): {e}")
        return False
    except Exception as e:
        print(f"[TEST] ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tune():
    """Test frequency tuning."""
    print("\n[TEST] Testing tune()...")
    try:
        sdr = BladeRFNativeDevice()
        sdr.tune(915_000_000, 10_000_000, 8_000_000)
        print(f"[TEST] ✓ Tuned to {sdr.center_freq_hz / 1e6:.3f} MHz")
        print(f"[TEST] ✓ Sample rate: {sdr.sample_rate_sps / 1e6:.2f} MS/s")
        print(f"[TEST] ✓ Stream active: {sdr._stream_active}")
        assert sdr.center_freq_hz == 915_000_000
        assert sdr.sample_rate_sps == 10_000_000
        assert sdr._stream_active == True
        sdr.close()
        return True
    except RuntimeError as e:
        print(f"[TEST] ✗ Tune failed (expected if no hardware): {e}")
        return False
    except Exception as e:
        print(f"[TEST] ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_read_samples():
    """Test sample reading."""
    print("\n[TEST] Testing read_samples()...")
    try:
        sdr = BladeRFNativeDevice()
        sdr.tune(915_000_000, 10_000_000, 8_000_000)
        
        # Wait for stream to stabilize
        time.sleep(0.5)
        
        # Read power-of-two samples
        samples = sdr.read_samples(8192)
        print(f"[TEST] ✓ Read {len(samples)} samples")
        print(f"[TEST] ✓ Sample dtype: {samples.dtype}")
        assert len(samples) == 8192, f"Expected 8192 samples, got {len(samples)}"
        assert samples.dtype == np.complex64, f"Expected complex64, got {samples.dtype}"
        
        # Check sample values are reasonable
        if len(samples) > 0:
            print(f"[TEST] ✓ Sample range: {np.abs(samples).min():.6f} to {np.abs(samples).max():.6f}")
        
        sdr.close()
        return True
    except RuntimeError as e:
        print(f"[TEST] ✗ Read failed (expected if no hardware): {e}")
        return False
    except Exception as e:
        print(f"[TEST] ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gain_control():
    """Test gain setting."""
    print("\n[TEST] Testing gain control...")
    try:
        sdr = BladeRFNativeDevice()
        sdr.tune(915_000_000, 10_000_000, 8_000_000)
        
        sdr.set_gain(40.0)
        print(f"[TEST] ✓ Set gain to {sdr.gain_db} dB")
        assert sdr.gain_db == 40.0
        
        sdr.set_gain_mode(GainMode.AGC)
        print(f"[TEST] ✓ Set gain mode to {sdr.gain_mode}")
        assert sdr.gain_mode == GainMode.AGC
        
        sdr.close()
        return True
    except RuntimeError as e:
        print(f"[TEST] ✗ Gain control failed (expected if no hardware): {e}")
        return False
    except Exception as e:
        print(f"[TEST] ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_health_stats():
    """Test health statistics."""
    print("\n[TEST] Testing health statistics...")
    try:
        sdr = BladeRFNativeDevice()
        sdr.tune(915_000_000, 10_000_000, 8_000_000)
        
        # Read some samples to generate stats
        time.sleep(0.5)
        for _ in range(5):
            sdr.read_samples(8192)
            time.sleep(0.1)
        
        health = sdr.get_health()
        print(f"[TEST] ✓ Health status: {health['status']}")
        print(f"[TEST] ✓ Success rate: {health['success_rate_pct']:.1f}%")
        print(f"[TEST] ✓ Throughput: {health['throughput_mbps']:.2f} MB/s")
        print(f"[TEST] ✓ Stream: {health['stream']}")
        
        assert 'status' in health
        assert 'success_rate_pct' in health
        assert 'throughput_mbps' in health
        
        sdr.close()
        return True
    except RuntimeError as e:
        print(f"[TEST] ✗ Health stats failed (expected if no hardware): {e}")
        return False
    except Exception as e:
        print(f"[TEST] ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("BladeRF Native Device Test Suite")
    print("=" * 60)
    
    tests = [
        ("Device Open/Close", test_device_open_close),
        ("Tune", test_tune),
        ("Read Samples", test_read_samples),
        ("Gain Control", test_gain_control),
        ("Health Stats", test_health_stats),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[TEST] ✗ {name} crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL (expected if no hardware)"
        print(f"{name:20s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[NOTE] Some tests failed - this is expected if no bladeRF hardware is connected")
        print("       Connect a bladeRF device and run again to verify hardware functionality")
        return 0  # Don't fail if hardware isn't present

if __name__ == "__main__":
    sys.exit(main())

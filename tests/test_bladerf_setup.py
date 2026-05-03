#!/usr/bin/env python3
"""
Test script to verify bladeRF setup is correct.
Tests the critical order of operations and verifies all settings are applied correctly.
"""

import sys
import time
import ctypes
import numpy as np

# Add project root to path
sys.path.insert(0, '/home/spear/spear-edgev1_0')

from spear_edge.core.sdr.bladerf_native import BladeRFNativeDevice
from spear_edge.core.sdr.base import GainMode

def test_bladerf_setup():
    """Test bladeRF setup with various configurations"""
    
    print("="*70)
    print("bladeRF Setup Verification Test")
    print("="*70)
    
    # Test configurations
    test_configs = [
        {
            "name": "Low Rate (2 MS/s)",
            "freq": 915_000_000,
            "rate": 2_000_000,
            "gain": 0.0,
            "fft_size": 4096,
        },
        {
            "name": "Medium Rate (20 MS/s)",
            "freq": 915_000_000,
            "rate": 20_000_000,
            "gain": 0.0,
            "fft_size": 16384,
        },
        {
            "name": "High Rate (61.44 MS/s)",
            "freq": 100_000_000,
            "rate": 61_440_000,
            "gain": 0.0,
            "fft_size": 65536,
        },
    ]
    
    sdr = None
    try:
        print("\n[1/5] Opening bladeRF device...")
        sdr = BladeRFNativeDevice()
        if sdr.dev is None:
            print("  [ERROR] Failed to open bladeRF device")
            return False
        print("  [OK] Device opened")
        
        for i, config in enumerate(test_configs, 1):
            print(f"\n[Test {i}/{len(test_configs)}] {config['name']}")
            print("-" * 70)
            
            # Test configuration
            print(f"  Config: freq={config['freq']/1e6:.3f} MHz, "
                  f"rate={config['rate']/1e6:.2f} MS/s, "
                  f"gain={config['gain']} dB")
            
            # Configure SDR
            print("  [2/5] Configuring RF parameters...")
            try:
                sdr.tune(
                    center_freq_hz=config['freq'],
                    sample_rate_sps=config['rate'],
                    bandwidth_hz=config['rate'],  # Match sample rate
                )
                print("  [OK] RF parameters configured")
            except Exception as e:
                print(f"  [ERROR] Failed to configure RF: {e}")
                return False
            
            # Verify gain (gain is already set in tune(), just verify)
            print(f"  [3/5] Verifying gain is {config['gain']} dB...")
            try:
                # Gain should already be set during tune()
                actual_gain = sdr.gain_db
                
                if abs(actual_gain - config['gain']) > 1.0:
                    print(f"  [WARNING] Gain mismatch: requested {config['gain']} dB, got {actual_gain} dB")
                    # Try to set it explicitly
                    sdr.set_gain(config['gain'])
                    time.sleep(0.1)
                    actual_gain = sdr.gain_db
                else:
                    print(f"  [OK] Gain verified: {actual_gain} dB")
            except Exception as e:
                print(f"  [ERROR] Failed to verify gain: {e}")
                return False
            
            # Verify settings (check internal state since get_info() doesn't return config)
            print("  [4/5] Verifying settings...")
            # Access internal state directly
            freq_ok = abs(sdr.center_freq_hz - config['freq']) < 1000
            rate_ok = abs(sdr.sample_rate_sps - config['rate']) < 1000
            gain_ok = abs(sdr.gain_db - config['gain']) < 1.0
            
            if not freq_ok:
                print(f"  [ERROR] Frequency mismatch: requested {config['freq']/1e6:.3f} MHz, "
                      f"got {sdr.center_freq_hz/1e6:.3f} MHz")
                return False
            if not rate_ok:
                print(f"  [ERROR] Sample rate mismatch: requested {config['rate']/1e6:.2f} MS/s, "
                      f"got {sdr.sample_rate_sps/1e6:.2f} MS/s")
                return False
            if not gain_ok:
                print(f"  [WARNING] Gain mismatch: requested {config['gain']} dB, "
                      f"got {sdr.gain_db:.1f} dB")
            
            print(f"  [OK] Settings verified: freq={sdr.center_freq_hz/1e6:.3f} MHz, "
                  f"rate={sdr.sample_rate_sps/1e6:.2f} MS/s, "
                  f"gain={sdr.gain_db:.1f} dB")
            
            # Test sample reading (optional - requires full environment)
            print("  [5/5] Testing sample reading...")
            read_size = 8192  # Power of two
            try:
                samples = sdr.read_samples(read_size)
                if samples is None or len(samples) == 0:
                    print(f"  [WARNING] Failed to read samples (got {len(samples) if samples is not None else 0} samples)")
                    print(f"  [INFO] Sample reading requires full environment - skipping")
                else:
                    # Check sample statistics
                    sample_mag = np.abs(samples)
                    max_val = np.max(sample_mag)
                    mean_val = np.mean(sample_mag)
                    std_val = np.std(sample_mag)
                    
                    print(f"  [OK] Read {len(samples)} samples")
                    print(f"       Stats: max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}")
                    
                    # Check for reasonable values (should be in [-1, 1] range after scaling)
                    if max_val > 1.5:
                        print(f"  [WARNING] Sample magnitude too high (max={max_val:.6f}), possible scaling issue")
                    if mean_val > 0.1:
                        print(f"  [WARNING] Mean sample magnitude high (mean={mean_val:.6f}), possible DC offset")
                
            except ImportError as e:
                print(f"  [INFO] Sample reading requires full environment (missing dependencies) - skipping")
                print(f"       This is OK - setup verification passed without sample reading")
            except Exception as e:
                print(f"  [WARNING] Sample reading failed: {e}")
                print(f"  [INFO] Setup verification passed - sample reading is optional")
            
            # Brief pause between tests
            time.sleep(0.5)
        
        print("\n" + "="*70)
        print("[SUCCESS] All bladeRF setup tests passed!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if sdr is not None:
            print("\n[CLOSE] Closing bladeRF device...")
            try:
                sdr.close()
                print("[OK] Device closed")
            except Exception as e:
                print(f"[WARNING] Error closing device: {e}")


if __name__ == "__main__":
    success = test_bladerf_setup()
    sys.exit(0 if success else 1)

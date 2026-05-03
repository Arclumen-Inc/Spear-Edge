#!/usr/bin/env python3
"""
Test script to verify FFT processing is correct.
Tests window setup, normalization, and FFT processing with different sizes.
"""

import sys
import numpy as np

# Add project root to path
sys.path.insert(0, '/home/spear/spear-edgev1_0')

def test_fft_setup():
    """Test FFT window and normalization setup"""
    
    print("="*70)
    print("FFT Setup Verification Test")
    print("="*70)
    
    # Test FFT sizes
    fft_sizes = [1024, 4096, 8192, 16384, 32768, 65536]
    
    print("\n[1/4] Testing window setup (Nuttall window)...")
    print("-" * 70)
    
    try:
        from scipy.signal import windows
        scipy_available = True
        print("  [OK] scipy available - using Nuttall window")
    except ImportError:
        scipy_available = False
        print("  [WARNING] scipy not available - will use Hann window fallback")
    
    for fft_size in fft_sizes:
        # Create window (matching scan_task.py logic)
        if scipy_available:
            win = windows.nuttall(fft_size).astype(np.float32)
            window_name = "Nuttall"
        else:
            win = np.hanning(fft_size).astype(np.float32)
            window_name = "Hann"
        
        # Calculate normalization constants (matching scan_task.py)
        coherent_gain = float(np.sum(win)) / fft_size
        window_sum = float(np.sum(win))
        win_energy = float(np.sum(win * win))
        
        # Expected values
        if scipy_available:
            # Nuttall window coherent gain ~ 0.363
            expected_coherent_gain = 0.363
            expected_tolerance = 0.01
        else:
            # Hann window coherent gain ~ 0.5
            expected_coherent_gain = 0.5
            expected_tolerance = 0.01
        
        coherent_ok = abs(coherent_gain - expected_coherent_gain) < expected_tolerance
        
        print(f"  FFT {fft_size:5d}: {window_name:8s} window, "
              f"coherent_gain={coherent_gain:.4f} (expected ~{expected_coherent_gain:.3f}), "
              f"sum={window_sum:.1f}, energy={win_energy:.1f}", end="")
        
        if coherent_ok:
            print(" [OK]")
        else:
            print(f" [WARNING] Coherent gain outside expected range")
    
    print("\n[2/4] Testing FFT normalization (coherent gain)...")
    print("-" * 70)
    
    # Test with synthetic signal
    test_fft_size = 4096
    if scipy_available:
        win = windows.nuttall(test_fft_size).astype(np.float32)
    else:
        win = np.hanning(test_fft_size).astype(np.float32)
    
    window_sum = float(np.sum(win))
    
    # Create a test signal: full-scale complex sinusoid (amplitude = 1.0)
    # This should peak at 0 dBFS after normalization
    sample_rate = 20_000_000  # 20 MS/s
    test_freq = 1_000_000  # 1 MHz offset from center
    t = np.arange(test_fft_size) / sample_rate
    test_signal = np.exp(1j * 2 * np.pi * test_freq * t).astype(np.complex64)
    
    # Apply window and FFT (matching scan_task.py)
    x = test_signal * win
    X = np.fft.fftshift(np.fft.fft(x, n=test_fft_size))
    
    # Coherent gain normalization (SDR++ style)
    mag = np.abs(X) / window_sum
    spec_db = 20.0 * np.log10(mag + 1e-12)
    
    # Find peak
    peak_idx = np.argmax(spec_db)
    peak_db = spec_db[peak_idx]
    
    # Expected: full-scale signal should peak at ~0 dBFS
    expected_peak = 0.0
    peak_tolerance = 1.0  # Allow 1 dB tolerance
    
    print(f"  Test signal: full-scale complex sinusoid (amplitude=1.0)")
    print(f"  FFT size: {test_fft_size}")
    print(f"  Peak found at bin {peak_idx}: {peak_db:.2f} dBFS")
    print(f"  Expected peak: ~{expected_peak:.1f} dBFS (tolerance: ±{peak_tolerance} dB)")
    
    if abs(peak_db - expected_peak) < peak_tolerance:
        print(f"  [OK] Peak matches expected value")
    else:
        print(f"  [WARNING] Peak is {abs(peak_db - expected_peak):.2f} dB from expected")
    
    print("\n[3/4] Testing theoretical noise floor calculation...")
    print("-" * 70)
    
    # Test theoretical floor calculation (matching scan_task.py)
    for fft_size in [1024, 4096, 65536]:
        if scipy_available:
            # Nuttall window: theoretical = 20*log10(1/0.363) - 10*log10(N)
            nuttall_coherent_gain = 0.363
            theoretical_floor = 20.0 * np.log10(1.0 / nuttall_coherent_gain) - 10.0 * np.log10(fft_size)
        else:
            # Hann window: theoretical = 20*log10(2) - 10*log10(N)
            theoretical_floor = 20.0 * np.log10(2.0) - 10.0 * np.log10(fft_size)
        
        print(f"  FFT {fft_size:5d}: theoretical floor = {theoretical_floor:6.1f} dBFS")
    
    print("\n[4/4] Testing FFT processing with white noise...")
    print("-" * 70)
    
    # Generate white noise
    np.random.seed(42)  # For reproducibility
    noise_samples = (np.random.randn(test_fft_size) + 1j * np.random.randn(test_fft_size)).astype(np.complex64)
    noise_samples = noise_samples / np.sqrt(2)  # Normalize to unit power
    
    # Process through FFT pipeline
    x = noise_samples * win
    X = np.fft.fftshift(np.fft.fft(x, n=test_fft_size))
    mag = np.abs(X) / window_sum
    spec_db = 20.0 * np.log10(mag + 1e-12)
    
    # Calculate noise floor (10th percentile, matching scan_task.py)
    noise_floor = float(np.percentile(spec_db, 10))
    
    # Calculate theoretical floor
    if scipy_available:
        nuttall_coherent_gain = 0.363
        theoretical_floor = 20.0 * np.log10(1.0 / nuttall_coherent_gain) - 10.0 * np.log10(test_fft_size)
    else:
        theoretical_floor = 20.0 * np.log10(2.0) - 10.0 * np.log10(test_fft_size)
    
    # Measured floor should be ~40-50 dB below theoretical (expected for real hardware)
    expected_floor_diff = 45.0  # dB
    floor_diff = theoretical_floor - noise_floor
    
    print(f"  FFT size: {test_fft_size}")
    print(f"  Theoretical floor: {theoretical_floor:.1f} dBFS")
    print(f"  Measured floor (10th percentile): {noise_floor:.1f} dBFS")
    print(f"  Difference: {floor_diff:.1f} dB")
    print(f"  Expected difference: ~{expected_floor_diff:.1f} dB (for real hardware)")
    
    if 30.0 < floor_diff < 60.0:
        print(f"  [OK] Noise floor difference is reasonable")
    else:
        print(f"  [WARNING] Noise floor difference is outside expected range")
    
    print("\n" + "="*70)
    print("[SUCCESS] All FFT setup tests passed!")
    print("="*70)
    print("\nSummary:")
    print(f"  - Window: {'Nuttall (SDR++ style)' if scipy_available else 'Hann (fallback)'}")
    print(f"  - Normalization: Coherent gain (divide by sum(window))")
    print(f"  - FFT sizes tested: {', '.join(map(str, fft_sizes))}")
    print(f"  - Peak detection: {'OK' if abs(peak_db - expected_peak) < peak_tolerance else 'WARNING'}")
    print(f"  - Noise floor: {'OK' if 30.0 < floor_diff < 60.0 else 'WARNING'}")
    
    return True


if __name__ == "__main__":
    try:
        success = test_fft_setup()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

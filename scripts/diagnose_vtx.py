#!/usr/bin/env python3
"""
VTX Signal Detection Diagnostic Tool
Tests bladeRF hardware directly and compares with Edge processing
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from spear_edge.core.sdr.bladerf_native import BladeRFNativeDevice
    from spear_edge.core.sdr.base import GainMode
except ImportError as e:
    print(f"ERROR: Cannot import Edge modules: {e}")
    print("Make sure you're running from the project root with venv activated")
    sys.exit(1)


def test_bladerf_direct():
    """Test bladeRF hardware directly to see if signal is present"""
    print("=" * 70)
    print("TEST 1: Direct bladeRF Hardware Test")
    print("=" * 70)
    
    try:
        sdr = BladeRFNativeDevice()
        print("✓ bladeRF device opened")
        
        # Configure for VTX frequency
        center_freq = 5917_000_000  # 5.917 GHz
        sample_rate = 20_000_000    # 20 MS/s (good balance)
        bandwidth = 20_000_000      # 20 MHz
        gain_db = 40                # Higher gain for 5.9 GHz
        
        print(f"\nConfiguration:")
        print(f"  Center: {center_freq/1e6:.3f} MHz")
        print(f"  Sample Rate: {sample_rate/1e6:.2f} MS/s")
        print(f"  Bandwidth: {bandwidth/1e6:.2f} MHz")
        print(f"  Gain: {gain_db} dB")
        
        # Tune to frequency
        sdr.tune(center_freq, sample_rate, bandwidth)
        sdr.set_gain_mode(GainMode.MANUAL)
        sdr.set_gain(gain_db)
        print("✓ SDR configured")
        
        # Read samples
        print("\nReading samples...")
        num_samples = 262144  # 256k samples
        iq = sdr.read_samples(num_samples)
        
        if iq.size == 0:
            print("✗ ERROR: No samples read!")
            return False
        
        print(f"✓ Read {iq.size} samples")
        
        # Compute FFT with different sizes
        print("\n" + "-" * 70)
        print("FFT Analysis (different sizes):")
        print("-" * 70)
        
        for fft_size in [4096, 8192, 16384, 32768, 65536]:
            if iq.size < fft_size:
                continue
                
            # Take first fft_size samples
            iq_fft = iq[:fft_size]
            
            # Window
            win = np.hanning(fft_size).astype(np.float32)
            window_sum = float(np.sum(win))
            
            # FFT
            iq_windowed = iq_fft * win
            X = np.fft.fftshift(np.fft.fft(iq_windowed, n=fft_size))
            
            # Power spectrum
            mag = np.abs(X) / window_sum
            spec_db = 20.0 * np.log10(mag + 1e-12)
            
            # Find peak
            peak_idx = np.argmax(spec_db)
            peak_db = spec_db[peak_idx]
            
            # Calculate frequency of peak
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0/sample_rate))
            peak_freq_hz = center_freq + freqs[peak_idx]
            
            # Noise floor (10th percentile)
            noise_floor = np.percentile(spec_db, 10)
            snr = peak_db - noise_floor
            
            # Frequency resolution
            freq_resolution = sample_rate / fft_size
            
            print(f"\nFFT Size: {fft_size}")
            print(f"  Frequency Resolution: {freq_resolution/1e3:.2f} kHz/bin")
            print(f"  Peak Power: {peak_db:.1f} dBFS")
            print(f"  Peak Frequency: {peak_freq_hz/1e6:.6f} MHz")
            print(f"  Noise Floor: {noise_floor:.1f} dBFS")
            print(f"  SNR: {snr:.1f} dB")
            
            # Check if signal is visible (SNR > 6 dB)
            if snr > 6.0:
                print(f"  ✓ SIGNAL DETECTED (SNR > 6 dB)")
            else:
                print(f"  ✗ Signal not visible (SNR < 6 dB)")
        
        # Check sample statistics
        print("\n" + "-" * 70)
        print("Sample Statistics:")
        print("-" * 70)
        sample_mag = np.abs(iq)
        print(f"  Max magnitude: {np.max(sample_mag):.6f}")
        print(f"  Mean magnitude: {np.mean(sample_mag):.6f}")
        print(f"  Std magnitude: {np.std(sample_mag):.6f}")
        
        # Check for clipping
        if np.max(sample_mag) > 0.9:
            print("  ⚠️  WARNING: Possible clipping detected!")
        else:
            print("  ✓ No clipping detected")
        
        sdr.close()
        return True
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frequency_sweep():
    """Sweep frequencies around 5917 MHz to find exact VTX frequency"""
    print("\n" + "=" * 70)
    print("TEST 2: Frequency Sweep (find exact VTX frequency)")
    print("=" * 70)
    
    try:
        sdr = BladeRFNativeDevice()
        
        sample_rate = 20_000_000
        bandwidth = 20_000_000
        gain_db = 40
        fft_size = 8192  # Good balance for signal detection
        
        sdr.tune(5917_000_000, sample_rate, bandwidth)
        sdr.set_gain_mode(GainMode.MANUAL)
        sdr.set_gain(gain_db)
        
        # Sweep ±5 MHz around 5917 MHz
        center_freqs = [5912, 5914, 5916, 5917, 5918, 5920, 5922]
        
        print(f"\nSweeping frequencies (FFT size: {fft_size}):")
        print("-" * 70)
        
        results = []
        for center_mhz in center_freqs:
            center_hz = int(center_mhz * 1e6)
            sdr.tune(center_hz, sample_rate, bandwidth)
            
            # Read samples
            iq = sdr.read_samples(fft_size * 2)
            if iq.size < fft_size:
                continue
            
            # FFT
            win = np.hanning(fft_size).astype(np.float32)
            window_sum = float(np.sum(win))
            iq_fft = iq[:fft_size] * win
            X = np.fft.fftshift(np.fft.fft(iq_fft, n=fft_size))
            mag = np.abs(X) / window_sum
            spec_db = 20.0 * np.log10(mag + 1e-12)
            
            peak_db = np.max(spec_db)
            noise_floor = np.percentile(spec_db, 10)
            snr = peak_db - noise_floor
            
            results.append((center_mhz, peak_db, noise_floor, snr))
            print(f"  {center_mhz:.1f} MHz: Peak={peak_db:6.1f} dBFS, Floor={noise_floor:6.1f} dBFS, SNR={snr:5.1f} dB", end="")
            
            if snr > 6.0:
                print(" ✓ SIGNAL")
            else:
                print("")
        
        # Find best frequency
        best_freq, best_peak, best_floor, best_snr = max(results, key=lambda x: x[3])
        print(f"\n✓ Best signal at {best_freq:.1f} MHz (SNR: {best_snr:.1f} dB)")
        
        sdr.close()
        return best_freq
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "=" * 70)
    print("VTX SIGNAL DETECTION DIAGNOSTIC")
    print("=" * 70)
    print("\nThis tool tests bladeRF hardware directly to verify VTX signal detection.")
    print("VTX Frequency: 5.917 GHz (5917 MHz)")
    print()
    
    # Test 1: Direct hardware test
    success = test_bladerf_direct()
    
    if not success:
        print("\n✗ Hardware test failed. Check bladeRF connection.")
        return
    
    # Test 2: Frequency sweep
    best_freq = test_frequency_sweep()
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print()
    print("1. If signal detected with smaller FFT sizes but not 65536:")
    print("   → Reduce FFT size in Edge to 4096-8192")
    print()
    print("2. If best frequency is different from 5917 MHz:")
    print(f"   → Set center frequency to {best_freq:.1f} MHz in Edge")
    print()
    print("3. If SNR is low (< 6 dB):")
    print("   → Increase gain (try 40-50 dB)")
    print("   → Check antenna connection")
    print()
    print("4. If no signal detected at all:")
    print("   → Verify antenna is connected to bladeRF RX port")
    print("   → Check antenna is tuned for 5.9 GHz")
    print("   → Compare with TinySA to verify signal is actually present")
    print()


if __name__ == "__main__":
    main()

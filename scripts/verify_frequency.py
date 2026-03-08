#!/usr/bin/env python3
"""
Verify actual tuned frequency by checking where known signals appear.
Uses FM radio stations as reference (they're at known frequencies).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import ctypes
from spear_edge.core.sdr.bladerf_native import (
    BladeRFNativeDevice, 
    _libbladerf,
    BLADERF_CHANNEL_RX
)
from spear_edge.core.sdr.base import SdrConfig, GainMode

def verify_frequency():
    """Verify actual tuned frequency using known signals."""
    print("=" * 80)
    print("FREQUENCY VERIFICATION - Using Known Signals")
    print("=" * 80)
    
    sdr = BladeRFNativeDevice()
    
    try:
        sdr.open()
        print("✓ Device opened\n")
        
        # Test 1: Tune to known FM station frequency
        # FM stations are at known frequencies (e.g., 100.1 MHz = 100100000 Hz)
        test_freq = 100_100_000  # 100.1 MHz (common FM station)
        sample_rate = 2_400_000  # 2.4 MS/s (low rate for FM)
        bandwidth = 2_400_000
        gain = 30.0
        
        print(f"[TEST 1] Tuning to known FM frequency: {test_freq/1e6:.3f} MHz")
        print(f"  Sample rate: {sample_rate/1e6:.2f} MS/s")
        print(f"  Bandwidth: {bandwidth/1e6:.2f} MHz")
        print(f"  Gain: {gain} dB\n")
        
        config = SdrConfig(
            center_freq_hz=test_freq,
            sample_rate_sps=sample_rate,
            bandwidth_hz=bandwidth,
            gain_mode=GainMode.MANUAL,
            gain_db=gain,
            rx_channel=0,
        )
        sdr.apply_config(config)
        time.sleep(0.5)
        
        # Read back frequency
        ch = BLADERF_CHANNEL_RX(0)
        actual_freq = ctypes.c_uint64()
        ret = _libbladerf.bladerf_get_frequency(sdr.dev, ch, ctypes.byref(actual_freq))
        
        print(f"  Requested: {test_freq/1e6:.6f} MHz")
        if ret == 0:
            print(f"  Readback: {actual_freq.value/1e6:.6f} MHz")
            print(f"  Difference: {abs(actual_freq.value - test_freq)/1e6:.3f} MHz")
        else:
            print(f"  Readback: FAILED (code {ret})")
        
        # Read samples and find peak
        print(f"\n  Reading samples and finding signal...")
        all_samples = []
        for i in range(5):
            iq = sdr.read_samples(8192)
            if len(iq) > 0:
                all_samples.append(iq)
        
        if all_samples:
            combined = np.concatenate(all_samples)
            fft_size = 2048
            
            # FFT
            window = np.hanning(fft_size)
            windowed = combined[:fft_size] * window
            fft = np.fft.fftshift(np.fft.fft(windowed))
            mag = np.abs(fft) / np.sum(window)
            spec_db = 20.0 * np.log10(mag + 1e-12)
            
            # Frequency axis
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate))
            freqs_hz = freqs + test_freq
            
            # Find peak
            peak_idx = np.argmax(spec_db)
            peak_freq = freqs_hz[peak_idx] / 1e6
            peak_power = spec_db[peak_idx]
            
            print(f"  Peak signal: {peak_power:.1f} dBFS @ {peak_freq:.3f} MHz")
            print(f"  Expected: {test_freq/1e6:.3f} MHz")
            print(f"  Offset: {peak_freq - test_freq/1e6:.3f} MHz")
            
            if abs(peak_freq - test_freq/1e6) < 0.1:
                print(f"  ✓ Frequency tuning is CORRECT (peak matches requested)")
            else:
                print(f"  ⚠️  Frequency tuning may be WRONG (peak doesn't match)")
        
        # Test 2: Now test at VTX frequency
        print(f"\n[TEST 2] Testing at VTX frequency: 5910 MHz")
        vtx_freq = 5910_000_000
        sample_rate_vtx = 40_000_000
        bandwidth_vtx = 40_000_000
        gain_vtx = 12.0
        
        config_vtx = SdrConfig(
            center_freq_hz=vtx_freq,
            sample_rate_sps=sample_rate_vtx,
            bandwidth_hz=bandwidth_vtx,
            gain_mode=GainMode.MANUAL,
            gain_db=gain_vtx,
            rx_channel=0,
        )
        sdr.apply_config(config_vtx)
        time.sleep(0.5)
        
        # Read back frequency
        actual_freq_vtx = ctypes.c_uint64()
        ret_vtx = _libbladerf.bladerf_get_frequency(sdr.dev, ch, ctypes.byref(actual_freq_vtx))
        
        print(f"  Requested: {vtx_freq/1e6:.6f} MHz")
        if ret_vtx == 0:
            print(f"  Readback: {actual_freq_vtx.value/1e6:.6f} MHz")
            print(f"  Difference: {abs(actual_freq_vtx.value - vtx_freq)/1e6:.3f} MHz")
            if abs(actual_freq_vtx.value - vtx_freq) > 100_000_000:  # > 100 MHz difference
                print(f"  ⚠️  WARNING: Large frequency difference! Hardware may not be tuned correctly.")
        else:
            print(f"  Readback: FAILED (code {ret_vtx})")
        
        # Read samples and analyze
        print(f"\n  Reading samples and analyzing signal location...")
        all_samples_vtx = []
        for i in range(5):
            iq = sdr.read_samples(196608)
            if len(iq) > 0:
                all_samples_vtx.append(iq)
        
        if all_samples_vtx:
            combined_vtx = np.concatenate(all_samples_vtx)
            fft_size_vtx = 65536
            
            # FFT
            window_vtx = np.hanning(fft_size_vtx)
            windowed_vtx = combined_vtx[:fft_size_vtx] * window_vtx
            fft_vtx = np.fft.fftshift(np.fft.fft(windowed_vtx))
            mag_vtx = np.abs(fft_vtx) / np.sum(window_vtx)
            spec_db_vtx = 20.0 * np.log10(mag_vtx + 1e-12)
            
            # Frequency axis
            freqs_vtx = np.fft.fftshift(np.fft.fftfreq(fft_size_vtx, d=1.0 / sample_rate_vtx))
            freqs_hz_vtx = freqs_vtx + vtx_freq
            
            # Find top 5 peaks
            peak_indices = np.argsort(spec_db_vtx)[::-1][:5]
            
            print(f"  Top 5 peaks:")
            for i, idx in enumerate(peak_indices):
                freq_mhz = freqs_hz_vtx[idx] / 1e6
                power = spec_db_vtx[idx]
                offset_mhz = freq_mhz - vtx_freq/1e6
                print(f"    {i+1}. {power:.1f} dBFS @ {freq_mhz:.3f} MHz (offset: {offset_mhz:+.3f} MHz)")
            
            # Check if signal is where expected (5917 MHz = +7 MHz offset)
            expected_vtx = 5917.0
            expected_offset = expected_vtx - vtx_freq/1e6  # Should be +7 MHz
            print(f"\n  Expected VTX: {expected_vtx} MHz (offset: {expected_offset:+.3f} MHz from center)")
            
            # Find closest peak to expected location
            closest_peak_idx = peak_indices[0]
            closest_offset = abs(freqs_hz_vtx[closest_peak_idx] / 1e6 - expected_vtx)
            for idx in peak_indices:
                offset = abs(freqs_hz_vtx[idx] / 1e6 - expected_vtx)
                if offset < closest_offset:
                    closest_offset = offset
                    closest_peak_idx = idx
            
            closest_freq = freqs_hz_vtx[closest_peak_idx] / 1e6
            closest_power = spec_db_vtx[closest_peak_idx]
            print(f"  Closest peak to {expected_vtx} MHz: {closest_power:.1f} dBFS @ {closest_freq:.3f} MHz")
            print(f"  Distance from expected: {closest_offset:.3f} MHz")
            
            if closest_offset < 2.0:  # Within 2 MHz
                print(f"  ✓ Signal is near expected location")
            else:
                print(f"  ⚠️  Signal is NOT at expected location - frequency tuning may be wrong")
            
            # Calculate what frequency we're actually tuned to based on signal location
            # If signal is at 5906 MHz but we requested 5910 MHz, we might be tuned to 5906-4 = 5902 MHz
            # Or there's a systematic offset
            if len(peak_indices) > 0:
                main_peak_freq = freqs_hz_vtx[peak_indices[0]] / 1e6
                apparent_center = main_peak_freq - (freqs_vtx[peak_indices[0]] / 1e6)
                print(f"\n  Signal location analysis:")
                print(f"    Main peak at: {main_peak_freq:.3f} MHz")
                print(f"    Peak's FFT offset: {freqs_vtx[peak_indices[0]]/1e6:.3f} MHz")
                print(f"    Apparent center freq: {apparent_center:.3f} MHz")
                print(f"    Requested center: {vtx_freq/1e6:.3f} MHz")
                print(f"    Apparent offset: {apparent_center - vtx_freq/1e6:.3f} MHz")
        
        print("\n" + "=" * 80)
        print("Verification complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sdr.close()

if __name__ == "__main__":
    verify_frequency()

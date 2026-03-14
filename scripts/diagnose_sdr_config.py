#!/usr/bin/env python3
"""
Diagnostic script to verify SDR configuration and check for issues.
Tests frequency tuning, channel selection, bandwidth, and signal presence.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from spear_edge.core.sdr.bladerf_native import BladeRFNativeDevice
from spear_edge.core.sdr.base import SdrConfig, GainMode
import time

def test_sdr_config():
    """Test SDR configuration and verify settings match what's requested."""
    print("=" * 60)
    print("SDR Configuration Diagnostic")
    print("=" * 60)
    
    sdr = BladeRFNativeDevice()
    
    try:
        print("\n[1] Opening device...")
        sdr.open()
        print("    ✓ Device opened")
        
        # Test configuration
        center_freq = 5914_000_000  # 5914 MHz
        sample_rate = 40_000_000     # 40 MS/s
        bandwidth = None  # Should default to sample_rate
        gain = 40.0
        
        print(f"\n[2] Applying configuration...")
        print(f"    Center Freq: {center_freq / 1e6:.3f} MHz")
        print(f"    Sample Rate: {sample_rate / 1e6:.2f} MS/s")
        print(f"    Bandwidth: {'Auto (sample_rate)' if bandwidth is None else f'{bandwidth/1e6:.2f} MHz'}")
        print(f"    Gain: {gain} dB")
        print(f"    RX Channel: 0")
        
        config = SdrConfig(
            center_freq_hz=center_freq,
            sample_rate_sps=sample_rate,
            bandwidth_hz=bandwidth,
            gain_mode=GainMode.MANUAL,
            gain_db=gain,
            rx_channel=0,
        )
        
        sdr.apply_config(config)
        print("    ✓ Configuration applied")
        
        # Verify settings
        print(f"\n[3] Verifying settings...")
        print(f"    Internal center_freq_hz: {sdr.center_freq_hz / 1e6:.3f} MHz")
        print(f"    Internal sample_rate_sps: {sdr.sample_rate_sps / 1e6:.2f} MS/s")
        print(f"    Internal bandwidth_hz: {sdr.bandwidth_hz / 1e6:.2f} MHz")
        print(f"    Internal gain_db: {sdr.gain_db:.1f} dB")
        print(f"    Internal rx_channel: {sdr.rx_channel}")
        print(f"    Stream active: {sdr._stream_active}")
        
        # Check if stream is set up
        if sdr._stream_active:
            print("\n[4] Stream is active - reading samples...")
            print("    Reading 8192 samples (power-of-two)...")
            
            # Read a few chunks to see if we get data
            for i in range(3):
                iq = sdr.read_samples(8192)
                if len(iq) > 0:
                    # Calculate power
                    power = np.mean(np.abs(iq) ** 2)
                    power_db = 10 * np.log10(power + 1e-12)
                    max_mag = np.max(np.abs(iq))
                    mean_mag = np.mean(np.abs(iq))
                    
                    print(f"    Chunk {i+1}: {len(iq)} samples, power={power_db:.2f} dB, max_mag={max_mag:.6f}, mean_mag={mean_mag:.6f}")
                    
                    # Quick FFT to see if there's signal
                    if len(iq) >= 1024:
                        fft = np.fft.fftshift(np.fft.fft(iq[:1024]))
                        fft_power = np.abs(fft) ** 2
                        fft_db = 10 * np.log10(fft_power + 1e-12)
                        peak_idx = np.argmax(fft_db)
                        peak_power = fft_db[peak_idx]
                        noise_floor = np.percentile(fft_db, 10)
                        snr = peak_power - noise_floor
                        
                        print(f"      FFT (1024): peak={peak_power:.1f} dBFS @ bin {peak_idx}, noise_floor={noise_floor:.1f} dBFS, SNR={snr:.1f} dB")
                else:
                    print(f"    Chunk {i+1}: EMPTY (no samples returned)")
                    time.sleep(0.1)
        else:
            print("\n[4] Stream is NOT active - this is the problem!")
            print("    The stream needs to be activated for reading samples.")
            print("    This happens in _setup_stream() which is called during tune().")
        
        # Check frequency range
        print(f"\n[5] Frequency range check...")
        freq_range_start = (center_freq - sample_rate // 2) / 1e6
        freq_range_end = (center_freq + sample_rate // 2) / 1e6
        print(f"    Capture range: {freq_range_start:.3f} to {freq_range_end:.3f} MHz")
        print(f"    VTX frequency (5914 MHz) should be in range: {freq_range_start <= 5914 <= freq_range_end}")
        
        # Check bandwidth
        print(f"\n[6] Bandwidth check...")
        print(f"    Bandwidth: {sdr.bandwidth_hz / 1e6:.2f} MHz")
        print(f"    VTX signal is ~20-30 MHz wide")
        print(f"    Bandwidth sufficient: {sdr.bandwidth_hz >= 30_000_000}")
        
    except Exception as e:
        print(f"\n[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[7] Closing device...")
        sdr.close()
        print("    ✓ Device closed")
        print("\n" + "=" * 60)

if __name__ == "__main__":
    test_sdr_config()

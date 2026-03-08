#!/usr/bin/env python3
"""
Test SDR configuration and signal detection using libbladeRF native.
Verifies frequency tuning, bandwidth, channel selection, and signal presence.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from spear_edge.core.sdr.bladerf_native import BladeRFNativeDevice
from spear_edge.core.sdr.base import SdrConfig, GainMode

def test_sdr_signal():
    """Test SDR configuration and detect signal."""
    print("=" * 70)
    print("SDR Signal Detection Test (libbladeRF Native)")
    print("=" * 70)
    
    sdr = BladeRFNativeDevice()
    
    try:
        print("\n[1] Opening bladeRF device...")
        sdr.open()
        print("    ✓ Device opened successfully")
        
        # Configuration matching Edge settings
        center_freq = 5915_000_000  # 5915 MHz (actual VTX center frequency)
        sample_rate = 40_000_000    # 40 MS/s
        bandwidth = None  # Auto (should default to sample_rate)
        gain = 40.0
        rx_channel = 0
        
        print(f"\n[2] Applying SDR configuration...")
        print(f"    Center Frequency: {center_freq / 1e6:.3f} MHz")
        print(f"    Sample Rate: {sample_rate / 1e6:.2f} MS/s")
        print(f"    Bandwidth: {'Auto (sample_rate)' if bandwidth is None else f'{bandwidth/1e6:.2f} MHz'}")
        print(f"    Gain: {gain} dB")
        print(f"    RX Channel: {rx_channel}")
        
        config = SdrConfig(
            center_freq_hz=center_freq,
            sample_rate_sps=sample_rate,
            bandwidth_hz=bandwidth,
            gain_mode=GainMode.MANUAL,
            gain_db=gain,
            rx_channel=rx_channel,
        )
        
        sdr.apply_config(config)
        print("    ✓ Configuration applied")
        
        # Verify internal state
        print(f"\n[3] Verifying internal state...")
        print(f"    center_freq_hz: {sdr.center_freq_hz / 1e6:.6f} MHz")
        print(f"    sample_rate_sps: {sdr.sample_rate_sps / 1e6:.6f} MS/s")
        print(f"    bandwidth_hz: {sdr.bandwidth_hz / 1e6:.6f} MHz")
        print(f"    gain_db: {sdr.gain_db:.1f} dB")
        print(f"    rx_channel: {sdr.rx_channel}")
        print(f"    stream_active: {sdr._stream_active}")
        
        # Check frequency range
        freq_range_start = (center_freq - sample_rate // 2) / 1e6
        freq_range_end = (center_freq + sample_rate // 2) / 1e6
        print(f"\n[4] Frequency range analysis...")
        print(f"    Capture range: {freq_range_start:.3f} to {freq_range_end:.3f} MHz")
        print(f"    VTX frequency (5915 MHz) in range: {freq_range_start <= 5915 <= freq_range_end}")
        print(f"    Bandwidth: {sdr.bandwidth_hz / 1e6:.2f} MHz")
        print(f"    VTX signal width: ~20-30 MHz")
        print(f"    Bandwidth sufficient: {sdr.bandwidth_hz >= 30_000_000}")
        
        if not sdr._stream_active:
            print("\n[ERROR] Stream is not active! Cannot read samples.")
            print("        This means _setup_stream() was not called or failed.")
            return
        
        print(f"\n[5] Reading samples and analyzing signal...")
        print("    Reading 3 chunks of 8192 samples each...")
        
        all_samples = []
        for i in range(3):
            iq = sdr.read_samples(8192)
            if len(iq) == 0:
                print(f"    Chunk {i+1}: EMPTY (no samples returned)")
                time.sleep(0.1)
                continue
            
            all_samples.append(iq)
            power = np.mean(np.abs(iq) ** 2)
            power_db = 10 * np.log10(power + 1e-12)
            max_mag = np.max(np.abs(iq))
            mean_mag = np.mean(np.abs(iq))
            std_mag = np.std(np.abs(iq))
            
            print(f"    Chunk {i+1}: {len(iq)} samples")
            print(f"      Power: {power_db:.2f} dB, max_mag={max_mag:.6f}, mean_mag={mean_mag:.6f}, std_mag={std_mag:.6f}")
        
        if not all_samples:
            print("\n[ERROR] No samples received! Cannot analyze signal.")
            return
        
        # Combine all samples for analysis
        combined_iq = np.concatenate(all_samples)
        print(f"\n[6] Combined analysis ({len(combined_iq)} total samples)...")
        
        # Calculate statistics
        power = np.mean(np.abs(combined_iq) ** 2)
        power_db = 10 * np.log10(power + 1e-12)
        max_mag = np.max(np.abs(combined_iq))
        mean_mag = np.mean(np.abs(combined_iq))
        std_mag = np.std(np.abs(combined_iq))
        
        print(f"    Total power: {power_db:.2f} dB")
        print(f"    Max magnitude: {max_mag:.6f}")
        print(f"    Mean magnitude: {mean_mag:.6f}")
        print(f"    Std magnitude: {std_mag:.6f}")
        
        # FFT analysis with different sizes
        print(f"\n[7] FFT analysis (looking for signal at 5915 MHz)...")
        
        fft_sizes = [1024, 2048, 4096]
        for fft_size in fft_sizes:
            if len(combined_iq) < fft_size:
                continue
            
            # Use first fft_size samples
            iq_segment = combined_iq[:fft_size]
            
            # Apply window
            window = np.hanning(len(iq_segment))
            windowed = iq_segment * window
            window_sum = np.sum(window)
            
            # FFT
            fft = np.fft.fftshift(np.fft.fft(windowed))
            mag = np.abs(fft) / window_sum
            spec_db = 20.0 * np.log10(mag + 1e-12)
            
            # Frequency axis
            freqs = np.fft.fftshift(np.fft.fftfreq(len(iq_segment), d=1.0 / sample_rate))
            freqs_hz = freqs + center_freq
            
            # Find peak
            peak_idx = np.argmax(spec_db)
            peak_power = spec_db[peak_idx]
            peak_freq = freqs_hz[peak_idx] / 1e6
            
            # Noise floor (10th percentile, excluding edges)
            exclude_pct = 0.05
            exclude_count = int(len(spec_db) * exclude_pct)
            center_spectrum = spec_db[exclude_count:-exclude_count] if exclude_count > 0 else spec_db
            noise_floor = np.percentile(center_spectrum, 10)
            
            # SNR
            snr = peak_power - noise_floor
            
            # Check if signal is near 5915 MHz
            target_freq = 5915.0
            freq_bin_width = (sample_rate / fft_size) / 1e6  # MHz per bin
            distance_from_target = abs(peak_freq - target_freq)
            
            print(f"\n    FFT Size: {fft_size}")
            print(f"      Peak: {peak_power:.1f} dBFS @ {peak_freq:.3f} MHz (bin {peak_idx})")
            print(f"      Noise floor: {noise_floor:.1f} dBFS")
            print(f"      SNR: {snr:.1f} dB")
            print(f"      Distance from 5915 MHz: {distance_from_target:.3f} MHz")
            print(f"      Bin width: {freq_bin_width:.3f} MHz")
            print(f"      Signal detected near 5915 MHz: {distance_from_target < freq_bin_width * 10}")
            
            # Check for wideband signal (power spread across bins)
            signal_threshold = noise_floor + 3.0  # 3 dB above noise
            signal_bins = np.sum(spec_db > signal_threshold)
            signal_pct = (signal_bins / len(spec_db)) * 100
            
            print(f"      Bins above noise+3dB: {signal_bins} ({signal_pct:.1f}%)")
            if signal_pct > 20:
                print(f"      → WIDEBAND SIGNAL DETECTED ({signal_pct:.1f}% of spectrum)")
            elif signal_pct > 5:
                print(f"      → Possible wideband signal ({signal_pct:.1f}% of spectrum)")
            else:
                print(f"      → Narrowband or no signal ({signal_pct:.1f}% of spectrum)")
        
        # Test channel 1 if channel 0 shows nothing
        if rx_channel == 0:
            print(f"\n[8] Testing RX Channel 1 (in case antenna is on wrong channel)...")
            print("    Reconfiguring to channel 1...")
            
            config_ch1 = SdrConfig(
                center_freq_hz=center_freq,
                sample_rate_sps=sample_rate,
                bandwidth_hz=bandwidth,
                gain_mode=GainMode.MANUAL,
                gain_db=gain,
                rx_channel=1,  # Channel 1
            )
            
            sdr.apply_config(config_ch1)
            print("    ✓ Channel 1 configured")
            
            # Read one chunk
            iq_ch1 = sdr.read_samples(8192)
            if len(iq_ch1) > 0:
                power_ch1 = np.mean(np.abs(iq_ch1) ** 2)
                power_db_ch1 = 10 * np.log10(power_ch1 + 1e-12)
                print(f"    Channel 1 power: {power_db_ch1:.2f} dB")
                print(f"    Channel 0 power: {power_db:.2f} dB")
                print(f"    Difference: {power_db_ch1 - power_db:.2f} dB")
                if power_db_ch1 > power_db + 3:
                    print("    → CHANNEL 1 HAS MORE POWER! Antenna might be on channel 1.")
            else:
                print("    Channel 1: No samples returned")
        
        print("\n" + "=" * 70)
        print("Test complete!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[9] Closing device...")
        sdr.close()
        print("    ✓ Device closed")

if __name__ == "__main__":
    test_sdr_signal()

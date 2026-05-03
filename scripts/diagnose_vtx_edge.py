#!/usr/bin/env python3
"""
Comprehensive VTX diagnostic using libbladeRF directly.
Matches Edge's exact configuration and shows hardware truth.
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

def diagnose_vtx_edge_config():
    """Diagnose VTX signal using Edge's exact configuration."""
    print("=" * 80)
    print("VTX DIAGNOSTIC - Matching Edge Configuration")
    print("=" * 80)
    
    # Edge's exact configuration from screenshot
    center_freq = 5910_000_000  # 5910 MHz (from UI)
    sample_rate = 40_000_000    # 40 MS/s
    bandwidth = 40_000_000      # 40 MHz
    gain = 12.0                 # 12 dB
    fft_size = 65536            # FFT size from UI
    rx_channel = 0
    
    print(f"\n[CONFIG] Edge Configuration:")
    print(f"  Center Frequency: {center_freq / 1e6:.3f} MHz")
    print(f"  Sample Rate: {sample_rate / 1e6:.2f} MS/s")
    print(f"  Bandwidth: {bandwidth / 1e6:.2f} MHz")
    print(f"  Gain: {gain} dB")
    print(f"  FFT Size: {fft_size}")
    print(f"  RX Channel: {rx_channel}")
    
    sdr = BladeRFNativeDevice()
    
    try:
        print(f"\n[1] Opening bladeRF device...")
        sdr.open()
        print("    ✓ Device opened")
        
        print(f"\n[2] Applying Edge configuration...")
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
        
        # Wait for stream to stabilize
        time.sleep(0.5)
        
        print(f"\n[3] HARDWARE TRUTH - Reading back actual values...")
        ch = BLADERF_CHANNEL_RX(rx_channel)
        
        # Read back frequency
        actual_freq = ctypes.c_uint64()
        ret_freq = _libbladerf.bladerf_get_frequency(sdr.dev, ch, ctypes.byref(actual_freq))
        if ret_freq == 0:
            print(f"  Frequency: req={center_freq/1e6:.6f} MHz, act={actual_freq.value/1e6:.6f} MHz")
            freq_diff = abs(actual_freq.value - center_freq)
            if freq_diff > 1000:  # > 1 kHz difference
                print(f"    ⚠️  WARNING: {freq_diff/1e6:.3f} MHz difference!")
        else:
            print(f"  Frequency: req={center_freq/1e6:.6f} MHz, act=READBACK_FAILED")
        
        # Read back sample rate
        actual_rate = ctypes.c_uint32()
        ret_rate = _libbladerf.bladerf_get_sample_rate(sdr.dev, ch, ctypes.byref(actual_rate))
        if ret_rate == 0:
            print(f"  Sample Rate: req={sample_rate/1e6:.2f} MS/s, act={actual_rate.value/1e6:.2f} MS/s")
            if actual_rate.value != sample_rate:
                print(f"    ⚠️  WARNING: {abs(actual_rate.value - sample_rate)/1e6:.2f} MS/s difference!")
        else:
            print(f"  Sample Rate: req={sample_rate/1e6:.2f} MS/s, act=READBACK_FAILED")
        
        # Read back bandwidth
        actual_bw = ctypes.c_uint32()
        ret_bw = _libbladerf.bladerf_get_bandwidth(sdr.dev, ch, ctypes.byref(actual_bw))
        if ret_bw == 0:
            print(f"  Bandwidth: req={bandwidth/1e6:.2f} MHz, act={actual_bw.value/1e6:.2f} MHz")
            if actual_bw.value < 30_000_000:
                print(f"    ⚠️  WARNING: Bandwidth < 30 MHz may not capture full VTX signal!")
        else:
            print(f"  Bandwidth: req={bandwidth/1e6:.2f} MHz, act=READBACK_FAILED")
        
        # Read back gain
        gain_ptr = ctypes.c_int()
        ret_gain = _libbladerf.bladerf_get_gain(sdr.dev, ch, ctypes.byref(gain_ptr))
        if ret_gain == 0:
            print(f"  Gain: req={gain:.1f} dB, act={gain_ptr.value} dB")
        else:
            print(f"  Gain: req={gain:.1f} dB, act=READBACK_FAILED")
        
        if not sdr._stream_active:
            print("\n[ERROR] Stream not active! Cannot read samples.")
            return
        
        print(f"\n[4] Reading samples (matching Edge's read size)...")
        # Edge uses adaptive chunk sizes, but for 40 MS/s it's around 192K samples
        read_size = 196608  # ~75% of 256K buffer for 40 MS/s
        print(f"  Reading {read_size} samples (power-of-two: {read_size & (read_size-1) == 0})")
        
        all_samples = []
        for i in range(5):  # Read 5 chunks
            iq = sdr.read_samples(read_size)
            if len(iq) == 0:
                print(f"    Chunk {i+1}: EMPTY")
                time.sleep(0.1)
                continue
            all_samples.append(iq)
            print(f"    Chunk {i+1}: {len(iq)} samples")
        
        if not all_samples:
            print("\n[ERROR] No samples received!")
            return
        
        combined_iq = np.concatenate(all_samples)
        print(f"\n[5] Combined IQ analysis ({len(combined_iq)} total samples)...")
        
        # Raw IQ statistics
        iq_power = np.mean(np.abs(combined_iq) ** 2)
        iq_power_db = 10 * np.log10(iq_power + 1e-12)
        iq_max = np.max(np.abs(combined_iq))
        iq_mean = np.mean(np.abs(combined_iq))
        iq_std = np.std(np.abs(combined_iq))
        
        print(f"  IQ Power: {iq_power_db:.2f} dB")
        print(f"  Max |IQ|: {iq_max:.6f}")
        print(f"  Mean |IQ|: {iq_mean:.6f}")
        print(f"  Std |IQ|: {iq_std:.6f}")
        print(f"  Signal ratio (std/mean): {iq_std/iq_mean:.2f}")
        
        # Use enough samples for FFT
        if len(combined_iq) < fft_size:
            print(f"\n[WARNING] Not enough samples for {fft_size}-point FFT, using {len(combined_iq)}")
            fft_size_actual = len(combined_iq)
        else:
            fft_size_actual = fft_size
            # Use first fft_size samples
            iq_fft = combined_iq[:fft_size_actual].copy()
        
        print(f"\n[6] FFT Analysis (size={fft_size_actual}, matching Edge processing)...")
        
        # Match Edge's exact FFT processing
        # 1. DC removal (if enabled - Edge has it disabled by default)
        from spear_edge.settings import settings
        if settings.DC_REMOVAL:
            dc_offset = np.mean(iq_fft)
            iq_fft = iq_fft - dc_offset
            print(f"  DC offset removed: {dc_offset:.6f}")
        else:
            print(f"  DC removal: DISABLED (Edge default)")
        
        # 2. Window (Hanning, matching Edge)
        window = np.hanning(fft_size_actual).astype(np.float32)
        windowed = iq_fft * window
        window_sum = np.sum(window)
        
        # 3. FFT (matching Edge)
        fft = np.fft.fftshift(np.fft.fft(windowed, n=fft_size_actual))
        mag = np.abs(fft) / window_sum
        spec_db = 20.0 * np.log10(mag + 1e-12)
        
        # 4. Frequency axis
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_size_actual, d=1.0 / sample_rate))
        freqs_hz = freqs + center_freq
        freqs_mhz = freqs_hz / 1e6
        
        # 5. Find peaks
        peak_idx = np.argmax(spec_db)
        peak_power = spec_db[peak_idx]
        peak_freq = freqs_mhz[peak_idx]
        
        print(f"\n  Peak Signal:")
        print(f"    Power: {peak_power:.1f} dBFS")
        print(f"    Frequency: {peak_freq:.3f} MHz")
        print(f"    Bin: {peak_idx}")
        print(f"    Offset from center: {peak_freq - center_freq/1e6:.3f} MHz")
        
        # 6. Noise floor (matching Edge's wideband detection)
        exclude_pct = 0.05
        exclude_count = int(len(spec_db) * exclude_pct)
        center_spectrum = spec_db[exclude_count:-exclude_count]
        
        # Edge uses 2nd percentile for wideband
        noise_floor_2nd = np.percentile(center_spectrum, 2)
        noise_floor_10th = np.percentile(center_spectrum, 10)
        
        print(f"\n  Noise Floor:")
        print(f"    2nd percentile (wideband): {noise_floor_2nd:.1f} dBFS")
        print(f"    10th percentile (narrowband): {noise_floor_10th:.1f} dBFS")
        print(f"    SNR (peak - 2nd pct): {peak_power - noise_floor_2nd:.1f} dB")
        
        # 7. Wideband detection (matching Edge)
        signal_threshold = noise_floor_2nd + 3.0
        signal_bins = np.sum(center_spectrum > signal_threshold)
        signal_pct = (signal_bins / len(center_spectrum)) * 100
        is_wideband = signal_pct > 20.0
        
        print(f"\n  Wideband Detection:")
        print(f"    Bins > noise+3dB: {signal_bins} ({signal_pct:.1f}%)")
        print(f"    Wideband: {is_wideband}")
        
        # 8. Edge bin analysis (matching Edge's edge zeroing)
        edge_zero_pct = 0.025  # 2.5% from each edge
        edge_zero_count = int(len(spec_db) * edge_zero_pct)
        
        print(f"\n  Edge Bin Analysis:")
        print(f"    Edge zero percentage: {edge_zero_pct*100:.1f}% ({edge_zero_count} bins each side)")
        print(f"    First {edge_zero_count} bins: min={np.min(spec_db[:edge_zero_count]):.1f} dBFS, "
              f"max={np.max(spec_db[:edge_zero_count]):.1f} dBFS, "
              f"mean={np.mean(spec_db[:edge_zero_count]):.1f} dBFS")
        print(f"    Last {edge_zero_count} bins: min={np.min(spec_db[-edge_zero_count:]):.1f} dBFS, "
              f"max={np.max(spec_db[-edge_zero_count:]):.1f} dBFS, "
              f"mean={np.mean(spec_db[-edge_zero_count:]):.1f} dBFS")
        
        # Check if peak is in edge region
        if peak_idx < edge_zero_count:
            print(f"    ⚠️  PEAK IS IN LEFT EDGE REGION (bin {peak_idx} < {edge_zero_count})")
            print(f"       This signal would be ZEROED by Edge's edge bin removal!")
        elif peak_idx >= len(spec_db) - edge_zero_count:
            print(f"    ⚠️  PEAK IS IN RIGHT EDGE REGION (bin {peak_idx} >= {len(spec_db) - edge_zero_count})")
            print(f"       This signal would be ZEROED by Edge's edge bin removal!")
        else:
            print(f"    ✓ Peak is in center region (not zeroed)")
        
        # 9. VTX frequency check
        vtx_freq = 5917.0  # Expected VTX frequency
        bin_width_mhz = (sample_rate / fft_size_actual) / 1e6
        vtx_bin = int((vtx_freq - center_freq/1e6) / bin_width_mhz) + fft_size_actual // 2
        
        print(f"\n  VTX Frequency Analysis:")
        print(f"    Expected VTX: {vtx_freq} MHz")
        print(f"    Bin width: {bin_width_mhz:.6f} MHz")
        print(f"    Expected VTX bin: {vtx_bin}")
        print(f"    Power at VTX bin: {spec_db[vtx_bin]:.1f} dBFS")
        print(f"    Distance from peak: {abs(vtx_bin - peak_idx)} bins ({abs(vtx_bin - peak_idx) * bin_width_mhz:.3f} MHz)")
        
        if vtx_bin < edge_zero_count or vtx_bin >= len(spec_db) - edge_zero_count:
            print(f"    ⚠️  VTX BIN IS IN EDGE REGION - would be zeroed!")
        
        # 10. Show top 10 peaks
        print(f"\n  Top 10 Peaks:")
        peak_indices = np.argsort(spec_db)[::-1][:10]
        for i, idx in enumerate(peak_indices):
            print(f"    {i+1}. {spec_db[idx]:.1f} dBFS @ {freqs_mhz[idx]:.3f} MHz (bin {idx})")
        
        print("\n" + "=" * 80)
        print("Diagnostic complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[7] Closing device...")
        sdr.close()
        print("    ✓ Device closed")

if __name__ == "__main__":
    diagnose_vtx_edge_config()
